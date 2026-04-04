"""
Improved Segmentation Training Script
DINOv2 Backbone + ConvNeXt Head
Fixes: Mask resize, Loss function, Optimizer, Augmentation, LR Scheduler
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch import nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
import cv2
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

# ============================================================================
# CONFIG — Yahan sab settings hain, sirf yeh badlo
# ============================================================================

BATCH_SIZE  = 2          # CPU pe 2 rakho
N_EPOCHS    = 20         # 40 epochs — enough for good convergence
LR          = 3e-4       # AdamW ke liye optimal
BACKBONE    = "small"    # small/base/large — small fast hai CPU pe
IMG_W       = int(((960 / 2) // 14) * 14)   # 476
IMG_H       = int(((540 / 2) // 14) * 14)   # 266

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths — apna path daal yahan
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(SCRIPT_DIR, '..', 'Offroad_Segmentation_Training_Dataset', 'train')
VAL_DIR     = os.path.join(SCRIPT_DIR, '..', 'Offroad_Segmentation_Training_Dataset', 'val')
OUTPUT_DIR  = os.path.join(SCRIPT_DIR, 'train_stats')
MODEL_PATH  = os.path.join(SCRIPT_DIR, 'segmentation_head_v2.pth')

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# Class Mapping
# ============================================================================

value_map = {
    0:     0,   # background
    100:   1,   # Trees
    200:   2,   # Lush Bushes
    300:   3,   # Dry Grass
    500:   4,   # Dry Bushes
    550:   5,   # Ground Clutter
    700:   6,   # Logs
    800:   7,   # Rocks
    7100:  8,   # Landscape
    10000: 9,   # Sky
}
n_classes = len(value_map)  # 10

CLASS_NAMES = [
    'Background', 'Trees', 'Lush Bushes', 'Dry Grass',
    'Dry Bushes', 'Ground Clutter', 'Logs',
    'Rocks', 'Landscape', 'Sky'
]

# Class weights — rare classes ko zyada weight (Logs, Flowers kam hote hain)
# Landscape/Sky dominant hain isliye unhe kam weight
CLASS_WEIGHTS = torch.tensor([
    0.5,   # Background
    2.0,   # Trees
    2.0,   # Lush Bushes
    1.5,   # Dry Grass
    2.0,   # Dry Bushes
    3.0,   # Ground Clutter — rare
    4.0,   # Logs — bahut rare
    2.5,   # Rocks
    0.5,   # Landscape — dominant, kam weight
    0.8,   # Sky — dominant, kam weight
], dtype=torch.float32)

# ============================================================================
# FIX 1: Mask Conversion — NEAREST interpolation se resize
# ============================================================================

def convert_mask_numpy(mask_path):
    """
    Load mask aur convert karo class IDs mein.
    NEAREST interpolation use karo — bilinear galat values deta hai masks ke liye.
    """
    mask = np.array(Image.open(mask_path))

    # Agar 3-channel hai toh single channel lo
    if len(mask.shape) == 3:
        mask = mask[:, :, 0]

    new_mask = np.zeros_like(mask, dtype=np.uint8)
    for raw_val, new_val in value_map.items():
        new_mask[mask == raw_val] = new_val

    # Resize with NEAREST — class IDs preserve hote hain
    new_mask = cv2.resize(new_mask, (IMG_W, IMG_H),
                          interpolation=cv2.INTER_NEAREST)
    return new_mask

# ============================================================================
# FIX 2: Augmentation — Albumentations se proper augmentation
# ============================================================================

train_aug = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(
        brightness_limit=0.2,
        contrast_limit=0.2,
        p=0.4
    ),
    A.HueSaturationValue(
        hue_shift_limit=10,
        sat_shift_limit=20,
        val_shift_limit=10,
        p=0.3
    ),
    A.GaussNoise(var_limit=(5.0, 20.0), p=0.2),
    A.RandomShadow(p=0.2),           # Desert mein shadows common hain
    A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
    ToTensorV2(),
])

val_aug = A.Compose([
    A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
    ToTensorV2(),
])

# ============================================================================
# Dataset
# ============================================================================

class MaskDataset(Dataset):
    def __init__(self, data_dir, augment=None):
        self.image_dir = os.path.join(data_dir, 'Color_Images')
        self.masks_dir = os.path.join(data_dir, 'Segmentation')
        self.augment   = augment
        self.data_ids  = sorted(os.listdir(self.image_dir))

    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, idx):
        data_id  = self.data_ids[idx]
        img_path = os.path.join(self.image_dir, data_id)

        # Mask same naam — extension match karo
        base = os.path.splitext(data_id)[0]
        mask_path = None
        for ext in ['.png', '.jpg', '.jpeg']:
            candidate = os.path.join(self.masks_dir, base + ext)
            if os.path.exists(candidate):
                mask_path = candidate
                break
        if mask_path is None:
            # Same naam try karo directly
            mask_path = os.path.join(self.masks_dir, data_id)

        # Image load
        image = np.array(
            Image.open(img_path).convert("RGB").resize(
                (IMG_W, IMG_H), Image.BILINEAR
            )
        )

        # Mask load — NEAREST resize (FIX 1)
        mask = convert_mask_numpy(mask_path)

        # Augmentation apply karo
        if self.augment:
            aug    = self.augment(image=image, mask=mask)
            image  = aug['image']   # (C, H, W) tensor
            mask   = aug['mask']    # (H, W) tensor
        else:
            # Val ke liye sirf normalize
            aug   = val_aug(image=image, mask=mask)
            image = aug['image']
            mask  = aug['mask']

        mask = mask.long()
        return image, mask

# ============================================================================
# Model: Improved Segmentation Head
# ============================================================================

class SegmentationHeadV2(nn.Module):
    """
    Improved ConvNeXt-style head:
    - Deeper network
    - Batch Normalization
    - Skip connections
    """
    def __init__(self, in_channels, out_channels, tokenW, tokenH):
        super().__init__()
        self.H = tokenH
        self.W = tokenW

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
        )

        # Block 1 — depthwise separable
        self.block1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=7, padding=3, groups=256),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Conv2d(256, 256, kernel_size=1),
            nn.GELU(),
        )

        # Block 2
        self.block2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=5, padding=2),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.GELU(),
        )

        # Dropout — overfitting rokne ke liye
        self.dropout = nn.Dropout2d(p=0.1)

        # Final classifier
        self.classifier = nn.Conv2d(128, out_channels, kernel_size=1)

        # Weight initialization
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(B, self.H, self.W, C).permute(0, 3, 1, 2)
        x = self.stem(x)
        x = self.block1(x) + x    # Skip connection
        x = self.block2(x)
        x = self.dropout(x)
        return self.classifier(x)

# ============================================================================
# FIX 3: Combined Loss — CrossEntropy + Dice
# ============================================================================

class CombinedLoss(nn.Module):
    """
    CrossEntropy + Dice Loss:
    - CrossEntropy: overall classification
    - Dice Loss: rare class performance improve karta hai
    - Class weights: imbalanced classes handle karta hai
    """
    def __init__(self, weights=None, dice_weight=0.5, ce_weight=0.5):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(
            weight=weights,
            ignore_index=255
        )
        self.dice_w = dice_weight
        self.ce_w   = ce_weight

    def dice_loss(self, pred, target, smooth=1e-6):
        pred   = torch.softmax(pred, dim=1)
        loss   = 0.0
        for c in range(pred.shape[1]):
            p          = pred[:, c].reshape(-1)
            t          = (target == c).float().reshape(-1)
            intersect  = (p * t).sum()
            loss      += 1 - (2 * intersect + smooth) / \
                         (p.sum() + t.sum() + smooth)
        return loss / pred.shape[1]

    def forward(self, pred, target):
        ce_loss   = self.ce(pred, target)
        dice_loss = self.dice_loss(pred, target)
        return self.ce_w * ce_loss + self.dice_w * dice_loss

# ============================================================================
# Metrics
# ============================================================================

def compute_iou(pred, target, num_classes=10):
    pred   = torch.argmax(pred, dim=1).view(-1)
    target = target.view(-1)
    ious   = []
    for c in range(num_classes):
        inter = ((pred == c) & (target == c)).sum().float()
        union = ((pred == c) | (target == c)).sum().float()
        if union > 0:
            ious.append((inter / union).item())
    return float(np.mean(ious)) if ious else 0.0


def compute_pixel_accuracy(pred, target):
    pred_cls = torch.argmax(pred, dim=1)
    return (pred_cls == target).float().mean().item()


def evaluate_full(model, backbone, loader, device, num_classes=10):
    model.eval()
    iou_scores, px_accs = [], []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            feats   = backbone.forward_features(imgs)["x_norm_patchtokens"]
            logits  = model(feats)
            outputs = F.interpolate(logits, size=imgs.shape[2:],
                                    mode="bilinear", align_corners=False)
            iou_scores.append(compute_iou(outputs, labels, num_classes))
            px_accs.append(compute_pixel_accuracy(outputs, labels))

    model.train()
    return float(np.mean(iou_scores)), float(np.mean(px_accs))


def per_class_iou(model, backbone, loader, device, num_classes=10):
    """Per class IoU compute karo — report ke liye"""
    model.eval()
    all_inter = np.zeros(num_classes)
    all_union = np.zeros(num_classes)

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            feats   = backbone.forward_features(imgs)["x_norm_patchtokens"]
            logits  = model(feats)
            outputs = F.interpolate(logits, size=imgs.shape[2:],
                                    mode="bilinear", align_corners=False)
            pred    = torch.argmax(outputs, dim=1).view(-1).cpu().numpy()
            tgt     = labels.view(-1).cpu().numpy()

            for c in range(num_classes):
                inter = ((pred == c) & (tgt == c)).sum()
                union = ((pred == c) | (tgt == c)).sum()
                all_inter[c] += inter
                all_union[c] += union

    model.train()
    ious = np.where(all_union > 0, all_inter / all_union, np.nan)
    return ious

# ============================================================================
# Plotting
# ============================================================================

def save_plots(history, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0,0].plot(history['train_loss'], label='Train', color='#e74c3c')
    axes[0,0].plot(history['val_loss'],   label='Val',   color='#3498db')
    axes[0,0].set_title('Loss'); axes[0,0].set_xlabel('Epoch')
    axes[0,0].legend(); axes[0,0].grid(True, alpha=0.3)

    axes[0,1].plot(history['train_iou'], label='Train IoU', color='#2ecc71')
    axes[0,1].plot(history['val_iou'],   label='Val IoU',   color='#f39c12')
    axes[0,1].set_title('IoU Score'); axes[0,1].set_xlabel('Epoch')
    axes[0,1].legend(); axes[0,1].grid(True, alpha=0.3)

    axes[1,0].plot(history['train_acc'], label='Train Acc', color='#9b59b6')
    axes[1,0].plot(history['val_acc'],   label='Val Acc',   color='#1abc9c')
    axes[1,0].set_title('Pixel Accuracy'); axes[1,0].set_xlabel('Epoch')
    axes[1,0].legend(); axes[1,0].grid(True, alpha=0.3)

    axes[1,1].plot(history['lr'], color='#e67e22')
    axes[1,1].set_title('Learning Rate'); axes[1,1].set_xlabel('Epoch')
    axes[1,1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=150)
    plt.close()
    print(f"Plots saved: {output_dir}/training_curves.png")


def save_per_class_chart(ious, class_names, output_dir):
    valid = [(n, v) for n, v in zip(class_names, ious) if not np.isnan(v)]
    names = [x[0] for x in valid]
    vals  = [x[1] for x in valid]

    colors = ['#2ecc71' if v >= 0.5 else '#f39c12' if v >= 0.2
              else '#e74c3c' for v in vals]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(names, vals, color=colors)
    ax.axhline(y=np.mean(vals), color='red', linestyle='--',
               label=f'Mean IoU: {np.mean(vals):.4f}')
    ax.set_ylim(0, 1)
    ax.set_title('Per-Class IoU Score')
    ax.set_ylabel('IoU')
    plt.xticks(rotation=45, ha='right')

    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'per_class_iou.png'), dpi=150)
    plt.close()
    print(f"Per-class chart saved.")


def save_history_txt(history, per_class_ious, output_dir):
    path = os.path.join(output_dir, 'evaluation_metrics.txt')
    with open(path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("TRAINING RESULTS\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Best Val IoU   : {max(history['val_iou']):.4f}"
                f" (Epoch {np.argmax(history['val_iou'])+1})\n")
        f.write(f"Final Val IoU  : {history['val_iou'][-1]:.4f}\n")
        f.write(f"Final Val Acc  : {history['val_acc'][-1]:.4f}\n")
        f.write(f"Final Val Loss : {history['val_loss'][-1]:.4f}\n\n")

        f.write("Per-Class IoU:\n")
        f.write("-" * 20 + "\n")
        for name, iou in zip(CLASS_NAMES, per_class_ious):
            val = f"{iou:.4f}" if not np.isnan(iou) else "N/A"
            f.write(f"  {name:<20}: {val}\n")

        f.write("\nPer-Epoch History:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Epoch':>6} {'TrLoss':>10} {'VaLoss':>10}"
                f" {'TrIoU':>8} {'VaIoU':>8} {'TrAcc':>8} {'VaAcc':>8}\n")
        f.write("-" * 80 + "\n")
        for i in range(len(history['train_loss'])):
            f.write(
                f"{i+1:>6} "
                f"{history['train_loss'][i]:>10.4f} "
                f"{history['val_loss'][i]:>10.4f} "
                f"{history['train_iou'][i]:>8.4f} "
                f"{history['val_iou'][i]:>8.4f} "
                f"{history['train_acc'][i]:>8.4f} "
                f"{history['val_acc'][i]:>8.4f}\n"
            )
    print(f"Metrics saved: {path}")

# ============================================================================
# Main Training
# ============================================================================

def main():
    print("=" * 60)
    print(f"Device     : {DEVICE}")
    print(f"Epochs     : {N_EPOCHS}")
    print(f"Batch Size : {BATCH_SIZE}")
    print(f"Image Size : {IMG_W}x{IMG_H}")
    print("=" * 60)

    # Datasets
    trainset = MaskDataset(DATA_DIR, augment=train_aug)
    valset   = MaskDataset(VAL_DIR,  augment=None)

    train_loader = DataLoader(trainset, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=0,
                              pin_memory=(DEVICE.type == 'cuda'))
    val_loader   = DataLoader(valset,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0,
                              pin_memory=(DEVICE.type == 'cuda'))

    print(f"Train: {len(trainset)} | Val: {len(valset)}")

    # DINOv2 Backbone load
    print("\nLoading DINOv2 backbone...")
    backbone_archs = {
        "small": "dinov2_vits14",
        "base":  "dinov2_vitb14",
        "large": "dinov2_vitl14",
    }
    backbone = torch.hub.load(
        "facebookresearch/dinov2",
        backbone_archs[BACKBONE]
    )
    backbone.eval()
    backbone.to(DEVICE)
    print("Backbone ready!")

    # Get embedding dim
    with torch.no_grad():
        dummy = torch.zeros(1, 3, IMG_H, IMG_W).to(DEVICE)
        out   = backbone.forward_features(dummy)["x_norm_patchtokens"]
        n_emb = out.shape[2]
        tH    = IMG_H // 14
        tW    = IMG_W // 14
    print(f"Embedding dim: {n_emb} | Token grid: {tH}x{tW}")

    # Model
    model = SegmentationHeadV2(
        in_channels=n_emb,
        out_channels=n_classes,
        tokenW=tW,
        tokenH=tH,
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {total_params:,}")

    # Loss — Combined CE + Dice with class weights
    criterion = CombinedLoss(
        weights=CLASS_WEIGHTS.to(DEVICE),
        dice_weight=0.5,
        ce_weight=0.5
    )

    # Optimizer — AdamW (SGD se better for this task)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=1e-4,
        betas=(0.9, 0.999)
    )

    # LR Scheduler — Cosine Annealing
    # Warmup 3 epochs + Cosine decay
    def lr_lambda(epoch):
        warmup = 3
        if epoch < warmup:
            return (epoch + 1) / warmup
        progress = (epoch - warmup) / (N_EPOCHS - warmup)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # History tracking
    history = {
        'train_loss': [], 'val_loss': [],
        'train_iou':  [], 'val_iou':  [],
        'train_acc':  [], 'val_acc':  [],
        'lr': []
    }

    best_iou    = 0.0
    best_epoch  = 0
    patience    = 10   # Early stopping — 10 epochs mein improve nahi hua toh stop
    no_improve  = 0

    print("\nStarting training...")
    print("=" * 60)

    for epoch in range(N_EPOCHS):
        current_lr = optimizer.param_groups[0]['lr']

        # ── Train ────────────────────────────────────────────────
        model.train()
        train_losses = []

        pbar = tqdm(train_loader,
                    desc=f"Epoch {epoch+1:2d}/{N_EPOCHS} [Train]",
                    leave=False)
        for imgs, labels in pbar:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            # DINOv2 features — no grad (backbone frozen)
            with torch.no_grad():
                feats = backbone.forward_features(imgs)["x_norm_patchtokens"]

            logits  = model(feats)
            outputs = F.interpolate(logits, size=imgs.shape[2:],
                                    mode="bilinear", align_corners=False)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping — training stable karta hai
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            train_losses.append(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        # ── Validation ───────────────────────────────────────────
        model.eval()
        val_losses = []

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                feats   = backbone.forward_features(imgs)["x_norm_patchtokens"]
                logits  = model(feats)
                outputs = F.interpolate(logits, size=imgs.shape[2:],
                                        mode="bilinear", align_corners=False)
                val_losses.append(criterion(outputs, labels).item())

        # ── Metrics ──────────────────────────────────────────────
        tr_iou, tr_acc = evaluate_full(model, backbone,
                                        train_loader, DEVICE, n_classes)
        va_iou, va_acc = evaluate_full(model, backbone,
                                        val_loader,   DEVICE, n_classes)

        t_loss = float(np.mean(train_losses))
        v_loss = float(np.mean(val_losses))

        history['train_loss'].append(t_loss)
        history['val_loss'].append(v_loss)
        history['train_iou'].append(tr_iou)
        history['val_iou'].append(va_iou)
        history['train_acc'].append(tr_acc)
        history['val_acc'].append(va_acc)
        history['lr'].append(current_lr)

        scheduler.step()

        print(f"Epoch {epoch+1:2d}/{N_EPOCHS} | "
              f"TrLoss: {t_loss:.4f} | VaLoss: {v_loss:.4f} | "
              f"TrIoU: {tr_iou:.4f} | VaIoU: {va_iou:.4f} | "
              f"VaAcc: {va_acc:.4f} | LR: {current_lr:.2e}")

        # ── Best Model Save ───────────────────────────────────────
        if va_iou > best_iou:
            best_iou   = va_iou
            best_epoch = epoch + 1
            no_improve = 0
            torch.save({
                'epoch':       epoch + 1,
                'model_state': model.state_dict(),
                'iou':         best_iou,
                'tH':          tH,
                'tW':          tW,
                'n_emb':       n_emb,
            }, MODEL_PATH)
            print(f"  >>> BEST MODEL SAVED! IoU = {best_iou:.4f} "
                  f"(Epoch {best_epoch})")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"\nEarly stopping at epoch {epoch+1} "
                      f"(no improvement for {patience} epochs)")
                break

    # ── Final Results ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print(f"Best Val IoU : {best_iou:.4f} (Epoch {best_epoch})")
    print("=" * 60)

    # Per-class IoU
    print("\nComputing per-class IoU...")
    pc_ious = per_class_iou(model, backbone, val_loader, DEVICE, n_classes)

    print("\nPer-Class IoU:")
    print("-" * 20)
    for name, iou in zip(CLASS_NAMES, pc_ious):
        bar = "█" * int((iou if not np.isnan(iou) else 0) * 20)
        val = f"{iou:.4f}" if not np.isnan(iou) else "  N/A"
        print(f"  {name:<20}: {val}  {bar}")

    # Save everything
    save_plots(history, OUTPUT_DIR)
    save_per_class_chart(pc_ious, CLASS_NAMES, OUTPUT_DIR)
    save_history_txt(history, pc_ious, OUTPUT_DIR)

    print(f"\nAll outputs saved to: {OUTPUT_DIR}")
    print(f"Model saved to      : {MODEL_PATH}")


if __name__ == "__main__":
    main()