<p align="center">
  <img src="./.github/readme-assets/signal.gif" alt="Animated signal / product visual for -LETVLER-yolo-hackathon" width="100%" />
</p>

<h1 align="center">-LETVLER-yolo-hackathon</h1>

<p align="center"><strong>Python scripts and utilities for offroad semantic segmentation (training, evaluation/visualization, and mask colorization). The repository contains several domain-specific utilities and artifacts but several critical runtime pieces are incomplete or missing, so it is not immediately runnable.</strong></p>

<p align="center"><code>REPO//SIGNAL</code> · <code>SIGNAL / PRODUCT</code> · <code>LOOPING README EXPERIENCE</code></p>

## Live signal

| Lens | Readout |
| --- | --- |
| Portfolio lane | **SIGNAL / PRODUCT** |
| Code surface | **17** tracked files observed |
| Primary materials | **Python, Markdown** |
| Verification | **1** test-related files observed |

> A moving scan of the project surface. The animated frame above is a lightweight visual signature; the sections below remain the source of truth for implementation details.

## Motion map

`SIGNAL` → `SHAPE` → `RELEASE`

Use the animated banner as the first signal, then move into the implementation dossier. The recommended next step is to verify the documented setup command against the repository scripts before extending the project.

<details open>
<summary><strong>Open the full project dossier</strong></summary>

## Overview
A small collection of Python scripts focused on semantic segmentation for an Offroad_Segmentation dataset. The codebase includes training/validation scripts, mask conversion and visualization utilities, and some saved artifacts (images, a checkpoint file). Several important functions and the model definition are incomplete or absent.

## What it does
- Provides a training script (train_segmentation.py) intended to train a segmentation head.
- Provides an evaluation/visualization script (test_segmentation.py) intended to run inference and save visualizations.
- Provides a standalone mask colorizer (visualize.py) that produces colorized PNGs from mask images.
- Includes utilities and constants for mapping raw mask values to class IDs, class names, and class weights.

## Key capabilities
- Explicit value-to-class-ID mapping (value_map) and fixed n_classes/class_names.
- Class weighting to address imbalance (CLASS_WEIGHTS).
- Utilities to convert mask data to color RGB images for visualization.
- Non-interactive matplotlib backend usage to support headless runs.
- Output path constants and a referenced model checkpoint file (segmentation_head_v2.pth).

## Technology
Observed/expected technologies referenced in the code and comments:
- Python 3.x
- PyTorch (torch, torchvision)
- NumPy
- OpenCV (cv2)
- Pillow (PIL)
- Albumentations (albumentations, albumentations.pytorch)
- matplotlib
- tqdm

## Repository structure
Top-level files observed:
- README.md (this file)
- train_segmentation.py — training script (incomplete pieces present)
- test_segmentation.py — evaluation / visualization script (contains bugs)
- visualize.py — standalone mask colorizer script
- segmentation_head_v2.pth — referenced model checkpoint (no loader/architecture included)
- req.txt — (dependency list file present; contents not shown here)
- evaluation_metrics.txt, training_curves.png, all_metrics_curves.png, iou_curves.png, dice_curves.png, per_class_iou.png, sample_0_comparison.png ... sample_4_comparison.png — saved figures / artifacts

Note: There is no explicit package layout (no src/), no separate model definition file, and no CI/test manifests beyond the single test file mentioned in audit data.

## Getting started
There are no runnable setup or usage instructions included in the repository. The code is not immediately runnable because of missing/incomplete pieces (see Development and quality notes). To inspect and begin working with the repository, contributors should:

- Open req.txt to review declared dependencies.
- Read train_segmentation.py, test_segmentation.py, and visualize.py to locate configuration constants (e.g., DATA_DIR, VAL_DIR, OUTPUT_DIR, MODEL_PATH) and to identify incomplete functions or obvious runtime errors.
- Examine segmentation_head_v2.pth to confirm a checkpoint artifact exists.
- Inspect sample images and saved metric/curve images to understand expected outputs and visualizations.

This inspection will let contributors determine what is needed to make the code runnable (data layout, missing model code, and missing dataset/loader implementations).

## Configuration
Observed expectations and conventions:
- Expected dataset layout (described by scripts/comments): a local dataset under Offroad_Segmentation_Training_Dataset/{train,val} with subfolders such as Color_Images and Segmentation for images and masks.
- Constants in scripts reference: DATA_DIR, VAL_DIR, IMG_W, IMG_H, OUTPUT_DIR, MODEL_PATH.
- Mask handling: code assumes nearest-neighbor resizing to preserve discrete class IDs.

No CLI flags, config file format, or environment specification is present beyond the visible constants and req.txt; contributors should open the scripts to confirm current hardcoded paths and values.

## Development and quality notes
Known functional gaps and defects (observed in code and audit):
- MaskDataset.__len__ in test_segmentation.py returns nothing (runtime error). It should return an integer length.
- convert_mask_numpy in train_segmentation.py is truncated and incomplete (the resize call is unfinished).
- No model architecture or code is present to build/load the described DINOv2 backbone + ConvNeXt head; MODEL_PATH is referenced but no loading code that matches it is present.
- No end-to-end training/validation loops, metric computations, or checkpoint save/load routines are fully implemented in the visible scripts.
- Data loading path and dataset __getitem__ implementations are missing or incomplete; hardcoded dataset paths exist.
- No unit tests or CI configuration are present beyond static artifacts.

Recommended immediate fixes to get code runnable (non-exhaustive, observed):
- Fix MaskDataset.__len__ to return len(self.data_ids).
- Complete convert_mask_numpy to perform the cv2.resize(..., interpolation=cv2.INTER_NEAREST) step.
- Implement MaskDataset.__getitem__ or confirm existing implementation to yield (image, mask) tensors with proper transforms/normalization.
- Add a model builder or loader that matches segmentation_head_v2.pth or include instructions to obtain matching pretrained weights.
- Move hardcoded paths to a configurable interface (CLI or config file) and validate input/output paths before writing.

## Safety and responsible use
- Scripts read and write files directly from configured paths and may overwrite output files if run without care; validate output paths before running any code.
- visualize.py and other scripts use cv2.imread and write images without robust input validation; processing untrusted/malformed files can cause crashes or denial-of-service. Add input validation and safe write policies before processing untrusted data.
- No secrets or credentials were observed in the repository.

## Contributing
Contributions that help make the repository reproducible and runnable are welcome. Useful contribution areas:
- Fixing runtime bugs (MaskDataset.__len__, convert_mask_numpy) and implementing dataset __getitem__.
- Adding or integrating a model definition and checkpoint load/save that matches segmentation_head_v2.pth.
- Adding a clear requirements or environment specification (if req.txt is incomplete) and a minimal README with reproducible quickstart steps.
- Adding simple unit tests for mask conversion utilities and data loaders.

To begin contributing, review the scripts and artifacts listed in Getting started, open issues describing fixes or improvements, and submit pull requests with focused changes. (Repository does not include contribution templates; follow standard GitHub issue/PR flow.)

(There is no license file present in the repository metadata; no license statement is included here.)

</details>

---

<p align="center"><sub>README motion system · visual layer by RepoSignal · implementation details remain project-specific</sub></p>
