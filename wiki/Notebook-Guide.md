# Notebook Guide

All notebooks are in `notes/`. They are numbered in the order the research progressed.

## Reading paths

**Just want the method:** `m10` -> `m11` -> `m13`, then `demo_showcase/model_download.ipynb`.

**Understanding DINOv3:** `m3` -> `m5` -> `m6` -> `m7` -> `m8`.

**Comparing against baselines:** `m2` -> `m4` -> `m9`.

## Overview

| Stage | Notebooks | Backbone |
| --- | --- | --- |
| Background | `m0`, `m1` | none / YOLOv8, RT-DETR |
| Classical post-hoc OOD | `m2` | MobileNetV3-Small |
| Feature analysis | `m3`-`m8` | ViT-S/16, ViT-7B/16 |
| Score comparison | `m4`, `m9` | ViT-S/16, ViT-7B/16 |
| BoVW + K-NN | `m10`-`m12` | ViT-S/16, ViT-L/16, ViT-7B/16 |
| OOD detection + benchmark | `m13` | ViT-L/16 |

## Tips
- Notebooks cache features to `.h5` / `.pt`; delete the cache to recompute.
- `notes/utils/hugging_face_util.py` wraps HuggingFace datasets for PyTorch loaders.
- `notes/utils/notebook_fixer.py` is a helper for notebook cleanup.
- ViT-7B notebooks need a big GPU.
