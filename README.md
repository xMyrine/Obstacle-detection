# Obstacle Detection & Out-of-Distribution Research

A Bachelor's thesis research project exploring **Out-of-Distribution (OOD) detection** for obstacle detection systems, leveraging Meta's **DINOv3** vision transformer as a feature backbone alongside classical object detection models.

---

## Overview

This repository contains research notebooks, utilities, and thesis materials investigating whether modern self-supervised vision models can reliably distinguish between known (in-distribution) and unknown (out-of-distribution) visual inputs — a critical property for safe obstacle detection in real-world environments.

The core research pipeline:

1. Extract rich feature representations using **DINOv3** (ViT-S/16)
2. Apply and benchmark multiple **post-hoc OOD scoring methods** (ODIN, Energy-based, Calibrated, etc.)
3. Visualize the feature space with **PCA**, **t-SNE**, and **attention maps**
4. Evaluate ID/OOD separability across class pairings

---

## Repository Structure

```
bachelors/
├── notes/                          # Research & experiment notebooks
│   ├── m0_concepts.ipynb           # Foundational concepts (IoU, PR curves, etc.)
│   ├── m1_baseline_interface.ipynb # Baseline object detection interface (YOLOv8, RT-DETR)
│   ├── m2_post_hoc_methods.ipynb   # MobileNet fine-tuning + OOD detection methods
│   ├── m3_feature_vectors_representation.ipynb  # DINOv3 feature extraction + PCA
│   ├── m4_grood_dino_implementation.ipynb       # GROOD method with DINOv3
│   ├── m5_dino_tsne_visualization.ipynb         # t-SNE dimensionality reduction
│   ├── m6_dino_attention_block.ipynb            # Attention head visualization
│   ├── m7_layer_wise_representation.ipynb       # Layer-wise [CLS] token trajectory
│   ├── m8_dino_visualisation.ipynb              # Enhanced DINO feature visualization
│   ├── m9_OOD_pairings_scores.ipynb            # OOD scoring comparisons (ODIN, Energy, ...)
│   ├── m10_patch_clustering.ipynb              # Patch-level clustering & Chi-Square distances
│   ├── data/                       # Raw datasets (CIFAR-10, etc.)
│   ├── features/                   # Pre-extracted feature files (.h5)
│   │   ├── cifar10.h5
│   │   ├── cifar100.h5
│   │   ├── food101.h5
│   │   ├── pokemon.h5
│   │   └── svhn.h5
│   └── utils/
│       ├── hugging_face_util.py    # HuggingFace model loading helpers
│       └── notebook_fixer.py      # Notebook utility helpers
├── image_generation/
│   └── cifar100_hiearchy.py        # CIFAR-100 class hierarchy visualization
├── prototype/                      # Prototype implementation (in progress)
├── thesis/                         # LaTeX thesis source (CTU template)
│   ├── thesis.tex
│   └── bibliography/
└── README.md
```

---

## Notebooks

| Notebook                            | Topic                                                                                |
| ----------------------------------- | ------------------------------------------------------------------------------------ |
| `m0_concepts`                       | Core metrics: IoU, Precision, Recall, PR curves, Average Precision                   |
| `m1_baseline_interface`             | Object detection baseline using **YOLOv8** and **RT-DETR**                           |
| `m2_post_hoc_methods`               | Transfer learning with **MobileNetV3** fine-tuned on CIFAR-10; post-hoc OOD methods  |
| `m3_feature_vectors_representation` | DINOv3 features + **PCA** class separability analysis                                |
| `m4_grood_dino_implementation`      | **GROOD** algorithm implemented on top of DINOv3                                     |
| `m5_dino_tsne_visualization`        | **t-SNE** visualization of DINOv3 embeddings                                         |
| `m6_dino_attention_block`           | Patch-level attention visualization across attention heads                           |
| `m7_layer_wise_representation`      | Layer-wise [CLS] token trajectory for class pairs (e.g. Cat vs Dog)                  |
| `m8_dino_visualisation`             | Enhanced spatial feature visualization with DINOv3                                   |
| `m9_OOD_pairings_scores`            | Comparative evaluation of **ODIN**, **Energy scoring**, **Calibrated OOD** detection |
| `m10_patch_clustering`              | Patch clustering (K-Means++) + Chi-Square distribution distances                     |

---

## Models Used

| Model                                      | Purpose                                        |
| ------------------------------------------ | ---------------------------------------------- |
| `facebook/dinov3-vits16-pretrain-lvd1689m` | Primary feature backbone (self-supervised ViT) |
| YOLOv8n / YOLOv8s                          | Object detection baseline                      |
| RT-DETR-L                                  | Transformer-based object detection baseline    |
| MobileNetV3-Small                          | Lightweight CNN fine-tuned for OOD baseline    |

---

## Datasets

| Dataset   | Use                                             |
| --------- | ----------------------------------------------- |
| CIFAR-10  | Primary in-distribution training/evaluation set |
| CIFAR-100 | OOD evaluation + hierarchy experiments          |
| Food-101  | OOD evaluation                                  |
| SVHN      | OOD evaluation (digit domain shift)             |
| Pokémon   | OOD evaluation (extreme domain shift)           |

---

## OOD Methods Evaluated

- **MSP** — Maximum Softmax Probability (baseline)
- **ODIN** — Out-of-DIstribution detector for Neural networks (temperature scaling + input perturbation)
- **Energy Score** — Energy-based OOD detection
- **Calibrated OOD Detection** — Confidence calibration approach
- **GROOD** — Graph-based representation OOD detection using DINOv3 features

---

## Setup

**Requirements:** Python 3.10+, PyTorch, Transformers (HuggingFace), scikit-learn, matplotlib, h5py

```bash
pip install torch torchvision transformers scikit-learn matplotlib h5py tqdm kagglehub
```

A HuggingFace account token is required to access the gated DINOv3 model:

```bash
export HF_TOKEN=your_token_here
```

---

## Thesis

The written thesis is compiled from LaTeX sources in the `thesis/` directory using the CTU thesis template (`ctuthesis.cls`). The bibliography is split into:

- `bib_ctu.bib` — CTU-related references
- `bib_research.bib` — Research paper references
