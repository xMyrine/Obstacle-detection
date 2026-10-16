# Visual Object Recognition with Unknown Input Detection

Bachelor's thesis, Czech Technical University in Prague (CTU FEE, Department of Cybernetics, Open Informatics - Artificial Intelligence and Computer Science).
Supervisor: Mgr. Ondřej Drbohlav, Ph.D.

*Czech title: Rozpoznávání vizuálních objektů s detekcí neznámých vstupů*

This repository contains the research notebooks, the LaTeX thesis and an interactive demo for **Out-of-Distribution (OOD) detection** with Meta's self-supervised **DINOv3** Vision Transformer. A recognition system that cannot say "I don't know" is unsafe in the real world (an obstacle detector that confidently mislabels an unseen object is worse than one that abstains), so this work asks whether a frozen foundation model can reliably separate known from unknown inputs.

---

## Key Idea

Instead of using one global embedding per image, we use DINOv3's **patch-level** features:

1. Extract patch tokens from a **frozen** DINOv3 backbone (no fine-tuning).
2. Cluster patches with K-Means / FAISS into a **visual codebook** (Bag of Visual Words, K = 1024 in the demo).
3. Represent each image as a normalized **histogram of visual words**.
4. Classify with **K-NN over histograms** using the **Chi-Square distance** (optionally TF-IDF weighted).
5. Flag an image as **OOD** with a **dual-condition heuristic**: it is rejected only if *both* hold
   - the Chi-Square distance to its nearest neighbour exceeds a threshold `T`, **and**
   - the number of distinct classes among its `K` nearest neighbours exceeds a chaos limit `C` (a cheap proxy for neighbourhood entropy).

```
image -> DINOv3 patch tokens -> nearest codebook word per patch -> histogram
      -> Chi-Square K-NN against ID reference histograms
      -> (dist > T) AND (unique labels in top-K > C) ? OOD : majority-vote class
```

Demo defaults (`demo_showcase/model/*_config.json`): `K_NN = 7`, `T = 0.5`, `C = 3`, `1024` clusters.

Beyond the proposed method, the thesis also analyses the DINOv3 feature space (PCA, t-SNE, layer-wise trajectories, attention maps) and compares against classical post-hoc OOD scores (MSP, Energy, ODIN, Mahalanobis, MC Dropout, Deep Ensembles, GROOD).

---

## Models Used

| Model | Purpose |
| --- | --- |
| `facebook/dinov3-vits16-pretrain-lvd1689m` (ViT-S/16, 384-d) | Lightweight backbone: feature/PCA/t-SNE notebooks and the **real-time demo** (12 layers) |
| `facebook/dinov3-vitl16-pretrain-lvd1689m` (ViT-L/16, 1024-d) | Mid-size backbone for the patch-clustering, K-NN and OOD benchmark notebooks |
| `facebook/dinov3-vit7b16-pretrain-lvd1689m` (ViT-7B/16, 4096-d) | Flagship backbone for high-resolution attention / layer-wise analysis and OOD score comparisons |
| DINOv2 (ViT-B/14) | Fallback in `m3` for users without access to the gated DINOv3 weights |
| YOLOv8n / YOLOv8s | Object detection baseline (`m1`) |
| RT-DETR-L | Transformer-based object detection baseline (`m1`) |
| MobileNetV3-Small | Lightweight CNN fine-tuned on CIFAR-10 as a post-hoc OOD baseline (`m2`) |
| Logistic Regression | Patch-level foreground/background segmentation on top of DINOv3 features |

---

## Repository Structure

```
.
├── notes/                    # Research notebooks (m0 - m13), shared utils, cached features
│   ├── features/             # Pre-extracted DINOv3 features (.h5)
│   ├── images/               # GIFs / figures produced by the notebooks
│   └── utils/                # HuggingFace dataset wrapper, feature splitter, notebook fixer
├── demo_showcase/            # Interactive OOD demo (webcam + Gradio)
│   ├── demo.py               # OODCameraDetector engine + OpenCV webcam app
│   ├── web.py                # Gradio web interface
│   ├── model_download.ipynb  # Builds the codebook + reference histograms
│   ├── model/                # FAISS codebook, K-NN reference DB, config JSONs
│   └── requirements.txt
├── image_generation/         # Scripts for thesis figures (e.g. CIFAR-100 hierarchy)
├── thesis/                   # LaTeX source (CTU template), figures, bibliography, thesis.pdf
└── README.md
```

---

## Notebooks

| Notebook | Topic |
| --- | --- |
| `m0_concepts` | Personal notes on core concepts: IoU, precision/recall, PR curves, AP/mAP, FPS, quantization, feature vectors, OOD |
| `m1_baseline_interface` | Object detection baselines (YOLOv8, RT-DETR) on webcam, COCO and video |
| `m2_post_hoc_methods` | MobileNetV3 fine-tuned on CIFAR-10; Energy, MSP, MC Dropout, Deep Ensembles vs. SVHN / CIFAR-100 |
| `m3_feature_vectors_representation` | DINOv3 feature extraction, mean images, PCA (2D/3D) |
| `m4_grood_dino_implementation` | GROOD (Linear Probe + Nearest Mean, Neyman-Pearson) on DINOv3; AUROC / FPR95 |
| `m5_dino_tsne_visualization` | t-SNE (2D/3D) of DINOv3 embeddings |
| `m6_dino_attention_block` | Attention-head visualization, including high-resolution images |
| `m7_layer_wise_representation` | Layer-wise `[CLS]` trajectories, cosine similarity, Pearson correlation, PCA "brain scan" GIFs |
| `m8_dino_visualisation` | Enhanced spatial visualization / video generation with the 40-layer flagship model |
| `m9_OOD_pairings_scores` | ID/OOD pairings (Food-101 vs. Pets, Textures, FGVC Aircraft): Energy, Mahalanobis, ODIN |
| `m10_patch_clustering` | Patch clustering (K-Means++), foreground segmentation, visual dictionary, Chi-Square distances |
| `m11_KNN_histogram_classification` | K-NN classification on BoVW histograms; FAISS, TF-IDF, cropped vs. uncropped |
| `m12_KNN_Classification_Tests` | K-NN benchmark on Caltech-101 (Chi-Square, FAISS L2, TF-IDF + cosine) |
| `m13_OOD_KNN_Detection` | OOD detection with K-NN and the dual-condition heuristic; benchmarking |

---

## Datasets

| Dataset | Use |
| --- | --- |
| Caltech-101 | ID set for K-NN classification and the demo |
| Food-101 | ID set for OOD score comparison; feature analysis |
| CIFAR-10 | ID set for the post-hoc / GROOD experiments |
| CIFAR-100 | OOD set; class hierarchy experiments |
| Oxford-IIIT Pet | Hierarchical (species -> breed) analysis; OOD set |
| DTD (Textures), FGVC Aircraft, Places | OOD sets |
| SVHN | OOD set (digit domain shift) |
| Pokémon | OOD set (extreme domain shift) |
| MNIST | Early K-NN histogram classification experiments |

---

## OOD Methods Evaluated

- **MSP**: Maximum Softmax Probability (baseline)
- **Energy score**: energy-based OOD detection
- **ODIN**: temperature scaling + input perturbation
- **Mahalanobis distance** on raw backbone features
- **MC Dropout** and **Deep Ensembles**: uncertainty-based baselines
- **GROOD**: calibrated OOD detection with a generic representation (Linear Probe + Nearest Mean + Neyman-Pearson)
- **Proposed**: BoVW + Chi-Square K-NN with the dual-condition (distance + label chaos) heuristic

---

## Demo

An interactive demo classifies an image (or webcam frame) into one of the Caltech-101 classes, or reports **UNKNOWN (OOD)**. It uses the compact ViT-S/16 backbone so that it runs at interactive speed.

```bash
cd demo_showcase
pip install -r requirements.txt

# DINOv3 is gated: request access on HuggingFace, then log in
huggingface-cli login

python demo.py   # OpenCV webcam window (press q to quit)
python web.py    # Gradio web interface
```

Scripts must be run from inside `demo_showcase/` because model assets are loaded with relative paths. If you want to rebuild the codebook and reference histograms, run `model_download.ipynb`.

---

## Setup (notebooks)

**Requirements:** Python 3.10+, PyTorch, Transformers (HuggingFace), scikit-learn, faiss, matplotlib, h5py.

```bash
pip install torch torchvision transformers scikit-learn faiss-cpu matplotlib h5py tqdm kagglehub ultralytics
```

A HuggingFace account with approved access is required for the gated DINOv3 weights. Provide the token through the environment, never commit it:

```bash
export HF_TOKEN=your_token_here
```

The ViT-7B model needs a large GPU; the notebooks that use it are meant for a GPU machine, while ViT-S/16 runs on a laptop.

---

## Thesis

The written thesis lives in `thesis/` (CTU template `ctuthesis.cls`, main file `thesis.tex`, compiled `thesis.pdf`). It needs `minted`, so compile with shell escape:

```bash
cd thesis
latexmk -pdf -shell-escape thesis.tex
```

Structure: Introduction, Related Work, Feature Representation and Analysis, Downstream Applications and Benchmarking, Proposed Dual-Condition OOD Detection, Real-World Demonstration, Conclusion.

Bibliography is split into `bib_ctu.bib` (template references) and `bib_research.bib` (research papers).

---

## Findings in Brief

- Frozen DINOv3 features form tight, object-centric clusters across diverse datasets without any fine-tuning.
- Distance-based scores struggle with **semantic overlap** (visually related ID/OOD classes) and with texture-biased cases such as SVHN.
- Patch-level BoVW histograms with the dual-condition rule give an interpretable, training-free OOD detector.
- Related classes share visual words (e.g. mammals, or scissors and swords), which motivates **hierarchical OOD detection** as future work.

---

## License

See [LICENSE](LICENSE).
