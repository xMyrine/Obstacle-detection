# Getting Started

## Requirements
- Python 3.10+
- PyTorch, Transformers, scikit-learn, faiss, matplotlib, h5py, tqdm

```bash
pip install torch torchvision transformers scikit-learn faiss-cpu matplotlib h5py tqdm kagglehub ultralytics
```

## DINOv3 access
The DINOv3 weights are **gated** on HuggingFace. Request access on the model page, create a read token, and expose it as an environment variable. Never hard-code or commit it.

```bash
export HF_TOKEN=your_token_here
```

If you cannot get access, `m3` shows a DINOv2 fallback.

## Which backbone for which hardware

| Backbone | Embedding dim | Hardware |
| --- | --- | --- |
| ViT-S/16 | 384 | Laptop / CPU is workable. Used by the demo. |
| ViT-L/16 | 1024 | Consumer GPU |
| ViT-7B/16 | 4096 | Large GPU. Used for high-res attention and OOD score comparisons. |

## Cached features
`notes/features/*.h5` holds pre-extracted features (CIFAR-10/100, Food-101, SVHN, Pokémon) so several notebooks can skip extraction.

## Suggested first run
1. `notes/m3_feature_vectors_representation.ipynb` (ViT-S, small and fast)
2. `demo_showcase/` (see [Demo Guide](Demo-Guide.md))
