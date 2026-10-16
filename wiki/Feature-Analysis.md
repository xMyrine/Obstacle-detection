# Feature Analysis

Before building the detector, the notebooks examine what DINOv3 has learned. All of this is done on a **frozen** model.

| Technique | Notebook | Takeaway |
| --- | --- | --- |
| Mean image / PCA (2D, 3D) | `m3` | Classes separate in CIFAR-10 and Food-101; SVHN does not separate linearly (texture bias) |
| t-SNE (2D, 3D) | `m5` | Much tighter, cleaner clusters than PCA |
| Attention heads | `m6` | Pick a patch and see which regions it attends to; high-res images give sharper maps |
| Layer-wise `[CLS]` trajectories | `m7` | How class pairs (cat vs dog, cat vs car) diverge with depth; cosine and Pearson comparisons |
| PCA "brain scan" GIFs | `m7`, `m8` | Layer-by-layer patch PCA, saved in `notes/images/` |
| Patch clustering | `m10` | Clusters correspond to parts (ears, legs, background) and are shared between related animals |
| Hierarchy | `image_generation/` | CIFAR-100 and Oxford-IIIT Pet class hierarchies used to reason about semantic overlap |

## Architecture notes (from m7)
DINOv3 is a ViT with 16x16 patches and 4 register tokens. The flagship 7B model has 40 layers; the demo model has 12 layers.
