# OOD Baselines

Methods compared against the proposed detector.

| Method | Idea | Where |
| --- | --- | --- |
| **MSP** | Low max softmax probability means OOD | `m2`, benchmarks |
| **Energy** | Negative log-sum-exp of logits; lower energy is more ID | `m2`, `m9` |
| **ODIN** | Temperature scaling + gradient input perturbation (slow, evaluated on subsets) | `m9` |
| **Mahalanobis** | Distance to class Gaussians on raw backbone features | `m9` |
| **MC Dropout** | Dropout at inference as an ensemble | `m2` |
| **Deep Ensembles** | Disagreement between independently trained models | `m2`, benchmarks |
| **GROOD** | Linear probe + nearest-mean scores modelled in 2D, Neyman-Pearson boundary | `m4` |

## Metrics
- **AUROC:** separability over all thresholds
- **AUPR:** precision/recall trade-off
- **FPR95:** fraction of OOD accepted when 95% of ID is accepted (lower is better)

## Setups
- `m2`: MobileNetV3-Small fine-tuned on CIFAR-10; OOD = SVHN, CIFAR-100.
- `m4`: GROOD on DINOv3 features.
- `m9`: Food-101 as ID; OOD = Oxford-IIIT Pet, FGVC Aircraft, DTD.
- `m13`: Caltech-101 as ID; OOD sets include DTD, SVHN, Pokémon and others (see the thesis).
