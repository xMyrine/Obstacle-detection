# Results

Figures here are taken from the thesis; see `thesis/thesis.pdf` for context and plots.

## Classification (Caltech-101, K-NN on histograms)
- Best accuracy at codebook K = 1024 and k = 9 neighbours; larger values only add cost.
- Cropping or filtering the foreground slightly lowered accuracy compared to the original images.
- Chi-Square performed at least as well as TF-IDF + cosine.

## Dual-condition OOD, parameter sweep (T = 0.5)

| K-NN | Chaos C | OOD recall | False alarms | ID accuracy |
| --- | --- | --- | --- | --- |
| 3 | 1 | 97.00% | 0.52% | 98.26% |
| 7 | 3 | 97.00% | 0.46% | 98.26% |
| 9 | 5 | 97.00% | 0.29% | 98.09% |
| 15 | 9 | 97.00% | 0.17% | 97.81% |
| 25 | 18 | 90.00% | 0.06% | 97.29% |

## Versus baselines (Caltech-101 as ID)
Average FPR95: proposed **0.84%**, MSP 1.98%, Energy 5.62%. Example rows:

| OOD set | Method | AUROC | FPR95 |
| --- | --- | --- | --- |
| DTD | Proposed | 99.41% | 1.09% |
| DTD | MSP | 99.08% | 2.02% |
| SVHN | Proposed | 99.45% | 0.81% |
| SVHN | MSP | 99.29% | 1.79% |
| Pokémon | Proposed | 99.40% | 0.81% |
| Pokémon | MSP | 99.84% | 0.29% |

MSP wins on Pokémon; the proposed method is more stable across datasets.

## Limitations
- **Low-resolution inputs (CIFAR):** too few patches; the method collapses (average AUROC 72.81%, FPR95 69.52% on CIFAR-100).
- **Cost:** self-attention is O(N^2) in tokens; K-NN needs the reference set in memory.
- **Texture bias / semantic overlap:** distance-based scores falter when ID and OOD share visual components; SVHN is not linearly separable in the features.
- In the Food-101 experiment, the Textures set gives lower AUROC than Pets or Aircraft.
