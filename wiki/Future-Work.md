# Future Work

## Hierarchical OOD detection (main direction)
Related classes share visual words: mammals share many clusters, and scissors and swords overlap strongly. Unknown inputs therefore land in a *related* region instead of random noise.

Idea: build a coarse-to-fine classifier that backs off to a super-category ("animal", "sharp tool") when fine-grained identification fails, instead of only answering "known" or "unknown".

Possible steps:
1. Measure codebook overlap between classes and derive a tree from it.
2. Use the CIFAR-100 and Oxford-IIIT Pet hierarchies in `image_generation/` as ground truth.
3. Replace the single chaos count with a per-level consistency check.

## Other ideas
- Adaptive thresholds `T` and `C` per class or per density.
- Smaller patch size / models for low-resolution input.
- Distilled or quantized backbones for edge devices.
- Approximate K-NN and prototype compression to cut memory.
- Tests on a real obstacle-detection dataset, tying back to the safety motivation.
