# Method: BoVW + Dual-Condition OOD Detection

## Pipeline
1. **Patch features.** Run an image through frozen DINOv3 and keep the patch tokens (drop `[CLS]` and the register tokens).
2. **Visual codebook.** Cluster patch tokens from the ID training set with K-Means / FAISS. Each centroid is a "visual word". The demo uses K = 1024.
3. **Histogram.** Assign each patch to its nearest word and count. Normalize to sum 1.
4. **Chi-Square K-NN.** Compare the histogram to all ID reference histograms with
   `chi2(a, b) = 0.5 * sum((a - b)^2 / (a + b + eps))`.
5. **Decision.** Take the K nearest neighbours.

## The dual-condition rule
An image is **OOD** if and only if

- `T`: distance to the closest neighbour > threshold, **and**
- `C`: number of distinct classes among the K neighbours > chaos limit.

Otherwise the prediction is the majority vote of the neighbours.

Why both? Distance alone breaks down because the ID space has uneven density. Label chaos (count of distinct classes, a proxy for neighbourhood entropy) catches queries that land between classes. Requiring both keeps false alarms low.

Default parameters: `K_NN = 7`, `T = 0.5`, `C = 3`. Scaling `C` in proportion to K keeps false alarms low; see [Results](Results.md).

## Variants explored (m10-m13)
- **Foreground segmentation:** a logistic regression on patch features masks the background. Slightly lowered accuracy because background carries context.
- **Filtered images:** background removed without zeroing pixels, comparable to cropping.
- **TF-IDF + cosine:** worked for classification but **failed for OOD**; the distributions stayed entangled.
- **FAISS L2 on Hellinger-transformed histograms:** a fast approximation of the Chi-Square comparison.
- **Codebook size K and neighbours k:** accuracy peaked around K = 1024 and k = 9, with no gain beyond.

## Known limitation
The method needs enough patches per image. On 32x32 inputs (CIFAR) a /16 ViT yields only 4 tokens, too few for a meaningful histogram.
