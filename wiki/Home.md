# Welcome

Wiki for the bachelor's thesis **Visual Object Recognition with Unknown Input Detection** (CTU Prague, FEE). It explains how the code and experiments fit together; the full theory is in `thesis/thesis.pdf`.

**The problem:** a recognition system should say "I don't know" when it sees something outside its training data (Out-of-Distribution, OOD).

**The idea:** use patch features from a frozen DINOv3 Vision Transformer, turn each image into a Bag-of-Visual-Words histogram, and reject an image only when it is *both* far from known data *and* surrounded by neighbours of many different classes.

## Pages

| Page | What is in it |
| --- | --- |
| [Getting Started](Getting-Started.md) | Install, HuggingFace access, hardware |
| [Method](Method.md) | BoVW pipeline and the dual-condition rule |
| [Feature Analysis](Feature-Analysis.md) | PCA, t-SNE, attention, layer-wise analysis |
| [OOD Baselines](OOD-Baselines.md) | MSP, Energy, ODIN, Mahalanobis, GROOD, ensembles |
| [Demo Guide](Demo-Guide.md) | Running and tuning the webcam / Gradio demo |
| [Notebook Guide](Notebook-Guide.md) | What each notebook does and the order to read them |
| [Results](Results.md) | Headline numbers and limitations |
| [Future Work](Future-Work.md) | Hierarchical OOD and other next steps |
