# Demo Guide

`demo_showcase/` contains a real-time classifier with OOD rejection. It uses ViT-S/16 (12 layers) for speed and Caltech-101 as the known set.

## Run
Run from inside `demo_showcase/` (assets are loaded by relative path), after logging in to HuggingFace.

```bash
cd demo_showcase
pip install -r requirements.txt
python demo.py   # webcam window, q to quit
python web.py    # Gradio web UI
```

## Files
| File | Role |
| --- | --- |
| `demo.py` | `OODCameraDetector` (engine) and `WebcamApp` |
| `web.py` | Gradio wrapper that overlays label, distance and chaos |
| `model/faiss_dictionary_small_k1024.npy` | Visual codebook (1024 x 384) |
| `model/knn_reference_small_database.npz` | ID histograms + labels |
| `model/model_small_config.json` | `K_NN`, `DISTANCE_THRESHOLD`, `CHAOS_THRESHOLD` |
| `model_download.ipynb` | Rebuilds the codebook and reference DB |

The files without `small` in the name (`model_config.json`, 4096-d assets) belong to the large backbone.

## Tuning
Edit the config JSON:
- Too many false "UNKNOWN": raise `DISTANCE_THRESHOLD` or `CHAOS_THRESHOLD`.
- Unknown objects accepted as known: lower them.

The overlay shows `Dist` (closest-neighbour Chi-Square) and `Chaos` (unique labels in the K neighbours) so you can see which condition triggers.

## Security notes
- Do not paste your HuggingFace token into `demo.py`; log in with `huggingface-cli login` or use `HF_TOKEN`.
- `*.pem` / `*.key` files are git-ignored. If you need local HTTPS, generate your own `cert.pem` / `key.pem` in `demo_showcase/` and never commit them.
