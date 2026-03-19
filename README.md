# 🔬 AnomalyLens — Industrial Defect Detection

A **Convolutional Autoencoder** trained on normal industrial part images to detect anomalies/defects via reconstruction error — built with PyTorch and Streamlit.

---

## 🧠 How It Works

```
Input Image
    ↓
 ENCODER  →  Conv2D × 4 blocks + MaxPool  →  Latent Bottleneck (8×8×256)
    ↓
 DECODER  →  ConvTranspose2D × 4 blocks   →  Reconstructed Image
    ↓
Reconstruction Error (MSE)  →  Anomaly Score
    ↓
Score > Threshold  →  ⚠ ANOMALY DETECTED
Score ≤ Threshold  →  ✅ NORMAL
```

The model is trained **only on normal images** (unsupervised). Defective parts look unfamiliar to the encoder, resulting in high reconstruction error — that's your anomaly signal.

---

## 📦 Installation

```bash
git clone <repo>
cd cae-anomaly-detector
pip install -r requirements.txt
```

---

## 🗂️ Dataset

Download **MVTec AD** (free for research):
👉 https://www.mvtec.com/company/research/datasets/mvtec-ad

Pick any category (e.g. `leather`, `screw`, `tile`) and use its `train/good/` folder.

---

## 🏋️ Training

```bash
python train.py \
  --data_dir ./mvtec/leather/train/good \
  --epochs 100 \
  --batch_size 16 \
  --lr 1e-3 \
  --save_path ./weights/autoencoder.pth
```

| Argument | Default | Description |
|---|---|---|
| `--data_dir` | required | Folder with normal training images |
| `--epochs` | 100 | Training epochs |
| `--batch_size` | 16 | Batch size |
| `--lr` | 0.001 | Learning rate |
| `--image_size` | 128 | Resize target (square) |
| `--save_path` | `./weights/autoencoder.pth` | Where to save weights |

Training will save:
- `./weights/autoencoder.pth` — model weights
- `./weights/loss_curve.png` — training loss plot

---

## 🚀 Running the App

```bash
streamlit run app.py
```

### App Features
| Feature | Description |
|---|---|
| 🖼️ Default Samples | 3 built-in industrial images — zero setup needed |
| 📤 Upload Own Image | Any format: JPG, PNG, BMP, TIFF, WebP |
| ⚙️ Threshold Slider | Adjust anomaly sensitivity live |
| 🖼️ Visual Analysis | Original vs Reconstructed vs Heatmap overlay |
| 📊 Error Gauge | Bar chart showing score vs threshold |
| 🗺️ Raw Error Map | Jet-colormap per-pixel error visualization |
| ⬇️ Download Report | Save full analysis as PNG |

---

## 📁 File Structure

```
cae-anomaly-detector/
├── app.py              ← Streamlit UI entry point
├── model.py            ← ConvAutoencoder (PyTorch)
├── train.py            ← Offline training script
├── utils.py            ← Preprocessing, scoring, heatmap
├── requirements.txt
├── weights/
│   └── autoencoder.pth ← Place your trained weights here
└── README.md
```

---

## 🔑 Key ML Concepts Demonstrated

| Concept | Implementation |
|---|---|
| Unsupervised learning | Trained on normal images only — no labels |
| Dimensionality reduction | Encoder: 128×128×3 → 8×8×256 latent |
| Latent space | Bottleneck at 8×8×256 |
| Reconstruction loss | MSE between input & output |
| Anomaly detection | Threshold on reconstruction error |
| Convolutional features | BatchNorm + ReLU + MaxPool / ConvTranspose |

---

## 🎯 Recommended Threshold Values

| Category | Suggested Threshold |
|---|---|
| Leather | 0.015 – 0.025 |
| Screw | 0.020 – 0.035 |
| Tile | 0.010 – 0.020 |
| Wood | 0.018 – 0.030 |

Tune using the sidebar slider based on your validation set.

---

## 📄 License
MIT — Free for academic and personal use.
