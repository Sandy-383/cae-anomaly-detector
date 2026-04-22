"""
app.py — Streamlit app for Industrial Anomaly Detection using ConvAutoencoder.

Run with:
    streamlit run app.py

Supports multiple materials: Screw, Wood, Leather, Bottle
"""

import os
import io
import requests
import numpy as np
import streamlit as st
import torch
from PIL import Image

from model import GANomalyNet
from utils import (
    load_image_tensor,
    compute_anomaly_score,
    build_comparison_figure,
    build_error_histogram,
)

# ─── Page Config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AnomalyLens — Industrial Defect Detection",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;600;700&family=Inter:wght@300;400;500&display=swap');

  :root {
    --bg-primary:   #0a0e14;
    --bg-card:      #111820;
    --bg-subtle:    #161d26;
    --accent:       #00e5ff;
    --accent-warn:  #f0a500;
    --accent-ok:    #00e676;
    --accent-bad:   #ff1744;
    --border:       #1e2d3d;
    --text-primary: #cdd9e5;
    --text-muted:   #546e7a;
    --font-mono:    'Share Tech Mono', monospace;
    --font-head:    'Rajdhani', sans-serif;
    --font-body:    'Inter', sans-serif;
  }

  html, body, [class*="css"] {
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
    font-family: var(--font-body) !important;
  }

  section[data-testid="stSidebar"] {
    background: var(--bg-card) !important;
    border-right: 1px solid var(--border) !important;
  }

  .anomaly-header {
    display: flex;
    align-items: center;
    gap: 18px;
    padding: 28px 0 18px 0;
    border-bottom: 1px solid var(--border);
    margin-bottom: 28px;
  }
  .anomaly-header .icon { font-size: 3rem; }
  .anomaly-header h1 {
    font-family: var(--font-head) !important;
    font-size: 2.6rem !important;
    font-weight: 700 !important;
    letter-spacing: 3px !important;
    color: var(--accent) !important;
    margin: 0 !important;
    text-shadow: 0 0 30px rgba(0,229,255,0.3);
  }
  .anomaly-header p {
    font-family: var(--font-mono) !important;
    font-size: 0.78rem !important;
    color: var(--text-muted) !important;
    margin: 4px 0 0 0 !important;
    letter-spacing: 1px;
  }

  .card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 22px 26px;
    margin-bottom: 18px;
  }
  .card-title {
    font-family: var(--font-head) !important;
    font-size: 1.1rem;
    font-weight: 600;
    letter-spacing: 2px;
    color: var(--accent);
    text-transform: uppercase;
    margin-bottom: 14px;
    border-bottom: 1px solid var(--border);
    padding-bottom: 8px;
  }

  /* Material selector cards */
  .material-card {
    background: var(--bg-subtle);
    border: 1.5px solid var(--border);
    border-radius: 8px;
    padding: 16px;
    text-align: center;
    cursor: pointer;
    transition: all 0.2s ease;
  }
  .material-card.active {
    border-color: var(--accent);
    background: rgba(0,229,255,0.08);
    box-shadow: 0 0 18px rgba(0,229,255,0.15);
  }
  .material-card .mat-icon { font-size: 2rem; margin-bottom: 6px; }
  .material-card .mat-name {
    font-family: var(--font-head);
    font-size: 1rem;
    font-weight: 600;
    letter-spacing: 2px;
    color: var(--text-primary);
    text-transform: uppercase;
  }
  .material-card .mat-status {
    font-family: var(--font-mono);
    font-size: 0.68rem;
    margin-top: 4px;
  }
  .mat-loaded  { color: var(--accent-ok); }
  .mat-missing { color: var(--accent-warn); }

  .verdict-normal {
    display: inline-block;
    background: rgba(0,230,118,0.12);
    border: 1.5px solid var(--accent-ok);
    color: var(--accent-ok);
    font-family: var(--font-head);
    font-size: 1.5rem;
    font-weight: 700;
    letter-spacing: 3px;
    padding: 14px 36px;
    border-radius: 6px;
    width: 100%;
    text-align: center;
    text-transform: uppercase;
  }
  .verdict-anomaly {
    display: inline-block;
    background: rgba(255,23,68,0.12);
    border: 1.5px solid var(--accent-bad);
    color: var(--accent-bad);
    font-family: var(--font-head);
    font-size: 1.5rem;
    font-weight: 700;
    letter-spacing: 3px;
    padding: 14px 36px;
    border-radius: 6px;
    width: 100%;
    text-align: center;
    text-transform: uppercase;
  }

  .metric-row { display: flex; gap: 14px; margin: 16px 0; }
  .metric-pill {
    flex: 1;
    background: var(--bg-subtle);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 14px;
    text-align: center;
  }
  .metric-pill .label {
    font-family: var(--font-mono);
    font-size: 0.7rem;
    color: var(--text-muted);
    letter-spacing: 1.5px;
    text-transform: uppercase;
  }
  .metric-pill .value {
    font-family: var(--font-mono);
    font-size: 1.3rem;
    color: var(--accent);
    font-weight: bold;
    margin-top: 4px;
  }

  .stButton > button {
    background: transparent !important;
    border: 1.5px solid var(--accent) !important;
    color: var(--accent) !important;
    font-family: var(--font-head) !important;
    font-size: 1rem !important;
    letter-spacing: 2px !important;
    font-weight: 600 !important;
    padding: 10px 28px !important;
    border-radius: 4px !important;
    transition: all 0.2s ease !important;
    text-transform: uppercase !important;
  }
  .stButton > button:hover {
    background: rgba(0,229,255,0.1) !important;
    box-shadow: 0 0 18px rgba(0,229,255,0.25) !important;
  }

  .info-box {
    background: rgba(0,229,255,0.06);
    border-left: 3px solid var(--accent);
    border-radius: 0 6px 6px 0;
    padding: 12px 16px;
    font-family: var(--font-mono);
    font-size: 0.8rem;
    color: var(--text-muted);
    margin: 12px 0;
    letter-spacing: 0.5px;
  }
  .warn-box {
    background: rgba(240,165,0,0.06);
    border-left: 3px solid var(--accent-warn);
    border-radius: 0 6px 6px 0;
    padding: 12px 16px;
    font-family: var(--font-mono);
    font-size: 0.8rem;
    color: var(--accent-warn);
    margin: 12px 0;
  }

  #MainMenu { visibility: hidden; }
  footer    { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─── Material Config ───────────────────────────────────────────────────────────

MATERIALS = {
    "Screw": {
        "icon": "🔩",
        "weights": "./weights/screw.pth",
        "threshold": 0.020,
        "default_img_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9e/Nut_and_bolt_-_1.jpg/320px-Nut_and_bolt_-_1.jpg",
        "train_cmd": "python train.py --data_dir ./screw/screw/train/good --save_path ./weights/screw.pth",
    },
    "Wood": {
        "icon": "🪵",
        "weights": "./weights/wood.pth",
        "threshold": 0.018,
        "default_img_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8b/Wood_floor_texture.jpg/320px-Wood_floor_texture.jpg",
        "train_cmd": "python train.py --data_dir ./wood/wood/train/good --save_path ./weights/wood.pth",
    },
    "Leather": {
        "icon": "👜",
        "weights": "./weights/leather.pth",
        "threshold": 0.015,
        "default_img_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6f/Leather_jacket_detail.jpg/320px-Leather_jacket_detail.jpg",
        "train_cmd": "python train.py --data_dir ./leather/leather/train/good --save_path ./weights/leather.pth",
    },
    "Bottle": {
        "icon": "🍾",
        "weights": "./weights/bottle.pth",
        "threshold": 0.022,
        "default_img_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Binocular_vision.jpg/320px-Binocular_vision.jpg",
        "train_cmd": "python train.py --data_dir ./bottle/bottle/train/good --save_path ./weights/bottle.pth",
    },
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─── Model Loader (cached per material) ───────────────────────────────────────

@st.cache_resource
def load_model(weights_path):
    model = GANomalyNet().to(DEVICE)
    if os.path.exists(weights_path):
        state = torch.load(weights_path, map_location=DEVICE)
        model.load_state_dict(state)
        model.eval()
        return model, True
    model.eval()
    return model, False


# ─── Header ───────────────────────────────────────────────────────────────────

st.markdown("""
<div class="anomaly-header">
  <div class="icon">🔬</div>
  <div>
    <h1>ANOMALYLENS</h1>
    <p>GANOMALY · MULTI-MATERIAL DEFECT DETECTION · UNSUPERVISED</p>
  </div>
</div>
""", unsafe_allow_html=True)


# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown('<div class="card-title">🏭 SELECT MATERIAL</div>', unsafe_allow_html=True)

    selected_material = st.selectbox(
        "Material",
        options=list(MATERIALS.keys()),
        format_func=lambda m: f"{MATERIALS[m]['icon']}  {m}",
        label_visibility="collapsed",
    )

    cfg = MATERIALS[selected_material]
    model, weights_loaded = load_model(cfg["weights"])

    # Weights status
    if weights_loaded:
        st.success(f"✅ {selected_material} weights loaded")
    else:
        st.warning(f"⚠️ No weights for {selected_material}")
        st.markdown(f'<div class="warn-box">Train first:<br><code>{cfg["train_cmd"]}</code></div>', unsafe_allow_html=True)

    st.markdown("---")

    # Threshold
    st.markdown('<div class="card-title">🎚 THRESHOLD</div>', unsafe_allow_html=True)
    threshold = st.slider(
        "Anomaly Score Threshold",
        min_value=0.001,
        max_value=0.200,
        value=cfg["threshold"],
        step=0.001,
        format="%.3f",
        help="Reconstruction error above this → ANOMALY",
    )
    st.markdown(f'<div class="info-box">MSE &gt; {threshold:.3f} → anomaly flagged</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="card-title">ℹ HOW IT WORKS</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    1. Encoder1 compresses image → latent z1<br>
    2. Decoder reconstructs the image<br>
    3. Encoder2 re-encodes reconstruction → z2<br>
    4. High MSE(z1, z2) = anomaly<br>
    5. Each material has its own model
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(f'<div class="info-box">Device: {str(DEVICE).upper()}</div>', unsafe_allow_html=True)


# ─── Material Status Overview ─────────────────────────────────────────────────

st.markdown('<div class="card-title">🏭 MATERIAL MODELS STATUS</div>', unsafe_allow_html=True)

cols = st.columns(len(MATERIALS))
for col, (mat_name, mat_cfg) in zip(cols, MATERIALS.items()):
    exists = os.path.exists(mat_cfg["weights"])
    active = mat_name == selected_material
    status_class = "mat-loaded" if exists else "mat-missing"
    status_text  = "WEIGHTS LOADED" if exists else "NOT TRAINED"
    active_class = "active" if active else ""
    col.markdown(f"""
    <div class="material-card {active_class}">
      <div class="mat-icon">{mat_cfg['icon']}</div>
      <div class="mat-name">{mat_name}</div>
      <div class="mat-status {status_class}">{status_text}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")


# ─── Image Source ─────────────────────────────────────────────────────────────

st.markdown(f'<div class="card-title">📷 IMAGE SOURCE — {cfg["icon"]} {selected_material.upper()}</div>', unsafe_allow_html=True)

col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    source_mode = st.radio(
        "Input mode",
        options=["🖼️  Use Default Sample", "📤  Upload My Own Image"],
        index=0,
        label_visibility="collapsed",
    )

input_image  = None
image_label  = ""

# ── Default Sample ────────────────────────────────────────────────────────────
if "Default" in source_mode:
    with col_right:
        st.markdown(f'<div class="info-box">Default sample: {cfg["icon"]} {selected_material}</div>', unsafe_allow_html=True)

    if st.button(f"▶  LOAD & ANALYSE {selected_material.upper()} SAMPLE"):
        with st.spinner("Fetching sample image..."):
            try:
                resp = requests.get(cfg["default_img_url"], timeout=10)
                resp.raise_for_status()
                input_image = Image.open(io.BytesIO(resp.content)).convert("RGB")
                image_label = f"{selected_material}_default"
            except Exception as e:
                st.error(f"Could not fetch sample: {e}")

# ── Upload ────────────────────────────────────────────────────────────────────
else:
    with col_right:
        uploaded = st.file_uploader(
            f"Upload a {selected_material} image",
            type=["jpg", "jpeg", "png", "bmp", "tiff", "webp"],
            label_visibility="collapsed",
        )
    if uploaded:
        input_image = Image.open(uploaded).convert("RGB")
        image_label = uploaded.name
        if st.button("▶  ANALYSE UPLOADED IMAGE"):
            pass


# ─── Inference ────────────────────────────────────────────────────────────────

if input_image is not None:

    if not weights_loaded:
        st.error(f"⚠️ No trained weights found for **{selected_material}**. Please train the model first.")
        st.code(cfg["train_cmd"])
        st.stop()

    st.markdown("---")

    with st.spinner(f"🔄 Running {selected_material} anomaly detection..."):
        tensor, img_resized = load_image_tensor(input_image)
        tensor = tensor.to(DEVICE)
        with torch.no_grad():
            x_hat, z1, z2 = model(tensor)
        score, error_map = compute_anomaly_score(tensor.cpu(), x_hat.cpu(), z1.cpu(), z2.cpu())
        is_anomaly = score > threshold

    # ── Verdict ──
    st.markdown(f'<div class="card-title">🎯 DETECTION RESULT — {cfg["icon"]} {selected_material.upper()}</div>', unsafe_allow_html=True)

    if is_anomaly:
        st.markdown('<div class="verdict-anomaly">⚠ ANOMALY DETECTED</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="verdict-normal">✓ NORMAL — NO DEFECT FOUND</div>', unsafe_allow_html=True)

    # ── Metrics ──
    confidence = min(abs(score - threshold) / threshold * 100, 99.9)
    st.markdown(f"""
    <div class="metric-row">
      <div class="metric-pill">
        <div class="label">Material</div>
        <div class="value">{cfg['icon']} {selected_material}</div>
      </div>
      <div class="metric-pill">
        <div class="label">Latent Score</div>
        <div class="value">{score:.5f}</div>
      </div>
      <div class="metric-pill">
        <div class="label">Threshold</div>
        <div class="value">{threshold:.5f}</div>
      </div>
      <div class="metric-pill">
        <div class="label">Δ from Threshold</div>
        <div class="value" style="color: {'#ff4c4c' if is_anomaly else '#00e676'}">
          {'▲' if is_anomaly else '▼'} {abs(score - threshold):.5f}
        </div>
      </div>
      <div class="metric-pill">
        <div class="label">Confidence</div>
        <div class="value">{confidence:.1f}%</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Visual Analysis ──
    st.markdown('<div class="card-title">🖼 VISUAL ANALYSIS</div>', unsafe_allow_html=True)
    comparison_fig = build_comparison_figure(img_resized, x_hat.cpu(), error_map, score, threshold)
    st.image(comparison_fig, use_container_width=True)

    # ── Error Gauge ──
    st.markdown('<div class="card-title">📊 ERROR GAUGE</div>', unsafe_allow_html=True)
    gauge_fig = build_error_histogram(score, threshold)
    st.image(gauge_fig, use_container_width=True)

    # ── Raw Error Map ──
    with st.expander("🗺️  View Raw Error Map"):
        import matplotlib.pyplot as plt
        e_min, e_max = error_map.min(), error_map.max()
        norm_map = (error_map - e_min) / (e_max - e_min + 1e-8)
        fig, ax = plt.subplots(figsize=(5, 5))
        fig.patch.set_facecolor("#0d1117")
        ax.imshow(norm_map, cmap="jet")
        ax.set_title(f"{selected_material} — Per-pixel Error", color="white", fontsize=11)
        ax.axis("off")
        st.pyplot(fig)
        plt.close(fig)

    # ── Download ──
    buf = io.BytesIO()
    comparison_fig.save(buf, format="PNG")
    st.download_button(
        label="⬇  Download Analysis Report (PNG)",
        data=buf.getvalue(),
        file_name=f"anomaly_{selected_material.lower()}_{image_label}.png",
        mime="image/png",
    )

else:
    st.markdown(f"""
    <div class="card" style="text-align:center; padding: 60px 20px;">
      <div style="font-size:3rem; margin-bottom:16px;">{cfg['icon']}</div>
      <div style="font-family:'Rajdhani',sans-serif; font-size:1.4rem; color:#00e5ff; letter-spacing:2px;">
        SELECT AN IMAGE SOURCE ABOVE
      </div>
      <div style="font-family:'Share Tech Mono',monospace; font-size:0.8rem; color:#546e7a; margin-top:10px;">
        Material selected: {selected_material} · Upload or use default sample to begin
      </div>
    </div>
    """, unsafe_allow_html=True)
