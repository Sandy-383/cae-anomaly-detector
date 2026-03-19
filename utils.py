import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import io


# ─── Image Preprocessing ──────────────────────────────────────────────────────

IMAGE_SIZE = 128

preprocess = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])


def load_image_tensor(image_source):
    """
    Load image from a PIL Image, file path, or BytesIO object.
    Returns: (tensor [1,3,H,W], original PIL Image resized to 128x128)
    """
    if isinstance(image_source, str):
        img = Image.open(image_source).convert("RGB")
    elif isinstance(image_source, Image.Image):
        img = image_source.convert("RGB")
    else:
        img = Image.open(image_source).convert("RGB")

    img_resized = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
    tensor = preprocess(img_resized).unsqueeze(0)  # [1, 3, 128, 128]
    return tensor, img_resized


# ─── Anomaly Scoring ──────────────────────────────────────────────────────────

def compute_anomaly_score(original_tensor, reconstructed_tensor):
    """
    Compute pixel-wise MSE between original and reconstructed.
    Returns:
        score (float): mean reconstruction error (anomaly score)
        error_map (np.ndarray): [H, W] per-pixel error map
    """
    with torch.no_grad():
        diff = (original_tensor - reconstructed_tensor) ** 2          # [1,3,H,W]
        error_map = diff.squeeze(0).mean(dim=0).cpu().numpy()         # [H,W]
        score = float(error_map.mean())
    return score, error_map


# ─── Heatmap Overlay ──────────────────────────────────────────────────────────

def create_heatmap_overlay(original_pil, error_map, alpha=0.55):
    """
    Blend the original image with a jet-colormap heatmap of the error map.
    Returns a PIL Image.
    """
    # Normalize error map to [0, 1]
    e_min, e_max = error_map.min(), error_map.max()
    if e_max > e_min:
        norm_map = (error_map - e_min) / (e_max - e_min)
    else:
        norm_map = np.zeros_like(error_map)

    # Apply colormap
    colormap = cm.get_cmap("jet")
    heatmap_rgba = colormap(norm_map)                        # [H,W,4] float
    heatmap_rgb = (heatmap_rgba[:, :, :3] * 255).astype(np.uint8)
    heatmap_pil = Image.fromarray(heatmap_rgb).resize(original_pil.size, Image.LANCZOS)

    # Blend
    overlay = Image.blend(original_pil.convert("RGB"), heatmap_pil, alpha=alpha)
    return overlay


# ─── Figure Builders ──────────────────────────────────────────────────────────

def build_comparison_figure(original_pil, reconstructed_tensor, error_map, score, threshold):
    """
    Build a matplotlib figure: Original | Reconstructed | Heatmap Overlay.
    Returns a PIL Image of the figure.
    """
    # Reconstructed PIL
    recon_np = reconstructed_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    recon_np = np.clip(recon_np, 0, 1)
    recon_pil = Image.fromarray((recon_np * 255).astype(np.uint8))

    # Heatmap
    overlay = create_heatmap_overlay(original_pil, error_map)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.patch.set_facecolor("#0d1117")

    panels = [
        (original_pil, "Original"),
        (recon_pil, "Reconstructed"),
        (overlay, "Anomaly Heatmap"),
    ]

    for ax, (img, title) in zip(axes, panels):
        ax.imshow(img)
        ax.set_title(title, color="white", fontsize=13, fontweight="bold", pad=10)
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")

    verdict = "⚠️  ANOMALY DETECTED" if score > threshold else "✅  NORMAL"
    color = "#ff4c4c" if score > threshold else "#4caf92"
    fig.suptitle(
        f"{verdict}   |   Reconstruction Error: {score:.5f}   |   Threshold: {threshold:.5f}",
        color=color, fontsize=14, fontweight="bold", y=1.02
    )
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    buf.seek(0)
    plt.close(fig)
    return Image.open(buf)


def build_error_histogram(score, threshold):
    """Single-bar gauge showing score vs threshold."""
    fig, ax = plt.subplots(figsize=(7, 2.2))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")

    bar_color = "#ff4c4c" if score > threshold else "#4caf92"
    ax.barh(["Error"], [score], color=bar_color, height=0.4, zorder=3)
    ax.axvline(threshold, color="#f0a500", linewidth=2, linestyle="--", label=f"Threshold ({threshold:.5f})", zorder=4)

    ax.set_xlabel("Reconstruction Error (MSE)", color="#8b949e")
    ax.tick_params(colors="#8b949e")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
    ax.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="white")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    buf.seek(0)
    plt.close(fig)
    return Image.open(buf)
