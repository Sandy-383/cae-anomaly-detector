"""
train.py — Offline training script for GANomalyNet on MVTec AD dataset.

Usage:
    python train.py --data_dir ./screw/screw/train/good --epochs 100 --batch_size 16

Features:
    - Three-part loss: adversarial + reconstruction + latent consistency
    - Checkpoint saved every 10 epochs automatically
    - Emergency save on Ctrl+C or terminal close
    - Hard kill of all child processes on forced exit
"""

import os
import sys
import signal
import atexit
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from model import GANomalyNet

# ─── Force-Kill Handlers ───────────────────────────────────────────────────────

def _force_exit(signum=None, frame=None):
    print("\n\n[TRAINER] ⛔ Interrupt received — shutting down all processes...")
    try:
        import psutil
        proc = psutil.Process(os.getpid())
        for child in proc.children(recursive=True):
            child.kill()
    except Exception:
        pass
    print("[TRAINER] ✅ All processes terminated. Exiting.")
    os._exit(0)

def _cleanup():
    print("[TRAINER] 🧹 Cleanup complete.")

atexit.register(_cleanup)
signal.signal(signal.SIGINT,  _force_exit)
signal.signal(signal.SIGTERM, _force_exit)
if hasattr(signal, 'SIGBREAK'):
    signal.signal(signal.SIGBREAK, _force_exit)


# ─── Dataset ──────────────────────────────────────────────────────────────────

class NormalImageDataset(Dataset):
    """Loads all images from a folder (only normal/good images for training)."""

    EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

    def __init__(self, folder_path, image_size=128):
        self.paths = [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if os.path.splitext(f)[1].lower() in self.EXTENSIONS
        ]
        if not self.paths:
            raise ValueError(f"No images found in: {folder_path}")

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)


# ─── Training Loop ─────────────────────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = NormalImageDataset(args.data_dir, image_size=args.image_size)
    loader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    print(f"Training on {len(dataset)} images — {len(loader)} batches/epoch")

    model = GANomalyNet().to(device)

    # Generator params = encoder1 + decoder + encoder2
    g_params = (
        list(model.encoder1.parameters()) +
        list(model.decoder.parameters()) +
        list(model.encoder2.parameters())
    )
    opt_g = torch.optim.Adam(g_params,                      lr=args.lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(model.discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))

    scheduler_g = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_g, patience=10, factor=0.5)

    criterion_bce = nn.BCEWithLogitsLoss()
    criterion_mse = nn.MSELoss()

    loss_history = []
    weights_dir  = os.path.dirname(args.save_path)
    os.makedirs(weights_dir, exist_ok=True)

    def save_checkpoint(epoch, reason="Checkpoint"):
        ckpt_path = args.save_path.replace(".pth", f"_epoch{epoch}.pth")
        torch.save(model.state_dict(), ckpt_path)
        print(f"[TRAINER] 💾 {reason} → {ckpt_path}")

    def save_final(epoch):
        torch.save(model.state_dict(), args.save_path)
        print(f"[TRAINER] ✅ Final weights saved → {args.save_path}")
        if loss_history:
            g_losses = [l[0] for l in loss_history]
            d_losses = [l[1] for l in loss_history]
            plt.figure(figsize=(10, 4))
            plt.plot(g_losses, color="#e05c5c", linewidth=2, label="Generator Loss")
            plt.plot(d_losses, color="#5c9ee0", linewidth=2, label="Discriminator Loss")
            plt.title("GANomaly Training Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.tight_layout()
            curve_path = os.path.join(weights_dir, "loss_curve.png")
            plt.savefig(curve_path, dpi=150)
            plt.close()
            print(f"[TRAINER] 📈 Loss curve saved → {curve_path}")

    try:
        for epoch in range(1, args.epochs + 1):
            model.train()
            epoch_loss_g = 0.0
            epoch_loss_d = 0.0

            for batch in loader:
                batch       = batch.to(device)
                B           = batch.size(0)
                real_labels = torch.ones(B,  device=device)
                fake_labels = torch.zeros(B, device=device)

                # ── Forward ──────────────────────────────────────────────────
                x_hat, z1, z2 = model(batch)

                # ── Train Discriminator ───────────────────────────────────────
                opt_d.zero_grad()
                pred_real = model.discriminator(batch)
                pred_fake = model.discriminator(x_hat.detach())
                loss_d = 0.5 * (
                    criterion_bce(pred_real, real_labels) +
                    criterion_bce(pred_fake, fake_labels)
                )
                loss_d.backward()
                opt_d.step()

                # ── Train Generator ───────────────────────────────────────────
                opt_g.zero_grad()
                pred_fake_g = model.discriminator(x_hat)
                loss_adv = criterion_bce(pred_fake_g, real_labels)  # fool D
                loss_rec = criterion_mse(x_hat, batch)              # pixel reconstruction
                loss_lat = criterion_mse(z2, z1.detach())           # latent consistency
                loss_g   = (args.w_adv * loss_adv +
                            args.w_rec * loss_rec +
                            args.w_lat * loss_lat)
                loss_g.backward()
                opt_g.step()

                epoch_loss_g += loss_g.item()
                epoch_loss_d += loss_d.item()

            avg_g = epoch_loss_g / len(loader)
            avg_d = epoch_loss_d / len(loader)
            loss_history.append((avg_g, avg_d))
            scheduler_g.step(avg_g)

            if epoch % 10 == 0 or epoch == 1:
                print(
                    f"Epoch [{epoch:>4}/{args.epochs}]  "
                    f"G: {avg_g:.5f}  D: {avg_d:.5f}  "
                    f"LR: {opt_g.param_groups[0]['lr']:.2e}"
                )

            if epoch % 10 == 0:
                save_checkpoint(epoch, reason="Auto checkpoint")

        save_final(args.epochs)

    except (KeyboardInterrupt, SystemExit):
        last_epoch = len(loss_history)
        print(f"\n[TRAINER] ⛔ Stopped at epoch {last_epoch}")
        if last_epoch > 0:
            save_checkpoint(last_epoch, reason="Emergency save")
            save_final(last_epoch)
        else:
            print("[TRAINER] No epochs completed — nothing to save.")
        sys.exit(0)


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GANomalyNet for anomaly detection")
    parser.add_argument("--data_dir",   type=str,   required=True,                       help="Path to folder with NORMAL training images")
    parser.add_argument("--epochs",     type=int,   default=100,                          help="Number of training epochs")
    parser.add_argument("--batch_size", type=int,   default=16,                           help="Batch size")
    parser.add_argument("--lr",         type=float, default=2e-4,                         help="Learning rate for both G and D")
    parser.add_argument("--image_size", type=int,   default=128,                          help="Resize images to this square size")
    parser.add_argument("--save_path",  type=str,   default="./weights/autoencoder.pth",  help="Where to save model weights")
    parser.add_argument("--w_adv",      type=float, default=1.0,                          help="Adversarial loss weight")
    parser.add_argument("--w_rec",      type=float, default=50.0,                         help="Reconstruction loss weight")
    parser.add_argument("--w_lat",      type=float, default=1.0,                          help="Latent consistency loss weight")
    args = parser.parse_args()
    train(args)
