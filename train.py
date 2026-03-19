"""
train.py — Offline training script for ConvAutoencoder on MVTec AD dataset.

Usage:
    python train.py --data_dir ./screw/screw/train/good --epochs 100 --batch_size 16

Features:
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
from model import ConvAutoencoder

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
    os._exit(0)  # Hard exit — no hanging

def _cleanup():
    print("[TRAINER] 🧹 Cleanup complete.")

# Register handlers
atexit.register(_cleanup)
signal.signal(signal.SIGINT,  _force_exit)    # Ctrl+C
signal.signal(signal.SIGTERM, _force_exit)    # Task kill / terminal close
if hasattr(signal, 'SIGBREAK'):
    signal.signal(signal.SIGBREAK, _force_exit)  # Windows Ctrl+Break


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

    # Dataset & DataLoader
    dataset = NormalImageDataset(args.data_dir, image_size=args.image_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    print(f"Training on {len(dataset)} images — {len(loader)} batches/epoch")

    # Model, loss, optimizer
    model = ConvAutoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    loss_history = []
    weights_dir = os.path.dirname(args.save_path)
    os.makedirs(weights_dir, exist_ok=True)

    # ── Helper: save checkpoint ──
    def save_checkpoint(epoch, reason="Checkpoint"):
        ckpt_path = args.save_path.replace(".pth", f"_epoch{epoch}.pth")
        torch.save(model.state_dict(), ckpt_path)
        print(f"[TRAINER] 💾 {reason} → {ckpt_path}")

    # ── Helper: save final weights + loss curve ──
    def save_final(epoch):
        torch.save(model.state_dict(), args.save_path)
        print(f"[TRAINER] ✅ Final weights saved → {args.save_path}")
        if loss_history:
            plt.figure(figsize=(10, 4))
            plt.plot(loss_history, color="#e05c5c", linewidth=2)
            plt.title("Training Loss (MSE)")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.tight_layout()
            curve_path = os.path.join(weights_dir, "loss_curve.png")
            plt.savefig(curve_path, dpi=150)
            plt.close()
            print(f"[TRAINER] 📈 Loss curve saved → {curve_path}")

    # ── Training ──
    try:
        for epoch in range(1, args.epochs + 1):
            model.train()
            epoch_loss = 0.0

            for batch in loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                output = model(batch)
                loss = criterion(output, batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(loader)
            loss_history.append(avg_loss)
            scheduler.step(avg_loss)

            if epoch % 10 == 0 or epoch == 1:
                print(f"Epoch [{epoch:>4}/{args.epochs}]  Loss: {avg_loss:.6f}  LR: {optimizer.param_groups[0]['lr']:.2e}")

            # Auto checkpoint every 10 epochs
            if epoch % 10 == 0:
                save_checkpoint(epoch, reason="Auto checkpoint")

        # Completed fully
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
    parser = argparse.ArgumentParser(description="Train ConvAutoencoder for anomaly detection")
    parser.add_argument("--data_dir",   type=str,   required=True,                        help="Path to folder with NORMAL training images")
    parser.add_argument("--epochs",     type=int,   default=100,                           help="Number of training epochs")
    parser.add_argument("--batch_size", type=int,   default=16,                            help="Batch size")
    parser.add_argument("--lr",         type=float, default=1e-3,                          help="Learning rate")
    parser.add_argument("--image_size", type=int,   default=128,                           help="Resize images to this square size")
    parser.add_argument("--save_path",  type=str,   default="./weights/autoencoder.pth",   help="Where to save model weights")
    args = parser.parse_args()
    train(args)
