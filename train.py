"""
train.py
========
Training loop for the LSTM Autoencoder log anomaly detector.

Responsibilities
----------------
  1. Load pre-built DataLoaders (real .npy files or synthetic fallback)
  2. Train the LSTMAutoencoder with MSE reconstruction loss
  3. Log train/val loss per epoch (stdout + optional TensorBoard)
  4. Save the best checkpoint (lowest val loss) and periodic checkpoints
  5. Stop early when val loss stops improving (configurable patience)

Usage
-----
    # Real data (default):
    python train.py --data_dir data/ --vocab_size 47

    # Synthetic smoke-test:
    python train.py --synthetic

    # Override any hyperparameter:
    python train.py --data_dir data/ --vocab_size 47 --lr 5e-4 --epochs 50

SLURM
-----
Submit via train_osc.sh. All stdout is captured by SLURM so every
print/log here ends up in your .log file for post-hoc inspection.
"""

import os
import sys
import time
import random
import logging

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from config import get_config
from autoencoder import LSTMAutoencoder

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """
    Fix all random seeds for reproducibility across runs.

    PyTorch, NumPy, and Python's random module each maintain independent
    PRNG states, so all three must be seeded. CUDNN's nondeterministic
    algorithms are disabled so GPU results are also reproducible (at a small
    speed cost — acceptable for a course project).
    """
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# DataLoaders
# ---------------------------------------------------------------------------

def get_dataloaders(config):
    """
    Load and return (train_loader, val_loader).

    Controlled by config.synthetic:
      False (default) — loads real .npy files from config.data_dir:
          X_train.npy  : (N, 20) int64, normal sequences only
          X_val.npy    : (N, 20) int64
          y_val.npy    : (N,)    int64, 0=normal 1=anomalous
      True — generates random synthetic sequences for smoke-testing.

    train_loader yields single-tensor batches : (B, T)
    val_loader   yields 2-tuple batches       : (B, T), (B,)

    Note: run_epoch() only uses batch[0] from val_loader, so the labels
    are silently unused during training — correct behaviour for an autoencoder.
    They are present so the same loader can be reused by evaluate.py.
    """
    if config.synthetic:
        logger.warning(
            "get_dataloaders() is using SYNTHETIC data (--synthetic flag set). "
            "Remove --synthetic to load real data from --data_dir."
        )
        n_train, n_val = 2000, 400

        def _make_unlabeled_loader(n):
            data = torch.randint(1, config.vocab_size, (n, config.window_size))
            dataset = torch.utils.data.TensorDataset(data)
            return torch.utils.data.DataLoader(
                dataset, batch_size=config.batch_size, shuffle=True,
                pin_memory=torch.cuda.is_available(),
            )

        def _make_labeled_loader(n, anomaly_fraction=0.10):
            n_anomaly = int(n * anomaly_fraction)
            n_normal  = n - n_anomaly
            normal_x  = torch.randint(1, config.vocab_size // 2,
                                      (n_normal, config.window_size))
            anomaly_x = torch.randint(config.vocab_size // 2, config.vocab_size,
                                      (n_anomaly, config.window_size))
            x      = torch.cat([normal_x, anomaly_x])
            labels = torch.cat([torch.zeros(n_normal),
                                torch.ones(n_anomaly)]).long()
            perm   = torch.randperm(n)
            dataset = torch.utils.data.TensorDataset(x[perm], labels[perm])
            return torch.utils.data.DataLoader(
                dataset, batch_size=config.batch_size, shuffle=False,
                pin_memory=torch.cuda.is_available(),
            )

        return _make_unlabeled_loader(n_train), _make_labeled_loader(n_val)

    # ------------------------------------------------------------------
    # Real data path
    # ------------------------------------------------------------------
    import numpy as np

    logger.info(f"Loading real data from {config.data_dir}")

    X_train = torch.from_numpy(
        np.load(os.path.join(config.data_dir, "X_train.npy"))
    )
    X_val = torch.from_numpy(
        np.load(os.path.join(config.data_dir, "X_val.npy"))
    )
    y_val = torch.from_numpy(
        np.load(os.path.join(config.data_dir, "y_val.npy"))
    )

    logger.info(
        f"Loaded — X_train: {tuple(X_train.shape)}, "
        f"X_val: {tuple(X_val.shape)}, y_val: {tuple(y_val.shape)}"
    )

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_train),
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_val, y_val),
        batch_size=config.batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(model, optimizer, epoch, val_loss, path):
    """
    Save a full training checkpoint.

    Saving optimizer state alongside model weights means training can be
    resumed from any checkpoint without losing momentum/learning-rate
    history — useful if an OSC job times out mid-training.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "config": config,
        },
        path,
    )
    logger.info(f"Checkpoint saved → {path}")


def load_checkpoint(path, model, optimizer=None):
    """
    Load a checkpoint back into model (and optionally optimizer).

    Returns the epoch and val_loss stored in the checkpoint so the caller
    knows where training left off.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    logger.info(
        f"Loaded checkpoint from {path} "
        f"(epoch {checkpoint['epoch']}, val_loss {checkpoint['val_loss']:.6f})"
    )
    return checkpoint["epoch"], checkpoint["val_loss"]


# ---------------------------------------------------------------------------
# Single-epoch helpers
# ---------------------------------------------------------------------------

def run_epoch(model, loader, optimizer, device, clip_grad_norm, train: bool):
    """
    Run one full pass over a DataLoader.

    Unified train/eval logic keeps the two modes in sync — any change to
    the forward pass is automatically reflected in both modes.

    Args:
        model:          LSTMAutoencoder
        loader:         DataLoader yielding (B, T) tensors or (B,T)+(B,) tuples
        optimizer:      Adam (ignored when train=False)
        device:         torch.device
        clip_grad_norm: float — max gradient norm (0 disables clipping)
        train:          bool — True = update weights, False = eval mode

    Returns:
        mean_loss: float — average MSE loss over all batches
    """
    model.train(train)

    total_loss = 0.0
    n_batches  = 0

    ctx = torch.enable_grad() if train else torch.no_grad()

    with ctx:
        for batch in loader:
            # batch[0] is always x; batch[1] (labels) is ignored here
            x = batch[0].to(device)   # (B, T)

            loss, _ = model(x)        # _ = latent (not needed for training)

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()

                if clip_grad_norm > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)

                optimizer.step()

            total_loss += loss.item()
            n_batches  += 1

    return total_loss / max(n_batches, 1)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(config):
    """
    Full training procedure.

    Steps:
      1. Setup: seed, device, model, optimizer, scheduler, TensorBoard
      2. Load DataLoaders
      3. Epoch loop: train → validate → log → checkpoint → early-stop check
    """
    set_seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    model = LSTMAutoencoder(config).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {n_params:,}")
    logger.info(f"\n{model}")

    optimizer = Adam(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=max(1, config.patience // 2),
    )

    writer = None
    if TENSORBOARD_AVAILABLE:
        os.makedirs(config.log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=config.log_dir)
        logger.info(f"TensorBoard logs → {config.log_dir}")
    else:
        logger.warning("TensorBoard not installed — skipping TB logging.")

    train_loader, val_loader = get_dataloaders(config)
    logger.info(
        f"DataLoaders ready — "
        f"train batches: {len(train_loader)}, val batches: {len(val_loader)}"
    )

    best_val_loss          = float("inf")
    epochs_without_improvement = 0
    best_ckpt_path         = os.path.join(config.checkpoint_dir, "best_model.pt")

    logger.info("=" * 60)
    logger.info(f"Starting training for up to {config.epochs} epochs")
    logger.info(f"Early stopping patience: {config.patience} epochs")
    logger.info("=" * 60)

    for epoch in range(1, config.epochs + 1):
        epoch_start = time.time()

        train_loss = run_epoch(
            model, train_loader, optimizer, device,
            config.clip_grad_norm, train=True,
        )
        val_loss = run_epoch(
            model, val_loader, optimizer, device,
            config.clip_grad_norm, train=False,
        )

        elapsed    = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        logger.info(
            f"Epoch {epoch:>3}/{config.epochs} | "
            f"train_loss: {train_loss:.6f} | "
            f"val_loss: {val_loss:.6f} | "
            f"lr: {current_lr:.2e} | "
            f"time: {elapsed:.1f}s"
        )

        if writer:
            writer.add_scalars("loss", {"train": train_loss, "val": val_loss}, epoch)
            writer.add_scalar("learning_rate", current_lr, epoch)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            save_checkpoint(model, optimizer, epoch, val_loss, best_ckpt_path)
            logger.info(f"  ↳ New best val_loss: {best_val_loss:.6f}")
        else:
            epochs_without_improvement += 1
            logger.info(
                f"  ↳ No improvement for "
                f"{epochs_without_improvement}/{config.patience} epochs"
            )

        if epoch % config.save_every == 0:
            periodic_path = os.path.join(
                config.checkpoint_dir, f"epoch_{epoch:04d}.pt"
            )
            save_checkpoint(model, optimizer, epoch, val_loss, periodic_path)

        if epochs_without_improvement >= config.patience:
            logger.info(
                f"Early stopping triggered after {epoch} epochs "
                f"({config.patience} epochs without val loss improvement)."
            )
            break

    logger.info("=" * 60)
    logger.info(f"Training complete. Best val_loss: {best_val_loss:.6f}")
    logger.info(f"Best checkpoint saved at: {best_ckpt_path}")
    logger.info("=" * 60)

    if writer:
        writer.close()

    return best_ckpt_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    import argparse

    config = get_config()

    extra_parser = argparse.ArgumentParser(add_help=False)
    extra_parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help=(
            "Early stopping patience: number of epochs with no val loss "
            "improvement before training stops."
        ),
    )
    extra_args, _ = extra_parser.parse_known_args()
    config.patience = extra_args.patience

    logger.info("=== Configuration ===")
    for key, value in sorted(vars(config).items()):
        logger.info(f"  {key:<25} = {value}")
    logger.info("=" * 60)

    train(config)


if __name__ == "__main__":
    main()
