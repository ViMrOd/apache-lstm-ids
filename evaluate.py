"""
evaluate.py
===========
Evaluation script for the LSTM Autoencoder log anomaly detector.

Pipeline
--------
  1. Load model checkpoint (weights + config)
  2. Score every sequence in val and test DataLoaders with per-sample MSE
  3. Sweep thresholds on the val set to find the one that maximises F1
  4. Evaluate on the test set: precision, recall, F1, ROC-AUC
  5. Save two diagnostic plots:
       <plot_dir>/roc_curve.png          – ROC curve with AUC annotation
       <plot_dir>/score_distribution.png – reconstruction score histograms
                                           split by true label (normal / anomalous)

Usage
-----
    # Real data, automatic threshold selection (recommended):
    python evaluate.py --checkpoint checkpoints/best_model.pt --data_dir data/ --vocab_size 47

    # Synthetic smoke-test:
    python evaluate.py --checkpoint checkpoints/best_model.pt --synthetic

    # Fix a threshold instead of sweeping:
    python evaluate.py --checkpoint checkpoints/best_model.pt --anomaly_threshold 0.05

    # Override plot output directory:
    python evaluate.py --checkpoint checkpoints/best_model.pt --plot_dir figures/
"""

import os
import sys
import logging
import argparse

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe on OSC / headless nodes
import matplotlib.pyplot as plt

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)

from config import get_config
from autoencoder import LSTMAutoencoder


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DataLoaders
# ---------------------------------------------------------------------------

def get_eval_dataloaders(config):
    """
    Load and return (val_loader, test_loader).

    Controlled by config.synthetic:
      False (default) — loads real .npy files from config.data_dir:
          X_val.npy  / y_val.npy   : val split with anomaly labels
          X_test.npy / y_test.npy  : test split with anomaly labels
      True — generates synthetic labeled data for smoke-testing.

    Both loaders yield 2-tuple batches: (B, T) int64, (B,) int64
    Labels: 0 = normal, 1 = anomalous.
    """
    if config.synthetic:
        logger.warning(
            "get_eval_dataloaders() is using SYNTHETIC data (--synthetic flag set). "
            "Remove --synthetic to load real data from --data_dir."
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

        return _make_labeled_loader(400), _make_labeled_loader(1000)

    # ------------------------------------------------------------------
    # Real data path
    # ------------------------------------------------------------------
    logger.info(f"Loading real eval data from {config.data_dir}")

    X_val  = torch.from_numpy(np.load(os.path.join(config.data_dir, "X_val.npy")))
    y_val  = torch.from_numpy(np.load(os.path.join(config.data_dir, "y_val.npy")))
    X_test = torch.from_numpy(np.load(os.path.join(config.data_dir, "X_test.npy")))
    y_test = torch.from_numpy(np.load(os.path.join(config.data_dir, "y_test.npy")))

    logger.info(
        f"Loaded — X_val: {tuple(X_val.shape)}, y_val: {tuple(y_val.shape)}, "
        f"X_test: {tuple(X_test.shape)}, y_test: {tuple(y_test.shape)}"
    )

    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_val, y_val),
        batch_size=config.batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_test, y_test),
        batch_size=config.batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )

    return val_loader, test_loader


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def collect_scores(model, loader, device):
    """
    Run the model over every batch and collect per-sample scores and labels.

    Args:
        model  : LSTMAutoencoder (already on `device`, in eval mode)
        loader : DataLoader yielding (x, labels) 2-tuples
        device : torch.device

    Returns:
        scores : np.ndarray, shape (N,)  — per-sample reconstruction MSE
        labels : np.ndarray, shape (N,)  — 0/1 ground-truth anomaly labels
    """
    all_scores = []
    all_labels = []

    model.eval()

    for batch in loader:
        x, batch_labels = batch[0].to(device), batch[1]
        scores = model.compute_reconstruction_loss(x)   # (B,) — no_grad inside
        all_scores.append(scores.cpu())
        all_labels.append(batch_labels)

    scores_np = torch.cat(all_scores).numpy()
    labels_np = torch.cat(all_labels).numpy().astype(int)
    return scores_np, labels_np


# ---------------------------------------------------------------------------
# Threshold selection
# ---------------------------------------------------------------------------

def find_best_threshold(scores, labels, n_thresholds=200):
    """
    Sweep threshold values and return the one that maximises F1 on the
    provided split (intended for the val set).

    Why sweep instead of using a fixed percentile?
      A percentile-based threshold assumes a known anomaly rate, which is
      rarely reliable in practice. Maximising F1 directly finds the threshold
      that best balances precision and recall for the actual score distribution,
      making it robust to class imbalance.

    Args:
        scores       : np.ndarray (N,)  — reconstruction MSE per sequence
        labels       : np.ndarray (N,)  — ground-truth 0/1 labels
        n_thresholds : int              — number of candidate thresholds

    Returns:
        best_threshold : float  — threshold maximising val F1
        best_f1        : float  — the corresponding F1 score
    """
    lo, hi     = np.percentile(scores, 1), np.percentile(scores, 99)
    candidates = np.linspace(lo, hi, n_thresholds)

    labels = (labels > 0).astype(int)  # binarize: 0=normal, >0=anomalous
    best_f1, best_threshold = -1.0, candidates[0]

    for thresh in candidates:
        preds = (scores >= thresh).astype(int)
        f1    = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_threshold = f1, thresh

    return best_threshold, best_f1


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_roc_curve(labels, scores, auc, save_path):
    """
    Plot and save the ROC curve with AUC annotated in the legend.
    """
    fpr, tpr, _ = roc_curve(labels, scores)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#2563eb", lw=2, label=f"LSTM-AE (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], color="#9ca3af", lw=1.2, linestyle="--",
            label="Random (AUC = 0.50)")

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve — Log Anomaly Detection", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info(f"ROC curve saved → {save_path}")


def plot_score_distribution(labels, scores, threshold, save_path):
    """
    Plot overlapping histograms of reconstruction scores split by ground-truth
    label, with a vertical line at the selected anomaly threshold.
    """
    normal_scores  = scores[labels == 0]
    anomaly_scores = scores[labels == 1]

    fig, ax = plt.subplots(figsize=(7, 4))

    bins = np.linspace(scores.min(), scores.max(), 60)
    ax.hist(normal_scores,  bins=bins, density=True, alpha=0.55,
            color="#22c55e", label=f"Normal (n={len(normal_scores):,})")
    ax.hist(anomaly_scores, bins=bins, density=True, alpha=0.55,
            color="#ef4444", label=f"Anomalous (n={len(anomaly_scores):,})")
    ax.axvline(threshold, color="#1d4ed8", lw=1.8, linestyle="--",
               label=f"Threshold = {threshold:.4f}")

    ax.set_xlabel("Reconstruction MSE", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Score Distribution — Normal vs Anomalous", fontsize=13,
                 fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info(f"Score distribution plot saved → {save_path}")


# ---------------------------------------------------------------------------
# Main evaluation procedure
# ---------------------------------------------------------------------------

def evaluate(config):
    """
    Full evaluation pipeline.

    Steps:
      1. Load model from checkpoint
      2. Score val set → find optimal threshold (or use config.anomaly_threshold)
      3. Score test set → compute precision / recall / F1 / ROC-AUC
      4. Save ROC curve and score-distribution plots
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    if not os.path.isfile(config.checkpoint):
        logger.error(f"Checkpoint not found: {config.checkpoint}")
        sys.exit(1)

    model = LSTMAutoencoder(config).to(device)
    checkpoint = torch.load(config.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info(
        f"Loaded checkpoint: {config.checkpoint} "
        f"(saved at epoch {checkpoint.get('epoch', '?')}, "
        f"val_loss {checkpoint.get('val_loss', float('nan')):.6f})"
    )
    model.eval()

    val_loader, test_loader = get_eval_dataloaders(config)
    logger.info(
        f"DataLoaders ready — "
        f"val batches: {len(val_loader)}, test batches: {len(test_loader)}"
    )

    # ---- Val set scoring -------------------------------------------------
    logger.info("Scoring val set …")
    val_scores, val_labels = collect_scores(model, val_loader, device)
    logger.info(
        f"Val scores — min: {val_scores.min():.6f}, "
        f"max: {val_scores.max():.6f}, mean: {val_scores.mean():.6f}"
    )
    val_labels = (val_labels > 0).astype(int)  # binarize multiclass labels
    logger.info(
        f"Val label distribution — "
        f"normal: {(val_labels == 0).sum():,}, "
        f"anomalous: {(val_labels == 1).sum():,}"
    )

    # ---- Threshold selection ---------------------------------------------
    if config.anomaly_threshold is not None:
        threshold = config.anomaly_threshold
        logger.info(f"Using fixed threshold from config: {threshold:.6f}")
    else:
        logger.info(
            f"Sweeping thresholds on val set to maximise F1 "
            f"(n_thresholds={config.threshold_sweep_steps}) …"
        )
        threshold, val_f1 = find_best_threshold(
            val_scores, val_labels, n_thresholds=config.threshold_sweep_steps
        )
        logger.info(f"Best val threshold: {threshold:.6f}  →  val F1: {val_f1:.4f}")

    # ---- Test set scoring ------------------------------------------------
    logger.info("Scoring test set …")
    test_scores, test_labels = collect_scores(model, test_loader, device)
    logger.info(
        f"Test scores — min: {test_scores.min():.6f}, "
        f"max: {test_scores.max():.6f}, mean: {test_scores.mean():.6f}"
    )
    test_labels_raw = test_labels.copy()  # save raw multiclass labels
    test_labels = (test_labels > 0).astype(int)  # binarize multiclass labels
    logger.info(
        f"Test label distribution — "
        f"normal: {(test_labels == 0).sum():,}, "
        f"anomalous: {(test_labels == 1).sum():,}"
    )

    # ---- Metrics ---------------------------------------------------------
    test_preds = (test_scores >= threshold).astype(int)

    precision = precision_score(test_labels, test_preds, zero_division=0)
    recall    = recall_score(test_labels, test_preds, zero_division=0)
    f1        = f1_score(test_labels, test_preds, zero_division=0)

    if len(np.unique(test_labels)) < 2:
        logger.warning("Test set has only one class — ROC-AUC undefined (set to 0.0).")
        auc = 0.0
    else:
        auc = roc_auc_score(test_labels, test_scores)

    logger.info("=" * 60)
    logger.info("TEST SET RESULTS")
    logger.info("=" * 60)
    logger.info(f"  Threshold  : {threshold:.6f}")
    logger.info(f"  Precision  : {precision:.4f}")
    logger.info(f"  Recall     : {recall:.4f}")
    logger.info(f"  F1 Score   : {f1:.4f}")
    logger.info(f"  ROC-AUC    : {auc:.4f}")
    logger.info("=" * 60)
    # ---- Per-category breakdown ------------------------------------------
    # Uses raw multiclass labels to show detection rate per anomaly type.
    # Load label_map if available for human-readable category names.
    label_map_path = os.path.join(config.data_dir, "label_map.json")
    if os.path.exists(label_map_path) and 'test_labels_raw' in locals():
        import json
        with open(label_map_path) as lf:
            label_map = json.load(lf)
        unique_cats = sorted(set(test_labels_raw.tolist()))
        anomaly_cats = [c for c in unique_cats if c != 0]
        if anomaly_cats:
            logger.info("=" * 60)
            logger.info("PER-CATEGORY BREAKDOWN (test set)")
            logger.info("=" * 60)
            for cat in anomaly_cats:
                mask = test_labels_raw == cat
                cat_scores = test_scores[mask]
                cat_preds  = (cat_scores >= threshold).astype(int)
                detected   = cat_preds.sum()
                total      = mask.sum()
                rate       = detected / total * 100 if total > 0 else 0.0
                name       = label_map.get(str(cat), f"label_{cat}")
                logger.info(
                    f"  [{cat:>3}] {name:<16} : {total:>5} seqs | "
                    f"mean_score: {cat_scores.mean():.4f} | "
                    f"detected: {detected}/{total} ({rate:.1f}%)"
                )
            logger.info("=" * 60)


    # ---- Plots -----------------------------------------------------------
    os.makedirs(config.plot_dir, exist_ok=True)

    plot_roc_curve(
        test_labels, test_scores, auc,
        save_path=os.path.join(config.plot_dir, "roc_curve.png"),
    )
    plot_score_distribution(
        test_labels, test_scores, threshold,
        save_path=os.path.join(config.plot_dir, "score_distribution.png"),
    )

    return {
        "threshold": threshold,
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
        "roc_auc":   auc,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    config = get_config()

    extra_parser = argparse.ArgumentParser(add_help=False)

    extra_parser.add_argument(
        "--checkpoint",
        type=str,
        default=os.path.join(config.checkpoint_dir, "best_model.pt"),
        help="Path to the model checkpoint (.pt) to evaluate.",
    )
    extra_parser.add_argument(
        "--plot_dir",
        type=str,
        default="plots/",
        help="Directory to save roc_curve.png and score_distribution.png.",
    )
    extra_parser.add_argument(
        "--threshold_sweep_steps",
        type=int,
        default=200,
        help=(
            "Number of candidate thresholds when sweeping for best F1. "
            "Ignored if --anomaly_threshold is set explicitly."
        ),
    )

    extra_args, _ = extra_parser.parse_known_args()

    config.checkpoint            = extra_args.checkpoint
    config.plot_dir              = extra_args.plot_dir
    config.threshold_sweep_steps = extra_args.threshold_sweep_steps

    logger.info("=== Evaluation Configuration ===")
    for key, value in sorted(vars(config).items()):
        logger.info(f"  {key:<28} = {value}")
    logger.info("=" * 60)

    evaluate(config)


if __name__ == "__main__":
    main()
