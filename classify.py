"""
classify.py
===========
Stage 2 of the detection pipeline: anomaly type classification.

Pipeline
--------
  Stage 1 (autoencoder): reconstruction score > threshold → ANOMALOUS
  Stage 2 (this script): latent vector → MLP → anomaly category

How it works
------------
  1. Load a trained LSTMAutoencoder checkpoint
  2. Extract 32-dim latent vectors for all labeled sequences
  3. Filter to anomalous sequences only (label > 0)
  4. Train a small MLP classifier: latent (32-dim) → category (N classes)
  5. Evaluate per-category accuracy on the test set
  6. Save the classifier checkpoint for use in the demo

Why train only on anomalous sequences?
  The classifier's job is to distinguish *between* anomaly types, not between
  normal and anomalous (that's the autoencoder's job). Training on normal
  sequences too would add noise and dilute the class boundaries.

Why use the latent space?
  The latent vector is the autoencoder's compressed representation of the
  sequence. Anomalous sequences that share the same failure mode should cluster
  together in latent space — the MLP learns to separate those clusters.

Datasets supported: BGL (22 anomaly types), Thunderbird (2 anomaly types)
Not applicable to HDFS (binary labels only).

Usage
-----
    # Train and evaluate on BGL:
    python classify.py --dataset bgl

    # Train and evaluate on Thunderbird:
    python classify.py --dataset thunderbird

    # Override paths:
    python classify.py --dataset bgl \
        --checkpoint checkpoints/bgl/best_model.pt \
        --data_dir data/bgl/ \
        --output_dir classifiers/bgl/

Output
------
    classifiers/<dataset>/
        classifier.pt       — trained MLP weights + class metadata
        confusion.png       — confusion matrix heatmap
        per_class.png       — per-category F1 bar chart
"""

import argparse
import json
import logging
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader, TensorDataset

from autoencoder import LSTMAutoencoder
from config import get_config

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
# MLP Classifier
# ---------------------------------------------------------------------------
class AnomalyClassifier(nn.Module):
    """
    Small MLP that maps a latent vector to an anomaly category.

    Architecture: latent_dim → 64 → 32 → n_classes
    Two hidden layers with ReLU + dropout for regularisation.
    Kept deliberately small — the latent space is already compressed
    and the classifier should not overfit on potentially small anomaly sets.
    """
    def __init__(self, latent_dim: int, n_classes: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, n_classes),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Latent vector extraction
# ---------------------------------------------------------------------------
@torch.no_grad()
def extract_latents(model: LSTMAutoencoder, X: np.ndarray,
                    batch_size: int, device: torch.device) -> np.ndarray:
    """
    Run the encoder over all sequences and return latent vectors.

    Args:
        model      : trained LSTMAutoencoder
        X          : (N, T) integer token array
        batch_size : batch size for inference
        device     : torch device

    Returns:
        latents : (N, latent_dim) float array
    """
    model.eval()
    dataset = TensorDataset(torch.from_numpy(X).long())
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_latents = []
    for (x,) in loader:
        x = x.to(device)
        latent, _ = model.encoder(x)
        all_latents.append(latent.cpu())
    return torch.cat(all_latents).numpy()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train_classifier(
    classifier: AnomalyClassifier,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    label_map: dict,
    device: torch.device,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
    patience: int = 10,
) -> AnomalyClassifier:
    """Train the MLP classifier with early stopping on val loss."""

    classifier = classifier.to(device)
    optimizer  = torch.optim.Adam(classifier.parameters(), lr=lr, weight_decay=1e-4)
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    X_tr = torch.tensor(X_train, dtype=torch.float32)
    y_tr = torch.tensor(y_train, dtype=torch.long)
    X_vl = torch.tensor(X_val,   dtype=torch.float32).to(device)
    y_vl = torch.tensor(y_val,   dtype=torch.long).to(device)

    dataset = TensorDataset(X_tr, y_tr)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    best_val_loss = float('inf')
    best_state    = None
    no_improve    = 0

    logger.info(f"Training classifier for up to {epochs} epochs ...")
    logger.info(f"  Train samples : {len(X_train)}")
    logger.info(f"  Val samples   : {len(X_val)}")
    logger.info(f"  Classes       : {len(label_map) - 1} anomaly types")

    for epoch in range(1, epochs + 1):
        classifier.train()
        epoch_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = classifier(xb)
            loss   = F.cross_entropy(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(xb)

        # Validation
        classifier.eval()
        with torch.no_grad():
            val_logits = classifier(X_vl)
            val_loss   = F.cross_entropy(val_logits, y_vl).item()
            val_preds  = val_logits.argmax(dim=1).cpu().numpy()
            val_f1     = f1_score(y_val, val_preds, average='macro', zero_division=0)

        scheduler.step(val_loss)

        if epoch % 10 == 0 or epoch == 1:
            logger.info(
                f"  Epoch {epoch:3d}/{epochs} | "
                f"train_loss: {epoch_loss/len(X_train):.4f} | "
                f"val_loss: {val_loss:.4f} | "
                f"val_macro_f1: {val_f1:.4f}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.clone() for k, v in classifier.state_dict().items()}
            no_improve    = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info(f"  Early stopping at epoch {epoch}")
                break

    classifier.load_state_dict(best_state)
    logger.info(f"  Best val_loss: {best_val_loss:.4f}")
    return classifier


# ---------------------------------------------------------------------------
# Evaluation + plots
# ---------------------------------------------------------------------------
def evaluate_classifier(
    classifier: AnomalyClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    label_map: dict,
    output_dir: str,
    device: torch.device,
) -> None:
    """Evaluate on test set and save confusion matrix + per-class F1 plots."""

    classifier.eval()
    X_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        preds = classifier(X_t).argmax(dim=1).cpu().numpy()

    # Remap label indices to class names for report
    # label_map keys are strings ('1', '2', ...) mapping to category names
    # y_test contains the original integer labels
    unique_labels = sorted(set(y_test.tolist()))
    target_names  = [label_map.get(str(l), f"label_{l}") for l in unique_labels]

    logger.info("=" * 60)
    logger.info("CLASSIFIER TEST RESULTS")
    logger.info("=" * 60)
    report = classification_report(
        y_test, preds,
        labels=unique_labels,
        target_names=target_names,
        zero_division=0,
    )
    for line in report.split('\n'):
        if line.strip():
            logger.info(f"  {line}")
    logger.info("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    # Confusion matrix
    cm = confusion_matrix(y_test, preds, labels=unique_labels)
    fig, ax = plt.subplots(figsize=(max(6, len(unique_labels)), max(5, len(unique_labels) - 2)))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(unique_labels)))
    ax.set_yticks(range(len(unique_labels)))
    ax.set_xticklabels(target_names, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(target_names, fontsize=8)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Anomaly Type Confusion Matrix')
    plt.tight_layout()
    cm_path = os.path.join(output_dir, 'confusion.png')
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Confusion matrix saved → {cm_path}")

    # Per-class F1 bar chart
    from sklearn.metrics import f1_score as f1_per_class
    per_class_f1 = f1_per_class(
        y_test, preds, labels=unique_labels, average=None, zero_division=0
    )
    fig, ax = plt.subplots(figsize=(max(8, len(unique_labels) * 0.6), 4))
    bars = ax.bar(target_names, per_class_f1, color='steelblue', edgecolor='white')
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('F1 Score')
    ax.set_title('Per-Category F1 Score')
    ax.axhline(y=per_class_f1.mean(), color='red', linestyle='--',
               label=f'Macro avg: {per_class_f1.mean():.3f}')
    ax.legend()
    plt.xticks(rotation=45, ha='right', fontsize=8)
    for bar, val in zip(bars, per_class_f1):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{val:.2f}', ha='center', va='bottom', fontsize=7)
    plt.tight_layout()
    f1_path = os.path.join(output_dir, 'per_class_f1.png')
    plt.savefig(f1_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Per-class F1 chart saved → {f1_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # ---- Argument parsing ------------------------------------------------
    parser = argparse.ArgumentParser(
        description="Train anomaly type classifier on autoencoder latent space.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset", type=str, required=True, choices=["bgl", "thunderbird"],
        help="Dataset to classify. HDFS is binary-only and not supported.",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to autoencoder checkpoint. Defaults to checkpoints/<dataset>/best_model.pt",
    )
    parser.add_argument(
        "--data_dir", type=str, default=None,
        help="Path to dataset directory. Defaults to data/<dataset>/",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Where to save classifier checkpoint and plots. Defaults to classifiers/<dataset>/",
    )
    parser.add_argument("--epochs",     type=int,   default=50)
    parser.add_argument("--batch_size", type=int,   default=32)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--patience",   type=int,   default=10)
    parser.add_argument("--seed",       type=int,   default=42)

    # Also parse autoencoder config flags (vocab_size, embed_dim, etc.)
    # We parse known args so extra autoencoder flags don't cause errors
    args, remaining = parser.parse_known_args()

    # Set defaults based on dataset
    if args.checkpoint is None:
        args.checkpoint = f"checkpoints/{args.dataset}/best_model.pt"
    if args.data_dir is None:
        args.data_dir = f"data/{args.dataset}/"
    if args.output_dir is None:
        args.output_dir = f"classifiers/{args.dataset}/"

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ---- Load autoencoder config + checkpoint ----------------------------
    logger.info(f"Loading autoencoder checkpoint: {args.checkpoint}")
    if not os.path.exists(args.checkpoint):
        logger.error(f"Checkpoint not found: {args.checkpoint}")
        logger.error("Has training completed? Run: bash run.sh status")
        sys.exit(1)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    ae_config  = checkpoint["config"]
    logger.info(
        f"Autoencoder config — vocab_size: {ae_config.vocab_size}, "
        f"latent_dim: {ae_config.latent_dim}"
    )

    model = LSTMAutoencoder(ae_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    logger.info("Autoencoder loaded successfully.")

    # ---- Load data -------------------------------------------------------
    logger.info(f"Loading data from: {args.data_dir}")
    X_train = np.load(os.path.join(args.data_dir, "X_train.npy"))
    X_val   = np.load(os.path.join(args.data_dir, "X_val.npy"))
    X_test  = np.load(os.path.join(args.data_dir, "X_test.npy"))
    y_val   = np.load(os.path.join(args.data_dir, "y_val.npy")).astype(int)
    y_test  = np.load(os.path.join(args.data_dir, "y_test.npy")).astype(int)

    # Load label map
    label_map_path = os.path.join(args.data_dir, "label_map.json")
    with open(label_map_path) as f:
        label_map = json.load(f)
    logger.info(f"Label map: {label_map}")

    # ---- Extract latent vectors ------------------------------------------
    logger.info("Extracting latent vectors from autoencoder encoder ...")
    latents_train = extract_latents(model, X_train, args.batch_size, device)
    latents_val   = extract_latents(model, X_val,   args.batch_size, device)
    latents_test  = extract_latents(model, X_test,  args.batch_size, device)
    logger.info(f"Latent shapes — train: {latents_train.shape}, val: {latents_val.shape}, test: {latents_test.shape}")

    # ---- We need labels for train too — use y_val structure --------------
    # Training set has no labels (unsupervised autoencoder training).
    # For the classifier, we use val set for training and test set for eval.
    # This is intentional: val labels are used to train the classifier,
    # test labels are held out for final evaluation.
    #
    # Filter to anomalous sequences only (label > 0)
    train_mask = y_val > 0
    test_mask  = y_test > 0

    X_clf_train = latents_val[train_mask]
    y_clf_train = y_val[train_mask]
    X_clf_test  = latents_test[test_mask]
    y_clf_test  = y_test[test_mask]

    logger.info(f"Classifier training samples (anomalous only): {len(X_clf_train)}")
    logger.info(f"Classifier test samples (anomalous only)    : {len(X_clf_test)}")

    if len(X_clf_train) == 0:
        logger.error("No anomalous samples found in val set. Cannot train classifier.")
        sys.exit(1)

    # ---- Label remapping -------------------------------------------------
    # Map original label integers to contiguous 0-indexed class IDs
    # e.g. [1, 4, 6, 13] → [0, 1, 2, 3]
    unique_classes = sorted(set(y_clf_train.tolist()) | set(y_clf_test.tolist()))
    class_to_idx   = {c: i for i, c in enumerate(unique_classes)}
    idx_to_class   = {i: c for c, i in class_to_idx.items()}

    y_clf_train_idx = np.array([class_to_idx[y] for y in y_clf_train])
    y_clf_test_idx  = np.array([class_to_idx[y] for y in y_clf_test])

    n_classes = len(unique_classes)
    logger.info(f"Number of anomaly classes: {n_classes}")

    # ---- Build + train classifier ----------------------------------------
    # Use a small portion of train for internal val during classifier training
    n_val = max(1, int(len(X_clf_train) * 0.2))
    idx   = np.random.permutation(len(X_clf_train))
    X_int_val, y_int_val = X_clf_train[idx[:n_val]], y_clf_train_idx[idx[:n_val]]
    X_int_tr,  y_int_tr  = X_clf_train[idx[n_val:]], y_clf_train_idx[idx[n_val:]]

    classifier = AnomalyClassifier(
        latent_dim=ae_config.latent_dim,
        n_classes=n_classes,
    )
    classifier = train_classifier(
        classifier, X_int_tr, y_int_tr, X_int_val, y_int_val,
        label_map, device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
    )

    # ---- Evaluate --------------------------------------------------------
    # Remap test labels back for evaluation with original label names
    evaluate_classifier(
        classifier, X_clf_test, y_clf_test_idx,
        {str(class_to_idx[int(k)]): v
         for k, v in label_map.items() if int(k) in class_to_idx},
        args.output_dir, device,
    )

    # ---- Save classifier checkpoint --------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)
    clf_path = os.path.join(args.output_dir, "classifier.pt")
    torch.save({
        "model_state_dict": classifier.state_dict(),
        "latent_dim":       ae_config.latent_dim,
        "n_classes":        n_classes,
        "class_to_idx":     class_to_idx,
        "idx_to_class":     idx_to_class,
        "label_map":        label_map,
        "dataset":          args.dataset,
    }, clf_path)
    logger.info(f"Classifier checkpoint saved → {clf_path}")
    logger.info("Done.")


if __name__ == "__main__":
    main()