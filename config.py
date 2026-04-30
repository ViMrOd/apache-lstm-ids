"""
config.py — Hyperparameter configuration for LSTM Autoencoder log anomaly detector.

All hyperparameters are exposed via argparse so experiments can be reproduced
exactly from the command line and logged to SLURM job output without touching
source code. Defaults are justified in inline comments.

Usage:
    # From Python
    from config import get_config
    args = get_config()

    # From CLI (e.g., override learning rate and epochs)
    python train.py --lr 0.0005 --epochs 50
"""

import argparse


def get_config() -> argparse.Namespace:
    """
    Parse and return all hyperparameters as an argparse Namespace.

    Returns
    -------
    argparse.Namespace
        Parsed arguments; all fields accessible as args.field_name.
    """
    parser = argparse.ArgumentParser(
        description="LSTM Autoencoder for log sequence anomaly detection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,  # prints defaults in --help
    )

    # ------------------------------------------------------------------
    # Data / vocabulary
    # ------------------------------------------------------------------
    data_group = parser.add_argument_group("Data & Vocabulary")

    data_group.add_argument(
        "--vocab_size",
        type=int,
        default=512,
        help=(
            "Number of unique log tokens (set by preprocessing). "
            "512 comfortably covers most system-log vocabularies (HDFS ~30, BGL ~300+) "
            "while leaving headroom for rare tokens. Must match the vocab file produced "
            "by your partner's pipeline."
        ),
    )

    data_group.add_argument(
        "--window_size",
        type=int,
        default=20,
        help=(
            "Number of log tokens per input sequence (sliding-window length). "
            "20 is a common choice in the literature (LogAnomaly, DeepLog) — long enough "
            "to capture multi-step failure patterns, short enough to keep training fast. "
            "Must match the window size used in preprocessing."
        ),
    )

    data_group.add_argument(
        "--padding_idx",
        type=int,
        default=0,
        help="Vocabulary index reserved for <PAD> tokens. 0 is the standard convention.",
    )

    data_group.add_argument(
        "--data_dir",
        type=str,
        default="data/",
        help="Directory containing train/val/test DataLoaders and vocab file.",
    )

    data_group.add_argument(
        "--synthetic",
        action="store_true",
        default=False,
        help=(
            "Use synthetic random data instead of real .npy files. "
            "Useful for smoke-testing the training loop without needing "
            "the preprocessed dataset. When False (default), --data_dir "
            "must contain X_train.npy, X_val.npy, y_val.npy, "
            "X_test.npy, y_test.npy."
        ),
    )

    # ------------------------------------------------------------------
    # Model architecture
    # ------------------------------------------------------------------
    model_group = parser.add_argument_group("Model Architecture")

    model_group.add_argument(
        "--embed_dim",
        type=int,
        default=64,
        help=(
            "Dimensionality of the token embedding layer. "
            "64 gives a reasonable dense representation for a vocab of ~512 without "
            "over-parameterising the model for a course-project-scale dataset. "
            "Rule of thumb: embed_dim ≈ vocab_size^(1/4) × 16."
        ),
    )

    model_group.add_argument(
        "--hidden_dim",
        type=int,
        default=128,
        help=(
            "Number of hidden units in each LSTM layer (encoder and decoder share this). "
            "128 is large enough to compress a 20-token sequence meaningfully while "
            "keeping the bottleneck tight — too large and the autoencoder memorises "
            "rather than generalises."
        ),
    )

    model_group.add_argument(
        "--num_layers",
        type=int,
        default=2,
        help=(
            "Number of stacked LSTM layers in the encoder (and mirrored in the decoder). "
            "2 layers capture more abstract temporal patterns than 1, while 3+ layers "
            "risk vanishing gradients on short sequences like ours."
        ),
    )

    model_group.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help=(
            "Dropout probability applied between LSTM layers (ignored when num_layers=1). "
            "0.2 is a light regulariser — heavier dropout (>0.5) hurts sequence models "
            "because it disrupts temporal dependencies."
        ),
    )

    model_group.add_argument(
        "--latent_dim",
        type=int,
        default=32,
        help=(
            "Dimensionality of the bottleneck (encoder's final hidden state projected "
            "down before being fed to the decoder). Forcing a narrower bottleneck "
            "encourages the model to learn compact normal-behaviour representations; "
            "anomalies then reconstruct poorly, raising the loss."
        ),
    )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    train_group = parser.add_argument_group("Training")

    train_group.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help=(
            "Number of sequences per mini-batch. "
            "64 fits comfortably in GPU memory for our sequence length and hidden dim, "
            "and gives stable gradient estimates. Halve this if you hit OOM on OSC."
        ),
    )

    train_group.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help=(
            "Initial learning rate for Adam. "
            "1e-3 is the canonical Adam default (Kingma & Ba, 2015) and works well "
            "for LSTMs on sequence reconstruction tasks. Decay is handled by the "
            "scheduler in train.py."
        ),
    )

    train_group.add_argument(
        "--epochs",
        type=int,
        default=30,
        help=(
            "Maximum number of training epochs. "
            "30 is usually sufficient for convergence on log datasets of this scale; "
            "early stopping (patience=5) in train.py will terminate sooner if needed."
        ),
    )

    train_group.add_argument(
        "--weight_decay",
        type=float,
        default=1e-5,
        help=(
            "L2 regularisation coefficient for Adam (AdamW-style). "
            "Small value to penalise large weights without interfering with learning."
        ),
    )

    train_group.add_argument(
        "--clip_grad_norm",
        type=float,
        default=1.0,
        help=(
            "Max norm for gradient clipping. "
            "LSTMs are prone to exploding gradients; clipping at 1.0 is standard "
            "practice (Pascanu et al., 2013)."
        ),
    )

    train_group.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (PyTorch, NumPy, Python random).",
    )

    train_group.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Early stopping patience: epochs with no val loss improvement before stopping.",
    )

    # ------------------------------------------------------------------
    # Anomaly detection
    # ------------------------------------------------------------------
    anomaly_group = parser.add_argument_group("Anomaly Detection")

    anomaly_group.add_argument(
        "--anomaly_threshold",
        type=float,
        default=None,
        help=(
            "Reconstruction-loss threshold above which a sequence is flagged as anomalous. "
            "Default None means the threshold is selected automatically during evaluation "
            "by maximising F1 on the validation set — recommended over a hand-picked value. "
            "Pass an explicit float (e.g. 0.05) to fix the threshold for inference."
        ),
    )

    anomaly_group.add_argument(
        "--threshold_percentile",
        type=float,
        default=95.0,
        help=(
            "If --anomaly_threshold is None, use this percentile of validation-set "
            "reconstruction losses as the starting search point. "
            "95th percentile assumes ~5%% of validation windows are anomalous."
        ),
    )

    # ------------------------------------------------------------------
    # Paths & logging
    # ------------------------------------------------------------------
    io_group = parser.add_argument_group("Paths & Logging")

    io_group.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints/",
        help="Directory to save model checkpoints.",
    )

    io_group.add_argument(
        "--log_dir",
        type=str,
        default="runs/",
        help="TensorBoard log directory.",
    )

    io_group.add_argument(
        "--vocab_file",
        type=str,
        default="data/vocab.json",
        help="Path to the vocabulary JSON file produced by the preprocessing pipeline.",
    )

    io_group.add_argument(
        "--save_every",
        type=int,
        default=5,
        help="Save a checkpoint every N epochs (in addition to best-val-loss checkpoint).",
    )

    io_group.add_argument(
        "--plot_dir",
        type=str,
        default="plots/",
        help="Directory to save roc_curve.png and score_distribution.png.",
    )

    io_group.add_argument(
        "--threshold_sweep_steps",
        type=int,
        default=200,
        help="Number of candidate thresholds to evaluate when sweeping for best F1.",
    )

    io_group.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_model.pt",
        help="Path to the model checkpoint to evaluate.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Quick self-test: python config.py --help  or  python config.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    args = get_config()
    print("=== Parsed Configuration ===")
    for key, value in sorted(vars(args).items()):
        print(f"  {key:<25} = {value}")
