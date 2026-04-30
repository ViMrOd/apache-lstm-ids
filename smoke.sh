#!/bin/bash
# ===========================================================================
# smoke.sh — Interactive smoke test for LSTM Autoencoder on OSC
# ===========================================================================
#
# USAGE
# -----
# Two modes depending on where you run it:
#
#   1. Login node (synthetic data, CPU only, ~30 seconds):
#        bash smoke.sh
#
#   2. Inside an sinteractive session (real data, GPU, ~2-3 minutes):
#        bash run.sh interactive
#        bash smoke.sh --real
#
# FLAGS
# -----
#   --real          Use real .npy files from DATA_DIR instead of synthetic data
#   --skip-eval     Skip the evaluation step (train only)
#   --no-cleanup    Keep /tmp checkpoint and plot files after the run
#   --help          Print this message and exit
#
# WHAT THIS TESTS
# ---------------
#   ✓ All imports and package availability
#   ✓ Config parsing and argument forwarding
#   ✓ Full train → checkpoint → evaluate pipeline
#   ✓ Shape correctness through encoder + decoder
#   ✓ Loss collapse (if loss hits 0.000000 in epoch 1-2, embedding fix needed)
#   ✓ Plot generation (roc_curve.png, score_distribution.png)
#   ✓ Real data loading and .npy shape compatibility (with --real)
#
# ===========================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Load credentials and paths from .env
# ---------------------------------------------------------------------------
PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ENV_FILE="$PROJECT_DIR/.env"

if [[ ! -f "$ENV_FILE" ]]; then
    echo "ERROR: .env file not found."
    echo "       Run:  cp .env.example .env"
    echo "       Then fill in your OSC credentials in .env"
    exit 1
fi

# shellcheck disable=SC1090
source "$ENV_FILE"

for var in OSC_ACCOUNT CONDA_ENV MODULE_CONDA MODULE_CUDA; do
    if [[ -z "${!var:-}" ]]; then
        echo "ERROR: $var is not set in .env"
        exit 1
    fi
done

# ---------------------------------------------------------------------------
# Defaults (smoke-test hyperparameters — intentionally tiny for speed)
# ---------------------------------------------------------------------------
DATA_DIR="$PROJECT_DIR/data/"
VOCAB_FILE="$DATA_DIR/vocab.json"

VOCAB_SIZE=50        # overridden to real vocab size when --real is set
WINDOW_SIZE=10       # overridden to match preprocessing when --real is set
HIDDEN_DIM=32
LATENT_DIM=8
NUM_LAYERS=1
DROPOUT=0.0
BATCH_SIZE=16
EPOCHS=3
PATIENCE=3

# Temp output dirs (cleaned up on exit unless --no-cleanup)
CKPT_DIR="/tmp/smoke_ckpt_$$"
LOG_DIR="/tmp/smoke_runs_$$"
PLOT_DIR="/tmp/smoke_plots_$$"

# ---------------------------------------------------------------------------
# Flag parsing
# ---------------------------------------------------------------------------
USE_REAL=false
SKIP_EVAL=false
CLEANUP=true

for arg in "$@"; do
    case $arg in
        --real)       USE_REAL=true ;;
        --skip-eval)  SKIP_EVAL=true ;;
        --no-cleanup) CLEANUP=false ;;
        --help)
            sed -n '3,40p' "$0"
            exit 0
            ;;
        *)
            echo "Unknown flag: $arg  (run with --help for usage)"
            exit 1
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Cleanup trap
# ---------------------------------------------------------------------------
cleanup() {
    if $CLEANUP; then
        rm -rf "$CKPT_DIR" "$LOG_DIR" "$PLOT_DIR"
        echo ""
        echo "Temp files cleaned up. (Pass --no-cleanup to keep them.)"
    else
        echo ""
        echo "Temp files kept:"
        echo "  Checkpoints : $CKPT_DIR"
        echo "  TensorBoard : $LOG_DIR"
        echo "  Plots       : $PLOT_DIR"
    fi
}
trap cleanup EXIT

# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------
echo "=========================================================="
echo "  LSTM Autoencoder — Smoke Test"
echo "=========================================================="
echo "  Mode        : $( $USE_REAL && echo 'REAL data (GPU expected)' || echo 'SYNTHETIC data (CPU)' )"
echo "  Project dir : $PROJECT_DIR"
echo "  Conda env   : $CONDA_ENV"
echo "  Date        : $(date)"
echo "=========================================================="
echo ""

# ---------------------------------------------------------------------------
# Environment check
# ---------------------------------------------------------------------------
echo ">>> Checking environment ..."

if [[ ! -f "$PROJECT_DIR/train.py" ]]; then
    echo "ERROR: train.py not found in $PROJECT_DIR"
    exit 1
fi

# Load modules and activate conda if not already in the right environment
if [[ "${CONDA_DEFAULT_ENV:-}" != "$CONDA_ENV" ]]; then
    echo "    Loading modules and activating conda env: $CONDA_ENV"
    module purge
    module load "$MODULE_CONDA"
    module load "$MODULE_CUDA"
    # shellcheck disable=SC1090
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
fi

echo "    Python      : $(which python)"
echo "    Torch       : $(python -c 'import torch; print(torch.__version__)')"
echo "    CUDA        : $(python -c 'import torch; print(torch.cuda.is_available())')"
if python -c 'import torch; exit(0 if torch.cuda.is_available() else 1)' 2>/dev/null; then
    echo "    GPU         : $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
fi
echo ""

# ---------------------------------------------------------------------------
# Real-data mode: validate files and pull real vocab/window sizes
# ---------------------------------------------------------------------------
if $USE_REAL; then
    echo ">>> Checking real data files ..."

    MISSING=false
    for f in X_train.npy X_val.npy y_val.npy X_test.npy y_test.npy; do
        if [[ -f "$DATA_DIR/$f" ]]; then
            SIZE=$(du -sh "$DATA_DIR/$f" | cut -f1)
            echo "    ✓  $f  ($SIZE)"
        else
            echo "    ✗  $f  — NOT FOUND"
            MISSING=true
        fi
    done

    if $MISSING; then
        echo ""
        echo "ERROR: One or more required data files are missing from $DATA_DIR"
        echo "       Run without --real to use synthetic data instead."
        exit 1
    fi

    # Pull actual window size from the .npy shape so args stay consistent
    WINDOW_SIZE=$(python -c "
import numpy as np
x = np.load('${DATA_DIR}/X_train.npy', mmap_mode='r')
print(x.shape[1])
")
    echo ""
    echo "    Detected window_size = $WINDOW_SIZE from X_train.npy"

    # Pull vocab size from vocab.json if available
    if [[ -f "$VOCAB_FILE" ]]; then
        VOCAB_SIZE=$(python -c "
import json
with open('${VOCAB_FILE}') as f:
    v = json.load(f)
print(len(v))
")
        echo "    Detected vocab_size  = $VOCAB_SIZE from vocab.json"
    else
        echo "    WARNING: vocab.json not found — using VOCAB_SIZE=$VOCAB_SIZE (may be wrong)"
    fi
    echo ""
fi

# ---------------------------------------------------------------------------
# Build shared argument string
# ---------------------------------------------------------------------------
COMMON_ARGS=(
    --vocab_size        "$VOCAB_SIZE"
    --window_size       "$WINDOW_SIZE"
    --hidden_dim        "$HIDDEN_DIM"
    --latent_dim        "$LATENT_DIM"
    --num_layers        "$NUM_LAYERS"
    --dropout           "$DROPOUT"
    --batch_size        "$BATCH_SIZE"
    --embed_dim         64
    --checkpoint_dir    "$CKPT_DIR"
    --log_dir           "$LOG_DIR"
)

if $USE_REAL; then
    COMMON_ARGS+=(--data_dir "$DATA_DIR" --vocab_file "$VOCAB_FILE")
else
    COMMON_ARGS+=(--synthetic)
fi

mkdir -p "$CKPT_DIR" "$LOG_DIR" "$PLOT_DIR"

# ---------------------------------------------------------------------------
# Step 1: Train
# ---------------------------------------------------------------------------
echo ">>> Running training smoke test ..."
echo "    epochs=$EPOCHS  batch=$BATCH_SIZE  hidden=$HIDDEN_DIM  latent=$LATENT_DIM"
echo ""

TRAIN_START=$(date +%s)

python "$PROJECT_DIR/train.py" \
    "${COMMON_ARGS[@]}" \
    --epochs         "$EPOCHS" \
    --patience       "$PATIENCE" \
    --save_every     1 \
    --lr             1e-3 \
    --weight_decay   1e-5 \
    --clip_grad_norm 1.0 \
    --seed           42

TRAIN_END=$(date +%s)
TRAIN_ELAPSED=$(( TRAIN_END - TRAIN_START ))

echo ""
echo "    Training finished in ${TRAIN_ELAPSED}s"

# Warn on loss collapse
FINAL_LOSS=$(python -c "
import torch
ckpt = torch.load('$CKPT_DIR/best_model.pt', map_location='cpu')
print(f\"{ckpt['val_loss']:.8f}\")
")
echo "    Best val_loss from checkpoint: $FINAL_LOSS"

if python -c "exit(0 if float('$FINAL_LOSS') < 1e-6 else 1)"; then
    echo ""
    echo "  WARNING: val_loss collapsed to ~0. The embedding shortcut is active."
    echo "     Apply the embedding freeze fix before your next real training run:"
    echo "       self.embedding.weight.requires_grad = False"
    echo ""
fi

# ---------------------------------------------------------------------------
# Step 2: Evaluate
# ---------------------------------------------------------------------------
if $SKIP_EVAL; then
    echo ""
    echo ">>> Skipping evaluation (--skip-eval set)."
else
    echo ""
    echo ">>> Running evaluation smoke test ..."
    echo ""

    EVAL_START=$(date +%s)

    python "$PROJECT_DIR/evaluate.py" \
        "${COMMON_ARGS[@]}" \
        --checkpoint            "$CKPT_DIR/best_model.pt" \
        --plot_dir              "$PLOT_DIR" \
        --threshold_sweep_steps 50 \
        --seed                  42

    EVAL_END=$(date +%s)
    EVAL_ELAPSED=$(( EVAL_END - EVAL_START ))
    echo ""
    echo "    Evaluation finished in ${EVAL_ELAPSED}s"

    echo ""
    echo "    Output plots:"
    for f in roc_curve.png score_distribution.png; do
        if [[ -f "$PLOT_DIR/$f" ]]; then
            echo "    ✓  $PLOT_DIR/$f"
        else
            echo "    ✗  $PLOT_DIR/$f  (not generated)"
        fi
    done
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "=========================================================="
echo "  Smoke test PASSED"
echo "  Total time : $(( $(date +%s) - TRAIN_START ))s"
echo ""
echo "  Next steps:"
if $USE_REAL; then
    echo "   • If loss collapsed → freeze embeddings in autoencoder.py"
    echo "   • If all looks good → bash run.sh train-eval"
else
    echo "   • Re-run with --real inside an sinteractive session to test"
    echo "     against your actual data before submitting to the queue"
    echo "   • bash run.sh interactive"
fi
echo "=========================================================="