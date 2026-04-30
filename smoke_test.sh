#!/bin/bash
# ===========================================================================
# smoke_test.sh — Fast end-to-end pipeline validation on OSC (real data)
# ===========================================================================
#
# Runs the FULL pipeline (train → evaluate) in ~5 min using:
#   - Real .npy files from your partner's preprocessing pipeline
#   - A small random slice of the data (N_TRAIN / N_VAL / N_TEST below)
#   - Tiny model (1 LSTM layer, small dims) — just enough to confirm no errors
#   - 3 epochs, then straight into evaluate.py
#   - Single GPU (avoids CPU-only slowness while keeping queue wait short)
#
# Purpose: catch dtype mismatches, wrong vocab_size, shape bugs, checkpoint
#          save/load errors, and plotting failures BEFORE burning a 4-hr slot.
#
# Recommended workflow:
#   1. Partner drops X_train.npy / X_val.npy / y_val.npy /
#      X_test.npy / y_test.npy into data/
#   2. sbatch smoke_test.sh
#      OR for instant feedback:
#      bash run.sh interactive
#      bash smoke_test.sh
#   3. Check logs/smoke_<JOBID>.log — look for PASSED / FAILED at the bottom
#   4. If all green: bash run.sh train-eval
#
# Monitor : squeue -u $USER
# Logs    : logs/smoke_<JOBID>.log
# ===========================================================================

#SBATCH --job-name=lstm_ae_smoke
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-node=1
#SBATCH --mem=8G
#SBATCH --output=logs/smoke_%j.log
#SBATCH --error=logs/smoke_%j.log
#SBATCH --mail-type=END,FAIL
# --account and --mail-user are passed by run.sh via CLI flags from .env

set -euo pipefail

# ---------------------------------------------------------------------------
# Load credentials and paths from .env
# ---------------------------------------------------------------------------
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ENV_FILE="$SCRIPT_DIR/.env"

if [[ ! -f "$ENV_FILE" ]]; then
    echo "ERROR: .env file not found."
    echo "       Run:  cp .env.example .env"
    echo "       Then fill in your OSC credentials in .env"
    exit 1
fi

# shellcheck disable=SC1090
source "$ENV_FILE"

for var in CONDA_ENV MODULE_CONDA MODULE_CUDA; do
    if [[ -z "${!var:-}" ]]; then
        echo "ERROR: $var is not set in .env"
        exit 1
    fi
done

# ---------------------------------------------------------------------------
# User config
# ---------------------------------------------------------------------------
DATA_DIR="${SCRIPT_DIR}/data"
SMOKE_DIR="${SCRIPT_DIR}/smoke_out"

# Subset sizes — small enough to finish in minutes on real data
N_TRAIN=500
N_VAL=200
N_TEST=200

# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------
echo "============================================================"
echo "  LSTM-AE Smoke Test — Real Data Subset"
echo "  Job ID : ${SLURM_JOB_ID:-local}"
echo "  Node   : ${SLURMD_NODENAME:-$(hostname)}"
echo "  Start  : $(date)"
echo "============================================================"
echo ""

# ---------------------------------------------------------------------------
# Load modules and activate conda
# ---------------------------------------------------------------------------
module purge
module load "$MODULE_CONDA"
module load "$MODULE_CUDA"

# shellcheck disable=SC1090
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

echo "Conda env : $CONDA_DEFAULT_ENV"
echo "Python    : $(which python)"
echo ""

# ---------------------------------------------------------------------------
# Check real data files exist before wasting queue time
# ---------------------------------------------------------------------------
echo "--- Checking source data files ---"
MISSING=0
for f in X_train.npy X_val.npy y_val.npy X_test.npy y_test.npy; do
    if [[ -f "$DATA_DIR/$f" ]]; then
        echo "  FOUND   $DATA_DIR/$f  ($(du -sh "$DATA_DIR/$f" | cut -f1))"
    else
        echo "  MISSING $DATA_DIR/$f"
        MISSING=1
    fi
done

if [[ $MISSING -eq 1 ]]; then
    echo ""
    echo "ERROR: One or more .npy files are missing from $DATA_DIR/."
    echo "       Get the files from your partner and resubmit."
    exit 1
fi
echo ""

# ---------------------------------------------------------------------------
# Prepare isolated smoke output dirs
# ---------------------------------------------------------------------------
mkdir -p logs "$SMOKE_DIR/data" "$SMOKE_DIR/checkpoints" \
         "$SMOKE_DIR/runs"      "$SMOKE_DIR/plots"

# ---------------------------------------------------------------------------
# Slice & validate real data
# ---------------------------------------------------------------------------
echo "--- Slicing and validating real data ---"

export DATA_DIR SMOKE_DIR N_TRAIN N_VAL N_TEST

python - <<'PYEOF'
import os, sys
import numpy as np

DATA_DIR  = os.environ["DATA_DIR"]
SMOKE_DIR = os.environ["SMOKE_DIR"]
N_TRAIN   = int(os.environ["N_TRAIN"])
N_VAL     = int(os.environ["N_VAL"])
N_TEST    = int(os.environ["N_TEST"])

def load_check(path, name, ndim, is_label=False):
    arr = np.load(path)
    print(f"  {name:<16} shape={arr.shape}  dtype={arr.dtype}  "
          f"min={arr.min()}  max={arr.max()}")
    if arr.ndim != ndim:
        print(f"  ERROR: expected {ndim}D, got {arr.ndim}D"); sys.exit(1)
    if is_label:
        bad = set(np.unique(arr).tolist()) - {0, 1}
        if bad:
            print(f"  ERROR: label values outside {{0,1}}: {bad}"); sys.exit(1)
    if arr.min() < 0:
        print(f"  ERROR: negative token indices found in {name}"); sys.exit(1)
    return arr

X_train = load_check(f"{DATA_DIR}/X_train.npy", "X_train", 2)
X_val   = load_check(f"{DATA_DIR}/X_val.npy",   "X_val",   2)
y_val   = load_check(f"{DATA_DIR}/y_val.npy",   "y_val",   1, is_label=True)
X_test  = load_check(f"{DATA_DIR}/X_test.npy",  "X_test",  2)
y_test  = load_check(f"{DATA_DIR}/y_test.npy",  "y_test",  1, is_label=True)

T = X_train.shape[1]
print(f"\n  Detected window_size T = {T}")

for Xname, X, yname, y in [("X_val", X_val, "y_val", y_val),
                             ("X_test", X_test, "y_test", y_test)]:
    if X.shape[0] != y.shape[0]:
        print(f"  ERROR: {Xname} ({X.shape[0]}) and {yname} ({y.shape[0]}) length mismatch")
        sys.exit(1)
    if X.shape[1] != T:
        print(f"  ERROR: {Xname} window size {X.shape[1]} != X_train window size {T}")
        sys.exit(1)

rng = np.random.default_rng(42)
train_idx = rng.choice(len(X_train), min(N_TRAIN, len(X_train)), replace=False)
val_idx   = rng.choice(len(X_val),   min(N_VAL,   len(X_val)),   replace=False)
test_idx  = rng.choice(len(X_test),  min(N_TEST,  len(X_test)),  replace=False)

Xs_train = X_train[train_idx]
Xs_val,  ys_val  = X_val[val_idx],   y_val[val_idx]
Xs_test, ys_test = X_test[test_idx], y_test[test_idx]

np.save(f"{SMOKE_DIR}/data/X_train.npy", Xs_train)
np.save(f"{SMOKE_DIR}/data/X_val.npy",   Xs_val)
np.save(f"{SMOKE_DIR}/data/y_val.npy",   ys_val)
np.save(f"{SMOKE_DIR}/data/X_test.npy",  Xs_test)
np.save(f"{SMOKE_DIR}/data/y_test.npy",  ys_test)

print(f"\n  Subsets saved to {SMOKE_DIR}/data/")
print(f"    X_train : {Xs_train.shape}")
print(f"    X_val   : {Xs_val.shape}   anomaly rate = {ys_val.mean():.1%}")
print(f"    X_test  : {Xs_test.shape}  anomaly rate = {ys_test.mean():.1%}")

if ys_val.mean() == 0.0 or ys_test.mean() == 0.0:
    print("\n  WARNING: anomaly rate is 0% in val or test slice.")
    print("           ROC-AUC and F1 will be undefined. Try larger N_VAL/N_TEST.")

with open(f"{SMOKE_DIR}/window_size.txt", "w") as f:
    f.write(str(T))
PYEOF

WINDOW_SIZE=$(cat "$SMOKE_DIR/window_size.txt")
echo ""
echo "  window_size = $WINDOW_SIZE  (auto-detected from X_train.npy)"
echo ""

# Pull vocab size from vocab.json
VOCAB_FILE="$DATA_DIR/vocab.json"
if [[ -f "$VOCAB_FILE" ]]; then
    VOCAB_SIZE=$(python -c "
import json
with open('${VOCAB_FILE}') as f:
    v = json.load(f)
print(len(v))
")
    echo "  vocab_size  = $VOCAB_SIZE  (auto-detected from vocab.json)"
else
    VOCAB_SIZE=512
    echo "  WARNING: vocab.json not found — using VOCAB_SIZE=$VOCAB_SIZE"
fi
echo ""

# ---------------------------------------------------------------------------
# Step 1: Train — 3 epochs, tiny model, real subset
# ---------------------------------------------------------------------------
echo "--- Step 1: Training (3 epochs) ---"
echo ""

python "$SCRIPT_DIR/train.py" \
    --data_dir          "$SMOKE_DIR/data/"        \
    --vocab_file        "$VOCAB_FILE"              \
    --checkpoint_dir    "$SMOKE_DIR/checkpoints/"  \
    --log_dir           "$SMOKE_DIR/runs/"          \
    --window_size       "$WINDOW_SIZE"              \
    --vocab_size        "$VOCAB_SIZE"               \
    --embed_dim         32                          \
    --hidden_dim        64                          \
    --latent_dim        16                          \
    --num_layers        1                           \
    --dropout           0.0                         \
    --batch_size        32                          \
    --lr                1e-3                        \
    --weight_decay      0                           \
    --clip_grad_norm    1.0                         \
    --epochs            3                           \
    --patience          99                          \
    --save_every        1                           \
    --seed              42

echo ""
if [[ -f "$SMOKE_DIR/checkpoints/best_model.pt" ]]; then
    echo "  Checkpoint: $(du -sh "$SMOKE_DIR/checkpoints/best_model.pt")"
else
    echo "  ERROR: best_model.pt not found — training failed."
    exit 1
fi
echo ""

# ---------------------------------------------------------------------------
# Step 2: Evaluate
# ---------------------------------------------------------------------------
echo "--- Step 2: Evaluating ---"
echo ""

python "$SCRIPT_DIR/evaluate.py" \
    --checkpoint            "$SMOKE_DIR/checkpoints/best_model.pt" \
    --data_dir              "$SMOKE_DIR/data/"                      \
    --vocab_file            "$VOCAB_FILE"                           \
    --plot_dir              "$SMOKE_DIR/plots/"                     \
    --window_size           "$WINDOW_SIZE"                          \
    --vocab_size            "$VOCAB_SIZE"                           \
    --embed_dim             32                                      \
    --hidden_dim            64                                      \
    --latent_dim            16                                      \
    --num_layers            1                                       \
    --dropout               0.0                                     \
    --batch_size            32                                      \
    --threshold_sweep_steps 50                                      \
    --seed                  42

echo ""

# ---------------------------------------------------------------------------
# Final report
# ---------------------------------------------------------------------------
echo "============================================================"
echo "  Smoke Test Results — $(date)"
echo ""

ALL_PASSED=1
for f in \
    "$SMOKE_DIR/checkpoints/best_model.pt" \
    "$SMOKE_DIR/plots/roc_curve.png" \
    "$SMOKE_DIR/plots/score_distribution.png"; do
    if [[ -f "$f" ]]; then
        echo "  PASSED  $f  ($(du -sh "$f" | cut -f1))"
    else
        echo "  FAILED  $f  (not produced)"
        ALL_PASSED=0
    fi
done

echo ""
if [[ $ALL_PASSED -eq 1 ]]; then
    echo "  All checks passed. Pipeline is healthy."
    echo "  Next step: bash run.sh train-eval"
else
    echo "  One or more outputs are missing — check the log above for errors."
fi
echo "============================================================"