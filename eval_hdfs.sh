#!/bin/bash
# ===========================================================================
# eval_osc.sh — SLURM job script for LSTM Autoencoder evaluation on OSC
# ===========================================================================
#
# Submit via run.sh (recommended — handles credentials from .env):
#   bash run.sh eval
#
# Or chain automatically after training:
#   bash run.sh train-eval
#
# Or submit directly:
#   sbatch --account=<YOUR_ACCOUNT> --mail-user=<YOUR_EMAIL> eval_osc.sh
#
# Monitor : squeue -u $USER
# Logs    : logs/lstm_ae_eval_<JOBID>.log
# ===========================================================================

# ---------------------------------------------------------------------------
# SLURM resource directives
# --account and --mail-user are intentionally omitted here — they are
# passed by run.sh via CLI flags sourced from .env, keeping credentials
# out of version control.
# ---------------------------------------------------------------------------
#SBATCH --job-name=lstm_ae_eval
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-node=1
#SBATCH --mem=16G
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log
#SBATCH --mail-type=END,FAIL

set -euo pipefail

# ---------------------------------------------------------------------------
# Load credentials and paths from .env
# ---------------------------------------------------------------------------
ENV_FILE="$(dirname "${BASH_SOURCE[0]}")/.env"

if [[ ! -f "$ENV_FILE" ]]; then
    echo "ERROR: .env file not found."
    echo "       Run:  cp .env.example .env"
    echo "       Then fill in your OSC credentials in .env"
    exit 1
fi

# shellcheck disable=SC1090
source "$ENV_FILE"

PROJECT_DIR="${PROJECT_DIR:-$SLURM_SUBMIT_DIR}"
CHECKPOINT="checkpoints/hdfs/best_model.pt"

# ---------------------------------------------------------------------------
# Job banner
# ---------------------------------------------------------------------------
echo "=========================================================="
echo "  OSC LSTM Autoencoder — Evaluation Job"
echo "=========================================================="
echo "  Job ID      : $SLURM_JOB_ID"
echo "  Node        : $SLURMD_NODENAME"
echo "  Start time  : $(date)"
echo "  Checkpoint  : $CHECKPOINT"
echo "  Working dir : $PROJECT_DIR"
echo "=========================================================="

# ---------------------------------------------------------------------------
# Load OSC modules (must match versions used during training)
# ---------------------------------------------------------------------------
module purge
module load "$MODULE_CONDA"
module load "$MODULE_CUDA"

# ---------------------------------------------------------------------------
# Activate conda environment
# ---------------------------------------------------------------------------
conda activate "$CONDA_ENV"
echo ""
echo "Conda env   : $CONDA_DEFAULT_ENV"
echo ""

# ---------------------------------------------------------------------------
# Navigate to project root and create output directories
# ---------------------------------------------------------------------------
cd "$PROJECT_DIR"
mkdir -p logs plots

# Verify the checkpoint exists before wasting queue time
if [[ ! -f "$CHECKPOINT" ]]; then
    echo "ERROR: Checkpoint not found at '$CHECKPOINT'"
    echo "       Did the training job complete successfully?"
    echo "       Check: ls -lh checkpoints/"
    exit 1
fi

echo "Checkpoint size : $(du -sh "$CHECKPOINT" | cut -f1)"
echo ""

# ---------------------------------------------------------------------------
# Run evaluation
#
# --anomaly_threshold is omitted so evaluate.py sweeps thresholds on the
# val set and picks the one that maximises F1 — recommended over a fixed value.
# To fix the threshold instead, add:  --anomaly_threshold 0.05
#
# Architecture flags must exactly match those used in train_osc.sh.
# ---------------------------------------------------------------------------
echo "Starting evaluation ..."
echo ""

python evaluate.py \
    --checkpoint            "$CHECKPOINT"   \
    --data_dir              data/           \
    --vocab_file            data/vocab.json \
    --plot_dir              plots/hdfs/          \
    --vocab_size            47              \
    --window_size           20              \
    --embed_dim             64              \
    --hidden_dim            128             \
    --latent_dim            32              \
    --num_layers            2               \
    --dropout               0.2             \
    --batch_size            64              \
    --threshold_sweep_steps 200             \
    --seed                  42              \
    --padding_idx           0

EXIT_CODE=$?

# ---------------------------------------------------------------------------
# Summarise outputs
# ---------------------------------------------------------------------------
echo ""
echo "=========================================================="
echo "  Evaluation complete"
echo "  Exit code : $EXIT_CODE"
echo "  End time  : $(date)"
echo ""
echo "  Output plots:"
for f in plots/hdfs/roc_curve.png plots/hdfs/score_distribution.png; do
    if [[ -f "$f" ]]; then
        echo "    ✓  $f  ($(du -sh "$f" | cut -f1))"
    else
        echo "    ✗  $f  (not found)"
    fi
done
echo "=========================================================="

exit $EXIT_CODE