#!/bin/bash
# ===========================================================================
# train_osc.sh — SLURM job script for LSTM Autoencoder training on OSC
# ===========================================================================
#
# Submit via run.sh (recommended — handles credentials from .env):
#   bash run.sh train
#
# Or submit directly (account and email passed by run.sh via CLI):
#   sbatch --account=<YOUR_ACCOUNT> --mail-user=<YOUR_EMAIL> train_osc.sh
#
# Monitor : squeue -u $USER
# Cancel  : scancel <JOBID>
# Logs    : logs/lstm_ae_train_<JOBID>.log
# ===========================================================================

# ---------------------------------------------------------------------------
# SLURM resource directives
# --account and --mail-user are intentionally omitted here — they are
# passed by run.sh via CLI flags sourced from .env, keeping credentials
# out of version control.
# ---------------------------------------------------------------------------
#SBATCH --job-name=lstm_ae_train
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log
#SBATCH --mail-type=BEGIN,END,FAIL

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

# ---------------------------------------------------------------------------
# Job banner
# ---------------------------------------------------------------------------
echo "=========================================================="
echo "  OSC LSTM Autoencoder — Training Job"
echo "=========================================================="
echo "  Job ID      : $SLURM_JOB_ID"
echo "  Job name    : $SLURM_JOB_NAME"
echo "  Node        : $SLURMD_NODENAME"
echo "  CPUs        : $SLURM_CPUS_PER_TASK"
echo "  Start time  : $(date)"
echo "  Working dir : $PROJECT_DIR"
echo "=========================================================="

# ---------------------------------------------------------------------------
# Load OSC modules
# ---------------------------------------------------------------------------
module purge
module load "$MODULE_CONDA"
module load "$MODULE_CUDA"

echo ""
echo "--- Environment ---"
echo "Python      : $(which python)"
echo "CUDA visible: $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'torch not yet loaded')"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null \
    && echo "" || echo "(nvidia-smi unavailable)"

# ---------------------------------------------------------------------------
# Activate conda environment
# ---------------------------------------------------------------------------
conda activate "$CONDA_ENV"
echo "Conda env   : $CONDA_DEFAULT_ENV"
echo "-------------------"
echo ""

# ---------------------------------------------------------------------------
# Navigate to project root and create expected directories
# ---------------------------------------------------------------------------
cd "$PROJECT_DIR"
mkdir -p logs checkpoints/hdfs runs/hdfs

# ---------------------------------------------------------------------------
# Launch training
# ---------------------------------------------------------------------------
echo "Starting training ..."
echo ""

python train.py \
    --data_dir          data/             \
    --vocab_file        data/vocab.json   \
    --checkpoint_dir    checkpoints/hdfs/      \
    --log_dir           runs/hdfs/             \
    --vocab_size        47                \
    --window_size       20                \
    --embed_dim         64                \
    --hidden_dim        128               \
    --latent_dim        32                \
    --num_layers        2                 \
    --dropout           0.2               \
    --batch_size        64                \
    --lr                1e-3              \
    --weight_decay      1e-5              \
    --clip_grad_norm    1.0               \
    --epochs            30                \
    --patience          5                 \
    --save_every        5                 \
    --seed              42

EXIT_CODE=$?

# ---------------------------------------------------------------------------
# Job footer
# ---------------------------------------------------------------------------
echo ""
echo "=========================================================="
echo "  Training complete"
echo "  Exit code  : $EXIT_CODE"
echo "  End time   : $(date)"
echo "  Checkpoint : checkpoints/hdfs/best_model.pt"
echo "=========================================================="

exit $EXIT_CODE