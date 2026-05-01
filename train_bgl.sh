#!/bin/bash
# ===========================================================================
# train_bgl.sh — SLURM training job for LSTM Autoencoder on BGL dataset
# ===========================================================================
# Submit via : bash run.sh train-eval-bgl   (recommended — chains eval after)
# Direct     : sbatch train_bgl.sh
# Monitor    : bash run.sh status
# Logs       : logs/lstm_ae_bgl_train_<JOBID>.log
# ===========================================================================

#SBATCH --job-name=lstm_ae_bgl
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=16G
#SBATCH --output=logs/lstm_ae_bgl_train_%j.log
#SBATCH --mail-type=END,FAIL
# --account and --mail-user are passed by run.sh via sbatch flags.
# Do not hardcode them here — they come from .env.

set -euo pipefail

# ---------------------------------------------------------------------------
# Load credentials from .env
# SLURM_SUBMIT_DIR is always the directory sbatch was called from.
# ---------------------------------------------------------------------------
ENV_FILE="$SLURM_SUBMIT_DIR/.env"
[[ -f "$ENV_FILE" ]] || { echo "ERROR: .env not found. Submit via: bash run.sh train-eval-bgl"; exit 1; }
# shellcheck disable=SC1090
source "$ENV_FILE"

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
module purge
module load "$MODULE_CONDA"
module load "$MODULE_CUDA"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs checkpoints/bgl runs/bgl plots/bgl

echo "=========================================="
echo "  BGL Training Job"
echo "  Job ID   : $SLURM_JOB_ID"
echo "  Node     : $SLURMD_NODENAME"
echo "  Conda    : $CONDA_ENV"
echo "  Start    : $(date)"
echo "  Python   : $(which python)"
echo "  GPU      : $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")')"
echo "=========================================="
echo ""

python train.py \
    --data_dir              data/bgl/            \
    --vocab_file            data/bgl/vocab.json   \
    --checkpoint_dir        checkpoints/bgl/      \
    --log_dir               runs/bgl/             \
    --vocab_size            267                   \
    --window_size           20                    \
    --embed_dim             64                    \
    --hidden_dim            128                   \
    --latent_dim            32                    \
    --num_layers            2                     \
    --dropout               0.2                   \
    --batch_size            64                    \
    --lr                    1e-3                  \
    --epochs                30                    \
    --patience              5                     \
    --weight_decay          1e-5                  \
    --clip_grad_norm        1.0                   \
    --seed                  42                    \
    --save_every            5

EXIT_CODE=$?
echo ""
echo "=========================================="
echo "  BGL Training complete | Exit: $EXIT_CODE | $(date)"
echo "=========================================="
exit $EXIT_CODE