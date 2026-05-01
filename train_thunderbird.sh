#!/bin/bash
# ===========================================================================
# train_thunderbird.sh — SLURM training job for Thunderbird dataset
# ===========================================================================
# Submit via : bash run.sh train-eval-tb   (recommended — chains eval after)
# Direct     : sbatch train_thunderbird.sh
# Monitor    : bash run.sh status
# Logs       : logs/lstm_ae_tb_train_<JOBID>.log
#
# NOTE: Thunderbird training set is small (3,170 sequences). Training will
# be fast. dropout and weight_decay are slightly higher than HDFS/BGL to
# compensate for the small dataset size.
# ===========================================================================

#SBATCH --job-name=lstm_ae_tb
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=16G
#SBATCH --output=logs/lstm_ae_tb_train_%j.log
#SBATCH --mail-type=END,FAIL

set -euo pipefail

ENV_FILE="$SLURM_SUBMIT_DIR/.env"
[[ -f "$ENV_FILE" ]] || { echo "ERROR: .env not found. Submit via: bash run.sh train-eval-tb"; exit 1; }
# shellcheck disable=SC1090
source "$ENV_FILE"

module purge
module load "$MODULE_CONDA"
module load "$MODULE_CUDA"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs checkpoints/thunderbird runs/thunderbird plots/thunderbird

echo "=========================================="
echo "  Thunderbird Training Job"
echo "  Job ID   : $SLURM_JOB_ID"
echo "  Node     : $SLURMD_NODENAME"
echo "  Conda    : $CONDA_ENV"
echo "  Start    : $(date)"
echo "  WARNING  : Small dataset (3,170 sequences) — watch for overfitting"
echo "  Python   : $(which python)"
echo "  GPU      : $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")')"
echo "=========================================="
echo ""

python train.py \
    --data_dir              data/thunderbird/            \
    --vocab_file            data/thunderbird/vocab.json   \
    --checkpoint_dir        checkpoints/thunderbird/      \
    --log_dir               runs/thunderbird/             \
    --vocab_size            1711                          \
    --window_size           20                            \
    --embed_dim             64                            \
    --hidden_dim            128                           \
    --latent_dim            32                            \
    --num_layers            2                             \
    --dropout               0.3                           \
    --batch_size            32                            \
    --lr                    1e-3                          \
    --epochs                30                            \
    --patience              7                             \
    --weight_decay          1e-4                          \
    --clip_grad_norm        1.0                           \
    --seed                  42                            \
    --save_every            5

EXIT_CODE=$?
echo ""
echo "=========================================="
echo "  Thunderbird Training complete | Exit: $EXIT_CODE | $(date)"
echo "=========================================="
exit $EXIT_CODE