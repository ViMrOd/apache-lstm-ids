#!/bin/bash
# ===========================================================================
# eval_thunderbird.sh — SLURM evaluation job for Thunderbird dataset
# ===========================================================================
# Submit via : bash run.sh train-eval-tb   (recommended — chains after train)
# Direct     : sbatch eval_thunderbird.sh
# ===========================================================================

#SBATCH --job-name=lstm_ae_tb_eval
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-node=1
#SBATCH --mem=8G
#SBATCH --output=logs/lstm_ae_tb_eval_%j.log
#SBATCH --mail-type=END,FAIL

set -euo pipefail

ENV_FILE="$SLURM_SUBMIT_DIR/.env"
[[ -f "$ENV_FILE" ]] || { echo "ERROR: .env not found. Submit via: bash run.sh eval-tb"; exit 1; }
# shellcheck disable=SC1090
source "$ENV_FILE"

module purge
module load "$MODULE_CONDA"
module load "$MODULE_CUDA"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs plots/thunderbird

CHECKPOINT="checkpoints/thunderbird/best_model.pt"
if [[ ! -f "$CHECKPOINT" ]]; then
    echo "ERROR: Checkpoint not found at $CHECKPOINT"
    echo "       Has training completed? Run: bash run.sh status"
    exit 1
fi

echo "=========================================="
echo "  Thunderbird Evaluation Job"
echo "  Job ID     : $SLURM_JOB_ID"
echo "  Checkpoint : $CHECKPOINT"
echo "  Start      : $(date)"
echo "=========================================="
echo ""

python evaluate.py \
    --checkpoint            $CHECKPOINT                   \
    --data_dir              data/thunderbird/             \
    --vocab_file            data/thunderbird/vocab.json   \
    --plot_dir              plots/thunderbird/            \
    --vocab_size            1711                          \
    --window_size           20                            \
    --embed_dim             64                            \
    --hidden_dim            128                           \
    --latent_dim            32                            \
    --num_layers            2                             \
    --batch_size            32                            \
    --threshold_sweep_steps 200                           \
    --seed                  42

EXIT_CODE=$?
echo ""
echo "=========================================="
echo "  Thunderbird Evaluation complete | Exit: $EXIT_CODE | $(date)"
echo "  Plots saved to: plots/thunderbird/"
echo "=========================================="
exit $EXIT_CODE