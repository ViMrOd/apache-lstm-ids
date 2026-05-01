#!/bin/bash
# ===========================================================================
# eval_bgl.sh — SLURM evaluation job for LSTM Autoencoder on BGL dataset
# ===========================================================================
# Submit via : bash run.sh train-eval-bgl   (recommended — chains after train)
# Direct     : sbatch eval_bgl.sh
# ===========================================================================

#SBATCH --job-name=lstm_ae_bgl_eval
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-node=1
#SBATCH --mem=8G
#SBATCH --output=logs/lstm_ae_bgl_eval_%j.log
#SBATCH --mail-type=END,FAIL

set -euo pipefail

ENV_FILE="$SLURM_SUBMIT_DIR/.env"
[[ -f "$ENV_FILE" ]] || { echo "ERROR: .env not found. Submit via: bash run.sh eval-bgl"; exit 1; }
# shellcheck disable=SC1090
source "$ENV_FILE"

module purge
module load "$MODULE_CONDA"
module load "$MODULE_CUDA"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs plots/bgl

CHECKPOINT="checkpoints/bgl/best_model.pt"
if [[ ! -f "$CHECKPOINT" ]]; then
    echo "ERROR: Checkpoint not found at $CHECKPOINT"
    echo "       Has training completed? Run: bash run.sh status"
    exit 1
fi

echo "=========================================="
echo "  BGL Evaluation Job"
echo "  Job ID     : $SLURM_JOB_ID"
echo "  Checkpoint : $CHECKPOINT"
echo "  Start      : $(date)"
echo "=========================================="
echo ""

python evaluate.py \
    --checkpoint            $CHECKPOINT           \
    --data_dir              data/bgl/             \
    --vocab_file            data/bgl/vocab.json   \
    --plot_dir              plots/bgl/            \
    --vocab_size            267                   \
    --window_size           20                    \
    --embed_dim             64                    \
    --hidden_dim            128                   \
    --latent_dim            32                    \
    --num_layers            2                     \
    --batch_size            64                    \
    --threshold_sweep_steps 200                   \
    --seed                  42

EXIT_CODE=$?
echo ""
echo "=========================================="
echo "  BGL Evaluation complete | Exit: $EXIT_CODE | $(date)"
echo "  Plots saved to: plots/bgl/"
echo "=========================================="
exit $EXIT_CODE