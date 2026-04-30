#!/bin/bash
# ===========================================================================
# run.sh — Master workflow script for LSTM Autoencoder on OSC
# ===========================================================================
#
# USAGE
# -----
#   bash run.sh <command>
#
# COMMANDS
# --------
#   smoke           Quick sanity check — synthetic data, login node (~30s)
#   smoke-real      Sanity check with real data sliced to 2k rows, login node (~30s)
#   interactive     Request a GPU interactive session (sinteractive)
#   train           Submit training job to SLURM queue (sbatch)
#   eval            Submit evaluation job to SLURM queue (sbatch)
#   train-eval      Submit training job; auto-queue eval to run after it finishes
#   status          Show your currently running/pending jobs
#   logs            Tail the most recent job log
#   help            Print this message
#
# TYPICAL WORKFLOW
# ----------------
#   1. cp .env.example .env && vim .env   ← fill in your OSC credentials once
#   2. bash run.sh smoke                  ← confirm everything imports and runs
#   3. bash run.sh smoke-real             ← confirm real data loads correctly
#   4. bash run.sh train-eval             ← submit both jobs, eval chains after train
#   5. bash run.sh status                 ← check job progress
#   6. bash run.sh logs                   ← tail the active log
#
# ===========================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Load credentials and paths from .env
# Never commit .env — it contains your OSC account ID and email.
# Copy .env.example → .env and fill in your values before running.
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

# Validate required variables are set and non-empty
for var in OSC_ACCOUNT OSC_EMAIL CONDA_ENV PROJECT_DIR MODULE_CONDA MODULE_CUDA; do
    if [[ -z "${!var:-}" ]]; then
        echo "ERROR: $var is not set in .env"
        exit 1
    fi
done

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
die()  { echo "ERROR: $*" >&2; exit 1; }
info() { echo ">>> $*"; }

require_project_dir() {
    [[ -f "$PROJECT_DIR/train.py" ]] \
        || die "train.py not found in $PROJECT_DIR — is PROJECT_DIR set correctly in .env?"
    cd "$PROJECT_DIR"
}

load_modules() {
    info "Loading modules ..."
    module load "$MODULE_CONDA"
    module load "$MODULE_CUDA"
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
    echo "    Python : $(which python)"
    echo "    Torch  : $(python -c 'import torch; print(torch.__version__)')"
    echo "    CUDA   : $(python -c 'import torch; print(torch.cuda.is_available())')"
    echo ""
}

# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

cmd_smoke() {
    require_project_dir
    load_modules
    info "Running smoke test (synthetic data) ..."
    bash "$PROJECT_DIR/smoke.sh"
}

cmd_smoke_real() {
    require_project_dir
    load_modules
    info "Running smoke test (real data subset) ..."
    bash "$PROJECT_DIR/smoke_test.sh"
}

cmd_interactive() {
    info "Requesting interactive GPU session ..."
    info "Once the session starts, run:  bash run.sh smoke-real"
    echo ""
    sinteractive -A "$OSC_ACCOUNT" -t 00:30:00 -N 1 -n 1 -g 1
}

cmd_train() {
    require_project_dir
    info "Submitting training job ..."
    mkdir -p logs checkpoints runs
    TRAIN_JOB=$(sbatch --parsable --account="$OSC_ACCOUNT" \
                       --mail-user="$OSC_EMAIL" train_osc.sh)
    echo ""
    echo "    Job ID  : $TRAIN_JOB"
    echo "    Logs    : logs/lstm_ae_train_${TRAIN_JOB}.log"
    echo "    Monitor : bash run.sh status"
    echo "    Tail log: bash run.sh logs"
}

cmd_eval() {
    require_project_dir
    [[ -f "checkpoints/best_model.pt" ]] \
        || die "checkpoints/best_model.pt not found — has training completed?"
    info "Submitting evaluation job ..."
    mkdir -p logs plots
    EVAL_JOB=$(sbatch --parsable --account="$OSC_ACCOUNT" \
                      --mail-user="$OSC_EMAIL" eval_osc.sh)
    echo ""
    echo "    Job ID  : $EVAL_JOB"
    echo "    Logs    : logs/lstm_ae_eval_${EVAL_JOB}.log"
    echo "    Monitor : bash run.sh status"
}

cmd_train_eval() {
    require_project_dir
    info "Submitting training job ..."
    mkdir -p logs checkpoints runs plots
    TRAIN_JOB=$(sbatch --parsable --account="$OSC_ACCOUNT" \
                       --mail-user="$OSC_EMAIL" train_osc.sh)
    echo "    Train job ID : $TRAIN_JOB"

    info "Chaining evaluation job to run after training completes ..."
    EVAL_JOB=$(sbatch --parsable --account="$OSC_ACCOUNT" \
                      --mail-user="$OSC_EMAIL" \
                      --dependency=afterok:"$TRAIN_JOB" eval_osc.sh)
    echo "    Eval job ID  : $EVAL_JOB  (starts after job $TRAIN_JOB finishes)"
    echo ""
    echo "    Logs    : logs/lstm_ae_train_${TRAIN_JOB}.log"
    echo "              logs/lstm_ae_eval_${EVAL_JOB}.log"
    echo "    Monitor : bash run.sh status"
    echo "    Tail log: bash run.sh logs"
}

cmd_status() {
    info "Your current jobs:"
    echo ""
    squeue -u "$USER" --format="%.10i %.20j %.8T %.10M %.10l %.6D %R"
}

cmd_logs() {
    require_project_dir
    LATEST_LOG=$(ls -t "$PROJECT_DIR"/logs/*.log 2>/dev/null | head -1)
    [[ -n "$LATEST_LOG" ]] || die "No log files found in $PROJECT_DIR/logs/"
    info "Tailing: $LATEST_LOG  (Ctrl+C to stop)"
    echo ""
    tail -f "$LATEST_LOG"
}

cmd_help() {
    sed -n '3,35p' "$0"
}

# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
COMMAND="${1:-help}"

case "$COMMAND" in
    smoke)        cmd_smoke ;;
    smoke-real)   cmd_smoke_real ;;
    interactive)  cmd_interactive ;;
    train)        cmd_train ;;
    eval)         cmd_eval ;;
    train-eval)   cmd_train_eval ;;
    status)       cmd_status ;;
    logs)         cmd_logs ;;
    help|--help)  cmd_help ;;
    *)
        echo "Unknown command: $COMMAND"
        echo ""
        cmd_help
        exit 1
        ;;
esac