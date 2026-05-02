#!/bin/bash
# ===========================================================================
# run.sh — Master workflow script for LSTM Autoencoder on OSC
# ===========================================================================
#
# USAGE
# -----
# bash run.sh <command>
#
# COMMANDS
# --------
# diagnose              Inspect vocab and data files for HDFS
# smoke                 Quick sanity check — synthetic data, login node (~30s)
# smoke-real            Sanity check with real HDFS data on login node (~30s)
# smoke-bgl             Sanity check with real BGL data on login node (~30s)
# smoke-thunderbird      Sanity check with real Thunderbird data on login node (~30s)
# interactive           Request a GPU interactive session (sinteractive)
# train-hdfs            Submit HDFS training job to SLURM
# eval-hdfs             Submit HDFS evaluation job to SLURM
# train-eval-hdfs       Submit HDFS training + evaluation (eval chains after train)
# train-bgl             Submit BGL training job to SLURM
# eval-bgl              Submit BGL evaluation job to SLURM
# train-eval-bgl        Submit BGL training + evaluation (eval chains after train)
# train-tb              Submit Thunderbird training job to SLURM
# eval-tb               Submit Thunderbird evaluation job to SLURM
# train-eval-tb         Submit Thunderbird training + evaluation (eval chains)
# train-all             Submit all three datasets at once
# classify-bgl          Run anomaly type classifier on BGL (after training)
# classify-tb           Run anomaly type classifier on Thunderbird (after training)
# classify-all          Run classifier on both BGL and Thunderbird
# status                Show your currently running/pending jobs
# logs                  Tail the most recent job log
# help                  Print this message
#
# ALIASES (backward compatible)
# train = train-hdfs | eval = eval-hdfs | train-eval = train-eval-hdfs
#
# TYPICAL WORKFLOW
# ----------------
# 1. cp .env.example .env && vim .env      <- fill in OSC credentials once
# 2. bash run.sh smoke                     <- confirm synthetic pipeline works
# 3. bash run.sh smoke-real                <- confirm HDFS data loads correctly
# 4. bash run.sh smoke-bgl                 <- confirm BGL data loads correctly
# 5. bash run.sh smoke-thunderbird         <- confirm Thunderbird data loads
# 6. bash run.sh train-all                 <- submit all three datasets
# 7. bash run.sh status                    <- check job progress
# 8. bash run.sh logs                      <- tail the active log
#
# ===========================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Load credentials from .env
# Uses the script's own directory — works regardless of where you call it from.
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

for var in OSC_ACCOUNT OSC_EMAIL CONDA_ENV MODULE_CONDA MODULE_CUDA; do
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

load_modules() {
    info "Loading modules ..."
    module load "$MODULE_CONDA"
    module load "$MODULE_CUDA"
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
    echo "   Python : $(which python)"
    echo "   Torch  : $(python -c 'import torch; print(torch.__version__)')"
    echo "   CUDA   : $(python -c 'import torch; print(torch.cuda.is_available())')"
    echo ""
}

# Submit a training job; prints and returns the job ID
submit_train() {
    local script="$1" label="$2"
    [[ -f "$PROJECT_DIR/$script" ]] || die "$script not found in $PROJECT_DIR"
    mkdir -p "$PROJECT_DIR/logs" "$PROJECT_DIR/checkpoints" "$PROJECT_DIR/runs"
    info "Submitting $label training job ..."
    local job_id
    job_id=$(sbatch --parsable \
        --account="$OSC_ACCOUNT" \
        --mail-user="$OSC_EMAIL" \
        "$PROJECT_DIR/$script")
    echo "   Job ID : $job_id"
    echo "   Logs   : logs/*_${job_id}.log"
    echo "$job_id"
}

# Submit an eval job, optionally chained after a train job
submit_eval() {
    local script="$1" label="$2" dep="${3:-}"
    [[ -f "$PROJECT_DIR/$script" ]] || die "$script not found in $PROJECT_DIR"
    mkdir -p "$PROJECT_DIR/logs" "$PROJECT_DIR/plots"
    info "Submitting $label evaluation job ..."
    local dep_flag=""
    [[ -n "$dep" ]] && dep_flag="--dependency=afterok:$dep"
    local job_id
    job_id=$(sbatch --parsable \
        --account="$OSC_ACCOUNT" \
        --mail-user="$OSC_EMAIL" \
        $dep_flag \
        "$PROJECT_DIR/$script")
    echo "   Job ID : $job_id"
    [[ -n "$dep" ]] && echo "   Starts after job $dep completes successfully"
    echo "$job_id"
}

# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

cmd_diagnose() {
    cd "$PROJECT_DIR"
    load_modules
    python diagnose.py --data_dir data/hdfs/ --vocab_file data/hdfs/vocab.json
}

cmd_smoke() {
    cd "$PROJECT_DIR"
    load_modules
    bash "$PROJECT_DIR/smoke.sh"
}

cmd_smoke_real() {
    cd "$PROJECT_DIR"
    load_modules
    bash "$PROJECT_DIR/smoke.sh" --real
}

cmd_smoke_bgl() {
    cd "$PROJECT_DIR"
    load_modules
    info "Running smoke test (real BGL data) ..."
    DATA_DIR="$PROJECT_DIR/data/bgl" \
    VOCAB_FILE="$PROJECT_DIR/data/bgl/vocab.json" \
    bash "$PROJECT_DIR/smoke.sh" --real
}

cmd_smoke_tb() {
    cd "$PROJECT_DIR"
    load_modules
    info "Running smoke test (real Thunderbird data) ..."
    DATA_DIR="$PROJECT_DIR/data/thunderbird" \
    VOCAB_FILE="$PROJECT_DIR/data/thunderbird/vocab.json" \
    bash "$PROJECT_DIR/smoke.sh" --real
}

cmd_interactive() {
    info "Requesting interactive GPU session ..."
    sinteractive -A "$OSC_ACCOUNT" -t 00:30:00 -N 1 -n 1 -g 1
}

# HDFS
cmd_train_eval() {
    local tid
    tid=$(submit_train "train_hdfs.sh" "HDFS")
    submit_eval "eval_hdfs.sh" "HDFS" "$tid" > /dev/null
    echo ""; echo "   Monitor: bash run.sh status | Tail: bash run.sh logs"
}

# BGL
cmd_train_eval_bgl() {
    local tid
    tid=$(submit_train "train_bgl.sh" "BGL")
    submit_eval "eval_bgl.sh" "BGL" "$tid" > /dev/null
    echo ""; echo "   Monitor: bash run.sh status | Tail: bash run.sh logs"
}

# Thunderbird
cmd_train_eval_tb() {
    local tid
    tid=$(submit_train "train_thunderbird.sh" "Thunderbird")
    submit_eval "eval_thunderbird.sh" "Thunderbird" "$tid" > /dev/null
    echo ""; echo "   Monitor: bash run.sh status | Tail: bash run.sh logs"
}

# All three at once
cmd_train_all() {
    info "Submitting all three datasets ..."
    echo ""
    local hdfs_tid bgl_tid tb_tid
    hdfs_tid=$(submit_train "train_hdfs.sh"        "HDFS")
    submit_eval "eval_hdfs.sh" "HDFS" "$hdfs_tid" > /dev/null

    bgl_tid=$(submit_train "train_bgl.sh"          "BGL")
    submit_eval "eval_bgl.sh" "BGL" "$bgl_tid" > /dev/null

    tb_tid=$(submit_train "train_thunderbird.sh"   "Thunderbird")
    submit_eval "eval_thunderbird.sh" "Thunderbird" "$tb_tid" > /dev/null

    echo ""
    echo "   All jobs submitted. Monitor: bash run.sh status"
}

cmd_classify() {
    local dataset="$1"
    cd "$PROJECT_DIR"
    load_modules
    local ckpt="checkpoints/${dataset}/best_model.pt"
    if [[ ! -f "$ckpt" ]]; then
        die "Checkpoint not found: $ckpt — has training completed? Run: bash run.sh status"
    fi
    info "Running anomaly type classifier on ${dataset^^} ..."
    mkdir -p "classifiers/$dataset"
    python classify.py --dataset "$dataset"
}

cmd_classify_all() {
    cmd_classify bgl
    cmd_classify thunderbird
}

cmd_status() {
    info "Your current jobs:"
    squeue -u "$USER" --format="%.10i %.25j %.8T %.10M %.10l %.6D %R"
}

cmd_logs() {
    local latest_log
    latest_log=$(ls -t "$PROJECT_DIR"/logs/*.log 2>/dev/null | head -1)
    [[ -n "$latest_log" ]] || die "No log files found in $PROJECT_DIR/logs/"
    info "Tailing: $latest_log (Ctrl+C to stop)"
    tail -f "$latest_log"
}

cmd_help() { sed -n '3,40p' "$0"; }

# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
COMMAND="${1:-help}"

case "$COMMAND" in
    diagnose)              cmd_diagnose ;;
    smoke)                 cmd_smoke ;;
    smoke-real)            cmd_smoke_real ;;
    smoke-bgl)             cmd_smoke_bgl ;;
    smoke-thunderbird)      cmd_smoke_tb ;;
    interactive)           cmd_interactive ;;
    train-hdfs)            submit_train "train_hdfs.sh"        "HDFS" > /dev/null ;;
    eval-hdfs)             submit_eval  "eval_hdfs.sh"         "HDFS" > /dev/null ;;
    train-eval-hdfs)       cmd_train_eval ;;
    train-bgl)             submit_train "train_bgl.sh"         "BGL"  > /dev/null ;;
    eval-bgl)              submit_eval  "eval_bgl.sh"          "BGL"  > /dev/null ;;
    train-eval-bgl)        cmd_train_eval_bgl ;;
    train-tb)              submit_train "train_thunderbird.sh"  "Thunderbird" > /dev/null ;;
    eval-tb)               submit_eval  "eval_thunderbird.sh"   "Thunderbird" > /dev/null ;;
    train-eval-tb)         cmd_train_eval_tb ;;
    train-all)             cmd_train_all ;;
    classify-bgl)          cmd_classify bgl ;;
    classify-tb)           cmd_classify thunderbird ;;
    classify-all)          cmd_classify_all ;;
    demo)                  bash "$PROJECT_DIR/demo_osc.sh" ;;
    status)                cmd_status ;;
    logs)                  cmd_logs ;;
    # Backward-compatible aliases
    train)                 submit_train "train_hdfs.sh"        "HDFS" > /dev/null ;;
    eval)                  submit_eval  "eval_hdfs.sh"         "HDFS" > /dev/null ;;
    train-eval)            cmd_train_eval ;;
    help|--help)           cmd_help ;;
    *)
        echo "Unknown command: $COMMAND"
        echo ""
        cmd_help
        exit 1
        ;;
esac