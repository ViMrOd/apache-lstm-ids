#!/bin/bash
# ===========================================================================
# demo_osc.sh — Launch the Streamlit demo on OSC
# ===========================================================================
#
# Usage: bash demo_osc.sh [port]
#
# Default port: 8501
# If 8501 is taken, try: bash demo_osc.sh 8502
#
# After running this, on your LOCAL machine run:
#   bash demo_local.sh <this_node> [port]
#
# ===========================================================================

set -euo pipefail

PORT="${1:-8501}"

# ---------------------------------------------------------------------------
# Load .env
# ---------------------------------------------------------------------------
PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ENV_FILE="$PROJECT_DIR/.env"

if [[ ! -f "$ENV_FILE" ]]; then
    echo "ERROR: .env not found. Run: cp .env.example .env"
    exit 1
fi
source "$ENV_FILE"

# ---------------------------------------------------------------------------
# Load modules and activate conda
# ---------------------------------------------------------------------------
module load "$MODULE_CONDA"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

# ---------------------------------------------------------------------------
# Check streamlit is installed
# ---------------------------------------------------------------------------
if ! python -c "import streamlit" 2>/dev/null; then
    echo ">>> Installing streamlit ..."
    pip install streamlit --quiet
fi

# ---------------------------------------------------------------------------
# Kill any existing streamlit on this port
# ---------------------------------------------------------------------------
EXISTING=$(lsof -ti :$PORT 2>/dev/null || true)
if [[ -n "$EXISTING" ]]; then
    echo ">>> Killing existing process on port $PORT (PID $EXISTING) ..."
    kill -9 $EXISTING 2>/dev/null || true
    sleep 1
fi

# ---------------------------------------------------------------------------
# Print connection info
# ---------------------------------------------------------------------------
NODE=$(hostname)
echo ""
echo "=========================================="
echo "  LSTM Anomaly IDS — Demo"
echo "  Node : $NODE"
echo "  Port : $PORT"
echo "=========================================="
echo ""
echo "  On your LOCAL machine, run:"
echo "  bash demo_local.sh $NODE $PORT"
echo ""
echo "  Then open: http://localhost:$PORT"
echo ""
echo "  Press Ctrl+C to stop the demo."
echo "=========================================="
echo ""

cd "$PROJECT_DIR"
streamlit run demo_app.py \
    --server.port "$PORT" \
    --server.headless true \
    --server.address 0.0.0.0 \
    --browser.gatherUsageStats false