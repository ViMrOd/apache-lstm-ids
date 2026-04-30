# LSTM Autoencoder — Log Anomaly Detection

Reconstruction-error-based anomaly detection for system logs using an LSTM Autoencoder. The model is trained on normal log sequences only; anomalies are flagged at inference time by high MSE between the input sequence and its reconstruction.

Implemented following Malhotra et al. (2016), *LSTM-based Encoder-Decoder for Multi-sensor Anomaly Detection*.

---

## Project structure

```
├── autoencoder.py      # Model architecture (LSTMEncoder, LSTMDecoder, LSTMAutoencoder)
├── config.py           # All hyperparameters via argparse
├── train.py            # Training loop with early stopping and checkpointing
├── evaluate.py         # Threshold sweep, F1/precision/recall/ROC-AUC, plots
├── diagnose.py         # Pre-training data and vocab sanity checks
├── run.sh              # Master workflow script — start here
├── train_osc.sh        # SLURM job script for training
├── eval_osc.sh         # SLURM job script for evaluation
├── smoke.sh            # Quick smoke test (synthetic data, ~30s)
├── smoke_test.sh       # Full pipeline smoke test on real data subset (~5 min)
├── .env.example        # Credential template — copy to .env and fill in
├── .gitignore
└── requirements.txt

## Prerequisites

- Access to the Ohio Supercomputer Center (OSC)
- A conda environment with PyTorch + CUDA (see setup below)

---

## Setup

### 1. Clone the repo

```bash
git clone <repo-url>
cd LSTM
```

### 2. Create the conda environment (once, on OSC)

```bash
module load miniconda3/24.1.2-py310
conda create -n log_anomaly python=3.10 \
    pytorch torchvision pytorch-cuda=11.8 \
    -c pytorch -c nvidia
conda activate log_anomaly
pip install -r requirements.txt
```

### 3. Configure your credentials

```bash
cp .env.example .env
```

Then open `.env` and fill in your values:

```bash
OSC_ACCOUNT=PAS1234              # your OSC project ID
OSC_EMAIL=you@email.osu.edu      # your OSC-registered email
CONDA_ENV=log_anomaly            # conda environment name
PROJECT_DIR=$HOME/LSTM           # absolute path to project root on OSC
MODULE_CONDA=miniconda3/24.1.2-py310
MODULE_CUDA=cuda/11.8.0
```

`.env` is gitignored and never committed. Every team member creates their own.

---

## Data

Place the output files in `data/`:

```
data/
├── X_train.npy    # (N, window_size) int64 — normal sequences only
├── X_val.npy      # (N, window_size) int64
├── y_val.npy      # (N,) int64 — 0=normal, 1=anomalous
├── X_test.npy     # (N, window_size) int64
├── y_test.npy     # (N,) int64 — 0=normal, 1=anomalous
└── vocab.json     # {token: index} mapping
```

The `data/` directory is gitignored — do not commit data files.

---

## Workflow

All commands go through `run.sh`:

```bash
bash run.sh <command>
```

### Typical workflow

```bash
# 1. Confirm the pipeline works end-to-end (no real data needed)
bash run.sh smoke

# 2. After data files are in place — check for vocab/shape mismatches
bash run.sh diagnose

# 3. Full pipeline check on a small real data subset (~5 min)
bash run.sh smoke-real

# 4. Submit training + evaluation to SLURM (eval auto-chains after training)
bash run.sh train-eval

# 5. Monitor progress
bash run.sh status

# 6. Tail the active log
bash run.sh logs
```

### All commands

| Command | What it does |
|---|---|
| `diagnose` | Inspect vocab and data files, report token ranges and anomaly rates |
| `smoke` | End-to-end test with synthetic data on the login node (~30s) |
| `smoke-real` | End-to-end test on a real data subset, sliced to 500/200/200 rows (~5 min) |
| `interactive` | Request a GPU interactive session via `sinteractive` |
| `train` | Submit training job to SLURM |
| `eval` | Submit evaluation job to SLURM |
| `train-eval` | Submit both; evaluation auto-starts after training completes |
| `status` | Show your running/pending SLURM jobs |
| `logs` | Tail the most recent log file |
| `help` | Print the command list |

---

## Outputs

After training and evaluation:

```
checkpoints/
└── best_model.pt          # Best checkpoint (lowest val loss)

plots/
├── roc_curve.png          # ROC curve with AUC annotation
└── score_distribution.png # Reconstruction score histogram (normal vs anomalous)

runs/                      # TensorBoard logs
logs/                      # SLURM job output logs
```

View TensorBoard logs locally:
```bash
tensorboard --logdir runs/
```

---

## Hyperparameters

All hyperparameters are CLI flags defined in `config.py`. Key defaults:

| Parameter | Default | Notes |
|---|---|---|
| `vocab_size` | 512 | Must match preprocessing vocab |
| `window_size` | 20 | Must match preprocessing sliding window |
| `embed_dim` | 64 | Token embedding dimension |
| `hidden_dim` | 128 | LSTM hidden state size |
| `latent_dim` | 32 | Bottleneck dimension |
| `num_layers` | 2 | Stacked LSTM layers |
| `dropout` | 0.2 | Between LSTM layers |
| `lr` | 1e-3 | Adam learning rate |
| `epochs` | 30 | Max epochs (early stopping at patience=5) |
| `batch_size` | 64 | Sequences per mini-batch |

Override any parameter via the CLI flags in `train_osc.sh` or `eval_osc.sh`.

---

## Model architecture

```
Input (B, T)
    ↓
Embedding (B, T, E)          frozen weights — padding_idx=0
    ↓
LSTM Encoder × 2 layers      final hidden state h_T
    ↓
Linear projection            (B, hidden_dim) → (B, latent_dim)
    ↓  [bottleneck]
Linear projection            (B, latent_dim) → (B, embed_dim)
    ↓ repeated T times
LSTM Decoder × 2 layers
    ↓
Linear projection            (B, T, hidden_dim) → (B, T, embed_dim)
    ↓
MSE vs original embeddings   → reconstruction loss / anomaly score
```

Anomaly score is the per-sample mean MSE. The detection threshold is selected automatically by sweeping values on the val set and maximising F1.

---

## Reference

Malhotra, P. et al. (2016). LSTM-based Encoder-Decoder for Multi-sensor Anomaly Detection. *ICML Anomaly Detection Workshop*. https://arxiv.org/abs/1607.00148