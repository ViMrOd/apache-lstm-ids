# LSTM Autoencoder — Log Anomaly Detection IDS

Reconstruction-error-based intrusion detection for system logs using an LSTM Autoencoder. The model is trained on normal log sequences only; anomalies are flagged at inference time by high MSE between the input sequence and its reconstruction. A secondary MLP classifier identifies the specific fault category for datasets with labeled anomaly types.

Implemented following Malhotra et al. (2016), *LSTM-based Encoder-Decoder for Multi-sensor Anomaly Detection*.

---

## Datasets

| Dataset | Environment | Vocab Size | Anomaly Types |
|---------|-------------|------------|---------------|
| HDFS | Hadoop Distributed File System | 47 | Binary (normal / anomalous) |
| BGL | Blue Gene/L Supercomputer (LLNL) | 267 | 22 fault categories |
| Thunderbird | Sandia National Labs supercomputer | 1711 | 2 fault categories |

---

## Project structure

```
├── autoencoder.py        # Model architecture (LSTMEncoder, LSTMDecoder, LSTMAutoencoder)
├── config.py             # All hyperparameters via argparse
├── train.py              # Training loop with early stopping and checkpointing
├── evaluate.py           # Threshold sweep, F1/precision/recall/ROC-AUC, per-category breakdown
├── classify.py           # Stage 2: MLP anomaly type classifier on latent space
├── diagnose.py           # Pre-training data and vocab sanity checks
├── demo_app.py           # Streamlit SOC dashboard demo
├── run.sh                # Master workflow script — start here
├── train_hdfs.sh         # SLURM training job — HDFS
├── train_bgl.sh          # SLURM training job — BGL
├── train_thunderbird.sh  # SLURM training job — Thunderbird
├── eval_hdfs.sh          # SLURM evaluation job — HDFS
├── eval_bgl.sh           # SLURM evaluation job — BGL
├── eval_thunderbird.sh   # SLURM evaluation job — Thunderbird
├── smoke.sh              # Quick smoke test (synthetic data, ~30s)
├── smoke_test.sh         # Full pipeline smoke test on real data subset
├── .env.example          # Credential template — copy to .env and fill in
├── .gitignore
└── requirements.txt
```

---

## Prerequisites

- Access to the Ohio Supercomputer Center (OSC)
- A conda environment with PyTorch + CUDA (see setup below)

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/ViMrOd/apache-lstm-ids.git
cd apache-lstm-ids
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

### 3. Configure credentials

```bash
cp .env.example .env
```

Fill in your values in `.env`:

```
OSC_ACCOUNT=PAS1234
OSC_EMAIL=you@email.osu.edu
CONDA_ENV=log_anomaly
PROJECT_DIR=/users/yourname/apache-lstm-ids
MODULE_CONDA=miniconda3/24.1.2-py310
MODULE_CUDA=cuda/11.8.0
```

`.env` is gitignored and never committed.

---

## Data

Place preprocessed files in `data/<dataset>/`:

```
data/
├── hdfs/
│   ├── X_train.npy     # (N, 20) int64 — normal sequences only
│   ├── X_val.npy       # (N, 20) int64
│   ├── y_val.npy       # (N,) int64 — 0=normal, 1=anomalous
│   ├── X_test.npy      # (N, 20) int64
│   ├── y_test.npy      # (N,) int64
│   ├── vocab.json      # {template_N: token_id}
│   └── label_map.json  # {token_id: category_name}
├── bgl/                # same structure, vocab_size=267, 22 anomaly types
└── thunderbird/        # same structure, vocab_size=1711, 2 anomaly types
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
# 1. Confirm the pipeline works (no real data needed)
bash run.sh smoke

# 2. Sanity check each dataset before submitting real jobs
bash run.sh smoke-real
bash run.sh smoke-bgl
bash run.sh smoke-thunderbird

# 3. Submit all three datasets at once
bash run.sh train-all          # submits train + eval jobs for HDFS, BGL, Thunderbird

# 4. Monitor progress
bash run.sh status
bash run.sh logs

# 5. Run anomaly type classifier (after BGL/Thunderbird training completes)
bash run.sh classify-bgl
bash run.sh classify-tb

# 6. Launch the demo
bash run.sh demo
```

### All commands

| Command | What it does |
|---------|-------------|
| `diagnose` | Inspect HDFS vocab and data files |
| `smoke` | Synthetic data smoke test on login node (~30s) |
| `smoke-real` | Real HDFS data smoke test on login node |
| `smoke-bgl` | Real BGL data smoke test on login node |
| `smoke-thunderbird` | Real Thunderbird data smoke test on login node |
| `interactive` | Request a GPU interactive session via `sinteractive` |
| `train-hdfs` | Submit HDFS training job to SLURM |
| `eval-hdfs` | Submit HDFS evaluation job to SLURM |
| `train-eval-hdfs` | Submit HDFS training + evaluation (eval chains after train) |
| `train-bgl` | Submit BGL training job to SLURM |
| `eval-bgl` | Submit BGL evaluation job to SLURM |
| `train-eval-bgl` | Submit BGL training + evaluation |
| `train-tb` | Submit Thunderbird training job to SLURM |
| `eval-tb` | Submit Thunderbird evaluation job to SLURM |
| `train-eval-tb` | Submit Thunderbird training + evaluation |
| `train-all` | Submit all three datasets at once |
| `classify-bgl` | Train anomaly type classifier on BGL latent space |
| `classify-tb` | Train anomaly type classifier on Thunderbird latent space |
| `classify-all` | Run classifier on both BGL and Thunderbird |
| `status` | Show running/pending SLURM jobs |
| `logs` | Tail the most recent log file |

Aliases: `train` = `train-hdfs`, `eval` = `eval-hdfs`, `train-eval` = `train-eval-hdfs`

---

## Interactive Demo

A Streamlit SOC (Security Operations Center) dashboard that runs the trained models live.

### Install Streamlit

```bash
module load miniconda3/24.1.2-py310
conda activate log_anomaly
pip install streamlit
```

### Run on OSC with port forwarding

**Step 1 — Start Streamlit on OSC** (note which login node you're on, e.g. `cardinal-login03`):

```bash
streamlit run demo_app.py --server.port 8501
```

**Step 2 — Forward the port** (run this on your local machine in a separate terminal):

```bash
ssh -L 8501:cardinal-login03:8501 your_username@cardinal.osc.edu
```

Replace `cardinal-login03` with whichever node Streamlit is running on.

**Step 3 — Open in browser:**

```
http://localhost:8501
```

### Demo features

- **Live Stream mode** — plays back real test sequences in real time, flags anomalies as they arrive
- **Custom mode** — build any sequence from the dataset vocabulary and score it live
- **Builder mode** — interactive block lifecycle builder (HDFS only)
- **Environment switcher** — switch between HDFS, BGL, and Thunderbird with one click
- **Anomaly type classification** — for BGL and Thunderbird, detected anomalies are labeled with their fault category (e.g. KERNSTOR, ECC::Server)
- **Score timeline** — rolling reconstruction score chart with threshold line
- **Alert feed** — running log of flagged sequences with scores and categories

---

## Outputs

After training and evaluation:

```
checkpoints/
├── hdfs/best_model.pt
├── bgl/best_model.pt
└── thunderbird/best_model.pt

plots/
├── hdfs/   roc_curve.png, score_distribution.png
├── bgl/    roc_curve.png, score_distribution.png, confusion.png, per_class_f1.png
└── thunderbird/  (same)

classifiers/
├── bgl/classifier.pt
└── thunderbird/classifier.pt

runs/        TensorBoard logs (per dataset)
logs/        SLURM job output logs
```

---

## Model architecture

### Stage 1 — LSTM Autoencoder (anomaly detection)

```
Input (B, T)
    ↓
Embedding (B, T, E)          frozen weights — padding_idx=0
    ↓
LSTM Encoder × 2 layers      final hidden state h_T
    ↓
Linear projection            (B, hidden_dim) → (B, latent_dim=32)
    ↓  [bottleneck]
Linear projection            (B, 32) → (B, embed_dim)
    ↓ repeated T times
LSTM Decoder × 2 layers
    ↓
Linear projection            (B, T, hidden_dim) → (B, T, embed_dim)
    ↓
MSE vs original embeddings   → reconstruction loss / anomaly score
```

### Stage 2 — MLP Classifier (fault categorization, BGL + Thunderbird only)

```
Latent vector (B, 32)
    ↓
Linear(32→64) → ReLU → Dropout
    ↓
Linear(64→32) → ReLU → Dropout
    ↓
Linear(32→N classes)
    ↓
Cross-entropy → predicted fault category
```

---

## Hyperparameters

| Parameter | HDFS | BGL | Thunderbird |
|-----------|------|-----|-------------|
| `vocab_size` | 47 | 267 | 1711 |
| `window_size` | 20 | 20 | 20 |
| `embed_dim` | 64 | 64 | 64 |
| `hidden_dim` | 128 | 128 | 128 |
| `latent_dim` | 32 | 32 | 32 |
| `dropout` | 0.2 | 0.2 | 0.3 |
| `batch_size` | 64 | 64 | 32 |
| `lr` | 1e-3 | 1e-3 | 1e-3 |
| `epochs` | 30 | 30 | 30 |
| `patience` | 5 | 5 | 7 |

---

## References

Malhotra, P. et al. (2016). LSTM-based Encoder-Decoder for Multi-sensor Anomaly Detection. *ICML Anomaly Detection Workshop*. https://arxiv.org/abs/1607.00148

He, S. et al. (2020). Loghub: A Large Collection of System Log Datasets towards Automated Log Analytics. *arXiv*. https://arxiv.org/abs/2008.06448