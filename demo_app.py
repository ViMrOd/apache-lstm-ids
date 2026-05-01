"""
demo_app.py — Multi-Dataset Log Anomaly Detection SOC Dashboard
===============================================================

Security Operations Center style dashboard for the LSTM Autoencoder
anomaly detection system. Supports HDFS, BGL, and Thunderbird datasets
with live log stream playback, anomaly type classification, and alert feed.

Run:
    streamlit run demo_app.py

Modes:
    LIVE STREAM  — plays back real test sequences from X_test.npy in real time
    BUILDER      — manually construct a sequence from the event palette (HDFS only)
"""

import json
import time
import threading
from pathlib import Path
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ---------------------------------------------------------------------------
# Dataset configurations
# ---------------------------------------------------------------------------
DATASETS = {
    "HDFS": {
        "checkpoint":    "checkpoints/hdfs/best_model.pt",
        "vocab_file":    "data/hdfs/vocab.json",
        "label_map":     "data/hdfs/label_map.json",
        "test_X":        "data/hdfs/X_test.npy",
        "test_y":        "data/hdfs/y_test.npy",
        "classifier":    None,  # HDFS is binary only
        "vocab_size":    47,
        "threshold":     0.008,
        "color":         "#38bdf8",
        "icon":          "🗄️",
        "description":   "Hadoop Distributed File System — block operation logs",
    },
    "BGL": {
        "checkpoint":    "checkpoints/bgl/best_model.pt",
        "vocab_file":    "data/bgl/vocab.json",
        "label_map":     "data/bgl/label_map.json",
        "test_X":        "data/bgl/X_test.npy",
        "test_y":        "data/bgl/y_test.npy",
        "classifier":    "classifiers/bgl/classifier.pt",
        "vocab_size":    267,
        "threshold":     0.01,
        "color":         "#f59e0b",
        "icon":          "⚡",
        "description":   "Blue Gene/L Supercomputer — kernel & hardware fault logs",
    },
    "Thunderbird": {
        "checkpoint":    "checkpoints/thunderbird/best_model.pt",
        "vocab_file":    "data/thunderbird/vocab.json",
        "label_map":     "data/thunderbird/label_map.json",
        "test_X":        "data/thunderbird/X_test.npy",
        "test_y":        "data/thunderbird/y_test.npy",
        "classifier":    "classifiers/thunderbird/classifier.pt",
        "vocab_size":    1711,
        "threshold":     0.01,
        "color":         "#a78bfa",
        "icon":          "🌩️",
        "description":   "Sandia National Labs Thunderbird — supercomputer fault logs",
    },
}

# HDFS template labels for the builder mode
TEMPLATE_LABELS = {
    "template_1":  ("Allocate Block",       "Block allocation request initiated"),
    "template_2":  ("Write Pipeline",       "Write block pipeline established"),
    "template_3":  ("Receive Block",        "Block received from DataNode"),
    "template_4":  ("Packet Sent",          "Data packet sent to downstream"),
    "template_5":  ("Packet Ack",           "Packet acknowledgment received"),
    "template_6":  ("Block Served",         "Block served to client"),
    "template_7":  ("Replicate Block",      "Block replication requested"),
    "template_8":  ("Replication Done",     "Block replication completed"),
    "template_9":  ("Block Verified",       "Block checksum verified OK"),
    "template_10": ("Delete Block",         "Block deletion requested"),
    "template_11": ("Block Deleted",        "Block successfully deleted"),
    "template_12": ("Close Block",          "Block write stream closed"),
    "template_13": ("Pipeline Setup",       "Replication pipeline being set up"),
    "template_14": ("Packet Responder End", "PacketResponder thread terminating"),
    "template_15": ("Connection Reset",     "Client connection reset"),
    "template_16": ("IOException",          "IOException during block transfer"),
    "template_17": ("Timeout",              "Operation timed out"),
    "template_18": ("Checksum Error",       "Block checksum mismatch detected"),
    "template_19": ("Corrupt Block",        "Block reported as corrupted"),
    "template_20": ("Replication Failed",   "Block replication attempt failed"),
    "template_21": ("Recover Block",        "Block recovery initiated"),
    "template_22": ("Recovery Done",        "Block recovery completed"),
    "template_23": ("NameNode Report",      "DataNode reports to NameNode"),
    "template_24": ("Heartbeat",            "DataNode heartbeat sent"),
    "template_25": ("Register",             "DataNode registration with NameNode"),
    "template_26": ("Decommission",         "DataNode decommission started"),
    "template_27": ("Under Replicated",     "Block flagged as under-replicated"),
    "template_28": ("Over Replicated",      "Block flagged as over-replicated"),
    "template_29": ("Finalize Block",       "Block finalization complete"),
    "template_30": ("Lease Recovery",       "Lease recovery initiated"),
    "template_31": ("Abandon Block",        "Block write abandoned by client"),
    "template_32": ("Retry Write",          "Write retry attempt"),
    "template_33": ("Read Block",           "Block read request received"),
    "template_34": ("Seek",                 "Block seek operation"),
    "template_35": ("Socket Exception",     "SocketException during transfer"),
    "template_36": ("Block Report",         "DataNode block report sent"),
    "template_37": ("Missing Block",        "Block missing from DataNode"),
    "template_38": ("Invalid Block",        "Invalid block ID referenced"),
    "template_39": ("Access Token",         "Block access token validated"),
    "template_40": ("Quota Exceeded",       "Namespace quota exceeded"),
    "template_41": ("Rename",               "File rename operation"),
    "template_42": ("Mkdir",                "Directory creation"),
    "template_43": ("Permission Denied",    "Permission check failed"),
    "template_44": ("Client Request",       "Client RPC request received"),
    "template_45": ("Safe Mode",            "NameNode entering/leaving safe mode"),
    "template_46": ("Unknown Event",        "Unclassified log event"),
}

HDFS_CATEGORIES = {
    "🟢 Normal Operations": [f"template_{i}" for i in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,24,25,29,33,34,36,39,41,42,44]],
    "🟡 Replication & Recovery": [f"template_{i}" for i in [15,21,22,23,26,27,28,30,31,32,45]],
    "🔴 Errors & Anomalies": [f"template_{i}" for i in [16,17,18,19,20,35,37,38,40,43,46]],
}

HDFS_SCENARIOS = {
    "✅ Normal Write": ["template_1","template_2","template_4","template_5","template_4","template_5","template_12","template_9","template_14","template_6"],
    "⚠️ Failed Replication": ["template_1","template_2","template_4","template_5","template_7","template_13","template_17","template_20","template_27","template_21"],
    "🚨 Corrupt Block": ["template_1","template_2","template_4","template_18","template_19","template_16","template_35","template_37","template_38","template_46"],
}

# ---------------------------------------------------------------------------
# Model definitions (must match autoencoder.py)
# ---------------------------------------------------------------------------
class LSTMEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config["vocab_size"], config["embed_dim"], padding_idx=config["padding_idx"])
        self.lstm = nn.LSTM(config["embed_dim"], config["hidden_dim"], num_layers=config["num_layers"], batch_first=True, dropout=config["dropout"] if config["num_layers"] > 1 else 0.0)
        self.hidden_to_latent = nn.Linear(config["hidden_dim"], config["latent_dim"])

    def forward(self, x):
        emb = self.embedding(x)
        _, (h_n, _) = self.lstm(emb)
        latent = self.hidden_to_latent(h_n[-1])
        return latent, emb


class LSTMDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.latent_to_hidden = nn.Linear(config["latent_dim"], config["embed_dim"])
        self.lstm = nn.LSTM(config["embed_dim"], config["hidden_dim"], num_layers=config["num_layers"], batch_first=True, dropout=config["dropout"] if config["num_layers"] > 1 else 0.0)
        self.output_projection = nn.Linear(config["hidden_dim"], config["embed_dim"])

    def forward(self, latent):
        B, T = latent.size(0), self.config["window_size"]
        proj = self.latent_to_hidden(latent).unsqueeze(1).expand(B, T, -1)
        out, _ = self.lstm(proj)
        return self.output_projection(out)


class LSTMAutoencoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = LSTMEncoder(config)
        self.decoder = LSTMDecoder(config)

    @torch.no_grad()
    def score(self, x):
        latent, target_emb = self.encoder(x)
        recon = self.decoder(latent)
        per_token = F.mse_loss(recon, target_emb, reduction="none").mean(dim=2)  # (B, T)
        overall = per_token.mean(dim=1)  # (B,)
        return overall[0].item(), per_token[0].cpu().numpy(), latent[0].cpu().numpy()


class AnomalyClassifier(nn.Module):
    def __init__(self, latent_dim, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32),         nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(32, n_classes),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
@st.cache_resource
def load_autoencoder(checkpoint_path):
    ckpt   = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg    = ckpt["config"]
    config = dict(
        vocab_size=cfg.vocab_size, embed_dim=cfg.embed_dim,
        hidden_dim=cfg.hidden_dim, latent_dim=cfg.latent_dim,
        num_layers=cfg.num_layers, dropout=cfg.dropout,
        window_size=cfg.window_size, padding_idx=0,
    )
    model = LSTMAutoencoder(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, config


@st.cache_resource
def load_classifier(classifier_path):
    ckpt = torch.load(classifier_path, map_location="cpu", weights_only=False)
    clf  = AnomalyClassifier(ckpt["latent_dim"], ckpt["n_classes"])
    clf.load_state_dict(ckpt["model_state_dict"])
    clf.eval()
    return clf, ckpt["idx_to_class"], ckpt["label_map"]


@st.cache_data
def load_test_data(test_X_path, test_y_path):
    X = np.load(test_X_path)
    y = np.load(test_y_path).astype(int)
    return X, y


@st.cache_data
def load_label_map(label_map_path):
    if Path(label_map_path).exists():
        with open(label_map_path) as f:
            return json.load(f)
    return {}


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------
def classify_anomaly(latent_vec, clf, idx_to_class, label_map):
    """Run classifier on latent vector, return category name."""
    x = torch.tensor(latent_vec, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        logits = clf(x)
        idx    = logits.argmax(dim=1).item()
    class_int = idx_to_class.get(idx, idx)
    return label_map.get(str(class_int), f"Type {class_int}")


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------
def score_gauge_svg(score, threshold, color):
    import math
    pct   = min(score / max(threshold * 2, 1e-6), 1.0)
    angle = -140 + pct * 280
    rad   = math.radians(angle - 90)
    nx    = 150 + 90 * math.cos(rad)
    ny    = 160 + 90 * math.sin(rad)
    c     = "#ef4444" if score >= threshold else "#22c55e"
    label = "ANOMALOUS" if score >= threshold else "NORMAL"
    return f"""
    <svg viewBox="0 0 300 200" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:300px;margin:auto;display:block">
      <defs>
        <linearGradient id="arc" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" stop-color="#22c55e"/>
          <stop offset="60%" stop-color="#f59e0b"/>
          <stop offset="100%" stop-color="#ef4444"/>
        </linearGradient>
      </defs>
      <path d="M 30 160 A 120 120 0 0 1 270 160" fill="none" stroke="#1e293b" stroke-width="18" stroke-linecap="round"/>
      <path d="M 30 160 A 120 120 0 0 1 270 160" fill="none" stroke="url(#arc)" stroke-width="14" stroke-linecap="round" opacity="0.85"/>
      <line x1="150" y1="160" x2="{nx:.1f}" y2="{ny:.1f}" stroke="{c}" stroke-width="4" stroke-linecap="round"/>
      <circle cx="150" cy="160" r="7" fill="{c}"/>
      <text x="150" y="135" text-anchor="middle" font-family="monospace" font-size="22" fill="{c}" font-weight="bold">{score:.4f}</text>
      <text x="150" y="192" text-anchor="middle" font-family="monospace" font-size="15" fill="{c}" font-weight="bold" letter-spacing="2">{label}</text>
      <text x="22" y="185" text-anchor="middle" font-family="monospace" font-size="10" fill="#64748b">0.0</text>
      <text x="278" y="185" text-anchor="middle" font-family="monospace" font-size="10" fill="#64748b">HIGH</text>
    </svg>"""


def timeline_figure(scores, threshold, color):
    """Rolling score timeline chart."""
    fig, ax = plt.subplots(figsize=(8, 2.2))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#0f172a")

    xs = list(range(len(scores)))
    ys = list(scores)

    # Fill under curve
    ax.fill_between(xs, ys, alpha=0.15, color=color)
    ax.plot(xs, ys, color=color, linewidth=1.5, alpha=0.9)

    # Threshold line
    ax.axhline(y=threshold, color="#ef4444", linestyle="--", linewidth=1, alpha=0.7, label=f"threshold={threshold:.3f}")

    # Mark anomalies
    for i, (x, y) in enumerate(zip(xs, ys)):
        if y >= threshold:
            ax.scatter(x, y, color="#ef4444", s=25, zorder=5)

    ax.set_xlim(0, max(len(scores) - 1, 1))
    ax.set_ylabel("MSE Score", color="#64748b", fontsize=9)
    ax.tick_params(colors="#475569", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#1e293b")
    ax.legend(fontsize=8, facecolor="#0f172a", labelcolor="#64748b", framealpha=0.5)
    fig.tight_layout(pad=0.5)
    return fig


def heatmap_figure(sequence, per_token_scores, vocab_inv, label_map=None):
    n      = len(sequence)
    scores = per_token_scores[:n]
    if label_map:
        labels = [label_map.get(str(t), str(t)) for t in sequence]
    else:
        labels = [TEMPLATE_LABELS.get(t, (str(t), ""))[0] for t in sequence]

    vmax = max(float(scores.max()), 1e-6)
    norm = mcolors.Normalize(vmin=0, vmax=vmax)
    cmap = plt.cm.RdYlGn_r

    fig, ax = plt.subplots(figsize=(max(6, n * 0.9), 2.2))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#0f172a")

    for i, (label, s) in enumerate(zip(labels, scores)):
        color = cmap(norm(float(s)))
        ax.add_patch(plt.Rectangle([i, 0], 0.9, 0.8, color=color, linewidth=0))
        ax.text(i + 0.45, 0.4, str(label)[:12], ha="center", va="center",
                fontsize=7, color="white", fontweight="bold", rotation=45)
        ax.text(i + 0.45, -0.15, f"{float(s):.3f}", ha="center", va="top",
                fontsize=7, color="#94a3b8")

    ax.set_xlim(0, n)
    ax.set_ylim(-0.35, 1.1)
    ax.axis("off")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.04, pad=0.05)
    cbar.ax.tick_params(colors="#94a3b8", labelsize=8)
    cbar.set_label("Reconstruction Error", color="#94a3b8", fontsize=9)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Log Anomaly IDS",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Outfit:wght@300;400;600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
    background-color: #080c14;
    color: #cbd5e1;
}
h1, h2, h3 { font-family: 'Outfit', sans-serif; font-weight: 800; }
code, .mono { font-family: 'Space Mono', monospace; }

.ds-pill {
    display: inline-block;
    border-radius: 20px;
    padding: 6px 18px;
    font-family: 'Space Mono', monospace;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 1px;
    cursor: pointer;
    border: 1.5px solid transparent;
    transition: all 0.2s;
}
.alert-item {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 8px 12px;
    margin: 4px 0;
    border-radius: 8px;
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    border-left: 3px solid;
}
.alert-anomaly {
    background: rgba(239,68,68,0.08);
    border-color: #ef4444;
    color: #fca5a5;
}
.alert-normal {
    background: rgba(34,197,94,0.05);
    border-color: #22c55e;
    color: #86efac;
}
.stat-box {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 10px;
    padding: 14px 16px;
    text-align: center;
}
.stat-val {
    font-family: 'Space Mono', monospace;
    font-size: 24px;
    font-weight: 700;
    line-height: 1;
}
.stat-lbl {
    font-size: 11px;
    color: #475569;
    margin-top: 4px;
    text-transform: uppercase;
    letter-spacing: 1px;
}
.verdict-normal {
    background: linear-gradient(135deg, #052e16, #0a3622);
    border: 2px solid #22c55e;
    border-radius: 12px;
    padding: 14px;
    text-align: center;
    font-size: 22px;
    font-weight: 800;
    color: #22c55e;
    letter-spacing: 3px;
    font-family: 'Space Mono', monospace;
}
.verdict-anomaly {
    background: linear-gradient(135deg, #2d0a0a, #3b0f0f);
    border: 2px solid #ef4444;
    border-radius: 12px;
    padding: 14px;
    text-align: center;
    font-size: 22px;
    font-weight: 800;
    color: #ef4444;
    letter-spacing: 3px;
    font-family: 'Space Mono', monospace;
    animation: pulse 1.2s infinite;
}
.category-badge {
    display: inline-block;
    background: rgba(245,158,11,0.15);
    border: 1px solid #f59e0b;
    border-radius: 6px;
    padding: 3px 10px;
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    color: #fbbf24;
    margin-top: 6px;
}
@keyframes pulse {
    0%   { box-shadow: 0 0 0 0 rgba(239,68,68,0.4); }
    70%  { box-shadow: 0 0 0 12px rgba(239,68,68,0); }
    100% { box-shadow: 0 0 0 0 rgba(239,68,68,0); }
}
.section-label {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: #334155;
    margin-bottom: 10px;
}
div[data-testid="stButton"] > button {
    background: #0f172a;
    color: #94a3b8;
    border: 1px solid #1e293b;
    border-radius: 8px;
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    transition: all 0.15s;
}
div[data-testid="stButton"] > button:hover {
    background: #1e293b;
    color: #e2e8f0;
    border-color: #334155;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
for key, default in [
    ("dataset", "HDFS"),
    ("mode", "LIVE STREAM"),
    ("stream_running", False),
    ("stream_idx", 0),
    ("scores_history", deque(maxlen=80)),
    ("alerts", deque(maxlen=20)),
    ("total_sequences", 0),
    ("total_anomalies", 0),
    ("builder_sequence", []),
    ("custom_sequence", []),
    ("custom_result", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🛡️ IDS Control Panel")
    st.markdown("---")

    # Dataset selector
    st.markdown('<div class="section-label">Environment</div>', unsafe_allow_html=True)
    for ds_name, ds_cfg in DATASETS.items():
        color = ds_cfg["color"]
        if st.button(
            f"{ds_cfg['icon']}  {ds_name}",
            key=f"ds_{ds_name}",
            use_container_width=True,
        ):
            if st.session_state.dataset != ds_name:
                st.session_state.dataset         = ds_name
                st.session_state.stream_running  = False
                st.session_state.stream_idx      = 0
                st.session_state.scores_history  = deque(maxlen=80)
                st.session_state.alerts          = deque(maxlen=20)
                st.session_state.total_sequences = 0
                st.session_state.total_anomalies = 0
                st.rerun()

    st.markdown("---")

    # Mode selector
    st.markdown('<div class="section-label">Mode</div>', unsafe_allow_html=True)
    mode = st.radio(
        "Mode", ["LIVE STREAM", "BUILDER", "CUSTOM"],
        index=["LIVE STREAM", "BUILDER", "CUSTOM"].index(
            st.session_state.mode if st.session_state.mode in ["LIVE STREAM", "BUILDER", "CUSTOM"] else "LIVE STREAM"
        ),
        label_visibility="collapsed",
    )
    st.session_state.mode = mode

    st.markdown("---")

    ds_cfg    = DATASETS[st.session_state.dataset]
    threshold = st.slider(
        "Anomaly threshold",
        min_value=0.001, max_value=0.1,
        value=ds_cfg["threshold"],
        step=0.001, format="%.3f",
    )

    if mode == "LIVE STREAM":
        stream_speed = st.slider("Stream speed (seq/sec)", 1, 10, 3)
    
    st.markdown("---")
    st.markdown(
        f'<div style="font-family:Space Mono,monospace;font-size:10px;color:#334155;">'
        f'{ds_cfg["description"]}</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Load resources for selected dataset
# ---------------------------------------------------------------------------
ds_cfg   = DATASETS[st.session_state.dataset]
ds_color = ds_cfg["color"]

model_loaded = False
clf_loaded   = False
model        = None
clf          = None
idx_to_class = {}
label_map_clf = {}

if Path(ds_cfg["checkpoint"]).exists():
    try:
        model, ae_config = load_autoencoder(ds_cfg["checkpoint"])
        model_loaded = True
    except Exception as e:
        st.sidebar.error(f"Model load failed: {e}")

if ds_cfg["classifier"] and Path(ds_cfg["classifier"]).exists():
    try:
        clf, idx_to_class, label_map_clf = load_classifier(ds_cfg["classifier"])
        clf_loaded = True
    except Exception as e:
        st.sidebar.warning(f"Classifier not loaded: {e}")

label_map = load_label_map(ds_cfg["label_map"])

test_X, test_y = None, None
if Path(ds_cfg["test_X"]).exists():
    test_X, test_y = load_test_data(ds_cfg["test_X"], ds_cfg["test_y"])


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
col_title, col_status = st.columns([3, 1])
with col_title:
    st.markdown(f"""
    <div style="padding-bottom:16px;border-bottom:1px solid #0f172a;">
      <h1 style="margin:0;font-size:1.9rem;color:#f8fafc;letter-spacing:-0.5px;">
        🛡️ Log Anomaly Detection IDS
      </h1>
      <p style="color:#334155;margin:4px 0 0;font-family:'Space Mono',monospace;font-size:11px;">
        {ds_cfg['icon']} {st.session_state.dataset} &nbsp;·&nbsp; LSTM Autoencoder &nbsp;·&nbsp; {st.session_state.mode}
      </p>
    </div>
    """, unsafe_allow_html=True)

with col_status:
    status_color = "#22c55e" if model_loaded else "#ef4444"
    status_text  = "ONLINE" if model_loaded else "OFFLINE"
    st.markdown(f"""
    <div style="text-align:right;padding-top:8px;">
      <span style="font-family:'Space Mono',monospace;font-size:11px;color:{status_color};">
        ● {status_text}
      </span>
      {"<br><span style='font-family:Space Mono,monospace;font-size:10px;color:#f59e0b;'>+ CLASSIFIER</span>" if clf_loaded else ""}
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='margin-top:16px;'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Stats row
# ---------------------------------------------------------------------------
s1, s2, s3, s4 = st.columns(4)
anomaly_rate = (st.session_state.total_anomalies / max(st.session_state.total_sequences, 1)) * 100

for col, val, lbl, color in [
    (s1, st.session_state.total_sequences, "Sequences Analyzed", ds_color),
    (s2, st.session_state.total_anomalies, "Anomalies Detected",  "#ef4444"),
    (s3, f"{anomaly_rate:.1f}%",           "Anomaly Rate",        "#f59e0b"),
    (s4, f"{threshold:.3f}",               "Threshold",           "#64748b"),
]:
    col.markdown(f"""
    <div class="stat-box">
      <div class="stat-val" style="color:{color};">{val}</div>
      <div class="stat-lbl">{lbl}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='margin-top:16px;'></div>", unsafe_allow_html=True)


# ===========================================================================
# LIVE STREAM MODE
# ===========================================================================
if st.session_state.mode == "LIVE STREAM":

    left, right = st.columns([1.6, 1], gap="large")

    with left:
        # Timeline
        st.markdown('<div class="section-label">Reconstruction Score Timeline</div>', unsafe_allow_html=True)
        timeline_placeholder = st.empty()

        # Current sequence heatmap
        st.markdown('<div class="section-label">Current Window — Per-Token Error</div>', unsafe_allow_html=True)
        heatmap_placeholder = st.empty()

    with right:
        # Gauge + verdict
        st.markdown('<div class="section-label">Current Sequence</div>', unsafe_allow_html=True)
        gauge_placeholder   = st.empty()
        verdict_placeholder = st.empty()
        category_placeholder = st.empty()

        st.markdown("<div style='margin-top:12px;'></div>", unsafe_allow_html=True)

        # Controls
        c1, c2 = st.columns(2)
        start_btn = c1.button("▶ Start Stream", use_container_width=True, disabled=not model_loaded or test_X is None)
        stop_btn  = c2.button("⏹ Stop",         use_container_width=True)

        if start_btn:
            st.session_state.stream_running = True
        if stop_btn:
            st.session_state.stream_running = False

        st.markdown("<div style='margin-top:12px;'></div>", unsafe_allow_html=True)

        # Alert feed
        st.markdown('<div class="section-label">Alert Feed</div>', unsafe_allow_html=True)
        alerts_placeholder = st.empty()

    # --- Render static state ---
    if st.session_state.scores_history:
        fig = timeline_figure(list(st.session_state.scores_history), threshold, ds_color)
        timeline_placeholder.pyplot(fig, use_container_width=True)
        plt.close(fig)

    def render_alerts():
        if not st.session_state.alerts:
            alerts_placeholder.markdown(
                '<p style="color:#1e293b;font-family:Space Mono,monospace;font-size:11px;">Waiting for stream...</p>',
                unsafe_allow_html=True,
            )
            return
        html = ""
        for alert in reversed(list(st.session_state.alerts)):
            cls  = "alert-anomaly" if alert["anomaly"] else "alert-normal"
            icon = "🚨" if alert["anomaly"] else "✅"
            cat  = f" · <b>{alert['category']}</b>" if alert.get("category") else ""
            html += f'<div class="alert-item {cls}">{icon} #{alert["idx"]} &nbsp; MSE={alert["score"]:.4f}{cat}</div>'
        alerts_placeholder.markdown(html, unsafe_allow_html=True)

    render_alerts()

    # --- Stream loop ---
    if st.session_state.stream_running and model_loaded and test_X is not None:
        idx = st.session_state.stream_idx % len(test_X)
        x_np = test_X[idx:idx+1]
        x    = torch.tensor(x_np, dtype=torch.long)

        score, per_token, latent = model.score(x)

        is_anomaly = score >= threshold
        st.session_state.scores_history.append(score)
        st.session_state.total_sequences += 1

        category = None
        if is_anomaly:
            st.session_state.total_anomalies += 1
            if clf_loaded:
                category = classify_anomaly(latent, clf, idx_to_class, label_map_clf)

        st.session_state.alerts.append({
            "idx":      st.session_state.total_sequences,
            "score":    score,
            "anomaly":  is_anomaly,
            "category": category,
        })

        st.session_state.stream_idx += 1

        # Render timeline
        fig = timeline_figure(list(st.session_state.scores_history), threshold, ds_color)
        timeline_placeholder.pyplot(fig, use_container_width=True)
        plt.close(fig)

        # Render heatmap
        seq_tokens = x_np[0].tolist()
        fig2 = heatmap_figure(seq_tokens, per_token, {}, label_map)
        heatmap_placeholder.pyplot(fig2, use_container_width=True)
        plt.close(fig2)

        # Gauge
        gauge_placeholder.markdown(score_gauge_svg(score, threshold, ds_color), unsafe_allow_html=True)

        # Verdict
        if is_anomaly:
            verdict_placeholder.markdown('<div class="verdict-anomaly">🚨 ANOMALOUS</div>', unsafe_allow_html=True)
        else:
            verdict_placeholder.markdown('<div class="verdict-normal">✅ NORMAL</div>', unsafe_allow_html=True)

        # Category badge (BGL / Thunderbird only)
        if category:
            category_placeholder.markdown(
                f'<div style="text-align:center;margin-top:8px;">'
                f'<span class="category-badge">⚠️ {category}</span></div>',
                unsafe_allow_html=True,
            )
        else:
            category_placeholder.empty()

        render_alerts()

        time.sleep(1.0 / stream_speed)
        st.rerun()

    elif not model_loaded:
        st.info(f"Checkpoint not found at `{ds_cfg['checkpoint']}` — waiting for training to complete.")
    elif test_X is None:
        st.info(f"Test data not found at `{ds_cfg['test_X']}`")


# ===========================================================================
# BUILDER MODE (HDFS only)
# ===========================================================================
elif st.session_state.mode == "BUILDER":
    if st.session_state.dataset != "HDFS":
        st.info("Builder mode is currently available for HDFS only. Switch to HDFS in the sidebar, or use Live Stream mode for BGL and Thunderbird.")
    else:
        left, right = st.columns([1.1, 1], gap="large")

        with left:
            st.markdown('<div class="section-label">Quick Scenarios</div>', unsafe_allow_html=True)
            s_cols = st.columns(len(HDFS_SCENARIOS))
            for col, (name, events) in zip(s_cols, HDFS_SCENARIOS.items()):
                if col.button(name, use_container_width=True):
                    st.session_state.builder_sequence = list(events)

            st.markdown("<div style='margin-top:12px;'></div>", unsafe_allow_html=True)
            st.markdown('<div class="section-label">Event Palette</div>', unsafe_allow_html=True)

            window_size = ae_config["window_size"] if model_loaded else 20

            for category, templates in HDFS_CATEGORIES.items():
                with st.expander(category, expanded=(category == "🟢 Normal Operations")):
                    rows = [templates[i:i+3] for i in range(0, len(templates), 3)]
                    for row in rows:
                        cols = st.columns(3)
                        for col, tmpl in zip(cols, row):
                            short, desc = TEMPLATE_LABELS.get(tmpl, (tmpl, ""))
                            if col.button(short, key=f"btn_{tmpl}", help=desc, use_container_width=True):
                                if len(st.session_state.builder_sequence) < window_size:
                                    st.session_state.builder_sequence.append(tmpl)

            st.markdown("<div style='margin-top:12px;'></div>", unsafe_allow_html=True)
            st.markdown('<div class="section-label">Current Sequence</div>', unsafe_allow_html=True)

            seq = st.session_state.builder_sequence
            if seq:
                tokens_html = "".join(
                    f'<span style="display:inline-block;background:#0f172a;border:1px solid #1e293b;'
                    f'border-radius:6px;padding:3px 8px;margin:2px;font-family:Space Mono,monospace;'
                    f'font-size:11px;color:{ds_color};">'
                    f'{TEMPLATE_LABELS.get(t,(t,""))[0]}</span>'
                    for t in seq
                )
                st.markdown(tokens_html, unsafe_allow_html=True)
                st.caption(f"{len(seq)} / {window_size} tokens")
            else:
                st.markdown('<p style="color:#1e293b;font-family:Space Mono,monospace;font-size:12px;">Add events above to begin...</p>', unsafe_allow_html=True)

            c1, c2 = st.columns(2)
            if c1.button("⬅️ Remove last", use_container_width=True):
                if st.session_state.builder_sequence:
                    st.session_state.builder_sequence.pop()
            if c2.button("🗑️ Clear", use_container_width=True):
                st.session_state.builder_sequence = []

        with right:
            st.markdown('<div class="section-label">Live Analysis</div>', unsafe_allow_html=True)
            seq = st.session_state.builder_sequence

            if not seq:
                st.markdown('<div style="color:#1e293b;font-family:Space Mono,monospace;font-size:12px;padding:60px 0;text-align:center;">Add events to see analysis</div>', unsafe_allow_html=True)
            else:
                if model_loaded:
                    vocab = json.load(open(ds_cfg["vocab_file"]))
                    ids   = [vocab.get(t, 0) for t in seq]
                    ids   = ids[:window_size] + [0] * max(0, window_size - len(ids))
                    x     = torch.tensor([ids], dtype=torch.long)
                    score, per_token, latent = model.score(x)
                else:
                    score     = float(np.random.uniform(0.01, 0.12))
                    per_token = np.random.uniform(0.001, 0.15, 20)
                    latent    = np.zeros(32)
                    st.info("Preview mode — load checkpoint for real scores.")

                is_anomaly = score >= threshold

                if is_anomaly:
                    st.markdown('<div class="verdict-anomaly">🚨 ANOMALOUS</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="verdict-normal">✅ NORMAL</div>', unsafe_allow_html=True)

                st.markdown("<div style='margin-top:10px;'></div>", unsafe_allow_html=True)
                st.markdown(score_gauge_svg(score, threshold, ds_color), unsafe_allow_html=True)

                st.markdown("<div style='margin-top:10px;'></div>", unsafe_allow_html=True)
                m1, m2, m3 = st.columns(3)
                m1.metric("MSE Score", f"{score:.4f}")
                m2.metric("Threshold", f"{threshold:.3f}")
                m3.metric("Margin",    f"{score - threshold:+.4f}", delta_color="inverse")

                st.markdown("<div style='margin-top:10px;'></div>", unsafe_allow_html=True)
                st.markdown('<div class="section-label">Per-Event Reconstruction Error</div>', unsafe_allow_html=True)
                fig = heatmap_figure(seq, per_token, {})
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

                if model_loaded and seq:
                    worst_idx   = int(per_token[:len(seq)].argmax())
                    worst_event = TEMPLATE_LABELS.get(seq[worst_idx], (seq[worst_idx], ""))[0]
                    st.markdown(
                        f'<p style="color:#475569;font-size:11px;font-family:Space Mono,monospace;">'
                        f'Highest error at position {worst_idx+1}: '
                        f'<strong style="color:#fbbf24">{worst_event}</strong> '
                        f'(MSE={per_token[worst_idx]:.4f})</p>',
                        unsafe_allow_html=True,
                    )

# ===========================================================================
# CUSTOM MODE — build and analyze any sequence for any dataset
# ===========================================================================
elif st.session_state.mode == "CUSTOM":

    st.markdown('<div class="section-label">Custom Sequence Analyzer</div>', unsafe_allow_html=True)
    st.markdown(
        '<p style="color:#475569;font-family:Space Mono,monospace;font-size:11px;margin-bottom:16px;">'
        'Build any sequence from the active dataset\'s vocabulary and score it live. '
        'Works for all three datasets — use this to demonstrate specific fault patterns.</p>',
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.2, 1], gap="large")

    with left:
        # Build vocab options from label_map + vocab file
        vocab_options = {}
        if Path(ds_cfg["vocab_file"]).exists():
            raw_vocab = json.load(open(ds_cfg["vocab_file"]))
            for template_key, token_id in raw_vocab.items():
                if template_key == "<PAD>":
                    continue
                # Use label_map name if available, else template key
                display_name = label_map.get(str(token_id), template_key)
                vocab_options[f"{template_key} — {display_name}"] = template_key

        if not vocab_options:
            st.warning("Vocab file not found. Cannot build custom sequences.")
        else:
            window_size = ae_config["window_size"] if model_loaded else 20

            st.markdown('<div class="section-label">Select Events (up to 20)</div>', unsafe_allow_html=True)

            # Multiselect — user picks templates by display name
            selected_display = st.multiselect(
                "Events",
                options=list(vocab_options.keys()),
                default=[list(vocab_options.keys())[i] for i in range(min(5, len(vocab_options)))],
                max_selections=window_size,
                label_visibility="collapsed",
                help="Select up to 20 log events in order. The model will score this sequence.",
            )

            # Convert display names back to template keys
            selected_templates = [vocab_options[d] for d in selected_display]
            st.session_state.custom_sequence = selected_templates

            st.markdown(f"<p style='color:#334155;font-family:Space Mono,monospace;font-size:10px;'>{len(selected_templates)}/{window_size} events selected</p>", unsafe_allow_html=True)

            st.markdown("<div style='margin-top:10px;'></div>", unsafe_allow_html=True)

            # Show selected sequence as token chips
            if selected_templates:
                chips = "".join(
                    f'<span style="display:inline-block;background:#0f172a;border:1px solid #1e293b;'
                    f'border-radius:6px;padding:3px 8px;margin:2px;font-family:Space Mono,monospace;'
                    f'font-size:10px;color:{ds_color};">'
                    f'{label_map.get(str(raw_vocab.get(t,0)), t)}</span>'
                    for t in selected_templates
                )
                st.markdown(chips, unsafe_allow_html=True)

            st.markdown("<div style='margin-top:14px;'></div>", unsafe_allow_html=True)

            analyze_btn = st.button(
                "🔍 Analyze Sequence",
                use_container_width=True,
                disabled=not model_loaded or len(selected_templates) == 0,
            )

            if analyze_btn and selected_templates and model_loaded:
                raw_vocab = json.load(open(ds_cfg["vocab_file"]))
                ids  = [raw_vocab.get(t, 0) for t in selected_templates]
                ids  = ids[:window_size] + [0] * max(0, window_size - len(ids))
                x    = torch.tensor([ids], dtype=torch.long)
                score, per_token, latent = model.score(x)

                category = None
                if score >= threshold and clf_loaded:
                    category = classify_anomaly(latent, clf, idx_to_class, label_map_clf)

                st.session_state.custom_result = {
                    "score":      score,
                    "per_token":  per_token,
                    "latent":     latent,
                    "sequence":   selected_templates,
                    "is_anomaly": score >= threshold,
                    "category":   category,
                }

    with right:
        result = st.session_state.custom_result

        if result is None:
            st.markdown(
                '<div style="color:#1e293b;font-family:Space Mono,monospace;font-size:12px;'
                'padding:80px 0;text-align:center;">Select events and click Analyze</div>',
                unsafe_allow_html=True,
            )
        else:
            score      = result["score"]
            per_token  = result["per_token"]
            is_anomaly = result["is_anomaly"]
            category   = result["category"]
            seq        = result["sequence"]

            # Verdict
            if is_anomaly:
                st.markdown('<div class="verdict-anomaly">🚨 ANOMALOUS</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="verdict-normal">✅ NORMAL</div>', unsafe_allow_html=True)

            # Category badge
            if category:
                st.markdown(
                    f'<div style="text-align:center;margin-top:8px;">'
                    f'<span class="category-badge">⚠️ {category}</span></div>',
                    unsafe_allow_html=True,
                )

            st.markdown("<div style='margin-top:10px;'></div>", unsafe_allow_html=True)
            st.markdown(score_gauge_svg(score, threshold, ds_color), unsafe_allow_html=True)

            st.markdown("<div style='margin-top:10px;'></div>", unsafe_allow_html=True)
            m1, m2, m3 = st.columns(3)
            m1.metric("MSE Score",  f"{score:.4f}")
            m2.metric("Threshold",  f"{threshold:.3f}")
            m3.metric("Margin",     f"{score - threshold:+.4f}", delta_color="inverse")

            st.markdown("<div style='margin-top:12px;'></div>", unsafe_allow_html=True)
            st.markdown('<div class="section-label">Per-Event Reconstruction Error</div>', unsafe_allow_html=True)

            fig = heatmap_figure(seq, per_token, {}, label_map)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            if seq:
                worst_idx  = int(per_token[:len(seq)].argmax())
                worst_name = label_map.get(
                    str(raw_vocab.get(seq[worst_idx], 0) if Path(ds_cfg["vocab_file"]).exists() else 0),
                    seq[worst_idx]
                )
                st.markdown(
                    f'<p style="color:#475569;font-size:11px;font-family:Space Mono,monospace;">'
                    f'Highest error at position {worst_idx+1}: '
                    f'<strong style="color:#fbbf24">{worst_name}</strong> '
                    f'(MSE={per_token[worst_idx]:.4f})</p>',
                    unsafe_allow_html=True,
                )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    '<p style="color:#1e293b;font-size:10px;font-family:Space Mono,monospace;text-align:center;">'
    'LSTM Autoencoder · Malhotra et al. (2016) · HDFS / BGL / Thunderbird · Graduate AI Course Project'
    '</p>',
    unsafe_allow_html=True,
)