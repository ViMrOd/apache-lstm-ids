"""
HDFS Log Preprocessing Pipeline — LSTM Autoencoder
===========================================================
Run:
    python hdfs_pipeline.py --batch-size 128   # override batch size
    python hdfs_pipeline.py --window-size 30   # override window size
"""

import re
import json
import argparse
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig
from drain3.masking import MaskingInstruction


# ── Paths ─────────────────────────────────────────────────────────────
HDFS_LOG_PATH   = "HDFS.log"
HDFS_LABEL_PATH = "anomaly_label.csv"
OUTPUT_DIR      = Path("lstm_data")

# ── Defaults ──────────────────────────────────────────────────────────
WINDOW_SIZE  = 20     # fixed sequence length expected by LSTM
TRAIN_RATIO  = 0.7    # normal sessions used for training
VAL_RATIO    = 0.15   # normal sessions used for validation
BATCH_SIZE   = 256
SEED         = 42


# ═══════════════════════════════════════════════════════════════════════
# 1.  DRAIN PARSING
# ═══════════════════════════════════════════════════════════════════════

def build_drain_config():
    cfg = TemplateMinerConfig()
    cfg.drain_sim_th       = 0.5
    cfg.drain_depth        = 4
    cfg.drain_max_children = 100
    cfg.drain_max_clusters = None
    cfg.masking_instructions = [
        MaskingInstruction(r"blk_-?\d+", "<BLK>"),
        MaskingInstruction(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(:\d+)?", "<IP>"),
        MaskingInstruction(
            r"(?<=[^A-Za-z0-9])(\-?\+?\d+)(?=[^A-Za-z0-9])|[0-9]+$", "<NUM>"),
    ]
    return cfg


def drain_parse(raw_lines, verbose=True):
    """
    Stage 1 - Drain Parsing
    -----------------------
    Converts free-form log text into integer template IDs.

    Drain assigns every unique log pattern a stable cluster_id starting
    at 1. Token ID 0 is intentionally left unused here - it is reserved
    as the PAD token in the LSTM vocabulary.

    Returns:
      structured  - list of dicts: {block_id, template_id, template}
      miner       - fitted TemplateMiner (carries the id->template map)
    """
    config  = build_drain_config()
    miner   = TemplateMiner(config=config)
    blk_re  = re.compile(r"(blk_-?\d+)")
    structured      = []
    template_counts = defaultdict(int)

    for line in raw_lines:
        line = line.strip()
        if not line:
            continue
        blk_match   = blk_re.search(line)
        block_id    = blk_match.group(1) if blk_match else None
        result      = miner.add_log_message(line)
        template_id = result["cluster_id"]   # integer >= 1, never 0
        template_counts[template_id] += 1
        structured.append({
            "block_id":    block_id,
            "template_id": template_id,
            "template":    result["template_mined"],
        })

    if verbose:
        n_tpl = len(miner.drain.id_to_cluster)
        print(f"\n[1] Drain Parsing")
        print(f"    Lines parsed    : {len(structured):,}")
        print(f"    Templates found : {n_tpl}  (IDs 1-{n_tpl}, 0 reserved for PAD)")
        top5 = sorted(template_counts.items(), key=lambda x: -x[1])[:5]
        print(f"    Top-5 templates:")
        for tid, cnt in top5:
            tmpl = miner.drain.id_to_cluster[tid].get_template()
            print(f"      [{tid:3d}] ({cnt:5d}x)  {tmpl[:75]}")

    return structured, miner


# ═══════════════════════════════════════════════════════════════════════
# 2.  BLOCK ID GROUPING
# ═══════════════════════════════════════════════════════════════════════

def group_by_block_id(structured, labels_dict, verbose=True):
    """
    Stage 2 - Block ID Grouping
    ---------------------------
    Assembles each block's full event sequence from the parsed rows.

    Output: list of dicts
      { block_id: str,
        event_sequence: [int, ...],   <- raw template IDs, variable length
        label: 0 | 1 }
    """
    sessions_map = defaultdict(list)
    for row in structured:
        if row["block_id"]:
            sessions_map[row["block_id"]].append(row["template_id"])

    sessions, skipped = [], 0
    for blk_id, seq in sessions_map.items():
        if blk_id not in labels_dict:
            skipped += 1
            continue
        sessions.append({
            "block_id":       blk_id,
            "event_sequence": seq,
            "label":          labels_dict[blk_id],
        })

    n_anom   = sum(1 for s in sessions if s["label"] == 1)
    n_normal = len(sessions) - n_anom

    if verbose:
        print(f"\n[2] Block ID Grouping")
        print(f"    Total sessions  : {len(sessions):,}")
        print(f"    Normal          : {n_normal:,}")
        print(f"    Anomaly         : {n_anom:,}  (label=1, kept out of training)")
        if skipped:
            print(f"    Skipped (no label): {skipped}")
        seq_lens = [len(s["event_sequence"]) for s in sessions]
        print(f"    Sequence length : min={min(seq_lens)}  "
              f"median={int(np.median(seq_lens))}  max={max(seq_lens)}")

    return sessions


# ═══════════════════════════════════════════════════════════════════════
# 3.  VOCAB
# ═══════════════════════════════════════════════════════════════════════

def build_vocab(miner, output_path=None, verbose=True):
    """
    Stage 3 - Vocabulary
    --------------------
    Maps every Drain template to a stable integer token ID.

    Convention (matches LSTM nn.Embedding expectation):
      0            -> <PAD>   (padding token - never produced by Drain)
      1 ... N      -> template_1 ... template_N  (Drain's own cluster IDs)

    Because Drain already assigns IDs starting at 1, the vocab is a
    direct pass-through - no remapping needed. We just materialise it
    explicitly so the LSTM can reconstruct the mapping at inference time.

    Saved as vocab.json:
      {"<PAD>": 0, "template_1": 1, "template_2": 2, ...}
    """
    vocab = {"<PAD>": 0}
    for tid in sorted(miner.drain.id_to_cluster.keys()):
        vocab[f"template_{tid}"] = tid

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(vocab, f, indent=2)

    if verbose:
        print(f"\n[3] Vocabulary")
        print(f"    Vocab size      : {len(vocab)}  (including <PAD>=0)")
        print(f"    Token ID range  : 0-{max(vocab.values())}")
        if output_path:
            print(f"    Saved to        : {output_path}")

    return vocab


# ═══════════════════════════════════════════════════════════════════════
# 4.  PAD / TRUNCATE  ->  FIXED-LENGTH WINDOWS
# ═══════════════════════════════════════════════════════════════════════

def pad_or_truncate(seq, window_size, pad_id=0):
    """
    Converts a variable-length event sequence to exactly window_size tokens.

    Truncation strategy: keep the LAST window_size tokens.
      Rationale: for HDFS, anomalous events tend to appear late in a
      block's lifecycle (during write/close). Keeping the tail preserves
      the most diagnostically relevant events.

    Padding strategy: right-pad with pad_id (0).
      Sequences shorter than window_size get zeros appended on the right.
      The LSTM's embedding layer maps 0 -> a learnable PAD vector that
      the model learns to ignore during training.
    """
    if len(seq) >= window_size:
        return seq[-window_size:]                      # truncate: keep tail
    return seq + [pad_id] * (window_size - len(seq))  # pad: append zeros


def build_windows(sessions, window_size, verbose=True):
    """
    Stage 4 - Fixed-Length Windows
    --------------------------------
    Applies pad_or_truncate to every session.

    Returns:
      X      - np.ndarray shape (N, window_size), dtype int64
      y      - np.ndarray shape (N,),             dtype int64
      ids    - list of block_id strings
    """
    X, y, ids = [], [], []
    for s in sessions:
        window = pad_or_truncate(s["event_sequence"], window_size)
        X.append(window)
        y.append(s["label"])
        ids.append(s["block_id"])

    X = np.array(X, dtype=np.int64)
    y = np.array(y, dtype=np.int64)

    if verbose:
        n_truncated = sum(
            1 for s in sessions if len(s["event_sequence"]) > window_size)
        n_padded = sum(
            1 for s in sessions if len(s["event_sequence"]) < window_size)
        print(f"\n[4] Fixed-Length Windows  (window_size={window_size})")
        print(f"    Output shape    : {X.shape}  (sessions x tokens)")
        print(f"    Truncated       : {n_truncated:,} sessions  "
              f"({100*n_truncated/len(sessions):.1f}%)")
        print(f"    Padded          : {n_padded:,} sessions  "
              f"({100*n_padded/len(sessions):.1f}%)")
        pad_counts = (X == 0).sum(axis=1)
        print(f"    Avg PAD tokens  : {pad_counts.mean():.1f} per session")

    return X, y, ids


# ═══════════════════════════════════════════════════════════════════════
# 5.  TRAIN / VAL / TEST SPLIT
# ═══════════════════════════════════════════════════════════════════════

def split_datasets(X, y, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO,
                   seed=SEED, verbose=True):
    """
    Stage 5 - Dataset Split
    -----------------------
    Autoencoder training requires a split strategy:

      TRAINING SET   -> normal sessions only (label=0)
                        The model learns what "normal" looks like.

      VALIDATION SET -> held-out normal sessions + half of all anomalies
                        Used to tune reconstruction-error threshold and
                        monitor training - needs both classes.

      TEST SET       -> remaining normal + remaining anomalies
                        Final held-out evaluation.

    The anomaly sessions are NEVER seen during training.

    Of the normal sessions:
      train_ratio (70%) -> train
      val_ratio   (15%) -> val
      remainder   (15%) -> test
    Anomalies split 50/50 between val and test.
    """
    rng = np.random.default_rng(seed)

    normal_idx = np.where(y == 0)[0]
    anom_idx   = np.where(y == 1)[0]

    rng.shuffle(normal_idx)
    rng.shuffle(anom_idx)

    n_train = int(len(normal_idx) * train_ratio)
    n_val   = int(len(normal_idx) * val_ratio)

    train_idx     = normal_idx[:n_train]
    val_norm_idx  = normal_idx[n_train : n_train + n_val]
    test_norm_idx = normal_idx[n_train + n_val:]

    n_anom_val    = len(anom_idx) // 2
    val_anom_idx  = anom_idx[:n_anom_val]
    test_anom_idx = anom_idx[n_anom_val:]

    val_idx  = np.concatenate([val_norm_idx,  val_anom_idx])
    test_idx = np.concatenate([test_norm_idx, test_anom_idx])

    X_train, y_train = X[train_idx], y[train_idx]
    X_val,   y_val   = X[val_idx],   y[val_idx]
    X_test,  y_test  = X[test_idx],  y[test_idx]

    if verbose:
        print(f"\n[5] Train / Val / Test Split")
        print(f"    Train  : {len(X_train):,} sessions  "
              f"({(y_train==0).sum():,} normal, {(y_train==1).sum()} anomaly)")
        print(f"    Val    : {len(X_val):,} sessions  "
              f"({(y_val==0).sum():,} normal, {(y_val==1).sum():,} anomaly)")
        print(f"    Test   : {len(X_test):,} sessions  "
              f"({(y_test==0).sum():,} normal, {(y_test==1).sum():,} anomaly)")
        if (y_train == 1).any():
            print("    WARNING: anomaly sessions leaked into training set!")

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


# ═══════════════════════════════════════════════════════════════════════
# 6.  PYTORCH DATASETS & DATALOADERS
# ═══════════════════════════════════════════════════════════════════════

class NormalSessionDataset(Dataset):
    """
    Training dataset - sequences only, no labels.
    The autoencoder receives x and tries to reconstruct x.
    Shape per item: (window_size,) int64
    """
    def __init__(self, X):
        self.X = torch.tensor(X, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]


class LabelledSessionDataset(Dataset):
    """
    Val / Test dataset - (sequence, label) pairs.
    Used to evaluate reconstruction error as an anomaly score:
      high error -> likely anomaly.
    Shape per item: (window_size,) int64, scalar int64
    """
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def build_dataloaders(X_train, y_train,
                      X_val,   y_val,
                      X_test,  y_test,
                      batch_size=BATCH_SIZE, verbose=True):
    """
    Stage 6 - DataLoaders
    ---------------------
    train_loader -> (B, window_size) int64, normal only, shuffled
    val_loader   -> ((B, window_size), (B,)) int64, not shuffled
    test_loader  -> ((B, window_size), (B,)) int64, not shuffled
    """
    train_ds = NormalSessionDataset(X_train)
    val_ds   = LabelledSessionDataset(X_val,  y_val)
    test_ds  = LabelledSessionDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, drop_last=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                              shuffle=False, drop_last=False)

    if verbose:
        print(f"\n[6] DataLoaders  (batch_size={batch_size})")
        print(f"    train_loader : {len(train_loader):,} batches  "
              f"-> x shape (B, {X_train.shape[1]}), dtype int64")
        print(f"    val_loader   : {len(val_loader):,} batches  "
              f"-> (x, label)")
        print(f"    test_loader  : {len(test_loader):,} batches  "
              f"-> (x, label)")

        # Spot-check shapes and dtypes
        x_batch = next(iter(train_loader))
        print(f"\n    train batch shape  : {tuple(x_batch.shape)}  "
              f"dtype={x_batch.dtype}")
        print(f"    train token range  : "
              f"{x_batch.min().item()}-{x_batch.max().item()}")
        assert x_batch.dtype == torch.int64, "dtype must be int64"
        assert x_batch.shape[1] == X_train.shape[1], "wrong window size"

        xv, yv = next(iter(val_loader))
        print(f"    val   x shape      : {tuple(xv.shape)}  dtype={xv.dtype}")
        print(f"    val   y shape      : {tuple(yv.shape)}  dtype={yv.dtype}")
        print(f"    val   label values : {yv.unique().tolist()}")

    return train_loader, val_loader, test_loader


# ═══════════════════════════════════════════════════════════════════════
# LOAD REAL DATA
# ═══════════════════════════════════════════════════════════════════════

def load_real_hdfs(log_path=HDFS_LOG_PATH, label_path=HDFS_LABEL_PATH):
    print(f"    Reading {log_path} ...")
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        raw_lines = f.readlines()
    print(f"    Reading {label_path} ...")
    df = pd.read_csv(label_path)
    df.columns = df.columns.str.strip()
    labels_dict = {
        row["BlockId"]: (0 if str(row["Label"]).strip() == "Normal" else 1)
        for _, row in df.iterrows()
    }
    print(f"    Lines loaded    : {len(raw_lines):,}")
    print(f"    Labels loaded   : {len(labels_dict):,}")
    return raw_lines, labels_dict


# ═══════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════

def run_pipeline(window_size=WINDOW_SIZE,
                 batch_size=BATCH_SIZE, output_dir=OUTPUT_DIR):

    print("=" * 60)
    print("  HDFS -> LSTM Autoencoder Preprocessing Pipeline")
    print(f"  window_size={window_size}  batch_size={batch_size}")
    print("=" * 60)

    output_dir.mkdir(parents=True, exist_ok=True)

    # 0. Data
    print("\n[0] Loading HDFS dataset ...")
    raw_lines, labels_dict = load_real_hdfs()

    # 1. Drain
    structured, miner = drain_parse(raw_lines)

    # 2. Group
    sessions = group_by_block_id(structured, labels_dict)

    # 3. Vocab
    vocab_path = output_dir / "vocab.json"
    vocab      = build_vocab(miner, output_path=vocab_path)

    # 4. Windows
    X, y, block_ids = build_windows(sessions, window_size)

    # 5. Split
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_datasets(X, y)
    
    # save arrays for LSTM
    np.save(output_dir / "X_train.npy", X_train)
    np.save(output_dir / "X_val.npy",   X_val)
    np.save(output_dir / "y_val.npy",   y_val)
    np.save(output_dir / "X_test.npy",  X_test)
    np.save(output_dir / "y_test.npy",  y_test)
    print(f"\n    Arrays saved to {output_dir}/")

    # 6. DataLoaders
    train_loader, val_loader, test_loader = build_dataloaders(
        X_train, y_train, X_val, y_val, X_test, y_test,
        batch_size=batch_size)

    print("\n" + "=" * 60)
    print("  Pipeline complete")
    print("=" * 60)
    print(f"  train_loader  -> (B, {window_size}) int64, normal only")
    print(f"  val_loader    -> (x=(B,{window_size}), label=(B,)) int64")
    print(f"  test_loader   -> (x=(B,{window_size}), label=(B,)) int64")
    print(f"  vocab_size    -> {len(vocab)}  (pass as num_embeddings to nn.Embedding)")
    print(f"  vocab.json    -> {vocab_path}")
    print()

    return {
        "train_loader": train_loader,
        "val_loader":   val_loader,
        "test_loader":  test_loader,
        "vocab":        vocab,
        "vocab_size":   len(vocab),
        "window_size":  window_size,
        "miner":        miner,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="HDFS preprocessing for LSTM autoencoder")
    parser.add_argument("--window-size", type=int, default=WINDOW_SIZE)
    parser.add_argument("--batch-size",  type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    result = run_pipeline(
        window_size=args.window_size,
        batch_size=args.batch_size,
    )
