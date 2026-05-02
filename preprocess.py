"""
preprocess.py  —  Multi-dataset LSTM Autoencoder Preprocessing Pipeline
========================================================================
Supports: HDFS, BGL, Spirit, Thunderbird

Each dataset is handled by a DatasetAdapter subclass that knows:
  • how to read raw log lines
  • how to parse a block/session key from each line
  • how to extract a fine-grained anomaly label (not just binary)
  • how to load ground-truth labels (inline flags or external CSV)

The rest of the pipeline (Drain parsing, vocab, windowing, stratified
split, PyTorch DataLoaders) is dataset-agnostic.

Usage
-----
    python preprocess.py --dataset hdfs      --log HDFS.log       --labels anomaly_label.csv
    python preprocess.py --dataset bgl       --log BGL.log
    python preprocess.py --dataset spirit    --log Spirit.log
    python preprocess.py --dataset thunderbird --log Thunderbird.log

Optional flags (apply to all datasets):
    --window-size 20   (default 20)
    --batch-size  256  (default 256)
    --output-dir  lstm_data

Multi-class label conventions
------------------------------
HDFS        : Normal | Anomaly  (external CSV; extend by replacing 'Anomaly'
              column with specific attack names if your CSV has them)
BGL         : Normal | (fault alert code, e.g. 'APPREAD', 'KERNSEG', ...)
              The BGL format encodes '-' for normal lines and the alert type
              for anomalous ones — no external label file needed.
Spirit      : Normal | (alert type extracted from inline flag field)
              Same inline convention as BGL.
Thunderbird : Normal | (component/severity-derived type from inline flag)
              Same inline convention as BGL/Spirit.

All adapters produce {block_id -> label_string} dicts consumed by the
shared encode_labels() step, so every dataset feeds a consistent
multi-class integer pipeline downstream.
"""

from __future__ import annotations

import abc
import re
import json
import argparse
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig
from drain3.masking import MaskingInstruction


# ── Global defaults ────────────────────────────────────────────────────
WINDOW_SIZE = 20
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
BATCH_SIZE  = 256
SEED        = 42


# ══════════════════════════════════════════════════════════════════════
# DATASET ADAPTERS
# ══════════════════════════════════════════════════════════════════════

class DatasetAdapter(abc.ABC):
    """
    Abstract base class.  Subclasses implement:
      load()  ->  (raw_lines: list[str], labels_dict: dict[str, str])

    labels_dict maps block_id -> label_string.
    'Normal' must be one of the label strings (case-sensitive).
    All other strings become separate anomaly classes.
    """

    name: str = "base"

    def __init__(self, log_path: str, label_path: str | None = None):
        self.log_path   = Path(log_path)
        self.label_path = Path(label_path) if label_path else None

    @abc.abstractmethod
    def load(self) -> tuple[list[str], dict[str, str]]:
        """Return (raw_log_lines, {block_id: label_string})."""
        ...

    # ── Shared helpers ─────────────────────────────────────────────────

    def _read_lines(self) -> list[str]:
        print(f"    Reading {self.log_path} ...")
        with open(self.log_path, encoding="utf-8", errors="replace") as f:
            return f.readlines()


# ── HDFS ───────────────────────────────────────────────────────────────

class HDFSAdapter(DatasetAdapter):
    """
    HDFS-1 dataset.

    Log format  : standard Hadoop log lines, each containing a blk_<id>.
    Labels      : external CSV with columns BlockId, Label.
                  Label can be 'Normal', 'Anomaly', or any finer-grained
                  string if your CSV already has attack categories.

    Session key : blk_<id>  extracted from every log line with a regex.
    One line can belong to multiple blocks (rare but possible); we assign
    the line to every matching block.
    """

    name = "hdfs"
    BLK_RE = re.compile(r"(blk_-?\d+)")

    def load(self) -> tuple[list[str], dict[str, str]]:
        raw_lines = self._read_lines()

        if self.label_path is None:
            raise ValueError("HDFS requires --labels <anomaly_label.csv>")

        print(f"    Reading {self.label_path} ...")
        df = pd.read_csv(self.label_path)
        df.columns = df.columns.str.strip()

        # Accept 'Label' column with arbitrary string values.
        # Binary datasets typically have 'Normal'/'Anomaly'; richer CSVs
        # can have 'DDoS', 'PortScan', etc.  All are handled identically.
        labels_dict: dict[str, str] = {
            str(row["BlockId"]).strip(): str(row["Label"]).strip()
            for _, row in df.iterrows()
        }

        print(f"    Lines loaded    : {len(raw_lines):,}")
        print(f"    Blocks labelled : {len(labels_dict):,}")
        print(f"    Unique labels   : {sorted(set(labels_dict.values()))}")
        return raw_lines, labels_dict

    def extract_block_id(self, line: str) -> list[str]:
        """Return all blk_<id> tokens found in a single log line."""
        return self.BLK_RE.findall(line)


class BGLAdapter(DatasetAdapter):
    """
    Blue Gene/L (BGL) supercomputer log dataset.

    Log format (space-separated fields):
        Field 0 : alert flag  ('-' = normal, else the alert/fault type,
                               e.g. 'APPREAD', 'KERNSEG', 'RAS', ...)
        Field 1 : timestamp (Unix epoch)
        Field 2 : date
        Field 3 : node (e.g. 'R02-M1-N0-C:J12-U11')
        Field 4 : time
        Field 5 : node (repeat)
        Field 6 : message type
        Field 7+: log message

    Session key : node identifier (field 3).
                  BGL logs come from supercomputer nodes; grouping by
                  node produces a natural session that represents the
                  health of one compute node over the log window.

    Labels      : derived inline from field 0.
                  '-' -> 'Normal'
                  anything else -> the raw alert code (e.g. 'APPREAD')
                  This gives fine-grained multi-class anomaly types
                  without any external file.

    Note: Some researchers use fixed-size sliding windows over the sorted
    log instead of node-based sessions.  Node grouping is used here
    because it maps cleanly to the shared session-based pipeline.
    """

    name = "bgl"

    def load(self) -> tuple[list[str], dict[str, str]]:
        raw_lines = self._read_lines()

        node_labels: dict[str, list[str]] = defaultdict(list)

        for line in raw_lines:
            parts = line.split()
            if len(parts) < 7:
                continue

            alert_flag = parts[0]          # '-' or alert code
            node_id    = parts[3]          # e.g. 'R02-M1-N0-C:J12-U11'

            label = "Normal" if alert_flag == "-" else alert_flag
            node_labels[node_id].append(label)

        # A node session is Anomalous if ANY line is flagged.
        # If anomalous, use the most common anomaly type as the session label
        # (preserves multi-class granularity even for mixed sessions).
        labels_dict: dict[str, str] = {}
        for node_id, lbls in node_labels.items():
            anomaly_lbls = [l for l in lbls if l != "Normal"]
            if anomaly_lbls:
                labels_dict[node_id] = Counter(anomaly_lbls).most_common(1)[0][0]
            else:
                labels_dict[node_id] = "Normal"

        print(f"    Lines loaded    : {len(raw_lines):,}")
        print(f"    Node sessions   : {len(labels_dict):,}")
        print(f"    Unique labels   : {sorted(set(labels_dict.values()))[:10]} "
              f"({'...' if len(set(labels_dict.values())) > 10 else ''})")
        return raw_lines, labels_dict

    def extract_block_id(self, line: str) -> list[str]:
        parts = line.split()
        return [parts[3]] if len(parts) >= 7 else []


class SpiritAdapter(DatasetAdapter):
    """
    Spirit supercomputer log dataset (Sandia National Laboratories).

    Log format (space-separated):
        Field 0 : label flag  ('-' = normal, else alert type, e.g. 'FATAL')
        Field 1 : Unix timestamp
        Field 2 : date string
        Field 3 : node / compute unit identifier
        Field 4 : time string
        Field 5+: log message

    Session key : node identifier (field 3), same rationale as BGL.
    Labels      : inline field 0, same convention as BGL.
                  Common Spirit anomaly types: 'FATAL', 'ERROR', 'FAILURE',
                  'WARNING' (depending on the release of the dataset).

    Spirit is significantly larger than BGL (~500 M lines in the full
    release).  If memory is a concern, pass --max-lines N to cap ingestion.
    """

    name = "spirit"

    def __init__(self, log_path: str, label_path: str | None = None,
                 max_lines: int | None = None):
        super().__init__(log_path, label_path)
        self.max_lines = max_lines

    def load(self) -> tuple[list[str], dict[str, str]]:
        print(f"    Reading {self.log_path} ...")
        raw_lines: list[str] = []
        with open(self.log_path, encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f):
                if self.max_lines and i >= self.max_lines:
                    break
                raw_lines.append(line)

        node_labels: dict[str, list[str]] = defaultdict(list)

        for line in raw_lines:
            parts = line.split()
            if len(parts) < 5:
                continue
            alert_flag = parts[0]
            node_id    = parts[3]
            label = "Normal" if alert_flag == "-" else alert_flag
            node_labels[node_id].append(label)

        labels_dict: dict[str, str] = {}
        for node_id, lbls in node_labels.items():
            anomaly_lbls = [l for l in lbls if l != "Normal"]
            if anomaly_lbls:
                labels_dict[node_id] = Counter(anomaly_lbls).most_common(1)[0][0]
            else:
                labels_dict[node_id] = "Normal"

        print(f"    Lines loaded    : {len(raw_lines):,}")
        print(f"    Node sessions   : {len(labels_dict):,}")
        print(f"    Unique labels   : {sorted(set(labels_dict.values()))[:10]}")
        return raw_lines, labels_dict

    def extract_block_id(self, line: str) -> list[str]:
        parts = line.split()
        return [parts[3]] if len(parts) >= 5 else []


class ThunderbirdAdapter(DatasetAdapter):
    """
    Thunderbird supercomputer log dataset (Sandia National Laboratories).

    Log format (space-separated):
        Field 0 : label flag  ('-' = normal, else alert type)
        Field 1 : Unix timestamp
        Field 2 : date (YYYY.MM.DD)
        Field 3 : user (often 'tbird-' prefixed node)
        Field 4 : month
        Field 5 : day
        Field 6 : time
        Field 7 : node / component identifier
        Field 8 : message level / component tag
        Field 9+: log message

    Session key : node/component identifier (field 7).
                  Thunderbird logs originate from a heterogeneous cluster;
                  field 7 identifies the specific node or daemon.

    Labels      : inline field 0 (same '-' / alert-code convention).
                  Common Thunderbird anomaly codes: 'FATAL', 'ERROR',
                  'FAILURE', 'kernel', 'tpmd', etc.

    Fine-grained label note: Thunderbird anomaly flags are coarser than
    BGL fault codes (fewer distinct types).  You can further split labels
    by combining the flag with field 8 (component tag) if you want
    component-level granularity — see _derive_label() below.
    """

    name = "thunderbird"

    def __init__(self, log_path: str, label_path: str | None = None,
                 component_split: bool = False,
                 max_lines: int | None = None):
        super().__init__(log_path, label_path)
        self.component_split = component_split  # combine flag + component tag
        self.max_lines       = max_lines

    def _derive_label(self, alert_flag: str, component_tag: str) -> str:
        if alert_flag == "-":
            return "Normal"
        if self.component_split:
            # e.g. 'FATAL::tpmd', 'ERROR::kernel'  — richer taxonomy
            return f"{alert_flag}::{component_tag}"
        return alert_flag

    def load(self) -> tuple[list[str], dict[str, str]]:
        print(f"    Reading {self.log_path} ...")
        raw_lines: list[str] = []
        with open(self.log_path, encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f):
                if self.max_lines and i >= self.max_lines:
                    break
                raw_lines.append(line)

        node_labels: dict[str, list[str]] = defaultdict(list)

        for line in raw_lines:
            parts = line.split()
            if len(parts) < 9:
                continue
            alert_flag    = parts[0]
            node_id       = parts[7]
            component_tag = parts[8]
            label = self._derive_label(alert_flag, component_tag)
            node_labels[node_id].append(label)

        labels_dict: dict[str, str] = {}
        for node_id, lbls in node_labels.items():
            anomaly_lbls = [l for l in lbls if l != "Normal"]
            if anomaly_lbls:
                labels_dict[node_id] = Counter(anomaly_lbls).most_common(1)[0][0]
            else:
                labels_dict[node_id] = "Normal"

        print(f"    Lines loaded    : {len(raw_lines):,}")
        print(f"    Node sessions   : {len(labels_dict):,}")
        print(f"    Unique labels   : {sorted(set(labels_dict.values()))[:10]}")
        return raw_lines, labels_dict

    def extract_block_id(self, line: str) -> list[str]:
        parts = line.split()
        return [parts[7]] if len(parts) >= 9 else []


# ── Registry ───────────────────────────────────────────────────────────

ADAPTER_REGISTRY: dict[str, type[DatasetAdapter]] = {
    "hdfs":        HDFSAdapter,
    "bgl":         BGLAdapter,
    "spirit":      SpiritAdapter,
    "thunderbird": ThunderbirdAdapter,
}


def get_adapter(dataset: str, log_path: str,
                label_path: str | None = None, **kwargs) -> DatasetAdapter:
    key = dataset.lower()
    if key not in ADAPTER_REGISTRY:
        raise ValueError(
            f"Unknown dataset '{dataset}'. "
            f"Choices: {sorted(ADAPTER_REGISTRY.keys())}")
    return ADAPTER_REGISTRY[key](log_path, label_path, **kwargs)


# ══════════════════════════════════════════════════════════════════════
# 0.  LABEL ENCODING
# ══════════════════════════════════════════════════════════════════════

def encode_labels(labels_dict: dict[str, str],
                  verbose: bool = True) -> tuple[dict[str, int], dict[int, str]]:
    """
    Convert {blk_id -> label_string} to {blk_id -> int} and a label_map.

    Encoding:
      0         -> 'Normal'  (always)
      1, 2, ... -> anomaly types sorted alphabetically (stable across runs)
    """
    unique_labels = sorted(set(labels_dict.values()))

    if "Normal" not in unique_labels:
        raise ValueError(
            "No 'Normal' class found. "
            "Verify your log/CSV uses 'Normal' for non-anomalous entries.")

    # Normal always first so it encodes as 0
    ordered = ["Normal"] + [l for l in unique_labels if l != "Normal"]
    str_to_id    = {cls: i for i, cls in enumerate(ordered)}
    encoded_dict = {blk: str_to_id[lbl] for blk, lbl in labels_dict.items()}
    label_map    = {i: cls for i, cls in enumerate(ordered)}

    assert label_map[0] == "Normal"

    if verbose:
        print(f"\n[1] Label Encoding")
        counts = Counter(labels_dict.values())
        print(f"    Classes found   : {len(label_map)}")
        for cls_id, cls_name in sorted(label_map.items()):
            tag = " <- class 0 (training only)" if cls_id == 0 else ""
            print(f"      {cls_id:>3}: {cls_name:<28} ({counts[cls_name]:>8,} sessions){tag}")

    return encoded_dict, label_map


def save_label_map(label_map: dict[int, str], path: Path) -> None:
    with open(path, "w") as f:
        json.dump({str(k): v for k, v in label_map.items()}, f, indent=2)
    print(f"    Saved to        : {path}")


# ══════════════════════════════════════════════════════════════════════
# 1.  DRAIN PARSING
# ══════════════════════════════════════════════════════════════════════

def _build_drain_config() -> TemplateMinerConfig:
    cfg = TemplateMinerConfig()
    cfg.drain_sim_th       = 0.5
    cfg.drain_depth        = 4
    cfg.drain_max_children = 100
    cfg.drain_max_clusters = None
    cfg.masking_instructions = [
        MaskingInstruction(r"blk_-?\d+",                                      "<BLK>"),
        MaskingInstruction(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(:\d+)?",    "<IP>"),
        MaskingInstruction(r"(?<=[^A-Za-z0-9])(\-?\+?\d+)(?=[^A-Za-z0-9])|[0-9]+$", "<NUM>"),
        # Node identifiers common in BGL/Spirit/Thunderbird
        MaskingInstruction(r"R\d{2}-M\d-N\d-[A-Z]:[A-Z]\d{2}-U\d{2}",       "<NODE>"),
    ]
    return cfg


def drain_parse(raw_lines: list[str],
                adapter: DatasetAdapter,
                verbose: bool = True):
    """
    Parse raw log lines with Drain3, using the adapter to extract the
    session key (block_id / node_id / etc.) from each line.

    Returns:
        structured : list of dicts with keys block_id, template_id, template
        miner      : fitted TemplateMiner (used later for vocab building)
    """
    miner           = TemplateMiner(config=_build_drain_config())
    structured      = []
    template_counts: Counter = Counter()

    for line in raw_lines:
        line = line.strip()
        if not line:
            continue

        block_ids   = adapter.extract_block_id(line)
        result      = miner.add_log_message(line)
        template_id = result["cluster_id"]
        template_counts[template_id] += 1

        if not block_ids:
            # Lines without a recognisable session key are still parsed
            # (they contribute to Drain's template learning) but won't
            # appear in any session.
            structured.append({
                "block_id":    None,
                "template_id": template_id,
                "template":    result["template_mined"],
            })
        else:
            for blk in block_ids:
                structured.append({
                    "block_id":    blk,
                    "template_id": template_id,
                    "template":    result["template_mined"],
                })

    if verbose:
        n_tpl = len(miner.drain.id_to_cluster)
        print(f"\n[2] Drain Parsing")
        print(f"    Lines parsed    : {len(raw_lines):,}")
        print(f"    Templates found : {n_tpl}  (IDs 1–{n_tpl}, 0 reserved for PAD)")
        top5 = template_counts.most_common(5)
        print(f"    Top-5 templates :")
        for tid, cnt in top5:
            tmpl = miner.drain.id_to_cluster[tid].get_template()
            print(f"      [{tid:3d}] ({cnt:6,}x)  {tmpl[:72]}")

    return structured, miner


# ══════════════════════════════════════════════════════════════════════
# 2.  SESSION GROUPING
# ══════════════════════════════════════════════════════════════════════

def group_by_session(structured: list[dict],
                     encoded_dict: dict[str, int],
                     label_map: dict[int, str],
                     verbose: bool = True) -> list[dict]:
    """
    Group parsed log lines by their session key (block_id / node_id).
    Each session becomes one training/eval sample.

    Lines whose block_id is absent from encoded_dict are skipped (no label).
    """
    sessions_map: dict[str, list[int]] = defaultdict(list)
    for row in structured:
        if row["block_id"] and row["block_id"] in encoded_dict:
            sessions_map[row["block_id"]].append(row["template_id"])

    sessions = [
        {
            "block_id":       blk_id,
            "event_sequence": seq,
            "label":          encoded_dict[blk_id],
        }
        for blk_id, seq in sessions_map.items()
    ]

    if verbose:
        label_counts = Counter(s["label"] for s in sessions)
        skipped = sum(1 for r in structured
                      if r["block_id"] and r["block_id"] not in encoded_dict)
        print(f"\n[3] Session Grouping")
        print(f"    Total sessions  : {len(sessions):,}")
        for cls_id, cls_name in sorted(label_map.items()):
            tag = " <- training only" if cls_id == 0 else ""
            print(f"      {cls_id:>3}: {cls_name:<28} "
                  f"{label_counts.get(cls_id, 0):>8,}{tag}")
        if skipped:
            print(f"    Skipped (no label) : {skipped:,}")
        seq_lens = [len(s["event_sequence"]) for s in sessions]
        print(f"    Seq length      : "
              f"min={min(seq_lens)}  "
              f"median={int(np.median(seq_lens))}  "
              f"max={max(seq_lens)}")

    return sessions


# ══════════════════════════════════════════════════════════════════════
# 3.  VOCABULARY
# ══════════════════════════════════════════════════════════════════════

def build_vocab(miner: TemplateMiner,
                output_path: Path | None = None,
                verbose: bool = True) -> dict[str, int]:
    vocab: dict[str, int] = {"<PAD>": 0}
    for tid in sorted(miner.drain.id_to_cluster.keys()):
        vocab[f"template_{tid}"] = tid

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(vocab, f, indent=2)

    if verbose:
        print(f"\n[4] Vocabulary")
        print(f"    Vocab size      : {len(vocab)}  (<PAD>=0)")
        print(f"    Token range     : 0–{max(vocab.values())}")
        if output_path:
            print(f"    Saved to        : {output_path}")

    return vocab


# ══════════════════════════════════════════════════════════════════════
# 4.  WINDOWING
# ══════════════════════════════════════════════════════════════════════

def _pad_or_truncate(seq: list[int], window_size: int, pad_id: int = 0) -> list[int]:
    if len(seq) >= window_size:
        return seq[-window_size:]
    return seq + [pad_id] * (window_size - len(seq))


def build_windows(sessions: list[dict],
                  window_size: int,
                  verbose: bool = True) -> tuple[np.ndarray, np.ndarray, list[str]]:
    X   = np.array([_pad_or_truncate(s["event_sequence"], window_size) for s in sessions], dtype=np.int64)
    y   = np.array([s["label"] for s in sessions],    dtype=np.int64)
    ids = [s["block_id"] for s in sessions]

    if verbose:
        n_trunc  = sum(1 for s in sessions if len(s["event_sequence"]) > window_size)
        n_padded = sum(1 for s in sessions if len(s["event_sequence"]) < window_size)
        print(f"\n[5] Fixed-Length Windows  (window_size={window_size})")
        print(f"    Output shape    : {X.shape}  (sessions × tokens)")
        print(f"    Truncated       : {n_trunc:,}  ({100*n_trunc/len(sessions):.1f}%)")
        print(f"    Padded          : {n_padded:,}  ({100*n_padded/len(sessions):.1f}%)")

    return X, y, ids


# ══════════════════════════════════════════════════════════════════════
# 5.  STRATIFIED SPLIT
# ══════════════════════════════════════════════════════════════════════

def split_datasets(X: np.ndarray,
                   y: np.ndarray,
                   label_map: dict[int, str],
                   train_ratio: float = TRAIN_RATIO,
                   val_ratio:   float = VAL_RATIO,
                   seed:        int   = SEED,
                   verbose:     bool  = True):
    """
    train  : class-0 (Normal) only, train_ratio fraction of normals
    val    : val_ratio of normals + 50% of each anomaly class
    test   : remainder of normals + 50% of each anomaly class

    Each anomaly class is split independently so rare classes appear in
    both val and test rather than accidentally landing entirely in one.
    """
    rng = np.random.default_rng(seed)

    # Normal
    norm_idx = np.where(y == 0)[0]
    rng.shuffle(norm_idx)
    n_train = int(len(norm_idx) * train_ratio)
    n_val   = int(len(norm_idx) * val_ratio)

    train_idx    = norm_idx[:n_train]
    val_norm_idx = norm_idx[n_train : n_train + n_val]
    tst_norm_idx = norm_idx[n_train + n_val :]

    # Anomalies — stratified per class
    val_anom, tst_anom = [], []
    for cls_id in sorted(k for k in label_map if k != 0):
        cls_idx = np.where(y == cls_id)[0]
        rng.shuffle(cls_idx)
        n_val_cls = max(1, len(cls_idx) // 2)
        val_anom.append(cls_idx[:n_val_cls])
        tst_anom.append(cls_idx[n_val_cls:])

    val_idx = np.concatenate([val_norm_idx, *val_anom])
    tst_idx = np.concatenate([tst_norm_idx, *tst_anom])

    splits = {
        "Train": (X[train_idx], y[train_idx]),
        "Val":   (X[val_idx],   y[val_idx]),
        "Test":  (X[tst_idx],   y[tst_idx]),
    }

    if verbose:
        print(f"\n[6] Stratified Train / Val / Test Split")
        for split_name, (x_arr, y_arr) in splits.items():
            counts = Counter(y_arr.tolist())
            print(f"\n    {split_name} ({len(x_arr):,} sessions):")
            for cls_id, cls_name in sorted(label_map.items()):
                cnt = counts.get(cls_id, 0)
                bar = "█" * min(30, max(1, int(30 * cnt / max(len(y_arr), 1))))
                print(f"      {cls_id:>3} {cls_name:<28} {cnt:>7,}  {bar}")

        if (splits["Train"][1] != 0).any():
            print("\n    ⚠  WARNING: non-Normal sessions in training set!")

    return splits["Train"], splits["Val"], splits["Test"]


# ══════════════════════════════════════════════════════════════════════
# 6.  PYTORCH DATASETS & DATALOADERS
# ══════════════════════════════════════════════════════════════════════

class UnlabelledDataset(Dataset):
    """Training split: yields x only (autoencoder reconstructs x from x)."""
    def __init__(self, X: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.long)
    def __len__(self):  return len(self.X)
    def __getitem__(self, idx): return self.X[idx]


class LabelledDataset(Dataset):
    """Val / Test: yields (x, label) where label is a multi-class integer."""
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):  return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]


def build_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test,
                      batch_size: int = BATCH_SIZE, verbose: bool = True):
    train_ds = UnlabelledDataset(X_train)
    val_ds   = LabelledDataset(X_val, y_val)
    test_ds  = LabelledDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, drop_last=False)

    if verbose:
        print(f"\n[7] DataLoaders  (batch_size={batch_size})")
        print(f"    train_loader : {len(train_loader):,} batches → x (B, {X_train.shape[1]}) int64")
        print(f"    val_loader   : {len(val_loader):,}   batches → (x, label)")
        print(f"    test_loader  : {len(test_loader):,}  batches → (x, label)")

        x_b = next(iter(train_loader))
        print(f"\n    train batch shape : {tuple(x_b.shape)}  "
              f"token range {x_b.min().item()}–{x_b.max().item()}")
        xv, yv = next(iter(val_loader))
        print(f"    val   batch shape : x={tuple(xv.shape)}  y={tuple(yv.shape)}")
        print(f"    val   classes     : {sorted(yv.unique().tolist())}  (0=Normal)")

    return train_loader, val_loader, test_loader


# ══════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════

def run_pipeline(dataset:     str,
                 log_path:    str,
                 label_path:  str | None = None,
                 window_size: int        = WINDOW_SIZE,
                 batch_size:  int        = BATCH_SIZE,
                 output_dir:  Path       = Path("lstm_data"),
                 **adapter_kwargs) -> dict:
    """
    Full preprocessing pipeline.  Returns a dict with:
        train_loader, val_loader, test_loader  — PyTorch DataLoaders
        vocab, vocab_size                      — token vocabulary
        label_map, num_classes                 — class integer ↔ name mapping
        window_size                            — tokens per session
        miner                                  — fitted Drain3 TemplateMiner
    """
    print("=" * 64)
    print(f"  LSTM Preprocessing Pipeline  |  dataset={dataset.upper()}")
    print(f"  window_size={window_size}  batch_size={batch_size}")
    print("=" * 64)

    output_dir.mkdir(parents=True, exist_ok=True)

    # 0. Adapter → raw lines + label strings
    print(f"\n[0] Loading {dataset.upper()} dataset ...")
    adapter = get_adapter(dataset, log_path, label_path, **adapter_kwargs)
    raw_lines, labels_dict = adapter.load()

    # 1. Encode labels
    encoded_dict, label_map = encode_labels(labels_dict)
    save_label_map(label_map, output_dir / "label_map.json")

    # 2. Drain parsing
    structured, miner = drain_parse(raw_lines, adapter)

    # 3. Session grouping
    sessions = group_by_session(structured, encoded_dict, label_map)

    # 4. Vocabulary
    vocab = build_vocab(miner, output_path=output_dir / "vocab.json")

    # 5. Windowing
    X, y, block_ids = build_windows(sessions, window_size)

    # 6. Stratified split
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = \
        split_datasets(X, y, label_map)

    # Save arrays
    np.save(output_dir / "X_train.npy", X_train)
    np.save(output_dir / "X_val.npy",   X_val);  np.save(output_dir / "y_val.npy",  y_val)
    np.save(output_dir / "X_test.npy",  X_test); np.save(output_dir / "y_test.npy", y_test)
    print(f"\n    Arrays saved to {output_dir}/")

    # 7. DataLoaders
    train_loader, val_loader, test_loader = build_dataloaders(
        X_train, y_train, X_val, y_val, X_test, y_test, batch_size=batch_size)

    n_classes = len(label_map)
    print("\n" + "=" * 64)
    print("  Pipeline complete")
    print("=" * 64)
    print(f"  num_classes  -> {n_classes}   (pass to LSTM output layer)")
    print(f"  vocab_size   -> {len(vocab)}  (pass to nn.Embedding)")
    print(f"  window_size  -> {window_size}")
    print(f"  label_map    -> {output_dir / 'label_map.json'}")
    print(f"  vocab.json   -> {output_dir / 'vocab.json'}")
    print()

    return {
        "train_loader": train_loader,
        "val_loader":   val_loader,
        "test_loader":  test_loader,
        "vocab":        vocab,
        "vocab_size":   len(vocab),
        "label_map":    label_map,
        "num_classes":  n_classes,
        "window_size":  window_size,
        "miner":        miner,
    }


# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multi-dataset LSTM autoencoder preprocessing pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required
    parser.add_argument(
        "--dataset", required=True,
        choices=sorted(ADAPTER_REGISTRY.keys()),
        help="Log dataset to process",
    )
    parser.add_argument(
        "--log", required=True, metavar="PATH",
        help="Path to the raw log file",
    )

    # Optional per-dataset
    parser.add_argument(
        "--labels", default=None, metavar="PATH",
        help="External label CSV (required for HDFS; ignored for BGL/Spirit/Thunderbird)",
    )
    parser.add_argument(
        "--component-split", action="store_true",
        help="Thunderbird only: derive label as ALERTCODE::component for finer classes",
    )
    parser.add_argument(
        "--max-lines", type=int, default=None, metavar="N",
        help="Spirit/Thunderbird only: cap number of log lines read (useful for huge files)",
    )

    # Pipeline hyper-parameters
    parser.add_argument("--window-size", type=int, default=WINDOW_SIZE)
    parser.add_argument("--batch-size",  type=int, default=BATCH_SIZE)
    parser.add_argument("--output-dir",  type=Path, default=Path("lstm_data"))

    args = parser.parse_args()

    # Extra kwargs forwarded to the adapter constructor
    adapter_kwargs: dict = {}
    if args.dataset == "thunderbird":
        adapter_kwargs["component_split"] = args.component_split
    if args.dataset in ("spirit", "thunderbird") and args.max_lines:
        adapter_kwargs["max_lines"] = args.max_lines

    run_pipeline(
        dataset     = args.dataset,
        log_path    = args.log,
        label_path  = args.labels,
        window_size = args.window_size,
        batch_size  = args.batch_size,
        output_dir  = args.output_dir,
        **adapter_kwargs,
    )
