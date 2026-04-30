"""
diagnose.py
===========
Inspect your data, vocabulary, and training setup to identify why the model
is collapsing to trivial reconstruction.

Run this BEFORE training to catch mismatches:
    python diagnose.py --data_dir data/ --vocab_file data/vocab.json
"""

import os
import sys
import json
import argparse
import numpy as np
import torch


def diagnose_vocab(vocab_file):
    """
    Load and inspect the vocabulary file.
    
    Expected format: either a JSON dict {token: idx} or list of tokens.
    """
    print("=" * 70)
    print("VOCABULARY INSPECTION")
    print("=" * 70)
    
    if not os.path.isfile(vocab_file):
        print(f"ERROR: vocab file not found: {vocab_file}")
        return None
    
    try:
        with open(vocab_file, 'r') as f:
            vocab_data = json.load(f)
    except Exception as e:
        print(f"ERROR: failed to load vocab: {e}")
        return None
    
    # Determine format
    if isinstance(vocab_data, dict):
        vocab_size = len(vocab_data)
        max_idx = max(int(v) if isinstance(v, (int, str)) else -1 for v in vocab_data.values())
        print(f"Vocab format    : dict (token → index)")
        print(f"Number of tokens: {vocab_size}")
        print(f"Max index       : {max_idx}")
        if max_idx + 1 != vocab_size:
            print(f"  ⚠️  WARNING: indices are not 0..{vocab_size-1}")
            print(f"      (gap or duplicate indices detected)")
    elif isinstance(vocab_data, list):
        vocab_size = len(vocab_data)
        print(f"Vocab format    : list")
        print(f"Number of tokens: {vocab_size}")
        print(f"First 5 tokens  : {vocab_data[:5]}")
    else:
        print(f"ERROR: unexpected vocab format: {type(vocab_data)}")
        return None
    
    print()
    return vocab_size


def diagnose_data(data_dir, vocab_size_expected=None):
    """
    Inspect the structure and ranges of the training data.
    """
    print("=" * 70)
    print("DATA INSPECTION")
    print("=" * 70)
    
    required_files = ["X_train.npy", "X_val.npy", "y_val.npy"]
    test_files = ["X_test.npy", "y_test.npy"]
    
    data = {}
    for fname in required_files:
        path = os.path.join(data_dir, fname)
        if not os.path.isfile(path):
            print(f"ERROR: required file missing: {fname}")
            return None
        try:
            arr = np.load(path)
            data[fname] = arr
            print(f"{fname:<20} shape: {arr.shape}  dtype: {arr.dtype}")
        except Exception as e:
            print(f"ERROR loading {fname}: {e}")
            return None
    
    # Optional test files
    for fname in test_files:
        path = os.path.join(data_dir, fname)
        if os.path.isfile(path):
            try:
                arr = np.load(path)
                data[fname] = arr
                print(f"{fname:<20} shape: {arr.shape}  dtype: {arr.dtype}")
            except Exception as e:
                print(f"WARNING: could not load {fname}: {e}")
    
    print()
    
    # Analyze token ranges
    print("TOKEN RANGE ANALYSIS")
    print("-" * 70)
    
    for arr_name in ["X_train.npy", "X_val.npy"]:
        if arr_name not in data:
            continue
        arr = data[arr_name]
        min_tok = arr.min()
        max_tok = arr.max()
        unique = len(np.unique(arr))
        print(
            f"{arr_name:<20} min: {min_tok:6}, max: {max_tok:6}, "
            f"unique: {unique:6}"
        )
        
        if vocab_size_expected is not None:
            if max_tok >= vocab_size_expected:
                print(
                    f"  ❌ CRITICAL: token index {max_tok} >= vocab_size {vocab_size_expected}"
                )
                print(f"     This causes undefined behavior (embedding table too small)!")
            elif max_tok >= vocab_size_expected * 0.9:
                print(
                    f"  ⚠️  WARNING: using ~{100*max_tok/vocab_size_expected:.0f}% of vocab table"
                )
    
    # Analyze labels
    if "y_val.npy" in data:
        y_val = data["y_val.npy"]
        n_normal = (y_val == 0).sum()
        n_anomaly = (y_val == 1).sum()
        anomaly_rate = n_anomaly / len(y_val) * 100
        print()
        print("ANOMALY LABEL DISTRIBUTION (y_val)")
        print("-" * 70)
        print(f"  Normal sequences  : {n_normal:8,} ({100-anomaly_rate:5.1f}%)")
        print(f"  Anomalous seqs    : {n_anomaly:8,} ({anomaly_rate:5.1f}%)")
        
        if n_anomaly == 0 or n_normal == 0:
            print(f"  ❌ PROBLEM: one class is missing!")
    
    print()
    return data


def diagnose_loss_scale(embed_dim, window_size, batch_size):
    """
    Estimate the typical loss scale for different reduction methods.
    """
    print("=" * 70)
    print("LOSS SCALE ANALYSIS")
    print("=" * 70)
    
    # Simulate random reconstructions with MSE ~1.0 per element
    B, T, E = batch_size, window_size, embed_dim
    
    # Random normal target and prediction (diff has std ~sqrt(2) ≈ 1.4)
    target = torch.randn(B, T, E)
    pred = target + torch.randn(B, T, E)
    mse_unreduced = torch.nn.functional.mse_loss(pred, target, reduction='none')
    
    # Method 1: reduction='mean' over all dims
    loss_mean_all = mse_unreduced.mean()
    
    # Method 2: per-sample mean (recommended)
    loss_per_sample = mse_unreduced.mean(dim=[1, 2]).mean()
    
    # Method 3: token-level mean
    loss_token_level = mse_unreduced.mean(dim=[0, 2]).mean()
    
    print(f"Config: B={B}, T={T}, E={E}")
    print(f"Total elements per batch: {B * T * E:,}")
    print()
    print("Loss scales for random predictions:")
    print(f"  reduction='mean' over all     : {loss_mean_all.item():.6f}")
    print(f"  per-sample mean (recommended) : {loss_per_sample.item():.6f}")
    print(f"  token-level mean              : {loss_token_level.item():.6f}")
    print()
    print("Why per-sample mean is better:")
    print("  - Independent of batch size, sequence length, embed_dim")
    print("  - Loss stays in the same range (typically 0.1–10.0)")
    print("  - Easier to interpret and debug")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose data/vocab issues before training"
    )
    parser.add_argument("--data_dir", type=str, default="data/",
                        help="Directory containing .npy files")
    parser.add_argument("--vocab_file", type=str, default="data/vocab.json",
                        help="Path to vocab file")
    parser.add_argument("--embed_dim", type=int, default=64,
                        help="Embedding dimension (for loss scale estimate)")
    parser.add_argument("--window_size", type=int, default=20,
                        help="Sequence length (for loss scale estimate)")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size (for loss scale estimate)")
    args = parser.parse_args()
    
    print("\n")
    
    # 1. Vocab
    vocab_size = diagnose_vocab(args.vocab_file)
    
    # 2. Data
    data = diagnose_data(args.data_dir, vocab_size_expected=vocab_size)
    
    # 3. Loss scale
    diagnose_loss_scale(args.embed_dim, args.window_size, args.batch_size)
    
    # Summary
    print("=" * 70)
    print("SUMMARY & RECOMMENDATIONS")
    print("=" * 70)
    if vocab_size is not None and data is not None:
        X_train = data.get("X_train.npy")
        if X_train is not None:
            max_tok = X_train.max()
            if max_tok >= vocab_size:
                print(f"❌ CRITICAL: vocab_size ({vocab_size}) is too small!")
                print(f"   Data contains tokens up to {max_tok}.")
                print(f"   Set vocab_size >= {max_tok + 1} in config or SLURM script.")
            else:
                print(f"✓ vocab_size is OK: {vocab_size} covers tokens 0..{max_tok}")
    
    print()
    print("Before retraining, ensure:")
    print("  1. vocab_size matches your data (no index out of bounds)")
    print("  2. Use per-sample loss reduction (already fixed in autoencoder.py)")
    print("  3. Anomaly rate is reasonable (~5-10%) in y_val")
    print("  4. Consider reducing latent_dim (32 → 16) for tighter bottleneck")
    print()


if __name__ == "__main__":
    main()