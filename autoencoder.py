"""
autoencoder.py
==============
LSTM Autoencoder for log anomaly detection via reconstruction error.

Architecture follows Malhotra et al. (2016) "LSTM-based Encoder-Decoder for
Multi-sensor Anomaly Detection":
  - Encoder: reads a windowed log sequence → compresses to a fixed latent vector
  - Decoder: reconstructs the sequence from that latent vector
  - Anomaly criterion: high MSE between input and reconstruction at inference time

Why an autoencoder (not a classifier)?
  Anomalies are rare and often undefined at training time. Training only on
  normal sequences lets the model learn "what normal looks like"; it will
  reconstruct normal sequences well but fail on out-of-distribution anomalies,
  making reconstruction error a natural, label-free anomaly score.

Hyperparameter contract
-----------------------
All hyperparameters arrive via a config namespace (argparse.Namespace or
equivalent). Expected fields:

  config.vocab_size      int   – vocabulary size (from preprocessing vocab file)
  config.embed_dim       int   – token embedding dimension
  config.hidden_dim      int   – LSTM hidden state size (encoder & decoder share)
  config.latent_dim      int   – bottleneck dimension (encoder output)
  config.num_layers      int   – number of LSTM layers in encoder & decoder
  config.dropout         float – dropout probability between LSTM layers
  config.window_size     int   – sequence length (T) from the sliding window
  config.padding_idx     int   – vocab index used for <PAD> tokens (usually 0)

Input / output shapes (all modules)
------------------------------------
  x : (B, T)         – integer token indices (batch, window length)
  embeddings: (B, T, E) – after embedding lookup, E = embed_dim
  encoded:    (B, latent_dim)
  decoded:    (B, T, E)   – reconstructed embedding sequence
  loss:       scalar
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# LSTMEncoder
# ---------------------------------------------------------------------------

class LSTMEncoder(nn.Module):
    """
    Encodes a windowed log sequence into a single fixed-size latent vector.

    Processing pipeline:
      token indices → Embedding → 2-layer LSTM → linear projection → latent h

    Why take the final hidden state (not the full output sequence)?
      The hidden state h_T summarises the entire sequence. Using it as the
      bottleneck forces the network to compress sequence information into a
      fixed-size vector — the compression is what makes reconstruction hard
      for out-of-distribution (anomalous) sequences.

    Why 2 layers?
      A single LSTM layer learns low-level temporal patterns. The second layer
      composes those patterns into higher-order features (e.g., recognising
      that "error after login" is different from "error after logout").
      More than 2 layers would likely overfit on typical log dataset sizes.
    """

    def __init__(self, config):
        super().__init__()

        self.config = config

        # ------------------------------------------------------------------
        # Token embedding
        # Converts discrete token IDs to dense vectors.
        # padding_idx ensures <PAD> tokens contribute zero gradient — critical
        # because padded positions should not influence the encoding.
        # ------------------------------------------------------------------
        self.embedding = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.embed_dim,
            padding_idx=config.padding_idx,
        )
        self.embedding.weight.requires_grad = False
        # ------------------------------------------------------------------
        # Stacked LSTM
        # batch_first=True → input/output shape is (B, T, H), matching the
        # DataLoader's default batch layout from preprocessing.
        # dropout is applied between layers (not after the last layer), so
        # it only takes effect when num_layers > 1.
        # ------------------------------------------------------------------
        self.lstm = nn.LSTM(
            input_size=config.embed_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
        )

        # ------------------------------------------------------------------
        # Projection from LSTM hidden dim → latent dim
        # A separate linear layer decouples the LSTM capacity (hidden_dim)
        # from the bottleneck width (latent_dim), letting us tune each
        # independently without changing the LSTM structure.
        # ------------------------------------------------------------------
        self.hidden_to_latent = nn.Linear(config.hidden_dim, config.latent_dim)

    def forward(self, x):
        """
        Args:
            x: (B, T) integer token tensor

        Returns:
            latent: (B, latent_dim) compressed sequence representation
            embeddings: (B, T, embed_dim) — passed to decoder to define the
                        reconstruction target in embedding space
        """
        # (B, T) → (B, T, E)
        embeddings = self.embedding(x)

        # Run LSTM; we only need the final hidden state h_n.
        # h_n shape: (num_layers, B, hidden_dim)
        # c_n (cell state) is discarded — we only propagate hidden state to decoder.
        _, (h_n, _) = self.lstm(embeddings)

        # Take the top-layer hidden state (index -1 along the layers dimension).
        # Shape: (B, hidden_dim)
        h_top = h_n[-1]

        # Project to latent space. Shape: (B, latent_dim)
        latent = self.hidden_to_latent(h_top)

        return latent, embeddings


# ---------------------------------------------------------------------------
# LSTMDecoder
# ---------------------------------------------------------------------------

class LSTMDecoder(nn.Module):
    """
    Reconstructs an embedding sequence from the encoder's latent vector.

    Processing pipeline:
      latent h → repeat T times → 2-layer LSTM → linear projection → (B, T, E)

    Why repeat the latent vector instead of using it as h_0?
      Initialising h_0 and feeding zeros as input is the standard seq2seq
      approach, but it forces all temporal information through the initial
      hidden state alone, which degrades over long sequences.  Repeating the
      latent vector as the input at every time step keeps the conditioning
      signal present throughout decoding, which stabilises training and
      produces better reconstructions on longer windows (Cho et al. 2014,
      "Learning Phrase Representations").

    Why reconstruct in embedding space (not token space)?
      Predicting the original token IDs would require a cross-entropy loss
      over the full vocabulary, which is noisy and sensitive to tokenisation
      choices. MSE in the continuous embedding space is smoother and better
      suited to measuring "how different is this sequence from normal".
    """

    def __init__(self, config):
        super().__init__()

        self.config = config

        # ------------------------------------------------------------------
        # Project latent vector back to LSTM input size.
        # The decoder LSTM expects embed_dim-sized inputs (to be symmetric
        # with the encoder embedding), so we project latent_dim → embed_dim.
        # ------------------------------------------------------------------
        self.latent_to_hidden = nn.Linear(config.latent_dim, config.embed_dim)

        # ------------------------------------------------------------------
        # 2-layer LSTM (mirrors encoder depth for architectural symmetry).
        # Input at each step: the repeated latent projection (embed_dim).
        # ------------------------------------------------------------------
        self.lstm = nn.LSTM(
            input_size=config.embed_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
        )

        # ------------------------------------------------------------------
        # Output projection: LSTM hidden dim → embedding dim.
        # This reconstructs a vector in the same space as the encoder's
        # embedding layer output — enabling direct MSE comparison.
        # ------------------------------------------------------------------
        self.output_projection = nn.Linear(config.hidden_dim, config.embed_dim)

    def forward(self, latent):
        """
        Args:
            latent: (B, latent_dim) encoder bottleneck vector

        Returns:
            reconstructed: (B, T, embed_dim) sequence of reconstructed embeddings
        """
        B = latent.size(0)
        T = self.config.window_size

        # Project latent → embed_dim, then repeat across T time steps.
        # latent_proj: (B, embed_dim)
        # decoder_input: (B, T, embed_dim)
        latent_proj = self.latent_to_hidden(latent)                  # (B, E)
        decoder_input = latent_proj.unsqueeze(1).expand(B, T, -1)   # (B, T, E)

        # Run LSTM. output shape: (B, T, hidden_dim)
        output, _ = self.lstm(decoder_input)

        # Project each time step's hidden state → embedding dim.
        # reconstructed: (B, T, embed_dim)
        reconstructed = self.output_projection(output)

        return reconstructed


# ---------------------------------------------------------------------------
# LSTMAutoencoder
# ---------------------------------------------------------------------------

class LSTMAutoencoder(nn.Module):
    """
    Full LSTM Autoencoder: chains LSTMEncoder → LSTMDecoder.

    This is the top-level module used by the training loop and inference code.
    It exposes two modes:
      - forward()                  → used during training (returns loss + latent)
      - compute_reconstruction_loss() → used at inference to score sequences

    Design rationale for two separate methods
    ------------------------------------------
    During training we want gradients through the full graph; `forward` keeps
    that straightforward. During inference we don't need gradients and want a
    per-sample scalar score — `compute_reconstruction_loss` wraps the forward
    pass in torch.no_grad() and returns per-sample MSE, which the evaluation
    script can threshold to produce anomaly predictions.
    """

    def __init__(self, config):
        super().__init__()

        self.encoder = LSTMEncoder(config)
        self.decoder = LSTMDecoder(config)

    def forward(self, x):
        """
        Full forward pass for training.

        Args:
            x: (B, T) integer token tensor

        Returns:
            loss:   scalar MSE averaged over batch, timesteps, and embed dims
            latent: (B, latent_dim) — useful for logging / t-SNE visualisation
        """
        # Encode: get latent vector and original embeddings (reconstruction target)
        latent, target_embeddings = self.encoder(x)

        # Decode: reconstruct embedding sequence from latent vector
        reconstructed = self.decoder(latent)

        # MSE between reconstructed and original embeddings.
        # Using the encoder's own embeddings as targets (rather than a separate
        # frozen embedding table) means the whole network can jointly learn an
        # embedding space where normal sequences are easy to reconstruct.
        # reduction='mean' averages over B × T × E — appropriate for a stable
        # loss scale regardless of batch size or window length.
        loss = F.mse_loss(reconstructed, target_embeddings, reduction='mean')

        return loss, latent

    @torch.no_grad()
    def compute_reconstruction_loss(self, x):
        """
        Compute per-sample reconstruction error for anomaly scoring.

        Called at inference time (no gradients needed).  Returns one scalar
        per sequence; the evaluation script thresholds these scores to produce
        binary anomaly predictions (used in F1 / ROC-AUC calculation).

        Args:
            x: (B, T) integer token tensor

        Returns:
            scores: (B,) per-sample MSE (mean over T and embed_dim)
                    Higher score → sequence is more anomalous.
        """
        latent, target_embeddings = self.encoder(x)
        reconstructed = self.decoder(latent)

        # Per-sample MSE: mean over time steps (dim=1) and embed dims (dim=2),
        # leaving a scalar per sequence in the batch.
        per_sample_mse = F.mse_loss(
            reconstructed, target_embeddings, reduction='none'
        ).mean(dim=[1, 2])   # shape: (B,)

        return per_sample_mse


# ---------------------------------------------------------------------------
# Quick smoke test (run this file directly to verify shapes)
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse

    # Minimal config mimicking what argparse / a config file would produce
    cfg = argparse.Namespace(
        vocab_size=500,
        embed_dim=64,
        hidden_dim=128,
        latent_dim=32,
        num_layers=2,
        dropout=0.2,
        window_size=20,
        padding_idx=0,
    )

    model = LSTMAutoencoder(cfg)
    print(model)
    print()

    # Fake batch: 8 sequences, each 20 tokens
    x = torch.randint(1, cfg.vocab_size, (8, cfg.window_size))

    # Training forward pass
    loss, latent = model(x)
    print(f"Training loss : {loss.item():.6f}")
    print(f"Latent shape  : {latent.shape}")   # expect (8, 32)

    # Inference scoring
    scores = model.compute_reconstruction_loss(x)
    print(f"Score shape   : {scores.shape}")   # expect (8,)
    print(f"Score range   : [{scores.min():.4f}, {scores.max():.4f}]")
