"""
File: src/models/encoder/whisper.py
Role: Minimal Whisper-like encoder stub used for experimentation/testing when full model isn't loaded.
"""

from __future__ import annotations

from typing import Dict

import torch
from torch import nn


class WhisperEncoderStub(nn.Module):
    """Lightweight stand-in for the Whisper encoder."""

    def __init__(self, cfg: Dict):
        super().__init__()
        hidden_size = cfg.get("hidden_size", 768)
        n_mels = cfg.get("n_mels", 80)

        self.proj = nn.Conv1d(n_mels, hidden_size, kernel_size=3, padding=1)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=8,
            dim_feedforward=hidden_size * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.layers = nn.TransformerEncoder(encoder_layer, num_layers=2)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Args:
            audio: (B, n_mels, frames)
        Returns:
            encoded: (B, frames, hidden_size)
        """
        x = self.proj(audio)  # (B, hidden, frames)
        x = x.transpose(1, 2)  # (B, frames, hidden)
        mask = None  # synthetic data doesn't need masking yet
        return self.layers(x, src_key_padding_mask=mask)
