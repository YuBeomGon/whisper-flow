"""
File: src/inference/audio.py
Role: Handles waveform loading/resampling prior to feature extraction for inference.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torchaudio


def load_audio(path: str | Path, sample_rate: int) -> torch.Tensor:
    """Load mono waveform and resample to target sample rate."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")
    waveform, sr = torchaudio.load(path)
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    waveform = waveform.squeeze(0)
    if sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
    return waveform
