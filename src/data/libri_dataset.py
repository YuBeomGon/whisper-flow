"""
File: src/data/libri_dataset.py
Role: Loads LibriSpeech manifest JSONL rows, processes audio/text, and outputs tensors for the model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torchaudio
from torch.utils.data import Dataset
from transformers import WhisperFeatureExtractor

from .tokenizer_helper import WhisperTokenizerHelper


class LibriSpeechManifestDataset(Dataset):
    """Dataset that reads LibriSpeech-style JSONL manifests."""

    def __init__(
        self,
        cfg: Dict,
        split: str,
        tokenizer: WhisperTokenizerHelper,
        feature_extractor: WhisperFeatureExtractor,
    ):
        dataset_cfg = cfg.get("dataset", {})
        manifest_path = dataset_cfg.get("manifests", {}).get(split)
        if manifest_path is None:
            raise ValueError(f"Manifest for split '{split}' not set in config.")
        path = Path(manifest_path)
        if not path.exists():
            raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

        self.entries = self._load_manifest(path)
        self.sample_rate = dataset_cfg.get("sample_rate", 16000)
        self.text_column = dataset_cfg.get("text_column", "text")
        self.audio_column = dataset_cfg.get("audio_column", "audio_path")
        self.language_column = dataset_cfg.get("language_column", "language")
        self.max_duration = dataset_cfg.get("max_duration_s", 30.0)
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor

    @staticmethod
    def _load_manifest(path: Path) -> List[Dict]:
        rows: List[Dict] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        if not rows:
            raise ValueError(f"Manifest {path} is empty.")
        return rows

    def __len__(self) -> int:
        return len(self.entries)

    def _load_audio(self, audio_path: str) -> torch.Tensor:
        waveform, sr = torchaudio.load(audio_path)
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        waveform = waveform.squeeze(0)
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        max_samples = int(self.sample_rate * self.max_duration)
        if waveform.shape[-1] > max_samples:
            waveform = waveform[:max_samples]
        return waveform

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        entry = self.entries[index]
        audio_path = entry[self.audio_column]
        waveform = self._load_audio(audio_path)
        features = self.feature_extractor(
            waveform.numpy(), sampling_rate=self.sample_rate, return_tensors="pt"
        ).input_features.squeeze(0)

        text = entry[self.text_column]
        language = entry.get(self.language_column)
        tokens, token_mask, flow_mask, length_mask = self.tokenizer.encode_text(text, language)
        return {
            "input_features": features,
            "tokens": tokens,
            "token_mask": token_mask,
            "flow_mask": flow_mask,
            "length_mask": length_mask,
        }

    @staticmethod
    def collate(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        features = torch.stack([item["input_features"] for item in batch], dim=0)
        tokens = torch.stack([item["tokens"] for item in batch], dim=0)
        token_mask = torch.stack([item["token_mask"] for item in batch], dim=0)
        flow_mask = torch.stack([item["flow_mask"] for item in batch], dim=0)
        length_mask = torch.stack([item["length_mask"] for item in batch], dim=0)
        return {
            "input_features": features,
            "tokens": tokens,
            "token_mask": token_mask,
            "flow_mask": flow_mask,
            "length_mask": length_mask,
        }
