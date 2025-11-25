"""
File: src/data/dummy_dataset.py
Role: Provides a random LibriSpeech-shaped dataset for smoke testing without real manifests.
"""

from __future__ import annotations

import random
from typing import Dict, List

import torch
from torch.utils.data import Dataset


class DummyLibriDataset(Dataset):
    """Synthetic dataset that mimics LibriSpeech shapes."""

    def __init__(self, cfg: Dict, split: str):
        dataset_cfg = cfg.get("dataset", {})
        tokenizer_cfg = cfg.get("tokenizer", {})

        dummy_counts = dataset_cfg.get("dummy_samples", {})
        self.length = dummy_counts.get(split, 512)
        self.n_mels = cfg.get("features", {}).get("n_mels", 80)
        self.n_frames = cfg.get("features", {}).get("n_frames", 3000)
        self.max_tokens = tokenizer_cfg.get("max_text_tokens", 448)
        self.vocab_size = tokenizer_cfg.get("vocab_size", 51865)
        self.pad_id = tokenizer_cfg.get("pad_token_id", 0)
        self.eot_id = tokenizer_cfg.get("eot_token_id", self.vocab_size - 1)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, _) -> Dict[str, torch.Tensor]:
        features = torch.randn(self.n_frames, self.n_mels).transpose(0, 1)
        seq_len = random.randint(5, self.max_tokens)
        tokens = torch.full((self.max_tokens,), self.pad_id, dtype=torch.long)
        random_tokens = torch.randint(0, self.vocab_size - 1, (seq_len - 1,))
        tokens[: seq_len - 1] = random_tokens
        tokens[seq_len - 1] = self.eot_id

        mask = torch.zeros(self.max_tokens, dtype=torch.float32)
        mask[:seq_len] = 1.0

        flow_mask = mask.clone()
        return {
            "input_features": features,
            "tokens": tokens,
            "token_mask": mask,
            "flow_mask": flow_mask,
        }

    @staticmethod
    def collate(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        tokens = torch.stack([item["tokens"] for item in batch], dim=0)
        mask = torch.stack([item["token_mask"] for item in batch], dim=0)
        flow_mask = torch.stack([item["flow_mask"] for item in batch], dim=0)
        features = torch.stack([item["input_features"] for item in batch], dim=0)
        return {
            "input_features": features,
            "tokens": tokens,
            "token_mask": mask,
            "flow_mask": flow_mask,
        }
