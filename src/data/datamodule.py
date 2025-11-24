from __future__ import annotations

from typing import Any, Dict, Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from transformers import WhisperFeatureExtractor

from .dummy_dataset import DummyLibriDataset
from .libri_dataset import LibriSpeechManifestDataset
from .tokenizer_helper import WhisperTokenizerHelper


class FlowDataModule(pl.LightningDataModule):
    """Lightning DataModule that wires train/val/test loaders."""

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        tokenizer_cfg = cfg.get("tokenizer", {})
        self.tokenizer_helper = WhisperTokenizerHelper(tokenizer_cfg)
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(
            tokenizer_cfg.get("hf_id", "openai/whisper-small"),
            cache_dir=tokenizer_cfg.get("cache_dir"),
        )
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

    def _build_dataset(self, split: str) -> Dataset:
        dataset_cfg = self.cfg.get("dataset", {})
        if dataset_cfg.get("synthetic", False):
            return DummyLibriDataset(self.cfg, split)
        return LibriSpeechManifestDataset(
            self.cfg,
            split,
            tokenizer=self.tokenizer_helper,
            feature_extractor=self.feature_extractor,
        )

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit"):
            self.train_dataset = self._build_dataset("train")
            self.val_dataset = self._build_dataset("val")
        if stage in (None, "test"):
            self.test_dataset = self._build_dataset("test")

    def _loader(self, dataset: Dataset, split: str) -> DataLoader:
        dl_cfg = (self.cfg.get("dataloader") or {}).get("train" if split == "train" else "eval", {})
        return DataLoader(
            dataset,
            batch_size=dl_cfg.get("batch_size", 4),
            shuffle=dl_cfg.get("shuffle", split == "train"),
            num_workers=dl_cfg.get("num_workers", 4),
            pin_memory=True,
            collate_fn=getattr(dataset, "collate"),
        )

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("call setup() before requesting train dataloader")
        return self._loader(self.train_dataset, "train")

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            raise RuntimeError("call setup() before requesting val dataloader")
        return self._loader(self.val_dataset, "val")

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            raise RuntimeError("call setup() before requesting test dataloader")
        return self._loader(self.test_dataset, "test")
