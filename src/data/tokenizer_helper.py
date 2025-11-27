"""
File: src/data/tokenizer_helper.py
Role: Wraps the Whisper tokenizer to build prefixes, training masks, and inference sampling tokens.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
from transformers import WhisperTokenizer


def _flatten_prompt(prompt: List[List[int]]) -> List[int]:
    flat: List[int] = []
    for part in prompt:
        flat.extend(part)
    return flat


class WhisperTokenizerHelper:
    """Utility wrapper to build decoder prefixes and tokenize transcripts."""

    def __init__(self, cfg: Dict):
        hf_id = cfg.get("hf_id", "openai/whisper-small")
        cache_dir = cfg.get("cache_dir")
        self.tokenizer = WhisperTokenizer.from_pretrained(hf_id, cache_dir=cache_dir)
        self.max_text_tokens = cfg.get("max_text_tokens", 448)
        self.pad_id = self.tokenizer.pad_token_id
        self.eot_id = self.tokenizer.eos_token_id
        mask_token = cfg.get("mask_token")
        if mask_token:
            self.tokenizer.add_special_tokens({"additional_special_tokens": [mask_token]})
            mask_id = self.tokenizer.convert_tokens_to_ids(mask_token)
        else:
            mask_id = None
        self.mask_token = mask_token
        self.mask_id = mask_id
        prefix_cfg = cfg.get("prefix", {})
        self.default_language = prefix_cfg.get("language")
        self.task = prefix_cfg.get("task", "transcribe")
        self.add_start = prefix_cfg.get("add_start_token", True)
        self.no_timestamps = prefix_cfg.get("no_timestamps", False)
        self.allowed_random_ids = self._build_allowed_random_ids()

    def build_prefix(self, language: Optional[str]) -> List[int]:
        lang = language or self.default_language
        prompt = self.tokenizer.get_decoder_prompt_ids(
            language=lang,
            task=self.task,
            no_timestamps=self.no_timestamps,
        )
        prefix = _flatten_prompt(prompt)
        if not self.add_start and prefix and prefix[0] == self.tokenizer.bos_token_id:
            prefix = prefix[1:]
        return prefix

    def encode_text(
        self, text: str, language: Optional[str]
    ) -> Tuple[torch.LongTensor, torch.FloatTensor, torch.FloatTensor]:
        prefix = self.build_prefix(language)
        text_tokens = self.tokenizer.encode(text, add_special_tokens=False)

        max_body = self.max_text_tokens - len(prefix) - 1  # leave room for EOT
        if max_body < 1:
            raise ValueError("max_text_tokens too small for prefix+EOT")
        text_tokens = text_tokens[:max_body]

        token_ids = prefix + text_tokens + [self.eot_id]
        token_mask = [1.0] * len(token_ids)
        flow_mask = [0.0] * len(prefix) + [1.0] * (len(token_ids) - len(prefix))

        pad_length = self.max_text_tokens - len(token_ids)
        if pad_length > 0:
            token_ids.extend([self.pad_id] * pad_length)
            token_mask.extend([0.0] * pad_length)
            flow_mask.extend([0.0] * pad_length)

        tokens = torch.tensor(token_ids, dtype=torch.long)
        token_mask_tensor = torch.tensor(token_mask, dtype=torch.float32)
        flow_mask_tensor = torch.tensor(flow_mask, dtype=torch.float32)
        return tokens, token_mask_tensor, flow_mask_tensor

    def build_sampling_tokens(
        self, language: Optional[str]
    ) -> Tuple[torch.LongTensor, torch.FloatTensor, torch.FloatTensor, int]:
        """Create prefix tokens/masks for inference."""
        prefix = self.build_prefix(language)
        prefix_len = len(prefix)
        if prefix_len >= self.max_text_tokens:
            raise ValueError("Prefix consumes the entire token budget.")

        token_ids = prefix + [self.pad_id] * (self.max_text_tokens - prefix_len)
        token_mask = [1.0] * prefix_len + [1.0] * (self.max_text_tokens - prefix_len)
        flow_mask = [0.0] * prefix_len + [1.0] * (self.max_text_tokens - prefix_len)

        tokens = torch.tensor(token_ids, dtype=torch.long)
        token_mask_tensor = torch.tensor(token_mask, dtype=torch.float32)
        flow_mask_tensor = torch.tensor(flow_mask, dtype=torch.float32)
        return tokens, token_mask_tensor, flow_mask_tensor, prefix_len
    def _build_allowed_random_ids(self) -> List[int]:
        special_ids = set(self.tokenizer.all_special_ids)
        timestamp_ids = set(getattr(self.tokenizer, "timestamp_ids", lambda: [])())
        forbidden = special_ids | timestamp_ids
        return [tid for tid in range(len(self.tokenizer)) if tid not in forbidden]
