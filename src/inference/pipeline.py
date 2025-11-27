"""
File: src/inference/pipeline.py
Role: Implements the end-to-end flow sampling path: audio preprocessing, ODE steps, and token decoding.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from src.data.tokenizer_helper import WhisperTokenizerHelper
from src.inference.audio import load_audio
from src.models.flow_whisper import FlowMatchingModel
from transformers import WhisperFeatureExtractor


class FlowInferencePipeline:
    """Utility to run single-sample inference with a trained flow decoder."""

    def __init__(
        self,
        model: FlowMatchingModel,
        data_cfg: Dict,
        sampler_cfg: Dict,
        device: torch.device,
    ):
        self.model = model.eval().to(device)
        self.device = device
        tokenizer_cfg = data_cfg.get("tokenizer", {})
        self.tokenizer_helper = WhisperTokenizerHelper(tokenizer_cfg)
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(
            tokenizer_cfg.get("hf_id", "openai/whisper-small"),
            cache_dir=tokenizer_cfg.get("cache_dir"),
        )
        self.sample_rate = data_cfg.get("dataset", {}).get("sample_rate", 16000)
        self.max_text_tokens = tokenizer_cfg.get("max_text_tokens", 448)
        self.sampler_cfg = sampler_cfg
        self.pad_id = tokenizer_cfg.get("pad_token_id", self.tokenizer_helper.pad_id)
        self.eot_id = tokenizer_cfg.get("eot_token_id", self.tokenizer_helper.eot_id)
        self.mask_token_id = self.tokenizer_helper.mask_id
        if self.mask_token_id is None:
            raise ValueError("Mask token must be defined for discrete decoding")
        schedule = list(sampler_cfg.get("mask_schedule", [1.0, 0.75, 0.5, 0.25, 0.0]))
        schedule = sorted(set(schedule), reverse=True)
        if schedule[0] != 1.0:
            schedule.insert(0, 1.0)
        if schedule[-1] != 0.0:
            schedule.append(0.0)
        self.mask_schedule = schedule
        topk_cfg = sampler_cfg.get("top_k_temperature", {})
        self.use_topk_temperature = bool(topk_cfg.get("enabled", False))
        self.high_mask_threshold = float(topk_cfg.get("high_mask_threshold", 0.7))
        self.mid_mask_threshold = float(topk_cfg.get("mid_mask_threshold", 0.3))
        self.high_mask_temperature = float(topk_cfg.get("high_mask_temperature", 1.3))
        self.mid_mask_temperature = float(topk_cfg.get("mid_mask_temperature", 1.1))
        self.low_mask_temperature = float(topk_cfg.get("low_mask_temperature", 1.0))
        self.high_mask_top_k = topk_cfg.get("high_mask_top_k", 0)
        self.mid_mask_top_k = topk_cfg.get("mid_mask_top_k", 0)
        self.low_mask_top_k = topk_cfg.get("low_mask_top_k", 0)
        refine_cfg = sampler_cfg.get("refine", {})
        self.refine_enabled = bool(refine_cfg.get("enabled", False))
        self.refine_tokens = int(refine_cfg.get("tokens", 0))
        self.refine_entropy_threshold = float(refine_cfg.get("entropy_threshold", 0.0))

    @torch.no_grad()
    def __call__(self, audio_path: str | Path, language: Optional[str] = None) -> Dict[str, str]:
        waveform = load_audio(audio_path, self.sample_rate)
        features = self.feature_extractor(
            waveform.numpy(), sampling_rate=self.sample_rate, return_tensors="pt"
        ).input_features.to(self.device)

        prefix_tokens, token_mask, flow_mask, prefix_len = (
            self.tokenizer_helper.build_sampling_tokens(language)
        )
        prefix_tokens = prefix_tokens.to(self.device).unsqueeze(0)
        token_mask = token_mask.to(self.device).unsqueeze(0)
        flow_mask = flow_mask.to(self.device).unsqueeze(0)

        encoder_hidden_states = self.model.encode(features)
        decoded_tokens = self._iterative_decode(
            prefix_tokens,
            token_mask,
            flow_mask,
            encoder_hidden_states,
        )
        predicted_ids = self._finalize_tokens(decoded_tokens, prefix_len)
        text = self.tokenizer_helper.tokenizer.decode(predicted_ids)
        return {
            "text": text,
            "tokens": " ".join(map(str, predicted_ids)),
        }

    def _iterative_decode(
        self,
        tokens: torch.Tensor,
        token_mask: torch.Tensor,
        flow_mask: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        decoded = tokens.clone()
        candidate_mask = flow_mask.bool()
        decoded[candidate_mask] = self.mask_token_id
        total_candidates = int(candidate_mask.sum().item())
        if total_candidates == 0:
            return decoded

        mode = (self.sampler_cfg.get("mode", "greedy").lower())
        sampling_enabled = mode.startswith("sample")
        for target_ratio in self.mask_schedule[1:]:
            mask_positions = (decoded == self.mask_token_id) & candidate_mask
            current_mask = int(mask_positions.sum().item())
            if current_mask == 0:
                break
            current_ratio = current_mask / total_candidates
            t_tensor = torch.tensor([current_ratio], device=self.device)
            logits = self.model.decode(decoded, token_mask, encoder_hidden_states, t_tensor)

            target_mask = int(max(0, math.ceil(target_ratio * total_candidates)))
            to_reveal = max(0, current_mask - target_mask)
            if to_reveal == 0:
                continue
            masked_indices = torch.nonzero(mask_positions[0], as_tuple=False).squeeze(-1)
            masked_logits = logits[0, masked_indices]
            num_pos = min(to_reveal, masked_indices.size(0))
            confidence = masked_logits.max(dim=-1).values
            _, pos_idx = torch.topk(confidence, k=num_pos)
            chosen_positions = masked_indices[pos_idx]
            vocab_ids = self._sample_tokens(masked_logits[pos_idx], num_pos, current_ratio, sampling_enabled)
            decoded[0, chosen_positions] = vocab_ids
        if self.refine_enabled and decoded.eq(self.mask_token_id).sum().item() == 0:
            decoded = self._refine_pass(decoded, token_mask, flow_mask, encoder_hidden_states, sampling_enabled)
        return decoded

    def _select_sampling_params(self, current_ratio: float, sampling_enabled: bool) -> Tuple[Optional[int], float]:
        if not sampling_enabled:
            return None, 1.0
        if not self.use_topk_temperature:
            return None, 1.0
        if current_ratio > self.high_mask_threshold:
            return (self.high_mask_top_k or None), self.high_mask_temperature
        if current_ratio > self.mid_mask_threshold:
            return (self.mid_mask_top_k or None), self.mid_mask_temperature
        return (self.low_mask_top_k or None), self.low_mask_temperature

    def _sample_tokens(self, masked_logits: torch.Tensor, to_reveal: int, current_ratio: float, sampling_enabled: bool) -> torch.Tensor:
        if to_reveal <= 0:
            return torch.tensor([], device=self.device, dtype=torch.long)
        top_k, temperature = self._select_sampling_params(current_ratio, sampling_enabled)
        logits = masked_logits.clone()
        if not sampling_enabled:
            vocab_ids = logits.argmax(dim=-1)
            return vocab_ids[:to_reveal]
        indices = None
        if top_k is not None and top_k > 0:
            top_k = min(top_k, logits.size(-1))
            values, idx = torch.topk(logits, k=top_k, dim=-1)
            logits = values
            indices = idx
        logits = logits / max(temperature, 1e-6)
        probs = torch.softmax(logits, dim=-1)
        sampled = torch.multinomial(probs, num_samples=1)
        if top_k is not None and top_k > 0 and indices is not None:
            vocab_ids = indices.gather(-1, sampled).squeeze(-1)
        else:
            vocab_ids = sampled.squeeze(-1)
        return vocab_ids[:to_reveal]

    def _refine_pass(
        self,
        decoded: torch.Tensor,
        token_mask: torch.Tensor,
        flow_mask: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        sampling_enabled: bool,
    ) -> torch.Tensor:
        if self.refine_tokens <= 0:
            return decoded
        candidate_mask = flow_mask.bool()[0]
        if candidate_mask.sum().item() == 0:
            return decoded
        t_tensor = torch.tensor([0.0], device=self.device)
        logits = self.model.decode(decoded, token_mask, encoder_hidden_states, t_tensor)[0]
        probs = torch.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs.clamp_min(1e-9))).sum(dim=-1)
        entropy = entropy * candidate_mask.float()
        if self.refine_entropy_threshold > 0.0:
            entropy = torch.where(entropy >= self.refine_entropy_threshold, entropy, torch.zeros_like(entropy))
        num_candidates = int(min(self.refine_tokens, (entropy > 0).sum().item()))
        if num_candidates <= 0:
            return decoded
        values, idx = torch.topk(entropy, k=num_candidates)
        idx = idx[values > 0]
        if idx.numel() == 0:
            return decoded
        decoded = decoded.clone()
        decoded[0, idx] = self.mask_token_id
        logits = self.model.decode(decoded, token_mask, encoder_hidden_states, t_tensor)[0]
        masked_logits = logits[idx]
        vocab_ids = self._sample_tokens(masked_logits, idx.numel(), 0.0, sampling_enabled)
        decoded[0, idx[: vocab_ids.size(0)]] = vocab_ids
        return decoded

    def _finalize_tokens(self, tokens: torch.Tensor, prefix_len: int) -> list[int]:
        seq = tokens[0].tolist()
        generated = seq[prefix_len:]
        trimmed: list[int] = []
        for tid in generated:
            if tid == self.eot_id:
                break
            if tid in (self.pad_id, self.mask_token_id):
                continue
            trimmed.append(tid)
        return trimmed
