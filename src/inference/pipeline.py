from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

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
        flow_mask = flow_mask.to(self.device).unsqueeze(0).unsqueeze(-1)

        prefix_emb = self.model.token_embedding(prefix_tokens)
        encoder_outputs = self.model.encoder(features)
        encoder_hidden_states = encoder_outputs.last_hidden_state

        latent = self._sample_latent(
            encoder_hidden_states,
            prefix_emb,
            token_mask,
            flow_mask,
        )
        predicted_ids = self._decode_latent(latent, prefix_len)
        text = self.tokenizer_helper.tokenizer.decode(predicted_ids)
        return {
            "text": text,
            "tokens": " ".join(map(str, predicted_ids)),
        }

    def _sample_latent(
        self,
        encoder_hidden_states: torch.Tensor,
        prefix_emb: torch.Tensor,
        token_mask: torch.Tensor,
        flow_mask: torch.Tensor,
    ) -> torch.Tensor:
        steps = max(1, int(self.sampler_cfg.get("steps", 16)))
        device = self.device
        # hidden_size = prefix_emb.size(-1)
        latent = torch.randn_like(prefix_emb, device=device)
        latent = latent * flow_mask + prefix_emb * (1.0 - flow_mask)

        t_values = torch.linspace(1.0, 0.0, steps + 1, device=device)
        for idx in range(steps):
            t_curr = t_values[idx]
            t_next = t_values[idx + 1]
            delta = t_curr - t_next
            t_tensor = torch.full((1, 1), t_curr.item(), device=device)
            time_emb = self.model.time_embedding(t_tensor).unsqueeze(1)
            decoder_inputs = latent + time_emb

            decoder_outputs = self.model.flow_decoder(
                inputs_embeds=decoder_inputs,
                attention_mask=token_mask,
                encoder_hidden_states=encoder_hidden_states,
                use_cache=False,
                output_attentions=False,
                return_dict=True,
            )
            velocity = self.model.velocity_head(decoder_outputs.last_hidden_state)
            latent = latent - delta * velocity
            latent = latent * flow_mask + prefix_emb * (1.0 - flow_mask)
        return latent

    def _decode_latent(self, latent: torch.Tensor, prefix_len: int) -> list[int]:
        logits = torch.matmul(latent, self.model.token_embedding.weight.T)
        token_ids = torch.argmax(logits, dim=-1)[0].tolist()
        generated = token_ids[prefix_len:]
        trimmed: list[int] = []
        for tid in generated:
            if tid == self.eot_id:
                break
            if tid == self.pad_id:
                continue
            trimmed.append(tid)
        return trimmed
