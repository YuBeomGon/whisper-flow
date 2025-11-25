"""
File: src/models/flow_whisper.py
Role: Wraps the Whisper encoder with a flow-matching decoder/velocity head for training/inference.
"""

from __future__ import annotations

from typing import Dict

import torch
from torch import nn
from transformers import WhisperModel

from .decoder.flow_decoder import FlowWhisperDecoder, SinusoidalTimeEmbedding


class FlowMatchingModel(nn.Module):
    """Composes Whisper encoder and flow decoder."""

    def __init__(
        self,
        data_cfg: Dict,
        encoder_cfg: Dict,
        decoder_cfg: Dict,
    ):
        super().__init__()
        hf_id = encoder_cfg.get("hf_id", "openai/whisper-small")
        cache_dir = encoder_cfg.get("cache_dir")
        torch_dtype = getattr(torch, encoder_cfg.get("dtype", "float32"))
        self.whisper = WhisperModel.from_pretrained(
            hf_id,
            cache_dir=cache_dir,
            torch_dtype=torch_dtype,
        )
        if encoder_cfg.get("freeze", True):
            for param in self.whisper.encoder.parameters():
                param.requires_grad = False

        self.encoder = self.whisper.encoder
        self.token_embedding = self.whisper.decoder.embed_tokens

        self.flow_decoder = FlowWhisperDecoder(self.whisper.config)
        self.flow_decoder.load_state_dict(self.whisper.decoder.state_dict())
        self.velocity_head = nn.Linear(self.whisper.config.d_model, self.whisper.config.d_model)

        time_dim = decoder_cfg.get("time_embedding", {}).get(
            "dim", self.whisper.config.d_model // 2
        )
        self.time_embedding = SinusoidalTimeEmbedding(self.whisper.config.d_model, time_dim)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        tokens = batch["tokens"]
        token_mask = batch["token_mask"]
        flow_mask = batch["flow_mask"]
        features = batch["input_features"]

        encoder_dtype = next(self.encoder.parameters()).dtype
        features = features.to(encoder_dtype)
        encoder_outputs = self.encoder(features)
        encoder_hidden_states = encoder_outputs.last_hidden_state

        token_emb = self.token_embedding(tokens)
        noise = torch.randn_like(token_emb)
        t = torch.rand(tokens.size(0), 1, 1, device=tokens.device)
        flow_mask_exp = flow_mask.unsqueeze(-1)
        mixed = token_emb * (1 - flow_mask_exp * t) + noise * (flow_mask_exp * t)

        time_emb = self.time_embedding(t.view(tokens.size(0), 1)).unsqueeze(1)
        decoder_inputs = mixed + time_emb

        attention_mask = token_mask.to(decoder_inputs.device)
        decoder_outputs = self.flow_decoder(
            inputs_embeds=decoder_inputs,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            use_cache=False,
            output_attentions=False,
            return_dict=True,
        )
        decoded = decoder_outputs.last_hidden_state
        v_pred = self.velocity_head(decoded)
        v_target = flow_mask_exp * (noise - token_emb)

        return {
            "v_pred": v_pred,
            "v_target": v_target,
            "flow_mask": flow_mask,
            "token_mask": token_mask,
        }
