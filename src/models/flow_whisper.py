"""
File: src/models/flow_whisper.py
Role: Wraps the Whisper encoder with a flow-matching decoder/velocity head for training/inference.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
from transformers import WhisperModel

from src.data.tokenizer_helper import WhisperTokenizerHelper

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
        tokenizer_cfg = data_cfg.get("tokenizer", {})
        self.tokenizer_helper = WhisperTokenizerHelper(tokenizer_cfg)
        self.mask_token_id = self.tokenizer_helper.mask_id

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

        vocab_size = len(self.tokenizer_helper.tokenizer)
        self.whisper.resize_token_embeddings(vocab_size)

        self.encoder = self.whisper.encoder
        self.token_embedding = self.whisper.decoder.embed_tokens

        self.flow_decoder = FlowWhisperDecoder(self.whisper.config)
        self.flow_decoder.load_state_dict(self.whisper.decoder.state_dict())

        time_dim = decoder_cfg.get("time_embedding", {}).get(
            "dim", self.whisper.config.d_model // 2
        )
        self.time_embedding = SinusoidalTimeEmbedding(self.whisper.config.d_model, time_dim)

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        encoder_dtype = next(self.encoder.parameters()).dtype
        features = features.to(encoder_dtype)
        encoder_outputs = self.encoder(features)
        return encoder_outputs.last_hidden_state

    def decode(
        self,
        decoder_tokens: torch.Tensor,
        token_mask: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        t_values: torch.Tensor,
    ) -> torch.Tensor:
        token_emb = self.token_embedding(decoder_tokens)
        time_emb = self.time_embedding(t_values.view(-1, 1)).unsqueeze(1)
        decoder_inputs = token_emb + time_emb

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
        logits = torch.matmul(decoded, self.token_embedding.weight.T)
        return logits

    def forward(
        self,
        decoder_tokens: torch.Tensor,
        token_mask: torch.Tensor,
        features: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        t_values: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if t_values is None:
            raise ValueError("t_values must be provided")
        if encoder_hidden_states is None:
            if features is None:
                raise ValueError("either features or encoder_hidden_states must be provided")
            encoder_hidden_states = self.encode(features)
        logits = self.decode(decoder_tokens, token_mask, encoder_hidden_states, t_values)
        return {
            "logits": logits,
            "encoder_hidden_states": encoder_hidden_states,
        }
