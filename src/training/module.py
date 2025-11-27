"""
File: src/training/module.py
Role: LightningModule that wraps FlowMatchingModel, computes flow loss, and configures optimizers/schedulers.
"""

from __future__ import annotations

import math
from typing import Any, Dict

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from src.data.tokenizer_helper import WhisperTokenizerHelper
from src.models.flow_whisper import FlowMatchingModel


class FlowMatchingModule(pl.LightningModule):
    """Lightning wrapper around FlowMatchingModel."""

    def __init__(
        self,
        train_cfg: Dict[str, Any],
        data_cfg: Dict[str, Any],
        encoder_cfg: Dict[str, Any],
        decoder_cfg: Dict[str, Any],
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["train_cfg", "data_cfg", "encoder_cfg", "decoder_cfg"])
        self.train_cfg = train_cfg
        self.data_cfg = data_cfg
        self.encoder_cfg = encoder_cfg
        self.decoder_cfg = decoder_cfg
        self.model = FlowMatchingModel(data_cfg, encoder_cfg, decoder_cfg)
        tokenizer_helper = WhisperTokenizerHelper(data_cfg.get("tokenizer", {}))
        if tokenizer_helper.mask_id is None:
            raise ValueError("Tokenizer must define a mask token for discrete training")
        self.mask_token_id = tokenizer_helper.mask_id
        self.allowed_random_ids = tokenizer_helper.allowed_random_ids
        self.loss_cfg = train_cfg.get("loss", {})
        self.masking_cfg = train_cfg.get("masking", {})
        schedule_cfg = self.masking_cfg.get("schedule", [])
        self.masking_schedule = sorted(
            schedule_cfg,
            key=lambda item: item.get("start_epoch", 0),
        )
        corruption_cfg = self.masking_cfg.get("corruption", {})
        self.corruption_probs = {
            "mask": float(corruption_cfg.get("mask", 1.0)),
            "random": float(corruption_cfg.get("random", 0.0)),
            "keep": float(corruption_cfg.get("keep", 0.0)),
        }
        total_prob = sum(self.corruption_probs.values())
        if total_prob <= 0:
            raise ValueError("corruption probabilities must sum to > 0")
        self.corruption_probs = {k: v / total_prob for k, v in self.corruption_probs.items()}
        self.stepwise_cfg = train_cfg.get("stepwise", {})
        self.stepwise_enabled = bool(self.stepwise_cfg.get("enabled", False))

    def _compute_ce_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask_positions: torch.Tensor,
        sample_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        vocab = logits.size(-1)
        ce = F.cross_entropy(
            logits.view(-1, vocab),
            targets.view(-1),
            reduction="none",
        ).view_as(mask_positions)
        weight_mask = mask_positions
        if sample_weights is not None:
            weight_mask = weight_mask * sample_weights.unsqueeze(-1)
        denom = weight_mask.sum().clamp_min(1.0)
        loss = (ce * weight_mask).sum() / denom
        return loss

    def _apply_mask_ratio(
        self,
        tokens: torch.Tensor,
        candidate_mask: torch.Tensor,
        ratios: torch.Tensor,
        min_masks: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        masked = tokens.clone()
        mask_positions = torch.zeros_like(tokens, dtype=torch.float32)
        actual_ratios = torch.zeros(tokens.size(0), device=tokens.device)
        probs = self.corruption_probs
        mask_prob = probs.get("mask", 1.0)
        rand_prob = probs.get("random", 0.0)
        for idx in range(tokens.size(0)):
            candidates = torch.nonzero(candidate_mask[idx], as_tuple=False).squeeze(-1)
            total = candidates.numel()
            if total == 0:
                continue
            ratio = float(torch.clamp(ratios[idx], min=0.0, max=1.0).item())
            num_to_mask = max(min_masks, int(math.ceil(ratio * total)))
            num_to_mask = min(num_to_mask, total)
            if num_to_mask == 0:
                continue
            perm = torch.randperm(total, device=tokens.device)
            selected = candidates[perm[:num_to_mask]]
            u = torch.rand(num_to_mask, device=tokens.device)
            mask_idx = selected[u < mask_prob]
            random_idx = selected[(u >= mask_prob) & (u < mask_prob + rand_prob)]
            masked[idx, mask_idx] = self.mask_token_id
            if random_idx.numel() > 0:
                random_ids = torch.randint(0, len(self.allowed_random_ids), (random_idx.numel(),), device=tokens.device)
                random_tokens = torch.tensor(self.allowed_random_ids, device=tokens.device)[random_ids]
                masked[idx, random_idx] = random_tokens
            changed_idx = torch.cat([mask_idx, random_idx])
            mask_positions[idx, changed_idx] = 1.0
            actual_ratios[idx] = num_to_mask / total
        return masked, mask_positions, actual_ratios

    def _current_mask_ratio_range(self) -> tuple[float, float]:
        if self.masking_schedule:
            current_epoch = getattr(self, "current_epoch", 0)
            selected = self.masking_schedule[0]
            for entry in self.masking_schedule:
                if current_epoch >= int(entry.get("start_epoch", 0)):
                    selected = entry
                else:
                    break
            if "ratio_range" in selected:
                ratio = selected["ratio_range"]
                return float(ratio[0]), float(ratio[1])
        base_range = self.masking_cfg.get("ratio_range", [0.05, 1.0])
        return float(base_range[0]), float(base_range[1])

    def on_train_epoch_start(self) -> None:
        min_ratio, max_ratio = self._current_mask_ratio_range()
        device = self.device if hasattr(self, "device") else torch.device("cpu")
        self.log(
            "train/mask_ratio_min",
            torch.tensor(min_ratio, device=device),
            prog_bar=True,
        )
        self.log(
            "train/mask_ratio_max",
            torch.tensor(max_ratio, device=device),
            prog_bar=True,
        )

    def _sample_mask_to_gt(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        min_ratio, max_ratio = self._current_mask_ratio_range()
        min_masks = int(self.masking_cfg.get("min_masks", 1))
        tokens = batch["tokens"]
        candidate_mask = (batch["flow_mask"] > 0.5) & (batch["token_mask"] > 0.5)
        ratios = torch.empty(tokens.size(0), device=tokens.device).uniform_(min_ratio, max_ratio)
        masked, mask_positions, actual_ratios = self._apply_mask_ratio(
            tokens, candidate_mask, ratios, min_masks
        )
        return {
            "decoder_tokens_mask": masked,
            "mask_positions_mask": mask_positions,
            "t_mask": actual_ratios.clamp_min(1e-3),
        }

    def _sample_stepwise_masks(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        tokens = batch["tokens"]
        candidate_mask = (batch["flow_mask"] > 0.5) & (batch["token_mask"] > 0.5)
        device = tokens.device
        hi_range = self.stepwise_cfg.get("hi_ratio", [0.4, 0.9])
        hi_min, hi_max = float(hi_range[0]), float(hi_range[1])
        lo_factor_range = self.stepwise_cfg.get("lo_factor", [0.5, 0.9])
        lo_min, lo_max = float(lo_factor_range[0]), float(lo_factor_range[1])
        min_masks = int(self.masking_cfg.get("min_masks", 1))

        masked_hi = tokens.clone()
        masked_lo = tokens.clone()
        mask_positions_hi = torch.zeros_like(tokens, dtype=torch.float32)
        mask_positions_focus = torch.zeros_like(tokens, dtype=torch.float32)
        actual_hi = torch.zeros(tokens.size(0), device=device)

        hi_ratios = torch.empty(tokens.size(0), device=device).uniform_(hi_min, hi_max)
        lo_factors = torch.empty(tokens.size(0), device=device).uniform_(lo_min, lo_max)

        for idx in range(tokens.size(0)):
            candidates = torch.nonzero(candidate_mask[idx], as_tuple=False).squeeze(-1)
            total = candidates.numel()
            if total == 0:
                continue
            ratio_hi = float(torch.clamp(hi_ratios[idx], min=0.0, max=1.0).item())
            num_hi = max(min_masks, int(math.ceil(ratio_hi * total)))
            num_hi = min(num_hi, total)
            ratio_lo = ratio_hi * float(lo_factors[idx].item())
            num_lo = int(math.ceil(ratio_lo * total))
            num_lo = max(0, min(num_lo, num_hi - 1))
            perm = torch.randperm(total, device=device)
            hi_select = candidates[perm[:num_hi]]
            lo_select = candidates[perm[:num_lo]] if num_lo > 0 else torch.empty(0, dtype=torch.long, device=device)

            masked_hi[idx, hi_select] = self.mask_token_id
            mask_positions_hi[idx, hi_select] = 1.0
            masked_lo[idx, lo_select] = self.mask_token_id
            mask_positions_focus[idx, hi_select] = 1.0
            if lo_select.numel() > 0:
                mask_positions_focus[idx, lo_select] = 0.0
            actual_hi[idx] = num_hi / total

        return {
            "decoder_tokens_step_hi": masked_hi,
            "decoder_tokens_step_lo": masked_lo,
            "mask_positions_step": mask_positions_focus,
            "t_step_hi": actual_hi.clamp_min(1e-3),
        }

    def _prepare_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        prepared = dict(batch)
        prepared.update(self._sample_mask_to_gt(batch))
        if self.stepwise_enabled:
            prepared.update(self._sample_stepwise_masks(batch))
        return prepared

    def _run_mask_to_gt_loss(
        self,
        batch: Dict[str, torch.Tensor],
        encoder_hidden_states: torch.Tensor,
        phase: str,
    ) -> torch.Tensor:
        logits = self.model.decode(
            batch["decoder_tokens_mask"],
            batch["token_mask"],
            encoder_hidden_states,
            batch["t_mask"],
        )
        sample_weights = None
        if self.loss_cfg.get("inverse_t_weight", False):
            sample_weights = 1.0 / batch["t_mask"].clamp_min(1e-3)
        loss = self._compute_ce_loss(
            logits,
            batch["tokens"],
            batch["mask_positions_mask"],
            sample_weights,
        )
        self.log(f"{phase}/mask_to_gt", loss, on_epoch=True, prog_bar=(phase == "train"))
        return loss

    def _run_stepwise_loss(
        self,
        batch: Dict[str, torch.Tensor],
        encoder_hidden_states: torch.Tensor,
        phase: str,
    ) -> torch.Tensor:
        logits = self.model.decode(
            batch["decoder_tokens_step_hi"],
            batch["token_mask"],
            encoder_hidden_states,
            batch["t_step_hi"],
        )
        loss = self._compute_ce_loss(
            logits,
            batch["decoder_tokens_step_lo"],
            batch["mask_positions_step"],
            None,
        )
        self.log(f"{phase}/stepwise", loss, on_epoch=True, prog_bar=False)
        weight = float(self.stepwise_cfg.get("loss_weight", 0.1))
        return weight * loss

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        batch = self._prepare_batch(batch)
        encoder_hidden = self.model.encode(batch["input_features"])
        total_loss = torch.tensor(0.0, device=encoder_hidden.device)
        if self.loss_cfg.get("mask_to_gt", True):
            total_loss = total_loss + self._run_mask_to_gt_loss(batch, encoder_hidden, "train")
        if self.stepwise_enabled:
            total_loss = total_loss + self._run_stepwise_loss(batch, encoder_hidden, "train")
        self.log("train/loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)
        return total_loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        batch = self._prepare_batch(batch)
        encoder_hidden = self.model.encode(batch["input_features"])
        total_loss = torch.tensor(0.0, device=encoder_hidden.device)
        if self.loss_cfg.get("mask_to_gt", True):
            total_loss = total_loss + self._run_mask_to_gt_loss(batch, encoder_hidden, "val")
        if self.stepwise_enabled:
            total_loss = total_loss + self._run_stepwise_loss(batch, encoder_hidden, "val")
        self.log("val/loss", total_loss, on_epoch=True, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        opt_cfg = self.train_cfg.get("optimizer", {})
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=opt_cfg.get("lr", 3e-4),
            betas=tuple(opt_cfg.get("betas", (0.9, 0.98))),
            eps=opt_cfg.get("eps", 1e-8),
            weight_decay=opt_cfg.get("weight_decay", 0.01),
        )

        sched_cfg = self.train_cfg.get("scheduler")
        if not sched_cfg:
            return optimizer

        warmup_steps = sched_cfg.get("warmup_steps", 1000)
        total_steps = sched_cfg.get("max_steps", 50000)
        min_lr = sched_cfg.get("min_lr", 1e-5)

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return (step + 1) / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            scale = 0.5 * (1 + math.cos(math.pi * progress))
            return max(min_lr / opt_cfg.get("lr", 3e-4), scale)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
