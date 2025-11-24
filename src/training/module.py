from __future__ import annotations

import math
from typing import Any, Dict

import pytorch_lightning as pl
import torch
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

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.model(batch)

    def compute_loss(self, outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        v_pred = outputs["v_pred"]
        v_target = outputs["v_target"]
        mask = outputs["flow_mask"].unsqueeze(-1)

        mse = (v_pred - v_target) ** 2
        loss = (mse * mask).sum() / mask.sum().clamp_min(1.0)
        return loss

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        outputs = self(batch)
        loss = self.compute_loss(outputs)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        outputs = self(batch)
        loss = self.compute_loss(outputs)
        self.log("val/loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)

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
