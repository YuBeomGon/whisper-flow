from __future__ import annotations

import os

# import torchaudio
# torchaudio.set_audio_backend("sox_io")
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger
from src.data.datamodule import FlowDataModule
from src.training.module import FlowMatchingModule
from src.utils.config import load_training_config, parse_args


def build_logger(cfg: dict) -> MLFlowLogger | None:
    mlflow_cfg = (cfg or {}).get("mlflow")
    if not mlflow_cfg:
        return None
    tracking_uri = mlflow_cfg.get("tracking_uri", "./mlruns")
    os.environ.setdefault("MLFLOW_TRACKING_URI", tracking_uri)
    return MLFlowLogger(
        experiment_name=mlflow_cfg.get("experiment", "flow-whisper"),
        tracking_uri=tracking_uri,
    )


def build_checkpoint_callback(cfg: dict) -> ModelCheckpoint | None:
    if not cfg:
        return None
    return ModelCheckpoint(
        dirpath=cfg.get("dirpath", "checkpoints"),
        filename="flow-whisper-{epoch:02d}",
        save_top_k=cfg.get("save_top_k", 1),
        monitor=cfg.get("monitor", "val/loss"),
        mode=cfg.get("mode", "min"),
    )


def run_training() -> None:
    args = parse_args()
    train_cfg = load_training_config(args.config)
    seed_everything(train_cfg.get("seed", 1337), workers=True)

    data_cfg = train_cfg["data"]
    model_cfgs = train_cfg.get("models", {})
    encoder_cfg = model_cfgs.get("encoder", {})
    decoder_cfg = model_cfgs.get("decoder", {})

    datamodule = FlowDataModule(data_cfg)
    module = FlowMatchingModule(train_cfg, data_cfg, encoder_cfg, decoder_cfg)

    logger = build_logger(train_cfg.get("logging", {}))
    checkpoint_cb = build_checkpoint_callback(train_cfg.get("checkpoints"))
    callbacks = [cb for cb in [checkpoint_cb] if cb is not None]

    trainer = Trainer(
        max_epochs=train_cfg.get("max_epochs", 5),
        accelerator=train_cfg.get("accelerator", "auto"),
        devices=train_cfg.get("devices", 1),
        strategy=train_cfg.get("strategy", "auto"),
        precision=train_cfg.get("precision", 32),
        gradient_clip_val=train_cfg.get("gradient_clip_val", 0.0),
        val_check_interval=train_cfg.get("val_check_interval", 1.0),
        log_every_n_steps=train_cfg.get("logging", {}).get("log_every_n_steps", 50),
        callbacks=callbacks,
        logger=logger,
    )

    trainer.fit(module, datamodule=datamodule)
