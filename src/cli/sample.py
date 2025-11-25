"""
File: src/cli/sample.py
Role: Command-line interface to run single-sample flow inference using a checkpoint and audio file.
"""

from __future__ import annotations

import argparse
from typing import Any, Dict

import torch
from src.inference.pipeline import FlowInferencePipeline
from src.training.module import FlowMatchingModule
from src.utils.config import load_training_config, load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run single-sample inference")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Inference config YAML",
    )
    parser.add_argument(
        "--audio",
        type=str,
        required=True,
        help="Path to audio file",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Override language code (default from config)",
    )
    return parser.parse_args()


def load_model(checkpoint: str, train_cfg: Dict[str, Any]) -> FlowMatchingModule:
    data_cfg = train_cfg["data"]
    model_cfgs = train_cfg.get("models", {})
    encoder_cfg = model_cfgs.get("encoder", {})
    decoder_cfg = model_cfgs.get("decoder", {})
    module = FlowMatchingModule.load_from_checkpoint(
        checkpoint,
        train_cfg=train_cfg,
        data_cfg=data_cfg,
        encoder_cfg=encoder_cfg,
        decoder_cfg=decoder_cfg,
        strict=False,
    )
    module.eval()
    return module


def main() -> None:
    args = parse_args()
    inf_cfg = load_yaml(args.config)
    train_cfg_path = inf_cfg.get("train_config")
    if not train_cfg_path:
        raise ValueError("Inference config must set 'train_config'")
    train_cfg = load_training_config(train_cfg_path)

    checkpoint = inf_cfg.get("checkpoint")
    if not checkpoint:
        raise ValueError("Inference config must set 'checkpoint'")

    module = load_model(checkpoint, train_cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module = module.to(device)

    pipeline = FlowInferencePipeline(
        model=module.model,
        data_cfg=train_cfg["data"],
        sampler_cfg=inf_cfg.get("sampler", {}),
        device=device,
    )

    result = pipeline(args.audio, language=args.language)
    print(f"Transcript: {result['text']}")


if __name__ == "__main__":
    main()
