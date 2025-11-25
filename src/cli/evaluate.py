"""
File: src/cli/evaluate.py
Role: CLI utility to run manifest-level inference and report WER/CER for a checkpoint.
"""

from __future__ import annotations

import argparse
import json

import torch
from src.cli.sample import load_model
from src.evaluation.evaluator import ManifestEvaluator
from src.inference.pipeline import FlowInferencePipeline
from src.utils.config import load_training_config, load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate checkpoint on LibriSpeech split")
    parser.add_argument(
        "--config",
        required=True,
        help="Inference config (reuses checkpoint/train_config)",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="val",
        help="Which manifest split to evaluate",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional JSONL path to store per-utterance predictions",
    )
    return parser.parse_args()


def resolve_manifest(train_cfg: dict, split: str) -> str:
    key = {"train": "train", "val": "val", "test": "test"}[split]
    manifests = train_cfg["data"]["dataset"]["manifests"]
    path = manifests.get(key)
    if not path:
        raise ValueError(f"Manifest path missing for split '{split}'")
    return path


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

    manifest_path = resolve_manifest(train_cfg, args.split)

    module = load_model(checkpoint, train_cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module = module.to(device)

    pipeline = FlowInferencePipeline(
        model=module.model,
        data_cfg=train_cfg["data"],
        sampler_cfg=inf_cfg.get("sampler", {}),
        device=device,
    )

    evaluator = ManifestEvaluator(
        pipeline=pipeline,
        manifest_path=manifest_path,
        output_path=args.output,
    )
    metrics = evaluator.run()
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
