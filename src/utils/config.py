import argparse
from pathlib import Path
from typing import Any, Dict

import yaml


def load_yaml(path: str | Path) -> Dict[str, Any]:
    """Return YAML content as a dict."""
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_training_config(path: str | Path) -> Dict[str, Any]:
    """Expand training config by loading referenced data/model configs."""
    train_cfg = load_yaml(path)

    data_cfg_path = train_cfg.get("data_config")
    if not data_cfg_path:
        raise ValueError("training config must define 'data_config'")
    train_cfg["data"] = load_yaml(data_cfg_path)

    model_cfgs: Dict[str, Any] = {}
    for name, cfg_path in (train_cfg.get("model_configs") or {}).items():
        model_cfgs[name] = load_yaml(cfg_path)
    train_cfg["models"] = model_cfgs
    return train_cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Flow-Whisper training entrypoint")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to training YAML config",
    )
    return parser.parse_args()
