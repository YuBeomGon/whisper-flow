"""
File: src/cli/train.py
Role: Entry point that delegates to run_training() so `python -m src.cli.train` starts training.
"""

from __future__ import annotations

from src.main import run_training

if __name__ == "__main__":
    run_training()
