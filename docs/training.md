# Training

Guidelines for running the Lightning training loop with MLflow logging.

## Framework
- PyTorch Lightning `LightningModule` wraps flow loss, optimizer, LR schedule.
- Mixed precision + DDP handled by Lightning Trainer flags.

## Loss
- Masked-token cross-entropy over corrupted positions (`[MASK]` or random replacements) with optional inverse-`t` weighting.
- Optional stepwise loss (`y_{t_hi}→y_{t_lo}`) focusing on positions newly unmasked between two mask ratios.
- Stage the mask ratios from low→mid→high so the model regularly trains on inference-like states.

## Optimizer / Schedules
- Start with AdamW (β₁=0.9, β₂=0.98, weight decay 0.01).
- Learning-rate warmup + cosine decay (configurable via YAML).

## Logging
- Set `MLFLOW_TRACKING_URI=./mlruns` (default local backend).
- Log metrics (loss, norm stats, GradNorm) and artifacts (checkpoints, configs).
- TensorBoard can be enabled alongside MLflow for quick scalar inspection if desired.

## Checkpoints
- Save Lightning checkpoints per epoch or best validation loss under `checkpoints/`.
