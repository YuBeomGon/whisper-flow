# Training

Guidelines for running the Lightning training loop with MLflow logging.

## Framework
- PyTorch Lightning `LightningModule` wraps flow loss, optimizer, LR schedule.
- Mixed precision + DDP handled by Lightning Trainer flags.

## Loss
- Velocity MSE over non-padding positions.
- Optional regularizers (EMA, auxiliary objectives) will be added after v0 bring-up.

## Optimizer / Schedules
- Start with AdamW (β₁=0.9, β₂=0.98, weight decay 0.01).
- Learning-rate warmup + cosine decay (configurable via YAML).

## Logging
- Set `MLFLOW_TRACKING_URI=./mlruns` (default local backend).
- Log metrics (loss, norm stats, GradNorm) and artifacts (checkpoints, configs).
- TensorBoard can be enabled alongside MLflow for quick scalar inspection if desired.

## Checkpoints
- Save Lightning checkpoints per epoch or best validation loss under `checkpoints/`.

