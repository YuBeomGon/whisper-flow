# Inference

How to run the sampler and evaluate outputs.

## Sampler
- Initialize latent `X_1 ~ 𝓝(0, I)`.
- Use Euler or Heun ODE solver with configurable step count (e.g., 8, 16, 32).
- Apply attention/flow masks to keep prefix tokens fixed during integration.

## Decoding
- After reaching `X_0`, compute logits via `X_0 Eᵀ`, apply softmax, and pick argmax or sample.
- Trim at first `<|endoftext|>` token; optionally keep timestamps for diagnostic purposes.

## CLI Usage
```bash
python scripts/sample.py \
  --checkpoint checkpoints/epoch=XX.ckpt \
  --audio path/to.wav \
  --ode-steps 16
```

## Debugging Tips
- Inspect intermediate latents via MLflow artifacts.
- Compare multiple step counts to study accuracy vs latency.

