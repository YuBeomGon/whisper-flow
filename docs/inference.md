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
python -m src.cli.sample \
  --config configs/inference/default.yaml \
  --audio path/to.wav
```
- `configs/inference/default.yaml` should point to the desired checkpoint and the training config.
- Override `--language` to force a specific language token if needed.

## Debugging Tips
- Inspect intermediate latents via MLflow artifacts.
- Compare multiple step counts to study accuracy vs latency.
- Ensure the tokenizer prefix matches the training task (transcribe vs translate).
