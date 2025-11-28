# Inference

How to run the sampler and evaluate outputs.

## Sampler
- Initialize all non-prefix positions as `[MASK]`.
- Define a mask schedule (e.g., `[1.0, 0.9, 0.8, ..., 0.0]`) that determines how many masks remain after each iteration.
- `sampler.mode` toggles decoding style:
  - `greedy` (default) uses pure argmax to keep decoding deterministic.
  - `sample` enables stochastic top-k/temperature sampling.
- Optional top-k/temperature schedule (`sampler.top_k_temperature`):
  - Enable in `configs/inference/default.yaml` under `sampler.top_k_temperature` to use a higher temperature and top-k sampling for early/high-mask steps, gradually annealing to greedy decoding as masks diminish.
- At each step:
  - Run the discrete diffusion decoder with the current sequence and timestep.
  - Select the `to_reveal = current_mask - target_mask` tokens via the configured sampling strategy (greedy or top-k/temperature) and unmask them.
  - Repeat until no masks remain.
- Optional partial re-masking (`sampler.refine`): after the main loop, re-mask a small number of low-confidence tokens (entropy/max-probility based) and run one more refinement step.

## Decoding
- After the final step, trim output at the first `<|endoftext|>`, skipping `<|mask|>`/`<|pad|>` tokens; optionally keep timestamps for diagnostics.

## CLI Usage
```bash
python -m src.cli.sample \
  --config configs/inference/default.yaml \
  --audio path/to.wav
```
- `configs/inference/default.yaml` should point to the desired checkpoint and the training config.
- Override `--language` to force a specific language token if needed.

## Batch Evaluation
- Evaluate an entire split and capture WER/CER:
```bash
python -m src.cli.evaluate \
  --config configs/inference/default.yaml \
  --split test \
  --output outputs/eval/test-clean.jsonl \
  --checkpoint checkpoints/libri100/flow-whisper-epoch=15.ckpt \
  --sampler-mode sample
```
- Metrics print to stdout; optionally review per-utterance predictions in the JSONL.
- `--checkpoint` / `--sampler-mode` let you swap checkpoints or toggle decoding style without editing the YAML config.

## Debugging Tips
- Log per-step predictions / mask ratios to ensure the schedule is honored.
- Compare multiple mask schedules for accuracy vs latency.
- Make sure the tokenizer prefix matches the training task (transcribe vs translate).
