# Inference

How to run the sampler and evaluate outputs.

## Sampler
- Initialize all non-prefix positions as `[MASK]`.
- Define a mask schedule (e.g., `[1.0, 0.9, 0.8, ..., 0.0]`) that determines how many masks remain after each iteration.
- At each step:
  - Run the discrete diffusion decoder with the current sequence and timestep.
  - Select the `to_reveal = current_mask - target_mask` most confident masked tokens (greedy for now) and unmask them.
  - Repeat until no masks remain.

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
    --output outputs/eval/test-clean.jsonl
  ```
- Metrics print to stdout; optionally review per-utterance predictions in the JSONL.

## Debugging Tips
- Log per-step predictions / mask ratios to ensure the schedule is honored.
- Compare multiple mask schedules for accuracy vs latency.
- Make sure the tokenizer prefix matches the training task (transcribe vs translate).
