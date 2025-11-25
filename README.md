# Flow Matching Whisper Decoder

Flow-matching based non-autoregressive ASR prototype that reuses the Whisper encoder as the acoustic front-end and adapts the Whisper decoder weights into a full-attention flow decoder that transports Gaussian noise to token embeddings, targeting LibriSpeech train-clean-100 for the initial bring-up.

## Overview
- **Encoder** – frozen Whisper encoder checkpoint loaded via Hugging Face Transformers (optional unfreeze hooks later).
- **Decoder** – Whisper decoder blocks with the causal mask removed, augmented with time-conditioning to predict flow velocities in continuous token-embedding space.
- **Training objective** – linear path flow-matching MSE between latent velocity and target `(Z₁ - X₀)`; no distillation or advanced probability paths in v0.
- **Logging** – MLflow tracking (default local backend) plus optional TensorBoard.
- **Current scope** – LibriSpeech 100h for bring-up, multilingual/timestamp fidelity and distillation come later.
- **Training stack** – PyTorch Lightning for the training loop, Hugging Face Transformers for model/tokenizer utilities.

## Getting Started
1. **Environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -U pip
   pip install -r requirements.txt  # PyTorch, torchaudio, pytorch-lightning, transformers, mlflow, etc.
   ```
2. **Data**
   - Download & extract splits: `./scripts/download_librispeech.sh data/raw/librispeech`.
   - Create manifests via `python scripts/make_manifest.py --root data/raw/librispeech/train-clean-100 --output data/manifests/librispeech/train-clean-100.jsonl` (repeat for other splits; details in `docs/data.md`).
   - Download a Whisper checkpoint (e.g., `openai/whisper-small`) into the Hugging Face cache or custom path.
3. **Configuration & Training**
   ```bash
   export MLFLOW_TRACKING_URI=./mlruns
   python -m src.cli.train --config configs/train/libri100.yaml
   ```
4. **Inference / Sampling**
   ```bash
   python -m src.cli.sample \
     --config configs/inference/default.yaml \
     --audio path/to/audio.wav
   ```
5. **Batch Evaluation (WER/CER)**
   ```bash
   python -m src.cli.evaluate \
     --config configs/inference/default.yaml \
     --split test \
     --output outputs/eval/test-clean.jsonl
   ```

## Development
- Install dev tools from `requirements.txt` (includes Black/Ruff).
- Formatting: `./scripts/format.sh src scripts`
- Lint check: `./scripts/lint.sh src scripts`
- Config parsing/logging relies on `pyproject.toml` (Black/Ruff settings) and MLflow env vars.
- Optional git hook: `pre-commit install` (runs Ruff/Black via `.pre-commit-config.yaml`).

## Repository Layout
- `configs/` – YAML configs for data/model/train/inference.
- `docs/` – deeper write-ups (overview, model, data handling, roadmap, references).
- `experiments/` – MLflow run exports, qualitative notes.
- `notebooks/` – exploratory analysis / plotting.
- `scripts/` – CLI entry points (`train.py`, `sample.py`, etc.).
- `src/`
  - `data/` – manifests, tokenizer wrappers, prefix builders.
  - `models/encoder/` – Whisper encoder loading/freeze utilities.
  - `models/decoder/` – flow decoder blocks, time embedding modules.
  - `flow/` – path definitions, ODE solvers, sampling utilities.
  - `training/` – Lightning modules, losses, optimizers.
  - `inference/` – decoding pipelines, length/EOT handling, logging hooks.
  - `evaluation/` – WER/CER metrics, timestamp scoring.
  - `utils/` – shared helpers, config parsing.
- `tests/` – smoke/unit tests for data + model components.

## Notes & Roadmap
- Padding positions are masked (loss + attention) so flow targets cover only meaningful tokens; EOT trimming during inference keeps decoding simple.
- Special tokens (language/task/start) are treated as fixed prefixes with no injected noise, whereas text/timestamp/EOT tokens participate in flow training.
- Immediate next steps:
  - [ ] Implement linear-path flow training loop with MLflow logging.
  - [ ] Bring up inference sampler (Euler + configurable steps).
  - [ ] Document data/model details under `docs/`.
- Future enhancements (not in v0): AR-to-flow distillation, audio-conditioned flow paths, EMA samplers, multilingual fine-tuning, better timestamp supervision.

## License
- Whisper checkpoints follow the original OpenAI license; ensure compliance when distributing weights.
- Code will adopt an open-source license (TBD) before release.
