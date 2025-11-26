# Masked Diffusion Whisper Decoder

Discrete masked-diffusion ASR prototype that reuses the Whisper encoder as the acoustic front-end and adapts the Whisper decoder weights into a full-attention diffusion decoder that iteratively unmasks tokens, now targeting LibriSpeech train-960 for the main bring-up.

## Overview
- **Encoder** – frozen Whisper encoder checkpoint loaded via Hugging Face Transformers (optional unfreeze hooks later).
- **Decoder** – Whisper decoder blocks with the causal mask removed, augmented with time-conditioning to run a masked diffusion process over discrete tokens (Whisfusion/MDM style).
- **Training objective** – cross-entropy on `[MASK]` positions with inverse-`t` weighting and staged mask ratios (low→mid→high mask) plus optional stepwise `y_{t_hi}→y_{t_lo}` loss.
- **Logging** – MLflow tracking (default local backend) plus optional TensorBoard.
- **Current scope** – LibriSpeech train-960 for bring-up; multilingual/timestamp fidelity and distillation come later.
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
   - Download & extract LibriSpeech splits: `./scripts/download_librispeech.sh data/raw/librispeech` (train-clean-100/360, train-other-500, dev/test clean+other).
   - Create manifests via `python scripts/make_manifest.py --root data/raw/librispeech/train-clean-100 --output data/manifests/librispeech/train-clean-100.jsonl` etc., then concatenate into `train-960.jsonl`, `dev-all.jsonl`, `test-all.jsonl` (details in `docs/data.md`).
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
  - `models/decoder/` – masked diffusion decoder blocks, time embedding modules.
  - `training/` – Lightning modules, staged masking + loss logic, optimizers.
  - `inference/` – iterative `[MASK]` decoding pipelines, length/EOT handling, logging hooks.
  - `evaluation/` – WER/CER metrics, timestamp scoring.
  - `utils/` – shared helpers, config parsing.
- `tests/` – smoke/unit tests for data + model components.

## Notes & Roadmap
- Padding positions are masked (loss + attention) so objectives cover only meaningful tokens; prefixes stay deterministic.
- `[MASK]` ratio schedule (low→mid→high) keeps training aligned with the inference trajectory; optional stepwise loss (`y_{t_hi}→y_{t_lo}`) can be enabled.
- Immediate next steps:
  - [ ] Tighten discrete diffusion docs (data/model/training/inference).
  - [ ] Add top-k/temperature/PDD-style sampler support.
  - [ ] Instrument train/inference parity on staged mask ratios.
- Future enhancements (not in v0): AR-to-diffusion distillation, audio-conditioned middle distributions, EMA samplers, multilingual fine-tuning, better timestamp supervision.

## License
- Whisper checkpoints follow the original OpenAI license; ensure compliance when distributing weights.
- Code will adopt an open-source license (TBD) before release.
- **Corruption** – Forward diffusion randomly turns tokens into `[MASK]` or other admissible text tokens (excluding special/timestamp IDs) according to configurable probabilities (`mask`, `random`, `keep`).
