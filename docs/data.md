# Data

Notes on preparing LibriSpeech splits, tokenizer usage, and prefix handling.

## LibriSpeech Setup
- Run `./scripts/download_librispeech.sh data/raw/librispeech` to fetch & extract `train-clean-100`, `train-clean-360`, `train-other-500`, `dev-clean`, `dev-other`, `test-clean`, `test-other`.
- Generate JSONL manifests with `scripts/make_manifest.py`:
  ```bash
  python scripts/make_manifest.py \
    --root data/raw/librispeech/train-clean-100 \
    --output data/manifests/librispeech/train-clean-100.jsonl
  ```
  Repeat for the other train splits plus dev/test (change `--root` and `--output` paths).
- Concatenate manifests into larger splits (examples):
  ```bash
  cat data/manifests/librispeech/train-clean-100.jsonl \
      data/manifests/librispeech/train-clean-360.jsonl \
      data/manifests/librispeech/train-other-500.jsonl \
      > data/manifests/librispeech/train-960.jsonl

  cat data/manifests/librispeech/dev-clean.jsonl \
      data/manifests/librispeech/dev-other.jsonl \
      > data/manifests/librispeech/dev-all.jsonl
  ```
- Each manifest line stores `audio_path`, `text`, `language` fields. Paths should be absolute or relative to the repo root.
- Normalize transcripts (lowercase/punctuation) during manifest creation if needed; Whisper tokenizer will ingest the provided text verbatim.

## Tokenizer & Prefix
- Load `WhisperTokenizer` from Hugging Face; share vocab with the diffusion decoder.
- Add a dedicated `[MASK]` special token (configurable via `mask_token`) so masked positions are easy to distinguish from padding.
- Prefix tokens:
  - `<|startoftranscript|>`
  - `<|lang:xx|>` (e.g., `<|ko|>`, `<|en|>`)
  - `<|transcribe|>` or `<|translate|>`
- Prefix is treated as deterministic context (no noise injected).

## Padding & Masks
- Pad sequences to `L_max = 448` tokens.
- Keep separate `attention_mask` and `flow_mask` to zero-out padded areas during loss computation and self-attention.

## Future Work
- Extend manifests for multilingual corpora.
- Add timestamp supervision details once v0 decoding stabilizes.
