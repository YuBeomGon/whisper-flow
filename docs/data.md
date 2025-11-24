# Data

Notes on preparing LibriSpeech 100h splits, tokenizer usage, and prefix handling.

## LibriSpeech Setup
- Run `./scripts/download_librispeech.sh data/raw/librispeech` to fetch & extract `train-clean-100`, `dev-clean`, `test-clean`.
- Generate JSONL manifests with `scripts/make_manifest.py`:
  ```bash
  python scripts/make_manifest.py \
    --root data/raw/librispeech/train-clean-100 \
    --output data/manifests/librispeech/train-clean-100.jsonl
  ```
  Repeat for dev/test splits (change `--root` and `--output` paths).
- Each manifest line stores `audio_path`, `text`, `language` fields. Paths should be absolute or relative to the repo root.
- Normalize transcripts (lowercase/punctuation) during manifest creation if needed; Whisper tokenizer will ingest the provided text verbatim.

## Tokenizer & Prefix
- Load `WhisperTokenizer` from Hugging Face; share vocab with the flow decoder.
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
