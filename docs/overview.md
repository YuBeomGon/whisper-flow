# Overview

Prototype ASR system that keeps the Whisper encoder frozen, replaces the decoder with a flow-matching generator over token embeddings, and targets LibriSpeech train-clean-100 for bring-up. This document summarizes the scope, motivation, and high-level roadmap.

## Goals
- Validate continuous flow matching on top of Whisper decoder weights without AR distillation.
- Establish reproducible training/inference scripts with MLflow tracking.
- Document data handling, tokenizer quirks, and padding/special-token strategy.

## Non-Goals (v0)
- Matching full Whisper WER on large multilingual corpora.
- Advanced probability paths, EMA samplers, or AR teacher distillation.
- Timestamp-specific heuristics beyond the default Whisper token format.

See `docs/model.md` and `docs/training.md` for deeper technical details.

