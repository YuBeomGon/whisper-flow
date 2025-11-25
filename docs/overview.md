# Overview

Prototype ASR system that keeps the Whisper encoder frozen, replaces the decoder with a flow-matching generator over token embeddings, and targets LibriSpeech train-clean-100 for bring-up. This document summarizes the scope, motivation, and high-level roadmap.

## Goals
- Validate continuous flow matching on top of Whisper decoder weights without AR distillation.
- Establish reproducible training/inference scripts with MLflow tracking.
- Document data handling, tokenizer quirks, and padding/special-token strategy.

## Current Findings
- Training loss (velocity MSE) drops steadily on Libri100 (≈60 → ≈5) yet decoding collapses even on training utterances. This indicates the model is reducing the average embedding error but not mapping to discrete tokens reliably.
- Continuous MSE on top of frozen Whisper embeddings lets multiple tokens share “midpoint” vectors. Recent text diffusion/flow LMs avoid this by either (a) re-training embedding+rounding jointly or (b) operating directly in discrete/categorical space.
- Before large architectural changes, we must add instrumentation:
  - single-sample overfit + reconstruction test (check \(X_0 = Z_1 - v_\theta\)),
  - GT-velocity ODE smoke test,
  - train vs inference forward-parity comparison.
- Plan to branch off for experiments (e.g., discrete diffusion/flow objectives or embedding+rounding losses) while keeping the current Whisper-flow pipeline as baseline.

## Non-Goals (v0)
- Matching full Whisper WER on large multilingual corpora.
- Advanced probability paths, EMA samplers, or AR teacher distillation.
- Timestamp-specific heuristics beyond the default Whisper token format.

See `docs/model.md` and `docs/training.md` for deeper technical details.
