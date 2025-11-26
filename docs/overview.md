# Overview

Prototype ASR system that keeps the Whisper encoder frozen, replaces the decoder with a masked-diffusion generator over token embeddings, and targets LibriSpeech train-960 for bring-up. This document summarizes the scope, motivation, and high-level roadmap.

## Goals
- Validate Whisfusion-style discrete diffusion on top of Whisper decoder weights without AR distillation.
- Establish reproducible training/inference scripts with staged mask ratios and MLflow tracking.
- Document data handling (train-960), tokenizer quirks (explicit `[MASK]`), and padding/special-token strategy.

## Current Findings
- Discrete masked diffusion with staged mask ratios reduces train-inference mismatch versus the previous continuous flow baseline.
- High-mask samples (70–100%) and the optional stepwise loss are essential for inference stability; low-mask-only training collapses at sampling time.
- Remaining work focuses on:
  - logging mask ratio / stepwise trajectories,
  - sampler upgrades (top-k, temperature, PDD),
  - reducing NaN/grad explosion at later epochs (LR scaling, clipping).
- Continuous flow documentation is kept only for historical reference; the active branch uses discrete diffusion throughout.

## Non-Goals (v0)
- Matching full Whisper WER on large multilingual corpora.
- Advanced probability paths, EMA samplers, or AR teacher distillation.
- Timestamp-specific heuristics beyond the default Whisper token format.

See `docs/model.md` and `docs/training.md` for deeper technical details.
