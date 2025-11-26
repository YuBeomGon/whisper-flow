# Roadmap

Planned milestones beyond the v0 prototype.

## Near-Term
- ✅ Masked diffusion training loop (Lightning + MLflow, staged mask ratios).
- ✅ Iterative `[MASK]` sampler with configurable schedules.
- ☐ Document discrete diffusion data/model/training details (keep this file in sync).
- ✅ Basic evaluation script for WER on `dev`/`test` splits.

## Debugging & Next Actions
- ☐ Train vs inference parity instrumentation (dump mask ratios, logits, stepwise stats).
- ☐ Top-k/temperature/PDD sampler options for better diversity + stability.
- ☐ Explore discrete/rounding-aware objectives (e.g., DiffusionBERT/D3PM-style corruption or joint embedding+rounding losses). Work on a dedicated branch so the current baseline remains intact.
- ☐ Document findings (failure modes, planned fixes) and link code/branch once experiments land.

## Mid-Term
- Add AR-to-flow distillation hooks.
- Explore audio-conditioned flow paths and EMA samplers.
- Implement multilingual data loaders.

## Long-Term
- Benchmark on larger corpora (Libri960, multilingual weakly-labeled data).
- Integrate timestamp-aware objectives.
- Release pretrained checkpoints + detailed docs.
