# Roadmap

Planned milestones beyond the v0 prototype.

## Near-Term
- ✅ Flow training loop (Lightning + MLflow).
- ✅ Inference sampler with configurable ODE steps.
- ☐ Populate `docs/data.md` and `docs/model.md` with finalized specs.
- ✅ Basic evaluation script for WER on `dev-clean`.

## Debugging & Next Actions
- ☐ Single-sample overfit harness + reconstruction check (`X0` vs `Z1 - vθ`) to verify the learned velocity field.
- ☐ GT-velocity ODE smoke test (reuse sampler loop but plug in `noise - token_emb`) to validate scheduling/masking code.
- ☐ Train vs inference forward-parity instrumentation (dump `Xt`, `t`, masks) to ensure both paths use identical logic.
- ☐ Explore discrete/rounding-aware objectives (e.g., DiffusionBERT/D3PM-style categorical corruption or joint embedding+rounding losses). Work on a dedicated branch so Whisper-flow baseline remains intact.
- ☐ Document findings (failure mode, planned fixes) and link code/branch once experiments land.

## Mid-Term
- Add AR-to-flow distillation hooks.
- Explore audio-conditioned flow paths and EMA samplers.
- Implement multilingual data loaders.

## Long-Term
- Benchmark on larger corpora (Libri960, multilingual weakly-labeled data).
- Integrate timestamp-aware objectives.
- Release pretrained checkpoints + detailed docs.
