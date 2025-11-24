# Model

Describes the encoder/decoder architecture and flow-matching formulation.

## Encoder
- Load Whisper checkpoint (e.g., `openai/whisper-small`) via `transformers`.
- Freeze parameters for v0; optional fine-tune hooks to be added later.
- Output hidden states `H ∈ ℝ^{T_enc × d}` used as condition for the decoder cross-attention.

## Decoder
- Start from Whisper decoder blocks:
  - Remove causal mask → allow full self-attention over the entire token sequence.
  - Inject sinusoidal or learned time-embedding `τ(t)` to every token position.
  - Output velocity tensor `v_θ(X_t, t, H)`.
- Keep token embedding matrix tied for final projection during inference.

## Flow Matching
- Latent path: `X_t = (1 - t) X_0 + t Z_1`, `t ~ Uniform(0, 1)`, `Z_1 ~ 𝓝(0, I)`.
- Target velocity: `v*(X_t) = Z_1 - X_0`.
- Loss: mean squared error over non-padding positions `||v_θ - v*||²`.
- Inference: integrate ODE from `t=1 → 0` (Euler/Heun) and decode tokens via `softmax(X_0 Eᵀ)`.

## Special Tokens
- Prefix tokens stay fixed (no noise).
- Text/timestamp/EOT tokens participate in the flow objective.

See `docs/training.md` for optimizer, logging, and Lightning integration details.

