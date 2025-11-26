# Model

Describes the encoder/decoder architecture and masked-diffusion formulation.

## Encoder
- Load Whisper checkpoint (e.g., `openai/whisper-small`) via `transformers`.
- Freeze parameters for v0; optional fine-tune hooks to be added later.
- Output hidden states `H ∈ ℝ^{T_enc × d}` used as condition for the decoder cross-attention.

## Decoder
- Start from Whisper decoder blocks:
  - Remove causal mask → allow full self-attention over the entire token sequence.
  - Inject a sinusoidal+MLP time-embedding `τ(t)` for each timestep.
  - Reuse Whisper cross-attention to condition on encoder features.
- Add an explicit `[MASK]` token to the tokenizer/embedding matrix; diffusion operates in discrete token space rather than continuous embeddings.

## Masked Diffusion Objective
- Sample `t ~ Uniform(min_ratio, max_ratio)` according to the staging schedule (low→mid→high mask).
- Apply `[MASK]` independently to `t · N` positions (prefix/padding excluded) to obtain `y_t`.
- Loss: cross-entropy over masked positions `CE(y_t → y_0)` with optional inverse-`t` weighting (`1 / (t + ε)`).
- Optional stepwise loss: sample `t_hi` and `t_lo < t_hi`, and train the model to map `y_{t_hi}` to `y_{t_lo}` on positions that become newly unmasked.

## Inference
- Initialize the sequence as `[PREFIX]+[MASK]`.
- Follow a predefined mask schedule (e.g., `[1.0, 0.9, ..., 0.0]`): at each step decode the current sequence, unmask the most confident subset, and repeat until no masks remain.
- Beam/PDD/top-k sampling are future additions; current implementation is greedy iterative decoding.

See `docs/training.md` for optimizer, logging, and Lightning integration details.
