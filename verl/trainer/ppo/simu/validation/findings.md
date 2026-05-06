# Simulator validation findings — 2026-05-06T21:46:22Z

Generated automatically by `run_validation.py` whenever any scenario is more than 30% off its published reference. Default tolerance is 50%; training and DeepSeek-V3 are tagged as sanity bounds (100%).

## Summary

| scenario | predicted | reference | error % | tolerance % | pass |
|---|---:|---:|---:|---:|:---:|
| llama2_7b_prefill_a100 | 57.6 ms | 50.0 ms | +15.2% | 50% | ✓ |
| llama2_7b_decode_a100 | 8.4 ms | 25.0 ms | -66.3% | 50% | ✗ |
| llama3_70b_prefill_8xa100_tp8 | 209.7 ms | 250.0 ms | -16.1% | 50% | ✓ |
| llama3_8b_training_8xa100_tp2_pp2_dp2 | 10660.9 ms | 4000.0 ms | +166.5% | 100% | ✗ |
| deepseek_v3_prefill_32xh100_ep32 | 1095.9 ms | 500.0 ms | +119.2% | 100% | ✗ |

## Per-scenario investigation

### llama2_7b_decode_a100

_Llama-2-7B decode, batch=1, single A100-80GB. Reference ~25ms/token (vLLM)._

- **Predicted:** 8.4 ms  **Reference:** 25.0 ms  **Error:** -66.3%  **Tolerance:** ±50% (exceeds tolerance)

**Dominant operators by wall-time fraction:**

- `Gemm` — 94.6%
- `PrefillAttention` — 3.0%
- `SwiGLUActivation` — 1.0%
- `RMSNorm` — 0.5%
- `RoPE` — 0.5%
- `DecodeAttention` — 0.3%

**Hypothesis:**

Pure roofline does not include kernel-launch overhead, KV-cache management, or sampling latency — these dominate at batch=1 decode (real wall time is typically 2-3× the analytical floor). Total memory traffic per decode step ≈ model_size + KV reads ≈ 13.4 GB on Llama-2-7B; at peak·η_mem ≈ 1.6 TB/s the floor is ~8 ms, matching the prediction. memory_efficiency=0.8 is also optimistic for tiny M=1 GEMMs — small_gemm_efficiency=0.4 currently triggers only when min(M,N,K) < 1024, which excludes K=4096 GEMMs even with M=1. Not a formula bug; the simulator's scope stops at the roofline.

### llama3_8b_training_8xa100_tp2_pp2_dp2

_Llama-3-8B training step, batch=64, microbatch=4, prompt=2048, 8xA100-80GB TP=2 PP=2 DP=2. Reference ~3-5s (Megatron-class). Sanity bound._

- **Predicted:** 10660.9 ms  **Reference:** 4000.0 ms  **Error:** +166.5%  **Tolerance:** ±100% (exceeds tolerance)

**Dominant operators by wall-time fraction:**

- `BackwardOp` — 72.8%
- `Gemm` — 19.2%
- `PrefillAttention` — 4.1%
- `AdamOptimizerOp` — 2.9%
- `SwiGLUActivation` — 0.5%
- `RMSNorm` — 0.3%

**Hypothesis:**

Above reference. Dominant cause: even-by-op-count PP partition splits a forward-then-backward pattern so stage 0 = all forward ops and stage 1 = all backward (3× heavier with recompute) + optimizer. max_stage is then the backward stage, and the (num_microbatches + pp - 1) pipeline multiplier amplifies the imbalance. Cost-balanced partition (the deferred TODO from the ModelMapping.__post_init__ comment) — interleaving forward and backward across stages — should roughly halve max_stage. Backward 3× with recompute is also conservative; FlashAttention backward + selective checkpointing reach ~2-2.5×.

### deepseek_v3_prefill_32xh100_ep32

_DeepSeek-V3 prefill, batch=1, prompt=4096, 32xH100-80GB EP=32 TP=1 PP=1. Reference ~500ms (looser bound)._

- **Predicted:** 1095.9 ms  **Reference:** 500.0 ms  **Error:** +119.2%  **Tolerance:** ±100% (exceeds tolerance)

**Dominant operators by wall-time fraction:**

- `PrefillAttention` — 80.3%
- `Gemm` — 18.8%
- `RMSNorm` — 0.5%
- `RoPE` — 0.3%
- `SwiGLUActivation` — 0.1%
- `TokenEmbedding` — 0.0%

**Hypothesis:**

Above reference. PrefillAttention dominates ~80% of predicted wall time. memory_bytes = 3·n_h·head_size·t²/B·dtype gives ~14 ms per attention layer for V3 (n_h=128, head_size=192, t=4096, B=64), or ~880 ms across 61 layers alone. Two refinements would close the gap: (a) memory_efficiency=0.8 is conservative for H100 HBM3 — measured 0.85-0.9 — bumping it to 0.9 trims ~10%; (b) FlashAttention-3 has lower effective memory traffic than the 3·l²/B·d_h tile model assumes (recomputes Q/K/V tiles instead of materializing). Re-deriving the FA3 memory factor on H100 would shave the rest.

## Suggested measurements to fit constants

These would let us replace the literature-typical defaults (0.7 / 0.8 / 0.4) with profiled values for the specific hardware target. **Do not auto-tune.**

- Profile a representative QKV GEMM at shape (M=2048, N=12288, K=4096) on the target A100 to fit `compute_efficiency` for compute-bound prefill.
- Profile a memory-bound GEMM at shape (M=1, N=4096, K=4096) on A100 to fit `memory_efficiency` and `small_gemm_efficiency` for decode.
- Measure end-to-end Llama-2-7B decode on a single A100 with kernel-launch overhead profiled (e.g., via NSight): if the gap to roofline is ~constant per token, bake it in as a per-step overhead constant rather than touching efficiencies.
- For multi-GPU TP scenarios, instrument NVLink utilization during a single AllReduce — current simulator assumes one shared spine; real NVLink switch fabric behaves differently under contention.
- For DeepSeek-V3, measure per-token AllToAll wall time on the target topology to fit a per-EP contention factor (single-spine model is approximate).

## What this run did NOT do

- No efficiency factors were modified.
- No reference numbers were adjusted.
- No new operators or formulas were added.

Review this report and decide whether to (a) profile to fit constants, (b) extend the operator/contention model, or (c) accept current accuracy and tighten the tolerances on stable scenarios.
