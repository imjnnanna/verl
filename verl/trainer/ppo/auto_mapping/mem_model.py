"""Memory accounting for the auto-mapping solver.

`get_min_alloc(g, Q, N_gpus, model_specs)` returns, for each placement
group, the minimum (t, p) such that the group's per-GPU footprint stays
within the budget Q. The solver consumes this as an a-priori lower bound
on submesh size before enumerating shapes.

Memory model (per group, per GPU):
  persistent  = sum_models( weight_bytes/(t*p) + optimizer_bytes/(t*p) )
  transient   = max_models( activation_bytes(t) + kv_cache_bytes(t) )
  footprint   = persistent + transient

Persistent state (weights, optimizer) sums across colocated models since
they all stay resident. Transient state (activations, KV cache) takes a
max because models in a hybrid engine alternate, not run concurrently.

When `model_specs is None` we return a permissive stub (t=p=1) so legacy
callers and tests still pass.
"""

from dataclasses import dataclass
from typing import Optional


DTYPE_BYTES = {"fp16": 2, "bf16": 2, "fp32": 4, "fp8": 1}


@dataclass
class ModelSpec:
    """Architecture + workload context the memory model needs."""
    num_layers: int
    hidden: int
    num_attention_heads: int
    num_kv_heads: int       # GQA: < num_attention_heads
    head_dim: int
    vocab: int
    inter_size: int         # MLP intermediate dim (≈ 4*hidden for vanilla, swiglu varies)

    seq_len: int            # prompt+response for training; prompt for prefill
    batch_per_gpu: int      # micro-batch size per DP rank

    is_training: bool = True
    is_generation: bool = False
    response_len: int = 0
    weight_dtype: str = "bf16"
    optimizer_bytes_per_param: int = 16  # Adam mixed-precision: m+v fp32 + master fp32 + grad fp32


@dataclass
class MinAlloc:
    """Minimum (t, p) for a placement group; n = t*p is the cluster-area lower bound."""
    t: int
    p: int
    n: int

    # Subscript protocol so callers that iterate as numeric tuples still work:
    #   m[0]=t, m[1]=p, m[2]=n
    def __getitem__(self, k):
        return (self.t, self.p, self.n)[k]


# --- memory accounting helpers ---------------------------------------------

def num_params(s: ModelSpec) -> int:
    """Approximate parameter count (transformer body + tied or untied embeddings)."""
    h, inter, L, V = s.hidden, s.inter_size, s.num_layers, s.vocab
    qkv_o = 4 * h * h                  # ignores GQA's KV savings
    mlp = 3 * h * inter                # gate + up + down (SwiGLU)
    return L * (qkv_o + mlp) + 2 * V * h


def weight_bytes(s: ModelSpec, t: int, p: int) -> int:
    return num_params(s) * DTYPE_BYTES[s.weight_dtype] // (t * p)


def optimizer_bytes(s: ModelSpec, t: int, p: int) -> int:
    if not s.is_training:
        return 0
    return num_params(s) * s.optimizer_bytes_per_param // (t * p)


def activation_bytes(s: ModelSpec, t: int) -> int:
    """Megatron-LM activation memory formula (Korthikanti et al., 2022)."""
    if not s.is_training:
        return 0
    return s.num_layers * s.seq_len * s.batch_per_gpu * s.hidden * 2 * (10 + 24 // max(t, 1))


def kv_cache_bytes(s: ModelSpec, t: int) -> int:
    """KV cache for autoregressive generation (per GPU after TP)."""
    if not s.is_generation:
        return 0
    seq = s.seq_len + s.response_len
    bpe = DTYPE_BYTES[s.weight_dtype]
    return 2 * s.num_kv_heads * s.head_dim * s.num_layers * seq * s.batch_per_gpu * bpe // max(t, 1)


def per_gpu_footprint(specs: list[ModelSpec], t: int, p: int) -> int:
    """Group footprint at parallelism (t, p).

    Persistent (weights + optimizer) sums across colocated models; transient
    (activations + KV cache) takes a max since hybrid engines time-multiplex
    training and generation.
    """
    persistent = sum(weight_bytes(s, t, p) + optimizer_bytes(s, t, p) for s in specs)
    transient = max((activation_bytes(s, t) + kv_cache_bytes(s, t) for s in specs), default=0)
    return persistent + transient


# --- public interface -------------------------------------------------------

class _Alloc:
    """List-like container of MinAlloc, indexed by group_idx."""
    def __init__(self, per_group: list[MinAlloc]):
        self._per = per_group

    def __len__(self):
        return len(self._per)

    def __getitem__(self, k):
        return self._per[k]

    def __iter__(self):
        return iter(self._per)


def get_min_alloc(
    g,
    Q: int,
    N_gpus: int,
    model_specs: Optional[dict] = None,
):
    """Find smallest (t, p) per group such that per-GPU footprint ≤ Q.

    Args:
        g: placement = tuple of role-id tuples.
        Q: per-GPU memory budget in bytes.
        N_gpus: total cluster GPUs (caps t*p search).
        model_specs: dict[role_id, ModelSpec]. If None, returns permissive
            stub (t=p=1) so legacy callers still work.

    Returns _Alloc of length len(g). For group i:
      A_min[i].t / .p — minimum tensor / pipeline parallelism
      A_min[i].n      — minimum GPU count = t * p
      A_min[i][2]     — same as .n (subscript fallback for legacy callers)
    """
    if model_specs is None:
        return _Alloc([MinAlloc(1, 1, 1) for _ in g])

    out: list[MinAlloc] = []
    for group in g:
        specs = [model_specs[r] for r in group]
        t, p = 1, 1
        max_t = 8  # conservative: typical NVLink island width per host
        while per_gpu_footprint(specs, t, p) > Q:
            if t * 2 <= max_t and t * p * 2 <= N_gpus:
                t *= 2
            elif t * (p * 2) <= N_gpus:
                p *= 2
            else:
                # Even at max (t, p), the group's footprint exceeds Q. Caller
                # should reject this placement; we still emit the maxed bounds.
                break
        out.append(MinAlloc(t=t, p=p, n=t * p))
    return _Alloc(out)
