from __future__ import annotations

from verl.trainer.ppo.simu.model import ArchitectureConfig
from verl.trainer.ppo.simu.network_requirement import (
    CollectiveKind,
    NetworkRequirement,
    ParallelismConfig,
    ParallelismGroup,
)
from verl.trainer.ppo.simu.operator import Operator
from verl.trainer.ppo.simu.operators.attention import DecodeAttention, PrefillAttention
from verl.trainer.ppo.simu.operators.builders import (
    OperatorWithReqs,
    PatternEntry,
    TaggedRepeat,
)
from verl.trainer.ppo.simu.operators.elementwise import RMSNorm, RoPE, SwiGLUActivation
from verl.trainer.ppo.simu.operators.embed import TokenEmbedding
from verl.trainer.ppo.simu.operators.gemm import Gemm
from verl.trainer.ppo.simu.operators.llama_config import LlamaConfig
from verl.trainer.ppo.simu.relation import Relation
from verl.trainer.ppo.simu.workload_context import TokenCount


def _phase_to_token_count(phase: str) -> TokenCount:
    if phase in ("prefill", "training"):
        return TokenCount.PREFILL
    if phase == "decode":
        return TokenCount.DECODE
    raise ValueError(f"Unknown phase: {phase!r}")


def _tp_all_reduce() -> NetworkRequirement:
    return NetworkRequirement(
        kind=CollectiveKind.ALL_REDUCE,
        relation=Relation.EXCLUSIVE,
        eta=1.0,
        group=ParallelismGroup.TP,
    )


def build_llama_layer(
    arch: LlamaConfig,
    phase: str,
    parallelism: ParallelismConfig,
) -> list[OperatorWithReqs]:
    """One Llama decoder layer with TP-sharded shapes and AR requirements.

    Megatron-style TP: heads are split across tp ranks for QKV/attn-output;
    intermediate (m) is split across tp ranks for the FFN. AllReduce after
    attn_out and ffn_down sums the partial outputs.
    """
    tp = parallelism.tp
    if arch.n_q % tp != 0:
        raise ValueError(f"n_q ({arch.n_q}) must be divisible by tp ({tp})")
    if arch.n_kv % tp != 0:
        raise ValueError(f"n_kv ({arch.n_kv}) must be divisible by tp ({tp})")
    if arch.m % tp != 0:
        raise ValueError(f"m ({arch.m}) must be divisible by tp ({tp})")

    n_q_local = arch.n_q // tp
    n_kv_local = arch.n_kv // tp
    m_local = arch.m // tp

    tc = _phase_to_token_count(phase)
    q_dim = n_q_local * arch.head_size
    kv_dim = n_kv_local * arch.head_size

    if tc is TokenCount.PREFILL:
        attention: Operator = PrefillAttention(
            num_heads=n_q_local,
            num_kv_heads=n_kv_local,
            head_size=arch.head_size,
            flash_block_size=arch.flash_block_size,
            dtype_bytes=arch.dtype_bytes,
        )
    else:
        attention = DecodeAttention(
            num_heads=n_q_local,
            num_kv_heads=n_kv_local,
            head_size=arch.head_size,
            dtype_bytes=arch.dtype_bytes,
        )

    no_req: list[NetworkRequirement] = []
    ar_req: list[NetworkRequirement] = [_tp_all_reduce()] if tp > 1 else []

    return [
        (RMSNorm(arch.h, tc, arch.dtype_bytes), no_req),
        (Gemm(n=q_dim + 2 * kv_dim, k=arch.h, token_count=tc, dtype_bytes=arch.dtype_bytes), no_req),
        (
            RoPE(
                num_heads=n_q_local + n_kv_local,
                head_size=arch.head_size,
                rope_dim=arch.rope_dim,
                token_count=tc,
                dtype_bytes=arch.dtype_bytes,
            ),
            no_req,
        ),
        (attention, no_req),
        # attn_out: emits TP AllReduce when tp > 1.
        (Gemm(n=arch.h, k=q_dim, token_count=tc, dtype_bytes=arch.dtype_bytes), ar_req),
        (RMSNorm(arch.h, tc, arch.dtype_bytes), no_req),
        (Gemm(n=m_local, k=arch.h, token_count=tc, dtype_bytes=arch.dtype_bytes), no_req),
        (Gemm(n=m_local, k=arch.h, token_count=tc, dtype_bytes=arch.dtype_bytes), no_req),
        (SwiGLUActivation(intermediate_size=m_local, token_count=tc, dtype_bytes=arch.dtype_bytes), no_req),
        # ffn_down: emits TP AllReduce when tp > 1.
        (Gemm(n=arch.h, k=m_local, token_count=tc, dtype_bytes=arch.dtype_bytes), ar_req),
    ]


def build_llama_pattern(
    arch: ArchitectureConfig,
    phase: str,
    parallelism: ParallelismConfig,
) -> list[PatternEntry]:
    """Full Llama model graph: embed, n_layers transformer blocks, final norm, LM head.

    Signature matches `BuildPatternFn`. Cast to LlamaConfig at runtime.
    """
    if not isinstance(arch, LlamaConfig):
        raise TypeError(f"build_llama_pattern requires LlamaConfig, got {type(arch).__name__}")

    layer = build_llama_layer(arch, phase, parallelism)
    tc = _phase_to_token_count(phase)
    return [
        (TokenEmbedding(vocab=arch.vocab, h=arch.h, token_count=tc, dtype_bytes=arch.dtype_bytes), []),
        TaggedRepeat(arch.n_layers, layer),
        (RMSNorm(arch.h, tc, arch.dtype_bytes), []),
        # TODO(phase4b): TP-shard the LM head along vocab and emit AllGather.
        (Gemm(n=arch.vocab, k=arch.h, token_count=tc, dtype_bytes=arch.dtype_bytes), []),
    ]
