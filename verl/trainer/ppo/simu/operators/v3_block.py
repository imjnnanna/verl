from __future__ import annotations
from typing import Callable, Optional

from verl.trainer.ppo.simu.model import ArchitectureConfig
from verl.trainer.ppo.simu.network_requirement import (
    CollectiveKind,
    NetworkRequirement,
    ParallelismConfig,
    ParallelismGroup,
)
from verl.trainer.ppo.simu.operator import Operator
from verl.trainer.ppo.simu.operators.builders import (
    OperatorWithReqs,
    PatternEntry,
    TaggedRepeat,
)
from verl.trainer.ppo.simu.operators.elementwise import RMSNorm, SwiGLUActivation
from verl.trainer.ppo.simu.operators.embed import TokenEmbedding
from verl.trainer.ppo.simu.operators.gemm import Gemm
from verl.trainer.ppo.simu.operators.mla import build_mla_decode, build_mla_prefill
from verl.trainer.ppo.simu.operators.moe import (
    CombineMarker,
    DispatchMarker,
    build_moe_ffn,
)
from verl.trainer.ppo.simu.operators.v3_config import V3Config
from verl.trainer.ppo.simu.relation import Relation
from verl.trainer.ppo.simu.workload_context import TokenCount, WorkloadContext


def _phase_to_token_count(phase) -> TokenCount:
    # Accept either a TokenCount (legacy callers from Phase 3 tests) or a
    # phase string ("prefill" / "decode" / "training").
    if isinstance(phase, TokenCount):
        return phase
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


def _ep_dispatch() -> NetworkRequirement:
    return NetworkRequirement(
        kind=CollectiveKind.ALL_TO_ALL_DISPATCH,
        relation=Relation.EXCLUSIVE,
        eta=1.0,
        group=ParallelismGroup.EP,
    )


def _ep_combine() -> NetworkRequirement:
    return NetworkRequirement(
        kind=CollectiveKind.ALL_TO_ALL_COMBINE,
        relation=Relation.EXCLUSIVE,
        eta=1.0,
        group=ParallelismGroup.EP,
    )


def _derive_t_for(phase: TokenCount) -> Callable[[WorkloadContext], float]:
    if phase is TokenCount.PREFILL:
        return lambda ctx: float(ctx.batch_size * ctx.prompt_len)
    return lambda ctx: float(ctx.batch_size)


def _dense_ffn(cfg: V3Config, phase: TokenCount) -> list[Operator]:
    return [
        Gemm(n=cfg.m_dense, k=cfg.h, token_count=phase, dtype_bytes=cfg.dtype_bytes),
        Gemm(n=cfg.m_dense, k=cfg.h, token_count=phase, dtype_bytes=cfg.dtype_bytes),
        SwiGLUActivation(
            intermediate_size=cfg.m_dense, token_count=phase, dtype_bytes=cfg.dtype_bytes
        ),
        Gemm(n=cfg.h, k=cfg.m_dense, token_count=phase, dtype_bytes=cfg.dtype_bytes),
    ]


def build_v3_dense_layer(
    cfg: V3Config,
    phase,
    parallelism: Optional[ParallelismConfig] = None,
) -> list[OperatorWithReqs]:
    """One V3 dense decoder layer.

    TODO(phase4b): TP-shard MLA heads / FFN intermediate. For now, shapes are
    full and only network requirements respond to parallelism.
    """
    parallelism = parallelism or ParallelismConfig()
    tc = _phase_to_token_count(phase)
    mla = build_mla_prefill(cfg) if tc is TokenCount.PREFILL else build_mla_decode(cfg)
    ffn = _dense_ffn(cfg, tc)

    no_req: list[NetworkRequirement] = []
    ar = [_tp_all_reduce()] if parallelism.tp > 1 else no_req

    # Layer order: pre-attn-norm, *MLA (last MLA op = attn_out), pre-ffn-norm, *FFN (last = ffn_down)
    out: list[OperatorWithReqs] = [
        (RMSNorm(hidden_size=cfg.h, token_count=tc, dtype_bytes=cfg.dtype_bytes), no_req),
    ]
    for i, op in enumerate(mla):
        is_attn_out = i == len(mla) - 1
        out.append((op, ar if is_attn_out else no_req))
    out.append((RMSNorm(hidden_size=cfg.h, token_count=tc, dtype_bytes=cfg.dtype_bytes), no_req))
    for i, op in enumerate(ffn):
        is_ffn_down = i == len(ffn) - 1
        out.append((op, ar if is_ffn_down else no_req))
    return out


def build_v3_moe_layer(
    cfg: V3Config,
    phase,
    parallelism: Optional[ParallelismConfig] = None,
) -> list[OperatorWithReqs]:
    parallelism = parallelism or ParallelismConfig()
    tc = _phase_to_token_count(phase)
    mla = build_mla_prefill(cfg) if tc is TokenCount.PREFILL else build_mla_decode(cfg)
    moe = build_moe_ffn(cfg, _derive_t_for(tc))

    no_req: list[NetworkRequirement] = []
    ar = [_tp_all_reduce()] if parallelism.tp > 1 else no_req
    dispatch = [_ep_dispatch()] if parallelism.ep > 1 else no_req
    combine = [_ep_combine()] if parallelism.ep > 1 else no_req

    out: list[OperatorWithReqs] = [
        (RMSNorm(hidden_size=cfg.h, token_count=tc, dtype_bytes=cfg.dtype_bytes), no_req),
    ]
    for i, op in enumerate(mla):
        is_attn_out = i == len(mla) - 1
        out.append((op, ar if is_attn_out else no_req))
    out.append((RMSNorm(hidden_size=cfg.h, token_count=tc, dtype_bytes=cfg.dtype_bytes), no_req))
    # MoE layout from build_moe_ffn:
    # [router, dispatch_marker, grouped_gate, grouped_up, grouped_swiglu, grouped_down,
    #  combine_marker, shared_gate, shared_up, shared_swiglu, shared_down]
    last_idx = len(moe) - 1
    for i, op in enumerate(moe):
        if isinstance(op, DispatchMarker):
            reqs = dispatch
        elif isinstance(op, CombineMarker):
            reqs = combine
        elif i == last_idx:
            # shared_down — emits TP AR (matches dense FFN behavior).
            reqs = ar
        else:
            reqs = no_req
        out.append((op, reqs))
    return out


def build_v3_model(
    cfg: V3Config,
    phase,
    parallelism: Optional[ParallelismConfig] = None,
) -> list[PatternEntry]:
    """Full V3 model graph: embed, dense layers, MoE layers, final norm, LM head."""
    parallelism = parallelism or ParallelismConfig()
    tc = _phase_to_token_count(phase)
    dense = build_v3_dense_layer(cfg, tc, parallelism)
    moe = build_v3_moe_layer(cfg, tc, parallelism)
    return [
        (TokenEmbedding(vocab=cfg.vocab, h=cfg.h, token_count=tc, dtype_bytes=cfg.dtype_bytes), []),
        TaggedRepeat(cfg.n_dense_layers, dense),
        TaggedRepeat(cfg.n_layers - cfg.n_dense_layers, moe),
        (RMSNorm(hidden_size=cfg.h, token_count=tc, dtype_bytes=cfg.dtype_bytes), []),
        (Gemm(n=cfg.vocab, k=cfg.h, token_count=tc, dtype_bytes=cfg.dtype_bytes), []),
    ]


def build_v3_pattern(
    arch: ArchitectureConfig,
    phase: str,
    parallelism: ParallelismConfig,
) -> list[PatternEntry]:
    """Adapter matching the BuildPatternFn signature."""
    if not isinstance(arch, V3Config):
        raise TypeError(f"build_v3_pattern requires V3Config, got {type(arch).__name__}")
    return build_v3_model(arch, phase, parallelism)
