from __future__ import annotations
from typing import Callable, Union

from verl.trainer.ppo.simu.operator import Operator
from verl.trainer.ppo.simu.operators.composite import Repeat
from verl.trainer.ppo.simu.operators.elementwise import RMSNorm, SwiGLUActivation
from verl.trainer.ppo.simu.operators.embed import TokenEmbedding
from verl.trainer.ppo.simu.operators.gemm import Gemm
from verl.trainer.ppo.simu.operators.mla import build_mla_decode, build_mla_prefill
from verl.trainer.ppo.simu.operators.moe import build_moe_ffn
from verl.trainer.ppo.simu.operators.v3_config import V3Config
from verl.trainer.ppo.simu.workload_context import TokenCount, WorkloadContext


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


def build_v3_dense_layer(cfg: V3Config, phase: TokenCount) -> list[Operator]:
    mla = build_mla_prefill(cfg) if phase is TokenCount.PREFILL else build_mla_decode(cfg)
    return [
        RMSNorm(hidden_size=cfg.h, token_count=phase, dtype_bytes=cfg.dtype_bytes),
        *mla,
        RMSNorm(hidden_size=cfg.h, token_count=phase, dtype_bytes=cfg.dtype_bytes),
        *_dense_ffn(cfg, phase),
    ]


def build_v3_moe_layer(cfg: V3Config, phase: TokenCount) -> list[Operator]:
    mla = build_mla_prefill(cfg) if phase is TokenCount.PREFILL else build_mla_decode(cfg)
    moe = build_moe_ffn(cfg, _derive_t_for(phase))
    return [
        RMSNorm(hidden_size=cfg.h, token_count=phase, dtype_bytes=cfg.dtype_bytes),
        *mla,
        RMSNorm(hidden_size=cfg.h, token_count=phase, dtype_bytes=cfg.dtype_bytes),
        *moe,
    ]


def build_v3_model(
    cfg: V3Config, phase: TokenCount
) -> list[Union[Operator, Repeat]]:
    """Full V3 model graph: embed, dense layers, MoE layers, final norm, LM head."""
    dense = build_v3_dense_layer(cfg, phase)
    moe = build_v3_moe_layer(cfg, phase)
    return [
        TokenEmbedding(vocab=cfg.vocab, h=cfg.h, token_count=phase, dtype_bytes=cfg.dtype_bytes),
        Repeat(cfg.n_dense_layers, dense),
        Repeat(cfg.n_layers - cfg.n_dense_layers, moe),
        RMSNorm(hidden_size=cfg.h, token_count=phase, dtype_bytes=cfg.dtype_bytes),
        Gemm(n=cfg.vocab, k=cfg.h, token_count=phase, dtype_bytes=cfg.dtype_bytes),
    ]
