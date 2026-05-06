from __future__ import annotations

from verl.trainer.ppo.simu.operator import Operator
from verl.trainer.ppo.simu.operators.attention import DecodeAttention, PrefillAttention
from verl.trainer.ppo.simu.operators.elementwise import RoPE
from verl.trainer.ppo.simu.operators.gemm import Gemm
from verl.trainer.ppo.simu.operators.v3_config import V3Config
from verl.trainer.ppo.simu.workload_context import TokenCount


def _mla_projections(cfg: V3Config, tc: TokenCount) -> list[Operator]:
    """The 4 MLA projection GEMMs + decoupled-RoPE op shared by prefill/decode."""
    return [
        Gemm(n=cfg.d_q_compress, k=cfg.h, token_count=tc, dtype_bytes=cfg.dtype_bytes),
        Gemm(
            n=cfg.n_h * (cfg.head_size + cfg.d_rope),
            k=cfg.d_q_compress,
            token_count=tc,
            dtype_bytes=cfg.dtype_bytes,
        ),
        Gemm(
            n=cfg.d_kv_compress + cfg.d_rope,
            k=cfg.h,
            token_count=tc,
            dtype_bytes=cfg.dtype_bytes,
        ),
        Gemm(
            n=cfg.n_h * 2 * cfg.head_size,
            k=cfg.d_kv_compress,
            token_count=tc,
            dtype_bytes=cfg.dtype_bytes,
        ),
        RoPE(
            num_heads=cfg.n_h,
            head_size=cfg.head_size,
            rope_dim=cfg.d_rope,
            token_count=tc,
            dtype_bytes=cfg.dtype_bytes,
        ),
    ]


def build_mla_prefill(cfg: V3Config) -> list[Operator]:
    """One MLA layer's operator sequence in prefill mode.

    Order: q_down, q_up, kv_down, kv_up, rope, attn_core, attn_out.
    Effective head_size for attn_core is head_size + d_rope (Q/K concat the
    decoupled rope dim for scoring); the value path keeps head_size, so
    attn_out's K is n_h * head_size.
    """
    tc = TokenCount.PREFILL
    ops = _mla_projections(cfg, tc)
    ops.append(
        PrefillAttention(
            num_heads=cfg.n_h,
            num_kv_heads=cfg.n_h,
            head_size=cfg.head_size + cfg.d_rope,
            flash_block_size=cfg.flash_block_size,
            dtype_bytes=cfg.dtype_bytes,
        )
    )
    ops.append(
        Gemm(
            n=cfg.h,
            k=cfg.n_h * cfg.head_size,
            token_count=tc,
            dtype_bytes=cfg.dtype_bytes,
        )
    )
    return ops


def build_mla_decode(cfg: V3Config) -> list[Operator]:
    """One MLA layer in decode mode.

    Same operator structure as prefill but decode-shaped tokens, and the key
    difference: DecodeAttention's kv_bytes_per_token is overridden to the
    MLA latent footprint (d_kv_compress + d_rope) * dtype_bytes — this is the
    dominant memory win MLA buys at decode.
    """
    tc = TokenCount.DECODE
    ops = _mla_projections(cfg, tc)
    ops.append(
        DecodeAttention(
            num_heads=cfg.n_h,
            num_kv_heads=cfg.n_h,
            head_size=cfg.head_size + cfg.d_rope,
            dtype_bytes=cfg.dtype_bytes,
            kv_bytes_per_token=(cfg.d_kv_compress + cfg.d_rope) * cfg.dtype_bytes,
        )
    )
    ops.append(
        Gemm(
            n=cfg.h,
            k=cfg.n_h * cfg.head_size,
            token_count=tc,
            dtype_bytes=cfg.dtype_bytes,
        )
    )
    return ops
