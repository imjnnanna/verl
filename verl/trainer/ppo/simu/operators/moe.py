from __future__ import annotations
from dataclasses import dataclass
from typing import Callable

from verl.trainer.ppo.simu.hardware import HardwareSpec
from verl.trainer.ppo.simu.operator import Operator
from verl.trainer.ppo.simu.operators.elementwise import SwiGLUActivation
from verl.trainer.ppo.simu.operators.gemm import Gemm
from verl.trainer.ppo.simu.operators.v3_config import V3Config
from verl.trainer.ppo.simu.workload_context import WorkloadContext


@dataclass(frozen=True)
class DispatchMarker(Operator):
    """Zero-cost attachment point for the dispatch all-to-all.

    ModelMapping recognizes this class and attaches the AllToAll network op
    to it. The operator itself contributes no compute or memory to the
    timing model.
    """

    def compute_flops(self, ctx: WorkloadContext) -> float:
        return 0.0

    def memory_bytes(self, ctx: WorkloadContext) -> float:
        return 0.0

    def is_compute_bound(self, ctx: WorkloadContext, hw: HardwareSpec) -> bool:
        return False


@dataclass(frozen=True)
class CombineMarker(Operator):
    """Zero-cost attachment point for the combine all-to-all (post-experts)."""

    def compute_flops(self, ctx: WorkloadContext) -> float:
        return 0.0

    def memory_bytes(self, ctx: WorkloadContext) -> float:
        return 0.0

    def is_compute_bound(self, ctx: WorkloadContext, hw: HardwareSpec) -> bool:
        return False


def build_moe_ffn(
    cfg: V3Config,
    derive_t: Callable[[WorkloadContext], float],
) -> list[Operator]:
    """One MoE FFN: router + dispatch + grouped routed experts + combine + shared expert.

    The grouped routed-expert ops carry n_replicas=n_routed_experts so
    parameter accounting reflects all expert weights, but compute/memory are
    sized to a single (bottleneck) expert's per-call work — the experts run
    in parallel on different ranks.
    """
    n_routed = cfg.n_routed_experts
    top_k = cfg.top_k
    imb = cfg.moe_imbalance_factor
    dtype = cfg.dtype_bytes

    # Per-expert M for the routed grouped GEMMs: average tokens-per-expert
    # scaled by imbalance to model the slowest-expert bottleneck.
    expert_M: Callable[[WorkloadContext], float] = (
        lambda ctx: derive_t(ctx) * top_k / n_routed * imb
    )

    return [
        # Router scores all tokens against every routed expert.
        Gemm(n=n_routed, k=cfg.h, dtype_bytes=dtype, derive_M=derive_t),
        DispatchMarker(),
        # Grouped routed experts (one Gemm per slice of the FFN).
        Gemm(
            n=cfg.m_expert, k=cfg.h, dtype_bytes=dtype,
            n_replicas=n_routed, replicas_active=top_k, derive_M=expert_M,
        ),
        Gemm(
            n=cfg.m_expert, k=cfg.h, dtype_bytes=dtype,
            n_replicas=n_routed, replicas_active=top_k, derive_M=expert_M,
        ),
        SwiGLUActivation(intermediate_size=cfg.m_expert, dtype_bytes=dtype, derive_t=expert_M),
        Gemm(
            n=cfg.h, k=cfg.m_expert, dtype_bytes=dtype,
            n_replicas=n_routed, replicas_active=top_k, derive_M=expert_M,
        ),
        CombineMarker(),
        # Shared expert (always active for every token), n_shared_experts == 1.
        Gemm(n=cfg.m_expert, k=cfg.h, dtype_bytes=dtype, derive_M=derive_t),
        Gemm(n=cfg.m_expert, k=cfg.h, dtype_bytes=dtype, derive_M=derive_t),
        SwiGLUActivation(intermediate_size=cfg.m_expert, dtype_bytes=dtype, derive_t=derive_t),
        Gemm(n=cfg.h, k=cfg.m_expert, dtype_bytes=dtype, derive_M=derive_t),
    ]
