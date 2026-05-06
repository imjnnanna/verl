from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Optional

from verl.trainer.ppo.simu.hardware import HardwareSpec
from verl.trainer.ppo.simu.operator import Operator
from verl.trainer.ppo.simu.workload_context import TokenCount, WorkloadContext


@dataclass(frozen=True)
class Gemm(Operator):
    n: int
    k: int
    # M (number of output rows) is derived per call. Either:
    #  - token_count: a TokenCount enum (PREFILL = B*L, DECODE = B), or
    #  - derive_M: a callable for cases the enum can't express (e.g. MoE
    #    expert grouping where M = t * top_k / n_experts * imbalance).
    # Exactly one must be supplied.
    token_count: Optional[TokenCount] = None
    dtype_bytes: int = 2
    # Weight replication (e.g. n_routed_experts in MoE). Compute and
    # memory_bytes model a single replica's per-call cost; replication only
    # affects parameter accounting.
    n_replicas: int = 1
    replicas_active: int = 1
    derive_M: Optional[Callable[[WorkloadContext], float]] = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        if (self.token_count is None) == (self.derive_M is None):
            raise ValueError(
                "Gemm requires exactly one of token_count or derive_M"
            )
        if self.replicas_active > self.n_replicas:
            raise ValueError(
                f"replicas_active ({self.replicas_active}) cannot exceed "
                f"n_replicas ({self.n_replicas})"
            )

    def _m(self, ctx: WorkloadContext) -> float:
        if self.derive_M is not None:
            return self.derive_M(ctx)
        return float(self.token_count.resolve(ctx))

    def compute_flops(self, ctx: WorkloadContext) -> float:
        return 2.0 * self._m(ctx) * self.n * self.k

    def memory_bytes(self, ctx: WorkloadContext) -> float:
        m = self._m(ctx)
        return float(self.dtype_bytes) * (m * self.k + self.k * self.n + m * self.n)

    def is_compute_bound(self, ctx: WorkloadContext, hw: HardwareSpec) -> bool:
        return self.compute_flops(ctx) / self.memory_bytes(ctx) > hw.ridge_flops_per_byte

    def is_small_dim(self, ctx: WorkloadContext) -> bool:
        return min(self._m(ctx), float(self.n), float(self.k)) < 1024

    def parameter_bytes(self) -> int:
        return self.n_replicas * self.n * self.k * self.dtype_bytes

    def activated_parameter_bytes(self) -> int:
        return self.replicas_active * self.n * self.k * self.dtype_bytes
