from __future__ import annotations
from dataclasses import dataclass

from verl.trainer.ppo.simu.operator import Operator
from verl.trainer.ppo.simu.workload_context import TokenCount, WorkloadContext

# A100 ridge point in FP16 FLOPs / byte — ~312 TFLOP/s / ~2 TB/s.
# TODO: derive from HardwareSpec once profiled values are wired in.
A100_RIDGE = 156.0


@dataclass(frozen=True)
class Gemm(Operator):
    n: int
    k: int
    token_count: TokenCount
    dtype_bytes: int = 2

    def _m(self, ctx: WorkloadContext) -> int:
        return self.token_count.resolve(ctx)

    def compute_flops(self, ctx: WorkloadContext) -> float:
        return 2.0 * self._m(ctx) * self.n * self.k

    def memory_bytes(self, ctx: WorkloadContext) -> float:
        m = self._m(ctx)
        return float(self.dtype_bytes) * (m * self.k + self.k * self.n + m * self.n)

    def is_compute_bound(self, ctx: WorkloadContext) -> bool:
        return self.compute_flops(ctx) / self.memory_bytes(ctx) > A100_RIDGE

    def is_small_dim(self, ctx: WorkloadContext) -> bool:
        return min(self._m(ctx), self.n, self.k) < 1024
