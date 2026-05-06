from __future__ import annotations
from dataclasses import dataclass

from verl.trainer.ppo.simu.operator import Operator
from verl.trainer.ppo.simu.workload_context import TokenCount, WorkloadContext


@dataclass(frozen=True)
class RMSNorm(Operator):
    hidden_size: int
    token_count: TokenCount
    dtype_bytes: int = 2

    def _t(self, ctx: WorkloadContext) -> int:
        return self.token_count.resolve(ctx)

    def compute_flops(self, ctx: WorkloadContext) -> float:
        # mean of squares (1) + rsqrt (~1) + per-element scale and weight mul (3)
        return 5.0 * self._t(ctx) * self.hidden_size

    def memory_bytes(self, ctx: WorkloadContext) -> float:
        # Read + write the activation tile; weight reuse fits in L2.
        return 2.0 * self._t(ctx) * self.hidden_size * self.dtype_bytes

    def is_compute_bound(self, ctx: WorkloadContext) -> bool:
        return False


@dataclass(frozen=True)
class SwiGLUActivation(Operator):
    intermediate_size: int
    token_count: TokenCount
    dtype_bytes: int = 2

    def _t(self, ctx: WorkloadContext) -> int:
        return self.token_count.resolve(ctx)

    def compute_flops(self, ctx: WorkloadContext) -> float:
        # silu(gate) * up: ~1 FLOP for silu approx + 1 for the multiply per element.
        return 2.0 * self._t(ctx) * self.intermediate_size

    def memory_bytes(self, ctx: WorkloadContext) -> float:
        # Read gate + read up + write output.
        return 3.0 * self._t(ctx) * self.intermediate_size * self.dtype_bytes

    def is_compute_bound(self, ctx: WorkloadContext) -> bool:
        return False


@dataclass(frozen=True)
class RoPE(Operator):
    """Rotary position embedding, applied to Q (n_q heads) and K (n_kv heads).

    `num_heads` here means the total number of heads being rotated — pass
    n_q + n_kv when used inside a transformer block.
    """

    num_heads: int
    head_size: int
    rope_dim: int
    token_count: TokenCount
    dtype_bytes: int = 2

    def _t(self, ctx: WorkloadContext) -> int:
        return self.token_count.resolve(ctx)

    def compute_flops(self, ctx: WorkloadContext) -> float:
        # 2 muls + 1 add per pair, applied to rope_dim/2 pairs => ~6 FLOPs per dim.
        return 6.0 * self._t(ctx) * self.num_heads * self.rope_dim

    def memory_bytes(self, ctx: WorkloadContext) -> float:
        # Read + write; cos/sin tables reused across tokens.
        return 2.0 * self._t(ctx) * self.num_heads * self.rope_dim * self.dtype_bytes

    def is_compute_bound(self, ctx: WorkloadContext) -> bool:
        return False
