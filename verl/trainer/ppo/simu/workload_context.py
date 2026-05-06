from __future__ import annotations
from dataclasses import dataclass
from enum import Enum


class Workload(Enum):
    GENERATION = "generation"
    PREPARATION = "preparation"
    TRAINING = "training"
    DORMANT = "dormant"


class TokenCount(Enum):
    """Resolves to the number of tokens an operator processes in a single invocation.

    PREFILL: every prompt token in flight (B * prompt_len). Also fits PREPARATION
        and TRAINING forward passes that walk the full sequence in one shot.
    DECODE:  one new token per request (B), the per-step cost during generation.
    """

    PREFILL = "prefill"
    DECODE = "decode"

    def resolve(self, ctx: "WorkloadContext") -> int:
        if self is TokenCount.PREFILL:
            return ctx.batch_size * ctx.prompt_len
        if self is TokenCount.DECODE:
            return ctx.batch_size
        raise ValueError(f"Unknown TokenCount: {self}")


@dataclass(frozen=True)
class WorkloadContext:
    """Per-invocation workload shape consumed by the operator graph.

    `batch_size × prompt_len` is the token count operators see in **one
    invocation** of the graph (i.e. one microbatch when pipelined). For
    PP-pipelined training, `batch_size == microbatch_size` and
    `num_microbatches` is the number of microbatches the simulator's
    pipeline formula loops the same graph over (per DP rank).

    For non-pipelined workloads (PREPARATION, single-step decode), set
    `batch_size = microbatch_size` and `num_microbatches = 1`.
    """

    workload_type: Workload
    batch_size: int
    microbatch_size: int
    prompt_len: int          # input sequence length per request
    response_len: int        # number of tokens to generate (0 for non-generation)
    num_microbatches: int    # pipeline multiplier; >= 1

    def __post_init__(self):
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.microbatch_size <= 0:
            raise ValueError(f"microbatch_size must be positive, got {self.microbatch_size}")
        if self.microbatch_size > self.batch_size:
            raise ValueError(
                f"microbatch_size ({self.microbatch_size}) cannot exceed "
                f"batch_size ({self.batch_size})"
            )
        if self.num_microbatches <= 0:
            raise ValueError(f"num_microbatches must be positive, got {self.num_microbatches}")
