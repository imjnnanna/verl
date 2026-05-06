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
    workload_type: Workload
    batch_size: int
    microbatch_size: int
    prompt_len: int          # input sequence length per request
    response_len: int        # number of tokens to generate (0 for non-generation workloads)
    num_microbatches: int    # derived but stored explicitly: batch_size // microbatch_size

    def __post_init__(self):
        if self.microbatch_size <= 0:
            raise ValueError(f"microbatch_size must be positive, got {self.microbatch_size}")
        if self.batch_size % self.microbatch_size != 0:
            raise ValueError(
                f"microbatch_size ({self.microbatch_size}) must divide batch_size "
                f"({self.batch_size}) evenly"
            )
        expected = self.batch_size // self.microbatch_size
        if self.num_microbatches != expected:
            raise ValueError(
                f"num_microbatches ({self.num_microbatches}) does not match "
                f"batch_size // microbatch_size ({expected})"
            )
