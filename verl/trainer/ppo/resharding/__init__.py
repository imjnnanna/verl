"""Resharding strategies for inter-stage parameter movement.

A `ReshardingStrategy` converts a (source, dest) pair of `ModelMapping`s
(with possibly different parallelism configs) into a list of `NetworkOp`s
that move parameter bytes between layouts. The strategies live outside
`simu/` so the policy is decoupled from the latency scorer; `simu` consumes
whatever NetworkOps the strategy returns.
"""

from verl.trainer.ppo.resharding.strategy import (
    NaiveP2PStrategy,
    ParameterTensor,
    ReshardingStrategy,
    llama_total_parameter_bytes,
)
from verl.trainer.ppo.resharding.zero_redundancy import ZeroRedundancyStrategy

__all__ = [
    "NaiveP2PStrategy",
    "ParameterTensor",
    "ReshardingStrategy",
    "ZeroRedundancyStrategy",
    "llama_total_parameter_bytes",
]
