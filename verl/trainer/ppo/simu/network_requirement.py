from __future__ import annotations
from dataclasses import dataclass
from enum import Enum

from verl.trainer.ppo.simu.relation import Relation


class CollectiveKind(Enum):
    ALL_REDUCE = "all_reduce"
    ALL_GATHER = "all_gather"
    REDUCE_SCATTER = "reduce_scatter"
    ALL_TO_ALL_DISPATCH = "all_to_all_dispatch"
    ALL_TO_ALL_COMBINE = "all_to_all_combine"
    P2P_PIPELINE_FORWARD = "p2p_pipeline_forward"
    P2P_PIPELINE_BACKWARD = "p2p_pipeline_backward"


class ParallelismGroup(Enum):
    TP = "tp"
    EP = "ep"
    PP = "pp"
    DP = "dp"


@dataclass(frozen=True)
class NetworkRequirement:
    """A declarative communication requirement an operator emits.

    Resolved at ModelMapping __post_init__ to a concrete NetworkOp over the
    shard set spanning `group`. Two requirements with identical fields share
    the same NetworkOp instance (deduped by equality).
    """

    kind: CollectiveKind
    relation: Relation
    eta: float
    group: ParallelismGroup


@dataclass(frozen=True)
class ParallelismConfig:
    tp: int = 1
    pp: int = 1
    dp: int = 1
    ep: int = 1
