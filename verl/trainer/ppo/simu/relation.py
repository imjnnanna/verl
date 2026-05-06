from __future__ import annotations
from dataclasses import dataclass
from enum import Enum

from verl.trainer.ppo.simu.network_op import NetworkOp


class Relation(Enum):
    EXCLUSIVE = "exclusive"
    CONCURRENT = "concurrent"


@dataclass(frozen=True)
class NetworkAssociation:
    network_op: NetworkOp
    relation: Relation
    eta: float = 1.0  # overlap efficiency, only used when relation is CONCURRENT.
                      # 1.0 = perfect overlap (max), 0.0 = no overlap (sum).

    def combine_times(self, kernel_time: float, network_time: float) -> float:
        if self.relation is Relation.EXCLUSIVE:
            return kernel_time + network_time
        return (
            (1.0 - self.eta) * (kernel_time + network_time)
            + self.eta * max(kernel_time, network_time)
        )
