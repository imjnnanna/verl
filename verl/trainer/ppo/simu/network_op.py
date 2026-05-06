from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar, Optional

from verl.trainer.ppo.simu.relation import NetworkPhase
from verl.trainer.ppo.simu.shard import Shard
from verl.trainer.ppo.simu.topo import HostTopo

if TYPE_CHECKING:
    # ModelMapping imports NetworkOp at runtime; defer this back-reference to
    # break the cycle.
    from verl.trainer.ppo.simu.model_mapping import ModelMapping

@dataclass(frozen=True)
class NetworkOp(ABC):
    logical_transfers: list[LogicalTransfer]
    # Per-instance contention class. Defaults to the subclass's DEFAULT_PHASE
    # (resolved in __post_init__) so the common case stays ergonomic, but any
    # call site can override — e.g. a P2P used as pipeline send/recv that
    # contends as a steady-state stream rather than a boundary one-shot.
    # kw_only so that @dataclass subclasses can still add positional fields
    # without defaults (CombinedNetworkOp.stages).
    phase: Optional[NetworkPhase] = field(default=None, kw_only=True)

    # Subclasses set this to declare the typical phase for the op type.
    # None means the caller MUST pass `phase=` explicitly.
    DEFAULT_PHASE: ClassVar[Optional[NetworkPhase]] = None

    def __post_init__(self) -> None:
        if self.phase is None and self.DEFAULT_PHASE is not None:
            object.__setattr__(self, "phase", self.DEFAULT_PHASE)

    @staticmethod
    def phase_of(op: NetworkOp) -> Optional[NetworkPhase]:
        return op.phase

    @staticmethod
    def time_simultaneous_system_flow(ops: list[NetworkOp], topo: HostTopo) -> list[float]:
        all_transfers = [transfer for op in ops for transfer in op.logical_transfers]
        topo_input = [((t.src_host_id, t.dst_host_id), t.data_GB) for t in all_transfers]
        flow_times = topo.get_simultaneous_system_flow_time(topo_input)
        transfer_times = dict(zip(all_transfers, flow_times))
        return [op.get_operator_time(transfer_times) for op in ops]

    @staticmethod
    def time_simultaneous_system_operators(ops: list[NetworkOp], topo: HostTopo) -> list[float]:
        all_transfers = [transfer for op in ops for transfer in op.logical_transfers]
        transfer_times = topo.get_simultaneous_logical_transfer_times(all_transfers)

        return [op.get_operator_time(transfer_times) for op in ops]

    @abstractmethod
    def get_operator_time(self, transfer_times: dict[LogicalTransfer, float]) -> float:
        raise NotImplementedError("Subclasses must implement get_operator_time")

@dataclass(frozen=True)
class CombinedNetworkOp(NetworkOp):
    stages: list[NetworkOp]

    def __init__(self, stages: list[NetworkOp], phase: Optional[NetworkPhase] = None):
        # Frozen dataclass forbids attribute assignment after __init__ runs;
        # bypass via object.__setattr__ to populate the inherited and own fields.
        # phase has no natural default for a composite (sub-stages may differ),
        # so callers pass one explicitly when they need phase_of() to resolve.
        object.__setattr__(self, "stages", stages)
        object.__setattr__(self, "logical_transfers", []) # Irrelevant for combined operations
        object.__setattr__(self, "phase", phase)

    def get_operator_time(self, transfer_times: dict[LogicalTransfer, float]) -> float:
        total_time = 0.0
        for stage in self.stages:
            stage_time = stage.get_operator_time(transfer_times)
            total_time += stage_time
        return total_time

class AllReduceRing(NetworkOp):
    DEFAULT_PHASE = NetworkPhase.STEADY

    @classmethod
    def generate(cls, model_mapping: ModelMapping, shards_ring: list[Shard], data_GB: float) -> AllReduceRing:
        """
        Sensitive to ring order (should have minor effect under tree structure, but could have major effect under multiple, highly differentiated paths)
        """
        logical_transfers = []
        n = len(shards_ring)
        if n > 1:
            for i in range(n):
                next_i = (i + 1) % n
                logical_transfers.append(LogicalTransfer(
                    src_host_id=model_mapping.shards_to_host_ids[shards_ring[i]],
                    dst_host_id=model_mapping.shards_to_host_ids[shards_ring[next_i]],
                    data_GB=data_GB / n
                ))
        return cls(logical_transfers=logical_transfers)

    def get_operator_time(self, transfer_times: dict[LogicalTransfer, float]) -> float:
        if not self.logical_transfers: return 0.0
        n = len(self.logical_transfers)
        max_time = max(transfer_times.get(t, 0.0) for t in self.logical_transfers)
        # Ring all-reduce is 2*(n-1) steps, each taking max bottleneck time of size data_GB/n
        return max_time * 2 * (n - 1) if n > 0 else 0.0

class AllGatherRing(NetworkOp):
    DEFAULT_PHASE = NetworkPhase.STEADY

    @classmethod
    def generate(cls, model_mapping: ModelMapping, shards_ring: list[Shard], data_GB: float) -> AllGatherRing:
        """
        Sensitive to ring order (should have minor effect under tree structure, but could have major effect under multiple, highly differentiated paths)
        """
        logical_transfers = []
        n = len(shards_ring)
        if n > 1:
            for i in range(n):
                next_i = (i + 1) % n
                logical_transfers.append(LogicalTransfer(
                    src_host_id=model_mapping.shards_to_host_ids[shards_ring[i]],
                    dst_host_id=model_mapping.shards_to_host_ids[shards_ring[next_i]],
                    data_GB=data_GB / n
                ))
        return cls(logical_transfers=logical_transfers)

    def get_operator_time(self, transfer_times: dict[LogicalTransfer, float]) -> float:
        if not self.logical_transfers: return 0.0
        n = len(self.logical_transfers)
        max_time = max(transfer_times.get(t, 0.0) for t in self.logical_transfers)
        # Ring all-gather takes n-1 steps, each moving data_GB/n over the ring.
        return max_time * (n - 1) if n > 0 else 0.0

class ReduceScatterRing(NetworkOp):
    DEFAULT_PHASE = NetworkPhase.STEADY

    @classmethod
    def generate(cls, model_mapping: ModelMapping, shards_ring: list[Shard], data_GB: float) -> ReduceScatterRing:
        """
        Sensitive to ring order (should have minor effect under tree structure, but could have major effect under multiple, highly differentiated paths)
        """
        # Same logical traffic volume pattern as all_gather in a ring
        base_op = AllGatherRing.generate(model_mapping, shards_ring, data_GB)
        return cls(logical_transfers=base_op.logical_transfers)

    def get_operator_time(self, transfer_times: dict[LogicalTransfer, float]) -> float:
        if not self.logical_transfers: return 0.0
        n = len(self.logical_transfers)
        max_time = max(transfer_times.get(t, 0.0) for t in self.logical_transfers)
        # Ring reduce-scatter takes n-1 steps.
        return max_time * (n - 1) if n > 0 else 0.0

class AllToAll(NetworkOp):
    DEFAULT_PHASE = NetworkPhase.STEADY

    @classmethod
    def generate(cls, model_mapping: ModelMapping, shards: list[Shard], data_GB: float) -> AllToAll:
        logical_transfers = []
        # In an all-to-all, every node sends data_GB/N to every other node
        for i in range(len(shards)):
            for j in range(len(shards)):
                if i != j:
                    logical_transfers.append(LogicalTransfer(
                        src_host_id=model_mapping.shards_to_host_ids[shards[i]],
                        dst_host_id=model_mapping.shards_to_host_ids[shards[j]],
                        data_GB=data_GB / len(shards)
                    ))
        return cls(logical_transfers=logical_transfers)

    def get_operator_time(self, transfer_times: dict[LogicalTransfer, float]) -> float:
        if not self.logical_transfers: return 0.0
        # In this logical model all connections happen simultaneously.
        # The time for the operator is just the maximum connection time.
        return max(transfer_times.get(t, 0.0) for t in self.logical_transfers)

class Broadcast(NetworkOp):
    DEFAULT_PHASE = NetworkPhase.BOUNDARY

    @classmethod
    def generate(cls, model_mapping: ModelMapping, src_shard: Shard, dst_shards: list[Shard], data_GB: float) -> Broadcast:
        logical_transfers = []
        src_host = model_mapping.shards_to_host_ids[src_shard]
        for dst in dst_shards:
            if src_shard != dst:
                logical_transfers.append(LogicalTransfer(
                    src_host_id=src_host,
                    dst_host_id=model_mapping.shards_to_host_ids[dst],
                    data_GB=data_GB
                ))
        return cls(logical_transfers=logical_transfers)

    def get_operator_time(self, transfer_times: dict[LogicalTransfer, float]) -> float:
        if not self.logical_transfers: return 0.0
        # A simple star broadcast time is dominated by the slowest link.
        return max(transfer_times.get(t, 0.0) for t in self.logical_transfers)

class P2P(NetworkOp):
    DEFAULT_PHASE = NetworkPhase.BOUNDARY

    @classmethod
    def generate(cls, model_mapping: ModelMapping, src_shard: Shard, dst_shard: Shard, data_GB: float) -> P2P:
        return cls(logical_transfers=[LogicalTransfer(
            src_host_id=model_mapping.shards_to_host_ids[src_shard],
            dst_host_id=model_mapping.shards_to_host_ids[dst_shard],
            data_GB=data_GB
        )])

    def get_operator_time(self, transfer_times: dict[LogicalTransfer, float]) -> float:
        if not self.logical_transfers: return 0.0
        return max(transfer_times.get(t, 0.0) for t in self.logical_transfers)

@dataclass(frozen=True)
class LogicalTransfer:
    src_host_id: int
    dst_host_id: int
    data_GB: float

    def get_connection_tuple(self) -> tuple[int, int]:
        return (self.src_host_id, self.dst_host_id)