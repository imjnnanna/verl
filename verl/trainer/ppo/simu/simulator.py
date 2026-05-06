from __future__ import annotations
from dataclasses import dataclass, field

from verl.trainer.ppo.simu.hardware import HardwareSpec
from verl.trainer.ppo.simu.stages import RLHFTimeline
from verl.trainer.ppo.simu.submesh_mapping import (
    SubmeshMapping,
    boundary_transfer_times,
    partition_steady_boundary,
    steady_transfer_times,
)
from verl.trainer.ppo.simu.topo import HostTopo
from verl.trainer.ppo.simu.workload_context import WorkloadContext


@dataclass
class IterationResult:
    total_time: float
    per_stage_times: list[float] = field(default_factory=list)
    per_boundary_times: list[float] = field(default_factory=list)


def simulate_stage(
    stage: list[SubmeshMapping],
    workload_ctx: WorkloadContext,
    hw: HardwareSpec,
    topo: HostTopo,
) -> float:
    """Wall time for a stage executed in parallel across submeshes.

    All submeshes share the cluster fabric, so steady (shared-bandwidth) and
    boundary (flow-time) snapshots are taken once across every NetworkOp from
    every ModelMapping in every SubmeshMapping. Each submesh then sums its
    own ModelMappings' simulate times against those snapshots, and the stage
    wall time is the slowest submesh.
    """
    if not stage:
        return 0.0

    all_mappings = [mm for sm in stage for mm in sm.model_mappings]
    steady_ops, boundary_ops = partition_steady_boundary(all_mappings)
    steady_times = steady_transfer_times(steady_ops, topo)
    boundary_times = boundary_transfer_times(boundary_ops, topo)

    submesh_times: list[float] = []
    for sm in stage:
        sm_time = sum(
            mm.simulate(workload_ctx, hw, steady_times, boundary_times).total_time
            for mm in sm.model_mappings
        )
        submesh_times.append(sm_time)
    return max(submesh_times)


def simulate_rlhf_iteration(
    timeline: RLHFTimeline,
    workload_ctx: WorkloadContext,
    hw: HardwareSpec,
    topo: HostTopo,
) -> IterationResult:
    """Sum stage wall times and boundary resharding times for one RLHF iteration."""
    total = 0.0
    per_stage_times: list[float] = []
    per_boundary_times: list[float] = []

    for i, stage in enumerate(timeline.stages):
        stage_time = simulate_stage(stage, workload_ctx, hw, topo)
        total += stage_time
        per_stage_times.append(stage_time)
        if i < len(timeline.boundaries):
            # All transitions in the boundary share one fabric simultaneously,
            # so contention is modeled with a single joint flow-time snapshot
            # covering every transition's transfers. StageBoundary.simulate
            # owns that scope.
            boundary_time = timeline.boundaries[i].simulate(topo, hw)
            total += boundary_time
            per_boundary_times.append(boundary_time)
    return IterationResult(
        total_time=total,
        per_stage_times=per_stage_times,
        per_boundary_times=per_boundary_times,
    )
