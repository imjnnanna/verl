from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

from verl.trainer.ppo.simu.hardware import HardwareSpec
from verl.trainer.ppo.simu.mesh import Mesh
from verl.trainer.ppo.simu.model import Model
from verl.trainer.ppo.simu.network_op import (
    AllGatherRing,
    AllReduceRing,
    AllToAll,
    LogicalTransfer,
    NetworkOp,
    P2P,
    ReduceScatterRing,
)
from verl.trainer.ppo.simu.network_requirement import (
    CollectiveKind,
    NetworkRequirement,
    ParallelismConfig,
    ParallelismGroup,
)
from verl.trainer.ppo.simu.operator import Operator
from verl.trainer.ppo.simu.operators.backward import derive_backward_pattern
from verl.trainer.ppo.simu.operators.builders import (
    OperatorWithReqs,
    PatternEntry,
    expand_tagged_pattern,
    split_pattern,
)
from verl.trainer.ppo.simu.operators.optimizer import AdamOptimizerOp
from verl.trainer.ppo.simu.relation import NetworkAssociation, NetworkPhase
from verl.trainer.ppo.simu.shard import Shard
from verl.trainer.ppo.simu.workload_context import Workload, WorkloadContext


@dataclass
class SimulationResult:
    total_time: float
    prefill_time: Optional[float] = None
    decode_time: Optional[float] = None
    training_time: Optional[float] = None
    pp_stage_times: list[float] = field(default_factory=list)
    per_op_breakdown: list[float] = field(default_factory=list)


def _even_partition(num_ops: int, num_stages: int) -> list[list[int]]:
    """Even split of operator indices into `num_stages` contiguous groups.

    TODO(phase5): replace with cost-balanced partitioning by per-op kernel_time
    at a reference workload.
    """
    if num_stages <= 0:
        raise ValueError("num_stages must be positive")
    if num_ops == 0:
        return [[] for _ in range(num_stages)]
    chunk, rem = divmod(num_ops, num_stages)
    out: list[list[int]] = []
    start = 0
    for i in range(num_stages):
        size = chunk + (1 if i < rem else 0)
        out.append(list(range(start, start + size)))
        start += size
    return out


@dataclass
class ModelMapping:
    """One model on one parallelism / workload assignment.

    `simulate` consumes pre-computed transfer-times dicts (steady + boundary)
    and returns a SimulationResult. NetworkOps are constructed once at
    __post_init__ with placeholder data_GB; the upstream stage-orchestration
    layer (Phase 4b) is responsible for populating realistic transfer times.
    """

    model: Model
    mesh: Mesh
    workload: Workload
    parallelism: ParallelismConfig
    shards_to_host_ids: dict[Shard, int]
    host_id_to_shard: dict[int, Shard]
    # Reference data_GB per collective kind for sizing the constructed
    # NetworkOps. Empty dict (default) treats every collective as 0-byte —
    # callers that want realistic contention should populate this with
    # estimated activation sizes per collective. Phase 4b's stage orchestrator
    # is the natural source.
    data_GB_estimates: dict["CollectiveKind", float] = field(default_factory=dict)

    # Computed in __post_init__:
    operator_pattern: list[Operator] = field(init=False, default_factory=list)
    per_op_network_requirements: list[list[NetworkRequirement]] = field(init=False, default_factory=list)
    network_ops: dict[NetworkRequirement, NetworkOp] = field(init=False, default_factory=dict)
    pipeline_partition: list[list[int]] = field(init=False, default_factory=list)
    # GENERATION holds a parallel decode-shaped graph; everything else is empty.
    decode_operator_pattern: list[Operator] = field(init=False, default_factory=list)
    decode_per_op_network_requirements: list[list[NetworkRequirement]] = field(init=False, default_factory=list)
    decode_pipeline_partition: list[list[int]] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        if self.workload is Workload.GENERATION:
            self._install_pattern(self._build_pattern("prefill"), is_decode=False)
            self._install_pattern(self._build_pattern("decode"), is_decode=True)
        elif self.workload is Workload.PREPARATION:
            self._install_pattern(self._build_pattern("prefill"), is_decode=False)
        elif self.workload is Workload.TRAINING:
            forward = self._build_pattern("training")
            backward = derive_backward_pattern(forward, recompute=True)
            forward_ops, _ = split_pattern(expand_tagged_pattern(forward))
            # Per-DP-rank parameter count = total local (already-tp/pp-sharded) params.
            # No ZeRO assumed; each DP rank holds and updates its own copy.
            dtype_bytes_param = 2
            total_param_bytes = sum(op.parameter_bytes() for op in forward_ops)
            num_params = total_param_bytes // dtype_bytes_param if dtype_bytes_param else 0
            opt_entry: OperatorWithReqs = (
                AdamOptimizerOp(num_parameters=num_params, dtype_bytes_param=dtype_bytes_param),
                [],
            )
            self._install_pattern(forward + backward + [opt_entry], is_decode=False)
        elif self.workload is Workload.DORMANT:
            pass  # leaves operator_pattern empty
        else:
            raise ValueError(f"Unknown workload: {self.workload!r}")

        # Materialize one NetworkOp per unique NetworkRequirement.
        for reqs_list in (self.per_op_network_requirements, self.decode_per_op_network_requirements):
            for reqs in reqs_list:
                for req in reqs:
                    if req not in self.network_ops:
                        self.network_ops[req] = self._resolve_network_op(req)

    # ----- pattern build / install --------------------------------------------------

    def _build_pattern(self, phase: str) -> list[PatternEntry]:
        return self.model.build_pattern(self.model.architecture, phase, self.parallelism)

    def _install_pattern(self, pattern: list[PatternEntry], is_decode: bool) -> None:
        expanded = expand_tagged_pattern(pattern)
        ops, reqs = split_pattern(expanded)
        partition = _even_partition(len(ops), max(self.parallelism.pp, 1))
        if is_decode:
            self.decode_operator_pattern = ops
            self.decode_per_op_network_requirements = reqs
            self.decode_pipeline_partition = partition
        else:
            self.operator_pattern = ops
            self.per_op_network_requirements = reqs
            self.pipeline_partition = partition

    # ----- network requirement resolution --------------------------------------------

    def _shards_in_group(self, group: ParallelismGroup) -> list[Shard]:
        """Return the shard set spanning the requested parallelism axis.

        TODO(phase4b): EP currently aliases the TP axis since Shard has no `ep`
        field. When EP becomes a separate axis, extend Shard and resolve here.
        """
        all_shards = list(self.shards_to_host_ids.keys())
        if not all_shards:
            return []
        ref = all_shards[0]
        if group is ParallelismGroup.TP or group is ParallelismGroup.EP:
            return sorted(
                (s for s in all_shards if s.dp == ref.dp and s.pp == ref.pp),
                key=lambda s: s.tp,
            )
        if group is ParallelismGroup.PP:
            return sorted(
                (s for s in all_shards if s.dp == ref.dp and s.tp == ref.tp),
                key=lambda s: s.pp,
            )
        if group is ParallelismGroup.DP:
            return sorted(
                (s for s in all_shards if s.pp == ref.pp and s.tp == ref.tp),
                key=lambda s: s.dp,
            )
        raise ValueError(f"Unknown ParallelismGroup: {group!r}")

    def _resolve_network_op(self, req: NetworkRequirement) -> NetworkOp:
        """Concrete NetworkOp from a declarative requirement.

        data_GB is a placeholder — Phase 4b's stage orchestrator computes
        realistic activation sizes and populates transfer-times dicts. Empty
        dicts return 0.0 via .get(t, 0.0).
        """
        shards = self._shards_in_group(req.group)
        kind = req.kind
        data_GB = self.data_GB_estimates.get(kind, 0.0)
        if kind is CollectiveKind.ALL_REDUCE:
            return AllReduceRing.generate(self, shards, data_GB=data_GB)
        if kind is CollectiveKind.ALL_GATHER:
            return AllGatherRing.generate(self, shards, data_GB=data_GB)
        if kind is CollectiveKind.REDUCE_SCATTER:
            return ReduceScatterRing.generate(self, shards, data_GB=data_GB)
        if kind in (CollectiveKind.ALL_TO_ALL_DISPATCH, CollectiveKind.ALL_TO_ALL_COMBINE):
            return AllToAll.generate(self, shards, data_GB=data_GB)
        if kind in (CollectiveKind.P2P_PIPELINE_FORWARD, CollectiveKind.P2P_PIPELINE_BACKWARD):
            if len(shards) < 2:
                # Degenerate — single-rank "pipeline". No transfer.
                return P2P(logical_transfers=[])
            return P2P.generate(self, shards[0], shards[1], data_GB=data_GB)
        raise ValueError(f"Unknown CollectiveKind: {kind!r}")

    # ----- simulation ----------------------------------------------------------------

    def simulate(
        self,
        workload_ctx: WorkloadContext,
        hw: HardwareSpec,
        transfer_times_steady: dict[LogicalTransfer, float],
        transfer_times_boundary: dict[LogicalTransfer, float],
    ) -> SimulationResult:
        if self.workload is Workload.GENERATION:
            prefill = self._simulate_one(
                self.operator_pattern,
                self.per_op_network_requirements,
                self.pipeline_partition,
                workload_ctx, hw,
                transfer_times_steady, transfer_times_boundary,
                pipelined=True,
            )
            # Per-token decode = sum(stage_times) — no pipelining benefit at
            # decode (one new token per request per step, no microbatching).
            decode_step = self._simulate_one(
                self.decode_operator_pattern,
                self.decode_per_op_network_requirements,
                self.decode_pipeline_partition,
                workload_ctx, hw,
                transfer_times_steady, transfer_times_boundary,
                pipelined=False,
            )
            decode_time = decode_step.total_time * max(workload_ctx.response_len, 0)
            return SimulationResult(
                total_time=prefill.total_time + decode_time,
                prefill_time=prefill.total_time,
                decode_time=decode_time,
                pp_stage_times=prefill.pp_stage_times,
                per_op_breakdown=prefill.per_op_breakdown + decode_step.per_op_breakdown,
            )

        if self.workload is Workload.DORMANT:
            return SimulationResult(total_time=0.0, pp_stage_times=[0.0])

        result = self._simulate_one(
            self.operator_pattern,
            self.per_op_network_requirements,
            self.pipeline_partition,
            workload_ctx, hw,
            transfer_times_steady, transfer_times_boundary,
            pipelined=True,
        )
        if self.workload is Workload.TRAINING:
            return SimulationResult(
                total_time=result.total_time,
                training_time=result.total_time,
                pp_stage_times=result.pp_stage_times,
                per_op_breakdown=result.per_op_breakdown,
            )
        # PREPARATION
        return result

    def _simulate_one(
        self,
        ops: list[Operator],
        reqs: list[list[NetworkRequirement]],
        partition: list[list[int]],
        workload_ctx: WorkloadContext,
        hw: HardwareSpec,
        transfer_times_steady: dict[LogicalTransfer, float],
        transfer_times_boundary: dict[LogicalTransfer, float],
        pipelined: bool,
    ) -> SimulationResult:
        if not ops:
            return SimulationResult(total_time=0.0, pp_stage_times=[0.0])

        per_op: list[float] = []
        for op, req_list in zip(ops, reqs):
            kernel = op.kernel_time(workload_ctx, hw)
            wall = kernel
            for req in req_list:
                net_op = self.network_ops[req]
                phase = NetworkOp.phase_of(net_op)
                tt = transfer_times_steady if phase is NetworkPhase.STEADY else transfer_times_boundary
                net_time = net_op.get_operator_time(tt)
                # Chain multiple requirements: each refines the running `wall`.
                assoc = NetworkAssociation(
                    network_op=net_op, relation=req.relation, eta=req.eta
                )
                wall = assoc.combine_times(wall, net_time)
            per_op.append(wall)

        stage_times = [sum(per_op[i] for i in stage) for stage in partition]

        if not pipelined or self.parallelism.pp <= 1:
            total = sum(stage_times)
            return SimulationResult(
                total_time=total,
                pp_stage_times=stage_times,
                per_op_breakdown=per_op,
            )

        # 1F1B-shaped iteration time. Bubble correction is implicit in
        # (num_microbatches + pp - 1).
        num_microbatches = max(workload_ctx.num_microbatches, 1)
        pp = self.parallelism.pp
        max_stage = max(stage_times)
        total = (num_microbatches + pp - 1) * max_stage
        return SimulationResult(
            total_time=total,
            pp_stage_times=stage_times,
            per_op_breakdown=per_op,
        )
