"""Bridge between the auto_mapping framework and the simu framework.

Translates auto_mapping's `Topology` + `GroupAssignment` placements into the
simulator's `HostTopo` + `Mesh` + `ModelMapping` + `RLHFTimeline` graph and
runs the simulator. The constructed `HostTopo` is cached on the bridge so
repeated `simulate(...)` calls don't rebuild it.

Two entry points used by auto_mapping:

- `simulate_per_model(parallelism, model_id, workload, device_mesh)`
  replaces `auto_parallel.simulate`; routes to
  `SubmeshMapping.simulate_isolated`.

- `simulate_iteration(g, l_parallel, workloads, assignments)` replaces
  `Solver.compute_cost`; routes to `simulator.simulate_rlhf_iteration`.

The bridge does NOT import from the auto_mapping package at runtime — it
duck-types the inputs (reads `.host_id`, `.host_ids`, `.gpus_per_host`,
`.compute_type`, `.d_in`, `.d_out`, etc.). Type checkers see the real
classes via `TYPE_CHECKING`.

Assumptions and limitations are documented in module docstrings on each
method that depends on one (workload mapping, single-stage-per-model,
global WorkloadContext, etc.).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Optional

from verl.trainer.ppo.simu.hardware import HardwareSpec
from verl.trainer.ppo.simu.mesh import Mesh
from verl.trainer.ppo.simu.model import ArchitectureConfig, BuildPatternFn, Model
from verl.trainer.ppo.simu.model_mapping import ModelMapping
from verl.trainer.ppo.simu.network_requirement import ParallelismConfig
from verl.trainer.ppo.simu.shard import Shard
from verl.trainer.ppo.simu.simulator import simulate_rlhf_iteration
from verl.trainer.ppo.simu.stages import RLHFTimeline, StageBoundary
from verl.trainer.ppo.simu.submesh_mapping import SubmeshMapping
from verl.trainer.ppo.simu.topo import HostTopo, Link, Path
from verl.trainer.ppo.simu.workload_context import Workload, WorkloadContext

if TYPE_CHECKING:
    from verl.trainer.ppo.auto_mapping.assignment import GroupAssignment
    from verl.trainer.ppo.auto_mapping.solver import Workload as AutoWorkload
    from verl.utils.topology import Topology


# Stage ordering for the RLHF dataflow built from auto_mapping placement groups.
# Each model is placed in exactly one stage based on its `compute_type`.
_STAGE_ORDER = ("generation", "inference", "training")
_AUTO_TO_SIMU_WORKLOAD: dict[str, Workload] = {
    "generation": Workload.GENERATION,
    "inference": Workload.PREPARATION,
    "training": Workload.TRAINING,
}


class AutoMappingBridge:
    """Translator + cached simulator topology for one cluster Topology.

    Construct once per cluster (the Topology is the cluster description and
    doesn't change across solver iterations); the bridge caches the
    simulator `HostTopo` derived from it. Each call to `simulate_iteration`
    or `simulate_per_model` reuses that cached topology and builds only the
    per-call `RLHFTimeline` / `SubmeshMapping`.
    """

    def __init__(
        self,
        topology: "Topology",
        hardware: HardwareSpec,
        model_architectures: dict[Any, ArchitectureConfig],
        model_build_patterns: dict[Any, BuildPatternFn],
        link_latency_ms: float = 0.1,
        intra_host_latency_ms: float = 0.0,
        default_microbatch_size: int = 1,
        default_num_microbatches: int = 1,
    ) -> None:
        self.topology = topology
        self.hardware = hardware
        self._archs = dict(model_architectures)
        self._build_patterns = dict(model_build_patterns)
        self._link_latency_ms = link_latency_ms
        self._intra_host_latency_ms = intra_host_latency_ms
        self._default_microbatch_size = default_microbatch_size
        self._default_num_microbatches = default_num_microbatches

        self._host_topo, self._host_id_to_int = self._build_host_topo()

    # ----- entry points ---------------------------------------------------

    def simulate(self, *args, **kwargs) -> float:
        """Variadic dispatcher for the auto_mapping `simulate(...)` callsite.

        auto_parallel.simulate(parallelism, l, W, device_mesh) — 4 args →
        routes to simulate_per_model. Other call shapes raise.
        """
        if len(args) >= 4:
            return self.simulate_per_model(args[0], args[1], args[2], args[3])
        raise TypeError(
            "AutoMappingBridge.simulate received unexpected args; expected "
            "(parallelism, model_id, workload, device_mesh)."
        )

    def simulate_per_model(
        self,
        parallelism_plan: tuple[int, int, int],  # (P, T, D)
        model_id: Any,
        workload: "AutoWorkload",
        device_mesh: Any,
    ) -> float:
        """Build a one-model `SubmeshMapping` and run `simulate_isolated`.

        Requires the device_mesh to carry an `assignment` attribute (a
        `GroupAssignment`). Solver attaches this when constructing the
        `LogicalDeviceMesh`.
        """
        assignment = getattr(device_mesh, "assignment", None)
        if assignment is None:
            raise RuntimeError(
                "AutoMappingBridge.simulate_per_model: device_mesh has no "
                "`assignment` attribute. Solver must construct LogicalDeviceMesh "
                "with the matching GroupAssignment so the bridge can place "
                "shards on physical hosts."
            )
        mesh = self._mesh_for_assignment(assignment)
        mm = self._model_mapping(model_id, parallelism_plan, workload, mesh, assignment)
        sm = SubmeshMapping(model_mappings=[mm], submesh=mesh)
        ctx = self._workload_context(workload)
        return sm.simulate_isolated(ctx, self.hardware, self._host_topo)

    def simulate_iteration(
        self,
        g: list[tuple[int, ...]],
        l_parallel: dict[Any, Any],
        workloads: dict[Any, "AutoWorkload"],
        assignments: list["GroupAssignment"],
    ) -> float:
        """Build a full `RLHFTimeline` and run `simulate_rlhf_iteration`.

        Returns total iteration wall time. Boundaries (resharding) are not
        modeled — auto_mapping doesn't currently provide that information.
        """
        timeline = self._build_timeline(g, l_parallel, workloads, assignments)
        ctx = self._reference_workload_context(workloads)
        result = simulate_rlhf_iteration(timeline, ctx, self.hardware, self._host_topo)
        return result.total_time

    # ----- topology accessors --------------------------------------------

    @property
    def host_topo(self) -> HostTopo:
        return self._host_topo

    def host_id_int(self, auto_host_id: str) -> int:
        return self._host_id_to_int[auto_host_id]

    # ----- internals: topology construction ------------------------------

    def _build_host_topo(self) -> tuple[HostTopo, dict[str, int]]:
        """Convert auto_mapping `Topology` to simulator `HostTopo`. Run once."""
        host_id_to_int: dict[str, int] = {h.host_id: i for i, h in enumerate(self.topology.hosts)}

        connections: dict[tuple[int, int], Path] = {}
        for h1 in self.topology.hosts:
            for h2 in self.topology.hosts:
                if h1.host_id == h2.host_id:
                    continue
                bw = self.topology.bandwidth_between(h1, h2)
                link = Link(
                    node_a_id=host_id_to_int[h1.host_id],
                    node_b_id=host_id_to_int[h2.host_id],
                    bandwidth=bw,
                    latency=self._link_latency_ms,
                )
                connections[(host_id_to_int[h1.host_id], host_id_to_int[h2.host_id])] = Path(link=[link])

        topo = HostTopo(
            intra_host_bandwidth=self.topology.intra_host_bw,
            intra_host_latency=self._intra_host_latency_ms,
            hosts_connections=connections,
        )
        return topo, host_id_to_int

    # ----- internals: per-call mesh / model construction -----------------

    def _mesh_for_assignment(self, assignment: "GroupAssignment") -> Mesh:
        sim_host_ids = [self._host_id_to_int[hid] for hid in assignment.host_ids]
        if not assignment.gpus_per_host:
            raise ValueError(f"GroupAssignment for group {assignment.group_index} has empty gpus_per_host")
        # Within a single GroupAssignment all hosts have the same per-host
        # GPU count by construction (assign_machines_greedy fills them
        # uniformly within a row or full block).
        gpus_per_host = max(assignment.gpus_per_host)
        return Mesh(host_ids=sim_host_ids, num_devices_per_host=gpus_per_host)

    def _model_mapping(
        self,
        model_id: Any,
        parallelism_plan: tuple[int, int, int],
        workload: "AutoWorkload",
        mesh: Mesh,
        assignment: "GroupAssignment",
    ) -> ModelMapping:
        ptd = _unpack_parallelism(parallelism_plan)
        p, t, d = ptd
        parallelism = ParallelismConfig(tp=t, pp=p, dp=d, ep=1)

        sim_workload = _AUTO_TO_SIMU_WORKLOAD.get(workload.compute_type)
        if sim_workload is None:
            raise ValueError(
                f"Unknown auto_mapping compute_type {workload.compute_type!r}; "
                f"expected one of {list(_AUTO_TO_SIMU_WORKLOAD)}"
            )

        arch = self._archs.get(model_id)
        build_pattern = self._build_patterns.get(model_id)
        if arch is None or build_pattern is None:
            raise KeyError(
                f"AutoMappingBridge: no architecture/build_pattern for model "
                f"{model_id!r}. Pass them via the model_architectures / "
                f"model_build_patterns args at bridge construction."
            )

        model = Model(
            name=f"auto_{model_id}",
            role=workload.compute_type,
            architecture=arch,
            build_pattern=build_pattern,
        )

        s2h, h2s = self._assign_shards(model, parallelism, mesh, assignment)
        return ModelMapping(
            model=model,
            mesh=mesh,
            workload=sim_workload,
            parallelism=parallelism,
            shards_to_host_ids=s2h,
            host_id_to_shard=h2s,
        )

    def _assign_shards(
        self,
        model: Model,
        parallelism: ParallelismConfig,
        mesh: Mesh,
        assignment: "GroupAssignment",
    ) -> tuple[dict[Shard, int], dict[int, Shard]]:
        """Map (pp, dp, tp) ranks to host_ids by row-major linear placement.

        Convention: TP is innermost (intra-host), DP middle, PP outermost
        (cross-host). For a (h-rows, m-cols) submesh:
          linear_rank = pp * (dp_size * tp_size) + dp * tp_size + tp
          host_idx    = linear_rank // num_devices_per_host

        For h=1 row submeshes (single host), all ranks land on the single
        host. For h>1 full-width blocks, hosts are consumed in PP×DP order.
        """
        n_ranks = parallelism.dp * parallelism.pp * parallelism.tp
        n_devices = sum(assignment.gpus_per_host)
        if n_ranks > n_devices:
            raise ValueError(
                f"Parallelism (P={parallelism.pp}, T={parallelism.tp}, "
                f"D={parallelism.dp}) needs {n_ranks} ranks but assignment "
                f"has {n_devices} devices."
            )

        sim_host_ids = mesh.host_ids
        m = mesh.num_devices_per_host
        s2h: dict[Shard, int] = {}
        h2s: dict[int, Shard] = {}

        for pp in range(parallelism.pp):
            for dp in range(parallelism.dp):
                for tp in range(parallelism.tp):
                    linear = (
                        pp * (parallelism.dp * parallelism.tp)
                        + dp * parallelism.tp
                        + tp
                    )
                    host_idx = linear // m if m > 0 else linear
                    if host_idx >= len(sim_host_ids):
                        raise ValueError(
                            f"Shard placement overran assignment hosts: "
                            f"linear={linear}, host_idx={host_idx}, "
                            f"available={len(sim_host_ids)}"
                        )
                    host = sim_host_ids[host_idx]
                    shard = Shard(model=model, dp=dp, pp=pp, tp=tp)
                    s2h[shard] = host
                    h2s.setdefault(host, shard)
        return s2h, h2s

    # ----- internals: workload context -----------------------------------

    def _workload_context(self, workload: "AutoWorkload") -> WorkloadContext:
        sim_workload = _AUTO_TO_SIMU_WORKLOAD.get(workload.compute_type, Workload.PREPARATION)
        microbatch = self._default_microbatch_size
        nmb = self._default_num_microbatches
        return WorkloadContext(
            workload_type=sim_workload,
            batch_size=microbatch * nmb,
            microbatch_size=microbatch,
            prompt_len=workload.d_in,
            response_len=workload.d_out,
            num_microbatches=nmb,
        )

    def _reference_workload_context(self, workloads: dict[Any, "AutoWorkload"]) -> WorkloadContext:
        """Pick a representative workload for the iteration-level simulate.

        Preference order: generation > training > inference. Generation
        sets the rollout's prompt/response shape, which is the canonical
        per-iteration size in RLHF.
        """
        for stage_type in ("generation", "training", "inference"):
            for w in workloads.values():
                if w.compute_type == stage_type:
                    return self._workload_context(w)
        if workloads:
            return self._workload_context(next(iter(workloads.values())))
        # No workloads at all — should never happen but be defensive.
        return WorkloadContext(
            workload_type=Workload.PREPARATION,
            batch_size=1, microbatch_size=1,
            prompt_len=1, response_len=0, num_microbatches=1,
        )

    # ----- internals: timeline construction ------------------------------

    def _build_timeline(
        self,
        g: list[tuple[int, ...]],
        l_parallel: dict[Any, Any],
        workloads: dict[Any, "AutoWorkload"],
        assignments: list["GroupAssignment"],
    ) -> RLHFTimeline:
        """One SubmeshMapping per (group × stage_type) cell with non-empty members.

        A given group's submesh appears in every stage that has at least one
        of its models. Stages run sequentially (the simulator sums them);
        within a stage, multiple submeshes run in parallel and contend on
        the shared fabric.
        """
        if len(assignments) != len(g):
            raise ValueError(
                f"len(assignments)={len(assignments)} ≠ len(g)={len(g)}; "
                "every placement group must have a corresponding GroupAssignment."
            )

        group_meshes: dict[int, Mesh] = {}
        cell_mappings: dict[tuple[int, str], list[ModelMapping]] = {}

        for group_idx, group in enumerate(g):
            assignment = assignments[group_idx]
            mesh = self._mesh_for_assignment(assignment)
            group_meshes[group_idx] = mesh
            for model_id in group:
                workload = workloads[model_id]
                stage_type = workload.compute_type
                if stage_type not in _AUTO_TO_SIMU_WORKLOAD:
                    raise ValueError(
                        f"Model {model_id} has unknown compute_type "
                        f"{stage_type!r}"
                    )
                ptd = _unpack_parallelism(l_parallel[model_id])
                mm = self._model_mapping(model_id, ptd, workload, mesh, assignment)
                cell_mappings.setdefault((group_idx, stage_type), []).append(mm)

        stages: list[list[SubmeshMapping]] = []
        for stage_type in _STAGE_ORDER:
            stage_submeshes: list[SubmeshMapping] = []
            for group_idx in range(len(g)):
                mms = cell_mappings.get((group_idx, stage_type))
                if not mms:
                    continue
                stage_submeshes.append(
                    SubmeshMapping(model_mappings=mms, submesh=group_meshes[group_idx])
                )
            if stage_submeshes:
                stages.append(stage_submeshes)

        # No resharding modeled: empty StageBoundary between consecutive stages.
        boundaries = [StageBoundary(transitions=[]) for _ in range(max(0, len(stages) - 1))]
        return RLHFTimeline(stages=stages, boundaries=boundaries)


def _unpack_parallelism(value: Any) -> tuple[int, int, int]:
    """Accept either (P, T, D) or (cost, (P, T, D)) and return (P, T, D)."""
    if isinstance(value, tuple) and len(value) == 3 and all(isinstance(x, int) for x in value):
        return value  # type: ignore[return-value]
    if isinstance(value, tuple) and len(value) == 2 and isinstance(value[1], tuple):
        inner = value[1]
        if len(inner) == 3 and all(isinstance(x, int) for x in inner):
            return inner  # type: ignore[return-value]
    raise ValueError(f"Cannot unpack parallelism plan from {value!r}")
