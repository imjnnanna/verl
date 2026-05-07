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
from verl.trainer.ppo.simu.operators.llama_builder import build_llama_pattern
from verl.trainer.ppo.simu.operators.llama_config import LlamaConfig
from verl.trainer.ppo.simu.shard import Shard
from verl.trainer.ppo.simu.simulator import simulate_rlhf_iteration
from verl.trainer.ppo.simu.stages import RLHFTimeline, StageBoundary
from verl.trainer.ppo.simu.submesh_mapping import SubmeshMapping
from verl.trainer.ppo.simu.topo import HostTopo, Link, Path
from verl.trainer.ppo.simu.workload_context import Workload, WorkloadContext

if TYPE_CHECKING:
    from verl.trainer.ppo.auto_mapping.assignment import GroupAssignment
    from verl.trainer.ppo.auto_mapping.solver import DataflowGraph, Workload as AutoWorkload
    from verl.utils.topology import Topology


# Stage ordering for the RLHF dataflow built from auto_mapping placement groups.
# Each model is placed in exactly one stage based on its `compute_type`.
_STAGE_ORDER = ("generation", "inference", "training")
_AUTO_TO_SIMU_WORKLOAD: dict[str, Workload] = {
    "generation": Workload.GENERATION,
    "inference": Workload.PREPARATION,
    "training": Workload.TRAINING,
}


# TODO(architecture-resolution): Hardwired Qwen2.5-0.5B-Instruct as the bridge's
# default model architecture. auto_mapping's `Workload` doesn't carry enough
# info to identify the actual model (only d_in/d_out/compute_type), so the
# bridge falls back to this when callers don't supply per-role architectures.
# Switch to one of:
#   (a) Resolve architecture from auto_mapping's `model_specs` channel once
#       its shape stabilizes (currently a stub in mem_model.py).
#   (b) Have the Solver caller (main_ppo.py) read each role's HF model path
#       from config and construct the per-role ArchitectureConfig externally,
#       passing model_architectures / model_build_patterns explicitly.
# Either path drops this hardwire — every caller-supplied architecture takes
# precedence over the fallback.
_QWEN2_5_0_5B_INSTRUCT = LlamaConfig(
    h=896,           # hidden_size
    n_layers=24,     # num_hidden_layers
    n_q=14,          # num_attention_heads
    n_kv=2,          # num_key_value_heads (GQA)
    head_size=64,    # = hidden_size // num_attention_heads
    m=4864,          # intermediate_size
    rope_dim=64,     # = head_size for Qwen2.5
    vocab=151_936,   # vocab_size
    flash_block_size=64,
    dtype_bytes=2,   # BF16
)
_QWEN_BUILD_PATTERN = build_llama_pattern  # Qwen2.5 = Llama-arch-compatible
                                            # (GQA + SwiGLU + RoPE + RMSNorm).
                                            # Bias on QKV and tied embeddings are
                                            # negligible for analytical timing.


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
        model_architectures: Optional[dict[Any, ArchitectureConfig]] = None,
        model_build_patterns: Optional[dict[Any, BuildPatternFn]] = None,
        link_latency_ms: float = 0.1,
        intra_host_latency_ms: float = 0.0,
        default_microbatch_size: int = 1,
        default_num_microbatches: int = 1,
    ) -> None:
        # `model_architectures` / `model_build_patterns` are optional. Roles
        # without an entry fall back to the Qwen2.5-0.5B hardwire defined at
        # module scope (see _QWEN2_5_0_5B_INSTRUCT TODO). Callers that
        # supply these dicts override the hardwire on a per-role basis.
        self.topology = topology
        self.hardware = hardware
        self._archs = dict(model_architectures or {})
        self._build_patterns = dict(model_build_patterns or {})
        self._link_latency_ms = link_latency_ms
        self._intra_host_latency_ms = intra_host_latency_ms
        self._default_microbatch_size = default_microbatch_size
        self._default_num_microbatches = default_num_microbatches

        self._host_topo, self._host_id_to_int = self._build_host_topo()

    # ----- entry points ---------------------------------------------------

    def simulate(self, *args, **kwargs) -> float:
        """Variadic dispatcher for the auto_mapping `simulate(...)` callsite.

        Accepts `simulate(parallelism, l, W, device_mesh, assignment=...)`
        — 4 positional args + optional `assignment` kwarg. Routes to
        simulate_per_model. Other call shapes raise.
        """
        if len(args) >= 4:
            assignment = kwargs.get("assignment")
            return self.simulate_per_model(
                args[0], args[1], args[2], args[3], assignment=assignment
            )
        raise TypeError(
            "AutoMappingBridge.simulate received unexpected args; expected "
            "(parallelism, model_id, workload, device_mesh, [assignment=...])."
        )

    def simulate_per_model(
        self,
        parallelism_plan: tuple[int, int, int],  # (P, T, D)
        model_id: Any,
        workload: "AutoWorkload",
        device_mesh: Any,
        assignment: Optional["GroupAssignment"] = None,
    ) -> float:
        """Build a one-model `SubmeshMapping` and run `simulate_isolated`.

        Requires a `GroupAssignment`: pass it explicitly via the `assignment`
        kwarg (preferred — Solver threads it through `auto_parallel`'s call),
        or attach it to `device_mesh.assignment` for bridge-aware DeviceMesh
        subclasses (legacy path).

        Returns `float("inf")` when the parallelism is infeasible for the
        chosen architecture (e.g. TP=4 against Qwen's `n_kv=2`, or a shard
        count that exceeds the assignment's device count). auto_parallel
        iterates over many candidates and treats inf as "skip", so this lets
        the search proceed without aborting on the first invalid shape.
        """
        if assignment is None:
            assignment = getattr(device_mesh, "assignment", None)
        if assignment is None:
            raise RuntimeError(
                "AutoMappingBridge.simulate_per_model: no GroupAssignment. "
                "Either pass `assignment=...` from the caller (Solver routes "
                "it through auto_parallel) or attach it to "
                "`device_mesh.assignment`."
            )
        try:
            mesh = self._mesh_for_assignment(assignment)
            mm = self._model_mapping(model_id, parallelism_plan, workload, mesh, assignment)
            sm = SubmeshMapping(model_mappings=[mm], submesh=mesh)
            ctx = self._workload_context(workload)
            print(f"AutoMappingBridge.simulate_per_model: simulating model_id={model_id} with parallelism_plan={parallelism_plan}")
            print(f"  model architecture: {mm.model.architecture}")
            print(f"  model build pattern: {mm.model.build_pattern.__name__}")
            print(f"  mesh: host_ids={mesh.host_ids} num_devices_per_host={mesh.num_devices_per_host}")
            print(f"  workload context: ")
            print(f"    workload_type={ctx.workload_type} batch_size={ctx.batch_size} microbatch_size={ctx.microbatch_size}")
            print(f"    prompt_len={ctx.prompt_len} response_len={ctx.response_len} num_microbatches={ctx.num_microbatches}")
            
            cost = sm.simulate_isolated(ctx, self.hardware, self._host_topo)
            print(f"Simulation result: cost={cost}")
            return cost
        except ValueError:
            # Infeasible plan for this architecture / assignment shape — let
            # the auto_parallel search skip this candidate. Programming errors
            # (KeyError, TypeError, etc.) still propagate.
            print(f"AutoMappingBridge.simulate_per_model: infeasible parallelism_plan={parallelism_plan} for model_id={model_id} with assignment={assignment}")
            return float("inf")

    def simulate_iteration(
        self,
        g: list[tuple[int, ...]],
        l_parallel: dict[Any, Any],
        workloads: dict[Any, "AutoWorkload"],
        assignments: list["GroupAssignment"],
        dataflow_graph: Optional["DataflowGraph"] = None,
    ) -> float:
        """Build a full `RLHFTimeline` and run `simulate_rlhf_iteration`.

        When `dataflow_graph` is supplied, stage construction uses
        `dataflow_graph.roles_in_stage(i)` directly — preferred path because
        it lets the same model appear in multiple stages and respects the
        caller's stage ordering. Without a dataflow_graph, falls back to
        grouping by `Workload.compute_type` against a hardcoded RLHF stage
        order.

        Boundaries (resharding) are not modeled — auto_mapping doesn't
        currently provide that information.
        """
        timeline = self._build_timeline(g, l_parallel, workloads, assignments, dataflow_graph)
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

        # TODO(architecture-resolution): Roles without an explicit entry fall
        # back to the Qwen2.5-0.5B-Instruct hardwire. Replace with proper
        # architecture resolution from `model_specs` (or caller-side wiring)
        # before treating the bridge as production-ready for non-Qwen runs.
        arch = self._archs.get(model_id, _QWEN2_5_0_5B_INSTRUCT)
        build_pattern = self._build_patterns.get(model_id, _QWEN_BUILD_PATTERN)

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
        dataflow_graph: Optional["DataflowGraph"] = None,
    ) -> RLHFTimeline:
        """Build the RLHFTimeline for one (g, l_parallel, assignments) plan.

        With `dataflow_graph`: stage construction uses `dag.roles_in_stage(i)`
        — a model can appear in multiple stages, stage ordering is the dag's,
        and `compute_type` only feeds the per-MM `Workload` enum (not stage
        placement).

        Without `dataflow_graph`: falls back to grouping by `compute_type`
        against `_STAGE_ORDER`, the legacy behavior.
        """
        if len(assignments) != len(g):
            raise ValueError(
                f"len(assignments)={len(assignments)} ≠ len(g)={len(g)}; "
                "every placement group must have a corresponding GroupAssignment."
            )
        if dataflow_graph is not None:
            return self._build_timeline_from_dag(
                g, l_parallel, workloads, assignments, dataflow_graph
            )
        return self._build_timeline_by_compute_type(
            g, l_parallel, workloads, assignments
        )

    def _build_timeline_by_compute_type(
        self,
        g: list[tuple[int, ...]],
        l_parallel: dict[Any, Any],
        workloads: dict[Any, "AutoWorkload"],
        assignments: list["GroupAssignment"],
    ) -> RLHFTimeline:
        """Legacy timeline construction: group by `Workload.compute_type` against
        `_STAGE_ORDER`. One stage per compute_type that has at least one model.
        """
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

        boundaries = [StageBoundary(transitions=[]) for _ in range(max(0, len(stages) - 1))]
        return RLHFTimeline(stages=stages, boundaries=boundaries)

    def _build_timeline_from_dag(
        self,
        g: list[tuple[int, ...]],
        l_parallel: dict[Any, Any],
        workloads: dict[Any, "AutoWorkload"],
        assignments: list["GroupAssignment"],
        dag: "DataflowGraph",
    ) -> RLHFTimeline:
        """Use the explicit dag.stages structure for stage placement.

        Each ModelMapping is built once per (group, model) pair, then placed
        into every stage `i` with `model_id in dag.roles_in_stage(i)`. A
        model that appears in multiple stages contributes its kernel time
        independently in each — accurate for actor-style models that both
        roll out and train within one iteration.
        """
        group_meshes: dict[int, Mesh] = {}
        # (group_idx, model_id) -> ModelMapping (built once, reused across stages)
        model_mappings: dict[tuple[int, Any], ModelMapping] = {}

        for group_idx, group in enumerate(g):
            assignment = assignments[group_idx]
            mesh = self._mesh_for_assignment(assignment)
            group_meshes[group_idx] = mesh
            for model_id in group:
                workload = workloads[model_id]
                ptd = _unpack_parallelism(l_parallel[model_id])
                mm = self._model_mapping(model_id, ptd, workload, mesh, assignment)
                model_mappings[(group_idx, model_id)] = mm

        stages: list[list[SubmeshMapping]] = []
        for stage_idx in range(dag.num_stages):
            roles_in_this_stage = set(dag.roles_in_stage(stage_idx))
            stage_submeshes: list[SubmeshMapping] = []
            for group_idx, group in enumerate(g):
                mms = [
                    model_mappings[(group_idx, mid)]
                    for mid in group
                    if mid in roles_in_this_stage
                ]
                if not mms:
                    continue
                stage_submeshes.append(
                    SubmeshMapping(model_mappings=mms, submesh=group_meshes[group_idx])
                )
            if stage_submeshes:
                stages.append(stage_submeshes)

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
