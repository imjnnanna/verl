# trainer/main_ppo.py will call solver.solve(config, role_worker_mapping) before creating ResourcePoolManager
# returns:
# resource_pool_spec: dict[str, list[int]]
# mapping: dict[Role, str]
# parallelism_overrides: dict[Role, dict]

from dataclasses import dataclass, field
from typing import Optional
import itertools
from .enumerators import enum_placement_groups, enum_submesh_shapes
from .auto_parallel import auto_parallel
from .mem_model import get_min_alloc
from .assignment import assign_machines_greedy
from .ray_config import export_solver_result
from . import simulators
import numpy as np

@dataclass
class Workload:
    d_in: int # input sequence length
    d_out: int # output sequence length
    compute_type: str # "training", "inference", or "generation"
    
class DeviceMesh:
    def __init__(self, id_mesh, mesh_alpha=None, mesh_beta=None):
        self.id_mesh = np.array(id_mesh) # np.array - logical grid of device IDs
        if mesh_alpha is None:
            mesh_alpha = [1.0] * len(id_mesh.shape)
        if mesh_beta is None:
            mesh_beta = [1.0] * len(id_mesh.shape)
        self.mesh_alpha = mesh_alpha # list[float] - per-mesh-dimension latency coefficients
        self.mesh_beta = mesh_beta # list[float] - per-mesh-dimension bandwidth coefficients

@dataclass
class DataflowGraph:
    edges: list[tuple[int, int]]
    stages: list[list[int]]  # list of stages, each stage is a list of LLM indices
    # Per-stage workload type label parallel to `stages`: one of
    # "generation" / "inference" / "training". Optional; required for the
    # bridge to model 3D-HybridEngine resharding (so a dual-layout role
    # uses GENERATION workload in the gen-stage and TRAINING in the
    # train-stage rather than its single per-model compute_type).
    stage_workload_types: Optional[list[str]] = None
    role_to_stage: dict[int, int] = field(init=False)

    def __post_init__(self):
        if self.stage_workload_types is not None and len(self.stage_workload_types) != len(self.stages):
            raise ValueError(
                f"stage_workload_types length {len(self.stage_workload_types)} != "
                f"stages length {len(self.stages)}"
            )
        # Last-write semantics for roles in multiple stages (dual-layout
        # actor appears in both gen and train stages). Callers needing
        # per-stage role placement should query `roles_in_stage(i)` directly.
        self.role_to_stage = {
            role: i
            for i, stage in enumerate(self.stages)
            for role in stage
        }

    @property
    def num_stages(self) -> int:
        return len(self.stages)

    def roles_in_stage(self, stage_idx: int) -> list[int]:
        return self.stages[stage_idx]
    
class Solver:
    def __init__(self, D, L, W, N, M, Q, topology=None, role_worker_mapping=None, model_specs=None, bridge=None):
        self.D = D # DataflowGraph - dataflow graph of the RLHF pipeline
        self.L = L # list[Role] - Roles in RLHF dataflow
        self.W = W # dict[int, Workload] - workload of roles in RLHF dataflow
        self.N = N # int - number of servers
        self.M = M # int - number of devices per server
        self.Q = Q # int - memory capacity per GPU
        self.topology = topology # topology - physical bandwidth tiers; for ray_config export
        self.role_worker_mapping = role_worker_mapping # dict[Role, WorkerType] - solver-id i in list(role_worker_mapping)[i]
        self.model_specs = model_specs # dict[role_id, ModelSpec] - architecture+workload info for memory accounting
        # Optional simu.bridge.AutoMappingBridge. When set, auto_parallel's
        # per-model `simulate(...)` routes to bridge.simulate_per_model, and
        # compute_cost dispatches to bridge.simulate_iteration for the
        # full-timeline cost.
        self.bridge = bridge

    def compute_cost(self, g, l_cost, l_parallel=None, assignments=None):
        """Score a candidate (g, submeshes, l_parallel).

        Returns `(cost, gen_parallel)` where `gen_parallel` is a
        `dict[role_id, (p_g, t_g, d_g_outer)]` for dual-layout roles
        (empty dict otherwise). The bridge path enumerates valid gen
        layouts and returns the argmin; the legacy aggregator returns
        `(cost, {})`.
        """
        # Bridge path: full RLHF iteration cost via simulate_rlhf_iteration.
        # Needs l_parallel and assignments because the bridge re-builds the
        # operator graph for each (g, submeshes) candidate; per-model l_cost
        # alone isn't enough.
        if self.bridge is not None and l_parallel is not None and assignments is not None:
            return self._compute_cost_bridge(g, l_parallel, assignments)

        # Aggregator over precomputed per-model l_cost (legacy path; also the
        # fallback when bridge isn't fully wired).
        s = self.D.num_stages
        c = [0] * s # computation cost per stage

        for group in g:
            c_g = [0] * s
            for i in range(s):
                for l in group:
                    if l in self.D.roles_in_stage(i):
                        c_g[i] += l_cost[l]
                c[i] = max(c[i], c_g[i])
        return sum(c), {}

    # ----- bridge-path: joint (train, gen) layout search ---------------

    def _compute_cost_bridge(self, g, l_parallel, assignments):
        """Bridge path: enumerate gen layouts for dual-layout roles.

        For each dual-layout role's train (p, t, d), enumerate all
        `(p_g, t_g)` divisor pairs of `(p, t)`. The dest gen layout
        is `(p_g, t_g, d * (p/p_g) * (t/t_g))` to keep total ranks
        equal between training and generation (same physical pool, just
        a relayout). Score every cartesian combination via
        `bridge.simulate_iteration` and take the argmin.

        Without dual-layout roles or without `stage_workload_types` on
        the dag, no enumeration is possible — fall back to a single
        `simulate_iteration` call (legacy bridge behavior, no resharding
        cost contribution).
        """
        stage_workload_types = self.D.stage_workload_types
        dual_layout_role_ids = self._dual_layout_role_ids(l_parallel)

        if not dual_layout_role_ids or stage_workload_types is None:
            cost = self.bridge.simulate_iteration(
                g, l_parallel, self.W, assignments,
                dataflow_graph=self.D,
                stage_workload_types=stage_workload_types,
            )
            return cost, {}

        per_role_options: dict[int, list[tuple[int, int, int]]] = {
            rid: self._enumerate_gen_layouts(*l_parallel[rid])
            for rid in dual_layout_role_ids
        }
        ordered_role_ids = list(per_role_options.keys())
        option_lists = [per_role_options[rid] for rid in ordered_role_ids]

        best_cost = float("inf")
        best_gen: dict[int, tuple[int, int, int]] = {}
        for combo in itertools.product(*option_lists):
            gen_overrides = dict(zip(ordered_role_ids, combo))
            stage_layouts = self._stage_layouts_for_gen(
                l_parallel, gen_overrides, stage_workload_types
            )
            cost = self.bridge.simulate_iteration(
                g, l_parallel, self.W, assignments,
                dataflow_graph=self.D,
                stage_layouts=stage_layouts,
                stage_workload_types=stage_workload_types,
            )
            if cost < best_cost:
                best_cost = cost
                best_gen = gen_overrides
        return best_cost, best_gen

    def _dual_layout_role_ids(self, l_parallel) -> list[int]:
        """Solver-id integers (keys of l_parallel) whose Role is dual-layout."""
        if not self.role_worker_mapping:
            return []
        roles = list(self.role_worker_mapping.keys())
        out: list[int] = []
        for rid in l_parallel:
            if 0 <= rid < len(roles) and roles[rid].is_dual_layout():
                out.append(rid)
        return out

    @staticmethod
    def _enumerate_gen_layouts(p_train: int, t_train: int, d_train: int) -> list[tuple[int, int, int]]:
        """Enumerate (p_g, t_g, d_g_outer) gen layouts for a given train (p, t, d).

        Constraints (from ZeroRedundancyStrategy):
          p_g divides p,  t_g divides t,  d_g_outer = d * (p/p_g) * (t/t_g).

        Always includes the train layout itself (p_g=p, t_g=t → no
        resharding) and the vLLM "no PP" heuristic (p_g=1, t_g=t).
        """
        p_divs = [d for d in range(1, p_train + 1) if p_train % d == 0]
        t_divs = [d for d in range(1, t_train + 1) if t_train % d == 0]
        return [
            (p_g, t_g, d_train * (p_train // p_g) * (t_train // t_g))
            for p_g in p_divs
            for t_g in t_divs
        ]

    def _stage_layouts_for_gen(
        self,
        l_parallel,
        gen_overrides: dict,
        stage_workload_types: list[str],
    ) -> list[dict]:
        """Build per-stage layout overrides: dual-layout roles use the gen
        layout in any "generation" stage, the train layout (`l_parallel`)
        elsewhere. Non-dual-layout roles aren't included (bridge falls back
        to `l_parallel` for them)."""
        stage_layouts: list[dict] = []
        for wt in stage_workload_types:
            stage_layout: dict = {}
            for rid, gen_layout in gen_overrides.items():
                stage_layout[rid] = gen_layout if wt == "generation" else l_parallel[rid]
            stage_layouts.append(stage_layout)
        return stage_layouts

    def solve(self) -> tuple[dict, dict, dict]:
        # return resource_pool_spec, mapping, parallelism_overrides
        G = enum_placement_groups(self.L, self.N * self.M)
        print(f"[solve] N={self.N}, M={self.M}, Q={self.Q}, |L|={len(self.L)}, |G|={len(G)} placements")
        best_cost = float('inf')
        best_mapping = None
        best_assignments = None
        best_gen_parallel: dict = {}

        submesh_cache = {}   # min_area -> list of submesh configs from enum_submesh_shapes
        # ap_cache key includes the assignment's host signature when the bridge
        # is active — different physical assignments span different bandwidth
        # tiers and produce different per-model costs even for identical
        # (l, min_area, h, w).
        ap_cache = {}        # (l, min_area, h, w, assignment_sig) -> (cost, parallel)

        # Register the bridge for the duration of solve(); auto_parallel.simulate
        # consults it via simulators.simulate's module-level dispatcher.
        prev_bridge = simulators.get_bridge()
        if self.bridge is not None:
            simulators.set_bridge(self.bridge)
        try:
            for g in G:
                A_min = get_min_alloc(g, self.Q, self.N * self.M, self.model_specs)
                min_area = tuple(A_min[i].n for i in range(len(g)))

                if min_area not in submesh_cache:
                    submesh_cache[min_area] = enum_submesh_shapes(self.N, self.M, list(min_area))

                n_submeshes = len(submesh_cache[min_area])
                print(f"[solve]   g={g} min_area={min_area} -> {n_submeshes} submesh options")

                for submeshes in submesh_cache[min_area]:
                    assignments = None
                    if self.topology is not None:
                        # skip non-packing submeshes
                        assignments = assign_machines_greedy(list(submeshes), self.topology)
                        if assignments is None:
                            print(f"[solve]     submeshes={submeshes} -> assignment FAILED")
                            continue
                        print(f"[solve]     submeshes={submeshes} -> assigned")
                    l_parallel = {}
                    l_cost = {}
                    for i, group in enumerate(g):
                        h, w = submeshes[i]
                        id_mesh = np.arange(h * w).reshape((h, w))
                        device_mesh = DeviceMesh(id_mesh=id_mesh)
                        assignment_for_group = assignments[i] if assignments is not None else None
                        assignment_sig = (
                            tuple(assignment_for_group.host_ids)
                            if (self.bridge is not None and assignment_for_group is not None)
                            else None
                        )
                        for l in group:
                            key = (l, min_area, h, w, assignment_sig)
                            if key not in ap_cache:
                                ap_cache[key] = auto_parallel(
                                    l, A_min[i], self.W[l], device_mesh,
                                    assignment=assignment_for_group,
                                )
                            l_cost[l], l_parallel[l] = ap_cache[key]
                    cost, gen_parallel = self.compute_cost(
                        g, l_cost, l_parallel=l_parallel, assignments=assignments
                    )
                    if cost < best_cost:
                        best_cost = cost
                        best_mapping = (g, submeshes, l_parallel)
                        best_assignments = assignments
                        best_gen_parallel = gen_parallel
        finally:
            # Restore prior bridge state so concurrent or nested solves don't
            # leak this Solver's bridge into other call sites.
            simulators.set_bridge(prev_bridge)
        
        if best_mapping is None:
            raise RuntimeError(
                f"auto_mapping solver found no feasible plan. "
                f"N={self.N}, M={self.M}, Q={self.Q}, |L|={len(self.L)}, "
                f"|G|={len(G)} placements tried. Likely causes: "
                f"(1) per-group min_area exceeds cluster capacity (model too big for budget), "
                f"(2) enum_submesh_shapes returned empty for every placement, or "
                f"(3) assign_machines_greedy rejected every submesh. "
                f"See `[solve]` log lines above for which step failed."
            )

        if self.topology is None or self.role_worker_mapping is None:
            return best_mapping
        g, submeshes, l_parallel = best_mapping
        if best_gen_parallel:
            print(f"[solve] best gen layouts (per dual-layout role): {best_gen_parallel}")
        return export_solver_result(
            g, submeshes, l_parallel, best_assignments, self.role_worker_mapping,
            l_gen_parallel=best_gen_parallel,
        )

# if __name__ == "__main__":
#     # Example usage
#   D = [(0, 1), (1, 2)] # example dataflow graph edges
#   L = [0, 1, 2] # example LLMs
#   W = {0: Workload(128, 128, "training"), 1: Workload(256, 256, "inference"), 2: Workload(512, 512, "generation")} # example workloads
#   N = 4 # number of servers
#   M = 8 # number of devices per server
#   Q = 40 # memory capacity per GPU in GB

#   solver = Solver(D, L, W, N, M, Q)
#   resource_pool_spec, mapping, parallelism_overrides = solver.solve()
