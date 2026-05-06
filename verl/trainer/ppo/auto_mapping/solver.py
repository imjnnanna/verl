# trainer/main_ppo.py will call solver.solve(config, role_worker_mapping) before creating ResourcePoolManager
# returns:
# resource_pool_spec: dict[str, list[int]]
# mapping: dict[Role, str]
# parallelism_overrides: dict[Role, dict]

from dataclasses import dataclass, field
from .enumerators import enum_placement_groups, enum_submesh_shapes
from .auto_parallel import auto_parallel
from .mem_model import get_min_alloc
from .assignment import assign_machines_greedy
from .ray_config import export_solver_result
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
    stages: list[list[int]] # list of stages, each stage is a list of LLM indices
    role_to_stage: dict[int, int] = field(init=False) 
    
    def __post_init__(self):
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
    def __init__(self, D, L, W, N, M, Q, topology=None, role_worker_mapping=None, model_specs=None):
        self.D = D # DataflowGraph - dataflow graph of the RLHF pipeline
        self.L = L # list[Role] - Roles in RLHF dataflow
        self.W = W # dict[int, Workload] - workload of roles in RLHF dataflow
        self.N = N # int - number of servers
        self.M = M # int - number of devices per server
        self.Q = Q # int - memory capacity per GPU
        self.topology = topology # topology - physical bandwidth tiers; for ray_config export
        self.role_worker_mapping = role_worker_mapping # dict[Role, WorkerType] - solver-id i in list(role_worker_mapping)[i]
        self.model_specs = model_specs # dict[role_id, ModelSpec] - architecture+workload info for memory accounting
    
    def compute_cost(self, g, l_cost):
        s = self.D.num_stages 
        c = [0] * s # computation cost per stage

        for group in g:
            c_g = [0] * s
            for i in range(s):
                for l in group:
                    if l in self.D.roles_in_stage(i):
                        c_g[i] += l_cost[l]
            c[i] = max(c[i], c_g[i])
        return sum(c)

    def solve(self) -> tuple[dict, dict, dict]:
        # return resource_pool_spec, mapping, parallelism_overrides
        G = enum_placement_groups(self.L, self.N * self.M)
        print(f"[solve] N={self.N}, M={self.M}, Q={self.Q}, |L|={len(self.L)}, |G|={len(G)} placements")
        best_cost = float('inf')
        best_mapping = None
        best_assignments = None

        submesh_cache = {}   # min_area -> list of submesh configs from enum_submesh_shapes
        ap_cache = {}        # (l, min_area, h, w) -> (cost, parallel) from auto_parallel

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
                    for l in group:
                        key = (l, min_area, h, w)
                        if key not in ap_cache:
                            ap_cache[key] = auto_parallel(l, A_min[i], self.W[l], device_mesh)
                        l_cost[l], l_parallel[l] = ap_cache[key]
                cost = self.compute_cost(g, l_cost)
                if cost < best_cost:
                    best_cost = cost
                    best_mapping = (g, submeshes, l_parallel)
                    best_assignments = assignments
        
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
        return export_solver_result(g, submeshes, l_parallel, best_assignments, self.role_worker_mapping)

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
