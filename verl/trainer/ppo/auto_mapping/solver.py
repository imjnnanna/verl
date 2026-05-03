# trainer/main_ppo.py will call solver.solve(config, role_worker_mapping) before creating ResourcePoolManager
# returns:
# resource_pool_spec: dict[str, list[int]]
# mapping: dict[Role, str]
# parallelism_overrides: dict[Role, dict]

from dataclasses import dataclass
from .enumerators import enum_placement_groups, enum_submesh
from .auto_parallel import auto_parallel
from .mem_model import get_min_alloc
from .simulators import simulate
from .assignment import assign_machines_greedy
from .ray_config import export_solver_result
import numpy as np

@dataclass
class Workload:
    d_in: int # input sequence length
    d_out: int # output sequence length
    compute_type: str # "training", "inference", or "generation"
    
class DeviceMesh:
	def __init__(self, host_ids, host_info, num_hosts, num_devices_per_host):
		self.host_ids = host_ids # list[int]
		self.host_info = host_info # dict[int, dict] - mapping from host_id to host specifications (e.g., CPU, memory, GPU type)
		self.num_hosts = num_hosts # int
		self.num_devices_per_host = num_devices_per_host # int
		self.num_devices = num_hosts * num_devices_per_host # int

class LogicalDeviceMesh:
	def __init__(self, physical_mesh, id_mesh, mesh_alpha=None, mesh_beta=None):
		self.physical_mesh = physical_mesh # PhysicalDeviceMesh
		self.id_mesh = np.array(id_mesh) # np.array - logical grid of device IDs
		self.flattened_id_mesh = tuple(int(x) for x in id_mesh.flatten()) # tuple[int] - flattened logical grid of device IDs
		if mesh_alpha is None:
			mesh_alpha = [1.0] * len(id_mesh.shape)
		if mesh_beta is None:
			mesh_beta = [1.0] * len(id_mesh.shape)
		self.mesh_alpha = mesh_alpha # list[float] - per-mesh-dimension latency coefficients
		self.mesh_beta = mesh_beta # list[float] - per-mesh-dimension bandwidth coefficients
    
class Solver:
	def __init__(self, D, L, W, N, M, Q, topology=None, role_worker_mapping=None):
		self.D = D # list[tuple[int, int]] - RLHF dataflow graph DAG edges
		self.L = L # list[Role] - LLMs in RLHF dataflow
		self.W = W # dict[int, Workload] - workload of LLMs in RLHF dataflow
		self.N = N # int - number of servers
		self.M = M # int - number of devices per server
		self.Q = Q # int - memory capacity per GPU
		self.topology = topology # topology - physical bandwidth tiers; for ray_config export
		self.role_worker_mapping = role_worker_mapping # dict[Role, WorkerType] - solver-id i in list(role_worker_mapping)[i]
	
	def compute_cost(self, g, l_parallel):
		s = 3 # number of stages in D
		c = [0] * s # computation cost per stage

		for group in g:
			c_g = [0] * s
			for i in range(s):
				for l in group:
					c_g[i] += simulate(l_parallel[l], self.W[l])
			c[i] = max(c[i], c_g[i])
		return sum(c)

	def solve(self) -> tuple[dict, dict, dict]:
		# return resource_pool_spec, mapping, parallelism_overrides
		G = enum_placement_groups(self.L, self.N * self.M)
		best_cost = float('inf')
		best_mapping = None
		best_assignments = None

		# calculate cost for each placement group and submesh shape, and find the best one
		# TODO: optimize by getting rid of redundant calculations
		# TODO: auto_parallel different workloads
		for g in G:
			A_min = get_min_alloc(g, self.Q, self.N * self.M)
			min_area = [model[2] for group in A_min for model in group]
			for submeshes in enum_submesh_shapes(self.N, self.M, A_min):
				assignments = None
				if self.topology is not None:
					# skip non-packing submeshes
					assignments = assign_machines_greedy(list(submeshes), self.topology)
					if assignments is None:
						continue
				l_parallel = {}
				l_cost = {}
				for i, group in enumerate(g):
					device_mesh = LogicalDeviceMesh(submeshes[i]) # TODO: fix this part
					for l in group:
						l_cost[l], l_parallel[l] = auto_parallel(l, A_min, self.W[l], device_mesh)
				cost = self.compute_cost(g, l_cost)
				if cost < best_cost:
					best_cost = cost
					best_mapping = (g, submeshes, l_parallel)
					best_assignments = assignments

		if self.topology is None or self.role_worker_mapping is None:
			return best_mapping
		g, submeshes, l_parallel = best_mapping
		return export_solver_result(g, submeshes, l_parallel, best_assignments, self.role_worker_mapping)

# if __name__ == "__main__":
#     # Example usage
# 	D = [(0, 1), (1, 2)] # example dataflow graph edges
# 	L = [0, 1, 2] # example LLMs
# 	W = {0: Workload(128, 128, "training"), 1: Workload(256, 256, "inference"), 2: Workload(512, 512, "generation")} # example workloads
# 	N = 4 # number of servers
# 	M = 8 # number of devices per server
# 	Q = 40 # memory capacity per GPU in GB

# 	solver = Solver(D, L, W, N, M, Q)
# 	resource_pool_spec, mapping, parallelism_overrides = solver.solve()