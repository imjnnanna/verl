# trainer/main_ppo.py will call solver.solve(config, role_worker_mapping) before creating ResourcePoolManager
# returns:
# resource_pool_spec: dict[str, list[int]]
# mapping: dict[Role, str]
# parallelism_overrides: dict[Role, dict]

from dataclasses import dataclass
from enumerators import enum_placement_groups, enum_submesh, get_min_alloc
from simulators import simulate, get_cost
import numpy as np

@dataclass
class Workload:
    d_in: int # input sequence length
    d_out: int # output sequence length
    compute_type: str # "training", "inference", or "generation"
    
class PhysicalDeviceMesh:
	def __init__(self, host_ids, host_info, num_hosts, num_devices_per_host):
		self.host_ids = host_ids # list[int]
		self.host_info = host_info # dict[int, dict] - mapping from host_id to host specifications (e.g., CPU, memory, GPU type)
		self.num_hosts = num_hosts # int
		self.num_devices_per_host = num_devices_per_host # int
		self.num_devices = num_hosts * num_devices_per_host # int

class LogicalDeviceMesh:
	def __init__(self, physical_mesh, id_mesh, mesh_alpha, mesh_beta):
		self.physical_mesh = physical_mesh # PhysicalDeviceMesh
		self.id_mesh = np.array(id_mesh) # np.array - logical grid of device IDs
		self.flattened_id_mesh = tuple(int(x) for x in id_mesh.flatten()) # tuple[int] - flattened logical grid of device IDs
		self.mesh_alpha = mesh_alpha # list[float] - per-mesh-dimension latency coefficients
		self.mesh_beta = mesh_beta # list[float] - per-mesh-dimension bandwidth coefficients
    
class Solver:
	def __init__(self, D, L, W, N, M, Q):
		self.D = D # list[tuple[int, int]] - RLHF dataflow graph DAG edges
		self.L = L # list[Role] - LLMs in RLHF dataflow
		self.W = W # dict[int, Workload] - workload of LLMs in RLHF dataflow
		self.N = N # int - number of servers
		self.M = M # int - number of devices per server
		self.Q = Q # int - memory capacity per GPU
		self.para_cost_cache = {} # dict[tuple[list, tuple[int, int]], float] - cache for parallelism cost simulations per placement group and physical mesh shape

	def solve(self) -> tuple[dict, dict, dict]:
		# return resource_pool_spec, mapping, parallelism_overrides

		

		return self.auto_device_mapping()