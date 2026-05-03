from solver import LogicalDeviceMesh, Workload
from simulators import simulate

def auto_parallel(l: int, A_min: list[tuple[int, int]], W: Workload, device_mesh: LogicalDeviceMesh) -> tuple[int, tuple[int, int, int]]:
	# return cost and (PP, DP, TP) for model l
	num_hosts = device_mesh.id_mesh.shape[0]
	num_devices_per_host = device_mesh.id_mesh.shape[1]
	num_devices = num_hosts * num_devices_per_host
	t_min = A_min[l].t
	p_min = A_min[l].p
	best_parallelism = None
	best_cost = float('inf')

	for t in range(t_min, num_devices_per_host + 1):
		for p in range(p_min, max(p_min + 1, num_hosts + 1)):
			d = num_devices // (t * p)
			if d < 1:
				continue
			parallelism_plan = (p, t, d)
			cost = simulate(parallelism_plan, l, W, device_mesh)
			if cost < best_cost:
				best_cost = cost
				best_parallelism = parallelism_plan
	return best_cost, best_parallelism