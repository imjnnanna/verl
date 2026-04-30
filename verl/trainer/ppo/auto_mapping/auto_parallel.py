from solver import LogicalDeviceMesh, Workload
from simulators import simulate

def auto_parallelism(l: int, A_min: list[tuple[int, int]], W: Workload, device_mesh: LogicalDeviceMesh) -> tuple[int, int, int]:
	# return (PP, DP, TP) for model l
	N = device_mesh.physical_mesh.num_devices
	U = device_mesh.physical_mesh.num_devices_per_host
	t_min = A_min[l].t
	p_min = A_min[l].p
	best_parallelism = None
	best_cost = float('inf')

	for t in range(t_min, U + 1):
		for p in range(p_min, max(p_min + 1, N // U + 1)):
			d = N // (t * p)
			if d < 1:
				continue
			parallelism_plan = (p, t, d)
			cost = simulate(parallelism_plan, l, W, device_mesh)
			if cost < best_cost:
				best_cost = cost
				best_parallelism = parallelism_plan
	return best_parallelism