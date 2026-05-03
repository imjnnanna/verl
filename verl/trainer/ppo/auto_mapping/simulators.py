# training/inference/generation

def simulate(placement_group, model_id, workload, physical_mesh) -> float:
	# return simulated cost for given placement group, submesh shape, workload, and physical mesh
	# TODO stub (smaller TP whenever possible) - not implemented!! for testing purposes
	if isinstance(placement_group, tuple):
		# (p, t, d) from auto_parallel
		if len(placement_group) == 3:
			p, t, d = placement_group
			return float(t*p) + 1.0 / max(d, 1)
	if len(placement_group) == 2 and isinstance(placement_group[1], tuple):
		_, (p, t, d) = placement_group
		return float(t*p) + 1.0 / max(d, 1)
	return 1.0