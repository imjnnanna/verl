
def intra_op_calc():
	# return resource_pool_spec, mapping, parallelism_overrides
	G = enum_placement_groups(self.D, self.L, self.N * self.M)
	best_cost = float('inf')
	best_mapping = None

	submesh_shapes = [(1, i) for i in range(1, self.M + 1)]  + [(i, self.M) for i in range(2, self.N + 1)] # (1, 1), (1, 2), ..., (1, M), (2, M), (3, M), ..., (N, M)

	# calculate cost for each placement group and submesh shape, and find the best one
	for g in G:
		for placement_group in g:
			A_min = get_min_alloc(placement_group, self.Q, self.N * self.M)
			for A in enum_submesh(self.N, A_min):
				cost = self.compute_cost(placement_group, A)
				if cost < best_cost:
					best_cost = cost
					best_mapping = (placement_group, A)

	return best_mapping

def inter_op_dp():
    pass