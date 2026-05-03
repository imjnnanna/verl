from auto_parallel import auto_parallel
from enumerators import enum_submesh_shapes

class SubmeshSolver:
    def __init__(self, N, M):
        self.N = N # int - number of servers
        self.M = M # int - number of devices per server
        self.mesh_cost_cache = {} # dict[tuple[list, tuple[int, int]], tuple[float, tuple[int, int, int]] - cache for parallelism cost and value simulations per placement group and physical mesh shape
        self.submesh_shapes = [(1, i) for i in range(1, M + 1)]  + [(i, M) for i in range(2, N + 1)] # (1, 1), (1, 2), ..., (1, M), (2, M), (3, M), ..., (N, M)
        
    def intra_op(self, placement_group, A_min, W):
		# TODO: remove redundant calculations across same placement groups
        min_gpus = A_min[0] * A_min[1]
        for (n, m) in self.submesh_shapes:
            A = n * m 
            if A < min_gpus:
                continue
            
            best_cost = 0
            for (n_l, m_l) in enum_submesh_shapes(n, m):
                logical_mesh = LogicalDeviceMesh() # TODO: replace with real values
                l_parallel = []
                total_cost = 0
                for l in placement_group:
                    cost, para = auto_parallel(l, A_min, W[l], logical_mesh) 
                    l_parallel.append(para)
                    total_cost += cost

                if total_cost < best_cost:
                    best_cost = total_cost
                    self.mesh_cost_cache[(placement_group, (n, m))] = (total_cost, l_parallel)
    
    def inter_op_dp():
        best_cost = float('inf')
        for t_max in sortedandfiltered(mesh_cost_cache.values()):
            if B * t_max >= best_cost:
                break
            
            dp(0, L + 1, 0, t_max) = 0
            for s in range(1, L + 1):
                for l in range(L, 0, -1):
                    for d in range(1, N * M + 1):
                        dp(s, l, d, t_max) = min(dp(s-1, k-1, d-delta_d, t_max) + cost(s, t_max), dp(s-1, k, d, t_max))

            best_cost_t_max = min()
			