from typing import List, Tuple

def get_min_alloc(g, Q: int, N_gpus: int) -> List[Tuple[Tuple[int, int, int],...]]:
    # return A_min for each model based on memory capacity Q and total number of GPUs N_gpus
	# A_min[l] is the minimum submesh shape (t, p) that can fit the placement group within memory constraints and the num_devices needed
	# order should follow the shape of g
	raise NotImplementedError()
