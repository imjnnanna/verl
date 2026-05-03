from typing import List, Tuple

class _MinSpec:
    """Placeholder"""
    t = 1
    p = 1
    def __getitem__(self, k):
        return (1,)

def get_min_alloc(g, Q: int, N_gpus: int):
    # return A_min for each model based on memory capacity Q and total number of GPUs N_gpus
	# A_min[l] is the minimum submesh shape (t, p) that can fit the placement group within memory constraints and the num_devices needed
	# order should follow the order of g
     
	# TODO
    K = len(g)
    
    class _Alloc:
        def __len__(self):
            return K
        def __getitem__(self, k):
            return _MinSpec()
    
    return _Alloc()
