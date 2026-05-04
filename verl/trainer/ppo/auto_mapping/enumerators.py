from typing import List, Tuple
from verl.trainer.ppo.utils import Role
import math
from functools import lru_cache

# TODO: add colocation constraints to enum_placement_groups, e.g. "model 0 and model 2 must be colocated"
def enum_placement_groups(L: List[Role], N_gpus: int, colocate_same_models=True) -> List[Tuple[Tuple[int, ...], ...]]:
    '''
    Enumerate all Bell partitions of models into colocated placement groups.
    
    Returns:
        List of groups:
            ((model_id, ...), ...) - each inner tuple is a colocated group of models
    '''
    placements = []
    
    def backtrack(i: int, groups: List[List[int]]) -> None:
        if i == len(L):
            placements.append(tuple(tuple(group) for group in groups))
            return
        
        for group in groups:
            group.append(i)
            backtrack(i + 1, groups)
            group.pop()
        
        if len(groups) < N_gpus:
            groups.append([i])
            backtrack(i + 1, groups)
            groups.pop()
            
    backtrack(0, [])
    return placements

def valid_submeshes(N: int, M: int, min_area: int) -> List[Tuple[int, int]]:
    # print(f"Finding valid submeshes for N={N}, M={M}, min_area={min_area}")
    submeshes = []

    # 1-row submeshes: (1, 1), ..., (1, m)
    for w in range(int(math.log2(M)), -1, -1):
        if 2**w < min_area:
            break
        submeshes.append((1, 2**w))
        # print(f"Added 1-row submesh: (1, {2**w})")

    # full-width multi-row submeshes: (2, m), ..., (n, m)
    for h in range(N, 1, -1):
        if h * M < min_area:
            break
        submeshes.append((h, M))
        # print(f"Added full-width submesh: ({h}, {M})")
            
    return submeshes
    
def enum_submesh_shapes(N: int, M: int, A_min: List[int]) -> List[List[Tuple[int, int]]]:
    """
    For each of the colocated groups, assign each element a rectangular submesh that fits inside an N x M grid.

    Allowed submesh shapes:
        (1, 1), (1, 2), (1, 4), ..., (1, M) - one-row submeshes with widths that are powers of 2
        (2, M), (3, M), (2, M), ..., (N, M) - full nodes

    Constraint:
        n_i * m_i >= A_min[i]

    Returns:
        List of device mesh shapes:
            (n_i, m_i)
    """
    K = len(A_min)
    total_area = N * M
    
    if sum(A_min) > total_area:
        return []

    shapes_by_i = [
        valid_submeshes(N, M, A_min[i])
        for i in range(K)
    ]

    @lru_cache(None)
    def dp(
        i: int,
        row_used: Tuple[int, ...],
        area_used: int,
    ) -> Tuple[Tuple[Tuple[int, int], ...], ...]:
        if i == K:
            if area_used == total_area and all(x == M for x in row_used):
                return ((),)
            return ()

        # Prune: already overfilled by area.
        if area_used > total_area:
            return ()

        # Prune: even using minimum required areas, cannot fit exactly anymore.
        remaining_min_area = sum(A_min[i:])
        if area_used + remaining_min_area > total_area:
            return ()

        results = set()

        for h, w in shapes_by_i[i]:
            shape_area = h * w

            if area_used + shape_area > total_area:
                continue

            if h == 1:
                # Place one-row chunk into any row with enough remaining width.
                for row in range(N):
                    if row_used[row] + w <= M:
                        new_row_used = list(row_used)
                        new_row_used[row] += w
                        new_state = tuple(new_row_used)

                        suffixes = dp(
                            i + 1,
                            new_state,
                            area_used + shape_area,
                        )

                        for suffix in suffixes:
                            results.add(((h, w),) + suffix)

            else:
                # Place full-width h-row chunk.
                # Requires h completely empty contiguous rows.
                for start_row in range(N - h + 1):
                    rows = range(start_row, start_row + h)

                    if all(row_used[r] == 0 for r in rows):
                        new_row_used = list(row_used)

                        for r in rows:
                            new_row_used[r] = M

                        new_state = tuple(new_row_used)

                        suffixes = dp(
                            i + 1,
                            new_state,
                            area_used + shape_area,
                        )

                        for suffix in suffixes:
                            results.add(((h, w),) + suffix)

        return tuple(sorted(results))

    initial_row_used = tuple([0] * N)
    tilings = dp(0, initial_row_used, 0)

    return [list(t) for t in tilings]

def enum_submesh(n, m, a_min):
    result = enum_submesh_shapes(n, m, a_min)
    if result is None:
        return
    yield list(result)

# if __name__ == "__main__":
#     N = 3
#     M = 16
#     A_min = [1, 2, 1, 1, 1]
#     print(enum_submesh_shapes(N, M, A_min))
