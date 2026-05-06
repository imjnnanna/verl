from typing import List, Tuple
from verl.trainer.ppo.utils import Role
import math
from functools import lru_cache

def enum_placement_groups(L: List[Role], N_gpus: int, colocate_same_models=True) -> List[Tuple[Tuple[int, ...], ...]]:
    '''
    Enumerate all Bell partitions of models into colocated placement groups.

    When colocate_same_models=True, any two roles that are both actors (is_actor())
    or both refs (is_ref()) are forced into the same placement group.

    Returns:
        List of groups:
            ((model_id, ...), ...) - each inner tuple is a colocated group of models
    '''
    n = len(L)

    # Build must-colocate adjacency: pairs that must share a group.
    must_colocate: List[set] = [set() for _ in range(n)]
    if colocate_same_models:
        for i in range(n):
            for j in range(i + 1, n):
                if (L[i].is_actor() and L[j].is_actor()) or (L[i].is_ref() and L[j].is_ref()):
                    must_colocate[i].add(j)
                    must_colocate[j].add(i)

    placements = []
    model_to_group: List[int] = [-1] * n

    def backtrack(i: int, groups: List[List[int]]) -> None:
        if i == n:
            placements.append(tuple(tuple(group) for group in groups))
            return

        # Determine if any already-placed partner constrains which group i must join.
        required_group = None
        for j in must_colocate[i]:
            if model_to_group[j] == -1:
                continue
            if required_group is None:
                required_group = model_to_group[j]
            elif required_group != model_to_group[j]:
                # Partners already split across groups — infeasible branch.
                return

        if required_group is not None:
            # i must join the group that contains its already-placed partners.
            groups[required_group].append(i)
            model_to_group[i] = required_group
            backtrack(i + 1, groups)
            groups[required_group].pop()
            model_to_group[i] = -1
        else:
            # No placed partners yet: i can join any existing group or start a new one.
            for g_idx, group in enumerate(groups):
                group.append(i)
                model_to_group[i] = g_idx
                backtrack(i + 1, groups)
                group.pop()
                model_to_group[i] = -1

            if len(groups) < N_gpus:
                groups.append([i])
                model_to_group[i] = len(groups) - 1
                backtrack(i + 1, groups)
                groups.pop()
                model_to_group[i] = -1

    backtrack(0, [])
    return placements

def valid_submeshes(N: int, M: int, min_area: int) -> List[Tuple[int, int]]:
    # print(f"Finding valid submeshes for N={N}, M={M}, min_area={min_area}")
    submeshes = []
    
    # if M not power of 2
    log2M = int(math.log2(M)) if M > 0 else 0
    if M >= min_area and (1 << log2M) != M:
        submeshes.append((1, M))

    # 1-row submeshes: (1, 1), ..., (1, m)
    for w in range(log2M, -1, -1):
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
    yield from result

# if __name__ == "__main__":
#     L = [Role.Actor, Role.Rollout, Role.Critic, Role.RefPolicy]
#     N = 3
    
#     print(enum_placement_groups(L, N, colocate_same_models=True))
