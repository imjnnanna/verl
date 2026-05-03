from typing import List, Tuple
# from enum import Enum
import math
from functools import lru_cache

# class Role(Enum):
    # """
    # To create more roles dynamically, you can subclass Role and add new members
    # """

    # Actor = 0
    # Rollout = 1
    # ActorRollout = 2
    # Critic = 3
    # RefPolicy = 4
    # RewardModel = 5
    # ActorRolloutRef = 6
    # Env = 7
    # TeacherModel = 8

    # def __str__(self):
    #     return self._get_role_string()

    # def _get_role_string(self):
    #     role_mapping = {
    #         Role.Actor: "actor",
    #         Role.Rollout: "rollout",
    #         Role.ActorRollout: "actor_rollout",
    #         Role.Critic: "critic",
    #         Role.RefPolicy: "ref",
    #         Role.RewardModel: "rm",
    #         Role.ActorRolloutRef: "actor_rollout_ref",
    #         Role.TeacherModel: "teacher",
    #     }
    #     return role_mapping.get(self, self.name.lower())

    # @classmethod
    # def from_string(cls, name: str):
    #     string_mapping = {
    #         "actor": cls.Actor,
    #         "rollout": cls.Rollout,
    #         "actor_rollout": cls.ActorRollout,
    #         "critic": cls.Critic,
    #         "ref": cls.RefPolicy,
    #         "rm": cls.RewardModel,
    #         "actor_rollout_ref": cls.ActorRolloutRef,
    #     }
    #     role = string_mapping.get(name.lower())
    #     if role is None:
    #         raise ValueError(f"No Role found for string: {name}")
    #     return role



def enum_placement_groups(L: List[Role], N_gpus: int) -> List[Tuple[Tuple[int, ...], ...]]:
    # enumerate all Bell partitions of models into colocated placement groups
    placements = []
    
    def backtrack(i: int, groups: List[List]) -> None:
        # print(f"Backtracking: i={i}, groups={groups}")
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

def valid_submeshes(n: int, m: int, min_area: int) -> List[Tuple[int, int]]:
    submeshes = []

    # 1-row submeshes: (1, 1), ..., (1, m)
    for w in range(0, math.log2(m) + 1):
        if 2**w >= min_area:
            submeshes.append((1, 2**w))

    # full-width multi-row submeshes: (2, m), ..., (n, m)
    for h in range(2, n + 1):
        if h * m >= min_area:
            submeshes.append((h, m))

    return submeshes
    
def enum_submesh_shapes(
    n: int,
    m: int,
    a_min: List[int],
):
    """
    Assign each element a rectangular chunk that fits inside an n x m grid.

    Allowed chunk shapes:
        (1, 1), (1, 2), ..., (1, m)
        (2, m), (3, m), ..., (n, m)

    Constraint:
        chunk_area >= a_min[i]

    Returns:
        List of placements:
            (top_row, left_col, height, width)
        or None if infeasible.
    """
    K = len(a_min)

    @lru_cache(None)
    def dp(i: int, row_used: Tuple[int, ...]):
        if i == K:
            return ()

        for h, w in valid_submeshes(n, m, a_min[i][0][0]):
            # Case 1: one-row submesh
            if h == 1:
                for row in range(n):
                    if row_used[row] + w <= m:
                        left_col = row_used[row]

                        new_row_used = list(row_used)
                        new_row_used[row] += w
                        new_row_used_tuple = tuple(new_row_used)

                        rest = dp(i + 1, new_row_used_tuple)
                        if rest is not None:
                            placement = (h, w)
                            return (placement,) + rest
            # Case 2: full-width multi-row chunk.
            else:
                # Need h completely empty rows.
                for start_row in range(n - h + 1):
                    rows = range(start_row, start_row + h)

                    if all(row_used[r] == 0 for r in rows):
                        new_row_used = list(row_used)

                        for r in rows:
                            new_row_used[r] = m

                        new_row_used_tuple = tuple(new_row_used)

                        rest = dp(i + 1, new_row_used_tuple)
                        if rest is not None:
                            placement = (h, w)
                            return (placement,) + rest

        return None

    initial_state = tuple([0] * n)
    result = dp(0, initial_state)

    if result is None:
        return None

    return list(result)

# if __name__ == "__main__":
#     L = [Role.Actor, Role.Critic, Role.RefPolicy]
#     N_gpus = 2
#     placements = enum_placement_groups(L, N_gpus)
#     print(placements)
