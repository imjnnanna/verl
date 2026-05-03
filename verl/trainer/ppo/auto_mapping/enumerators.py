from typing import List, Tuple
from enum import Enum

class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6
    Env = 7
    TeacherModel = 8

    def __str__(self):
        return self._get_role_string()

    def _get_role_string(self):
        role_mapping = {
            Role.Actor: "actor",
            Role.Rollout: "rollout",
            Role.ActorRollout: "actor_rollout",
            Role.Critic: "critic",
            Role.RefPolicy: "ref",
            Role.RewardModel: "rm",
            Role.ActorRolloutRef: "actor_rollout_ref",
            Role.TeacherModel: "teacher",
        }
        return role_mapping.get(self, self.name.lower())

    @classmethod
    def from_string(cls, name: str):
        string_mapping = {
            "actor": cls.Actor,
            "rollout": cls.Rollout,
            "actor_rollout": cls.ActorRollout,
            "critic": cls.Critic,
            "ref": cls.RefPolicy,
            "rm": cls.RewardModel,
            "actor_rollout_ref": cls.ActorRolloutRef,
        }
        role = string_mapping.get(name.lower())
        if role is None:
            raise ValueError(f"No Role found for string: {name}")
        return role



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

def enum_submesh_shapes(n, m):
    raise NotImplementedError()

if __name__ == "__main__":
    L = [Role.Actor, Role.Critic, Role.RefPolicy]
    N_gpus = 2
    placements = enum_placement_groups(L, N_gpus)
    print(placements)
