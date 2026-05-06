from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from verl.trainer.ppo.simu.mesh import Mesh
from verl.trainer.ppo.simu.model import Model
from verl.trainer.ppo.simu.shard import Shard

@dataclass
class ModelMapping:
    model: Model
    mesh: Mesh
    workload: Workload
    tp: int # tensor parallel degree
    pp: int # pipeline parallel degree
    dp: int # data parallel degree
    shards_to_host_ids: dict[Shard, int]
    host_id_to_shard: dict[int, Shard]

class Workload(Enum):
    GENERATION = "generation"
    PREPARATION = "preparation"
    TRAINING = "training"
    DORMANT = "dormant"