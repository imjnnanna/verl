from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from enum import Enum

from verl.trainer.ppo.simu.mesh import Mesh
from verl.trainer.ppo.simu.model_mapping import ModelMapping

@dataclass
class SubmeshMapping:
    model_mappings: list[ModelMapping]
    submesh: Mesh