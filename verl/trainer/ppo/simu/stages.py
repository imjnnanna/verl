from __future__ import annotations
from dataclasses import dataclass
from enum import Enum

from verl.trainer.ppo.simu.model import Model
from verl.trainer.ppo.simu.model_mapping import ModelMapping
from verl.trainer.ppo.simu.submesh_mapping import SubmeshMapping

@dataclass
class Stages:
    stages: list[list[SubmeshMapping]] # stages with submesh mappings of models that run sequentially within a stage