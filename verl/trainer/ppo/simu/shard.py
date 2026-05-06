from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

from verl.trainer.ppo.simu.model import Model

@dataclass
class Shard:
    model: Model
    dp: int # data parallel rank of the shard
    pp: int # pipeline parallel rank of the shard
    tp: int # tensor parallel rank of the shard