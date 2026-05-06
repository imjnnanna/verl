from __future__ import annotations
from dataclasses import dataclass

from verl.trainer.ppo.simu.model import Model


@dataclass(frozen=True)
class Shard:
    model: Model
    dp: int  # data parallel rank
    pp: int  # pipeline parallel rank
    tp: int  # tensor parallel rank
