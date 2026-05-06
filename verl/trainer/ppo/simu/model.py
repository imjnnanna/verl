from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from enum import Enum

@dataclass
class Model:
    name: str
    role: str