from __future__ import annotations
from dataclasses import dataclass, field
from typing import Union

from verl.trainer.ppo.simu.network_requirement import NetworkRequirement
from verl.trainer.ppo.simu.operator import Operator


# An operator paired with the network requirements it emits at this position.
# `tuple` matches the signature shape the brief specifies for layer builders.
OperatorWithReqs = tuple[Operator, list[NetworkRequirement]]


@dataclass(frozen=True)
class TaggedRepeat:
    """Repeated block of (operator, requirements) tuples.

    Counterpart to operators.composite.Repeat but for the "with-requirements"
    pattern carried through ModelMapping.
    """

    count: int
    members: list[OperatorWithReqs] = field(default_factory=list)

    def expand(self) -> list[OperatorWithReqs]:
        return list(self.members) * self.count


PatternEntry = Union[OperatorWithReqs, TaggedRepeat]


def expand_tagged_pattern(pattern: list[PatternEntry]) -> list[OperatorWithReqs]:
    out: list[OperatorWithReqs] = []
    for item in pattern:
        if isinstance(item, TaggedRepeat):
            out.extend(item.expand())
        else:
            out.append(item)
    return out


def split_pattern(
    expanded: list[OperatorWithReqs],
) -> tuple[list[Operator], list[list[NetworkRequirement]]]:
    """Split parallel lists out of a fully-expanded tagged pattern."""
    ops: list[Operator] = [t[0] for t in expanded]
    reqs: list[list[NetworkRequirement]] = [t[1] for t in expanded]
    return ops, reqs
