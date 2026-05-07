# training/inference/generation
#
# Module-level dispatcher: when a bridge is registered via `set_bridge()`,
# `simulate(...)` routes to it; otherwise the legacy stub answers (used by
# tests that don't construct a real bridge).

from __future__ import annotations

from typing import Optional

_bridge: Optional[object] = None  # AutoMappingBridge; avoid runtime import to dodge cycles


def set_bridge(bridge) -> None:
    """Register the active AutoMappingBridge for `simulate(...)` to delegate to.

    Pass `None` to reset to the legacy stub. The bridge lives in
    `verl.trainer.ppo.simu.bridge.AutoMappingBridge` and is owned by the
    `Solver`.
    """
    global _bridge
    _bridge = bridge


def get_bridge():
    return _bridge


# def simulate(placement_group, model_id, workload, physical_mesh, assignment=None) -> float:
def simulate(*args, **kwargs) -> float:
    """Cost of a per-model placement.

    With a bridge registered: dispatched to `AutoMappingBridge.simulate`,
    which routes auto_parallel's 4-arg form (plus optional `assignment`
    kwarg) into `SubmeshMapping.simulate_isolated`.

    Without a bridge: returns the legacy stub (a TP×PP-favoring score), kept
    for backward compatibility with tests that haven't wired in a real bridge.
    """
    if _bridge is not None:
        return _bridge.simulate(*args, **kwargs)

    # Legacy stub (smaller TP whenever possible, not implemented for real).
    placement = args[0] if args else None
    if isinstance(placement, tuple):
        if len(placement) == 3 and all(isinstance(x, int) for x in placement):
            p, t, d = placement
            return float(t * p) + 1.0 / max(d, 1)
    if len(placement) == 2 and isinstance(placement[1], tuple):
        _, (p, t, d) = placement
        return float(t * p) + 1.0 / max(d, 1)
    return 1.0
