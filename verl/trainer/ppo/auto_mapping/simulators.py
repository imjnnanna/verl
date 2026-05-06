# training/inference/generation
#
# Module-level dispatcher: when a bridge is registered via `set_bridge()`,
# `simulate(...)` routes to it; otherwise the legacy stub answers (used by
# tests that don't construct a real bridge).

#def simulate(placement_group, model_id, workload, physical_mesh) -> float:
def simulate(*args, **kwargs) -> float:
    # return simulated cost for given placement group, submesh shape, workload, and physical mesh
    # TODO stub (smaller TP whenever possible) - not implemented!! for testing purposes
    placement = args[0] if args else None
    if isinstance(placement, tuple):
        if len(placement) == 3 and all(isinstance(x, int) for x in placement):
            p, t, d = placement
            return float(t*p) + 1.0 / max(d, 1)
    if len(placement) == 2 and isinstance(placement[1], tuple):
        _, (p, t, d) = placement
        return float(t*p) + 1.0 / max(d, 1)
    return 1.0
