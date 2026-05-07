# Ray config exporter
# get_topology(config) - resolve a Topology (config or live Ray)
# export_solver_result(...) - convert (g, submeshes, l_parallel, assignments) into (resource_pool_spec, mapping, overrides) for RayPPOTrainer
# apply_parallelism_overrides(config, overrides) - OmegaConf paths to overwrite for a given Role

from __future__ import annotations

from typing import Any
from omegaconf import OmegaConf

from verl.utils.topology import Topology, get_topology_from_config, get_topology_from_ray
from verl.trainer.ppo.utils import Role

def get_topology(config) -> Topology:
    am = config.trainer.get("auto_mapping", {})
    if am.get("topology") is not None:
        return get_topology_from_config(am.topology)
    bw = am.get("bandwidth", {})
    # default bandwidth estimates for H100s, TODO make these configurable
    return get_topology_from_ray(
        intra_host_bw=float(bw.get("intra_host", 600.0)),
        intra_block_bw=float(bw.get("intra_block", 25.0)),
        inter_block_bw=float(bw.get("inter_block", 12.0)),
        node_label_key=str(am.get("node_label_key", "rack")),
    )

def export_solver_result(
    g,
    submeshes,
    l_parallel,
    assignments,
    role_worker_mapping: dict[Any, Any],
    l_gen_parallel: dict[int, tuple] | None = None,
):
    """Convert solver output into ResourcePoolManager-shaped data.

    `l_gen_parallel` (optional) maps dual-layout role_id → (p_g, t_g, d_g_outer)
    for the generation side. When present, the per-role overrides emit the gen
    values for rollout-side OmegaConf paths and the train values for
    Megatron-side paths. When absent (or empty), gen falls back to using the
    train layout — preserving the legacy single-layout output.
    """
    # roles = role_order_mapping.keys() in iteration order; solver-id i corresponds to roles[i]
    roles = list(role_worker_mapping.keys())
    resource_pool_spec: dict[str, list[int]] = {}
    mapping: dict[Any, str] = {}
    overrides: dict[Any, dict[str, int]] = {}
    l_gen_parallel = l_gen_parallel or {}

    for group_idx, group in enumerate(g):
        assignment = assignments[group_idx]
        role_strs = sorted(str(roles[r]) for r in group)
        pool_name = f"auto_pool_{group_idx}_" + "_".join(role_strs)
        resource_pool_spec[pool_name] = list(assignment.gpus_per_host)

        for role_id in group:
            role = roles[role_id]
            mapping[role] = pool_name
            if role_id in l_parallel:
                p, t, d = l_parallel[role_id]
                if role_id in l_gen_parallel:
                    p_g, t_g, _d_g_outer = l_gen_parallel[role_id]
                else:
                    p_g, t_g = p, t
                overrides[role] = _parallelism_keys_for_role(
                    role, p=p, t=t, d=d, p_gen=p_g, t_gen=t_g,
                )

    return resource_pool_spec, mapping, overrides

# config overrides for RayPPOTrainer
def apply_parallelism_overrides(config, overrides: dict[Any, dict[str, int]]) -> None:
    """Write the solver-chosen parallelism into the OmegaConf graph.

    For ablation experiments that need to deploy a layout the solver did NOT
    pick, set the env var `AUTO_MAPPING_SKIP_OVERRIDES` to a comma-separated
    list of dotted paths. Those paths keep whatever value Hydra's CLI args
    landed in the config — useful for forcing e.g. TP=1 in a split placement
    when the solver wanted TP=2.
    """
    if not overrides:
        return
    import os
    skip = {p for p in os.environ.get("AUTO_MAPPING_SKIP_OVERRIDES", "").split(",") if p}
    is_omega = OmegaConf.is_config(config)
    for _role, kvs in overrides.items():
        for path, value in kvs.items():
            if path in skip:
                print(f"[auto_mapping] skipping override of {path} (env AUTO_MAPPING_SKIP_OVERRIDES)")
                continue
            if is_omega:
                parent = path.rsplit(".", 1)[0]
                if OmegaConf.select(config, parent, default=None) is None:
                    continue
                OmegaConf.update(config, path, value, merge=True)
            else:
                _setattr_path(config, path, value)

def _parallelism_keys_for_role(
    role, *, p: int, t: int, d: int, p_gen: int | None = None, t_gen: int | None = None,
) -> dict[str, int]:
    """OmegaConf paths to override for a given Role's selected parallelism.

    `p`, `t`, `d` are the train-side (Megatron) parallelism. `p_gen` and
    `t_gen` are the generation-side (vLLM rollout) parallelism for
    dual-layout roles. When `p_gen` / `t_gen` are None, the gen side uses
    the train values (legacy single-layout behavior).
    """
    if p_gen is None:
        p_gen = p
    if t_gen is None:
        t_gen = t
    if role in (Role.ActorRollout, Role.ActorRolloutRef, Role.Actor):
        return {
            "actor_rollout_ref.actor.megatron.tensor_model_parallel_size": t,
            "actor_rollout_ref.actor.megatron.pipeline_model_parallel_size": p,
            # vLLM / sglang honor rollout PP at runtime (default 1 in
            # rollout.yaml). trtllm asserts ==1; if we ever target it we'll
            # need to clamp p_gen=1 in the search. For now, surface what the
            # solver picked so the deployed layout matches the simulator's
            # cost model.
            "actor_rollout_ref.rollout.tensor_model_parallel_size": t_gen,
            "actor_rollout_ref.rollout.pipeline_model_parallel_size": p_gen,
        }
    if role == Role.Critic:
        return {
            "critic.megatron.tensor_model_parallel_size": t,
            "critic.megatron.pipeline_model_parallel_size": p,
        }
    if role == Role.RefPolicy:
        return {
            "actor_rollout_ref.ref.megatron.tensor_model_parallel_size": t,
            "actor_rollout_ref.ref.megatron.pipeline_model_parallel_size": p,
        }
    if role == Role.RewardModel:
        return {
            "reward.reward_model.megatron.tensor_model_parallel_size": t,
            "reward.reward_model.megatron.pipeline_model_parallel_size": p,
        }
    return {}

def _setattr_path(obj, path: str, value) -> None:
    parts = path.split(".")
    for p in parts[:-1]:
        obj = getattr(obj, p)
    setattr(obj, parts[-1], value)
