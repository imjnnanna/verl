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
):
    # roles = role_order_mapping.keys() in iteration order; solver-id i corresponds to roles[i]
    roles = list(role_worker_mapping.keys())
    resource_pool_spec: dict[str, list[int]] = {}
    mapping: dict[Any, str] = {}
    overrides: dict[Any, dict[str, int]] = {}

    for group_idx, group in enumerate(g):
        assignment = assignments[group_idx]
        role_strs = sorted(str(roles[r]) for r in group)
        pool_name = f"auto_pool_{group_idx}_" + "_".join(role_strs)
        resource_pool_spec[pool_name] = list(assignment.gpus_per_host)

        for role_id in group:
            role = roles[role_id]
            mapping[role] = pool_name
            if role_id in l_parallel:
                _, (p, t, d) = l_parallel[role_id]
                overrides[role] = _parallelism_keys_for_role(role, p=p, t=t, d=d)

    return resource_pool_spec, mapping, overrides

# config overrides for RayPPOTrainer
def apply_parallelism_overrides(config, overrides: dict[Any, dict[str, int]]) -> None:
    if not overrides:
        return
    is_omega = OmegaConf.is_config(config)
    for _role, kvs in overrides.items():
        for path, value in kvs.items():
            if is_omega:
                OmegaConf.update(config, path, value, merge=True)
            else:
                _setattr_path(config, path, value)

def _parallelism_keys_for_role(role, *, p: int, t: int, d: int) -> dict[str, int]:
    if role in (Role.ActorRollout, Role.ActorRolloutRef, Role.Actor):
        return {
            "actor_rollout_ref.actor.megatron.tensor_model_parallel_size": t,
            "actor_rollout_ref.actor.megatron.pipeline_model_parallel_size": p,
            "actor_rollout_ref.rollout.tensor_model_parallel_size": t,
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
