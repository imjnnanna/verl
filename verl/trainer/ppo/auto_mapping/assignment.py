# greedy machine assignment heuristic for the auto-mapper

# 1. sort submeshes by decreasing "demand" (number of machines)
# 2. for each submesh, pair all host-sets and score with (intra-set bandwidth) + alpha * (cross-set bandwidth to neighbors of same group)
# 3. greedy commit; if no candidate fits, fall back to wider bandwidth tier

# O(G*H) (submesh, host) pairs

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from verl.utils.topology import HostSpec, Topology

@dataclass
class GroupAssignment:
    group_index: int
    submesh_shape: tuple[int, int] # (h, w) from enum_submesh_shapes
    host_ids: list[str] # h hosts when w == m, else 1
    gpus_per_host: list[int] # for this group
    block_ids: list[str] # scoring

def assign_machines_greedy(
    submesh_shapes: list[tuple[int, int]],
    topology: Topology,
    alpha: float = 0.1, # weight on cross-group bandwidth; higher = prefer packing groups into blocks, lower = prefer spreading out
) -> Optional[list[GroupAssignment]]:
    # returns None if no assignment is possible; caller should try next submesh enumeration

    m = topology.gpus_per_host()
    for h, w in submesh_shapes:
        if h == 1:
            if w > m:
                return None
        elif w != m:
            return None

    # row-submeshes share a host, block-submeshes share a block; track residual capacity accordingly
    host_residual: dict[str, int] = {h.host_id: h.num_gpus for h in topology.hosts}
    by_id: dict[str, HostSpec] = {h.host_id: h for h in topology.hosts}

    order = sorted(
        range(len(submesh_shapes)),
        key=lambda i: (
            -submesh_shapes[i][0] * submesh_shapes[i][1],
            -(submesh_shapes[i][0] if submesh_shapes[i][1] == m else 0),
        ),
    )

    assignments: list[Optional[GroupAssignment]] = [None] * len(submesh_shapes)

    for idx in order:
        h, w = submesh_shapes[idx]
        if w == m and h >= 1 and (h > 1 or w == m):
            chosen = _pick_full_width_block(h, host_residual, by_id, topology, alpha)
        else:
            chosen = _pick_row_submesh(w, host_residual, by_id, topology, alpha)

        if chosen is None:
            return None

        host_ids, per_host_gpus = chosen
        assignments[idx] = GroupAssignment(
            group_index=idx,
            submesh_shape=(h, w),
            host_ids=host_ids,
            gpus_per_host=per_host_gpus,
            block_ids=[by_id[hid].block_id for hid in host_ids],
        )

    out: list[GroupAssignment] = []
    for a in assignments:
        assert a is not None
        out.append(a)
    return out


def _pick_full_width_block(
    h: int,
    host_residual: dict[str, int],
    by_id: dict[str, HostSpec],
    topology: Topology,
    alpha: float,
) -> Optional[tuple[list[str], list[int]]]:
    m = topology.gpus_per_host()
    # try every block that has h whole hosts free + best score
    candidates: list[tuple[float, list[str]]] = []
    for block_id, hosts in topology.hosts_by_block().items():
        whole = [hh.host_id for hh in hosts if host_residual[hh.host_id] == m]
        if len(whole) >= h:
            picked = whole[:h]
            score = _score_block(picked, by_id, topology, alpha)
            candidates.append((score, picked))

    # cross-block
    if not candidates:
        whole_all = [hid for hid, r in host_residual.items() if r == m]
        if len(whole_all) >= h:
            whole_all_sorted = sorted(
                whole_all,
                key=lambda hid: (by_id[hid].block_id, hid),
            )
            picked = whole_all_sorted[:h]
            candidates.append((_score_block(picked, by_id, topology, alpha), picked))

    if not candidates:
        return None

    candidates.sort(key=lambda x: -x[0])
    picked = candidates[0][1]
    for hid in picked:
        host_residual[hid] = 0
    return picked, [m] * h


def _pick_row_submesh(
    w: int,
    host_residual: dict[str, int],
    by_id: dict[str, HostSpec],
    topology: Topology,
    alpha: float,
) -> Optional[tuple[list[str], list[int]]]:
    # pack w GPUs onto 1 host with best score
    candidates: list[tuple[float, str]] = []
    for hid, residual in host_residual.items():
        if residual >= w:
            score = topology.intra_host_bw
            score += alpha * sum(
                topology.bandwidth_between(by_id[hid], by_id[other])
                for other in host_residual
                if other != hid
            ) / max(1, len(host_residual) - 1)
            candidates.append((score, hid))

    if not candidates:
        return None

    candidates.sort(key=lambda x: (-x[0], x[1]))
    picked = candidates[0][1]
    host_residual[picked] -= w
    return [picked], [w]


def _score_block(
    picked: list[str],
    by_id: dict[str, HostSpec],
    topology: Topology,
    alpha: float,
) -> float:
    # higher = better
    # average intra-set BW + alpha * average cross-set BW
    # adjust alpha to prefer tighter packing (higher) vs spreading out (lower)
    if len(picked) <= 1:
        intra = topology.intra_host_bw
    else:
        pairs = 0
        total = 0.0
        for i in range(len(picked)):
            for j in range(i + 1, len(picked)):
                total += topology.bandwidth_between(by_id[picked[i]], by_id[picked[j]])
                pairs += 1
        intra = total / pairs

    others = [hid for hid in by_id if hid not in picked]
    if not others:
        cross = 0.0
    else:
        total = 0.0
        for hid in picked:
            for o in others:
                total += topology.bandwidth_between(by_id[hid], by_id[o])
        cross = total / (len(picked) * len(others))

    return intra + alpha * cross
