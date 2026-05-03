# get_topology_from_ray() vs get_topology_from_config()
# use second for testing, first for live Ray cluster introspection

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class HostSpec:
    """A single physical machine in the cluster."""

    host_id: str            # stable identifier (Ray NodeID, hostname, or index)
    block_id: str           # rack / switch / pod the host belongs to
    num_gpus: int           # GPUs available on this host
    gpu_type: str = "unknown"


@dataclass
class Topology:
    """Bandwidth-tiered view of the cluster."""

    hosts: list[HostSpec]
    intra_host_bw: float    # NVLink / NVSwitch within a host
    intra_block_bw: float   # IB within the same rack / leaf switch
    inter_block_bw: float   # cross-rack / spine, typically oversubscribed
    block_pair_bw: dict[tuple[str, str], float] = field(default_factory=dict) # optional block override

    def num_hosts(self) -> int:
        return len(self.hosts)

    def num_gpus(self) -> int:
        return sum(h.num_gpus for h in self.hosts)

    def gpus_per_host(self) -> int:
        per = {h.num_gpus for h in self.hosts}
        if len(per) != 1:
            raise ValueError(f"Topology has heterogeneous gpus_per_host: {per}")
        return per.pop()

    def hosts_by_block(self) -> dict[str, list[HostSpec]]:
        out: dict[str, list[HostSpec]] = {}
        for h in self.hosts:
            out.setdefault(h.block_id, []).append(h)
        return out

    def bandwidth_between(self, h1: HostSpec, h2: HostSpec) -> float:
        if h1.host_id == h2.host_id:
            return self.intra_host_bw
        if h1.block_id == h2.block_id:
            return self.intra_block_bw
        key = (h1.block_id, h2.block_id)
        if key in self.block_pair_bw:
            return self.block_pair_bw[key]
        if (key[1], key[0]) in self.block_pair_bw:
            return self.block_pair_bw[(key[1], key[0])]
        return self.inter_block_bw


def get_topology_from_config(cfg) -> Topology:
    """Build a `Topology` from `config.trainer.auto_mapping.topology`."""
    hosts = [
        HostSpec(
            host_id=str(h.host_id),
            block_id=str(h.block_id),
            num_gpus=int(h.num_gpus),
            gpu_type=str(h.get("gpu_type", "unknown")),
        )
        for h in cfg.hosts
    ]
    block_pair_bw: dict[tuple[str, str], float] = {}
    for entry in cfg.get("block_pair_bw", []) or []:
        block_pair_bw[(str(entry.a), str(entry.b))] = float(entry.bw)

    return Topology(
        hosts=hosts,
        intra_host_bw=float(cfg.intra_host_bw),
        intra_block_bw=float(cfg.intra_block_bw),
        inter_block_bw=float(cfg.inter_block_bw),
        block_pair_bw=block_pair_bw,
    )


def get_topology_from_ray(
    intra_host_bw: float,
    intra_block_bw: float,
    inter_block_bw: float,
    block_of: Optional[Callable[[dict], str]] = None,
    node_label_key: Optional[str] = "rack",
) -> Topology:
    """Each Ray node becomes a `HostSpec`, derive block ids from `block_of` or `node_label_key`."""
    import ray

    if block_of is None:
        def block_of(node: dict) -> str:  # type: ignore[no-redef]
            labels = node.get("Resources", {})
            for k, v in labels.items():
                if isinstance(k, str) and k.startswith(f"{node_label_key}:"):
                    return k.split(":", 1)[1]
            return node.get("NodeManagerHostname", "block0")

    hosts: list[HostSpec] = []
    for node in ray.nodes():
        if not node.get("Alive", False):
            continue
        resources = node.get("Resources", {})
        n_gpu = int(resources.get("GPU", 0) or resources.get("NPU", 0))
        if n_gpu == 0:
            continue
        hosts.append(
            HostSpec(
                host_id=node["NodeID"],
                block_id=block_of(node),
                num_gpus=n_gpu,
            )
        )

    if not hosts:
        raise RuntimeError("get_topology_from_ray: no GPU nodes detected in Ray cluster")

    return Topology(
        hosts=hosts,
        intra_host_bw=intra_host_bw,
        intra_block_bw=intra_block_bw,
        inter_block_bw=inter_block_bw,
    )
