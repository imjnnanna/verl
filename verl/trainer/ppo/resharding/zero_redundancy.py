"""HybridFlow zero-redundancy resharding (training → generation).

Implements the rank-mapping derivation for a 3D-HybridEngine sync, generalized
to any factorization of d_g into (a, b) such that a | p and b | t. This is
strictly more general than the original HybridFlow paper (Section 5.3), which
implicitly treats t_g and d_g as divisors of t and d respectively.

ModelMapping convention assumed by `compute_network_ops`:
    source: training,   ParallelismConfig(dp=d,        pp=p,   tp=t)
    dest:   generation, ParallelismConfig(dp=d * d_g,  pp=p_g, tp=t_g)
            with p = p_g · a, t = t_g · b, d_g = a · b.

Rank mapping (one P2P per training rank, no duplication):
    Each training shard (d_i, p_i, t_i) sends to the generation rank with
        d_i_dest = d_i · d_g + d_g_i
        p_g_i    = p_i // a
        t_g_i    = t_i // b
        δ_p      = p_i %  a
        δ_t      = t_i %  b
        d_g_i    = δ_p · b + δ_t
    where (δ_p, δ_t) is the position of the source shard inside its (a × b)
    micro-DP tile of the source p × t grid.

After the P2P phase, each gen rank holds 1/d_g of its target generation
shard. A d_g-way AllGather inside each (d_i, p_g_i, t_g_i) micro-DP group
assembles the full shard. We emit both phases as BOUNDARY-phase NetworkOps
so they participate in `StageBoundary`'s flow-time contention snapshot.

Bytes moved per training rank during the P2P phase: total_param_bytes /
(p · t). Bytes moved per AllGather ring: d_g · per_rank_bytes (split across
the d_g members; final shard size = d_g · per_rank_bytes per rank).
"""

from __future__ import annotations
from collections import defaultdict
from typing import TYPE_CHECKING

from verl.trainer.ppo.resharding.strategy import (
    ReshardingStrategy,
    llama_total_parameter_bytes,
)
from verl.trainer.ppo.simu.network_op import (
    AllGatherRing,
    LogicalTransfer,
    NetworkOp,
    P2P,
)
from verl.trainer.ppo.simu.operators.llama_config import LlamaConfig
from verl.trainer.ppo.simu.relation import NetworkPhase
from verl.trainer.ppo.simu.shard import Shard

if TYPE_CHECKING:
    from verl.trainer.ppo.simu.model_mapping import ModelMapping


class ZeroRedundancyStrategy(ReshardingStrategy):
    """Training → generation resharding with one P2P per training rank.

    Auto-derives (a, b, d_g) from source/dest ParallelismConfig under the
    convention documented at module top. Validates divisibility and the
    dest.dp == source.dp · d_g invariant before emitting ops.
    """

    def compute_network_ops(
        self,
        source: "ModelMapping",
        dest: "ModelMapping",
    ) -> list[NetworkOp]:
        arch = source.model.architecture
        if not isinstance(arch, LlamaConfig):
            raise NotImplementedError(
                "ZeroRedundancyStrategy currently relies on llama_total_parameter_bytes; "
                "extend to V3/MoE by routing through an architecture-specific byte counter."
            )

        d, p, t = source.parallelism.dp, source.parallelism.pp, source.parallelism.tp
        d_dest, p_g, t_g = dest.parallelism.dp, dest.parallelism.pp, dest.parallelism.tp

        if p_g <= 0 or t_g <= 0:
            raise ValueError(f"dest parallelism must be positive; got pp={p_g}, tp={t_g}")
        if p % p_g != 0 or t % t_g != 0:
            raise ValueError(
                f"ZeroRedundancyStrategy requires p_g | p and t_g | t; "
                f"got source(pp={p}, tp={t}) dest(pp={p_g}, tp={t_g})"
            )
        a = p // p_g
        b = t // t_g
        d_g = a * b
        if d_dest != d * d_g:
            raise ValueError(
                f"ZeroRedundancyStrategy requires dest.dp == source.dp · d_g; "
                f"got dest.dp={d_dest}, source.dp={d}, d_g={d_g} (a={a}, b={b})"
            )

        total_param_bytes = llama_total_parameter_bytes(arch)
        per_rank_bytes = total_param_bytes // (p * t)

        ops: list[NetworkOp] = []
        ops.extend(self._p2p_ops(source, dest, d, p, t, a, b, d_g, per_rank_bytes))
        if d_g > 1:
            ops.extend(self._allgather_ops(dest, d, p_g, t_g, d_g, per_rank_bytes))
        return ops

    def _p2p_ops(
        self,
        source: "ModelMapping",
        dest: "ModelMapping",
        d: int,
        p: int,
        t: int,
        a: int,
        b: int,
        d_g: int,
        per_rank_bytes: int,
    ) -> list[NetworkOp]:
        agg: dict[tuple[int, int], int] = defaultdict(int)
        for d_i in range(d):
            for p_i in range(p):
                for t_i in range(t):
                    p_g_i, delta_p = divmod(p_i, a)
                    t_g_i, delta_t = divmod(t_i, b)
                    d_g_i = delta_p * b + delta_t
                    src_shard = Shard(model=source.model, dp=d_i, pp=p_i, tp=t_i)
                    dst_shard = Shard(
                        model=dest.model,
                        dp=d_i * d_g + d_g_i,
                        pp=p_g_i,
                        tp=t_g_i,
                    )
                    src_host = source.shards_to_host_ids[src_shard]
                    dst_host = dest.shards_to_host_ids[dst_shard]
                    agg[(src_host, dst_host)] += per_rank_bytes

        ops: list[NetworkOp] = []
        for (src_host, dst_host), bytes_total in agg.items():
            if bytes_total <= 0:
                continue
            transfer = LogicalTransfer(
                src_host_id=src_host,
                dst_host_id=dst_host,
                data_GB=bytes_total / 1e9,
            )
            ops.append(P2P(logical_transfers=[transfer]))
        return ops

    def _allgather_ops(
        self,
        dest: "ModelMapping",
        d: int,
        p_g: int,
        t_g: int,
        d_g: int,
        per_rank_bytes: int,
    ) -> list[NetworkOp]:
        # Per-ring data_GB is the gathered shard size; AllGatherRing.generate
        # divides it across the d_g ring members internally.
        ring_data_GB = (d_g * per_rank_bytes) / 1e9
        ops: list[NetworkOp] = []
        for d_i in range(d):
            for p_g_i in range(p_g):
                for t_g_i in range(t_g):
                    ring_shards = [
                        Shard(
                            model=dest.model,
                            dp=d_i * d_g + d_g_i,
                            pp=p_g_i,
                            tp=t_g_i,
                        )
                        for d_g_i in range(d_g)
                    ]
                    ag = AllGatherRing.generate(
                        model_mapping=dest,
                        shards_ring=ring_shards,
                        data_GB=ring_data_GB,
                    )
                    if not ag.logical_transfers:
                        continue
                    # AllGatherRing defaults to STEADY; resharding ops must
                    # be BOUNDARY (validated by StageTransition).
                    ops.append(
                        AllGatherRing(
                            logical_transfers=ag.logical_transfers,
                            phase=NetworkPhase.BOUNDARY,
                        )
                    )
        return ops
