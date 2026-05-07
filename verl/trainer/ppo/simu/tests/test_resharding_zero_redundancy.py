"""Unit tests for ZeroRedundancyStrategy (HybridFlow micro-DP resharding).

Validates the rank-mapping math without needing a topology, bridge, or
running PPO. Good first sanity check before any cluster smoke test.

Run on the cluster with:
    pytest verl/trainer/ppo/simu/tests/test_resharding_zero_redundancy.py -v
"""
from __future__ import annotations

import pytest

from verl.trainer.ppo.simu.mesh import Mesh
from verl.trainer.ppo.simu.model import Model
from verl.trainer.ppo.simu.model_mapping import ModelMapping
from verl.trainer.ppo.simu.network_op import AllGatherRing, NetworkOp, P2P
from verl.trainer.ppo.simu.network_requirement import ParallelismConfig
from verl.trainer.ppo.simu.operators.llama_builder import build_llama_pattern
from verl.trainer.ppo.simu.operators.llama_config import LlamaConfig
from verl.trainer.ppo.simu.relation import NetworkPhase
from verl.trainer.ppo.simu.shard import Shard
from verl.trainer.ppo.simu.workload_context import Workload
from verl.trainer.ppo.resharding import (
    ZeroRedundancyStrategy,
    llama_total_parameter_bytes,
)


# n_kv=4 so it divides tp ∈ {1,2,4,8}. dtype_bytes=2 → BF16.
TINY_LLAMA = LlamaConfig(
    h=128,
    n_layers=2,
    n_q=4,
    n_kv=4,
    head_size=32,
    m=512,
    rope_dim=32,
    vocab=1024,
    flash_block_size=64,
    dtype_bytes=2,
)


def _model() -> Model:
    return Model(
        name="actor",
        role="dual_layout",
        architecture=TINY_LLAMA,
        build_pattern=build_llama_pattern,
    )


def _dormant_mapping(
    model: Model, p: int, t: int, d: int, host_ids: list[int]
) -> ModelMapping:
    """Construct a ModelMapping with Workload.DORMANT — skips operator-pattern
    building (we only need parallelism + shard-to-host metadata for the
    resharding strategy)."""
    parallelism = ParallelismConfig(tp=t, pp=p, dp=d, ep=1)
    expected = p * t * d
    if len(host_ids) != expected:
        raise ValueError(
            f"need {expected} host_ids for ParallelismConfig(pp={p}, tp={t}, dp={d}), "
            f"got {len(host_ids)}"
        )
    s2h: dict[Shard, int] = {}
    h2s: dict[int, Shard] = {}
    idx = 0
    # Order matches bridge._assign_shards: linear = pp * (dp * tp) + dp * tp + tp,
    # but for unit tests we just need consistent (s2h, h2s) maps.
    for pp in range(p):
        for dp in range(d):
            for tp in range(t):
                shard = Shard(model=model, dp=dp, pp=pp, tp=tp)
                s2h[shard] = host_ids[idx]
                h2s.setdefault(host_ids[idx], shard)
                idx += 1
    mesh = Mesh(host_ids=sorted(set(host_ids)), num_devices_per_host=1)
    return ModelMapping(
        model=model,
        mesh=mesh,
        workload=Workload.DORMANT,
        parallelism=parallelism,
        shards_to_host_ids=s2h,
        host_id_to_shard=h2s,
    )


def _bytes_total(ops: list[NetworkOp]) -> int:
    """Sum of all logical-transfer payloads across every emitted op."""
    return sum(
        int(round(t.data_GB * 1e9))
        for op in ops
        for t in op.logical_transfers
    )


def _split_by_kind(ops: list[NetworkOp]) -> tuple[list[P2P], list[AllGatherRing]]:
    p2ps = [op for op in ops if isinstance(op, P2P)]
    ags = [op for op in ops if isinstance(op, AllGatherRing)]
    other = [op for op in ops if not isinstance(op, (P2P, AllGatherRing))]
    assert not other, f"unexpected op types: {[type(o).__name__ for o in other]}"
    return p2ps, ags


# ---- tests -----------------------------------------------------------------


def test_identity_layout_emits_only_intrahost_p2p_no_allgather():
    """Train layout == gen layout → a=b=d_g=1 → AllGather skipped, P2Ps all
    src==dst (intra-host, zero-cost on any reasonable topology)."""
    m = _model()
    src = _dormant_mapping(m, p=1, t=1, d=4, host_ids=[0, 1, 2, 3])
    dst = _dormant_mapping(m, p=1, t=1, d=4, host_ids=[0, 1, 2, 3])

    ops = ZeroRedundancyStrategy().compute_network_ops(src, dst)
    p2ps, ags = _split_by_kind(ops)

    assert ags == [], "d_g=1 must skip the AllGather phase"
    for op in p2ps:
        for t in op.logical_transfers:
            assert t.src_host_id == t.dst_host_id, (
                f"identity layout produced cross-host P2P: {t}"
            )


def test_pure_tp_widening_at_gen_emits_one_p2p_per_train_rank():
    """train (p=1, t=2, d=2) → gen (p_g=1, t_g=1, d_g_outer=4) [a=1, b=2, d_g=2].

    Each of the 4 train ranks sends its full per-rank slice to one of 4 gen
    ranks. After P2P, each (d_i, p_g_i, t_g_i) micro-DP group of size 2
    AllGathers to assemble the full gen shard.
    """
    m = _model()
    # 4 train ranks across 4 hosts; same hosts repurposed for gen layout
    # (host_ids=[0,1,2,3] in both).
    src = _dormant_mapping(m, p=1, t=2, d=2, host_ids=[0, 1, 2, 3])
    dst = _dormant_mapping(m, p=1, t=1, d=4, host_ids=[0, 1, 2, 3])

    ops = ZeroRedundancyStrategy().compute_network_ops(src, dst)
    p2ps, ags = _split_by_kind(ops)

    # 2 micro-DP groups (one per d_i), each gathers d_g=2 shards.
    assert len(ags) == 2, f"expected 2 AllGather rings (one per d_i), got {len(ags)}"
    # P2Ps: aggregated by (src_host, dst_host). Lower bound: at least 1.
    # Upper bound: 4 train ranks × 1 dst each = 4 unique pairs at most.
    assert 1 <= len(p2ps) <= 4
    print(f"[zero-redundancy] tp-widen: P2Ps={len(p2ps)} AllGathers={len(ags)}")


def test_p2p_total_bytes_matches_full_parameter_set():
    """Across all P2Ps, total bytes moved == total parameter bytes per DP
    group × number of DP replicas. Each train rank sends its full per-rank
    slice (= total_params / (p · t)), and there are p · t · d such ranks
    → total = total_params × d."""
    m = _model()
    p, t, d = 1, 4, 2
    n = p * t * d
    src = _dormant_mapping(m, p=p, t=t, d=d, host_ids=list(range(n)))
    dst = _dormant_mapping(m, p=1, t=1, d=p * t * d, host_ids=list(range(n)))

    ops = ZeroRedundancyStrategy().compute_network_ops(src, dst)
    p2ps, _ = _split_by_kind(ops)

    total_p2p_bytes = _bytes_total(p2ps)
    expected = llama_total_parameter_bytes(TINY_LLAMA) * d
    # Per-rank bytes is computed via integer division in the strategy
    # (total // (p*t)); the d-fold repetition picks up the same per-rank
    # bytes per DP replica. Allow exact match modulo integer rounding.
    per_rank = llama_total_parameter_bytes(TINY_LLAMA) // (p * t)
    expected_int = per_rank * (p * t * d)
    print(
        f"[zero-redundancy] p2p total: {total_p2p_bytes:,} bytes; "
        f"expected per-rank·n_ranks = {expected_int:,}; "
        f"full param set × d = {expected:,}"
    )
    assert total_p2p_bytes == expected_int, (
        f"sum of P2P bytes ({total_p2p_bytes}) "
        f"≠ per_rank ({per_rank}) × n_ranks ({p*t*d}) = {expected_int}"
    )


def test_all_emitted_ops_are_boundary_phase():
    """`StageTransition.__post_init__` validates that every resharding op is
    BOUNDARY phase. This test catches regressions where a STEADY-default
    op (e.g., AllGatherRing) leaks through without a phase override."""
    m = _model()
    src = _dormant_mapping(m, p=1, t=2, d=2, host_ids=[0, 1, 2, 3])
    dst = _dormant_mapping(m, p=1, t=1, d=4, host_ids=[0, 1, 2, 3])

    ops = ZeroRedundancyStrategy().compute_network_ops(src, dst)
    assert ops, "expected at least one op for a non-trivial reshard"
    for op in ops:
        assert NetworkOp.phase_of(op) is NetworkPhase.BOUNDARY, (
            f"{type(op).__name__} has phase {NetworkOp.phase_of(op)}, "
            f"must be BOUNDARY for resharding ops"
        )


def test_invalid_t_g_not_dividing_t_train_raises():
    m = _model()
    src = _dormant_mapping(m, p=1, t=4, d=1, host_ids=[0, 1, 2, 3])
    # t_g=3 doesn't divide t=4 — strategy must reject before doing any work.
    dst = _dormant_mapping(m, p=1, t=3, d=1, host_ids=[0, 1, 2])
    with pytest.raises(ValueError, match="p_g \\| p and t_g \\| t"):
        ZeroRedundancyStrategy().compute_network_ops(src, dst)


def test_invalid_dest_dp_invariant_raises():
    m = _model()
    src = _dormant_mapping(m, p=1, t=2, d=2, host_ids=[0, 1, 2, 3])
    # Required: dest.dp == source.dp · d_g where d_g=(p/p_g)·(t/t_g).
    # Here a=1, b=2, d_g=2, so dest.dp must be 4. Pass 3 to trigger.
    dst = _dormant_mapping(m, p=1, t=1, d=3, host_ids=[0, 1, 2])
    with pytest.raises(ValueError, match="dest.dp == source.dp"):
        ZeroRedundancyStrategy().compute_network_ops(src, dst)


def test_d_g_one_emits_zero_allgather_ops():
    """When d_g=1 (a=b=1), each gen shard's full content arrives via the P2P
    alone — no intra-micro-DP gather needed. Strategy must skip AllGather."""
    m = _model()
    src = _dormant_mapping(m, p=1, t=2, d=2, host_ids=[0, 1, 2, 3])
    dst = _dormant_mapping(m, p=1, t=2, d=2, host_ids=[0, 1, 2, 3])

    ops = ZeroRedundancyStrategy().compute_network_ops(src, dst)
    _, ags = _split_by_kind(ops)
    assert ags == [], f"d_g=1 should skip AllGather, got {len(ags)} rings"
