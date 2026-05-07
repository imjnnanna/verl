"""End-to-end bridge tests for 3D-HybridEngine resharding accuracy.

Validates that the bridge (1) builds per-stage ModelMappings with distinct
parallelism / workload, (2) emits the right StageTransition at the
TRAIN→GEN boundary, (3) returns a `simulate_iteration` total_time that
includes the resharding cost. Useful as a cluster-side smoke test before
running the full PPO loop.

Run with:
    pytest verl/trainer/ppo/simu/tests/test_bridge_3d_hybridengine.py -v
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from verl.trainer.ppo.resharding import NaiveP2PStrategy, ZeroRedundancyStrategy
from verl.trainer.ppo.simu.bridge import AutoMappingBridge
from verl.trainer.ppo.simu.hardware import HardwareSpec
from verl.trainer.ppo.simu.operators.llama_builder import build_llama_pattern
from verl.trainer.ppo.simu.operators.llama_config import LlamaConfig
from verl.trainer.ppo.simu.workload_context import Workload


# ---- fakes for the bridge's ducked-typed inputs -----------------------------
# AutoMappingBridge type-checks via `if TYPE_CHECKING:`, so at runtime we
# only need objects exposing the attributes/methods it consults.


@dataclass(frozen=True)
class _FakeHost:
    host_id: str
    block_id: str = "rack0"
    num_gpus: int = 1
    gpu_type: str = "test"


@dataclass
class _FakeTopology:
    hosts: list[_FakeHost]
    intra_host_bw: float = 1000.0
    intra_block_bw: float = 100.0
    inter_block_bw: float = 50.0

    def bandwidth_between(self, h1: _FakeHost, h2: _FakeHost) -> float:
        if h1.host_id == h2.host_id:
            return self.intra_host_bw
        if h1.block_id == h2.block_id:
            return self.intra_block_bw
        return self.inter_block_bw

    def num_hosts(self) -> int:
        return len(self.hosts)

    def gpus_per_host(self) -> int:
        return self.hosts[0].num_gpus if self.hosts else 0


@dataclass
class _FakeAssignment:
    """Mimics auto_mapping.assignment.GroupAssignment's read surface."""
    group_index: int
    host_ids: list[str]
    gpus_per_host: list[int]
    submesh_shape: tuple[int, int] = (1, 1)
    block_ids: list[str] = field(default_factory=list)


@dataclass
class _FakeWorkload:
    """Mimics auto_mapping.solver.Workload."""
    d_in: int
    d_out: int
    compute_type: str


@dataclass
class _FakeDag:
    """Mimics auto_mapping.solver.DataflowGraph (read surface only)."""
    stages: list[list[int]]
    stage_workload_types: list[str]

    @property
    def num_stages(self) -> int:
        return len(self.stages)

    def roles_in_stage(self, i: int) -> list[int]:
        return self.stages[i]


# ---- shared fixtures --------------------------------------------------------

# n_kv=4 divides tp ∈ {1, 2, 4}; small dimensions keep simulate_iteration fast.
_TINY_LLAMA = LlamaConfig(
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

_A100 = HardwareSpec(
    peak_compute_flops=312e12,
    peak_memory_bandwidth=2.0e12,
    ridge_flops_per_byte=156.0,
)


def _bridge(num_hosts: int = 4, num_gpus_per_host: int = 1) -> AutoMappingBridge:
    hosts = [
        _FakeHost(host_id=f"h{i}", num_gpus=num_gpus_per_host)
        for i in range(num_hosts)
    ]
    topo = _FakeTopology(hosts=hosts)
    return AutoMappingBridge(
        topology=topo,
        hardware=_A100,
        model_architectures={0: _TINY_LLAMA},
        model_build_patterns={0: build_llama_pattern},
    )


def _assignment(host_count: int) -> _FakeAssignment:
    return _FakeAssignment(
        group_index=0,
        host_ids=[f"h{i}" for i in range(host_count)],
        gpus_per_host=[1] * host_count,
        submesh_shape=(host_count, 1),
        block_ids=["rack0"] * host_count,
    )


# ---- tests ------------------------------------------------------------------


def test_dual_layout_role_emits_zero_redundancy_at_train_to_gen_boundary():
    """[infer, train, gen]: actor in stages 1 + 2 with different parallelism →
    bridge emits one StageTransition with ZeroRedundancyStrategy at boundary 1→2."""
    bridge = _bridge(num_hosts=4)
    g = [(0,)]
    workloads = {0: _FakeWorkload(d_in=64, d_out=8, compute_type="training")}
    assignments = [_assignment(4)]
    # Actor only — role 0 in train (stage 1) and gen (stage 2). Empty infer (stage 0).
    dag = _FakeDag(
        stages=[[], [0], [0]],
        stage_workload_types=["inference", "training", "generation"],
    )

    # train (p=1, t=2, d=2) → gen (p_g=1, t_g=1, d_g_outer=4) [a=1, b=2, d_g=2]
    stage_layouts = [
        {},                  # infer (empty)
        {0: (1, 2, 2)},      # train layout: P=1, T=2, D=2
        {0: (1, 1, 4)},      # gen layout:   P=1, T=1, D=4 (d * d_g)
    ]
    timeline = bridge._build_timeline(
        g, l_parallel={0: (1, 2, 2)},
        workloads=workloads, assignments=assignments,
        dataflow_graph=dag,
        stage_layouts=stage_layouts,
        stage_workload_types=dag.stage_workload_types,
    )

    # Empty infer is filtered out → 2 timeline stages, 1 boundary.
    assert len(timeline.stages) == 2
    assert len(timeline.boundaries) == 1

    transitions = timeline.boundaries[0].transitions
    assert len(transitions) == 1, (
        f"expected 1 transition (actor train→gen), got {len(transitions)}"
    )
    transition = transitions[0]
    assert isinstance(transition.strategy, ZeroRedundancyStrategy), (
        f"strategy is {type(transition.strategy).__name__}, expected ZeroRedundancyStrategy"
    )
    assert transition.source.workload is Workload.TRAINING
    assert transition.dest.workload is Workload.GENERATION
    print(
        f"[bridge] emitted {len(transition.network_ops)} network ops at the "
        f"train→gen boundary using ZeroRedundancyStrategy"
    )


def test_identical_layouts_emit_no_transitions():
    """Same parallelism + same workload across stages → no resharding needed."""
    bridge = _bridge(num_hosts=4)
    g = [(0,)]
    workloads = {0: _FakeWorkload(d_in=64, d_out=8, compute_type="training")}
    assignments = [_assignment(4)]
    dag = _FakeDag(
        stages=[[0], [0]],
        stage_workload_types=["training", "training"],
    )
    stage_layouts = [{0: (1, 2, 2)}, {0: (1, 2, 2)}]

    timeline = bridge._build_timeline(
        g, l_parallel={0: (1, 2, 2)},
        workloads=workloads, assignments=assignments,
        dataflow_graph=dag,
        stage_layouts=stage_layouts,
        stage_workload_types=dag.stage_workload_types,
    )

    assert len(timeline.boundaries) == 1
    assert timeline.boundaries[0].transitions == [], (
        "identical layouts should emit no transitions"
    )


def test_naive_p2p_picked_for_non_train_to_gen_transitions():
    """Strategy selector: only TRAINING → GENERATION uses ZeroRedundancyStrategy.
    GENERATION → TRAINING (the reverse) falls back to NaiveP2PStrategy."""
    bridge = _bridge(num_hosts=4)
    g = [(0,)]
    workloads = {0: _FakeWorkload(d_in=64, d_out=8, compute_type="training")}
    assignments = [_assignment(4)]
    # Order [gen, train] flips the workload pair on the only boundary.
    dag = _FakeDag(
        stages=[[0], [0]],
        stage_workload_types=["generation", "training"],
    )
    stage_layouts = [{0: (1, 1, 4)}, {0: (1, 2, 2)}]

    timeline = bridge._build_timeline(
        g, l_parallel={0: (1, 2, 2)},
        workloads=workloads, assignments=assignments,
        dataflow_graph=dag,
        stage_layouts=stage_layouts,
        stage_workload_types=dag.stage_workload_types,
    )

    transitions = timeline.boundaries[0].transitions
    assert len(transitions) == 1
    assert isinstance(transitions[0].strategy, NaiveP2PStrategy)


def test_simulate_iteration_includes_resharding_cost():
    """End-to-end: total iter time with reshard > total iter time without.

    Constructs the same compute graph twice (train+gen for the same actor),
    once with identical parallelism (no reshard) and once with a real layout
    swap. The latter must be strictly larger; the delta is the boundary cost.
    """
    bridge = _bridge(num_hosts=4)
    g = [(0,)]
    workloads = {0: _FakeWorkload(d_in=64, d_out=8, compute_type="training")}
    assignments = [_assignment(4)]
    dag = _FakeDag(
        stages=[[0], [0]],
        stage_workload_types=["training", "generation"],
    )

    # No-reshard baseline: both stages use the SAME layout.
    no_reshard_layouts = [{0: (1, 2, 2)}, {0: (1, 2, 2)}]
    # Some unused arg permutation isn't valid here — for "no reshard" we need
    # stage_layouts to declare gen layout == train layout, satisfying
    # ZeroRedundancy's invariants (a=b=d_g=1, dest.dp == source.dp).
    t_no_reshard = bridge.simulate_iteration(
        g, l_parallel={0: (1, 2, 2)},
        workloads=workloads, assignments=assignments,
        dataflow_graph=dag,
        stage_layouts=no_reshard_layouts,
        stage_workload_types=dag.stage_workload_types,
    )

    # Real reshard: train (1, 2, 2) → gen (1, 1, 4).
    reshard_layouts = [{0: (1, 2, 2)}, {0: (1, 1, 4)}]
    t_with_reshard = bridge.simulate_iteration(
        g, l_parallel={0: (1, 2, 2)},
        workloads=workloads, assignments=assignments,
        dataflow_graph=dag,
        stage_layouts=reshard_layouts,
        stage_workload_types=dag.stage_workload_types,
    )

    print(
        f"[bridge] no-reshard total = {t_no_reshard:.6f}s; "
        f"with-reshard total = {t_with_reshard:.6f}s; "
        f"delta = {t_with_reshard - t_no_reshard:.6f}s"
    )
    assert t_with_reshard > t_no_reshard, (
        f"reshard cost should make iteration slower; got "
        f"no_reshard={t_no_reshard}, with_reshard={t_with_reshard}"
    )
    # The compute portion is identical between the two runs (same operator
    # graphs, same hardware), so the entire delta is the boundary cost.
    assert (t_with_reshard - t_no_reshard) > 0


def test_models_are_reused_across_stages_for_stage_transition_identity():
    """StageTransition does `source.model is dest.model`. With per-stage
    ModelMappings, the bridge must reuse one Model instance per model_id —
    otherwise transition construction raises."""
    bridge = _bridge(num_hosts=4)
    g = [(0,)]
    workloads = {0: _FakeWorkload(d_in=64, d_out=8, compute_type="training")}
    assignments = [_assignment(4)]
    dag = _FakeDag(
        stages=[[0], [0]],
        stage_workload_types=["training", "generation"],
    )

    # Different layouts force two distinct ModelMappings with the same Model.
    stage_layouts = [{0: (1, 2, 2)}, {0: (1, 1, 4)}]
    # If Model identity isn't preserved, _build_timeline raises ValueError
    # from StageTransition.__post_init__("StageTransition requires identical
    # model on both sides"). So this just-runs assertion is the test.
    timeline = bridge._build_timeline(
        g, l_parallel={0: (1, 2, 2)},
        workloads=workloads, assignments=assignments,
        dataflow_graph=dag,
        stage_layouts=stage_layouts,
        stage_workload_types=dag.stage_workload_types,
    )
    transitions = timeline.boundaries[0].transitions
    assert transitions[0].source.model is transitions[0].dest.model


@pytest.mark.parametrize(
    "train_ptd,gen_ptd",
    [
        ((1, 1, 4), (1, 1, 4)),  # identity
        ((1, 2, 2), (1, 1, 4)),  # tp split
        ((1, 4, 1), (1, 1, 4)),  # full tp → all into d_g
        ((2, 2, 1), (1, 1, 4)),  # both p and t collapsed
    ],
)
def test_simulate_iteration_runs_for_assorted_layouts(train_ptd, gen_ptd):
    """Parametric smoke check: every supported (train, gen) pair returns a
    finite, positive iteration time. Useful for catching regressions in
    the strategy without specific value assertions."""
    bridge = _bridge(num_hosts=4)
    g = [(0,)]
    workloads = {0: _FakeWorkload(d_in=64, d_out=8, compute_type="training")}
    assignments = [_assignment(4)]
    dag = _FakeDag(
        stages=[[0], [0]],
        stage_workload_types=["training", "generation"],
    )
    stage_layouts = [{0: train_ptd}, {0: gen_ptd}]

    total = bridge.simulate_iteration(
        g, l_parallel={0: train_ptd},
        workloads=workloads, assignments=assignments,
        dataflow_graph=dag,
        stage_layouts=stage_layouts,
        stage_workload_types=dag.stage_workload_types,
    )
    assert 0 < total < float("inf"), (
        f"iter time {total} for train={train_ptd} gen={gen_ptd}"
    )
    print(f"[bridge] train={train_ptd} gen={gen_ptd} → {total:.6f}s")
