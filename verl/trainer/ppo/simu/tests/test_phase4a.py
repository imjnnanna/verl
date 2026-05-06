from __future__ import annotations

import pytest

from verl.trainer.ppo.simu.hardware import HardwareSpec
from verl.trainer.ppo.simu.mesh import Mesh
from verl.trainer.ppo.simu.model import Model
from verl.trainer.ppo.simu.model_mapping import ModelMapping
from verl.trainer.ppo.simu.network_op import AllReduceRing
from verl.trainer.ppo.simu.network_requirement import (
    CollectiveKind,
    NetworkRequirement,
    ParallelismConfig,
    ParallelismGroup,
)
from verl.trainer.ppo.simu.operators.backward import BackwardOp
from verl.trainer.ppo.simu.operators.llama_builder import build_llama_pattern
from verl.trainer.ppo.simu.operators.llama_config import LlamaConfig
from verl.trainer.ppo.simu.operators.optimizer import AdamOptimizerOp
from verl.trainer.ppo.simu.shard import Shard
from verl.trainer.ppo.simu.workload_context import Workload, WorkloadContext


SMALL_LLAMA = LlamaConfig(
    h=512,
    n_layers=4,
    n_q=8,
    n_kv=2,
    head_size=64,
    m=2048,
    rope_dim=64,
    vocab=32000,
    flash_block_size=64,
    dtype_bytes=2,
)


A100 = HardwareSpec(
    peak_compute_flops=312e12,
    peak_memory_bandwidth=2.0e12,
    ridge_flops_per_byte=156.0,
)


def _make_model() -> Model:
    return Model(
        name="small_llama",
        role="actor",
        architecture=SMALL_LLAMA,
        build_pattern=build_llama_pattern,
    )


def _make_mapping(workload: Workload, parallelism: ParallelismConfig) -> ModelMapping:
    """Constructs a 1-host-per-shard mesh sized to (dp * pp * tp).

    Shards laid out by (dp, pp, tp) → host_id deterministically.
    """
    model = _make_model()
    shards: list[Shard] = []
    shards_to_host: dict[Shard, int] = {}
    host_to_shard: dict[int, Shard] = {}
    host_id = 0
    for d in range(parallelism.dp):
        for p in range(parallelism.pp):
            for t in range(parallelism.tp):
                s = Shard(model=model, dp=d, pp=p, tp=t)
                shards.append(s)
                shards_to_host[s] = host_id
                host_to_shard[host_id] = s
                host_id += 1
    mesh = Mesh(host_ids=list(range(host_id)), num_devices_per_host=1)
    return ModelMapping(
        model=model,
        mesh=mesh,
        workload=workload,
        parallelism=parallelism,
        shards_to_host_ids=shards_to_host,
        host_id_to_shard=host_to_shard,
    )


def _ctx(batch: int = 2, prompt_len: int = 64, response_len: int = 8) -> WorkloadContext:
    return WorkloadContext(
        workload_type=Workload.PREPARATION,
        batch_size=batch,
        microbatch_size=batch,
        prompt_len=prompt_len,
        response_len=response_len,
        num_microbatches=1,
    )


def test_tp2_emits_all_reduce_network_ops_on_correct_shards():
    mm = _make_mapping(Workload.PREPARATION, ParallelismConfig(tp=2, pp=1, dp=1, ep=1))

    ar_req = NetworkRequirement(
        kind=CollectiveKind.ALL_REDUCE,
        relation=mm.per_op_network_requirements[0][0].relation
        if mm.per_op_network_requirements and mm.per_op_network_requirements[0]
        else None,
        eta=1.0,
        group=ParallelismGroup.TP,
    )
    # Find the single dedup'd AR requirement actually in the dict.
    ar_keys = [
        k for k in mm.network_ops.keys()
        if k.kind is CollectiveKind.ALL_REDUCE and k.group is ParallelismGroup.TP
    ]
    assert len(ar_keys) == 1, f"expected 1 unique TP-AR requirement, got {len(ar_keys)}"
    ar_op = mm.network_ops[ar_keys[0]]
    assert isinstance(ar_op, AllReduceRing)
    # Ring on 2 TP shards → 2 logical transfers (host 0 → 1 → 0).
    assert len(ar_op.logical_transfers) == 2
    # Verify the transfers reference the TP-group shards (host 0 and host 1).
    src_hosts = {t.src_host_id for t in ar_op.logical_transfers}
    dst_hosts = {t.dst_host_id for t in ar_op.logical_transfers}
    assert src_hosts == {0, 1}
    assert dst_hosts == {0, 1}

    # Each layer emits 2 AR requirements (attn_out + ffn_down) × 4 layers = 8.
    total_ar_uses = sum(
        1
        for reqs in mm.per_op_network_requirements
        for r in reqs
        if r.kind is CollectiveKind.ALL_REDUCE
    )
    assert total_ar_uses == 2 * SMALL_LLAMA.n_layers


def test_simulate_pp1_returns_positive_total_with_empty_transfer_times():
    mm = _make_mapping(Workload.PREPARATION, ParallelismConfig(tp=2, pp=1, dp=1, ep=1))
    result = mm.simulate(_ctx(), A100, transfer_times_steady={}, transfer_times_boundary={})
    assert result.total_time > 0.0
    assert len(result.pp_stage_times) == 1


def test_simulate_pp2_partition_and_stage_times():
    mm = _make_mapping(Workload.PREPARATION, ParallelismConfig(tp=2, pp=2, dp=1, ep=1))
    # Pipeline partition: roughly even split across 2 stages.
    assert len(mm.pipeline_partition) == 2
    n_ops = len(mm.operator_pattern)
    sizes = [len(stage) for stage in mm.pipeline_partition]
    assert sum(sizes) == n_ops
    # Stages should differ by at most 1 op (even-by-count partition).
    assert max(sizes) - min(sizes) <= 1

    result = mm.simulate(_ctx(), A100, transfer_times_steady={}, transfer_times_boundary={})
    assert len(result.pp_stage_times) == 2
    assert result.total_time > 0.0


def test_training_includes_backward_and_optimizer():
    mm = _make_mapping(Workload.TRAINING, ParallelismConfig(tp=2, pp=1, dp=1, ep=1))

    backward_ops = [op for op in mm.operator_pattern if isinstance(op, BackwardOp)]
    assert backward_ops, "expected BackwardOp instances in training pattern"

    # Spot-check 3x scaling on a backward op: pick one and compare to its forward.
    sample = backward_ops[0]
    ctx = _ctx()
    fwd_flops = sample.forward.compute_flops(ctx)
    bwd_flops = sample.compute_flops(ctx)
    if fwd_flops > 0:
        assert bwd_flops == pytest.approx(3.0 * fwd_flops)

    optimizers = [op for op in mm.operator_pattern if isinstance(op, AdamOptimizerOp)]
    assert len(optimizers) == 1
    assert optimizers[0].num_parameters > 0


def test_generation_simulate_returns_prefill_and_decode():
    mm = _make_mapping(Workload.GENERATION, ParallelismConfig(tp=2, pp=1, dp=1, ep=1))

    # Both patterns should be populated.
    assert mm.operator_pattern, "prefill pattern empty"
    assert mm.decode_operator_pattern, "decode pattern empty"

    ctx = _ctx(batch=1, prompt_len=64, response_len=8)
    result = mm.simulate(ctx, A100, transfer_times_steady={}, transfer_times_boundary={})

    assert result.prefill_time is not None
    assert result.decode_time is not None
    assert result.prefill_time > 0.0
    assert result.decode_time > 0.0
    assert result.total_time == pytest.approx(result.prefill_time + result.decode_time)
