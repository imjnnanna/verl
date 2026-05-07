#!/usr/bin/env bash
# Auto-mapping ablation suite.
#
# Runs four configurations back-to-back on the same 8-GPU box and writes each
# run's full stdout to its own log file under ./ablation_runs/<timestamp>/.
# At the end, dumps a side-by-side TSV summary with the headline metrics
# (per-step time, throughput, MFU, final reward, final entropy) plus the
# solver's per-candidate cost predictions for the simulator-accuracy check.
#
# Runs:
#   E1-default     : verl baseline, no auto_mapping (K=1 colocated, all TP=1).
#   E1-auto        : auto_mapping enabled; solver picks the layout.
#   E2-auto-tp1    : auto_mapping picks placement, but TP forced to 1 via env
#                    AUTO_MAPPING_SKIP_OVERRIDES + Hydra CLI. Tests "what if
#                    the solver had picked split + TP=1 instead".
#   E2-auto-altgen : auto_mapping picks layout, but rollout TP forced to 1
#                    (gen layout != train layout). Tests the gen-side decision.
#
# Useful comparisons:
#   E1-default vs E1-auto        → end-to-end auto-mapping speedup (E1).
#   E1-auto vs E2-auto-tp1       → TP=2 vs TP=1 with same split (E3a ranking).
#   E1-auto vs E2-auto-altgen    → same train layout, different gen (resharding).
#
# Tail of each .log has the last `step:` line (final metrics). The summary
# at the end also pulls `[solve]` lines for sim-vs-runtime comparison.

set -uo pipefail

ROOT="${ROOT:-/root/verl}"
RUNS_DIR="${RUNS_DIR:-${ROOT}/ablation_runs/$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${RUNS_DIR}"
echo "Output directory: ${RUNS_DIR}"

# Common Hydra args (kept identical across runs so comparisons are clean).
COMMON_ARGS=(
    algorithm.adv_estimator=gae
    data.train_files=/root/data/gsm8k/train.parquet
    data.val_files=/root/data/gsm8k/test.parquet
    data.train_batch_size=64
    data.max_prompt_length=512
    data.max_response_length=512
    data.filter_overlong_prompts=True
    data.truncation=error
    actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.actor.ppo_mini_batch_size=32
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4
    critic.optim.lr=1e-5
    critic.model.use_remove_padding=True
    critic.model.path=Qwen/Qwen2.5-0.5B-Instruct
    critic.ppo_micro_batch_size_per_gpu=2
    critic.fsdp.param_offload=False
    critic.fsdp.optimizer_offload=False
    algorithm.use_kl_in_reward=False
    trainer.balance_batch=False
    trainer.critic_warmup=0
    trainer.logger=[console]
    trainer.project_name=auto_mapping_ablation
    trainer.n_gpus_per_node=8
    trainer.nnodes=1
    trainer.total_epochs=1
    trainer.save_freq=-1
    trainer.test_freq=-1
)

AUTO_ARGS=(
    +trainer.auto_mapping.enable=true
    +trainer.auto_mapping.per_gpu_budget_gb=80
    +trainer.auto_mapping.bandwidth.intra_host=600
    +trainer.auto_mapping.bandwidth.intra_block=25
    +trainer.auto_mapping.bandwidth.inter_block=12
)

run_one() {
    local NAME="$1"
    shift
    local LOG="${RUNS_DIR}/${NAME}.log"
    echo
    echo "=========================================================="
    echo "Running ${NAME}"
    echo "Log:    ${LOG}"
    echo "Started: $(date)"
    echo "=========================================================="
    local START=$(date +%s)
    (
        cd "${ROOT}" || exit 1
        # AUTO_MAPPING_SKIP_OVERRIDES is consumed by ray_config.apply_parallelism_overrides.
        # Set inline per-run via the env-var passed in by callers; default empty.
        env "${ENV_OVERRIDES[@]}" python -m verl.trainer.main_ppo \
            "${COMMON_ARGS[@]}" \
            trainer.experiment_name="${NAME}" \
            "$@"
    ) 2>&1 | tee "${LOG}"
    local END=$(date +%s)
    echo "${NAME} elapsed: $((END - START))s" | tee -a "${LOG}"
    echo "Finished: $(date)"
}

# -------------------- E1-default: no auto_mapping --------------------
ENV_OVERRIDES=()
run_one "E1-default" \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1

# -------------------- E1-auto: solver picks --------------------
ENV_OVERRIDES=()
run_one "E1-auto" \
    "${AUTO_ARGS[@]}"

# -------------------- E2-auto-tp1: split placement, TP=1 --------------------
# Solver still chooses placement (likely K=2 split). We override BOTH
# train-side and gen-side TP back to 1 by (a) supplying CLI values and
# (b) telling apply_parallelism_overrides to skip those paths so the CLI
# values survive. Critic same.
ENV_OVERRIDES=(
    "AUTO_MAPPING_SKIP_OVERRIDES=actor_rollout_ref.actor.megatron.tensor_model_parallel_size,actor_rollout_ref.rollout.tensor_model_parallel_size,critic.megatron.tensor_model_parallel_size"
)
run_one "E2-auto-tp1" \
    "${AUTO_ARGS[@]}" \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    critic.megatron.tensor_model_parallel_size=1

# -------------------- E2-auto-altgen: keep train TP=2, force gen TP=1 --------------------
# Solver picks (1,2,2) for both train and gen. We force gen TP=1 so train
# layout (1,2,2) ≠ gen layout (1,1,4) → ZeroRedundancyStrategy fires for
# real. Train TP override stays from the solver.
ENV_OVERRIDES=(
    "AUTO_MAPPING_SKIP_OVERRIDES=actor_rollout_ref.rollout.tensor_model_parallel_size"
)
run_one "E2-auto-altgen" \
    "${AUTO_ARGS[@]}" \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1

# -------------------- summary --------------------
SUMMARY="${RUNS_DIR}/summary.tsv"
echo "Writing summary to ${SUMMARY}"
{
    printf "run\tstep_time_s\tthroughput_tok_s\tactor_mfu\tcritic_mfu\tactor_infer_mfu\tcritic_score_mean\tactor_entropy\ttiming_gen\ttiming_old_log_prob\ttiming_values\ttiming_update_actor\ttiming_update_critic\ttiming_update_weights\n"
    for run in E1-default E1-auto E2-auto-tp1 E2-auto-altgen; do
        LOG="${RUNS_DIR}/${run}.log"
        if [[ ! -f "${LOG}" ]]; then
            printf "%s\t(no log)\n" "${run}"
            continue
        fi
        # Take the last "step:N - ..." line (final training step).
        LAST=$(grep -E "step:[0-9]+ -" "${LOG}" | tail -1)
        # Helper to grep one metric out of the last step line.
        x() { echo "${LAST}" | grep -oE "$1:[^ ]*" | tail -1 | cut -d: -f2; }
        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
            "${run}" \
            "$(x timing_s/step)" \
            "$(x perf/throughput)" \
            "$(x perf/mfu/actor)" \
            "$(x perf/mfu/critic)" \
            "$(x perf/mfu/actor_infer)" \
            "$(x critic/score/mean)" \
            "$(x actor/entropy)" \
            "$(x timing_s/gen)" \
            "$(x timing_s/old_log_prob)" \
            "$(x timing_s/values)" \
            "$(x timing_s/update_actor)" \
            "$(x timing_s/update_critic)" \
            "$(x timing_s/update_weights)"
    done
} | tee "${SUMMARY}"

echo
echo "Solver predictions (E1-auto run only — for E3a sim-vs-measured ratio):"
grep -E "^\(TaskRunner.*\) (\[solve\]|\[auto_mapping\]|Simulation result)" \
    "${RUNS_DIR}/E1-auto.log" \
    | tee "${RUNS_DIR}/E1-auto.solver.log" \
    || true

echo
echo "All runs complete. Summary file: ${SUMMARY}"
echo "Per-run logs in: ${RUNS_DIR}"
echo
echo "For the report, use:"
echo "  E1 = E1-default vs E1-auto                    (auto-mapping speedup)"
echo "  E2-tp = E1-auto vs E2-auto-tp1                (value of TP=2 over TP=1)"
echo "  E2-gen = E1-auto vs E2-auto-altgen            (value of train==gen choice)"
echo "  E3a   = solver-predicted ratios in E1-auto.solver.log vs measured"
echo "          step times in summary.tsv"

