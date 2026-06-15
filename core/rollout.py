"""K-step LLM belief rollout (readme §9.1 / §9.2 / §2.2).

核心实现 (Day-1 SETUP_DAY1, 所有 harness 共用):
  step 0: 把 task instruction + harness 给的 observation 喂给 LLM → belief_0
  step t: belief_{t-1} + (假想/真实) action_{t-1} + step-t observation → belief_t
  ...
  step K: belief_K

每一步都是一次独立 LLM call (绝不把整个 rollout 塞一个 prompt)。
每一步输出严格 JSON, 通过 BELIEF_OUTPUT_SCHEMA 校验; 校验失败 → 客户端 retry
(最多 2 次), 全失败 → 用 empty_belief_output() fallback 且写 schema_fail=True 到
log, 不中断 rollout (这样 anchor_2 schema pass rate 才能被准确度量)。

注意:
- 这是 imaginary rollout: agent 在 step t 选了一个 candidate action, world
  model rollout 在不真正执行 action 的前提下 imagine belief_{t+1}; 这是
  multi-step LLM world model 的标准做法 (readme §9.2 "predicted belief_{t+1}")
- 这区别于 H0 真实环境 step 推进: Day-1 暂不接环境, observation 在 step t≥1
  时 = "agent 假设刚才选了 action X, harness 假想反馈" (由 harness 决定)
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict
from typing import Any, Optional

from .belief_schema import (
    BELIEF_OUTPUT_SCHEMA,
    empty_belief_output,
    validate_belief,
    validate_step_log,
)
from .harness_base import Harness, Observation
from .jsonl_logger import JSONLLogger, now_jst_iso
from .llm_client import LLMClient


SYSTEM_PROMPT = """你是一个软件 agent 的 multi-step LLM world model。
你的任务: 阅读当前观测 + 历史, 输出当前 belief_state, 预测未来 horizon 步,
并推荐 next action。

你必须只输出一个合法的 JSON 对象, 严格匹配以下 schema:
{
  "belief_state": {
    "task_progress": "none|weak|partial|strong|complete",
    "known_constraints": [string],
    "satisfied_constraints": [string],
    "violated_constraints": [string],
    "risk_state": "low|medium|high",
    "recoverability": "high|medium|low",
    "uncertainty": float in [0,1],
    "likely_failure_mode": "none|search_loop|test_loop|wrong_file_patch|retry_loop|policy_violation|destructive_action|form_loop"
  },
  "predicted_future": {
    "horizon": int,
    "success_probability": float in [0,1],
    "failure_attractor_probability": float in [0,1],
    "risk_accumulation": float >= 0,
    "expected_cost": float >= 0,
    "expected_repair_need": float in [0,1]
  },
  "next_action_recommendation": {
    "action": string,
    "reason": string,
    "verification_target": string
  }
}

不要输出任何 markdown、解释、注释、代码块。只输出 JSON 对象本身。"""


def _build_step_prompt(
    task: dict,
    obs: Observation,
    prev_belief: Optional[dict],
    action_history: list[dict],
    step: int,
    horizon: int,
    harness_meta: dict,
) -> list[dict[str, str]]:
    """每一步独立调用; prompt 包含 task / 当前 obs / 上一步 belief / 历史."""
    user_payload = {
        "task_id": task["task_id"],
        "task_instruction": task["instruction"],
        "harness_metadata": harness_meta,
        "current_step": step,
        "rollout_horizon": horizon,
        "current_observation": {
            "raw_text": obs.raw_text,
            "structured": obs.structured,
            "harness_meta": obs.meta,
        },
        "previous_belief_state": prev_belief,
        "action_history": action_history[-5:],  # 最近 5 步避免 prompt 爆
    }
    user_msg = (
        f"## Step {step} of {horizon}\n\n"
        f"以下是当前 multi-step world model rollout 的输入 (JSON):\n```json\n"
        f"{json.dumps(user_payload, ensure_ascii=False, indent=2)}\n```\n\n"
        "请输出当前 belief_state、predicted_future、next_action_recommendation。"
        "严格 JSON, 无 markdown。"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]


def run_kstep_rollout(
    *,
    task: dict,
    harness: Harness,
    llm: LLMClient,
    horizon: int,
    logger: JSONLLogger,
    benchmark_id: str = "HIBench-Code",
    environment_id: str = "E_default_v0",
    seed: int = 0,
    run_id: Optional[str] = None,
) -> dict[str, Any]:
    """跑一条 K-step rollout, 写 K+1 条 step log; 返回 summary."""
    run_id = run_id or f"{harness.harness_id}_{task['task_id']}_K{horizon}_seed{seed}_{uuid.uuid4().hex[:6]}"
    summary: dict[str, Any] = {
        "run_id": run_id,
        "task_id": task["task_id"],
        "harness_id": harness.harness_id,
        "horizon": horizon,
        "seed": seed,
        "started_jst": now_jst_iso(),
        "steps_written": 0,
        "schema_pass": 0,
        "schema_fail": 0,
        "llm_calls": 0,
        "total_tokens": 0,
        "total_latency_s": 0.0,
        "step_log_validation_errors": [],
    }

    prev_belief: Optional[dict] = None
    action_history: list[dict] = []
    # Inject horizon + run-level metadata into task dict so harnesses (e.g. H4)
    # can adapt selective-verification policy to the actual K. Non-mutating
    # contract: we set on a per-run shallow copy to avoid side effects across
    # K calls when caller reuses the task object.
    task = dict(task)
    task["_rollout_horizon"] = horizon
    task["_run_id"] = run_id

    for step in range(horizon + 1):  # 0..K inclusive
        observation = harness.make_observation(task, step, action_history)

        prompt = _build_step_prompt(
            task=task,
            obs=observation,
            prev_belief=prev_belief,
            action_history=action_history,
            step=step,
            horizon=horizon,
            harness_meta=harness.metadata(),
        )

        belief_obj: dict
        schema_fail = False
        llm_err: Optional[str] = None
        stats_dump: dict = {}
        try:
            # Per-step seed = base seed + step idx, makes the 3-seed Phase 1
            # design produce distinct call streams (Azure 不强制 determinism
            # but it changes the request body so prompt-cache hashes vary)
            step_seed = (seed * 1000 + step) if seed is not None else None
            belief_obj, stats = llm.chat_json(prompt, max_tokens=1200, seed=step_seed)
            stats_dump = asdict(stats)
            stats_dump.pop("raw_response", None)  # 避免日志爆量
            summary["llm_calls"] += 1
            summary["total_tokens"] += stats.total_tokens
            summary["total_latency_s"] += stats.latency_s
        except Exception as e:  # noqa: BLE001
            llm_err = f"{type(e).__name__}: {e}"
            belief_obj = empty_belief_output(horizon=horizon)
            schema_fail = True

        # belief schema check
        b_errs = validate_belief(belief_obj)
        if b_errs:
            # 一次硬修复: 缺字段补 default, 再校验; 若仍不过, 用 fallback
            patched = _patch_belief(belief_obj, horizon)
            if not validate_belief(patched):
                belief_obj = patched
            else:
                belief_obj = empty_belief_output(horizon=horizon)
                schema_fail = True

        if schema_fail or b_errs:
            summary["schema_fail"] += 1
        else:
            summary["schema_pass"] += 1

        # harness gate: 拿 belief 里推荐的 action, 跑 candidate generation
        rec_action = belief_obj["next_action_recommendation"]["action"]
        candidate_actions = [rec_action]  # Day-1 简化: 单候选; Day-2 起 H6 会
                                          # rollout 3 candidate path
        decision = harness.gate_action(task, rec_action, candidate_actions)

        # verifier (harness 决定是否运行)
        ver = harness.run_verifier(task, step, decision.selected_action)

        # repair: Day-1 不触发, 留给 H3
        repair = harness.attempt_repair(task, decision.selected_action, {})

        step_record = {
            "task_id": task["task_id"],
            "benchmark": benchmark_id,
            "environment_id": environment_id,
            "harness_id": harness.harness_id,
            "base_llm": "gpt-5.4-mini",
            "rollout_horizon": horizon,
            "step": step,
            "observation": observation.raw_text,
            "canonical_belief_input": {
                "structured_observation": observation.structured,
                "harness_observation_meta": observation.meta,
                "previous_belief": prev_belief,
                "action_history_tail": action_history[-5:],
            },
            "belief_output": belief_obj,
            "candidate_actions": decision.candidate_actions,
            "selected_action": decision.selected_action,
            "blocked_actions": decision.blocked_actions,
            "blocking_reasons": decision.blocking_reasons,
            "verification_mask": {
                "verified": ver.verified,
                "verifier_type": ver.verifier_type,
                "cost": ver.cost,
                "extras": ver.extras,
            },
            "repair_event": {
                "occurred": repair.occurred,
                "repair_action": repair.repair_action,
                "extras": repair.extras,
            },
            "shadow_execution": {},
            "downstream_result": {},
            "timestamp_jst": now_jst_iso(),
            "run_id": run_id,
            "llm_stats": stats_dump,
            "seed": seed,
            "schema_fail": schema_fail,
            "llm_error": llm_err,
        }
        step_record = harness.filter_log(step_record)

        # validate full step log; collect errors but don't crash
        log_errs = validate_step_log(step_record)
        if log_errs:
            summary["step_log_validation_errors"].append({"step": step, "errs": log_errs[:5]})

        logger.write(step_record)
        summary["steps_written"] += 1

        # bookkeeping
        action_history.append(
            {"step": step, "selected_action": decision.selected_action, "blocked": decision.blocked_actions}
        )
        prev_belief = belief_obj["belief_state"]

    summary["finished_jst"] = now_jst_iso()
    return summary


def _patch_belief(obj: Any, horizon: int) -> dict:
    """尽力补齐 belief 缺字段; 不改 LLM 已填的合理值."""
    template = empty_belief_output(horizon=horizon)
    if not isinstance(obj, dict):
        return template
    out = template.copy()
    for top_key in ("belief_state", "predicted_future", "next_action_recommendation"):
        sub = obj.get(top_key)
        if isinstance(sub, dict):
            merged = template[top_key].copy()
            merged.update(sub)
            out[top_key] = merged
    return out
