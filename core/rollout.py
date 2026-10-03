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
        "action_history": action_history[-5:],
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
    task = dict(task)
    task["_rollout_horizon"] = horizon
    task["_run_id"] = run_id

    for step in range(horizon + 1):
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
            step_seed = (seed * 1000 + step) if seed is not None else None
            belief_obj, stats = llm.chat_json(prompt, max_tokens=1200, seed=step_seed)
            stats_dump = asdict(stats)
            stats_dump.pop("raw_response", None)
            summary["llm_calls"] += 1
            summary["total_tokens"] += stats.total_tokens
            summary["total_latency_s"] += stats.latency_s
        except Exception as e:
            llm_err = f"{type(e).__name__}: {e}"
            belief_obj = empty_belief_output(horizon=horizon)
            schema_fail = True

        b_errs = validate_belief(belief_obj)
        if b_errs:
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

        rec_action = belief_obj["next_action_recommendation"]["action"]
        candidate_actions = [rec_action]
        decision = harness.gate_action(task, rec_action, candidate_actions)

        ver = harness.run_verifier(task, step, decision.selected_action)

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

        log_errs = validate_step_log(step_record)
        if log_errs:
            summary["step_log_validation_errors"].append({"step": step, "errs": log_errs[:5]})

        logger.write(step_record)
        summary["steps_written"] += 1

        action_history.append(
            {"step": step, "selected_action": decision.selected_action, "blocked": decision.blocked_actions}
        )
        prev_belief = belief_obj["belief_state"]

    summary["finished_jst"] = now_jst_iso()
    return summary


def _patch_belief(obj: Any, horizon: int) -> dict:
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
