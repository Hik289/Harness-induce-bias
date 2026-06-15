"""Belief / log JSON schema (readme §9.1 + §15).

为下游 data_scientist 的 D_belief / ECE / AUROC 模块提供稳定 schema。
本 schema 用 jsonschema Draft-2020-12 表达, validator 暴露 validate_belief() /
validate_step_log()。

设计选择:
- belief_state 的枚举值严格对齐 readme §9.1 (task_progress / risk_state /
  recoverability / likely_failure_mode), data_scientist 可直接按 categorical
  mismatch 算 D_belief
- predicted_future 中所有概率 0..1, uncertainty 0..1
- 允许 belief_state.extras / predicted_future.extras 字段, harness 可附加未来
  分析需要的额外信息, 不影响 schema 校验
"""
from __future__ import annotations

from typing import Any
import jsonschema


# -- belief_state ---------------------------------------------------
TASK_PROGRESS_ENUM = ["none", "weak", "partial", "strong", "complete"]
RISK_STATE_ENUM = ["low", "medium", "high"]
RECOVERABILITY_ENUM = ["high", "medium", "low"]
FAILURE_MODE_ENUM = [
    "none",
    "search_loop",
    "test_loop",
    "wrong_file_patch",
    "retry_loop",
    "policy_violation",
    "destructive_action",
    "form_loop",
]

BELIEF_STATE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": [
        "task_progress",
        "known_constraints",
        "satisfied_constraints",
        "violated_constraints",
        "risk_state",
        "recoverability",
        "uncertainty",
        "likely_failure_mode",
    ],
    "properties": {
        "task_progress": {"type": "string", "enum": TASK_PROGRESS_ENUM},
        "known_constraints": {"type": "array", "items": {"type": "string"}},
        "satisfied_constraints": {"type": "array", "items": {"type": "string"}},
        "violated_constraints": {"type": "array", "items": {"type": "string"}},
        "risk_state": {"type": "string", "enum": RISK_STATE_ENUM},
        "recoverability": {"type": "string", "enum": RECOVERABILITY_ENUM},
        "uncertainty": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "likely_failure_mode": {"type": "string", "enum": FAILURE_MODE_ENUM},
        "extras": {"type": "object"},
    },
    "additionalProperties": True,
}

# -- predicted_future -----------------------------------------------
PREDICTED_FUTURE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": [
        "horizon",
        "success_probability",
        "failure_attractor_probability",
        "risk_accumulation",
        "expected_cost",
        "expected_repair_need",
    ],
    "properties": {
        "horizon": {"type": "integer", "minimum": 0},
        "success_probability": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "failure_attractor_probability": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
        },
        "risk_accumulation": {"type": "number", "minimum": 0.0},
        "expected_cost": {"type": "number", "minimum": 0.0},
        "expected_repair_need": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "extras": {"type": "object"},
    },
    "additionalProperties": True,
}

# -- next_action_recommendation -------------------------------------
NEXT_ACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["action", "reason", "verification_target"],
    "properties": {
        "action": {"type": "string"},
        "reason": {"type": "string"},
        "verification_target": {"type": "string"},
    },
    "additionalProperties": True,
}

# -- top-level belief output ----------------------------------------
BELIEF_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["belief_state", "predicted_future", "next_action_recommendation"],
    "properties": {
        "belief_state": BELIEF_STATE_SCHEMA,
        "predicted_future": PREDICTED_FUTURE_SCHEMA,
        "next_action_recommendation": NEXT_ACTION_SCHEMA,
    },
    "additionalProperties": True,
}

# -- step log (readme §15) -------------------------------------------
STEP_LOG_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": [
        "task_id",
        "benchmark",
        "environment_id",
        "harness_id",
        "base_llm",
        "rollout_horizon",
        "step",
        "observation",
        "canonical_belief_input",
        "belief_output",
        "candidate_actions",
        "selected_action",
        "blocked_actions",
        "blocking_reasons",
        "verification_mask",
        "repair_event",
        "shadow_execution",
        "downstream_result",
    ],
    "properties": {
        "task_id": {"type": "string"},
        "benchmark": {"type": "string"},
        "environment_id": {"type": "string"},
        "harness_id": {"type": "string"},
        "base_llm": {"type": "string"},
        "rollout_horizon": {"type": "integer", "minimum": 1},
        "step": {"type": "integer", "minimum": 0},
        "observation": {"type": ["string", "object", "null"]},
        "canonical_belief_input": {"type": "object"},
        "belief_output": BELIEF_OUTPUT_SCHEMA,
        "candidate_actions": {"type": "array"},
        "selected_action": {"type": ["string", "object", "null"]},
        "blocked_actions": {"type": "array"},
        "blocking_reasons": {"type": "array"},
        "verification_mask": {
            "type": "object",
            "required": ["verified", "verifier_type", "cost"],
            "properties": {
                "verified": {"type": "boolean"},
                "verifier_type": {"type": "string"},
                "cost": {"type": "number", "minimum": 0.0},
            },
            "additionalProperties": True,
        },
        "repair_event": {
            "type": "object",
            "required": ["occurred"],
            "properties": {
                "occurred": {"type": "boolean"},
                "repair_action": {"type": ["string", "object", "null"]},
            },
            "additionalProperties": True,
        },
        "shadow_execution": {"type": "object"},
        "downstream_result": {"type": "object"},
        # 实际跑会附加的字段
        "timestamp_jst": {"type": "string"},
        "run_id": {"type": "string"},
        "llm_stats": {"type": "object"},
        "seed": {"type": "integer"},
    },
    "additionalProperties": True,
}


_belief_validator = jsonschema.Draft202012Validator(BELIEF_OUTPUT_SCHEMA)
_step_validator = jsonschema.Draft202012Validator(STEP_LOG_SCHEMA)


def validate_belief(obj: Any) -> list[str]:
    """返回 error message 列表; 空表示通过."""
    return [f"{'/'.join(map(str, e.path))}: {e.message}" for e in _belief_validator.iter_errors(obj)]


def validate_step_log(obj: Any) -> list[str]:
    return [f"{'/'.join(map(str, e.path))}: {e.message}" for e in _step_validator.iter_errors(obj)]


def empty_belief_output(horizon: int = 0) -> dict:
    """生成一个 schema-valid 的空 belief (供 fallback / 测试用)."""
    return {
        "belief_state": {
            "task_progress": "none",
            "known_constraints": [],
            "satisfied_constraints": [],
            "violated_constraints": [],
            "risk_state": "medium",
            "recoverability": "medium",
            "uncertainty": 1.0,
            "likely_failure_mode": "none",
        },
        "predicted_future": {
            "horizon": horizon,
            "success_probability": 0.5,
            "failure_attractor_probability": 0.5,
            "risk_accumulation": 0.0,
            "expected_cost": 0.0,
            "expected_repair_need": 0.0,
        },
        "next_action_recommendation": {
            "action": "noop",
            "reason": "fallback",
            "verification_target": "none",
        },
    }
