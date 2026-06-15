"""H5 Cost-Aware Harness (readme §8 H5 + §10.1 H0 vs H5 expected K=5 high divergence).

与 H0 Raw 的核心差异 (prompt-side, 必须是 LLM 看得见的差异):
- observation 被**截断**到 cost budget: 长输出 (test log 等) 被 summary 替换;
  raw_text 上面附 "[BUDGET=low] omitted ~N chars" 标注; LLM 看见 budget 标注
  应当主动节省后续 verifier 调用, 但实测往往会让 belief 过自信 (因为关键
  细节被砍掉)
- 显式 metadata: "verifier 因 cost 跳过", "expensive observation 已截断", 
  "重型 LLM review 已禁用"
- verifier 几乎不跑 (random 1/3 概率跑 cheap)
"""
from __future__ import annotations

import hashlib
from typing import Any

from ..core.harness_base import (
    ActionDecision,
    Harness,
    Observation,
    RepairEvent,
    VerificationResult,
)


_BUDGET_HEADER = (
    "💸 [BUDGET=low] 本环境运行在 cost-aware harness:\n"
    "  - observation 长输出会被截断为 summary (节省 LLM context)\n"
    "  - verifier 大多数 step 被跳过 (节省 test 运行 cost)\n"
    "  - expensive API call / strong-model review 禁用\n"
    "  - 你应在低成本前提下规划 next action; 但请注意: 关键细节可能被截断, "
    "  请用低置信度估计 task_progress."
)

_OBS_CHAR_BUDGET = 200  # 字符上限


def _truncate(text: str, n: int) -> tuple[str, int]:
    if len(text) <= n:
        return text, 0
    keep = text[:n]
    omitted = len(text) - n
    return keep + f"\n…[{omitted} chars truncated]…", omitted


def _step_runs_verifier(step: int, task: dict) -> bool:
    """每 3 步跑 1 次 cheap verifier; 用 task_id+step 决定 (deterministic)."""
    key = (task.get("task_id", "") + f"_step{step}").encode("utf-8")
    h = int.from_bytes(hashlib.sha1(key).digest()[:4], "big")
    return (h % 3) == 0


class H5CostAwareHarness(Harness):
    harness_id = "H5_cost_aware"

    def make_observation(self, task: dict, step: int, history: list[dict]) -> Observation:
        raw = task.get("raw_observation", "")
        truncated, omitted = _truncate(raw, _OBS_CHAR_BUDGET)
        last = history[-1] if history else {}
        last_act = last.get("selected_action") or "(none)"
        if step == 0:
            body = (
                f"{_BUDGET_HEADER}\n\n"
                f"[step 0] truncated observation:\n{truncated}\n"
                f"(verifier_hint: {task.get('verifier_hint','n/a')})"
            )
        else:
            body = (
                f"{_BUDGET_HEADER}\n\n"
                f"[step {step}] 上一步 action: {last_act}\n"
                f"truncated 反馈:\n{truncated}\n"
                f"(imagined; budget=low, 关键细节可能被截断)"
            )
        return Observation(
            raw_text=body,
            structured={
                "harness_view": "cost_aware",
                "budget": "low",
                "observation_truncation": {
                    "char_budget": _OBS_CHAR_BUDGET,
                    "omitted_chars": omitted,
                    "summary_only": omitted > 0,
                },
                "verifier_policy": "skip_most_steps",
                "task_constraints": {
                    "target_state": task.get("target_state", ""),
                    "expected_failure_modes": task.get("expected_failure_modes", []),
                },
            },
            meta={
                "harness": "cost_aware",
                "step": step,
                "budget_active": True,
                "truncation_omitted_chars": omitted,
            },
        )

    def gate_action(self, task, candidate_action, all_candidates) -> ActionDecision:
        return ActionDecision(
            selected_action=candidate_action,
            candidate_actions=all_candidates,
            blocked_actions=[],
            blocking_reasons=[],
        )

    def run_verifier(self, task, step, action) -> VerificationResult:
        # 不跑 verifier 除非 deterministic hash 命中
        if _step_runs_verifier(step, task):
            return VerificationResult(
                verified=False, verifier_type="cheap",
                cost=0.05,
                extras={"reason": "budget-allowed cheap check"},
            )
        return VerificationResult(
            verified=False, verifier_type="none", cost=0.0,
            extras={"reason": "skipped to save cost"},
        )

    def attempt_repair(self, task, last_action, failure_info) -> RepairEvent:
        return RepairEvent(occurred=False)
