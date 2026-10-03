from __future__ import annotations

from typing import Any

from ..core.harness_base import (
    ActionDecision,
    Harness,
    Observation,
    RepairEvent,
    VerificationResult,
)


_REPAIR_NARRATIVE = (
    "🔁 [auto-repair active] 本环境会自动 retry 失败的 action, "
    "并在失败时 rollback 到上一个 checkpoint, refresh 工具状态, "
    "再 reissue action. 你在 history 里看到的'成功'结果可能是 1-3 次自动重试后达成的; "
    "本 harness 默认不暴露每次 retry 细节, 只展示净结果。"
)


class H3RepairHeavyHarness(Harness):
    harness_id = "H3_repair_heavy"

    def make_observation(self, task: dict, step: int, history: list[dict]) -> Observation:
        raw = task.get("raw_observation", "")
        if step == 0:
            body = (
                f"{_REPAIR_NARRATIVE}\n\n"
                f"[step 0] Raw observation:\n{raw}\n"
                f"(verifier_hint: {task.get('verifier_hint','n/a')})"
            )
        else:
            collapsed = []
            for i, h in enumerate(history[-5:]):
                act = h.get("selected_action") or "(noop)"
                collapsed.append(
                    f"  step {h.get('step', i)}: action={act!r} → "
                    f"succeeded (auto-repaired, 1 retry hidden)"
                )
            body = (
                f"{_REPAIR_NARRATIVE}\n\n"
                f"[step {step}] history (collapsed view):\n"
                + "\n".join(collapsed)
                + f"\n\n当前环境反馈 (raw, 已 repair):\n{raw}\n"
                "(imagined rollout; 注意 history 中的成功可能掩盖了底层失败)"
            )
        return Observation(
            raw_text=body,
            structured={
                "harness_view": "repair_heavy",
                "repair_policy": {
                    "auto_retry": True,
                    "auto_rollback": True,
                    "max_retries": 3,
                    "expose_failure_details": False,
                },
                "collapsed_history_length": min(len(history), 5),
                "task_constraints": {
                    "target_state": task.get("target_state", ""),
                    "expected_failure_modes": task.get("expected_failure_modes", []),
                },
            },
            meta={
                "harness": "repair_heavy",
                "step": step,
                "history_collapsed": True,
                "narrative_active": True,
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
        return VerificationResult(verified=False, verifier_type="none", cost=0.0)

    def attempt_repair(self, task, last_action, failure_info) -> RepairEvent:
        if last_action is None:
            return RepairEvent(occurred=False)
        return RepairEvent(
            occurred=True,
            repair_action="auto_retry+rollback",
            extras={
                "hidden_from_agent": True,
                "retry_count_simulated": 1,
                "rollback_to_checkpoint": "auto",
            },
        )
