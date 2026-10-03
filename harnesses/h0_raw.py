from __future__ import annotations

from typing import Any

from ..core.harness_base import (
    ActionDecision,
    Harness,
    Observation,
    RepairEvent,
    VerificationResult,
)


class H0RawHarness(Harness):
    harness_id = "H0_raw"

    def make_observation(self, task: dict, step: int, history: list[dict]) -> Observation:
        raw = task.get("raw_observation", "")
        if step == 0:
            raw_text = f"[step 0] 你刚拿到任务。Raw observation:\n{raw}"
        else:
            last = history[-1] if history else {}
            last_act = last.get("selected_action") or "(none)"
            raw_text = (
                f"[step {step}] 上一步你选择了 action: {last_act}\n"
                f"环境反馈 (raw, 未结构化):\n{raw}\n"
                f"(注: Day-1 rollout 不执行真实动作, 这里 obs 与 step 0 相同; "
                f"这是 imagined rollout, 你需要 imagine 选择 action 后的 belief 更新)"
            )
        return Observation(
            raw_text=raw_text,
            structured={},
            meta={"harness": "raw", "step": step},
        )

    def gate_action(
        self, task: dict, candidate_action: str, all_candidates: list[str]
    ) -> ActionDecision:
        return ActionDecision(
            selected_action=candidate_action,
            candidate_actions=all_candidates,
            blocked_actions=[],
            blocking_reasons=[],
        )

    def run_verifier(self, task: dict, step: int, action) -> VerificationResult:
        return VerificationResult(verified=False, verifier_type="none", cost=0.0)

    def attempt_repair(self, task, last_action, failure_info) -> RepairEvent:
        return RepairEvent(occurred=False)
