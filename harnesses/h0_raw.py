"""H0 Raw Harness (readme §8 H0).

特征:
- 尽量完整执行动作 (Day-1 不接环境, 不真实执行; 但 observation 不裁剪)
- 完整记录 observation / action / result (raw, 不结构化)
- 不阻止任何动作 (gate 永远放行)
- 不强制 verifier (verifier_type = "none", Day-1)
- 不做 repair
- log 完整透传

这是近似 reference harness。与 H1 (Structured) 的区别: observation 是 raw
text, 不暴露 traceback parser / call graph / structured fields。
"""
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
        # raw obs = task 描述的 raw_observation 字段 + 上一步选的 action 文本
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
            structured={},  # H0 不暴露结构化
            meta={"harness": "raw", "step": step},
        )

    def gate_action(
        self, task: dict, candidate_action: str, all_candidates: list[str]
    ) -> ActionDecision:
        # 全放行
        return ActionDecision(
            selected_action=candidate_action,
            candidate_actions=all_candidates,
            blocked_actions=[],
            blocking_reasons=[],
        )

    def run_verifier(self, task: dict, step: int, action) -> VerificationResult:
        # Day-1 H0 不跑 verifier
        return VerificationResult(verified=False, verifier_type="none", cost=0.0)

    def attempt_repair(self, task, last_action, failure_info) -> RepairEvent:
        return RepairEvent(occurred=False)
