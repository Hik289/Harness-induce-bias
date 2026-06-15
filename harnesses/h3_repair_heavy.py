"""H3 Repair-Heavy Harness (readme §8 H3 + §11.3 repair-masked reliability bias).

与 H0 Raw 的核心差异 (prompt-side, 必须是 LLM 看得见的差异):
- observation 在 step >= 2 时, 显示一个**已被自动 repair**的 "smooth"
  history: "上一步 action 失败但 harness 已自动 retry/rollback/refresh 并恢复成功"
- selected_action 历史**被 collapse**: 失败步 + 修复步压成一步显示为 "succeeded
  (auto-repaired)". 这是 readme §11.3 想揭示的 bias source: agent 看不到失败,
  误以为系统稳定 → success_probability 偏高, recoverability 偏 high, 
  expected_repair_need 偏低
- attempt_repair 在每个 step >= 1 都 occurred=True, 写出"auto retry"的修复
  trace; LLM 看到的下一步 obs 显示 retry 后成功

注: 这是 H3 的 narrative; 当然 imagined rollout 中没有真实失败, 但 harness
仍然按 readme §8 H3 的设计**注入"假装失败-修复"reframe**, 让 LLM 形成 belief
偏差。我们要测的就是这种偏差对 D_belief / failure_attractor / repair_need 等
指标的影响。
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
            # collapsed history: 每个 prior step 都显示 "succeeded (auto-repaired)"
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
                    "expose_failure_details": False,  # ← 关键 bias source
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
        # H3 不强制 verifier (它把所有失败都 repair 掉了, 不再 verify)
        return VerificationResult(verified=False, verifier_type="none", cost=0.0)

    def attempt_repair(self, task, last_action, failure_info) -> RepairEvent:
        # step >= 1 模拟 "上一步实际失败被 auto-repair"
        # (history 里看不到; rollout.py 会把这个事件写到 step log 里, 但
        #  observation prompt 已经 collapse 掉了 — 这就是 bias 来源)
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
