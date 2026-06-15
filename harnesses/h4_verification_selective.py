"""H4 Verification-Selective Harness (readme §8 H4 + §11.4 verification mask).

与 H0 Raw 的核心差异 (prompt-side, 必须是 LLM 看得见的差异):
- observation 显式标注 "verification policy": 只有第 K 步 (最终步) 才跑 full
  verifier; 中间 step 写 "verification skipped (cost-deferred)"
- 中间 step 的反馈写为 "unverified — outcome not confirmed; treat as
  optimistic estimate"; agent 看到这种字样, belief 的 uncertainty 应当上升,
  task_progress 不应轻易升到 strong/complete (但实测里 LLM 可能就是会, 这是
  bias 来源)
- 在 step == K (最终步) 才返回 verifier_type=full + verified flag (基于 fake
  symbol: imagined rollout 里我们没真实 result, 用 LLM 自报的 success_prob 做
  proxy decision; 但下游 ds 只会消费 verifier_type / cost 字段, 不会读 verified)
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


_POLICY_TEXT = (
    "🧪 [VERIFICATION POLICY] 本环境采用 selective verification:\n"
    "  - 中间 step (step < K): 跳过 full verifier (成本高), 反馈视为 *unverified*\n"
    "  - 中间 step 只跑 weak signal check (syntax-level), 不保证语义正确\n"
    "  - 最终 step (step == K): 运行 full verifier (targeted_test + full_test)\n"
    "  - 你看到的中间步 success 反馈仅为 weak proxy, 真实 task_progress "
    "应保守估计."
)


class H4VerificationSelectiveHarness(Harness):
    harness_id = "H4_verification_selective"

    def make_observation(self, task: dict, step: int, history: list[dict]) -> Observation:
        raw = task.get("raw_observation", "")
        ver_target = task.get("verifier_hint", "n/a")
        last = history[-1] if history else {}
        last_act = last.get("selected_action") or "(none)"
        if step == 0:
            body = (
                f"{_POLICY_TEXT}\n\n"
                f"[step 0] Raw observation:\n{raw}\n"
                f"Verifier (will run at final step): {ver_target}"
            )
        else:
            verified_note = (
                "✅ [step K final] full verifier RAN: "
                f"`{ver_target}` (cost=high)" if step == _get_horizon(task) else
                "⚠️  verification skipped (cost-deferred). 这条反馈未经 verifier 确认; "
                "请保守估计 task_progress"
            )
            body = (
                f"{_POLICY_TEXT}\n\n"
                f"[step {step}] 上一步 action: {last_act}\n"
                f"反馈 (raw, imagined):\n{raw}\n"
                f"{verified_note}"
            )
        return Observation(
            raw_text=body,
            structured={
                "harness_view": "verification_selective",
                "verification_policy": {
                    "intermediate_verifier": "none_or_weak",
                    "final_verifier": "targeted_test+full_test",
                    "current_step_verified": step != 0 and step == _get_horizon(task),
                },
                "task_constraints": {
                    "target_state": task.get("target_state", ""),
                    "expected_failure_modes": task.get("expected_failure_modes", []),
                },
            },
            meta={
                "harness": "verification_selective",
                "step": step,
                "verification_mask_visible": True,
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
        # 中间步: cheap weak verifier 或 none; 最终步: full
        horizon = _get_horizon(task)
        if step == 0:
            return VerificationResult(verified=False, verifier_type="none", cost=0.0)
        if step == horizon:
            return VerificationResult(
                verified=True, verifier_type="full",
                cost=1.0,
                extras={"reason": "final-step full check"},
            )
        return VerificationResult(
            verified=False, verifier_type="cheap",
            cost=0.05,
            extras={"reason": "intermediate weak signal only"},
        )

    def attempt_repair(self, task, last_action, failure_info) -> RepairEvent:
        return RepairEvent(occurred=False)


def _get_horizon(task: dict) -> int:
    """从 task 提取当前 rollout horizon. 没有显式存 horizon, 用 sentinel 8.
    Day-2 rollout.py 不向 harness 透传 horizon, 这里只能用最大 K=8 作为 final-step
    判定的 conservative 上界 (因为只有 step==horizon 时才需要 'final'); 实际
    rollout 跑 K=3 时 step 永远不会到 8, full verifier 不触发 — 这是 H4
    设计的 cost-deferred 副作用 (短 horizon 永远 unverified)。"""
    return task.get("_rollout_horizon", 8)
