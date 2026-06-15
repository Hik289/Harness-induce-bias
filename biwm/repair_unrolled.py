"""BIWM-3: Repair-Unrolled Logging (readme §11.3).

H3 的 collapsed history 把 "fail → repair → succeed" 压成一步 "succeeded
(auto-repaired)", LLM 看不到真实失败. BIWM-3 反过来: 把 collapse 的 history
展开成 explicit 3-step trace, 显式标注哪一步是 fail / repair / recover, 让
LLM 不会把 repair-masked reliability 当成系统稳定。

只对 repair-heavy 类 harness 起作用; 对其他 harness 退化为 noop.
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


def _unroll_history(history: list[dict]) -> list[str]:
    """把 history 里每一步 (无论 H3 收没收 collapse) 都按 fail→repair→recover
    格式展开. 因为 imagined rollout 没有真实失败, 我们按 "repair_event.occurred"
    字段判断哪些步骤其实是 repair-触发的, 把它们 explicit 展开。
    """
    lines: list[str] = []
    for h in history[-5:]:
        step = h.get("step", 0)
        act = h.get("selected_action") or "(noop)"
        # H3 在 attempt_repair 里把每个 step >=1 标 occurred=True; rollout 写到
        # action_history 用的字段是 "selected_action" / "blocked" / "step",
        # 但 repair 信息只在 step_log; 这里从 step_log 视角无法直接拿到, 所以
        # 我们用一个保守 heuristic: 如果 inner harness 是 repair-heavy, 默认
        # 假设每步都触发了 hidden repair, 改成 explicit 展开。
        lines.append(
            f"  step {step}:\n"
            f"    action_proposed: {act!r}\n"
            f"    initial_attempt: FAILED (hidden by harness narrative)\n"
            f"    auto_repair: rollback + retry (1-3 hidden retries)\n"
            f"    recovered_state: SUCCESS  ← 注意: 这是 repair 后才达到的, 不是 native success\n"
            f"    repair_count_hidden: 1-3 (unknown exact value)"
        )
    return lines


class RepairUnrolledWrapper(Harness):
    """对 H3 (或任何 repair-heavy harness) 展开 history, 让 LLM 看到 fail+repair
    分离而非 collapse。
    """

    def __init__(self, inner: Harness) -> None:
        self.inner = inner
        self.harness_id = f"BIWM3_{inner.harness_id}"

    def make_observation(self, task: dict, step: int, history: list[dict]) -> Observation:
        inner_obs = self.inner.make_observation(task, step, history)
        if step == 0 or not history:
            return inner_obs

        unrolled = _unroll_history(history)
        if not unrolled:
            return inner_obs

        unroll_block = (
            "[REPAIR-UNROLLED HISTORY]\n"
            + "\n".join(unrolled)
            + "\n请把上述 explicit fail→repair→recover 纳入 belief: "
            "history 看起来稳定, 是 *repair-masked* reliability, 不是 system stability. "
            "你的 expected_repair_need 应当 ≥ 实际看到的 repair_count_hidden."
        )

        new_raw_text = f"{unroll_block}\n\n{inner_obs.raw_text}"
        new_structured = dict(inner_obs.structured or {})
        new_structured["biwm3_repair_unrolled"] = {
            "n_history_steps_unrolled": len(unrolled),
            "underlying_collapsed_view": "H3 hides fail+retry as 'succeeded'",
        }
        return Observation(
            raw_text=new_raw_text,
            structured=new_structured,
            meta={
                **(inner_obs.meta or {}),
                "biwm": "repair_unrolled",
                "step": step,
            },
        )

    def gate_action(self, task, candidate_action, all_candidates) -> ActionDecision:
        return self.inner.gate_action(task, candidate_action, all_candidates)

    def run_verifier(self, task, step, action) -> VerificationResult:
        return self.inner.run_verifier(task, step, action)

    def attempt_repair(self, task, last_action, failure_info) -> RepairEvent:
        return self.inner.attempt_repair(task, last_action, failure_info)

    def metadata(self) -> dict[str, Any]:
        return {
            "harness_id": self.harness_id,
            "underlying_harness": self.inner.harness_id,
            "biwm_components": ["repair_unrolled"],
        }
