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
    lines: list[str] = []
    for h in history[-5:]:
        step = h.get("step", 0)
        act = h.get("selected_action") or "(noop)"
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
