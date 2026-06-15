"""BIWM-4: Verification Mask (readme §11.4).

H4 selective verification 的 bias 来源: agent 不能区分 "已 verify success" 和
"未 verify, 当 success" — 容易过乐观. BIWM-4 在 observation 里**强制**显示每
个声称 success 的中间步的 verification status (verified/cheap-only/unverified)
+ verifier_type + cost, 让 LLM 必须把 "unverified" 当 uncertain 处理。
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


class VerificationMaskWrapper(Harness):
    def __init__(self, inner: Harness) -> None:
        self.inner = inner
        self.harness_id = f"BIWM4_{inner.harness_id}"
        # 内部状态: 每步的 verification 结果 (在 run_verifier 写, make_observation 读)
        self._last_verifier: dict[int, VerificationResult] = {}

    def make_observation(self, task: dict, step: int, history: list[dict]) -> Observation:
        inner_obs = self.inner.make_observation(task, step, history)
        if step == 0:
            return inner_obs
        # 把过往 verification 状态汇成一个 mask
        mask_lines = ["[VERIFICATION MASK]"]
        for h in history[-6:]:
            s = h.get("step", 0)
            v = self._last_verifier.get(s)
            if v is None:
                vlabel = "verified: UNKNOWN  verifier_type: unspecified  cost: 0"
            else:
                vlabel = (
                    f"verified: {v.verified}  verifier_type: {v.verifier_type}  "
                    f"cost: {v.cost}"
                )
            act = h.get("selected_action", "")[:60] if h.get("selected_action") else "(none)"
            mask_lines.append(f"  step {s}: action={act!r}  {vlabel}")
        mask_lines.append(
            "请把上述 verification status 纳入 belief: 对 verifier_type='none' 或 'cheap' 的步骤, "
            "不要把 success 当 verified — uncertainty 应当显著升高, task_progress 应保守."
        )
        mask_block = "\n".join(mask_lines)

        new_raw_text = f"{mask_block}\n\n{inner_obs.raw_text}"
        new_structured = dict(inner_obs.structured or {})
        new_structured["biwm4_verification_mask"] = {
            "n_steps_in_mask": min(len(history), 6),
            "policy": "treat unverified as uncertain",
        }
        return Observation(
            raw_text=new_raw_text,
            structured=new_structured,
            meta={
                **(inner_obs.meta or {}),
                "biwm": "verification_mask",
                "step": step,
            },
        )

    def gate_action(self, task, candidate_action, all_candidates) -> ActionDecision:
        return self.inner.gate_action(task, candidate_action, all_candidates)

    def run_verifier(self, task, step, action) -> VerificationResult:
        result = self.inner.run_verifier(task, step, action)
        self._last_verifier[step] = result
        return result

    def attempt_repair(self, task, last_action, failure_info) -> RepairEvent:
        return self.inner.attempt_repair(task, last_action, failure_info)

    def metadata(self) -> dict[str, Any]:
        return {
            "harness_id": self.harness_id,
            "underlying_harness": self.inner.harness_id,
            "biwm_components": ["verification_mask"],
        }
