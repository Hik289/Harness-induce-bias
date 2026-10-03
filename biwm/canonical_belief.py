from __future__ import annotations

import re
from typing import Any

from ..core.harness_base import (
    ActionDecision,
    Harness,
    Observation,
    RepairEvent,
    VerificationResult,
)


_DECORATION_PATTERNS = (
    r"\[POLICY\][^\n]*",
    r"\[BUDGET=[^\]]+\][^\n]*",
    r"\[VERIFICATION POLICY\][^\n]*",
    r"\[auto-repair active\][^\n]*",
    r"💸[^\n]*",
    r"🧪[^\n]*",
    r"🔁[^\n]*",
    r"⚠️[^\n]*",
    r"⛔[^\n]*",
    r"✅[^\n]*",
    r"^\s*-\s+\w[^\n]*$",
)


def _strip_decorations(text: str) -> str:
    for pat in _DECORATION_PATTERNS:
        text = re.sub(pat, "", text, flags=re.MULTILINE)
    text = re.sub(r"\n\s*\n+", "\n\n", text).strip()
    return text


def _extract_canonical_fields(text: str, task: dict) -> dict[str, Any]:
    failing = re.findall(r"(?:File\s+([\w./_-]+),\s*line\s*\d+|([\w/]+\.py))", text)
    files = list({a or b for a, b in failing if (a or b)})[:5]
    excs = list(set(re.findall(r"\b([A-Z][A-Za-z]+(?:Error|Exception))\b", text)))[:5]
    pytest_lines = list(set(re.findall(r"pytest[^\n]*", text)))[:3]
    return {
        "task_id": task["task_id"],
        "task_instruction": task.get("instruction", "")[:300],
        "target_state": task.get("target_state", "")[:200],
        "candidate_files": files,
        "candidate_exceptions": excs,
        "verifier_targets": pytest_lines or [task.get("verifier_hint", "")],
        "expected_failure_modes": task.get("expected_failure_modes", []),
        "safe_actions_hint": task.get("safe_actions", [])[:5],
    }


class CanonicalBeliefWrapper(Harness):

    def __init__(self, inner: Harness) -> None:
        self.inner = inner
        self.harness_id = f"BIWM1_{inner.harness_id}"

    def make_observation(self, task: dict, step: int, history: list[dict]) -> Observation:
        raw_obs = self.inner.make_observation(task, step, history)
        stripped = _strip_decorations(raw_obs.raw_text)
        canon = _extract_canonical_fields(stripped, task)
        canonical_text = (
            f"[step {step}] [canonical belief input]\n"
            f"- task: {canon['task_instruction']}\n"
            f"- target_state: {canon['target_state']}\n"
            f"- candidate_files: {canon['candidate_files']}\n"
            f"- candidate_exceptions: {canon['candidate_exceptions']}\n"
            f"- verifier_targets: {canon['verifier_targets']}\n"
            f"- expected_failure_modes (catalog): {canon['expected_failure_modes']}\n"
            f"- safe_actions_hint: {canon['safe_actions_hint']}\n\n"
            f"(harness-specific decorations stripped; if relevant, "
            f"this rollout passed through {self.inner.harness_id} but you should "
            f"report belief in task-canonical form.)"
        )
        return Observation(
            raw_text=canonical_text,
            structured={
                "harness_view": "biwm1_canonical",
                "underlying_harness": self.inner.harness_id,
                "canonical_belief_input": canon,
            },
            meta={
                "biwm": "canonical_belief",
                "underlying_harness": self.inner.harness_id,
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
            "biwm_components": ["canonical_belief"],
        }
