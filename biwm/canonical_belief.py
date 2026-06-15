"""BIWM-1: Canonical Belief (readme §11.1).

不同 harness 暴露不同 observation, 但在 belief rollout 前先把"原始 observation"
通过 LLM 翻译成 canonical 形态: 一个**所有 harness 共享**的固定 schema, 内容
仅取与 task 状态有关的信息, 剥离 harness 自带的 narrative/policy header/
budget banner 等"装饰".

实施位置: harness wrapper (覆盖 make_observation 的输出, 不动 harness 原始
逻辑). 这样所有 harness 进入 rollout 前都先经过同一规范化层。

为了避免每步多花一次 LLM call (canonical 步会让总 token 翻倍), 我们用
**deterministic rule-based canonicalizer**, 不调 LLM. 这与 readme §11.1 描
述的 "raw observation -> canonical task belief" 一致, readme 没规定必须 LLM 跑.
"""
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


# 在 raw_text 前缀里出现的 harness-specific 装饰行模式
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
    r"^\s*-\s+\w[^\n]*$",  # bullet items in policy lists
)


def _strip_decorations(text: str) -> str:
    for pat in _DECORATION_PATTERNS:
        text = re.sub(pat, "", text, flags=re.MULTILINE)
    # collapse blank lines
    text = re.sub(r"\n\s*\n+", "\n\n", text).strip()
    return text


def _extract_canonical_fields(text: str, task: dict) -> dict[str, Any]:
    """从 stripped observation 提取一组 canonical 字段, 跨 harness 通用."""
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
    """把任意 harness 包成一个新 harness, 在 make_observation 里做规范化。

    使用:
        wrapped = CanonicalBeliefWrapper(H2RiskGatedHarness())
        # rollout 仍然用 wrapped 跑, 但 LLM 看到的 prompt 是 canonical 后的
    """

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
