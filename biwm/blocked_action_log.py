"""BIWM-2: Blocked-Action Logging (readme §11.2).

H2 风险阻断的关键 bias 来源: agent 不知道哪些 action 被阻塞, 也不知道阻塞的
原因 → blocked-branch censorship. BIWM-2 在 observation 里**显式补回**:
- 上一步 candidate action 是什么
- 哪些被阻塞 (string + reason)
- 风险估计 (如果有)
- "要安全执行需要什么补充" (heuristic, e.g. "限定 path 在 logs/ 目录" 或 "加 dry-run")

只对 risk-gated 类 harness 起作用 (本项目 H2). 对其他 harness 退化为 noop.
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


def _safe_execution_hint(reason: str) -> str:
    r = reason.lower()
    if "rm -rf" in r or "rmtree" in r:
        return "用 os.remove(path) 限定 path, 或先 dry-run"
    if "drop " in r or "truncate" in r:
        return "限制到非生产表 / 先 EXPLAIN, 用 transaction + rollback"
    if "force push" in r:
        return "改用 PR + review; 加 --force-with-lease 限定 ref"
    if "/etc/" in r or "/etc/shadow" in r or "/etc/passwd" in r:
        return "禁止访问系统文件; 改用 task-specific 路径"
    if "send_email" in r or "submit_order" in r:
        return "改用 dry-run / staging endpoint, 加 confirmation 字段"
    return "降级到 risky_actions list 之外的等效 action; 或为 sandbox/shadow 模式申请例外"


class BlockedActionLogWrapper(Harness):
    """包一层 harness, 在 make_observation 时把上一步的 blocked actions 显式
    写进 observation, 给出风险原因和安全执行 hint。
    """

    def __init__(self, inner: Harness) -> None:
        self.inner = inner
        self.harness_id = f"BIWM2_{inner.harness_id}"

    def make_observation(self, task: dict, step: int, history: list[dict]) -> Observation:
        inner_obs = self.inner.make_observation(task, step, history)
        if step == 0:
            return inner_obs  # no prior block to report

        last = history[-1] if history else {}
        blocked = last.get("blocked") or []
        if not blocked:
            return inner_obs

        # 重组: 加 blocked-action log 块到 raw_text 前面
        block_log_lines: list[str] = ["[BLOCKED-ACTION LOG]"]
        for ba in blocked[:5]:
            hint = _safe_execution_hint(ba)
            block_log_lines.append(
                f"  - candidate: {ba!r}\n"
                f"    blocked: True\n"
                f"    risk_class: high\n"
                f"    safe_execution_hint: {hint}"
            )
        block_log_lines.append(
            "请把上述 blocked 信息纳入 belief: 它们没有真正执行, 你看不到执行结果, "
            "但这些 branch 在 risk-gated harness 里是 censored 的 — 不要把 "
            "'failure_mode=none' 解读为风险消失."
        )
        prefix = "\n".join(block_log_lines)

        new_raw_text = f"{prefix}\n\n{inner_obs.raw_text}"
        new_structured = dict(inner_obs.structured or {})
        new_structured["biwm2_blocked_log"] = {
            "n_blocked_last_step": len(blocked),
            "blocked_candidates": blocked[:5],
        }
        return Observation(
            raw_text=new_raw_text,
            structured=new_structured,
            meta={
                **(inner_obs.meta or {}),
                "biwm": "blocked_action_log",
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
            "biwm_components": ["blocked_action_log"],
        }
