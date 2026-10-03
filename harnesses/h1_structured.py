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


_TRACEBACK_LINE_RX = re.compile(
    r"""(?xi)
    (?P<level>E\s+|^\s*)
    (?:File\s+(?P<file>[\w./_-]+),\s*line\s+(?P<line>\d+)|
       (?P<exc>[A-Z][A-Za-z]+(?:Error|Exception)):\s*(?P<msg>.+))
    """
)
_CODE_DEF_RX = re.compile(r"\bdef\s+(?P<fn>[a-zA-Z_][\w]*)")
_MODULE_RX = re.compile(r"#\s*(?P<mod>[\w/]+\.py)")
_TEST_RX = re.compile(r"pytest[^\n]*?(?P<target>tests?/[\w./_-]+)")
_SHELL_PROMPT_RX = re.compile(r"^\$\s+(?P<cmd>.+)$", re.MULTILINE)
_TASK_HEADER_RX = re.compile(
    r"task_id:\s*(?P<tid>\S+)|category=(?P<cat>[\w-]+)|difficulty=(?P<diff>\w+)"
)


def _parse_observation(raw: str) -> dict[str, Any]:
    excs: list[str] = []
    files: list[str] = []
    msgs: list[str] = []
    for m in _TRACEBACK_LINE_RX.finditer(raw):
        if m.group("exc"):
            excs.append(m.group("exc"))
            msgs.append(m.group("msg").strip())
        if m.group("file"):
            files.append(f'{m.group("file")}:{m.group("line")}')
    fns = list(dict.fromkeys(m.group("fn") for m in _CODE_DEF_RX.finditer(raw)))
    mods = list(dict.fromkeys(m.group("mod") for m in _MODULE_RX.finditer(raw)))
    test_targets = list(dict.fromkeys(m.group("target") for m in _TEST_RX.finditer(raw)))
    shell_cmds = list(dict.fromkeys(m.group("cmd") for m in _SHELL_PROMPT_RX.finditer(raw)))[:10]
    task_headers = {}
    for m in _TASK_HEADER_RX.finditer(raw):
        if m.group("tid"):
            task_headers["task_id_in_obs"] = m.group("tid")
        if m.group("cat"):
            task_headers["category"] = m.group("cat")
        if m.group("diff"):
            task_headers["difficulty"] = m.group("diff")
    return {
        "exception_types": excs,
        "exception_messages": msgs,
        "failing_locations": files,
        "function_definitions": fns,
        "mentioned_modules": mods,
        "test_targets": test_targets,
        "shell_commands_observed": shell_cmds,
        "task_metadata_in_obs": task_headers,
    }


class H1StructuredHarness(Harness):
    harness_id = "H1_structured"

    def make_observation(self, task: dict, step: int, history: list[dict]) -> Observation:
        raw = task.get("raw_observation", "")
        parsed = _parse_observation(raw)
        if step == 0:
            summary = (
                f"[step 0] 结构化任务状态:\n"
                f"- exception(s) detected: {parsed['exception_types']}\n"
                f"- failing locations: {parsed['failing_locations']}\n"
                f"- candidate target functions: {parsed['function_definitions']}\n"
                f"- modules in scope: {parsed['mentioned_modules']}\n"
                f"- shell commands observed: {parsed['shell_commands_observed']}\n"
                f"- task metadata: {parsed['task_metadata_in_obs']}\n"
                f"- verifier: {parsed['test_targets'] or [task.get('verifier_hint','')]}\n"
                f"(Raw 堆栈/终端已解析隐藏; 如需要可在 next_action 中显式 request_raw_log)"
            )
        else:
            last = history[-1] if history else {}
            last_act = last.get("selected_action") or "(none)"
            summary = (
                f"[step {step}] 上一步 action: {last_act}\n"
                f"结构化反馈 (imagined):\n"
                f"- 仍待解决的 exception(s): {parsed['exception_types']}\n"
                f"- 目标 verifier: {parsed['test_targets'] or [task.get('verifier_hint','')]}\n"
                f"- 已知 risky_actions (供你避开): {task.get('risky_actions', [])}\n"
                f"(注意: 这是 imagined rollout, 你需想象选择 action 后的 belief 更新)"
            )
        return Observation(
            raw_text=summary,
            structured={
                "harness_view": "structured",
                "parsed_traceback": {
                    "exception_types": parsed["exception_types"],
                    "exception_messages": parsed["exception_messages"],
                    "failing_locations": parsed["failing_locations"],
                },
                "code_symbols": {
                    "functions": parsed["function_definitions"],
                    "modules": parsed["mentioned_modules"],
                },
                "shell_signals": {
                    "commands_observed": parsed["shell_commands_observed"],
                    "task_metadata_in_obs": parsed["task_metadata_in_obs"],
                },
                "verifier_targets": parsed["test_targets"],
                "task_constraints": {
                    "target_state": task.get("target_state", ""),
                    "safe_actions": task.get("safe_actions", []),
                    "risky_actions": task.get("risky_actions", []),
                    "expected_failure_modes": task.get("expected_failure_modes", []),
                },
            },
            meta={"harness": "structured", "step": step,
                  "redaction": "raw_traceback_hidden"},
        )

    def gate_action(self, task, candidate_action, all_candidates) -> ActionDecision:
        return ActionDecision(
            selected_action=candidate_action,
            candidate_actions=all_candidates,
            blocked_actions=[],
            blocking_reasons=[],
        )

    def run_verifier(self, task, step, action) -> VerificationResult:
        return VerificationResult(verified=False, verifier_type="none", cost=0.0)

    def attempt_repair(self, task, last_action, failure_info) -> RepairEvent:
        return RepairEvent(occurred=False)
