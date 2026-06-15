"""H1 Structured Harness (readme §8 H1).

与 H0 Raw 的核心差异 (prompt-side, 必须是 LLM 看得见的差异):
- observation 不再是 raw 终端文本; 改为**结构化解析后的 JSON**:
    * parsed_traceback: 从 raw_observation 抽出 exception type / file / line / 关键行
    * code_symbols: 从 raw_observation 抽出函数名 / 模块名 / 类名
    * test_targets: verifier_hint 转成结构化字段
    * structured 字段全部填实, raw_text 同时给一个**简短**的高层描述 (不再是堆栈贴片)
- 暴露这种 "结构化解析" 是 readme §6.1 / §8 H1 描述的核心: H1 给 agent 一个
  "干净的中间 state", 期望减少 wrong-file 误判但可能丧失一些 raw 细节
- gate / verifier / repair 与 H0 一致 (全放行 / 不验证 / 不修复); 差异只在
  observation 暴露层
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
# Day 6 G2 Terminal-Bench patch: 识别 terminal prompt + cmd
_SHELL_PROMPT_RX = re.compile(r"^\$\s+(?P<cmd>.+)$", re.MULTILINE)
_TASK_HEADER_RX = re.compile(
    r"task_id:\s*(?P<tid>\S+)|category=(?P<cat>[\w-]+)|difficulty=(?P<diff>\w+)"
)


def _parse_observation(raw: str) -> dict[str, Any]:
    """Extract structured fields from the raw observation text.

    本函数是 H1 的"信息暴露"决策: 抽哪些, 不抽哪些。当前抽:
    - exception_type / exception_message
    - failing_file (file:line)
    - function_names defined in shown code
    - mentioned modules
    """
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
    # Day 6 patch: 识别 terminal-style shell prompts (Terminal-Bench tasks)
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
        # 高层概要 (非堆栈贴片): 一句 task 状态 + 当前 step 处于哪
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
