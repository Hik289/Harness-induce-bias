"""Harness 抽象接口 (readme §8 + §2.1).

设计原则:
- harness 不"执行任务", 它只决定 agent 看到什么 (observation), 能做什么
  (action_space), 哪些 action 被允许 (gate), 验证用什么 (verifier), 修复
  策略 (repair), 日志策略 (log_policy)
- 同一 task 在不同 harness 下跑, base LLM 和任务都一样, 只有 harness 不同
- 本 Day-1 骨架 (SETUP_DAY1) 只有 H0 端到端可跑; H1-H5 暴露同样接口但 Day-2
  填充
- Day-1 阶段不接真实 environment (不改 file / 不执行 shell), observation =
  task spec 静态片段; downstream_result 由 harness/runner 用 simple test
  函数判定 (HIBench-Code v0 = unit test pass/fail). 这符合 Director "不要把
  environment 改动混入这一阶段"。

关键不变量:
- 所有 harness 的 belief rollout protocol 完全一致 (rollout.py 实现一次, 所
  有 harness 共用), harness 只改写传给 LLM 的 prompt / observation. 这是 RQ1
  能成立的前提: 同 LLM 同任务, 唯一变量是 harness
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Observation:
    """harness 给 LLM 的当前观测; 既给 belief prompt, 又给 candidate-action 提
    prompt."""

    raw_text: str
    structured: dict[str, Any] = field(default_factory=dict)
    # harness 自报的元数据 (e.g., verification result, blocked action, etc.)
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class ActionDecision:
    """LLM 选定的 action 经过 harness gate 后的最终决定."""

    selected_action: Optional[str]
    candidate_actions: list[str]
    blocked_actions: list[str] = field(default_factory=list)
    blocking_reasons: list[str] = field(default_factory=list)


@dataclass
class VerificationResult:
    verified: bool
    verifier_type: str  # none|cheap|targeted|full|human|strong_model
    cost: float
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class RepairEvent:
    occurred: bool
    repair_action: Optional[str] = None
    extras: dict[str, Any] = field(default_factory=dict)


class Harness(ABC):
    """所有 harness 继承; readme §8 H0-H6."""

    harness_id: str  # e.g. "H0_raw"

    # 接收 readme 9.1 schema 的 observation 表示形式, harness 决定怎么暴露给
    # LLM。
    @abstractmethod
    def make_observation(self, task: dict, step: int, history: list[dict]) -> Observation:
        """决定 agent 在 step t 看到什么。harness 可以裁剪 / 加结构 / 过滤。"""

    # gate: 拿到 LLM 候选 action 后, harness 决定是否阻止
    @abstractmethod
    def gate_action(
        self, task: dict, candidate_action: str, all_candidates: list[str]
    ) -> ActionDecision:
        """返回 ActionDecision; 若 selected_action=None 表示该步被阻止 (用于 H2)."""

    # verifier: 每步可选, 用于 H4 selective verification
    @abstractmethod
    def run_verifier(self, task: dict, step: int, action: Optional[str]) -> VerificationResult:
        """harness 决定本步是否调用 verifier, 用哪种 verifier."""

    # repair: H3 自动 repair
    def attempt_repair(
        self, task: dict, last_action: Optional[str], failure_info: dict
    ) -> RepairEvent:
        """默认: 不做 repair. H3 override."""
        return RepairEvent(occurred=False)

    # log policy: 每个 harness 可决定 step log 里哪些字段保留, 哪些剔除
    def filter_log(self, step_record: dict) -> dict:
        """默认: 透传. H5 cost-aware 可裁掉一些重日志, H6 bias-aware 补齐."""
        return step_record

    # harness 自报 metadata, 写入每条 step log
    def metadata(self) -> dict[str, Any]:
        return {"harness_id": self.harness_id}
