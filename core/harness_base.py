from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Observation:

    raw_text: str
    structured: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class ActionDecision:

    selected_action: Optional[str]
    candidate_actions: list[str]
    blocked_actions: list[str] = field(default_factory=list)
    blocking_reasons: list[str] = field(default_factory=list)


@dataclass
class VerificationResult:
    verified: bool
    verifier_type: str
    cost: float
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class RepairEvent:
    occurred: bool
    repair_action: Optional[str] = None
    extras: dict[str, Any] = field(default_factory=dict)


class Harness(ABC):

    harness_id: str

    @abstractmethod
    def make_observation(self, task: dict, step: int, history: list[dict]) -> Observation:
        pass

    @abstractmethod
    def gate_action(
        self, task: dict, candidate_action: str, all_candidates: list[str]
    ) -> ActionDecision:
        pass

    @abstractmethod
    def run_verifier(self, task: dict, step: int, action: Optional[str]) -> VerificationResult:
        pass

    def attempt_repair(
        self, task: dict, last_action: Optional[str], failure_info: dict
    ) -> RepairEvent:
        return RepairEvent(occurred=False)

    def filter_log(self, step_record: dict) -> dict:
        return step_record

    def metadata(self) -> dict[str, Any]:
        return {"harness_id": self.harness_id}
