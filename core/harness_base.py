"""Harness interface described in README Sections 8 and 2.1.

The harness does not execute the task. It controls the observation, action
space, action gate, verifier, repair policy, and logging policy. Experiments
hold the task and base model fixed and vary only the harness. The benchmark
uses static task observations and a deterministic downstream evaluator, so
environment mutations cannot confound this comparison.

All harnesses share the rollout implementation in ``rollout.py``; a harness
may change only the prompt and observation presented to the model.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Observation:
    """Observation used for both belief and candidate-action prompts."""

    raw_text: str
    structured: dict[str, Any] = field(default_factory=dict)
    # Harness-provided metadata, such as verification or blocked-action state.
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class ActionDecision:
    """Action decision after applying the harness gate."""

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
    """Base class for the H0-H6 harness variants."""

    harness_id: str  # e.g. "H0_raw"

    @abstractmethod
    def make_observation(self, task: dict, step: int, history: list[dict]) -> Observation:
        """Build the observation shown to the agent at this step."""

    @abstractmethod
    def gate_action(
        self, task: dict, candidate_action: str, all_candidates: list[str]
    ) -> ActionDecision:
        """Apply the action gate; ``None`` means that the action was blocked."""

    @abstractmethod
    def run_verifier(self, task: dict, step: int, action: Optional[str]) -> VerificationResult:
        """Run the verifier selected by this harness, if any."""

    def attempt_repair(
        self, task: dict, last_action: Optional[str], failure_info: dict
    ) -> RepairEvent:
        """Return no repair by default; H3 overrides this behavior."""
        return RepairEvent(occurred=False)

    def filter_log(self, step_record: dict) -> dict:
        """Filter a step record; the default preserves every field."""
        return step_record

    def metadata(self) -> dict[str, Any]:
        """Return metadata attached to each step record."""
        return {"harness_id": self.harness_id}
