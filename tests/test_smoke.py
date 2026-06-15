"""本地 smoke (无 LLM 调用), 验证: imports / schema / logger / harness 实例化."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from skeleton.benchmark.hibench_loader import load_tasks
from skeleton.core.belief_schema import (
    empty_belief_output,
    validate_belief,
    validate_step_log,
)
from skeleton.core.jsonl_logger import JSONLLogger
from skeleton.harnesses import (
    H0RawHarness,
    H1StructuredHarness,
    H2RiskGatedHarness,
    H3RepairHeavyHarness,
    H4VerificationSelectiveHarness,
    H5CostAwareHarness,
    HARNESS_REGISTRY,
)


def test_empty_belief_passes_schema():
    b = empty_belief_output(horizon=5)
    assert validate_belief(b) == []


def test_step_log_validates():
    b = empty_belief_output(horizon=3)
    record = {
        "task_id": "t1",
        "benchmark": "HIBench-Code",
        "environment_id": "E_default",
        "harness_id": "H0_raw",
        "base_llm": "gpt-5.4-mini",
        "rollout_horizon": 3,
        "step": 0,
        "observation": "raw",
        "canonical_belief_input": {},
        "belief_output": b,
        "candidate_actions": ["a"],
        "selected_action": "a",
        "blocked_actions": [],
        "blocking_reasons": [],
        "verification_mask": {"verified": False, "verifier_type": "none", "cost": 0.0},
        "repair_event": {"occurred": False, "repair_action": None},
        "shadow_execution": {},
        "downstream_result": {},
    }
    errs = validate_step_log(record)
    assert errs == [], errs


def test_load_tasks_8():
    tasks = load_tasks()
    assert len(tasks) >= 5, len(tasks)
    for t in tasks:
        assert "task_id" in t
        assert "raw_observation" in t


def test_all_harnesses_instantiate():
    for cls in (
        H0RawHarness,
        H1StructuredHarness,
        H2RiskGatedHarness,
        H3RepairHeavyHarness,
        H4VerificationSelectiveHarness,
        H5CostAwareHarness,
    ):
        h = cls()
        assert h.harness_id
        obs = h.make_observation({"task_id": "x", "raw_observation": "raw"}, 0, [])
        assert obs.raw_text
        dec = h.gate_action({"task_id": "x"}, "noop", ["noop"])
        assert dec.selected_action == "noop"


def test_harness_observation_differences():
    """H0/H1/H2/H3/H4/H5 在 step 0 produce **observably different** raw_text/structured.

    这是 Day-2 关键不变量: 如果两个 harness 的 prompt 完全相同, 后续 D_belief
    任何差异就只是 LLM nondeterminism, 不是 harness 效果. 必须 fail-fast.
    """
    task = load_tasks()[6]  # toy_007_destructive_action_trap (含 risky_actions)
    raw_texts = {}
    structureds = {}
    for hid, cls in HARNESS_REGISTRY.items():
        obs = cls().make_observation(task, 0, [])
        raw_texts[hid] = obs.raw_text
        structureds[hid] = obs.structured
    # 所有 raw_text 必须两两不同
    keys = list(raw_texts.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = keys[i], keys[j]
            assert raw_texts[a] != raw_texts[b], f"{a} vs {b} 的 raw_text 相同 — harness 未注入差异"
    # structured.harness_view 必须不同
    views = {s.get("harness_view") for s in structureds.values()}
    # H0 没设 harness_view, 其他 5 个有 5 种不同 view => set size >= 6 (含 None)
    assert len(views) == 6, f"harness_view 应每个 harness 唯一; got {views}"


def test_h2_gate_blocks_risky_action():
    """H2 必须能阻断 risky action; H0 必须不阻断."""
    task = load_tasks()[6]  # toy_007 含 risky_actions=["运行 rotate('/etc/passwd')", "直接 os.system rm", ...]
    risky_cand = "运行 rotate('/etc/passwd') 然后检查 fixture"
    h2 = H2RiskGatedHarness()
    dec = h2.gate_action(task, risky_cand, [risky_cand])
    assert dec.selected_action is None, "H2 应阻断 risky action"
    assert dec.blocked_actions == [risky_cand]
    assert dec.blocking_reasons

    h0 = H0RawHarness()
    dec0 = h0.gate_action(task, risky_cand, [risky_cand])
    assert dec0.selected_action == risky_cand, "H0 不应阻断"
    assert dec0.blocked_actions == []


def test_logger():
    with tempfile.TemporaryDirectory() as d:
        log = JSONLLogger(Path(d) / "test.jsonl")
        log.write({"a": 1})
        log.write({"b": 2})
        text = (Path(d) / "test.jsonl").read_text()
        lines = [json.loads(x) for x in text.strip().split("\n")]
        assert lines == [{"a": 1}, {"b": 2}]


if __name__ == "__main__":
    test_empty_belief_passes_schema()
    test_step_log_validates()
    test_load_tasks_8()
    test_all_harnesses_instantiate()
    test_harness_observation_differences()
    test_h2_gate_blocks_risky_action()
    test_logger()
    print("ALL OK")
