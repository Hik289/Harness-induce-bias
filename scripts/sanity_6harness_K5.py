"""6-harness × 1-task × K=5 sanity: 把所有 6 harness 跑同一 task 同一 seed K=5,
把 belief JSON 摆在一起看, 确认 harness 间差异是 categorical (action / failure_mode
/ risk_state / progress) 而不只是 numeric noise. Director 指定的 Day-2 最后一道
sanity (anchor_4 之外的扩展 check).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

_HERE = Path(__file__).resolve()
_EXPERIMENTS = _HERE.parents[2]
_SKELETON = _HERE.parents[1]
for _p in (str(_EXPERIMENTS.parent), str(_SKELETON), str(_EXPERIMENTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from skeleton.benchmark.hibench_loader import load_tasks  # noqa: E402
from skeleton.core.jsonl_logger import JSONLLogger  # noqa: E402
from skeleton.core.llm_client import LLMClient  # noqa: E402
from skeleton.core.rollout import run_kstep_rollout  # noqa: E402
from skeleton.harnesses import HARNESS_REGISTRY  # noqa: E402
from metrics.d_belief import d_belief_components  # noqa: E402

JST = timezone(timedelta(hours=9))


def _read_last_belief(jsonl_path: Path) -> dict:
    last = ""
    with jsonl_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                last = line
    return json.loads(last)["belief_output"]


def main(out_dir: str, task_id: str, K: int, seed: int) -> int:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tasks = load_tasks()
    task = next((t for t in tasks if t["task_id"] == task_id), None)
    if task is None:
        raise SystemExit(f"task_id 不存在: {task_id}; 可用: {[t['task_id'] for t in tasks]}")

    llm = LLMClient(min_interval_s=0.35, max_retries=3)
    beliefs: dict[str, dict] = {}
    summaries: dict[str, dict] = {}
    t0 = time.time()
    for hid, cls in HARNESS_REGISTRY.items():
        harness = cls()
        log_path = out / f"{hid}_{task_id}_K{K}_seed{seed}.jsonl"
        logger = JSONLLogger(log_path)
        summary = run_kstep_rollout(
            task=task,
            harness=harness,
            llm=llm,
            horizon=K,
            logger=logger,
            benchmark_id="HIBench-Code-v0_toy",
            environment_id="E_default_v0",
            seed=seed,
        )
        beliefs[hid] = _read_last_belief(log_path)
        summaries[hid] = {
            "log": str(log_path),
            "tokens": summary["total_tokens"],
            "latency_s": round(summary["total_latency_s"], 2),
        }
        bs = beliefs[hid]["belief_state"]
        act = beliefs[hid]["next_action_recommendation"]["action"][:80]
        print(
            f"  [{hid:<28s}] progress={bs['task_progress']:<8s} risk={bs['risk_state']:<6s} "
            f"recov={bs['recoverability']:<6s} fail_mode={bs['likely_failure_mode']:<22s} "
            f"unc={bs['uncertainty']:.2f}",
            flush=True,
        )

    # pairwise D_belief across the 6
    hids = list(HARNESS_REGISTRY.keys())
    pairwise: list[dict] = []
    for i in range(len(hids)):
        for j in range(i + 1, len(hids)):
            a, b = hids[i], hids[j]
            comp = d_belief_components(beliefs[a], beliefs[b])
            pairwise.append({"pair": [a, b], **comp})

    cat_view = {
        hid: {
            "task_progress": beliefs[hid]["belief_state"]["task_progress"],
            "risk_state": beliefs[hid]["belief_state"]["risk_state"],
            "recoverability": beliefs[hid]["belief_state"]["recoverability"],
            "likely_failure_mode": beliefs[hid]["belief_state"]["likely_failure_mode"],
            "uncertainty": beliefs[hid]["belief_state"]["uncertainty"],
            "success_probability": beliefs[hid]["predicted_future"]["success_probability"],
            "failure_attractor_probability": beliefs[hid]["predicted_future"]["failure_attractor_probability"],
            "expected_repair_need": beliefs[hid]["predicted_future"]["expected_repair_need"],
            "next_action_first80": beliefs[hid]["next_action_recommendation"]["action"][:80],
        }
        for hid in hids
    }

    # 判 "categorical 差异 vs numeric noise"
    progress_set = {v["task_progress"] for v in cat_view.values()}
    risk_set = {v["risk_state"] for v in cat_view.values()}
    fmode_set = {v["likely_failure_mode"] for v in cat_view.values()}
    action_unique = len({v["next_action_first80"] for v in cat_view.values()})
    categorical_signals = (
        len(progress_set) +
        len(risk_set) +
        len(fmode_set) +
        action_unique
    )
    # 6 harness 之间, 任何一种 categorical 字段至少出现 2 个值算"有差异"
    categorical_diff_present = (
        len(progress_set) >= 2 or len(risk_set) >= 2 or
        len(fmode_set) >= 2 or action_unique >= 3
    )

    overall = {
        "spec": "6-harness × 1-task × K=5 sanity (Day-2)",
        "benchmark": "HIBench-Code-v0_toy",
        "task_id": task_id,
        "K": K,
        "seed": seed,
        "harnesses": hids,
        "categorical_diff_present": categorical_diff_present,
        "categorical_summary": {
            "task_progress_values": sorted(progress_set),
            "risk_state_values": sorted(risk_set),
            "likely_failure_mode_values": sorted(fmode_set),
            "n_unique_next_actions": action_unique,
        },
        "per_harness_belief_summary": cat_view,
        "pairwise_d_belief": pairwise,
        "summaries": summaries,
        "elapsed_s": round(time.time() - t0, 2),
        "ended_jst": datetime.now(JST).isoformat(timespec="seconds"),
    }
    summary_path = out / "sanity_summary.json"
    summary_path.write_text(json.dumps(overall, ensure_ascii=False, indent=2))
    print("\n=== sanity summary ===")
    print(json.dumps({
        k: v for k, v in overall.items()
        if k not in ("per_harness_belief_summary", "pairwise_d_belief", "summaries")
    }, ensure_ascii=False, indent=2))
    return 0 if categorical_diff_present else 1


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="logs/sanity_6harness_K5")
    p.add_argument("--task", default="toy_007_destructive_action_trap")
    p.add_argument("--K", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    sys.exit(main(a.out, a.task, a.K, a.seed))
