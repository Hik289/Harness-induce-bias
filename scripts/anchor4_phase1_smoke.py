from __future__ import annotations

import argparse
import json
import math
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

from skeleton.benchmark.hibench_loader import load_tasks
from skeleton.core.jsonl_logger import JSONLLogger
from skeleton.core.llm_client import LLMClient
from skeleton.core.rollout import run_kstep_rollout
from skeleton.harnesses import H0RawHarness, H2RiskGatedHarness
from metrics.d_belief import d_belief_components

JST = timezone(timedelta(hours=9))


def _read_last_belief(jsonl_path: Path) -> dict:
    last_line = ""
    with jsonl_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                last_line = line
    rec = json.loads(last_line)
    return rec["belief_output"]


def _binomial_one_sided_pmf(k: int, n: int, p: float = 0.5) -> float:
    from math import comb
    return sum(comb(n, i) * (p ** i) * ((1 - p) ** (n - i)) for i in range(k, n + 1))


def main(out_dir: str, tasks_path: str | None, seed: int, n_tasks: int) -> int:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tasks = load_tasks(tasks_path)[:n_tasks]

    llm = LLMClient(min_interval_s=0.35, max_retries=3)
    harnesses = {
        "H0_raw": H0RawHarness(),
        "H2_risk_gated": H2RiskGatedHarness(),
    }

    t_start = time.time()
    per_task: list[dict] = []

    for task in tasks:
        task_row: dict = {"task_id": task["task_id"]}
        beliefs: dict[tuple[str, int], dict] = {}

        for K in (1, 3):
            for hid, harness in harnesses.items():
                log_path = out / f"{hid}_{task['task_id']}_K{K}_seed{seed}.jsonl"
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
                beliefs[(hid, K)] = _read_last_belief(log_path)
                task_row[f"{hid}_K{K}_log"] = str(log_path)
                task_row[f"{hid}_K{K}_tokens"] = summary["total_tokens"]
                task_row[f"{hid}_K{K}_latency_s"] = round(summary["total_latency_s"], 2)

        c1 = d_belief_components(beliefs[("H0_raw", 1)], beliefs[("H2_risk_gated", 1)])
        c3 = d_belief_components(beliefs[("H0_raw", 3)], beliefs[("H2_risk_gated", 3)])
        task_row["D_K1"] = c1["D_belief"]
        task_row["D_K3"] = c3["D_belief"]
        task_row["delta"] = c3["D_belief"] - c1["D_belief"]
        task_row["direction_consistent"] = c3["D_belief"] > c1["D_belief"]
        task_row["components_K1"] = c1
        task_row["components_K3"] = c3
        per_task.append(task_row)
        print(
            f"  [{task['task_id']:<32s}] D_K1={c1['D_belief']:.3f} "
            f"D_K3={c3['D_belief']:.3f} delta={task_row['delta']:+.3f} "
            f"consistent={task_row['direction_consistent']}",
            flush=True,
        )

    n = len(per_task)
    k = sum(1 for r in per_task if r["direction_consistent"])
    p = _binomial_one_sided_pmf(k, n, 0.5)
    anchor_passed = (k == n)

    overall = {
        "anchor": "anchor_4",
        "spec": "Phase 1 smoke: D_belief(H0,H2,K=3) > D_belief(H0,H2,K=1) on 5 toy tasks",
        "benchmark": "HIBench-Code-v0_toy",
        "harness_pair": ["H0_raw", "H2_risk_gated"],
        "K_values": [1, 3],
        "seed": seed,
        "n_tasks": n,
        "n_consistent": k,
        "binomial_p_one_sided_ge_k": p,
        "anchor_passed_strict_5of5": anchor_passed,
        "elapsed_s": round(time.time() - t_start, 2),
        "ended_jst": datetime.now(JST).isoformat(timespec="seconds"),
        "tasks": per_task,
    }
    summary_path = out / "anchor4_summary.json"
    summary_path.write_text(json.dumps(overall, ensure_ascii=False, indent=2))
    print("\n=== anchor_4 summary ===")
    print(json.dumps({k: v for k, v in overall.items() if k != "tasks"}, ensure_ascii=False, indent=2))
    return 0 if anchor_passed else 1


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="logs/anchor4_phase1_smoke")
    p.add_argument("--tasks", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-tasks", type=int, default=5)
    a = p.parse_args()
    sys.exit(main(a.out, a.tasks, a.seed, a.n_tasks))
