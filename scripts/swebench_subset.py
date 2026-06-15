"""Day 8 Task 2b: SWE-bench Verified subset descriptive replication.

10 task × H0/H1/H2 × K∈{3, 5} × seed=42 = 60 new run.
Imagined rollout (Director Day-2 决策, 不在 hpc 跑真实 docker).
"""
from __future__ import annotations

import argparse
import json
import statistics
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

from skeleton.benchmark.swebench_adapter import load_swebench_tasks  # noqa: E402
from skeleton.core.jsonl_logger import JSONLLogger  # noqa: E402
from skeleton.core.llm_client import LLMClient  # noqa: E402
from skeleton.core.rollout import run_kstep_rollout  # noqa: E402
from skeleton.harnesses import HARNESS_REGISTRY  # noqa: E402
from metrics.d_belief import d_belief_components  # noqa: E402

JST = timezone(timedelta(hours=9))

HARNESS_SUBSET = ["H0_raw", "H1_structured", "H2_risk_gated"]
K_VALUES = [3, 5]
SEED = 42


def _read_last_belief(p: Path) -> dict:
    last = ""
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            s = line.strip()
            if s:
                last = s
    return json.loads(last)["belief_output"]


def main(out_dir: str, n_tasks: int) -> int:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"[swebench subset] loading tasks ...", flush=True)
    tasks = load_swebench_tasks(n=n_tasks, seed=SEED)
    print(f"  loaded {len(tasks)} task: {[t['task_id'] for t in tasks]}", flush=True)
    harnesses = {hid: HARNESS_REGISTRY[hid]() for hid in HARNESS_SUBSET}
    llm = LLMClient(min_interval_s=0.35, max_retries=3)

    n_expected = len(tasks) * len(harnesses) * len(K_VALUES)
    print(f"[swebench subset] starting: {len(tasks)} task × {len(harnesses)} harness × {len(K_VALUES)} K = {n_expected} run", flush=True)
    t_start = time.time()
    runs: list[dict] = []
    crashed: list[dict] = []
    bel_idx: dict[tuple[str, str, int], dict] = {}
    total_tokens = 0
    completed = 0

    for task in tasks:
        for K in K_VALUES:
            for hid, harness in harnesses.items():
                log_path = out / f"{hid}_{task['task_id']}_K{K}_seed{SEED}.jsonl"
                if log_path.exists() and log_path.stat().st_size > 0:
                    try:
                        bel_idx[(hid, task["task_id"], K)] = _read_last_belief(log_path)
                        completed += 1
                        continue
                    except Exception:
                        log_path.unlink(missing_ok=True)
                logger = JSONLLogger(log_path)
                try:
                    s = run_kstep_rollout(
                        task=task, harness=harness, llm=llm, horizon=K, logger=logger,
                        benchmark_id="SWE-bench-Verified", environment_id="E_default_swe",
                        seed=SEED,
                    )
                    bel_idx[(hid, task["task_id"], K)] = _read_last_belief(log_path)
                    s["log_path"] = str(log_path)
                    runs.append(s)
                    total_tokens += s["total_tokens"]
                except Exception as e:
                    crashed.append({"task_id": task["task_id"], "harness_id": hid, "K": K, "error": str(e)})
                    print(f"  [CRASH] {task['task_id']} {hid} K={K}: {e}", flush=True)
                completed += 1
                if completed % 10 == 0 or completed == n_expected:
                    elapsed = time.time() - t_start
                    eta = elapsed / completed * (n_expected - completed)
                    print(f"  [{completed}/{n_expected}] elapsed={elapsed/60:.1f}min eta={eta/60:.1f}min tokens={total_tokens/1000:.1f}K crashed={len(crashed)}", flush=True)

    # D_belief 描述性: H0 vs H1 / H0 vs H2 per K
    print("\n=== D_belief per pair × K ===", flush=True)
    table: dict[str, dict] = {}
    for hx in ("H1_structured", "H2_risk_gated"):
        for K in K_VALUES:
            comps = []
            for task in tasks:
                a = bel_idx.get(("H0_raw", task["task_id"], K))
                b = bel_idx.get((hx, task["task_id"], K))
                if a is None or b is None:
                    continue
                comps.append(d_belief_components(a, b))
            if not comps:
                table[f"H0_vs_{hx}_K{K}"] = {"n": 0}
                continue
            table[f"H0_vs_{hx}_K{K}"] = {
                "n": len(comps),
                "D_belief_mean": round(statistics.fmean(c["D_belief"] for c in comps), 4),
                "D_belief_std": round(statistics.pstdev(c["D_belief"] for c in comps) if len(comps) > 1 else 0, 4),
                "cat_mismatch_mean": round(statistics.fmean(c["cat_mismatch"] for c in comps), 4),
                "failure_mode_mismatch_mean": round(statistics.fmean(c["failure_mode_mismatch"] for c in comps), 4),
                "set_distance_mean": round(statistics.fmean(c["set_distance"] for c in comps), 4),
                "num_distance_mean": round(statistics.fmean(c["num_distance"] for c in comps), 4),
                "action_mismatch_mean": round(statistics.fmean(c["action_mismatch"] for c in comps), 4),
            }
            v = table[f"H0_vs_{hx}_K{K}"]
            print(f"  H0 vs {hx} K={K}  n={v['n']}  D={v['D_belief_mean']:.3f}  cat={v['cat_mismatch_mean']:.3f}  fail={v['failure_mode_mismatch_mean']:.3f}  num={v['num_distance_mean']:.3f}", flush=True)

    elapsed = time.time() - t_start
    overall = {
        "phase": "DAY8_SWEbench_subset",
        "benchmark": "SWE-bench-Verified",
        "harnesses": HARNESS_SUBSET, "K_values": K_VALUES,
        "n_tasks": len(tasks), "seed": SEED,
        "n_expected": n_expected, "n_runs_new": len(runs), "n_crashed": len(crashed),
        "total_tokens_new": total_tokens,
        "elapsed_s": round(elapsed, 1),
        "ended_jst": datetime.now(JST).isoformat(timespec="seconds"),
        "tasks": [{"task_id": t["task_id"], "category": t["category"]} for t in tasks],
        "pairwise_D": table,
        "crashed": crashed,
    }
    (out / "swebench_summary.json").write_text(json.dumps(overall, ensure_ascii=False, indent=2))
    print(f"\n=== summary written ===")
    print(json.dumps({k: v for k, v in overall.items() if k not in ("crashed", "tasks", "pairwise_D")}, ensure_ascii=False, indent=2))
    return 0 if (len(crashed) == 0 and len(runs) == n_expected) else 1


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="logs/swebench_subset")
    p.add_argument("--n-tasks", type=int, default=10)
    a = p.parse_args()
    sys.exit(main(a.out, a.n_tasks))
