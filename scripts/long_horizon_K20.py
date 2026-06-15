"""Day 8 Task 1: K=20 long horizon on HIBench-Code v0_toy.

Goal: 测 D_belief 在 K∈{12, 16, 20} 上是否还在增长还是趋于饱和.
Phase 1 主表已经覆盖 K∈{1, 3, 5, 8}, 现补 K∈{12, 16, 20}.

Setup:
- 6 harness × 8 task × {K=12, 16, 20} × 1 seed = 144 new run
- seed=42 (与 Phase 1 主表 seed=42 那一组对齐, 便于跨 K 拼接)
- imagined rollout 不变 (Director Day-2 决策)

Output:
- logs/long_horizon_K20/{harness}_{task}_K{k}_seed42.jsonl (144 files)
- logs/long_horizon_K20/long_horizon_summary.json (含 per-(harness, K) tokens/lat + 跨 K trajectory)
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import traceback
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

JST = timezone(timedelta(hours=9))

K_VALUES_NEW = [12, 16, 20]   # 新跑的 K
SEED = 42


def main(out_dir: str, n_tasks: int | None = None) -> int:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tasks = load_tasks()
    if n_tasks is not None:
        tasks = tasks[:n_tasks]
    harnesses = {hid: cls() for hid, cls in HARNESS_REGISTRY.items()}
    llm = LLMClient(min_interval_s=0.35, max_retries=3)

    n_expected = len(tasks) * len(harnesses) * len(K_VALUES_NEW)
    print(
        f"[long_horizon] starting: {len(tasks)} task × {len(harnesses)} harness × "
        f"{len(K_VALUES_NEW)} K (new) = {n_expected} run", flush=True
    )
    t_start = time.time()
    runs: list[dict] = []
    crashed: list[dict] = []
    total_steps = total_pass = total_fail_schema = 0
    total_tokens = 0
    completed = 0

    # task-major, then K asc, then harness. Long-K runs are slow so we want
    # broad coverage before finishing all 20-step runs.
    for task in tasks:
        for K in K_VALUES_NEW:
            for hid, harness in harnesses.items():
                log_path = out / f"{hid}_{task['task_id']}_K{K}_seed{SEED}.jsonl"
                if log_path.exists() and log_path.stat().st_size > 0:
                    # resume-skip
                    try:
                        with log_path.open("r", encoding="utf-8") as fh:
                            lines = [json.loads(l) for l in fh if l.strip()]
                        if len(lines) >= K + 1:
                            runs.append({
                                "run_id": lines[-1].get("run_id", ""),
                                "task_id": task["task_id"],
                                "harness_id": hid,
                                "horizon": K, "seed": SEED,
                                "steps_written": len(lines),
                                "schema_pass": sum(1 for l in lines if not l.get("schema_fail", False)),
                                "schema_fail": sum(1 for l in lines if l.get("schema_fail", False)),
                                "total_tokens": sum(
                                    l.get("llm_stats", {}).get("total_tokens", 0) for l in lines
                                ),
                                "total_latency_s": sum(
                                    l.get("llm_stats", {}).get("latency_s", 0.0) for l in lines
                                ),
                                "log_path": str(log_path),
                                "resumed": True,
                            })
                            total_steps += len(lines)
                            total_pass += runs[-1]["schema_pass"]
                            total_fail_schema += runs[-1]["schema_fail"]
                            total_tokens += runs[-1]["total_tokens"]
                            completed += 1
                            continue
                    except Exception:  # noqa: BLE001
                        log_path.unlink(missing_ok=True)

                logger = JSONLLogger(log_path)
                try:
                    s = run_kstep_rollout(
                        task=task, harness=harness, llm=llm, horizon=K, logger=logger,
                        benchmark_id="HIBench-Code-v0_toy",
                        environment_id="E_default_v0",
                        seed=SEED,
                    )
                    s["log_path"] = str(log_path)
                    runs.append(s)
                    total_steps += s["steps_written"]
                    total_pass += s["schema_pass"]
                    total_fail_schema += s["schema_fail"]
                    total_tokens += s["total_tokens"]
                except Exception as e:  # noqa: BLE001
                    crashed.append({
                        "task_id": task["task_id"], "harness_id": hid,
                        "K": K, "error": f"{type(e).__name__}: {e}",
                        "traceback": traceback.format_exc()[:500],
                    })
                    print(f"  [CRASH] {task['task_id']} {hid} K={K}: {e}", flush=True)

                completed += 1
                elapsed = time.time() - t_start
                if completed % 12 == 0 or completed == n_expected:
                    pct = 100.0 * completed / n_expected
                    eta = elapsed / completed * (n_expected - completed)
                    print(
                        f"  [{completed:>3d}/{n_expected}] {pct:5.1f}% "
                        f"elapsed={elapsed/60:.1f}min eta={eta/60:.1f}min "
                        f"steps={total_steps} schema_ok={total_pass}/{total_steps} "
                        f"tokens={total_tokens/1000:.1f}K crashed={len(crashed)}",
                        flush=True,
                    )

    # ---- aggregate per-(harness, K) ----
    print("\n=== aggregating ===", flush=True)
    per_hk: dict[tuple[str, int], dict] = {}
    for r in runs:
        key = (r["harness_id"], r["horizon"])
        bucket = per_hk.setdefault(key, {
            "n_runs": 0, "tokens": [], "latency": [], "steps": [],
            "schema_pass": [], "schema_fail": [],
        })
        bucket["n_runs"] += 1
        bucket["tokens"].append(r["total_tokens"])
        bucket["latency"].append(r["total_latency_s"])
        bucket["steps"].append(r["steps_written"])
        bucket["schema_pass"].append(r["schema_pass"])
        bucket["schema_fail"].append(r["schema_fail"])

    per_hk_summary = {}
    for (hid, K), b in per_hk.items():
        per_hk_summary[f"{hid}_K{K}"] = {
            "n_runs": b["n_runs"],
            "mean_tokens": round(statistics.fmean(b["tokens"]), 1),
            "mean_latency_s": round(statistics.fmean(b["latency"]), 2),
            "mean_steps": round(statistics.fmean(b["steps"]), 2),
            "total_schema_pass": sum(b["schema_pass"]),
            "total_schema_fail": sum(b["schema_fail"]),
        }

    elapsed = time.time() - t_start
    overall = {
        "phase": "DAY8_long_horizon_K20",
        "spec": "K∈{12,16,20} on HIBench-Code v0_toy 8 task × 6 harness × seed=42",
        "benchmark": "HIBench-Code-v0_toy",
        "K_values_new": K_VALUES_NEW,
        "seed": SEED,
        "n_expected": n_expected,
        "n_runs_recorded": len(runs),
        "n_crashed": len(crashed),
        "total_steps": total_steps,
        "total_schema_pass": total_pass,
        "total_schema_fail": total_fail_schema,
        "schema_pass_rate": (total_pass / total_steps) if total_steps else 0.0,
        "total_tokens": total_tokens,
        "elapsed_s": round(elapsed, 1),
        "ended_jst": datetime.now(JST).isoformat(timespec="seconds"),
        "tasks": [t["task_id"] for t in tasks],
        "harnesses": list(harnesses.keys()),
        "per_harness_K": per_hk_summary,
        "crashed": crashed,
    }
    out_summary = out / "long_horizon_summary.json"
    out_summary.write_text(json.dumps(overall, ensure_ascii=False, indent=2))
    print(f"\n=== summary written: {out_summary} ===")
    print(json.dumps({
        k: v for k, v in overall.items()
        if k not in ("per_harness_K", "crashed", "tasks")
    }, ensure_ascii=False, indent=2))
    return 0 if (len(crashed) == 0 and len(runs) == n_expected) else 1


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="logs/long_horizon_K20")
    p.add_argument("--n-tasks", type=int, default=None)
    a = p.parse_args()
    sys.exit(main(a.out, a.n_tasks))
