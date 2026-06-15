"""anchor_2: HIBench-Code v0 toy 8 tasks × K∈{1,3,5,8} 在 H0 Raw harness 下
端到端跑通; 全部 step 的 belief 满足 readme §9.1 schema; logs 满足 §15 JSONL
格式 (100%).

每个 (task, K) 写一个独立的 JSONL 文件; 总 summary 写到
SETUP_DAY1_anchor2_summary.json。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from skeleton.benchmark.hibench_loader import load_tasks  # noqa: E402
from skeleton.core.jsonl_logger import JSONLLogger  # noqa: E402
from skeleton.core.llm_client import LLMClient  # noqa: E402
from skeleton.core.rollout import run_kstep_rollout  # noqa: E402
from skeleton.harnesses.h0_raw import H0RawHarness  # noqa: E402

JST = timezone(timedelta(hours=9))
K_VALUES = [1, 3, 5, 8]


def main(out_dir: str, tasks_path: str | None, seed: int, n_tasks: int | None) -> int:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tasks = load_tasks(tasks_path)
    if n_tasks is not None:
        tasks = tasks[:n_tasks]

    llm = LLMClient(min_interval_s=0.35, max_retries=3)
    harness = H0RawHarness()

    t_start = time.time()
    run_summaries: list[dict] = []
    total_steps = 0
    total_schema_pass = 0
    total_schema_fail = 0
    total_tokens = 0
    total_llm_calls = 0
    crashed: list[dict] = []

    for task in tasks:
        for K in K_VALUES:
            log_path = out / f"H0_raw_{task['task_id']}_K{K}_seed{seed}.jsonl"
            logger = JSONLLogger(log_path)
            try:
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
                summary["log_path"] = str(log_path)
                run_summaries.append(summary)
                total_steps += summary["steps_written"]
                total_schema_pass += summary["schema_pass"]
                total_schema_fail += summary["schema_fail"]
                total_tokens += summary["total_tokens"]
                total_llm_calls += summary["llm_calls"]
                print(
                    f"  [{task['task_id']:<28s} K={K}] "
                    f"steps={summary['steps_written']} "
                    f"schema_ok={summary['schema_pass']}/{summary['steps_written']} "
                    f"tokens={summary['total_tokens']} "
                    f"latency={summary['total_latency_s']:.1f}s",
                    flush=True,
                )
            except Exception as e:  # noqa: BLE001
                err = {
                    "task_id": task["task_id"],
                    "K": K,
                    "error": f"{type(e).__name__}: {e}",
                    "log_path": str(log_path),
                }
                crashed.append(err)
                print(f"  [CRASH] {err}", flush=True)

    overall = {
        "anchor": "anchor_2",
        "benchmark": "HIBench-Code-v0_toy",
        "harness": "H0_raw",
        "K_values": K_VALUES,
        "n_tasks": len(tasks),
        "n_runs_expected": len(tasks) * len(K_VALUES),
        "n_runs_completed": len(run_summaries),
        "n_runs_crashed": len(crashed),
        "total_steps_written": total_steps,
        "total_schema_pass": total_schema_pass,
        "total_schema_fail": total_schema_fail,
        "schema_pass_rate": (
            total_schema_pass / total_steps if total_steps > 0 else 0.0
        ),
        "total_llm_calls": total_llm_calls,
        "total_tokens": total_tokens,
        "elapsed_s": round(time.time() - t_start, 2),
        "ended_jst": datetime.now(JST).isoformat(timespec="seconds"),
        "anchor_passed": (
            len(crashed) == 0
            and total_steps > 0
            and total_schema_pass == total_steps
        ),
        "crashed": crashed,
        "runs": run_summaries,
    }
    summary_path = out / "anchor2_summary.json"
    summary_path.write_text(json.dumps(overall, ensure_ascii=False, indent=2))
    print("\n=== anchor_2 summary ===")
    print(json.dumps({k: v for k, v in overall.items() if k not in ("runs", "crashed")}, ensure_ascii=False, indent=2))
    if crashed:
        print("crashes:", json.dumps(crashed, ensure_ascii=False, indent=2))
    return 0 if overall["anchor_passed"] else 1


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="logs/anchor2_h0_smoke")
    p.add_argument("--tasks", default=None, help="explicit tasks.json path")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-tasks", type=int, default=None)
    a = p.parse_args()
    sys.exit(main(a.out, a.tasks, a.seed, a.n_tasks))
