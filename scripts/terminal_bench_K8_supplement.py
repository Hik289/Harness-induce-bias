"""Day 8 Task 2a: Terminal-Bench K=8 supplement.

Day 6 G2 已经跑 K∈{1, 5}. 现在补 K=8.
10 task × 6 harness × K=8 × seed=42 = 60 new run.
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

from skeleton.benchmark.terminal_bench_adapter import load_terminal_bench_tasks  # noqa: E402
from skeleton.core.jsonl_logger import JSONLLogger  # noqa: E402
from skeleton.core.llm_client import LLMClient  # noqa: E402
from skeleton.core.rollout import run_kstep_rollout  # noqa: E402
from skeleton.harnesses import HARNESS_REGISTRY  # noqa: E402

JST = timezone(timedelta(hours=9))


def main(out_dir: str) -> int:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tasks = load_terminal_bench_tasks(n=10, seed=42)
    harnesses = {hid: cls() for hid, cls in HARNESS_REGISTRY.items()}
    llm = LLMClient(min_interval_s=0.35, max_retries=3)

    K = 8
    SEED = 42
    n_expected = len(tasks) * len(harnesses)
    print(f"[TB K=8 supplement] {len(tasks)} task × {len(harnesses)} harness × K=8 = {n_expected} run", flush=True)
    t_start = time.time()
    runs: list[dict] = []
    crashed: list[dict] = []
    total_tokens = 0
    completed = 0

    for task in tasks:
        for hid, harness in harnesses.items():
            log_path = out / f"{hid}_{task['task_id']}_K{K}_seed{SEED}.jsonl"
            if log_path.exists() and log_path.stat().st_size > 0:
                try:
                    with log_path.open("r", encoding="utf-8") as fh:
                        lines = [l for l in fh if l.strip()]
                    if len(lines) >= K + 1:
                        runs.append({
                            "task_id": task["task_id"], "harness_id": hid,
                            "horizon": K, "seed": SEED, "steps_written": len(lines),
                            "schema_pass": K + 1, "schema_fail": 0,
                            "total_tokens": 0, "total_latency_s": 0.0,
                            "log_path": str(log_path), "resumed": True,
                        })
                        completed += 1
                        continue
                except Exception:  # noqa: BLE001
                    log_path.unlink(missing_ok=True)
            logger = JSONLLogger(log_path)
            try:
                s = run_kstep_rollout(
                    task=task, harness=harness, llm=llm, horizon=K, logger=logger,
                    benchmark_id="Terminal-Bench-v0", environment_id="E_default_tb",
                    seed=SEED,
                )
                s["log_path"] = str(log_path)
                runs.append(s)
                total_tokens += s["total_tokens"]
            except Exception as e:  # noqa: BLE001
                crashed.append({"task_id": task["task_id"], "harness_id": hid, "error": str(e)})
                print(f"  [CRASH] {task['task_id']} {hid}: {e}", flush=True)
            completed += 1
            if completed % 10 == 0 or completed == n_expected:
                elapsed = time.time() - t_start
                eta = elapsed / completed * (n_expected - completed)
                print(f"  [{completed}/{n_expected}] elapsed={elapsed/60:.1f}min eta={eta/60:.1f}min tokens={total_tokens/1000:.1f}K", flush=True)

    elapsed = time.time() - t_start
    overall = {
        "phase": "DAY8_TB_K8_supplement",
        "benchmark": "Terminal-Bench-v0", "K": K, "seed": SEED,
        "n_expected": n_expected, "n_runs_recorded": len(runs),
        "n_crashed": len(crashed),
        "total_tokens_new": total_tokens,
        "elapsed_s": round(elapsed, 1),
        "ended_jst": datetime.now(JST).isoformat(timespec="seconds"),
        "tasks": [t["task_id"] for t in tasks],
        "crashed": crashed,
    }
    (out / "tb_k8_summary.json").write_text(json.dumps(overall, ensure_ascii=False, indent=2))
    print(f"\n=== summary written ===")
    print(json.dumps({k: v for k, v in overall.items() if k not in ("crashed",)}, ensure_ascii=False, indent=2))
    return 0 if (len(crashed) == 0 and len(runs) == n_expected) else 1


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="logs/tb_K8_supplement")
    a = p.parse_args()
    sys.exit(main(a.out))
