"""DAY3 Step A: Phase 1 main table = 6 harness × 8 toy task × {K=1,3,5,8} × 3 seed
= 576 run.

每 run 写 logs/phase1_main/<harness>_<task>_K<k>_seed<s>.jsonl (K+1 step).
跑完写 phase1_summary.json 含:
- per-(harness, K) 平均 tokens / latency / steps
- per-(harness, K) D_belief 5 分量 group-mean (相对 H0_raw 同 task 同 K 同 seed 的 baseline)
- 总 LLM call / tokens / wall / est cost
- 576 run 完成率 (任何 fail 全 stack 报)
- toy_007 H2 dominance 多 seed 复现 check (跨 3 seed H2 是否依然出现 risk: high→medium / failure_mode: destructive_action→none)

不算 G1 strict test (那是 ds 的活, 用升级版 D_arrival/D_growth).
不中断 on fail: 单 run crash 记进 crashed[] 但 main loop 继续.

进度日志: 每完成 1 run 打 1 行; 每完成 48 run (= 1/12) 打 checkpoint 行 + flush.
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
from metrics.d_belief import d_belief_components  # noqa: E402

JST = timezone(timedelta(hours=9))

K_VALUES = [1, 3, 5, 8]
SEEDS = [42, 43, 44]


def _read_last_belief(p: Path) -> dict:
    last = ""
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            s = line.strip()
            if s:
                last = s
    return json.loads(last)["belief_output"]


def main(out_dir: str, n_tasks: int | None) -> int:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tasks = load_tasks()
    if n_tasks is not None:
        tasks = tasks[:n_tasks]
    harnesses = {hid: cls() for hid, cls in HARNESS_REGISTRY.items()}

    llm = LLMClient(min_interval_s=0.35, max_retries=3)

    n_expected = len(tasks) * len(K_VALUES) * len(SEEDS) * len(harnesses)
    print(f"phase1_main starting: {len(tasks)} task × {len(K_VALUES)} K × {len(SEEDS)} seed × "
          f"{len(harnesses)} harness = {n_expected} run", flush=True)
    t_start = time.time()
    runs: list[dict] = []
    crashed: list[dict] = []
    total_steps = total_pass = total_fail_schema = 0
    total_tokens = 0
    total_calls = 0
    completed = 0

    # iterate task-major so each task's all 24 runs (6 har × 4 K × 3 seed) finish
    # before moving on; this makes interrupted partial states easy to resume.
    # within a task, iterate seed→harness→K so seeds interleave (less risk of
    # one seed dominating context-cache patterns).
    for task in tasks:
        for seed in SEEDS:
            for hid, harness in harnesses.items():
                for K in K_VALUES:
                    log_path = out / f"{hid}_{task['task_id']}_K{K}_seed{seed}.jsonl"
                    if log_path.exists() and log_path.stat().st_size > 0:
                        # resume-skip: someone re-ran us; reuse existing log
                        try:
                            with log_path.open("r", encoding="utf-8") as fh:
                                lines = [json.loads(l) for l in fh if l.strip()]
                            if len(lines) >= K + 1:
                                runs.append({
                                    "run_id": lines[-1].get("run_id", ""),
                                    "task_id": task["task_id"], "harness_id": hid,
                                    "horizon": K, "seed": seed,
                                    "steps_written": len(lines),
                                    "schema_pass": sum(1 for l in lines if not l.get("schema_fail", False)),
                                    "schema_fail": sum(1 for l in lines if l.get("schema_fail", False)),
                                    "llm_calls": sum(1 for l in lines if not l.get("schema_fail", False)),
                                    "total_tokens": sum(l.get("llm_stats", {}).get("total_tokens", 0) for l in lines),
                                    "total_latency_s": sum(l.get("llm_stats", {}).get("latency_s", 0.0) for l in lines),
                                    "log_path": str(log_path),
                                    "resumed": True,
                                })
                                total_steps += len(lines)
                                total_pass += runs[-1]["schema_pass"]
                                total_fail_schema += runs[-1]["schema_fail"]
                                total_tokens += runs[-1]["total_tokens"]
                                total_calls += runs[-1]["llm_calls"]
                                completed += 1
                                continue
                        except Exception:  # noqa: BLE001
                            log_path.unlink(missing_ok=True)

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
                        runs.append(summary)
                        total_steps += summary["steps_written"]
                        total_pass += summary["schema_pass"]
                        total_fail_schema += summary["schema_fail"]
                        total_tokens += summary["total_tokens"]
                        total_calls += summary["llm_calls"]
                    except Exception as e:  # noqa: BLE001
                        crashed.append({
                            "task_id": task["task_id"], "harness_id": hid,
                            "K": K, "seed": seed,
                            "error": f"{type(e).__name__}: {e}",
                            "traceback": traceback.format_exc()[:500],
                        })
                        print(f"  [CRASH] {task['task_id']} {hid} K={K} seed={seed}: {e}",
                              flush=True)

                    completed += 1
                    elapsed = time.time() - t_start
                    if completed % 24 == 0 or completed == n_expected:
                        pct = 100.0 * completed / n_expected
                        eta = elapsed / completed * (n_expected - completed)
                        print(
                            f"  [{completed:>3d}/{n_expected}] {pct:5.1f}% "
                            f"elapsed={elapsed/60:.1f}min eta={eta/60:.1f}min "
                            f"steps={total_steps} schema_ok={total_pass}/{total_steps} "
                            f"tokens={total_tokens/1000:.1f}K "
                            f"crashed={len(crashed)}",
                            flush=True,
                        )

    elapsed = time.time() - t_start

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

    # ---- D_belief 5-component group-mean, per (paired_harness, K) ----
    # for each (task, K, seed), compute D(H0, Hx) for x in {H1..H5}; group mean across (task, seed)
    print("=== computing D_belief 5 分量 group-mean ===", flush=True)
    pairwise: dict[tuple[str, int], list[dict]] = {}
    target_pairs = [
        ("H0_raw", h) for h in
        ("H1_structured", "H2_risk_gated", "H3_repair_heavy",
         "H4_verification_selective", "H5_cost_aware")
    ]
    # we need to load the final belief of each run; build index first
    bel_idx: dict[tuple[str, str, int, int], dict] = {}
    for task in tasks:
        for seed in SEEDS:
            for K in K_VALUES:
                for hid in harnesses:
                    log_path = out / f"{hid}_{task['task_id']}_K{K}_seed{seed}.jsonl"
                    if not log_path.exists() or log_path.stat().st_size == 0:
                        continue
                    try:
                        bel_idx[(hid, task["task_id"], K, seed)] = _read_last_belief(log_path)
                    except Exception:  # noqa: BLE001
                        pass

    for (h0, hx) in target_pairs:
        for K in K_VALUES:
            comps: list[dict] = []
            for task in tasks:
                for seed in SEEDS:
                    a = bel_idx.get((h0, task["task_id"], K, seed))
                    b = bel_idx.get((hx, task["task_id"], K, seed))
                    if a is None or b is None:
                        continue
                    c = d_belief_components(a, b)
                    c["task_id"] = task["task_id"]
                    c["seed"] = seed
                    comps.append(c)
            pairwise[(hx, K)] = comps

    pairwise_summary = {}
    for (hx, K), comps in pairwise.items():
        if not comps:
            pairwise_summary[f"H0_vs_{hx}_K{K}"] = {"n": 0}
            continue
        agg = {
            "n": len(comps),
            "D_belief_mean": round(statistics.fmean(c["D_belief"] for c in comps), 4),
            "D_belief_std": round(statistics.pstdev(c["D_belief"] for c in comps), 4)
                            if len(comps) > 1 else 0.0,
            "cat_mismatch_mean": round(statistics.fmean(c["cat_mismatch"] for c in comps), 4),
            "failure_mode_mismatch_mean": round(
                statistics.fmean(c["failure_mode_mismatch"] for c in comps), 4),
            "set_distance_mean": round(statistics.fmean(c["set_distance"] for c in comps), 4),
            "num_distance_mean": round(statistics.fmean(c["num_distance"] for c in comps), 4),
            "action_mismatch_mean": round(statistics.fmean(c["action_mismatch"] for c in comps), 4),
        }
        pairwise_summary[f"H0_vs_{hx}_K{K}"] = agg

    # ---- H2 dominance check on toy_007 across 3 seeds × all K ----
    h2_check_t007 = {}
    for seed in SEEDS:
        for K in K_VALUES:
            b0 = bel_idx.get(("H0_raw", "toy_007_destructive_action_trap", K, seed))
            b2 = bel_idx.get(("H2_risk_gated", "toy_007_destructive_action_trap", K, seed))
            if b0 and b2:
                bs0 = b0["belief_state"]
                bs2 = b2["belief_state"]
                pf0 = b0["predicted_future"]
                pf2 = b2["predicted_future"]
                h2_check_t007[f"K{K}_seed{seed}"] = {
                    "H0_risk": bs0["risk_state"],
                    "H2_risk": bs2["risk_state"],
                    "H0_failure_mode": bs0["likely_failure_mode"],
                    "H2_failure_mode": bs2["likely_failure_mode"],
                    "H0_P_success": pf0["success_probability"],
                    "H2_P_success": pf2["success_probability"],
                    "censorship_signal": (
                        bs0["likely_failure_mode"] == "destructive_action"
                        and bs2["likely_failure_mode"] != "destructive_action"
                    ),
                }

    n_censor = sum(1 for v in h2_check_t007.values() if v.get("censorship_signal"))
    n_check = len(h2_check_t007)

    # ---- final summary ----
    overall = {
        "phase": "DAY3_step_A",
        "spec": "Phase 1 main table 6 har × 8 task × {1,3,5,8} × 3 seed",
        "benchmark": "HIBench-Code-v0_toy",
        "n_expected": n_expected,
        "n_completed": completed,
        "n_runs_recorded": len(runs),
        "n_crashed": len(crashed),
        "completion_rate": round(len(runs) / n_expected, 4),
        "total_steps": total_steps,
        "total_schema_pass": total_pass,
        "total_schema_fail": total_fail_schema,
        "schema_pass_rate": (total_pass / total_steps) if total_steps else 0.0,
        "total_llm_calls": total_calls,
        "total_tokens": total_tokens,
        "elapsed_s": round(elapsed, 1),
        "ended_jst": datetime.now(JST).isoformat(timespec="seconds"),
        "seeds": SEEDS,
        "K_values": K_VALUES,
        "harnesses": list(harnesses.keys()),
        "tasks": [t["task_id"] for t in tasks],
        "per_harness_K": per_hk_summary,
        "pairwise_d_belief_mean": pairwise_summary,
        "h2_censorship_toy007_check": {
            "n_data_points": n_check,
            "n_censorship_signal_present": n_censor,
            "censorship_rate": (n_censor / n_check) if n_check else 0.0,
            "per_point": h2_check_t007,
        },
        "crashed": crashed,
    }
    summary_path = out / "phase1_summary.json"
    summary_path.write_text(json.dumps(overall, ensure_ascii=False, indent=2))

    # print only key fields (full file on disk)
    print("\n=== Phase 1 summary ===", flush=True)
    print(json.dumps({
        k: v for k, v in overall.items()
        if k not in ("per_harness_K", "pairwise_d_belief_mean",
                     "h2_censorship_toy007_check", "crashed")
    }, ensure_ascii=False, indent=2))

    return 0 if (len(crashed) == 0 and completed == n_expected) else 1


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="logs/phase1_main")
    p.add_argument("--n-tasks", type=int, default=None)
    a = p.parse_args()
    sys.exit(main(a.out, a.n_tasks))
