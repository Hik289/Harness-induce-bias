"""Day 6 G2: Terminal-Bench descriptive replication.

Director 派单 §8.3 最便宜方案:
- 10 TB task (stratified by difficulty), seed=42 single, K ∈ {1, 5}
- 6 harness baseline + 5 BIWM-wrapped + BIWM-full + post-hoc cross-harness aligned
- 描述性 only, 无 p / Bonferroni / CI

阶段:
A. 6 harness × 10 task × K∈{1,5} × seed=42 = **120 base run** → Table 1 G2 复现
B. 5 wrapper + BIWM-full × 10 task × K=5 × seed=42 = **60 BIWM run** → Table 2 G2 复现
C. post-hoc cross-harness alignment (n=20: 10 task × 2 K) → BIWM-6/7 G2 复现

不算 G1 / G3 strict test. 只报描述性 mean Δ + +/- consistency.
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
from skeleton.biwm import (  # noqa: E402
    CanonicalBeliefWrapper, BlockedActionLogWrapper, RepairUnrolledWrapper,
    VerificationMaskWrapper, ShadowExecutionWrapper, biwm_full,
    align_beliefs, self_consistency_score,
)
from metrics.d_belief import d_belief_components  # noqa: E402


JST = timezone(timedelta(hours=9))
K_VALUES = [1, 5]
BIWM_K = 5
SEED = 42
N_TASKS = 10


GROUP_A_BIWM = [
    ("BIWM1_canonical", "H1_structured", CanonicalBeliefWrapper),
    ("BIWM2_blocked_log", "H2_risk_gated", BlockedActionLogWrapper),
    ("BIWM3_repair_unrolled", "H3_repair_heavy", RepairUnrolledWrapper),
    ("BIWM4_verification_mask", "H4_verification_selective", VerificationMaskWrapper),
    ("BIWM5_shadow", "H5_cost_aware", ShadowExecutionWrapper),
]


def _read_last_belief(p: Path) -> dict:
    last = ""
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            s = line.strip()
            if s:
                last = s
    return json.loads(last)["belief_output"]


def main(out_dir: str) -> int:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tasks = load_terminal_bench_tasks(n=N_TASKS, seed=SEED)
    print(f"[g2] tasks: {[t['task_id'] for t in tasks]}", flush=True)

    llm = LLMClient(min_interval_s=0.35, max_retries=3)
    t0 = time.time()
    crashed: list[dict] = []
    total_new = 0
    total_tokens = 0

    # ============== Stage 1: 6 harness baseline (Table 1 G2) ==============
    print("\n=== Stage 1: 6 harness baseline 10 task × K∈{1,5} ===", flush=True)
    base_idx: dict[tuple[str, str, int], dict] = {}  # (hid, task_id, K) → belief_K
    base_summaries: list[dict] = []
    expected_base = len(tasks) * len(HARNESS_REGISTRY) * len(K_VALUES)
    done_base = 0
    for task in tasks:
        for hid, cls in HARNESS_REGISTRY.items():
            for K in K_VALUES:
                log_path = out / f"BASE_{hid}_{task['task_id']}_K{K}_seed{SEED}.jsonl"
                if log_path.exists() and log_path.stat().st_size > 0:
                    try:
                        belief = _read_last_belief(log_path)
                        base_idx[(hid, task["task_id"], K)] = belief
                        done_base += 1
                        continue
                    except Exception:  # noqa: BLE001
                        log_path.unlink(missing_ok=True)
                harness = cls()
                logger = JSONLLogger(log_path)
                try:
                    s = run_kstep_rollout(
                        task=task, harness=harness, llm=llm, horizon=K, logger=logger,
                        benchmark_id="Terminal-Bench-v0",
                        environment_id="E_default_tb", seed=SEED,
                    )
                    base_idx[(hid, task["task_id"], K)] = _read_last_belief(log_path)
                    base_summaries.append({**s, "log": str(log_path)})
                    total_new += 1
                    total_tokens += s["total_tokens"]
                except Exception as e:  # noqa: BLE001
                    crashed.append({
                        "stage": "base", "task_id": task["task_id"],
                        "hid": hid, "K": K, "error": f"{type(e).__name__}: {e}"
                    })
                    print(f"  [CRASH] base {hid} {task['task_id']} K{K}: {e}", flush=True)
                done_base += 1
                if done_base % 20 == 0:
                    elapsed = time.time() - t0
                    eta = elapsed / done_base * (expected_base - done_base)
                    print(f"  [base {done_base}/{expected_base}] elapsed={elapsed/60:.1f}min eta={eta/60:.1f}min tokens={total_tokens/1000:.0f}K", flush=True)

    # ----- Table 1 (G2) per-harness D mean by K (5 components) -----
    print("\n=== Table 1 (G2): D_belief 5 components per (Hx, K) pairs ===", flush=True)
    table1: dict[str, dict] = {}
    target_pairs = [
        ("H0_raw", h) for h in ("H1_structured", "H2_risk_gated", "H3_repair_heavy",
                                "H4_verification_selective", "H5_cost_aware")
    ]
    for h0, hx in target_pairs:
        for K in K_VALUES:
            entries: list[dict] = []
            for task in tasks:
                a = base_idx.get((h0, task["task_id"], K))
                b = base_idx.get((hx, task["task_id"], K))
                if a is None or b is None:
                    continue
                c = d_belief_components(a, b)
                entries.append({"task_id": task["task_id"], **c})
            if not entries:
                table1[f"H0_vs_{hx}_K{K}"] = {"n": 0}
                continue
            agg = {
                "n": len(entries),
                "D_belief_mean": round(statistics.fmean(e["D_belief"] for e in entries), 4),
                "D_belief_std": round(statistics.pstdev(e["D_belief"] for e in entries) if len(entries) > 1 else 0, 4),
                "cat_mismatch_mean": round(statistics.fmean(e["cat_mismatch"] for e in entries), 4),
                "failure_mode_mismatch_mean": round(statistics.fmean(e["failure_mode_mismatch"] for e in entries), 4),
                "set_distance_mean": round(statistics.fmean(e["set_distance"] for e in entries), 4),
                "num_distance_mean": round(statistics.fmean(e["num_distance"] for e in entries), 4),
                "action_mismatch_mean": round(statistics.fmean(e["action_mismatch"] for e in entries), 4),
            }
            table1[f"H0_vs_{hx}_K{K}"] = agg
            print(f"  H0 vs {hx:<28s} K={K}  n={agg['n']}  D={agg['D_belief_mean']:.3f}  cat={agg['cat_mismatch_mean']:.3f}  fail={agg['failure_mode_mismatch_mean']:.3f}  num={agg['num_distance_mean']:.3f}", flush=True)

    # Quick K-amplification check (descriptive, no p)
    print("\n=== G1 G2 描述性 K-放大 (D K5/K1 ratio per pair, scalar D) ===", flush=True)
    k_amp_table = {}
    for h0, hx in target_pairs:
        D_K1 = table1.get(f"H0_vs_{hx}_K1", {}).get("D_belief_mean")
        D_K5 = table1.get(f"H0_vs_{hx}_K5", {}).get("D_belief_mean")
        if D_K1 is None or D_K5 is None or D_K1 < 1e-6:
            k_amp_table[f"H0_vs_{hx}"] = {"D_K1": D_K1, "D_K5": D_K5}
            continue
        k_amp_table[f"H0_vs_{hx}"] = {
            "D_K1": D_K1, "D_K5": D_K5,
            "K5_minus_K1": round(D_K5 - D_K1, 4),
            "K5_over_K1": round(D_K5 / D_K1, 3),
        }
        print(f"  H0 vs {hx:<28s}  D_K1={D_K1:.3f}  D_K5={D_K5:.3f}  Δ=+{D_K5-D_K1:.3f}  ratio={D_K5/D_K1:.2f}x", flush=True)

    # ============== Stage 2: BIWM Group A (5 single wrappers) + Group B (BIWM-full) on K=5 ==============
    print("\n=== Stage 2: BIWM Group A (single) + Group B (full) at K=5 ===", flush=True)
    group_a_results: dict[str, dict] = {}
    for label, hx_id, wrapper_cls in GROUP_A_BIWM:
        per: list[dict] = []
        for task in tasks:
            b0 = base_idx.get(("H0_raw", task["task_id"], BIWM_K))
            bx_base = base_idx.get((hx_id, task["task_id"], BIWM_K))
            if b0 is None or bx_base is None:
                continue
            log_path = out / f"{label}_on_{hx_id}_{task['task_id']}_K{BIWM_K}_seed{SEED}.jsonl"
            if log_path.exists() and log_path.stat().st_size > 0:
                try:
                    biwm_belief = _read_last_belief(log_path)
                except Exception:  # noqa: BLE001
                    log_path.unlink(missing_ok=True)
                    biwm_belief = None
            else:
                biwm_belief = None
            if biwm_belief is None:
                wrapped = wrapper_cls(HARNESS_REGISTRY[hx_id]())
                wrapped.harness_id = f"{label}_on_{hx_id}"
                logger = JSONLLogger(log_path)
                try:
                    s = run_kstep_rollout(
                        task=task, harness=wrapped, llm=llm, horizon=BIWM_K, logger=logger,
                        benchmark_id="Terminal-Bench-v0",
                        environment_id="E_default_tb", seed=SEED,
                    )
                    biwm_belief = _read_last_belief(log_path)
                    total_new += 1
                    total_tokens += s["total_tokens"]
                except Exception as e:  # noqa: BLE001
                    crashed.append({"stage": "biwm_A", "label": label,
                                    "task_id": task["task_id"], "error": str(e)})
                    print(f"  [CRASH] {label} {task['task_id']}: {e}", flush=True)
                    continue
            d_base = d_belief_components(b0, bx_base)
            d_biwm = d_belief_components(b0, biwm_belief)
            per.append({
                "task_id": task["task_id"],
                "D_baseline": d_base["D_belief"],
                "D_biwm": d_biwm["D_belief"],
                "delta_D": d_biwm["D_belief"] - d_base["D_belief"],
                "cat_delta": d_biwm["cat_mismatch"] - d_base["cat_mismatch"],
                "fail_delta": d_biwm["failure_mode_mismatch"] - d_base["failure_mode_mismatch"],
                "num_delta": d_biwm["num_distance"] - d_base["num_distance"],
            })
        if not per:
            group_a_results[f"{label}_on_{hx_id}"] = {"n": 0}
            continue
        m_base = statistics.fmean(p["D_baseline"] for p in per)
        m_biwm = statistics.fmean(p["D_biwm"] for p in per)
        deltas = [p["delta_D"] for p in per]
        group_a_results[f"{label}_on_{hx_id}"] = {
            "n": len(per),
            "D_baseline_mean": round(m_base, 4),
            "D_biwm_mean": round(m_biwm, 4),
            "delta_D_mean": round(m_biwm - m_base, 4),
            "delta_D_std": round(statistics.pstdev(deltas) if len(deltas) > 1 else 0, 4),
            "n_delta_positive": sum(1 for d in deltas if d > 0),
            "n_delta_negative": sum(1 for d in deltas if d < 0),
            "consistency_ratio_positive": round(sum(1 for d in deltas if d > 0) / len(deltas), 3),
            "delta_cat_mean": round(statistics.fmean(p["cat_delta"] for p in per), 4),
            "delta_fail_mean": round(statistics.fmean(p["fail_delta"] for p in per), 4),
            "delta_num_mean": round(statistics.fmean(p["num_delta"] for p in per), 4),
        }
        print(f"  {label:<22s} on {hx_id:<28s} n={len(per)}  D_base={m_base:.3f}  D_biwm={m_biwm:.3f}  Δ={m_biwm-m_base:+.3f}  (+/-: {group_a_results[f'{label}_on_{hx_id}']['n_delta_positive']}/{group_a_results[f'{label}_on_{hx_id}']['n_delta_negative']})", flush=True)

    # Group B: BIWM-full
    print("\n=== Group B G2: BIWM-full stacking on 5 non-H0 base ===", flush=True)
    group_b_results: dict[str, dict] = {}
    for hx_id in ("H1_structured", "H2_risk_gated", "H3_repair_heavy",
                  "H4_verification_selective", "H5_cost_aware"):
        per: list[dict] = []
        for task in tasks:
            b0 = base_idx.get(("H0_raw", task["task_id"], BIWM_K))
            bx_base = base_idx.get((hx_id, task["task_id"], BIWM_K))
            if b0 is None or bx_base is None:
                continue
            log_path = out / f"BIWMfull_on_{hx_id}_{task['task_id']}_K{BIWM_K}_seed{SEED}.jsonl"
            biwm_belief = None
            if log_path.exists() and log_path.stat().st_size > 0:
                try:
                    biwm_belief = _read_last_belief(log_path)
                except Exception:
                    log_path.unlink(missing_ok=True)
            if biwm_belief is None:
                wrapped = biwm_full(HARNESS_REGISTRY[hx_id]())
                logger = JSONLLogger(log_path)
                try:
                    s = run_kstep_rollout(
                        task=task, harness=wrapped, llm=llm, horizon=BIWM_K, logger=logger,
                        benchmark_id="Terminal-Bench-v0",
                        environment_id="E_default_tb", seed=SEED,
                    )
                    biwm_belief = _read_last_belief(log_path)
                    total_new += 1
                    total_tokens += s["total_tokens"]
                except Exception as e:  # noqa: BLE001
                    crashed.append({"stage": "biwm_B", "hid": hx_id,
                                    "task_id": task["task_id"], "error": str(e)})
                    print(f"  [CRASH] BIWMfull {hx_id} {task['task_id']}: {e}", flush=True)
                    continue
            d_base = d_belief_components(b0, bx_base)
            d_full = d_belief_components(b0, biwm_belief)
            per.append({
                "task_id": task["task_id"],
                "D_baseline": d_base["D_belief"],
                "D_biwm_full": d_full["D_belief"],
                "delta_D": d_full["D_belief"] - d_base["D_belief"],
                "cat_delta": d_full["cat_mismatch"] - d_base["cat_mismatch"],
                "fail_delta": d_full["failure_mode_mismatch"] - d_base["failure_mode_mismatch"],
                "num_delta": d_full["num_distance"] - d_base["num_distance"],
            })
        if not per:
            group_b_results[hx_id] = {"n": 0}
            continue
        m_base = statistics.fmean(p["D_baseline"] for p in per)
        m_full = statistics.fmean(p["D_biwm_full"] for p in per)
        deltas = [p["delta_D"] for p in per]
        group_b_results[hx_id] = {
            "n": len(per),
            "D_baseline_mean": round(m_base, 4),
            "D_biwm_full_mean": round(m_full, 4),
            "delta_D_mean": round(m_full - m_base, 4),
            "delta_D_std": round(statistics.pstdev(deltas) if len(deltas) > 1 else 0, 4),
            "n_delta_positive": sum(1 for d in deltas if d > 0),
            "n_delta_negative": sum(1 for d in deltas if d < 0),
            "consistency_ratio_positive": round(sum(1 for d in deltas if d > 0) / len(deltas), 3),
            "delta_cat_mean": round(statistics.fmean(p["cat_delta"] for p in per), 4),
            "delta_fail_mean": round(statistics.fmean(p["fail_delta"] for p in per), 4),
            "delta_num_mean": round(statistics.fmean(p["num_delta"] for p in per), 4),
        }
        print(f"  BIWMfull on {hx_id:<28s} n={len(per)}  D_base={m_base:.3f}  D_full={m_full:.3f}  Δ={m_full-m_base:+.3f}  (+/-: {group_b_results[hx_id]['n_delta_positive']}/{group_b_results[hx_id]['n_delta_negative']})", flush=True)

    # ============== Stage 3: post-hoc Group C alignment ==============
    print("\n=== Group C G2: BIWM-6/7 cross-harness alignment (post-hoc) ===", flush=True)
    align_records: list[dict] = []
    for task in tasks:
        for K in K_VALUES:
            h0 = base_idx.get(("H0_raw", task["task_id"], K))
            if h0 is None:
                continue
            non_h0_views = [
                base_idx[(hid, task["task_id"], K)]
                for hid in HARNESS_REGISTRY if hid != "H0_raw"
                and (hid, task["task_id"], K) in base_idx
            ]
            if len(non_h0_views) < 2:
                continue
            aligned = align_beliefs(non_h0_views)
            consistency = self_consistency_score(non_h0_views)
            ds_base = [d_belief_components(h0, v)["D_belief"] for v in non_h0_views]
            d_align = d_belief_components(h0, aligned)["D_belief"]
            align_records.append({
                "task_id": task["task_id"], "K": K, "n_views": len(non_h0_views),
                "mean_D_baseline_5pair": statistics.fmean(ds_base),
                "D_H0_vs_aligned": d_align,
                "delta_D_align_minus_base": d_align - statistics.fmean(ds_base),
                "consistency_signal": consistency["signal"],
                "categorical_disagreement": consistency["categorical_disagreement"],
            })
    group_c: dict = {"n": len(align_records)}
    if align_records:
        for K in K_VALUES:
            sub = [r for r in align_records if r["K"] == K]
            if not sub:
                continue
            group_c[f"K{K}"] = {
                "n": len(sub),
                "D_baseline_5pair_mean": round(statistics.fmean(r["mean_D_baseline_5pair"] for r in sub), 4),
                "D_H0_vs_aligned_mean": round(statistics.fmean(r["D_H0_vs_aligned"] for r in sub), 4),
                "delta_mean": round(statistics.fmean(r["delta_D_align_minus_base"] for r in sub), 4),
                "categorical_disagreement_mean": round(statistics.fmean(r["categorical_disagreement"] for r in sub), 4),
            }
            v = group_c[f"K{K}"]
            print(f"  K={K}: n={v['n']}  D_base={v['D_baseline_5pair_mean']:.3f}  D_aligned={v['D_H0_vs_aligned_mean']:.3f}  Δ={v['delta_mean']:+.3f}  cat_disagree={v['categorical_disagreement_mean']:.2f}", flush=True)

    # ----- Final summary -----
    elapsed = time.time() - t0
    overall = {
        "phase": "DAY6_G2_TerminalBench",
        "spec": "Terminal-Bench v0 subset 10 task, K∈{1,5}, seed=42, descriptive only",
        "n_tasks": len(tasks),
        "seeds": [SEED],
        "K_values": K_VALUES,
        "harnesses": list(HARNESS_REGISTRY.keys()),
        "n_runs_new": total_new,
        "n_runs_crashed": len(crashed),
        "total_tokens_new": total_tokens,
        "elapsed_s": round(elapsed, 1),
        "ended_jst": datetime.now(JST).isoformat(timespec="seconds"),
        "tasks_loaded": [{"task_id": t["task_id"], "difficulty": t["difficulty"],
                          "category": t["category"]} for t in tasks],
        "table1_pairwise_D": table1,
        "g1_g2_K_amplification": k_amp_table,
        "group_A_single_components": group_a_results,
        "group_B_biwm_full": group_b_results,
        "group_C_alignment": group_c,
        "crashed": crashed,
    }
    out_summary = out / "g2_terminal_bench_summary.json"
    out_summary.write_text(json.dumps(overall, ensure_ascii=False, indent=2))
    print(f"\n=== summary written: {out_summary} ===")
    print(f"new runs: {total_new}, crashed: {len(crashed)}, tokens: {total_tokens:,}, elapsed: {elapsed:.0f}s")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="logs/g2_terminal_bench")
    a = p.parse_args()
    sys.exit(main(a.out))
