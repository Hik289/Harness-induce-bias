"""Day 5 extension: Group A + Group B at n=24 (8 task × 3 seed × K=5).

复用已有 anchor5_biwm_smoke 的 30 个 jsonl (3 task × 5 wrapper × seed=42 +
3 task × 5 BIWMfull × seed=42), 只跑剩余 cells.

Group A: BIWM-{1,2,3,4,5} wrapper × {pair: H1/H2/H3/H4/H5} × 8 task × 3 seed × K=5
Group B: BIWM-full(Hx) × Hx ∈ {H1,...,H5} × 8 task × 3 seed × K=5

每个 BIWM 跑完后, 从 Phase 1 主表读 D_base = D(H0_K5_seed, Hx_K5_seed), 然后
读 BIWM jsonl 的 belief_K = D(H0_K5_seed, BIWM(Hx)_K5_seed). 计算 ΔD per task
+ 5 分量.

Director 派单要求只做描述性, 不算 p 值; 报 mean / per-task delta / 5 分量.
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

from skeleton.benchmark.hibench_loader import load_tasks  # noqa: E402
from skeleton.core.jsonl_logger import JSONLLogger  # noqa: E402
from skeleton.core.llm_client import LLMClient  # noqa: E402
from skeleton.core.rollout import run_kstep_rollout  # noqa: E402
from skeleton.harnesses import HARNESS_REGISTRY  # noqa: E402
from skeleton.biwm import (  # noqa: E402
    CanonicalBeliefWrapper, BlockedActionLogWrapper, RepairUnrolledWrapper,
    VerificationMaskWrapper, ShadowExecutionWrapper, biwm_full,
)
from metrics.d_belief import d_belief_components  # noqa: E402

JST = timezone(timedelta(hours=9))


def _read_last_belief(p: Path) -> dict:
    last = ""
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            s = line.strip()
            if s:
                last = s
    return json.loads(last)["belief_output"]


def _read_phase1(p1: Path, hid: str, tid: str, K: int, seed: int):
    f = p1 / f"{hid}_{tid}_K{K}_seed{seed}.jsonl"
    if not f.exists() or f.stat().st_size == 0:
        return None
    try:
        return _read_last_belief(f)
    except Exception:  # noqa: BLE001
        return None


# Group A mapping (component label, pair Hx, wrapper_cls)
GROUP_A = [
    ("BIWM1_canonical", "H1_structured", CanonicalBeliefWrapper),
    ("BIWM2_blocked_log", "H2_risk_gated", BlockedActionLogWrapper),
    ("BIWM3_repair_unrolled", "H3_repair_heavy", RepairUnrolledWrapper),
    ("BIWM4_verification_mask", "H4_verification_selective", VerificationMaskWrapper),
    ("BIWM5_shadow", "H5_cost_aware", ShadowExecutionWrapper),
]


def main(out_dir: str, reuse_dir: str, phase1_dir: str, K: int, seeds: list[int]) -> int:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    reuse = Path(reuse_dir)
    phase1 = Path(phase1_dir)
    tasks = load_tasks()
    llm = LLMClient(min_interval_s=0.35, max_retries=3)

    print(f"[extend_n24] starting: 8 task × {len(seeds)} seed × K={K}", flush=True)
    print(f"  reuse from: {reuse}", flush=True)
    print(f"  phase1 from: {phase1}", flush=True)
    t_start = time.time()

    total_new = 0
    total_reused = 0
    total_tokens = 0

    # ----- Group A -----
    print("\n=== Group A (extended): single-component wrappers ===", flush=True)
    group_a_results = {}
    for label, hx_id, wrapper_cls in GROUP_A:
        per = []
        for task in tasks:
            for seed in seeds:
                # baselines from phase1
                b0 = _read_phase1(phase1, "H0_raw", task["task_id"], K, seed)
                bx_base = _read_phase1(phase1, hx_id, task["task_id"], K, seed)
                if b0 is None or bx_base is None:
                    print(f"  skip {label} {task['task_id']} seed{seed}: missing phase1 belief", flush=True)
                    continue
                # BIWM rollout: reuse if exists in `reuse` (anchor5_biwm_smoke), else new run in `out`
                fname = f"{label}_on_{hx_id}_{hx_id}_{task['task_id']}_K{K}_seed{seed}.jsonl"
                reuse_path = reuse / fname
                out_path = out / fname
                if reuse_path.exists() and reuse_path.stat().st_size > 0:
                    biwm_belief = _read_last_belief(reuse_path)
                    biwm_tokens = 0
                    total_reused += 1
                elif out_path.exists() and out_path.stat().st_size > 0:
                    biwm_belief = _read_last_belief(out_path)
                    biwm_tokens = 0
                else:
                    wrapped = wrapper_cls(HARNESS_REGISTRY[hx_id]())
                    wrapped.harness_id = f"{label}_on_{hx_id}_{hx_id}"
                    logger = JSONLLogger(out_path)
                    try:
                        summary = run_kstep_rollout(
                            task=task, harness=wrapped, llm=llm, horizon=K, logger=logger,
                            benchmark_id="HIBench-Code-v0_toy",
                            environment_id="E_default_v0", seed=seed,
                        )
                        biwm_belief = _read_last_belief(out_path)
                        biwm_tokens = summary["total_tokens"]
                        total_tokens += biwm_tokens
                        total_new += 1
                    except Exception as e:  # noqa: BLE001
                        print(f"  CRASH {label} {task['task_id']} seed{seed}: {e}", flush=True)
                        continue

                d_base = d_belief_components(b0, bx_base)
                d_biwm = d_belief_components(b0, biwm_belief)
                per.append({
                    "task_id": task["task_id"], "seed": seed,
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
        # SE of paired delta
        deltas = [p["delta_D"] for p in per]
        s_delta = statistics.pstdev(deltas) if len(deltas) > 1 else 0.0
        group_a_results[f"{label}_on_{hx_id}"] = {
            "n": len(per),
            "D_baseline_mean": round(m_base, 4),
            "D_biwm_mean": round(m_biwm, 4),
            "delta_D_mean": round(m_biwm - m_base, 4),
            "delta_D_std": round(s_delta, 4),
            "n_delta_positive": sum(1 for d in deltas if d > 0),
            "n_delta_negative": sum(1 for d in deltas if d < 0),
            "delta_cat_mean": round(statistics.fmean(p["cat_delta"] for p in per), 4),
            "delta_fail_mean": round(statistics.fmean(p["fail_delta"] for p in per), 4),
            "delta_num_mean": round(statistics.fmean(p["num_delta"] for p in per), 4),
        }
        print(f"  {label:<22s} on {hx_id:<28s} n={len(per)}  D_base={m_base:.3f}  D_biwm={m_biwm:.3f}  Δ={m_biwm-m_base:+.3f} (std={s_delta:.3f}, +/-: {group_a_results[f'{label}_on_{hx_id}']['n_delta_positive']}/{group_a_results[f'{label}_on_{hx_id}']['n_delta_negative']})", flush=True)

    # ----- Group B -----
    print("\n=== Group B (extended): BIWM-full stacks ===", flush=True)
    group_b_results = {}
    for hx_id in ("H1_structured", "H2_risk_gated", "H3_repair_heavy",
                  "H4_verification_selective", "H5_cost_aware"):
        per = []
        for task in tasks:
            for seed in seeds:
                b0 = _read_phase1(phase1, "H0_raw", task["task_id"], K, seed)
                bx_base = _read_phase1(phase1, hx_id, task["task_id"], K, seed)
                if b0 is None or bx_base is None:
                    continue
                fname = f"BIWMfull_{hx_id}_{task['task_id']}_K{K}_seed{seed}.jsonl"
                reuse_path = reuse / fname
                out_path = out / fname
                if reuse_path.exists() and reuse_path.stat().st_size > 0:
                    biwm_belief = _read_last_belief(reuse_path)
                    total_reused += 1
                elif out_path.exists() and out_path.stat().st_size > 0:
                    biwm_belief = _read_last_belief(out_path)
                else:
                    wrapped = biwm_full(HARNESS_REGISTRY[hx_id]())
                    logger = JSONLLogger(out_path)
                    try:
                        summary = run_kstep_rollout(
                            task=task, harness=wrapped, llm=llm, horizon=K, logger=logger,
                            benchmark_id="HIBench-Code-v0_toy",
                            environment_id="E_default_v0", seed=seed,
                        )
                        biwm_belief = _read_last_belief(out_path)
                        total_tokens += summary["total_tokens"]
                        total_new += 1
                    except Exception as e:  # noqa: BLE001
                        print(f"  CRASH BIWMfull {hx_id} {task['task_id']} seed{seed}: {e}", flush=True)
                        continue
                d_base = d_belief_components(b0, bx_base)
                d_full = d_belief_components(b0, biwm_belief)
                per.append({
                    "task_id": task["task_id"], "seed": seed,
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
        s_delta = statistics.pstdev(deltas) if len(deltas) > 1 else 0.0
        group_b_results[hx_id] = {
            "n": len(per),
            "D_baseline_mean": round(m_base, 4),
            "D_biwm_full_mean": round(m_full, 4),
            "delta_D_mean": round(m_full - m_base, 4),
            "delta_D_std": round(s_delta, 4),
            "n_delta_positive": sum(1 for d in deltas if d > 0),
            "n_delta_negative": sum(1 for d in deltas if d < 0),
            "delta_cat_mean": round(statistics.fmean(p["cat_delta"] for p in per), 4),
            "delta_fail_mean": round(statistics.fmean(p["fail_delta"] for p in per), 4),
            "delta_num_mean": round(statistics.fmean(p["num_delta"] for p in per), 4),
        }
        print(f"  BIWMfull on {hx_id:<28s} n={len(per)}  D_base={m_base:.3f}  D_full={m_full:.3f}  Δ={m_full-m_base:+.3f} (std={s_delta:.3f}, +/-: {group_b_results[hx_id]['n_delta_positive']}/{group_b_results[hx_id]['n_delta_negative']})", flush=True)

    elapsed = time.time() - t_start
    overall = {
        "phase": "DAY5_BIWM_extend",
        "spec": "Group A/B extended to n=24 (8 task × 3 seed × K=5)",
        "K": K,
        "seeds": seeds,
        "total_new_runs": total_new,
        "total_reused_runs": total_reused,
        "total_new_tokens": total_tokens,
        "elapsed_s": round(elapsed, 2),
        "ended_jst": datetime.now(JST).isoformat(timespec="seconds"),
        "group_A_single_components": group_a_results,
        "group_B_biwm_full": group_b_results,
    }
    out_summary = out / "anchor5_extend_summary.json"
    out_summary.write_text(json.dumps(overall, ensure_ascii=False, indent=2))
    print(f"\n=== summary written: {out_summary} ===")
    print(f"new runs: {total_new}, reused: {total_reused}, tokens: {total_tokens}, elapsed: {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="logs/day5_biwm_extend")
    p.add_argument("--reuse", default="logs/anchor5_biwm_smoke")
    p.add_argument("--phase1", default="logs/phase1_main")
    p.add_argument("--K", type=int, default=5)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    a = p.parse_args()
    sys.exit(main(a.out, a.reuse, a.phase1, a.K, a.seeds))
