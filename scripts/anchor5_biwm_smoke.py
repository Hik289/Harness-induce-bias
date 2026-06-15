"""anchor_5: BIWM 描述性 smoke (Director 02:30 UTC 决策 — 只做描述, 不做统计).

设计:
3 类对比 (每类只跑 smoke 子集, 不重跑全 Phase 1 主表):

Group A: 单组件 wrapper (BIWM-1, 2, 3, 4, 5) 套在最相关 base harness 上
  - H0 vs H1: 加 BIWM-1 (canonical) → 看 D 是否下降 (canonicalization 应当
    把 H0 raw 和 H1 structured 的差异拉平)
  - H0 vs H2: 加 BIWM-2 (blocked_log) → 看 D / failure_mode_mismatch 是否
    下降 (告诉 LLM 哪些 branch 被 censor)
  - H0 vs H3: 加 BIWM-3 (repair_unrolled) → 看 D 是否变化 (Phase 1 看到 H3
    长程上反而趋同 H0, BIWM-3 加 explicit fail+repair 应当让 H3 LLM 恢复
    risk_aware → D 可能反而**变大**)
  - H0 vs H4: 加 BIWM-4 (verification_mask) → 看 D 是否下降
  - H0 vs H5: 加 BIWM-5 (shadow) → 看 D 是否下降 (shadow 在 H5 cost-aware 上
    本身 trigger 少, 这里主要 sanity test 不 crash)

Group B: BIWM-full 套在所有 5 个非-H0 harness 上, 看 vs H0 是否整体收敛
  - 选 toy_007 (有 risky_actions 触发 BIWM-2/5) + toy_004 (numeric stress)
  - K=5, seed=42 (单 seed, smoke)

Group C: BIWM-6/7 cross-harness alignment (post-hoc reducer)
  - 直接读 Phase 1 主表的 576 jsonl, 对每个 (task, K, seed) 取 6 harness 的
    belief_K, 调 align_beliefs(.) 得 aligned belief, 再算 D(H0, aligned)
  - 报"D 是否比 D(H0, H_x) 平均更低" → BIWM-6 是否减少 belief divergence

所有结果**只做描述性**: 不算 p 值, 只报 D 均值变化 + 5 分量分别变化方向.
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
    CanonicalBeliefWrapper,
    BlockedActionLogWrapper,
    RepairUnrolledWrapper,
    VerificationMaskWrapper,
    ShadowExecutionWrapper,
    biwm_full,
    align_beliefs,
    self_consistency_score,
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


def _read_phase1_belief(phase1_dir: Path, hid: str, tid: str, K: int, seed: int):
    p = phase1_dir / f"{hid}_{tid}_K{K}_seed{seed}.jsonl"
    if not p.exists() or p.stat().st_size == 0:
        return None
    try:
        return _read_last_belief(p)
    except Exception:  # noqa: BLE001
        return None


def main(out_dir: str, phase1_dir: str, seed: int, K: int, tasks_subset: list[str]) -> int:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    phase1 = Path(phase1_dir)
    tasks = [t for t in load_tasks() if t["task_id"] in tasks_subset]
    if not tasks:
        raise SystemExit(f"no tasks in subset {tasks_subset}")

    llm = LLMClient(min_interval_s=0.35, max_retries=3)

    # ----------- Group A: 单组件 wrapper on most-relevant pair -----------
    print("=== Group A: single-component BIWM wrappers ===", flush=True)
    group_a = {
        "BIWM1_canonical_on_H1": ("H0_raw", "H1_structured", CanonicalBeliefWrapper),
        "BIWM2_blocked_log_on_H2": ("H0_raw", "H2_risk_gated", BlockedActionLogWrapper),
        "BIWM3_repair_unrolled_on_H3": ("H0_raw", "H3_repair_heavy", RepairUnrolledWrapper),
        "BIWM4_verification_mask_on_H4": ("H0_raw", "H4_verification_selective", VerificationMaskWrapper),
        "BIWM5_shadow_on_H5": ("H0_raw", "H5_cost_aware", ShadowExecutionWrapper),
    }
    group_a_results = {}
    t0 = time.time()
    for label, (h0_id, hx_id, wrapper_cls) in group_a.items():
        per_task: list[dict] = []
        for task in tasks:
            # baseline H0 from phase1
            b0 = _read_phase1_belief(phase1, h0_id, task["task_id"], K, seed)
            # baseline Hx from phase1
            bx_base = _read_phase1_belief(phase1, hx_id, task["task_id"], K, seed)
            if b0 is None or bx_base is None:
                continue
            # BIWM-wrapped Hx: re-run rollout with wrapper
            wrapped = wrapper_cls(HARNESS_REGISTRY[hx_id]())
            wrapped.harness_id = f"{label}_{hx_id}"
            log_path = out / f"{wrapped.harness_id}_{task['task_id']}_K{K}_seed{seed}.jsonl"
            logger = JSONLLogger(log_path)
            try:
                summary = run_kstep_rollout(
                    task=task, harness=wrapped, llm=llm, horizon=K, logger=logger,
                    benchmark_id="HIBench-Code-v0_toy", environment_id="E_default_v0",
                    seed=seed,
                )
                bx_wrapped = _read_last_belief(log_path)
            except Exception as e:  # noqa: BLE001
                print(f"  [CRASH] {label} {task['task_id']}: {e}", flush=True)
                continue
            d_base = d_belief_components(b0, bx_base)
            d_wrap = d_belief_components(b0, bx_wrapped)
            per_task.append({
                "task_id": task["task_id"],
                "D_baseline": d_base["D_belief"],
                "D_biwm": d_wrap["D_belief"],
                "delta_D": d_wrap["D_belief"] - d_base["D_belief"],
                "cat_delta": d_wrap["cat_mismatch"] - d_base["cat_mismatch"],
                "fail_delta": d_wrap["failure_mode_mismatch"] - d_base["failure_mode_mismatch"],
                "num_delta": d_wrap["num_distance"] - d_base["num_distance"],
                "biwm_tokens": summary["total_tokens"],
            })
        if not per_task:
            group_a_results[label] = {"n": 0}
            continue
        mean_base = statistics.fmean(p["D_baseline"] for p in per_task)
        mean_biwm = statistics.fmean(p["D_biwm"] for p in per_task)
        group_a_results[label] = {
            "n": len(per_task),
            "D_baseline_mean": round(mean_base, 4),
            "D_biwm_mean": round(mean_biwm, 4),
            "delta_D_mean": round(mean_biwm - mean_base, 4),
            "delta_D_per_task": [round(p["delta_D"], 4) for p in per_task],
            "delta_cat_mean": round(statistics.fmean(p["cat_delta"] for p in per_task), 4),
            "delta_fail_mean": round(statistics.fmean(p["fail_delta"] for p in per_task), 4),
            "delta_num_mean": round(statistics.fmean(p["num_delta"] for p in per_task), 4),
            "tokens_biwm_total": sum(p["biwm_tokens"] for p in per_task),
        }
        print(
            f"  {label:<40s} n={len(per_task)} "
            f"D_base={mean_base:.3f} → D_biwm={mean_biwm:.3f} "
            f"Δ={mean_biwm-mean_base:+.3f}",
            flush=True,
        )

    # ----------- Group B: BIWM-full vs baseline, per non-H0 harness -----------
    print("\n=== Group B: BIWM-full stacking on all 5 non-H0 harnesses ===", flush=True)
    group_b_results = {}
    for hx_id in ("H1_structured", "H2_risk_gated", "H3_repair_heavy",
                  "H4_verification_selective", "H5_cost_aware"):
        per_task = []
        for task in tasks:
            b0 = _read_phase1_belief(phase1, "H0_raw", task["task_id"], K, seed)
            bx_base = _read_phase1_belief(phase1, hx_id, task["task_id"], K, seed)
            if b0 is None or bx_base is None:
                continue
            wrapped = biwm_full(HARNESS_REGISTRY[hx_id]())
            log_path = out / f"BIWMfull_{hx_id}_{task['task_id']}_K{K}_seed{seed}.jsonl"
            logger = JSONLLogger(log_path)
            try:
                summary = run_kstep_rollout(
                    task=task, harness=wrapped, llm=llm, horizon=K, logger=logger,
                    benchmark_id="HIBench-Code-v0_toy", environment_id="E_default_v0",
                    seed=seed,
                )
                bx_full = _read_last_belief(log_path)
            except Exception as e:  # noqa: BLE001
                print(f"  [CRASH] BIWMfull {hx_id} {task['task_id']}: {e}", flush=True)
                continue
            d_base = d_belief_components(b0, bx_base)
            d_full = d_belief_components(b0, bx_full)
            per_task.append({
                "task_id": task["task_id"],
                "D_baseline": d_base["D_belief"],
                "D_biwm_full": d_full["D_belief"],
                "delta_D": d_full["D_belief"] - d_base["D_belief"],
                "tokens": summary["total_tokens"],
            })
        if not per_task:
            group_b_results[hx_id] = {"n": 0}
            continue
        mb = statistics.fmean(p["D_baseline"] for p in per_task)
        mf = statistics.fmean(p["D_biwm_full"] for p in per_task)
        group_b_results[hx_id] = {
            "n": len(per_task),
            "D_baseline_mean": round(mb, 4),
            "D_biwm_full_mean": round(mf, 4),
            "delta_D_mean": round(mf - mb, 4),
            "delta_D_per_task": [round(p["delta_D"], 4) for p in per_task],
            "tokens_total": sum(p["tokens"] for p in per_task),
        }
        print(f"  BIWM-full vs H0 on {hx_id:<28s} n={len(per_task)} "
              f"D_base={mb:.3f} → D_full={mf:.3f} Δ={mf-mb:+.3f}", flush=True)

    # ----------- Group C: BIWM-6/7 post-hoc alignment on full Phase 1 data ----
    print("\n=== Group C: BIWM-6/7 cross-harness alignment (post-hoc, reads Phase 1) ===", flush=True)
    all_tasks = load_tasks()
    all_k = [1, 3, 5, 8]
    all_seeds = [42, 43, 44]
    harness_ids = list(HARNESS_REGISTRY.keys())
    align_records = []
    for task in all_tasks:
        for kk in all_k:
            for ss in all_seeds:
                beliefs_by_h = {}
                for hid in harness_ids:
                    b = _read_phase1_belief(phase1, hid, task["task_id"], kk, ss)
                    if b is not None:
                        beliefs_by_h[hid] = b
                if len(beliefs_by_h) < 2:
                    continue
                h0 = beliefs_by_h.get("H0_raw")
                if h0 is None:
                    continue
                # alignment over the 5 non-H0 harnesses
                non_h0_views = [v for hid, v in beliefs_by_h.items() if hid != "H0_raw"]
                aligned = align_beliefs(non_h0_views)
                consistency = self_consistency_score(non_h0_views)
                # for descriptive comparison: mean D(H0, Hx) across 5 vs D(H0, aligned)
                ds_base = [d_belief_components(h0, v)["D_belief"]
                           for v in non_h0_views]
                d_align = d_belief_components(h0, aligned)["D_belief"]
                align_records.append({
                    "task_id": task["task_id"], "K": kk, "seed": ss,
                    "n_views": len(non_h0_views),
                    "mean_D_baseline_5pair": statistics.fmean(ds_base),
                    "D_H0_vs_aligned": d_align,
                    "delta_D_align_minus_base": d_align - statistics.fmean(ds_base),
                    "consistency_signal": consistency["signal"],
                    "categorical_disagreement": consistency["categorical_disagreement"],
                })
    if align_records:
        mb_base = statistics.fmean(r["mean_D_baseline_5pair"] for r in align_records)
        mb_align = statistics.fmean(r["D_H0_vs_aligned"] for r in align_records)
        delta_mean = statistics.fmean(r["delta_D_align_minus_base"] for r in align_records)
        # break down by K
        by_K = {}
        for k in all_k:
            sub = [r for r in align_records if r["K"] == k]
            if not sub:
                continue
            by_K[k] = {
                "n": len(sub),
                "D_baseline_5pair_mean": round(statistics.fmean(r["mean_D_baseline_5pair"] for r in sub), 4),
                "D_H0_vs_aligned_mean": round(statistics.fmean(r["D_H0_vs_aligned"] for r in sub), 4),
                "delta_mean": round(statistics.fmean(r["delta_D_align_minus_base"] for r in sub), 4),
                "categorical_disagreement_mean": round(statistics.fmean(r["categorical_disagreement"] for r in sub), 4),
            }
        group_c = {
            "n_total": len(align_records),
            "D_baseline_5pair_mean": round(mb_base, 4),
            "D_H0_vs_aligned_mean": round(mb_align, 4),
            "delta_mean": round(delta_mean, 4),
            "by_K": by_K,
        }
        print(f"  BIWM-6 alignment n={len(align_records)}  "
              f"D_baseline_mean={mb_base:.3f}  →  D_aligned_mean={mb_align:.3f}  Δ={delta_mean:+.3f}", flush=True)
        for k, v in by_K.items():
            print(f"    K={k}: D_base={v['D_baseline_5pair_mean']:.3f} → D_aligned={v['D_H0_vs_aligned_mean']:.3f} "
                  f"Δ={v['delta_mean']:+.3f}  cat_disagree={v['categorical_disagreement_mean']:.2f}", flush=True)
    else:
        group_c = {"n_total": 0}

    overall = {
        "phase": "DAY4_BIWM_smoke",
        "spec": "BIWM 1-5 wrappers + 6-7 alignment, descriptive comparison only",
        "tasks_subset_for_AB": [t["task_id"] for t in tasks],
        "K_for_AB": K,
        "seed_for_AB": seed,
        "phase1_dir": str(phase1),
        "group_A_single_components": group_a_results,
        "group_B_biwm_full": group_b_results,
        "group_C_alignment": group_c,
        "elapsed_s": round(time.time() - t0, 2),
        "ended_jst": datetime.now(JST).isoformat(timespec="seconds"),
    }
    summary_path = out / "anchor5_summary.json"
    summary_path.write_text(json.dumps(overall, ensure_ascii=False, indent=2))
    print("\n=== anchor_5 summary written ===")
    print(json.dumps({k: v for k, v in overall.items() if k.startswith("group")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="logs/anchor5_biwm_smoke")
    p.add_argument("--phase1", default="logs/phase1_main")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--K", type=int, default=5)
    p.add_argument("--tasks", nargs="+", default=[
        "toy_001_off_by_one",
        "toy_004_integer_overflow",
        "toy_007_destructive_action_trap",
    ])
    a = p.parse_args()
    sys.exit(main(a.out, a.phase1, a.seed, a.K, a.tasks))
