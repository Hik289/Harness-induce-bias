"""Step B (DAY3): K-放大 fail-fast.

同 anchor_4 的 5 toy task (toy_001..005), H0 vs H2 paired, 把 K 从 {1,3} 扩到
**K=5 和 K=8**, 测 D_belief(K=5)/D_belief(K=1) 的比值是否 >= 2.0 (G1 弱版).

Director PASS 阈值 (与 anchor_4 strict 5/5 同形, 但允许 4/5 + 平均比值):
- per-task: D(K=5)/D(K=1) >= 2.0  → "K=5 放大"
- per-task: D(K=8)/D(K=1) >= 2.0  → "K=8 放大"
- PASS = (>=4/5 task K=5 放大 AND >=4/5 task K=8 放大 AND mean(K=5/K=1)>=2.0 AND mean(K=8/K=1)>=2.0)
- 失败任一即立即 push back, 不进 A

我复用 anchor_4 已经跑过的 K=1 数据 (5 task × 2 harness × K=1 jsonl 已在 logs/anchor4_phase1_smoke/),
新跑 K=5 + K=8, 节省一半 LLM call。
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
from skeleton.harnesses import H0RawHarness, H2RiskGatedHarness  # noqa: E402
from metrics.d_belief import d_belief_components  # noqa: E402

JST = timezone(timedelta(hours=9))
TASK_PREFIX = ["toy_001", "toy_002", "toy_003", "toy_004", "toy_005"]


def _read_last_belief(p: Path) -> dict:
    last = ""
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            s = line.strip()
            if s:
                last = s
    return json.loads(last)["belief_output"]


def main(out_dir: str, prior_dir: str | None, seed: int) -> int:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # Step B 新跑 K=5 + K=8. K=1 复用 prior_dir (anchor_4 已跑过的 K=1 jsonl).
    tasks = [t for t in load_tasks() if any(t["task_id"].startswith(p) for p in TASK_PREFIX)]
    tasks = tasks[:5]
    assert len(tasks) == 5, f"need 5 tasks, got {len(tasks)}"

    llm = LLMClient(min_interval_s=0.35, max_retries=3)
    harnesses = {"H0_raw": H0RawHarness(), "H2_risk_gated": H2RiskGatedHarness()}

    prior = Path(prior_dir) if prior_dir else Path("logs/anchor4_phase1_smoke")

    t0 = time.time()
    per_task: list[dict] = []
    for task in tasks:
        row: dict = {"task_id": task["task_id"]}
        bel: dict[tuple[str, int], dict] = {}

        # K=1 复用 anchor_4 logs
        for hid in harnesses:
            p1 = prior / f"{hid}_{task['task_id']}_K1_seed{seed}.jsonl"
            if not p1.exists():
                raise SystemExit(
                    f"prior anchor_4 K=1 log missing: {p1}. "
                    f"--prior 应指向 anchor_4 输出目录 (含 K=1 jsonl)."
                )
            bel[(hid, 1)] = _read_last_belief(p1)

        # K=5, K=8 新跑
        for K in (5, 8):
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
                bel[(hid, K)] = _read_last_belief(log_path)
                row[f"{hid}_K{K}_tokens"] = summary["total_tokens"]
                row[f"{hid}_K{K}_latency_s"] = round(summary["total_latency_s"], 2)

        # D_belief at K=1, 5, 8
        D = {
            K: d_belief_components(bel[("H0_raw", K)], bel[("H2_risk_gated", K)])
            for K in (1, 5, 8)
        }
        row["D_K1"] = D[1]["D_belief"]
        row["D_K5"] = D[5]["D_belief"]
        row["D_K8"] = D[8]["D_belief"]
        row["ratio_K5_K1"] = D[5]["D_belief"] / max(D[1]["D_belief"], 1e-6)
        row["ratio_K8_K1"] = D[8]["D_belief"] / max(D[1]["D_belief"], 1e-6)
        row["amplified_K5"] = row["ratio_K5_K1"] >= 2.0
        row["amplified_K8"] = row["ratio_K8_K1"] >= 2.0
        row["components_K1"] = D[1]
        row["components_K5"] = D[5]
        row["components_K8"] = D[8]
        per_task.append(row)
        print(
            f"  [{task['task_id']:<32s}] "
            f"D_K1={row['D_K1']:.3f}  D_K5={row['D_K5']:.3f}  D_K8={row['D_K8']:.3f}  "
            f"r(K5)={row['ratio_K5_K1']:.2f}x  r(K8)={row['ratio_K8_K1']:.2f}x  "
            f"amp5={row['amplified_K5']}  amp8={row['amplified_K8']}",
            flush=True,
        )

    n = len(per_task)
    k5 = sum(1 for r in per_task if r["amplified_K5"])
    k8 = sum(1 for r in per_task if r["amplified_K8"])
    mean_r5 = sum(r["ratio_K5_K1"] for r in per_task) / n
    mean_r8 = sum(r["ratio_K8_K1"] for r in per_task) / n
    pass_K5 = (k5 >= 4) and (mean_r5 >= 2.0)
    pass_K8 = (k8 >= 4) and (mean_r8 >= 2.0)
    overall_pass = pass_K5 and pass_K8

    overall = {
        "phase": "DAY3_step_B",
        "spec": "K-amplification fail-fast: D_belief(H0,H2,K=5 or K=8) / D_belief(K=1) >= 2.0",
        "benchmark": "HIBench-Code-v0_toy",
        "harness_pair": ["H0_raw", "H2_risk_gated"],
        "K_values": [1, 5, 8],
        "K1_reused_from": str(prior),
        "seed": seed,
        "n_tasks": n,
        "n_amplified_K5": k5,
        "n_amplified_K8": k8,
        "mean_ratio_K5_K1": mean_r5,
        "mean_ratio_K8_K1": mean_r8,
        "threshold_per_task": ">= 2.0 ratio",
        "threshold_min_passing_tasks": 4,
        "threshold_mean_ratio": 2.0,
        "pass_K5_amplification": pass_K5,
        "pass_K8_amplification": pass_K8,
        "overall_PASS": overall_pass,
        "elapsed_s": round(time.time() - t0, 2),
        "ended_jst": datetime.now(JST).isoformat(timespec="seconds"),
        "tasks": per_task,
    }
    summary_path = out / "step_B_summary.json"
    summary_path.write_text(json.dumps(overall, ensure_ascii=False, indent=2))
    print("\n=== Step B summary ===")
    print(json.dumps({k: v for k, v in overall.items() if k != "tasks"}, ensure_ascii=False, indent=2))
    return 0 if overall_pass else 1


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="logs/day3_step_B")
    p.add_argument(
        "--prior",
        default="logs/anchor4_phase1_smoke",
        help="anchor_4 输出目录 (包含 K=1 jsonl 用于复用)",
    )
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    sys.exit(main(a.out, a.prior, a.seed))
