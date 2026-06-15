"""BIWM-6 + BIWM-7: Cross-Harness Belief Alignment + Self-Consistency.

§11.6 Cross-harness alignment: 对同一 (task, K, seed), 在多个 harness view 下
跑 belief rollout, 然后**对齐**: 用规则把多 belief 合并成 1 个 aligned belief,
disagree 时 uncertainty 拉高, 多数 vote 决定 categorical fields, 平均 numeric.

§11.7 Self-consistency: 把 disagreement 量化成 epistemic uncertainty signal:
跨 harness view 的 belief 越散, aligned uncertainty 越高 (这是 BIWM 自己的
calibration signal).

实施: 不调 LLM, 输入是已有 belief_output 列表, 输出 1 个 aligned belief_output.
这样 BIWM-6+7 是一个**纯函数 reducer**, 跑在 Phase 1 主表 jsonl 之上.
"""
from __future__ import annotations

import statistics
from collections import Counter
from typing import Iterable


def _most_common(values: list[str]) -> str:
    if not values:
        return ""
    c = Counter(values)
    return c.most_common(1)[0][0]


def _mean_float(values: list[float], default: float = 0.5) -> float:
    cleaned = [v for v in values if isinstance(v, (int, float))]
    return statistics.fmean(cleaned) if cleaned else default


def _disagreement_score(values: list[str]) -> float:
    """0 (全同) - 1 (全不同)."""
    if not values:
        return 0.0
    c = Counter(values)
    top = c.most_common(1)[0][1]
    return 1.0 - (top / len(values))


def align_beliefs(belief_outputs: list[dict]) -> dict:
    """合并 K 个 harness 的 belief_output → 1 个 aligned belief_output.

    Inputs:
        belief_outputs: list of belief_output dicts (BELIEF_OUTPUT_SCHEMA 形态)

    Returns:
        aligned belief_output (相同 schema)
    """
    if not belief_outputs:
        raise ValueError("align_beliefs 需要至少 1 个 belief_output")

    progresses = [b["belief_state"]["task_progress"] for b in belief_outputs]
    risks = [b["belief_state"]["risk_state"] for b in belief_outputs]
    recovs = [b["belief_state"]["recoverability"] for b in belief_outputs]
    fmodes = [b["belief_state"]["likely_failure_mode"] for b in belief_outputs]
    actions = [b["next_action_recommendation"]["action"] for b in belief_outputs]

    # 各 categorical 字段的 disagreement, 用于 boost uncertainty
    disagree = max(
        _disagreement_score(progresses),
        _disagreement_score(risks),
        _disagreement_score(fmodes),
    )

    # 数字字段平均
    uncs = [b["belief_state"]["uncertainty"] for b in belief_outputs]
    succ = [b["predicted_future"]["success_probability"] for b in belief_outputs]
    fa = [b["predicted_future"]["failure_attractor_probability"] for b in belief_outputs]
    ra = [b["predicted_future"]["risk_accumulation"] for b in belief_outputs]
    ec = [b["predicted_future"]["expected_cost"] for b in belief_outputs]
    ern = [b["predicted_future"]["expected_repair_need"] for b in belief_outputs]

    # known/satisfied/violated 取并集 (谨慎 — 任一 harness 报的约束都纳入)
    def union_constraints(field: str) -> list[str]:
        s: set[str] = set()
        for b in belief_outputs:
            for c in b["belief_state"].get(field, []):
                if isinstance(c, str):
                    s.add(c.strip())
        return sorted(s)

    aligned = {
        "belief_state": {
            "task_progress": _most_common(progresses),
            "risk_state": _most_common(risks),
            "recoverability": _most_common(recovs),
            "likely_failure_mode": _most_common(fmodes),
            # 关键: aligned uncertainty = max(mean(input unc), disagreement)
            # → 跨 harness 不一致时强制 uncertainty 升高
            "uncertainty": max(_mean_float(uncs, 0.5), disagree),
            "known_constraints": union_constraints("known_constraints"),
            "satisfied_constraints": union_constraints("satisfied_constraints"),
            "violated_constraints": union_constraints("violated_constraints"),
        },
        "predicted_future": {
            "horizon": belief_outputs[0]["predicted_future"]["horizon"],
            "success_probability": _mean_float(succ, 0.5),
            "failure_attractor_probability": _mean_float(fa, 0.5),
            "risk_accumulation": _mean_float(ra, 0.0),
            "expected_cost": _mean_float(ec, 0.0),
            "expected_repair_need": _mean_float(ern, 0.0),
        },
        "next_action_recommendation": {
            # 多数 vote, tie 时取第一个
            "action": _most_common(actions),
            "reason": f"cross-harness aligned (BIWM-6) over {len(belief_outputs)} views; "
                      f"categorical disagreement={disagree:.2f}",
            "verification_target": "aggregated across harness views",
        },
        "extras": {
            "biwm6_alignment": {
                "n_views": len(belief_outputs),
                "disagreement_max_categorical": disagree,
                "harness_views_unique_task_progress": sorted(set(progresses)),
                "harness_views_unique_risk_state": sorted(set(risks)),
                "harness_views_unique_failure_mode": sorted(set(fmodes)),
            }
        },
    }
    return aligned


def self_consistency_score(belief_outputs: list[dict]) -> dict:
    """BIWM-7: 跨 harness view 一致性 quantified, 输出 epistemic uncertainty
    signal."""
    if len(belief_outputs) < 2:
        return {"signal": "none", "disagreement": 0.0, "n_views": len(belief_outputs)}
    progresses = [b["belief_state"]["task_progress"] for b in belief_outputs]
    risks = [b["belief_state"]["risk_state"] for b in belief_outputs]
    fmodes = [b["belief_state"]["likely_failure_mode"] for b in belief_outputs]
    actions = [b["next_action_recommendation"]["action"] for b in belief_outputs]
    cat_disagree = max(
        _disagreement_score(progresses),
        _disagreement_score(risks),
        _disagreement_score(fmodes),
    )
    act_disagree = _disagreement_score(actions)
    return {
        "signal": "high" if cat_disagree > 0.6 else ("medium" if cat_disagree > 0.3 else "low"),
        "categorical_disagreement": cat_disagree,
        "action_disagreement": act_disagree,
        "n_views": len(belief_outputs),
        "epistemic_uncertainty_boost": cat_disagree,
    }
