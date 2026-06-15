"""Phase-1 Table 1 — descriptive (pilot / mechanism study) version.

Per human-researcher decision 2026-06-11 02:29 UTC (branch c3):
- Phase 1 is recast as a pilot / mechanism study.
- Statistical inference (p-values, Bonferroni, bootstrap CI, Cohen's d) is
  removed from all Phase-1 outputs and deferred to Phase 2 on public
  benchmarks.
- This script reads the per-row CSV produced by phase1_table1.py and emits a
  descriptive markdown document with means, K-trend arrows, and a brief
  mechanism summary for the paper §17 Table 1 candidate.

Editorial decisions:
- All 5 H0-vs-Hx pairs are reported with their numbers, **no framing words**
  like "backfire", "negative", "falsified", or "unexpected". Trend arrows are
  mechanical: ↑ if D(K=8) > D(K=1) + 0.005, ↓ if D(K=8) < D(K=1) - 0.005,
  → otherwise. This is the minimum bar for a descriptive Phase-1 table that
  does not selectively omit harness pairs (see ds push-back to Director
  2026-06-11 02:32 UTC; default = report all, internal interpretation kept
  in `analysis/internal/h3_h5_polarity_internal.md`).
- H2 censorship on toy_007 is reported as a brief mechanism case study with
  P(success) gap by K, no significance testing.
- The "scope" footer states that statistical validation is deferred to
  Phase 2 on public benchmarks (G2 family).

Run:
    python3 analysis/phase1_table1_descriptive.py
"""
from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "analysis" / "phase1_table1.csv"
JSON_PATH = ROOT / "analysis" / "phase1_results.json"  # has H2 raw audit
OUT_MD = ROOT / "analysis" / "phase1_table1_descriptive.md"
OUT_INTERNAL = ROOT / "analysis" / "internal" / "h3_h5_polarity_internal.md"

H0 = "H0_raw"
H_TARGETS = ["H1_structured", "H2_risk_gated", "H3_repair_heavy",
             "H4_verification_selective", "H5_cost_aware"]
KS = [1, 3, 5, 8]
TREND_THRESHOLD = 0.005  # |Δ| below this is rendered as →


def short(h: str) -> str:
    return {
        "H0_raw": "H0",
        "H1_structured": "H1 (structured)",
        "H2_risk_gated": "H2 (risk-gated)",
        "H3_repair_heavy": "H3 (repair-heavy)",
        "H4_verification_selective": "H4 (verification-selective)",
        "H5_cost_aware": "H5 (cost-aware)",
    }[h]


def trend(d_k1: float, d_k8: float) -> str:
    diff = d_k8 - d_k1
    if diff > TREND_THRESHOLD:
        return "↑"
    if diff < -TREND_THRESHOLD:
        return "↓"
    return "→"


def load_csv_rows() -> list[dict]:
    out = []
    with CSV_PATH.open() as f:
        r = csv.DictReader(f)
        for row in r:
            if row["status"] != "ok":
                continue
            for k in ("D_belief", "D_arrival", "D_growth"):
                row[k] = float(row[k])
            row["K"] = int(row["K"])
            out.append(row)
    return out


def aggregate(rows: list[dict]) -> dict:
    """{(hb, K): {D_arrival_mean, D_growth_mean, D_belief_mean, n}}."""
    agg = {}
    for hb in H_TARGETS:
        for k in KS:
            cell = [r for r in rows
                    if r["ha"] == H0 and r["hb"] == hb and r["K"] == k]
            if not cell:
                continue
            agg[(hb, k)] = {
                "n": len(cell),
                "D_belief_mean": statistics.fmean(r["D_belief"] for r in cell),
                "D_arrival_mean": statistics.fmean(r["D_arrival"] for r in cell),
                "D_growth_mean": statistics.fmean(r["D_growth"] for r in cell),
            }
    return agg


def render_md(agg, h2_audit):
    md = []
    md.append("# Phase-1 Table 1 — descriptive (pilot / mechanism study)\n\n")
    md.append("| Field | Value |\n| --- | --- |\n")
    md.append("| **Owner** | data_scientist |\n")
    md.append("| **Phase** | Phase-1 (pilot / mechanism study; branch c3 per human researcher 2026-06-11 02:29 UTC) |\n")
    md.append("| **Statistical inference** | **none** — deferred to Phase 2 on public benchmarks (G2 family); see §5 scope |\n")
    md.append("| **Metric** | $D_{\\mathrm{belief}}$ v1.1 decomposition: $D = w_A D_{\\mathrm{arrival}} + w_G D_{\\mathrm{growth}}$, $w_A = 0.30$, $w_G = 0.70$ (see METRICS_SPEC §10) |\n")
    md.append("| **Data** | 576 runs (6 harness × 8 task × 4 K × 3 seed); 0 missing, 100% schema-pass |\n")
    md.append("| **Unit of observation** | final-step belief (`step == rollout_horizon`), averaged over 24 (task, seed) pairs per cell |\n")
    md.append("| **Trend symbol** | ↑ = mean increases K=1→K=8 by ≥ 0.005; ↓ = decreases by ≥ 0.005; → = within ±0.005 |\n\n")

    md.append("## 1. K-horizon belief divergence (group mean over 24 (task, seed) cells per K)\n\n")
    md.append("All 5 H0-vs-Hx pairs reported. Three sub-scalars per pair: the scalar $D_{\\mathrm{belief}}$\n")
    md.append("(paper §16.1 headline), the arrival floor $D_{\\mathrm{arrival}}$, and the growth-axis\n")
    md.append("$D_{\\mathrm{growth}}$. Trend column compares K=1 vs K=8 means.\n\n")

    # Scalar
    md.append("### 1.1 $D_{\\mathrm{belief}}$ scalar by horizon $K$\n\n")
    md.append("| pair | K=1 | K=3 | K=5 | K=8 | K=1 → K=8 |\n")
    md.append("| --- | ---: | ---: | ---: | ---: | :---: |\n")
    for hb in H_TARGETS:
        d = {k: agg[(hb, k)]["D_belief_mean"] for k in KS}
        t = trend(d[1], d[8])
        md.append(f"| H0 vs {short(hb)} | {d[1]:.3f} | {d[3]:.3f} | "
                  f"{d[5]:.3f} | {d[8]:.3f} | {t} |\n")
    md.append("\n")

    # Arrival
    md.append("### 1.2 $D_{\\mathrm{arrival}}$ (on-arrival shift) by horizon $K$\n\n")
    md.append("Components: `set_distance` + `action_mismatch`. Captures divergence present immediately at K=1 due to harness prompt-context rewrites.\n\n")
    md.append("| pair | K=1 | K=3 | K=5 | K=8 | K=1 → K=8 |\n")
    md.append("| --- | ---: | ---: | ---: | ---: | :---: |\n")
    for hb in H_TARGETS:
        d = {k: agg[(hb, k)]["D_arrival_mean"] for k in KS}
        t = trend(d[1], d[8])
        md.append(f"| H0 vs {short(hb)} | {d[1]:.3f} | {d[3]:.3f} | "
                  f"{d[5]:.3f} | {d[8]:.3f} | {t} |\n")
    md.append("\n")

    # Growth
    md.append("### 1.3 $D_{\\mathrm{growth}}$ (K-amplification axis) by horizon $K$\n\n")
    md.append("Components: `cat_mismatch` + `failure_mode_mismatch` + `num_distance`. The sub-scalar the H0 hypothesis is loaded on (METRICS_SPEC §10.4).\n\n")
    md.append("| pair | K=1 | K=3 | K=5 | K=8 | K=1 → K=8 |\n")
    md.append("| --- | ---: | ---: | ---: | ---: | :---: |\n")
    for hb in H_TARGETS:
        d = {k: agg[(hb, k)]["D_growth_mean"] for k in KS}
        t = trend(d[1], d[8])
        md.append(f"| H0 vs {short(hb)} | {d[1]:.3f} | {d[3]:.3f} | "
                  f"{d[5]:.3f} | {d[8]:.3f} | {t} |\n")
    md.append("\n")

    # Per-pair sentence summary (mechanical, no framing)
    md.append("## 2. Per-pair sentence summary (mechanical readout from §1.3)\n\n")
    for hb in H_TARGETS:
        d = {k: agg[(hb, k)]["D_growth_mean"] for k in KS}
        md.append(f"- **H0 vs {short(hb)}, K=5**: "
                  f"$D_{{\\mathrm{{growth}}}} = {d[5]:.3f}$ "
                  f"(K=1 baseline = {d[1]:.3f}, trend {trend(d[1], d[5])}).\n")
        md.append(f"  K=8: $D_{{\\mathrm{{growth}}}} = {d[8]:.3f}$ (trend K=1→K=8: {trend(d[1], d[8])}).\n")
    md.append("\n")

    # H2 toy_007 mechanism vignette
    md.append("## 3. H2 blocked-branch mechanism vignette — toy_007\n\n")
    md.append("`toy_007_destructive_action_trap` is the task designed to trigger H2's risk-gate.\n")
    md.append("This pair (H0 vs H2) on this single task shows a textbook blocked-branch effect:\n")
    md.append("the risk-gate prevents the LLM from observing the destructive failure mode, so the\n")
    md.append("agent's belief about $P(\\mathrm{success})$ drifts upward as K grows.\n\n")
    md.append("### 3.1 Failure-mode label distribution (n = 12: 4 K × 3 seeds)\n\n")
    md.append("| harness | distribution |\n| --- | --- |\n")
    md.append(f"| H0 | `destructive_action`: {h2_audit['h0_fm_destructive_count']} / 12 |\n")
    md.append(f"| H2 | `destructive_action`: {h2_audit['h2_fm_destructive_count']} / 12, "
              f"`policy_violation`: {h2_audit['h2_fm_policy_violation_count']} / 12 |\n\n")
    md.append(f"H2 re-labels the same task to `policy_violation` on "
              f"{h2_audit['h2_fm_policy_violation_count']} / 12 cells "
              f"(relabel rate {h2_audit['h2_relabel_rate_to_policy_violation']*100:.0f}%).\n\n")
    md.append("### 3.2 $P(\\mathrm{success})$ gap by K (H2 − H0, group mean across 3 seeds)\n\n")
    md.append("| K | H0 mean $P(\\mathrm{success})$ | H2 mean $P(\\mathrm{success})$ | gap (H2 − H0) |\n")
    md.append("| ---: | ---: | ---: | ---: |\n")
    for k in KS:
        h0 = h2_audit['p_success_H0_mean_by_K'][str(k)]
        h2 = h2_audit['p_success_H2_mean_by_K'][str(k)]
        gap = h2_audit['p_success_gap_H2_minus_H0_by_K'][str(k)]
        md.append(f"| {k} | {h0:.3f} | {h2:.3f} | **{gap:+.3f}** |\n")
    md.append("\n")
    g = h2_audit["p_success_gap_H2_minus_H0_by_K"]
    md.append(f"The gap rises from K=1 ({g['1']:+.2f}) to K=5 ({g['5']:+.2f}) and then closes at K=8 ({g['8']:+.2f}).\n\n")

    # Mechanism summary (single paragraph)
    md.append("## 4. Mechanism summary (paragraph form)\n\n")
    md.append("The five H0-vs-Hx pairs span an arrival floor of $D_{\\mathrm{arrival}} \\approx 0.99$ "
              "across all K and all pairs: each non-trivial harness immediately rewrites the constraint "
              "set and recommended action at K=1, with negligible additional movement as K grows. "
              "The K-step amplification signal is carried entirely by $D_{\\mathrm{growth}}$. Across "
              "the v0_toy benchmark family, $D_{\\mathrm{growth}}$ exhibits harness-specific dynamics "
              "rather than a universal pattern: some pairs trend upward with K, others do not. "
              "Section 3 illustrates one specific blocked-branch mechanism on a single task design "
              "(toy_007), where H2's risk-gate produces a monotone $P(\\mathrm{success})$ inflation "
              "of +0.07 (K=1) → +0.45 (K=5) before partial K=8 reversion. These observations are "
              "*descriptive*; statistical validation is deferred to Phase 2 on public benchmarks "
              "(see §5).\n\n")

    # Scope
    md.append("## 5. Scope of this document\n\n")
    md.append("- **Pilot / mechanism study only.** Phase-1 is reframed (human researcher decision "
              "2026-06-11 02:29 UTC) as a pilot exploration on the v0_toy benchmark; statistical "
              "validation of the H0 hypothesis (K-amplification of $D_{\\mathrm{growth}}$) is "
              "deferred to **Phase 2 on $\\geq 2$ public benchmarks** (G2 family: SWE-bench, "
              "Terminal-Bench).\n")
    md.append("- **No statistical inference reported here.** No p-values, no significance tests, "
              "no Bonferroni, no bootstrap CIs, no Cohen's d. The previous pre-registration document "
              "`analysis/phase1_stats_protocol.md` is **deprecated** (banner at top of that file).\n")
    md.append("- **All numbers are reproducible** from the per-row CSV `analysis/phase1_table1.csv` "
              "and the source jsonls in `experiments/logs/phase1_main/`. Deterministic, no RNG.\n")
    md.append("- **Identity check.** Every row in the source CSV satisfies "
              "$D_{\\mathrm{belief}} = 0.30\\, D_{\\mathrm{arrival}} + 0.70\\, D_{\\mathrm{growth}}$ "
              "to machine epsilon (max residual = 2.22 × 10⁻¹⁶ across 1440 rows; see "
              "`analysis/phase1_results.json` field `identity_audit`).\n\n")

    md.append("## 6. Source files\n\n")
    md.append("- `analysis/phase1_table1.csv` — per-row source data (1440 rows, no aggregation).\n")
    md.append("- `experiments/logs/phase1_main/` — 576 step-jsonl rollout logs (raw).\n")
    md.append("- `experiments/metrics/d_belief.py` — metric implementation (v1.1; 77/77 unit tests).\n")
    md.append("- `analysis/METRICS_SPEC.md` §10 — decomposition definition.\n")

    OUT_MD.write_text("".join(md))


def render_internal_notes(agg):
    """Internal-only mechanism notes for the 5-pair K-trend pattern.

    Not for paper. Lives under `analysis/internal/` so future ds (or theorist
    on a follow-up) can see what the K=1→K=8 numbers do *across* harnesses
    without losing the audit trail.
    """
    OUT_INTERNAL.parent.mkdir(parents=True, exist_ok=True)
    md = []
    md.append("# Internal notes — Phase-1 K-trend pattern across 5 harness pairs\n\n")
    md.append("**Status**: internal, not for paper.  \n")
    md.append("**Purpose**: keep mechanism interpretation of the descriptive Table 1 §1.3 "
              "available to future ds / theorist without putting it in paper-facing prose.\n\n")
    md.append("## K=1 vs K=8 growth-axis movement\n\n")
    md.append("| pair | $D_G$(K=1) | $D_G$(K=8) | Δ | direction |\n")
    md.append("| --- | ---: | ---: | ---: | --- |\n")
    for hb in H_TARGETS:
        d_k1 = agg[(hb, 1)]["D_growth_mean"]
        d_k8 = agg[(hb, 8)]["D_growth_mean"]
        d = d_k8 - d_k1
        direction = "growth up" if d > TREND_THRESHOLD else ("growth down" if d < -TREND_THRESHOLD else "flat")
        md.append(f"| H0 vs {short(hb)} | {d_k1:.3f} | {d_k8:.3f} | {d:+.3f} | {direction} |\n")
    md.append("\n## Interpretation (internal)\n\n")
    md.append("The Phase-1 v0_toy benchmark shows the K-amplification pattern is **not uniform** "
              "across harness types. Some harness types (e.g. those that produce richer "
              "structured exposures) move $D_G$ upward with K; others (e.g. those that produce "
              "collapsed-history narratives, repair-heavy traces, or cost-aware truncation) "
              "move $D_G$ downward with K. The downward movement is consistent with mechanisms "
              "where the harness narrative *converges* the LLM's belief about failure modes and "
              "uncertainty as K grows, rather than amplifying differences.\n\n")
    md.append("This pattern is left out of the paper-facing Table 1 prose by editorial decision "
              "(human researcher 2026-06-11 02:29 UTC); the numbers themselves remain in §1.3. "
              "If a future pre-registration on Phase 2 wants to use the v0_toy pattern as a prior, "
              "this internal note is the place to look.\n")
    OUT_INTERNAL.write_text("".join(md))


def main() -> int:
    rows = load_csv_rows()
    agg = aggregate(rows)
    j = json.loads(JSON_PATH.read_text())
    h2_audit = j["h2_censorship_toy_007"]
    render_md(agg, h2_audit)
    render_internal_notes(agg)
    print(json.dumps({
        "wrote": [str(OUT_MD.relative_to(ROOT)),
                  str(OUT_INTERNAL.relative_to(ROOT))],
        "n_rows_read": len(rows),
        "n_cells": len(agg),
        "phase1_protocol_deprecated": True,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
