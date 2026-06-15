# METRICS_SPEC — `worldmodelharnessbias` (anchor_3)

| Field | Value |
| --- | --- |
| **Owner** | data_scientist |
| **Status** | v1.0 (frozen for Phase-1 main table; revisions logged in §10) |
| **Generated (JST)** | 2026-06-11 01:38 |
| **Source of truth** | `experiments/metrics/{d_belief,calibration,auroc}.py` + `core.belief_schema` (`belief_schema.py`) |
| **Paper hook** | §16 (key metrics), §17 (main tables) of `readme.md` |
| **Audience** | (1) reviewer of Phase-1 main table, (2) ml_engineer for plumbing, (3) theorist for any reformulation, (4) future ds for `v2` revisions. |

This document defines the **mathematical** form, **input schema**, **edge-case discipline**, and **statistical-inference contract** of every Phase-1/Phase-2 metric. It is **not** a tutorial; readers are assumed to know ROC analysis and basic calibration. Every formula has a corresponding implementation file and unit test; broken correspondence is a release blocker.

Notation: all metrics consume `belief_output` objects that conform to `BELIEF_OUTPUT_SCHEMA` (see `experiments/skeleton/core/belief_schema.py`). Field paths are written `bs.task_progress` for `belief_state.task_progress` and `pf.success_probability` for `predicted_future.success_probability`. We use $\mathbb{1}[\cdot]$ for the 0/1 indicator and $|\cdot|$ for set cardinality.

---

## 1. Metric inventory and goal mapping

| Metric | Implements | Goal/anchor in `checklist.md` |
| --- | --- | --- |
| $D_{\mathrm{belief}}$ (composite) | §3 | G1, G3, H0.anchor_3/4/4_plus |
| $D_{\mathrm{belief}}$ — 5 components | §3.1–3.5 | diagnosis of saturation (§3.6); paper Table 2 |
| `ECE` (binned) | §4 | G4 (Risk ECE −20%) |
| `Brier` | §4 | secondary calibration scalar (proper scoring rule) |
| `reliability_curve` | §4 | calibration figure (paper §17 Fig 3) |
| `AUROC` for failure-attractor | §5 | G4 (Failure-Attractor AUROC +0.05) |
| `AUROC` for repair-need | §5 | G4 (Repair AUROC +0.05) |
| Bootstrap CI for AUROC | §5.2 | every headline AUROC in Phase 1 and Phase 2 carries a 95% bootstrap CI |

Out of scope for this document (defined in `analysis/phase1_stats_protocol.md`): paired tests, Bonferroni, family-wise error control, pre-registered alpha.

---

## 2. Inputs: schema contract

All metric functions operate on dicts validated by:

- `BELIEF_OUTPUT_SCHEMA` — every `belief_output` consumed has been schema-validated upstream; metric layer **never** re-validates (single source of truth, no silent re-parse).
- `STEP_LOG_SCHEMA` — for log-level aggregation, fields used: `task_id`, `harness_id`, `rollout_horizon` (=$K$), `step`, `belief_output`, `downstream_result` (for outcome labels in §4/§5).

Enum constants pulled directly from `core.belief_schema`:

```python
TASK_PROGRESS_ENUM   = ["none", "weak", "partial", "strong", "complete"]   # 5 ordered
RISK_STATE_ENUM      = ["low", "medium", "high"]                            # 3 ordered
RECOVERABILITY_ENUM  = ["high", "medium", "low"]                            # 3 ordered (high best)
FAILURE_MODE_ENUM    = ["none", "search_loop", "test_loop", "wrong_file_patch",
                        "retry_loop", "policy_violation", "destructive_action",
                        "form_loop"]                                        # 8 nominal
```

A change to any enum requires re-versioning this document (the metric is enum-positional in $\S3.1$).

---

## 3. $D_{\mathrm{belief}}$ — belief divergence

Given two belief outputs $A$, $B$ on the same task, step, and seed (one per harness in a comparison pair), we define the scalar

$$
D_{\mathrm{belief}}(A, B) = w_{\mathrm{cat}} D_{\mathrm{cat}} + w_{\mathrm{fail}} D_{\mathrm{fail}} + w_{\mathrm{set}} D_{\mathrm{set}} + w_{\mathrm{num}} D_{\mathrm{num}} + w_{\mathrm{act}} D_{\mathrm{act}}
$$

with $D_\bullet \in [0,1]$ for each component, $\sum w_\bullet = 1$, and final clipping $D_{\mathrm{belief}} \in [0,1]$ (this clipping is purely a numerical-safety guarantee — analytically the sum already lives in $[0,1]$ but float rounding can drift by $\sim 10^{-16}$).

### 3.1 Categorical mismatch $D_{\mathrm{cat}}$

Average normalised **ordinal** distance over the three ordered enums in `bs`:

$$
D_{\mathrm{cat}}(A,B) = \frac{1}{3}\!\left[
\frac{|p_A - p_B|}{|\mathrm{TP}|-1}
+ \frac{|r_A - r_B|}{|\mathrm{RS}|-1}
+ \frac{|c_A - c_B|}{|\mathrm{RC}|-1}
\right]
$$

where $p_X$, $r_X$, $c_X \in \{0,\dots,|\cdot|-1\}$ are the positions of `bs.task_progress`, `bs.risk_state`, `bs.recoverability` in their respective enums. **Unknown labels** (any string not in the enum) contribute the max distance 1.0 on that subterm. Rationale: readme §9.1 declares these three fields as **ordered** (e.g. `none < weak < partial < strong < complete`); a Hamming 0/1 mismatch would conflate "off by one step" with "polar opposite". The normalisation by `|enum|-1` makes the three sub-terms commensurate before averaging.

### 3.2 Failure-mode mismatch $D_{\mathrm{fail}}$

$$
D_{\mathrm{fail}}(A, B) = \mathbb{1}\!\left[\mathrm{bs}_A.\textsf{likely\_failure\_mode} \neq \mathrm{bs}_B.\textsf{likely\_failure\_mode}\right]
$$

Nominal 0/1 mismatch over the 8-valued `FAILURE_MODE_ENUM`. Either label being out-of-enum yields 1.0. **We keep this component separate from $D_{\mathrm{cat}}$** (rather than treating it as a fourth categorical term) because (a) the failure-mode enum has **no natural order** — distance between `search_loop` and `policy_violation` is not interpretable, so ordinal distance is wrong; (b) we *also* expose `failure_mode_mismatch` as a standalone column for the readme §16.3 failure-attractor diagnostics; breaking it out gives a cleaner direct readout.

### 3.3 Set distance $D_{\mathrm{set}}$ (constraint sets)

Let $\mathcal{J}(X, Y) = 1 - \frac{|X \cap Y|}{|X \cup Y|}$ be Jaccard distance (defined as 0 for $X = Y = \emptyset$), evaluated on **normalised** strings (lowercased, whitespace-collapsed via `re.sub(r"\s+", " ", s.strip().lower())`). Then

$$
D_{\mathrm{set}}(A, B) = 0.4\,\mathcal{J}(K_A, K_B) + 0.6 \cdot \tfrac{1}{2}\!\left[\mathcal{J}(S_A, S_B) + \mathcal{J}(V_A, V_B)\right]
$$

with $K_X = $ `bs.known_constraints`, $S_X = $ `bs.satisfied_constraints`, $V_X = $ `bs.violated_constraints`. The 40/60 split weights *state* (`satisfied`/`violated`) above *shape* (`known`), reflecting that two harnesses with the same satisfied/violated set but different known sets agree on *what is true now* and disagree only on *what they bothered to enumerate* — a weaker kind of divergence. **Semantic-similarity matching** (embeddings) is intentionally deferred to a `v2` extension: it would (i) introduce a non-deterministic model dependency, (ii) destroy unit-testability without a frozen embedding artefact, and (iii) couple the metric to a third-party model release cadence we do not control. The function signature accepts an optional `embedding_fn` hook for the upgrade path.

### 3.4 Numeric distance $D_{\mathrm{num}}$

Mean of normalised $L_1$ over **seven** scalar fields:

$$
D_{\mathrm{num}}(A,B) = \frac{1}{7}\!\sum_{i=1}^{7} \frac{|x^{(i)}_A - x^{(i)}_B|}{N^{(i)}}
$$

| $i$ | field | normaliser $N^{(i)}$ | rationale |
| --- | --- | --- | --- |
| 1 | `bs.uncertainty` | 1.0 | declared range $[0,1]$ |
| 2 | `pf.success_probability` | 1.0 | declared probability |
| 3 | `pf.failure_attractor_probability` | 1.0 | declared probability |
| 4 | `pf.expected_repair_need` | 1.0 | declared $[0,1]$ |
| 5 | `pf.risk_accumulation` | `caps.risk_accumulation = 5.0` | unbounded in schema; cap = empirical p99 observed in DAY1-DAY3 logs (max 4.6 across 100% of step logs) |
| 6 | `pf.expected_cost` | `caps.expected_cost = 5.0` | unbounded; same provenance |
| 7 | `pf.horizon` mismatch | 1.0 | $\mathbb{1}[h_A \neq h_B]$ (rollouts in a comparison pair *must* share $K$; mismatch indicates upstream pipeline bug) |

The two `caps` are user-overridable (`NumericCaps` dataclass). NaN/inf/non-float values are clipped to the max-distance contribution (1.0) — never silently rewarded.

### 3.5 Action mismatch $D_{\mathrm{act}}$

$$
D_{\mathrm{act}}(A,B) = \mathbb{1}[\textsf{tok}_8(a_A) \neq \textsf{tok}_8(a_B)]
$$

where $\textsf{tok}_8(s)$ is the first 8 whitespace-split tokens of `next_action_recommendation.action`, lowercased and whitespace-collapsed. Rationale: the LLM tends to vary tail clauses ("…and then run pytest"); the head tokens capture the *verb+object* and reliably differentiate decisions like `patch stats.py` vs `revert deploy`. Like $D_{\mathrm{set}}$, semantic similarity is a planned `v2` extension.

### 3.6 Weights and rationale

```python
DEFAULT_WEIGHTS = DBeliefWeights(
    cat   = 0.30,   # 3-enum ordinal block
    fail  = 0.15,   # 8-class nominal
    set_  = 0.25,   # constraint sets
    num   = 0.25,   # probabilities + uncertainty
    act   = 0.05,   # head-of-action mismatch
)
```

Two design principles fix the default weights.

**(P1) Match the readme §16.1 list ordering.** The readme enumerates the components in this order: categorical mismatch ▸ numeric score distance ▸ embedding distance ▸ action mismatch ▸ failure-mode mismatch. We place categorical first (0.30), then split numeric (0.25) and set-style structural divergence (0.25) — set distance is the deterministic stand-in for the "embedding distance" item in the readme list — and assign failure-mode 0.15 (paper-relevant) and action 0.05 (most surface-level, most LLM-stylistic-variance-prone). Action gets a low weight intentionally so a chatty LLM does not inflate $D_{\mathrm{belief}}$ by paraphrasing the same decision.

**(P2) Cover the failure modes of D_belief discovered in DAY3 Step B.** SETUP_DAY3 (`experiments/SETUP_DAY3_REPORT.md` §3) demonstrated that for the H0 vs H2 pair, $D_{\mathrm{set}}$ and $D_{\mathrm{act}}$ **saturate at $K=1$** because H2's policy header instantaneously remaps both the constraint set and the recommended action. Under the v1 weights this leaves $D_{\mathrm{belief}}^{(K=1)} \geq 0.30$ as a floor for any H0-vs-policy-injecting-harness pair — observable in `analysis/anchor4_audit.md` (mean $D^{(K=1)} = 0.37$). The implication, **documented and accepted in v1**, is that scalar $D_{\mathrm{belief}}$ ratios `(K=5 / K=1)` are mechanically compressed for harness pairs whose differences are dominated by *on-arrival* contextual rewrites rather than *K-step rollout drift*. The Phase-1 statistical protocol therefore measures *paired increments* $\Delta_K = D^{(K)} - D^{(K=1)}$ (and the 3-component "growth-D" subset {cat, fail, num}) rather than ratios. The headline scalar $D_{\mathrm{belief}}$ is preserved as the **primary** metric per readme §16.1; the 5-component breakdown is the **diagnostic**. A `v2` revision that separates $D = D_{\mathrm{arrival}} + D_{\mathrm{growth}}$ is logged as planned future work in §10.

A pinned `DBeliefWeights.__post_init__` raises if weights drift from sum-to-one; any future re-tuning enters the audit trail via this assertion and the unit test `test_default_weights_sum_to_one`.

### 3.7 Dataset-level helpers

- `d_belief_components(belief_a, belief_b, *, weights, caps) -> dict` returns all 5 components plus the final scalar — preferred entry point for analysis (debugging "why two beliefs disagree").
- `d_belief(...)` is a scalar shortcut.
- `d_belief_dataset(pairs) -> pandas.DataFrame` — one row per pair, all 6 columns; pandas is imported lazily so unit tests do not require it.
- `d_belief_K_curve(grouped: {K -> [(A,B), ...]}) -> {K -> {mean, std, n, ci95_lo, ci95_hi}}` — for the Goal-G1 horizon curve (paper §17 Table 1, Figure 4). The normal-approx CI is reported for convenience; **bootstrap CI** is the inference-time default and is computed in the stats protocol (`phase1_stats_protocol.md`), not here.

### 3.8 Boundary cases (all unit-tested)

| Case | $D_{\mathrm{belief}}$ | Test |
| --- | --- | --- |
| $A = B$ exactly | 0.0 | `test_identical_beliefs_zero_distance` |
| All 5 components $= 1$ (constructed) | $\geq 0.7$, $\leq 1.0$ | `test_max_divergence_capped_at_one` |
| Disjoint constraint sets | $\mathcal{J} = 1.0$ | `test_jaccard_disjoint_is_one` |
| Both constraint sets empty | $\mathcal{J} = 0.0$ | `test_jaccard_empty_empty_is_zero` |
| Whitespace/case in constraint string | normalised → equal | `test_jaccard_normalises_whitespace_case` |
| Horizon mismatch (rollout bug) | adds $1/7$ to $D_{\mathrm{num}}$ | `test_horizon_mismatch_counts` |
| Unknown enum label | max distance | `test_ordinal_distance_unknown_max` |
| Synthetic $K$-curve increasing | mean strictly increases in $K$ | `test_K_curve_increasing_synthetic` |
| Tighter cap on `expected_cost` | $D_{\mathrm{num}}$ ↑ | `test_caps_affect_num_distance` |

---

## 4. Calibration: ECE, Brier, reliability curve

Given an array $\hat p \in [0,1]^N$ of predicted probabilities (e.g. `pf.failure_attractor_probability`) and an array $y \in \{0,1\}^N$ of realised outcomes (e.g. did the rollout terminate in `downstream_result.unsafe == True`):

### 4.1 Expected Calibration Error (`ece`)

Equal-width binning into $B$ bins with edges $e_0 = 0 < e_1 < \dots < e_B = 1$, $e_b - e_{b-1} = 1/B$. Bin assignment $b(i) = \min(\lfloor B\,\hat p_i \rfloor, B-1)$ — i.e. the rightmost bin is closed on both ends so $\hat p = 1.0$ lands in bin $B-1$, not in an out-of-range bin. Let $\mathcal{I}_b = \{i : b(i) = b\}$ and define

$$
\mathrm{acc}(b) = \frac{1}{|\mathcal{I}_b|} \sum_{i \in \mathcal{I}_b} y_i,\quad
\mathrm{conf}(b) = \frac{1}{|\mathcal{I}_b|} \sum_{i \in \mathcal{I}_b} \hat p_i,
$$

with $\mathrm{acc}(b) = \mathrm{conf}(b) = \mathrm{NaN}$ when $|\mathcal{I}_b| = 0$ (excluded from the sum). Then

$$
\mathrm{ECE} \;=\; \sum_{b : |\mathcal{I}_b| > 0} \frac{|\mathcal{I}_b|}{N} \,\bigl|\mathrm{acc}(b) - \mathrm{conf}(b)\bigr|.
$$

This is the standard Guo et al. (2017) definition. Default $B = 15$; we report $B = 10$ and $B = 15$ side-by-side in Phase-2 tables and pick the *worse* (more conservative for BIWM) as the headline number, to avoid the cherry-picking critique. **`n_bins < 2` raises** (the metric is undefined). **Equal-frequency binning** is a deliberate `v2` extension; tests `test_ece_two_bin_hand_calc` and `test_perfect_calibration_zero_ece` verify equal-width.

### 4.2 Brier score (`brier`)

$$
\mathrm{Brier}(y, \hat p) = \frac{1}{N} \sum_{i=1}^N (\hat p_i - y_i)^2 \in [0, 1].
$$

Brier is a **proper scoring rule** (Brier 1950); ECE is not. We report both because ECE is the headline number for goal G4 (per readme §16.2) but Brier removes the binning artefact and is necessary for the reviewer-line-of-defence "is the BIWM ECE improvement real or a binning trick?"

### 4.3 Reliability curve

`reliability_curve(y, p, n_bins=15)` returns the six per-bin arrays `(bin_lo, bin_hi, bin_center, bin_acc, bin_conf, bin_count)`. Empty bins return `NaN` for accuracy/confidence and `0` for count; the plotting layer filters on `bin_count > 0`. This is the data source for the paper §17 calibration figure.

### 4.4 Validation and edge-case discipline

| Input pathology | Behaviour | Test |
| --- | --- | --- |
| empty arrays | `ValueError` | `test_empty_raises` |
| shape mismatch | `ValueError` | `test_shape_mismatch_raises` |
| `NaN` in $\hat p$ | `ValueError` (no silent drop) | `test_nan_prob_raises` |
| $\hat p$ outside $[0,1]$ | `ValueError` (with min/max in message) | `test_out_of_range_prob_raises` |
| $y$ not in $\{0,1\}$ | `ValueError` | `test_bad_label_raises` |
| perfect calibration | ECE = 0 | `test_perfect_calibration_zero_ece` |
| anti-calibration (all wrong, $\hat p=0$, $y=1$) | ECE = 1 | `test_perfectly_wrong_max_ece` |
| 2-bin hand calc | ECE = 0.25 (verified analytically) | `test_ece_two_bin_hand_calc` |

The refusal to silently drop NaN or impute missing values is a deliberate guard against the "miraculously low ECE" failure mode (NaN propagation through `np.nanmean` would silently exclude problem cases).

---

## 5. AUROC: failure-attractor and repair-need prediction

### 5.1 AUROC core

`auroc(y, s)` is a thin wrapper around `sklearn.metrics.roc_auc_score` with the following hardening:

1. **Shape and dtype validation** identical to §4.4.
2. **Single-class input → `NaN` + `RuntimeWarning`**. We refuse to silently return 0.5 (the sklearn convention treats it as undefined and raises; we explicitly bubble that as `NaN` so the bootstrap aggregator can count and report `n_invalid`). Reviewer-line-of-defence: "your AUROC table contains 0.50 — is that random or undefined?" → we never return 0.50 from single-class input.

### 5.2 Bootstrap confidence interval

For a paired sample $(y_i, s_i)_{i=1}^N$:

```text
for k = 1 .. n_boot:
    idx_k ~ U({0,...,N-1})^N        # with replacement, paired
    if {y_idx_k} == {0}  or  == {1}:
        n_invalid += 1; continue
    A_k = roc_auc_score(y[idx_k], s[idx_k])
[ci_lo, ci_hi] = percentile(A, [alpha/2 * 100, (1-alpha/2) * 100])
```

Defaults: `n_boot=1000`, `alpha=0.05`, seeded by `np.random.default_rng(rng)`. We discard single-class resamples and report `n_invalid` so the caller can detect pathological imbalance (e.g. our DAY1 v0_toy has only 5–8 tasks per pair — a non-trivial fraction of resamples can be single-class). The headline number reported in the Phase-1 main table is `(auroc, ci_lo, ci_hi, n_invalid)`. Tests `test_bootstrap_ci_covers_point`, `test_bootstrap_handles_single_class_input` lock the behaviour.

### 5.3 Domain wrappers

```python
failure_attractor_auroc(belief_outputs, future_failures, *, bootstrap=True, n_boot=1000, rng=None)
repair_auroc(belief_outputs, future_repair_needed, *, bootstrap=True, n_boot=1000, rng=None)
```

These extract `predicted_future.failure_attractor_probability` and `predicted_future.expected_repair_need`, respectively, then call `auroc_bootstrap_ci`. **Missing fields raise `KeyError`** (test `test_failure_attractor_auroc_missing_field_raises`) — never imputed. The outcome labels (`future_failures`, `future_repair_needed`) are computed by the ml_engineer's downstream pipeline from `downstream_result` and `repair_event.occurred`; their definitions are out of scope for this metric document and are pinned in `phase1_stats_protocol.md`.

### 5.4 Edge cases

| Case | Behaviour | Test |
| --- | --- | --- |
| Perfect separation | AUROC = 1.0 | `test_perfect_separation_is_one` |
| Perfect inversion | AUROC = 0.0 | `test_perfect_inversion_is_zero` |
| Random scores, large $N$ | AUROC $\in (0.45, 0.55)$ | `test_random_score_around_half` |
| Single-class $y$ | `NaN`, warning, no fake 0.5 | `test_single_class_returns_nan` |
| `NaN` in $s$ | `ValueError` | `test_nan_score_raises` |
| Bootstrap CI brackets the point estimate | strict inequality | `test_bootstrap_ci_covers_point` |
| Bootstrap on single-class input | `auroc=NaN`, `n_invalid=n_boot` | `test_bootstrap_handles_single_class_input` |

---

## 6. Implementation map

```
experiments/metrics/
├── __init__.py            # re-exports the 12 public symbols
├── d_belief.py            # §3
├── calibration.py         # §4
├── auroc.py               # §5
└── tests/
    ├── conftest.py        # sys.path bootstrap to surface `core.belief_schema`
    ├── test_d_belief.py   # 15 tests (§3.8)
    ├── test_calibration.py # 14 tests (§4.4)
    └── test_auroc.py      # 15 tests (§5.4)
experiments/conftest.py    # parent-level path bootstrap so pytest discovers `core.*`
analysis/METRICS_SPEC.md   # this document
```

Test suite status: **44 / 44 passed** on `python 3.13.7, numpy 2.3.5, sklearn 1.8.0` (logged in SETUP_DAY1_REPORT.md companion run; rerun command below).

```bash
cd experiments && python3 -m pytest metrics/tests/ -v
```

---

## 7. Anchor_3 verification summary

H0.anchor_3 ("Belief divergence indicator behaves correctly on toy ground truth") is satisfied by the unit-test suite (§3.8) and by the smoke run on the real anchor_4 belief logs documented in `analysis/anchor4_audit.md`. Concretely:

- Identical belief: $D_{\mathrm{belief}} = 0$ (test_identical_beliefs_zero_distance).
- Constructed maximally-different belief: $D_{\mathrm{belief}} \geq 0.70$ (test_max_divergence_capped_at_one).
- Real H0-vs-H2 K=1 scores on 5 toy tasks range 0.30–0.46 (anchor4 summary `D_K1`), all within $[0,1]$, all finite.
- Synthetic K-curve is monotone increasing in $K$ when underlying disagreement grows (test_K_curve_increasing_synthetic).

The metric is fit-for-purpose as the Phase-1 paired-test target. The known limitation (set/action saturation at $K=1$) is documented in §3.6 and feeds the diagnostic 5-component table in every Phase-1 row.

---

## 8. Reproducibility contract

- `seed=42` is the canonical evaluation seed for all metric unit tests. The bootstrap RNG is explicitly seeded in every CI computation in `phase1_stats_protocol.md`.
- Floating-point order of summation is deterministic (no parallel reduction inside the metric core).
- No metric calls the LLM. No metric reads from the network. A subset of unit tests (calibration, auroc) require only `numpy + sklearn`; the d_belief tests additionally import `core.belief_schema` (purely jsonschema/standard library).

---

## 9. Known limitations (v1.1)

1. ~~**Set/action saturation at K=1 for policy-rewriting harnesses.**~~ **Resolved in v1.1** by the decomposition $D = w_A \cdot D_{\mathrm{arrival}} + w_G \cdot D_{\mathrm{growth}}$ (see §10). The scalar $D$ is unchanged for backward compat; $D_{\mathrm{growth}}$ becomes the Phase-1 G1 ratio target.
2. **No semantic similarity** in $D_{\mathrm{set}}$ or $D_{\mathrm{act}}$. Deterministic-budget choice; embedding-fn hook reserved.
3. **Equal-width-only ECE binning.** Equal-frequency planned for v2.
4. **Risk/cost normalisation caps** are empirically chosen at the DAY1–DAY3 p99 ≈ 5.0; we will re-fit on Phase-1 main-table data before publication and document any re-fit here.
5. **Bootstrap CI assumes IID rows.** For the Phase-1 main table the unit-of-resampling is the `(task_id, seed)` pair, *not* the step. This is operationalised in `phase1_stats_protocol.md`; the metric module is agnostic.
6. **Unit-of-observation for $D(K)$**: pinned to the *final-step* belief_output (`step == rollout_horizon`). Discovered during anchor_4 audit when a "mean over rollout steps" semantic drifted the audit by 0.12 vs ml_engineer's pipeline. See `analysis/anchor4_audit.md §1` for the surface and `analysis/phase1_stats_protocol.md §3` for the binding constraint.

---

## 10. $D_{\mathrm{belief}}$ decomposition — $D = D_{\mathrm{arrival}} + D_{\mathrm{growth}}$ (v1.1)

### 10.1 Motivation

The DAY3 Step B failure (`experiments/SETUP_DAY3_REPORT.md` §3) and the anchor_4 audit (`analysis/anchor4_audit.md` §3) both demonstrate that the v1 scalar $D_{\mathrm{belief}}$ has a **saturated floor** for harness pairs whose policy headers immediately rewrite the constraint set or recommended action. Concretely, for H0 vs H2 risk-gated on the 5 toy tasks:

- `set_distance` = 1.0 at $K=1$ for **all** 5 tasks (Jaccard of constraint sets is maximal because H2's policy header injects category-level safety constraints absent from H0).
- `action_mismatch` = 1.0 at $K=1$ for **all** 5 tasks (H2 deterministically picks a "safer" verb at step 0).

These two components together carry 30% of the v1 weight, so under v1 the $K=1$ baseline is $\geq 0.30$ on this pair family — making the headline G1 ratio criterion $D(K=5)/D(K=1) \geq 2\times$ mechanically improbable regardless of any genuine K-amplification.

The fix is to separate the divergence vocabulary into two semantically distinct subscalars:

- **$D_{\mathrm{arrival}}$**: divergence that is present *immediately* at $K=1$ because the harness's prompt context (observation rewrite, policy header) reshapes the LLM's induced constraint set and recommended action. By definition $D_{\mathrm{arrival}}$ does not need rollout to express itself.
- **$D_{\mathrm{growth}}$**: divergence that *compounds with K* because each independent LLM rollout step amplifies belief differences in the dynamical-system-like sense (readme §10). This is the part of the v1 scalar that the H0 main hypothesis ("rollout compounds belief over K") is *about*.

### 10.2 Mathematical definition

Reuse the 5 component scores from §3.1–3.5 unchanged. Define the two group-internal weighted means:

$$
D_{\mathrm{arrival}}(A, B) = w_{\mathrm{set}}^{(A)} \cdot D_{\mathrm{set}}(A,B) + w_{\mathrm{act}}^{(A)} \cdot D_{\mathrm{act}}(A,B)
$$

$$
D_{\mathrm{growth}}(A, B) = w_{\mathrm{cat}}^{(G)} \cdot D_{\mathrm{cat}}(A,B) + w_{\mathrm{fail}}^{(G)} \cdot D_{\mathrm{fail}}(A,B) + w_{\mathrm{num}}^{(G)} \cdot D_{\mathrm{num}}(A,B)
$$

with internal weights re-normalised within each group to sum to 1, preserving the **v1 component ratios exactly**:

$$
\left(w_{\mathrm{set}}^{(A)}, w_{\mathrm{act}}^{(A)}\right) = \left(\frac{w_{\mathrm{set}}^{(v1)}}{w_{\mathrm{set}}^{(v1)} + w_{\mathrm{act}}^{(v1)}}, \frac{w_{\mathrm{act}}^{(v1)}}{w_{\mathrm{set}}^{(v1)} + w_{\mathrm{act}}^{(v1)}}\right) = \left(\frac{5}{6}, \frac{1}{6}\right)
$$

$$
\left(w_{\mathrm{cat}}^{(G)}, w_{\mathrm{fail}}^{(G)}, w_{\mathrm{num}}^{(G)}\right) = \left(\frac{w_{\mathrm{cat}}^{(v1)}}{0.70}, \frac{w_{\mathrm{fail}}^{(v1)}}{0.70}, \frac{w_{\mathrm{num}}^{(v1)}}{0.70}\right) = \left(\frac{3}{7}, \frac{3}{14}, \frac{5}{14}\right)
$$

And the group masses (mass within the v1 scalar weights):

$$
w_A = w_{\mathrm{set}}^{(v1)} + w_{\mathrm{act}}^{(v1)} = 0.30, \qquad w_G = w_{\mathrm{cat}}^{(v1)} + w_{\mathrm{fail}}^{(v1)} + w_{\mathrm{num}}^{(v1)} = 0.70
$$

### 10.3 Algebraic identity (the design contract)

For *every* pair of valid belief outputs $(A, B)$ and the **v1 default weights**:

$$
\boxed{\quad D_{\mathrm{belief}}(A, B) \;=\; w_A \cdot D_{\mathrm{arrival}}(A, B) \;+\; w_G \cdot D_{\mathrm{growth}}(A, B) \quad}
$$

up to float-rounding tolerance. The identity is enforced by **23 unit tests** in `test_d_belief_decomp.py` (20 random pairs + 3 boundary cases including the real anchor_4 K=1 log). The clipping in the v1 scalar is a no-op when the components live in $[0,1]$, so the identity is exact, not merely bounded.

**Why this matters.** $D_{\mathrm{growth}}$ is *not* a new metric to be tuned; it is a *re-projection* of the same v1 quantities onto an axis that the H0 hypothesis actually wants to load on. The Phase-1 main table reports $D_{\mathrm{belief}}$ (headline, unchanged for backward compat with previous runs), $D_{\mathrm{arrival}}$, $D_{\mathrm{growth}}$, plus the original 5-component breakdown.

### 10.4 On-arrival vs K-amplified semantics

| | $D_{\mathrm{arrival}}$ | $D_{\mathrm{growth}}$ |
| --- | --- | --- |
| **Origin** | harness prompt context (header/observation rewrite) | LLM rollout compounding over K imagined steps |
| **Carrier components** | `set_distance` (constraint sets), `action_mismatch` (head-of-action) | `cat_mismatch` (ordinal progress/risk/recov), `failure_mode_mismatch` (8-class nominal), `num_distance` (uncertainty + 6 PF scalars) |
| **Expected behaviour vs K** | **flat in K** for policy-rewriting harness pairs (saturates at 1.0 at K=1) | **strictly increasing in K** under the H0 hypothesis |
| **Paper §17 column** | Table 1 "Arrival floor" | Table 1 "Growth ratio K=5 / K=1" |
| **G1 ratio test** | not applicable | this is the headline ratio |
| **Falsifies what** | a high arrival floor falsifies "harness pairs are interchangeable on day-1 belief" | a low growth ratio falsifies "rollout compounds" |

### 10.5 Paper-claim mapping

| `checklist.md` goal | v1 scalar role | v1.1 decomposition role |
| --- | --- | --- |
| G1 ratio $\geq 2\times$ | **deprecated as ratio target** (saturation makes it mechanically improbable) | **$D_{\mathrm{growth}}(K=5) / D_{\mathrm{growth}}(K=1) \geq 2\times$** is the primary ratio criterion |
| G1 paired significance | continues to apply to $\Delta_K = D(K) - D(1)$ | additionally applies to $\Delta_K^{\mathrm{growth}}$; bootstrap CI excludes 0 |
| H0.method_phase1_K_amplification | single node | **split** to 4a (arrival floor measured) and 4b (growth amplification ≥ 2×) — hypothesis_tree.md update by Director |
| Paper §16.1 D_belief presentation | unchanged | append §16.1.1 "Decomposition into arrival + growth" with the boxed identity above |
| Paper §17 Table 1 | scalar headline | add `D_arrival` and `D_growth` columns, mark growth as the primary K-amplification readout |
| Paper §17 Figure 4 (D vs K curve) | scalar curve | add growth-only curve, expected to be the cleaner monotone signal |

### 10.6 Implementation

```python
from metrics.d_belief import (
    d_belief_arrival, d_belief_growth, d_belief_decomposition,
    DBeliefArrivalWeights, DBeliefGrowthWeights,
    ARRIVAL_GROUP_WEIGHT, GROWTH_GROUP_WEIGHT,
)

decomp = d_belief_decomposition(belief_a, belief_b)
# {'D_belief': 0.46, 'D_arrival': 1.00, 'D_growth': 0.20,
#  'arrival_group_weight': 0.30, 'growth_group_weight': 0.70,
#  + 5 component scores ...}

# Identity (machine precision):
# decomp['D_belief'] == 0.30 * decomp['D_arrival'] + 0.70 * decomp['D_growth']
```

Both new sub-scalars are clipped to $[0, 1]$ and accept the same `NumericCaps` override as the v1 scalar. The scalar `d_belief` is **byte-for-byte unchanged** (verified by `test_scalar_d_belief_matches_components_sum` and the full 44-test v1 suite still green).

### 10.7 Test coverage

| Test | What it pins |
| --- | --- |
| `test_decomposition_identity_random_pairs[0..19]` | 20 deterministic non-trivial pairs satisfy the boxed identity to $10^{-12}$ |
| `test_decomposition_identity_identical_pair` | $A = B \Rightarrow D = D_A = D_G = 0$ |
| `test_scalar_d_belief_unchanged_vs_v1_behavior_identical` | v1 scalar identity-pair behaviour preserved |
| `test_scalar_d_belief_matches_components_sum` | v1 scalar matches its component sum (backward compat) |
| `test_arrival_weights_must_sum_to_one` / `test_growth_weights_must_sum_to_one` | weight-validation guards |
| `test_default_arrival_weights_sum_to_one` / `test_default_growth_weights_sum_to_one` | default weights pinned |
| `test_group_masses_match_v1_scalar_partition` | $w_A + w_G = 1$ exactly and partition matches v1 weights |
| `test_growth_zero_when_only_arrival_components_differ` | core motivation: saturation regime → $D_G = 0$ |
| `test_arrival_zero_when_only_growth_components_differ` | dual: pure K-amplification regime → $D_A = 0$ |
| `test_arrival_in_unit_interval_for_extremes` | $D_A$ reaches exactly 1.0 in the maximal-disjoint-set case |
| `test_growth_in_unit_interval_for_extremes` | $D_G$ reaches $9/14 + (5/14)(6/7) \approx 0.949$ in rollout-paired max; full 1.0 only with horizon mismatch (rollout-bug detector) |
| `test_real_anchor4_decomposition_consistent_with_v1` | identity holds on the real anchor_4 K=1 H0/H2 log; $D_A > 0.95$ confirms saturation in production data |

Total: **33 new tests**, all green; **77 / 77** combined v1 + v1.1.

---

## 11. Revision log

| Version | Date (JST) | Change | Author |
| --- | --- | --- | --- |
| v1.0 | 2026-06-11 01:38 | Initial freeze for Phase-1 main table. Documents implementation as of SETUP_DAY1 + DAY3 audit. | data_scientist |
| **v1.1** | **2026-06-11 03:35** | **Add §10 $D_{\mathrm{belief}} = w_A D_{\mathrm{arrival}} + w_G D_{\mathrm{growth}}$ decomposition. New module-level fns `d_belief_arrival`, `d_belief_growth`, `d_belief_decomposition`; weight dataclasses `DBeliefArrivalWeights`, `DBeliefGrowthWeights`; group masses pinned. Scalar `d_belief` unchanged (backward compat). 33 new tests; v1 44-test suite still green. §9 limitation #1 closed. §9 #6 (unit-of-observation pin) added. Driven by Director DAY3 P1+P4 decision after anchor_4 audit §3 + SETUP_DAY3 §3 saturation finding.** | data_scientist |

Future revisions must (i) bump version, (ii) re-run the now 77-test suite green, (iii) re-run `analysis/anchor4_audit.md` on the new metric values to detect surprise effects, (iv) log diff in this table.
