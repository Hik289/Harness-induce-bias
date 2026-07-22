# Artifact Guide

Operational notes for reproducing `Measuring Harness-Induced Belief Divergence in Multi-Step LLM Agents` from the public `Harness-induce-bias` repository.

## Review Path

- `benchmark/`: Project-specific implementation subtree.
- `biwm/`: Project-specific implementation subtree.
- `core/`: Project-specific implementation subtree.
- `harnesses/`: Project-specific implementation subtree.
- `scripts/`: Command-line entry points for experiments, analysis, or reproduction.
- `tests/`: Local tests or smoke checks for fresh checkouts.
- `figures/`: README and paper-facing figures.
- `analysis/`: Post-processing, table, and figure-generation scripts.

## Environment Files

- `requirements.txt`: Primary Python dependency list.

## Smoke Checks

Run these checks before long jobs:

```bash
python -m compileall -q .
python -m pytest tests -q
python scripts/anchor1_api_smoke.py
python scripts/anchor4_phase1_smoke.py
python scripts/anchor5_biwm_smoke.py
python scripts/sanity_6harness_K5.py
python tests/test_smoke.py
```

## Reproduction Entry Points

Main tracked entry points for paper-scale or benchmark-scale runs:

- `python analysis/biwm_v2_recompute.py`
- `python analysis/phase1_table1.py`
- `python analysis/phase1_table1_descriptive.py`

## Figure Assets

- `figures/intuition.pdf`
- `figures/intuition.png`

## Data And Outputs

- API-backed runs should read credentials from environment variables or local `.env` files only; never commit real keys or provider-specific secrets.
- Record provider endpoint, model/deployment name, sampling parameters, and execution date for every API-backed table or figure.
- Treat generated JSONL files, logs, caches, model checkpoints, and benchmark downloads as local artifacts unless explicitly tracked as fixtures.
- For stochastic experiments, record seeds, task counts, dataset splits, and the exact git commit used for the run.

## Reporting Checklist

- `git rev-parse HEAD`
- Python version and dependency-install command
- Full command line for every table, figure, or benchmark cell
- Paths to raw outputs and aggregation scripts
- External data, benchmark, or API-backed steps that were intentionally skipped
