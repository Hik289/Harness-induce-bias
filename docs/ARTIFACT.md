# Artifact Guide

This guide maps the public `Harness-induce-bias` repository to a reviewer-friendly artifact workflow for `Measuring Harness-Induced Belief Divergence in Multi-Step LLM Agents`. It is meant to make the release easier to inspect in the style of ICML, ICLR, NeurIPS, and similar artifact-review processes.

## What To Inspect First

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

## Minimal Verification

Run these checks in a fresh environment before launching expensive jobs:

```bash
python -m compileall -q .
python -m pytest tests -q
python scripts/anchor1_api_smoke.py
python scripts/anchor4_phase1_smoke.py
python scripts/anchor5_biwm_smoke.py
python scripts/sanity_6harness_K5.py
python tests/test_smoke.py
```

## Reproduction And Analysis Entry Points

These are the main tracked files to inspect for paper-scale or benchmark-scale reproduction. Some require arguments, credentials, downloaded benchmarks, or local data paths described in the README.

- `python analysis/biwm_v2_recompute.py`
- `python analysis/phase1_table1.py`
- `python analysis/phase1_table1_descriptive.py`

## Figure Assets

- `figures/intuition.pdf`
- `figures/intuition.png`

## Data, Credentials, And Generated Outputs

- API-backed runs should read credentials from environment variables or local `.env` files only; never commit real keys or provider-specific secrets.
- Record provider endpoint, model/deployment name, sampling parameters, and execution date for every API-backed table or figure.
- Treat generated JSONL files, logs, caches, model checkpoints, and benchmark downloads as local artifacts unless explicitly tracked as fixtures.
- For stochastic experiments, record seeds, task counts, dataset splits, and the exact git commit used for the run.

## Reviewer Reporting Checklist

- `git rev-parse HEAD`
- Python version and dependency-install command
- Full command line for every table, figure, or benchmark cell
- Paths to raw outputs and aggregation scripts
- External data, benchmark, or API-backed steps that were intentionally skipped
