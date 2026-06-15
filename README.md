# Harness-Induced Belief Bias in LLM Software Agents

Code release for the paper:

> **Harness-Induced Belief Bias: How Agent Execution Interfaces Shape LLM World-Model Trajectories**
> Anonymous Authors, 2026

---

## Repository Layout

```
code/
├── core/
│   ├── belief_schema.py          # JSON belief schema (5 blocks)
│   ├── harness_base.py           # Abstract Harness (O,A,V,G,R,L six-tuple)
│   ├── rollout.py                # K-step LLM belief rollout
│   ├── llm_client.py             # OpenAI/Azure client (key via env var)
│   └── jsonl_logger.py           # Per-step JSONL logging
├── harnesses/
│   ├── h0_raw.py                 # H0: raw baseline
│   ├── h1_structured.py          # H1: structured observations
│   ├── h2_risk_gated.py          # H2: risk gate (blocks destructive actions)
│   ├── h3_repair_heavy.py        # H3: repair-heavy
│   ├── h4_verification_selective.py
│   └── h5_cost_aware.py
├── biwm/
│   ├── canonical_belief.py       # BIWM-1: canonical schema
│   ├── blocked_action_log.py     # BIWM-2: blocked-action logging
│   ├── repair_unrolled.py        # BIWM-3: unrolled repair transitions
│   ├── verification_mask.py      # BIWM-4: verification mask
│   ├── shadow_execution.py       # BIWM-5: shadow execution
│   └── cross_harness_align.py    # BIWM-6: cross-harness alignment
├── benchmark/
│   ├── hibench_loader.py         # HIBench-Code v0 loader
│   ├── terminal_bench_adapter.py # Terminal-Bench adapter
│   └── swebench_adapter.py       # SWE-bench Verified adapter
├── scripts/
│   ├── phase1_main.py            # Main experiment (192 rollouts)
│   ├── long_horizon_K20.py       # Long-horizon K=1→20
│   ├── swebench_subset.py        # SWE-bench 10-task cross-benchmark
│   ├── g2_terminal_bench.py      # Terminal-Bench replication
│   └── anchor*.py                # Smoke / sanity checks
├── analysis/
│   ├── METRICS_SPEC.md           # D_belief metric specification
│   ├── phase1_table1.py          # Reproduce Table 1
│   ├── g4_recompute.py           # AUROC / calibration
│   └── ...
├── figures/
│   ├── make_pipeline.py          # System pipeline figure
│   ├── make_intuition.py         # Intuition figure
│   ├── make_long_horizon.py      # K=1→20 trajectory figure
│   └── make_figures.py           # Phase 1 + BIWM figures
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone https://github.com/YOUR_ORG/harness-belief-bias.git
cd harness-belief-bias
conda create -n hibench_env python=3.11 -y
conda activate hibench_env
pip install -r requirements.txt
```

---

## Configuration

Set credentials via environment variables — **never hardcode keys**:

```bash
export AZURE_OPENAI_API_KEY="your-key-here"
export AZURE_OPENAI_ENDPOINT="https://your-resource.services.ai.azure.com/openai/v1"
export AZURE_OPENAI_DEPLOYMENT="gpt-4o"
```

Standard OpenAI also works:
```bash
export AZURE_OPENAI_ENDPOINT="https://api.openai.com/v1"
export AZURE_OPENAI_API_KEY="sk-..."
```

---

## Quickstart

```bash
# Smoke test (~30 sec)
python scripts/anchor2_h0_endtoend.py --tasks 1 --K 1 3

# Full Phase 1 — 8 tasks × 6 harnesses × K={1,3,5,8} (~45 min)
python scripts/phase1_main.py \
    --tasks data/hibench_code/v0_toy/tasks.json \
    --horizons 1 3 5 8 --seeds 42 \
    --output logs/phase1_main/

# Reproduce Table 1
python analysis/phase1_table1.py --log-dir logs/phase1_main/

# Long-horizon K=20 (~2 h)
python scripts/long_horizon_K20.py \
    --tasks data/hibench_code/v0_toy/tasks.json \
    --horizons 12 16 20 --output logs/long_horizon_K20/

# Reproduce all figures
python figures/make_pipeline.py
python figures/make_intuition.py
python figures/make_long_horizon.py
python figures/make_figures.py
```

---

## Using Pre-computed Data

Unzip the companion data release (`harness-belief-bias-data.zip`):

```bash
unzip harness-belief-bias-data.zip   # creates data/
python analysis/phase1_table1.py --log-dir data/logs/phase1_main/
```

---

## Metric: D_belief

Full specification in `analysis/METRICS_SPEC.md`. Summary:

| Component | Weight | Description |
|-----------|--------|-------------|
| D_cat | 0.15 | Ordinal distance on (progress, risk, recoverability) |
| D_fail | 0.20 | Failure-mode label mismatch |
| D_set | 0.35 | Jaccard distance on constraint sets |
| D_num | 0.20 | Normalised L1 on 7 numeric predictions |
| D_act | 0.10 | Next-action recommendation mismatch |

---

## Citation

```bibtex
@article{anonymous2026harnessbias,
  title={Harness-Induced Belief Bias: How Agent Execution Interfaces
         Shape {LLM} World-Model Trajectories},
  author={Anonymous Authors},
  journal={arXiv preprint},
  year={2026}
}
```

---

## License

MIT License.
