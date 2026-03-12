# Autoresearch Optuna: Meta-Optimization of the TPE Sampler

## Overview

An autoresearch loop (inspired by Karpathy's autoresearch) that autonomously optimizes Optuna's TPE sampler. An AI agent modifies a single file (`train.py`), runs an evaluation harness, measures improvement, and keeps or discards changes. The metric is geometric mean trials-to-target across 5 standard benchmark functions.

## Architecture

```
autoresearch-optuna/
├── prepare.py          # (Immutable) Benchmark functions + evaluation harness
├── train.py            # (Agent-modified) TPE sampler configuration
├── program.md          # (Human-modified) Agent instructions
├── pyproject.toml      # Dependencies
├── .gitignore
├── results.jsonl       # Auto-generated: experiment log
├── progress.png        # Auto-generated: live graph (updated every run)
└── progress_X.png      # Auto-generated: snapshot every 25 experiments
```

## Metric

Geometric mean of median trials-to-target across:
- Sphere (10D, target 0.01)
- Rastrigin (10D, target 1.0)
- Rosenbrock (10D, target 1.0)
- Ackley (10D, target 1.0)
- Levy (10D, target 0.5)

Each function evaluated 10 seeds, median taken. Max 500 trials per run.
Baseline TPE scores ~80-120. Lower is better.

## Cycle Time

~1-2 minutes per experiment (5 functions x 10 seeds x ~1-3s each).

## Key Constraint

Agent only modifies `train.py`. Must export `create_sampler()` returning a valid Optuna sampler.
