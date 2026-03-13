# Autoresearch Optuna

AI-driven meta-optimization of [Optuna](https://github.com/optuna/optuna) samplers on the BBOB benchmark suite.

An AI agent iteratively evolves `train.py` to find the best-performing sampler configuration, evaluated on all 24 BBOB (Black-Box Optimization Benchmarking) functions in 5 dimensions across 10 random seeds.

## Result

**0.1501 mean normalized regret** — 85% better than random sampling, 25% better than the CMA-ES baseline (0.2004), achieved through 97 experiments of systematic and creative search.

### Architecture: Sobol → CMA-ES → Multi-stage Refinement

| Phase | Trials | Strategy |
|-------|--------|----------|
| 1. Initialization | 0–7 | Sobol QMC (quasi-random space-filling) |
| 2. Optimization | 8–139 | CMA-ES (popsize=6, sigma0=0.2) |
| 3a. Medium refinement | 140–169 | Gaussian perturbation, sigma=1% of range |
| 3b. Tight refinement | 170–199 | Gaussian perturbation, sigma=0.2% of range |

### Key Insight

CMA-ES converges by ~trial 140 on most functions. The remaining trial budget is better spent on targeted local search around the best point than continuing CMA-ES with diminishing returns. Since `study.best_value` tracks the global best, refinement improvements are kept while failed perturbations don't hurt.

### Per-Category Results

| Category | Regret | Functions |
|----------|--------|-----------|
| Separable | 0.1161 | f1–f5 |
| Low conditioning | 0.0311 | f6–f9 |
| High conditioning | 0.0511 | f10–f14 |
| Multimodal (global) | 0.1663 | f15–f19 |
| Multimodal (weak) | 0.3623 | f20–f24 |
| **Mean** | **0.1501** | **All 24** |

## How It Works

```
train.py    — the ONLY file the AI agent modifies (exports create_sampler())
prepare.py  — immutable evaluation harness (BBOB × 10 seeds × 200 trials)
results.jsonl — experiment log
progress.png  — regret over time
```

Run an experiment:
```bash
python prepare.py
```

## Acknowledgements

This project is directly inspired by [Andrej Karpathy](https://github.com/karpathy)'s [autoresearch](https://x.com/karpathy/status/1895901790498996510) concept — using AI agents in an autonomous loop to conduct research by iteratively modifying code and evaluating results. The core idea of `train.py` (agent-modifiable) + `prepare.py` (immutable evaluator) comes from his framework.

Built with [Optuna](https://github.com/optuna/optuna), [cmaes](https://github.com/CyberAgentAILab/cmaes), and [OptunaHub BBOB benchmark](https://hub.optuna.org/benchmarks/bbob/).

## License

MIT
