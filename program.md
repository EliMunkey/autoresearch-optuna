# Autoresearch: Optuna TPE Sampler Optimization

## Goal

Minimize the geometric mean trials-to-target across 5 benchmark functions
(Sphere, Rastrigin, Rosenbrock, Ackley, Levy). Lower is better.

You are optimizing the optimizer itself. This is meta-optimization.

## Rules

- Only modify `train.py`
- `create_sampler()` must return a valid Optuna sampler
- Do not modify `prepare.py`
- Each experiment: edit train.py, then run `python prepare.py`
- Keep changes that improve the score, discard those that don't
- Record your reasoning in the git commit message

## Current baseline

Default TPESampler scores ~80-120 on the composite metric.

## Ideas to explore

Go wild. Novel approaches are encouraged. Even crazy ideas are worth trying —
the evaluation loop is fast, so there's no cost to being bold. Some starting points:

### Conservative tweaks
- Tune n_startup_trials (fewer = faster convergence, riskier)
- Tune n_ei_candidates (more = better EI approximation, slower per step)
- Adjust the good/bad observation split ratio (gamma)
- Toggle multivariate vs independent sampling

### Structural changes
- Override internal TPE methods (_sample, _log_pdf)
- Custom kernel density estimation with different bandwidths
- Adaptive strategies that change parameters mid-optimization
- Hybrid approaches: TPE + periodic random restarts
- Warm-starting from prior distributions based on early observations

### Wild ideas — try these
- Evolutionary mutation of the search distribution
- Bandit-style exploration bonuses on underexplored regions
- Momentum-based sampling (bias toward directions that improved)
- Curiosity-driven exploration (sample where uncertainty is highest)
- Population-based TPE (run multiple TPE instances, share information)
- Simulated annealing of the TPE temperature over time
- Gradient-free local search around best-found points
- Lévy flight perturbations for escaping local optima
- Information-theoretic acquisition functions (entropy search)
- Neural network surrogate models instead of KDE
- Learned acquisition functions from the optimization trajectory
- Multi-fidelity approaches (cheap approximations early, precise later)

## What NOT to do

- Don't add external dependencies beyond optuna, numpy, and scipy
- Don't make the sampler problem-specific (it must be general-purpose)
- Don't exceed 500 trials per function (prepare.py enforces this)
- Don't modify prepare.py or this file

## Philosophy

The best optimizer is one that converges fast on ALL types of landscapes —
convex, multimodal, deceptive, high-dimensional. Don't overfit to one function.
Surprise us. The crazier the idea, the more interesting the result, win or lose.
