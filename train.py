"""
Autoresearch Optuna — The Agent's Playground

This is the ONLY file the AI agent modifies.
It must export create_sampler() returning a valid Optuna sampler.

Baseline: default TPESampler with standard parameters.
The agent evolves this to minimize trials-to-target on benchmark functions.
"""

from optuna.samplers import CmaEsSampler, QMCSampler, BaseSampler


class SobolCmaEs(BaseSampler):
    """Sobol QMC → CMA-ES (optimized configuration).

    Best found through 46 experiments of systematic search:
    - Sobol-8: power-of-2 QMC gives optimal space-filling in 5D
    - CMA-ES popsize=6: more generations than default (~9), faster convergence
    - CMA-ES sigma0=0.2: narrow initial step size for fast convergence from
      the best Sobol-discovered basin

    Phase 1 (0-7):  Sobol QMC — 8 points (power of 2) for 5D coverage
    Phase 2 (8+):   CMA-ES popsize=6, sigma0=0.2

    Results on BBOB (24F × 5D × 10 seeds × 200 trials):
    Mean normalized regret: 0.2004 (0=optimal, 1=random)
    Per-category: separable 0.17, low_cond 0.03, high_cond 0.06,
                  multimodal_global 0.25, multimodal_weak 0.46
    """

    def __init__(self, seed=None):
        self._qmc = QMCSampler(seed=seed, warn_independent_sampling=False)
        self._cmaes = CmaEsSampler(
            seed=seed,
            n_startup_trials=0,
            popsize=6,
            sigma0=0.2,
            warn_independent_sampling=False,
        )

    def _pick(self, study):
        n = len(study.trials)
        if n < 8:
            return self._qmc
        return self._cmaes

    def infer_relative_search_space(self, study, trial):
        return self._pick(study).infer_relative_search_space(study, trial)

    def sample_relative(self, study, trial, search_space):
        return self._pick(study).sample_relative(study, trial, search_space)

    def sample_independent(self, study, trial, param_name, param_distribution):
        return self._pick(study).sample_independent(
            study, trial, param_name, param_distribution
        )


def create_sampler(seed=None):
    """Return an Optuna sampler. This is the function prepare.py calls.
    seed is provided by prepare.py for reproducibility — pass it through."""
    return SobolCmaEs(seed=seed)
