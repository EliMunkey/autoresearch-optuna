"""
Autoresearch Optuna — The Agent's Playground

This is the ONLY file the AI agent modifies.
It must export create_sampler() returning a valid Optuna sampler.

Baseline: default TPESampler with standard parameters.
The agent evolves this to minimize trials-to-target on benchmark functions.
"""

import math
import optuna
from optuna.samplers import TPESampler, CmaEsSampler, QMCSampler, BaseSampler


class SobolTPECmaEs(BaseSampler):
    """Three-phase with Sobol quasi-random startup:
    Phase 1 (0-4):  Sobol QMC — optimal space-filling startup
    Phase 2 (5-24): Tuned multivariate TPE — global search
    Phase 3 (25+):  CMA-ES — local refinement
    """

    def __init__(self, seed=None):
        self._qmc = QMCSampler(seed=seed, warn_independent_sampling=False)
        self._tpe = TPESampler(
            n_startup_trials=0,  # no random startup — QMC already covered it
            n_ei_candidates=48,
            multivariate=True,
            seed=seed,
            gamma=lambda n: max(1, int(math.ceil(0.20 * n))),
            consider_endpoints=True,
            warn_independent_sampling=False,
        )
        self._cmaes = CmaEsSampler(
            seed=seed,
            n_startup_trials=1,
            warn_independent_sampling=False,
        )

    def _pick(self, study):
        n = len(study.trials)
        if n < 5:
            return self._qmc
        elif n < 25:
            return self._tpe
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
    return SobolTPECmaEs(seed=seed)
