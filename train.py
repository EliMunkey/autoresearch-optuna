"""
Autoresearch Optuna — The Agent's Playground

This is the ONLY file the AI agent modifies.
It must export create_sampler() returning a valid Optuna sampler.

Baseline: default TPESampler with standard parameters.
The agent evolves this to minimize trials-to-target on benchmark functions.
"""

import math
import optuna
from optuna.samplers import TPESampler, CmaEsSampler, BaseSampler


class TPEThenCmaEs(BaseSampler):
    """Start with tuned TPE for global search, switch to CMA-ES for local refinement."""

    def __init__(self, seed=None, switch_at=30):
        self._tpe = TPESampler(
            n_startup_trials=5,
            n_ei_candidates=48,
            multivariate=True,
            seed=seed,
            gamma=lambda n: max(1, int(math.ceil(0.20 * n))),  # top 20%
            consider_endpoints=True,
            warn_independent_sampling=False,
        )
        self._cmaes = CmaEsSampler(
            seed=seed,
            n_startup_trials=1,
            warn_independent_sampling=False,
        )
        self._switch_at = switch_at

    def _pick(self, study):
        return self._cmaes if len(study.trials) >= self._switch_at else self._tpe

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
    return TPEThenCmaEs(seed=seed, switch_at=30)
