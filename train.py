"""
Autoresearch Optuna — The Agent's Playground

This is the ONLY file the AI agent modifies.
It must export create_sampler() returning a valid Optuna sampler.

Baseline: default TPESampler with standard parameters.
The agent evolves this to minimize trials-to-target on benchmark functions.
"""

import math
import numpy as np
import optuna
from optuna.samplers import TPESampler, BaseSampler
from optuna.distributions import FloatDistribution


class AdaptiveTPE(BaseSampler):
    """TPE with adaptive exploration-exploitation balance.

    Phase 1 (trials 0-7): Independent sampling for broad exploration
    Phase 2 (trials 8+): Multivariate TPE with aggressive gamma for exploitation
    Additionally: dynamically adjusts n_ei_candidates based on progress.
    """

    def __init__(self, seed=None):
        self._explore_tpe = TPESampler(
            n_startup_trials=5,
            n_ei_candidates=16,
            multivariate=False,  # independent for exploration
            seed=seed,
            gamma=lambda n: max(1, int(math.ceil(0.25 * n))),
        )
        self._exploit_tpe = TPESampler(
            n_startup_trials=5,
            n_ei_candidates=32,
            multivariate=True,  # multivariate for exploitation
            seed=seed,
            gamma=lambda n: max(1, int(math.ceil(0.15 * n))),
        )
        self._switch_at = 8

    def _pick_sampler(self, study):
        n = len(study.trials)
        if n < self._switch_at:
            return self._explore_tpe
        return self._exploit_tpe

    def infer_relative_search_space(self, study, trial):
        return self._pick_sampler(study).infer_relative_search_space(study, trial)

    def sample_relative(self, study, trial, search_space):
        return self._pick_sampler(study).sample_relative(study, trial, search_space)

    def sample_independent(self, study, trial, param_name, param_distribution):
        return self._pick_sampler(study).sample_independent(
            study, trial, param_name, param_distribution
        )


def create_sampler(seed=None):
    """Return an Optuna sampler. This is the function prepare.py calls.
    seed is provided by prepare.py for reproducibility — pass it through."""
    return AdaptiveTPE(seed=seed)
