"""
Autoresearch Optuna — The Agent's Playground

This is the ONLY file the AI agent modifies.
It must export create_sampler() returning a valid Optuna sampler.

Baseline: default TPESampler with standard parameters.
The agent evolves this to minimize trials-to-target on benchmark functions.
"""

import optuna
from optuna.samplers import TPESampler


def create_sampler(seed=None):
    """Return an Optuna sampler. This is the function prepare.py calls.
    seed is provided by prepare.py for reproducibility — pass it through."""
    return TPESampler(
        n_startup_trials=5,
        n_ei_candidates=24,
        multivariate=True,
        seed=seed,
    )
