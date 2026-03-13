"""
Autoresearch Optuna — The Agent's Playground

This is the ONLY file the AI agent modifies.
It must export create_sampler() returning a valid Optuna sampler.

Baseline: default TPESampler with standard parameters.
The agent evolves this to minimize trials-to-target on benchmark functions.
"""

from optuna.samplers import CmaEsSampler, QMCSampler, BaseSampler
from scipy.stats import norm
from scipy.stats.qmc import Sobol
import numpy as np


class SobolCmaRefine(BaseSampler):
    """Sobol-8 → CMA-ES → quasi-random Gaussian refinement.

    Exp 134: Replace pseudo-random Gaussian perturbation with
    quasi-random Sobol-based Gaussian for better 5D coverage.
    64 Sobol points (power of 2) transformed via inverse CDF.
    Sigma: 0.13 * exp(-0.11 * (n - 140)).
    Mean normalized regret: 0.1284 (-8.7% vs 0.1406 pseudo-random).
    """

    def __init__(self, seed=None):
        self._seed = seed
        self._qmc = QMCSampler(seed=seed, warn_independent_sampling=False)
        self._cmaes = CmaEsSampler(
            seed=seed,
            n_startup_trials=0,
            popsize=6,
            sigma0=0.2,
            warn_independent_sampling=False,
        )
        self._rng = np.random.RandomState(seed if seed is not None else 0)
        self._refinement_z = None
        self._param_order = None

    def _init_refinement(self, study):
        """Pre-generate 64 quasi-random Gaussian vectors for refinement."""
        self._param_order = sorted(study.best_trial.params.keys())
        d = len(self._param_order)
        sobol_engine = Sobol(
            d=d, scramble=True,
            seed=self._seed if self._seed is not None else 0,
        )
        # Generate 64 Sobol points (power of 2 for optimal balance)
        u = sobol_engine.random(64)
        # Transform to standard normal via inverse CDF
        self._refinement_z = norm.ppf(np.clip(u, 1e-10, 1 - 1e-10))

    def _pick(self, study):
        n = len(study.trials)
        if n < 8:
            return self._qmc
        if n < 140:
            return self._cmaes
        return None

    def infer_relative_search_space(self, study, trial):
        sampler = self._pick(study)
        if sampler is None:
            return {}
        return sampler.infer_relative_search_space(study, trial)

    def sample_relative(self, study, trial, search_space):
        sampler = self._pick(study)
        if sampler is None:
            return {}
        return sampler.sample_relative(study, trial, search_space)

    def sample_independent(self, study, trial, param_name, param_distribution):
        n = len(study.trials)

        if n >= 140:
            # Initialize quasi-random refinement vectors on first call
            if self._refinement_z is None:
                self._init_refinement(study)

            trial_idx = n - 140
            best_trial = study.best_trial

            if (trial_idx < 64
                    and param_name in best_trial.params
                    and self._param_order is not None
                    and param_name in self._param_order):
                dim_idx = self._param_order.index(param_name)
                z = self._refinement_z[trial_idx, dim_idx]

                best_val = best_trial.params[param_name]
                low = param_distribution.low
                high = param_distribution.high
                rng = high - low

                sigma_frac = 0.13 * np.exp(-0.11 * (n - 140))
                spread = rng * sigma_frac
                val = best_val + z * spread
                return max(low, min(high, val))

            # Consume RNG to keep state aligned for edge cases
            self._rng.normal(0, 1)
            return study.best_trial.params.get(
                param_name,
                self._qmc.sample_independent(
                    study, trial, param_name, param_distribution))

        sampler = self._pick(study)
        if sampler is not None:
            return sampler.sample_independent(
                study, trial, param_name, param_distribution)
        return self._qmc.sample_independent(
            study, trial, param_name, param_distribution)


def create_sampler(seed=None):
    return SobolCmaRefine(seed=seed)
