"""
Autoresearch Optuna — Evaluation Harness (IMMUTABLE)

Do not modify this file. The AI agent only modifies train.py.

Benchmark: BBOB (Black-Box Optimization Benchmarking) — the gold standard
for evaluating continuous black-box optimizers. 24 functions spanning
5 difficulty categories, used in GECCO competitions worldwide.

Metrics:
- Raw score: sum of median best values across all 24 functions (lower = better)
- Normalized regret: (sampler_best - fopt) / (random_best - fopt) per function
  0.0 = optimal, 1.0 = random-level, <1.0 = better than random

This script:
1. Imports the sampler from train.py
2. Runs all 24 BBOB functions (5D) × 10 seeds
3. Computes raw score and normalized regret
4. Outputs results and logs to results.jsonl
5. Updates progress.png (and progress_X.png every 25 experiments)
"""

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import optuna
import optunahub

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------------------------------------------------------
# BBOB Configuration
# ---------------------------------------------------------------------------

BBOB_MODULE = None  # lazy-loaded

DIMENSION = 5
N_TRIALS = 200
N_SEEDS = 10
FUNCTION_IDS = list(range(1, 25))  # all 24 BBOB functions

CATEGORIES = {
    "separable": [1, 2, 3, 4, 5],
    "low_conditioning": [6, 7, 8, 9],
    "high_conditioning": [10, 11, 12, 13, 14],
    "multimodal_global": [15, 16, 17, 18, 19],
    "multimodal_weak": [20, 21, 22, 23, 24],
}

# Pre-computed optimal values (via scipy differential_evolution, 5 restarts)
FOPTS = {
    1: 79.48, 2: -209.88, 3: -462.09, 4: -461.10, 5: -9.21,
    6: 35.90, 7: 92.94, 8: 149.15, 9: 123.83, 10: -54.94,
    11: 76.27, 12: -621.11, 13: 29.97, 14: -52.35, 15: 1000.00,
    16: 71.35, 17: -16.94, 18: -16.94, 19: -102.53, 20: -546.50,
    21: 40.78, 22: -1000.00, 23: 6.87, 24: 108.60,
}

# Pre-computed random baseline medians (10 seeds, 200 trials each)
RANDOM_BASELINES = {
    1: 84.95, 2: 31972.38, 3: -400.37, 4: -388.67, 5: 17.46,
    6: 103.06, 7: 115.97, 8: 1068.83, 9: 1028.07, 10: 52162.56,
    11: 374.15, 12: 2227285.61, 13: 424.14, 14: -49.62, 15: 1062.84,
    16: 82.29, 17: -12.94, 18: 1.59, 19: -95.34, 20: -534.79,
    21: 50.63, 22: -991.76, 23: 9.20, 24: 147.80,
}


def get_bbob():
    global BBOB_MODULE
    if BBOB_MODULE is None:
        BBOB_MODULE = optunahub.load_module("benchmarks/bbob")
    return BBOB_MODULE


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_single(function_id, sampler_factory, seed):
    """Run one BBOB function with given sampler, return best value."""
    bbob = get_bbob()
    problem = bbob.Problem(
        function_id=function_id,
        dimension=DIMENSION,
        instance_id=1,
    )
    sampler = sampler_factory(seed)
    study = optuna.create_study(directions=problem.directions, sampler=sampler)
    study.optimize(problem, n_trials=N_TRIALS)
    return study.best_value


def normalized_regret(sampler_best, fopt, random_best):
    """Compute normalized regret. 0.0 = optimal, 1.0 = random-level."""
    denom = random_best - fopt
    if abs(denom) < 1e-10:
        return 0.0 if abs(sampler_best - fopt) < 1e-10 else 1.0
    return max(0.0, (sampler_best - fopt) / denom)


def run_evaluation():
    """Run full BBOB evaluation, return results dict."""
    sys.path.insert(0, str(Path(__file__).parent))
    try:
        from train import create_sampler
    except Exception as e:
        print(f"ERROR: Failed to import create_sampler from train.py: {e}")
        sys.exit(1)

    sampler_factory = lambda seed: create_sampler(seed=seed)

    results = {}
    all_medians = []
    all_regrets = []
    print(f"Running BBOB evaluation (24 functions x 5D x {N_SEEDS} seeds)...")
    print("-" * 70)

    for cat_name, fids in CATEGORIES.items():
        cat_medians = []
        cat_regrets = []
        for fid in fids:
            seed_values = []
            for s in range(N_SEEDS):
                seed = 42 + s
                val = evaluate_single(fid, sampler_factory, seed)
                seed_values.append(val)

            median_val = float(np.median(seed_values))
            fopt = FOPTS[fid]
            rand_base = RANDOM_BASELINES[fid]
            regret = normalized_regret(median_val, fopt, rand_base)

            cat_medians.append(median_val)
            cat_regrets.append(regret)
            all_medians.append(median_val)
            all_regrets.append(regret)
            results[f"f{fid}"] = {
                "median": median_val,
                "regret": round(regret, 4),
                "seeds": seed_values,
            }

        cat_sum = sum(cat_medians)
        cat_regret = float(np.mean(cat_regrets))
        print(f"  {cat_name:>22s}: sum={cat_sum:12.2f}  regret={cat_regret:.4f}  ({len(fids)}F)")

    raw_score = sum(all_medians)
    mean_regret = float(np.mean(all_regrets))
    median_regret = float(np.median(all_regrets))

    print("-" * 70)
    print(f"  {'RAW SCORE':>22s}: {raw_score:.2f}")
    print(f"  {'MEAN NORM. REGRET':>22s}: {mean_regret:.4f}  (0=optimal, 1=random)")
    print(f"  {'MEDIAN NORM. REGRET':>22s}: {median_regret:.4f}")
    print()

    return raw_score, mean_regret, median_regret, results


# ---------------------------------------------------------------------------
# Logging & graphing
# ---------------------------------------------------------------------------


def get_experiment_number(log_path):
    if not log_path.exists():
        return 1
    with open(log_path) as f:
        lines = [l for l in f if l.strip()]
    return len(lines) + 1


def log_result(log_path, experiment_num, raw_score, mean_regret, median_regret, results):
    entry = {
        "experiment": experiment_num,
        "raw_score": round(raw_score, 2),
        "mean_regret": round(mean_regret, 4),
        "median_regret": round(median_regret, 4),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "per_function": {
            name: {"median": round(data["median"], 2), "regret": data["regret"]}
            for name, data in results.items()
        },
    }
    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def plot_progress(log_path, output_dir, experiment_num):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not log_path.exists():
        return

    experiments = []
    regrets = []
    with open(log_path) as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                experiments.append(entry["experiment"])
                regrets.append(entry["mean_regret"])

    if not experiments:
        return

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    colors = []
    best_so_far = float("inf")
    for r in regrets:
        if r < best_so_far:
            colors.append("#22c55e")
            best_so_far = r
        else:
            colors.append("#ef4444")

    ax.scatter(experiments, regrets, c=colors, s=40, zorder=3, edgecolors="white", linewidth=0.5)
    ax.plot(experiments, regrets, color="#94a3b8", linewidth=0.8, alpha=0.5, zorder=2)

    best_regrets = []
    best = float("inf")
    for r in regrets:
        best = min(best, r)
        best_regrets.append(best)
    ax.plot(experiments, best_regrets, color="#22c55e", linewidth=2, linestyle="--",
            label=f"Best: {best:.4f}", zorder=2)

    ax.axhline(y=1.0, color="#ef4444", linewidth=1, linestyle=":",
               label="Random baseline (1.0)", alpha=0.7)

    if len(regrets) > 0:
        baseline = regrets[0]
        ax.axhline(y=baseline, color="#f59e0b", linewidth=1, linestyle=":",
                    label=f"Default TPE: {baseline:.4f}", alpha=0.7)

    ax.set_xlabel("Experiment", fontsize=12)
    ax.set_ylabel("Mean Normalized Regret (0=optimal, 1=random)", fontsize=12)
    ax.set_title(
        f"Autoresearch Optuna [BBOB 24F x 5D x {N_SEEDS}seeds] — Exp {experiment_num} — Best: {min(regrets):.4f}",
        fontsize=14,
    )
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    plt.tight_layout()

    fig.savefig(output_dir / "progress.png", dpi=150)
    if experiment_num % 25 == 0:
        snapshot = output_dir / f"progress_{experiment_num}.png"
        fig.savefig(snapshot, dpi=150)
        print(f"Snapshot saved: {snapshot.name}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    project_dir = Path(__file__).parent
    log_path = project_dir / "results.jsonl"

    experiment_num = get_experiment_number(log_path)
    print(f"=== Experiment {experiment_num} ===")
    print()

    start = time.time()
    raw_score, mean_regret, median_regret, results = run_evaluation()
    elapsed = time.time() - start

    log_result(log_path, experiment_num, raw_score, mean_regret, median_regret, results)
    plot_progress(log_path, project_dir, experiment_num)

    print(f"score: {raw_score:.2f}  |  mean_regret: {mean_regret:.4f}  |  median_regret: {median_regret:.4f}")
    print(f"Time: {elapsed:.1f}s")
    print(f"Logged to results.jsonl (experiment {experiment_num})")


if __name__ == "__main__":
    main()
