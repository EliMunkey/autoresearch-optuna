"""
Autoresearch Optuna — Evaluation Harness (IMMUTABLE)

Do not modify this file. The AI agent only modifies train.py.

Benchmark: BBOB (Black-Box Optimization Benchmarking) — the gold standard
for evaluating continuous black-box optimizers. 24 functions spanning
5 difficulty categories, used in GECCO competitions worldwide.

This script:
1. Imports the sampler from train.py
2. Runs all 24 BBOB functions (5D) × 3 seeds
3. Computes score = sum of best values (lower = better)
4. Outputs the score and logs to results.jsonl
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
N_SEEDS = 3
FUNCTION_IDS = list(range(1, 25))  # all 24 BBOB functions

CATEGORIES = {
    "separable": [1, 2, 3, 4, 5],
    "low_conditioning": [6, 7, 8, 9],
    "high_conditioning": [10, 11, 12, 13, 14],
    "multimodal_global": [15, 16, 17, 18, 19],
    "multimodal_weak": [20, 21, 22, 23, 24],
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
    all_values = []
    print("Running BBOB evaluation (24 functions × 5D × 3 seeds)...")
    print("-" * 60)

    for cat_name, fids in CATEGORIES.items():
        cat_values = []
        for fid in fids:
            seed_values = []
            for s in range(N_SEEDS):
                seed = 42 + s
                val = evaluate_single(fid, sampler_factory, seed)
                seed_values.append(val)
            median_val = float(np.median(seed_values))
            cat_values.append(median_val)
            all_values.append(median_val)
            results[f"f{fid}"] = {
                "median": median_val,
                "seeds": seed_values,
            }

        cat_sum = sum(cat_values)
        print(f"  {cat_name:>22s}: sum={cat_sum:12.2f}  ({len(fids)} functions)")

    score = sum(all_values)
    print("-" * 60)
    print(f"  {'TOTAL SCORE':>22s}: {score:.2f}")
    print()

    return score, results


# ---------------------------------------------------------------------------
# Logging & graphing
# ---------------------------------------------------------------------------


def get_experiment_number(log_path):
    if not log_path.exists():
        return 1
    with open(log_path) as f:
        lines = [l for l in f if l.strip()]
    return len(lines) + 1


def log_result(log_path, experiment_num, score, results):
    entry = {
        "experiment": experiment_num,
        "score": round(score, 2),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "per_function": {
            name: round(data["median"], 2) for name, data in results.items()
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
    scores = []
    with open(log_path) as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                experiments.append(entry["experiment"])
                scores.append(entry["score"])

    if not experiments:
        return

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    colors = []
    best_so_far = float("inf")
    for s in scores:
        if s < best_so_far:
            colors.append("#22c55e")
            best_so_far = s
        else:
            colors.append("#ef4444")

    ax.scatter(experiments, scores, c=colors, s=40, zorder=3, edgecolors="white", linewidth=0.5)
    ax.plot(experiments, scores, color="#94a3b8", linewidth=0.8, alpha=0.5, zorder=2)

    best_scores = []
    best = float("inf")
    for s in scores:
        best = min(best, s)
        best_scores.append(best)
    ax.plot(experiments, best_scores, color="#22c55e", linewidth=2, linestyle="--",
            label=f"Best: {best:.0f}", zorder=2)

    if len(scores) > 0:
        baseline = scores[0]
        ax.axhline(y=baseline, color="#f59e0b", linewidth=1, linestyle=":",
                    label=f"Baseline: {baseline:.0f}", alpha=0.7)

    ax.set_xlabel("Experiment", fontsize=12)
    ax.set_ylabel("Score (sum of best values, lower = better)", fontsize=12)
    ax.set_title(
        f"Autoresearch Optuna [BBOB 24F×5D] — Exp {experiment_num} — Best: {min(scores):.0f}",
        fontsize=14,
    )
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
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
    score, results = run_evaluation()
    elapsed = time.time() - start

    log_result(log_path, experiment_num, score, results)
    plot_progress(log_path, project_dir, experiment_num)

    print(f"score: {score:.2f}")
    print(f"Time: {elapsed:.1f}s")
    print(f"Logged to results.jsonl (experiment {experiment_num})")


if __name__ == "__main__":
    main()
