"""
Autoresearch Optuna — Evaluation Harness (IMMUTABLE)

Do not modify this file. The AI agent only modifies train.py.

This script:
1. Imports the sampler from train.py
2. Runs 5 benchmark functions × 10 seeds each
3. Computes geometric mean of median trials-to-target
4. Outputs the score and logs to results.jsonl
5. Updates progress.png (and progress_X.png every 25 experiments)
"""

import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------------------------------------------------------
# Benchmark functions
# ---------------------------------------------------------------------------

BENCHMARKS = {
    "sphere": {
        "dims": 5,
        "target": 1.0,
        "bounds": (-5.12, 5.12),
    },
    "rastrigin": {
        "dims": 5,
        "target": 20.0,
        "bounds": (-5.12, 5.12),
    },
    "rosenbrock": {
        "dims": 5,
        "target": 100.0,
        "bounds": (-5.0, 10.0),
    },
    "ackley": {
        "dims": 5,
        "target": 5.0,
        "bounds": (-5.0, 5.0),
    },
    "levy": {
        "dims": 5,
        "target": 2.0,
        "bounds": (-10.0, 10.0),
    },
}

MAX_TRIALS = 500
N_SEEDS = 10


def sphere(params):
    return sum(x**2 for x in params.values())


def rastrigin(params):
    A = 10
    vals = list(params.values())
    n = len(vals)
    return A * n + sum(x**2 - A * math.cos(2 * math.pi * x) for x in vals)


def rosenbrock(params):
    vals = list(params.values())
    return sum(
        100 * (vals[i + 1] - vals[i] ** 2) ** 2 + (1 - vals[i]) ** 2
        for i in range(len(vals) - 1)
    )


def ackley(params):
    vals = list(params.values())
    n = len(vals)
    sum_sq = sum(x**2 for x in vals)
    sum_cos = sum(math.cos(2 * math.pi * x) for x in vals)
    return (
        -20 * math.exp(-0.2 * math.sqrt(sum_sq / n))
        - math.exp(sum_cos / n)
        + 20
        + math.e
    )


def levy(params):
    vals = list(params.values())
    w = [1 + (x - 1) / 4 for x in vals]
    term1 = math.sin(math.pi * w[0]) ** 2
    terms = sum(
        (wi - 1) ** 2 * (1 + 10 * math.sin(math.pi * wi + 1) ** 2)
        for wi in w[:-1]
    )
    term3 = (w[-1] - 1) ** 2 * (1 + math.sin(2 * math.pi * w[-1]) ** 2)
    return term1 + terms + term3


FUNC_MAP = {
    "sphere": sphere,
    "rastrigin": rastrigin,
    "rosenbrock": rosenbrock,
    "ackley": ackley,
    "levy": levy,
}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_single(func_name, config, sampler_factory, seed):
    """Run one optimization, return trials-to-target (or MAX_TRIALS if not reached)."""
    func = FUNC_MAP[func_name]
    dims = config["dims"]
    lo, hi = config["bounds"]
    target = config["target"]

    sampler = sampler_factory(seed)

    study = optuna.create_study(direction="minimize", sampler=sampler)

    def objective(trial):
        params = {f"x{i}": trial.suggest_float(f"x{i}", lo, hi) for i in range(dims)}
        return func(params)

    best_so_far = float("inf")
    trials_to_target = MAX_TRIALS

    for t in range(MAX_TRIALS):
        study.optimize(objective, n_trials=1)
        val = study.best_value
        if val < best_so_far:
            best_so_far = val
        if best_so_far <= target:
            trials_to_target = t + 1
            break

    return trials_to_target


def make_sampler_factory(create_sampler):
    """Wrap create_sampler to pass seed for reproducibility."""

    def factory(seed):
        return create_sampler(seed=seed)

    return factory


def run_evaluation():
    """Run full evaluation, return results dict."""
    # Import the sampler from train.py
    sys.path.insert(0, str(Path(__file__).parent))
    try:
        from train import create_sampler
    except Exception as e:
        print(f"ERROR: Failed to import create_sampler from train.py: {e}")
        sys.exit(1)

    sampler_factory = make_sampler_factory(create_sampler)

    results = {}
    print("Running evaluation...")
    print("-" * 50)

    for func_name, config in BENCHMARKS.items():
        trials_list = []
        for seed in range(N_SEEDS):
            t = evaluate_single(func_name, config, sampler_factory, seed + 42)
            trials_list.append(t)
        median_trials = float(np.median(trials_list))
        results[func_name] = {
            "median": median_trials,
            "all_seeds": trials_list,
        }
        status = "HIT" if median_trials < MAX_TRIALS else "MISS"
        print(f"  {func_name:>12s}: median {median_trials:6.1f} trials  [{status}]  (range: {min(trials_list)}-{max(trials_list)})")

    # Geometric mean of medians
    medians = [results[f]["median"] for f in BENCHMARKS]
    geo_mean = float(np.exp(np.mean(np.log(medians))))

    print("-" * 50)
    print(f"  {'SCORE':>12s}: {geo_mean:.2f}")
    print()

    return geo_mean, results


# ---------------------------------------------------------------------------
# Logging & graphing
# ---------------------------------------------------------------------------


def get_experiment_number(log_path):
    """Get next experiment number from results log."""
    if not log_path.exists():
        return 1
    with open(log_path) as f:
        lines = [l for l in f if l.strip()]
    return len(lines) + 1


def log_result(log_path, experiment_num, score, results):
    """Append result to jsonl log."""
    entry = {
        "experiment": experiment_num,
        "score": round(score, 2),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "per_function": {
            name: round(data["median"], 1) for name, data in results.items()
        },
    }
    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def plot_progress(log_path, output_dir, experiment_num):
    """Update progress.png and save numbered snapshot every 25 experiments."""
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

    # Color points: green if improvement over previous best, red otherwise
    colors = []
    best_so_far = float("inf")
    for s in scores:
        if s < best_so_far:
            colors.append("#22c55e")  # green
            best_so_far = s
        else:
            colors.append("#ef4444")  # red

    ax.scatter(experiments, scores, c=colors, s=40, zorder=3, edgecolors="white", linewidth=0.5)
    ax.plot(experiments, scores, color="#94a3b8", linewidth=0.8, alpha=0.5, zorder=2)

    # Best score line
    best_scores = []
    best = float("inf")
    for s in scores:
        best = min(best, s)
        best_scores.append(best)
    ax.plot(experiments, best_scores, color="#22c55e", linewidth=2, linestyle="--", label=f"Best: {best:.2f}", zorder=2)

    # Baseline reference
    if len(scores) > 0:
        baseline = scores[0]
        ax.axhline(y=baseline, color="#f59e0b", linewidth=1, linestyle=":", label=f"Baseline: {baseline:.2f}", alpha=0.7)

    ax.set_xlabel("Experiment", fontsize=12)
    ax.set_ylabel("Score (geometric mean trials-to-target)", fontsize=12)
    ax.set_title(f"Autoresearch Optuna — Experiment {experiment_num} — Best: {min(scores):.2f}", fontsize=14)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Always save progress.png
    progress_path = output_dir / "progress.png"
    fig.savefig(progress_path, dpi=150)

    # Save numbered snapshot every 25 experiments
    if experiment_num % 25 == 0:
        snapshot_path = output_dir / f"progress_{experiment_num}.png"
        fig.savefig(snapshot_path, dpi=150)
        print(f"Snapshot saved: {snapshot_path.name}")

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
