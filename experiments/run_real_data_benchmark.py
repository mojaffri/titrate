"""Real-data validation: same benchmark methodology as run_benchmark.py, run
against a GP emulator fit on real Suzuki-Miyaura HTE data (Reizman et al.,
2016) instead of the physics simulator. See data/README.md for provenance
and titrate/environments/reizman_suzuki_env.py for how the emulator works.

Usage:
    python experiments/run_real_data_benchmark.py
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from titrate.environments.reizman_suzuki_env import ReizmanSuzukiEnvironment  # noqa: E402
from titrate.evaluation.benchmark import run_benchmark  # noqa: E402
from titrate.evaluation.metrics import summarize_experiments_to_threshold  # noqa: E402
from titrate.evaluation.plotting import (  # noqa: E402
    plot_convergence,
    plot_experiments_to_threshold,
    plot_final_yield_distribution,
)

BUDGET = 20
N_SEEDS = 30
BO_N_INITIAL = 5
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "real_data"


def main() -> None:
    warnings.filterwarnings("ignore", category=Warning, module="sklearn")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    env = ReizmanSuzukiEnvironment()
    rmse = env.emulator_holdout_rmse()
    optimum = env.true_optimum()
    print(f"Real-data environment: {env.n_real_experiments} real experiments (catalyst {env.catalyst})")
    print(f"Emulator 5-fold holdout RMSE: {rmse * 100:.1f} percentage points of yield")
    print(f"Emulator-predicted optimum: yield={optimum.objective * 100:.1f}% at x={optimum.x}")

    start = time.time()
    trials = run_benchmark(env, budget=BUDGET, n_seeds=N_SEEDS, bo_n_initial=BO_N_INITIAL)
    elapsed = time.time() - start
    print(f"Benchmark finished in {elapsed:.1f}s ({len(trials)} rows)")

    trials.to_csv(RESULTS_DIR / "benchmark_trials.csv", index=False)
    summary = summarize_experiments_to_threshold(trials, true_optimum=optimum.objective)
    summary.to_csv(RESULTS_DIR / "benchmark_summary.csv", index=False)

    headline = {}
    for strategy, group in summary.groupby("strategy"):
        headline[strategy] = {
            "median_experiments_to_90pct": float(group["experiments_to_90pct"].median()),
            "median_experiments_to_95pct": float(group["experiments_to_95pct"].median()),
            "median_final_best_feasible_yield": float(group["final_best_feasible_yield"].median()),
            "n_seeds_reached_90pct": int(group["experiments_to_90pct"].notna().sum()),
            "n_seeds": int(len(group)),
        }
    headline["emulator"] = {
        "n_real_experiments": env.n_real_experiments,
        "catalyst": env.catalyst,
        "holdout_rmse": rmse,
        "source": "Reizman et al., React. Chem. Eng. 2016, 1, 658-666",
    }
    headline["true_optimum"] = {
        "yield": optimum.objective,
        "x": optimum.x.tolist(),
        "dimension_names": list(env.dimension_names),
    }
    headline["benchmark_config"] = {"budget": BUDGET, "n_seeds": N_SEEDS, "bo_n_initial": BO_N_INITIAL}
    (RESULTS_DIR / "headline_results.json").write_text(json.dumps(headline, indent=2))

    plot_convergence(trials, optimum.objective, RESULTS_DIR / "convergence.png")
    plot_experiments_to_threshold(summary, 90, RESULTS_DIR / "experiments_to_90pct.png")
    plot_final_yield_distribution(summary, optimum.objective, RESULTS_DIR / "final_yield_distribution.png")

    print("\n=== Real-data headline results (median over seeds) ===")
    for strategy in ("random", "grid", "lhs", "bo_unconstrained", "bo_constrained"):
        row = headline[strategy]
        print(
            f"{strategy:20s} exp->90%: {row['median_experiments_to_90pct']:.1f}  "
            f"final yield: {row['median_final_best_feasible_yield']:.3f}"
        )
    print(f"\nWrote results to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
