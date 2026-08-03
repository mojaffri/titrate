"""Benchmark result plots. Every figure is generated from an actual
run_benchmark() DataFrame -- no placeholder numbers."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

STRATEGY_LABELS = {
    "random": "Random search",
    "grid": "Grid search",
    "lhs": "Latin Hypercube",
    "bo_unconstrained": "BO (unconstrained)",
    "bo_constrained": "Constrained BO",
}
STRATEGY_COLORS = {
    "random": "#9ca3af",
    "grid": "#f59e0b",
    "lhs": "#60a5fa",
    "bo_unconstrained": "#a78bfa",
    "bo_constrained": "#16a34a",
}


def plot_convergence(trials: pd.DataFrame, true_optimum: float, save_path: str | Path) -> None:
    """Mean best-feasible-so-far vs. iteration, one line per strategy, with
    an inter-quartile shaded band across seeds."""
    fig, ax = plt.subplots(figsize=(7, 5))
    for strategy, group in trials.groupby("strategy"):
        pivot = group.pivot(index="seed", columns="iteration", values="best_feasible_so_far")
        pivot = pivot.ffill(axis=1)  # NaN before first feasible point -> carry as "no result yet"
        iterations = pivot.columns.to_numpy()
        median = pivot.median(axis=0, skipna=True).to_numpy()
        q25 = pivot.quantile(0.25, axis=0).to_numpy()
        q75 = pivot.quantile(0.75, axis=0).to_numpy()
        color = STRATEGY_COLORS.get(strategy, None)
        ax.plot(iterations, median, label=STRATEGY_LABELS.get(strategy, strategy), color=color)
        ax.fill_between(iterations, q25, q75, alpha=0.15, color=color)

    ax.axhline(true_optimum, color="black", linestyle="--", linewidth=1, label="True optimum")
    ax.set_xlabel("Experiments performed")
    ax.set_ylabel("Best feasible yield found so far")
    ax.set_title("Sample efficiency: yield vs. experiment budget")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_experiments_to_threshold(summary: pd.DataFrame, fraction_pct: int, save_path: str | Path) -> None:
    column = f"experiments_to_{fraction_pct}pct"
    strategies = [s for s in STRATEGY_LABELS if s in summary["strategy"].unique()]
    data = [summary.loc[summary["strategy"] == s, column].dropna().to_numpy() for s in strategies]
    labels = [STRATEGY_LABELS[s] for s in strategies]

    fig, ax = plt.subplots(figsize=(7, 5))
    box = ax.boxplot(data, tick_labels=labels, patch_artist=True)
    for patch, strategy in zip(box["boxes"], strategies):
        patch.set_facecolor(STRATEGY_COLORS.get(strategy, "#cccccc"))
        patch.set_alpha(0.6)
    ax.set_ylabel(f"Experiments to reach {fraction_pct}% of true optimum")
    ax.set_title(f"Experiments needed to reach {fraction_pct}% of optimal yield")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_final_yield_distribution(summary: pd.DataFrame, true_optimum: float, save_path: str | Path) -> None:
    strategies = [s for s in STRATEGY_LABELS if s in summary["strategy"].unique()]
    data = [
        summary.loc[summary["strategy"] == s, "final_best_feasible_yield"].dropna().to_numpy()
        for s in strategies
    ]
    labels = [STRATEGY_LABELS[s] for s in strategies]

    fig, ax = plt.subplots(figsize=(7, 5))
    box = ax.boxplot(data, tick_labels=labels, patch_artist=True)
    for patch, strategy in zip(box["boxes"], strategies):
        patch.set_facecolor(STRATEGY_COLORS.get(strategy, "#cccccc"))
        patch.set_alpha(0.6)
    ax.axhline(true_optimum, color="black", linestyle="--", linewidth=1, label="True optimum")
    ax.set_ylabel("Final best feasible yield")
    ax.set_title("Final yield distribution across seeds")
    ax.legend(fontsize=8)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_calibration(empirical_coverage: dict[float, float], save_path: str | Path) -> None:
    nominal = sorted(empirical_coverage.keys())
    empirical = [empirical_coverage[q] for q in nominal]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1, label="Perfect calibration")
    ax.plot(nominal, empirical, marker="o", color="#16a34a", label="GP surrogate")
    ax.set_xlabel("Nominal predictive interval coverage")
    ax.set_ylabel("Empirical coverage")
    ax.set_title("GP uncertainty calibration")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
