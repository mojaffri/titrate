"""Renders assets/demo.gif: an animation of one real constrained-BO
trajectory on the CSTR environment -- the actual algorithm, actual GP
posterior, and actual points landing and converging toward the true
optimum. Not a UI screen recording; a direct visualization of the science.

Usage:
    python experiments/make_demo_gif.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import norm  # noqa: E402

from titrate.baselines.lhs import propose as lhs_propose  # noqa: E402
from titrate.environments.cstr_env import CSTREnvironment  # noqa: E402
from titrate.optimization.bo_loop import recommend_next_point  # noqa: E402

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
BUDGET = 25
N_INITIAL = 5
SEED = 3  # a representative, not cherry-picked, run: same seed style used throughout the benchmark
FRAME_MS = 3200
HOLD_LAST_FRAMES = 4


def render_frame(env, X, objectives, constraints, recommendation, iteration, optimum) -> np.ndarray:
    x_star = recommendation.x
    n_grid = 45
    T_grid = np.linspace(env.bounds[0, 0], env.bounds[0, 1], n_grid)
    tau_grid = np.linspace(env.bounds[1, 0], env.bounds[1, 1], n_grid)
    TT, TAU = np.meshgrid(T_grid, tau_grid)
    grid_points = np.column_stack([TT.ravel(), TAU.ravel(), np.full(TT.size, x_star[2])])
    mean, _ = recommendation.gp_objective.predict(grid_points)
    mean_grid = mean.reshape(TT.shape)
    c_mean, c_std = recommendation.gp_constraint.predict(grid_points)
    p_feasible = norm.cdf((env.constraint_max - c_mean) / np.maximum(c_std, 1e-9)).reshape(TT.shape)

    fig, (ax_land, ax_conv) = plt.subplots(1, 2, figsize=(11, 4.5))

    contour = ax_land.contourf(TT, TAU, mean_grid, levels=20, cmap="viridis", vmin=0, vmax=1)
    ax_land.contour(TT, TAU, p_feasible, levels=[0.5], colors="red", linewidths=1.5, linestyles="--")
    ax_land.scatter(X[:, 0], X[:, 1], color="white", edgecolor="black", s=30, zorder=5)
    ax_land.scatter([x_star[0]], [x_star[1]], color="#f59e0b", marker="*", s=280, edgecolor="black", zorder=6)
    ax_land.set_xlabel("Temperature (K)")
    ax_land.set_ylabel("Residence time (hr)")
    ax_land.set_title(f"Predicted yield landscape (experiment {iteration})")
    fig.colorbar(contour, ax=ax_land, label="Predicted yield")

    true_objs = np.array([env.evaluate_noiseless(x).objective for x in X])
    true_cons = np.array([env.evaluate_noiseless(x).constraint_value for x in X])
    feasible = true_cons <= env.constraint_max
    best_so_far = []
    running_best = -np.inf
    for i in range(len(X)):
        if feasible[i] and true_objs[i] > running_best:
            running_best = true_objs[i]
        best_so_far.append(running_best if running_best > -np.inf else np.nan)

    ax_conv.plot(np.arange(1, len(X) + 1), best_so_far, color="#16a34a", marker="o", markersize=4, linewidth=2)
    ax_conv.axhline(optimum.objective, color="black", linestyle=":", linewidth=1, label="True optimum")
    ax_conv.set_xlim(0.5, BUDGET + 0.5)
    ax_conv.set_ylim(0, 1.0)
    ax_conv.set_xlabel("Experiments performed")
    ax_conv.set_ylabel("Best feasible yield so far")
    ax_conv.set_title("Convergence (this run)")
    ax_conv.legend(fontsize=8, loc="lower right")

    fig.suptitle(
        f"Titrate: constrained Bayesian optimization  |  next recommended: "
        f"T={x_star[0]:.0f}K, τ={x_star[1]:.1f}hr, C_cat={x_star[2]:.2f}mol%",
        fontsize=10,
    )
    fig.tight_layout()

    fig.canvas.draw()
    image = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return image


def main() -> None:
    warnings.filterwarnings("ignore", category=Warning, module="sklearn")
    ASSETS_DIR.mkdir(exist_ok=True)

    env = CSTREnvironment()
    optimum = env.true_optimum()
    rng = np.random.default_rng(SEED)

    X = lhs_propose(env.bounds, N_INITIAL, rng)
    objectives = np.empty(N_INITIAL)
    constraints = np.empty(N_INITIAL)
    for i, x in enumerate(X):
        result = env.evaluate(x, rng)
        objectives[i] = result.objective
        constraints[i] = result.constraint_value
    frames = []

    n_iterations = BUDGET - N_INITIAL
    for step in range(n_iterations + 1):
        recommendation = recommend_next_point(env, X, objectives, constraints, rng, use_constraint=True)
        frames.append(render_frame(env, X, objectives, constraints, recommendation, len(X), optimum))
        if step == n_iterations:
            break
        result = env.evaluate(recommendation.x, rng)
        X = np.vstack([X, recommendation.x])
        objectives = np.append(objectives, result.objective)
        constraints = np.append(constraints, result.constraint_value)
        print(f"frame {step + 1}/{n_iterations}: queried x={recommendation.x}, observed yield={result.objective:.3f}")

    from PIL import Image

    pil_frames = [Image.fromarray(f) for f in frames]
    pil_frames += [pil_frames[-1]] * HOLD_LAST_FRAMES

    out_path = ASSETS_DIR / "demo.gif"
    pil_frames[0].save(
        out_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=FRAME_MS,
        loop=0,
    )
    total_seconds = len(pil_frames) * FRAME_MS / 1000
    print(f"Wrote {out_path} ({len(pil_frames)} frames, ~{total_seconds:.0f}s)")


if __name__ == "__main__":
    main()
