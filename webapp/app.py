"""Titrate live demo: a working constrained-BO loop, not a mockup. Every
number on screen comes from the real titrate package -- the same GP
surrogate and from-scratch constrained-EI acquisition used in the
benchmark reported in the repository README.

Three data sources, same underlying code path (the whole point of the
ExperimentEnvironment abstraction):
  - the physics-based CSTR simulator, with adjustable engineering constraints
  - real Suzuki-Miyaura reaction data (Reizman et al., 2016)
  - a user-uploaded CSV, fit on the fly as a GP emulator

Run with:  streamlit run webapp/app.py
"""

from __future__ import annotations

import io
import json
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
LOGO_PATH = REPO_ROOT / "assets" / "titrate-logo.png"

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from scipy.stats import norm  # noqa: E402

from titrate.baselines.lhs import propose as lhs_propose  # noqa: E402
from titrate.environments.base import ExperimentEnvironment  # noqa: E402
from titrate.environments.cstr_env import CSTREnvironment  # noqa: E402
from titrate.environments.reizman_suzuki_env import ReizmanSuzukiEnvironment  # noqa: E402
from titrate.environments.tabular_env import TabularEmulatorEnvironment  # noqa: E402
from titrate.environments.validation import validate_experiment_table  # noqa: E402
from titrate.optimization.bo_loop import recommend_next_point  # noqa: E402

st.set_page_config(page_title="Titrate", page_icon=str(LOGO_PATH), layout="wide")

N_INITIAL = 5
MIN_UPLOAD_ROWS = 8

DIM_LABEL_PRESETS = {
    "temperature_K": "Temperature (K)",
    "residence_time_hr": "Residence time (hr)",
    "catalyst_loading_mol_pct": "Catalyst loading (mol%)",
    "residence_time_s": "Residence time (s)",
    "temperature_C": "Temperature (°C)",
}


def pretty_dim_name(name: str) -> str:
    return DIM_LABEL_PRESETS.get(name, name.replace("_", " ").strip().title())


OBJECTIVE_NAME_HINTS = ("yield", "conversion", "result", "output", "target", "score", "response", "purity")


def _guess_objective_column(numeric_cols: list[str]) -> str:
    """Best-effort default for which column a user probably wants to
    maximize -- prefers a name that sounds like an outcome (yield,
    conversion, ...), falling back to the last column (a common dataset
    convention) if nothing matches. Always overridable in the sidebar."""
    for col in numeric_cols:
        if any(hint in col.lower() for hint in OBJECTIVE_NAME_HINTS):
            return col
    return numeric_cols[-1]


@dataclass
class DisplayConfig:
    source_key: str
    value_name: str
    is_percent: bool
    optimum_label: str
    curve_label: str
    show_benchmark_overlay: bool


@dataclass(frozen=True)
class AppSettings:
    seed: int = 0
    n_initial: int = N_INITIAL
    xi: float = 0.01
    acquisition_restarts: int = 10


def fmt_value(display: DisplayConfig, value: float) -> str:
    return f"{value * 100:.1f}%" if display.is_percent else f"{value:.3g}"


def fmt_magnitude(display: DisplayConfig, value: float) -> str:
    return f"{value * 100:.2f} pp" if display.is_percent else f"{value:.3g}"


def fmt_delta(display: DisplayConfig, value: float) -> str:
    return f"+{fmt_magnitude(display, value)}"


# --------------------------------------------------------------------------
# Environment loading (one cached constructor per data source)
# --------------------------------------------------------------------------


@st.cache_resource
def load_cstr_environment(t_max: float, tau_max: float, ccat_max: float, impurity_max: float) -> CSTREnvironment:
    env = CSTREnvironment(
        temperature_bounds=(320.0, t_max),
        residence_time_bounds=(0.1, tau_max),
        catalyst_bounds=(0.0, ccat_max),
        impurity_max=impurity_max,
    )
    env.true_optimum()  # trigger + cache the differential_evolution solve once
    return env


@st.cache_resource
def load_reizman_environment() -> ReizmanSuzukiEnvironment:
    env = ReizmanSuzukiEnvironment()
    env.true_optimum()
    return env


@st.cache_resource
def load_uploaded_environment(
    file_bytes: bytes,
    input_cols: tuple[str, ...],
    objective_col: str,
    constraint_col: str | None,
    constraint_max: float,
    objective_direction: str,
    constraint_operator: str,
) -> TabularEmulatorEnvironment:
    df = pd.read_csv(io.BytesIO(file_bytes))
    needed = list(input_cols) + [objective_col] + ([constraint_col] if constraint_col else [])
    df = df[needed].dropna()
    env = TabularEmulatorEnvironment(
        df,
        input_columns=list(input_cols),
        objective_column=objective_col,
        constraint_column=constraint_col,
        constraint_max=constraint_max if constraint_col else float("inf"),
        objective_direction=objective_direction,
        constraint_operator=constraint_operator,
    )
    env.true_optimum()
    return env


@st.cache_data
def load_benchmark_trials() -> pd.DataFrame | None:
    path = REPO_ROOT / "results" / "benchmark_trials.csv"
    return pd.read_csv(path) if path.exists() else None


# --------------------------------------------------------------------------
# Sidebar: pick a data source, get back an environment + a display config
# --------------------------------------------------------------------------


def sidebar_pick_environment() -> tuple[ExperimentEnvironment, DisplayConfig, tuple, AppSettings] | None:
    st.sidebar.header("Problem setup")
    source = st.sidebar.radio(
        "Data source",
        ["Synthetic CSTR simulator", "Real Suzuki-Miyaura data (Reizman et al. 2016)", "Upload your own CSV"],
    )
    with st.sidebar.expander("Advanced settings"):
        seed = st.number_input("Random seed", min_value=0, max_value=2_147_483_647, value=0)
        n_initial = st.slider("Initial design size", 3, 12, N_INITIAL)
        xi = st.number_input("Exploration (xi)", min_value=0.0, max_value=1.0, value=0.01, step=0.01)
        acquisition_restarts = st.slider("Acquisition optimizer restarts", 3, 30, 10)
        st.caption("Defaults match the published benchmark unless an export says otherwise.")
    settings = AppSettings(int(seed), int(n_initial), float(xi), int(acquisition_restarts))

    if source == "Synthetic CSTR simulator":
        st.sidebar.subheader("Engineering constraints")
        t_max = st.sidebar.slider("Max temperature (K)", 360, 450, 400, step=5)
        tau_max = st.sidebar.slider("Max residence time (hr)", 1.0, 8.0, 5.0, step=0.5)
        ccat_max = st.sidebar.slider("Max catalyst loading (mol%)", 0.5, 4.0, 2.0, step=0.25)
        impurity_max = st.sidebar.slider("Max impurity (mol/L)", 0.02, 0.15, 0.05, step=0.01)
        env = load_cstr_environment(t_max, tau_max, ccat_max, impurity_max)
        display = DisplayConfig(
            source_key="cstr",
            value_name="Yield",
            is_percent=True,
            optimum_label="True optimum (ground-truth physics)",
            curve_label="True yield (ground-truth simulator)",
            show_benchmark_overlay=True,
        )
        return env, display, ("cstr", t_max, tau_max, ccat_max, impurity_max, settings), settings

    if source == "Real Suzuki-Miyaura data (Reizman et al. 2016)":
        st.sidebar.info(
            "96 real automated flow-chemistry experiments. A GP emulator is fit on the 37 runs with the "
            "most-sampled catalyst. No purity/engineering constraint is recorded in this dataset, so "
            "constrained and unconstrained BO behave identically here — see the README's real-data validation "
            "section for why."
        )
        env = load_reizman_environment()
        display = DisplayConfig(
            source_key="reizman",
            value_name="Yield",
            is_percent=True,
            optimum_label="Optimum (GP emulator estimate)",
            curve_label="Emulator mean (fit on real data)",
            show_benchmark_overlay=False,
        )
        return env, display, ("reizman", settings), settings

    # Upload your own CSV
    st.sidebar.caption(
        "Upload a CSV with at least 2 numeric input (decision variable) columns and one numeric objective "
        "column to optimize. A GP emulator is fit on your data on the fly, and the same optimization engine "
        "runs on it."
    )
    uploaded = st.sidebar.file_uploader("CSV file", type="csv")
    if uploaded is None:
        st.info(
            "⬆️ Upload a CSV in the sidebar to try Titrate on your own data — e.g. any small "
            "design-of-experiments table with a few numeric input columns and a result you want to maximize."
        )
        st.stop()

    file_bytes = uploaded.getvalue()
    try:
        df = pd.read_csv(io.BytesIO(file_bytes))
    except Exception as exc:  # noqa: BLE001
        st.error(f"Couldn't read that CSV: {exc}")
        st.stop()

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 2:
        st.error("Need at least 2 numeric columns in the CSV (≥1 input + 1 objective).")
        st.stop()

    guessed_objective = _guess_objective_column(numeric_cols)
    default_inputs = [c for c in numeric_cols if c != guessed_objective][:3]
    input_cols = st.sidebar.multiselect(
        "Input (decision variable) columns", numeric_cols, default=default_inputs
    )
    remaining = [c for c in numeric_cols if c not in input_cols]
    if not input_cols or not remaining:
        st.warning("Pick at least 1 input column and leave at least 1 numeric column free for the objective.")
        st.stop()
    objective_default_idx = remaining.index(guessed_objective) if guessed_objective in remaining else 0
    objective_col = st.sidebar.selectbox("Objective column", remaining, index=objective_default_idx)
    objective_direction = st.sidebar.radio(
        "Objective direction", ["Maximize", "Minimize"], horizontal=True
    ).lower()

    use_constraint = st.sidebar.checkbox("Add a constraint column?")
    constraint_col, constraint_max, constraint_operator = None, float("inf"), "<="
    if use_constraint:
        constraint_candidates = [c for c in numeric_cols if c not in input_cols and c != objective_col]
        if constraint_candidates:
            constraint_col = st.sidebar.selectbox("Constraint column", constraint_candidates)
            constraint_operator = st.sidebar.radio("Constraint rule", ["<=", ">="], horizontal=True)
            col_min, col_max = float(df[constraint_col].min()), float(df[constraint_col].max())
            if col_min < col_max:
                constraint_max = st.sidebar.slider(
                    f"Threshold for {constraint_col}", col_min, col_max, col_max
                )
            else:
                constraint_max = col_max
        else:
            st.sidebar.caption("No other numeric columns available to use as a constraint.")

    try:
        _, health = validate_experiment_table(
            df, list(input_cols), objective_col, constraint_col, min_rows=MIN_UPLOAD_ROWS
        )
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    with st.expander("Dataset health", expanded=True):
        h1, h2, h3, h4 = st.columns(4)
        h1.metric("Rows uploaded", health.raw_rows)
        h2.metric("Usable rows", health.complete_rows)
        h3.metric("Unique conditions", health.unique_conditions)
        h4.metric("Dropped rows", health.dropped_rows)
        for warning in health.warnings:
            st.warning(warning)

    try:
        env = load_uploaded_environment(
            file_bytes, tuple(input_cols), objective_col, constraint_col, constraint_max,
            objective_direction, constraint_operator,
        )
    except Exception as exc:  # noqa: BLE001
        st.error(f"Couldn't fit a model on that data: {exc}")
        st.stop()

    display = DisplayConfig(
        source_key="upload",
        value_name=objective_col,
        is_percent=False,
        optimum_label="Optimum (GP emulator estimate on your data)",
        curve_label="Emulator mean (fit on your data)",
        show_benchmark_overlay=False,
    )
    diagnostics = env.emulator_cross_validation()
    st.caption(
        f"Cross-validated emulator fit: RMSE {diagnostics['rmse']:.3g}, "
        f"MAE {diagnostics['mae']:.3g}, R² {diagnostics['r2']:.2f}. "
        "These describe the fitted proxy, not experimental measurement noise."
    )
    config_key = (
        "upload", uploaded.name, tuple(input_cols), objective_col, objective_direction,
        constraint_col, constraint_operator, constraint_max, settings,
    )
    return env, display, config_key, settings


# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------


def ensure_session_state(env: ExperimentEnvironment, config_key: tuple, settings: AppSettings) -> None:
    if st.session_state.get("env_key") == config_key:
        return
    rng = np.random.default_rng(settings.seed)
    n_initial = min(settings.n_initial, max(3, env.n_dims + 2))
    X = lhs_propose(env.bounds, n_initial, rng)
    objectives = np.empty(n_initial)
    constraints = np.empty(n_initial)
    for i, x in enumerate(X):
        result = env.evaluate(x, rng)
        objectives[i] = result.objective
        constraints[i] = result.constraint_value
    st.session_state.env_key = config_key
    st.session_state.rng = rng
    st.session_state.X = X
    st.session_state.objectives = objectives
    st.session_state.constraints = constraints


def run_recommended_experiment(env: ExperimentEnvironment, x_next: np.ndarray) -> None:
    result = env.evaluate(x_next, st.session_state.rng)
    st.session_state.X = np.vstack([st.session_state.X, x_next])
    st.session_state.objectives = np.append(st.session_state.objectives, result.objective)
    st.session_state.constraints = np.append(st.session_state.constraints, result.constraint_value)


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def render_header(env: ExperimentEnvironment, optimum, display: DisplayConfig) -> None:
    logo_col, title_col = st.columns([1, 10])
    logo_col.image(str(LOGO_PATH), width=72)
    title_col.title("Titrate")
    st.markdown("### Reach better experimental conditions with fewer runs")
    st.caption(
        "Constrained Bayesian optimization for chemical process design — interactive: swap data sources, "
        "adjust constraints, or upload your own CSV in the sidebar. "
        "Choose a process, learn from completed experiments, and repeat an uncertainty-aware recommendation. "
        "See the [GitHub repo](https://github.com/mojaffri/titrate) for multi-seed benchmarks and methodology."
    )

    X, objectives, constraints = st.session_state.X, st.session_state.objectives, st.session_state.constraints
    feasible = env.feasible_mask(constraints)
    if feasible.any():
        best = objectives[feasible].max() if env.objective_direction == "maximize" else objectives[feasible].min()
    else:
        best = float("nan")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(f"Current best {display.value_name.lower()} (observed)", fmt_value(display, best))
    col2.metric("Experiments performed", f"{len(X)}")
    col3.metric(display.optimum_label, fmt_value(display, optimum.objective))
    col4.metric("Feasible experiments so far", f"{int(feasible.sum())}/{len(X)}")


def render_constraints(env: ExperimentEnvironment) -> None:
    with st.expander("Search domain & constraints", expanded=False):
        lines = [
            f"- {pretty_dim_name(name)}: {low:.2f} – {high:.2f}"
            for name, (low, high) in zip(env.dimension_names, env.bounds)
        ]
        if np.isfinite(env.constraint_max):
            lines.append(
                f"- **{env.constraint_name}** {env.constraint_operator} {env.constraint_max:.3f} — a *soft*, nonlinear "
                f"constraint (not a simple bound); this is what makes constrained BO matter."
            )
        else:
            lines.append(
                "- No engineering constraint recorded for this data source — constrained and unconstrained "
                "BO are mathematically equivalent here."
            )
        st.markdown("\n".join(lines))


def render_recommendation(env: ExperimentEnvironment, display: DisplayConfig, settings: AppSettings):
    recommendation = recommend_next_point(
        env,
        st.session_state.X,
        st.session_state.objectives,
        st.session_state.constraints,
        st.session_state.rng,
        use_constraint=True,
        xi=settings.xi,
        n_acquisition_restarts=settings.acquisition_restarts,
    )

    st.subheader("Recommended next experiment")
    cols = st.columns(env.n_dims + 2)
    for col, dim_name, value in zip(cols, env.dimension_names, recommendation.x):
        col.metric(pretty_dim_name(dim_name), f"{value:.3g}")
    cols[-2].metric("P(feasible)", f"{recommendation.probability_feasible:.0%}")
    cols[-1].metric("Expected improvement", fmt_delta(display, recommendation.acquisition_value))
    direction = "high predicted performance" if env.objective_direction == "maximize" else "low predicted objective"
    st.info(
        f"Why this point: it balances {direction} "
        f"({fmt_value(display, recommendation.predicted_mean)}) with model uncertainty "
        f"(±{fmt_magnitude(display, recommendation.predicted_std)}), while the constraint model estimates "
        f"a {recommendation.probability_feasible:.0%} chance of feasibility."
    )
    with st.expander("How this recommendation was computed"):
        st.markdown(
            "Titrate fits Gaussian-process models to the objective and constraint, then maximizes "
            "**Expected Improvement × Probability of Feasibility**. The uncertainty shown is model "
            "uncertainty; it is not claimed to be laboratory measurement error."
        )

    if st.button("▶ Run this experiment", type="primary"):
        run_recommended_experiment(env, recommendation.x)
        st.rerun()

    return recommendation


def render_data_table(env: ExperimentEnvironment, display: DisplayConfig) -> None:
    X, objectives, constraints = st.session_state.X, st.session_state.objectives, st.session_state.constraints
    df = pd.DataFrame(X, columns=[pretty_dim_name(d) for d in env.dimension_names])
    df[f"Observed {display.value_name.lower()}"] = objectives
    if np.isfinite(env.constraint_max):
        df[f"Observed {env.constraint_name}"] = constraints
        df["Feasible"] = env.feasible_mask(constraints)
    df.index = pd.RangeIndex(1, len(df) + 1, name="Experiment #")
    st.dataframe(df.style.format(precision=3), width="stretch")
    st.download_button(
        "Download experiment history (CSV)",
        df.reset_index().to_csv(index=False).encode("utf-8"),
        file_name="titrate_experiment_history.csv",
        mime="text/csv",
    )


def render_gp_slice(env: ExperimentEnvironment, recommendation, display: DisplayConfig) -> None:
    x_star = recommendation.x
    dim_index = st.selectbox(
        "Slice dimension", range(env.n_dims), format_func=lambda i: pretty_dim_name(env.dimension_names[i])
    )
    dim0_grid = np.linspace(env.bounds[dim_index, 0], env.bounds[dim_index, 1], 100)
    slice_X = np.tile(x_star, (100, 1))
    slice_X[:, dim_index] = dim0_grid
    mean, std = recommendation.gp_objective.predict(slice_X)
    true_curve = np.array([env.evaluate_noiseless(x).objective for x in slice_X])

    X = st.session_state.X
    other_dims = [i for i in range(env.n_dims) if i != dim_index]
    if other_dims:
        other_dims_range = np.maximum(env.bounds[other_dims, 1] - env.bounds[other_dims, 0], 1e-9)
        normalized_dist = np.linalg.norm((X[:, other_dims] - x_star[other_dims]) / other_dims_range, axis=1)
        alphas = np.clip(1.0 - normalized_dist, 0.08, 1.0)
    else:
        alphas = np.ones(len(X))

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(dim0_grid, true_curve, color="black", linestyle="--", linewidth=1, label=display.curve_label)
    ax.plot(dim0_grid, mean, color="#16a34a", label="GP prediction")
    ax.fill_between(dim0_grid, mean - 2 * std, mean + 2 * std, color="#16a34a", alpha=0.15, label="95% interval")
    for xi, yi, alpha in zip(X[:, dim_index], st.session_state.objectives, alphas):
        ax.scatter([xi], [yi], color="#374151", s=20, zorder=5, alpha=alpha)
    label = "Observed data" + (" (faded = far from this slice)" if env.n_dims > 1 else "")
    ax.scatter([], [], color="#374151", s=20, label=label)
    ax.axvline(x_star[dim_index], color="#a78bfa", linestyle=":", label="Recommended point")
    ax.set_xlabel(pretty_dim_name(env.dimension_names[dim_index]))
    ax.set_ylabel(display.value_name)
    title_rest = ", ".join(
        f"{pretty_dim_name(env.dimension_names[i])}={x_star[i]:.2g}" for i in other_dims
    )
    ax.set_title(f"GP surrogate slice{' (' + title_rest + ')' if title_rest else ''}")
    ax.legend(fontsize=7, loc="lower center")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def render_landscape(env: ExperimentEnvironment, recommendation) -> None:
    if env.n_dims < 2:
        st.info("Add a 2nd input column to see the 2D optimization landscape.")
        return

    x_star = recommendation.x
    axis_options = list(range(env.n_dims))
    select_x, select_y = st.columns(2)
    d0 = select_x.selectbox(
        "Horizontal dimension", axis_options, index=0,
        format_func=lambda i: pretty_dim_name(env.dimension_names[i]),
    )
    y_options = [i for i in axis_options if i != d0]
    d1 = select_y.selectbox(
        "Vertical dimension", y_options, index=0,
        format_func=lambda i: pretty_dim_name(env.dimension_names[i]),
    )
    n_grid = 40
    d0_grid = np.linspace(env.bounds[d0, 0], env.bounds[d0, 1], n_grid)
    d1_grid = np.linspace(env.bounds[d1, 0], env.bounds[d1, 1], n_grid)
    G0, G1 = np.meshgrid(d0_grid, d1_grid)
    grid_points = np.tile(x_star, (G0.size, 1))
    grid_points[:, d0] = G0.ravel()
    grid_points[:, d1] = G1.ravel()

    mean, _ = recommendation.gp_objective.predict(grid_points)
    mean_grid = mean.reshape(G0.shape)

    feasibility_grid = None
    if recommendation.gp_constraint is not None and np.isfinite(env.constraint_max):
        c_mean, c_std = recommendation.gp_constraint.predict(grid_points)
        z = (env.constraint_max - c_mean) / np.maximum(c_std, 1e-9)
        p_feasible = norm.cdf(z) if env.constraint_operator == "<=" else norm.cdf(-z)
        feasibility_grid = p_feasible.reshape(G0.shape)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    contour = ax.contourf(G0, G1, mean_grid, levels=20, cmap="viridis")
    fig.colorbar(contour, ax=ax, label="Predicted value")
    if feasibility_grid is not None:
        ax.contour(G0, G1, feasibility_grid, levels=[0.5], colors="red", linewidths=1.5, linestyles="--")
    ax.scatter(
        st.session_state.X[:, d0], st.session_state.X[:, d1], color="white", edgecolor="black", s=25, label="Observed"
    )
    ax.scatter([x_star[d0]], [x_star[d1]], color="#f59e0b", marker="*", s=250, edgecolor="black", label="Recommended", zorder=6)
    ax.set_xlabel(pretty_dim_name(env.dimension_names[d0]))
    ax.set_ylabel(pretty_dim_name(env.dimension_names[d1]))
    ax.set_title("Predicted value landscape")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    if feasibility_grid is not None:
        st.caption("Red dashed line: the 50% feasibility contour from the constraint GP.")


def render_convergence(env: ExperimentEnvironment, optimum, display: DisplayConfig) -> None:
    X = st.session_state.X
    true_objectives = np.array([env.evaluate_noiseless(x).objective for x in X])
    true_constraints = np.array([env.evaluate_noiseless(x).constraint_value for x in X])
    is_feasible = env.feasible_mask(true_constraints)

    best_so_far = np.full(len(X), np.nan)
    maximize = env.objective_direction == "maximize"
    running_best = -np.inf if maximize else np.inf
    for i in range(len(X)):
        improved = true_objectives[i] > running_best if maximize else true_objectives[i] < running_best
        if is_feasible[i] and improved:
            running_best = true_objectives[i]
        if np.isfinite(running_best):
            best_so_far[i] = running_best

    fig, ax = plt.subplots(figsize=(6, 4.5))
    if display.show_benchmark_overlay:
        trials = load_benchmark_trials()
        if trials is not None:
            for strategy, label, color in [
                ("random", "Random search (benchmark median)", "#9ca3af"),
                ("lhs", "Latin Hypercube (benchmark median)", "#60a5fa"),
            ]:
                group = trials[trials["strategy"] == strategy]
                median = group.groupby("iteration")["best_feasible_so_far"].median()
                ax.plot(median.index, median.values, color=color, linestyle="--", linewidth=1, label=label)

    ax.plot(np.arange(1, len(X) + 1), best_so_far, color="#16a34a", linewidth=2, label="This session (constrained BO)")
    ax.axhline(optimum.objective, color="black", linestyle=":", linewidth=1, label=display.optimum_label)
    ax.set_xlabel("Experiments performed")
    ax.set_ylabel(f"Best feasible {display.value_name.lower()} found so far")
    ax.set_title("This session's convergence")
    ax.legend(fontsize=7)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption(
        "Scored against the noiseless ground-truth function, not the noisy values actually observed during "
        "the session — matching the benchmark methodology in the README."
    )


def main() -> None:
    picked = sidebar_pick_environment()
    if picked is None:
        return
    env, display, config_key, settings = picked
    optimum = env.true_optimum()
    ensure_session_state(env, config_key, settings)

    render_header(env, optimum, display)
    render_constraints(env)

    if st.button("↺ Reset session"):
        for key in ("env_key", "rng", "X", "objectives", "constraints"):
            st.session_state.pop(key, None)
        st.rerun()

    recommendation = render_recommendation(env, display, settings)

    export_payload = {
        "source": display.source_key,
        "objective": display.value_name,
        "objective_direction": env.objective_direction,
        "constraint": None if not np.isfinite(env.constraint_max) else {
            "name": env.constraint_name,
            "operator": env.constraint_operator,
            "threshold": env.constraint_max,
        },
        "settings": settings.__dict__,
        "recommended_condition": dict(zip(env.dimension_names, recommendation.x.tolist())),
        "predicted_objective": recommendation.predicted_mean,
        "model_std": recommendation.predicted_std,
        "probability_feasible": recommendation.probability_feasible,
        "acquisition_value": recommendation.acquisition_value,
    }
    st.download_button(
        "Download recommendation (JSON)",
        json.dumps(export_payload, indent=2),
        file_name="titrate_recommendation.json",
        mime="application/json",
    )

    if display.source_key == "reizman":
        st.info(
            "Data provenance: Reizman et al. (2016), 96 published automated flow-chemistry runs; "
            "this demo fits a GP to the 37 runs using the most-sampled catalyst. New in-app outcomes "
            "are emulator-generated, not additional laboratory measurements."
        )

    tab_data, tab_slice, tab_landscape, tab_convergence = st.tabs(
        ["Experimental data", "GP prediction slice", "Optimization landscape", "Convergence"]
    )
    with tab_data:
        render_data_table(env, display)
    with tab_slice:
        render_gp_slice(env, recommendation, display)
    with tab_landscape:
        render_landscape(env, recommendation)
    with tab_convergence:
        render_convergence(env, optimum, display)


if __name__ == "__main__":
    main()
