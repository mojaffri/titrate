"""Held-out GP versus PyTorch model comparison."""

from __future__ import annotations

import base64
import sys
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
LOGO_PATH = REPO_ROOT / "assets" / "titrate-logo.png"
LOGO_DATA_URI = "data:image/png;base64," + base64.b64encode(LOGO_PATH.read_bytes()).decode("ascii")
STYLE_PATH = REPO_ROOT / "webapp" / "styles.css"

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from titrate.environments.cstr_env import CSTREnvironment  # noqa: E402
from titrate.evaluation.model_comparison import (  # noqa: E402
    ComparisonResult,
    LearningCurvePoint,
    build_cstr_dataset,
    build_learning_curve,
    compare_surrogates,
    held_out_split,
)

st.set_page_config(
    page_title="Titrate | Model evidence lab",
    page_icon=str(LOGO_PATH),
    layout="wide",
    initial_sidebar_state="auto",
)
st.markdown(f"<style>{STYLE_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def run_lab(
    total_samples: int,
    torch_epochs: int,
    seed: int,
) -> tuple[ComparisonResult, list[LearningCurvePoint]]:
    env = CSTREnvironment()
    X, y = build_cstr_dataset(total_samples, seed)
    X_train, X_test, y_train, y_test = held_out_split(X, y, seed=seed)
    comparison = compare_surrogates(
        env.bounds,
        X_train,
        y_train,
        X_test,
        y_test,
        seed=seed,
        torch_epochs=torch_epochs,
        torch_patience=max(20, torch_epochs // 4),
        gp_restarts=1,
    )
    train_count = len(X_train)
    curve_sizes = tuple(sorted({20, 40, min(80, train_count), train_count}))
    learning_curve = build_learning_curve(
        env.bounds,
        X_train,
        y_train,
        X_test,
        y_test,
        curve_sizes,
        seed=seed,
        torch_epochs=max(40, min(80, torch_epochs)),
    )
    return comparison, learning_curve


def render_metric_table(comparison: ComparisonResult) -> None:
    rows = []
    for metric in comparison.metrics:
        row = asdict(metric)
        row["95% interval coverage"] = row.pop("interval_95_coverage")
        row["mean uncertainty"] = row.pop("mean_predictive_std")
        row["fit time (s)"] = row.pop("fit_seconds")
        rows.append(row)
    frame = pd.DataFrame(rows).set_index("model")
    st.dataframe(
        frame.style.format(
            {
                "rmse": "{:.4f}",
                "mae": "{:.4f}",
                "r2": "{:.3f}",
                "mean uncertainty": "{:.4f}",
                "95% interval coverage": "{:.1%}",
                "fit time (s)": "{:.2f}",
            }
        ),
        width="stretch",
    )


def render_prediction_evidence(comparison: ComparisonResult) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True, sharey=True)
    limits = [float(comparison.y_test.min()), float(comparison.y_test.max())]
    for ax, title, mean, std, color in [
        (axes[0], "Gaussian process", comparison.gp_mean, comparison.gp_std, "#0f766e"),
        (axes[1], "PyTorch MLP + MC dropout", comparison.torch_mean, comparison.torch_std, "#7c3aed"),
    ]:
        ax.errorbar(
            comparison.y_test,
            mean,
            yerr=1.96 * std,
            fmt="o",
            color=color,
            alpha=0.6,
            markersize=4,
            elinewidth=0.7,
            capsize=1,
        )
        ax.plot(limits, limits, linestyle="--", color="#64748b", linewidth=1)
        ax.set_title(title)
        ax.set_xlabel("True held-out yield")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Predicted yield (95% interval)")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def render_learning_curve(points: list[LearningCurvePoint]) -> None:
    frame = pd.DataFrame([asdict(point) for point in points])
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    colors = {"Gaussian process": "#0f766e", "PyTorch MLP": "#7c3aed"}
    for model, group in frame.groupby("model"):
        group = group.sort_values("train_samples")
        ax.plot(
            group["train_samples"],
            group["rmse"],
            marker="o",
            linewidth=2,
            label=model,
            color=colors[str(model)],
        )
    ax.set_xlabel("Training observations")
    ax.set_ylabel("Held-out RMSE (lower is better)")
    ax.set_title("Sample efficiency on the same held-out set")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def render_training_curve(comparison: ComparisonResult) -> None:
    history = comparison.torch_history
    frame = pd.DataFrame(
        {
            "epoch": np.arange(1, len(history.train_loss) + 1),
            "training loss": history.train_loss,
            "validation loss": history.val_loss,
        }
    ).set_index("epoch")
    st.line_chart(frame, color=["#7c3aed", "#f59e0b"])
    st.caption(
        f"Early stopping restored epoch {history.best_epoch + 1}; the chart shows every epoch evaluated."
    )


def render_interactive_prediction(comparison: ComparisonResult) -> None:
    env = CSTREnvironment()
    st.subheader("Try an operating condition")
    st.caption("Both models see the same point. The simulator value is shown only for honest comparison.")
    columns = st.columns(3)
    values = []
    labels = ("Temperature (K)", "Residence time (hr)", "Catalyst loading (mol%)")
    defaults = (360.0, 2.5, 1.0)
    steps = (1.0, 0.1, 0.05)
    for column, label, bounds, default, step in zip(columns, labels, env.bounds, defaults, steps):
        with column:
            values.append(
                st.slider(
                    label,
                    min_value=float(bounds[0]),
                    max_value=float(bounds[1]),
                    value=default,
                    step=step,
                )
            )

    point = np.asarray([values], dtype=float)
    gp_mean, gp_std = comparison.gp.predict(point)
    torch_mean, torch_std = comparison.torch.predict(point, mc_samples=100)
    truth = env.evaluate_noiseless(point[0])
    result = pd.DataFrame(
        [
            {"model": "Gaussian process", "predicted yield": gp_mean[0], "uncertainty (1 std)": gp_std[0]},
            {"model": "PyTorch MLP", "predicted yield": torch_mean[0], "uncertainty (1 std)": torch_std[0]},
            {"model": "Physics simulator", "predicted yield": truth.objective, "uncertainty (1 std)": 0.0},
        ]
    ).set_index("model")
    st.dataframe(result.style.format("{:.4f}"), width="stretch")
    if truth.constraint_value > env.constraint_max:
        st.warning(
            f"This point violates the impurity limit: {truth.constraint_value:.4f} > {env.constraint_max:.4f} mol/L."
        )
    else:
        st.success(f"This point is feasible: impurity = {truth.constraint_value:.4f} mol/L.")


st.markdown(
    f"""
    <section class="tt-hero">
        <div class="tt-hero-grid">
            <div class="tt-logo-wrap">
                <img src="{LOGO_DATA_URI}" alt="Titrate logo">
            </div>
            <div>
                <div class="tt-eyebrow">Held-out surrogate evaluation</div>
                <h1>Model evidence lab</h1>
                <p>
                    Compare a Gaussian process and a PyTorch MLP on identical observations
                    and one untouched test split. Metrics, uncertainty intervals, learning
                    curves, and predictions remain visible side by side.
                </p>
                <div class="tt-tags">
                    <span class="tt-tag">Shared held-out split</span>
                    <span class="tt-tag">GP posterior</span>
                    <span class="tt-tag">MC dropout</span>
                    <span class="tt-tag">Learning curves</span>
                </div>
            </div>
        </div>
    </section>
    """,
    unsafe_allow_html=True,
)
st.caption(
    "The GP remains Titrate's small-data BO default; the PyTorch path is evaluated as a "
    "larger-data surrogate and serving alternative."
)

with st.sidebar:
    st.header("Comparison settings")
    total_samples = st.slider("Total simulated observations", 120, 480, 160, step=40)
    torch_epochs = st.slider("Maximum PyTorch epochs", 60, 240, 100, step=20)
    seed = st.number_input("Random seed", min_value=0, max_value=10_000, value=42, step=1)
    st.caption("Changing these settings retrains both models on an identical split.")

with st.spinner("Training both models and measuring sample efficiency…"):
    comparison, learning_curve = run_lab(total_samples, torch_epochs, int(seed))

st.markdown('<div class="tt-section">Held-out evidence</div>', unsafe_allow_html=True)
st.caption(
    "Metrics use the same untouched 20% test split. Uncertainty is the GP posterior standard deviation "
    "or PyTorch MC-dropout standard deviation; coverage reports how often the 95% interval contains truth."
)
render_metric_table(comparison)
render_prediction_evidence(comparison)

left, right = st.columns(2)
with left:
    render_learning_curve(learning_curve)
with right:
    st.markdown("#### PyTorch optimization history")
    render_training_curve(comparison)

render_interactive_prediction(comparison)

with st.expander("Methodology and interpretation"):
    st.markdown(
        "- A seeded Latin Hypercube design covers the CSTR operating region.\n"
        "- The physics simulator is evaluated without measurement noise so this page isolates approximation error.\n"
        "- Both models receive identical training rows and are scored on one untouched test set.\n"
        "- Learning-curve fits use nested training subsets and the same test set.\n"
        "- This supervised comparison does not replace the BO benchmark: the GP is still the default acquisition model."
    )
