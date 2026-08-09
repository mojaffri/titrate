"""Validation and diagnostics for user-provided experimental tables."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DatasetHealth:
    raw_rows: int
    complete_rows: int
    unique_conditions: int
    duplicate_rows: int
    dropped_rows: int
    warnings: tuple[str, ...]


def validate_experiment_table(
    data: pd.DataFrame,
    input_columns: list[str],
    objective_column: str,
    constraint_column: str | None = None,
    min_rows: int = 8,
) -> tuple[pd.DataFrame, DatasetHealth]:
    """Return a finite, complete table or raise a user-actionable ValueError."""
    selected = input_columns + [objective_column] + ([constraint_column] if constraint_column else [])
    if not input_columns:
        raise ValueError("Select at least one input (decision variable) column.")
    if len(set(selected)) != len(selected):
        raise ValueError("Inputs, objective, and constraint must be different columns.")
    missing = [name for name in selected if name not in data.columns]
    if missing:
        raise ValueError(f"Missing selected columns: {', '.join(missing)}")

    numeric = data[selected].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    clean = numeric.dropna().copy()
    if len(clean) < min_rows:
        raise ValueError(
            f"Only {len(clean)} finite, complete rows remain; at least {min_rows} are required."
        )
    constant = [name for name in input_columns if clean[name].nunique(dropna=True) < 2]
    if constant:
        raise ValueError(f"Input columns must vary; constant: {', '.join(constant)}")

    unique_conditions = int(clean[input_columns].drop_duplicates().shape[0])
    if unique_conditions < max(4, len(input_columns) + 1):
        raise ValueError(
            f"Only {unique_conditions} unique conditions remain; add more distinct experiments."
        )
    duplicates = len(clean) - unique_conditions
    warnings: list[str] = []
    near_constant = [
        name for name in input_columns
        if clean[name].nunique() <= max(2, int(0.05 * len(clean)))
    ]
    if near_constant:
        warnings.append("Low-cardinality inputs: " + ", ".join(near_constant))
    if duplicates:
        warnings.append(
            f"{duplicates} repeated condition row(s) will be averaged when fitting the emulator."
        )
    health = DatasetHealth(
        raw_rows=len(data),
        complete_rows=len(clean),
        unique_conditions=unique_conditions,
        duplicate_rows=duplicates,
        dropped_rows=len(data) - len(clean),
        warnings=tuple(warnings),
    )
    return clean, health
