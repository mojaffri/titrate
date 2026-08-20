"""Suzuki-Miyaura reaction environment backed by a GP emulator.

Data source:
Reizman, B. J.; Wang, Y.-M.; Buchwald, S. L.; Jensen, K. F.
"Suzuki-Miyaura cross-coupling optimization enabled by automated feedback."
React. Chem. Eng. 2016, 1, 658-666.

Sequential optimization requires a queryable response surface. This environment fits
a GP to the published measurements and queries that emulator between measured points.
The emulator error is reported by ``emulator_holdout_rmse()`` and in the README.

The source dataset does not include a purity or process constraint, so
``constraint_max`` is infinite and all points are feasible.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from titrate.environments.tabular_env import TabularEmulatorEnvironment

DATA_PATH = Path(__file__).resolve().parent.parent.parent.parent / "data" / "reizman_suzuki_case1.csv"
DEFAULT_CATALYST = "P1-L4"  # the most-sampled catalyst in the dataset (37 of 96 runs)


def _load_catalyst_subset(catalyst: str = DEFAULT_CATALYST) -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, skiprows=[1])  # row 1 contains TYPE metadata
    df["yld_frac"] = df["yld"] / 100.0
    return df[df["catalyst"] == catalyst].reset_index(drop=True)


class ReizmanSuzukiEnvironment(TabularEmulatorEnvironment):
    def __init__(self, catalyst: str = DEFAULT_CATALYST, random_state: int = 0) -> None:
        data = _load_catalyst_subset(catalyst)
        super().__init__(
            data=data,
            input_columns=["t_res", "temperature", "catalyst_loading"],
            objective_column="yld_frac",
            dimension_names=("residence_time_s", "temperature_C", "catalyst_loading_mol_pct"),
            clip_range=(0.0, 1.0),
            random_state=random_state,
        )
        self.catalyst = catalyst
