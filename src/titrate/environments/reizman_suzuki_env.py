"""A real-data experiment environment: a GP emulator fit on real Suzuki-Miyaura
cross-coupling flow-chemistry data, wrapped behind the same ExperimentEnvironment
interface as the CSTR simulator (a thin specialization of TabularEmulatorEnvironment).

Data: Reizman, B. J.; Wang, Y.-M.; Buchwald, S. L.; Jensen, K. F. "Suzuki-Miyaura
cross-coupling optimization enabled by automated feedback." React. Chem. Eng.
2016, 1, 658-666. See data/README.md for provenance.

Why an emulator, not the raw table directly: a sequential optimization strategy
needs to query points that weren't in the original 96-experiment dataset. Fitting
a GP regression model on the real measurements and treating its predictions as
the queryable "ground truth" is the same technique the Summit benchmarking
package (Felton et al., 2021) uses for exactly this purpose -- it turns a fixed
real dataset into a continuously queryable benchmark function, at the cost of
the emulator's own regression error, which is reported (not hidden) via
`emulator_holdout_rmse()` and in the README's real-data validation section.

This dataset has no purity/impurity specification, so unlike the CSTR
environment there is no real engineering constraint here -- constraint_max is
+inf (always satisfied), which makes constrained BO on this environment
mathematically reduce to plain BO. This is stated explicitly rather than
inventing a constraint that isn't in the source data.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from titrate.environments.tabular_env import TabularEmulatorEnvironment

DATA_PATH = Path(__file__).resolve().parent.parent.parent.parent / "data" / "reizman_suzuki_case1.csv"
DEFAULT_CATALYST = "P1-L4"  # the most-sampled catalyst in the dataset (37 of 96 runs)


def _load_catalyst_subset(catalyst: str = DEFAULT_CATALYST) -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, skiprows=[1])  # row 1 is a "TYPE" metadata row, not data
    df["yld_frac"] = df["yld"] / 100.0  # yield reported as 0-100%, rescale to [0,1]
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
