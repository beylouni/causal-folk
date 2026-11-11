"""Data loading utilities for Folktables (ACS PUMS)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from folktables import (
    ACSDataSource,
    ACSEmployment,
    ACSIncome,
    ACSMobility,
    ACSPublicCoverage,
    ACSTravelTime,
)

TASKS = {
    "ACSIncome": ACSIncome,
    "ACSEmployment": ACSEmployment,
    "ACSPublicCoverage": ACSPublicCoverage,
    "ACSMobility": ACSMobility,
    "ACSTravelTime": ACSTravelTime,
}


def load_acs_data(
    state: str = "CA",
    year: str = "2018",
    task_name: str = "ACSIncome",
    data_dir: str | Path = "data/raw",
    sample_size: int | None = None,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Download (if needed) and load Folktables data for a specific US state.

    Parameters
    ----------
    state : str
        Two-letter US state postal code (e.g., 'CA', 'TX', 'MI').
    year : str
        Survey year (e.g., '2018').
    task_name : str
        Folktables benchmark task ('ACSIncome', 'ACSEmployment', etc.).
    data_dir : str or Path
        Directory where raw data files are cached.
    sample_size : int, optional
        Subsample size (useful for quick experiments or TabPFN context limits).
    random_state : int
        Random seed for subsampling reproducibility.

    Returns
    -------
    X : pd.DataFrame
        Covariates / features.
    y : pd.Series
        Target variable (binary 0/1).
    group : pd.Series
        Demographic group attribute (e.g., race or sex).
    """
    if task_name not in TASKS:
        raise ValueError(f"Unknown task '{task_name}'. Available: {list(TASKS.keys())}")

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    data_source = ACSDataSource(
        survey_year=str(year),
        horizon="1-Year",
        survey="person",
        root_dir=str(data_dir),
    )

    acs_data = data_source.get_data(states=[state], download=True)
    task = TASKS[task_name]
    features, label, group = task.df_to_pandas(acs_data)

    X = features.reset_index(drop=True)
    y = pd.Series(label, name="target").reset_index(drop=True)
    group = pd.Series(group, name="group").reset_index(drop=True)

    if sample_size is not None and len(X) > sample_size:
        rng = np.random.RandomState(random_state)
        idx = rng.choice(len(X), size=sample_size, replace=False)
        X = X.iloc[idx].reset_index(drop=True)
        y = y.iloc[idx].reset_index(drop=True)
        group = group.iloc[idx].reset_index(drop=True)

    return X, y, group
