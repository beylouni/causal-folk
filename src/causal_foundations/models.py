"""Model helpers for TabPFN and baseline classifiers."""

from __future__ import annotations

from typing import Any

from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tabpfn import TabPFNClassifier


def get_tabpfn_classifier(
    device: str = "cpu",
    n_estimators: int = 4,
    random_state: int = 42,
    **kwargs: Any,
) -> TabPFNClassifier:
    """Instantiate a TabPFN classifier with convenient defaults."""
    return TabPFNClassifier(
        device=device,
        n_estimators=n_estimators,
        random_state=random_state,
        **kwargs,
    )


def get_baseline_classifiers(random_state: int = 42) -> dict[str, Any]:
    """Return dictionary of standard baseline models."""
    return {
        "LogisticRegression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(random_state=random_state, max_iter=1000)),
            ]
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=100,
            learning_rate=0.05,
            random_state=random_state,
            verbose=-1,
        ),
    }
