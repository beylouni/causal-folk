"""Basic sanity checks for package setup and core dependencies."""

import numpy as np
import pandas as pd

from causal_foundations import (
    generate_synthetic_target,
    get_baseline_classifiers,
    get_tabpfn_classifier,
    inject_spurious_shortcut,
)


def test_imports() -> None:
    """Verify that core third-party packages import cleanly."""
    import folktables
    import lightgbm
    import shap
    import tabpfn
    import torch

    assert folktables is not None
    assert lightgbm is not None
    assert shap is not None
    assert tabpfn is not None
    assert torch is not None


def test_synthetic_scm_helpers() -> None:
    """Verify synthetic SCM target and spurious shortcut generator."""
    rng = np.random.RandomState(42)
    df = pd.DataFrame(
        {
            "AGEP": rng.randint(18, 70, size=100),
            "SCHL": rng.randint(1, 24, size=100),
            "WKHP": rng.randint(10, 50, size=100),
        }
    )

    y, _prob = generate_synthetic_target(
        df, causal_parents=["AGEP", "SCHL"], weights=[0.5, 0.8], random_state=42
    )
    assert len(y) == 100
    assert set(y.unique()).issubset({0, 1})

    spurious = inject_spurious_shortcut(y, correlation=0.9, random_state=42)
    assert len(spurious) == 100
    corr = np.corrcoef(spurious, y)[0, 1]
    assert corr > 0.75


def test_model_instantiation() -> None:
    """Verify model factory functions instantiate without error."""
    baselines = get_baseline_classifiers()
    assert "LogisticRegression" in baselines
    assert "LightGBM" in baselines

    tabpfn_model = get_tabpfn_classifier(device="cpu", n_estimators=2)
    assert tabpfn_model is not None
