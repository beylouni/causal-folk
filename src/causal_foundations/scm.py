"""Structural Causal Model (SCM) injection helpers and starter functions."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def generate_synthetic_target(
    X: pd.DataFrame,
    causal_parents: Sequence[str],
    weights: Sequence[float],
    noise_std: float = 0.1,
    random_state: int = 42,
) -> tuple[pd.Series, pd.Series]:
    """Generate a controlled synthetic binary target Y using a logistic SCM.

    P(Y = 1 | Pa(Y)) = sigmoid(sum(w_i * X_i) + eps)

    Parameters
    ----------
    X : pd.DataFrame
        Covariates from Folktables.
    causal_parents : list of str
        Features chosen as ground-truth direct causes.
    weights : list of float
        Structural coefficients corresponding to each causal parent.
    noise_std : float
        Standard deviation of Gaussian noise.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    y : pd.Series
        Binary synthetic target.
    prob : pd.Series
        Underlying probability before Bernoulli sampling.
    """
    rng = np.random.RandomState(random_state)
    parent_vals = X[list(causal_parents)].values.astype(float)

    # Standardize parents internally so weights act on a comparable scale
    mean = np.mean(parent_vals, axis=0)
    std = np.std(parent_vals, axis=0) + 1e-8
    norm_vals = (parent_vals - mean) / std

    logits = np.dot(norm_vals, np.asarray(weights)) + rng.normal(0, noise_std, size=len(X))
    prob = 1.0 / (1.0 + np.exp(-logits))
    y = (rng.rand(len(prob)) < prob).astype(int)

    return pd.Series(y, name="target", index=X.index), pd.Series(prob, name="prob", index=X.index)


def inject_spurious_shortcut(
    y: pd.Series | np.ndarray,
    correlation: float = 0.85,
    random_state: int = 42,
) -> pd.Series:
    """Generate a continuous feature correlated with Y to test shortcut learning.

    Parameters
    ----------
    y : array-like
        Target variable.
    correlation : float
        Target correlation alpha in [-1.0, 1.0].
        - Close to 1.0: Strong shortcut.
        - Close to 0.0: Independent noise.
        - Negative: Inverted correlation (anti-shortcut under shift).
    """
    rng = np.random.RandomState(random_state)
    y_arr = np.asarray(y, dtype=float)

    y_std = np.std(y_arr)
    y_norm = (y_arr - np.mean(y_arr)) / (y_std if y_std > 0 else 1.0)
    noise = rng.normal(0, 1, size=len(y_arr))

    alpha = np.clip(correlation, -1.0, 1.0)
    signal = alpha * y_norm + np.sqrt(max(0.0, 1.0 - alpha**2)) * noise
    return pd.Series(signal, name="SPURIOUS_SIGNAL")
