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


PERTURBATIONS = ("negate", "scale", "shift", "permute", "constant", "noise")


def perturb_feature(
    X: pd.DataFrame,
    feature: str,
    kind: str,
    value: float = 1.0,
    random_state: int = 42,
) -> pd.DataFrame:
    """Return a copy of ``X`` with a single feature perturbed (test-time intervention).

    Only ``feature`` changes; every other column (and the true target) stays the same,
    so any change in the model's behaviour comes from its reliance on that feature.

    Parameters
    ----------
    X : pd.DataFrame
        Test covariates.
    feature : str
        Column to perturb (e.g. the spurious feature ``"S"``).
    kind : str
        One of:
        - ``"negate"``: ``S -> -S`` (inverts the correlation with Y).
        - ``"scale"``: ``S -> value * S`` (``value < 1`` weakens, ``> 1`` strengthens).
        - ``"shift"``: ``S -> S + value`` (moves the whole distribution).
        - ``"permute"``: shuffles ``S`` across rows (breaks the link to Y, keeps marginal).
        - ``"constant"``: ``do(S = value)`` for every row.
        - ``"noise"``: ``S -> S + N(0, value^2)`` (dilutes the correlation).
    value : float
        Parameter of the perturbation (ignored by ``"negate"`` and ``"permute"``).
    random_state : int
        Seed for ``"permute"`` and ``"noise"``.
    """
    if kind not in PERTURBATIONS:
        raise ValueError(f"Unknown perturbation '{kind}'. Available: {list(PERTURBATIONS)}")
    if feature not in X.columns:
        raise KeyError(f"Feature '{feature}' not in X.")

    rng = np.random.RandomState(random_state)
    X_new = X.copy()
    col = X_new[feature].to_numpy(dtype=float)

    if kind == "negate":
        col = -col
    elif kind == "scale":
        col = value * col
    elif kind == "shift":
        col = col + value
    elif kind == "permute":
        col = rng.permutation(col)
    elif kind == "constant":
        col = np.full_like(col, value)
    elif kind == "noise":
        col = col + rng.normal(0, value, size=len(col))

    X_new[feature] = col
    return X_new


# Structural equation of Y in the prototype SCM (data/prototype/scm_geracao_dados.ipynb):
# Y := 1[1.5 * X1 - 1.0 * X2 + eps_Y > 0],  eps_Y ~ Logistic(0, 1)
PROTOTYPE_Y_WEIGHTS = {"X1": 1.5, "X2": -1.0}


def logistic_scm_probability(
    X: pd.DataFrame,
    weights: dict[str, float] | None = None,
) -> pd.Series:
    """True causal probability ``P(Y = 1 | Pa(Y))`` of a logistic-noise SCM.

    With ``Y := 1[sum(w_i * X_i) + eps_Y > 0]`` and ``eps_Y ~ Logistic(0, 1)``,
    ``P(Y = 1 | Pa(Y)) = sigmoid(sum(w_i * X_i))``. Columns outside ``weights`` (e.g. the
    spurious ``S``) are ignored, since they are not causes of Y.
    """
    weights = PROTOTYPE_Y_WEIGHTS if weights is None else weights
    logit = sum(w * X[col].to_numpy(dtype=float) for col, w in weights.items())
    return pd.Series(1.0 / (1.0 + np.exp(-logit)), index=X.index, name="p_true")


def prototype_counterfactual(
    df: pd.DataFrame,
    alpha: float,
    interventions: dict[str, float | np.ndarray],
    weights: dict[str, float] | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Counterfactual version of each row of the prototype SCM under ``do(...)``.

    Follows Pearl's three steps for ``Y := 1[w . Pa(Y) + eps_Y > 0]`` and
    ``S := alpha * (2Y - 1) + eps_S``:

    1. **Abduction**: recover ``eps_S = S - alpha * (2Y - 1)`` exactly, and sample
       ``eps_Y`` from its posterior given the observed Y (a truncated logistic).
    2. **Action**: overwrite the intervened columns (``X1``, ``X2`` or ``Z``).
    3. **Prediction**: recompute Y and then S with the same noise terms.

    Parameters
    ----------
    df : pd.DataFrame
        Rows from one environment, with columns ``X1, X2, S, Z, Y``.
    alpha : float
        Strength of ``Y -> S`` in that environment.
    interventions : dict
        ``{column: value}``; the value can be a scalar or an array (one per row).
    weights : dict, optional
        Weights of Y's structural equation (defaults to the prototype's).
    random_state : int
        Seed for sampling ``eps_Y`` in the abduction step.

    Returns
    -------
    pd.DataFrame
        Same columns as ``df``, with ``Y`` and ``S`` propagated through the SCM.
    """
    weights = PROTOTYPE_Y_WEIGHTS if weights is None else weights
    for col in interventions:
        if col in ("Y", "S"):
            raise ValueError("Intervene on exogenous columns (X1, X2, Z); Y and S are recomputed.")

    rng = np.random.RandomState(random_state)
    y = df["Y"].to_numpy(dtype=int)

    # 1) Abduction. Y = 1 iff eps_Y > -logit, so eps_Y | Y is a truncated logistic.
    logit = sum(w * df[c].to_numpy(dtype=float) for c, w in weights.items())
    cdf_cut = 1.0 / (1.0 + np.exp(logit))  # F(-logit) for the standard logistic
    u = np.where(
        y == 1,
        rng.uniform(cdf_cut, 1.0),
        rng.uniform(0.0, cdf_cut),
    )
    u = np.clip(u, 1e-12, 1 - 1e-12)
    eps_y = np.log(u / (1.0 - u))
    eps_s = df["S"].to_numpy(dtype=float) - alpha * (2 * y - 1)

    # 2) Action
    cf = df.copy()
    for col, value in interventions.items():
        cf[col] = np.broadcast_to(np.asarray(value, dtype=float), len(cf))

    # 3) Prediction
    logit_cf = sum(w * cf[c].to_numpy(dtype=float) for c, w in weights.items())
    y_cf = (logit_cf + eps_y > 0).astype(int)
    cf["Y"] = y_cf
    cf["S"] = alpha * (2 * y_cf - 1) + eps_s
    return cf
