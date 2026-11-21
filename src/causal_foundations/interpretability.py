"""SHAP helpers for comparing model attributions against the ground-truth causal graph."""

from __future__ import annotations

from collections.abc import Sequence
from math import factorial
from typing import Any

import numpy as np
import pandas as pd
import shap

MAX_EXACT_FEATURES = 10


def compute_shap_values(
    model: Any,
    X_explain: pd.DataFrame,
    X_background: pd.DataFrame,
    n_background: int = 50,
    batch_size: int = 25,
    positive_class: int = 1,
    random_state: int = 42,
) -> shap.Explanation:
    """Exact interventional SHAP values for ``P(Y = positive_class | x)``.

    Gives the same result as ``shap.ExactExplainer`` with an ``Independent`` masker,
    but evaluates every coalition of every row in a few large ``predict_proba`` calls.
    This matters for TabPFN, where each call re-runs the transformer over the whole
    context; ``shap``'s explainers call the model once per explained row.

    The cost is ``len(X_explain) * 2**n_features * n_background`` predicted rows, so it
    is meant for small feature sets like the prototype (4 features -> 16 coalitions).

    Parameters
    ----------
    model : fitted classifier
        Anything with ``predict_proba`` (TabPFN, LightGBM, sklearn pipelines...).
    X_explain : pd.DataFrame
        Rows to explain.
    X_background : pd.DataFrame
        Reference distribution used to "remove" features (usually the training set).
    n_background : int
        Number of background rows sampled from ``X_background``.
    batch_size : int
        Number of explained rows evaluated per ``predict_proba`` call.
    positive_class : int
        Column of ``predict_proba`` to explain.
    random_state : int
        Seed for the background subsample.

    Returns
    -------
    shap.Explanation
        Works with ``shap.plots.beeswarm``, ``shap.plots.bar``, etc.
    """
    feature_names = list(X_explain.columns)
    n_features = len(feature_names)
    if n_features > MAX_EXACT_FEATURES:
        raise ValueError(
            f"Exact SHAP needs 2**n_features coalitions; got {n_features} features "
            f"(max {MAX_EXACT_FEATURES}). Use shap.PermutationExplainer instead."
        )

    background = X_background[feature_names]
    if len(background) > n_background:
        background = background.sample(n=n_background, random_state=random_state)
    bg = background.to_numpy(dtype=float)
    x = X_explain.to_numpy(dtype=float)

    # Coalition c contains feature j iff bit j of c is set, so masks[c] <-> code c.
    codes = np.arange(2**n_features)
    masks = ((codes[:, None] >> np.arange(n_features)) & 1).astype(bool)

    def predict(data: np.ndarray) -> np.ndarray:
        frame = pd.DataFrame(data, columns=feature_names)
        return np.asarray(model.predict_proba(frame))[:, positive_class]

    # v[n, c] = E_bg[ f(x_n on coalition c, background elsewhere) ]
    n_rows, n_coal, n_bg = len(x), len(masks), len(bg)
    v = np.empty((n_rows, n_coal))
    for start in range(0, n_rows, batch_size):
        xb = x[start : start + batch_size]
        mixed = np.where(masks[None, :, None, :], xb[:, None, None, :], bg[None, None, :, :])
        preds = predict(mixed.reshape(-1, n_features))
        v[start : start + len(xb)] = preds.reshape(len(xb), n_coal, n_bg).mean(axis=2)

    sizes = masks.sum(axis=1)
    weights = np.array(
        [
            factorial(s) * factorial(n_features - s - 1) / factorial(n_features)
            for s in range(n_features)
        ]
    )
    values = np.zeros((n_rows, n_features))
    for j in range(n_features):
        without_j = np.flatnonzero(~masks[:, j])
        values[:, j] = (
            (v[:, without_j | (1 << j)] - v[:, without_j]) * weights[sizes[without_j]]
        ).sum(axis=1)

    base_value = v[:, 0]  # empty coalition: mean prediction over the background
    return shap.Explanation(
        values=values,
        base_values=base_value,
        data=x,
        feature_names=feature_names,
    )


def shap_importance(explanation: shap.Explanation) -> pd.Series:
    """Mean absolute SHAP value per feature, sorted from most to least important."""
    importance = np.abs(np.asarray(explanation.values)).mean(axis=0)
    return pd.Series(importance, index=explanation.feature_names, name="mean_abs_shap").sort_values(
        ascending=False
    )


def causal_precision_at_k(
    importance: pd.Series,
    causal_parents: Sequence[str],
    k: int | None = None,
) -> float:
    """Fraction of the top-k most important features that are true causal parents.

    ``k`` defaults to the number of causal parents, so 1.0 means the model's top
    features are exactly ``Pa(Y)``.
    """
    k = len(causal_parents) if k is None else k
    top_k = importance.sort_values(ascending=False).index[:k]
    return float(np.mean([f in set(causal_parents) for f in top_k]))
