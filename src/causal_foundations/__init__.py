"""CausalFoundations: Investigating Causal Alignment in Tabular Foundation Models."""

from causal_foundations.data import load_acs_data, load_prototype_environments
from causal_foundations.interpretability import (
    causal_precision_at_k,
    compute_shap_values,
    shap_importance,
)
from causal_foundations.models import get_baseline_classifiers, get_tabpfn_classifier
from causal_foundations.scm import (
    generate_synthetic_target,
    inject_spurious_shortcut,
    logistic_scm_probability,
    perturb_feature,
    prototype_counterfactual,
)

__version__ = "0.1.0"

__all__ = [
    "causal_precision_at_k",
    "compute_shap_values",
    "generate_synthetic_target",
    "get_baseline_classifiers",
    "get_tabpfn_classifier",
    "inject_spurious_shortcut",
    "load_acs_data",
    "load_prototype_environments",
    "logistic_scm_probability",
    "perturb_feature",
    "prototype_counterfactual",
    "shap_importance",
]
