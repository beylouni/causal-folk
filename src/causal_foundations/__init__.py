"""CausalFoundations: Investigating Causal Alignment in Tabular Foundation Models."""

from causal_foundations.data import load_acs_data
from causal_foundations.models import get_baseline_classifiers, get_tabpfn_classifier
from causal_foundations.scm import generate_synthetic_target, inject_spurious_shortcut

__version__ = "0.1.0"

__all__ = [
    "generate_synthetic_target",
    "get_baseline_classifiers",
    "get_tabpfn_classifier",
    "inject_spurious_shortcut",
    "load_acs_data",
]
