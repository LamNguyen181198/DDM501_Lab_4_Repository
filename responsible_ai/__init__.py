"""
Responsible AI Module
=====================
Provides SHAP-based model explainability and basic fairness utilities
for the Wine Quality classification pipeline.
"""

from .explainability import explain_prediction
from .fairness import check_feature_bias

__all__ = ["explain_prediction", "check_feature_bias"]
