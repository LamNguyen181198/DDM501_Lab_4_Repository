"""
Fairness & Bias Utilities
==========================
Light-weight checks to detect potential bias in feature usage.

These are intentionally simple statistical checks — for production use,
consider AIF360 or Fairlearn for demographic-parity / equalised-odds metrics.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def check_feature_bias(
    model,
    feature_names: list[str],
    *,
    concentration_threshold: float = 0.5,
) -> dict:
    """
    Flag features that dominate model decisions (concentration bias).

    A model that puts > `concentration_threshold` of total importance on
    a single feature may be vulnerable to:
    - spurious correlations
    - disparate impact on under-represented groups

    Parameters
    ----------
    model                  : fitted scikit-learn estimator with feature_importances_
    feature_names          : column names in training order
    concentration_threshold: importance fraction that triggers a warning

    Returns
    -------
    {
        "feature_importances": {feature: float, ...},
        "top_feature": str,
        "top_feature_importance": float,
        "warnings": [str, ...]
    }
    """
    if not hasattr(model, "feature_importances_"):
        return {
            "feature_importances": {},
            "top_feature": None,
            "top_feature_importance": None,
            "warnings": ["Model does not expose feature_importances_."],
        }

    importances = model.feature_importances_
    total = importances.sum()
    normalised = importances / total if total > 0 else importances

    importance_dict = {
        name: round(float(val), 6)
        for name, val in sorted(
            zip(feature_names, normalised), key=lambda x: x[1], reverse=True
        )
    }

    top_feature = max(importance_dict, key=importance_dict.get)  # type: ignore[arg-type]
    top_val = importance_dict[top_feature]

    warnings: list[str] = []
    if top_val > concentration_threshold:
        warnings.append(
            f"Feature '{top_feature}' accounts for {top_val:.1%} of model importance "
            f"(threshold: {concentration_threshold:.0%}). "
            "Review whether this feature is a proxy for a protected attribute."
        )

    # Bottom-n features with near-zero importance (potential noise)
    near_zero = [k for k, v in importance_dict.items() if v < 0.01]
    if near_zero:
        warnings.append(
            f"{len(near_zero)} feature(s) have <1% importance: {near_zero}. "
            "Consider removing them to reduce overfitting risk."
        )

    return {
        "feature_importances": importance_dict,
        "top_feature": top_feature,
        "top_feature_importance": top_val,
        "warnings": warnings,
    }


def data_privacy_report(df: pd.DataFrame, pii_columns: list[str] | None = None) -> dict:
    """
    Basic data-privacy scan.

    Checks that known PII columns are absent from the dataframe.
    Wine-quality dataset has no PII; this function is a template for
    use cases that do (e.g., credit-fraud).

    Returns
    -------
    {"pii_found": [...], "safe": bool}
    """
    pii_columns = pii_columns or [
        "name", "email", "phone", "ssn", "credit_card",
        "address", "dob", "date_of_birth",
    ]
    found = [col for col in df.columns if col.lower() in pii_columns]
    return {
        "pii_found": found,
        "safe": len(found) == 0,
    }
