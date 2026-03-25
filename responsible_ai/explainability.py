"""
SHAP-based Model Explainability
================================
Uses TreeExplainer (zero-dependency on a running API) to compute
local feature-attribution values for any scikit-learn tree ensemble
(RandomForest, GradientBoosting, etc.).

Usage
-----
    from responsible_ai.explainability import explain_prediction

    result = explain_prediction(model, feature_names, input_data)
"""
from __future__ import annotations

import numpy as np

try:
    import shap
    _SHAP_AVAILABLE = True
except ImportError:
    _SHAP_AVAILABLE = False


def explain_prediction(
    model,
    feature_names: list[str],
    input_array: np.ndarray,
    *,
    use_shap: bool = True,
) -> dict:
    """
    Return a feature-attribution dictionary for a single prediction.

    Parameters
    ----------
    model        : fitted scikit-learn estimator (tree-based preferred)
    feature_names: list of column names, same order as input_array
    input_array  : 2-D numpy array, shape (1, n_features)
    use_shap     : if True and shap is installed, use SHAP TreeExplainer;
                   otherwise fall back to model.feature_importances_

    Returns
    -------
    {
        "method": "shap" | "feature_importance",
        "class_names": [...],
        "predicted_class": 0 | 1 | 2,
        "attributions": {feature: value, ...}   # SHAP for predicted class
        "base_value": float                      # expected model output
    }
    """
    input_array = np.array(input_array, dtype=float)
    if input_array.ndim == 1:
        input_array = input_array.reshape(1, -1)

    pred_class = int(model.predict(input_array)[0])
    class_names = [str(c) for c in model.classes_] if hasattr(model, "classes_") else []

    if use_shap and _SHAP_AVAILABLE:
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(input_array)

            # shap_values shape: (n_classes, n_samples, n_features)  for multi-class RF
            # or (n_samples, n_features) for binary/single output
            if isinstance(shap_values, list):
                # multi-class: pick slice for predicted class
                sv = shap_values[pred_class][0]
                base_val = float(explainer.expected_value[pred_class])
            else:
                sv = shap_values[0]
                base_val = float(explainer.expected_value)

            attributions = {
                name: round(float(val), 6)
                for name, val in zip(feature_names, sv)
            }
            return {
                "method": "shap",
                "class_names": class_names,
                "predicted_class": pred_class,
                "base_value": base_val,
                "attributions": attributions,
            }
        except Exception as exc:  # noqa: BLE001 – graceful fallback
            pass  # fall through to feature_importances_

    # ── Fallback: global feature importances (model-level, not local) ──
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        importances = np.ones(len(feature_names)) / len(feature_names)

    attributions = {
        name: round(float(val), 6)
        for name, val in zip(feature_names, importances)
    }
    return {
        "method": "feature_importance",
        "class_names": class_names,
        "predicted_class": pred_class,
        "base_value": None,
        "attributions": attributions,
    }
