"""
evaluation.py – Bước 4: Đánh giá model và log metrics vào MLflow
Tích hợp từ Lab 2: ML Pipeline & Experiment Tracking
"""
import os
import mlflow
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")


def evaluate_model(
    model: RandomForestClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    run_id: str,
) -> dict:
    """
    Đánh giá model trên test set và log metrics vào MLflow run đã tạo.
    Returns: metrics dict
    """
    y_pred = model.predict(X_test)

    metrics = {
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "test_f1_macro": float(f1_score(y_test, y_pred, average="macro")),
        "test_precision_macro": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
        "test_recall_macro": float(recall_score(y_test, y_pred, average="macro")),
    }

    print("[evaluation] Test Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    # Log vào MLflow run đang active (nếu có) hoặc dùng run_id
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    with mlflow.start_run(run_id=run_id):
        for k, v in metrics.items():
            mlflow.log_metric(k, v)

    # In classification report để debug
    print("\n[evaluation] Classification Report:")
    print(classification_report(y_test, y_pred))

    return metrics
