"""
training.py – Bước 3: Huấn luyện model với MLflow tracking
Tích hợp từ Lab 2: ML Pipeline & Experiment Tracking
"""
import os
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
EXPERIMENT_NAME = os.getenv("EXPERIMENT_NAME", "credit_fraud_experiment")


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list,
    n_estimators: int = 100,
    max_depth: int = None,
    min_samples_split: int = 2,
    random_state: int = 42,
) -> tuple:
    """
    Huấn luyện RandomForest và log toàn bộ vào MLflow.
    Returns: model, run_id
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=f"RF_n{n_estimators}_d{max_depth}") as run:
        # Log parameters
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("min_samples_split", min_samples_split)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("n_features", len(feature_names))
        mlflow.log_param("n_train_samples", len(X_train))

        # Train
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        # Train accuracy (quick sanity check)
        train_pred = model.predict(X_train)
        train_acc = accuracy_score(y_train, train_pred)
        mlflow.log_metric("train_accuracy", train_acc)

        # Log feature importances
        importances = model.feature_importances_
        for fname, imp in zip(feature_names, importances):
            mlflow.log_metric(f"importance_{fname[:20]}", float(imp))

        # Log model
        mlflow.sklearn.log_model(
            model,
            "model",
            registered_model_name=None,  # Registration done in registry.py
        )

        run_id = run.info.run_id
        print(f"[training] Run ID: {run_id}")
        print(f"[training] Train accuracy: {train_acc:.4f}")

    return model, run_id
