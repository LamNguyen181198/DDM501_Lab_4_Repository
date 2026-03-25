"""
registry.py – Bước 5: Đăng ký model tốt nhất vào MLflow Model Registry
Tích hợp từ Lab 2: ML Pipeline & Experiment Tracking
"""
import os
import mlflow
from mlflow.tracking import MlflowClient

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
MODEL_NAME = os.getenv("MODEL_NAME", "credit_fraud_model")
ACCURACY_THRESHOLD = float(os.getenv("ACCURACY_THRESHOLD", "0.85"))


def register_if_best(run_id: str, metrics: dict) -> dict:
    """
    Nếu model mới có accuracy >= ACCURACY_THRESHOLD thì đăng ký vào Registry
    và promote lên Production, archive version cũ.
    Returns: registration result dict
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    accuracy = metrics.get("test_accuracy", 0.0)
    result = {"registered": False, "version": None, "reason": ""}

    if accuracy < ACCURACY_THRESHOLD:
        result["reason"] = (
            f"Accuracy {accuracy:.4f} < threshold {ACCURACY_THRESHOLD}"
        )
        print(f"[registry] ❌ Model NOT registered: {result['reason']}")
        return result

    # Đăng ký model mới
    model_uri = f"runs:/{run_id}/model"
    mv = mlflow.register_model(model_uri, MODEL_NAME)
    version = mv.version

    # Transition các version cũ về Archived
    existing_versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])
    for old_v in existing_versions:
        if old_v.version != version:
            client.transition_model_version_stage(
                name=MODEL_NAME,
                version=old_v.version,
                stage="Archived",
            )
            print(f"[registry] Archived old version {old_v.version}")

    # Promote version mới lên Production
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=version,
        stage="Production",
    )

    result.update({
        "registered": True,
        "version": version,
        "reason": f"Accuracy {accuracy:.4f} >= threshold {ACCURACY_THRESHOLD}",
    })
    print(f"[registry] ✅ Registered {MODEL_NAME} v{version} → Production")
    return result
