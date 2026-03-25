"""
============================================
TRAIN & REGISTER MODEL TO MLFLOW
Credit Card Fraud Detection
============================================

This script:
1. Generates synthetic credit card fraud dataset
2. Trains RandomForest classifier
3. Logs to MLFlow
4. Registers to MLFlow Model Registry
5. Promotes to Production stage
"""

import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import os

# ============================================
# CONFIGURATION
# ============================================

MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5001')
MODEL_NAME = os.getenv('MODEL_NAME', "credit_fraud_model")
EXPERIMENT_NAME = os.getenv('EXPERIMENT_NAME', "credit_fraud_experiment")

FEATURE_NAMES = [
    "amount", "time_of_day", "day_of_week", "merchant_risk_score",
    "distance_from_home_km", "distance_from_last_txn_km",
    "ratio_to_median_amount", "repeat_merchant", "used_chip",
    "used_pin", "online_order", "foreign_transaction", "txn_velocity_1h",
]


# Configure MinIO S3 for artifacts
os.environ['MLFLOW_S3_ENDPOINT_URL'] = os.getenv('MLFLOW_S3_ENDPOINT_URL', 'http://localhost:9000')
os.environ['AWS_ACCESS_KEY_ID'] = os.getenv('AWS_ACCESS_KEY_ID', 'minio')
os.environ['AWS_SECRET_ACCESS_KEY'] = os.getenv('AWS_SECRET_ACCESS_KEY', 'minio123')
os.environ['MLFLOW_S3_IGNORE_TLS'] = os.getenv('MLFLOW_S3_IGNORE_TLS', 'true')

print(f"🔧 MLFlow Tracking URI: {MLFLOW_TRACKING_URI}")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# ============================================
# GENERATE DATA
# ============================================

print("📊 Generating credit card fraud dataset...")
X, y = make_classification(
    n_samples=5000,
    n_features=13,
    n_informative=8,
    n_redundant=2,
    n_repeated=0,
    n_classes=2,
    weights=[0.8, 0.2],
    random_state=42,
    flip_y=0,
)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"   Total samples: {len(X)}")
print(f"   Training samples: {len(X_train)}")
print(f"   Test samples: {len(X_test)}")
print(f"   Fraud rate: {y.mean():.1%}")

# ============================================
# TRAIN MODEL
# ============================================

print("\n🎯 Training model...")

mlflow.set_experiment(EXPERIMENT_NAME)

print(f"🎯 Experiment: {EXPERIMENT_NAME}")

with mlflow.start_run(run_name="RandomForest_FraudDetection_v1") as run:

    params = {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 2,
        'class_weight': 'balanced',
        'random_state': 42
    }

    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    accuracy  = accuracy_score(y_test, y_pred)
    f1        = f1_score(y_test, y_pred, average='binary')
    precision = precision_score(y_test, y_pred, average='binary')
    recall    = recall_score(y_test, y_pred, average='binary')
    auc_roc   = roc_auc_score(y_test, y_pred_proba)

    print("\n📈 Model Performance:")
    print(f"   Accuracy:  {accuracy:.4f}")
    print(f"   F1 Score:  {f1:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   AUC-ROC:   {auc_roc:.4f}")

    mlflow.log_params(params)
    mlflow.log_metrics({
        'accuracy':  accuracy,
        'f1_score':  f1,
        'precision': precision,
        'recall':    recall,
        'auc_roc':   auc_roc,
    })

    mlflow.sklearn.log_model(
        model,
        artifact_path="model",
        registered_model_name=MODEL_NAME,
        signature=mlflow.models.infer_signature(X_train, y_pred),
        input_example=X_train[:5]
    )

    run_id = run.info.run_id
    print("\n✅ Model logged to MLFlow!")
    print(f"   Run ID: {run_id}")

# ============================================
# PROMOTE TO PRODUCTION
# ============================================

print("\n🚀 Promoting model to Production stage...")

client = mlflow.tracking.MlflowClient()
model_versions = client.get_latest_versions(MODEL_NAME, stages=["None"])

if model_versions:
    latest_version = model_versions[0].version

    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=latest_version,
        stage="Production",
        archive_existing_versions=True
    )

    print("✅ Model promoted to Production!")
    print(f"   Model: {MODEL_NAME}")
    print(f"   Version: {latest_version}")
    print("   Stage: Production")
else:
    print("❌ No model versions found")

# ============================================
# SUMMARY
# ============================================

print("\n" + "="*50)
print("✅ SETUP COMPLETE!")
print("="*50)
print("\n Next steps:")
print("   1. Start API: docker compose up -d api")
print("   2. Health check: curl http://localhost:8000/health")
print("   3. Make prediction: curl -X POST http://localhost:8000/predict \\")
print("      -H 'Content-Type: application/json' \\")
print(f"      -d '{{\"features\": {X_test[0].tolist()}}}'")
print("   4. Explain prediction: curl -X POST http://localhost:8000/explain \\")
print("      -H 'Content-Type: application/json' \\")
print(f"      -d '{{\"features\": {X_test[0].tolist()}}}'")
print("   5. View Grafana:  http://localhost:3000")
print("   6. View MLFlow:   http://localhost:5001")