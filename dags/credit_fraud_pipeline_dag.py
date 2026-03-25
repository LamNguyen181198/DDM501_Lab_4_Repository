"""
credit_fraud_pipeline_dag.py – Airflow DAG cho Credit Card Fraud Detection Training Pipeline
Tích hợp từ Lab 2: ML Pipeline & Experiment Tracking

DAG orchestrates:
  1. Ingest data (generate synthetic fraud dataset)
  2. Preprocess (StandardScaler)
  3. Train model (RandomForest + MLflow)
  4. Evaluate model (accuracy, F1, AUC-ROC)
  5. Register best model (promote to Production if accuracy >= 0.85)

Schedule: weekly (@weekly) – retraining mỗi tuần
"""

import os
import sys
import pickle
from datetime import datetime, timedelta

# Thêm project root vào PYTHONPATH để import pipeline module
sys.path.insert(0, "/opt/airflow")

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator

# ============================================================
# Default arguments
# ============================================================
default_args = {
    "owner": "mlops-team",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

DATA_DIR = "/tmp/fraud_pipeline"

# ============================================================
# DAG Definition
# ============================================================
dag = DAG(
    dag_id="credit_fraud_training_pipeline",
    default_args=default_args,
    description="Weekly automated retraining pipeline for Credit Card Fraud Detection model",
    schedule_interval="@weekly",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["ml", "training", "fraud-detection", "mlflow"],
)


# ============================================================
# Task Functions
# ============================================================

def task_ingest_data(**context):
    """Task 1: Generate synthetic fraud dataset và chia train/test."""
    from pipeline.data_ingestion import load_and_split, save_data

    X_train, X_test, y_train, y_test, feature_names, stats = load_and_split()
    save_data(X_train, X_test, y_train, y_test, feature_names, DATA_DIR)

    context["ti"].xcom_push(key="data_stats", value=stats)
    context["ti"].xcom_push(key="data_dir", value=DATA_DIR)
    print(f"[DAG] Data ingested: {stats['n_total']} samples")
    return "Data ingested"


def task_preprocess(**context):
    """Task 2: StandardScaler fit + transform."""
    from pipeline.data_ingestion import load_data
    from pipeline.preprocessing import preprocess, save_scaler

    data_dir = context["ti"].xcom_pull(key="data_dir")
    splits = load_data(data_dir)

    X_train_s, X_test_s, scaler, report = preprocess(
        splits["X_train"], splits["X_test"], splits["feature_names"]
    )

    # Lưu lại data đã scale
    os.makedirs(data_dir, exist_ok=True)
    for name, arr in [("X_train_s", X_train_s), ("X_test_s", X_test_s)]:
        with open(f"{data_dir}/{name}.pkl", "wb") as f:
            pickle.dump(arr, f)

    save_scaler(scaler, data_dir)
    context["ti"].xcom_push(key="preprocessing_report", value=report)
    print(f"[DAG] Preprocessing done. Missing values: {report['missing_train']}")
    return "Preprocessing done"


def task_train_model(**context):
    """Task 3: Train RandomForest với MLflow tracking."""
    from pipeline.data_ingestion import load_data
    from pipeline.training import train_model
    import pickle

    data_dir = context["ti"].xcom_pull(key="data_dir")
    splits = load_data(data_dir)

    # Load scaled data
    with open(f"{data_dir}/X_train_s.pkl", "rb") as f:
        X_train_s = pickle.load(f)

    model, run_id = train_model(
        X_train=X_train_s,
        y_train=splits["y_train"],
        feature_names=splits["feature_names"],
        n_estimators=100,
        random_state=42,
    )

    # Lưu model tạm
    with open(f"{data_dir}/model.pkl", "wb") as f:
        pickle.dump(model, f)

    context["ti"].xcom_push(key="run_id", value=run_id)
    print(f"[DAG] Training complete. MLflow run_id: {run_id}")
    return run_id


def task_evaluate_model(**context):
    """Task 4: Đánh giá model trên test set."""
    from pipeline.data_ingestion import load_data
    from pipeline.evaluation import evaluate_model
    import pickle

    data_dir = context["ti"].xcom_pull(key="data_dir")
    run_id = context["ti"].xcom_pull(key="run_id")
    splits = load_data(data_dir)

    with open(f"{data_dir}/model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(f"{data_dir}/X_test_s.pkl", "rb") as f:
        X_test_s = pickle.load(f)

    metrics = evaluate_model(model, X_test_s, splits["y_test"], run_id)
    context["ti"].xcom_push(key="metrics", value=metrics)
    print(f"[DAG] Evaluation done. Accuracy: {metrics['test_accuracy']:.4f}")
    return metrics


def branch_register(**context):
    """Branch: register model only nếu accuracy đủ cao."""
    metrics = context["ti"].xcom_pull(key="metrics")
    threshold = float(os.getenv("ACCURACY_THRESHOLD", "0.90"))

    if metrics and metrics.get("test_accuracy", 0) >= threshold:
        return "register_model"
    return "skip_registration"


def task_register_model(**context):
    """Task 5a: Đăng ký model vào MLflow Registry → Production."""
    from pipeline.registry import register_if_best

    run_id = context["ti"].xcom_pull(key="run_id")
    metrics = context["ti"].xcom_pull(key="metrics")

    result = register_if_best(run_id, metrics)
    context["ti"].xcom_push(key="registration_result", value=result)
    print(f"[DAG] Registered: {result}")
    return result


# ============================================================
# Task Definitions
# ============================================================
with dag:
    ingest = PythonOperator(
        task_id="ingest_data",
        python_callable=task_ingest_data,
    )

    preprocess = PythonOperator(
        task_id="preprocess_data",
        python_callable=task_preprocess,
    )

    train = PythonOperator(
        task_id="train_model",
        python_callable=task_train_model,
    )

    evaluate = PythonOperator(
        task_id="evaluate_model",
        python_callable=task_evaluate_model,
    )

    branch = BranchPythonOperator(
        task_id="check_accuracy_threshold",
        python_callable=branch_register,
    )

    register = PythonOperator(
        task_id="register_model",
        python_callable=task_register_model,
    )

    skip = EmptyOperator(task_id="skip_registration")

    end = EmptyOperator(
        task_id="pipeline_complete",
        trigger_rule="none_failed_min_one_success",
    )

    # Pipeline flow
    ingest >> preprocess >> train >> evaluate >> branch
    branch >> register >> end
    branch >> skip >> end
