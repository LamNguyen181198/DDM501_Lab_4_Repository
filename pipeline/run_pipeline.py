"""
run_pipeline.py – Chạy toàn bộ ML pipeline từ đầu đến cuối
Có thể chạy trực tiếp: python -m pipeline.run_pipeline
Hoặc gọi từ Airflow DAG
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.data_ingestion import load_and_split, save_data
from pipeline.preprocessing import preprocess, save_scaler
from pipeline.training import train_model
from pipeline.evaluation import evaluate_model
from pipeline.registry import register_if_best

DATA_DIR = "/tmp/fraud_pipeline"


def run_full_pipeline(
    n_estimators: int = 100,
    max_depth=None,
    min_samples_split: int = 2,
):
    """Chạy toàn bộ pipeline: load → preprocess → train → evaluate → register."""
    print("=" * 60)
    print("🚀 CREDIT CARD FRAUD DETECTION ML PIPELINE")
    print("=" * 60)

    # Bước 1: Tải dữ liệu
    print("\n[Step 1] Data Ingestion")
    X_train, X_test, y_train, y_test, feature_names, stats = load_and_split()
    save_data(X_train, X_test, y_train, y_test, feature_names, DATA_DIR)

    # Bước 2: Tiền xử lý
    print("\n[Step 2] Preprocessing")
    X_train_s, X_test_s, scaler, report = preprocess(X_train, X_test, feature_names)
    save_scaler(scaler, DATA_DIR)

    # Bước 3: Huấn luyện
    print("\n[Step 3] Training")
    model, run_id = train_model(
        X_train_s, y_train, feature_names,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
    )

    # Bước 4: Đánh giá
    print("\n[Step 4] Evaluation")
    metrics = evaluate_model(model, X_test_s, y_test, run_id)

    # Bước 5: Đăng ký model
    print("\n[Step 5] Model Registry")
    reg_result = register_if_best(run_id, metrics)

    print("\n" + "=" * 60)
    print("✅ Pipeline hoàn thành!")
    print(f"   Test Accuracy: {metrics['test_accuracy']:.4f}")
    print(f"   Registered: {reg_result['registered']}")
    if reg_result.get("version"):
        print(f"   Model Version: {reg_result['version']} → Production")
    print("=" * 60)

    return {"metrics": metrics, "run_id": run_id, "registration": reg_result}


if __name__ == "__main__":
    run_full_pipeline()
