"""
preprocessing.py – Bước 2: Tiền xử lý dữ liệu
Tích hợp từ Lab 2: ML Pipeline & Experiment Tracking
"""
import os
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler


def preprocess(X_train: np.ndarray, X_test: np.ndarray, feature_names: list):
    """
    Chuẩn hóa features bằng StandardScaler.
    Fit trên train, transform cả train và test.
    Returns: X_train_scaled, X_test_scaled, scaler, preprocessing_report
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Thống kê kiểm tra dữ liệu
    report = {
        "n_features": X_train.shape[1],
        "feature_names": feature_names,
        "train_mean": X_train_scaled.mean(axis=0).tolist(),
        "train_std": X_train_scaled.std(axis=0).tolist(),
        "missing_train": int(np.isnan(X_train_scaled).sum()),
        "missing_test": int(np.isnan(X_test_scaled).sum()),
    }

    print(f"[preprocessing] Scaled {report['n_features']} features")
    print(f"[preprocessing] Missing values - train: {report['missing_train']}, "
          f"test: {report['missing_test']}")

    return X_train_scaled, X_test_scaled, scaler, report


def save_scaler(scaler: StandardScaler, output_dir: str = "/tmp/wine_pipeline"):
    """Lưu scaler để dùng lại khi inference."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "scaler.pkl")
    with open(path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"[preprocessing] Scaler saved to {path}")
    return path


def load_scaler(data_dir: str = "/tmp/wine_pipeline") -> StandardScaler:
    """Load scaler từ file pickle."""
    path = os.path.join(data_dir, "scaler.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)
