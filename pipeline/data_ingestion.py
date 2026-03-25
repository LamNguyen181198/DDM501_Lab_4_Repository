"""
data_ingestion.py – Bước 1: Tải và chia dữ liệu Credit Card Fraud Detection
Tích hợp từ Lab 2: ML Pipeline & Experiment Tracking
"""
import os
import pickle
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

FEATURE_NAMES = [
    "amount",
    "time_of_day",
    "day_of_week",
    "merchant_risk_score",
    "distance_from_home_km",
    "distance_from_last_txn_km",
    "ratio_to_median_amount",
    "repeat_merchant",
    "used_chip",
    "used_pin",
    "online_order",
    "foreign_transaction",
    "txn_velocity_1h",
]
TARGET_NAMES = ["legitimate", "fraud"]


def load_and_split(test_size: float = 0.2, random_state: int = 42):
    """
    Generate synthetic credit card fraud dataset và chia train/test.
    Binary classification: 0 = legitimate, 1 = fraud
    Returns: X_train, X_test, y_train, y_test, feature_names, stats
    """
    X, y = make_classification(
        n_samples=5000,
        n_features=13,
        n_informative=8,
        n_redundant=2,
        n_repeated=0,
        n_classes=2,
        weights=[0.8, 0.2],
        random_state=random_state,
        flip_y=0,
    )
    feature_names = list(FEATURE_NAMES)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    fraud_rate = float(y.mean())
    stats = {
        "n_total": len(X),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_features": X.shape[1],
        "n_classes": 2,
        "feature_names": feature_names,
        "target_names": TARGET_NAMES,
        "fraud_rate": fraud_rate,
    }

    print(f"[data_ingestion] Generated fraud dataset: {stats['n_total']} samples, "
          f"{stats['n_features']} features, fraud_rate={fraud_rate:.1%}")
    print(f"[data_ingestion] Train: {stats['n_train']}, Test: {stats['n_test']}")

    return X_train, X_test, y_train, y_test, feature_names, stats


def save_data(X_train, X_test, y_train, y_test, feature_names, output_dir: str = "/tmp/fraud_pipeline"):
    """Lưu dữ liệu đã chia ra file pickle."""
    os.makedirs(output_dir, exist_ok=True)
    splits = {
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "feature_names": feature_names,
    }
    for name, obj in splits.items():
        path = os.path.join(output_dir, f"{name}.pkl")
        with open(path, "wb") as f:
            pickle.dump(obj, f)
    print(f"[data_ingestion] Data saved to {output_dir}")
    return output_dir


def load_data(data_dir: str = "/tmp/fraud_pipeline"):
    """Load dữ liệu từ file pickle."""
    splits = {}
    for name in ["X_train", "X_test", "y_train", "y_test", "feature_names"]:
        path = os.path.join(data_dir, f"{name}.pkl")
        with open(path, "rb") as f:
            splits[name] = pickle.load(f)
    return splits
