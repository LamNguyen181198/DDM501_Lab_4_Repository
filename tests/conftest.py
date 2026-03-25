"""
conftest.py – Shared pytest fixtures
Tích hợp từ Lab 3: Testing & CI/CD
"""
import os
import sys
import pytest
import numpy as np
from unittest.mock import MagicMock

# Thêm project root vào path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="session")
def fraud_data():
    """Generate credit card fraud dataset một lần cho toàn bộ test session."""
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    X, y = make_classification(
        n_samples=5000, n_features=13, n_informative=8,
        n_redundant=2, n_classes=2, weights=[0.8, 0.2],
        random_state=42, flip_y=0,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    feature_names = [
        "amount", "time_of_day", "day_of_week", "merchant_risk_score",
        "distance_from_home_km", "distance_from_last_txn_km",
        "ratio_to_median_amount", "repeat_merchant", "used_chip",
        "used_pin", "online_order", "foreign_transaction", "txn_velocity_1h",
    ]
    return {
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "feature_names": feature_names,
        "target_names": ["legitimate", "fraud"],
    }


@pytest.fixture(scope="session")
def trained_model(fraud_data):
    """Trained RandomForest model – dùng lại giữa các tests."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(fraud_data["X_train"])
    model = RandomForestClassifier(n_estimators=50, class_weight="balanced", random_state=42)
    model.fit(X_tr, fraud_data["y_train"])
    return model, scaler


@pytest.fixture
def sample_features():
    """Sample credit card transaction features để test prediction (13 features)."""
    return [0.52, -0.31, 1.14, 0.78, -0.45, 0.93, -1.20, 0.38, 0.61, -0.77, 0.14, 0.89, -0.23]


@pytest.fixture
def api_client():
    """FastAPI test client (không cần MLflow thật)."""
    try:
        from fastapi.testclient import TestClient
        from unittest.mock import patch, MagicMock

        # Patch model loading để tránh cần MLflow thật
        with patch("mlflow.pyfunc.load_model"), \
             patch("mlflow.set_tracking_uri"), \
             patch("mlflow.tracking.MlflowClient"):
            import importlib
            import api.main as main_module
            importlib.reload(main_module)
            # Inject a mock model
            mock_model = MagicMock()
            mock_model.predict.return_value = np.array([1])
            main_module.model_manager.model = mock_model
            main_module.model_manager.model_version = "test"
            client = TestClient(main_module.app)
            return client
    except Exception:
        return None
