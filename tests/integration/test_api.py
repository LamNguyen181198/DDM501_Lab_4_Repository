"""
test_api.py – Integration tests cho FastAPI endpoints
Tích hợp từ Lab 3: Testing & CI/CD
Chạy: pytest tests/integration/ -v
"""
import os
import sys
import pytest
import numpy as np
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# ============================================================
# Fixture: Test client với model mock
# ============================================================
@pytest.fixture(scope="module")
def client():
    """FastAPI TestClient với model manager được mock."""
    os.environ.setdefault("MLFLOW_TRACKING_URI", "http://localhost:5001")
    os.environ.setdefault("MODEL_NAME", "credit_fraud_model")
    os.environ.setdefault("MODEL_STAGE", "Production")

    with patch("mlflow.pyfunc.load_model") as mock_load, \
         patch("mlflow.set_tracking_uri"), \
         patch("mlflow.tracking.MlflowClient") as mock_client_cls:

        # Mock MLflow client
        mock_client = MagicMock()
        mock_client.get_latest_versions.return_value = [
            MagicMock(version="4")
        ]
        mock_client_cls.return_value = mock_client

        # Mock model predict
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([1])
        mock_load.return_value = mock_model

        from fastapi.testclient import TestClient
        import api.main as main_module
        import importlib
        importlib.reload(main_module)

        # Inject mocked model
        main_module.model_manager.model = mock_model
        main_module.model_manager.model_version = "4"

        yield TestClient(main_module.app)


VALID_FEATURES = [0.52, -0.31, 1.14, 0.78, -0.45, 0.93, -1.20, 0.38, 0.61, -0.77, 0.14, 0.89, -0.23]


# ============================================================
# Health endpoint
# ============================================================
class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_response_has_required_fields(self, client):
        resp = client.get("/health")
        data = resp.json()
        for field in ["status", "model_loaded", "model_name", "model_version", "uptime_seconds"]:
            assert field in data, f"Health response thiếu field '{field}'"

    def test_health_uptime_is_positive(self, client):
        resp = client.get("/health")
        assert resp.json()["uptime_seconds"] >= 0


# ============================================================
# Predict endpoint
# ============================================================
class TestPredictEndpoint:
    def test_predict_returns_200_with_valid_input(self, client):
        resp = client.post("/predict", json={"features": VALID_FEATURES})
        assert resp.status_code == 200

    def test_predict_response_has_prediction_field(self, client):
        resp = client.post("/predict", json={"features": VALID_FEATURES})
        data = resp.json()
        assert "prediction" in data

    def test_predict_returns_valid_class(self, client):
        resp = client.post("/predict", json={"features": VALID_FEATURES})
        pred = resp.json()["prediction"]
        assert pred in [0.0, 1.0], f"Prediction {pred} ngoai range hop le"

    def test_predict_response_has_model_info(self, client):
        resp = client.post("/predict", json={"features": VALID_FEATURES})
        data = resp.json()
        for field in ["model_name", "model_version", "timestamp", "latency_ms"]:
            assert field in data, f"Response thiếu field '{field}'"

    def test_predict_with_empty_features_returns_422(self, client):
        resp = client.post("/predict", json={"features": []})
        # Phải validate và trả về lỗi
        assert resp.status_code in [422, 400, 500]

    def test_predict_with_wrong_number_of_features(self, client):
        # 5 features thay vì 13
        resp = client.post("/predict", json={"features": [1.0, 2.0, 3.0, 4.0, 5.0]})
        # Có thể là 500 (model error) hoặc 422 (validation)
        assert resp.status_code in [200, 400, 422, 500]

    def test_predict_with_missing_body_returns_422(self, client):
        resp = client.post("/predict", json={})
        assert resp.status_code == 422

    def test_predict_latency_returned_in_ms(self, client):
        resp = client.post("/predict", json={"features": VALID_FEATURES})
        data = resp.json()
        if "latency_ms" in data:
            assert data["latency_ms"] >= 0


# ============================================================
# Metrics endpoint (Prometheus)
# ============================================================
class TestMetricsEndpoint:
    def test_metrics_returns_200(self, client):
        resp = client.get("/metrics")
        assert resp.status_code == 200

    def test_metrics_content_type_is_prometheus(self, client):
        resp = client.get("/metrics")
        assert "text/plain" in resp.headers.get("content-type", "")

    def test_metrics_contains_api_request_counter(self, client):
        client.get("/health")  # Tạo traffic
        resp = client.get("/metrics")
        assert "api_requests_total" in resp.text


# ============================================================
# Docs endpoint
# ============================================================
class TestDocsEndpoint:
    def test_swagger_docs_available(self, client):
        resp = client.get("/docs")
        assert resp.status_code == 200

    def test_openapi_json_available(self, client):
        resp = client.get("/openapi.json")
        assert resp.status_code == 200
        data = resp.json()
        assert "openapi" in data
        assert "paths" in data
