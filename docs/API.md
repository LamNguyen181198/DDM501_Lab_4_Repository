# ML Model Serving API — Reference Documentation

**Base URL:** `http://localhost:8000`  
**Interactive Swagger UI:** `http://localhost:8000/docs`  
**ReDoc:** `http://localhost:8000/redoc`  
**OpenAPI spec (YAML):** [`docs/openapi.yaml`](openapi.yaml) — import into Postman / Insomnia / Swagger Editor

---

## Overview

REST API that serves the **Credit Card Fraud Detection** classification model (RandomForest, binary: 0 = legitimate, 1 = fraud) registered in the MLflow model registry. All requests/responses are JSON.

| Service | URL |
|---------|-----|
| **API** (this) | http://localhost:8000 |
| **MLflow UI** | http://localhost:5001 |
| **Grafana** | http://localhost:3000 (admin/admin) |
| **Prometheus** | http://localhost:9090 |
| **Evidently** | http://localhost:8001 |
| **MinIO Console** | http://localhost:9001 (minio/minio123) |
| **Airflow** | http://localhost:8080 (airflow/airflow) |

---

## Endpoints

### `GET /`

Service discovery. Returns available endpoint paths.

**Response 200**
```json
{
  "message": "Credit Card Fraud Detection API",
  "version": "1.0.0",
  "endpoints": {
    "predict": "/predict",
    "explain": "/explain",
    "health": "/health",
    "metrics": "/metrics",
    "model_info": "/model/info"
  }
}
```

---

### `GET /health`

Liveness + readiness check. Use for container health probes.

**Response 200 — healthy**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_name": "credit_fraud_model",
  "model_version": "4",
  "uptime_seconds": 3612.5
}
```

**Response 200 — unhealthy** (model failed to load)
```json
{
  "status": "unhealthy",
  "model_loaded": false,
  "model_name": "credit_fraud_model",
  "model_version": "unknown",
  "uptime_seconds": 12.1
}
```

---

### `POST /predict`

Runs inference and returns whether the transaction is fraudulent.

#### Input features

The model expects **exactly 13 numeric features** in this order:

| # | Name | Description | Typical range |
|---|------|-------------|---------------|
| 0 | `amount` | Normalized transaction amount | -3 – 5 |
| 1 | `time_of_day` | Hour of day (standardized) | -2 – 2 |
| 2 | `day_of_week` | Day of week (standardized) | -2 – 2 |
| 3 | `merchant_risk_score` | Merchant fraud risk score | -2 – 4 |
| 4 | `distance_from_home_km` | Distance from cardholderʼs home (standardized) | -2 – 5 |
| 5 | `distance_from_last_txn_km` | Distance from last transaction (standardized) | -2 – 5 |
| 6 | `ratio_to_median_amount` | Ratio of txn to median spend (standardized) | -3 – 5 |
| 7 | `repeat_merchant` | Whether merchant was used before (standardized) | -2 – 2 |
| 8 | `used_chip` | Chip was used for transaction (standardized) | -2 – 2 |
| 9 | `used_pin` | PIN was entered (standardized) | -2 – 2 |
| 10 | `online_order` | Transaction is online (standardized) | -2 – 2 |
| 11 | `foreign_transaction` | Transaction in foreign country (standardized) | -2 – 2 |
| 12 | `txn_velocity_1h` | Number of transactions in last 1 hour (standardized) | -2 – 5 |

**Output classes:**  `0` = legitimate,  `1` = fraud

#### Request body

```json
{
  "features": [0.52, -0.31, 1.14, 0.78, -0.45, 0.93, -1.20, 0.38, 0.61, -0.77, 0.14, 0.89, -0.23],
  "feature_names": ["amount", "time_of_day", "day_of_week", "merchant_risk_score",
                    "distance_from_home_km", "distance_from_last_txn_km",
                    "ratio_to_median_amount", "repeat_merchant", "used_chip",
                    "used_pin", "online_order", "foreign_transaction", "txn_velocity_1h"]
}
```

> `feature_names` is optional. Omitting it is valid; the API uses default fraud feature names.

#### Response 200

```json
{
  "prediction": 1.0,
  "model_name": "credit_fraud_model",
  "model_version": "4",
  "timestamp": "2026-03-23T10:15:30.123456",
  "latency_ms": 3.21
}
```

#### cURL example

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": [0.52, -0.31, 1.14, 0.78, -0.45, 0.93, -1.20, 0.38, 0.61, -0.77, 0.14, 0.89, -0.23]
  }'
```

#### Error responses

| Code | Cause |
|------|-------|
| 400 | Non-numeric values or wrong feature count |
| 503 | Model not yet loaded (container starting up) |
| 500 | Unexpected server error |

---

### `POST /explain`

Returns **SHAP per-feature attribution values** for a single prediction.

Uses `shap.TreeExplainer` when available; falls back to global `feature_importances_` otherwise.

#### How to interpret results

- **Positive attribution** → feature pushes prediction *toward* fraud (class 1)
- **Negative attribution** → feature pushes prediction *away* from fraud (toward legitimate)
- `base_value` is the average model output (a.k.a. SHAP expected value)

#### Request body

Same schema as `/predict`.

```json
{
  "features": [0.52, -0.31, 1.14, 0.78, -0.45, 0.93, -1.20, 0.38, 0.61, -0.77, 0.14, 0.89, -0.23]
}
```

#### Response 200

```json
{
  "method": "shap",
  "predicted_class": 1,
  "base_value": -1.2034,
  "attributions": {
    "amount":                       0.312,
    "time_of_day":                 -0.041,
    "day_of_week":                  0.008,
    "merchant_risk_score":          0.265,
    "distance_from_home_km":        0.195,
    "distance_from_last_txn_km":    0.143,
    "ratio_to_median_amount":       0.389,
    "repeat_merchant":             -0.015,
    "used_chip":                   -0.009,
    "used_pin":                    -0.022,
    "online_order":                 0.118,
    "foreign_transaction":          0.072,
    "txn_velocity_1h":              0.041
  },
  "top_features": [
    { "feature": "ratio_to_median_amount", "attribution": 0.389 },
    { "feature": "amount",                 "attribution": 0.312 },
    { "feature": "merchant_risk_score",    "attribution": 0.265 }
  ],
  "model_name": "credit_fraud_model",
  "model_version": "4"
}
```

| `method` value | Meaning |
|----------------|---------|
| `shap` | Local SHAP values — sample-specific, recommended |
| `feature_importance` | Global RF importances — fallback when SHAP unavailable |

#### cURL example

```bash
curl -X POST http://localhost:8000/explain \
  -H "Content-Type: application/json" \
  -d '{
    "features": [0.52, -0.31, 1.14, 0.78, -0.45, 0.93, -1.20, 0.38, 0.61, -0.77, 0.14, 0.89, -0.23]
  }'
```

---

### `GET /model/info`

Returns metadata about the currently loaded model.

**Response 200**
```json
{
  "model_name": "credit_fraud_model",
  "model_version": "4",
  "model_uri": "models:/credit_fraud_model/Production",
  "load_time_seconds": 1.23,
  "tracking_uri": "http://mlflow:5000"
}
```

---

### `POST /model/reload`

Hot-reloads the Production model from MLflow without restarting the container.

**Response 200**
```json
{
  "status": "success",
  "message": "Model reloaded successfully",
  "model_version": "5"
}
```

---

### `GET /metrics`

Returns Prometheus metrics in standard text exposition format.
Scraped every **15 s** by Prometheus.

**Key metrics**

| Metric name | Type | Description |
|-------------|------|-------------|
| `api_requests_total` | Counter | Total HTTP requests — labels: `method`, `endpoint`, `status` |
| `api_request_latency_seconds` | Histogram | End-to-end request latency |
| `model_predictions_total` | Counter | Total predictions — labels: `model_name`, `model_version` |
| `model_prediction_latency_seconds` | Histogram | Inference-only latency |
| `model_prediction_value` | Histogram | Distribution of predicted class values (0/1) |
| `model_prediction_errors_total` | Counter | Errors — label: `error_type` |
| `model_version_info` | Gauge | Currently loaded model version number |
| `model_load_time_seconds` | Gauge | Seconds taken to load the model |
| `model_feature_value` | Histogram | Per-feature input distribution (used by Evidently for drift detection) |

**Response** — `Content-Type: text/plain; charset=utf-8`
```
# HELP api_requests_total Total API requests
# TYPE api_requests_total counter
api_requests_total{endpoint="/predict",method="POST",status="200"} 142.0
...
```

---

## Python client example

```python
import requests

BASE = "http://localhost:8000"

# Fraud transaction features (standardized)
features = [0.52, -0.31, 1.14, 0.78, -0.45, 0.93, -1.20, 0.38, 0.61, -0.77, 0.14, 0.89, -0.23]

# Predict
resp = requests.post(f"{BASE}/predict", json={"features": features})
print(resp.json())
# → {"prediction": 1.0, "model_version": "4", ...}  (1 = fraud)

# Explain
resp = requests.post(f"{BASE}/explain", json={"features": features})
data = resp.json()
print(f"Top feature: {data['top_features'][0]}")
# → {"feature": "ratio_to_median_amount", "attribution": 0.389}
```

---

## Postman / Insomnia import

Import [docs/openapi.yaml](openapi.yaml) directly into Postman or Insomnia:

- **Postman**: File → Import → select `docs/openapi.yaml`
- **Insomnia**: Application → Import → From File → select `docs/openapi.yaml`
- **Swagger Editor**: paste contents at https://editor.swagger.io

---

## Error schema

All errors follow the standard FastAPI error shape:

```json
{
  "detail": "Human-readable error message"
}
