# 🎯 Hệ thống Giám sát Mô hình ML

**Dự án này phục vụ mục đích học tập và demo.**
Một hệ thống giám sát machine learning cấp production sử dụng MLFlow, Prometheus, Grafana và Evidently để quan sát mô hình và theo dõi hiệu năng theo thời gian thực.

---

## 📋 Mục lục

- [Tổng quan](#-tổng-quan)
- [Kiến trúc hệ thống](#-kiến-trúc-hệ-thống)
- [Tính năng](#-tính-năng)
- [Yêu cầu cài đặt](#-yêu-cầu-cài-đặt)
- [Khởi động nhanh](#-khởi-động-nhanh)
- [Cấu hình](#-cấu-hình)
- [Huấn luyện mô hình](#-huấn-luyện-mô-hình)
- [Giám sát & Dashboards](#-giám-sát--dashboards)
- [Evidently & Phát hiện Drift](#-evidently--phát-hiện-drift)
- [Simulation & Load Testing](#-simulation--load-testing)
- [Tài liệu API](#-tài-liệu-api)
- [Airflow DAG Orchestration](#-airflow-dag-orchestration)
- [Xử lý sự cố](#-xử-lý-sự-cố)
- [Cấu trúc dự án](#-cấu-trúc-dự-án)


---

## 🎯 Tổng quan

Dự án này minh họa một MLOps pipeline hoàn chỉnh với giám sát thời gian thực cho mô hình phát hiện gian lận thẻ tín dụng trong môi trường production. Bao gồm:

- **Huấn luyện & Registry mô hình**: Train và quản lý phiên bản mô hình bằng MLFlow
- **Phục vụ mô hình (Model Serving)**: REST API dựa trên FastAPI để thực hiện dự đoán
- **Thu thập metrics**: Prometheus scrape metrics của model, API và drift
- **Trực quan hóa**: Grafana dashboards giám sát thời gian thực và phân tích drift
- **Giám sát Drift & Chất lượng dữ liệu**: Evidently service phát hiện data drift và sinh báo cáo HTML
- **Lưu trữ**: MinIO (tương thích S3) cho model artifacts
- **Cơ sở dữ liệu**: PostgreSQL làm backend store cho MLFlow

**Các use case:**
- Giám sát hiệu năng mô hình trong production
- Theo dõi latency và throughput của dự đoán
- Phát hiện data drift và sự suy giảm của mô hình
- So sánh các phiên bản mô hình và A/B testing
- Cảnh báo khi có bất thường hoặc vấn đề hiệu năng

---

## 🏗 Kiến trúc hệ thống

```
┌──────────────────────────────────────────────────────────────────────┐
│                          MONITORING STACK                           │
│                                                                      │
│  ┌──────────┐      ┌──────────┐      ┌──────────┐                  │
│  │ Grafana  │─────▶│Prometheus│◀─────│ FastAPI  │                  │
│  │  :3000   │      │  :9090   │      │  :8000   │                  │
│  └──────────┘      └──────────┘      └────┬─────┘                  │
│       ▲                     ▲             │                         │
│       │                     │             │                         │
│       │            ┌────────┴──────┐      │                         │
│       │            │  Evidently    │◀─────┘  (prediction capture)   │
│       │            │  :8001        │  exposes drift/data-quality    │
│       │            └──────────────-┘  metrics to Prometheus         │
│       │                                    │                        │
│       │            ┌──────────┐            │                        │
│       └───────────▶│  MLFlow  │◀───────────┘                        │
│                    │  :5000   │                                     │
│                    └────┬─────┘                                     │
│                         │                                           │
│              ┌──────────┴──────────┐                                │
│              │                     │                                │
│         ┌────▼─────┐         ┌────▼────┐                            │
│         │PostgreSQL│         │  MinIO  │                            │
│         │  :5432   │         │  :9000  │                            │
│         └──────────┘         └─────────┘                            │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### Các thành phần

| Service | Port | Mô tả |
|---------|------|-------|
| **MLFlow** | 5001 | Model registry và experiment tracking |
| **FastAPI** | 8000 | API phục vụ mô hình với Prometheus metrics |
| **Evidently** | 8001 | Giám sát data drift & chất lượng dữ liệu (Prometheus + báo cáo HTML) |
| **Prometheus** | 9090 | Thu thập và lưu trữ metrics |
| **Grafana** | 3000 | Trực quan hóa và dashboards (API + drift) |
| **MinIO** | 9000 | Object storage tương thích S3 cho model artifacts |
| **MinIO Console** | 9001 | Giao diện web của MinIO |
| **PostgreSQL** | 5432 | Cơ sở dữ liệu backend cho MLFlow |
| **Airflow** | 8080 | Điều phối pipeline theo DAG (admin/admin) |

---

## Tính năng

### Giám sát mô hình
- Theo dõi latency dự đoán thời gian thực (p50, p95, p99)
- Giám sát throughput và error rate
- Theo dõi và so sánh phiên bản mô hình
- Phân tích phân phối dự đoán
- Phát hiện feature drift (qua Prometheus + Evidently)

### Thu thập Metrics
- Metrics API (số lượng request, latency, status code)
- Metrics mô hình (số dự đoán, latency, lỗi)
- Phân phối giá trị feature để phát hiện drift
- Thời gian load mô hình và thông tin phiên bản
- Custom business metrics
- Metrics drift & chất lượng dữ liệu từ Evidently (drift score, số feature bị drift, missing values)

### Trực quan hóa
- Grafana dashboards cấu hình sẵn cho API và model metrics
- Dashboard Grafana riêng cho Evidently drift monitoring
- Cập nhật metrics theo thời gian thực
- Luật cảnh báo cho bất thường và drift
- Phân tích xu hướng lịch sử
- So sánh đa mô hình

### Sẵn sàng Production
- Điều phối bằng Docker Compose
- Health check cho tất cả services
- Khởi tạo MinIO bucket tự động
- Volumes dữ liệu bền vững
- Cấu hình dựa trên biến môi trường

---

## Yêu cầu cài đặt

Trước khi bắt đầu, đảm bảo đã cài đặt:

- **Docker** (≥ 20.10.0)
- **Docker Compose** (≥ 2.0.0)
- **Python** (≥ 3.10) — để train mô hình
---

## Quick Start

> Thực hiện **đúng thứ tự** các bước dưới đây. Toàn bộ stack gồm 11 containers.


---

### Bước 1 — Chuẩn bị môi trường

```bash
# Vào thư mục project
cd "/Users/dodoannang/Documents/Thạc sĩ MSE/MLOpsFinal/DDM501_Lab_4_Repository"

# Kiểm tra Docker đang chạy
docker info | head -5

# Kiểm tra Docker Compose
docker compose version   # hoặc: docker-compose version
```

---

### Bước 2 — Khởi động Infrastructure (PostgreSQL + MinIO + MLflow + Grafana + Prometheus)

```bash
# Bước 2a: Start core infrastructure
docker-compose up -d postgres minio minio-init mlflow prometheus grafana

# Bước 2b: Init Airflow DB (có thể chạy song song với 2a)
docker-compose up -d airflow-postgres airflow-init
```

Chờ khoảng **60–90 giây** rồi kiểm tra trạng thái tất cả services:

```bash
docker-compose ps
```

Xác nhận từng service đã healthy:

```bash
# MLflow (port 5001)
curl -s http://localhost:5001/health
# Expected: OK

# MinIO (port 9001)
curl -s http://localhost:9000/minio/health/live
# Expected: HTTP 200

# Prometheus
curl -s http://localhost:9090/-/healthy
# Expected: Prometheus Server is Healthy.

# Xem MinIO đã tạo bucket chưa
docker-compose logs --tail=20 minio-init
# Expected dòng: "Bucket created successfully" hoặc "already exists"

# Xem Airflow init xong chưa
docker-compose logs --tail=20 airflow-init
# Expected dòng: "Admin user admin created" rồi container exit(0)
```

> **Nếu MLflow không healthy**: `docker-compose restart mlflow` và chờ thêm 30s.

---

### Bước 3 — Train & Register Model vào MLflow

> ⚠️ **QUAN TRỌNG**: API container sẽ crash-loop nếu chưa có model trong Registry. **Bắt buộc** chạy bước này trước.

Bước này chạy **trên máy local** (ngoài Docker):

```bash
# Cài dependencies (chỉ cần 1 lần)
pip install -r scripts/requirements.txt

# Train model — script tự đọc MLFLOW_TRACKING_URI từ docker-compose .env
python scripts/training.py
```

**Output mong đợi:**

```
 MLFlow Tracking URI: http://localhost:5001
  Loading credit card fraud detection dataset...
  Training samples: 4000 | Test samples: 1000
  Training model...
 Model Performance:
   Accuracy:  0.8750
   F1 Score:  0.8420
   Precision: 0.8600
   Recall:    0.8300
   ROC AUC:   0.9210
 Model promoted to Production!
   Model: credit_fraud_model  |  Version: 1  |  Stage: Production
 SHAP explainer ready. /explain endpoint active.
```

Script tự động:
1. Tạo synthetic credit card fraud dataset (5 000 samples, 13 features)
2. Train `RandomForestClassifier` với `class_weight="balanced"`
3. Log metrics & artifacts vào experiment `credit_fraud_experiment`
4. Register `credit_fraud_model` và promote lên **Production**
5. Lưu artifact vào MinIO bucket `mlflow-artifacts`

Xác nhận model đã có trong Registry:

```bash
curl -s "http://localhost:5001/api/2.0/mlflow/registered-models/get?name=credit_fraud_model" \
  | python3 -m json.tool | grep '"name"'
# Expected: "name": "credit_fraud_model"
```

Xem trong UI: http://localhost:5001 → **Models** → `credit_fraud_model` → Stage: **Production**

---

### Bước 4 — Start API (Model Serving) + Evidently (Drift Detection)

```bash
docker-compose up -d api evidently
```

Chờ ~30 giây để API load model, rồi verify:

```bash
# ── Health check ──
curl http://localhost:8000/health
# Expected: {"status":"healthy","model_loaded":true,"model_name":"credit_fraud_model",...}

# ── Test predict (13 features) ──
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [0.52, -0.31, 1.14, 0.78, -0.45, 0.93, -1.20, 0.38, 0.61, -0.77, 0.14, 0.89, -0.23]}'
# Expected: {"prediction":0,"probability":0.12,"model_name":"credit_fraud_model",...}

# ── SHAP Explain ──
curl -X POST http://localhost:8000/explain \
  -H "Content-Type: application/json" \
  -d '{"features": [0.52, -0.31, 1.14, 0.78, -0.45, 0.93, -1.20, 0.38, 0.61, -0.77, 0.14, 0.89, -0.23]}'
# Expected: {"method":"shap","predicted_class":0,"top_features":[...]}

# ── Evidently health ──
curl http://localhost:8001/health
# Expected: {"status":"healthy",...}
```

Xem API docs đầy đủ (Swagger): http://localhost:8000/docs

> Nếu API báo "model not loaded": `docker-compose logs api` — nếu thấy lỗi registry, chạy lại Bước 3 rồi `docker-compose restart api`.

---

### Bước 5 — Kích hoạt Airflow (Pipeline Orchestration)

Airflow init đã chạy từ Bước 2. Giờ start webserver + scheduler:

```bash
docker-compose up -d airflow-webserver airflow-scheduler
```

Chờ ~60 giây để webserver sẵn sàng:

```bash
curl http://localhost:8080/health
# Expected: {"metadatabase":{"status":"healthy"},"scheduler":{"status":"healthy",...}}
```

**Kích hoạt DAG qua Airflow UI:**

1. Mở http://localhost:8080 → đăng nhập **admin / admin**
2. Tìm DAG **`credit_fraud_training_pipeline`**
3. Bật **toggle ON** (cột trái ngoài cùng)
4. Nhấn nút **▶ Trigger DAG** (cột Actions)
5. Nhấn tên DAG → tab **Graph** để theo dõi từng task

DAG chạy 5 tasks tuần tự:

```
ingest_data → preprocess_data → train_model → evaluate_model → register_model
```

> Airflow scheduler mất ~90s để scan và load DAG mới. Nếu DAG chưa hiện → `docker-compose restart airflow-scheduler` rồi chờ thêm 60s.

---

### Bước 6 — Chạy Simulation (Generate Traffic & Drift)

Simulation gửi predict requests tới API, thu thập dữ liệu phân phối và trigger Evidently phân tích drift. Kết quả sẽ hiện trên Grafana dashboards.

```bash
cd simulations
pip install -r requirements.txt
```

**Smoke test – xác nhận kết nối:**

```bash
chmod +x quick_test.sh
./quick_test.sh
# Gửi 20 requests, in prediction + probability của từng request
```

**Simulation chính – tạo traffic + drift signal:**

```bash
# Normal traffic (baseline)
python run_simulation.py -n 150 -s normal --analyze

# Drift nhẹ – moderate drift (xem drift score tăng trên Grafana)
python run_simulation.py -n 200 -s moderate_drift --analyze

# Drift nặng – severe drift (có thể kích hoạt Grafana alerts)
python run_simulation.py -n 300 -s severe_drift --analyze

# Burst traffic – test latency p95
python run_simulation.py -n 100 -p burst -s normal
```

**Chạy tất cả scenarios tự động (demo đầy đủ):**

```bash
python scenarios.py   # ~5–10 phút, chạy tất cả 6 scenarios
```

```bash
cd ..  # quay về root
```

Sau khi simulation xong, xem Evidently HTML report: http://localhost:8001/reports

---

### Bước 7 — Xem Dashboards & Kết quả

| Service | URL | Login | Nội dung chính |
|---------|-----|-------|---------------|
| **API Swagger** | http://localhost:8000/docs | — | Test `/predict`, `/explain`, `/health` trực tiếp |
| **MLflow** | http://localhost:5001 | — | Experiments, runs, model versions, artifacts |
| **Grafana** | http://localhost:3000 | admin / admin | ML Monitoring + Evidently Drift dashboards |
| **Prometheus** | http://localhost:9090 | — | Raw metrics, PromQL queries |
| **Airflow** | http://localhost:8080 | admin / admin | DAG runs, task logs, XCom |
| **Evidently Reports** | http://localhost:8001/reports | — | HTML drift report sau simulation |
| **Evidently API** | http://localhost:8001/docs | — | REST API cho drift analysis |
| **MinIO Console** | http://localhost:9001 | minio / minio123 | Model artifacts, bucket `mlflow-artifacts` |

**Trong Grafana (http://localhost:3000):**
- Dashboard **"ML Monitoring"** → request rate, latency p95/p99, prediction class distribution, error rate
- Dashboard **"Evidently – Data Drift Monitoring"** → drift score theo thời gian, số feature bị drift, missing values

**PromQL mẫu trong Prometheus (http://localhost:9090):**

```promql
# Request rate
rate(http_requests_total{job="credit-fraud-api"}[5m])

# Latency p95
histogram_quantile(0.95, rate(request_duration_seconds_bucket[5m]))

# Prediction distribution
sum by (prediction) (credit_fraud_prediction_total)
```

---

### Tắt toàn bộ stack

```bash
# Dừng tất cả containers, giữ data volumes
docker-compose down

# Dừng và XÓA toàn bộ data (reset hoàn toàn về zero)
docker-compose down -v
```

---

### Troubleshooting nhanh

| Triệu chứng | Lệnh kiểm tra | Cách fix |
|-------------|--------------|---------|
| API crash-loop ("model not found") | `docker-compose logs api` | Chạy `python scripts/training.py` rồi `docker-compose restart api` |
| MLflow không healthy | `docker-compose logs mlflow` | `docker-compose restart mlflow` |
| Airflow DAG không hiện | `docker-compose logs airflow-scheduler` | `docker-compose restart airflow-scheduler`, chờ 90s |
| MinIO bucket trống | `docker-compose logs minio-init` | `docker-compose up minio-init` |
| Grafana không có data | `curl http://localhost:9090/-/healthy` | `docker-compose restart prometheus grafana` |
| Port conflict | `lsof -i :8000` hoặc `-i :5001` | Dừng process đang dùng port, hoặc sửa `.env` |

---

## Cấu hình

### Biến môi trường

Toàn bộ cấu hình được quản lý qua file `.env`:

```bash
# Cấu hình người dùng
USER=dongnd                    # Username của bạn (dùng đặt tên container)

# Cấu hình MLFlow
MLFLOW_PORT=5000              # Port của MLFlow server
MODEL_NAME=credit_fraud_model # Tên mô hình trong registry
MODEL_STAGE=Production        # Stage của mô hình cần serve

# Cấu hình MinIO
MINIO_ROOT_USER=minio
MINIO_ROOT_PASSWORD=minio123 
MINIO_BUCKET_NAME=mlflow-artifacts

# Cấu hình Grafana
GRAFANA_ADMIN_USER=admin
GRAFANA_ADMIN_PASSWORD=admin

# Xem .env.example để biết thêm tùy chọn
```

### Grafana Datasource

Datasource Prometheus được provision tự động với:
- **UID**: `prometheus-datasource` (tham chiếu ổn định)
- **URL**: `http://prometheus:9090`
- **Query Timeout**: 60s
- **HTTP Method**: POST (truy vấn nhanh hơn)
- **Incremental Querying**: Bật

---

## Huấn luyện mô hình

### Sử dụng script training có sẵn

Script `scripts/training.py` train mô hình RandomForest trên tập dữ liệu phát hiện gian lận thẻ tín dụng:

```bash
# Sử dụng cơ bản
python scripts/training.py
```

### Tự viết script training

Để train mô hình của riêng bạn:

1. **Tạo script training** theo mẫu sau:

```python
import mlflow
import mlflow.sklearn
import os

# Configure MLFlow
mlflow.set_tracking_uri('http://localhost:5000')
os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://localhost:9000'
os.environ['AWS_ACCESS_KEY_ID'] = 'minio'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'minio123'

# Train your model
with mlflow.start_run(run_name="my_model_v1"):
    # ... train model ...
    
    # Log model
    mlflow.sklearn.log_model(
        model,
        artifact_path="model",
        registered_model_name="my_model",
        signature=mlflow.models.infer_signature(X_train, predictions)
    )
    
    run_id = mlflow.active_run().info.run_id

# Promote to Production
client = mlflow.tracking.MlflowClient()
model_versions = client.get_latest_versions("my_model", stages=["None"])
if model_versions:
    client.transition_model_version_stage(
        name="my_model",
        version=model_versions[0].version,
        stage="Production"
    )
```

2. **Cập nhật biến môi trường** trong `docker-compose.yml`:

```yaml
api:
  environment:
    - MODEL_NAME=my_model 
    - MODEL_STAGE=Production
```

3. **Khởi động lại API**:

```bash
docker-compose restart api
```

### Yêu cầu đối với mô hình

Mô hình của bạn phải:
- Được đăng ký trong MLFlow Model Registry
- Có phiên bản được promote lên stage chỉ định (mặc định: Production)
- Tương thích với `mlflow.pyfunc.load_model()`
- Chấp nhận đầu vào dạng mảng số để thực hiện dự đoán

---

## Giám sát & Dashboards

### Truy cập Grafana

1. Mở http://localhost:3000
2. Đăng nhập: `admin` / `admin`
3. Vào **Dashboards** → **ML Monitoring** (metrics API / model)
4. Vào **Dashboards** → **Evidently - Data Drift Monitoring** (drift metrics)

### Danh sách metrics có sẵn

API expose Prometheus metrics tại `http://localhost:8000/metrics`:

#### Metrics API
```promql
# Số lượng request theo method, endpoint và status
api_requests_total{method="POST",endpoint="/predict",status="200"}

# Histogram latency request
api_request_latency_seconds{method="POST",endpoint="/predict"}
```

#### Metrics mô hình
```promql
# Số dự đoán theo tên và phiên bản mô hình
model_predictions_total{model_name="credit_fraud_model",model_version="1"}

# Histogram latency dự đoán
model_prediction_latency_seconds{model_name="credit_fraud_model"}

# Phân phối giá trị dự đoán (để phát hiện drift)
model_prediction_value{model_name="credit_fraud_model"}

# Lỗi dự đoán theo loại
model_prediction_errors_total{model_name="credit_fraud_model",error_type="..."}
```

#### Metrics hệ thống
```promql
# Phiên bản mô hình hiện tại
model_version_info{model_name="credit_fraud_model",version="1"}

# Thời gian load mô hình
model_load_time_seconds{model_name="credit_fraud_model"}
```

### Ví dụ câu truy vấn PromQL

```promql
# Request per second
rate(api_requests_total[5m])

# Latency ở percentile 95
histogram_quantile(0.95, rate(model_prediction_latency_seconds_bucket[5m]))

# Tỷ lệ lỗi
rate(model_prediction_errors_total[5m]) / rate(model_predictions_total[5m])

# Giá trị dự đoán trung bình (theo dõi drift)
rate(model_prediction_value_sum[5m]) / rate(model_prediction_value_count[5m])
```

---

## Evidently & Phát hiện Drift

Evidently chạy như một service riêng biệt (port **8001**) và cung cấp:

- **REST API** để capture dữ liệu dự đoán và trigger phân tích
- **Báo cáo HTML** chi tiết về drift và chất lượng dữ liệu
- **Prometheus metrics** hiển thị trên Grafana drift dashboard

### Các endpoint chính

- `GET /health` – kiểm tra sức khỏe service và tóm tắt trạng thái
- `POST /capture` – capture một dự đoán (features + prediction)
- `POST /capture/batch` – capture một batch dự đoán
- `POST /analyze` – chạy phân tích drift trên dữ liệu production gần nhất
- `GET /reports` – liệt kê các báo cáo HTML drift được lưu
- `GET /reports/{name}` – xem báo cáo cụ thể
- `GET /metrics` – Prometheus metrics (ví dụ: `evidently_data_drift_detected`, `evidently_drift_score`)

### Quy trình giám sát drift điển hình

1. **API** phục vụ dự đoán và expose metrics tại `/metrics`.
2. **Script simulation** (hoặc ứng dụng thực) gửi traffic tới `/predict`.
3. Simulator tuy ỳ chọn **capture** từng request/response sang Evidently (`/capture`).
4. Theo lịch định kỳ (hoặc thủ công), Evidently `/analyze` so sánh dữ liệu production gần nhất với dữ liệu tham chiếu và:
   - Cập nhật Prometheus metrics (trạng thái drift, score, số feature bị drift, missing values...).
   - Tạo báo cáo HTML drift.
5. **Prometheus** scrape Evidently, **Grafana** hiển thị dashboard drift và chất lượng dữ liệu.

### Drift Dashboard trong Grafana

- Dashboard UID: `evidently-drift`
- File: `config/grafana/dashboards/evidently-drift-monitoring.json`
- Hiển thị:
  - Trạng thái drift hiện tại và số feature bị drift
  - Drift score theo thời gian
  - Ma trận drift theo từng feature
  - Latency và tần suất phân tích
  - Tỷ lệ missing values theo từng feature

### Cảnh báo

Luật cảnh báo Prometheus cho Evidently được định nghĩa trong:

- `config/prometheus/evidently_alerts.yml`

Ví dụ:

- `DataDriftDetected` – drift bị phát hiện ≥ 5 phút
- `MultipleDriftedFeatures` – từ 3 feature trở lên bị drift
- `HighDriftScore` – drift score > 0.5
- `HighMissingValues` – tỷ lệ missing values > 20%
- `SlowDriftAnalysis` – phân tích mất quá nhiều thời gian

Các cảnh báo này có thể kết nối với Alertmanager / Slack / email theo nhu cầu.

---

## Simulation & Load Testing

Để tạo traffic thực tế và các kịch bản drift, sử dụng các công cụ trong thư mục `simulations/`.

### Cài đặt

```bash
cd simulations
pip install -r requirements.txt
```

### Smoke test nhanh

```bash
cd simulations
./quick_test.sh       # Gửi 20 requests để xác nhận mọi thứ hoạt động
cd ..
```

### Sử dụng CLI

```bash
cd simulations

# 100 requests bình thường với tốc độ 2 req/s
python run_simulation.py -n 100 -s normal

# 200 requests với moderate drift + chạy phân tích Evidently
python run_simulation.py -n 200 -s moderate_drift --analyze

# Burst traffic
python run_simulation.py -p burst -s normal

# Tăng traffic dần dần
python run_simulation.py -p gradual -s normal
```

### Các kịch bản có sẵn

```bash
cd simulations

# Chạy một kịch bản cụ thể
python scenarios.py 1  # Traffic ngày bình thường
python scenarios.py 2  # Drift tăng dần
python scenarios.py 3  # Thay đổi phân phối đột ngột
python scenarios.py 6  # Stress test

# Chạy tất cả kịch bản tuần tự (để demo)
python scenarios.py
```

Các simulation này:

- Gửi request tới endpoint `/predict` của API.
- Tuy chọn **capture** từng dự đoán sang Evidently.
- Tự động **đưa metrics và drift signal** vào Prometheus & Grafana.

---

## Tài liệu API

### Các Endpoint

#### `GET /`
Endpoint gốc trả về thông tin API.

#### `GET /health`
Kiểm tra sức khỏe của API.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_name": "credit_fraud_model",
  "model_version": "1",
  "uptime_seconds": 123.45
}
```

#### `POST /predict`
Thực hiện dự đoán với mô hình đang chạy.

**Request:**
```json
{
  "features": [0.52, -0.31, 1.14, 0.78, -0.45, 0.93, -1.20, 0.38, 0.61, -0.77, 0.14, 0.89, -0.23],
  "feature_names": ["amount", "time_of_day", ...] // tùy chọn
}
```

**Response:**
```json
{
  "prediction": 1.0,
  "model_name": "credit_fraud_model",
  "model_version": "1",
  "timestamp": "2025-11-19T12:00:00",
  "latency_ms": 15.23
}
```

#### `GET /model/info`
Lấy thông tin về mô hình đang được nạp.

#### `POST /model/reload`
Tải lại mô hình từ Model Registry.

#### `GET /metrics`
Endpoint Prometheus metrics.

### Tài liệu API tương tác

Truy cập Swagger UI tại: http://localhost:8000/docs

---

## Xử lý sự cố

### Các lỗi thường gặp

#### 1. API khởng khởi động được: "Model not loaded"

**Nguyên nhân**: Chưa có mô hình nào được đăng ký trong MLFlow hoặc sai tên / stage mô hình.

**Cách sử a:**
```bash
# Kiểm tra model tồn tại chưa
curl http://localhost:5000/api/2.0/mlflow/registered-models/get?name=credit_fraud_model

# Train và đăng ký model
python scripts/training.py

# Restart API
docker-compose restart api
```

#### 2. MLFlow healthcheck thất bại

**Nguyên nhân**: Mài mông port hoặc MLFlow chưa khởi động xong.

**Cách sử a:**
```bash
# Xem log MLFlow
docker-compose logs mlflow

# Kiểm tra port (đúng là 5001)
docker-compose ps mlflow

# Restart nếu cần
docker-compose restart mlflow
```

#### 3. MinIO bucket chưa được tạo

**Nguyên nhân**: Service `minio-init` bị lỗi khi khởi động.

**Cách sử a:**
```bash
# Xem log init
docker-compose logs minio-init

# Tạo bucket thủ công
docker-compose exec minio mc alias set myminio http://localhost:9000 minio minio123
docker-compose exec minio mc mb myminio/mlflow-artifacts --ignore-existing
```

#### 4. Grafana datasource not working

**Problem**: Prometheus not accessible or wrong configuration.

**Solution:**
```bash
# Test Prometheus
curl http://localhost:9090/-/healthy

# Check Grafana datasource
# Go to: http://localhost:3000/datasources
# Click Prometheus → Test

# Verify configuration in:
cat config/grafana/provisioning/datasources/prometheus.yml
```

### Service Health Checks

```bash
# Check all services
docker-compose ps

# Check specific service logs
docker-compose logs -f mlflow
docker-compose logs -f api
docker-compose logs -f prometheus
docker-compose logs -f grafana

# Test endpoints
curl http://localhost:5000/health    # MLFlow
curl http://localhost:8000/health    # API
curl http://localhost:9090/-/healthy # Prometheus
curl http://localhost:3000/api/health # Grafana
```

### Reset Everything

```bash
# Stop all services
docker-compose down

# Remove volumes (️deletes all data!)
docker-compose down -v

# Rebuild and start fresh
docker-compose build

# Step 1: Infrastructure
docker-compose up -d postgres minio minio-init mlflow prometheus grafana

# Step 2: Train model (wait for mlflow to be healthy first)
python scripts/training.py

# Step 3: API
docker-compose up -d api evidently

# Step 4: Airflow
docker-compose up -d airflow-postgres airflow-init
docker-compose up -d airflow-webserver airflow-scheduler
```

---

## Cấu trúc dự án

```
DDM501_Lab_4_Repository/
├── api/                          # FastAPI phục vụ mô hình
│   ├── Dockerfile
│   ├── main.py                   # API + SHAP /explain + Prometheus metrics
│   └── requirements.txt
├── mlflow/                       # MLFlow server
│   └── Dockerfile
├── evidently/                    # Evidently drift service
│   ├── Dockerfile
│   ├── main.py
│   └── requirements.txt
├── pipeline/                     # Các module ML pipeline tái sử dụng
│   ├── data_ingestion.py         # Load dữ liệu gian lận thẻ tín dụng
│   ├── training.py               # Train mô hình RandomForest
│   ├── registry.py               # Đăng ký mô hình vào MLFlow
│   └── run_pipeline.py           # Pipeline chạy toàn bộ từ đầu đến cuối
├── dags/                         # Airflow DAGs
│   └── credit_fraud_pipeline_dag.py  # DAG credit_fraud_training_pipeline
├── simulations/                  # Load testing & simulation drift
│   ├── run_simulation.py         # CLI runner
│   ├── simulator.py              # Core simulator
│   ├── scenarios.py              # Các kịch bản drift cấu hình sẵn
│   ├── data_generator.py         # Tạo dữ liệu tổng hợp
│   ├── config.yaml               # Cài đặt simulation
│   ├── quick_test.sh             # Smoke test nhanh
│   └── requirements.txt
├── tests/                        # Bộ test (pytest)
│   ├── conftest.py               # Fixtures dùng chung (fraud_data, trained_model)
│   ├── unit/test_pipeline.py     # Unit test cho pipeline modules
│   ├── data/test_data_quality.py # Kiểm tra chất lượng dữ liệu
│   ├── model/test_model_behavior.py  # Kiểm tra hành vi & tính bất biến của mô hình
│   └── integration/test_api.py   # Integration test cho FastAPI
├── scripts/                      # Script training và tiện ích
│   ├── training.py               # Script train mô hình độc lập
│   └── requirements.txt
├── config/                       # File cấu hình
│   ├── prometheus.yml            # Cấu hình scrape của Prometheus
│   └── grafana/
│       ├── alerts.yml
│       ├── provisioning/
│       │   ├── datasources/      # Datasource Grafana
│       │   └── dashboards/       # Provision dashboard
│       └── dashboards/           # File JSON dashboard
│           ├── ml-monitoring.json
│           └── evidently-drift-monitoring.json
├── docs/                         # Tài liệu API
│   ├── API.md
│   └── openapi.yaml
├── docker-compose.yml            # Điều phối toàn bộ stack
└── README.md                     # File này

```

---

## 📄 Giấy phép

Dự án này được cung cấp "as-is" phục vụ mục đích học tập và demo.