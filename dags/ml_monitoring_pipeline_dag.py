"""Airflow DAG for orchestrating the ML monitoring demo pipeline.

Pipeline steps:
1. Validate repository paths and required files.
2. Start core infra services (Postgres, MinIO, MLflow, Prometheus, Grafana).
3. Train and register the model in MLflow.
4. Start API and Evidently services.
5. Wait for API and Evidently health endpoints.
6. Run prediction simulation and trigger drift analysis.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

from airflow import DAG
from airflow.operators.python import PythonOperator


PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[1]))
COMPOSE_FILE = PROJECT_ROOT / "docker-compose.yml"
TRAINING_SCRIPT = PROJECT_ROOT / "scripts" / "training.py"
SIMULATION_SCRIPT = PROJECT_ROOT / "simulations" / "run_simulation.py"
SIMULATION_CONFIG = PROJECT_ROOT / "simulations" / "config.yaml"
ENABLE_DOCKER_COMPOSE_TASKS = os.getenv("ENABLE_DOCKER_COMPOSE_TASKS", "false").lower() == "true"
API_HEALTH_URL = os.getenv("API_HEALTH_URL", "http://api:8000/health")
EVIDENTLY_HEALTH_URL = os.getenv("EVIDENTLY_HEALTH_URL", "http://evidently:8001/health")


def _compose_command() -> list[str]:
    """Return available docker compose command as tokenized list."""
    if shutil.which("docker"):
        check = subprocess.run(
            ["docker", "compose", "version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if check.returncode == 0:
            return ["docker", "compose"]

    if shutil.which("docker-compose"):
        return ["docker-compose"]

    raise RuntimeError(
        "Neither 'docker compose' nor 'docker-compose' is available in PATH."
    )


def _run_command(command: list[str], cwd: Path | None = None) -> None:
    """Run shell command with logging and fail fast on non-zero exit."""
    cmd_text = " ".join(shlex.quote(part) for part in command)
    print(f"\n[cmd] {cmd_text}")
    subprocess.run(command, cwd=str(cwd) if cwd else None, check=True)


def _check_url(url: str, timeout_seconds: int = 5) -> bool:
    try:
        with urlopen(url, timeout=timeout_seconds) as response:
            return 200 <= response.status < 400
    except URLError:
        return False


def validate_workspace() -> None:
    required = [
        COMPOSE_FILE,
        TRAINING_SCRIPT,
        SIMULATION_SCRIPT,
        SIMULATION_CONFIG,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Required files not found for DAG execution: " + ", ".join(missing)
        )


def start_core_services() -> None:
    if not ENABLE_DOCKER_COMPOSE_TASKS:
        print("Skipping compose startup for core services (ENABLE_DOCKER_COMPOSE_TASKS=false).")
        return

    compose = _compose_command()
    _run_command(
        compose
        + [
            "-f",
            str(COMPOSE_FILE),
            "up",
            "-d",
            "postgres",
            "minio",
            "minio-init",
            "mlflow",
            "prometheus",
            "grafana",
        ],
        cwd=PROJECT_ROOT,
    )


def train_and_register_model() -> None:
    _run_command(["python", str(TRAINING_SCRIPT)], cwd=PROJECT_ROOT)


def start_inference_services() -> None:
    if not ENABLE_DOCKER_COMPOSE_TASKS:
        print("Skipping compose startup for inference services (ENABLE_DOCKER_COMPOSE_TASKS=false).")
        return

    compose = _compose_command()
    _run_command(
        compose + ["-f", str(COMPOSE_FILE), "up", "-d", "api", "evidently"],
        cwd=PROJECT_ROOT,
    )


def wait_for_service_health(url: str, wait_seconds: int = 300, interval: int = 10) -> None:
    deadline = time.time() + wait_seconds
    while time.time() < deadline:
        if _check_url(url):
            print(f"Service healthy: {url}")
            return
        print(f"Waiting for healthy service: {url}")
        time.sleep(interval)

    raise TimeoutError(f"Service did not become healthy in time: {url}")


def wait_for_api_health() -> None:
    wait_for_service_health(API_HEALTH_URL)


def wait_for_evidently_health() -> None:
    wait_for_service_health(EVIDENTLY_HEALTH_URL)


def run_simulation_with_analysis() -> None:
    _run_command(
        [
            "python",
            str(SIMULATION_SCRIPT),
            "--config",
            str(SIMULATION_CONFIG),
            "--scenario",
            "normal",
            "--requests",
            "200",
            "--rps",
            "2",
            "--analyze",
            "--window",
            "100",
        ],
        cwd=PROJECT_ROOT,
    )


default_args = {
    "owner": "ml-monitoring",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="ml_monitoring_orchestration",
    description="Train model, start services, and run traffic/drift simulation.",
    default_args=default_args,
    start_date=datetime(2025, 1, 1),
    schedule="@daily",
    catchup=False,
    max_active_runs=1,
    tags=["ml", "monitoring", "mlflow", "evidently"],
) as dag:
    validate_workspace_task = PythonOperator(
        task_id="validate_workspace",
        python_callable=validate_workspace,
    )

    start_core_services_task = PythonOperator(
        task_id="start_core_services",
        python_callable=start_core_services,
    )

    train_model_task = PythonOperator(
        task_id="train_and_register_model",
        python_callable=train_and_register_model,
    )

    start_inference_services_task = PythonOperator(
        task_id="start_inference_services",
        python_callable=start_inference_services,
    )

    wait_api_task = PythonOperator(
        task_id="wait_for_api_health",
        python_callable=wait_for_api_health,
    )

    wait_evidently_task = PythonOperator(
        task_id="wait_for_evidently_health",
        python_callable=wait_for_evidently_health,
    )

    run_simulation_task = PythonOperator(
        task_id="run_simulation_with_analysis",
        python_callable=run_simulation_with_analysis,
    )

    (
        validate_workspace_task
        >> start_core_services_task
        >> train_model_task
        >> start_inference_services_task
        >> [wait_api_task, wait_evidently_task]
        >> run_simulation_task
    )
