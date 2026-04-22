# airflow/dags/plantdisease_pipeline.py
# DA5402 — A6: Orchestrated Web Scraper Pipeline
# Adapted for Plant Disease Detection project

import os
import csv
import json
import sqlite3
import logging
import requests
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.email import EmailOperator
from airflow.sensors.filesystem import FileSensor
from airflow.operators.empty import EmptyOperator

# ─────────────────────────────────────────
# Default Args
# ─────────────────────────────────────────
default_args = {
    "owner": "ajazhsn",
    "depends_on_past": False,
    "start_date": datetime(2026, 4, 1),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=1),
    "retry_exponential_backoff": True,
}

# ─────────────────────────────────────────
# Config
# ─────────────────────────────────────────
INPUT_DIR = "/mnt/d/Plant Disease Detection/data/airflow_input"
DB_PATH = "/mnt/d/Plant Disease Detection/data/plantdisease.db"
API_URL = "http://localhost:8000/predict"
BATCH_THRESHOLD = 10

# ─────────────────────────────────────────
# Task Functions
# ─────────────────────────────────────────


def init_database(**context):
    """Initialize SQLite database for storing predictions."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            image_path TEXT,
            predicted_class TEXT,
            confidence REAL,
            inference_time_ms REAL,
            processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            status TEXT DEFAULT 'success',
            error_message TEXT
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS pipeline_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            csv_file TEXT,
            total_images INTEGER,
            successful INTEGER,
            failed INTEGER,
            status TEXT
        )
    """)
    conn.commit()
    conn.close()
    logging.info("Database initialized successfully")
    return "DB initialized"


def read_csv_and_get_images(**context):
    """Read CSV file from input directory and get image paths."""
    os.makedirs(INPUT_DIR, exist_ok=True)

    # Find CSV files
    csv_files = list(Path(INPUT_DIR).glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {INPUT_DIR}")

    csv_file = csv_files[0]
    logging.info(f"Processing CSV: {csv_file}")

    image_paths = []
    with open(csv_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_path = row.get(
                "image_path", row.get("path", ""))
            if image_path and Path(image_path).exists():
                image_paths.append(image_path)
            else:
                logging.warning(
                    f"Image not found: {image_path}")

    logging.info(f"Found {len(image_paths)} valid images")

    # Push to XCom
    context["ti"].xcom_push(
        key="image_paths", value=image_paths)
    context["ti"].xcom_push(
        key="csv_file", value=str(csv_file))

    return len(image_paths)


def process_image(image_path: str) -> dict:
    """Send single image to FastAPI for prediction."""
    try:
        with open(image_path, "rb") as f:
            response = requests.post(
                API_URL,
                files={"file": (
                    Path(image_path).name, f, "image/jpeg")},
                timeout=30
            )

        if response.status_code == 200:
            result = response.json()
            return {
                "filename": Path(image_path).name,
                "image_path": image_path,
                "predicted_class": result["predicted_class"],
                "confidence": result["confidence"],
                "inference_time_ms": result[
                    "inference_time_ms"],
                "status": "success"
            }
        else:
            return {
                "filename": Path(image_path).name,
                "image_path": image_path,
                "status": "error",
                "error_message": (
                    f"HTTP {response.status_code}")
            }
    except Exception as e:
        return {
            "filename": Path(image_path).name,
            "image_path": image_path,
            "status": "error",
            "error_message": str(e)
        }


def run_batch_predictions(**context):
    """Process all images and store results in DB."""
    ti = context["ti"]
    image_paths = ti.xcom_pull(
        key="image_paths",
        task_ids="read_csv_task")

    if not image_paths:
        logging.warning("No images to process")
        return 0

    results = []
    failed_images = []

    for image_path in image_paths:
        result = process_image(image_path)
        results.append(result)
        if result["status"] == "error":
            failed_images.append(result)
        logging.info(
            f"Processed: {result['filename']} "
            f"→ {result.get('predicted_class', 'ERROR')}")

    # Store in DB
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    for r in results:
        cursor.execute("""
            INSERT INTO predictions
            (filename, image_path, predicted_class,
             confidence, inference_time_ms, status,
             error_message)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            r["filename"],
            r["image_path"],
            r.get("predicted_class"),
            r.get("confidence"),
            r.get("inference_time_ms"),
            r["status"],
            r.get("error_message")
        ))

    # Log pipeline run
    successful = sum(
        1 for r in results if r["status"] == "success")
    failed = len(results) - successful

    cursor.execute("""
        INSERT INTO pipeline_runs
        (csv_file, total_images, successful, failed, status)
        VALUES (?, ?, ?, ?, ?)
    """, (
        ti.xcom_pull(
            key="csv_file", task_ids="read_csv_task"),
        len(results),
        successful,
        failed,
        "completed"
    ))

    conn.commit()
    conn.close()

    # Push results for downstream tasks
    ti.xcom_push(key="results", value=results)
    ti.xcom_push(key="failed_images", value=failed_images)
    ti.xcom_push(key="successful_count", value=successful)
    ti.xcom_push(key="failed_count", value=failed)

    logging.info(
        f"Batch complete: {successful} success, "
        f"{failed} failed")
    return successful


def check_failures(**context):
    """Check for failed predictions and raise if any."""
    ti = context["ti"]
    failed = ti.xcom_pull(
        key="failed_images",
        task_ids="batch_predict_task")

    if failed:
        failed_list = "\n".join(
            [f"- {f['filename']}: {f['error_message']}"
             for f in failed])
        logging.error(
            f"Failed predictions:\n{failed_list}")
        # Store failed images info for email
        ti.xcom_push(
            key="failed_summary", value=failed_list)
        return "has_failures"
    return "no_failures"


def get_db_stats(**context):
    """Get current database statistics."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        "SELECT COUNT(*) FROM predictions "
        "WHERE status='success'")
    total_success = cursor.fetchone()[0]

    cursor.execute(
        "SELECT COUNT(*) FROM predictions "
        "WHERE status='error'")
    total_errors = cursor.fetchone()[0]

    cursor.execute("""
        SELECT predicted_class, COUNT(*) as count
        FROM predictions
        WHERE status='success'
        GROUP BY predicted_class
        ORDER BY count DESC
    """)
    class_distribution = cursor.fetchall()

    cursor.execute(
        "SELECT AVG(confidence) FROM predictions "
        "WHERE status='success'")
    avg_confidence = cursor.fetchone()[0]

    conn.close()

    stats = {
        "total_success": total_success,
        "total_errors": total_errors,
        "avg_confidence": round(
            avg_confidence or 0, 4),
        "class_distribution": {
            row[0]: row[1]
            for row in class_distribution
        }
    }

    context["ti"].xcom_push(
        key="db_stats", value=stats)

    logging.info(f"DB Stats: {stats}")

    # Check if batch threshold reached
    ti = context["ti"]
    successful = ti.xcom_pull(
        key="successful_count",
        task_ids="batch_predict_task") or 0

    if successful >= BATCH_THRESHOLD:
        return "threshold_reached"
    return "below_threshold"


def archive_csv(**context):
    """Move processed CSV to archive folder."""
    ti = context["ti"]
    csv_file = ti.xcom_pull(
        key="csv_file", task_ids="read_csv_task")

    if csv_file:
        archive_dir = Path(INPUT_DIR) / "processed"
        archive_dir.mkdir(exist_ok=True)
        src = Path(csv_file)
        dst = archive_dir / (
            src.stem +
            f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            + src.suffix)
        src.rename(dst)
        logging.info(f"Archived CSV to {dst}")


# ─────────────────────────────────────────
# DAG Definition
# ─────────────────────────────────────────
with DAG(
    dag_id="plant_disease_pipeline",
    default_args=default_args,
    description=(
        "Plant disease detection pipeline: "
        "sense CSV → predict images → store DB → notify"
    ),
    schedule_interval=timedelta(hours=1),
    catchup=False,
    tags=["plant-disease", "mlops", "da5402"],
    max_active_runs=1,
) as dag:

    # Start
    start = EmptyOperator(task_id="start")

    # Sense for CSV file
    sense_csv = FileSensor(
        task_id="sense_csv_file",
        filepath=INPUT_DIR + "/*.csv",
        fs_conn_id="fs_default",
        poke_interval=30,
        timeout=60 * 60 * 12,  # 12 hours
        mode="poke",
        soft_fail=False,
    )

    # Initialize DB
    init_db = PythonOperator(
        task_id="init_db_task",
        python_callable=init_database,
        pool="default_pool",
    )

    # Read CSV
    read_csv = PythonOperator(
        task_id="read_csv_task",
        python_callable=read_csv_and_get_images,
        pool="scraper_pool",
    )

    # Batch predictions
    batch_predict = PythonOperator(
        task_id="batch_predict_task",
        python_callable=run_batch_predictions,
        pool="scraper_pool",
    )

    # Check failures
    check_fail = PythonOperator(
        task_id="check_failures_task",
        python_callable=check_failures,
        pool="default_pool",
    )

    # Get DB stats
    db_stats = PythonOperator(
        task_id="db_stats_task",
        python_callable=get_db_stats,
        pool="default_pool",
    )

    # Archive CSV
    archive = PythonOperator(
        task_id="archive_csv_task",
        python_callable=archive_csv,
        pool="default_pool",
    )

    # Email: Batch collection stats
    notify_batch = EmailOperator(
        task_id="notify_batch_complete",
        to="da25m006@smail.iitm.ac.in",
        subject=(
            "[Plant Disease Pipeline] "
            "Batch Collection Statistics"
        ),
        html_content="""
        <h2>Plant Disease Pipeline — Batch Report</h2>
        <p>A batch of images has been processed.</p>
        <h3>Statistics</h3>
        <p>Check Airflow logs for detailed stats.</p>
        <p>Dashboard: 
        <a href='http://localhost:3000'>Grafana</a></p>
        """,
        trigger_rule="all_success",
    )

    # Email: Broken links / failures
    notify_failure = EmailOperator(
        task_id="notify_failures",
        to="da25m006@smail.iitm.ac.in",
        subject=(
            "[Plant Disease Pipeline] "
            "⚠️ Prediction Failures Detected"
        ),
        html_content="""
        <h2>Pipeline Failure Alert</h2>
        <p>Some images failed to process.</p>
        <p>Check Airflow logs for details.</p>
        """,
        trigger_rule="all_done",
    )

    # End
    end = EmptyOperator(task_id="end")

    # ─────────────────────────────────────
    # Task Dependencies (The DAG Structure)
    # ─────────────────────────────────────
    start >> sense_csv >> init_db >> read_csv
    read_csv >> batch_predict
    batch_predict >> check_fail
    batch_predict >> db_stats
    check_fail >> notify_failure
    db_stats >> notify_batch
    notify_batch >> archive >> end
    notify_failure >> end
