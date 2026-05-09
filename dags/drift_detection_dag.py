"""
Weekly Airflow DAG that:
1. Finds latest model in MLflow
2. Runs drift detection on current data
3. Evaluates model performance
4. Alerts if retraining is needed

Runs every Monday at 01:00 UTC automatically.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator

logger = logging.getLogger(__name__)

# DAG settings
DEFAULT_ARGS = {
    "owner":            "mlops",
    "retries":          1,
    "retry_delay":      timedelta(minutes=5),
    "email_on_failure": False,
}

REFERENCE_DATA_PATH = "data/processed/train.csv"
CURRENT_DATA_PATH   = "data/processed/current.csv"
SAMPLE_SIZE         = 50_000
BASELINE_AUC        = 0.7686


# Task 1: Find the latest model
def _get_latest_model_uri(**context) -> str:
    """
    Looks inside MLflow for the most recent training run.
    Pushes the model URI into XCom so other tasks can use it.
    """
    import mlflow

    try:
        # First try: look for a registered Production model
        client = mlflow.MlflowClient()
        versions = client.get_latest_versions("ctr_xgboost", stages=["Production"])
        if versions:
            uri = "models:/ctr_xgboost/Production"
            logger.info(f"Found production model: {uri}")
        else:
            # Second try: find most recent training run
            runs = mlflow.search_runs(
                filter_string="tags.pipeline_stage = 'training'",
                order_by=["start_time DESC"],
                max_results=1,
            )
            if runs.empty:
                raise RuntimeError("No training runs found in MLflow.")
            run_id = runs.iloc[0]["run_id"]
            uri = f"runs:/{run_id}/model"
            logger.info(f"Found model from run: {uri}")

    except Exception as e:
        logger.warning(f"MLflow lookup failed: {e}")
        uri = "models:/ctr_xgboost/Production"

    # Push URI into XCom so downstream tasks can read it
    context["ti"].xcom_push(key="model_uri", value=uri)
    return uri


# Task 2: Run drift detection
def _run_drift_detection(**context) -> None:
    """
    Loads reference and current data.
    Calculates PSI for all features.
    Saves report and logs to MLflow.
    Pushes summary into XCom.
    """
    import pandas as pd
    import sys
    import os
    sys.path.insert(0, os.getcwd())

    from src.monitoring.drift_detector import DriftDetector

    ref_df = pd.read_csv(REFERENCE_DATA_PATH, nrows=SAMPLE_SIZE)
    cur_df = pd.read_csv(CURRENT_DATA_PATH,   nrows=SAMPLE_SIZE)

    detector = DriftDetector(reference_data=ref_df)
    report   = detector.run(current_data=cur_df)

    detector.log_to_mlflow(report, run_name=f"drift_{context['ds_nodash']}")
    detector.save_report(report)

    # Push summary so check_alerts task can read it
    context["ti"].xcom_push(key="drift_report", value=report["summary"])
    logger.info(f"Drift detection done: {report['summary']}")


# Task 3: Run model evaluation
def _run_model_evaluation(**context) -> None:
    """
    Pulls model URI from XCom (set by Task 1).
    Runs model on current data.
    Computes AUC and other metrics.
    Pushes results into XCom.
    """
    import sys
    import os
    sys.path.insert(0, os.getcwd())

    from src.model.evaluate import evaluate, log_to_mlflow

    # Pull model URI that Task 1 saved
    model_uri = context["ti"].xcom_pull(
        key="model_uri",
        task_ids="get_latest_model_uri"
    )

    metrics = evaluate(
        data_path=CURRENT_DATA_PATH,
        model_uri=model_uri,
        sample=SAMPLE_SIZE,
        baseline_auc=BASELINE_AUC,
    )

    log_to_mlflow(metrics, run_name=f"eval_{context['ds_nodash']}")

    # Push metrics so check_alerts task can read them
    context["ti"].xcom_push(key="eval_metrics", value=metrics)


# Task 4: Check if we need to alert
def _check_alerts(**context) -> str:
    """
    Reads drift and evaluation results from XCom.
    Returns the name of the next task to run.
    This is the branching decision point.
    """
    ti = context["ti"]

    drift_summary = ti.xcom_pull(key="drift_report",  task_ids="run_drift_detection")
    eval_metrics  = ti.xcom_pull(key="eval_metrics",  task_ids="run_model_evaluation")

    needs_retrain = (
        drift_summary.get("retrain_recommended", False)
        or eval_metrics.get("retrain_recommended", False)
    )

    if needs_retrain:
        logger.warning("Retraining needed — routing to alert task.")
        return "alert_retrain"
    else:
        logger.info("Everything healthy — no retraining needed.")
        return "pipeline_healthy"


# Task 5a: Send retrain alert
def _send_telegram_alert(token: str, chat_id: str, message: str) -> bool:
    """
    Sends a message via Telegram Bot API.
    Returns True if successful, False otherwise.
    """
    import requests

    url      = f"https://api.telegram.org/bot{token}/sendMessage"
    payload  = {"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}
    response = requests.post(url, json=payload, timeout=10)
    return response.status_code == 200


def _build_alert_message(drift: dict, eval_m: dict) -> str:
    """
    Builds a clean, professional alert message.
    Plain text — no emojis, no formatting gimmicks.
    """
    severe_features = drift.get("severe_features", [])
    lines = [
        "ALERT: Model Retraining Required",
        f"Project  : ad-ctr-mlops",
        f"Date     : {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC",
        "",
        "Drift Report",
        f"  Mean PSI         : {drift.get('mean_psi', 'N/A')}",
        f"  Max PSI          : {drift.get('max_psi', 'N/A')}",
        f"  Severe features  : {', '.join(severe_features) if severe_features else 'None'}",
        f"  Moderate features: {', '.join(drift.get('moderate_features', [])) or 'None'}",
        "",
        "Model Performance",
        f"  Current AUC      : {eval_m.get('auc_roc', 'N/A')}",
        f"  Baseline AUC     : {eval_m.get('baseline_auc', 'N/A')}",
        f"  AUC Degradation  : {eval_m.get('auc_degradation', 'N/A')}",
        f"  Log Loss         : {eval_m.get('log_loss', 'N/A')}",
        "",
        "Recommended Action: Trigger retraining pipeline.",
    ]
    return "
".join(lines)


def _alert_retrain(**context) -> None:
    """
    Fires when drift or model degradation is detected.
    Sends a structured alert via Telegram and logs to Airflow.
    """
    from dotenv import load_dotenv
    import os

    load_dotenv(dotenv_path="/Users/deekshith/Desktop/ad-ctr-mlops/.env")

    token   = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    ti     = context["ti"]
    drift  = ti.xcom_pull(key="drift_report", task_ids="run_drift_detection")
    eval_m = ti.xcom_pull(key="eval_metrics", task_ids="run_model_evaluation")

    message = _build_alert_message(drift, eval_m)
    logger.warning(f"Retrain alert triggered.
{message}")

    success = _send_telegram_alert(token, chat_id, message)
    if success:
        logger.info("Telegram alert delivered successfully.")
    else:
        logger.error("Telegram alert delivery failed.")


# DAG definition
with DAG(
    dag_id="drift_detection_weekly",
    description="Weekly CTR model drift detection and evaluation",
    default_args=DEFAULT_ARGS,
    start_date=datetime(2024, 1, 1),
    schedule_interval="0 1 * * 1",
    catchup=False,
    tags=["mlops", "monitoring", "drift"],
) as dag:

    start = EmptyOperator(task_id="start")

    get_model = PythonOperator(
        task_id="get_latest_model_uri",
        python_callable=_get_latest_model_uri,
    )

    drift_task = PythonOperator(
        task_id="run_drift_detection",
        python_callable=_run_drift_detection,
    )

    eval_task = PythonOperator(
        task_id="run_model_evaluation",
        python_callable=_run_model_evaluation,
    )

    check = BranchPythonOperator(
        task_id="check_alerts",
        python_callable=_check_alerts,
        trigger_rule="all_done",
    )

    alert_retrain = PythonOperator(
        task_id="alert_retrain",
        python_callable=_alert_retrain,
    )

    healthy = EmptyOperator(task_id="pipeline_healthy")

    end = EmptyOperator(
        task_id="end",
        trigger_rule="none_failed_min_one_success"
    )

    #Wire the tasks together
    start >> get_model >> [drift_task, eval_task] >> check >> [alert_retrain, healthy] >> end

