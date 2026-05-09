from __future__ import annotations
import logging
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator

logger = logging.getLogger(__name__)

DEFAULT_ARGS = {
    "owner": "mlops",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}

REFERENCE_DATA_PATH = "data/processed/train.csv"
CURRENT_DATA_PATH = "data/processed/current.csv"
SAMPLE_SIZE = 50_000
BASELINE_AUC = 0.7589


def _get_latest_model_uri(**context):
    import mlflow
    mlflow.set_tracking_uri("mlruns")
    mlflow.set_experiment("ctr-prediction")
    try:
        runs = mlflow.search_runs(order_by=["start_time DESC"], max_results=1)
        if runs.empty:
            raise RuntimeError("No training runs found.")
        run_id = runs.iloc[0]["run_id"]
        uri = f"runs:/{run_id}/model"
        logger.info(f"Model URI: {uri}")
    except Exception as e:
        logger.warning(f"MLflow lookup failed: {e}")
        uri = "models:/ctr_xgboost/Production"
    context["ti"].xcom_push(key="model_uri", value=uri)
    return uri


def _run_drift_detection(**context):
    import pandas as pd
    import sys, os
    sys.path.insert(0, os.getcwd())
    from src.monitoring.drift_detector import DriftDetector
    ref_df = pd.read_csv(REFERENCE_DATA_PATH, nrows=SAMPLE_SIZE)
    cur_df = pd.read_csv(CURRENT_DATA_PATH, nrows=SAMPLE_SIZE)
    detector = DriftDetector(reference_data=ref_df)
    report = detector.run(current_data=cur_df)
    detector.log_to_mlflow(report, run_name="drift_" + context["ds_nodash"])
    detector.save_report(report)
    context["ti"].xcom_push(key="drift_report", value=report["summary"])


def _run_model_evaluation(**context):
    import sys, os
    sys.path.insert(0, os.getcwd())
    from src.model.evaluate import evaluate, log_to_mlflow
    model_uri = context["ti"].xcom_pull(key="model_uri", task_ids="get_latest_model_uri")
    metrics = evaluate(
        data_path=CURRENT_DATA_PATH,
        model_uri=model_uri,
        sample=SAMPLE_SIZE,
        baseline_auc=BASELINE_AUC,
    )
    log_to_mlflow(metrics, run_name="eval_" + context["ds_nodash"])
    context["ti"].xcom_push(key="eval_metrics", value=metrics)


def _check_alerts(**context):
    ti = context["ti"]
    drift = ti.xcom_pull(key="drift_report", task_ids="run_drift_detection")
    eval_m = ti.xcom_pull(key="eval_metrics", task_ids="run_model_evaluation")
    needs_retrain = (
        drift.get("retrain_recommended", False)
        or eval_m.get("retrain_recommended", False)
    )
    if needs_retrain:
        logger.warning("Retraining needed.")
        return "alert_retrain"
    else:
        logger.info("All checks passed.")
        return "pipeline_healthy"


def _alert_retrain(**context):
    import requests
    from dotenv import load_dotenv
    import os
    load_dotenv(dotenv_path="/Users/deekshith/Desktop/ad-ctr-mlops/.env")
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    ti = context["ti"]
    drift = ti.xcom_pull(key="drift_report", task_ids="run_drift_detection")
    eval_m = ti.xcom_pull(key="eval_metrics", task_ids="run_model_evaluation")
    severe = drift.get("severe_features", [])
    moderate = drift.get("moderate_features", [])
    lines = [
        "ALERT: Model Retraining Required",
        "Project  : ad-ctr-mlops",
        "Date     : " + datetime.utcnow().strftime("%Y-%m-%d %H:%M") + " UTC",
        "",
        "Drift Report",
        "  Mean PSI         : " + str(drift.get("mean_psi", "N/A")),
        "  Max PSI          : " + str(drift.get("max_psi", "N/A")),
        "  Severe features  : " + (", ".join(severe) if severe else "None"),
        "  Moderate features: " + (", ".join(moderate) if moderate else "None"),
        "",
        "Model Performance",
        "  Current AUC      : " + str(eval_m.get("auc_roc", "N/A")),
        "  Baseline AUC     : " + str(eval_m.get("baseline_auc", "N/A")),
        "  AUC Degradation  : " + str(eval_m.get("auc_degradation", "N/A")),
        "  Log Loss         : " + str(eval_m.get("log_loss", "N/A")),
        "",
        "Recommended Action: Trigger retraining pipeline.",
    ]
    message = chr(10).join(lines)
    logger.warning("Retrain alert triggered.")
    response = requests.post(
        f"https://api.telegram.org/bot{token}/sendMessage",
        json={"chat_id": chat_id, "text": message},
        timeout=10,
    )
    if response.status_code == 200:
        logger.info("Telegram alert delivered successfully.")
    else:
        logger.error("Telegram alert delivery failed.")


with DAG(
    dag_id="drift_detection_weekly",
    description="Weekly CTR model drift detection and evaluation",
    default_args=DEFAULT_ARGS,
    start_date=datetime(2024, 1, 1),
    schedule="0 1 * * 1",
    catchup=False,
    tags=["mlops", "monitoring", "drift"],
) as dag:

    start = EmptyOperator(task_id="start")
    get_model = PythonOperator(task_id="get_latest_model_uri", python_callable=_get_latest_model_uri)
    drift_task = PythonOperator(task_id="run_drift_detection", python_callable=_run_drift_detection)
    eval_task = PythonOperator(task_id="run_model_evaluation", python_callable=_run_model_evaluation)
    check = BranchPythonOperator(task_id="check_alerts", python_callable=_check_alerts, trigger_rule="all_done")
    alert_retrain = PythonOperator(task_id="alert_retrain", python_callable=_alert_retrain)
    healthy = EmptyOperator(task_id="pipeline_healthy")
    end = EmptyOperator(task_id="end", trigger_rule="none_failed_min_one_success")

    start >> get_model >> [drift_task, eval_task] >> check >> [alert_retrain, healthy] >> end
