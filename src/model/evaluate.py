"""
src/model/evaluate.py

Evaluates the live XGBoost model on current data.
Answers: "Is our model still as accurate as when we trained it?"
"""

import argparse
import logging
from pathlib import Path

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    log_loss,
    average_precision_score,
    classification_report,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FEATURE_COLS = [
    "intCol_0", "intCol_1", "intCol_2", "intCol_3", "intCol_4",
    "intCol_5", "intCol_6", "intCol_7", "intCol_8", "intCol_9",
    "intCol_10", "intCol_11", "intCol_12",
    "catCol_0", "catCol_1", "catCol_2", "catCol_3", "catCol_4",
    "catCol_5", "catCol_6", "catCol_7", "catCol_8", "catCol_9",
    "catCol_10", "catCol_11", "catCol_12", "catCol_13", "catCol_14",
    "catCol_15", "catCol_16", "catCol_17", "catCol_18", "catCol_19",
    "catCol_20", "catCol_21", "catCol_22", "catCol_23", "catCol_24",
    "catCol_25",
]
LABEL_COL = "target"


def load_data(path: str, sample: int = 100_000):
    """
    Load current data from CSV.
    Only keeps the feature columns the model was trained on.
    Warns if any columns are missing.
    """
    df = pd.read_csv(path, nrows=sample)
    available = [f for f in FEATURE_COLS if f in df.columns]
    missing = set(FEATURE_COLS) - set(available)
    if missing:
        logger.warning(f"Missing columns: {missing}")
    X = df[available]
    y = df[LABEL_COL]
    return X, y


def compute_metrics(y_true: pd.Series, y_prob: np.ndarray) -> dict:
    """
    Compute all evaluation metrics.

    Why each metric:
    - auc_roc       : main model quality score (was 0.7686 at training)
    - log_loss      : how confident and correct predictions are
    - avg_precision : useful when clicks are rare (imbalanced data)
    - ctr_predicted : average predicted click rate
    - ctr_actual    : real click rate in current data
    """
    y_pred = (y_prob >= 0.5).astype(int)
    report = classification_report(y_true, y_pred, output_dict=True)
    return {
        "auc_roc":          round(roc_auc_score(y_true, y_prob), 4),
        "log_loss":         round(log_loss(y_true, y_prob), 4),
        "avg_precision":    round(average_precision_score(y_true, y_prob), 4),
        "accuracy":         round(report["accuracy"], 4),
        "precision_class1": round(report["1"]["precision"], 4),
        "recall_class1":    round(report["1"]["recall"], 4),
        "f1_class1":        round(report["1"]["f1-score"], 4),
        "ctr_predicted":    round(float(y_prob.mean()), 4),
        "ctr_actual":       round(float(y_true.mean()), 4),
    }


def evaluate(
    data_path: str,
    model_uri: str,
    sample: int = 100_000,
    baseline_auc: float = 0.7686,
) -> dict:
    """
    Main evaluation function.

    Steps:
    1. Load model from MLflow
    2. Load current data
    3. Run predictions
    4. Compute metrics
    5. Compare against baseline AUC
    6. Flag if retraining is needed
    """
    logger.info(f"Loading model from MLflow: {model_uri}")
    model = mlflow.xgboost.load_model(model_uri)

    logger.info(f"Loading data from: {data_path}")
    X, y = load_data(data_path, sample)

    # Encode categorical columns exactly like train.py does
    from sklearn.preprocessing import LabelEncoder
    for col in X.select_dtypes(include="object").columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    # Model is XGBClassifier (sklearn API) — pass DataFrame directly
    y_prob = model.predict_proba(X)[:, 1]

    # Compute all metrics
    metrics = compute_metrics(y, y_prob)

    # Compare against your Week 1 baseline
    metrics["baseline_auc"]        = baseline_auc
    metrics["auc_degradation"]     = round(baseline_auc - metrics["auc_roc"], 4)
    metrics["retrain_recommended"] = metrics["auc_degradation"] > 0.02

    _log_summary(metrics)
    return metrics


def log_to_mlflow(metrics: dict, run_name: str = "model_evaluation") -> None:
    """
    Send all metrics to MLflow.
    You'll see these as graphs in your MLflow dashboard over time.
    """
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("pipeline_stage", "evaluation")
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(k, float(v))
            elif isinstance(v, bool):
                mlflow.log_metric(k, int(v))
    logger.info("Evaluation metrics logged to MLflow.")


def _log_summary(m: dict) -> None:
    logger.info("── Evaluation Results ─────────────────────────")
    logger.info(f"  AUC-ROC        : {m['auc_roc']}  (baseline {m['baseline_auc']})")
    logger.info(f"  Log Loss       : {m['log_loss']}")
    logger.info(f"  Avg Precision  : {m['avg_precision']}")
    logger.info(f"  Accuracy       : {m['accuracy']}")
    logger.info(f"  CTR actual     : {m['ctr_actual']}  | predicted: {m['ctr_predicted']}")
    logger.info(f"  AUC Degradation: {m['auc_degradation']}")
    if m["retrain_recommended"]:
        logger.warning("  RETRAIN RECOMMENDED — AUC degraded more than 2%")
    else:
        logger.info("  Model performance is healthy.")
    logger.info("───────────────────────────────────────────────")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",         required=True)
    parser.add_argument("--model-uri",    required=True)
    parser.add_argument("--sample",       type=int,   default=100_000)
    parser.add_argument("--baseline-auc", type=float, default=0.7686)
    args = parser.parse_args()

    metrics = evaluate(
        data_path=args.data,
        model_uri=args.model_uri,
        sample=args.sample,
        baseline_auc=args.baseline_auc,
    )
    log_to_mlflow(metrics)
