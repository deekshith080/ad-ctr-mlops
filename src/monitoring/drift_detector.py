"""
src/monitoring/drift_detector.py

PSI-based drift detector for ad-ctr-mlops.
Compares training data distribution vs current/production data.
"""

import numpy as np
import pandas as pd
import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Optional
import mlflow

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# PSI thresholds (industry standard)
PSI_NEGLIGIBLE = 0.1   # stable - no action
PSI_MODERATE   = 0.2   # investigate
PSI_SEVERE     = 0.25  # retrain now

NUMERIC_FEATURES = [
    "intCol_0", "intCol_1", "intCol_2", "intCol_3", "intCol_4",
    "intCol_5", "intCol_6", "intCol_7", "intCol_8", "intCol_9",
    "intCol_10", "intCol_11", "intCol_12",
]

CATEGORICAL_FEATURES = [
    "catCol_0", "catCol_1", "catCol_2", "catCol_3", "catCol_4",
    "catCol_5", "catCol_6", "catCol_7", "catCol_8", "catCol_9",
    "catCol_10", "catCol_11", "catCol_12", "catCol_13", "catCol_14",
    "catCol_15", "catCol_16", "catCol_17", "catCol_18", "catCol_19",
    "catCol_20", "catCol_21", "catCol_22", "catCol_23", "catCol_24",
    "catCol_25",
]


def _psi_numeric(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """
    PSI for continuous features.
    We use the REFERENCE data to define the bins — this is important.
    If we used current data to define bins, we'd be cheating.
    """
    breakpoints = np.percentile(reference.dropna(), np.linspace(0, 100, bins + 1))
    breakpoints = np.unique(breakpoints)
    if len(breakpoints) < 2:
        return 0.0

    def _bin_counts(series):
        counts, _ = np.histogram(series.dropna(), bins=breakpoints)
        proportions = counts / counts.sum()
        # Replace 0s with small number to avoid log(0) = error
        proportions = np.where(proportions == 0, 1e-4, proportions)
        return proportions

    ref_prop = _bin_counts(reference)
    cur_prop = _bin_counts(current)

    psi = np.sum((cur_prop - ref_prop) * np.log(cur_prop / ref_prop))
    return float(psi)


def _psi_categorical(reference: pd.Series, current: pd.Series) -> float:
    """
    PSI for categorical features.
    Handles unseen categories in current data gracefully.
    """
    ref_counts = reference.value_counts(normalize=True)
    cur_counts = current.value_counts(normalize=True)
    all_cats = set(ref_counts.index) | set(cur_counts.index)

    psi = 0.0
    for cat in all_cats:
        ref_p = ref_counts.get(cat, 1e-4)
        cur_p = cur_counts.get(cat, 1e-4)
        psi += (cur_p - ref_p) * np.log(cur_p / ref_p)
    return float(psi)


def _severity(psi: float) -> str:
    if psi < PSI_NEGLIGIBLE:
        return "stable"
    elif psi < PSI_MODERATE:
        return "moderate_drift"
    else:
        return "severe_drift"


class DriftDetector:
    """
    Main drift detection class.
    
    How to use:
        detector = DriftDetector(reference_data=train_df)
        report = detector.run(current_data=new_df)
        detector.log_to_mlflow(report)
    """

    def __init__(self, reference_data, numeric_features=None, categorical_features=None, bins=10):
        self.reference = reference_data
        self.numeric_features = numeric_features or NUMERIC_FEATURES
        self.categorical_features = categorical_features or CATEGORICAL_FEATURES
        self.bins = bins
        logger.info(f"DriftDetector ready — {len(self.numeric_features)} numeric, {len(self.categorical_features)} categorical features.")

    def run(self, current_data: pd.DataFrame) -> dict:
        """Run PSI for all features. Returns a full report dict."""
        scores = {}

        for feat in self.numeric_features:
            if feat in self.reference.columns and feat in current_data.columns:
                psi = _psi_numeric(self.reference[feat], current_data[feat], self.bins)
                scores[feat] = {"psi": round(psi, 4), "type": "numeric", "severity": _severity(psi)}

        for feat in self.categorical_features:
            if feat in self.reference.columns and feat in current_data.columns:
                psi = _psi_categorical(self.reference[feat], current_data[feat])
                scores[feat] = {"psi": round(psi, 4), "type": "categorical", "severity": _severity(psi)}

        report = self._build_report(scores)
        self._log_summary(report)
        return report

    def log_to_mlflow(self, report: dict, run_name: str = "drift_detection") -> None:
        """Log all PSI scores to MLflow so you can track drift over time."""
        with mlflow.start_run(run_name=run_name):
            mlflow.set_tag("pipeline_stage", "drift_detection")
            for feat, info in report["feature_scores"].items():
                mlflow.log_metric(f"psi_{feat}", info["psi"])
            s = report["summary"]
            mlflow.log_metric("psi_mean", s["mean_psi"])
            mlflow.log_metric("psi_max", s["max_psi"])
            mlflow.log_metric("retrain_recommended", int(s["retrain_recommended"]))
            report_path = Path("/tmp/drift_report.json")
            report_path.write_text(json.dumps(report, indent=2))
            mlflow.log_artifact(str(report_path), artifact_path="drift")
        logger.info("Drift report logged to MLflow.")

    def save_report(self, report: dict, output_dir: str = "reports/drift") -> Path:
        """Save the drift report as a JSON file for auditing."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        path = out / f"drift_{ts}.json"
        path.write_text(json.dumps(report, indent=2))
        logger.info(f"Drift report saved → {path}")
        return path

    def _build_report(self, scores: dict) -> dict:
        psi_values = [v["psi"] for v in scores.values()]
        severe   = [f for f, v in scores.items() if v["severity"] == "severe_drift"]
        moderate = [f for f, v in scores.items() if v["severity"] == "moderate_drift"]
        stable   = [f for f, v in scores.items() if v["severity"] == "stable"]
        return {
            "run_timestamp": datetime.utcnow().isoformat(),
            "feature_scores": scores,
            "summary": {
                "mean_psi":            round(float(np.mean(psi_values)), 4) if psi_values else 0.0,
                "max_psi":             round(float(np.max(psi_values)), 4)  if psi_values else 0.0,
                "n_stable":            len(stable),
                "n_moderate_drift":    len(moderate),
                "n_severe_drift":      len(severe),
                "severe_features":     severe,
                "moderate_features":   moderate,
                "retrain_recommended": len(severe) > 0,
            },
        }

    def _log_summary(self, report: dict) -> None:
        s = report["summary"]
        logger.info("── Drift Summary ──────────────────────────────")
        logger.info(f"  Mean PSI : {s['mean_psi']:.4f}  |  Max PSI: {s['max_psi']:.4f}")
        logger.info(f"  Stable: {s['n_stable']}  |  Moderate: {s['n_moderate_drift']}  |  Severe: {s['n_severe_drift']}")
        if s["retrain_recommended"]:
            logger.warning(f"  RETRAIN RECOMMENDED — severe drift in: {s['severe_features']}")
        else:
            logger.info("  No severe drift detected.")
        logger.info("───────────────────────────────────────────────")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--current",   required=True)
    parser.add_argument("--sample",    type=int, default=50_000)
    args = parser.parse_args()

    ref_df = pd.read_csv(args.reference, nrows=args.sample)
    cur_df = pd.read_csv(args.current,   nrows=args.sample)

    detector = DriftDetector(reference_data=ref_df)
    report   = detector.run(current_data=cur_df)
    detector.save_report(report)
    detector.log_to_mlflow(report)
