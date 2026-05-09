"""
train.py
--------
Trains an XGBoost CTR prediction model on reference data.
Logs parameters, metrics, and model artifact to MLflow.
Saves model to models/ directory.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import mlflow
import mlflow.xgboost
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.preprocessing import LabelEncoder

# Paths
REFERENCE_DATA_PATH = Path("data/processed/reference.csv")
MODEL_DIR           = Path("models")
MLFLOW_TRACKING_URI = "mlruns"

# Target column
TARGET_COL = "target"

# Model parameters
XGB_PARAMS = {
    "n_estimators":     300,
    "max_depth":        6,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "use_label_encoder": False,
    "eval_metric":      "logloss",
    "random_state":     42,
    "n_jobs":           -1,
}


def load_data(path: Path) -> pd.DataFrame:
    """Load processed reference data."""
    print(f"Loading reference data from {path} ...")
    df = pd.read_csv(path)
    print(f"Loaded {len(df):,} rows, {df.shape[1]} columns")
    return df


def prepare_features(df: pd.DataFrame):
    """
    Separate features from target.
    Encode any remaining string columns.
    """
    print("Preparing features ...")

    # Drop target
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].astype(int)

    # Encode any remaining object columns with LabelEncoder
    for col in X.select_dtypes(include="object").columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    print(f"Features shape: {X.shape}")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    return X, y


def train_model(X_train, y_train, X_val, y_val):
    """Train XGBoost model with early stopping."""
    print("Training XGBoost model ...")

    model = xgb.XGBClassifier(**XGB_PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=100,
    )
    return model


def evaluate_model(model, X_val, y_val):
    """Compute AUC and log loss on validation set."""
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    auc      = roc_auc_score(y_val, y_pred_proba)
    logloss  = log_loss(y_val, y_pred_proba)
    print(f"\nValidation AUC:      {auc:.4f}")
    print(f"Validation Log Loss: {logloss:.4f}")
    return auc, logloss


def save_model(model, model_dir: Path):
    """Save model artifact to disk."""
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "ctr_model.json"
    model.save_model(str(model_path))
    print(f"Model saved → {model_path}")
    return model_path


def main():
    print("=" * 55)
    print("  CTR Model Training Pipeline")
    print("=" * 55)

    # Set MLflow tracking
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("ctr-prediction")

    # Load and prepare data
    df       = load_data(REFERENCE_DATA_PATH)
    X, y     = prepare_features(df)

    # Time-based split — first 80% train, last 20% val
    split_idx  = int(len(X) * 0.8)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"\nTrain size: {len(X_train):,}")
    print(f"Val size:   {len(X_val):,}")

    # Train with MLflow tracking
    with mlflow.start_run(run_name="xgboost-ctr-v1"):

        # Log parameters
        mlflow.log_params(XGB_PARAMS)
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("val_size",   len(X_val))
        mlflow.log_param("split_type", "time_based_80_20")

        # Train
        model = train_model(X_train, y_train, X_val, y_val)

        # Evaluate
        auc, logloss = evaluate_model(model, X_val, y_val)

        # Log metrics
        mlflow.log_metric("val_auc",      auc)
        mlflow.log_metric("val_logloss",  logloss)

        # Save and log model
        model_path = save_model(model, MODEL_DIR)
        mlflow.xgboost.log_model(model, "model")

        print(f"\n Training complete.")
        print(f"   AUC:      {auc:.4f}")
        print(f"   Log Loss: {logloss:.4f}")
        print(f"   MLflow run logged to: {MLFLOW_TRACKING_URI}")


if __name__ == "__main__":
    main()