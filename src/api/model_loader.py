"""
Loads the XGBoost model from MLflow once at API startup.
Keeps it in memory so predictions are fast (<100ms).
"""

import logging
import mlflow
import mlflow.xgboost
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

CATEGORICAL_FEATURES = [
    "catCol_0", "catCol_1", "catCol_2", "catCol_3", "catCol_4",
    "catCol_5", "catCol_6", "catCol_7", "catCol_8", "catCol_9",
    "catCol_10", "catCol_11", "catCol_12", "catCol_13", "catCol_14",
    "catCol_15", "catCol_16", "catCol_17", "catCol_18", "catCol_19",
    "catCol_20", "catCol_21", "catCol_22", "catCol_23", "catCol_24",
    "catCol_25",
]

NUMERIC_FEATURES = [
    "intCol_0", "intCol_1", "intCol_2", "intCol_3", "intCol_4",
    "intCol_5", "intCol_6", "intCol_7", "intCol_8", "intCol_9",
    "intCol_10", "intCol_11", "intCol_12",
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


class ModelLoader:
    """
    Loads and holds the XGBoost model in memory.
    Also fits LabelEncoders on training data so categorical
    features are encoded the same way as during training.
    """

    def __init__(self):
        self.model       = None
        self.encoders    = {}
        self.model_uri   = None
        self.run_id      = None
        self.is_loaded   = False
        self.baseline_auc = 0.7589

    def load(self, model_uri: str, reference_data_path: str = "data/processed/train.csv"):
        """
        Load model from MLflow and fit encoders on reference data.

        Why reference data:
        LabelEncoders must be fit on the same data the model was trained on.
        If we fit on new data, category → number mapping would be different
        and predictions would be completely wrong.
        """
        logger.info(f"Loading model from: {model_uri}")
        mlflow.set_tracking_uri("mlruns")

        import xgboost as xgb
        m = xgb.XGBClassifier()
        m.load_model(model_uri)
        self.model = m
        self.model_uri = model_uri
        self.run_id = "local"

        logger.info("Fitting label encoders on reference data...")
        ref_df = pd.read_csv(reference_data_path, nrows=70_000)

        for col in CATEGORICAL_FEATURES:
            if col in ref_df.columns:
                le = LabelEncoder()
                le.fit(ref_df[col].astype(str))
                self.encoders[col] = le

        self.is_loaded = True
        logger.info(f"Model loaded successfully. Run ID: {self.run_id}")

    def predict(self, features: dict) -> float:
        """
        Takes a dictionary of features and returns CTR probability.

        Steps:
        1. Convert dict to DataFrame (model expects DataFrame)
        2. Encode categorical columns using fitted LabelEncoders
        3. Handle unseen categories gracefully
        4. Run prediction
        5. Return probability
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        df = pd.DataFrame([features])

        for col in CATEGORICAL_FEATURES:
            if col in df.columns and col in self.encoders:
                le = self.encoders[col]
                val = str(df[col].iloc[0])
                if val in le.classes_:
                    df[col] = le.transform([val])
                else:
                    # Unseen category — use 0 as fallback
                    # Better than crashing in production
                    logger.warning(f"Unseen category in {col}: {val}. Using fallback.")
                    df[col] = 0

        df = df[ALL_FEATURES]
        probability = float(self.model.predict_proba(df)[0][1])
        return probability


# Global singleton — loaded once, used by all requests
model_loader = ModelLoader()
