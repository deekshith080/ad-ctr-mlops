"""
Defines the request and response shapes for the CTR prediction API.
Pydantic validates all incoming data automatically.
"""

from pydantic import BaseModel, Field
from typing import Optional


class PredictRequest(BaseModel):
    """
    Features the model needs to make a CTR prediction.
    These match exactly what the model was trained on.
    """
    # Numeric features
    intCol_0:  float = Field(..., description="Numeric feature 0")
    intCol_1:  float = Field(..., description="Numeric feature 1")
    intCol_2:  float = Field(..., description="Numeric feature 2")
    intCol_3:  float = Field(..., description="Numeric feature 3")
    intCol_4:  float = Field(..., description="Numeric feature 4")
    intCol_5:  float = Field(..., description="Numeric feature 5")
    intCol_6:  float = Field(..., description="Numeric feature 6")
    intCol_7:  float = Field(..., description="Numeric feature 7")
    intCol_8:  float = Field(..., description="Numeric feature 8")
    intCol_9:  float = Field(..., description="Numeric feature 9")
    intCol_10: float = Field(..., description="Numeric feature 10")
    intCol_11: float = Field(..., description="Numeric feature 11")
    intCol_12: float = Field(..., description="Numeric feature 12")

    # Categorical features
    catCol_0:  str = Field(..., description="Categorical feature 0")
    catCol_1:  str = Field(..., description="Categorical feature 1")
    catCol_2:  str = Field(..., description="Categorical feature 2")
    catCol_3:  str = Field(..., description="Categorical feature 3")
    catCol_4:  str = Field(..., description="Categorical feature 4")
    catCol_5:  str = Field(..., description="Categorical feature 5")
    catCol_6:  str = Field(..., description="Categorical feature 6")
    catCol_7:  str = Field(..., description="Categorical feature 7")
    catCol_8:  str = Field(..., description="Categorical feature 8")
    catCol_9:  str = Field(..., description="Categorical feature 9")
    catCol_10: str = Field(..., description="Categorical feature 10")
    catCol_11: str = Field(..., description="Categorical feature 11")
    catCol_12: str = Field(..., description="Categorical feature 12")
    catCol_13: str = Field(..., description="Categorical feature 13")
    catCol_14: str = Field(..., description="Categorical feature 14")
    catCol_15: str = Field(..., description="Categorical feature 15")
    catCol_16: str = Field(..., description="Categorical feature 16")
    catCol_17: str = Field(..., description="Categorical feature 17")
    catCol_18: str = Field(..., description="Categorical feature 18")
    catCol_19: str = Field(..., description="Categorical feature 19")
    catCol_20: str = Field(..., description="Categorical feature 20")
    catCol_21: str = Field(..., description="Categorical feature 21")
    catCol_22: str = Field(..., description="Categorical feature 22")
    catCol_23: str = Field(..., description="Categorical feature 23")
    catCol_24: str = Field(..., description="Categorical feature 24")
    catCol_25: str = Field(..., description="Categorical feature 25")

    class Config:
        json_schema_extra = {
            "example": {
                "intCol_0": 1, "intCol_1": 1, "intCol_2": 5,
                "intCol_3": 0, "intCol_4": 1382, "intCol_5": 4,
                "intCol_6": 15, "intCol_7": 2, "intCol_8": 181,
                "intCol_9": 1, "intCol_10": 2, "intCol_11": 1,
                "intCol_12": 1,
                "catCol_0": "05db9164", "catCol_1": "08d6d899",
                "catCol_2": "a99f214a", "catCol_3": "5b392875",
                "catCol_4": "43b19349", "catCol_5": "6f6d9be8",
                "catCol_6": "bcdee96c", "catCol_7": "cada4365",
                "catCol_8": "001f3601", "catCol_9": "07d13a8f",
                "catCol_10": "1f89b562", "catCol_11": "a7b606c4",
                "catCol_12": "06367733", "catCol_13": "None",
                "catCol_14": "None", "catCol_15": "32c7478e",
                "catCol_16": "3fdb382b", "catCol_17": "None",
                "catCol_18": "None", "catCol_19": "None",
                "catCol_20": "None", "catCol_21": "None",
                "catCol_22": "None", "catCol_23": "None",
                "catCol_24": "None", "catCol_25": "None",
            }
        }


class PredictResponse(BaseModel):
    """What the API returns after making a prediction."""
    ctr_probability: float = Field(..., description="Predicted click probability (0 to 1)")
    will_click:      bool  = Field(..., description="True if probability >= 0.5")
    model_version:   str   = Field(..., description="MLflow run ID of the model used")


class HealthResponse(BaseModel):
    """Response for the /health endpoint."""
    status:        str = Field(..., description="API status")
    model_loaded:  bool = Field(..., description="Whether model is loaded")


class ModelInfoResponse(BaseModel):
    """Response for the /model/info endpoint."""
    model_uri:     str = Field(..., description="MLflow model URI")
    run_id:        str = Field(..., description="MLflow run ID")
    baseline_auc:  float = Field(..., description="AUC at training time")
