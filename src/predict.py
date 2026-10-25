"""Scoring logic, kept separate from the web framework so it can be tested alone."""
from functools import lru_cache
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from pydantic import BaseModel, Field

from features import clean

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "churn_model.joblib"

YesNo = Literal["Yes", "No"]
Addon = Literal["Yes", "No", "No internet service"]


class Customer(BaseModel):
    gender: Literal["Male", "Female"]
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0, le=100, description="Months with the company")
    PhoneService: YesNo
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Addon
    OnlineBackup: Addon
    DeviceProtection: Addon
    TechSupport: Addon
    StreamingTV: Addon
    StreamingMovies: Addon
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal["Electronic check", "Mailed check", "Bank transfer (automatic)",
                           "Credit card (automatic)"]
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)


class Prediction(BaseModel):
    churn_probability: float
    contact_for_retention: bool
    threshold: float


@lru_cache(maxsize=1)
def load_bundle():
    return joblib.load(MODEL_PATH)


def score(customers: list[Customer]) -> list[Prediction]:
    bundle = load_bundle()
    df = clean(pd.DataFrame([c.model_dump() for c in customers]))
    proba = bundle["model"].predict_proba(df[bundle["features"]])[:, 1]
    t = bundle["threshold"]
    return [Prediction(churn_probability=round(float(p), 4), contact_for_retention=bool(p >= t),
                       threshold=round(t, 2)) for p in proba]
