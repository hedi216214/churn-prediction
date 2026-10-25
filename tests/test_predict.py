"""Run with:  python -m pytest tests   (after python src/train.py)"""
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from predict import Customer, score  # noqa: E402

RISKY = dict(gender="Female", SeniorCitizen=1, Partner="No", Dependents="No", tenure=2,
             PhoneService="Yes", MultipleLines="No", InternetService="Fiber optic",
             OnlineSecurity="No", OnlineBackup="No", DeviceProtection="No", TechSupport="No",
             StreamingTV="Yes", StreamingMovies="Yes", Contract="Month-to-month",
             PaperlessBilling="Yes", PaymentMethod="Electronic check",
             MonthlyCharges=99.5, TotalCharges=199.0)
LOYAL = dict(RISKY, SeniorCitizen=0, Partner="Yes", tenure=68, InternetService="DSL",
             OnlineSecurity="Yes", TechSupport="Yes", Contract="Two year",
             PaperlessBilling="No", PaymentMethod="Bank transfer (automatic)",
             MonthlyCharges=60.0, TotalCharges=4080.0)


def test_risky_customer_scores_higher_than_loyal():
    risky, loyal = score([Customer(**RISKY), Customer(**LOYAL)])
    assert 0 <= loyal.churn_probability < risky.churn_probability <= 1
    assert risky.contact_for_retention


def test_invalid_input_is_rejected():
    with pytest.raises(ValidationError):
        Customer(**dict(RISKY, Contract="Weekly"))
