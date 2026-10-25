"""Data cleaning and the preprocessing pipeline shared by training and the API."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "Churn"
ID_COL = "customerID"
NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
CATEGORICAL = ["gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
               "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
               "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
               "PaperlessBilling", "PaymentMethod"]
FEATURES = NUMERIC + CATEGORICAL


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Fix the known quirks of the Telco schema."""
    df = df.copy()
    # TotalCharges is text in the source file, blank for customers with tenure 0.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)
    if TARGET in df:
        df[TARGET] = (df[TARGET].astype(str).str.strip() == "Yes").astype(int)
    return df


def preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
