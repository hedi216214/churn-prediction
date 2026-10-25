"""Generate a simulated telecom customer dataset with the same columns as the
public IBM "Telco Customer Churn" dataset.

The real dataset can be downloaded from Kaggle or IBM's GitHub; train.py works
with either file. This generator exists so the project runs end to end without
any download, and so the churn logic is known and documented.

Usage:
    python data/make_dataset.py --rows 7043 --out data/telco_churn_simulated.csv
"""
import argparse

import numpy as np
import pandas as pd


def generate(n: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    gender = rng.choice(["Male", "Female"], n)
    senior = rng.binomial(1, 0.16, n)
    partner = rng.choice(["Yes", "No"], n, p=[0.48, 0.52])
    dependents = np.where(partner == "Yes",
                          rng.choice(["Yes", "No"], n, p=[0.5, 0.5]),
                          rng.choice(["Yes", "No"], n, p=[0.1, 0.9]))
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.21, 0.24])

    # Tenure depends on contract type: long contracts mean older customers.
    base = {"Month-to-month": 14, "One year": 36, "Two year": 56}
    tenure = np.array([int(np.clip(rng.normal(base[c], 14), 0, 72)) for c in contract])

    phone = rng.choice(["Yes", "No"], n, p=[0.9, 0.1])
    multiple = np.where(phone == "No", "No phone service", rng.choice(["Yes", "No"], n, p=[0.42, 0.58]))
    internet = rng.choice(["DSL", "Fiber optic", "No"], n, p=[0.34, 0.44, 0.22])

    def addon(p_yes):
        v = rng.choice(["Yes", "No"], n, p=[p_yes, 1 - p_yes])
        return np.where(internet == "No", "No internet service", v)

    security, backup, protection = addon(0.29), addon(0.34), addon(0.34)
    support, tv, movies = addon(0.29), addon(0.38), addon(0.39)

    paperless = rng.choice(["Yes", "No"], n, p=[0.59, 0.41])
    payment = rng.choice(["Electronic check", "Mailed check", "Bank transfer (automatic)",
                          "Credit card (automatic)"], n, p=[0.34, 0.23, 0.22, 0.21])

    monthly = 20 + (phone == "Yes") * 5 + (multiple == "Yes") * 5
    monthly = monthly + np.select([internet == "DSL", internet == "Fiber optic"], [25, 50], 0)
    for col in (security, backup, protection, support, tv, movies):
        monthly = monthly + (col == "Yes") * rng.uniform(4, 10, n)
    monthly = np.round(monthly + rng.normal(0, 2, n), 2)
    total = np.round(monthly * np.maximum(tenure, 0) * rng.uniform(0.95, 1.05, n), 2)

    # Documented churn logic (log-odds), plus noise so the problem is not trivial.
    logit = (-1.6
             + 1.35 * (contract == "Month-to-month") - 0.9 * (contract == "Two year")
             - 0.045 * tenure
             + 0.75 * (internet == "Fiber optic")
             + 0.55 * (payment == "Electronic check")
             - 0.45 * (security == "Yes") - 0.45 * (support == "Yes")
             + 0.30 * (paperless == "Yes") + 0.30 * senior
             + 0.012 * (monthly - 65)
             + rng.normal(0, 0.8, n))
    churn = rng.binomial(1, 1 / (1 + np.exp(-logit)))

    df = pd.DataFrame({
        "customerID": [f"{i:04d}-SIM" for i in range(n)],
        "gender": gender, "SeniorCitizen": senior, "Partner": partner, "Dependents": dependents,
        "tenure": tenure, "PhoneService": phone, "MultipleLines": multiple,
        "InternetService": internet, "OnlineSecurity": security, "OnlineBackup": backup,
        "DeviceProtection": protection, "TechSupport": support, "StreamingTV": tv,
        "StreamingMovies": movies, "Contract": contract, "PaperlessBilling": paperless,
        "PaymentMethod": payment, "MonthlyCharges": monthly,
        "TotalCharges": total.astype(str), "Churn": np.where(churn == 1, "Yes", "No"),
    })
    # The real dataset stores TotalCharges as text with blanks for new customers.
    df.loc[df["tenure"] == 0, "TotalCharges"] = " "
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=7043)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/telco_churn_simulated.csv")
    a = ap.parse_args()
    d = generate(a.rows, a.seed)
    d.to_csv(a.out, index=False)
    print(f"Wrote {len(d)} rows to {a.out} (churn rate {(d.Churn == 'Yes').mean():.1%})")
