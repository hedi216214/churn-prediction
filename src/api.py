"""REST API serving the churn model.

Run locally:
    cd src && uvicorn api:app --reload
Then open http://127.0.0.1:8000/docs to try the endpoints.
"""
from fastapi import FastAPI

from predict import Customer, Prediction, load_bundle, score

app = FastAPI(title="Churn prediction API", version="1.0.0")


@app.get("/health")
def health() -> dict:
    bundle = load_bundle()
    return {"status": "ok", "threshold": round(bundle["threshold"], 2)}


@app.post("/predict", response_model=Prediction)
def predict_one(customer: Customer) -> Prediction:
    return score([customer])[0]


@app.post("/predict/batch", response_model=list[Prediction])
def predict_batch(customers: list[Customer]) -> list[Prediction]:
    return score(customers)
