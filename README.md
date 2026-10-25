# Customer churn prediction with a profit-based decision threshold

Predict which telecom customers are likely to leave, explain why, and decide **who should get a retention offer** by maximising expected profit rather than accuracy. The model is served through a FastAPI REST API and packaged with Docker.

## Why this project

A churn model is only useful if it changes a business decision. This project goes from raw customer data to a concrete rule: *"contact customers whose churn probability is above X"*, where X is chosen from the cost of an offer and the revenue it can save.

## Pipeline

```
customer data (CSV) → cleaning → preprocessing (scaling + one-hot)
   → model comparison (5-fold CV) → test-set evaluation
   → profit-based threshold → permutation importance
   → saved model → FastAPI /predict → Docker
```

## Data

The code uses the column layout of the public **IBM Telco Customer Churn** dataset (7,043 customers, 21 columns).

- `data/generate_data.py` creates a **simulated** dataset with that same layout, so the project runs with no download. Its churn rules are written in the file: short tenure, month-to-month contracts, fiber internet and electronic-check payment raise the risk, and security and tech-support add-ons lower it.
- To use the real data, download `WA_Fn-UseC_-Telco-Customer-Churn.csv` from Kaggle and run `python src/train.py --data path/to/file.csv`.

## Results on the simulated data

| Step | Result |
|---|---|
| 5-fold CV ROC-AUC, logistic regression | 0.833 ± 0.005 (selected) |
| 5-fold CV ROC-AUC, gradient boosting | 0.817 ± 0.005 |
| Test ROC-AUC / PR-AUC | 0.812 / 0.600 (churn rate 27.4%) |
| Best threshold | 0.15: contact 815 of 1,409 test customers, recall 92%, precision 44% |
| Expected profit | 52,085 vs 30,643 if every customer is contacted (+70%) |

Logistic regression beats gradient boosting here because the simulated churn rules are close to linear. On the real dataset the comparison can go the other way, which is why the code compares models with cross-validation instead of hard-coding one.

The profit assumptions are in `src/train.py`: an offer costs 50, it keeps 30% of real churners, and a saved customer brings 12 months of revenue.

![ROC and PR curves](reports/roc_pr.png)
![Feature importance](reports/feature_importance.png)
![Profit by threshold](reports/threshold_profit.png)

## Run it

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python data/generate_data.py          # simulated data
python src/train.py                   # trains, writes models/ and reports/
python -m pytest -q tests             # 2 tests
cd src && uvicorn api:app --reload    # API on http://127.0.0.1:8000/docs
```

With Docker (after training):

```bash
docker build -t churn-api . && docker run -p 8000:8000 churn-api
```

Example request:

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{
  "gender":"Female","SeniorCitizen":1,"Partner":"No","Dependents":"No","tenure":2,
  "PhoneService":"Yes","MultipleLines":"No","InternetService":"Fiber optic",
  "OnlineSecurity":"No","OnlineBackup":"No","DeviceProtection":"No","TechSupport":"No",
  "StreamingTV":"Yes","StreamingMovies":"Yes","Contract":"Month-to-month",
  "PaperlessBilling":"Yes","PaymentMethod":"Electronic check",
  "MonthlyCharges":99.5,"TotalCharges":199.0}'
# → {"churn_probability": 0.86, "contact_for_retention": true, "threshold": 0.15}
```

## Project structure

```
data/generate_data.py   simulated data with the Telco schema
src/features.py         cleaning + preprocessing pipeline
src/train.py            CV comparison, evaluation, threshold, explainability, saving
src/predict.py          input validation (Pydantic) and scoring
src/api.py              FastAPI endpoints: /health, /predict, /predict/batch
tests/                  unit tests
```

## Next steps

- Run on the real IBM dataset and compare with the simulated results.
- Track experiments with MLflow (parameters, metrics, model versions).
- Add SHAP values for per-customer explanations.
- Build a Power BI page on the scored customers (risk segments, expected revenue at risk).
