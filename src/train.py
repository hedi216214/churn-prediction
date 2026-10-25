"""Train and evaluate churn models, pick a decision threshold from business costs,
explain the model, and save everything the API needs.

Usage:
    python src/train.py --data data/telco_churn_simulated.csv
    python src/train.py --data data/WA_Fn-UseC_-Telco-Customer-Churn.csv   # real IBM file

Outputs (in models/ and reports/):
    churn_model.joblib, metrics.json, roc_pr.png, feature_importance.png, threshold_profit.png
"""
import argparse
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, precision_recall_curve, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from features import FEATURES, TARGET, clean, preprocessor

ROOT = Path(__file__).resolve().parents[1]

# Business assumptions for choosing the threshold (edit to match the real business):
OFFER_COST = 50.0        # cost of a retention offer sent to one customer
SAVE_RATE = 0.30         # share of real churners the offer keeps
MONTHS_SAVED = 12        # months of revenue kept when a churner is saved


def candidates():
    return {
        "logistic_regression": Pipeline([("prep", preprocessor()),
                                         ("clf", LogisticRegression(max_iter=2000, C=0.5))]),
        "gradient_boosting": Pipeline([("prep", preprocessor()),
                                       ("clf", HistGradientBoostingClassifier(
                                           learning_rate=0.05, max_iter=300, max_leaf_nodes=15,
                                           l2_regularization=1.0, random_state=42))]),
    }


def profit_curve(y, proba, monthly):
    """Expected profit of contacting every customer above each threshold."""
    thresholds = np.linspace(0.05, 0.95, 91)
    profits = []
    for t in thresholds:
        target = proba >= t
        saved_value = (y[target] * monthly[target] * MONTHS_SAVED * SAVE_RATE).sum()
        profits.append(saved_value - OFFER_COST * target.sum())
    return thresholds, np.array(profits)


def main(data_path: str):
    df = clean(pd.read_csv(data_path))
    X, y = df[FEATURES], df[TARGET].values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # 1. Compare models with 5-fold cross-validation on the training set.
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    cv_scores = {}
    for name, model in candidates().items():
        s = cross_val_score(model, X_tr, y_tr, cv=cv, scoring="roc_auc")
        cv_scores[name] = {"roc_auc_mean": round(s.mean(), 4), "roc_auc_std": round(s.std(), 4)}
        print(f"{name:22s} CV ROC-AUC {s.mean():.3f} ± {s.std():.3f}")
    best_name = max(cv_scores, key=lambda k: cv_scores[k]["roc_auc_mean"])

    # 2. Fit the best model and evaluate once on the held-out test set.
    model = candidates()[best_name].fit(X_tr, y_tr)
    proba = model.predict_proba(X_te)[:, 1]
    test = {"roc_auc": round(roc_auc_score(y_te, proba), 4),
            "pr_auc": round(average_precision_score(y_te, proba), 4),
            "churn_rate": round(float(y_te.mean()), 4)}

    # 3. Choose the threshold that maximises expected retention profit.
    th, profit = profit_curve(y_te, proba, X_te["MonthlyCharges"].values)
    best_t = float(th[profit.argmax()])
    pred = proba >= best_t
    tp = int((pred & (y_te == 1)).sum())
    test.update({
        "threshold": round(best_t, 2),
        "precision_at_threshold": round(tp / max(pred.sum(), 1), 4),
        "recall_at_threshold": round(tp / max((y_te == 1).sum(), 1), 4),
        "customers_contacted": int(pred.sum()),
        "test_customers": int(len(y_te)),
        "expected_profit": round(float(profit.max()), 2),
        "profit_contact_everyone": round(float(profit_curve(y_te, np.ones_like(proba),
                                                            X_te["MonthlyCharges"].values)[1][0]), 2),
    })

    # 4. Explain the model with permutation importance (model-agnostic).
    imp = permutation_importance(model, X_te, y_te, scoring="roc_auc", n_repeats=10, random_state=42)
    importance = (pd.Series(imp.importances_mean, index=FEATURES)
                  .sort_values(ascending=False).round(4))

    # 5. Save the model, metrics and figures.
    (ROOT / "models").mkdir(exist_ok=True)
    (ROOT / "reports").mkdir(exist_ok=True)
    joblib.dump({"model": model, "threshold": best_t, "features": FEATURES},
                ROOT / "models" / "churn_model.joblib")
    metrics = {"data": Path(data_path).name, "rows": int(len(df)), "best_model": best_name,
               "cross_validation": cv_scores, "test": test,
               "assumptions": {"offer_cost": OFFER_COST, "save_rate": SAVE_RATE,
                               "months_saved": MONTHS_SAVED},
               "top_features": importance.head(8).to_dict()}
    (ROOT / "reports" / "metrics.json").write_text(json.dumps(metrics, indent=2))

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    fpr, tpr, _ = roc_curve(y_te, proba)
    ax[0].plot(fpr, tpr, label=f"AUC {test['roc_auc']:.3f}")
    ax[0].plot([0, 1], [0, 1], "--", color="grey")
    ax[0].set(title="ROC curve", xlabel="False positive rate", ylabel="True positive rate")
    ax[0].legend()
    p, r, _ = precision_recall_curve(y_te, proba)
    ax[1].plot(r, p, label=f"AP {test['pr_auc']:.3f}")
    ax[1].axhline(test["churn_rate"], ls="--", color="grey", label="baseline")
    ax[1].set(title="Precision-recall curve", xlabel="Recall", ylabel="Precision")
    ax[1].legend()
    fig.tight_layout(); fig.savefig(ROOT / "reports" / "roc_pr.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    importance.head(10)[::-1].plot.barh(ax=ax)
    ax.set(title="Permutation importance (drop in ROC-AUC)", xlabel="Importance")
    fig.tight_layout(); fig.savefig(ROOT / "reports" / "feature_importance.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(th, profit)
    ax.axvline(best_t, ls="--", color="grey")
    ax.set(title="Expected retention profit by threshold", xlabel="Threshold", ylabel="Profit")
    fig.tight_layout(); fig.savefig(ROOT / "reports" / "threshold_profit.png", dpi=120); plt.close(fig)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "telco_churn_simulated.csv"))
    main(ap.parse_args().data)
