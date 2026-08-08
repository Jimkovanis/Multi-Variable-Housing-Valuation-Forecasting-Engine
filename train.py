"""
train.py
Trains and compares Linear, Ridge, and Lasso regression on the processed
Ames Housing data, then saves the best-performing model + scaler + feature
columns to disk for use by predict.py.

Target is log1p(SalePrice) -- see data_prep.py for why.
"""

import json
import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_prep import run_pipeline, load_raw

RANDOM_STATE = 42
TEST_SIZE = 0.2

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "AmesHousing.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")


def load_data(raw_csv_path: str):
    df, stats = run_pipeline(raw_csv_path, return_stats=True)
    X = df.drop(columns=["SalePrice"])
    y = np.log1p(df["SalePrice"])
    return X, y, stats


def evaluate(name, model, X_train, X_test, y_train, y_test):
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    test_pred_dollars = np.expm1(test_pred)
    y_test_dollars = np.expm1(y_test)

    metrics = {
        "train_r2": round(r2_score(y_train, train_pred), 4),
        "test_r2": round(r2_score(y_test, test_pred), 4),
        "test_rmse_log": round(np.sqrt(mean_squared_error(y_test, test_pred)), 4),
        "test_mae_dollars": round(mean_absolute_error(y_test_dollars, test_pred_dollars), 0),
    }
    print(f"=== {name} ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    return metrics


def main():
    X, y, imputation_stats = load_data(raw_csv_path=DATA_PATH)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    candidates = {
        "Linear Regression": LinearRegression(),
        "Ridge": RidgeCV(alphas=np.logspace(-3, 3, 50), cv=5),
        "Lasso": LassoCV(alphas=np.logspace(-4, 1, 50), cv=5, max_iter=20000),
    }

    results = {}
    fitted = {}
    for name, model in candidates.items():
        model.fit(X_train_scaled, y_train)
        results[name] = evaluate(name, model, X_train_scaled, X_test_scaled, y_train, y_test)
        fitted[name] = model
        print()

    # Pick the model with the best test R^2 (generalizes best to unseen houses)
    best_name = max(results, key=lambda n: results[n]["test_r2"])
    best_model = fitted[best_name]
    print(f"Best model: {best_name} (test R2 = {results[best_name]['test_r2']})")

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(best_model, os.path.join(MODELS_DIR, "best_model.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
    with open(os.path.join(MODELS_DIR, "feature_columns.json"), "w") as f:
        json.dump(list(X.columns), f)
    with open(os.path.join(MODELS_DIR, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Save the raw (pre-encoding) column schema too, so predict.py can
    # backfill any fields a caller doesn't supply before running the
    # cleaning/encoding pipeline on a single new house.
    raw_columns = [c for c in load_raw(DATA_PATH).columns if c != "SalePrice"]
    with open(os.path.join(MODELS_DIR, "raw_columns.json"), "w") as f:
        json.dump(raw_columns, f)
    with open(os.path.join(MODELS_DIR, "imputation_stats.json"), "w") as f:
        json.dump(imputation_stats, f)

    print("\nSaved model, scaler, feature list, and metrics to models/")


if __name__ == "__main__":
    main()
