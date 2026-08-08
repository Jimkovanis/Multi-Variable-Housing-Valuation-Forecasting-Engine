"""
predict.py
The "forecasting engine": takes a dict of raw house features (same schema
as the original Ames CSV columns) and returns a predicted sale price.

Handles alignment with the one-hot-encoded training columns automatically,
so callers only need to supply raw, human-readable feature values.
"""

import json
import os
import sys
import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_prep import handle_missing_values, engineer_features, encode_categoricals

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
FEATURES_PATH = os.path.join(MODELS_DIR, "feature_columns.json")
RAW_COLUMNS_PATH = os.path.join(MODELS_DIR, "raw_columns.json")
IMPUTATION_STATS_PATH = os.path.join(MODELS_DIR, "imputation_stats.json")


class HousingValuationEngine:
    def __init__(self,
                 model_path: str = MODEL_PATH,
                 scaler_path: str = SCALER_PATH,
                 features_path: str = FEATURES_PATH,
                 raw_columns_path: str = RAW_COLUMNS_PATH,
                 imputation_stats_path: str = IMPUTATION_STATS_PATH):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        with open(features_path) as f:
            self.feature_columns = json.load(f)
        with open(raw_columns_path) as f:
            self.raw_columns = json.load(f)
        with open(imputation_stats_path) as f:
            self.imputation_stats = json.load(f)

    def predict(self, house: dict) -> float:
        """
        house: dict of raw feature values, e.g.
            {"Gr Liv Area": 1800, "Overall Qual": 7, "Neighborhood": "NAmes", ...}
        Missing fields are fine -- they'll be treated as missing and imputed
        the same way the training pipeline handled them.

        Returns: predicted sale price in dollars (float).
        """
        df = pd.DataFrame([house])
        # Backfill any raw columns the caller didn't supply, as NaN, so the
        # cleaning pipeline (which expects the full raw schema) can run
        # unmodified on a single partially-specified house.
        df = df.reindex(columns=self.raw_columns, fill_value=np.nan)

        df = handle_missing_values(df, stats=self.imputation_stats)
        df = engineer_features(df)
        df = encode_categoricals(df)

        # Align to the exact training feature set: add any missing one-hot
        # columns as 0, drop any extras, and enforce the training column order.
        df = df.reindex(columns=self.feature_columns, fill_value=0)

        X_scaled = self.scaler.transform(df)
        log_pred = self.model.predict(X_scaled)[0]
        return float(np.expm1(log_pred))


if __name__ == "__main__":
    engine = HousingValuationEngine()

    example_house = {
        "Gr Liv Area": 1800,
        "Overall Qual": 7,
        "Overall Cond": 5,
        "Total Bsmt SF": 900,
        "1st Flr SF": 1000,
        "2nd Flr SF": 800,
        "Garage Cars": 2,
        "Garage Area": 480,
        "Full Bath": 2,
        "Half Bath": 1,
        "Bedroom AbvGr": 3,
        "Year Built": 2005,
        "Year Remod/Add": 2005,
        "Yr Sold": 2010,
        "Neighborhood": "NAmes",
        "MS Zoning": "RL",
    }

    price = engine.predict(example_house)
    print(f"Predicted sale price: ${price:,.0f}")
