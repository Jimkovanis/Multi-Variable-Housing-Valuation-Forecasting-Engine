"""
data_prep.py
Cleaning and feature engineering pipeline for the Ames Housing dataset.

Design notes (learned during EDA):
- Many "missing" values aren't actually missing -- e.g. NaN in `Pool QC`
  means "this house has no pool", not "we don't know the pool quality".
- We fill those with the string "NoFeature" rather than "None", because
  pandas' read_csv silently re-parses the literal text "None" as NaN on
  reload -- a subtle bug that will resurrect missing values if you're
  not careful.
- SalePrice is right-skewed, so we model log1p(SalePrice) and invert
  with expm1() when we want a dollar prediction.
- Quality/condition columns (Ex/Gd/TA/Fa/Po) have a natural order, so
  they're ordinal-encoded rather than one-hot encoded, to preserve that
  ranking information.
"""

import pandas as pd
import numpy as np

QUALITY_SCALE = {"NoFeature": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
BSMT_EXPOSURE_SCALE = {"NoFeature": 0, "No": 1, "Mn": 2, "Av": 3, "Gd": 4}

QUALITY_COLS = [
    "Exter Qual", "Exter Cond", "Bsmt Qual", "Bsmt Cond", "Heating QC",
    "Kitchen Qual", "Fireplace Qu", "Garage Qual", "Garage Cond", "Pool QC",
]

NONE_COLS = [
    "Pool QC", "Misc Feature", "Alley", "Fence", "Fireplace Qu",
    "Garage Type", "Garage Finish", "Garage Qual", "Garage Cond",
    "Bsmt Qual", "Bsmt Cond", "Bsmt Exposure", "BsmtFin Type 1",
    "BsmtFin Type 2", "Mas Vnr Type",
]

ZERO_COLS = [
    "Garage Yr Blt", "Garage Area", "Garage Cars",
    "Bsmt Full Bath", "Bsmt Half Bath", "Total Bsmt SF",
    "BsmtFin SF 1", "BsmtFin SF 2", "Bsmt Unf SF", "Mas Vnr Area",
]


def load_raw(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def compute_imputation_stats(df: pd.DataFrame) -> dict:
    """
    Learn imputation values from a (training) dataset. These must be
    computed once on training data and reused at prediction time --
    recomputing a median/mode from a single new house is meaningless
    (or crashes outright on an all-NaN column).

    Includes column-level medians/modes for every raw column, as a
    catch-all fallback for fields a prediction-time caller didn't supply
    and that aren't covered by the specific NONE_COLS/ZERO_COLS rules
    (e.g. porch square footage, misc columns).
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()

    return {
        "lot_frontage_by_neighborhood": df.groupby("Neighborhood")["Lot Frontage"]
            .median().to_dict(),
        "lot_frontage_overall": float(df["Lot Frontage"].median()),
        "electrical_mode": df["Electrical"].mode()[0],
        "general_numeric_medians": {c: float(df[c].median()) for c in numeric_cols
                                     if c != "SalePrice"},
        "general_categorical_modes": {c: df[c].mode()[0] for c in cat_cols
                                       if df[c].notna().any()},
    }


def handle_missing_values(df: pd.DataFrame, stats: dict = None) -> pd.DataFrame:
    df = df.copy()

    df[NONE_COLS] = df[NONE_COLS].fillna("NoFeature")
    df[ZERO_COLS] = df[ZERO_COLS].fillna(0)

    if stats is None:
        # Fitting on a full dataset (e.g. during training): safe to learn
        # stats directly from it.
        stats = compute_imputation_stats(df)

    by_neigh = stats["lot_frontage_by_neighborhood"]
    df["Lot Frontage"] = df.apply(
        lambda row: by_neigh.get(row["Neighborhood"], stats["lot_frontage_overall"])
        if pd.isna(row["Lot Frontage"]) else row["Lot Frontage"],
        axis=1,
    )
    df["Lot Frontage"] = df["Lot Frontage"].fillna(stats["lot_frontage_overall"])
    df["Electrical"] = df["Electrical"].fillna(stats["electrical_mode"])

    # Catch-all: any remaining NaNs (fields a caller simply didn't supply)
    # fall back to the training-set median/mode for that column.
    for col, val in stats["general_numeric_medians"].items():
        if col in df.columns:
            df[col] = df[col].fillna(val)
    for col, val in stats["general_categorical_modes"].items():
        if col in df.columns:
            df[col] = df[col].fillna(val)

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["House Age"] = (df["Yr Sold"] - df["Year Built"]).clip(lower=0)
    df["Years Since Remodel"] = (df["Yr Sold"] - df["Year Remod/Add"]).clip(lower=0)
    df["Was Remodeled"] = (df["Year Built"] != df["Year Remod/Add"]).astype(int)

    df["Total SF"] = df["Total Bsmt SF"] + df["1st Flr SF"] + df["2nd Flr SF"]
    df["Total Bath"] = (
        df["Full Bath"] + 0.5 * df["Half Bath"]
        + df["Bsmt Full Bath"] + 0.5 * df["Bsmt Half Bath"]
    )

    df["Has Garage"] = (df["Garage Area"] > 0).astype(int)
    df["Has Pool"] = (df["Pool Area"] > 0).astype(int)
    df["Has Fireplace"] = (df["Fireplaces"] > 0).astype(int)

    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in QUALITY_COLS:
        df[col] = df[col].map(QUALITY_SCALE)
    df["Bsmt Exposure"] = df["Bsmt Exposure"].map(BSMT_EXPOSURE_SCALE)

    drop_cols = [c for c in ["Order", "PID"] if c in df.columns]
    df = df.drop(columns=drop_cols)

    remaining_cat = df.select_dtypes(include=["object", "string"]).columns.tolist()
    df = pd.get_dummies(df, columns=remaining_cat, drop_first=True)

    return df


def run_pipeline(raw_csv_path: str, return_stats: bool = False):
    """Full pipeline: raw CSV -> model-ready DataFrame with encoded features.

    If return_stats=True, also returns the imputation stats learned from
    this data, so callers (train.py) can save them for later reuse on new,
    single-house predictions (predict.py).
    """
    df = load_raw(raw_csv_path)
    stats = compute_imputation_stats(df)
    df = handle_missing_values(df, stats=stats)
    df = engineer_features(df)
    df = encode_categoricals(df)

    if return_stats:
        return df, stats
    return df


if __name__ == "__main__":
    import os
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(project_root, "data", "AmesHousing.csv")
    out_path = os.path.join(project_root, "data", "ames_final.csv")

    df = run_pipeline(csv_path)
    df.to_csv(out_path, index=False)
    print(f"Saved processed dataset: {df.shape}")
