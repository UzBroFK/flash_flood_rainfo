"""Data loading, validation, and physics feature engineering.

Reuses the calibrated hydrological physics from backend/app/models/feature_engineering.py
(SCS-CN, API, Kirpich) so training features are computed by exactly the same code
the serving layer uses — no drift between train and inference.
"""

import logging
import os

import numpy as np
import pandas as pd

from ml.pipeline.config import (
    RAW_DATA_PATH, INCIDENTS_PATH, PROCESSED_DATA_DIR, FEATURE_COLS, REQUIRED_COLUMNS,
)

logger = logging.getLogger(__name__)

from app.models.feature_engineering import (
    calculate_api,
    adjust_cn_for_moisture,
    calculate_potential_retention,
    calculate_scs_runoff,
    calculate_time_concentration,
)


def load_and_merge_datasets() -> pd.DataFrame:
    """Load the base hydrological dataset and append real historical flood incidents."""
    df_base = pd.read_csv(RAW_DATA_PATH)
    logger.info("Loaded %d base records from %s", len(df_base), RAW_DATA_PATH)

    if os.path.exists(INCIDENTS_PATH):
        df_inc = pd.read_csv(INCIDENTS_PATH)
        # Normalize soil moisture scale (fraction -> percentage)
        if df_inc["soil_moisture"].max() <= 1.0:
            df_inc["soil_moisture"] = df_inc["soil_moisture"] * 100.0
        df_inc["flood_occurred"] = 1
        for col in ["state", "lat", "lon", "casualties", "damage_estimate_inr"]:
            if col not in df_inc.columns:
                df_inc[col] = 0
        cols = [c for c in df_base.columns if c in df_inc.columns]
        df = pd.concat([df_base, df_inc[cols]], ignore_index=True)
        logger.info("Merged %d historical flood incidents from %s", len(df_inc), INCIDENTS_PATH)
    else:
        df = df_base

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    return df


def compute_physics_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply SCS-CN / API / Kirpich physics to produce the 12-feature training matrix."""
    r24 = df["rainfall_24h"].fillna(df["rainfall_6h"] * 2)
    sm = df["soil_moisture"].astype(float)
    ant = df["antecedent_rain_3d"].astype(float)
    cn_adj = [adjust_cn_for_moisture(c, s) for c, s in zip(df["cn"], sm)]
    cn_adj_s = pd.Series(cn_adj, index=df.index)

    df_feat = pd.DataFrame({
        "rainfall_6h": df["rainfall_6h"].astype(float),
        "rainfall_24h": r24.astype(float),
        "soil_moisture": sm,
        "slope": df["slope"].astype(float),
        "elevation": df["elevation"].astype(float),
        "cn": df["cn"].astype(float),
        "antecedent_rain_3d": ant,
        "api": [calculate_api(a) for a in ant],
        "cn_adjusted": cn_adj_s,
        "potential_retention_mm": [calculate_potential_retention(c) for c in cn_adj],
        "expected_runoff_mm": [calculate_scs_runoff(p, c) for p, c in zip(df["rainfall_6h"], cn_adj)],
        "time_concentration_min": [
            calculate_time_concentration(sl, el)
            for sl, el in zip(df["slope"], df["elevation"])
        ],
        "flood_occurred": df["flood_occurred"].astype(int),
    })

    # Trust-boundary validation: drop rows with NaN/inf features (fail loudly if systemic)
    bad = df_feat[FEATURE_COLS].isin([np.inf, -np.inf]).any(axis=1) | df_feat[FEATURE_COLS].isna().any(axis=1)
    n_bad = int(bad.sum())
    if n_bad:
        logger.warning("Dropping %d rows with missing/invalid feature values", n_bad)
        df_feat = df_feat[~bad].reset_index(drop=True)

    if df_feat["flood_occurred"].nunique() < 2:
        raise ValueError("After cleaning, only one class remains — cannot train a classifier.")

    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    out_csv = os.path.join(PROCESSED_DATA_DIR, "features.csv")
    df_feat.to_csv(out_csv, index=False)
    logger.info("Saved %d engineered feature rows to %s", len(df_feat), out_csv)
    return df_feat


def data_summary(df_feat: pd.DataFrame) -> dict:
    """Small dataset fingerprint written into the metrics report."""
    n = len(df_feat)
    n_flood = int(df_feat["flood_occurred"].sum())
    return {
        "n_records": n,
        "n_flood": n_flood,
        "n_no_flood": n - n_flood,
        "flood_ratio": round(n_flood / n, 4),
        "features": FEATURE_COLS,
    }
