# Central configuration for the RAINFO ML training pipeline.
# Single source of truth: paths, seed, features, and model registry.

import os
import sys

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

SEED = 42
TEST_SIZE = 0.20
CV_SPLITS = 5

# ---- Paths ----
RAW_DATA_PATH = os.path.join(BASE_DIR, "dataset.csv")
INCIDENTS_PATH = os.path.join(BASE_DIR, "India_Flood_Incidents_Data.csv")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "ml", "data", "processed")
REPORTS_DIR = os.path.join(BASE_DIR, "ml", "reports")
MODELS_DIR = os.path.join(BASE_DIR, "ml_models")
BACKEND_MODELS_DIR = os.path.join(BASE_DIR, "backend", "app", "models")

# Make backend physics functions importable (reused, not re-implemented)
sys.path.insert(0, os.path.join(BASE_DIR, "backend"))

REQUIRED_COLUMNS = [
    "rainfall_6h", "rainfall_24h", "soil_moisture", "slope",
    "elevation", "cn", "antecedent_rain_3d", "flood_occurred",
]

# The exact 12-feature vector used at training AND serving time. Order matters.
FEATURE_COLS = [
    "rainfall_6h",
    "rainfall_24h",
    "soil_moisture",
    "slope",
    "elevation",
    "cn",
    "antecedent_rain_3d",
    "api",
    "cn_adjusted",
    "potential_retention_mm",
    "expected_runoff_mm",
    "time_concentration_min",
]

# Model registry: name -> estimator factory (all seeded for reproducibility).
def build_model_registry():
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier

    return {
        "LogisticRegression": lambda: LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=SEED,
        ),
        "RandomForest": lambda: RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_split=4, min_samples_leaf=2,
            class_weight="balanced", n_jobs=-1, random_state=SEED,
        ),
        "GradientBoosting": lambda: GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=3,
            subsample=0.9, random_state=SEED,
        ),
        "LightGBM": lambda: LGBMClassifier(
            n_estimators=300, learning_rate=0.05, num_leaves=15, max_depth=6,
            min_child_samples=10, subsample=0.9, colsample_bytree=0.9,
            reg_lambda=1.0, class_weight="balanced", verbose=-1, random_state=SEED,
        ),
        "XGBoost": lambda: XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=4,
            min_child_weight=2, subsample=0.9, colsample_bytree=0.9,
            reg_lambda=1.0, eval_metric="logloss", random_state=SEED,
        ),
    }

# Ensemble members: deliberately diverse strong learners.
ENSEMBLE_MEMBERS = ["RandomForest", "LightGBM", "XGBoost"]

CV_SCORING = ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc", "average_precision"]
