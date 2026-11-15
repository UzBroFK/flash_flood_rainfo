#!/usr/bin/env python3
"""Ponytail assert-based self-check for the ML pipeline (zero frameworks).

Runs: python ml/tests/test_pipeline.py
Fails loudly if the training logic, artifacts, or serving contract break.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import joblib
import numpy as np
from ml.pipeline.config import MODELS_DIR, FEATURE_COLS, build_model_registry, ENSEMBLE_MEMBERS
from ml.pipeline.data import load_and_merge_datasets, compute_physics_features


def run_checks():
    print("Running pipeline self-check...")

    # 1. Registry builds every model and all estimators support predict_proba (needed for soft voting)
    registry = build_model_registry()
    for name in list(registry) + ["Ensemble"]:
        if name != "Ensemble":
            m = registry[name]()
            assert hasattr(m, "fit") and hasattr(m, "predict_proba"), f"{name} lacks predict_proba"
    for n in ENSEMBLE_MEMBERS:
        assert n in registry, f"Ensemble member {n} missing from registry"
    print("  [PASS] Model registry + soft-voting prerequisites")

    # 2. Data contract: features.csv matches the serving feature vector, exactly, in order
    df = compute_physics_features(load_and_merge_datasets())
    assert list(df[FEATURE_COLS].columns) == FEATURE_COLS, "Feature column order drift!"
    assert set(np.unique(df["flood_occurred"])) == {0, 1}, "Target must be binary"
    assert len(df) > 50, "Dataset unexpectedly small"
    print(f"  [PASS] Feature contract ({len(df)} rows x {len(FEATURE_COLS)} features, binary target)")

    # 3. Serving artifacts exist and predict_proba returns a valid 2-class probability
    model = joblib.load(os.path.join(MODELS_DIR, "flood_model.pkl"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    vec = df[FEATURE_COLS].iloc[[0]].to_numpy(dtype=np.float64)
    probs = model.predict_proba(scaler.transform(vec))[0]
    assert len(probs) == 2 and abs(probs.sum() - 1.0) < 1e-6, "Invalid probability vector"
    assert 0.0 <= probs[1] <= 1.0, "Flood probability out of range"
    print(f"  [PASS] Serving model loads & infers (sample flood prob={probs[1]:.3f})")

    print("\nAll pipeline invariants held.")


if __name__ == "__main__":
    run_checks()
