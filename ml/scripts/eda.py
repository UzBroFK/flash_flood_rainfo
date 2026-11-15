"""EDA-only entrypoint: generates charts + eda_summary.json without training.

Run:  python ml/scripts/eda.py
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import logging
from ml.pipeline.data import load_and_merge_datasets, compute_physics_features
from ml.pipeline.eda import run_eda

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s %(message)s")
    df_raw = load_and_merge_datasets()
    df_feat = compute_physics_features(df_raw)
    summary = run_eda(df_feat)
    print(f"EDA complete -> {len(summary['charts'])} charts + eda_summary.json in ml/reports/")
