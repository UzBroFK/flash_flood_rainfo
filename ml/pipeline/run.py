"""RAINFO reproducible ML pipeline — end-to-end orchestrator.

Usage:
    python -m ml.pipeline.run            # EDA + training + artifacts
    python -m ml.pipeline.run --no-eda   # training only (faster re-runs)
"""

import argparse
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s %(name)s: %(message)s")
logger = logging.getLogger("ml.pipeline")


def main() -> int:
    ap = argparse.ArgumentParser(description="RAINFO ML training pipeline")
    ap.add_argument("--no-eda", action="store_true", help="Skip EDA chart generation")
    args = ap.parse_args()

    import pandas as pd  # noqa: F401  (ensures pandas init before heavy imports on some envs)
    from ml.pipeline.data import load_and_merge_datasets, compute_physics_features
    from ml.pipeline.train import train_and_evaluate, build_serving_model, fit_serving_model, save_artifacts, print_report

    print("=" * 76)
    print("RAINFO ML PIPELINE  —  flash flood early warning model training")
    print("=" * 76)

    print("\n[1/4] Loading & merging datasets...")
    df_raw = load_and_merge_datasets()
    print(f"      {len(df_raw)} records, floods: {(df_raw['flood_occurred'] == 1).sum()}")

    print("[2/4] Engineering physics features (SCS-CN, API, Kirpich)...")
    df_feat = compute_physics_features(df_raw)
    print(f"      {len(df_feat)} clean rows, 12 features -> ml/data/processed/features.csv")

    if not args.no_eda:
        print("[3/4] Running EDA...")
        from ml.pipeline.eda import run_eda
        summary = run_eda(df_feat)
        print(f"      class balance: {summary['n_flood']} floods / {summary['n_no_flood']} non-floods "
              f"({summary['flood_ratio']:.1%} positive),  charts -> ml/reports/")
    else:
        print("[3/4] EDA skipped (--no-eda)")

    print("[4/4] Training & evaluating models...")
    result = train_and_evaluate(df_feat)
    print_report(result["metrics"])

    model, scaler = fit_serving_model(df_feat, build_serving_model(result["metrics"]))

    save_artifacts(model, scaler, result["metrics"])
    print("\n[SUCCESS] Pipeline complete. Artifacts:")
    print("          ml_models/flood_model.pkl + scaler.pkl + metrics.json  (serving)")
    print("          ml/model_report.json, ml/reports/  (reports & EDA charts)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
