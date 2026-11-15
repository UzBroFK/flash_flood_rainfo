"""Model training, evaluation, ensembling, and artifact export.

Trains a baseline + strong learners (RandomForest, GradientBoosting, LightGBM,
XGBoost), evaluates each with the full metric suite under stratified CV, builds
a soft-voting ensemble of the strongest diverse members, and exports the best
serving model for the FastAPI backend.
"""

import json
import logging
import os
import platform
from collections import OrderedDict
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
import lightgbm
import xgboost
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, fbeta_score, roc_auc_score, average_precision_score,
    matthews_corrcoef, cohen_kappa_score, log_loss, brier_score_loss,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_validate
from sklearn.preprocessing import StandardScaler

from ml.pipeline.config import (
    SEED, TEST_SIZE, CV_SPLITS, FEATURE_COLS, REPORTS_DIR, MODELS_DIR,
    BACKEND_MODELS_DIR, build_model_registry, ENSEMBLE_MEMBERS, CV_SCORING,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------- helpers

def _round(v, n=4):
    return round(float(v), n)


def _safe_auc(fn, y_true, y_prob):
    """AUC metrics need both classes present; degrade gracefully instead of crashing."""
    try:
        return _round(fn(y_true, y_prob))
    except ValueError:
        return None


def _metric_suite(y_true, y_prob, threshold: float = 0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "accuracy": _round(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": _round(balanced_accuracy_score(y_true, y_pred)),
        "precision": _round(precision_score(y_true, y_pred, zero_division=0)),
        "recall": _round(recall_score(y_true, y_pred, zero_division=0)),
        "f1": _round(f1_score(y_true, y_pred, zero_division=0)),
        "f2": _round(fbeta_score(y_true, y_pred, beta=2, zero_division=0)),
        "roc_auc": _safe_auc(roc_auc_score, y_true, y_prob),
        "pr_auc": _safe_auc(average_precision_score, y_true, y_prob),
        "mcc": _round(matthews_corrcoef(y_true, y_pred)),
        "cohen_kappa": _round(cohen_kappa_score(y_true, y_pred)),
        "log_loss": _round(log_loss(y_true, np.clip(y_prob, 1e-6, 1 - 1e-6))),
        "brier_score": _round(brier_score_loss(y_true, y_prob)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def _curve_points(y_true, y_prob, max_points=60):
    """Downsample ROC/PR curves so they can ship inside metrics.json for the UI."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    idx_r = np.unique(np.linspace(0, len(fpr) - 1, min(max_points, len(fpr))).astype(int))
    idx_p = np.unique(np.linspace(0, len(rec) - 1, min(max_points, len(rec))).astype(int))
    return {
        "roc": {"fpr": np.round(fpr[idx_r], 4).tolist(), "tpr": np.round(tpr[idx_r], 4).tolist()},
        "pr": {"recall": np.round(rec[idx_p], 4).tolist(), "precision": np.round(prec[idx_p], 4).tolist()},
    }


def _best_threshold(y_true, y_prob):
    """Threshold that maximises F1 — the early-warning operating point."""
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-9, None)
    return _round(float(thr[np.argmax(f1[:-1])]), 3)


def _importances(model, feature_cols):
    if hasattr(model, "feature_importances_"):
        vals = np.asarray(model.feature_importances_, dtype=np.float64)
    elif hasattr(model, "coef_"):
        vals = np.abs(np.asarray(model.coef_[0], dtype=np.float64))
    else:
        return []
    total = vals.sum()
    if total > 0:
        vals = vals / total  # normalize to sum-to-1 so every model is comparable
    pairs = sorted(zip(feature_cols, vals), key=lambda p: p[1], reverse=True)
    return [{"feature": k, "importance": _round(v)} for k, v in pairs]


# ---------------------------------------------------------------- training

def train_and_evaluate(df_feat: pd.DataFrame) -> dict:
    X = df_feat[FEATURE_COLS].to_numpy(dtype=np.float64)
    y = df_feat["flood_occurred"].to_numpy(dtype=int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y,
    )
    logger.info("Split: %d train / %d test (stratified)", len(X_train), len(X_test))

    scaler = StandardScaler().fit(X_train)
    Xtr, Xte = scaler.transform(X_train), scaler.transform(X_test)

    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=SEED)
    registry = build_model_registry()
    results: "OrderedDict[str, dict]" = OrderedDict()

    def evaluate(name: str, make_model, members=None) -> dict:
        model = make_model()
        cv_res = cross_validate(model, Xtr, y_train, cv=cv, scoring=CV_SCORING, n_jobs=-1)
        model.fit(Xtr, y_train)
        y_prob = model.predict_proba(Xte)[:, 1]

        test = _metric_suite(y_test, y_prob)
        f1_threshold = _best_threshold(y_train, model.predict_proba(Xtr)[:, 1])
        test_at_opt = _metric_suite(y_test, y_prob, threshold=f1_threshold)

        cv_summary = {}
        for m in CV_SCORING:
            scores = cv_res[f"test_{m}"]
            cv_summary[m] = {"mean": _round(scores.mean()), "std": _round(scores.std(), 3)}

        entry = {
            "name": name,
            "members": members or [name],
            "test_metrics": test,
            "test_metrics_at_best_f1_threshold": test_at_opt,
            "best_f1_threshold": f1_threshold,
            f"cv_{CV_SPLITS}fold": cv_summary,
            "curves": _curve_points(y_test, y_prob),
        }
        results[name] = entry
        logger.info("%-18s test: acc=%.3f recall=%.3f f1=%.3f roc_auc=%.3f pr_auc=%.3f",
                    name, test["accuracy"], test["recall"], test["f1"],
                    test["roc_auc"] or -1, test["pr_auc"] or -1)
        return entry

    # --- Individual models ---
    for name, factory in registry.items():
        evaluate(name, factory)

    # --- Selection key: CV F1 first, then recall, then AUCs, then prefer the ensemble ---
    def selection_key(entry):
        cv = entry[f"cv_{CV_SPLITS}fold"]
        t = entry["test_metrics"]
        return (
            cv["f1"]["mean"], cv["recall"]["mean"],
            t["roc_auc"] or 0.0, t["pr_auc"] or 0.0,
            entry["name"] == "Ensemble",
        )

    # --- Ensemble: soft-vote the TOP 3 single models (ties broken by AUC) ---
    top_singles = sorted(
        (e for e in results.values() if e["name"] != "Ensemble"),
        key=selection_key, reverse=True,
    )[:3]
    member_names = [e["name"] for e in top_singles]

    def make_ensemble():
        estimators = [(n.lower(), registry[n]()) for n in member_names]
        return VotingClassifier(estimators=estimators, voting="soft", n_jobs=-1)

    evaluate("Ensemble", make_ensemble, members=member_names)

    # --- Model selection: recall-weighted CV score (early warning: misses are costly) ---
    def selection_score(entry):
        cv = entry[f"cv_{CV_SPLITS}fold"]
        return round(0.6 * cv["f1"]["mean"] + 0.4 * cv["recall"]["mean"], 4)

    ranked = sorted(results.items(), key=lambda kv: selection_key(kv[1]), reverse=True)

    # Serving model: the ensemble of the best members (variance reduction at
    # deployment); full per-model comparison above stays available for review.
    best_entry = results["Ensemble"]
    best_name = best_entry["name"]

    # Per-model importances; ensemble = mean of its members
    for name, entry in results.items():
        if name == "Ensemble":
            member_imps = [_importances(registry[n]().fit(Xtr, y_train), FEATURE_COLS) for n in entry["members"]]
            agg = {}
            for imp in member_imps:
                for item in imp:
                    agg[item["feature"]] = agg.get(item["feature"], 0.0) + item["importance"]
            entry["feature_importances"] = [
                {"feature": k, "importance": _round(v / len(member_imps))}
                for k, v in sorted(agg.items(), key=lambda p: p[1], reverse=True)
            ]
        else:
            fitted = registry[name]().fit(Xtr, y_train)
            entry["feature_importances"] = _importances(fitted, FEATURE_COLS)

    comparison = [
        {
            "model": name,
            "selection_score": selection_score(entry),
            "accuracy": entry["test_metrics"]["accuracy"],
            "precision": entry["test_metrics"]["precision"],
            "recall": entry["test_metrics"]["recall"],
            "f1": entry["test_metrics"]["f1"],
            "roc_auc": entry["test_metrics"]["roc_auc"],
            "pr_auc": entry["test_metrics"]["pr_auc"],
            "mcc": entry["test_metrics"]["mcc"],
        }
        for name, entry in ranked
    ]

    metrics = {
        "generated_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "pipeline": "ml.pipeline (RAINFO reproducible training pipeline)",
        "serving_model": {
            "name": best_name,
            "type": "soft_voting_ensemble" if best_name == "Ensemble" else "single_model",
            "members": best_entry["members"],
            "best_f1_threshold": best_entry["best_f1_threshold"],
        },
        "data": {
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "features": FEATURE_COLS,
        },
        "environment": {
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "lightgbm": lightgbm.__version__,
            "numpy": np.__version__,
            "seed": SEED,
        },
        "comparison": comparison,
        "models": results,
        "feature_importances": best_entry["feature_importances"],
    }
    return {"metrics": metrics, "scaler": scaler, "model": None, "best_name": best_name}


def build_serving_model(metrics: dict):
    """Instantiate + fit the selected serving model (ensemble or single)."""
    registry = build_model_registry()
    if metrics["serving_model"]["type"] == "soft_voting_ensemble":
        estimators = [(n.lower(), registry[n]()) for n in metrics["serving_model"]["members"]]
        return VotingClassifier(estimators=estimators, voting="soft", n_jobs=-1)
    return registry[metrics["serving_model"]["name"]]()


def fit_serving_model(df_feat: pd.DataFrame, model):
    """Refit the serving model on the FULL dataset (train+test) for final deployment."""
    X = df_feat[FEATURE_COLS].to_numpy(dtype=np.float64)
    y = df_feat["flood_occurred"].to_numpy(dtype=int)
    scaler = StandardScaler().fit(X)
    model.fit(scaler.transform(X), y)
    return model, scaler


def save_artifacts(model, scaler, metrics: dict):
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(BACKEND_MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    joblib.dump(model, os.path.join(MODELS_DIR, "flood_model.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
    with open(os.path.join(MODELS_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Backend-serving copies (same bytes, two formats for compatibility)
    joblib.dump(model, os.path.join(BACKEND_MODELS_DIR, "flood_model.joblib"))
    joblib.dump(scaler, os.path.join(BACKEND_MODELS_DIR, "scaler.joblib"))
    joblib.dump(model, os.path.join(BACKEND_MODELS_DIR, "flood_model.pkl"))
    joblib.dump(scaler, os.path.join(BACKEND_MODELS_DIR, "scaler.pkl"))

    with open(os.path.join(os.path.dirname(MODELS_DIR), "ml", "model_report.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(REPORTS_DIR, "training_report.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Artifacts saved: %s, %s, ml/model_report.json", MODELS_DIR, BACKEND_MODELS_DIR)


def print_report(metrics: dict) -> None:
    print("\n" + "=" * 76)
    print(f"MODEL COMPARISON  (serving: {metrics['serving_model']['name']})")
    print("=" * 76)
    hdr = f"{'Model':<18}{'Acc':>7}{'Prec':>7}{'Rec':>7}{'F1':>7}{'ROC-AUC':>9}{'PR-AUC':>8}{'MCC':>7}"
    print(hdr)
    print("-" * len(hdr))
    for row in metrics["comparison"]:
        fmt = lambda v: f"{v:.3f}" if v is not None else "  n/a"
        print(f"{row['model']:<18}{fmt(row['accuracy']):>7}{fmt(row['precision']):>7}"
              f"{fmt(row['recall']):>7}{fmt(row['f1']):>7}{fmt(row['roc_auc']):>9}"
              f"{fmt(row['pr_auc']):>8}{fmt(row['mcc']):>7}")
    best = metrics["models"][metrics["serving_model"]["name"]]
    cm = best["test_metrics"]["confusion_matrix"]
    print(f"\nServing model confusion matrix [[TN FP][FN TP]]: {cm}")
    print(f"Recall (floods caught): {best['test_metrics']['recall']:.1%}   "
          f"Best-F1 operating threshold: {best['best_f1_threshold']}")
    print(f"Feature importances (top 5): "
          + ", ".join(f"{i['feature']} ({i['importance']:.2f})" for i in metrics["feature_importances"][:5]))
