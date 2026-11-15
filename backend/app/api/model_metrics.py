"""Serves the training pipeline's metrics report and EDA summary to the UIs."""

import json
import logging
import os
from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)
router = APIRouter()

_MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "ml_models"))
_REPORTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "ml", "reports"))


def _load_json(path: str) -> dict:
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Report not found — run the training pipeline first: python -m ml.pipeline.run")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@router.get("/model/metrics")
async def get_model_metrics() -> dict:
    """Full training metrics report: model comparison, CV scores, curves, importances."""
    metrics = _load_json(os.path.join(_MODELS_DIR, "metrics.json"))
    return {"success": True, "metrics": metrics}


@router.get("/model/eda")
async def get_eda_summary() -> dict:
    """EDA summary statistics and chart filenames (charts served at /model-reports/)."""
    summary = _load_json(os.path.join(_REPORTS_DIR, "eda_summary.json"))
    return {"success": True, "eda": summary}
