"""Exploratory Data Analysis: charts + machine-readable summary.

Outputs to ml/reports/:
  eda_class_balance.png, eda_distributions.png, eda_boxplots.png,
  eda_correlation.png, eda_scatter_rainfall.png, eda_summary.json
"""

import json
import logging
import os

import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ml.pipeline.config import REPORTS_DIR, FEATURE_COLS

logger = logging.getLogger(__name__)

_NAVY = "#003366"
_RED = "#b91c1c"
_GREEN = "#138808"


def _fig_path(name: str) -> str:
    return os.path.join(REPORTS_DIR, name)


def _class_palette(y: pd.Series):
    return np.where(y.values == 1, _RED, _NAVY)


def run_eda(df: pd.DataFrame) -> dict:
    os.makedirs(REPORTS_DIR, exist_ok=True)
    y = df["flood_occurred"]
    X = df[FEATURE_COLS]

    # ---------- 1. Class balance ----------
    fig, ax = plt.subplots(figsize=(5, 3.2))
    counts = y.value_counts().sort_index()
    ax.bar(["No Flood (0)", "Flood (1)"], [counts.get(0, 0), counts.get(1, 0)],
           color=[_NAVY, _RED], edgecolor="white")
    for i, v in enumerate([counts.get(0, 0), counts.get(1, 0)]):
        ax.text(i, v, str(v), ha="center", va="bottom", fontweight="bold")
    ax.set_title("Target Class Balance", fontsize=11, fontweight="bold", color=_NAVY)
    ax.set_ylabel("Records")
    fig.tight_layout()
    fig.savefig(_fig_path("eda_class_balance.png"), dpi=120)
    plt.close(fig)

    # ---------- 2. Feature distributions by class ----------
    fig, axes = plt.subplots(3, 4, figsize=(13, 7.5))
    for axi, col in zip(axes.flat, FEATURE_COLS):
        for label, mask, color in [(0, y == 0, _NAVY), (1, y == 1, _RED)]:
            axi.hist(df.loc[mask, col], bins=22, alpha=0.62, color=color, label=f"class {label}")
        axi.set_title(col, fontsize=8.5)
        axi.tick_params(labelsize=7)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Feature Distributions by Flood Class (red = flood)", fontsize=12, fontweight="bold", color=_NAVY)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(_fig_path("eda_distributions.png"), dpi=120)
    plt.close(fig)

    # ---------- 3. Boxplots by class ----------
    fig, axes = plt.subplots(3, 4, figsize=(13, 7.5))
    for axi, col in zip(axes.flat, FEATURE_COLS):
        groups = [df.loc[y == 0, col], df.loc[y == 1, col]]
        bp = axi.boxplot(groups, tick_labels=["0", "1"], patch_artist=True, widths=0.5)
        for patch, color in zip(bp["boxes"], [_NAVY, _RED]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        axi.set_title(col, fontsize=8.5)
        axi.tick_params(labelsize=7)
    fig.suptitle("Feature Spread by Flood Class", fontsize=12, fontweight="bold", color=_NAVY)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(_fig_path("eda_boxplots.png"), dpi=120)
    plt.close(fig)

    # ---------- 4. Correlation heatmap ----------
    corr = X.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(8.2, 7))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), FEATURE_COLS, rotation=90, fontsize=7)
    ax.set_yticks(range(len(corr)), FEATURE_COLS, fontsize=7)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=5.5,
                    color="white" if abs(corr.values[i, j]) > 0.55 else "black")
    fig.colorbar(im, fraction=0.046)
    ax.set_title("Feature Correlation Matrix", fontsize=11, fontweight="bold", color=_NAVY)
    fig.tight_layout()
    fig.savefig(_fig_path("eda_correlation.png"), dpi=120)
    plt.close(fig)

    # ---------- 5. Rainfall vs flood scatter ----------
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.scatter(df["rainfall_6h"], df["rainfall_24h"], c=_class_palette(y), s=32, alpha=0.65, edgecolors="white", linewidths=0.4)
    ax.axhline(df.loc[y == 1, "rainfall_24h"].median(), color=_RED, ls="--", lw=1, label="flood median (24h)")
    ax.set_xlabel("Rainfall 6h (mm)"); ax.set_ylabel("Rainfall 24h (mm)")
    ax.set_title("Rainfall vs Flood Occurrence", fontsize=11, fontweight="bold", color=_NAVY)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(_fig_path("eda_scatter_rainfall.png"), dpi=120)
    plt.close(fig)

    # ---------- 6. Machine-readable summary ----------
    n = len(df)
    n_flood = int(y.sum())
    # Top correlations with the target, and most collinear feature pairs
    target_corr = X.assign(_y=y).corr()["_y"].drop("_y").sort_values(key=abs, ascending=False)
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack()
    collinear = upper[abs(upper) > 0.85].sort_values(ascending=False).round(3).head(6)

    summary = {
        "n_records": n,
        "n_flood": n_flood,
        "n_no_flood": n - n_flood,
        "flood_ratio": round(n_flood / n, 4),
        "missing_values": int(df[FEATURE_COLS].isna().sum().sum()),
        "target_correlation": {k: round(float(v), 4) for k, v in target_corr.items()},
        "highly_collinear_pairs": {f"{a} ~ {b}": round(float(v), 3) for (a, b), v in collinear.items()},
        "describe": json.loads(
            df.groupby("flood_occurred")[FEATURE_COLS].mean().round(2).to_json(orient="index")
        ),
        "charts": [
            "eda_class_balance.png", "eda_distributions.png", "eda_boxplots.png",
            "eda_correlation.png", "eda_scatter_rainfall.png",
        ],
    }
    with open(_fig_path("eda_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("EDA complete: %d charts + summary saved to %s", len(summary["charts"]), REPORTS_DIR)
    return summary
