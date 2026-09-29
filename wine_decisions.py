"""Evaluate a fixed tasting budget, without choosing thresholds on test labels."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def shortlist_metrics(labels, probabilities, fraction: float) -> dict:
    """Rank by score and report a predeclared fraction of the held-out samples.

    Tied scores preserve input order. Round upward so a positive budget always
    selects at least one sample. A zero-positive test set has undefined recall
    and lift, represented by NaN rather than a misleading zero.
    """
    y = np.asarray(labels)
    scores = np.asarray(probabilities, dtype=float)
    if y.ndim != 1 or scores.ndim != 1 or len(y) != len(scores) or not len(y):
        raise ValueError("Labels and scores must be nonempty, equally sized vectors.")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("Labels must be binary.")
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Probabilities must be finite values between 0 and 1.")
    if not math.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError("Budget fraction must be in (0, 1].")
    k = math.ceil(len(y) * fraction)
    selected = np.argsort(-scores, kind="stable")[:k]
    found = int(y[selected].sum())
    positives = int(y.sum())
    precision = found / k
    prevalence = positives / len(y)
    return {
        "budget_fraction": fraction,
        "selected": k,
        "good_found": found,
        "total_good": positives,
        "precision": precision,
        "recall": found / positives if positives else float("nan"),
        "lift": precision / prevalence if positives else float("nan"),
        "random_expected_good": k * prevalence,
    }


def outlier_audit(df: pd.DataFrame) -> pd.DataFrame:
    """Count 1.5-IQR flags for numeric predictors; flags never delete observations."""
    numeric = df.select_dtypes(include="number").drop(
        columns=["quality", "good"], errors="ignore"
    )
    rows = []
    for name in numeric:
        values = numeric[name]
        q1, q3 = values.quantile([0.25, 0.75])
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        rows.append(
            {
                "feature": name,
                "missing": int(values.isna().sum()),
                "lower_fence": low,
                "upper_fence": high,
                "flagged": int(((values < low) | (values > high)).sum()),
            }
        )
    return pd.DataFrame(rows)


def write_decision_report(df: pd.DataFrame, results: dict, output_dir: Path) -> None:
    """Write reproducible, predeclared budget comparisons and an outlier audit."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "model": name,
            **shortlist_metrics(
                results["y_test"], results[name]["probabilities"], budget
            ),
        }
        for name in ("logistic_regression", "random_forest")
        for budget in (0.05, 0.10, 0.20, 0.30, 0.50, 1.0)
    ]
    metrics = pd.DataFrame(rows)
    metrics.to_csv(output_dir / "tasting_budget.csv", index=False, float_format="%.6f")
    outlier_audit(df).to_csv(
        output_dir / "outlier_audit.csv", index=False, float_format="%.6f"
    )
    print("\nTASTING BUDGET: fixed top 20% of held-out wines")
    print(metrics[metrics.budget_fraction == 0.20].to_string(index=False))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, group in metrics.groupby("model", sort=False):
        ax.plot(
            group.budget_fraction * 100,
            group.recall * 100,
            marker="o",
            label=name.replace("_", " "),
        )
    ax.plot(
        [0, 100],
        [0, 100],
        linestyle="--",
        color="gray",
        label="random selection (expectation)",
    )
    ax.set(
        xlabel="Tasting budget (% of held-out wines)",
        ylabel="Good wines found (%)",
        title="Finding good wines with limited tasting capacity",
        xlim=(0, 100),
        ylim=(0, 102),
    )
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "tasting_budget.png", dpi=150)
    plt.close(fig)
