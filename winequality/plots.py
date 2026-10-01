"""Figures written by analysis.py."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write PNGs without a display (CI, Docker)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import precision_score, recall_score  # noqa: E402


def plot_overview(df: pd.DataFrame, importances: pd.Series, out_path: Path) -> Path:
    """Quality distribution, alcohol by quality, and top feature importances."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    counts = df["quality"].value_counts().sort_index()
    axes[0].bar(counts.index.astype(str), counts.values, color="#7a3b3b")
    axes[0].set(title="Distribution of quality scores", xlabel="quality", ylabel="number of wines")

    scores = sorted(df["quality"].unique())
    axes[1].boxplot([df.loc[df["quality"] == s, "alcohol"] for s in scores])
    axes[1].set_xticks(range(1, len(scores) + 1), [str(s) for s in scores])
    axes[1].set(
        title="Alcohol content by quality score", xlabel="quality", ylabel="alcohol (% ABV)"
    )

    top = importances.head(8).sort_values()
    axes[2].barh(top.index, top.values, color="#4a6b8a")
    axes[2].set(title="Top features (random forest)", xlabel="importance")

    return _save(fig, out_path)


def plot_threshold_tradeoff(
    y_true, proba, groups, chosen: float, target_recall: float, out_path: Path
) -> Path:
    """Precision and recall of the forest across cut-offs, overall and by wine type."""
    y_true, proba, groups = np.asarray(y_true), np.asarray(proba), np.asarray(groups)
    grid = np.round(np.arange(0.05, 0.86, 0.01), 2)  # above ~0.85 almost nothing is flagged
    fig, ax = plt.subplots(figsize=(8.5, 4.8))

    styles = {"all": ("#222222", "-"), "red": ("#9b2335", "--"), "white": ("#c9a227", ":")}
    for name, (color, ls) in styles.items():
        mask = np.ones_like(y_true, bool) if name == "all" else groups == name
        if not mask.any():
            continue
        rec = [recall_score(y_true[mask], proba[mask] >= t, zero_division=0) for t in grid]
        prec = [precision_score(y_true[mask], proba[mask] >= t, zero_division=np.nan) for t in grid]
        ax.plot(grid, rec, color=color, ls=ls, lw=2, label=f"recall · {name}")
        ax.plot(grid, prec, color=color, ls=ls, lw=1, alpha=0.55, label=f"precision · {name}")

    ax.axvline(0.5, color="#999999", lw=1)
    ax.axvline(chosen, color="#2a6f4e", lw=1.5)
    ax.axhline(target_recall, color="#2a6f4e", lw=0.8, ls="--")
    ax.text(0.505, 0.03, "default 0.5", color="#777777", fontsize=8)
    ax.text(chosen + 0.005, 0.03, f"chosen {chosen:.2f}", color="#2a6f4e", fontsize=8)
    ax.set(
        title="Random forest on the test set: precision/recall vs cut-off",
        xlabel="flag wine as 'good' if P(good) ≥ cut-off",
        ylim=(0, 1.02),
    )
    ax.legend(fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.15), frameon=False)
    return _save(fig, out_path)


def _save(fig, out_path: Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path
