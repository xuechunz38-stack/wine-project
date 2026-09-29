"""Reusable modelling steps, separated from printing and plotting.

Keep the original 80/20 split and model settings so earlier results remain
comparable. Missing/nonfinite features fail explicitly; they are never silently
filled with values learned from a test set.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data_utils import QUALITY_THRESHOLD

RANDOM_STATE = 42


def build_target(df: pd.DataFrame) -> pd.DataFrame:
    """Turn the 0-10 quality score into a binary 'good wine' label."""
    if "quality" not in df or df.empty:
        raise ValueError("A nonempty quality column is required.")
    quality = pd.to_numeric(df["quality"], errors="coerce")
    if (
        quality.isna().any()
        or not quality.between(0, 10).all()
        or (quality % 1 != 0).any()
    ):
        raise ValueError("Quality scores must be whole numbers from 0 to 10.")
    out = df.drop_duplicates().copy()
    out["quality"] = quality.loc[out.index]
    out["good"] = (out["quality"] >= QUALITY_THRESHOLD).astype(int)
    return out


def select_features(data: pd.DataFrame) -> list[str]:
    """Numeric columns other than the target. Text columns such as `type` are skipped."""
    return [
        c
        for c in data.columns
        if c not in ("quality", "good") and pd.api.types.is_numeric_dtype(data[c])
    ]


def majority_baseline(y: pd.Series) -> float:
    """Accuracy of always predicting the most common class.

    Written as max(p, 1 - p) rather than 1 - p: the latter silently assumes the
    negative class is the majority, which is true for this dataset but not in
    general (a unit test caught this on a positive-majority sample).
    """
    if y.empty or not y.isin([0, 1]).all():
        raise ValueError("Baseline labels must be a nonempty binary sequence.")
    p = float(y.mean())
    return max(p, 1.0 - p)


def evaluate(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Score a fitted classifier. Returned as a dict so tests can check the numbers."""
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, proba),
        "predictions": pred,
        "probabilities": proba,
        "report": classification_report(
            y_test, pred, target_names=["not good", "good"], zero_division=0
        ),
    }


def fit_models(X_train, y_train):
    """Fit both reproducible baselines using training data only."""
    models = {
        "logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced"),
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            n_jobs=-1,
        ),
    }
    for model in models.values():
        model.fit(X_train, y_train)
    return models


def train_and_evaluate(df):
    """Run the shared split/train/evaluate computation without output side effects."""
    data = build_target(df)
    features = select_features(data)
    if not features or not np.isfinite(data[features].to_numpy(dtype=float)).all():
        raise ValueError(
            "Numeric features must exist and contain no missing/infinite values."
        )
    y = data["good"]
    counts = y.value_counts()
    if len(counts) != 2 or counts.min() < 5:
        raise ValueError(
            "At least five unique rows in each quality class are required."
        )
    X_train, X_test, y_train, y_test = train_test_split(
        data[features], y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    models = fit_models(X_train, y_train)
    return {
        "features": features,
        "n_rows": len(data),
        "class_counts": counts.to_dict(),
        "n_test": len(y_test),
        "y_test": y_test.to_numpy(),
        "baseline": majority_baseline(y_test),
        **{name: evaluate(model, X_test, y_test) for name, model in models.items()},
        "importances": pd.Series(
            models["random_forest"].feature_importances_, index=features
        ).sort_values(ascending=False),
    }
