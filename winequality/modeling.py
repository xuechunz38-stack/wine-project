"""Model training, evaluation and decision-threshold selection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from winequality.config import RANDOM_STATE, TEST_SIZE


@dataclass
class Split:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def split_data(data: pd.DataFrame, features: list[str], target: str = "good") -> Split:
    """Stratified train/test split. Needs at least two examples of each class."""
    counts = data[target].value_counts()
    if len(counts) < 2 or counts.min() < 2:
        raise ValueError(
            "Need at least two 'good' and two 'not good' wines to split the data; "
            f"got class counts {counts.to_dict()}."
        )
    X_train, X_test, y_train, y_test = train_test_split(
        data[features],
        data[target],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=data[target],
    )
    return Split(X_train, X_test, y_train, y_test)


def make_logistic_regression():
    """Scaled logistic regression; the scaler is fitted on training data only."""
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced"),
    )


def make_random_forest(n_estimators: int = 300):
    return RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1,
    )


def majority_baseline(y: pd.Series) -> float:
    """Accuracy of always predicting the most common class, i.e. max(p, 1 - p)."""
    p = float(pd.Series(y).mean())
    return max(p, 1.0 - p)


def evaluate(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Score a fitted classifier at the default 0.5 threshold."""
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, proba),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "predictions": pred,
        "probabilities": proba,
        "report": classification_report(
            y_test, pred, target_names=["not good", "good"], zero_division=0
        ),
    }


def out_of_fold_probabilities(model, X: pd.DataFrame, y: pd.Series, folds: int = 5) -> np.ndarray:
    """Cross-validated P(good) for every *training* row.

    Each row is scored by a model that never saw it, so a threshold chosen on
    these probabilities is not tuned on the test set.
    """
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    return cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]


def choose_threshold(y_true, proba, target_recall: float, step: float = 0.01) -> float:
    """Highest probability cut-off whose recall reaches `target_recall`.

    The highest such cut-off flags the fewest wines, i.e. the best precision
    for the recall the team asked for.
    """
    if not 0 < target_recall <= 1:
        raise ValueError("target_recall must be in (0, 1].")
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    if y_true.sum() == 0:
        raise ValueError("Cannot choose a recall threshold without any positive examples.")
    for threshold in np.round(np.arange(1.0, 0.0, -step), 4):
        if recall_score(y_true, proba >= threshold) >= target_recall:
            return float(threshold)
    return 0.0  # flag everything: recall is then 1.0


def metrics_at_threshold(y_true, proba, threshold: float) -> dict:
    """Precision, recall and workload when wines with P(good) >= threshold are flagged."""
    y_true = np.asarray(y_true)
    flagged = np.asarray(proba) >= threshold
    return {
        "threshold": threshold,
        "n_wines": int(len(y_true)),
        "n_good": int(y_true.sum()),
        "n_flagged": int(flagged.sum()),
        "good_found": int((flagged & (y_true == 1)).sum()),
        "precision": precision_score(y_true, flagged, zero_division=0),
        "recall": recall_score(y_true, flagged, zero_division=0),
    }


def metrics_by_group(y_true, proba, groups, threshold: float) -> pd.DataFrame:
    """`metrics_at_threshold` plus ROC-AUC for each group (e.g. red vs white).

    ROC-AUC is NaN for a group that contains only one class, instead of raising.
    """
    frame = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(proba), "g": np.asarray(groups)})
    rows = []
    for name, part in frame.groupby("g", sort=True):
        row = {"group": name, **metrics_at_threshold(part["y"], part["p"], threshold)}
        row["roc_auc"] = roc_auc_score(part["y"], part["p"]) if part["y"].nunique() == 2 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)
