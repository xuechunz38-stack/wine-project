"""Validation, target construction and outlier auditing."""

from __future__ import annotations

import numpy as np
import pandas as pd

from winequality.config import FEATURE_COLUMNS, QUALITY_THRESHOLD


def validate_wine_frame(df: pd.DataFrame) -> None:
    """Fail early, with a readable message, if the data cannot be modelled.

    Checks for an empty table, missing required columns, and missing or
    infinite values in the columns the models use.
    """
    if df.empty:
        raise ValueError("The wine table is empty.")
    required = FEATURE_COLUMNS + ["quality"]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required column(s): {', '.join(missing_cols)}")
    values = df[required].apply(pd.to_numeric, errors="coerce")
    bad = values.isna() | np.isinf(values)
    if bad.any().any():
        cols = bad.any()[bad.any()].index.tolist()
        raise ValueError(f"Missing, non-numeric or infinite values in: {', '.join(cols)}")


def build_target(df: pd.DataFrame, threshold: int = QUALITY_THRESHOLD) -> pd.DataFrame:
    """Drop exact duplicates and add a binary `good` label (quality >= threshold).

    Duplicates are dropped *before* any train/test split: an identical row in
    both sets would inflate test scores without the model learning anything.
    The input frame is not modified.
    """
    out = df.drop_duplicates().copy()
    out["good"] = (out["quality"] >= threshold).astype(int)
    return out


def select_features(data: pd.DataFrame) -> list[str]:
    """Numeric columns other than the target. Text columns such as `type` are skipped."""
    return [
        c
        for c in data.columns
        if c not in ("quality", "good") and pd.api.types.is_numeric_dtype(data[c])
    ]


def iqr_outlier_report(df: pd.DataFrame, columns: list[str], k: float = 1.5) -> pd.DataFrame:
    """Count values outside [Q1 - k*IQR, Q3 + k*IQR] for each column.

    This *flags* outliers for documentation; it does not delete them. A column
    whose IQR is zero (e.g. a constant column) gets zero flagged values rather
    than flagging everything that differs from the constant by a hair.
    """
    rows = []
    for col in columns:
        series = df[col].dropna()
        q1, q3 = series.quantile([0.25, 0.75])
        iqr = q3 - q1
        low, high = q1 - k * iqr, q3 + k * iqr
        n_out = 0 if iqr == 0 else int(((series < low) | (series > high)).sum())
        rows.append(
            {
                "column": col,
                "lower_fence": round(float(low), 4),
                "upper_fence": round(float(high), 4),
                "n_outliers": n_out,
                "share": round(n_out / len(series), 4) if len(series) else 0.0,
                "max": float(series.max()) if len(series) else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("n_outliers", ascending=False, ignore_index=True)
