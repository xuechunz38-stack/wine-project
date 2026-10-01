"""Validation, target construction, feature selection and the outlier audit."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from winequality import cleaning
from winequality.config import FEATURE_COLUMNS, QUALITY_THRESHOLD


# ---------------------------------------------------------------- validation
def test_validate_accepts_clean_frame(clean_frame):
    cleaning.validate_wine_frame(clean_frame)  # no exception


def test_validate_rejects_empty_frame():
    with pytest.raises(ValueError, match="empty"):
        cleaning.validate_wine_frame(pd.DataFrame())


def test_validate_names_the_missing_column(clean_frame):
    with pytest.raises(ValueError, match="alcohol"):
        cleaning.validate_wine_frame(clean_frame.drop(columns="alcohol"))


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, "n/a"])
def test_validate_rejects_missing_infinite_or_text_values(clean_frame, bad_value):
    frame = clean_frame.astype({"chlorides": object})
    frame.loc[frame.index[3], "chlorides"] = bad_value
    with pytest.raises(ValueError, match="chlorides"):
        cleaning.validate_wine_frame(frame)


def test_validate_ignores_problems_in_unused_text_column(clean_frame):
    frame = clean_frame.copy()
    frame.loc[frame.index[0], "type"] = None  # `type` is not a model input
    cleaning.validate_wine_frame(frame)


# -------------------------------------------------------------- build_target
def test_build_target_labels_at_threshold():
    df = pd.DataFrame({"quality": [3, 5, 6, 7, 8, 9], "alcohol": [9, 10, 11, 12, 13, 14]})
    expected = [int(q >= QUALITY_THRESHOLD) for q in df["quality"]]
    assert cleaning.build_target(df)["good"].tolist() == expected


def test_build_target_boundary_is_inclusive():
    """quality == 7 is 'good'; quality == 6 is not. Guards against an off-by-one."""
    df = pd.DataFrame({"quality": [6, 7], "alcohol": [10.0, 10.1]})
    assert cleaning.build_target(df)["good"].tolist() == [0, 1]


def test_build_target_drops_duplicates_and_keeps_input_unchanged():
    df = pd.DataFrame({"quality": [5, 5, 7], "alcohol": [9.0, 9.0, 12.0]})
    before = df.copy()
    assert len(cleaning.build_target(df)) == 2
    pd.testing.assert_frame_equal(df, before)


def test_build_target_keeps_original_index_for_joining_back(clean_frame):
    out = cleaning.build_target(clean_frame)
    assert out.index.isin(clean_frame.index).all()


def test_select_features_excludes_target_and_text_columns(clean_frame):
    features = cleaning.select_features(cleaning.build_target(clean_frame))
    assert sorted(features) == sorted(FEATURE_COLUMNS)


# ------------------------------------------------------------ outlier audit
def test_iqr_report_flags_a_planted_outlier():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0, 100.0]})
    report = cleaning.iqr_outlier_report(df, ["x"])
    assert report.loc[0, "n_outliers"] == 1
    assert report.loc[0, "max"] == 100.0


def test_iqr_report_flags_nothing_in_uniform_data():
    df = pd.DataFrame({"x": np.arange(100, dtype=float)})
    assert cleaning.iqr_outlier_report(df, ["x"]).loc[0, "n_outliers"] == 0


def test_iqr_report_handles_constant_column():
    """IQR = 0 would otherwise flag every value that differs at all."""
    df = pd.DataFrame({"x": [5.0] * 20 + [5.1]})
    assert cleaning.iqr_outlier_report(df, ["x"]).loc[0, "n_outliers"] == 0


def test_iqr_report_does_not_remove_rows(clean_frame):
    before = len(clean_frame)
    cleaning.iqr_outlier_report(clean_frame, FEATURE_COLUMNS)
    assert len(clean_frame) == before


def test_iqr_report_is_sorted_by_count(clean_frame):
    report = cleaning.iqr_outlier_report(clean_frame, FEATURE_COLUMNS)
    assert report["n_outliers"].is_monotonic_decreasing
    assert set(report["column"]) == set(FEATURE_COLUMNS)
