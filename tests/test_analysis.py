"""Core workflow: loading, preprocessing, grouping, model training and evaluation, plotting."""

from __future__ import annotations

import pandas as pd
import pytest

import analysis
from data_utils import QUALITY_THRESHOLD


# ------------------------------------------------------------------- loading
def test_load_data_normalises_column_names(data_dir):
    df = analysis.load_data()
    assert "volatile_acidity" in df.columns
    assert "ph" in df.columns
    assert all(c == c.lower() and " " not in c for c in df.columns)


def test_load_data_reads_semicolon_file_identically(data_dir, raw_frame):
    """The UCI original is semicolon-separated; it must load the same as the comma version."""
    comma = analysis.load_data()
    for f in data_dir.glob("*.csv"):
        f.unlink()
    raw_frame.to_csv(data_dir / "wine_semicolon.csv", sep=";", index=False)
    semicolon = analysis.load_data()
    pd.testing.assert_frame_equal(comma, semicolon)


def test_load_data_keeps_every_row(data_dir, raw_frame):
    assert len(analysis.load_data()) == len(raw_frame)


# ------------------------------------------------------------- preprocessing
def test_build_target_labels_at_threshold():
    df = pd.DataFrame({"quality": [3, 5, 6, 7, 8, 9], "alcohol": [9, 10, 11, 12, 13, 14]})
    out = analysis.build_target(df)
    expected = [1 if q >= QUALITY_THRESHOLD else 0 for q in df["quality"]]
    assert out["good"].tolist() == expected


def test_build_target_boundary_is_inclusive():
    """quality == 7 is 'good'; quality == 6 is not. Guards against an off-by-one."""
    df = pd.DataFrame({"quality": [6, 7], "alcohol": [10.0, 10.1]})
    assert analysis.build_target(df)["good"].tolist() == [0, 1]


def test_build_target_drops_duplicate_rows():
    df = pd.DataFrame({"quality": [5, 5, 7], "alcohol": [9.0, 9.0, 12.0]})
    assert len(analysis.build_target(df)) == 2


def test_build_target_does_not_modify_input():
    df = pd.DataFrame({"quality": [5, 5, 7], "alcohol": [9.0, 9.0, 12.0]})
    before = df.copy()
    analysis.build_target(df)
    pd.testing.assert_frame_equal(df, before)


def test_select_features_excludes_target_and_text_columns(clean_frame):
    data = analysis.build_target(clean_frame)
    features = analysis.select_features(data)
    assert "quality" not in features
    assert "good" not in features
    assert "type" not in features       # text column: skipped
    assert len(features) == 11          # the eleven physicochemical measurements


@pytest.mark.parametrize(
    "labels, expected",
    [
        ([0, 0, 0, 1], 0.75),   # negative majority (like the real data)
        ([1, 1, 1, 0], 0.75),   # positive majority: the case the old formula got wrong
        ([0, 1], 0.5),          # perfectly balanced
        ([0, 0, 0], 1.0),       # single class
    ],
)
def test_majority_baseline(labels, expected):
    assert analysis.majority_baseline(pd.Series(labels)) == pytest.approx(expected)


# ---------------------------------------------------------------- grouping
def test_filter_and_group_counts_add_up(clean_frame):
    grouped = analysis.filter_and_group(clean_frame)
    assert grouped["n"].sum() == len(clean_frame)


def test_filter_and_group_is_sorted_by_quality(clean_frame):
    grouped = analysis.filter_and_group(clean_frame)
    assert list(grouped.index) == sorted(grouped.index)


def test_filter_and_group_means_match_manual_calculation(clean_frame):
    grouped = analysis.filter_and_group(clean_frame)
    for score, row in grouped.iterrows():
        manual = clean_frame.loc[clean_frame["quality"] == score, "alcohol"].mean()
        assert row["mean_alcohol"] == pytest.approx(manual, abs=1e-3)


def test_filter_and_group_warns_about_small_groups(clean_frame, capsys):
    analysis.filter_and_group(clean_frame)
    assert "fewer than 50 samples" in capsys.readouterr().out


# ------------------------------------------------ model training & evaluation
@pytest.fixture
def model_results(clean_frame):
    return analysis.explore_model(clean_frame)


def test_explore_model_returns_all_parts(model_results):
    for key in ("features", "baseline", "logistic_regression", "random_forest", "importances"):
        assert key in model_results


def test_metrics_are_valid_probabilities(model_results):
    for name in ("logistic_regression", "random_forest"):
        m = model_results[name]
        assert 0.0 <= m["accuracy"] <= 1.0
        assert 0.0 <= m["roc_auc"] <= 1.0


def test_one_prediction_per_test_row(model_results):
    for name in ("logistic_regression", "random_forest"):
        preds = model_results[name]["predictions"]
        assert len(preds) == model_results["n_test"]
        assert set(preds) <= {0, 1}


def test_models_beat_random_ranking(model_results):
    """On synthetic data where alcohol drives quality, both models must beat a coin flip."""
    assert model_results["logistic_regression"]["roc_auc"] > 0.7
    assert model_results["random_forest"]["roc_auc"] > 0.7


def test_importances_are_a_valid_distribution(model_results):
    imp = model_results["importances"]
    assert len(imp) == len(model_results["features"])
    assert (imp >= 0).all()
    assert imp.sum() == pytest.approx(1.0)


def test_model_finds_the_planted_signal(model_results):
    """Alcohol is the only informative feature in the synthetic data, so it should rank first."""
    assert model_results["importances"].index[0] == "alcohol"


def test_explore_model_is_reproducible(clean_frame):
    first = analysis.explore_model(clean_frame)
    second = analysis.explore_model(clean_frame)
    assert first["random_forest"]["accuracy"] == second["random_forest"]["accuracy"]
    pd.testing.assert_series_equal(first["importances"], second["importances"])


# ------------------------------------------------------------ visualisation
def test_visualise_writes_png(clean_frame, fig_dir):
    importances = pd.Series({"alcohol": 0.5, "density": 0.3, "ph": 0.2})
    path = analysis.visualise(clean_frame, importances)
    assert path.exists()
    assert path.suffix == ".png"
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"  # real PNG, not an empty file
