"""The analysis workflow: grouping, model comparison and threshold tuning."""

from __future__ import annotations

import pandas as pd
import pytest

import analysis
from winequality import cleaning, exploration, modeling


# ---------------------------------------------------------------- grouping
def test_summary_counts_add_up(clean_frame):
    assert exploration.summarise_by_quality(clean_frame)["n"].sum() == len(clean_frame)


def test_summary_is_sorted_and_matches_manual_means(clean_frame):
    grouped = exploration.summarise_by_quality(clean_frame)
    assert list(grouped.index) == sorted(grouped.index)
    for score, row in grouped.iterrows():
        manual = clean_frame.loc[clean_frame["quality"] == score, "alcohol"].mean()
        assert row["mean_alcohol"] == pytest.approx(manual, abs=1e-3)


def test_filter_and_group_warns_about_small_groups(clean_frame, capsys):
    exploration.filter_and_group(clean_frame)
    assert "fewer than 50" in capsys.readouterr().out


# ------------------------------------------------------- model comparison
@pytest.fixture
def results(clean_frame):
    data = cleaning.build_target(clean_frame)
    return analysis.compare_models(data, cleaning.select_features(data))


def test_compare_models_returns_valid_metrics(results):
    for name in ("logistic_regression", "random_forest"):
        m = results[name]
        assert 0 <= m["accuracy"] <= 1 and 0 <= m["roc_auc"] <= 1
        assert len(m["predictions"]) == len(results["split"].y_test)
        assert set(m["predictions"]) <= {0, 1}


def test_models_find_the_planted_signal(results):
    """Alcohol is the only informative feature in the synthetic data."""
    assert results["logistic_regression"]["roc_auc"] > 0.7
    assert results["random_forest"]["roc_auc"] > 0.7
    assert results["importances"].index[0] == "alcohol"
    assert results["importances"].sum() == pytest.approx(1.0)


def test_compare_models_is_reproducible(clean_frame, results):
    data = cleaning.build_target(clean_frame)
    again = analysis.compare_models(data, cleaning.select_features(data))
    pd.testing.assert_series_equal(results["importances"], again["importances"])


# --------------------------------------------------------- threshold tuning
def test_tune_threshold_uses_training_data_only(results, monkeypatch):
    """The cut-off must be chosen from training labels; test labels must never be passed in."""
    test_index = set(results["split"].y_test.index)
    seen = []
    real = modeling.choose_threshold

    def spy(y_true, proba, target_recall, step=0.01):
        seen.append(set(pd.Series(y_true).index))
        return real(y_true, proba, target_recall, step)

    monkeypatch.setattr(modeling, "choose_threshold", spy)
    analysis.tune_threshold(results)
    assert seen and not (seen[0] & test_index)


def test_tune_threshold_meets_target_on_training_folds(results):
    tuned = analysis.tune_threshold(results, target_recall=0.7)
    assert tuned["train_cv"]["recall"] >= 0.7
    assert 0 < tuned["threshold"] <= 1
    # lowering the cut-off below 0.5 can only flag more wines
    if tuned["threshold"] < 0.5:
        assert tuned["test"]["n_flagged"] >= tuned["test_default"]["n_flagged"]
