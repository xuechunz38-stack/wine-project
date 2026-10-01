"""Model helpers: baseline, splitting, evaluation and threshold selection."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from winequality import cleaning, modeling


@pytest.mark.parametrize(
    "labels, expected",
    [
        ([0, 0, 0, 1], 0.75),  # negative majority (like the real data)
        ([1, 1, 1, 0], 0.75),  # positive majority: 1 - p would be wrong here
        ([0, 1], 0.5),
        ([0, 0, 0], 1.0),  # single class
    ],
)
def test_majority_baseline(labels, expected):
    assert modeling.majority_baseline(pd.Series(labels)) == pytest.approx(expected)


# ------------------------------------------------------------------ split
def test_split_is_stratified_and_sized(clean_frame):
    data = cleaning.build_target(clean_frame)
    split = modeling.split_data(data, cleaning.select_features(data))
    assert len(split.X_test) == pytest.approx(0.2 * len(data), abs=1)
    assert split.y_test.mean() == pytest.approx(data["good"].mean(), abs=0.03)
    assert not set(split.X_train.index) & set(split.X_test.index)


@pytest.mark.parametrize("labels", [[0] * 20, [0] * 19 + [1]])
def test_split_rejects_data_without_two_examples_per_class(labels):
    data = pd.DataFrame({"x": range(len(labels)), "good": labels})
    with pytest.raises(ValueError, match="at least two"):
        modeling.split_data(data, ["x"])


# --------------------------------------------------------- choose_threshold
def test_choose_threshold_reaches_target_recall():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.6, 0.4, 0.5, 0.8, 0.9])
    t = modeling.choose_threshold(y, p, target_recall=0.75)
    assert t == pytest.approx(0.5)  # flags 0.5, 0.6, 0.8, 0.9 -> 3 of 4 positives
    assert modeling.metrics_at_threshold(y, p, t)["recall"] >= 0.75


def test_choose_threshold_prefers_the_highest_qualifying_cutoff():
    y = np.array([0, 1, 1])
    p = np.array([0.1, 0.7, 0.8])
    assert modeling.choose_threshold(y, p, target_recall=1.0) == pytest.approx(0.7)


def test_choose_threshold_full_recall_with_tied_scores():
    y = np.array([0, 1, 0, 1])
    p = np.array([0.3, 0.3, 0.3, 0.3])
    t = modeling.choose_threshold(y, p, target_recall=1.0)
    assert modeling.metrics_at_threshold(y, p, t)["recall"] == 1.0


@pytest.mark.parametrize("bad_target", [0, -0.1, 1.5])
def test_choose_threshold_rejects_invalid_target(bad_target):
    with pytest.raises(ValueError, match="target_recall"):
        modeling.choose_threshold([0, 1], [0.2, 0.8], bad_target)


def test_choose_threshold_rejects_no_positive_examples():
    with pytest.raises(ValueError, match="positive"):
        modeling.choose_threshold([0, 0, 0], [0.2, 0.5, 0.8], 0.7)


# ----------------------------------------------------- metrics_at_threshold
def test_metrics_at_threshold_counts():
    y = np.array([1, 1, 0, 0, 1])
    p = np.array([0.9, 0.2, 0.8, 0.1, 0.6])
    m = modeling.metrics_at_threshold(y, p, 0.5)
    assert (m["n_flagged"], m["good_found"], m["n_good"]) == (3, 2, 3)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)


def test_metrics_at_threshold_when_nothing_is_flagged():
    m = modeling.metrics_at_threshold([1, 0], [0.2, 0.1], 0.9)
    assert m["n_flagged"] == 0 and m["precision"] == 0 and m["recall"] == 0


# --------------------------------------------------------- metrics_by_group
def test_metrics_by_group_splits_correctly():
    y = [1, 0, 1, 0, 1, 0]
    p = [0.9, 0.1, 0.8, 0.7, 0.3, 0.2]
    g = ["red", "red", "white", "white", "white", "white"]
    out = modeling.metrics_by_group(y, p, g, 0.5).set_index("group")
    assert out.loc["red", "n_wines"] == 2 and out.loc["white", "n_wines"] == 4
    assert out.loc["red", "recall"] == 1.0
    assert out.loc["white", "recall"] == 0.5


def test_metrics_by_group_auc_is_nan_for_single_class_group():
    out = modeling.metrics_by_group([0, 0, 1, 0], [0.1, 0.2, 0.9, 0.3], ["a", "a", "b", "b"], 0.5)
    auc = out.set_index("group")["roc_auc"]
    assert np.isnan(auc["a"]) and auc["b"] == 1.0


# ---------------------------------------------------------- out-of-fold
def test_out_of_fold_probabilities_cover_every_row(clean_frame):
    data = cleaning.build_target(clean_frame)
    features = cleaning.select_features(data)
    oof = modeling.out_of_fold_probabilities(
        modeling.make_random_forest(n_estimators=30), data[features], data["good"]
    )
    assert oof.shape == (len(data),)
    assert ((oof >= 0) & (oof <= 1)).all()
