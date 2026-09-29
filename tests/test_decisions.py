"""Hand-calculated policy outcomes and invalid-data protection."""

import numpy as np
import pandas as pd
import pytest

from wine_decisions import outlier_audit, shortlist_metrics
from wine_models import build_target, majority_baseline, train_and_evaluate


def test_shortlist_matches_hand_calculation():
    result = shortlist_metrics([1, 0, 1, 0, 0], [0.9, 0.8, 0.7, 0.2, 0.1], 0.4)
    assert result["selected"] == 2
    assert result["good_found"] == 1
    assert result["precision"] == result["recall"] == 0.5
    assert result["lift"] == 1.25
    assert result["random_expected_good"] == 0.8


def test_ties_are_stable_and_small_budget_rounds_up():
    result = shortlist_metrics([0, 1, 1], [0.5, 0.5, 0.5], 0.01)
    assert result["selected"] == 1
    assert result["good_found"] == 0


def test_full_budget_recovers_all_positives():
    result = shortlist_metrics([0, 1, 0, 1], [0.8, 0.1, 0.9, 0.2], 1)
    assert result["recall"] == 1
    assert result["lift"] == 1


def test_no_positive_labels_have_undefined_recall():
    result = shortlist_metrics([0, 0], [0.2, 0.3], 0.5)
    assert np.isnan(result["recall"])
    assert np.isnan(result["lift"])


@pytest.mark.parametrize(
    "y,scores,fraction",
    [
        ([], [], 0.2),
        ([1], [0.2, 0.3], 0.2),
        ([2], [0.5], 0.2),
        ([1], [np.nan], 0.2),
        ([1], [np.inf], 0.2),
        ([1], [1.1], 0.2),
        ([1], [0.2], 0),
        ([1], [0.2], 1.1),
        ([1], [0.2], np.nan),
        ([[1]], [[0.2]], 0.5),
    ],
)
def test_invalid_budget_inputs_fail(y, scores, fraction):
    with pytest.raises(ValueError):
        shortlist_metrics(y, scores, fraction)


def test_outliers_are_flagged_without_deleting_rows():
    df = pd.DataFrame({"alcohol": [10, 10, 10, 10, 100], "quality": [5] * 5})
    original = df.copy()
    result = outlier_audit(df)
    assert result.iloc[0]["flagged"] == 1
    assert result.feature.tolist() == ["alcohol"]
    pd.testing.assert_frame_equal(df, original)


@pytest.mark.parametrize("quality", [[np.nan], [np.inf], [-1], [11], [6.5], []])
def test_invalid_quality_is_not_silently_labeled_bad(quality):
    with pytest.raises(ValueError, match="quality|Quality"):
        build_target(pd.DataFrame({"quality": quality}))


def test_missing_features_fail_before_training(clean_frame):
    clean_frame.loc[0, "alcohol"] = np.nan
    with pytest.raises(ValueError, match="missing/infinite"):
        train_and_evaluate(clean_frame)


def test_single_class_fails_clearly(clean_frame):
    clean_frame["quality"] = 5
    with pytest.raises(ValueError, match="each quality class"):
        train_and_evaluate(clean_frame)


def test_empty_baseline_fails():
    with pytest.raises(ValueError):
        majority_baseline(pd.Series([], dtype=int))
