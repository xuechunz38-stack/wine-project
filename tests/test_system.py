"""End-to-end: run the whole pipeline the way a user would."""

from __future__ import annotations

import pandas as pd
import pytest

import analysis
from winequality import cleaning, config

REAL_DATA = config.PROJECT_ROOT / "data"
EXPECTED_OUTPUTS = (
    "wine_overview.png",
    "threshold_tradeoff.png",
    "outlier_report.csv",
    "threshold_by_type.csv",
)


def test_pipeline_runs_end_to_end_on_synthetic_data(data_dir, fig_dir, capsys):
    analysis.main()
    out = capsys.readouterr().out
    for step in ("1. IMPORTING", "2. INSPECTING", "4. MACHINE LEARNING", "6. RED VS WHITE", "DONE"):
        assert step in out, f"pipeline never reached: {step}"
    for name in EXPECTED_OUTPUTS:
        assert (fig_dir / name).exists(), name
    assert (fig_dir / "wine_overview.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_pipeline_fails_clearly_on_a_broken_file(data_dir, fig_dir, raw_frame):
    for f in data_dir.glob("*.csv"):
        f.unlink()
    raw_frame.drop(columns="alcohol").to_csv(data_dir / "wine_broken.csv", index=False)
    with pytest.raises(ValueError, match="alcohol"):
        analysis.main()


@pytest.mark.skipif(not any(REAL_DATA.glob("*.csv")), reason="real dataset not present")
def test_real_data_reproduces_readme_numbers(fig_dir):
    """If these fail, the data or code changed and the README findings are stale."""
    df = analysis.load_data()
    assert df.shape == (6497, 13)
    assert int(df.duplicated().sum()) == 1177

    data = cleaning.build_target(df)
    results = analysis.compare_models(data, cleaning.select_features(data))
    assert results["baseline"] == pytest.approx(0.810, abs=0.005)
    assert results["random_forest"]["roc_auc"] == pytest.approx(0.879, abs=0.01)
    assert results["logistic_regression"]["roc_auc"] == pytest.approx(0.816, abs=0.01)
    assert results["importances"].index[0] == "alcohol"

    tuned = analysis.tune_threshold(results)
    assert tuned["threshold"] == pytest.approx(0.25, abs=0.03)
    assert tuned["test"]["recall"] == pytest.approx(0.76, abs=0.04)
    assert tuned["test"]["precision"] == pytest.approx(0.50, abs=0.04)

    outliers = cleaning.iqr_outlier_report(df, config.FEATURE_COLUMNS).set_index("column")
    assert isinstance(outliers, pd.DataFrame) and outliers["n_outliers"].sum() > 0
