"""End-to-end: run the whole pipeline the way a user would."""

from __future__ import annotations

from pathlib import Path

import pytest

import analysis

REAL_DATA = Path(__file__).resolve().parent.parent / "data"


def test_pipeline_runs_end_to_end_on_synthetic_data(data_dir, fig_dir, capsys):
    analysis.main()
    out = capsys.readouterr().out

    for step in ("1. IMPORTING", "2. INSPECTING", "3. FILTERING", "4. MACHINE LEARNING", "5. VISUALISATION", "DONE"):
        assert step in out, f"pipeline never reached: {step}"
    assert (fig_dir / "wine_overview.png").exists()


@pytest.mark.skipif(not any(REAL_DATA.glob("*.csv")), reason="real dataset not present")
def test_pipeline_on_real_data_reproduces_readme_numbers(fig_dir, capsys):
    """Runs on the committed CSV and checks the figures quoted in the README.

    If any of these fail, either the data or the code changed and the README's
    findings are no longer accurate.
    """
    df = analysis.load_data()
    assert df.shape == (6497, 13)
    assert df.isnull().sum().sum() == 0
    assert int(df.duplicated().sum()) == 1177

    results = analysis.explore_model(df)
    assert results["baseline"] == pytest.approx(0.810, abs=0.005)
    assert results["random_forest"]["accuracy"] == pytest.approx(0.847, abs=0.005)
    assert results["random_forest"]["roc_auc"] == pytest.approx(0.879, abs=0.005)
    assert results["logistic_regression"]["accuracy"] == pytest.approx(0.735, abs=0.005)
    assert results["importances"].index[0] == "alcohol"

    analysis.visualise(df, results["importances"])
    assert (fig_dir / "wine_overview.png").exists()
