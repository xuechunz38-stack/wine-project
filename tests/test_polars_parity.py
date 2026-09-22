"""Pandas and Polars pipelines must compute the same answers.

Skipped automatically if Polars is not installed.
"""

from __future__ import annotations

import pytest

pl = pytest.importorskip("polars")

import analysis_polars as ap  # noqa: E402  (import after importorskip on purpose)


@pytest.fixture
def csv_path(data_dir):
    return next(data_dir.glob("*.csv"))


def test_read_gives_same_shape_and_columns(csv_path):
    pdf = ap.pandas_read(csv_path, ",")
    pldf = ap.polars_read(csv_path, ",")
    assert pdf.shape == pldf.shape
    assert list(pdf.columns) == list(pldf.columns)


def test_filter_gives_same_row_count(csv_path):
    pdf = ap.pandas_read(csv_path, ",")
    pldf = ap.polars_read(csv_path, ",")
    assert len(ap.pandas_filter(pdf)) == ap.polars_filter(pldf).height


def test_groupby_results_match(csv_path):
    pdf = ap.pandas_read(csv_path, ",")
    pldf = ap.polars_read(csv_path, ",")
    assert ap.results_match(ap.pandas_group(pdf), ap.polars_group(pldf))


def test_results_match_catches_a_wrong_value(csv_path):
    """The comparison must actually fail when the numbers differ, or it proves nothing."""
    pdf = ap.pandas_group(ap.pandas_read(csv_path, ","))
    pldf = ap.polars_group(ap.polars_read(csv_path, ","))
    pdf.loc[0, "mean_alcohol"] += 1.0
    assert not ap.results_match(pdf, pldf)
