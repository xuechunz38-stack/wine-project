"""Shared fixtures.

The unit tests run on a small synthetic dataset rather than the real CSV, so
they are fast, deterministic, and do not depend on the data file being present.
Only the system tests in test_system.py touch the real data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import data_utils

# Column names exactly as they appear in the Kaggle CSV (spaces, mixed case),
# so the tests also exercise the normalisation step.
RAW_COLUMNS = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
    "pH", "sulphates", "alcohol", "quality", "type",
]


def make_wine_frame(n: int = 300, seed: int = 0) -> pd.DataFrame:
    """A small wine-like table in which alcohol is the only real signal.

    Quality is driven by alcohol plus noise, and every other feature is pure
    noise. That gives the model tests a known right answer: a working model
    should rank alcohol first and beat chance.
    """
    rng = np.random.default_rng(seed)
    alcohol = rng.uniform(8.0, 14.0, n)
    # Offset chosen so fewer than half the wines score >= 7, as in the real data.
    quality = np.clip(np.round(alcohol - 5.3 + rng.normal(0, 0.6, n)), 3, 9).astype(int)
    df = pd.DataFrame({
        "fixed acidity": rng.normal(7.2, 1.3, n),
        "volatile acidity": rng.normal(0.34, 0.16, n).clip(0.05),
        "citric acid": rng.normal(0.32, 0.14, n).clip(0),
        "residual sugar": rng.gamma(2.0, 2.7, n),
        "chlorides": rng.normal(0.056, 0.03, n).clip(0.005),
        "free sulfur dioxide": rng.normal(30, 17, n).clip(1),
        "total sulfur dioxide": rng.normal(115, 56, n).clip(5),
        "density": rng.normal(0.995, 0.003, n),
        "pH": rng.normal(3.22, 0.16, n),
        "sulphates": rng.normal(0.53, 0.15, n).clip(0.2),
        "alcohol": alcohol.round(1),
        "quality": quality,
        "type": rng.choice(["red", "white"], n),
    })
    return df[RAW_COLUMNS]


@pytest.fixture
def raw_frame() -> pd.DataFrame:
    return make_wine_frame()


@pytest.fixture
def clean_frame(raw_frame: pd.DataFrame) -> pd.DataFrame:
    """The synthetic frame with column names normalised, as load_data returns it."""
    out = raw_frame.copy()
    out.columns = [data_utils.normalise(c) for c in out.columns]
    return out


@pytest.fixture
def data_dir(tmp_path, monkeypatch, raw_frame):
    """A temporary data/ folder holding the synthetic CSV.

    data_utils.find_dataset() reads the module-level DATA_DIR at call time, so
    pointing it here makes every loader in the project use the synthetic file.
    """
    folder = tmp_path / "data"
    folder.mkdir()
    raw_frame.to_csv(folder / "wine_quality_test.csv", index=False)
    monkeypatch.setattr(data_utils, "DATA_DIR", folder)
    return folder


@pytest.fixture
def fig_dir(tmp_path, monkeypatch):
    """Redirect figure output so tests never overwrite the committed figure."""
    import analysis

    folder = tmp_path / "figures"
    monkeypatch.setattr(analysis, "FIG_DIR", folder)
    return folder
