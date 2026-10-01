"""Locating, sniffing and loading the wine CSV.

Shared by the Pandas and the Polars pipeline, so both read the same file with
the same delimiter and the same column names.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

from winequality import config


def find_dataset(data_dir: Path | None = None) -> Path:
    """Return the wine CSV in `data_dir`, preferring a file with 'wine' in its name."""
    folder = Path(data_dir) if data_dir is not None else config.DATA_DIR
    candidates = sorted(folder.glob("*.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"No CSV found in {folder}. Download the dataset from Kaggle "
            "and place it there (see README.md)."
        )
    for path in candidates:
        if "wine" in path.name.lower():
            return path
    return candidates[0]


def detect_separator(path: Path) -> str:
    """The UCI original is semicolon-separated; most Kaggle mirrors use commas."""
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        sample = handle.read(4096)
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t").delimiter
    except csv.Error:
        return ";" if sample.count(";") > sample.count(",") else ","


def normalise_column(name: str) -> str:
    """'Volatile Acidity' -> 'volatile_acidity' so column names are predictable."""
    return name.strip().strip('"').lower().replace(" ", "_").replace("-", "_")


def read_wine_csv(
    path: Path | None = None, sep: str | None = None
) -> tuple[pd.DataFrame, Path, str]:
    """Read the CSV with Pandas and normalise its headers.

    Returns the frame together with the path and separator that were used, so
    callers can report them without re-detecting. Pass `sep` to skip sniffing.
    """
    path = Path(path) if path is not None else find_dataset()
    sep = sep or detect_separator(path)
    df = pd.read_csv(path, sep=sep)
    df.columns = [normalise_column(c) for c in df.columns]
    return df, path, sep
