"""Shared helpers for locating and normalising the wine-quality dataset.

Both the Pandas and the Polars pipeline import from here so that they are
guaranteed to read the same file, with the same delimiter and the same
column names.  Any difference in the benchmark then comes from the library,
not from the loading logic.
"""

from __future__ import annotations

import csv
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
QUALITY_THRESHOLD = 7  
ALCOHOL_FILTER = 11.0  


def find_dataset() -> Path:
    """Return the path to the wine CSV, whatever it happens to be called."""
    candidates = sorted(DATA_DIR.glob("*.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"No CSV found in {DATA_DIR}. Download the dataset from Kaggle "
            "and place it there (see README.md)."
        )
    
    for path in candidates:
        if "wine" in path.name.lower():
            return path
    return candidates[0]


def detect_separator(path: Path) -> str:
    """The UCI original is semicolon-separated; most Kaggle mirrors are commas."""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        sample = handle.read(4096)
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t").delimiter
    except csv.Error:
        return ";" if sample.count(";") > sample.count(",") else ","


def normalise(name: str) -> str:
    """'Volatile Acidity' -> 'volatile_acidity' so column names are predictable."""
    return name.strip().strip('"').lower().replace(" ", "_").replace("-", "_")
