"""Project-wide constants and paths.

Output location can be overridden with the WINE_OUTPUT_DIR environment variable,
which is how the Docker container writes figures to a mounted volume.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

QUALITY_THRESHOLD = 7  # quality >= 7 counts as a "good" wine
ALCOHOL_FILTER = 11.0  # % ABV used for the high-alcohol subset
RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET_RECALL = 0.70  # recall the tasting team wants when shortlisting wines

FEATURE_COLUMNS = [
    "fixed_acidity",
    "volatile_acidity",
    "citric_acid",
    "residual_sugar",
    "chlorides",
    "free_sulfur_dioxide",
    "total_sulfur_dioxide",
    "density",
    "ph",
    "sulphates",
    "alcohol",
]


def output_dir() -> Path:
    """Folder for figures and CSV outputs (created on demand)."""
    folder = Path(os.environ.get("WINE_OUTPUT_DIR", PROJECT_ROOT / "figures"))
    folder.mkdir(parents=True, exist_ok=True)
    return folder
