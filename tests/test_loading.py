"""Loading helpers: column normalisation, delimiter detection, file discovery."""

from __future__ import annotations

import pandas as pd
import pytest

from winequality import config
from winequality.loading import detect_separator, find_dataset, normalise_column, read_wine_csv


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Volatile Acidity", "volatile_acidity"),
        ("fixed acidity", "fixed_acidity"),
        ("pH", "ph"),
        ("free-sulfur dioxide", "free_sulfur_dioxide"),
        ('  "alcohol"  ', "alcohol"),  # stray whitespace and quotes from a messy header
        ("quality", "quality"),  # already clean: unchanged
    ],
)
def test_normalise_column(raw, expected):
    assert normalise_column(raw) == expected


def test_normalise_column_is_idempotent():
    once = normalise_column("Total Sulfur Dioxide")
    assert normalise_column(once) == once


@pytest.mark.parametrize("sep", [",", ";", "\t"])
def test_detect_separator(tmp_path, sep):
    path = tmp_path / "wine.csv"
    header = sep.join(["fixed acidity", "alcohol", "quality"])
    rows = [sep.join(["7.4", "9.4", "5"]), sep.join(["7.8", "9.8", "6"])]
    path.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")
    assert detect_separator(path) == sep


def test_detect_separator_ignores_byte_order_mark(tmp_path):
    """Excel often saves UTF-8 CSVs with a BOM; it must not confuse detection."""
    path = tmp_path / "wine.csv"
    path.write_text("alcohol;quality\n9.4;5\n9.8;6\n", encoding="utf-8-sig")
    assert detect_separator(path) == ";"


def test_detect_separator_falls_back_on_single_column(tmp_path):
    path = tmp_path / "wine.csv"
    path.write_text("quality\n5\n6\n", encoding="utf-8")
    assert detect_separator(path) in {",", ";"}


def test_find_dataset_raises_when_folder_is_empty(tmp_path):
    with pytest.raises(FileNotFoundError, match="No CSV found"):
        find_dataset(tmp_path)


def test_find_dataset_prefers_file_named_wine(tmp_path):
    (tmp_path / "aaa_other.csv").write_text("x\n1\n")
    (tmp_path / "wine_quality.csv").write_text("x\n1\n")
    assert find_dataset(tmp_path).name == "wine_quality.csv"


def test_find_dataset_falls_back_to_first_csv(tmp_path):
    (tmp_path / "b.csv").write_text("x\n1\n")
    (tmp_path / "a.csv").write_text("x\n1\n")
    assert find_dataset(tmp_path).name == "a.csv"


def test_find_dataset_ignores_non_csv_files(tmp_path):
    (tmp_path / "notes.txt").write_text("not data")
    (tmp_path / "wine.csv").write_text("x\n1\n")
    assert find_dataset(tmp_path).suffix == ".csv"


def test_find_dataset_uses_config_data_dir_by_default(data_dir):
    assert find_dataset().parent == data_dir == config.DATA_DIR


def test_read_wine_csv_normalises_headers_and_keeps_rows(data_dir, raw_frame):
    df, path, sep = read_wine_csv()
    assert sep == ","
    assert len(df) == len(raw_frame)
    assert "volatile_acidity" in df.columns and "ph" in df.columns


def test_semicolon_and_comma_files_load_identically(tmp_path, raw_frame):
    """The UCI original is semicolon-separated; it must load like the Kaggle comma file."""
    comma, semi = tmp_path / "wine_a.csv", tmp_path / "wine_b.csv"
    raw_frame.to_csv(comma, index=False)
    raw_frame.to_csv(semi, sep=";", index=False)
    pd.testing.assert_frame_equal(read_wine_csv(comma)[0], read_wine_csv(semi)[0])
