"""Data loading helpers: column-name normalisation, delimiter detection, file discovery."""

from __future__ import annotations

import pytest

from data_utils import detect_separator, find_dataset, normalise
import data_utils


# ------------------------------------------------------------------ normalise
@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Volatile Acidity", "volatile_acidity"),
        ("fixed acidity", "fixed_acidity"),
        ("pH", "ph"),
        ("free-sulfur dioxide", "free_sulfur_dioxide"),
        ('  "alcohol"  ', "alcohol"),  # stray whitespace and quotes from a messy header
        ("quality", "quality"),          # already clean: unchanged
    ],
)
def test_normalise(raw, expected):
    assert normalise(raw) == expected


def test_normalise_is_idempotent():
    once = normalise("Total Sulfur Dioxide")
    assert normalise(once) == once


# ----------------------------------------------------------- detect_separator
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
    """A file with no delimiter at all should still return something usable."""
    path = tmp_path / "wine.csv"
    path.write_text("quality\n5\n6\n", encoding="utf-8")
    assert detect_separator(path) in {",", ";"}


# --------------------------------------------------------------- find_dataset
def test_find_dataset_raises_when_folder_is_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(data_utils, "DATA_DIR", tmp_path)
    with pytest.raises(FileNotFoundError, match="No CSV found"):
        find_dataset()


def test_find_dataset_prefers_file_named_wine(tmp_path, monkeypatch):
    (tmp_path / "aaa_other.csv").write_text("x\n1\n")
    (tmp_path / "wine_quality.csv").write_text("x\n1\n")
    monkeypatch.setattr(data_utils, "DATA_DIR", tmp_path)
    assert find_dataset().name == "wine_quality.csv"


def test_find_dataset_falls_back_to_first_csv(tmp_path, monkeypatch):
    (tmp_path / "b.csv").write_text("x\n1\n")
    (tmp_path / "a.csv").write_text("x\n1\n")
    monkeypatch.setattr(data_utils, "DATA_DIR", tmp_path)
    assert find_dataset().name == "a.csv"


def test_find_dataset_ignores_non_csv_files(tmp_path, monkeypatch):
    (tmp_path / "notes.txt").write_text("not data")
    (tmp_path / "wine.csv").write_text("x\n1\n")
    monkeypatch.setattr(data_utils, "DATA_DIR", tmp_path)
    assert find_dataset().suffix == ".csv"
