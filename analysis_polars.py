"""Polars version of the pipeline, plus a head-to-head benchmark against Pandas.

The point of this script is not to show that one library "wins". It is to run
the *same three operations* (read, filter, group-by) through both libraries,
check that they produce identical numbers, and time them honestly -- first on
the real dataset, then on an artificially enlarged copy, because the size of
the data is what decides which library is faster.

Run with:  python analysis_polars.py
"""

from __future__ import annotations

import statistics
import time
from pathlib import Path

import pandas as pd
import polars as pl

from data_utils import ALCOHOL_FILTER, detect_separator, find_dataset, normalise

REPEATS = 15  # timed runs per operation; we report the median
SCALED_ROWS = 1_000_000
TMP_DIR = Path(__file__).parent / ".bench_tmp"


def banner(text: str) -> None:
    print(f"\n{'=' * 70}\n{text}\n{'=' * 70}")


def time_it(fn, repeats: int = REPEATS) -> tuple[float, object]:
    """Warm up once, then time `repeats` runs and return (median_ms, last_result)."""
    result = fn()  # warm-up: fills the OS page cache, triggers any lazy imports
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        timings.append((time.perf_counter() - start) * 1000)
    return statistics.median(timings), result



def pandas_read(path: Path, sep: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=sep)
    df.columns = [normalise(c) for c in df.columns]
    return df


def polars_read(path: Path, sep: str) -> pl.DataFrame:
    df = pl.read_csv(path, separator=sep)
    return df.rename({c: normalise(c) for c in df.columns})


def pandas_filter(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["alcohol"] > ALCOHOL_FILTER]


def polars_filter(df: pl.DataFrame) -> pl.DataFrame:
    return df.filter(pl.col("alcohol") > ALCOHOL_FILTER)


def pandas_group(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("quality")
        .agg(
            n=("quality", "size"),
            mean_alcohol=("alcohol", "mean"),
            mean_volatile_acidity=("volatile_acidity", "mean"),
            mean_sulphates=("sulphates", "mean"),
        )
        .reset_index()
        .sort_values("quality")
        .reset_index(drop=True)
    )


def polars_group(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.group_by("quality")
        .agg(
            pl.len().alias("n"),
            pl.col("alcohol").mean().alias("mean_alcohol"),
            pl.col("volatile_acidity").mean().alias("mean_volatile_acidity"),
            pl.col("sulphates").mean().alias("mean_sulphates"),
        )
        .sort("quality")
    )


def polars_lazy_group(path: Path, sep: str) -> pl.DataFrame:
    """The whole pipeline expressed lazily, so Polars can optimise it as a unit."""
    scan = pl.scan_csv(path, separator=sep)
    renamed = {c: normalise(c) for c in scan.collect_schema().names()}
    return (
        scan.rename(renamed)
        .filter(pl.col("alcohol") > ALCOHOL_FILTER)
        .group_by("quality")
        .agg(
            pl.len().alias("n"),
            pl.col("alcohol").mean().alias("mean_alcohol"),
        )
        .sort("quality")
        .collect()
    )



def results_match(pdf: pd.DataFrame, pldf: pl.DataFrame, tol: float = 1e-6) -> bool:
    """Confirm both libraries computed the same group-by.

    Deliberately avoids `pldf.to_pandas()`, which would pull in pyarrow as an
    extra dependency. Comparing plain Python values is enough here and keeps the
    check independent of either library's internals.
    """
    if list(pdf.columns) != list(pldf.columns):
        print(f"  column mismatch: {list(pdf.columns)} vs {list(pldf.columns)}")
        return False

    if len(pdf) != pldf.height:
        print(f"  row count mismatch: {len(pdf)} vs {pldf.height}")
        return False

    pandas_rows = pdf.to_dict(orient="records")
    polars_rows = pldf.to_dicts()

    for i, (left, right) in enumerate(zip(pandas_rows, polars_rows)):
        for col in pdf.columns:
            a, b = left[col], right[col]
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                if abs(float(a) - float(b)) > tol:
                    print(f"  row {i}, column {col!r}: {a} vs {b}")
                    return False
            elif a != b:
                print(f"  row {i}, column {col!r}: {a!r} vs {b!r}")
                return False
    return True



def run_benchmark(path: Path, sep: str, label: str) -> list[dict]:
    print(f"\n--- {label} ---")
    rows = []

    read_pd_ms, pdf = time_it(lambda: pandas_read(path, sep))
    read_pl_ms, pldf = time_it(lambda: polars_read(path, sep))
    rows.append({"operation": "read_csv", "pandas_ms": read_pd_ms, "polars_ms": read_pl_ms})

    filt_pd_ms, filt_pd = time_it(lambda: pandas_filter(pdf))
    filt_pl_ms, filt_pl = time_it(lambda: polars_filter(pldf))
    rows.append({"operation": "filter", "pandas_ms": filt_pd_ms, "polars_ms": filt_pl_ms})

    grp_pd_ms, grp_pd = time_it(lambda: pandas_group(pdf))
    grp_pl_ms, grp_pl = time_it(lambda: polars_group(pldf))
    rows.append({"operation": "groupby+agg", "pandas_ms": grp_pd_ms, "polars_ms": grp_pl_ms})

    print(f"rows: {len(pdf):,}   filtered rows: {len(filt_pd):,} (pandas) / "
          f"{filt_pl.height:,} (polars)")
    print(f"results identical: {results_match(grp_pd, grp_pl)}")

    print(f"\n{'operation':<14}{'pandas (ms)':>14}{'polars (ms)':>14}{'speed-up':>12}")
    for row in rows:
        ratio = row["pandas_ms"] / row["polars_ms"]
        row["speedup"] = ratio
        row["scale"] = label
        print(f"{row['operation']:<14}{row['pandas_ms']:>14.3f}"
              f"{row['polars_ms']:>14.3f}{ratio:>11.2f}x")

    return rows


def make_scaled_copy(path: Path, sep: str) -> Path:
    """Repeat the dataset until it has roughly SCALED_ROWS rows."""
    TMP_DIR.mkdir(exist_ok=True)
    out = TMP_DIR / "wine_scaled.csv"
    df = pandas_read(path, sep)
    factor = max(1, SCALED_ROWS // len(df))
    big = pd.concat([df] * factor, ignore_index=True)
    big.to_csv(out, sep=sep, index=False)
    size_mb = out.stat().st_size / 1024 / 1024
    print(f"\nBuilt a scaled copy: {len(big):,} rows, {size_mb:.1f} MB "
          f"(original repeated {factor}x)")
    return out


def main() -> None:
    banner("PANDAS vs POLARS")
    print(f"pandas {pd.__version__}   polars {pl.__version__}")
    print(f"Each operation is warmed up once, then run {REPEATS} times; "
          "the median is reported.")

    path = find_dataset()
    sep = detect_separator(path)

    all_rows = run_benchmark(path, sep, f"original dataset ({path.name})")

    scaled = make_scaled_copy(path, sep)
    all_rows += run_benchmark(scaled, sep, f"scaled to ~{SCALED_ROWS:,} rows")

    banner("LAZY API")
    lazy_ms, lazy_result = time_it(lambda: polars_lazy_group(scaled, sep))
    print(f"scan_csv -> filter -> group_by -> collect on the scaled file: "
          f"{lazy_ms:.1f} ms")
    print("Lazy mode lets Polars push the filter down into the CSV scan, so rows "
          "that fail the predicate are never fully materialised.")
    print(lazy_result)

    summary = pd.DataFrame(all_rows)[["scale", "operation", "pandas_ms", "polars_ms", "speedup"]]
    summary.to_csv(Path(__file__).parent / "benchmark_results.csv", index=False)

    banner("SUMMARY")
    print(summary.round(3).to_string(index=False))
    print("\nWritten to benchmark_results.csv -- paste these numbers into the README.")

    scaled.unlink(missing_ok=True)
    TMP_DIR.rmdir()


if __name__ == "__main__":
    main()
