"""Descriptive steps: inspection, filtering and grouping."""

from __future__ import annotations

import pandas as pd

from winequality.config import ALCOHOL_FILTER

SMALL_GROUP = 50  # group means based on fewer wines than this are flagged as unstable


def inspect(df: pd.DataFrame) -> dict:
    """Print head/info/describe, missing values, duplicates and the target distribution."""
    print("\n--- .head() ---")
    print(df.head())
    print("\n--- .info() ---")
    df.info()
    print("\n--- .describe() ---")
    print(df.describe().T.round(3))

    missing = df.isnull().sum()
    print("\n--- missing values per column ---")
    print(missing[missing > 0] if missing.any() else "No missing values in any column.")

    n_dupes = int(df.duplicated().sum())
    print(
        f"\n--- duplicates ---\nFully duplicated rows: {n_dupes} "
        f"({n_dupes / len(df):.1%} of the dataset)"
    )
    if n_dupes:
        print("Kept for EDA, dropped before modelling so a row cannot sit in both train and test.")

    counts = df["quality"].value_counts().sort_index()
    print("\n--- target distribution (quality) ---")
    print(pd.DataFrame({"count": counts, "share": (counts / len(df)).round(3)}))
    return {"n_missing": int(missing.sum()), "n_duplicates": n_dupes}


def summarise_by_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Count and chemistry means for each quality score, sorted by score."""
    return (
        df.groupby("quality")
        .agg(
            n=("quality", "size"),
            mean_alcohol=("alcohol", "mean"),
            mean_volatile_acidity=("volatile_acidity", "mean"),
            mean_sulphates=("sulphates", "mean"),
            mean_density=("density", "mean"),
        )
        .round(3)
    )


def filter_and_group(df: pd.DataFrame) -> pd.DataFrame:
    """Compare high-alcohol wines with the rest and print the per-score summary."""
    high_alcohol = df[df["alcohol"] > ALCOHOL_FILTER]
    print(f"\nFilter: alcohol > {ALCOHOL_FILTER}% ABV")
    print(f"  {len(high_alcohol)} of {len(df)} wines ({len(high_alcohol) / len(df):.1%})")
    print(f"  mean quality in this subset: {high_alcohol['quality'].mean():.3f}")
    print(f"  mean quality overall:        {df['quality'].mean():.3f}")

    grouped = summarise_by_quality(df)
    print("\nGroup by quality score:")
    print(grouped)

    trough_score = grouped["mean_alcohol"].idxmin()
    print(
        f"\nMean alcohol is lowest at quality {trough_score} "
        f"({grouped['mean_alcohol'].min():.2f}%), not at the lowest score."
    )
    small = grouped[grouped["n"] < SMALL_GROUP]
    if not small.empty:
        print(
            f"\nCaution: quality score(s) {list(small.index)} have fewer than {SMALL_GROUP} "
            f"samples ({small['n'].to_dict()}); their group means are unstable."
        )
    return grouped
