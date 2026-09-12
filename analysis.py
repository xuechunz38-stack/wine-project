"""Wine quality: exploratory data analysis and a first machine-learning model.

Covers the five required steps of the assignment:
    1. Import the dataset
    2. Inspect the data (head / info / describe / missing values / duplicates)
    3. Basic filtering and grouping
    4. Explore a machine-learning algorithm
    5. Visualisation

Run with:  python analysis.py
Outputs:   figures/wine_overview.png  and  a printed report on stdout
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data_utils import (
    ALCOHOL_FILTER,
    QUALITY_THRESHOLD,
    detect_separator,
    find_dataset,
    normalise,
)

FIG_DIR = Path(__file__).parent / "figures"
RANDOM_STATE = 42


def banner(text: str) -> None:
    print(f"\n{'=' * 70}\n{text}\n{'=' * 70}")



def load_data() -> pd.DataFrame:
    path = find_dataset()
    sep = detect_separator(path)
    df = pd.read_csv(path, sep=sep)
    df.columns = [normalise(c) for c in df.columns]
    print(f"Loaded {path.name}  (separator {sep!r})  ->  {df.shape[0]} rows x {df.shape[1]} columns")
    return df



def inspect(df: pd.DataFrame) -> None:
    banner("2. INSPECTING THE DATA")

    print("\n--- .head() ---")
    print(df.head())

    print("\n--- .info() ---")
    df.info()

    print("\n--- .describe() ---")
    print(df.describe().T.round(3))

    print("\n--- missing values per column ---")
    missing = df.isnull().sum()
    print(missing[missing > 0] if missing.any() else "No missing values in any column.")

    n_dupes = int(df.duplicated().sum())
    print(f"\n--- duplicates ---\nFully duplicated rows: {n_dupes} "
          f"({n_dupes / len(df):.1%} of the dataset)")
    if n_dupes:
        print("These are kept for the EDA but dropped before model training, "
              "so that identical rows cannot appear in both train and test sets.")

    print("\n--- target distribution (quality) ---")
    counts = df["quality"].value_counts().sort_index()
    print(pd.DataFrame({"count": counts, "share": (counts / len(df)).round(3)}))



def filter_and_group(df: pd.DataFrame) -> pd.DataFrame:
    banner("3. FILTERING AND GROUPING")

    high_alcohol = df[df["alcohol"] > ALCOHOL_FILTER]
    print(f"\nFilter: alcohol > {ALCOHOL_FILTER}% ABV")
    print(f"  {len(high_alcohol)} of {len(df)} wines ({len(high_alcohol) / len(df):.1%})")
    print(f"  mean quality in this subset: {high_alcohol['quality'].mean():.3f}")
    print(f"  mean quality overall:        {df['quality'].mean():.3f}")

    grouped = (
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
    print("\nGroup by quality score:")
    print(grouped)

    lo, hi = grouped["mean_alcohol"].iloc[0], grouped["mean_alcohol"].iloc[-1]
    trough_score = grouped["mean_alcohol"].idxmin()
    trough = grouped["mean_alcohol"].min()
    print(f"\nMean alcohol at the lowest quality score:  {lo:.2f}%")
    print(f"Mean alcohol at the highest quality score: {hi:.2f}%")
    print(f"Minimum is at quality {trough_score}, not at the bottom: {trough:.2f}%")
    if trough_score not in (grouped.index[0], grouped.index[-1]):
        print(
            f"So the relationship is U-shaped, not monotonic: alcohol falls to "
            f"{trough:.2f}% at quality {trough_score}, then rises {hi - trough:.2f} "
            f"percentage points to {hi:.2f}% at quality {grouped.index[-1]}."
        )

    small = grouped[grouped["n"] < 50]
    if not small.empty:
        print(
            f"\nCaution: quality score(s) {list(small.index)} have fewer than 50 "
            f"samples ({small['n'].to_dict()}). Their group means are unstable "
            "and should not be used to support a conclusion."
        )

    return grouped



def build_target(df: pd.DataFrame) -> pd.DataFrame:
    """Turn the 3-8 quality score into a binary 'good wine' label."""
    out = df.drop_duplicates().copy()
    out["good"] = (out["quality"] >= QUALITY_THRESHOLD).astype(int)
    return out


def explore_model(df: pd.DataFrame) -> pd.Series:
    banner("4. MACHINE LEARNING: IS THIS A GOOD WINE?")

    data = build_target(df)
    feature_cols = [c for c in data.columns if c not in ("quality", "good")]
    feature_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(data[c])]

    X = data[feature_cols]
    y = data["good"]

    print(f"\nTask       : binary classification, good = quality >= {QUALITY_THRESHOLD}")
    print(f"Rows used  : {len(data)} (after dropping duplicates)")
    print(f"Features   : {len(feature_cols)} -> {', '.join(feature_cols)}")
    print(f"Class split: {y.value_counts().to_dict()}  "
          f"(positive class = {y.mean():.1%} of rows)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    baseline = 1 - y_test.mean()  
    print(f"\nMajority-class baseline accuracy: {baseline:.3f}")
    print("Any model has to beat this number to be worth anything.")

    
    logreg = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced"),
    )
    logreg.fit(X_train, y_train)
    pred_lr = logreg.predict(X_test)

    print("\n--- Logistic regression ---")
    print(f"accuracy: {accuracy_score(y_test, pred_lr):.3f}   "
          f"ROC-AUC: {roc_auc_score(y_test, logreg.predict_proba(X_test)[:, 1]):.3f}")
    print(classification_report(y_test, pred_lr, target_names=["not good", "good"]))

    
    forest = RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1,
    )
    forest.fit(X_train, y_train)
    pred_rf = forest.predict(X_test)

    print("--- Random forest ---")
    print(f"accuracy: {accuracy_score(y_test, pred_rf):.3f}   "
          f"ROC-AUC: {roc_auc_score(y_test, forest.predict_proba(X_test)[:, 1]):.3f}")
    print(classification_report(y_test, pred_rf, target_names=["not good", "good"]))

    importances = (
        pd.Series(forest.feature_importances_, index=feature_cols)
        .sort_values(ascending=False)
    )
    print("Feature importance (random forest):")
    print(importances.round(4).to_string())

    print(
        "\nNote: accuracy alone is misleading on this dataset because the classes "
        "are imbalanced. Recall on the 'good' class is the number that actually "
        "says whether the model finds good wines."
    )
    return importances



def visualise(df: pd.DataFrame, importances: pd.Series) -> Path:
    banner("5. VISUALISATION")

    FIG_DIR.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    counts = df["quality"].value_counts().sort_index()
    axes[0].bar(counts.index.astype(str), counts.values, color="#7a3b3b")
    axes[0].set_title("Distribution of quality scores")
    axes[0].set_xlabel("quality")
    axes[0].set_ylabel("number of wines")

    scores = sorted(df["quality"].unique())
    axes[1].boxplot([df.loc[df["quality"] == s, "alcohol"] for s in scores])
    axes[1].set_xticks(range(1, len(scores) + 1), [str(s) for s in scores])
    axes[1].set_title("Alcohol content by quality score")
    axes[1].set_xlabel("quality")
    axes[1].set_ylabel("alcohol (% ABV)")

    top = importances.head(8).sort_values()
    axes[2].barh(top.index, top.values, color="#4a6b8a")
    axes[2].set_title("Top features (random forest)")
    axes[2].set_xlabel("importance")

    fig.tight_layout()
    out_path = FIG_DIR / "wine_overview.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"Figure written to {out_path}")
    return out_path


def main() -> None:
    banner("1. IMPORTING THE DATASET")
    df = load_data()

    inspect(df)
    filter_and_group(df)
    importances = explore_model(df)
    visualise(df, importances)

    banner("DONE")


if __name__ == "__main__":
    main()
