"""Wine quality: can chemistry help a tasting team shortlist good wines?

Steps
    1. Import            4. Model comparison at the default 0.5 cut-off
    2. Inspect + clean   5. Choose a cut-off on training data for a target recall
    3. Filter and group  6. Check the chosen cut-off separately for red and white wines
                         7. Figures and CSV outputs

Run with:  python analysis.py
Outputs:   figures/ (or $WINE_OUTPUT_DIR): wine_overview.png, threshold_tradeoff.png,
           outlier_report.csv, threshold_by_type.csv
"""

from __future__ import annotations

import pandas as pd

from winequality import cleaning, config, exploration, modeling, plots
from winequality.loading import read_wine_csv
from winequality.reporting import print_section


def load_data() -> pd.DataFrame:
    df, path, sep = read_wine_csv()
    print(f"Loaded {path.name} (separator {sep!r}) -> {df.shape[0]} rows x {df.shape[1]} columns")
    return df


def compare_models(data: pd.DataFrame, features: list[str]) -> dict:
    """Fit both models on the same split and score them at the default cut-off."""
    split = modeling.split_data(data, features)
    logreg = modeling.make_logistic_regression().fit(split.X_train, split.y_train)
    forest = modeling.make_random_forest().fit(split.X_train, split.y_train)
    importances = pd.Series(forest.feature_importances_, index=features).sort_values(
        ascending=False
    )
    return {
        "split": split,
        "features": features,
        "baseline": modeling.majority_baseline(split.y_test),
        "logistic_regression": modeling.evaluate(logreg, split.X_test, split.y_test),
        "random_forest": modeling.evaluate(forest, split.X_test, split.y_test),
        "importances": importances,
    }


def print_model_comparison(results: dict) -> None:
    print(f"\nMajority-class baseline accuracy: {results['baseline']:.3f}")
    for key, label in (
        ("logistic_regression", "Logistic regression"),
        ("random_forest", "Random forest"),
    ):
        m = results[key]
        print(f"\n--- {label} ---")
        print(
            f"accuracy {m['accuracy']:.3f} | ROC-AUC {m['roc_auc']:.3f} | "
            f"precision {m['precision']:.2f} | recall {m['recall']:.2f}"
        )
        print(m["report"])
    print("Feature importance (random forest):")
    print(results["importances"].round(4).to_string())


def tune_threshold(results: dict, target_recall: float = config.TARGET_RECALL) -> dict:
    """Pick the forest's cut-off from out-of-fold training probabilities, then test it."""
    split = results["split"]
    oof = modeling.out_of_fold_probabilities(
        modeling.make_random_forest(), split.X_train, split.y_train
    )
    threshold = modeling.choose_threshold(split.y_train, oof, target_recall)
    test_proba = results["random_forest"]["probabilities"]
    return {
        "threshold": threshold,
        "target_recall": target_recall,
        "train_cv": modeling.metrics_at_threshold(split.y_train, oof, threshold),
        "test": modeling.metrics_at_threshold(split.y_test, test_proba, threshold),
        "test_default": modeling.metrics_at_threshold(split.y_test, test_proba, 0.5),
    }


def main() -> None:
    pd.set_option("display.width", 140)
    pd.set_option("display.max_columns", 20)
    out = config.output_dir()

    print_section("1. IMPORTING THE DATASET")
    df = load_data()

    print_section("2. INSPECTING AND CLEANING THE DATA")
    exploration.inspect(df)
    cleaning.validate_wine_frame(df)
    outliers = cleaning.iqr_outlier_report(df, config.FEATURE_COLUMNS)
    outliers.to_csv(out / "outlier_report.csv", index=False)
    print("\n--- 1.5 x IQR outlier audit (flagged, not removed) ---")
    print(outliers.to_string(index=False))

    print_section("3. FILTERING AND GROUPING")
    exploration.filter_and_group(df)

    print_section("4. MACHINE LEARNING: IS THIS A GOOD WINE?")
    data = cleaning.build_target(df)
    features = cleaning.select_features(data)
    print(
        f"Rows used: {len(data)} after dropping duplicates; {len(features)} features; "
        f"positive class = {data['good'].mean():.1%}"
    )
    results = compare_models(data, features)
    print_model_comparison(results)

    print_section("5. CHOOSING A CUT-OFF WITHOUT TOUCHING THE TEST SET")
    tuned = tune_threshold(results)
    print(
        f"Target recall {tuned['target_recall']:.0%} -> cut-off {tuned['threshold']:.2f} "
        "(chosen on 5-fold out-of-fold training probabilities)"
    )
    for label in ("test_default", "test"):
        m = tuned[label]
        print(
            f"  test @ {m['threshold']:.2f}: flag {m['n_flagged']} of {m['n_wines']}, "
            f"find {m['good_found']}/{m['n_good']} good wines, "
            f"precision {m['precision']:.3f}, recall {m['recall']:.3f}"
        )

    print_section("6. RED VS WHITE AT THE CHOSEN CUT-OFF")
    split = results["split"]
    wine_type = df.loc[split.X_test.index, "type"] if "type" in df.columns else None
    by_type = pd.DataFrame()
    if wine_type is not None:
        by_type = modeling.metrics_by_group(
            split.y_test, results["random_forest"]["probabilities"], wine_type, tuned["threshold"]
        )
        by_type.round(3).to_csv(out / "threshold_by_type.csv", index=False)
        print(by_type.round(3).to_string(index=False))
    else:
        print("No `type` column: skipping the red/white comparison.")

    print_section("7. VISUALISATION")
    print(f"Wrote {plots.plot_overview(df, results['importances'], out / 'wine_overview.png')}")
    if wine_type is not None:
        path = plots.plot_threshold_tradeoff(
            split.y_test,
            results["random_forest"]["probabilities"],
            wine_type,
            tuned["threshold"],
            tuned["target_recall"],
            out / "threshold_tradeoff.png",
        )
        print(f"Wrote {path}")

    print_section("DONE")


if __name__ == "__main__":
    main()
