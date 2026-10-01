# Data, method, and limitations

The committed [merged red/white wine dataset](https://www.kaggle.com/datasets/amirmohamadrezaie/red-and-white-wine-quality)
is a mirror of [UCI Wine Quality](https://archive.ics.uci.edu/dataset/186/wine+quality):
**6,497 rows**, 11 chemical predictors, quality, and wine type.

- **Missing data:** the committed data has none. Modelling rejects missing or
  infinite numeric features and invalid quality labels with an explicit error;
  no full-dataset imputation can leak information into evaluation.
- **Duplicates:** EDA retains all observations; modelling removes **1,177 exact
  duplicate rows before splitting**, leaving 5,320 observations.
- **Outliers:** the report flags numeric predictors outside 1.5-IQR fences but
  retains them. A univariate flag alone does not prove measurement error, and
  automatic trimming could remove rare wine styles. See
  [outlier audit](../figures/outlier_audit.csv).
- **Evaluation:** fixed, stratified 80/20 split (`random_state=42`); 1,064 test
  wines, including 202 good wines. Logistic regression uses training-only
  standardisation; both classifiers use balanced class weights.
- **Scope:** wine type is excluded to keep earlier results comparable. These
  observational results do not establish causation or generalise automatically
  to other regions, vintages, or tasting panels.


## Interpretation

Alcohol has the largest random-forest impurity importance. This is predictive association, not a causal effect. Quality scores 3 and 9 have only 30 and 5 observations, so their group averages are particularly uncertain. Default-threshold accuracy alone is not a useful comparison when the tasting workload differs. The decision report compares an explicit budget grid (5%, 10%, 20%, 30%, 50%, 100%) and records both found and missed good wines. A future deployment needs validation data for budget selection and a fresh final test.

## Earlier work and limitations

`python analysis_polars.py` checks Pandas/Polars results and benchmarks loading,
filtering, and grouping on the original and repeated data; timings depend on
the machine. The [Rust notebook](../rust_vs_python_intro.ipynb) and
[earlier analysis notes](previous-analysis.md) are retained from previous
mini-assignments. The notebook needs a separate Rust/evcxr Jupyter environment;
it is not part of the Python container or CI.

This is a single held-out split, with no uncertainty intervals or external
validation. Feature importance can be distorted by correlated predictors.
Future work should compare wine types separately and validate the ranking on
new samples. AI assistance was used for the Week 4 refactoring, tests, and
writing; the linked code, commits, and execution evidence make those changes
inspectable.
