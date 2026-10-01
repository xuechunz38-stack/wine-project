# Pandas vs Polars benchmark (Week 2–3)

Run with `python analysis_polars.py` (or `make benchmark`). Numbers below are from the original Week 3 run on a MacBook and are saved in `benchmark_results.csv`.


Both pipelines run the same three operations -- `read_csv`, a filter on
`alcohol > 11`, and a group-by with four aggregations. The script verifies that
the two group-by results are numerically identical before reporting any timings,
so the comparison is between two implementations of the same computation. Both
also agree on the filter: 1,969 of 6,497 rows, and 301,257 of 994,041 on the
scaled file.

Method: one untimed warm-up run per operation (to fill the OS page cache and
absorb first-call overhead), then 15 timed runs, reporting the **median** rather
than the mean so a single scheduling hiccup does not dominate.

The dataset is only 6,497 rows, so the benchmark is also run on a copy repeated
153 times to ~994,041 rows (60.5 MB) -- large enough for the scaling behaviour to
show.

## Original dataset (6,497 rows)

| Operation | Pandas (ms) | Polars (ms) | Speed-up |
|---|---|---|---|
| `read_csv` | 5.197 | 1.532 | 3.39x |
| `filter` | 0.214 | 0.885 | **0.24x** |
| `groupby + agg` | 3.129 | 0.763 | 4.10x |

## Scaled to ~994,041 rows

| Operation | Pandas (ms) | Polars (ms) | Speed-up |
|---|---|---|---|
| `read_csv` | 623.856 | 139.988 | 4.46x |
| `filter` | 16.072 | 5.620 | **2.86x** |
| `groupby + agg` | 46.562 | 5.510 | 8.45x |

## Interpretation

**The result is not "Polars is faster". It depends on the operation and on the
data size, and one cell above goes the other way.**

*Filtering is slower in Polars on the small dataset* -- 0.885 ms against 0.214 ms,
about four times worse. Filtering 6,497 rows is almost no work: Pandas builds a
boolean mask over a contiguous NumPy array in vectorised C and is done in a
fifth of a millisecond. Polars pays a fixed cost first -- constructing the
expression, handing it to its execution engine, coordinating worker threads --
and here that setup costs more than the filtering itself. The absolute gap is
0.67 ms, which no user would ever notice, but it is a clean illustration of
fixed overhead dominating when there is not enough work to amortise it.

*The same filter flips to 2.86x in Polars' favour at a million rows.* The setup
cost has not changed; the amount of real work has grown ~150x, so the overhead
stops mattering and the parallel columnar execution takes over. This single row
of the table, read across both scales, is the whole argument about when to reach
for Polars.

*Reading and grouping favour Polars even at 6,497 rows* (3.4x and 4.1x). Unlike
the filter, these do substantial per-column work regardless of row count -- CSV
parsing means tokenising and type-inferring every field, and a group-by means
hashing keys and maintaining accumulators. Polars is written in Rust, stores
data in Apache Arrow columnar format, and parallelises across cores by default,
so there is enough work here for that to pay off immediately. Pandas is largely
single-threaded.

*The group-by advantage widens with size*, from 4.1x to 8.45x, which is the
expected direction: more rows per group means more work to parallelise.

**A caveat on precision.** Running the benchmark twice on the same machine gave
3.22x and 8.45x for the scaled group-by. Timings on a laptop compete with
background processes, thermal throttling and cache state; the median over 15
runs suppresses outliers but does not make the number reproducible to two
decimal places. These results support "Polars is several times faster on this
workload at this scale", not any exact multiplier.

**Practical conclusion.** For a 6,497-row dataset the entire pipeline runs in
under 10 ms either way, so performance is not a reason to choose between them --
the deciding factors at this scale are API ergonomics and the expression system.
The scaling behaviour is what would justify Polars on a larger project.

## Lazy execution

`analysis_polars.py` also times `scan_csv(...).filter(...).group_by(...)
.collect()` on the scaled file: **75.4 ms**, against 139.988 ms for Polars just
to *read* the same file eagerly. The full pipeline finishing in roughly half the
time of one of its own steps is the point: in lazy mode Polars builds a query
plan before executing anything, sees that a filter follows the scan, and pushes
the predicate down into the CSV reader, so the ~693,000 rows that fail
`alcohol > 11` are never fully materialised. This is what a query planner does in
a database, and it is where the gap over eager Pandas is widest.

Note that the lazy pipeline's group-by output shows different counts and means
from the eager benchmark above -- that is expected, not a discrepancy: the lazy
pipeline applies the `alcohol > 11` filter before grouping, while the eager
group-by benchmark groups the unfiltered table.

