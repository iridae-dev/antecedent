# Analyze Parquet, Arrow IPC and tables larger than memory

Use this guide when your data are in a file rather than an in-memory table, or when the
table does not fit in memory. Antecedent reads Apache Parquet files and directories of
Parquet files that share one schema. It also reads Arrow IPC files (Feather version 2) and
Arrow IPC streams. Opening a file reads its schema and footer only. The rows are read later,
in bounded chunks, and only for the columns an analysis needs. The estimate from a file is
the estimate from the same rows in memory. The file changes how rows are read. It does not
change what is estimated or assumed.

Runnable example: [`examples/python/parquet_e2e.py`](https://github.com/iridae-dev/antecedent/blob/v3.1.0/examples/python/parquet_e2e.py)
analyzes a Parquet file, compares the result with the in-memory one, and streams a
summary through a memory pool smaller than the table.

## Open a file

```python
import antecedent as ant

orders = ant.data.from_parquet("orders.parquet")          # or a directory of .parquet files
events = ant.data.from_arrow_ipc("events.arrow")          # .arrows is read as a stream
print(orders.schema)       # name, kind, nullability and Arrow type of every column
print(orders.num_rows)     # from the Parquet footer; None for Arrow IPC

result = ant.analyze(orders, query=ant.AverageEffect("t", "y"), graph=[("z", "t"), ("z", "y"), ("t", "y")])
```

These calls also take the path itself (`"orders.parquet"`, a `pathlib.Path`, or a directory):

- `analyze`
- `prepare`
- `ant.rebind(study, data=...)`

A path whose extension is not `.parquet`, `.parq`, `.pq`, `.arrow`, `.feather`, `.ipc` or
`.arrows`, and that is not a directory, is refused with `reason_code="invalid_argument"`.
The error is a `CausalValueError`. See [refusal codes and remedies](../reference/refusal-codes.md#structured-refusals-and-their-remedies).
Arrow C streams are accepted too. That is any `__arrow_c_stream__` exporter, such as a
PyArrow, Polars or DuckDB table.

The handle is a read-only mapping of column name to array. Indexing one name reads that
column. `read(columns)` reads several in one pass. `from_parquet(path, columns=[...])`
hides every other column. It refuses a name that is not in the file or is repeated.
`from_arrow_ipc(path, stream=True)` reads a stream whatever the extension.

## Which columns are read

A read uses the columns the scientific target names, together with every validator, support,
weight, split and complete-case column. Only those columns are decoded. A projection that
would drop one of them is refused before any byte is read. The refusal names the column to
retain. With an edge list or a `Dag`, an analysis reads the graph's nodes and the columns
the query and estimator configuration name. Other columns, including nested ones, are never
decoded.

## Check a column projection or a row filter against the cohort

When you read fewer columns, or push a row filter into your own scan, the result must be the same set of rows the analysis is entitled to. Those are the rows observed on every column the analysis reads. Call `antecedent.cohort_pushdown` before you rely on the cheaper read. A cell is observed when it is not `NaN`, `NaT` or `None`. `source` is any mapping of column name to array. A `TableFile` works. Only the columns the analysis reads, and the columns you keep, are read.

In the six-row table below, `y` is the target, `w` the weight and `x` a complete-case column; rows 0, 3 and 5 observe all three.

```python
import numpy as np
from antecedent import cohort_pushdown as cp

NAN = float("nan")
source = {
    "y": np.array([1.0, NAN, 3.0, 4.0, 5.0, 6.0]),
    "w": np.array([1.0, 1.0, 1.0, 1.0, NAN, 1.0]),
    "x": np.array([1.0, 2.0, NAN, 4.0, 5.0, 6.0]),
    "extra": np.array([NAN, 1.0, 1.0, 1.0, 1.0, 1.0]),
}
closure = cp.ColumnClosure(target=("y",), weights=("w",), complete_case=("x",))

cp.checked_projection(source, closure, ["y", "w", "x"])      # array([0, 3, 5])
drop = [False, True, True, False, True, False]
cp.checked_push_filter(source, closure, "extra", drop)      # array([0, 3, 5])
```

Both return the cohort's row indices. A projection that drops a required column raises `CausalValueError` and names the column. Keeping `extra` does the same: completing cases on it would also drop row 0. A filter that would drop a cohort row is refused and names the filter column. That column still decides membership. A mask that is not one entry per row is refused too. The check changes nothing about the analysis. It only tells you whether your cheaper read delivers the cohort.

## Types are kept exactly

| File column | Read as | Missing cells |
| --- | --- | --- |
| float64, float32, float16 | float64 (narrower floats widen exactly) | `NaN`, invalid |
| any integer width | int64; an unsigned value above `2**63 - 1` is refused | `None` in an object array |
| boolean | bool | `None` in an object array |
| timestamp (any unit, any time zone), date32, date64 | nanoseconds (`datetime64[ns]`); an overflow is refused | `NaT` |
| dictionary of strings or integers, string | category labels | `None` |
| decimal, binary, list, struct, map, union | refused with the next attempt | |

A dictionary keeps its order as the category domain, unused levels included. An ordered
dictionary keeps its order flag. An ordered dictionary that changes between record batches
is refused. String columns take their levels in order of first appearance. Null cells are
validity masks. A missing float cell is invalid and holds `NaN`, which is what the
[missing-data policies](missing-data.md) read. Integers above `2**53` still refuse the
numeric ingest of an analysis, as they do for in-memory data.

## Memory: chunks, quotes and refusals

Each chunk is 65,536 rows by default. Set `chunk_rows=` on `from_parquet` or
`from_arrow_ipc`. The chunk is charged to the admission pool before it is decoded and again
before its library-owned columns are built. Pass
`execution=Execution(limits=ResourceLimits(memory_bytes=...))` to bound that pool. The
execution's `cancel` token stops a read. When the pool cannot hold one chunk the read is
refused with the bytes needed, the bytes available and the next attempt. The next attempt is
a smaller chunk, or a larger pool. Nothing stays charged. Arrow IPC record batches keep the
size their writer chose. A batch larger than the chunk quote is charged as soon as it is
decoded. It is refused when the pool cannot hold it.

An analysis that needs the whole table resident is quoted first. Call `quote(columns)`. You
get a `MemoryQuote`: `total_bytes`, `resident_bytes`, `chunk_bytes` and `chunk_rows`. The
quote covers the resident columns, their masks, one chunk of scratch and the copy that seals
each column. When the pool is smaller, the read is refused before reading with
`CausalResourceError`. Its `remedy` names the two next attempts: stream a
sufficient-statistics summary, or raise the pool.

## Tables larger than memory

Call `summarize_weighted_gaussian` to stream a file through the bounded weighted Gaussian
and OLS summary. It holds one chunk at a time. It reads the file twice: cross-products,
then diagnostics. It refuses a file whose rows changed between the passes. You get the
weighted coefficients, the `HC0` covariance, the cross-products `X'WX`, `X'Wy` and `y'Wy`,
and leverage and residual checks:

```python
summary = orders.summarize_weighted_gaussian(["t", "z"], "y", weight="w", batch_rows=65_536)
summary.design_columns        # names in coefficient order
summary.coefficients          # intercept first
summary.robust_covariance     # HC0
summary.rows_used, summary.rows_dropped
```

The coefficients are a weighted least-squares regression of the response on the
covariates. They describe that regression. They are not an identified causal effect
unless you separately justify the adjustment set. An `analyze` call with a graph does
that. This summary does not. The `HC0` covariance is heteroskedasticity-consistent for
independent rows. The summary reports no interval and no calibration status.

Covariates, response and weight must be float, integer or boolean columns. A missing
value is refused with its column and row. `missing="refuse"` is the default. Pass
`missing="complete_case"` to leave the row out and count it in `rows_dropped`.
`intercept=False` omits the intercept. In Rust, call
`antecedent::analysis::file_table::summarize_weighted_gaussian`.

## Row ids above `2**32`

Rows are addressed by checked 64-bit global ids. A table may hold more than `2**32` rows.
A row is narrowed to a 32-bit offset only inside its own bounded chunk. An id that does not
fit is refused rather than truncated.

## Identity

Every read computes the full-content identity as it streams. Call `content_identity()`.
It covers the row count, the column names and kinds, the category domains and every
value. It does not depend on the chunk size, the row-group layout or the container. A
Parquet file and an Arrow IPC file with the same rows have one identity. Any changed value
changes it. It is issued only after a complete first scan.

## What this does not change

Reading a large table does not extend an analysis's calibrated range. An interval is
`calibrated` only when its coverage record's measured range contains the row count.
A larger table runs. Its intervals report `outside_measured_scope` (see
[what the routes do not do](../user-facing-limits.md#sample-size-series-length-and-the-measured-range)).
Polars is not a backend. Antecedent reads Arrow directly.
