# Analyze Parquet, Arrow IPC and tables larger than memory

An analysis can start from a file instead of an in-memory table. Antecedent reads
Apache Parquet files, directories of Parquet files that share one schema, Arrow IPC
files (Feather version 2) and Arrow IPC streams. Opening a file reads its schema and
footer only; the rows are read later, in bounded chunks, and only for the columns an
analysis needs.

## Open a file

```python
import antecedent as ant

orders = ant.data.from_parquet("orders.parquet")          # or a directory of .parquet files
events = ant.data.from_arrow_ipc("events.arrow")          # .arrows is read as a stream
print(orders.schema)       # name, kind, nullability and Arrow type of every column
print(orders.num_rows)     # from the Parquet footer; None for Arrow IPC

result = ant.analyze(orders, query=ant.AverageEffect("t", "y"), graph=[("z", "t"), ("z", "y"), ("t", "y")])
```

`analyze`, `prepare`, and `ant.rebind(study, data=...)`
also take the path itself (`"orders.parquet"`, a `pathlib.Path`, or a directory). A path
whose extension is not `.parquet`, `.parq`, `.pq`, `.arrow`, `.feather`, `.ipc` or
`.arrows`, and that is not a directory, is refused with `reason_code="invalid_argument"`.
Arrow C streams (`__arrow_c_stream__` exporters such as PyArrow, Polars and DuckDB
tables) are accepted as before.

The handle is a read-only mapping of column name to array. Indexing one name reads that
column; `read(columns)` reads several in one pass. `from_parquet(path, columns=[...])`
hides every other column.

## Which columns are read

A read is planned from the required-column closure: the columns the scientific target
names together with every validator, support, weight, split and complete-case column.
Only those columns are decoded. A projection that would drop one of them is refused
before any byte is read, naming the column to retain. With an edge list or a `Dag`, an
analysis reads the graph's nodes and the columns the query and estimator configuration
name; other columns, including nested ones, are never decoded.

## Types are kept exactly

| File column | Read as | Missing cells |
| --- | --- | --- |
| float64, float32, float16 | float64 (narrower floats widen exactly) | `NaN`, invalid |
| any integer width | int64; an unsigned value above `2**63 - 1` is refused | `None` in an object array |
| boolean | bool | `None` in an object array |
| timestamp (any unit, any time zone), date32, date64 | nanoseconds (`datetime64[ns]`); an overflow is refused | `NaT` |
| dictionary of strings or integers, string | category labels | `None` |
| decimal, binary, list, struct, map, union | refused with the next attempt | |

A dictionary keeps its order as the category domain, unused levels included, and an
ordered dictionary keeps its order flag; an ordered dictionary that changes between
record batches is refused. String columns take their levels in order of first
appearance. Null cells are validity masks: a missing float cell is invalid and holds
`NaN`, which is what the [missing-data policies](missing-data.md) read. Integers above
`2**53` still refuse the numeric ingest of an analysis, as they do for in-memory data.

## Memory: chunks, quotes and refusals

Each chunk (65,536 rows by default; `chunk_rows=`) is charged to the admission pool
before it is decoded and again before its library-owned columns are built.
`execution=Execution(limits=ResourceLimits(memory_bytes=...))` bounds that pool, and the execution's `cancel` token stops a read. When the pool cannot hold one chunk the read is
refused with the bytes needed, the bytes available and the next attempt (a smaller
chunk, or a larger pool); nothing stays charged. Arrow IPC record batches keep the
size their writer chose, so a batch larger than the chunk quote is charged as soon as
it is decoded and refused when the pool cannot hold it.

A route that needs the whole table resident is quoted first: `quote(columns)` returns
the bytes of the resident columns, their masks, one chunk of scratch and the copy that
seals each column. When the pool is smaller, the read is refused before reading with
`CausalResourceError` whose `remedy` names the two next attempts: stream a
sufficient-statistics summary route, or raise the pool.

## Tables larger than memory

`summarize_weighted_gaussian` streams a file through the bounded weighted Gaussian /
OLS capacity route. It holds one chunk at a time, reads the file twice (cross-products,
then diagnostics), refuses a file whose rows changed between the passes, and returns the
weighted coefficients, the `HC0` covariance, the cross-products `X'WX`, `X'Wy`, `y'Wy`,
and leverage and residual checks:

```python
summary = orders.summarize_weighted_gaussian(["t", "z"], "y", weight="w", batch_rows=65_536)
summary.coefficients          # intercept first
summary.robust_covariance     # HC0
```

Covariates, response and weight must be float, integer or boolean columns. A missing
value is refused with its column and row unless `missing="complete_case"`, which leaves
the row out and counts it in `rows_dropped`. From Rust the same route is
`antecedent::analysis::file_table::summarize_weighted_gaussian`.

## Row ids above `2**32`

Rows are addressed by checked 64-bit global ids, so a table may hold more than
`2**32` rows; a row is narrowed to a 32-bit offset only inside its own bounded chunk,
and an id that does not fit is refused rather than truncated.

## Identity

Every read computes the full-content identity as it streams (`content_identity()`).
It covers the row count, the column names and kinds, the category domains and every
value. It does not depend on the chunk size, the row-group layout or the container (a
Parquet file and an Arrow IPC file with the same rows have one identity), and any
changed value changes it. It is issued only after a complete first scan.

## What this does not change

Reading a large table does not extend a route's calibrated range. Intervals are
calibrated up to 100,000 rows for the measured constructions; larger tables run and
report `scope_not_assessed` (see
[what the routes do not do](../user-facing-limits.md#sample-size-series-length-and-the-measured-range)).
Polars is not a backend; direct Arrow remains the preparation path.
