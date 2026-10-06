# Sparse and columnar ingest evaluation (2.2 E7)

**Decision (pending the measurement below): dense fallback retained, evaluation recorded.**
No library code changes. The accepted schema, the estimand and the single design builder are
untouched; a CSC pattern exists only inside the one-off evaluation program and nothing in the
library consumes it (no second panel builder).

## Question

For a wide, mostly-zero binary adjustment set (about 3,000 columns), how much does the dense
path copy and allocate to build a design matrix, and how much would a CSC-style
representation save for that build, without changing what is estimated?

## The dense path as it exists

`TabularData` stores every column as dense `f64` (a `Boolean` column type exists, but
`float64_masked` and the propensity preparer read `Float64` only). For a propensity/AIPW fit,
`propensity::prepare` (`crates/antecedent-estimate/src/propensity/prepare.rs`) does, per
adjustment column:

1. `data.float64_masked(z, &row_mask)`: one masked copy of the column (`n` `f64`);
2. a copy of that copy into the packed column-major design (`n x (1 + p)`);
3. `Arc::from(col)`: the column kept as its own owned slice;

and finally `Arc::from(design)`, which copies the packed design once more. For `n` rows and
`p` columns that is about `4 n p` `f64` written and about `3 n p` `f64` live at the peak, on
top of the `n p` `f64` of the stored table. For 5,000 rows and 3,000 columns, `n p * 8` bytes
is 114.4 MiB per copy by arithmetic.

A CSC pattern of a binary block stores one `u32` row index per 1 and one offset per column:
`4 nnz + 8 (p + 1)` bytes, 1.2 MB at 2% density for the same shape, again by arithmetic.

## Measurement

One command, a few seconds on a laptop (release build; defaults 5,000 rows, 3,000 columns,
density 0.02):

```sh
cargo run --release -p antecedent-data --example sparse_ingest_eval
# optional: -- ROWS COLS DENSITY, e.g. -- 5000 3000 0.02
```

The program (`crates/antecedent-data/examples/sparse_ingest_eval.rs`) builds a synthetic
`TabularData` with `COLS` binary columns, a binary treatment and an outcome, then reports
median-of-three wall time, peak live bytes above the starting level and cumulative bytes
allocated (a counting global allocator) for:

* the dense design build, written exactly as `propensity::prepare` does it over the public
  data API;
* a CSC pattern built straight from the stored columns (one pass per column, no per-column
  `f64` copy);
* the CSC-to-dense scatter a dense consumer would need;
* the first use, `X'y`, from each representation.

It asserts that the scattered CSC reproduces the dense design bit for bit and that `X'y` agrees
exactly, so both paths describe one estimand.

### Results

Measured on the development laptop (release build, one run; wall ms, peak MiB and allocated MiB
per step; `nnz`, retained dense and CSC MiB; the build speed-up and retained-memory ratio).
Numbers vary by machine; rerun the command above to reproduce:

```text
rows=5000 binary_columns=3000 target_density=0.02

resident TabularData (accepted dense schema, unchanged): 117.0 MiB (15010000 f64 cells)

nnz=300219 measured_density=0.0200; retained after build: dense 228.9 MiB, CSC 1.168 MiB

dense design build (current path)                   40.5 ms      343.6 MiB peak      458.0 MiB allocated
CSC pattern build from stored columns               34.1 ms        3.0 MiB peak        4.0 MiB allocated
CSC -> dense materialization (dense consumer)        2.0 ms      114.5 MiB peak      114.5 MiB allocated

first use X'y, dense                                13.7 ms        0.0 MiB peak        0.0 MiB allocated
first use X'y, CSC                                   0.2 ms        0.0 MiB peak        0.0 MiB allocated

Dense X'y == CSC X'y exactly (3000 columns). Build speed-up CSC vs dense: 1.2x; retained-memory ratio dense/CSC: 196x.
```

## How the result is read

* The resident `TabularData` stays dense in either case: the accepted schema is dense
  `Float64`, and changing it would change what the library accepts. A CSC pattern could only
  be derived from those columns, so it saves the **per-fit copies and the retained design**,
  not the table.
* The estimators and learners consume a dense column-major `DesignView`. A fit through a
  CSC pattern needs either a sparse-aware solver for every estimator and learner (a second
  ML stack, out of scope) or a scatter back to dense at the fit, which makes the fit's own
  peak dense again. The measured scatter row is that cost.
* A change would be justified only if the measured build and retained-memory savings are
  large **and** can be taken without a second builder or a changed schema (for example
  dropping the redundant `Arc::from(col)` copies of columns that the packed design already
  holds). If the numbers show a clear, local, estimand-neutral saving of that kind, it is
  recorded here and made as a small change on the existing builder; otherwise the dense
  fallback stays.

## Status

Measured and recorded; **the dense path is retained and the library is unchanged.**

* The resident table is 117 MiB dense in both cases, because the accepted schema is dense
  `Float64` and stays so.
* A CSC pattern keeps 1.2 MiB where the dense design keeps 229 MiB (196x), and building it is
  only 1.2x faster than the current build. The saving is in what is *retained* between fits,
  not in build time.
* Every estimator and learner consumes a dense column-major design, so a fit scatters the
  pattern back to dense (114.5 MiB peak for 5,000 rows by 3,000 columns at 2 percent
  density). The peak of a fit is therefore dense either way; only a sparse-aware solver for
  each estimator would remove it, and that is a second ML stack outside this scope.
* The numbers describe one synthetic shape on one machine. They do not show a local,
  estimand-neutral change to the existing builder that is worth making, so none was made.
