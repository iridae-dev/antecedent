# Bring your own table

Start with a small reviewed table and an explicit scientific question. The recipe below creates a temporary CSV so it can run without shipping a dataset; replace that path with your own file.

## Declare the meaning before estimating

| Input | Declare and check |
| --- | --- |
| Treatment | What contrast is being requested and how treatment is encoded. This recipe uses numeric 0/1. |
| Outcome | Units and observation mechanism; censoring and missingness require their own assumptions and supported routes. |
| Adjustment columns | Pre-treatment variables justified by the graph, not a convenient list of predictors. |
| Unit IDs | Stable observational or assignment units; a row index is not automatically a subject identity. |
| Target population | The default all-observed population or an explicit supported population declaration. |
| Design | For a randomized study, declare assignment probabilities and units through the experiment design API. |

<!-- example: examples/python/your_data.py -->

The estimate should be close to 2 in this simulated table. The recipe rejects missing required values rather than silently filling or dropping them. For your data, choose and document an observation policy before preparing the analysis; see [the observation contract](../observation-contract.md).

Use numeric columns for this recipe and inspect categorical encodings before generalizing it. See [Python workflow options](../python-options.md) for populations and data settings, and [preflight diagnostics](preflight-diagnostics.md) for fit-free checks.

## Columns the analysis reads

The table may hold more than the analysis uses. Strings, dates and identifier columns that the query, the graph and the estimator configuration do not name are never converted, so they cannot refuse the numeric ingest. Columns are chosen by name, never by position, and a pandas index is not exported as a column; a filtered frame with a non-default index analyses the same rows as a reset one.

- A `Dag` or an edge list binds columns by name: the analysis reads the graph's nodes and every column the query or estimator configuration names, in the table's order.
- An `Admg`, `Pag` or `Cpdag` binds columns by name and order: the analysis reads exactly its nodes, in the graph's order, wherever they sit in the table.
- With no graph (discovery), every column is read and must be numeric.
- Design queries (panel, staggered, synthetic-control and similar) read only the columns they name; their identifier columns keep their labels.
- `ant.rebind(study, data=new_data)` reduces a new table to the prepared columns by name, so extra columns and a different column order are accepted.

A required column that is missing, and a selected column name that occurs twice (in a pandas frame or an Arrow table), are refused with the column names. A duplicated name among columns the analysis does not read is ignored.

## Files and large tables

`analyze`, `prepare` and `ant.rebind(study, data=...)` also take a Parquet file, a directory of Parquet files, or an Arrow IPC file or stream, by path or as a handle from `antecedent.data.from_parquet` / `from_arrow_ipc`. Only the columns above are decoded, in bounded chunks charged to the admission pool; see [Parquet, Arrow IPC and large tables](large-tables.md).

## Cluster identifiers

`LinearAdjustment(se="cluster", cluster="site")` and `GlmAdjustment(se="cluster", cluster="site")` name the column of cluster labels instead of passing `cluster_ids`. The column may hold strings or whole numbers; a missing or empty label is refused. Labels are encoded in first-seen order over the rows of the table you pass, and the estimator keeps each label with its row: a row dropped for a missing treatment, outcome or adjustment value, or by a target-population predicate, takes its label with it. `cluster_ids`, by contrast, must already be aligned to the rows the estimator uses.

The prepared study freezes the labels and a digest of their row assignment, and the estimator identity digests them. `ant.rebind` refuses a table whose `site` column holds other labels or the same labels on other rows, before anything runs and with the previous result intact; call `ant.prepare(new_data, ...)` to declare new identifiers. The refusal carries a `receipt`: every row whose label moved (row, frozen label, supplied label), the labels a column of unique participants now repeats (a duplicate identifier on rebind), and the standing `recompute` of any claim computed under the frozen identifiers. `Aipw`, `FrontdoorLinearTwoStage` and, through `ClusterDml(cluster=...)`, `Aipw`, `DML` and `DRLearner` with `cluster_dml=` take named clusters the same way; other estimators take `cluster_ids` aligned to their analysis rows, or refuse a named cluster with the reason their variance does not read cluster labels.

`participant="pid"` on `prepare` / `analyze` names a column of participant labels (strings or whole numbers). `study.participants` returns the original label of every row of the prepared table; the labels are frozen with the study and `ant.rebind` refuses a table with other labels or another row count. `SampleSplit.participants(data, column)` returns the labels of the discovery and analysis rows by their original row ids and refuses a participant on both sides. Transport studies have no table rows and refuse `participant=`.

## Rebind or prepare again?

`ant.rebind` keeps the prepared scientific question and retained design. For named experimental assignments and IDs, supply outcome-only updates when the design is unchanged. Changed assignments, IDs, row order, graph or query require the corresponding explicit new preparation. [Notebook to system](notebook-to-system.md) demonstrates both operations and a refused design change.
