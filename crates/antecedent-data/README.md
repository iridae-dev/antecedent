# antecedent-data

Stable library-owned tabular, temporal, and related data views. Arrow is an
optional adapter backend; Arrow crate types are not the public causal API
(ADR 0004).

Optional features `parquet` and `ipc` add `file_source`: Parquet files and
datasets, and Arrow IPC files and streams, read in bounded chunks under the
admission pool, projected to the required-column closure, with a streaming
content identity. The `antecedent` facade and the Python extension enable both.
