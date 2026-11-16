# antecedent-data

Library-owned views of the data a study reads: tabular, time-series, panel, event,
multi-environment and network data, typed and categorical columns, and missingness
declarations. Arrow is an optional adapter backend (the default `arrow` feature); Arrow types
never appear in the causal API ([ADR 0004](https://github.com/iridae-dev/antecedent/blob/v3.0.0/adr/0004-data-api.md)).

The optional `parquet` and `ipc` features add `file_source`: Parquet files and datasets, and
Arrow IPC files and streams, read in bounded chunks under the admission pool, projected to the
columns the analysis needs, with a streaming content identity. The `antecedent` facade and the
Python extension enable both.

Most users want the [`antecedent`](https://crates.io/crates/antecedent) facade, which re-exports the data types in `antecedent::prelude` and under `antecedent::data`. Depend on this crate directly only to prepare or validate data without the analysis layers.

Main types: `TabularData`, `TimeSeriesData`, `PanelData`, `EventData`, `MultiEnvironmentData`, `NetworkData`. API reference: <https://docs.rs/antecedent-data>.

## License

MIT OR Apache-2.0.
