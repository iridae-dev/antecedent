# antecedent-stats

Statistical algorithms and the `faer` dense linear-algebra backend
([ADR 0001](https://github.com/iridae-dev/antecedent/blob/v3.0.0/adr/0001-linear-algebra-backend.md)): least squares and GLMs, M-estimation,
conditional-independence tests, FDR control, cluster and covariance estimators, matching
indexes, survival fits and identified-set utilities.

Most users want the [`antecedent`](https://crates.io/crates/antecedent) facade, which re-exports the test and GLM types it exposes under `antecedent::inference`. Depend on this crate directly only to use a statistical routine on its own.

Main types: `FaerBackend`, `LeastSquaresFit`, `MatchingIndex`, `StatsError`. API reference: <https://docs.rs/antecedent-stats>.

## License

MIT OR Apache-2.0.
