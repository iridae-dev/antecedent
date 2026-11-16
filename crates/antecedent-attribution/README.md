# antecedent-attribution

Attribution under a fitted causal model: anomaly scores, distribution-, unit- and
mechanism-change attribution, Shapley and path decompositions, intrinsic causal influence,
feature relevance and graph-sensitive root-cause ranking.

Most users want the [`antecedent`](https://crates.io/crates/antecedent) facade, which runs these as `Attribute` queries on a study and re-exports the types under `antecedent::gcm`. Depend on this crate directly only to call an attribution engine on a model you fitted yourself.

Main types: `ChangeAttribution`, `ChangeAttributionResult`, `AnomalyScores`, `ShapleyEstimate`, `RootCauseRank`. API reference: <https://docs.rs/antecedent-attribution>.

## License

MIT OR Apache-2.0.
