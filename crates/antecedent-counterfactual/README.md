# antecedent-counterfactual

Unit-level counterfactuals by abduction, action and prediction over invertible structural
causal models. Worlds share immutable model structure and apply interventions as overlays
instead of copying the model.

Most users want the [`antecedent`](https://crates.io/crates/antecedent) facade, which runs counterfactual queries as compiled study routes and re-exports the types under `antecedent::gcm`. Depend on this crate directly only to evaluate counterfactuals on a model you compiled yourself.

Main types: `CounterfactualEngine`, `CounterfactualWorld`, `CounterfactualResult`, `CompiledCounterfactualPlan`. API reference: <https://docs.rs/antecedent-counterfactual>.

## License

MIT OR Apache-2.0.
