# antecedent-model

Probabilistic and structural causal models. DAGs compile to topological execution plans;
sampling applies interventions as overlays rather than cloning the model. The crate also holds
mechanism families and their held-out selection, noise replay, do-samplers and model
evaluation.

Most users want the [`antecedent`](https://crates.io/crates/antecedent) facade, which fits models through `Simulate` and `Attribute` study queries and re-exports the types under `antecedent::gcm`. Depend on this crate directly only to fit and sample a model outside a study.

Main types: `ProbabilisticCausalModel`, `InvertibleStructuralCausalModel`, `CompiledCausalModel`, `InterventionOverlay`, `ModelEvaluator`. API reference: <https://docs.rs/antecedent-model>.

## License

MIT OR Apache-2.0.
