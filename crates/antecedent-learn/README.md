# antecedent-learn

The prediction contract for Antecedent's nuisance models: `DesignView`, `LearnerFactory` and
`FittedPredictor`. The crate does not identify effects or compute orthogonal scores; causal
estimation stays in `antecedent-estimate`
([ADR 0023](https://github.com/iridae-dev/antecedent/blob/v3.0.0/adr/0023-learner-substrate.md)).

Optional learners: `ml-gbdt` (gradient-boosted trees), `ml-forest` (random forests), `ml-full`
(both) and `ml-neural`, which enables `neural_net` through `antecedent-learn-burn` and is not
part of `ml-full`.

Most users want the [`antecedent`](https://crates.io/crates/antecedent) facade, which turns these learners on with the same feature names. Depend on this crate directly only to supply your own learner through `LearnerFactory`.

Main types: `DesignView`, `LearnerFactory`, `FittedPredictor`. API reference: <https://docs.rs/antecedent-learn>.

## License

MIT OR Apache-2.0.
