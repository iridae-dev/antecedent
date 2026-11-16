# antecedent-validate

Effect refuters and validation diagnostics: placebo treatment, random common cause, dummy
outcome, bootstrap and data-subset refuters, unobserved-common-cause and E-value sensitivity
(linear through nonparametric, including Riesz-based checks for the binary average effect),
overlap diagnostics and rules, discovery-stability validators, Bayesian predictive checks and
composable refutation suites.

Refuters report applicability explicitly: a validator that does not apply to an estimator
returns `NotApplicable`, not a vacuous pass.

Most users want the [`antecedent`](https://crates.io/crates/antecedent) facade, which runs refutation suites on a study's estimate and re-exports the types under `antecedent::validate`. Depend on this crate directly only to run a refuter on an estimate produced elsewhere.

Main types: `PlaceboTreatment`, `RandomCommonCause`, `DummyOutcome`, `EValue`, `OverlapRefuter`, `ValidationError`. API reference: <https://docs.rs/antecedent-validate>.

## License

MIT OR Apache-2.0.
