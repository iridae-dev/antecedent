# antecedent-estimate

Estimators for identified causal functionals. An estimator consumes an `IdentifiedEstimand`;
it never chooses confounders or asserts identifiability.

Frequentist estimators: linear and GLM adjustment, g-computation, propensity methods
(weighting, matching, AIPW), instrumental variables (Wald, 2SLS), the front-door plug-in and
its linear two-stage form, sharp regression discontinuity, and temporal adjustment, mediation
and prediction. Continuous causal responses (doubly robust dose-response curves, derivatives,
elasticities) are in `response`. Bayesian estimation covers g-computation, HMC GLMs, prior
transfer and graph-by-effect posterior envelopes.

Observation mechanisms (`observation`), trial-to-target transport (`transport`) and
randomized interference (`interference`) are explicit stage APIs: they change what identifies
the estimand and are never inferred from the data.

Standard errors are analytic or bootstrap; overlap diagnostics and clipping policies are
reported, never applied silently. The full estimator inventory is in
[docs/capabilities.md](https://github.com/iridae-dev/antecedent/blob/v3.0.0/docs/capabilities.md).

Most users want the [`antecedent`](https://crates.io/crates/antecedent) facade, which selects an estimator for the identified estimand and re-exports the types under `antecedent::estimate`. Depend on this crate directly only to run an estimator on an estimand you identified yourself.

Main types: `EffectEstimate`, `CausalPosterior`, `LinearAdjustmentAte`, `PropensityWeighting`, `AipwAte`, `DmlAte`. API reference: <https://docs.rs/antecedent-estimate>.

## License

MIT OR Apache-2.0.
