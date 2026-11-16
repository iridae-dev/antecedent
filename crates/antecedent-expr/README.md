# antecedent-expr

An arena-backed representation of identified causal functionals, independent of any
estimator: the identified estimand and its method tags, compiled evaluators, algebraic
simplification, exact discrete laws, and plain-text and LaTeX display.

Most users want the [`antecedent`](https://crates.io/crates/antecedent) facade, which re-exports `IdentifiedEstimand` in its prelude and the exact-law types under `antecedent::estimate`. Depend on this crate directly only to build or evaluate identified functionals without the estimators.

Main types: `IdentifiedEstimand`, `ExactDiscreteLaw`, `Assignment`. API reference: <https://docs.rs/antecedent-expr>.

## License

MIT OR Apache-2.0.
