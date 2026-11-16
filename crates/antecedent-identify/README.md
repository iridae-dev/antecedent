# antecedent-identify

Identification algorithms: backdoor, efficient, front-door, instrumental-variable and sharp
regression-discontinuity identifiers; generalized adjustment envelopes; temporal pulse,
sustained and mediation identification; the Shpitser-Pearl ID and IDC algorithms
(`IdIdentifier`, `IdcIdentifier`, `AutoIdentifier`); path-specific effects; sharp binary-IV
Balke-Pearl bounds; pairwise backdoor for continuous-response functionals; and a certified
subset of selection-diagram transport, where an unsupported case returns `NotCertified`
rather than a false claim of non-transportability.

Most users want the [`antecedent`](https://crates.io/crates/antecedent) facade, which identifies every study before estimation and re-exports the results under `antecedent::identify`. Depend on this crate directly only to call an identifier on a graph and query without a study.

Main types: `BackdoorIdentifier`, `FrontDoorIdentifier`, `IdIdentifier`, `IdcIdentifier`, `AutoIdentifier`, `IdentificationResult`. API reference: <https://docs.rs/antecedent-identify>.

## License

MIT OR Apache-2.0.
