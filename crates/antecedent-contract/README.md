# antecedent-contract

The contract layer above `antecedent-core`: the portable claim envelope and its composition
rules, external provider contracts with their verification and binding to a checked program,
evidence obligations, composition provenance, the reasoning view, the capability report and
the physical-plan records. No estimator or identification crate depends on it.

Most users want the [`antecedent`](https://crates.io/crates/antecedent) facade, which re-exports the public names under `antecedent::contract` and reports a contract with every prepared study. Depend on this crate directly only to read or produce claim and provider records without the rest of the library.

Main types: the claim envelope, the capability report and the physical-plan records. API reference: <https://docs.rs/antecedent-contract>.

## License

MIT OR Apache-2.0.
