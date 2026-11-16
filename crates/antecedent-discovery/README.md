# antecedent-discovery

Static and temporal causal discovery. Static: PC, FCI and RFCI (returning a PAG), GES
(a CPDAG), DirectLiNGAM and NOTEARS (a DAG), and Bayesian graph posteriors (exact DAG
posterior, structure and order MCMC, CI-screened and DBN posteriors). Temporal: the PCMCI
family (PCMCI, PCMCI+, LPCMCI, J-PCMCI+, regime-specific RPCMCI) with compiled constraint
masks (`required`, `forbidden`, `tiers`), FDR control over the full MCI family, target-wise
parallelism and `TemporalGraphReview` output.

Discovery returns equivalence classes and posteriors, not a single guessed DAG; no
orientation is invented downstream.

Most users want the [`antecedent`](https://crates.io/crates/antecedent) facade, which runs discovery as a compiled study route and requires the result to be accepted before estimation. Depend on this crate directly only to run a search engine on its own; its output then carries no contract or claim.

Main types: `Pc`, `Fci`, `Ges`, `Pcmci`, `PcmciPlus`, `Lpcmci`, `ExactDagPosterior`, `StructureMcmc`. API reference: <https://docs.rs/antecedent-discovery>.

## License

MIT OR Apache-2.0.
