# antecedent-prob

Probability distributions, columnar posterior storage, prior specifications, weighted graph
samples and inference backends (Laplace and HMC GLMs, conjugate models). Draws are stored
by column and in batches, never one heap object per draw.

Most users want the [`antecedent`](https://crates.io/crates/antecedent) facade, which configures Bayesian studies through `antecedent::BayesianConfig` and re-exports these types under `antecedent::inference`. Depend on this crate directly only to work with posteriors or priors outside a study.

Main types: `PriorSet`, `PriorSpec`, `PosteriorDraws`, `PosteriorSummary`, `WeightedGraphSamples`. API reference: <https://docs.rs/antecedent-prob>.

## License

MIT OR Apache-2.0.
