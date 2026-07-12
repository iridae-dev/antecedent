# causal-library

A Rust causal computation library with Python bindings.

The library implements causal discovery, identification, estimation, structural
causal models, counterfactuals, attribution, and related primitives. It targets
functional parity with DoWhy (excluding EconML) and Tigramite, plus a
Bayesian-first extension that preserves frequentist parity.

This is a library and Python extension, not a hosted service, workflow system,
dashboard, or deployment platform.

## Design

See [DESIGN.md](DESIGN.md) for the technical design, phase plan, and
non-negotiable rules. Accepted architecture decisions live in [`adr/`](adr/).

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE)), or
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Contributions require Developer
Certificate of Origin (DCO) sign-off.
