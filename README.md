# Antecedent

[![CI](https://github.com/iridae-dev/antecedent/actions/workflows/ci.yml/badge.svg)](https://github.com/iridae-dev/antecedent/actions/workflows/ci.yml) [![Crates.io](https://img.shields.io/crates/v/antecedent)](https://crates.io/crates/antecedent) [![PyPI](https://img.shields.io/pypi/v/antecedent)](https://pypi.org/project/antecedent/) [![GitHub Release](https://img.shields.io/github/v/release/iridae-dev/antecedent)](https://github.com/iridae-dev/antecedent/releases/latest) [![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21556247-blue)](https://doi.org/10.5281/zenodo.21556247)

A causal inference engine in Rust with a first-class Python API, built for **causal
inference under structural uncertainty**.

* **One engine, whole workflow.** Discovery, identification, estimation, Bayesian
  inference, interventions, counterfactuals, attribution, validation, and experimental
  design share one API and one set of guarantees. Assumptions are not lost at the seams
  between libraries.
* **Structure is evidence, not ground truth.** A CPDAG, a PAG, a posterior over graphs —
  the uncertainty propagates through estimation rather than being resolved by assumption.
* **Temporal and online.** Temporal graphs with their own semantics, PCMCI-family
  discovery, temporal identification and estimation, and incremental `CausalState` for
  streaming.

## Try it

Three notebooks, runnable in Colab:

| Notebook | |
| ----- | ----- |
| [Paid-search attribution](examples/notebooks/marketing_channel_structural_uncertainty.ipynb) — a naive dashboard overstates paid search by crediting demand that would have existed anyway. Adjust for it and get a decision-ready estimate of incremental pipeline. | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/iridae-dev/antecedent/blob/main/examples/notebooks/marketing_channel_structural_uncertainty.ipynb) |
| [Campaign evidence transfer](examples/notebooks/sales_campaign_prior_transfer.ipynb) — reuse a previous campaign's treatment-effect posterior without assuming the new campaign is identical, then let current data update it. | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/iridae-dev/antecedent/blob/main/examples/notebooks/sales_campaign_prior_transfer.ipynb) |
| [Experiment design](examples/notebooks/marketing_experiment_design.ipynb) — holdout, better intent data, or more CRM records? Find the best feasible action under a £40,000 budget, and why more of the same data would not fix the attribution problem. | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/iridae-dev/antecedent/blob/main/examples/notebooks/marketing_experiment_design.ipynb) |

More Rust and Python examples in [`examples/`](examples/).

## Capabilities

Full inventory in [docs/capabilities.md](docs/capabilities.md). The highlights:

* **Graphs.** DAG, ADMG, CPDAG, PAG and temporal variants; d-/m-separation, latent
  projection, Markov-equivalence operations. Static and temporal semantics stay distinct.
  Interchange via NetworkX, DOT, JSON, GML, versioned CBOR.
* **Discovery.** PC, FCI, RFCI, GES, DirectLiNGAM, NOTEARS; the temporal PCMCI family
  (PCMCI, PCMCI+, LPCMCI, J-PCMCI+, regime-specific RPCMCI); Bayesian structure posteriors
  that propagate downstream; stability validators.
* **Identification.** Backdoor, front-door, IV, sharp RD, ID/IDC on DAGs and ADMGs,
  generalized adjustment for partial graphs, temporal strategies. Every query comes back
  identified, partially identified, graph-dependent, or not identified.
* **Estimation.** Regression, g-computation, IPW, matching, AIPW, 2SLS, RD and temporal
  estimators; Bayesian g-computation, HMC GLMs, prior transfer, graph-by-effect posterior
  envelopes.
* **Interventions and counterfactuals.** Hard, soft, stochastic, sequenced and policy
  interventions; abduction–action–prediction counterfactuals, nested counterfactuals,
  temporal trajectories.
* **Attribution.** Anomaly, distribution-shift, change-point and unit-level attribution;
  Shapley-based root-cause ranking.
* **Validation and sensitivity.** Placebo, common-cause, bootstrap and data-subset
  refuters; overlap diagnostics; E-values; linear through nonparametric sensitivity;
  Bayesian predictive checks.
* **Experimental design.** Rank measure/intervene/observe actions by expected information
  gain, probability of identification, or decision utility.
* **Incremental state.** `CausalState` for online work: streaming sufficient statistics,
  particle filters, prepared analyses, and invalidation that never silently reruns an
  analysis.
* **Data and artifacts.** NumPy, pandas and Arrow; tabular, time-series, panel and
  multi-environment data; schema-versioned CBOR with memory-mapped access.

## Scientific scope

Seven constraints the library will not bend:

1. Priors do not upgrade nonparametric identification.
2. Discovery results are not assumed to be ground truth.
3. Static and temporal graph semantics are not interchangeable.
4. Unidentified graph-posterior mass is preserved.
5. Partial graphs are not silently completed.
6. PAG-native full ID and IDC are not claimed.
7. Unsupervised regime discovery is outside the RPCMCI workflow.

## How this is verified

Traceability and behavioural verification are merge requirements here, not release-time
aspirations.

* **Provenance.** Every significant algorithm cites its scientific sources and records any
  upstream implementation consulted, and whether it was referenced directly or used only as
  a black-box comparator. Current upstream comparisons are black-box only.
  [`provenance/`](provenance/)
* **Conformance.** Implementations are output-verified on documented fixtures against
  reference libraries — DoWhy, scikit-learn, Tigramite where applicable — with no
  unexplained divergence permitted. [`conformance/`](conformance/)
* **Parity.** Applicable public behaviour stays in parity across the Rust and Python APIs.
  [`parity/`](parity/)

As of 0.4.0:

| | |
|---|---|
| Rust tests | 1323 |
| Python tests | 507 |
| Coverage floor | 85%, enforced in CI |
| Conformance fixtures | 130 documented pages |
| Platforms | CPython 3.11–3.14 on Linux, macOS, Windows; Rust 1.85+ |

Every commit runs both test suites, both lint gates, CodeQL, and the domain gates covering
conformance fixtures and cross-language parity. A scheduled gate runs what is too slow for
every commit: whether nominal 95% intervals actually cover 95%, whether null p-values are
actually uniform, and discovery false-positive rates.

This establishes algorithmic lineage, agreement on reference cases, and consistent
cross-language behaviour. It does not validate your causal assumptions, and implies no
endorsement by the referenced projects.

The 0.4.0 correctness audit found and fixed twenty-five defects — see
[the release notes](docs/release-notes/v0.4.0.md) for what they were and why they mattered.

## Install

```bash
pip install antecedent        # CPython 3.11–3.14, Linux/macOS/Windows
cargo add antecedent          # Rust 1.85+
```

Wheels are on PyPI and attached to each GitHub Release. No other language bindings are
provided.

## Documentation

[Capabilities](docs/capabilities.md) · [Architecture](docs/architecture.md) ·
[Comparison with DoWhy, EconML, Tigramite, causal-learn](docs/comparison.md) ·
[Development](docs/development.md) · [API naming](docs/api_naming.md) ·
[ADRs](adr/README.md)

Narrative docs and the Python API reference are on
[Read the Docs](https://antecedent.readthedocs.io/)
([Python API](https://antecedent.readthedocs.io/en/latest/python/antecedent.html)); the Rust
API is on [docs.rs/antecedent](https://docs.rs/antecedent).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). DCO sign-off required.

## License

MIT OR Apache-2.0 — see `LICENSE-MIT` and `LICENSE-APACHE`.
