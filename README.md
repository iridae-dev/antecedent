# Antecedent

[![CI](https://github.com/iridae-dev/antecedent/actions/workflows/ci.yml/badge.svg)](https://github.com/iridae-dev/antecedent/actions/workflows/ci.yml) [![Crates.io](https://img.shields.io/crates/v/antecedent)](https://crates.io/crates/antecedent) [![PyPI](https://img.shields.io/pypi/v/antecedent)](https://pypi.org/project/antecedent/) [![GitHub Release](https://img.shields.io/github/v/release/iridae-dev/antecedent)](https://github.com/iridae-dev/antecedent/releases/latest) [![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21556247-blue)](https://doi.org/10.5281/zenodo.21556247)

A causal inference engine in Rust with a first-class Python API, built for **causal
inference under structural uncertainty** — including continuous causal responses,
not only binary contrasts.

* **One engine, explicit contracts.** Discovery, identification, estimation, Bayesian
  inference, interventions, counterfactuals, attribution, validation, and experimental
  design share one API. The support matrix says which analysis combinations are licensed;
  implemented primitives outside those cells do not inherit a blanket guarantee.
* **Structure is evidence, not ground truth.** A CPDAG, a PAG, a posterior over graphs —
  selected licensed effect paths preserve that uncertainty rather than silently treating a
  learned structure as ground truth. This is not available for every query family.
* **Responses are first-class.** Mean curves, derivatives, elasticities, and Jacobians
  keep structural identification, empirical support, and uncertainty kind as separate
  axes at their native APIs. Derivative query types are importable but have no licensed
  `analyze` cell in 0.9; observation, transport, and interference remain explicit stage
  contracts.
* **Temporal and online.** Temporal graphs with their own semantics, PCMCI-family
  discovery, temporal identification and estimation, and incremental `CausalState` for
  streaming.

## Try it

Notebooks runnable in Colab:

| Notebook | |
| ----- | ----- |
| [Paid-search attribution](examples/notebooks/marketing_channel_structural_uncertainty.ipynb) — a naive dashboard overstates paid search by crediting demand that would have existed anyway. Adjust for it and get a decision-ready estimate of incremental pipeline. | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/iridae-dev/antecedent/blob/main/examples/notebooks/marketing_channel_structural_uncertainty.ipynb) |
| [Campaign evidence transfer](examples/notebooks/sales_campaign_prior_transfer.ipynb) — reuse a previous campaign's treatment-effect posterior without assuming the new campaign is identical, then let current data update it. | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/iridae-dev/antecedent/blob/main/examples/notebooks/sales_campaign_prior_transfer.ipynb) |
| [Experiment design](examples/notebooks/marketing_experiment_design.ipynb) — holdout, better intent data, or more CRM records? Find the best feasible action under a £40,000 budget, and why more of the same data would not fix the attribution problem. | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/iridae-dev/antecedent/blob/main/examples/notebooks/marketing_experiment_design.ipynb) |
| [Continuous causal response](examples/notebooks/continuous_causal_response.ipynb) — estimate a nonlinear dose–response curve, local derivative, elasticity, and observed-law average derivative, reading identification, support, and uncertainty as separate axes. | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/iridae-dev/antecedent/blob/main/examples/notebooks/continuous_causal_response.ipynb) |
| [Pricing, availability, and latent demand](examples/notebooks/pricing_availability_latent_demand.ipynb) — inventory-limited sales are not demand; compare a naive observed-sales curve with an explicit censoring mechanism and the fail-closed boundary for observation-aware response. | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/iridae-dev/antecedent/blob/main/examples/notebooks/pricing_availability_latent_demand.ipynb) |

More Rust and Python examples in [`examples/`](examples/).

## Capabilities

Maintained inventory in [docs/capabilities.md](docs/capabilities.md). The highlights:

* **Graphs.** DAG, ADMG, CPDAG, PAG and temporal variants; d-/m-separation, latent
  projection, Markov-equivalence operations. Static and temporal semantics stay distinct.
  Interchange via NetworkX, DOT, JSON, GML, versioned CBOR.
* **Discovery.** PC, FCI, RFCI, GES, DirectLiNGAM, NOTEARS; the temporal PCMCI family
  (PCMCI, PCMCI+, LPCMCI, J-PCMCI+, regime-specific RPCMCI); Bayesian structure posteriors
  that propagate downstream; stability validators.
* **Identification.** Backdoor, front-door, IV, sharp RD, an explicitly incomplete ID/IDC
  subset on DAGs and ADMGs,
  generalized adjustment for partial graphs, temporal strategies, pairwise backdoor for
  continuous-response functionals, sharp binary-IV Balke–Pearl ATE bounds, and a sound
  certified subset of single-source selection-diagram transport. Every query comes back
  identified, partially identified, graph-dependent, not identified, or — for transport
  outside the certified subset — `NotCertified`.
* **Estimation.** Regression, g-computation, IPW, matching, AIPW, 2SLS, RD and temporal
  estimators; Kennedy-style doubly robust response curves, Riesz average derivatives,
  low-dimensional GAM Jacobians; Bayesian g-computation, HMC GLMs, prior transfer,
  graph-by-effect posterior envelopes.
* **Observation, transport, and interference.** Explicit complete, censored, truncated,
  and selected observation mechanisms (assumptions never inferred from columns);
  trial-to-target IPW/AIPW under selection diagrams; randomized interference with
  Horvitz–Thompson / Hájek contrasts. These stay stage APIs — they change what identifies
  the estimand and are not folded into ordinary `analyze` flags.
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
  multi-environment data; schema-versioned CBOR with memory-mapped access, including
  response artifact format 0.4 with migrations from 0.1, 0.2, and 0.3.

## Scientific scope

Constraints the library will not bend:

1. Priors do not upgrade nonparametric identification.
2. Discovery results are not assumed to be ground truth.
3. Static and temporal graph semantics are not interchangeable.
4. Unidentified graph-posterior mass is preserved.
5. Partial graphs are not silently completed.
6. PAG-native full ID and IDC are not claimed.
7. Unsupervised regime discovery is outside the RPCMCI workflow.
8. Observation mechanisms do not imply their identifying assumptions.
9. Multi-source meta-transport, cyclic/equilibrium models, and observational network
   interference are outside the current contract.
10. General multi-node sID recursion is not claimed; unsupported transport returns
    `NotCertified`, not a false non-transportability certificate.

## How this is verified

Traceability and behavioural verification are merge requirements here, not release-time
aspirations.

* **Provenance.** Every significant algorithm cites its scientific sources and records any
  upstream implementation consulted, and whether it was referenced directly or used only as
  a black-box comparator. Current upstream comparisons are black-box only.
  [`provenance/`](provenance/)
* **Conformance.** Selected outputs are checked on frozen, scoped fixtures. Pinned external
  producers include DoWhy, Tigramite, causal-learn, lingam, statsmodels, `bpbounds`, and a
  supported `causaleffect` transport subset; other clean-room fixtures test internal known
  truths. A fixture match is evidence for the fields and cases asserted by its consuming
  test, not whole-method parity or proof of correctness.
  [`conformance/`](conformance/)
* **Parity.** Named cross-language contracts are checked across the Rust and Python APIs;
  this is capability parity, not identical surface syntax or proof that every public path is
  paired.
  [`parity/`](parity/)

Pull requests and main-branch pushes run both test suites, linting, CodeQL, and the domain
gates covering conformance fixtures and cross-language parity. Release gates include
response-calibration and causal-artifact round-trip checks; the broader statistical suite
runs weekly (and can be dispatched before a release). It checks interval coverage against
declared finite-Monte-Carlo tolerances, null p-value calibration, and discovery
false-positive rates on specified simulated designs. Those checks do not certify universal
calibration.

This establishes algorithmic lineage, agreement on reference cases, and consistent
cross-language behaviour. It does not validate your causal assumptions, and implies no
endorsement by the referenced projects.

The 0.4.0 correctness audit found and fixed twenty-five defects — see
[the release notes](docs/release-notes/v0.4.0.md) for what they were and why they mattered.
The 0.7.0 temporal-response cut is described in
[docs/release-notes/v0.7.0.md](docs/release-notes/v0.7.0.md); the 0.6.0
contract cut in
[docs/release-notes/v0.6.0.md](docs/release-notes/v0.6.0.md).

## Install

```bash
pip install antecedent        # CPython 3.11–3.14, Linux/macOS/Windows
cargo add antecedent          # Rust 1.85+
```

Wheels are on PyPI and attached to each GitHub Release. No other language bindings are
provided. This branch is package version **0.9.0** (crates.io / PyPI publish on tag).

## Documentation

[Capabilities](docs/capabilities.md) · [Causal responses](docs/causal-responses.md) ·
[Transport and interference](docs/transport-interference.md) ·
[Architecture](docs/architecture.md) ·
[Comparison with DoWhy, EconML, Tigramite, causal-learn](docs/comparison.md) ·
[Roadmap](ROADMAP.md) · [Development](docs/development.md) ·
[API naming](docs/api_naming.md) · [ADRs](adr/README.md)

Narrative docs and the Python API reference are on
[Read the Docs](https://antecedent.readthedocs.io/)
([Python API](https://antecedent.readthedocs.io/en/latest/python/antecedent.html)); the Rust
API is on [docs.rs/antecedent](https://docs.rs/antecedent).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). DCO sign-off required.

## License

MIT OR Apache-2.0 — see `LICENSE-MIT` and `LICENSE-APACHE`.
