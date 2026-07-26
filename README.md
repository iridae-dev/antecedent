# Antecedent

[![CI](https://github.com/iridae-dev/antecedent/actions/workflows/ci.yml/badge.svg)](https://github.com/iridae-dev/antecedent/actions/workflows/ci.yml) [![Crates.io](https://img.shields.io/crates/v/antecedent)](https://crates.io/crates/antecedent) [![PyPI](https://img.shields.io/pypi/v/antecedent)](https://pypi.org/project/antecedent/) [![GitHub Release](https://img.shields.io/github/v/release/iridae-dev/antecedent)](https://github.com/iridae-dev/antecedent/releases/latest) [![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21556247-blue)](https://doi.org/10.5281/zenodo.21556247)

Antecedent is a causal inference engine written in Rust with a first-class Python API. It is built for **causal inference under structural uncertainty**. 

* **One engine, whole workflow.** Discovery, identification, estimation, Bayesian inference, interventions, counterfactuals, attribution, validation, and experimental design share one API and one set of guarantees — assumptions are not lost at the seams between libraries.
* Discovered structure — a CPDAG, a PAG, a posterior over graphs — is treated as evidence about the causal graph, and that uncertainty is propagated throughout estimation.
* **Temporal and online analysis.** Temporal graphs with their own semantics, PCMCI-family discovery, temporal identification and estimation, and incremental `CausalState` for streaming workflows.

## Try it!

For the fastest introduction, we've written some notebooks that you can open and run in Google Colab to see how Antecedent works:

* [Paid-search attribution](examples/notebooks/marketing_channel_structural_uncertainty.ipynb) See how a naive marketing dashboard can materially overstate paid-search impact by crediting the campaign for demand that would have existed anyway. Antecedent adjusts for market demand and produces a decision-ready estimate of incremental pipeline.
* [Campaign evidence transfer](examples/notebooks/sales_campaign_prior_transfer.ipynb) Use evidence from a previous sales campaign without assuming the new campaign is identical. Antecedent transfers the historical treatment-effect posterior into a different target model, then lets current data update it.
* [Marketing experiment design](examples/notebooks/marketing_experiment_design.ipynb) Compare a holdout experiment, better intent data and additional CRM records to determine which investment actually resolves the causal question. Antecedent identifies the best feasible action under a £40,000 budget and shows why collecting more of the same data would not fix the attribution problem.

We also have a library of examples in both Rust and Python in [`examples/`](examples/).

## Capabilities

The full inventory — every graph class, algorithm, estimator, refuter, and
mechanism — lives in [docs/capabilities.md](docs/capabilities.md). The highlights:

* **Graphs.** DAG, ADMG, CPDAG, PAG, and their temporal variants, with
  d-/m-separation, latent projection, and Markov-equivalence operations.
  Static and temporal semantics are distinct. Interchange via NetworkX, DOT,
  JSON, GML, and versioned CBOR.
* **Discovery.** PC, FCI, RFCI, GES, DirectLiNGAM, NOTEARS; the temporal
  PCMCI family (PCMCI, PCMCI+, LPCMCI, J-PCMCI+, regime-specific RPCMCI);
  Bayesian structure posteriors that propagate into downstream effect
  analyses; discovery stability validators.
* **Identification.** Backdoor, front-door, IV, sharp RD, ID/IDC on DAGs and
  ADMGs, generalized adjustment for partial graphs, and temporal strategies.
  Every query is reported as identified, partially identified,
  graph-dependent, or not identified.
* **Estimation.** Regression, g-computation, IPW, matching, AIPW, 2SLS, RD,
  and temporal estimators on the frequentist side; Bayesian g-computation,
  HMC GLMs, prior transfer, and graph-by-effect posterior envelopes on the
  Bayesian side.
* **Interventions and counterfactuals.** An SCM layer with hard, soft,
  stochastic, sequenced, and policy interventions; abduction–action–prediction
  counterfactuals, nested counterfactuals, and temporal trajectories.
* **Attribution and diagnostics.** Anomaly, distribution-shift, change-point,
  and unit-level attribution; Shapley-based root-cause ranking.
* **Validation and sensitivity.** Placebo, common-cause, bootstrap, and
  data-subset refuters; overlap diagnostics; E-values; linear through
  nonparametric sensitivity; Bayesian predictive checks.
* **Experimental design.** Rank measure/intervene/observe actions by expected
  information gain, probability of identification, or decision utility.
* **Incremental state.** `CausalState` for online workflows: streaming
  sufficient statistics, particle filters, prepared analyses, and explicit
  invalidation that never silently reruns an analysis.
* **Data and artifacts.** NumPy, pandas, and Arrow in Python; tabular,
  time-series, panel, and multi-environment data; schema-versioned CBOR
  artifacts with memory-mapped access.

## Scientific scope

Antecedent follows several explicit constraints:

1. Priors do not upgrade nonparametric identification.
2. Discovery results are not assumed to be ground truth.
3. Static and temporal graph semantics are not interchangeable.
4. Unidentified graph-posterior mass is preserved.
5. Partial graphs are not silently completed.
6. PAG-native full ID and IDC are not claimed.
7. Unsupervised regime discovery is outside the RPCMCI workflow.

## Platform support

Python wheels cover CPython 3.11–3.14 on Linux, macOS, and Windows:
`pip install antecedent` (PyPI; also GitHub Release wheels). The scientific
engine and native API are written in Rust: `cargo add antecedent` (crates.io) —
see [docs/development.md](docs/development.md). No other language bindings are
currently provided.

## Documentation

- [Full capabilities](docs/capabilities.md)
- [Architecture](docs/architecture.md) 
- [Comparison with DoWhy, EconML, Tigramite, causal-learn](docs/comparison.md)

- Rust docs and API are on [docs.rs](https://docs.rs/antecedent/latest/antecedent/)
- Docs and the Python API reference are on
[Read the Docs](https://antecedent.readthedocs.io/) ([Python API](https://antecedent.readthedocs.io/en/latest/python/antecedent.html));
the Rust API is on [docs.rs/antecedent](https://docs.rs/antecedent). 

## Citation

If you use Antecedent in research, please cite it. Citation metadata is in
[`CITATION.cff`](CITATION.cff); the archived release is
[doi:10.5281/zenodo.21556247](https://doi.org/10.5281/zenodo.21556247).

```bibtex
@software{hinshaw_antecedent_2026,
  author       = {Hinshaw, Charles and Antecedent Contributors},
  title        = {Antecedent},
  version      = {0.3.0},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.21556247},
  url          = {https://doi.org/10.5281/zenodo.21556247}
}
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). DCO sign-off required.

Also: [Development](docs/development.md) · [API naming](docs/api_naming.md) · [ADRs](adr/README.md)

## License

MIT OR Apache-2.0 — see `LICENSE-MIT` and `LICENSE-APACHE`.

