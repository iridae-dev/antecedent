# Antecedent

[![CI](https://github.com/iridae-dev/antecedent/actions/workflows/ci.yml/badge.svg)](https://github.com/iridae-dev/antecedent/actions/workflows/ci.yml) [![Crates.io](https://img.shields.io/crates/v/antecedent)](https://crates.io/crates/antecedent) [![PyPI](https://img.shields.io/pypi/v/antecedent)](https://pypi.org/project/antecedent/) [![GitHub Release](https://img.shields.io/github/v/release/iridae-dev/antecedent)](https://github.com/iridae-dev/antecedent/releases/latest) [![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21556247-blue)](https://doi.org/10.5281/zenodo.21556247)

Antecedent is an identification-first causal inference engine in Rust with a first-class Python API. It unifies causal discovery, structural uncertainty, identification, estimation, validation, interventions, temporal analysis, and durable artifacts in one typed workflow.

Give it data, a causal question, and a graph or discovery strategy. Antecedent determines what is identified, runs a compatible inference path, or refuses claims the available evidence does not warrant.

Results preserve assumptions, uncertainty, diagnostics, and provenance across scalar effects, response curves, interventional distributions, and temporal trajectories—so analyses can be reviewed, reused, and audited without changing their scientific meaning.

## Try it in Colab

Five decision-focused notebooks run without local setup:

| Notebook | Run |
| --- | --- |
| [Paid-search attribution](examples/notebooks/marketing_channel_structural_uncertainty.ipynb) — see how a naive dashboard can overstate paid-search impact by crediting the campaign for demand that would have existed anyway. | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/iridae-dev/antecedent/blob/main/examples/notebooks/marketing_channel_structural_uncertainty.ipynb) |
| [Campaign evidence transfer](examples/notebooks/sales_campaign_prior_transfer.ipynb) — reuse evidence from a previous campaign without assuming the new campaign is identical, then let current data update or contradict it. | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/iridae-dev/antecedent/blob/main/examples/notebooks/sales_campaign_prior_transfer.ipynb) |
| [Experiment design](examples/notebooks/marketing_experiment_design.ipynb) — compare a holdout, better intent data, and more CRM records to find the best feasible action under a £40,000 budget. | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/iridae-dev/antecedent/blob/main/examples/notebooks/marketing_experiment_design.ipynb) |
| [Continuous causal response](examples/notebooks/continuous_causal_response.ipynb) — estimate a nonlinear dose–response curve and examine identification, empirical support, and uncertainty as separate result axes. | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/iridae-dev/antecedent/blob/main/examples/notebooks/continuous_causal_response.ipynb) |
| [Pricing, availability, and latent demand](examples/notebooks/pricing_availability_latent_demand.ipynb) — compare observed sales with an explicit censoring mechanism and see the fail-closed boundary for observation-aware response. | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/iridae-dev/antecedent/blob/main/examples/notebooks/pricing_availability_latent_demand.ipynb) |

The [examples directory](examples/README.md) also contains paired Python and Rust workflows for discovery, propensity weighting, temporal response, Bayesian prior transfer, design ranking, incremental state, class-preserving CPDAG estimation, and end-to-end analysis.

## What Can I Do With Antecedent?

Most causal libraries center on an estimator, a graph, or a particular stage of analysis. Antecedent centers on the **causal analysis itself**.

The question, structural assumptions, identification status, uncertainty, estimation path, diagnostics, and provenance move through the system together. That lets Antecedent connect work that is often split across separate tools:

- causal discovery and graph review;
- identification before estimation;
- frequentist and Bayesian inference;
- scalar effects, distributions, response curves, and temporal trajectories;
- structural uncertainty from CPDAGs, PAGs, and graph posteriors;
- observation mechanisms, transport, interference, and interventions;
- durable artifacts with explicit contracts for the scientific fields they retain.

The point is not simply breadth. It is that distinctions established upstream remain meaningful downstream, so unearned certainty cannot creep in at the seams. Most choices in Antecedent come from that and it has become a central organizing principle: a scientific claim must not become stronger merely because context was dropped while moving through discovery, identification, estimation, serialization, or the Rust/Python boundary.

See [**Capabilities**](docs/capabilities.md) for the full inventory and [**Support Matrix**](docs/support-matrix.md) for the analysis combinations licensed in the current release.

Artifact contents depend on the payload. Scalar posterior exports retain draws
and identification metadata but omit the complete assumption and validation
ledger; keep the analysis result alongside them. See the
[artifact contracts](docs/artifacts.md#exporting-prepared-results).

## Epistemic Honesty

Antecedent does not rely on a single notion of “tested.” It triangulates its claims from several directions, then makes those constraints executable.

- [**External oracles**](parity/README.md) record pinned upstream implementations, exact generation recipes, frozen results, comparison rules, and consuming conformance tests. A fixture sitting on disk is not evidence until the code actually exercises it.
- [**Algorithm provenance**](provenance/README.md) records primary literature, upstream exposure, test sources, and implementation deviations… including where the implementation does *not* inherit the guarantees of the cited method.
- [**The support matrix**](docs/support-matrix.md) **is a license, not a feature list.** A capability existing somewhere in the codebase does not mean the public analysis workflow may use it. Each licensed combination carries an explicit evidence contract; everything else is typed impossible or refuses by default.
- **Evidence itself is typed.** Implementation tests, analytic known truths, internal cross-checks, frozen external oracles, behavioral parity, and stronger equivalence claims remain distinct so weaker evidence cannot quietly acquire a stronger label.

For example, consider an average treatment effect estimated with AIPW. The public analysis route is licensed only where Antecedent has evidence for the full causal contract. The legacy full-sample AIPW path has a provenance record that explicitly excludes cross-fitting guarantees. The 1.5 score path adds cross-fitting, whose inference still requires positivity and suitable nuisance convergence rates; the review ledger records its current evidence and gaps. The legacy path’s numerical behavior is separately checked against pinned external implementations through reproducible oracle fixtures, and repository gates verify that those fixtures are actually exercised by conformance tests. None of those facts is allowed to stand in for the others: an external match does not prove identification, a citation does not grant an unimplemented theorem, and code that happens to run does not make an analysis supported. If the requested combination falls outside the licensed contract, Antecedent refuses it instead.

**None of this proves Antecedent correct. Together, these mechanisms make accidental overclaiming harder, make unsupported claims fail closed, and leave a visible trail for someone trying to prove the software wrong.**

## Project status and documentation

The working package version is **1.5.0**. Its checklist is not yet complete.
Implemented additions include
retargetable prepared plans, exceedance functionals, cell-saturated joint
AIPW, and tier-background identification on the existing licensed cells. The
[1.5.0 release notes](docs/release-notes/v1.5.0.md) and
[evidence ledger](docs/v1.5-evidence.md) describe their implemented scope and remaining release requirements.

[Documentation](https://antecedent.readthedocs.io/) ·
[Python API](https://antecedent.readthedocs.io/en/latest/python/antecedent.html) ·
[Rust API](https://docs.rs/antecedent) ·
[Capabilities](docs/capabilities.md) ·
[Causal responses](docs/causal-responses.md) ·
[Transport and interference](docs/transport-interference.md) ·
[Architecture](docs/architecture.md) ·
[Artifacts](docs/artifacts.md) ·
[Comparison](docs/comparison.md) ·
[ADRs](adr/README.md) ·
[Development](docs/development.md)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Contributions require DCO sign-off.

## License

MIT OR Apache-2.0 — see [LICENSE-MIT](LICENSE-MIT) and [LICENSE-APACHE](LICENSE-APACHE).
