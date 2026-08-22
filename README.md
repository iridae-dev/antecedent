# Antecedent

[![CI](https://github.com/iridae-dev/antecedent/actions/workflows/ci.yml/badge.svg)](https://github.com/iridae-dev/antecedent/actions/workflows/ci.yml) [![Crates.io](https://img.shields.io/crates/v/antecedent)](https://crates.io/crates/antecedent) [![PyPI](https://img.shields.io/pypi/v/antecedent)](https://pypi.org/project/antecedent/) [![GitHub Release](https://img.shields.io/github/v/release/iridae-dev/antecedent)](https://github.com/iridae-dev/antecedent/releases/latest) [![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21556247-blue)](https://doi.org/10.5281/zenodo.21556247)

Antecedent is an identification-first causal inference engine in Rust with a
first-class Python API. It provides one workflow for causal discovery, graph
review, identification, frequentist and Bayesian estimation, validation,
interventions, temporal analysis, durable artifacts, and experimental design.

Give it data, a typed causal question, and either a graph or a discovery
strategy. Antecedent determines whether and how the query is identified, runs a
compatible estimator or posterior path, and returns a structured result. Questions
can be scalar effects, interventional distributions, path-specific effects,
continuous causal responses, or temporal effects and intervention trajectories.

It is designed for work where the graph may be uncertain, the estimand may be a
curve or trajectory rather than one number, and an analysis needs to survive
review, reuse, and serialization without changing its scientific meaning.

## Try it in Colab

Five decision-focused notebooks run without local setup:

| Notebook | |
| --- | --- |
| [Paid-search attribution](examples/notebooks/marketing_channel_structural_uncertainty.ipynb) — see how a naive dashboard can overstate paid-search impact by crediting the campaign for demand that would have existed anyway. | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/iridae-dev/antecedent/blob/main/examples/notebooks/marketing_channel_structural_uncertainty.ipynb) |
| [Campaign evidence transfer](examples/notebooks/sales_campaign_prior_transfer.ipynb) — reuse evidence from a previous campaign without assuming the new campaign is identical, then let current data update or contradict it. | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/iridae-dev/antecedent/blob/main/examples/notebooks/sales_campaign_prior_transfer.ipynb) |
| [Experiment design](examples/notebooks/marketing_experiment_design.ipynb) — compare a holdout, better intent data, and more CRM records to find the best feasible action under a £40,000 budget. | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/iridae-dev/antecedent/blob/main/examples/notebooks/marketing_experiment_design.ipynb) |
| [Continuous causal response](examples/notebooks/continuous_causal_response.ipynb) — estimate a nonlinear dose–response curve and examine identification, empirical support, and uncertainty as separate result axes. | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/iridae-dev/antecedent/blob/main/examples/notebooks/continuous_causal_response.ipynb) |
| [Pricing, availability, and latent demand](examples/notebooks/pricing_availability_latent_demand.ipynb) — compare observed sales with an explicit censoring mechanism and see the fail-closed boundary for observation-aware response. | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/iridae-dev/antecedent/blob/main/examples/notebooks/pricing_availability_latent_demand.ipynb) |

The [examples directory](examples/README.md) also contains paired Python and Rust
workflows for discovery, propensity weighting, temporal response, Bayesian prior
transfer, design ranking, incremental state, and end-to-end analysis.

## Scientific claims should not get stronger in transit

Its governing rule is simple: a scientific claim must not become stronger merely
because context was dropped while moving through discovery, identification,
estimation, serialization, or the Rust/Python boundary. Claims are typed, and
the evidence behind those claims is typed too.

When context cannot be preserved, the safe outcomes are a weaker status, an
explicit limitation, or a refusal—not promotion. “The model ran” is not an
adequate scientific conclusion.

```text
typed question + typed structural evidence
                    │
                    ▼
     identification status + assumptions
                    │
                    ▼
 estimate / posterior + support + uncertainty
                    │
                    ▼
      validation evidence + provenance
                    │
                    ▼
    API view / artifact without semantic upgrade
```

What makes Antecedent different:

- **Claim strength survives the workflow.** Nonparametric identification,
  identification under parametric or prior restrictions, partial identification,
  graph dependence, and non-identification are different states—not labels that
  may collapse to “identified” downstream.
- **Structure can remain uncertain.** DAGs, ADMGs, PAGs, accepted discovery
  results, and selected graph posteriors retain their distinct semantics.
  Unidentified posterior mass is not silently renormalized away.
- **Causal responses and time are native objects.** Antecedent models dose–response
  curves, intervention paths, and dose-by-horizon surfaces alongside ordinary
  treatment contrasts.
- **The result explains its own limits.** Identification status, estimator,
  diagnostics, validation, execution plan, and provenance remain inspectable.
  Python effect views expose assumption counts, response views expose assumption
  labels, and durable artifacts retain the full typed and scoped records.
  Response results also keep empirical support distinct from statistical
  uncertainty.
- **Refusal is part of the API.** Unsupported combinations fail as typed
  refusals instead of drifting into a nearby method with different assumptions.
- **Analysis can become a system.** Prepared analyses, accepted graph versions,
  incremental state, durable artifacts, prior transfer, and design ranking
  support repeated decisions—not only one-off notebook runs.

## Install

```bash
pip install antecedent        # CPython 3.11–3.14, Linux/macOS/Windows
cargo add antecedent          # Rust 1.85+
```

Python wheels are published on PyPI and attached to GitHub releases. Heavy
computation runs in Rust; Python exposes the same workflows with NumPy, pandas,
and Arrow inputs.

## A first analysis

The one-shot API takes data, a causal graph, and a typed question. This example
estimates an average treatment effect with explicit propensity weighting and a
cheap validation suite:

```python
import numpy as np
from antecedent import AverageEffect, analyze

rng = np.random.default_rng(7)
n = 1_200
z = rng.normal(size=n)
p_treated = 1 / (1 + np.exp(-(0.3 * z)))
t = (rng.random(n) < p_treated).astype(float)
y = 2.0 * t + z + rng.normal(scale=0.4, size=n)

query = AverageEffect("t", "y")
graph = [("z", "t"), ("z", "y"), ("t", "y")]

result = analyze(
    {"t": t, "y": y, "z": z},
    graph=graph,
    query=query,
    estimator="propensity.weighting",
    refute="cheap",
    bootstrap=100,
    seed=11,
)

print(result.identification)  # method, status, adjustment set
print(result.estimate)        # effect, uncertainty, estimator, overlap
print(result.validation)      # named checks and their outcomes
```

`analyze()` is convenience over the staged workflow. When identification itself
is the decision point, keep it explicit:

```python
from antecedent.identify import identify

identified = identify(graph=graph, query=query, names=["z", "t", "y"])
if not identified:
    raise RuntimeError(f"Not identified: {identified.status}")

result = identified.estimate({"t": t, "y": y, "z": z}, bootstrap=100)
```

The same engine is available through Rust’s typed `Study` builder. See the
[paired Python and Rust examples](examples/README.md).

## A causal result is more than a number

Antecedent preserves the distinctions that determine what can legitimately be
said about a result:

| Layer | What remains explicit |
| --- | --- |
| **Question** | Estimand, treatment, outcome, intervention, target population, and temporal coordinates |
| **Structure** | Graph class, source, version, unresolved marks, and—where licensed—posterior mass over graphs |
| **Identification** | Nonparametric, parametric-restricted, prior-restricted, partial, graph-dependent, or not identified; required assumptions remain scoped to identification |
| **Estimation** | Estimator and method, estimation-scoped restrictions, diagnostics, overlap or support, and the uncertainty actually computed |
| **Validation** | Which checks ran, what they compared, whether they were informative, and what passed or failed |
| **Transport through software** | Artifacts and wire conversions retain statuses, assumptions, evidence, geometry, and provenance; incoherent combinations fail closed |

That last row matters. A partial result must not become point identified because
an enum was flattened to a string. A parametric result must not lose the
restriction that made it identifiable. An internal consistency check must not
turn into known-truth evidence because a fixture name survived but its scope did
not. Antecedent treats these as correctness properties of the software stack.

## One engine across the causal workflow

Antecedent is deliberately broader than an estimator collection and more
connected than a discovery toolbox. The implemented system includes:

| Area | What is available |
| --- | --- |
| **Graphs** | DAG, ADMG, CPDAG, PAG, and temporal graph representations; d-/m-separation, latent projection, equivalence operations, temporal unfolding, and intervention overlays |
| **Discovery** | PC, FCI, RFCI, GES, DirectLiNGAM, NOTEARS; PCMCI-family temporal discovery; exact and MCMC-based graph posteriors; stability diagnostics |
| **Identification** | Backdoor, front-door, IV, sharp RD, an explicitly incomplete ID/IDC subset, generalized adjustment, temporal strategies, binary-IV bounds, and a certified single-source transport subset |
| **Estimation** | Regression and g-computation, IPW, matching, AIPW, 2SLS, RD, functional plug-ins, continuous response curves, intervention responses, and temporal estimators |
| **Bayesian analysis** | Conjugate, Laplace, and HMC GLMs; Bayesian g-computation; graph-by-effect envelopes; prior catalogs, compatibility checks, conflict-sensitive transfer, and predictive checks |
| **SCMs and explanations** | Hard, soft, stochastic, policy, and sequenced interventions; abduction–action–prediction; nested and temporal counterfactual machinery; anomaly, distribution-shift, change-point, unit, and Shapley root-cause attribution |
| **Study conditions** | Explicit observation mechanisms, single-source structural transport, trial-to-target estimation, and design-based randomized interference in dedicated stage APIs |
| **Decision workflows** | Experimental-design ranking, prepared analyses, accepted graph review and versioning, incremental `CausalState`, cancellation and compute budgets |
| **Data and artifacts** | NumPy, pandas, Arrow, tabular, time-series, panel, and multi-environment data; graph interchange; schema-versioned CBOR artifacts with migration and memory-mapped access |

This table is an implementation tour, not a promise that every cross-product of
query, graph class, inference mode, structure source, and validation suite is
valid. The [support matrix](docs/support-matrix.md) is the authoritative runtime
license for those combinations.

## Built for structural uncertainty

Many causal workflows discover a graph and immediately treat it as truth.
That is one common way claim strength increases in transit. Antecedent keeps the
transition explicit:

1. Discover or import structural evidence.
2. Review unresolved orientations and constraints.
3. Accept and version the structure used for analysis.
4. Identify the query against that graph class.
5. Estimate only within a licensed analysis cell.
6. Reuse the accepted graph or prepared analysis without silently rediscovering.

For selected Bayesian effect paths, Antecedent can carry a posterior over DAGs
through identification and estimation. Graph atoms that do not identify the
query remain visible as unidentified mass. Priors can change inference within
an identified model; they cannot turn non-identification into identification.

Partial graphs are not aliases for DAGs. Static and temporal graphs are not
interchangeable. Completing or orienting a graph creates a new structural
coordinate rather than quietly changing the meaning of the old one.

## Responses, interventions, and time

An average treatment effect is one useful causal question, not the universal
shape of one. Antecedent also represents function-valued queries such as

```text
a ↦ E[Y | do(A = a)]
```

and temporal surfaces such as

```text
(dose, horizon) ↦ E[Y at horizon | intervention policy at dose].
```

Static and temporal `ResponseCurve` and `InterventionResponse` paths report the
requested grid together with identification, assumptions, point support,
extrapolation diagnostics, and the uncertainty kind actually computed. Hard,
shift, stochastic, soft, and sequenced interventions live in the same typed
vocabulary, with unsupported policy/estimand combinations refused explicitly.

The lower-level response and SCM stages expose additional derivative,
counterfactual, and mechanism operations. Importability is not an `analyze()`
license: the current matrix intentionally has no licensed derivative or root
counterfactual analysis cells.

## From one answer to repeated decisions

Antecedent includes the machinery needed when causal analysis becomes part of a
product or an operating process:

- **Accepted graphs** separate discovery from human or programmatic structural
  review, retain an algorithm/version record, and can be held across estimates.
- **Prepared analyses** compile a licensed study once and support repeat
  estimation without changing the causal question.
- **`CausalState`** versions appended or replaced data, marks registered queries
  stale, and refreshes only when explicitly requested under a cache budget.
- **Artifacts** retain queries, assumptions, identification, estimates,
  diagnostics, posterior summaries, provenance, and response geometry across
  Rust and Python.
- **Design ranking** compares candidate measurements, interventions, sampling
  changes, or environments by information gain, identification probability, or
  decision utility.

This execution model is designed for auditability: data changes do not silently
rerun analyses, discovered structure does not silently refresh, and migration
does not fabricate evidential context missing from an older artifact.

## Evidence is typed too

Evidence is not a Boolean in Antecedent. Capability records distinguish what a
test or reference actually demonstrates:

| Evidence kind | Claim it supports |
| --- | --- |
| `implementation_exists` | The code path exists and has ordinary unit coverage; no numerical truth claim |
| `internal_cross_check` | Two Antecedent paths agree; consistency evidence, not independent truth |
| `internal_known_truth` | The implementation matches a closed-form, analytic, or clean-room reference case |
| `frozen_external_oracle` | The named fields match a frozen run from a pinned upstream implementation |
| `behavioral_parity` | Agreement with an upstream implementation across a range of inputs |
| `contract_equivalence` | A theorem-level or method-contract argument establishes the named equivalence |

The categories are deliberately not interchangeable. A frozen external fixture
does not prove a method correct, an internal cross-check is not calibration, and
ordinary tests do not become known-truth evidence through confident prose.
Recorded limitations scope an evidence claim to the proposition it actually
supports.

This typed evidence model is backed by repository-wide traceability:

- The [support matrix](docs/support-matrix.md) records which public analysis
  coordinates are licensed, not applicable, or refused—and why. Licensed rows
  name their evidence kind and limitations.
- The [provenance ledger](provenance/) records scientific sources and any
  upstream implementation consulted.
- [Conformance fixtures](conformance/) pin selected known-truth cases and
  black-box outputs from versioned external producers. Tests assert the named
  fields; a match is not a claim of whole-library parity.
- [Parity manifests](parity/) track named Rust/Python capability contracts and
  the evidence that exercises them.
- Statistical gates check declared finite-simulation calibration targets on
  specified designs. They do not establish universal interval coverage or
  validate causal assumptions from observed data.
- CI runs Rust and Python tests, linting, domain gates, and CodeQL. Release and
  scheduled workflows add artifact round trips, calibration suites, benchmark
  smoke tests, provenance closure, and dependency/security review.

This creates a typed chain from public claim to implementation to executable
evidence. It does not replace domain knowledge, study design, or scrutiny of the
assumptions attached to a result.

## Scientific boundaries

Antecedent’s scope is intentionally explicit:

- no prior can rescue a nonidentified estimand;
- no automatic assumption inference from column presence;
- no complete PAG-native ID/IDC or general multi-node sID recursion;
- no ML CATE, causal forests, or general policy-learning surface;
- no cyclic/equilibrium models or observational network contagion;
- no plotting subsystem or bindings beyond Rust and Python through 1.0.

Observation, transport, and interference remain dedicated stage APIs because
they change what identifies the estimand. Unsupported transport outside the
certified subset returns `NotCertified`, not a false proof of
non-transportability. See [Capabilities](docs/capabilities.md) and
[Comparison](docs/comparison.md) for the precise current boundaries.

## Project status and documentation

The current package version is **0.9.0**. This release consolidates the public
surface, evidence contracts, artifact format, and refusal boundaries on the path
to 1.0. The [roadmap](ROADMAP.md) describes that compatibility boundary and the
project’s deliberate non-goals.

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

MIT OR Apache-2.0 — see [LICENSE-MIT](LICENSE-MIT) and
[LICENSE-APACHE](LICENSE-APACHE).
