# Antecedent

[![CI](https://github.com/iridae-dev/antecedent/actions/workflows/ci.yml/badge.svg)](https://github.com/iridae-dev/antecedent/actions/workflows/ci.yml) [![Crates.io](https://img.shields.io/crates/v/antecedent)](https://crates.io/crates/antecedent) [![PyPI](https://img.shields.io/pypi/v/antecedent)](https://pypi.org/project/antecedent/) [![GitHub Release](https://img.shields.io/github/v/release/iridae-dev/antecedent)](https://github.com/iridae-dev/antecedent/releases/latest) [![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21556247-blue)](https://doi.org/10.5281/zenodo.21556247)

Antecedent is designed for causal inference as a composable systems primitive, built to preserve the epistemic correctness of causal analysis. It unifies discovery, structural uncertainty, identification, estimation, validation, interventions, temporal analysis, and durable artifacts in one typed workflow.

Antecedent applies high-assurance engineering principles to causal inference: claims are explicitly scoped, unsupported combinations fail closed, evidence is classified, implementations are checked against independent oracles where available, and provenance and traceability are machine-audited.

Give it data, a causal question, and a graph or discovery strategy. Antecedent determines what is identified, runs only a licensed inference path, or refuses claims the available evidence does not warrant. Results preserve assumptions, uncertainty, diagnostics, and provenance across system boundaries, so analyses can be composed, reviewed, reused, and audited without losing their scientific meaning.

## Python: one call, reusable study

```python
import antecedent as ant

result = ant.analyze(data, graph=graph, query=ant.AverageEffect("treatment", "outcome"))
study = result.study
updated = study.refresh(new_data)
report = result.inspect().to_dict()
loaded = ant.load(result.export())
```

A typical causal library returns an estimate. This returns a study: the same
program can run again, the report still carries the assumptions, and the
export is the claim you can hand to another process.

Paste this and run it. Two seasons of confounded price and sales; true
effect 1.5:

```python
import numpy as np
import antecedent as ant


def simulate(seed, n=2000):
    rng = np.random.default_rng(seed)
    season = rng.normal(size=n)
    price = 0.7 * season + rng.normal(size=n)
    sales = 1.5 * price + 2.0 * season + rng.normal(size=n)
    return {"season": season, "treatment": price, "outcome": sales}


data, new_data = simulate(0), simulate(1)
graph = [("season", "treatment"), ("season", "outcome"), ("treatment", "outcome")]
query = ant.AverageEffect("treatment", "outcome")

result = ant.analyze(data, graph=graph, query=query)
study = result.study
updated = study.refresh(new_data)
report = result.inspect().to_dict()
loaded = ant.load(result.export())

print(result.answer)
print(updated.answer)
```

The same analysis, with the identification in hand:

```python
ident = ant.identify(graph=graph, query=query, names=list(data))
result = ident.estimate(data)
updated = result.study.refresh(new_data)
report = updated.inspect().to_dict()
loaded = ant.load(updated.export())
```

`analyze` is those verbs in one call. `ident.validate(data)` is a refute
run, not a preflight. Read `result.answer.kind` (`point`, `bounds`,
`partial`, `response`, `structured`, `unavailable`); do not treat `.effect`
as the claim. A loaded result has no study — the export moved a contracted
execution, not a live object.

See the [Python workflow](docs/python-workflow.md) and
[examples](examples/README.md) for discovery review and the exception surface.

## The claim does not get stronger at the seams

Discovery, identification, estimation, and serialization are the places a
typical library drops context and the number gets more certain than the
evidence. Antecedent keeps the typed workflow intact: a CPDAG stays a
CPDAG, a partial answer stays partial, and `export` / `load` cannot invent
a study or a point.

The [support matrix](docs/support-matrix.md) is a license, not a feature
list. A combination that exists in the codebase but is not licensed
refuses. [Capabilities](docs/capabilities.md) is the inventory; the matrix
is what `analyze` may claim.

## Evidence is classified

Antecedent does not rely on a single notion of “tested.” It triangulates,
then makes those constraints executable.

- [**External oracles**](parity/README.md) record pinned upstream implementations, exact recipes, frozen results, and the tests that consume them. A fixture on disk is not evidence until the code exercises it.
- [**Algorithm provenance**](provenance/README.md) records the literature, upstream exposure, and where the implementation does *not* inherit the cited method’s guarantees.
- **Evidence itself is typed.** Implementation tests, analytic known truths, internal cross-checks, frozen oracles, behavioral parity, and stronger equivalence claims stay distinct, so a weaker label cannot become a stronger one.

An AIPW average treatment effect is licensed only where the full causal
contract has evidence. The legacy full-sample path’s provenance excludes
cross-fitting; its point is checked against a synthetic SCM, and the
DoWhy fixture beside it used weighting, not AIPW. An external match does
not prove identification, a citation does not grant an unimplemented
theorem, and code that runs does not make an analysis supported.

**None of this proves Antecedent correct. Together, these mechanisms make accidental overclaiming harder, make unsupported claims fail closed, and leave a visible trail for someone trying to prove the software wrong.**

## On a real decision

The notebooks are the same contract on a decision, not a second API.
Each one keeps a distinction the opening paragraphs refuse to drop.
Install their Python dependencies using the
[example setup](examples/README.md#python-environment):

| Notebook | What it keeps visible |
| --- | --- |
| [Paid-search attribution](examples/notebooks/marketing_channel_structural_uncertainty.ipynb) — a dashboard can credit paid search for demand that would have existed anyway. | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/iridae-dev/antecedent/blob/1.10.0/examples/notebooks/marketing_channel_structural_uncertainty.ipynb) |
| [Campaign evidence transfer](examples/notebooks/sales_campaign_prior_transfer.ipynb) — reuse a previous campaign’s evidence without assuming the new campaign is identical. | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/iridae-dev/antecedent/blob/1.10.0/examples/notebooks/sales_campaign_prior_transfer.ipynb) |
| [Experiment design](examples/notebooks/marketing_experiment_design.ipynb) — rank a holdout, better intent data, and more CRM records under a £40,000 budget. | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/iridae-dev/antecedent/blob/1.10.0/examples/notebooks/marketing_experiment_design.ipynb) |
| [Continuous causal response](examples/notebooks/continuous_causal_response.ipynb) — a nonlinear dose–response, with identification, support, and uncertainty as separate axes. | [![Open in Colab](https://colab.research.google.com/github/iridae-dev/antecedent/blob/1.10.0/examples/notebooks/continuous_causal_response.ipynb) |
| [Pricing, availability, and latent demand](examples/notebooks/pricing_availability_latent_demand.ipynb) — observed sales are not demand; the observation-aware route fails closed. | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/iridae-dev/antecedent/blob/1.10.0/examples/notebooks/pricing_availability_latent_demand.ipynb) |

Paired Python and Rust scripts for the same workflows live in
[examples](examples/README.md).

## 1.10.0

The current package version is **1.10.0**.
1.10 is a composition release of the existing 341 licensed cells (1423
meaningful combinations). Every licensed cell has a first-class
inspect/contract coordinate and completes the Rust compiler path
inspect → preview → execute → claim → consume. That is composition-seam
evidence, not inherited estimator-oracle truth.
Every Antecedent analysis retains a reusable study and exports a contracted execution; custom validator results travel as caller-attested, not re-verifiable, evidence, and a row-weight retarget re-executes only on its own data snapshot.
Every reported interval states its calibration: calibrated when a coverage record matches the execution and the execution is inside that record's scope; scope_not_assessed when a record matches but the execution is outside its scope or the record is a boundary; unavailable with a reason code when no record exists.
Identities are distinct and stable: every IdentityDomain plus target_weights is domain-separated and registered in parity/identity.toml.
See the
[1.10.0 release notes](docs/release-notes/v1.10.0.md), [1.9.0 calibration
notes](docs/release-notes/v1.9.0.md), [support matrix](docs/support-matrix.md),
and [conformance index](docs/conformance/README.md). The [1.5 Python
walkthrough](docs/local-distributional-joint.md) remains the guide for
retargeting, CDFs, and joint interventions; see the [release
checklist](docs/development.md#releases) for release requirements.

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
