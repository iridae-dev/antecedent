# Antecedent

```python
import antecedent as ant

result = ant.analyze(data, graph=graph, query=ant.AverageEffect("treatment", "outcome"))
study = result.study
updated = study.refresh(new_data)
report = result.inspect().to_dict()
loaded = ant.load(result.export())
```

Antecedent is an identification-first causal inference engine for Python (with
a native Rust core). `analyze` checks that the effect is identified before
anything is estimated and runs refuters against the estimate. `result.study`
keeps the compiled study, so `study.refresh(new_data)` re-executes the same
program on new data. `result.inspect().to_dict()` is the whole report as
JSON-safe data: answer, identification, support, uncertainty, assumptions,
identities and calibration. `ant.load(result.export())` round-trips the
contracted execution through the Rust semantic consumer.

Uncertainty about the causal graph (a CPDAG, a PAG, a posterior over graphs) is
carried through to the effect instead of being resolved by fiat, and discovered
graphs are never silently treated as ground truth.

Requires CPython 3.11–3.14. Install from PyPI:

```bash
pip install antecedent
```

Paste this block and run it. It simulates two confounded datasets and runs the
five lines on them:

```python
import numpy as np
import antecedent as ant


def simulate(seed, n=2000):
    rng = np.random.default_rng(seed)
    season = rng.normal(size=n)  # confounder
    price = 0.7 * season + rng.normal(size=n)  # treatment
    sales = 1.5 * price + 2.0 * season + rng.normal(size=n)  # outcome, true effect = 1.5
    return {"season": season, "treatment": price, "outcome": sales}


data, new_data = simulate(0), simulate(1)
graph = [("season", "treatment"), ("season", "outcome"), ("treatment", "outcome")]

result = ant.analyze(data, graph=graph, query=ant.AverageEffect("treatment", "outcome"))
study = result.study
updated = study.refresh(new_data)
report = result.inspect().to_dict()
loaded = ant.load(result.export())

print(result.answer)  # Answer(kind='point', value=1.48..., ...)
print(updated.answer)  # the same program on new_data
assert loaded.acceptance.verified and loaded.answer == result.answer
```

`result.answer` is the safe way to read the number: a set-identified or
partially identified analysis answers with `bounds` or `partial`, never an
unrestricted scalar. The same `analyze()` call scales to temporal dose × horizon
``ResponseCurve`` surfaces, pulse and sustained contrasts, Bayesian
graph-posterior mixtures that report unidentified structure mass, mediation,
counterfactuals, and root-cause attribution — see the
[project README](https://github.com/iridae-dev/antecedent#readme) for worked
examples, the
[Python workflow guide](https://github.com/iridae-dev/antecedent/blob/main/docs/python-workflow.md)
for studies, reports and portable executions, and the
[documentation](https://antecedent.readthedocs.io/) for the full API.

## 1.11.0

The
[1.11.0 release notes](https://github.com/iridae-dev/antecedent/blob/main/docs/release-notes/v1.11.0.md)
are the 1.x close-out on the 1.10 composition contract: licensed
graph-posterior cells that now earn, GAC 40, parallel coverage seeds, and
the [1.11 finding closeout](https://github.com/iridae-dev/antecedent/blob/1.11/docs/reviews/v1.11-finding-closeout.md).
Every Antecedent analysis retains a reusable study and exports a contracted execution; custom validator results travel as caller-attested, not re-verifiable, evidence, and a row-weight retarget re-executes only on its own data snapshot.
Every reported interval states its calibration: calibrated when a coverage record matches the execution and the execution is inside that record's scope; scope_not_assessed when a record matches but the execution is outside its scope or the record is a boundary; unavailable with a reason code when no record exists.
Identities are distinct and stable: every IdentityDomain plus target_weights is domain-separated and registered in parity/identity.toml.

## Earlier releases

These summaries describe the releases when they shipped. For current support,
use the [support matrix](https://github.com/iridae-dev/antecedent/blob/1.11/docs/support-matrix.md).

### 1.10.0

The
[1.10.0 release notes](https://github.com/iridae-dev/antecedent/blob/main/docs/release-notes/v1.10.0.md)
cover composition of the existing 341 licensed cells: inspect/contract
coordinates, the Rust inspect → preview → execute → claim → consume path
for every licensed cell, retained studies on ordinary prepared routes, and
portable claims.

### 1.9.0

The
[1.9.0 release notes](https://github.com/iridae-dev/antecedent/blob/main/docs/release-notes/v1.9.0.md)
cover the calibration of licensed intervals (a two-sided repeated-sampling
coverage gate, with boundary records disclosed at runtime), ADMG
interventional distributions, accepted-Dag
counterfactuals, Frequentist DBN-posterior mediation, and staged
attribution / transport / interference cells (Rust Study API only).
Behaviour changes:
`rd.sharp` defaults to the HC1 SE, NaN and null float cells are missing
values, and the default bootstrap count is 199.

### 1.8.0

The
[1.8.0 release notes](https://github.com/iridae-dev/antecedent/blob/main/docs/release-notes/v1.8.0.md)
cover the Bayesian remainder of the staged handle: functional path,
distribution, and ADMG ATE, static mediation, counterfactuals, derivatives,
and ConditionalEffect graph-posterior mixtures.

### 1.7.0

The
[1.7.0 release notes](https://github.com/iridae-dev/antecedent/blob/main/docs/release-notes/v1.7.0.md)
cover class-preserving Bayesian analyses on incomplete temporal graphs.
The
[1.6.0 release notes](https://github.com/iridae-dev/antecedent/blob/main/docs/release-notes/v1.6.0.md)
cover per-horizon temporal identification, sequential overlays, and
observation-adjusted temporal curves. The
[Python walkthrough for local, distributional, and joint effects](https://github.com/iridae-dev/antecedent/blob/main/docs/local-distributional-joint.md)
remains the guide for retargeting, CDFs, and joint interventions.

## Development

For local development you need a Rust 1.85 toolchain. CI builds and smoke-tests
wheels for the supported CPython range on Linux x86_64/aarch64 (manylinux),
macOS arm64, and Windows x86_64 (default `faer` path; no system BLAS). Tagged
releases publish wheels to PyPI and GitHub Release assets (see
[docs/development.md](https://github.com/iridae-dev/antecedent/blob/main/docs/development.md)).

```bash
cd python
uv venv && source .venv/bin/activate
uv sync --group dev
maturin develop
pytest
```

Lint and types (local gate only — not part of wheel CI):

```bash
bash ../scripts/gate_python_lint.sh   # ruff check/format + mypy
```

## Public API

Primary entry point is the OO facade:

```python
import antecedent

g = antecedent.Dag.from_edges(["z", "t", "y"], [("z", "t"), ("z", "y"), ("t", "y")])
result = antecedent.analyze(
    data,  # dict[str, array] or pandas DataFrame
    graph=g,  # or an edge list
    query=antecedent.AverageEffect(treatment="t", outcome="y"),
    inference=antecedent.Frequentist(),  # or antecedent.Bayesian(...)
)
study = result.study
updated = study.refresh(new_data)
report = result.inspect().to_dict()
loaded = antecedent.load(result.export())
assert loaded.acceptance.verified

# Or stop before estimation with the same ordinary defaults:
prepared = antecedent.prepare(data, graph=g, query=antecedent.AverageEffect("t", "y"))
repeated = prepared.estimate()

# Identify without estimating:
id_only = antecedent.identify(
    graph=g,
    query=antecedent.AverageEffect(treatment="t", outcome="y"),
    identifier=antecedent.Identifier.BACKDOOR_ADJUSTMENT,
)

gcm = antecedent.model.fit_gcm(["z", "t", "y"], columns, list(g.edges()))
draws = gcm.sample_do({"t": 1.0}, n=200)

# Or discover then fit (never invents orientations; refuses incomplete PAG/FCI):
fitted, edges = antecedent.gcm.fit_gcm_discovered(
    data, discovery=antecedent.discovery.PC(alpha=0.05)
)
```

The root namespace contains 56 names: the analyze contract plus
`prepare`, `load`, `InterferenceQuery`, and `Analysis`. `AnomalyAttribution`
and `ChangeAttribution` run their licensed Dag cells on `analyze()` and retain
a study. `InterferenceQuery` and `antecedent.transport.advanced.TransportQuery`
run their licensed cells on `analyze()` and retain a study like every other
licensed route; `antecedent.transport` / `antecedent.interference` hold the
2.0 compiler, selection diagram, designs, exposure mappings, the transport
identification stage, and the unlicensed `estimate_trial_effect` / `estimate`
utilities, which return bare numbers with no study or license.
Everything else is reached through a stage module (`antecedent.discovery`, `antecedent.priors`, `antecedent.errors`, …).

Also exposed:

- Typed graphs: `Dag` / `Cpdag` / `Pag` / `Admg` / `TemporalDag`
  (`d_separated` / `latent_project` on `Dag`; `m_separated` on `Admg` / `Pag`)
- `Identifier` / `Estimator` enums (wire ids) plus string kwargs
- `identify(graph=…, query=AverageEffect(…))` — identify without estimating
- Queries: `AverageEffect`, `MediationEffect`, `Counterfactual`, `PulseEffect`, `SustainedEffect`,
  `InterventionalDistribution`, `PathSpecificEffect`, `ConditionalEffect`,
  `TemporalMediationEffect`, `InterventionResponse`, plus the response family
  (`ResponseCurve`, `AverageDerivative`, `PointDerivative`, `Elasticity`,
  `SemiElasticity`, `DirectionalDerivative`, `ResponseJacobian`), plus
  `AnomalyAttribution` / `ChangeAttribution` (root types; licensed
  `analyze(data, graph=Dag, query=...)` at validation `none`). Temporal
  dose × horizon uses the same `ResponseCurve` /
  `InterventionResponse` types with keyword-only `horizons`, `policy`, and
  `treatment_lag` (see `examples/python/temporal_response_curve.py`).
- `antecedent.discovery` — PC, GES, LiNGAM, NOTEARS, FCI/RFCI, PCMCI family, Bayesian posteriors
- `antecedent.validation.validate_pcmci_*` — discovery stability (block bootstrap, FPR, grids, …)
- `antecedent.model` / `antecedent.counterfactual` — `FittedGcm`, `sample_do`, `counterfactual_ite`
- `antecedent.population` — `PopulationRegistry` / `target_*` for named predicates and custom-distribution IPW
- `antecedent.gcm` — `fit_gcm_discovered` / `attribute_*_discovered` discover-then-attribute composition
- `antecedent.state.CausalState` — incremental state with retained batches, events, suff-stats, particle filter
- `refute="full"|"placebo"|False` on static `analyze` (`refute=True` is a
  `TypeError`; leave it unset for the licensed default). Static and temporal
  `ResponseCurve` / `InterventionResponse` license validation `none` only;
  requesting a scalar refuter suite raises `CausalUnsupportedError`.
- RD: `estimator="rd.sharp"` with `running_variable` / `cutoff` / `bandwidth`
- Graph interchange on the classes: `Dag.from_dot` / `.to_dot` and the JSON / GML / NetworkX peers
- Design / state examples: [`examples/python/rank_designs.py`](https://github.com/iridae-dev/antecedent/blob/main/examples/python/rank_designs.py),
  [`examples/python/causal_state_workflow.py`](https://github.com/iridae-dev/antecedent/blob/main/examples/python/causal_state_workflow.py),
  [`examples/python/temporal_response_curve.py`](https://github.com/iridae-dev/antecedent/blob/main/examples/python/temporal_response_curve.py),
  [`examples/python/staged_static_kinds.py`](https://github.com/iridae-dev/antecedent/blob/main/examples/python/staged_static_kinds.py)
  (see ADR 0016 — no auto-rerun); catalog in [`examples/README.md`](https://github.com/iridae-dev/antecedent/blob/main/examples/README.md)

Build artifacts (`_native.*.so`) are gitignored; always `maturin develop` (or install a wheel) on a fresh checkout.

In 1.6, multi-step Sustained supports `none`/`cheap`/`full` validation in both
inference modes, including DBN posterior mixtures. Full perturbations refit
the sequential model; Bayesian checks retain mechanism PPC and composed-effect
prior sensitivity. Temporal Soft `multiplicative` and `truncated_shift` compose
across single, joint, and multi-step schedules; truncation applies to the
propagated conditional mean, not a stochastic draw. Bayesian temporal
observation curves and Sequence use an observed-data Gaussian SEM with
latent-trajectory Gibbs sampling under an explicit ignorable trajectory
coarsening assumption and distinct priors. See the
[observation contract](https://github.com/iridae-dev/antecedent/blob/main/docs/observation-contract.md)
for the conditioning assumptions; this is not Bayesian IPCW weighting.

In 1.3, `antecedent.estimation.PreparedAnalysis` also stages Frequentist DAG
derivatives, static natural mediation, explicit-DAG unit counterfactuals, and
the published observation-pair contract (selected AIPW, marginal KM, conditional
Cox IPCW). Derivative cheap/full stay n/a; counterfactual sampling uncertainty
is unavailable. See the
[1.3 evidence ledger](https://github.com/iridae-dev/antecedent/blob/main/docs/v1.3-evidence.md)
and the 1.2 ledger below for earlier Bayesian and sequential forms.

In 1.2, `antecedent.estimation.PreparedAnalysis` also supports the licensed
Bayesian conditional, temporal-mediation and response forms, and multi-step
`SustainedEffect(..., window=(-2, -1))`. Choose the validation suite when
preparing these analyses; second-click `refute()` remains AverageEffect-only.
`export_artifact()` retains the fitted posterior or response, while
`export_artifact(payload="query")` retains the original query kind and variable
IDs. Scalar posterior exports do not include the complete assumption or
validation ledger; retain the analysis result alongside them. See the
[1.2 evidence ledger](https://github.com/iridae-dev/antecedent/blob/main/docs/v1.2-evidence.md)
for model restrictions and evidence.

Typed exceptions (`CausalError` and subclasses) mirror Rust `CausalError` categories.
The native module `antecedent._native` remains available for advanced FFI use
(including the flat `AteAnalysisResult` DTO; prefer nested `AnalysisResult`).
