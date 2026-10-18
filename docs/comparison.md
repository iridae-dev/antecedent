# Antecedent and other causal libraries

This page is a selection aid, not a competitor benchmark. It describes
Antecedent from contracts verified in this repository and links to other
projects for capabilities that may fit a different workflow. Claims about
another project should be checked against that project's current documentation.

Antecedent is an identification-first Rust engine with Python bindings. Its
distinctive workflow is to keep the query, graph class, identification status,
empirical support, inference mode, and validation contract explicit through
estimation.

## How Antecedent is different: epistemic type safety

Antecedent is built to make certain scientifically invalid transformations as difficult as type errors are in a programming language.

For example, Antecedent explicitly tries to prevent these equivalences:

- `estimated == identified`
- `confidence interval == structural uncertainty`
- `discovered graph == known graph`
- `search failed == impossible`
- `implemented algorithm == supported scientific claim`
- `same numeric answer == same causal analysis`

That means that a result isn't fundamentally:

```
ATE = 0.23 ± 0.04
```

It is closer to:

```
Claim:
    question = ...
    target population = ...
    structural knowledge = ...
    identification = ...
    estimator/inference binding = ...
    empirical support = ...
    uncertainty represented = ...
    uncertainty NOT represented = ...
    assumptions = ...
    data snapshot = ...
    provenance = ...
    calibration evidence = ...
    answer shape = point | bounds | partial | response | ...
```

Antecedent explicitly has result shapes such as point, bounds, partial, response, structured, and unavailable. It does not want a partially identified problem to become a scalar simply because a consumer expects a float.

That distinction is much deeper than having better diagnostics.

Compared to a typical causal influence library:

| Dimension              | Typical estimator-oriented library                 | Antecedent                                                                                      |
| ---------------------- | -------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| Main object            | Estimator/model                                    | **Causal question / contract**                                                                  |
| Starting point         | `Y, T, X, W, Z`                                    | Question + structure + evidence + population                                                    |
| Identification         | Often assumed by estimator setup                   | **Explicit compiler stage**                                                                     |
| Confounders            | User chooses covariates                            | Estimator is forbidden from deciding identification                                             |
| Graph                  | Often absent or a DAG                              | DAG / ADMG / CPDAG / PAG / temporal classes remain semantically distinct                        |
| Discovery              | Often separate package                             | Discovery produces **evidence**, not automatic truth                                            |
| Structural uncertainty | Often resolved before estimation                   | Can propagate through to bounds/envelopes/partial answers                                       |
| Unsupported case       | Error, NaN, undefined use, or documentation caveat | **Typed refusal with reason**                                                                   |
| Uncertainty            | Usually sampling/model CI                          | Sampling, graph, orientation, identification, mechanism, regime, measurement etc. kept separate |
| Output                 | Estimate/model                                     | **Claim + contract + diagnostics + provenance**                                                 |
| Reuse                  | Refit estimator                                    | Re-execute a compiled `Study`                                                                   |
| Serialization          | Model/result pickle/object                         | Versioned semantic artifact intended to preserve claim meaning                                  |
| Correctness evidence   | Tests                                              | Tests + provenance + external oracles + conformance + calibration records                       |


## What Antecedent is licensed to run

The [support matrix](support-matrix.md) is authoritative. A capability present
in the codebase is not necessarily a licensed `analyze()` combination.

The licensed coordinates are the generated inventory below. Each combination
fixes the question, graph class, structure source, inference method, and
validation level. Licensed does not mean interval coverage was measured: the
support matrix counts the cells with no coverage measurement for their
estimator. A successful licensed Python analysis retains a study and
exports its result.

Use [supported analyses](supported-analyses.md) to find a starting point and
[capabilities](capabilities.md) for the methods behind each path. The
[2.0 release notes](release-notes/v2.0.0.md) describe the release and its calibration boundary.

<!-- generated:query-inventory:begin -->
Generated from `parity/support_axes.toml`, `parity/support_licensed.toml`,
and `parity/support_n_a.toml` by `scripts/generate_claim_inventory.py`.
Licensed coordinates are grouped only where the cell lists the full product
of the structures, inferences, and validations shown. A coordinate that is
not listed is n/a or refused. Refusal is the default;
`parity/support_closed.toml` names reasons and does not create a fourth state.
The [support matrix](support-matrix.md) is the license.
Do not edit this block by hand.

| Query | Licensed | n/a | Refused |
|---|---:|---:|---:|
| AnomalyAttribution | 4 | 36 | 140 |
| AverageDerivative | 20 | 68 | 92 |
| AverageEffect | 92 | 88 | 0 |
| ChangeAttribution | 4 | 36 | 140 |
| ConditionalEffect | 54 | 90 | 36 |
| Counterfactual | 20 | 36 | 124 |
| DirectionalDerivative | 20 | 68 | 92 |
| Elasticity | 20 | 68 | 92 |
| InterventionalDistribution | 36 | 36 | 108 |
| InterventionResponse | 99 | 81 | 0 |
| MediationEffect | 12 | 90 | 78 |
| NestedCounterfactualEffect | 18 | 36 | 126 |
| PathSpecificEffect | 28 | 36 | 116 |
| PointDerivative | 20 | 68 | 92 |
| PulseEffect | 54 | 108 | 18 |
| ResponseCurve | 48 | 116 | 16 |
| ResponseJacobian | 20 | 68 | 92 |
| SemiElasticity | 20 | 68 | 92 |
| SustainedEffect | 54 | 108 | 18 |
| TemporalConditionalPulse | 16 | 108 | 56 |
| TemporalConditionalSustained | 16 | 108 | 56 |
| TemporalMediationEffect | 36 | 108 | 36 |
| TransportQuery | 4 | 90 | 86 |
| InterferenceQuery | 2 | 90 | 88 |

717 licensed cells.

| Query | Graph class | Structures | Inferences | Validations | Cells |
|---|---|---|---|---|---:|
| AnomalyAttribution | Dag | explicit | Frequentist, Bayesian | none | 2 |
| AnomalyAttribution | TemporalDag | explicit | Frequentist, Bayesian | none | 2 |
| AverageDerivative | Dag | explicit, accepted | Frequentist, Bayesian | none | 4 |
| AverageDerivative | TemporalDag | explicit, accepted | Frequentist, Bayesian | none | 4 |
| AverageDerivative | TemporalCpdag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none | 6 |
| AverageDerivative | TemporalPag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none | 6 |
| AverageEffect | Dag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| AverageEffect | Admg | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| AverageEffect | Cpdag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| AverageEffect | Pag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| AverageEffect | TemporalAdmg | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| AverageEffect | CoDetermined | explicit | Frequentist | none | 1 |
| AverageEffect | Unknown | explicit | Frequentist | none | 1 |
| ChangeAttribution | Dag | explicit | Frequentist, Bayesian | none | 2 |
| ChangeAttribution | TemporalDag | explicit | Frequentist, Bayesian | none | 2 |
| ConditionalEffect | Dag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| ConditionalEffect | Cpdag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| ConditionalEffect | Pag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| Counterfactual | Dag | explicit, accepted | Frequentist, Bayesian | none | 4 |
| Counterfactual | TemporalDag | explicit, accepted | Frequentist, Bayesian | none | 4 |
| Counterfactual | TemporalCpdag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none | 6 |
| Counterfactual | TemporalPag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none | 6 |
| DirectionalDerivative | Dag | explicit, accepted | Frequentist, Bayesian | none | 4 |
| DirectionalDerivative | TemporalDag | explicit, accepted | Frequentist, Bayesian | none | 4 |
| DirectionalDerivative | TemporalCpdag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none | 6 |
| DirectionalDerivative | TemporalPag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none | 6 |
| Elasticity | Dag | explicit, accepted | Frequentist, Bayesian | none | 4 |
| Elasticity | TemporalDag | explicit, accepted | Frequentist, Bayesian | none | 4 |
| Elasticity | TemporalCpdag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none | 6 |
| Elasticity | TemporalPag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none | 6 |
| InterventionalDistribution | Dag | explicit, accepted | Frequentist, Bayesian | none, cheap, full | 12 |
| InterventionalDistribution | Admg | explicit, accepted | Frequentist, Bayesian | none | 4 |
| InterventionalDistribution | TemporalDag | explicit, accepted | Frequentist, Bayesian | none | 4 |
| InterventionalDistribution | TemporalCpdag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none | 6 |
| InterventionalDistribution | TemporalPag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none | 6 |
| InterventionalDistribution | TemporalAdmg | explicit, accepted | Frequentist, Bayesian | none | 4 |
| InterventionResponse | Dag | explicit | Frequentist | none | 1 |
| InterventionResponse | Dag | explicit | Frequentist | cheap | 1 |
| InterventionResponse | Dag | explicit | Frequentist | full | 1 |
| InterventionResponse | Dag | explicit | Bayesian | none | 1 |
| InterventionResponse | Dag | accepted | Frequentist | none | 1 |
| InterventionResponse | Dag | accepted | Frequentist | cheap | 1 |
| InterventionResponse | Dag | accepted | Frequentist | full | 1 |
| InterventionResponse | Dag | accepted | Bayesian | none | 1 |
| InterventionResponse | Dag | graph_posterior | Frequentist | none | 1 |
| InterventionResponse | Dag | graph_posterior | Frequentist | cheap | 1 |
| InterventionResponse | Dag | graph_posterior | Frequentist | full | 1 |
| InterventionResponse | Dag | graph_posterior | Bayesian | none | 1 |
| InterventionResponse | Admg | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| InterventionResponse | Cpdag | explicit | Frequentist, Bayesian | none | 2 |
| InterventionResponse | Cpdag | accepted | Frequentist, Bayesian | none | 2 |
| InterventionResponse | Cpdag | graph_posterior | Frequentist, Bayesian | none, cheap, full | 6 |
| InterventionResponse | Pag | explicit | Frequentist, Bayesian | none | 2 |
| InterventionResponse | Pag | accepted | Frequentist, Bayesian | none | 2 |
| InterventionResponse | Pag | graph_posterior | Frequentist, Bayesian | none, cheap, full | 6 |
| InterventionResponse | TemporalDag | explicit | Frequentist, Bayesian | none | 2 |
| InterventionResponse | TemporalDag | accepted | Frequentist, Bayesian | none | 2 |
| InterventionResponse | TemporalDag | graph_posterior | Frequentist, Bayesian | none, cheap, full | 6 |
| InterventionResponse | TemporalCpdag | explicit | Frequentist, Bayesian | none | 2 |
| InterventionResponse | TemporalCpdag | accepted | Frequentist, Bayesian | none | 2 |
| InterventionResponse | TemporalCpdag | graph_posterior | Frequentist, Bayesian | none, cheap, full | 6 |
| InterventionResponse | TemporalPag | explicit | Frequentist, Bayesian | none | 2 |
| InterventionResponse | TemporalPag | accepted | Frequentist, Bayesian | none | 2 |
| InterventionResponse | TemporalPag | graph_posterior | Frequentist, Bayesian | none, cheap, full | 6 |
| InterventionResponse | TemporalAdmg | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| InterventionResponse | CoDetermined | explicit | Frequentist | none | 1 |
| MediationEffect | Dag | explicit, accepted | Frequentist, Bayesian | none, cheap, full | 12 |
| NestedCounterfactualEffect | Dag | explicit | Frequentist, Bayesian | none | 2 |
| NestedCounterfactualEffect | TemporalDag | explicit, accepted | Frequentist, Bayesian | none | 4 |
| NestedCounterfactualEffect | TemporalCpdag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none | 6 |
| NestedCounterfactualEffect | TemporalPag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none | 6 |
| PathSpecificEffect | Dag | explicit, accepted | Frequentist, Bayesian | none, cheap, full | 12 |
| PathSpecificEffect | TemporalDag | explicit, accepted | Frequentist, Bayesian | none | 4 |
| PathSpecificEffect | TemporalCpdag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none | 6 |
| PathSpecificEffect | TemporalPag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none | 6 |
| PointDerivative | Dag | explicit, accepted | Frequentist, Bayesian | none | 4 |
| PointDerivative | TemporalDag | explicit, accepted | Frequentist, Bayesian | none | 4 |
| PointDerivative | TemporalCpdag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none | 6 |
| PointDerivative | TemporalPag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none | 6 |
| PulseEffect | TemporalDag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| PulseEffect | TemporalCpdag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| PulseEffect | TemporalPag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| ResponseCurve | Dag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none | 6 |
| ResponseCurve | Admg | explicit, accepted, graph_posterior | Frequentist, Bayesian | none | 6 |
| ResponseCurve | Cpdag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none | 6 |
| ResponseCurve | Pag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none | 6 |
| ResponseCurve | TemporalDag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none | 6 |
| ResponseCurve | TemporalCpdag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none | 6 |
| ResponseCurve | TemporalPag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none | 6 |
| ResponseCurve | TemporalAdmg | explicit, accepted, graph_posterior | Frequentist, Bayesian | none | 6 |
| ResponseJacobian | Dag | explicit, accepted | Frequentist, Bayesian | none | 4 |
| ResponseJacobian | TemporalDag | explicit, accepted | Frequentist, Bayesian | none | 4 |
| ResponseJacobian | TemporalCpdag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none | 6 |
| ResponseJacobian | TemporalPag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none | 6 |
| SemiElasticity | Dag | explicit, accepted | Frequentist, Bayesian | none | 4 |
| SemiElasticity | TemporalDag | explicit, accepted | Frequentist, Bayesian | none | 4 |
| SemiElasticity | TemporalCpdag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none | 6 |
| SemiElasticity | TemporalPag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none | 6 |
| SustainedEffect | TemporalDag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| SustainedEffect | TemporalCpdag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| SustainedEffect | TemporalPag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| TemporalConditionalPulse | TemporalDag | explicit, accepted | Frequentist, Bayesian | none | 4 |
| TemporalConditionalPulse | TemporalCpdag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none | 6 |
| TemporalConditionalPulse | TemporalPag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none | 6 |
| TemporalConditionalSustained | TemporalDag | explicit, accepted | Frequentist, Bayesian | none | 4 |
| TemporalConditionalSustained | TemporalCpdag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none | 6 |
| TemporalConditionalSustained | TemporalPag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none | 6 |
| TemporalMediationEffect | TemporalDag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| TemporalMediationEffect | TemporalCpdag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| TransportQuery | Admg | explicit | Frequentist, Bayesian | none | 2 |
| TransportQuery | TemporalAdmg | explicit | Frequentist, Bayesian | none | 2 |
| InterferenceQuery | Dag | explicit | Frequentist, Bayesian | none | 2 |
<!-- generated:query-inventory:end -->

Some important distinctions when choosing a workflow:

- Static DAG graph-posterior paths include average effects, conditional effects,
  response curves, and one-coordinate intervention responses. Each has its own
  inference and validation restrictions.
- Temporal graph-posterior paths include Pulse and Sustained effects under
  both inference modes, and temporal mediation on `TemporalDag` and
  `TemporalCpdag` under both inference modes at validation
  `none`/`cheap`/`full`. `TemporalPag` mediation is not licensed.
- Frequentist and Bayesian envelopes on incomplete `TemporalCpdag` and
  `TemporalPag` graphs are supported for licensed queries. A caller-supplied
  `ClassPrior` can provide mixture weights; enumerating completions alone does
  not assign probabilities.
- Derivatives on explicit or accepted DAGs support Frequentist and Bayesian
  inference at validation `none`. Partial-graph derivatives remain refused.
- ADMG interventional distributions run through Python `analyze`/`prepare` and
  Rust `Study` at validation `none`.
- Licensed anomaly/change attribution paths run through Python `analyze`/`prepare`
  and Rust `Study` at validation `none`.
- Licensed transport and interference queries run through both Python `analyze`
  and Rust `Study`, with their explicit design assumptions.

`ResponseCurve` and `InterventionResponse` are licensed on temporal DAG,
`TemporalCpdag` and `TemporalPag` graph-posterior cells (see the matrix for
the validation levels each admits). ADMG graph-posterior AverageEffect at validation `none` identifies each atom with `general.id` and
estimates `functional.effect`; cheap/full stay closed. These are different
requests from incomplete-class envelopes and from CPDAG/PAG graph-posterior
AverageEffect, which evaluates each atom with the class ATE envelope and
combines only under a shared estimand. Unidentified structural mass stays
visible; priors do not establish identification.

## What the repository compares externally

Antecedent is independently implemented. Selected conformance fixtures record
black-box outputs from pinned external packages:

- DoWhy 0.14 supplies scoped identification and estimation reference fixtures.
  Executing tests compare only the fields named by each evidence contract.
  Some fixtures carry a recorded DoWhy value that is intentionally not asserted;
  their limitations say so.
- Tigramite 5.2.1.30 supplies the core temporal discovery and conditional-
  independence fixtures. J-PCMCI+ and fixed-regime RPCMCI-related fixtures use
  5.2.9.7. Evidence ranges from frozen examples to behavioral comparisons and
  is not a claim of whole-library parity.
- causal-learn 0.1.4.3 supplies the frozen FCI and GES reference cases, while
  lingam 1.9.1 supplies the DirectLiNGAM order/coefficient reference in the
  same static-discovery fixture.
- statsmodels 0.14.4 supplies the conditional-effect and multiplicity
  reference outputs claimed by the capability manifests.
- bpbounds 0.1.8 supplies the canonical binary-IV Balke–Pearl bounds fixture;
  causaleffect 1.3.15 supplies formula-class references only for Antecedent's
  explicitly scoped sound catalog-bound sID subset.

Each external claim has an immutable record under `parity/baselines`, a frozen
fixture, and an executing conformance test. See
[ADR 0009](https://github.com/iridae-dev/antecedent/blob/main/adr/0009-parity-baselines.md), the parity manifests, and each
licensed row's evidence kind and limitations. A shared algorithm name is not
evidence of equivalent behavior.

## Choosing a workflow

### DoWhy

DoWhy and Antecedent overlap in model–identify–estimate–refute workflows and
graphical causal-model tooling. Evaluate DoWhy when its Python ecosystem,
examples, estimator integrations, or GCM workflows are the primary need.

Antecedent differs by making its support matrix and typed refusals part of the
runtime contract. It also carries accepted partial graphs and selected graph
posteriors into licensed downstream analyses. That does not imply broad feature
parity with DoWhy.

Where the two share a semantic, the comparison is executed rather than asserted: against
the version pinned in `parity/baselines/dowhy.toml`, the suite computes DoWhy's interventional
means (constant and functional interventions), unit counterfactuals, intrinsic causal influence,
graph permutation falsification and automatic mechanism assignment at test time and checks
agreement within stated tolerances (`python/tests/test_dowhy_comparison.py`; skipped when DoWhy is
not installed). The overlap and the differences are stated on the
[intervention programs](guides/intervention-programs.md#differences-from-dowhy),
[mechanism selection](guides/mechanism-selection.md#differences-from-dowhy),
[intrinsic influence](guides/intrinsic-influence.md#differences-from-dowhy) and
[graph falsification](guides/graph-falsification.md#differences-from-dowhy) guides.

### EconML

EconML focuses on machine-learning estimators for heterogeneous treatment
effects and policy-oriented workflows. Antecedent now ships native
cross-fitted DML (`estimators.DML`, AIPW or Robinson PLR) and a DR-Learner
CATE path (`estimators.DRLearner`) and a native honest causal forest
(`estimators.CausalForest`). DR-Learner reports the marginal ATE and its IID
interval from cross-fitted AIPW scores. A linear final stage also reports HC0
pointwise CATE standard errors of those orthogonal scores; penalized or
nonlinear finals withhold CATE intervals. The honest forest publishes no
`cate_se`; its per-row `cate_leaf_dispersion` is a diagnostic (the root mean of
two-sample leaf variances across trees). It omits between-tree and
adaptive-neighbourhood variability, so it is not a pointwise standard error and
`cate ± 1.96·cate_leaf_dispersion` is not a confidence band; forest CATE
predictions carry no pointwise intervals. Robinson PLR
requires a constant conditional effect to interpret its slope as the ATE.
The initial known-truth fixture in `conformance/estimate/learner_ate/fixture.json`
covers a binary-treatment linear SCM at one sample size; it does not license
nonlinear-provider or CATE interval calibration. Read each result's calibration status and matching coverage record;
coverage for a marginal ATE does not license pointwise CATE intervals.

That is not a 12-estimator EconML clone:
learners never choose the adjustment set, and the EconML handoff remains
an identification-set export.

`antecedent.handoff.adjustment(result)` emits the adjustment set and identification
status for point-identified backdoor / generalized-adjustment estimands
(`handoff.econml` is the same function),
including lag-aligned temporal columns when the certificate carries offsets.
`spec.attach(...)` records a caller-fitted estimate as attested, uncalibrated
evidence; Antecedent does not wrap the learner or inherit native calibration.
A scalar `label="ate"` effect is bound through `external.bind`, so the learner,
its declared version, support labels and lineage come along as `.claim`.
Front-door, IV, general-ID, partial-identification, and graph-posterior
results refuse rather than pretending they are a set. The adapter does not
wrap EconML learners or absorb an external CATE.

A worked handoff is in
[`examples/python/econml_cate_handoff.py`](https://github.com/iridae-dev/antecedent/blob/v2.4.1/examples/python/econml_cate_handoff.py):
Antecedent identifies and estimates ATE, then the caller optionally fits
EconML `LinearDML` on `spec.columns(data)` when `econml` is installed.


### Tigramite

Tigramite is the upstream reference used for selected PCMCI-family and
conditional-independence conformance fixtures. Evaluate it directly for
time-series discovery research, its current algorithm set, and its native
visualization workflow.

Antecedent's temporal discovery implementations are useful when discovery must
feed a licensed effect, intervention, prepared-analysis, or artifact workflow in
the same engine. External comparisons are fixture-specific: for example,
fixed-label per-regime tests are not parity evidence for unsupervised RPCMCI
regime learning.

### causal-learn

causal-learn provides a broad Python causal-discovery surface. Evaluate it when
discovery itself is the endpoint or when the required discovery algorithm is
outside Antecedent's implemented set.

Graphs can be imported through supported interchange formats, but importing a
graph does not bypass Antecedent's matrix. Its graph class, query, structure
source, inference mode, and validation suite still determine whether the
downstream analysis is licensed, not applicable, or refused.

## Current Antecedent boundaries

The following are current product boundaries or explicit matrix refusals:

- no 12-estimator EconML clone (native DML, DRLearner, and honest causal-forest CATE; the EconML adapter still emits a set, not a learner);
- no plotting module;
- no R, Julia, or JavaScript bindings;
- no complete PAG-native ID/IDC;
- transport identification is complete only within the experimental-information
  families stated in [transport scope](guides/transport-scope.md); catalog-bound
  search is sound and incomplete, and `NotCertified` is not a
  non-transportability proof;
- for incomplete-class response uncertainty, use the query-specific contracts in
  [causal responses](causal-responses.md);
- no partial-graph derivative cells;
- no exact DAG pseudo-posterior enumeration beyond six nodes;
- no automatic estimator choice and no prior that can rescue identification.

These statements are not interchangeable:

- **not applicable** means a matrix coordinate does not denote;
- **refused** means the coordinate is meaningful but this release does not
  license it;
- **outside product scope** names a broader capability Antecedent does not aim
  to provide in this release line.

For exact current behavior, use the [support matrix](support-matrix.md). For the
implemented building blocks behind it, use [Capabilities](capabilities.md).
