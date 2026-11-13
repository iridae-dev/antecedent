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


## What Antecedent is supported to run

The [support matrix](support-matrix.md) is authoritative. A capability present
in the codebase is not necessarily a supported `analyze()` combination.

The supported coordinates are the generated inventory below. Each combination
fixes the question, graph class, structure source, inference method, and
validation level. Supported does not mean interval coverage was measured: the
support matrix counts the cells with no coverage measurement for their
estimator. A successful supported Python analysis retains a study and
exports its result.

Use [supported analyses](supported-analyses.md) to find a starting point and
[capabilities](capabilities.md) for the methods behind each path. The
[release notes](release-notes/v3.0.0.md) describe what changed in the claims.

<!-- generated:query-inventory:begin -->
Generated from `parity/support_axes.toml`, `parity/support_cells.toml`,
and `parity/support_n_a.toml` by `scripts/generate_claim_inventory.py`.
Supported coordinates are grouped only where the cell lists the full product
of the structures, inferences, and validations shown. A coordinate that is
not listed is n/a or refused. Refusal is the default;
`parity/support_refusal_reasons.toml` names reasons and does not create a fourth state.
The [support matrix](support-matrix.md) is the support.
Do not edit this block by hand.

| Query | Supported | n/a | Refused |
|---|---:|---:|---:|
| AnomalyAttribution | 144 | 36 | 0 |
| AverageDerivative | 144 | 36 | 0 |
| AverageEffect | 148 | 32 | 0 |
| ChangeAttribution | 48 | 36 | 96 |
| ConditionalEffect | 90 | 90 | 0 |
| Counterfactual | 132 | 36 | 12 |
| DirectionalDerivative | 144 | 36 | 0 |
| Elasticity | 144 | 36 | 0 |
| InterventionalDistribution | 144 | 36 | 0 |
| InterventionResponse | 146 | 34 | 0 |
| MediationEffect | 90 | 90 | 0 |
| NestedCounterfactualEffect | 142 | 36 | 2 |
| PathSpecificEffect | 144 | 36 | 0 |
| PointDerivative | 144 | 36 | 0 |
| PulseEffect | 72 | 108 | 0 |
| ResponseCurve | 144 | 36 | 0 |
| ResponseJacobian | 144 | 36 | 0 |
| SemiElasticity | 144 | 36 | 0 |
| SustainedEffect | 72 | 108 | 0 |
| TemporalConditionalPulse | 72 | 108 | 0 |
| TemporalConditionalSustained | 72 | 108 | 0 |
| TemporalMediationEffect | 72 | 108 | 0 |
| TransportQuery | 90 | 90 | 0 |
| InterferenceQuery | 90 | 90 | 0 |

**Supported units: 3084 (matrix cells 2776, graphless rows 41, capabilities outside the matrix 267); not supported: 0 (owed records only 0, other 0).**

The 2776 matrix cells below are the matrix part of that count; graphless rows (`parity/support_graphless.toml`) and capabilities outside the matrix (`parity/capability_records.toml`) make up the rest. Entry-point routes are not counted.

| Query | Graph class | Structures | Inferences | Validations | Cells |
|---|---|---|---|---|---:|
| AnomalyAttribution | Dag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| AnomalyAttribution | Admg | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| AnomalyAttribution | Cpdag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| AnomalyAttribution | Pag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| AnomalyAttribution | TemporalDag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| AnomalyAttribution | TemporalCpdag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| AnomalyAttribution | TemporalPag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| AnomalyAttribution | TemporalAdmg | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| AverageDerivative | Dag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| AverageDerivative | Admg | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| AverageDerivative | Cpdag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| AverageDerivative | Pag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| AverageDerivative | TemporalDag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| AverageDerivative | TemporalCpdag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| AverageDerivative | TemporalPag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| AverageDerivative | TemporalAdmg | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| AverageEffect | Dag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| AverageEffect | Admg | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| AverageEffect | Cpdag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| AverageEffect | Pag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| AverageEffect | TemporalDag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| AverageEffect | TemporalCpdag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| AverageEffect | TemporalPag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| AverageEffect | TemporalAdmg | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| AverageEffect | CoDetermined | explicit | Frequentist, Bayesian | none | 2 |
| AverageEffect | Unknown | explicit | Frequentist, Bayesian | none | 2 |
| ChangeAttribution | Dag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none | 6 |
| ChangeAttribution | Admg | explicit, accepted, graph_posterior | Frequentist, Bayesian | none | 6 |
| ChangeAttribution | Cpdag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none | 6 |
| ChangeAttribution | Pag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none | 6 |
| ChangeAttribution | TemporalDag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none | 6 |
| ChangeAttribution | TemporalCpdag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none | 6 |
| ChangeAttribution | TemporalPag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none | 6 |
| ChangeAttribution | TemporalAdmg | explicit, accepted, graph_posterior | Frequentist, Bayesian | none | 6 |
| ConditionalEffect | Dag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| ConditionalEffect | Admg | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| ConditionalEffect | Cpdag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| ConditionalEffect | Pag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| ConditionalEffect | TemporalAdmg | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| Counterfactual | Dag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| Counterfactual | Admg | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| Counterfactual | Cpdag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| Counterfactual | Pag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none | 6 |
| Counterfactual | TemporalDag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| Counterfactual | TemporalCpdag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| Counterfactual | TemporalPag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| Counterfactual | TemporalAdmg | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| DirectionalDerivative | Dag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| DirectionalDerivative | Admg | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| DirectionalDerivative | Cpdag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| DirectionalDerivative | Pag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| DirectionalDerivative | TemporalDag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| DirectionalDerivative | TemporalCpdag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| DirectionalDerivative | TemporalPag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| DirectionalDerivative | TemporalAdmg | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| Elasticity | Dag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| Elasticity | Admg | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| Elasticity | Cpdag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| Elasticity | Pag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| Elasticity | TemporalDag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| Elasticity | TemporalCpdag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| Elasticity | TemporalPag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| Elasticity | TemporalAdmg | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| InterventionalDistribution | Dag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| InterventionalDistribution | Admg | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| InterventionalDistribution | Cpdag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| InterventionalDistribution | Pag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| InterventionalDistribution | TemporalDag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| InterventionalDistribution | TemporalCpdag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| InterventionalDistribution | TemporalPag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| InterventionalDistribution | TemporalAdmg | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| InterventionResponse | Dag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| InterventionResponse | Admg | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| InterventionResponse | Cpdag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| InterventionResponse | Pag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| InterventionResponse | TemporalDag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| InterventionResponse | TemporalCpdag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| InterventionResponse | TemporalPag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| InterventionResponse | TemporalAdmg | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| InterventionResponse | CoDetermined | explicit | Frequentist, Bayesian | none | 2 |
| MediationEffect | Dag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| MediationEffect | Admg | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| MediationEffect | Cpdag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| MediationEffect | Pag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| MediationEffect | TemporalAdmg | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| NestedCounterfactualEffect | Dag | explicit | Frequentist, Bayesian | none, cheap, full | 6 |
| NestedCounterfactualEffect | Dag | accepted | Frequentist, Bayesian | none, cheap, full | 6 |
| NestedCounterfactualEffect | Dag | graph_posterior | Frequentist, Bayesian | none, cheap | 4 |
| NestedCounterfactualEffect | Admg | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| NestedCounterfactualEffect | Cpdag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| NestedCounterfactualEffect | Pag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| NestedCounterfactualEffect | TemporalDag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| NestedCounterfactualEffect | TemporalCpdag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| NestedCounterfactualEffect | TemporalPag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| NestedCounterfactualEffect | TemporalAdmg | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| PathSpecificEffect | Dag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| PathSpecificEffect | Admg | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| PathSpecificEffect | Cpdag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| PathSpecificEffect | Pag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| PathSpecificEffect | TemporalDag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| PathSpecificEffect | TemporalCpdag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| PathSpecificEffect | TemporalPag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| PathSpecificEffect | TemporalAdmg | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| PointDerivative | Dag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| PointDerivative | Admg | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| PointDerivative | Cpdag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| PointDerivative | Pag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| PointDerivative | TemporalDag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| PointDerivative | TemporalCpdag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| PointDerivative | TemporalPag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| PointDerivative | TemporalAdmg | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| PulseEffect | TemporalDag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| PulseEffect | TemporalCpdag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| PulseEffect | TemporalPag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| PulseEffect | TemporalAdmg | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| ResponseCurve | Dag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| ResponseCurve | Admg | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| ResponseCurve | Cpdag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| ResponseCurve | Pag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| ResponseCurve | TemporalDag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| ResponseCurve | TemporalCpdag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| ResponseCurve | TemporalPag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| ResponseCurve | TemporalAdmg | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| ResponseJacobian | Dag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| ResponseJacobian | Admg | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| ResponseJacobian | Cpdag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| ResponseJacobian | Pag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| ResponseJacobian | TemporalDag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| ResponseJacobian | TemporalCpdag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| ResponseJacobian | TemporalPag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| ResponseJacobian | TemporalAdmg | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| SemiElasticity | Dag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| SemiElasticity | Admg | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| SemiElasticity | Cpdag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| SemiElasticity | Pag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| SemiElasticity | TemporalDag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| SemiElasticity | TemporalCpdag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| SemiElasticity | TemporalPag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| SemiElasticity | TemporalAdmg | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| SustainedEffect | TemporalDag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| SustainedEffect | TemporalCpdag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| SustainedEffect | TemporalPag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| SustainedEffect | TemporalAdmg | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| TemporalConditionalPulse | TemporalDag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| TemporalConditionalPulse | TemporalCpdag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| TemporalConditionalPulse | TemporalPag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| TemporalConditionalPulse | TemporalAdmg | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| TemporalConditionalSustained | TemporalDag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| TemporalConditionalSustained | TemporalCpdag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| TemporalConditionalSustained | TemporalPag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| TemporalConditionalSustained | TemporalAdmg | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| TemporalMediationEffect | TemporalDag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| TemporalMediationEffect | TemporalCpdag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| TemporalMediationEffect | TemporalPag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| TemporalMediationEffect | TemporalAdmg | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| TransportQuery | Dag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| TransportQuery | Admg | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| TransportQuery | Cpdag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| TransportQuery | Pag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| TransportQuery | TemporalAdmg | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| InterferenceQuery | Dag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| InterferenceQuery | Admg | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| InterferenceQuery | Cpdag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| InterferenceQuery | Pag | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
| InterferenceQuery | TemporalAdmg | explicit, accepted, graph_posterior | Frequentist, Bayesian | none, cheap, full | 18 |
<!-- generated:query-inventory:end -->

Some important distinctions when choosing a workflow:

- Static DAG graph-posterior paths include average effects, conditional effects,
  response curves, and one-coordinate intervention responses. Each has its own
  inference and validation restrictions.
- Temporal graph-posterior paths include Pulse and Sustained effects and
  temporal mediation on `TemporalDag`, `TemporalCpdag`, `TemporalPag` and
  `TemporalAdmg` under both inference modes at validation
  `none`/`cheap`/`full`.
- Frequentist and Bayesian envelopes on incomplete `TemporalCpdag` and
  `TemporalPag` graphs are supported for supported queries. A caller-supplied
  `ClassPrior` can provide mixture weights; enumerating completions alone does
  not assign probabilities.
- Derivatives run on `Dag`, `Cpdag`, `Pag` and `Admg` structures (explicit,
  accepted or a graph posterior) under both inference modes; a class answers
  through its identified completions.
- ADMG interventional distributions run through Python `analyze`/`prepare` and
  Rust `Study`.
- Supported anomaly and change attribution paths run through Python `analyze`/`prepare`
  and Rust `Study`; change attribution runs at validation `none` only.
- Supported transport and interference queries run through both Python `analyze`
  and Rust `Study`, with their explicit design assumptions.

`ResponseCurve` and `InterventionResponse` are supported on temporal DAG,
`TemporalCpdag` and `TemporalPag` graph-posterior cells (see the matrix for
the validation levels each admits). ADMG graph-posterior AverageEffect identifies each atom with `general.id` and
estimates `functional.effect`. These are different
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
supported row's evidence kind and limitations. A shared algorithm name is not
evidence of equivalent behavior.

## Choosing a workflow

### DoWhy

DoWhy and Antecedent overlap in model–identify–estimate–refute workflows and
graphical causal-model tooling. Evaluate DoWhy when its Python ecosystem,
examples, estimator integrations, or GCM workflows are the primary need.

Antecedent differs by making its support matrix and typed refusals part of the
runtime contract. It also carries accepted partial graphs and selected graph
posteriors into supported downstream analyses. That does not imply broad feature
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
effects and policy-oriented workflows. Antecedent ships native
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
The known-truth fixture in `conformance/estimate/learner_ate/fixture.json`
covers a binary-treatment linear SCM at one sample size; it does not support
nonlinear-provider or CATE interval calibration. Read each result's calibration status and matching coverage record;
coverage for a marginal ATE does not support pointwise CATE intervals.

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
[`examples/python/econml_cate_handoff.py`](https://github.com/iridae-dev/antecedent/blob/v3.0.0/examples/python/econml_cate_handoff.py):
Antecedent identifies and estimates ATE, then the caller optionally fits
EconML `LinearDML` on `spec.columns(data)` when `econml` is installed.


### Tigramite

Tigramite is the upstream reference used for selected PCMCI-family and
conditional-independence conformance fixtures. Evaluate it directly for
time-series discovery research, its current algorithm set, and its native
visualization workflow.

Antecedent's temporal discovery implementations are useful when discovery must
feed a supported effect, intervention, prepared-analysis, or artifact workflow in
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
downstream analysis is supported, not applicable, or refused.

## Current Antecedent boundaries

The following are current product boundaries or explicit matrix refusals:

- no 12-estimator EconML clone (native DML, DRLearner, and honest causal-forest CATE; the EconML adapter emits a set, not a learner);
- no plotting module;
- no R, Julia, or JavaScript bindings;
- no complete PAG-native ID/IDC;
- transport identification is complete only within the experimental-information
  families stated in [transport scope](guides/transport-scope.md); catalog-bound
  search is sound and incomplete, and `NotCertified` is not a
  non-transportability proof;
- for incomplete-class response uncertainty, use the query-specific contracts in
  [causal responses](causal-responses.md);
- no exact DAG pseudo-posterior enumeration beyond six nodes;
- no automatic estimator choice and no prior that can rescue identification.

These statements are not interchangeable:

- **not applicable** means a matrix coordinate does not denote;
- **refused** means the coordinate is meaningful but this release does not
  support it;
- **outside product scope** names a broader capability Antecedent does not aim
  to provide in this release line.

For exact current behavior, use the [support matrix](support-matrix.md). For the
implemented building blocks behind it, use [Capabilities](capabilities.md).
