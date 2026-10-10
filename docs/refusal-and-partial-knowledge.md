# Refusal and partial knowledge

Sometimes the correct causal result is not a number. Antecedent distinguishes
states because they imply different next actions.

- **Not identified:** the declared structure and evidence do not determine the
  request; obtain different evidence or justify stronger assumptions.
- **Missing evidence/provider:** a formula may exist, but a required joint law
  or statistical implementation is absent.
- **Support failure:** the causal question may be identified while the actual
  data do not cover the requested intervention or population.
- **Not certified/licensed:** a meaningful request is outside the implemented
  theorem or evidence-backed execution surface.
- **Budget or numerical failure:** execution did not complete; this is not a
  scientific impossibility claim.

Structural transport makes this especially visible through typed outcomes such
as `ProvenNonTransportable`, `NotCertified`, `MissingEvidence`,
`MissingProvider`, `UnsupportedEvaluator`, `SupportFailure`,
`NumericalFailure`, and `BudgetCancel`. The principle applies everywhere:
information that is unavailable must remain unavailable, not become an empty
success object, `NaN`, or a generic error string.

## Structured refusals and their remedies

In 2.3 scientific stage refusals share the structured
base, `antecedent.errors.StructuredRefusal` (itself a `CausalUnsupportedError`).
It exposes machine-readable fields, so you branch on fields rather than parse
a message. Input type, value and resource errors can instead raise ordinary
`CausalTypeError`, `CausalValueError` or `CausalResourceError` without a scientific
reason code.

| Field | Meaning |
| --- | --- |
| `code` (alias `reason_code`) | the registered refusal code, such as `cell_not_licensed`, `route_not_supported`, `transport_proven_non_transportable` or `invalid_argument` |
| `detail` | the namespaced `family.slot`, such as `msm_sensitivity.lambda_below_one` or `composition_bundle.node_not_found` |
| `offending` | the field, quantity, node or identity the refusal is about, or `None` |
| `remedy` | what the caller can change in order to proceed, or `None` |
| `stage` | the refusing stage |
| `message` | human-readable context |

```python
from antecedent import msm_sensitivity
from antecedent.errors import StructuredRefusal

try:
    msm_sensitivity.msm_ate_sensitivity(strata, 0.5)
except StructuredRefusal as refusal:
    if refusal.code == "invalid_argument":
        ...                      # a malformed argument: fix the call
    elif refusal.code == "cell_not_licensed":
        ...                      # a meaningful request outside the licensed surface
    print(refusal.detail, refusal.offending, refusal.remedy)
```

A refusal whose code is `invalid_argument` rejects a malformed argument rather than
a scientific route, and is also a `CausalValueError`, so `except CausalValueError`
catches it. A refusal is registered: its code is one of the reasons in the registry
and its detail is a literal under its route's namespace.

The 2.3 families built on the base are: `DecisionRefusal` (and its
`ScenarioDecisionRefusal` and `CompositionRefusal` subtypes), `SensitivityRefusal`,
`MsmSensitivityRefusal`, `MechanismDiscrepancyRefusal`, `TransportedCounterfactualRefusal`
(with a two-model `witness` when it proves non-transportability),
`ScenarioInvarianceRefusal`, `RepairRefusal`, `RecalcRefusal` (with
`RecalcUnavailable`, `ScoreResumeRefusal` and `ScoreResumeUnavailable`),
`CompositionBundleRefusal` (with one subclass per stage, including
`NodeNotFoundRefusal`, `TamperedQuantityRefusal` and `ExpectedIdentityMismatchRefusal`),
and the earlier estimator refusals `VectorTreatmentRefusal`,
`CategoricalTreatmentRefusal` and `CompactExportRefusal`. Many of them are also importable
lazily from `antecedent.errors`. `ExternalRefusal` and `DesignRankingRefusal` (with
`SignalProviderRefusal`, `CostUnitsRefusal`, `SourceOverlapRefusal`) are
`CausalUnsupportedError` subclasses that carry the same `reason_code`, `remedy`,
`stage`, `detail` and `offending` fields. To handle any 2.3 refusal generically,
catch `CausalUnsupportedError` and read those attributes.

Common remedies:

| Refusal | What to change |
| --- | --- |
| `cell_not_licensed` / `*.composition_not_licensed` | Report the sampling interval next to the assumption range; do not compose them. No composition method is licensed. |
| `cell_not_licensed` / `learned_joint_transport.route_frozen` | Use the checked `transport.advanced.learned_joint_transport` adapter inside its measured protocol. The separate `prepare_learned_continuous` IID route offers a point estimate or its scoped analytic interval via `prepared.interval()`; these methods have distinct contracts. |
| `cell_not_licensed` / `transported_counterfactual.nonadditive_mechanism` and its siblings | Declare the premise in `Premises` if you can defend it, or choose an assignment inside the affine-additive class. |
| `transport_proven_non_transportable` | Read `witness`: two models agree on the source and differ on the target. Remove the selection on that mechanism or supply target evidence for it. |
| `decision_contract_unsatisfied` / `native_claims.source_not_supplied` | A mean cannot answer a quantile or probability. Supply aligned joint draws. |
| `route_not_supported` / `decision_evaluation.mean_source_not_replayable` | `Decision.export()` requires aligned joint draws; `decision.replay` uses that same law. For a bound external mean claim, `composition_bundle.mean_decision` supplies a distinct point-only bundle artifact. Native mean workflows retain their own source-bound recalculation/replay contracts. |
| `design_cost_units_mismatch` | Supply a `CostMap` from the study cost unit to the decision's utility unit, or consume under the cost map the ranking used. |
| `score_table_unavailable` / `recalc.unavailable_data` | The request needs the data or a fit; supply the data and refit, or ask only for a same-row reweighting of the frozen scores. |
| `composition_bundle.node_not_found` | Add the node before connecting it (`NodeNotFoundRefusal.offending` is the missing id). |
| `invalid_argument` | Fix the argument named in `detail` and `offending`. |

See also the [2.3 lifecycle](2_3-lifecycle.md) for the refusal-handling pattern in
context.

## Remedies

Some refusals also name what the caller can change to proceed. The remedy is
a separate, optional text field, never part of the message, and it never adds
or changes a reason code: Rust reads it with `EstimationError::remedy()` (and
`CausalError::remedy()` at the facade), Python with the `remedy` attribute
every `CausalError` carries. A refusal that names no remedy reads `None`.
Estimation errors are not serialized into artifacts, so no artifact or wire
format changes with it.

| Refusal | Remedy |
| --- | --- |
| Additive-GAM non-convergence (`additive GAM target did not converge`, `additive GAM nuisance did not converge`, and the weighted nuisance form) | Standardize the treatment and adjustment columns, drop or coarsen near-collinear or near-constant adjustment covariates, or raise `nuisance_lambda` / lower `nuisance_basis` in the response options. |
| `plug-in response Jacobian supports at most two treatments` | Query the Jacobian over at most two treatments at a time (one `ResponseJacobian` per pair, with the remaining treatments in the adjustment set where the graph licenses it), or an `AverageDerivative` per treatment. |
| `plug-in directional derivative supports at most two treatments` | Restrict the direction to at most two treatments, or query a `ResponseJacobian` per pair and take the inner product with the direction. |
| `ConditionalLinearAdjustment currently supports one effect modifier` (and `conditional arm scores require AllObserved and exactly one modifier` when several modifiers were asked for) | The frequentist interaction regression and its calibrated interval cover one modifier. For several, use the Bayesian conditional estimator (`.estimator(EstimatorId::BayesianConditional).inference(InferenceMode::Bayesian(..))`), hand the identified adjustment set to an external CATE learner with `antecedent.handoff.econml(result, modifiers=[...])`, or fit one `ConditionalEffect` per modifier. |
| Python `ConditionalEffect takes one modifier column` (a sequence passed as `modifier`) | Use `antecedent.handoff.econml(result, modifiers=[...])` or one `ConditionalEffect` per modifier; the multi-modifier Bayesian conditional estimator is reachable from Rust (`EstimatorId::BayesianConditional`). |

```python
query = antecedent.ResponseJacobian(["a", "b", "c"], ["y"], at=[0.0, 0.0, 0.0])
try:
    antecedent.analyze(data, query=query, graph=graph)
except antecedent.CausalError as err:
    print(err)         # plug-in response Jacobian supports at most two treatments
    print(err.remedy)  # query the Jacobian over at most two treatments at a time ...
```
