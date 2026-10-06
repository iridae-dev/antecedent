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
