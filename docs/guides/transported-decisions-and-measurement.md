# Transported cohort decisions and measurement models

Two routes carry something already learned in one population to a declared
target population. Both are point-only and both state the premise they stand on.
Neither inspects target outcomes, and neither produces an interval.

| Route | Rust entry point | What it returns |
| --- | --- | --- |
| Retained effect model applied to a target cohort | `antecedent_learn::transported_effect::transport_cohort_effect` | `TransportedCohortEffect`: one weighted cohort-mean contrast, point-only |
| Measurement model carried to another population | `antecedent_estimate::measurement_transport::transport_measurement_model` | `TransportedMeasurementModel` plus a receipt; applied through `apply` |

Python bindings follow the Python API refactor; this page names the Rust
surface only. Provenance records: `transport.covariate_shift_cohort_effect`,
`transport.measurement_model`.

## Cohort effect under a selection premise

A fitted effect model predicts a conditional contrast `tau(x)` and keeps the box
of feature values it was fitted on. Applying it to a cohort from another
population is a transport claim, issued only when two premises are met:

1. A `SelectionShiftDeclaration` states that the selection node separating the
   populations points only into the declared effect modifiers and that those
   modifiers are exactly the model's features. This is declared, never tested.
2. An `OverlapAssessment` of the target rows against the model's support box
   finds no feature disjoint from the fitted range and no row outside it. A box
   is a necessary condition for positivity, not an overlap or density guarantee.

The value is `sum_i w_i tau(x_i)` over named cohort rows with normalized weights.
It is labelled extrapolative. No standard error, interval or utility variance of
the source fit is inherited.

## Measurement model transport

A measurement model (sensitivity and specificity, a confusion matrix, an error
variance) is established for one population and refuses data from any other.
`transport_measurement_model` carries it only under a declared
`PopulationErrorStructure`:

- `Invariant` names every error parameter the model's route uses as the same in
  source and target. Naming specificity alone does not license a
  sensitivity-and-specificity correction.
- `Differential` states that the error differs, so nothing is carried. Only a
  model established for the target is used; a missing one is refused rather
  than replaced by another population's rates.

The result is the source model re-bound to the target together with a receipt
that binds the source model identity, both populations, the premise and the
basis. The corrected functional is never presented as native to the target.
Only the parameter-estimation uncertainty the correction already propagates is
reported, it is conditional on the premise and uncalibrated, and the uncertainty
of the premise itself is not propagated.

## What is refused, and the remedy

| Refused request | Reason code | Remedy |
| --- | --- | --- |
| No selection-shift declaration | `transport_missing_evidence` | Declare the selection node and the effect modifiers it enters |
| Declaration does not match the model, populations or selection structure (selection enters the mechanism, modifiers differ from features) | `transport_not_certified` | Correct the declaration or refit the model with the modifiers the diagram requires |
| No overlap, or rows outside the support box | `transport_support_failure` | Restrict the cohort to the fitted box or refit on data that covers it |
| Interval, standard error or law of the cohort effect | `cell_not_licensed` / `decision_contract_unsatisfied` | Use the point value; an interval needs a measured record at that coordinate |
| Measurement parameters not declared invariant, or only some of them | `transport_missing_evidence` / `transport_not_certified` | Name every parameter the route uses, or establish the model in the target |
| Differential error with no target model | `differential_error_not_pooled` | Validate the measurement model in the target population |
| Replicated measurements or an external latent posterior | `route_not_supported` | Replicates re-estimate in the target; a posterior is bound to its source snapshot |
| Data from another population or regime than the model's | `measurement_binding_mismatch` | Re-bind through the transport route, or use a target model |
