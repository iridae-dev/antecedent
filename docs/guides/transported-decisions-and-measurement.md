# Transport into decisions and measurement models

You can carry something already learned in one population, such as a fitted effect model, a measurement model's error rates, a policy, an adaptive trial or a provider's output, into a declared target population and use it in a decision. Each route states the premise it stands on, inspects no target outcomes and returns a point with that premise attached. None produces an interval.

## Run a small example

Carry sensitivity and specificity from a source clinic under an explicit invariance premise and compare the target correction with the closed form.

<!-- example: examples/python/transported_measurement.py -->

The sections below give the contracts for each route and the requests that are refused.

## Routes

| You hold | Route | Rust entry point | You get |
| --- | --- | --- | --- |
| A retained effect model | Cohort effect under a selection premise | `antecedent_learn::transported_effect::transport_cohort_effect` | One weighted cohort-mean contrast, point-only |
| A measurement model (sensitivity, specificity, error variance) | Measurement model transport | `antecedent_estimate::measurement_transport::transport_measurement_model` | The model re-bound to the target plus a receipt |
| Validation samples from several populations with differing error | Pooled differential error | `antecedent_estimate::measurement_transport_pooled` | A prevalence correction with the invariance premise's uncertainty propagated |
| A fixed contingent policy | Transported policy | `antecedent_design::transported_policy::evaluate_transported_policy` | A sequential g-formula value in the target |
| An adaptive trial | Adaptive trial as a source | `antecedent_design::adaptive_trial_transport::transport_adaptive_trial` | Arm means standardized to a target stratum law |
| A policy and an earlier period | Period shift | `antecedent_design::temporal_policy_transport::evaluate_temporal_policy` | The same g-formula with a shifted initial state |
| An external provider's output | Provider-sourced transport | `antecedent_contract::provider_transport::ProviderTransportSource::admit` | An admitted source and a portable handoff |

A cohort effect is read in Python as a `cohort_decision` source and, under a declared covariate shift, becomes a transported response with `antecedent.transported_response.TransportedResponse.from_shifted_cohort`. It is decided on with `Contract.evaluate_transported` or read by an inverse query through `antecedent.inverse_sources.cohort_forward_source` (see [inverse queries and decisions](inverse-outcome.md#inverse-queries-on-composed-cohort-and-transported-sources)). The measurement route is reached through `antecedent.measurement_model`. Policy, adaptive-trial and period values reach a decision as transported responses through `antecedent.transported_constructors.{from_policy, from_adaptive_trial, from_period_shift}`, each with a re-deriving verifier (`verify_policy`, `verify_adaptive_trial`, `verify_period_shift`) that refuses a value which does not match. The typed constructors copy populations, identities, assumptions and the support label from the producing result; none is typed by the caller.

## Cohort effect under a selection premise

A fitted effect model predicts a conditional contrast `tau(x)` and keeps the box of feature values it was fitted on. Applying it to a cohort from another population is a transport claim, issued only when two premises hold:

1. A `SelectionShiftDeclaration` states that the selection node separating the populations points only into the declared effect modifiers, and that those modifiers are exactly the model's features. This is declared, never tested.
2. An `OverlapAssessment` of the target rows against the model's support box finds no feature disjoint from the fitted range and no row outside it. A box is a necessary condition for positivity, not an overlap or density guarantee.

The value is `sum_i w_i tau(x_i)` over named cohort rows with normalized weights, labelled extrapolative. No standard error, interval or utility variance of the source fit is inherited. A model that names a feature twice is refused before anything is applied.

## Measurement model transport

A measurement model is established for one population and refuses data from any other. `transport_measurement_model` carries it only under a declared `PopulationErrorStructure`:

- `Invariant` names every error parameter the model's route uses as the same in source and target. Naming specificity alone does not license a sensitivity-and-specificity correction.
- `Differential` states that the error differs, so nothing is carried. Only a model established for the target is used; a missing one is refused rather than replaced by another population's rates.

The result is the source model re-bound to the target, with a receipt binding the source model identity, both populations, the premise and the basis. The corrected functional is never presented as native to the target. Only the parameter-estimation uncertainty the correction already propagates is reported; it is conditional on the premise and uncalibrated, and the uncertainty of the premise itself is not propagated. When several models are held, the one established for the requested population, regime and arm is used.

### Pooled differential error

When error rates differ by population and several populations hold their own validation samples, `transport_pooled_differential_prevalence` pools them instead of refusing (Python: `antecedent.measurement_models.transport_pooled_differential`). The logits of sensitivity and specificity are modelled as draws from a population distribution whose between-population variance `tau^2` is estimated (DerSimonian and Laird); invariance is the special case `tau^2 = 0`, so the premise's uncertainty is propagated rather than assumed away. The target's rates are the predictive law of that distribution, updated by the target's own validation sample when one exists. The Rogan-Gladen correction of the recorded prevalence is propagated by simulation. A draw with a non-positive Youden index is discarded and counted; more than five percent refuse (`differential_error_not_pooled`; next step: validate in the target). The interval is an uncalibrated simulation interval, for binary sensitivity and specificity only.

## Transported policy

You declare, for every mechanism (initial state, each action's transition, each observation channel, the state utility), whether a selection node points to it (`Shifted`) or it is `Invariant`. An undeclared mechanism is refused by name. The value is `V_T(pi) = sum over h of P_T(h | pi) * realized_T(h)`: a sequential g-formula under the source's invariant mechanisms and your target initial-state law, whose support must lie inside the source's so that every reachable target history was reachable in the source.

The initial-state law may be shifted, and so may the utility when you supply the target's own (`TargetUtility`). What an outcome is worth is a valuation, not a causal mechanism, so the source cannot supply it. The target utility covers exactly the source model's actions and latent states, every value finite and every cost non-negative, and it names its provenance, a source label and a 64-character content digest. Both are bound into the evaluation identity, so an edited value does not replay through the verifier. A shifted utility relaxes nothing else: transitions and observation channels must still be invariant.

## Adaptive trial as a source

The trial is analysed through its `AdaptiveDesignContract` (assignment history, realized probabilities, exposure history, outcome availability and delay, interim analyses, stopping rule). Without it the trial is refused, never analysed as if it were fixed. The request names the trial's own population (`trial_population`): non-blank, different from the target, and the selection diagram's source population. A stratum law declared invariant is checked against the trial's own stratum law over its enrolled units (tolerance 1e-9); a declaration the trial contradicts is refused. The stratum law may be shifted, but the outcome given stratum and arm must be invariant. For every target stratum with positive mass and both compared arms, the trial must hold units whose realized assignment probability is at or above the declared floor. The estimator is the self-normalized inverse-realized-probability mean per stratum and arm, standardized to the target law, with no interval.

## Period shift

Time invariance is declared per mechanism and never assumed. The horizon and the action and observation alphabets must match, and the policy table is never re-keyed implicitly. Only the initial state may have moved, and the period route does not accept a shifted utility.

## Provider-sourced transport

An external provider can supply a quantity as source evidence for a transport, and a transported claim can be handed to an external consumer. Both go through the interface contract and are decided from declarations before any number is computed; nothing here executes a provider, callback or declared code. Python: `antecedent.transport_handoff` (`TransportIdentity`, `TransportedClaimOffer`, `hand_to_callback`, `consume_for_callback`, `TransportHandoff`) over a `provider_callbacks.CallbackDeclaration`.

Check the connection between a declared transported offer and a callback input, persist the receipt, and refuse a native-only consumer. This recipe covers handoff rather than source admission or numerical transport identification.

<!-- example: examples/python/provider_transport.py -->

**Admission.** A source declares one provider object at an exact request, its trust (attestation and exact-request verification stay apart), the source population and regime of the quantity, the invariance and selection assumptions it relies on, and the evidence it rests on. `admit` refuses a missing population or regime, a quantity the provider does not declare, a native licence, a verification receipt that does not cover the object, and a provider that is only attested where the route requires verification. An admitted source is an `external_claim` in the evidence ledger and never native target data. Connect an admitted source with `connect_bound_provider_source` (and `hand_to_consumer_from_sources` for a handoff), which first requires the source to be bound to that transport's population and carried assumptions.

**Handoff.** A `TransportedClaimOffer` carries its `TransportIdentity` (source populations, selection diagram, assumptions). A consumer that accepts only native target data refuses it. An accepted handoff stays a transported external claim and its calibration label is never raised. The portable record (`transport_handoff_v1`) is re-derived by the consumer, which rebuilds the offer and contract, re-runs the connection decision and recomputes identities. A stored handoff cannot be relabelled upward. A declared population is the producer's declaration, not proof that data came from it, and a verification receipt covers provider behaviour at exact requests, not the truth of an invariance assumption.

## What is refused, and what to do

| Refused request | Reason code | Next step |
| --- | --- | --- |
| No selection-shift declaration | `transport_missing_evidence` | Declare the selection node and the effect modifiers it enters |
| Declaration does not match the model, populations or selection structure | `transport_not_certified` | Correct the declaration or refit with the modifiers the diagram requires |
| A model that names a feature twice | `invalid_argument` (`transported_effect.model_invalid`) | Give every feature one name |
| No overlap, or rows outside the support box | `transport_support_failure` | Restrict the cohort to the fitted box or refit on data that covers it |
| Interval, standard error or law of a cohort effect | `cell_not_licensed` / `decision_contract_unsatisfied` | Use the point; an interval needs a measured record at that coordinate |
| Measurement parameters not declared invariant, or only some of them | `transport_missing_evidence` / `transport_not_certified` | Name every parameter the route uses, or establish the model in the target |
| Differential error with no target model, or pooling with fewer than two validated sources | `differential_error_not_pooled` | Validate in the target population or in a second source |
| Replicated measurements or an external latent posterior | `route_not_supported` | Replicates re-estimate in the target; a posterior is bound to its source snapshot |
| Data from another population or regime than the model's | `measurement_binding_mismatch` | Re-bind through the transport route, or use a target model |
| Mechanism left undeclared, or no selection diagram | `transport_missing_evidence` | Declare every mechanism |
| Shifted transition or observation mechanism | `transport_missing_evidence` (names the mechanism) | Re-evaluate in the target or use a model fitted there |
| Utility declared shifted with no target utility, or a target utility with the utility declared invariant | `transport_missing_evidence` (`target_utility_missing`), `invalid_argument` (`utility_declared_invariant`) | Supply the target's utility with provenance, or mark the utility invariant |
| Target utility with no source label, a malformed digest, other actions or states than the model's, or a non-finite value | `target_utility_provenance_missing`, `target_utility_digest`, `target_utility_support`, `target_utility_value` | Declare the target's utility over the model's own actions and states |
| Target initial law outside the source support | `transport_support_failure` | Restrict the target law to the source support |
| Different horizon or alphabet across periods | `invalid_argument` | Build a model for the target period |
| Adaptive trial with no retained contract | `adaptive_contract_incomplete` | Supply the `AdaptiveDesignContract` |
| Blank trial population, target equal to the trial population, or a diagram whose source is another population | `adaptive_transport.population_label`, `target_is_source`, `diagram_binding_mismatch` | Name the trial's population and the diagram built for it |
| Stratum law declared invariant that differs from the trial's own | `adaptive_transport.stratum_law_declared_invariant` | Declare it shifted, or correct the declaration |
| Contract not licensed for weighted scores, or a fixed design | `route_not_supported` | Use the fixed-design transport route |
| Provider with no population or regime; attested-only provider where verification is required | none issued | Declare both; obtain a verification receipt covering the object |
| Provider output claimed as native target data; a handoff to a native-only consumer | none issued | Present it as a transported external claim; use a consumer that accepts transported claims |
| A stored handoff consumed against another callback declaration | `provider_transport.identity_mismatch` | Consume it against the declaration it was sealed for |
| Relabelled calibration, provenance or evidence in a stored artifact | none issued | Re-seal from the offer; the consumer re-derives it |
| Interval, posterior or calibration from any route above | none issued | Use the point as labelled; an interval needs a measured record at the target |

## Related

- [Carry an effect to another population](transport/carry-an-effect.md)
- [Assumption ranges and revalidation](assumption-range-and-revalidation.md): carry the remaining premise into the decision as a range
- [Provider callbacks](../provider-callbacks.md) and [measurement models](../measurement-models.md)
- [Refusal codes](../reference/refusal-codes.md)
