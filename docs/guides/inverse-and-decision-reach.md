# Inverse queries and decisions on composed, cohort and transported sources

Inverse queries and decisions can consume a composed mechanism, a cohort-mean
contrast and a transported response. Each source states what it cannot supply
before the engine runs, and its identity travels into the result.

| Source | Rust surface | Supplies |
| --- | --- | --- |
| Composed mechanisms | `antecedent_design::inverse_forward_source::composed_forward_source` | Linear-Gaussian: per-action interventional mean (point). Finite table: the exact finite law |
| Cohort-mean contrast | `cohort_forward_source` | A point; under a declared covariate shift it is a transported response |
| Transported response | `transported_forward_source`, `decision_transported_source::evaluate_transported_decision` | A point, or a transported law |
| Observational scenarios | `inverse_observational_scenarios::evaluate_observational_scenarios` | `P(S)`, `P(S and E)`, `P(E | S)`, `P(S | E)`, lift |

Python entries: `antecedent.inverse_sources` (`composed_forward_source`,
`cohort_forward_source`, `transported_forward_source`, `ForwardSource.evaluate`,
`.seal`, `.seal_per_action`, `observational_scenarios`),
`antecedent.transported_response` (`TransportedResponse`,
`TransportDeclaration`, `evaluate_transported`) and
`Contract.evaluate_transported`. Provenance records:
`decision.transported_source`, `inverse.observational_scenarios`.

## Point-only means point-only

A mean supplies the expectation of an affine utility and nothing else. A
quantile, a probability, a nonlinear utility, a hard constraint, regret or an
outcome-law input needs a law and is refused. No source supplies an interval
region: composition variance, a cohort model's standard error and the transport
step have no measured coverage. A source's own variance is kept as a diagnostic
and never becomes a feasibility statement. A Gaussian composition does not
supply a law the engine can integrate.

## Transported acceptance

A transported response is labelled at best extrapolative. A consumer that does
not accept a transported input refuses it with one code, `cell_not_licensed`,
whichever source carried it: an inverse query reading a cohort, composed or
response source under the native-target-only policy
(`inverse_forward_source.transported_not_accepted`), and a decision under the
default support policy, which admits only supported coordinates
(`transported_decision.transported_not_accepted`). Accepting is explicit: the
inverse query is told to accept transported inputs
(`accepts_transported=True` in Python), and a decision uses a policy that names
the weaker status. The receipt (source populations, target, diagram identities,
assumptions) travels into the result and the sealed artifact. A support rule of
the contract stronger than the label refuses `transport_support_failure`
(`transported_decision.support_rule_unmet`); declared exclusions remove actions
before the comparison and are part of the bound decision's identity.

A carried composition is a transported claim. A composed model whose
mechanisms were admitted into the target (carried from another population)
reaches an inverse query only with the receipt that carried it and an explicit
accept: its origin is labelled extrapolative with the receipt's support, never
supported. A composition with no such receipt, or a receipt of another route,
is refused.

## Typed sources and exported decisions

A transported point enters a decision through the typed constructors, which
take populations and identities from the producing result. The generic
`declared_point` / `declared_law` constructors are for a declared transport and
refuse the typed routes (policy, shifted cohort, synthesis); every typed route
requires digest transport identities, so a label typed by hand cannot stand in
for one. A decision on a transported point can be exported and consumed
(`transported_decision_v1`): the consumer rebuilds the contract, validates the
receipt, re-evaluates, and requires the identity and the action values to
match. A law is not exportable (`law_not_exportable`).

## Identity

`ForwardOrigin` records the source kind, identity, representation, support
label, assumed mechanisms or the whole receipt, and a digest of the claim handed
to the engine. `seal_on_source` binds it into the inverse-query artifact, and a
consumer re-derives the digest and refuses an origin that does not describe the
embedded claim. Replay checks the numbers against the origin; it does not
re-execute the producing route.

## Observational scenarios

`P(X in scenario | Y event)` from an aligned observational law over declared
scenario cells. It is associational: it ranks no action and recommends none.
The mass of the event no scenario explains is reported, not renormalized away.

## What is refused, and the remedy

| Refused request | Reason code or detail | Remedy |
| --- | --- | --- |
| Quantile, probability or nonlinear utility on a point source | `PointOnlyLawRequested` / `decision_contract_unsatisfied` | Supply a law (draws or a finite table) |
| Quantile or probability on a Gaussian composition | `ComposedLawNotMaterialized` | Draw the intervened law |
| Interval region from any source | `IntervalNotLicensed` / `cell_not_licensed` | Use the point as labelled |
| Transported input to a consumer that does not accept one (inverse query, cohort forward claim, decision under the default policy) | `cell_not_licensed` (`inverse_forward_source.transported_not_accepted`, `transported_decision.transported_not_accepted`) | Accept transported inputs explicitly, naming the weaker support status in the policy |
| Support rule stronger than the transported label | `transport_support_failure` (`transported_decision.support_rule_unmet`) | Weaken the rule for this input or measure in the target |
| Typed route through the declared constructor, or a hand-typed identity | `transported_decision.route_needs_typed_source`, `transport_identity_not_digest` | Use the typed constructor of the route |
| Native or calibrated label on a transported value | `transport_not_certified` | Keep the transport label |
| Scenario query on an interventional law or independent marginals | `joint_law_required` | Supply an observational aligned law |
| Event with no mass | typed refusal | Choose a supported event |
