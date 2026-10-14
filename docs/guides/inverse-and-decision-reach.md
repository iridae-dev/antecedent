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

Provenance records: `decision.transported_source`,
`inverse.observational_scenarios`.

## Point-only means point-only

A mean supplies the expectation of an affine utility and nothing else. A
quantile, a probability, a nonlinear utility, a hard constraint, regret or an
outcome-law input needs a law and is refused. No source supplies an interval
region: composition variance, a cohort model's standard error and the transport
step have no measured coverage. A source's own variance is kept as a diagnostic
and never becomes a feasibility statement. A Gaussian composition does not
supply a law the engine can integrate.

## Transported acceptance

A transported response is labelled at best extrapolative. The default support
policy admits only supported coordinates, so a decision on a transported
response needs a policy that names the weaker status; that is your acceptance of
a transported claim. The receipt (source populations, target, diagram identities,
assumptions) travels into the decision result.

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
| Transported response under the default support policy | `SupportBelowPolicy` | Name the weaker support status in the policy |
| Native or calibrated label on a transported value | `transport_not_certified` | Keep the transport label |
| Scenario query on an interventional law or independent marginals | `joint_law_required` | Supply an observational aligned law |
| Event with no mass | typed refusal | Choose a supported event |
