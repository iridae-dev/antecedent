# Transported policy values, adaptive trials and period shifts

Three bounded routes evaluate a supplied object in a target population or
period under a declared selection diagram. All return point values; none learns
anything from the target.

| Route | Rust entry point | Value |
| --- | --- | --- |
| Fixed contingent policy in a target population | `antecedent_design::transported_policy::evaluate_transported_policy` | Sequential g-formula under the source's invariant mechanisms, the target's initial-state law and, when declared shifted, the target's own utility |
| Adaptive trial as a transport source | `antecedent_design::adaptive_trial_transport::transport_adaptive_trial` | Arm means standardized to a finite target stratum law |
| Policy across periods | `antecedent_design::temporal_policy_transport::evaluate_temporal_policy` | The same g-formula with time-invariant dynamics and a shifted initial state |

Each has a `verify_*` function that re-evaluates and refuses a value that does
not match. Provenance records: `transport.policy_evaluation`,
`transport.adaptive_trial_source`, `transport.period_shift_policy`.

Each value reaches a decision, an inverse query or an assumption range as a
transported response. The typed constructors in the facade
(`transported_policy_response`, `temporal_policy_response`,
`adaptive_trial_response`, `transported_synthesis_response`,
`surrogate_bridge_response`, in `analysis::transported_constructors`) re-run the
route and copy the populations, identities, assumptions and support label from
the producing result; none is typed by the caller. In Python the response is
read as `antecedent.transported_response.TransportedResponse`. The policy,
adaptive-trial and period routes are run through
`antecedent.transported_constructors.{from_policy, from_adaptive_trial, from_period_shift}`,
whose `evaluation` mapping carries the route's values, identity and premises, and each has a
re-deriving verifier, `verify_policy`, `verify_adaptive_trial` and `verify_period_shift`
(`python/tests/test_transported_constructors.py`,
`python/tests/test_transported_route_verification.py`).

## Transported policy

You declare, for every mechanism (initial state, each action's transition, each
observation channel, the state utility), whether a selection node points to it
(`Shifted`) or it is `Invariant`. An undeclared mechanism is refused by name.
The initial-state law may be shifted, and so may the utility when you supply the
target's own (`TargetUtility`). What an outcome is worth is a valuation, not a
causal mechanism, so the source cannot supply it: the target utility covers
exactly the source model's actions and latent states (per-action cost and
immediate value, the delayed value where the source action has a delayed outcome,
and the terminal value), every value finite and every cost non-negative, and it
names its provenance, a source label and a 64-character content digest. Both the
values and the provenance are bound into the evaluation identity and named in
the response's assumptions, so an edited value or provenance does not replay
through the verifier. A utility declared shifted with none supplied, or supplied
while declared invariant, is refused. A shifted utility relaxes nothing else:
transitions and observation channels must still be invariant and the target
initial law must still lie inside the source's support. You supply the target
initial-state law and its support must lie inside the source's, so every
reachable target history was reachable in the source. The value is
`V_T(pi) = sum over h of P_T(h | pi) * realized_T(h)`, where `realized_T` uses the
target utility when one is supplied and the source's otherwise. The period route
does not accept a shifted utility.

## Adaptive trial as a source

The trial is analysed through its `AdaptiveDesignContract` (assignment policy
history, realized probabilities, exposure history, outcome availability and
delay, interim analyses, stopping rule). Without it the trial is refused, never
analysed as if it were fixed. The request names the trial's own population
(`trial_population`, required): it must be non-blank, must differ from the
target and must be the selection diagram's source population, and it enters the
estimate identity. A stratum law declared invariant is checked against the
trial's empirical stratum law over its enrolled units (tolerance 1e-9); a
declaration the trial itself contradicts is refused rather than trusted. The contract's procedure cell must be licensed for
adaptively weighted scores. The stratum law may be shifted, the outcome given
stratum and arm must be invariant. For every target stratum with positive mass
and both compared arms the trial must hold analysed units whose realized
assignment probability is at or above the declared floor. The estimator is the
self-normalized inverse-realized-probability mean per stratum and arm,
standardized to the target law. It carries no interval and no sampling
distribution.

## Period shift

Time invariance is declared per mechanism and never assumed. The horizon and
the action and observation alphabets must match, and the policy table is never
re-keyed implicitly. Only the initial state may have moved.

## What is refused, and the remedy

| Refused request | Reason code | Remedy |
| --- | --- | --- |
| Mechanism left undeclared, or no selection diagram | `transport_missing_evidence` | Declare every mechanism |
| Shifted transition or observation mechanism | `transport_missing_evidence` (names the mechanism) | Re-evaluate in the target or use a model fitted there |
| Utility declared shifted with no target utility, or a target utility with the utility declared invariant | `transport_missing_evidence` (`target_utility_missing`), `invalid_argument` (`utility_declared_invariant`) | Supply the target's utility with its provenance, or mark the utility invariant |
| Target utility with no source label, a malformed digest, other actions or states than the model's, or a non-finite value | `transport_missing_evidence` (`target_utility_provenance_missing`), `invalid_argument` (`target_utility_digest`, `target_utility_support`, `target_utility_value`) | Declare the target's utility over the model's own actions and states |
| Shifted utility mechanism on the period route | `transport_missing_evidence` (names the mechanism) | Declare the utility time-invariant |
| Target initial law outside the source support | `transport_support_failure` | Restrict the target law to the source support |
| Different horizon or alphabet across periods | `invalid_argument` | Build a model for the target period |
| Adaptive trial with no retained contract | `adaptive_contract_incomplete` | Supply the `AdaptiveDesignContract` |
| Blank trial population, target equal to the trial population, or a diagram whose source is another population | `invalid_argument` (`adaptive_transport.population_label`, `target_is_source`), `transport_missing_evidence` (`diagram_binding_mismatch`) | Name the trial's population and the diagram built for it |
| Stratum law declared invariant that differs from the trial's own | `invalid_argument` (`adaptive_transport.stratum_law_declared_invariant`) | Declare the stratum law shifted, or correct the declaration |
| Contract cell not licensed for weighted scores, or fixed design | `route_not_supported` | Use the fixed-design transport route |
| Shifted outcome mechanism, or unsupported stratum/arm | `transport_missing_evidence`, `transport_support_failure` | Declare invariance honestly or collect the missing cell |
| Interval or posterior | none issued | Use the point as labelled; an interval needs a measured record |
