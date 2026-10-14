# Transported policy values, adaptive trials and period shifts

Three bounded routes evaluate a supplied object in a target population or
period under a declared selection diagram. All return point values; none learns
anything from the target.

| Route | Rust entry point | Value |
| --- | --- | --- |
| Fixed contingent policy in a target population | `antecedent_design::transported_policy::evaluate_transported_policy` | Sequential g-formula under the source's invariant mechanisms and the target's initial-state law |
| Adaptive trial as a transport source | `antecedent_design::adaptive_trial_transport::transport_adaptive_trial` | Arm means standardized to a finite target stratum law |
| Policy across periods | `antecedent_design::temporal_policy_transport::evaluate_temporal_policy` | The same g-formula with time-invariant dynamics and a shifted initial state |

Each has a `verify_*` function that re-evaluates and refuses a value that does
not match. Provenance records: `transport.policy_evaluation`,
`transport.adaptive_trial_source`, `transport.period_shift_policy`.

## Transported policy

You declare, for every mechanism (initial state, each action's transition, each
observation channel, the state utility), whether a selection node points to it
(`Shifted`) or it is `Invariant`. An undeclared mechanism is refused by name.
Only the initial-state law may be shifted. You supply the target initial-state
law and its support must lie inside the source's, so every reachable target
history was reachable in the source. The value is
`V_T(pi) = sum over h of P_T(h | pi) * realized(h)`.

## Adaptive trial as a source

The trial is analysed through its `AdaptiveDesignContract` (assignment policy
history, realized probabilities, exposure history, outcome availability and
delay, interim analyses, stopping rule). Without it the trial is refused, never
analysed as if it were fixed. The contract's procedure cell must be licensed for
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
| Shifted transition, observation or utility mechanism | `transport_missing_evidence` (names the mechanism) | Re-evaluate in the target or use a model fitted there |
| Target initial law outside the source support | `transport_support_failure` | Restrict the target law to the source support |
| Different horizon or alphabet across periods | `invalid_argument` | Build a model for the target period |
| Adaptive trial with no retained contract | `adaptive_contract_incomplete` | Supply the `AdaptiveDesignContract` |
| Contract cell not licensed for weighted scores, or fixed design | `route_not_supported` | Use the fixed-design transport route |
| Shifted outcome mechanism, or unsupported stratum/arm | `transport_missing_evidence`, `transport_support_failure` | Declare invariance honestly or collect the missing cell |
| Interval or posterior | none issued | Use the point as labelled; an interval needs a measured record |
