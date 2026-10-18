# Observational policy value

`antecedent.observational_policy.policy_value` (Rust:
`antecedent::analysis::observational_policy::ObservationalPolicyValue`) estimates the value of
a treatment policy from **observational** data: the mean outcome if every unit were given the
action the policy picks from its covariates. The randomized route in
[Doubly robust policy evaluation](../doubly-robust-policy-evaluation.md) takes precomputed
recommendations and known randomization propensities; this one estimates the propensities from
the data over a declared treatment support, and says where the policy came from.

## Declarations without defaults

- `support`: the actions the policy may take, two or more distinct levels. Every observed
  treatment value must be one of them (`treatment_level_undeclared` otherwise) and every
  action must have positive propensity.
- `objective`: `"maximize"` or `"minimize"`. It directs learned policies.
- `evaluation`: how the policy is obtained and scored.

An absent declaration refuses `required_option_missing` with its own detail
(`policy_value.support_missing`, `.objective_missing`, `.evaluation_missing`,
`.policy_actions_missing`, `.policy_origin_missing`, `.evaluation_fraction_missing`).

## Evaluations

| `evaluation` | Policy | Scored on | What the value is |
|---|---|---|---|
| `"prespecified"` | the caller's `policy_actions`, one action per row | every row; propensities and outcome models cross-fitted | `E[Y(pi(Z))]` of a policy fixed without these rows |
| `"held_out"` | the greedy policy (best predicted outcome) learned on a seeded learning part | independent held-out rows (`evaluation_fraction`, stratified by treatment level) | the value of the one learned policy; the interval conditions on it |
| `"cross_fitted_learning"` | learned on the other folds for each fold | each fold, with the policy that never saw it | selection-aware value of the learning procedure at `(K - 1) / K` of the data, not of one final policy |

A prespecified policy must be declared `policy_origin="prespecified"`. Declaring
`policy_origin="learned_on_evaluation_rows"` refuses `policy_learned_on_evaluation_rows`: a
policy chosen to fit the rows that score it has an inflated value, and the refusal names
`held_out` and `cross_fitted_learning` as the honest alternatives. The test suite shows the
inflation: with no treatment effect and 29 noise covariates on 300 rows, the in-sample greedy plug-in
value is biased upward while the cross-fitted value is not.

## Estimator and assumptions

For each split a multinomial-logit propensity `e_a(Z)` and a per-action outcome regression
`m_a(Z)` are fitted on the training rows. The score of a unit scored with action `pi(Z)` is
`m_pi(Z) + 1{T = pi(Z)} (Y - m_pi(Z)) / e_pi(Z)`; the value is the mean score and the
standard error the standard deviation of the scores over the square root of the scored rows.
The value of giving every unit each declared action, and (with `reference=`) the paired gain of
the policy over one action, come from the same scores.

Identification needs conditional exchangeability given the `adjust` columns (owned by the
caller; no graph is checked), positivity of every declared action, consistency and no
interference. The estimate is consistent if either the propensity or the outcome models are.
`known_propensity=[0.5, 0.5]` (one probability per support action, or one column per action) declares the
assignment probabilities of a randomized design and replaces the fitted propensity model; the report
states `propensity_source`. The positivity floor `min_propensity` (default 0.01) applies to every declared action at every
scored row.

## Refusals

| Reason code | Detail | When |
|---|---|---|
| `positivity_violated` | `policy_value.action_outside_support` | The policy takes an action outside the declared support; the action is named. |
| `positivity_violated` | `policy_value.action_propensity_below_floor` | A declared action's propensity falls below the floor at a scored row. |
| `policy_learned_on_evaluation_rows` | `policy_value.policy_learned_on_evaluation_rows` | The policy is declared learned on the scoring rows. |
| `treatment_level_undeclared` | `policy_value.undeclared_level` | An observed treatment outside the support. |
| `arm_not_populated` | `policy_value.action_not_populated`, `.training_split_missing_action`, `.sparse_action` | An action has no rows, or a training split has none or too few. |
| `invalid_argument` | `policy_value.invalid_*`, `.invalid_known_propensity`, `.reference_outside_support`, `.too_few_*` | Malformed support, reference, fraction, folds, floor or columns. |

## What is not claimed

The standard errors are influence-function standard errors labelled `calibration ==
"unmeasured"`; no interval is produced and no coverage claim is made until a coverage record
exists (the designs are written as ignored calibration tests). For `cross_fitted_learning` the
standard error pools the scores of all folds and treats the fold policies as fixed. That is valid
when the learned policy stabilises across folds (the cross-fitting argument of Athey and Wager,
2021; Zhou, Athey and Wager, 2023), as a low-dimensional parametric rule does at moderate sample
sizes; when the rule changes materially between folds the standard error is approximate and no
correction is applied. The learned policy class is the
greedy rule over per-action linear outcome models, not an optimal policy search. This route
creates no support-matrix license.

Verification against known truth: with `E[Y | A = 0, z] = 1 + z` and effect `1.5 z` the policy
"treat when z > 0" has value `1 + 1.5 phi(0)`; a three-action process gives a second closed
form (`observational_policy_value.rs`, `python/tests/test_observational_policy.py`).
