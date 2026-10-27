# Estimate heterogeneous effects and evaluate treatment policies

Ask who benefits, and what a rule that treats some units and not others would be worth. This page covers four routes: a pointwise conditional effect (CATE) at a covariate profile, a conditional dose-response at chosen doses, the value of a fixed policy under randomized assignment, and the value of a policy when assignment was observational. They share one discipline: the interval or standard error describes the quantity named in the table, and none carries a coverage measurement.

| You want | Route | Needs |
|---|---|---|
| The effect at a covariate profile | `DrLearner::fit_pointwise_profiles` (Rust) | Repeated covariate profiles present in both arms |
| Response at chosen doses within baseline groups | `policy.ConditionalDoseResponse` | A dose density you supply |
| The value of a fixed binary policy, randomized data | `policy.evaluate_policy_doubly_robust` | Known assignment propensities and outcome predictions |
| The value of a policy, observational data | `observational_policy.policy_value` | A declared action support and adjustment columns |

## Observational policy value

Estimate the mean outcome if every unit received the action a policy picks from its covariates. This example is simulated: one confounder `z` drives treatment and outcome, `E[Y(0) | z] = 1 + z` and the effect is `1.5 z`, so the policy "treat when z > 0" has the closed-form value `1 + 1.5 phi(0)`. The known truth lets you check the estimate.

```python
import numpy as np
from antecedent import Execution, observational_policy

rng = np.random.default_rng(21)
n = 6000
z = rng.normal(size=n)
a = (rng.uniform(size=n) < 1 / (1 + np.exp(-z))).astype(float)
y = 1.0 + z + 1.5 * z * a + 0.5 * rng.normal(size=n)

result = observational_policy.policy_value(
    y,
    a,
    support=[0.0, 1.0],
    objective="maximize",
    evaluation="prespecified",
    policy_actions=(z > 0).astype(float),
    policy_origin="prespecified",
    reference=0.0,
    adjust={"z": z},
    execution=Execution(seed=0),
)
truth = 1.0 + 1.5 / np.sqrt(2.0 * np.pi)
assert abs(result.value - truth) < 4 * result.se
assert result.calibration == "unmeasured"
```

The Rust entry is `antecedent::analysis::observational_policy::ObservationalPolicyValue`. The test suite (`python/tests/test_observational_policy.py`, `observational_policy_value.rs`) checks this closed form and a three-action process.

### Declarations without defaults

- `support`: the actions the policy may take, two or more distinct levels. Every observed treatment value must be one of them (`treatment_level_undeclared` otherwise) and every action must have positive propensity.
- `objective`: `"maximize"` or `"minimize"`; it directs learned policies.
- `evaluation`: how the policy is obtained and scored.

An absent declaration refuses `required_option_missing` with its own detail (`policy_value.support_missing`, `.objective_missing`, `.evaluation_missing`, `.policy_actions_missing`, `.policy_origin_missing`, `.evaluation_fraction_missing`).

### Choose an evaluation

| `evaluation` | Policy | Scored on | What the value is |
|---|---|---|---|
| `"prespecified"` | the caller's `policy_actions`, one action per row | every row; propensities and outcome models cross-fitted | `E[Y(pi(Z))]` of a policy fixed without these rows |
| `"held_out"` | the greedy policy (best predicted outcome) learned on a seeded learning part | independent held-out rows (`evaluation_fraction`, stratified by treatment level) | the value of the one learned policy; the interval conditions on it |
| `"cross_fitted_learning"` | learned on the other folds for each fold | each fold, with the policy that never saw it | selection-aware value of the learning procedure at `(K - 1) / K` of the data, not of one final policy |

A prespecified policy must be declared `policy_origin="prespecified"`. Declaring `policy_origin="learned_on_evaluation_rows"` refuses `policy_learned_on_evaluation_rows`: a policy chosen to fit the rows that score it has an inflated value. The refusal names `held_out` and `cross_fitted_learning` as the alternatives. The test suite shows the inflation: with no treatment effect and 29 noise covariates on 300 rows, the in-sample greedy plug-in value is biased upward while the cross-fitted value is not.

### Estimator and assumptions

For each split a multinomial-logit propensity `e_a(Z)` and a per-action outcome regression `m_a(Z)` are fitted on the training rows. The score of a unit scored with action `pi(Z)` is `m_pi(Z) + 1{T = pi(Z)} (Y - m_pi(Z)) / e_pi(Z)`; the value is the mean score and the standard error is the standard deviation of the scores over the square root of the scored rows. The value of giving every unit each declared action, and (with `reference=`) the paired gain of the policy over one action, come from the same scores.

Identification needs conditional exchangeability given the `adjust` columns (you own them; no graph is checked), positivity of every declared action, consistency and no interference. The estimate is consistent if either the propensity or the outcome models are. `known_propensity=[0.5, 0.5]` (one probability per support action, or one column per action) declares the assignment probabilities of a randomized design and replaces the fitted propensity model; the report states `propensity_source`. The positivity floor `min_propensity` (default 0.01) applies to every declared action at every scored row.

### Refusals

| Reason code | Detail | When |
|---|---|---|
| `positivity_violated` | `policy_value.action_outside_support` | The policy takes an action outside the declared support; the action is named. |
| `positivity_violated` | `policy_value.action_propensity_below_floor` | A declared action's propensity falls below the floor at a scored row. |
| `policy_learned_on_evaluation_rows` | `policy_value.policy_learned_on_evaluation_rows` | The policy is declared learned on the scoring rows. |
| `treatment_level_undeclared` | `policy_value.undeclared_level` | An observed treatment outside the support. |
| `arm_not_populated` | `policy_value.action_not_populated`, `.training_split_missing_action`, `.sparse_action` | An action has no rows, or a training split has none or too few. |
| `invalid_argument` | `policy_value.invalid_*`, `.invalid_known_propensity`, `.reference_outside_support`, `.too_few_*` | Malformed support, reference, fraction, folds, floor or columns. |

### What the standard errors support

They are influence-function standard errors labelled `calibration == "unmeasured"`; no interval is produced and no coverage claim is made. For `cross_fitted_learning` the standard error pools the scores of all folds and treats the fold policies as fixed. That is reasonable when the learned policy stabilises across folds (the cross-fitting argument of Athey and Wager, 2021; Zhou, Athey and Wager, 2023), as a low-dimensional parametric rule does at moderate sample sizes. When the rule changes materially between folds the standard error is approximate and no correction is applied. The learned policy class is the greedy rule over per-action linear outcome models, not an optimal policy search. This route creates no support-matrix license.

## Evaluate a fixed policy under randomized assignment

`policy.evaluate_policy_doubly_robust` evaluates fixed binary recommendations with randomized assignment propensities and supplied `mu0`/`mu1` outcome predictions. For each row it combines the predicted potential outcome under the recommendation with the inverse-propensity weighted residual for the observed randomized action. It reports policy, reference and incremental values, net costs, and row-score standard errors for all three.

Provide unique evaluation subject IDs and one of two ownership contracts: training subject IDs disjoint from evaluation subjects, or fold IDs matching the fold excluded when each row's predictions and recommendation were produced. These checks validate only the supplied metadata; they do not inspect model training records. Randomization propensities must be strictly between zero and one. The standard errors assume independent evaluation subjects and condition on the supplied nuisance predictions and policy; no clustered standard error or calibrated interval is claimed.

The route evaluates a precomputed policy and does not train it. Identification requires the declared randomized assignment, consistency and no interference. It creates no support-matrix license. When propensities are unknown or the policy may have been learned from the same rows, use the observational route above.

## Conditional dose-response at chosen doses

`policy.ConditionalDoseResponse`, through `analyze(data, query=...).continuous_dose_response`, estimates response levels at dose targets you specify within baseline groups you supply. The native estimator uses an Epanechnikov kernel, 0.75(1 - u²), centered at each target and inverse supplied density weights, then reports a Hajek local mean. You provide the continuous-dose density evaluated at each observed dose and declare whether it is known or externally estimated.

Each group and target result reports local row count, effective sample size, minimum local density, maximum normalized weight and descriptive local outcome spread. The route refuses targets without the configured minimum local rows and rejects non-positive densities. Bandwidth sets the local window. The route fits no outcome or density model.

Identification requires conditional exchangeability given the supplied baseline groups, consistency, no interference, a correct density, and adequate continuous-dose overlap at each target. The result is point-only; the local outcome spread is not a standard error. It describes a conditional dose-response curve and does not select or evaluate a learned continuous-dose policy. For the unconditional response over a dose grid, see [Causal responses](../causal-responses.md).

## Pointwise CATE at a covariate profile

`DrLearner::fit_pointwise_profiles` is a deliberately narrow route for an unpenalized linear final stage. Each profile lists every adjustment variable in the order stored by the prepared problem. A profile is accepted only when the complete covariate tuple appears exactly among retained observations in both treatment arms and every matching row passes the configured propensity clip. Profiles outside this empirical joint support are refused: the route does not extrapolate or infer support from coordinate-wise ranges. That makes it useful for categorical or otherwise repeated covariate profiles; continuous profiles with no exactly repeated tuple are refused.

The estimand is the best linear projection of the CATE onto the adjustment variables (Semenova and Chernozhukov), evaluated at the profile. It equals the true pointwise CATE only under a correctly specified linear or saturated final stage. The estimate is the linear prediction of the cross-fitted doubly robust score; its standard error is the HC0 sandwich at that profile from the same score fit. The `lower_95` and `upper_95` fields are pointwise normal-approximation bounds. They are uncalibrated: no coverage record is attached, and they do not give simultaneous coverage across profiles.

This is a Rust API on `antecedent_estimate::DrLearner`; Python has no binding for it. A Python result exposes rowwise `result.estimate.cate` and `cate_se` instead, which are not profile intervals, and the rowwise `EffectEstimate.heterogeneity` outputs and forest leaf dispersion must not be substituted for the profile result.

## Related

- A unit-level counterfactual effect is a different estimand; read its per-unit intervals and support flags in [Read counterfactual unit effects](../python-options.md#read-counterfactual-unit-effects).
- Compare several average effects with simultaneous (max-t) intervals and BH/BY-adjusted p-values in [Estimate a family of average effects](../python-options.md#estimate-a-family-of-average-effects).
- [Learner-backed estimation with DML scores](dml-scores.md)
