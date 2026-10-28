# Optimise history-dependent policies

A scheduled intervention fixes a value for each step in advance. An adaptive policy
chooses each step's action from what has been observed: the current value of a
variable, or a value several steps back. Antecedent searches a bounded family of
such rules and evaluates every candidate through a compatible causal response
model, so the best rule is compared with the best fixed schedule on the same draws.

## Declare a family

A family names the features a rule may read, the rule's shape and a closed box for
each parameter. Every action is clipped into the family's action interval.

| Rule | Parameters |
| --- | --- |
| `Schedule` | one action per step (the fixed-schedule baseline) |
| `Threshold` | threshold, action below, action at or above |
| `Piecewise` | `k` ordered cut points and `k + 1` levels on one feature |
| `LinearFeedback` | `u = clip(a + b . features)` |
| `Tree` | thresholds and leaf actions of a complete tree of depth 1 to 3 |

The same parameter vector is available directly (`AdaptivePolicyFamily::action`) and
lowered to a value program (`AdaptivePolicyFamily::to_value_expr`) that a temporal
rollout executes as an assigned intervention. The two agree exactly. A rule may read
only variables the response source declares observed when the policy acts; reading
anything else is refused with `adaptive_policy_unobserved_feature`.

## Evaluate through the response model

`ModelPolicySource` rolls a fitted model out under the lowered policy, with one model
per parameter or posterior draw. Every policy meets the same parameter draws and the
same exogenous noise (common random numbers), so policy values and their paired
contrasts share draws. The optimisation and held-out splits use independent noise.
Any other simulator implements `PolicyRolloutSource`; in Python the response source is
an object with `rollout(family_id, theta, split)` (see
`antecedent.adaptive_policy`).

From Python, `adaptive_policy.search_model(fitted, families, ModelSource(...))` rolls a
fitted graphical causal model out directly. The policy decides every
`decision_interval` steps; at the steps in between the action holds its last realised
value, so the action variable is always assigned explicitly. Parameter uncertainty of the
fit is not propagated: the shared draws are the model's own noise.

## Constraints

Constraints are checked on rollout paths, never as an objective penalty:

- cumulative cost and cumulative exposure, each with a tolerated violation
  probability (zero is a hard constraint);
- per-step actions, checked on the realised actions of every draw.

Each constraint reports its violations, the observed violation rate, a Wilson 95%
upper bound on the violation probability (an approximation, not a guarantee) and the
largest observed value. A hard constraint is satisfied at the observed rate only; a
finite sample never certifies a zero violation probability. If no policy satisfies the
constraints the result carries no selected family and says so.

## Search and its record

The search is derivative-free: a seeded elite-refit evolution strategy over the
parameter box, followed by a coordinate pattern polish. It has a fixed evaluation
budget, retains every proposal (failed source evaluations stay visible), is admitted
against the memory limit and honours cancellation. Exhausting the evaluation budget
before convergence returns an incomplete result, never a silent best-so-far. A
`record_identity` digests the problem and every proposal and value; re-running with the
same seed and source reproduces it exactly.

## What the result means

The result is a **fitted-model optimisation**: the best policy found inside the fitted
response model, conditional on its assumptions. It is labelled an **identified policy
effect** only when you request that label and the response source states an
identification basis for the policy; otherwise the request is refused with
`adaptive_policy_identification_unsupported`. The label does not strengthen because the
fit is good.

The selected policy is chosen on the optimisation draws, so its in-sample value is
optimistic. The held-out value is computed on independent draws or held-out units that
selection never touched, and the difference is reported as `selection_optimism`. When the
study design supplies neither, `held_out="required"` refuses
(`adaptive_policy_heldout_unsupported`) and `held_out="preferred"` flags the result.

## Independent reference

`conformance/design/adaptive_policy_search/reference.py` derives two closed forms and
checks them against its own Monte Carlo: the treat-above-a-threshold optimum for a
stationary Gaussian state (the best fixed schedule treats never), and the linear-quadratic
Riccati feedback optimum against the best open-loop schedule.
