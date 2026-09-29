# Doubly robust binary policy evaluation

`policy.evaluate_policy_doubly_robust` evaluates fixed binary recommendations
with randomized assignment propensities and supplied `mu0`/`mu1` outcome
predictions. For each row it combines the predicted potential outcome under
the recommendation with the inverse-propensity weighted residual for the
observed randomized action. It reports policy, reference and incremental
values, net costs, and row-score standard errors for all three.

The caller must provide unique evaluation subject IDs and one of two ownership
contracts: training subject IDs disjoint from evaluation subjects, or fold IDs
matching the fold excluded when each row's predictions/recommendation were
produced. These checks validate only the supplied metadata; they do not inspect
model training records. Randomization propensities must be strictly between
zero and one. The standard errors assume independent evaluation subjects and
condition on the supplied nuisance predictions and policy. No clustered
standard error or calibrated interval is claimed.

This API evaluates a precomputed policy and does not train it. Identification
requires the declared randomized assignment, consistency and no interference.
It does not create a support-matrix license.
