# Observation-conditioned scientific queries

Conditioning on an observation and intervening are different claims, also when
both are computed from the same causal model. `P(Y | S = s)` reads the
observational law and keeps every association between `S` and `Y`, including
the one a common cause creates. `P(Y | do(S = s))` cuts the arrows into `S`.
This page describes the bounded queries that keep the two apart in the claim's
identity and in its reported value, how a measured signal differs from the
construct it measures, and what is closed.

All values here are exact given a declared model, or a labelled point value of
a fitted one. Nothing on this page is an interval or a calibration claim.

## Boundary

| Item | Supported | Refused |
| --- | --- | --- |
| Model | a composed finite model (declared or fitted conditional tables, enumerated within the finite-state bound), a declared finite hypothesis space, a declared Beta-Bernoulli model | a linear-Gaussian composition (`route_not_supported`), continuous or sampled conditioning |
| Conditioning | `P(outcome | recorded values)` with the probability of the evidence; paired with `P(outcome | do(assignment))` on the same model | an assigned or hypothetical value in a conditioning slot, an intervention supplied to a conditioning query, observations mixed with an intervention |
| Measured signal | the recorded value of one indicator of a construct, through a declared non-differential binary-misclassification or categorical-confusion measurement model in its own population, regime, arm and representation version | the recorded value used as the construct; error that differs by arm, population or time; a route with no confusion matrix; more than one measured signal |
| Threshold crossing | which observed levels of one signal move `P(outcome = level)` to or past a level, and the probability of observing one | a level no observed level reaches (refused, with the most extreme value attained) |
| Posterior update | exact Bayes' rule on a declared finite space (exchangeable observations); Beta-Bernoulli conjugate update with predictive success probability | a prior or likelihood that is not a distribution; observations of probability zero |
| Evidence required | the exact Bayes factor; the fewest identical conditionally independent readings of a binary signal; the fewest consecutive successes or failures for a Beta-Bernoulli predictive probability | a level the signal cannot reach, or that needs more than the declared bound (refused, with the supremum or infimum reachable) |
| Uncertainty | none | an interval, a credible interval, a calibration statement |

## Observing is not intervening

On a composed finite model both are computed from one executable joint law:

- `P(y | observed)` is the ratio of two finite sums of the model's observational
  law: `P(y, s) / P(s)`.
- `P(y | do(s))` is the model's truncated factorization.

The pair is reported side by side with its largest absolute difference. In the
confounded fixture of the test suite (`Z -> S`, `Z -> Y`, `S -> Y`, all binary,
`P(Z = 1) = 0.5`, `P(S = 1 | Z) = 0.2, 0.8`, `P(Y = 1 | S, Z) = 0.1, 0.5, 0.3,
0.9` keyed `(S, Z)`):

| Quantity | Value |
| --- | --- |
| `P(S = 1)` | 0.5 |
| `P(Y = 1 | S = 1)` | 0.78 |
| `P(Y = 1 | do(S = 1))` | 0.6 |
| `P(Y = 1 | S = 0)` | 0.18 |
| `P(Y = 1 | do(S = 0))` | 0.3 |

With the arrow `Z -> S` removed the two laws agree, so a disagreement is a
property of the model and not of the code.

Each query names its operation, and the operation is part of the claim's
identity. Conditioning, intervention and the paired comparison of the same
variables and levels on the same model are three different identities with
three differently shaped results; a conditioning claim has no interventional
value and an interventional claim has no conditional one.

A value placed in a conditioning slot must have been recorded. An assigned or
hypothetical value is refused as an intervention smuggled into a conditioning
slot (`observation_query.intervention_in_conditioning_slot`), as is an
intervention handed to a conditioning query. The same value stated as an
intervention is answered. The distribution that conditions on observations
after an intervention, `P(y | do(s), z)`, is a different query with no bounded
route; it is refused (`observation_query.mixed_conditioning_and_intervention`).

## Measured versus latent

A recorded indicator value is not the construct it measures. Offering it as the
construct without a measurement model is refused
(`observation_query.measured_as_construct`, code `proxy_offered_as_construct`).
With a declared model the construct's posterior is the Bayes inversion of the
declared likelihood against the composed model's own law:

```
P(c | m, o)  proportional to  P(m | c) P(c, o)
P(y | m, o)  =  sum_c P(y | c, o) P(c | m, o)
```

The premise is that the indicator is conditionally independent of the outcome
and of every other observed variable given the construct (non-differential
error). Hand-checkable fixture: prevalence 0.1, sensitivity 0.9, specificity
0.8, `P(Y = 1 | C) = 0.1, 0.7`.

| Quantity | Value |
| --- | --- |
| `P(M = 1)` | 0.27 |
| `P(C = 1 | M = 1)` | 1/3 |
| `P(Y = 1 | M = 1)` | 0.3 |
| `P(C = 1 | M = 0)` | 1/73 |
| `P(Y = 1 | M = 0)` | 7.9/73 |
| `P(Y = 1 | C = 1)` (the measured value read as the construct) | 0.7 |

The model must name this indicator and this construct, and be established for
the composed model's population and regime and arm and the indicator's
representation version; each mismatch is its own refusal
(`observation_query.indicator_mismatch`, `construct_mismatch`,
`population_mismatch`, `regime_mismatch`, `arm_mismatch`); a changed
representation version needs an explicit mapping
(`representation_mapping_required`). Error declared to differ by arm,
population or time is never pooled with the non-differential route
(`observation_query.differential_error_unsupported`).

## Thresholds and evidence required

*Threshold crossing.* For each observed level of a signal the query reports the
exact conditional probability and the probability of observing that level, and
the total probability of a crossing observation. A threshold no level reaches is
refused as `evidence_cannot_reach_level`, with the requested level and the most
extreme conditional probability attained stated.

*Evidence required.* Posterior odds are prior odds times the Bayes factor, so
moving a probability from `p0` to `q` needs the factor
`[q / (1 - q)] / [p0 / (1 - p0)]`. With `n` conditionally independent readings
of one kind, each of Bayes factor `B`, the minimum `n` is the smallest integer
count reaching the level; for a Beta(a, b) predictive probability, `n`
successes give `(a + n) / (a + b + n)` and `n` failures give
`a / (a + b + n)`. A level matched within `1e-12` in log-odds (probability, for
the conjugate solve) counts as reached; one fewer observation is verified not to
reach it.

| Request | Answer |
| --- | --- |
| prior 0.2 to level 0.8 | Bayes factor 16 |
| prior 0.01 to 0.5, rates 0.9 and 0.05 (factor 18) | 2 readings (one gives odds 18/99) |
| Beta(1,1) to predictive 0.9 | 8 successes |
| Beta(1,1) to predictive 0.99, bound 50 | refused; supremum reachable 51/52 |
| any prior to a level with an uninformative signal | refused; the prior is the supremum |

## What a value means

| Reading | When | Claim |
| --- | --- | --- |
| `exact_given_declared_model` | every mechanism of the composed model was written down by the analyst, or a declared Bayes model | exact statement about the declared model |
| `point_only_sample_fitted` | a mechanism was fitted to a sample or supplied by a provider, or the caller declares the prior and likelihood fitted | a point value; no claim about how often it is right |

The reading is derived from the mechanisms' origin and bound into the identity;
a resealed upgrade to the exact reading is refused by the consumer.

### Calibration decision

A sample-fitted probability whose mechanisms are not the count-fitted tables of
the route below, and every provider-supplied model, stays a labelled point
value. The count-fitted interval is measured as a whole-method design:

- *Method*: fit conditional tables from a sample, compose, condition (or update),
  report `P(Y | S = s)` or a posterior probability.
- *Data-generating process*: a known finite structural model with a common cause
  of the signal and the outcome so that `P(Y | S)` and `P(Y | do(S))` differ;
  a measured-signal variant with known sensitivity and specificity.
- *Nominal level*: coverage of a stated interval for the conditional probability
  at 0.95, and calibration of stated posterior probabilities.
- *Support range*: a grid of sample sizes including conditioning cells that are
  empty or sparse after conditioning.
- *Probes*: conditioning events of probability near zero; a measured signal near
  uninformative (Youden index near zero); a small observation bound.

A declared model's posterior probability is an exact number with no sampling
distribution, so a sampling interval for it is logically closed with
`cell_not_licensed` (`observation_query.declared_model_has_no_sampling_interval`,
`observation_query.calibrated_probability_closed`). The interval is earned only
where the number does have a sampling distribution: tables fitted to counts. The
count-fitted route (`antecedent.analysis.count_fitted_observation`) retains
per-parent-configuration counts and the row count `n`, places an independent
Dirichlet posterior over every conditional, and reports the equal-tailed
posterior quantiles of `P(Y | evidence)` computed by Monte Carlo over joint
table draws and exact enumeration. The interval is a measured scalar bound to a
currently attesting coverage record; a consumer replays the draws from the
retained counts, prior, seed and level. Measured: a confounded `Z -> S`,
`Z -> Y`, `S -> Y` model and the same model with a measured signal of known
sensitivity and specificity, levels 0.90 and 0.95, around 300 rows; sparse
parent configurations and rare conditioning events are named boundaries. A
declared node (a measured signal with known error rates) carries no sampling
uncertainty and is not redrawn.

## Layers

- `antecedent_model::observation_query`: exact numerics on a `FiniteLaw` /
  `ComposedFinite`, `FiniteBayesModel`, `BetaBernoulli` and the evidence solves.
  No claim vocabulary and no contract dependency.
- `antecedent_contract::observation_query`: slot semantics
  (`check_observation_slots`), the measured-signal likelihood
  (`measured_likelihood`, built on the measurement model contract), the
  reading (`InferenceBasis`) and the closed-route refusals.
- `antecedent_io_core::observation_query_artifact`: the `observation_query_v1`
  artifact.
- `antecedent::analysis::observation_query`, `observation_posterior` and
  `observation_evidence`: the facade. `PreparedMechanismComposition` gains
  `observe`, `condition_on`, `intervene_on`, `condition_versus_intervene` and
  `threshold_crossing`; declared-model queries are free functions.

## Python

`antecedent.observation_query` exposes the facade: `condition_on`, `intervene_on`,
`condition_versus_intervene` and `threshold_crossing` take a prepared
`mechanism_composition.PreparedMechanismComposition`; `finite_posterior`,
`conjugate_posterior`, `required_bayes_factor`, `required_observations_binary_signal` and
`required_observations_conjugate` take a declared model; `consume_observation_query_artifact`
replays an artifact against a retained identity. Slots are `Recorded`, `Assigned`,
`Hypothetical` and `Measured` (a recorded indicator value with its
`measurement_models.MeasurementModel`). Every claim is an `ObservationClaim` with its
`identity`, `operation`, `inference_basis` and sealed `artifact`; `interval()` and
`calibrated_probability()` raise `cell_not_licensed`. Every refusal is an
`ObservationQueryRefusal` with a registered `reason_code`, an `observation_query.*` `detail`,
`expected`/`supplied` (the requested level and the reachable extremum of an unreachable-level
refusal) and a `remedy`.

## Artifact

`observation_query_v1` keeps the composed premises (the mechanism-composition
premises wire), the query, the operation, the reading and the result as IEEE bit
patterns. A consumer recomputes the identity, recompiles the premises under the
coherence checker and replays the query to identical bits. A refused query (an
unreachable level, a measured value without an applicable model) cannot be
sealed.

Replay protects against corruption, edits that are not resealed and resealed
edits that change the answer. It does not protect against a producer that seals
wrong but coherent premises, or a bug shared by producer and consumer; the
independent oracles are in the test suites.

## Lifecycle

Refreshing the composition with refitted mechanisms
(`PreparedMechanismComposition::refresh_mechanisms`) returns a new composition.
A query asked of it is a new claim with a new identity; the earlier claim and
its artifact are unchanged and still replay against their own premises.
Observation-conditioned claims do not survive a change of mechanism numbers: a
consumer holding the old premises replays the old answer, and a claim for the
new composition must be asked again.

## Refusals

Details are under `observation_query.*`; the full table with codes and remedies
is in [refusal-and-partial-knowledge.md](refusal-and-partial-knowledge.md).
