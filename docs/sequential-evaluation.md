# Evaluate a policy that changes with what you observe

Use this route to compute the value of a contingent decision policy (act now, wait, collect an observation and choose differently depending on what it shows) over a finite horizon, on a model you declare. You supply the policy; Antecedent computes its value, compares named policies and prices the value of collecting more evidence. It is not a planner, a workflow engine or a policy learner, and it never searches for a better policy.

You need a finite model: discrete latent states with a prior, actions with their transition laws and utilities, and a [measurement model](measurement-models.md) for each observation channel. The value is conditional on that model and on three premises the data cannot test: exchangeability, consistency and positivity ([below](#routes)). It is a point value for the model's own population and period, with uncalibrated diagnostics.

## Evaluate a policy

A unit is `sound` or `faulty` with probability one half each. You may `inspect` it at cost 0.25 (an alarm that fires with probability 0.75 on a faulty unit and 0.25 on a sound one), `ship` it (+10 if sound, -10 if faulty) or `repair` it (+6 if sound, -4 if faulty). The policy inspects once, repairs after an alarm and ships otherwise.

```python
from antecedent import measurement_models as mm
from antecedent import sequential_evaluation as se

alarm = mm.MeasurementModel(
    indicator=mm.MeasurementVariable("alarm", "noisy_misclassified", "category"),
    construct=mm.MeasurementVariable("unit_condition", "latent_scientific", "category"),
    parameters=mm.BinaryParameters(
        rates=(mm.BinaryRates(sensitivity=0.75, specificity=0.75),), min_youden=0.1
    ),
    population_id="plant_a",
    regime_id="operations",
    evidence=mm.ParameterEvidence.declared("reliability_engineering"),
)
IDENTITY = ((1.0, 0.0), (0.0, 1.0))
model = se.FiniteModel(
    population="plant_a", regime="operations", period="2026",
    state_variable="unit_condition", states=("sound", "faulty"), initial=(0.5, 0.5),
    horizon=2,
    actions=(
        se.Action("inspect", "continue", transition=IDENTITY, utility_cost=0.25, channel=alarm),
        se.Action("ship", "commit", transition=IDENTITY, immediate_utility=(10.0, -10.0)),
        se.Action("repair", "commit", transition=IDENTITY, immediate_utility=(6.0, -4.0)),
    ),
    terminal_utility=(0.0, 0.0),
)
INSPECT, SHIP, REPAIR = 0, 1, 2
NO_ALARM, ALARM = 0, 1
policy = se.Policy.deterministic("inspect_then_act_v1", {
    (): INSPECT,
    ((INSPECT, ALARM),): REPAIR,
    ((INSPECT, NO_ALARM),): SHIP,
})
evaluation = se.evaluate(model, policy)
print(evaluation.value)           # 1.5
print(evaluation.assumptions)     # the exchangeability, consistency and positivity premises
```

A history is a tuple of `(action, observation)` pairs, actions by index (`Policy.deterministic` also accepts action labels when you pass `model=`). The value is `-0.25 + 0.5 * (-1.5) + 0.5 * 5 = 1.5`: the alarm fires with probability 0.5, after which `P(faulty) = 0.75` and repair is worth `-1.5`; with no alarm `P(faulty) = 0.25` and shipping is worth `5`. `evaluation.histories` and `evaluation.decisions` hold the receipts that add up to this value, and `evaluation.inference` names the claim (`point_only`). No interval is attached, because the value is exact given the declared model.

## Boundary

| Item | Supported | Refused |
| --- | --- | --- |
| Model | discrete latent states, discrete actions, a finite horizon (see [Bounds](#bounds)), per-action transition and observation laws | continuous states, unbounded horizons, laws that are not interventional given the information set |
| Policy | a fixed table from observed histories to action distributions, with a version label | a policy that conditions on the latent state, changes during the evaluation, or is learned from the data under evaluation (`policy_change_refused`) |
| Static scores | none | a reweighted static score table has no information sets or histories (`static_score_table`) |
| Observations | typed measurement models: binary misclassification or categorical confusion, non-differential, bound to the model's population and regime | a measured column named as the state, another construct, population or regime, differential error, a route with no finite likelihood |
| Evidence | each action and observation recorded once in the evidence ledger | a repeat, a partly overlapping log, a repeated trajectory identity (`evidence_double_counted`) |
| Positivity | an action in a reachable history needs declared support, and a logged-data route needs logging-policy probability above zero | `history_support_violated`; no extrapolation across an unsupported history |
| Transport | none: values are for the model's own population and period | another population or period: a typed refusal naming the missing selection-diagram and invariance contract |
| Uncertainty | a point value; a Wald interval for the off-policy estimate; uncalibrated diagnostics | an interval of an exact or rollout value (`uncertainty_representation_insufficient`), a posterior or a coverage statement |

## The model

A `FiniteModel` (Rust: `FiniteSequentialModel`) declares the latent states with their prior, the
horizon, the actions, the terminal utility and the declared support. Each
action is either `continue` or `commit`:

- a **continue** action costs utility and resource, moves the state through
  its transition, and produces an observation through its channel;
- a **commit** action ends the episode. Its immediate utility is realized
  at the decision time, its transition governs the state until its delayed
  outcome falls due, and no further observation is made.

An outcome may be **delayed** by a number of decision times; it is credited
when it falls due, and a commit evolves everything still outstanding under its
own transition. An outcome due after the horizon is refused or forfeited and
reported, as the model's censoring rule says. A **resource budget** bounds the
episode, and a **stopping boundary** on the posterior mass of a set of states
forces a commit action before the policy is consulted. The boundary is part of
the problem and applies to every policy equally.

## Belief updates and the once-only record

An observation's likelihood is the confusion law of the measurement model
bound to the action. Its construct must be the model's state variable, its
population and regime the model's, and its error non-differential. The
parameters are used as declared; their estimation uncertainty is not
propagated, and the receipt says so.

`BeliefSession` (Rust: `BeliefTracker`) follows one realized history: it applies the Bayes update and
records each action and each observation once in an evidence ledger, with
identities scoped to the session. The same record again is refused and the
belief and the ledger are left as they were. Evaluating a fixed policy needs
no observation, so it records nothing; collecting a new observation is a
separate act.

## Routes

**Sequential g-formula** (`evaluate`; Rust `evaluate_policy`). Exact enumeration of the
observed histories reachable under the policy. The value is

```text
V(pi) = sum over histories h of  P(h | pi) * realized(h)
```

where `pi` is the policy, `P(h | pi)` the probability of reaching the terminal history `h`
under `pi` and the declared model, and `realized(h)` the utility realized along `h` (costs,
immediate and delayed outcomes, terminal utility). Beliefs come from Bayes' rule on the declared model. It is conditional on
three declared premises that the data cannot test:

- exchangeability: the declared per-action laws are the interventional laws
  given the information set;
- consistency: the realized outcome of an action is its declared outcome;
- positivity: every action taken with positive probability in a reachable
  history is supported there.

Receipts record, for every reachable history, its reach probability, belief,
realized utility and continuation value, and for every decision the action
distribution, the boundary that forced it and the resource used. Reach times
realized utility sums to the value, and the enumeration refuses itself when it
does not.

**Off-policy from logged data** (`evaluate_off_policy`, the same name in Rust). Sequential importance
weighting of logged finite trajectories with their logged assignment
probabilities. The weight of a trajectory is the product over policy steps of
`pi(a | h) / mu(a | h)`, where `mu` is the logging (behaviour) policy; steps forced by a boundary weigh one. The ordinary
mean of `w G`, with `G` the trajectory's realized return, is unbiased for `V(pi)` when `mu > 0` wherever `pi > 0`, under
the same premises. Positivity is checked on every history the target policy
can reach in the model, not only on logged ones, and a logged probability must
equal the declared logging policy's. The standard error treats the weighted
returns as independent and is an uncalibrated diagnostic; the self-normalized
value and the effective sample size are reported with it. Re-evaluating
another policy on the same recorded log re-uses its records; a log that
overlaps what the ledger holds without being the same log is refused.

**Rollouts on common random streams** (`rollout`; Rust `rollout_policy`). Every uniform a
rollout consumes is keyed by its semantic stream identity
`(seed, rollout, time, role)`, so two policies see the same randomness for the
same unit at the same time and a policy differing only at histories it never
reaches is bit-identical. The estimate is a diagnostic of the enumeration with
an uncalibrated Monte Carlo standard error.

## Comparisons and the value of collecting evidence

`compare` (Rust `compare_policies`) evaluates named policies and checks each implements its
declared kind: act now (commit first), wait (continue without an observation),
collect evidence (continue through a channel) or choose differently after an
observation (different decisions after different observations). The report
orders the supplied comparisons; it makes no claim of optimality over
policies.

The value of *collecting* an observation (`value_of_collecting_evidence`) is not computed a second time. It is
the expected value of sample information scored by the existing design ranker
(`DesignRanker` with the `ReduceDecisionRegret` objective) over the finite
channel, and checked against the enumerated value of the collect-then-act
policy:

```text
V(collect, then Bayes action) = max_a E[U_a] + EVSI - cost
```

The cross-check covers one collection followed by a commit with no stopping
boundary; a disagreement is refused rather than reported.

## Artifact

`sequential_evaluation_v1` carries the model with its bound measurement
models, the policy version and decisions, the exact enumeration with its
history and decision receipts, and optionally the off-policy estimate with its
logged trajectories and logging policy, and the rollout diagnostic with its
seed. Every real is an IEEE-754 bit pattern. A consumer rebuilds the model,
policy, logged data and rollout configuration, re-runs the enumeration, the
weighting (recording the log once in a fresh ledger) and the rollouts, and
refuses unless every stored number, receipt and identity equals the
recomputation bit for bit. A consumer that retained an identity independently
also refuses a consistently resealed artifact of another evaluation.

## Decision notes

- **Transported policy evaluation.** A value is for the model's own
  population unless it is carried by the bounded transport route
  (`antecedent_design::transported_policy`, `docs/guides/transported-decisions-and-measurement.md`).
  That route evaluates a fixed policy in a target population under a declared
  selection diagram: transition and observation mechanisms must be invariant,
  the target initial-state law must lie inside the source's support, and the
  utility is either invariant or replaced by the target's own declared utility
  with its provenance (source label and content digest), which is bound into the
  evaluation identity. A request with no selection diagram is refused with that
  contract named (`transported_policy_contract_missing`); an undeclared
  mechanism, a shifted transition or observation mechanism or a target utility
  that does not match the model is refused by name.
- **Temporal transport.** Refused with the same typed refusal
  (`temporal_transport_contract_missing`). The model's laws are
  time-homogeneous within its horizon by construction; carrying them to
  another period is an invariance claim across periods that is not declared
  or checked here.
- **Uncertainty.** The exact value is a point. The off-policy estimate carries a 95% Wald interval from the importance-weighted standard error (`mean(wG) +- z * sqrt(var(wG) / n)`), reported with its calibration status (`interval.calibration` in Python); it needs two trajectories. The rollout standard error is an uncalibrated diagnostic, and an interval of an exact or rollout value is refused (`uncertainty_representation_insufficient`).

## Facade

`antecedent::analysis::sequential_evaluation` re-exports the contract and adds
`evaluate`, `seal`, `export` and `consume`. A refusal is a
`SequentialEvaluationError` with its registered `code`, the
`sequential_evaluation.*` `detail`, a message and a remedy;
`to_refusal` gives the shared structured shape.

## Python surface

`antecedent.sequential_evaluation` builds the declarations as frozen dataclasses
(`FiniteModel`, `Action`, `Delayed`, `StoppingBoundary`, `SupportOverride`,
`Policy`, `LoggedTrajectory`) and calls the native layer. `evaluate` returns a
`PolicyEvaluation` with its `HistoryReceipt` and `DecisionReceipt` tuples,
`evaluate_off_policy` an `OffPolicyEstimate`, `rollout` a `RolloutEstimate`
(`stream_uniform` exposes one stream identity), `compare` a `ComparisonReport`
and `value_of_collecting_evidence` a `ValueOfEvidence`. `BeliefSession`
follows one realized history, `seal` and `consume_artifact` export and
re-evaluate the artifact, and `ReweightedStaticScores` and
`LearnedFromEvaluationData` are the declarations that are refused by name.
Every refusal is a `SequentialEvaluationRefusal` with its registered
`reason_code` and the `sequential_evaluation.*` `detail`.

## Bounds

At most 4096 latent states, 1024 actions, 1024 decision times, 65536 policy entries,
150000 logged trajectories and 10000000 rollouts. The cost of one backward pass,
`states * states * actions * horizon`, is bounded by 2^34, and an exact enumeration
visits at most 2000000 history nodes by default (`max_nodes`). A declaration beyond a
bound is refused (`invalid_argument`, detail `sequential_evaluation.limit_exceeded`),
never truncated.

## Refusals

Every refusal carries a registered `reason_code` and a `sequential_evaluation.*` `detail`.
See [refusal codes](reference/refusal-codes.md#structured-refusals-and-their-remedies) for
remedies.

| `reason_code` | `detail` | When |
| --- | --- | --- |
| `route_not_supported` | `static_score_table`, `policy_change_refused` | a `ReweightedStaticScores` table or a `LearnedFromEvaluationData` policy |
| `route_not_supported` | `channel_route`, `channel_rates` | a channel that is not binary misclassification or categorical confusion, or a binary channel without one shared pair of rates on a two-state model |
| `differential_error_not_pooled` | `differential_channel` | a channel whose error is declared differential |
| `route_not_supported` | `voi_scope` | a value-of-evidence request outside one collection followed by a commit with no stopping boundary |
| [`history_support_violated`](reference/refusal-codes.md#other-registered-refusal-codes) | `action_not_supported`, `behaviour_support_missing`, `no_overlap` | a positivity failure in the model or the logging policy |
| [`evidence_double_counted`](reference/refusal-codes.md#evidence-ledger) | `observation_recorded_twice`, `trajectory_recorded_twice`, `record_already_counted`, `log_partially_counted` | a record entered twice, or a log that partly overlaps the ledger |
| [`measurement_binding_mismatch`](reference/refusal-codes.md#measurement-models) | `channel_construct_mismatch`, `channel_binding_mismatch`, `channel_levels_mismatch` | a channel bound to another construct, population or regime, or with the wrong number of levels |
| `transport_missing_evidence` | `transported_policy_contract_missing`, `temporal_transport_contract_missing` | a value requested for another population or period (`check_request`) |
| `uncertainty_representation_insufficient` | `interval_not_supported` | an interval of an exact or rollout value (`request_interval`) |
| `invalid_argument` | for example `policy_incomplete`, `limit_exceeded` | a malformed declaration, a reachable history without a decision, a bound exceeded |

## Related

- [Measurement models](measurement-models.md) to declare an observation channel.

- [Transported policy evaluation](guides/transported-decisions-and-measurement.md) to carry a fixed policy to a target population.
- [Value of information and design ranking](design-ranking.md) for the evidence-collection score.
- [Adaptive and group-sequential inference](adaptive-inference.md) for analysing data from adaptive experiments.
