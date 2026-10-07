# Population, time and uncertainty routes (2.3, A section)

This page summarizes the 2.3 A-section routes that extend graph classes,
transport models, temporal sequences, counterfactuals and effect-constancy
testing. For each it states the question it answers, the exact scope bounds, the
claim label, the Python entry point and the namespace of its refusal details.

The authoritative status of each route (`licensed` or `closed`, with its reason
code) is its record in `parity/promotion_2_3.toml` and its row in
`parity/transport_stages.toml`; this page does not override them. No route here
returns a confidence interval, credible interval or coverage statement. A route
whose claim would be a calibrated interval is closed, and its calibration is
unmeasured.

## Claim labels

- **Structural envelope.** An unweighted range over a finite, enumerated set of
  scenarios. It is not a probability statement and carries no sampling
  uncertainty.
- **Point only.** A value (or a matrix of values) with no interval and no
  coverage claim. Where a test is reported, its Type I error and power are
  unmeasured.
- **Closed pending calibration.** The Rust core exists and replays through an
  independent artifact, but the public producer validates its request and then
  refuses with `cell_not_licensed` and a `route_frozen` detail. Nothing is
  returned.

Every refusal is a pair of a registered reason code and a detail of the form
`<namespace>.<snake_case>`; callers switch on the pair.

## A1: DAG completions of a CPDAG and shared-row covariance

**Completion scenarios.** Does a target effect remain identified, and what range
does it span, across every DAG completion of a supplied CPDAG when each
completion receives evidence bound to its own assumptions?

- Scope: a maximally oriented CPDAG over at most six fully observed nodes (no
  selection or latent structure) and supplied exact laws (counted laws refuse).
  Enumeration and per-completion decisions share one search budget; a stopped
  search keeps the completions found, marks undecided ones `unevaluated` and
  counts never-enumerated ones.
- Output: every completion is kept whatever its status, with identified,
  unidentified, unevaluated and not-enumerated counts kept apart and never
  renormalized, plus an unweighted envelope over the identified completions.
  No graph probability is assigned and no interval across completions exists.
- Claim: structural envelope.
- Entry points: `antecedent.transport.advanced.cpdag_completion_scenarios` and
  `consume_cpdag_scenarios_artifact`.
- Refusals: `cpdag_scenarios.*`, for example `selection_or_latent`,
  `bounds_exceeded`, `not_a_cpdag` and `exact_laws_only`.

**Shared-row covariance.** How correlated are point estimates from identified
completions that reuse the same observed units?

- Scope: scenarios on one complete-row snapshot with one shared replicate id and
  row selection across scenarios. `shared_row_bootstrap` (seeded whole-row
  resampling; a replicate where any estimator fails is dropped for all
  scenarios and counted) or `exact_enumeration` (tiny tables only: at most 24
  rows and a bounded number of compositions). Scenarios on different snapshots
  or unit lists, declared independent, or with repeated unit ids refuse; a
  diagonal-only matrix is never a fallback.
- Claim: point only. The matrix is not an interval.
- Entry points: `antecedent.transport.advanced.scenario_shared_covariance` and
  `consume_scenario_covariance_artifact`.
- Refusals: `scenario_covariance.*`, for example `unknown_dependence`.

## A2: joint Bayesian transport (closed)

What target-effect posterior follows from a declared joint source and target
model and the supplied transport evidence?

- Scope of the pilot: the fixed DAG of a separately licensed 2.2 transport
  derivation. An ADMG or graph-posterior query refuses; draws are bounded to
  1..=100000 and the model to 256 parameters; undeclared or overlapping source
  dependence and a missing target law refuse with their own typed details.
- Claim: closed pending calibration. A valid request raises
  `cell_not_licensed` with `bayesian_transport.route_frozen`; calibration is
  unmeasured and no posterior is published.
- Entry point: `antecedent.transport.advanced.joint_bayesian_transport`, which
  never returns.
- Refusals: `bayesian_transport.*` (`unsupported_graph`, `route_frozen`).

## A3: binary nested-Markov pilot (closed)

Can one bounded binary ADMG transport functional be evaluated through a
correctly normalized nested-Markov likelihood?

- Scope: exactly one four-variable binary graph (`X1 -> X2 -> X3 -> X4` with
  `X2 <-> X4`) and the observational regime. Any other ADMG, regime, non-binary
  domain or more than six observed variables refuses, and this is never a
  nonidentification claim.
- Claim: closed pending calibration; no interval or posterior is published.
- Entry point: `antecedent.transport.advanced.binary_nested_markov`, which never
  returns.
- Refusals: `nested_markov.*` (`outside_binary_pilot`, `route_frozen`).

## A4: temporal extensions and effect constancy

**Uncertain initial state.** How does uncertainty in the pre-action state change
the two-step target response?

- Scope: the 2.2 fixed two-slice graph with a finite observed pre-action state
  and a two-step sequence. The value is the target two-step mean marginalized
  over a finite target initial-state law. A source law or a single state does
  not answer it (a fixed-state response is reported under its own label), and a
  state with positive target mass and no history support refuses.
- Claim: point only.
- Entry points: `antecedent.transport.advanced.temporal_initial_state` and
  `consume_temporal_initial_state_artifact` (registry route
  `antecedent.temporal.initial_state`).
- Refusals: `initial_state.*` (`target_law_missing`, `support_gap`).

**New-period refresh.** Can a held result accept a new observation period
without reusing stale time or selection premises?

- Scope: the horizon stays two, and the graph, lag alignment, intervention
  history, selection targets, regimes and proof are unchanged while both the
  period and the snapshot are new. The value is re-evaluated on the replacement
  panel, never copied. Any interval reported for the old window is invalidated.
- Claim: point only.
- Entry points: `antecedent.transport.advanced.temporal_new_period_refresh` and
  `consume_temporal_refresh_artifact` (registry route
  `antecedent.temporal.new_period_refresh`).
- Refusals: `temporal_refresh.*` (`horizon_changed`, `lag_alignment_changed`,
  `intervention_history_changed`, `premises_changed`, `stale_snapshot`).

**Dependent-sampling interval (closed).** What sampling interval surrounds a
two-step sequence effect when observations share a repeated unit?

- Scope: arguments are validated (unit map, minimum units, replicate bound,
  supported history) and a well-formed request then refuses.
- Claim: closed pending calibration; a point estimate is never relabeled as an
  interval.
- Entry point: `antecedent.transport.advanced.temporal_dependent_interval`,
  which never returns.
- Refusals: `temporal_interval.*` (`unknown_units`, `too_few_units`,
  `too_many_replicates`, `unsupported_history`, `route_frozen`).

**Effect constancy (F18).** Can a test that one effect is the same across a
declared partition (time periods or regions) be consumed under a checked
contract?

- Null: the effect is equal across all declared partitions. It is not
  configurable.
- Statistic: for partitions with independent evidence, Cochran's Q. When the
  caller supplies the full covariance of the partition estimates, a Wald
  chi-square of the contrasts against the first partition in label order (equal
  to Q for a diagonal covariance). Tails use the chi-square distribution.
  Dependence is declared, never assumed from silence, and an unusable
  covariance refuses rather than falling back to independence.
- Contrasts: two-sided normal p-values over a declared family (all pairs, or
  every partition against a named reference) with Holm adjustment at a fixed,
  caller-supplied level. Partitions are put in label order before any
  arithmetic, so input order does not change the result.
- Scope: at most 1024 partitions that share one effect, unit, regime and
  population; a partition with unsupported coordinate support refuses.
- Frozen acceptance: equal effects give difference 0 and statistic 0; effects 1
  and 2 with variance 1/4 each give statistic 2.
- Claim: point only. Non-rejection does not prove constancy, the test's Type I
  error and power are unmeasured (calibration coordinate `unmeasured`), and no
  power claim is made.
- Artifact: `effect_constancy_v1` keeps partition identities, covariance,
  multiplicity family, the null and the calibration coordinate; a consumer
  recomputes the result and requires every stored value to match. A changed
  identity is refused only against an `expected` identity that the consumer
  retained itself; without one, a consistently resealed artifact is checked for
  internal consistency only.
- Entry points: `antecedent.temporal.effect_constancy` and
  `consume_effect_constancy_artifact`; the conclusion is
  `ConstancyConclusion.NOT_REJECTED` or `REJECTED`.
- Refusals: `effect_constancy.*` with `invalid_argument` (`invalid_alpha`,
  `too_few_partitions`, `too_many_partitions`, `non_finite_estimate`,
  `invalid_standard_error`, `invalid_covariance`, `unknown_reference`,
  `estimand_missing`, `invalid_request`, `limits_exceeded`, `invalid_identity`)
  or `route_not_supported` (`incompatible_partitions`, `unsupported_partition`,
  `wrong_contract`). The frozen text named `temporal_effect_constancy.*`
  details; the emitted namespace is `effect_constancy`.

## A5: temporal counterfactual and the closed transported composition

**Fixed-population temporal counterfactual.** What would each observed unit's
final outcome have been under a different two-step action history?

- Scope: one fixed, fully observed, Markovian two-slice DAG
  `covariate_0 -> action_0 -> covariate_1 -> action_1 -> outcome` with
  additive-noise linear mechanisms and no latent confounding. Each unit's
  exogenous history is recovered exactly from its factual trajectory once, and
  both named action histories are replayed against that same history. Fresh
  noise per world would answer an interventional question and is never drawn.
  A history that refutes the fitted mechanism refuses.
- Output: per-unit final outcomes in both worlds and their sample mean contrast,
  with a shared-abduction receipt that an independent consumer replays.
- Claim: point only.
- Entry points: `antecedent.temporal_counterfactual.temporal_fixed_population` and
  `antecedent.temporal_counterfactual.consume_temporal_counterfactual_artifact`
  (registry route `antecedent.cross_world.temporal_fixed_population`).
- Refusals: `temporal_counterfactual.*`, with `route_not_supported`
  (`unpaired_histories`, `time_misaligned`, `shared_history_missing`,
  `latent_confounding`, `refuting_history`) and `invalid_argument`
  (`horizon_exceeded`, `too_many_units`, `invalid_graph`, `fit_mismatch`,
  `history_name_invalid`, `non_finite_history`).

**Transported path-specific counterfactual (closed).** Can a static
path-specific counterfactual answer be transported to a named target?

- Scope: transporting a counterfactual needs both an identified target transport
  functional and a fixed-population cross-world theorem for the same quantity.
  Neither a transported mean nor a source counterfactual alone licenses the
  composition, and the joint theorem has not passed. Nothing is evaluated
  whatever is supplied; the refusal names the missing prerequisite gates and
  absent regime factors.
- Claim: closed pending the joint theorem.
- Entry point: `antecedent.temporal_counterfactual.transported_path_specific`,
  which never returns (registry route
  `antecedent.cross_world.transported_path_specific`).
- Refusals: `transported_counterfactual.*` (`route_frozen` as
  `cell_not_licensed`; `factor_missing` as `transport_missing_evidence`).

## A6: sampled observation recovery (closed)

What sampling uncertainty surrounds an effect computed from the exact binary
observation-recovery formula?

- Scope: the one binary m-graph and exactly recoverable effect query of the 2.2
  observation-recovery route, at most six binary variables. An m-graph with a
  verified nonrecoverability witness, a zero complete-case cell (a zero
  denominator of the recovery formula) or a malformed row refuses.
- Claim: closed pending calibration; the percentile interval's whole-method
  calibration is unmeasured and nothing is returned.
- Entry point: `antecedent.transport.advanced.sampled_observation_recovery`,
  which never returns.
- Refusals: `sampled_recovery.*` (`unrecoverable_pattern`, `bounds_exceeded`,
  `route_frozen`).

The CPDAG search charges completion storage as well as orientation attempts. A
stopped enumeration reports an upper bound on unseen completions, labeled
`cpdag_completions_not_enumerated_upper_bound` in its receipt; Python's
`counts.exact` is false. It never restarts search to discover an exact remainder
after cancellation or budget exhaustion. Directed edges that are reversible
within the equivalence class are rejected, as are undirected edges whose
orientation is compelled.

F18 also executes a known-Gaussian reference experiment (2,000 replicates per
cell; n=20,80,320; 2,3,8 partitions; independent and positively/negatively
correlated errors), checking null Type I error, power and both Holm families.
This measures the test composition under known covariance. It does not certify
an arbitrary supplied effect estimator or estimated covariance; those results
retain their `unmeasured` calibration label.
