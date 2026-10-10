# Population, time and uncertainty routes (2.3)

This page summarizes the 2.3 routes that extend graph classes,
transport models, temporal sequences, counterfactuals and effect-constancy
testing. For each it states the question it answers, the exact scope bounds, the
claim label, the Python entry point and the namespace of its refusal details.

The authoritative status of each route (`licensed` or `closed`, with its reason
code) is its record in `parity/promotion_2_3.toml` and its row in
`parity/transport_stages.toml`; this page does not override them. The statistical
adapters described below return a factory-only `inference.MeasuredInference`
only when their actual scalar bases resolve to current attesting records. Missing or stale evidence refuses. Underlying model artifacts retain their
original `unmeasured` standing outside the named measured scalar intervals.

## Claim labels

- **Structural envelope.** An unweighted range over a finite, enumerated set of
  scenarios. It is not a probability statement and carries no sampling
  uncertainty.
- **Point only.** A value (or a matrix of values) with no interval and no
  coverage claim. Where a test is reported, its Type I error and power are
  unmeasured.
- **Measured scalar inference.** The native producer and fresh consumer derive
  each reported scalar's full method-specific basis and require a current,
  passing record. Only the named scalar intervals receive that standing.
  Unknown sampling/model assumptions remain declarations, and covariance,
  posterior draws and simultaneous intervals receive no blanket license.
- **Closed or unlicensed.** A missing implementation, unsupported protocol or
  nonattesting record produces a typed refusal. An underlying model report and its
  source artifact never supply a measured interval by themselves.

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

## A2: joint Bayesian transport

What target-effect posterior follows from a declared joint source and target
model and the supplied original transport evidence?

- Measured adapter scope: original checked fixed-DAG direct/standardization
  proof; one covariate; zero-mean declared isotropic Gaussian priors of variance
  1000; known source variances 1 and (when a second source exists) 2.25;
  disjoint independent source samples of equal size 150..600; original covariates
  in [-1,1]; fixed four-row target covariates [-0.25,0.25,0.25,0.55]. The four
  measured configurations bind intercept/full varying blocks and
  independent/shared block declarations. Posterior draws are 4096..100000 and
  the nominal level is 0.95. Prior-bank variants cannot borrow this license.
- Output: `MeasuredInference.scalar("target_effect")`. The original joint
  posterior retains its full covariance and model-conditional diagnostic
  standing. Gaussian model correctness and IID sampling are declared assumptions.
- Entry point: `antecedent.transport.advanced.joint_bayesian_transport`.
  Fresh consumption uses `inference.MeasuredInference.load` with the retained
  expected identity; source proof, populations, data and model are replayed.
- Refusals include unsupported graph/transport proof, overlapping or unknown
  evidence dependence, out-of-protocol priors/designs and nonattesting records.

## A3: binary nested-Markov posterior and Fisher inference

The original graph is exactly `X1 -> X2 -> X3 -> X4`, with `X2 <-> X4`, four
binary variables and one observational sixteen-cell table of positive integer
counts. Both methods execute original checked general-ID before fitting;
nonparametric identification and likelihood-model adequacy remain distinct.
Original variable labels have a 256 UTF-8 byte limit. Default fit options are
50000 iterations, tolerance 1e-11 and no empirical-residual refusal threshold.

- `antecedent.transport.advanced.binary_nested_markov_fisher_interval` reports
  named `mean0`, `mean1` and `contrast` scalars at 0.95, using the original
  expected multinomial Fisher information and full delta covariance. Measured
  sample-size scope is 1000..4000; singular/boundary information refuses.
- `antecedent.transport.advanced.binary_nested_markov` uses the original full
  eleven-dimensional feasible Möbius model with all eleven raw-coordinate
  Beta(1,1) or all Beta(2,2) shapes. Its sampler is four chains, 2048 warmup,
  4096 retained draws per chain, a 5000000 proposal bound and 0.95 equal-tailed
  quantiles. All fourteen parameters/derived estimands must satisfy rank/folded
  Rhat <=1.01 and bulk/tail ESS >=400. Measured count scope is 2000..8000.
- Both return `inference.MeasuredInference` only if all three actual scalar
  bases have current records. Posterior summaries are Monte Carlo estimates;
  parameter-coordinate or simultaneous intervals are not licensed by these
  three scalar records. Correct model specification and IID sampling cannot
  be established from the observed table alone.
- The fresh measured consumer replays the original point fit plus the complete
  information or posterior receipt. Changing prior, fit/sampler settings,
  counts, identity or source-report standing refuses. An MLE or Fisher
  record cannot supply a Bayesian license. See the
  [Bayesian scientific design](nested-markov-bayesian-calibration.md).

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

**Dependent-sampling interval.** The measured direct-panel adapter uses a
whole-unit equal-tailed studentized bootstrap, B500 at level 0.95, on complete
balanced binary `(S0,A1,L2,A2)` histories. Each unit retains all sixteen history
cells; units and raw per-unit scores, standard errors and resample pivots travel
with the source. The fixed target initial law is (0.3,0.7) and sequence (0,0).
The measured direct unit-count span is 75..200; the construction has native
20..4096 unit bounds, minimum units 20 and failed fraction at most 0.05.

`antecedent.transport.advanced.temporal_dependent_interval` resolves the
`response` scalar's actual basis before returning `MeasuredInference`.
`TemporalSession.dependent_interval` uses the original checked source,
functional, active/control executions and paired unit scores; its measured
checked span is 64..256 units and its functional basis is separate from a
direct panel. Fresh consumption replays that checked source rather than
substituting a plain panel artifact. Arbitrary temporal callbacks or incomplete
histories receive no studentization certificate.

Response intervals use the studentized method. Checked paired effects have
separate studentized, percentile and basic records; one method does not supply
evidence for another. See [prepared recalculation](2_3-recalculation-capabilities.md).

**Effect constancy.** Can a test that one effect is the same across a
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
- Checked example: equal effects give difference 0 and statistic 0; effects 1
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

## A6: sampled observation recovery

The measured adapter reruns the original checked binary observation-recovery
formula and standardized recovered effect on complete whole-row bootstrap
samples. Its exact protocol is B2000, level 0.95, zero failed bootstrap draws,
normalization tolerance 0.1, small-cell count 5, treated level 1 and control 0.
The `bootstrap_bca` interval binds midrank bias correction, the exact grouped
all-row delete-one jackknife and adjusted quantile probabilities in a version-3
source artifact. Measured row-count scope is 1000..4000. Support loss, degenerate
jackknife variance, invalid bias/acceleration or unresolved adjusted tails refuse.

`antecedent.transport.advanced.sampled_observation_recovery` returns a
`MeasuredInference` for the named `recovered_effect` only when the actual
original derivation/method basis resolves to a current record. Independent
consumption rechecks the recovery proof and every bootstrap/jackknife receipt;
correct missingness premises and independent observations remain declarations.
See the [BCa scientific design](sampled-recovery-bca.md).

The CPDAG search charges completion storage as well as orientation attempts. A
stopped enumeration reports an upper bound on unseen completions, labeled
`cpdag_completions_not_enumerated_upper_bound` in its receipt; Python's
`counts.exact` is false. It never restarts search to discover an exact remainder
after cancellation or budget exhaustion. Directed edges that are reversible
within the equivalence class are rejected, as are undirected edges whose
orientation is compelled.

The known-Gaussian reference experiment (2,000 replicates per
cell; n=20,80,320; 2,3,8 partitions; independent and positively/negatively
correlated errors), checking null Type I error, power and both Holm families.
This measures the test composition under known covariance. It does not certify
an arbitrary supplied effect estimator or estimated covariance; those results
retain their `unmeasured` calibration label.

## Effect constancy in downstream reviews

`antecedent.effect_constancy_review` consumes the original portable effect-constancy artifact
against its independently retained identity before invoking downstream engines.
`transport_diagnostic` returns an original oriented, covariance-aware Holm contrast;
it always requires a separate transport identification check.

`rank_prior_sources` calls the original prior-bank compatibility filter and ranks
usable entries by explicitly declared partition-effect proximity. This preference
is data dependent. It does not synthesize a posterior from sampling standard errors,
pool observations or authorize transfer of a selected prior.

`policy_review` calls the original affine mean decision engine on every fully
supported original partition. It preserves exactly tied leaders and reports an
empty common-leader set when partitions disagree. Each decision retains its own
original partition coordinate and source identity, even when two point effects
are equal. Nonlinear utilities, unavailable distribution requirements and semantic
coordinate mismatches are refused.

These consumers retain the original point-only, unmeasured status and original
non-rejection/power caveats. Agreement among point decisions supplies no statistical
optimality or policy-generalization guarantee.
