# Python workflow reference

Start with the [Python quickstart](start/first-analysis.md) for a complete example.
Use this page when you need to configure or integrate an analysis.

## Prepare explicitly when useful

```python
study = ant.prepare(data, graph=graph, query=query, execution=ant.Execution(seed=19))
report = study.inspect()        # includes cached identification; does not estimate
result = ant.estimate(study)   # uses the owned data and the execution declared at preparation
```

### Ask a policy question

`ant.prepare(data, graph=dag, query=PolicyProgramValue(programs, outcome, n_draws=...))`
(from `antecedent.intervention_programs`) prepares the value of an intervention program such as
`price := max(1, 0.85 * price)` on a fitted structural model. The program is
compiled during preparation, while mechanism fitting and Monte Carlo evaluation require
explicit execution. Use `ant.estimate(study)`, `ant.rebind(study, data=new_data)`, or
`ant.reconfigure(study, execution=ant.Execution(seed=...))` for execution and independent
branches. The result reports the support decision of the
simulation, counterfactual and identification routes, the contract (the structural assumptions the
value rests on) and the provenance record. Estimand options such as `estimator=` and `refute=`
do not apply and are refused; see the [intervention programs guide](guides/intervention-programs.md).

### Scientific defaults and execution policy

`analyze` and `prepare` resolve the same native scientific defaults. Declare
bootstrap counts and posterior draws on `Frequentist` or `Bayesian`; validation
budgets belong to the applicable validation declaration. Scientific profiles
such as `Frequentist(profile="interactive")` select presets only for omitted
values. An explicit scientific budget takes precedence over its profile.

Seeds, threads, cancellation, resource limits, cache policy, and observers
belong to reusable `Execution` declarations. Replace a policy through an
immutable `reconfigure` branch. The shared default table is available through
`antecedent._native.omitted_defaults()`; Rust owns the resolved numerical defaults.

### Inspect and preview a study

Use `study.inspect()` to read what the study already knows: cached
identification, contract identities, and the reasoning report. Until estimation
runs, calibration is `unavailable` with the reason `not_executed`.

`study.preflight()` gives the cheaper structural view from before identification.
It has no `program_id`, because that identity also depends on identification.

Use `study.preview_transform(intent)` to check a proposed change without
executing it. Intents include `compatible_data_replace`, `retarget`,
`filter_population`, `new_conditional_query`, `change_graph`, and
`display_precision`. If the change cannot run, the preview reports `refused`
with the same `refusal` and `refusal_code` as the operation itself. For example,
retargeting without prepared scores returns `score_table_unavailable`.

### Refusal codes

A refusal carries a registered `reason_code` (and `error.report.code`). An
estimator that does not implement the requested inference is refused with
`estimator_inference_mismatch`; a question with no identified estimand raises
`antecedent.errors.EffectNotIdentified` with `identification_status` and
`search_complete`.

`result.inspect().to_dict()` and `ant.load(result.export()).inspect().to_dict()`
report the same portable record: `contract` is the exported contract section,
and the identification method and adjustment set, the validation verdict
(including a failed one) and counterfactual `unit_effects` are read from the
exported body.

## Choose a target population

A query owns its target population:
`ant.AverageEffect("t", "y", target_population=Treated())`. Every query whose
Rust kind is population-scoped carries the same keyword-only field
(`AverageEffect`, `ConditionalEffect`, `MediationEffect`,
`TemporalMediationEffect`, `PathSpecificEffect`, `InterventionalDistribution`,
`PulseEffect`, `SustainedEffect`, `InterventionResponse`, `ResponseCurve` and
the six derivative queries) and accepts the `antecedent.population` types
(`AllRows`, `Treated`, `Untreated`, `Named`, `Rows`, `CustomDistribution`, with
named and custom targets bound by `population_registry=`). `None` means the
all-observed population. The Rust study builder licenses the declaration: an
`AverageEffect` estimates the targets its estimator supports (AIPW and the
propensity estimators take treated, untreated, predicate and custom-distribution
targets; `linear.adjustment.ate` and Bayesian g-computation take only the
all-observed population on every graph class). Every other population-scoped
query refuses a declared population with `reason_code="population_not_estimable"`
instead of answering for the all-observed rows; for a reweighted response or
average effect, prepare the all-observed AIPW or cell-AIPW study and `retarget`
its frozen scores. `estimation.analyze_many` and `estimation.prepare_many` preserve each
query's own population declaration.

## Read counterfactual unit effects

A `Counterfactual` result (a two-world GCM ITE; see
[interventions and counterfactuals](capabilities.md#interventions-and-counterfactuals))
carries its per-unit output on the result itself, aligned row for row:

- `result.unit_effects`: one effect per unit.
- `result.unit_effect_intervals`: per-unit `(lower, upper)` pairs, with
  `unit_effect_intervals_level` and `unit_effect_intervals_method`. They are
  published under `inference=ant.Bayesian()`, as posterior quantiles of each
  unit's draws (`unit_posterior_quantile`). On the frequentist path all three
  are `None`; the `gcm.counterfactual.uncertainty_unavailable` diagnostic says
  why. Rust callers read the same fields on `IteResult`.
- `result.unit_extrapolative`: support flags, `True` where the unit's
  prediction into the arm it did not receive leaves that arm's observed
  support (its covariate cell or its abducted disturbance). The
  `gcm.counterfactual.support` diagnostic counts each kind.
- `result.estimate.unit_effects_homogeneous`: `True` when no
  effect-modifying mechanism family was fit, so equal unit effects hold by
  construction rather than as a finding.

A unit ITE is not a conditional average effect. For a pointwise interval on
the CATE `tau(x)` at prespecified covariate profiles, use the Rust
[`DrLearner::fit_pointwise_profiles`](guides/heterogeneous-effects-and-policies.md#pointwise-cate-at-a-covariate-profile).

## Estimate a family of average effects

`ant.estimation.prepare_many(data, graph=..., queries=[...], inference=...,
execution=ant.Execution(seed=19))` owns one table and prepares an immutable batch
without fitting. `ant.estimate(study)` returns `Outcome[BatchResult]`;
`require_success(outcome).members` contains one typed outcome per query, in query order.
`ant.estimation.analyze_many(...)` combines preparation and execution.
The same operations accept homogeneous `AverageEffect` queries or static discrete joint
`InterventionResponse` cells. Select `batch.member(index)` explicitly for downstream
composition or export. Branch with `ant.rebind(batch.study, data=new_data)` or
`ant.reconfigure(batch.study, queries=other_queries)`; earlier completions stay unchanged.
Run other query families through `ant.analyze`.

- **One seed for the family.** Every query cross-fits with the batch's `seed`
  and fold count, so it draws the same fold plan as the same query passed to
  `ant.analyze(..., execution=ant.Execution(seed=seed))` and returns the same estimate.
- **Shared nuisance fits.** Queries that would fit the identical cross-fitted
  AIPW nuisance share one fit. The propensity model is shared when the
  treatment coding, adjustment set, complete-case rows, fold plan and learner
  options all match (for example, one treatment against several outcomes); the
  outcome regressions are shared when the outcome column matches as well. A
  fit is reused only on bit-identical inputs, so a shared result is
  bit-identical to the query's own `ant.analyze` fit, and different data,
  seeds, folds or learners never share. Each result's `batch.shared_design`
  diagnostic says whether its propensity (`shares_propensity`) and outcome
  regressions (`shares_outcome_residualization`) were shared. Bootstrap
  replicates and joint-cell (`cell.aipw`) nuisances are fit per query.
- **Simultaneous intervals (max-t).** With two or more queries on the
  all-observed population, the batch stacks each query's influence function
  on the shared rows into one covariance (`result.estimate.joint_covariance`)
  and sets `result.estimate.simultaneous_interval` to `(lower, upper, 0.95)`.
  The critical value is the 95% quantile of `max_j |Z_j|` for Gaussian `Z`
  with the family's correlation, so the intervals hold jointly across the
  family instead of one query at a time.
- **Multiplicity-adjusted p-values (BH/BY).** `result.estimate.adjusted_p_values`
  is `(bh, by)`: Benjamini–Hochberg and Benjamini–Yekutieli adjustments of
  each query's two-sided Wald test of a zero effect across the family. BY
  allows arbitrary dependence among the estimates; BH assumes positive
  dependence. A degenerate standard error gives `NaN`, which is never ranked
  as a discovery.
- **Screen, then estimate.** `candidate_screen=CandidateScreen(screen_id,
  procedure, screen_rows, estimate_rows)` records a screen/estimate split, with
  `procedure` one of `"max_t"`, `"bh"`, `"by"` or `"unrecorded"`;
  `result.estimate.candidate_selection` reports the winning index, the family
  size and whether the two row sets are disjoint.

When the family cannot be formed (a declared target population, or a query
without an influence function), the joint fields stay `None` and each result
carries `batch.joint_if.unavailable`. If only the max-t critical value fails,
the p-values remain and each result carries `batch.joint_if.max_t_unavailable`.
For joint cells, `simultaneous_interval` is on the cell levels, and the
p-values test `family=ant.estimation.CellFamily(contrast="cell_minus_control")`
(the default); `CellFamily(contrast=None)` publishes no p-values.
`CellFamily(posterior_draws=300)` requests aligned score-posterior draws for joint
cells. Supplying `family` to an average-effect batch raises an argument error.

**What a batch covers.** `estimation.analyze_many` and `estimation.prepare_many`
accept homogeneous `AverageEffect` queries or static discrete joint
`InterventionResponse` cells (two or more `Set` interventions). Other queries have
`reason_code="route_not_supported"` and a remedy naming `ant.analyze`.
The concise execution boundary returns a typed refusal; malformed arguments and
operational failures raise typed errors. Native identification refusals remain visible
on prepared handles before explicit execution. No unsupported query silently executes
as another scientific method.

**Configured batches and retarget.** The batch entry points take `estimator=` as a string
id or a typed configuration (`Aipw(...)`) and the `estimator_config=` dict, exactly as
`ant.analyze` does; one configuration governs the batch. The completed `BatchResult.retarget` returns `Outcome[BatchRetarget]` with the
points, joint score covariance and named contrasts of a declared family over one row snapshot
(points and plug-in covariance remain distinct from the simultaneous interval). Preparation has no
scores to retarget; every completed result keeps its own row snapshot. See
[the guide](guides/batch-retarget.md).

## Configure response estimates

`PulseEffect` and `SustainedEffect` take `control_level` as well as
`active_level`; the effect is the contrast between the two. Continuous-response
`estimator_config` accepts `bandwidth`, `confidence_level`, `folds`,
`nuisance_basis`, `nuisance_lambda` and `minimum_local_ess` on curves and
derivatives, plus `simultaneous_replicates`, `multiplier_seed` and
`export_row_diagnostics` on a `ResponseCurve`, on a `Dag`, a `Cpdag` / `Pag`
envelope and a graph-posterior mixture alike; an unknown key is refused.

## Set the temporal history window

`PulseEffect` and `SustainedEffect` take `max_history_lag` (default `None`):
the number of steps back the temporal unfolding may look for an adjustment
set. When identification needs an older covariate, including a treatment
parent under the pulse parent-adjustment fallback, the refusal names
`max_history_lag`, and raising it on the query is the remedy. With the
default, the unfolding grows to the graph's own chain bound, and a refusal
there (a lagged cycle) is not helped by a cap.

## Declare delayed feedback

A temporal `ResponseCurve` or `InterventionResponse` takes a keyword-only
`feedback=FeedbackMechanism(delay=, level=, autoregression=, input_gain=,
innovation_variance=, initial_law=None, initial_state=0.0)`. It declares the
dynamics `s <- a s + b u + e`; they are declared, not fitted. `delay` is the
feedback delay in ticks (at least 1) and is not `treatment_lag`; `level` is the
dose the pulse or sustained policy holds; `initial_law` is `None` for the
contemporaneous observed law or the name of a law, and `initial_state` is its
level at the origin. With a declaration the response is the finite-horizon
rollout of those dynamics at every requested horizon
(`model.delayed_feedback.finite_horizon`), with no band; without one the fitted
dose-by-horizon surface is published. Feedback requires `horizons`. An
off-cadence horizon, a delay that arrives after the horizon, and an ancestry
deeper than the declared `max_history_lag` are refused.

## Choose a Bayesian likelihood

`Bayesian(likelihood=...)` chooses the g-computation outcome model:
`"gaussian"` (identity link, the default), `"logit"` or `"probit"` (Bernoulli)
and `"poisson"` (log link). A non-Gaussian likelihood is fitted for a tabular
`AverageEffect` mean on a `Dag` with the `laplace` or `hmc` backend under the
isotropic `prior_scale`. The effect is the average over rows of the inverse-link
contrast, a mean difference on the outcome scale. Every other route, the
`conjugate` backend and `prior_from=` refuse a non-Gaussian likelihood with
`reason_code="likelihood_not_supported"` rather than fit a Gaussian model. The
likelihood is part of the inference-binding identity and of the calibration
key's posterior construction. A Bayesian fit under the Gaussian likelihood to
an outcome that is 0/1 or nonnegative-integer valued reports the warning
diagnostic `estimate.bayesian.gaussian_likelihood_discrete_outcome`.

## Bound the draws behind a Bayesian interval

An interval from posterior draws is a finite-sample quantile, so each endpoint
carries a Monte Carlo standard error that belongs to the draws, not the
posterior. `Bayesian(n_draws=..., draw_controls="measured")` bounds it for a
tabular `AverageEffect` on a `Dag`:

- `n_draws` must reach the backend's supported minimum, 2000 for `laplace` and
  `conjugate` and 3000 for `hmc` (the floor its chain diagnostics need). A
  smaller count, or the default count, is refused when the study is built with
  `reason_code="posterior_draws_below_supported"`, naming the minimum. The
  count is never rewritten.
- The Monte Carlo standard error of the larger interval endpoint must be at most
  a twentieth of the interval width. Independent draws (`laplace`, `conjugate`)
  carry the order-statistic error of the draw count. An `hmc` chain's draws are
  autocorrelated, so the same error is discounted by the tail effective sample
  size; a sampler that mixes poorly is refused with
  `reason_code="posterior_draws_not_converged"`, naming R-hat and the effective
  sample sizes (or the endpoint errors and the tolerance). The reported errors
  are the `estimate.bayesian.draw_controls` diagnostic.
- The actual draw count is part of the inference-binding identity, the controls
  enter it only when requested, and the seed is part of the execution identity,
  so changing either changes the identity and replaying with both reproduces
  every draw.

Any other route refuses the option with `reason_code="invalid_argument"` rather
than ignore it. The interval stays `scope_not_assessed` until a coverage record
covers the draw count and sample size.

## Clip or trim propensity scores

The propensity-score estimators (`PropensityWeighting`, `PropensityMatching`,
`PropensityStratification`, `DistanceMatching` and `Aipw` in
`antecedent.estimators`) take `overlap=Overlap(clip=..., trim=...)`, or the
`estimator_config` key `overlap={"clip": ..., "trim": ...}`. By default
propensities are clipped into `[0.01, 0.99]` and no unit is trimmed, and
`Overlap()` spells out that default. `clip` bounds the propensities used in
the weights, and `trim` drops units whose propensity lies outside
`[trim, 1 - trim]`. Trimming narrows the population the effect describes.
`None` turns either one off. A non-default policy changes the inference-binding
identity and is part of the calibration key, so it reports
`scope_not_assessed` unless a coverage record measured that policy.

## Penalize the AIPW propensity

`Aipw(propensity_penalty=PropensityPenalty())` fits a ridge-logistic propensity on each
cross-fit fold's training rows, with the penalty chosen from a fixed grid by seeded inner
cross-validation on those rows only; `PropensityPenalty(kind="lasso")` fits an L1-penalized
one and also selects its support on those rows (recorded per fold in
the `penalized_support` diagnostic). The route keeps the score table and retargeting (with its
covariance) and publishes the cross-fitted influence-function SE; with `bootstrap > 0` it
publishes a bootstrap SE whose every replicate repeats penalty selection and nuisance
fitting (`penalized_bootstrap` lists the penalties each replicate selected). Both
rest on a stated remainder condition and are calibration-wired and unmeasured, so
they report `scope_not_assessed`. `nuisance_fallback="ridge_logistic"` / `"lasso"` (or a
`PropensityPenalty` for its tuning) re-runs the route with that propensity when the plain
GLM fit fails and records the failed fit in `penalized_fallback`;
`nuisance_fallback="ml"` stays closed (`nuisance_fallback_not_licensed`) and a lasso outside
the cross-fitted untrimmed mean ATE is refused (`selection_inference_not_licensed`). This is
distinct from `GlmOptions.ridge_on_separation`. See [the guide](guides/penalized-aipw.md).

## Declare the independence unit of the AIPW cross-fit

`Aipw(bootstrap=0, cluster_dml=ClusterDml(cluster_ids=..., min_clusters=20))` cross-fits
with whole clusters owning folds (not an IID cross-fit followed by `se="cluster"`). It
reports the cross-fitted point, keeps the score table and reports the cluster-sandwich
standard error of the scores as `se_analytic`: a bootstrap or a non-default `se` is refused
up front (`cluster_interval_not_licensed`), fewer clusters than
`min_clusters` refuse `too_few_clusters`. `ClusterDml(unit="dyad", cluster_ids=...,
second_cluster_ids=...)` declares two-way dependence: folds own whole connected components of
the endpoint graph (no endpoint crosses folds) and `se_analytic` is the two-way sandwich; an entity used in both
endpoint roles or one giant component refuses `dyadic_dependence_not_licensed`, and fewer
than `min_components_per_fold` components per fold refuse `too_few_clusters`. `DML` and
`DRLearner` declare dependence through `cluster_dml` only, and `CausalForest` through
`cluster="site"` (whole-cluster subsampling, honesty and cross-fitting; see
[the guide](guides/clustered-dml.md#clusters-for-propensity-weighting-stratification-and-the-causal-forest)): a row-aligned `cluster_ids`, `cluster_dml` or
`multiway_ids` key in their `estimator_config` refuses `route_not_supported`.
`PropensityWeighting` and `PropensityStratification` take `cluster="site"` for a cluster
sandwich with the propensity-estimation correction and a whole-cluster bootstrap.
`CandidateScreen.from_units(...)` builds a
deterministic screen/estimate split from entity labels, or from dyad endpoints so that
connected endpoints never cross the split, and records the seed and a unit digest in
`screen_id`. See [the guide](guides/clustered-dml.md).

## Sampling weights

`weights="w"` on `LinearAdjustment`, `GlmAdjustment`, `Aipw`, `PropensityWeighting`,
`PropensityMatching`, `DistanceMatching`, `PropensityStratification`, `DML` and `DRLearner`, on
`RandomizedEffect`, on `PanelDifferenceInDifferences` and on
`PulseEffect` names a column of sampling weights (or gives one weight per row). The estimate
speaks for the weighted population, weights are normalized to mean one (so their scale changes
nothing), rows of weight zero are excluded and counted, a negative, NaN or infinite weight
refuses naming its row (`sampling_weights_invalid`), and `result.estimate.sampling_weights`
reports the rows used, the zero-weight rows, the Kish effective sample size and the declared
weight sum. A construction with no weighted contrast refuses the weights
(`option_not_applicable`, `sampling_weights.route_unsupported`). See
[the guide](guides/sampling-weights.md).

## Read identification details

`identify(...)` returns an `Identification` whose `statement`, `verdict`,
`qualified_verdict`, `assumption_statements`, and `derivation_statements` are
human-readable state on the object (`identification.to_dict()` includes them).
`qualified_verdict` keeps the restriction an identified verdict holds under;
`statement` uses it. That is readable identification data, not a notebook
renderer.

## Reuse data and branch a study

| Operation | Data used | Returned object |
|---|---|---|
| `ant.estimate(study)` | Retained data | A new outcome |
| `ant.rebind(study, data=new_data)` | Newly declared observations | A new prepared branch |
| `ant.reconfigure(study, query=other_query)` | Compatible retained observations | A new prepared branch |
| `result.inspect()` | That outcome's scientific record | An inspection report |
| `result.export()` | The explicitly selected scientific product | A typed artifact |

Earlier studies, outcomes, and exported bytes stay unchanged. Inspect
`branch.reuse` for retained and invalidated work. When native reuse is unsupported
(for example a changed frozen design), `ant.rebind` refuses and names the remedy:
call `ant.prepare` again with the new declarations. A branch never silently
substitutes another scientific method. Streaming and adaptive sessions keep
their explicitly stateful domain interfaces.

## Software and agent consumers

```python
report = result.inspect().to_dict()
# json.dumps(report, allow_nan=False) is supported.

branch = ant.rebind(study, data=new_data)
next_result = ant.estimate(branch)
if next_result.refusal is not None:
    print(next_result.refusal.code, next_result.refusal.message)
    print(next_result.refusal.obligations, next_result.refusal.next_actions)
```

Scientific refusals retain stable codes, details, obligations and available next actions.
Unknown scientific evidence stays unavailable. Invalid arguments, resource failures,
cancellation and implementation failures raise typed errors; callback exceptions
retain their original object and traceback. These failures do not publish a successful
scientific outcome.

Reports retain identification mass, uncertainty sources and targets (including
omitted components), assumption obligations, and support evidence. A licensed
matrix cell does not establish empirical support or validate its assumptions.

## Portable executions

```python
artifact = result.export()
encoded = bytes(artifact)
loaded = ant.load(encoded, expected=artifact.expectation)
loaded.acceptance.status  # "verified", "sealed", or "unavailable"
loaded.acceptance.unresolved  # sealed checked operations the consumer cannot replay
loaded.answer
loaded.inspect().to_dict()
assert bytes(loaded.export()) == encoded
```

`load` runs the Rust semantic consumer. Malformed contracts fail closed;
missing or unrecognized contracts yield explicitly unavailable semantic views
that can still be forwarded. A `verified` load replayed a verified program. A
`sealed` load recognized the artifact, verified its contract and identities,
kept the recorded answer, and names the sealed checked operation it cannot
replay from bytes alone; `acceptance.verified` and `acceptance.replayable`
stay `False` for it. Acceptance verifies the artifact's semantic
contract, not the truth of causal assumptions. Loading does not reconstruct a
live study or recover the source dataset. Use the common inspection report to read acceptance and retained scientific standing;
requested specialized conversions are explicit materialization boundaries.

For prior transfer, select `posterior = result.export(quantity="posterior")` and
pass that typed artifact as `Bayesian(prior_from=posterior)`. This selection
materializes the retained posterior without another fit. The full result artifact
preserves its own execution and acceptance standing. See [artifact examples](artifacts.md#exporting-prepared-results).

`program_id` is the compiled program identity (`contract["program"]`).
`claim_id` is the execution claim identity (`contract["claim"]["claim_id"]`).
`data_snapshot_id` is the identity of the data the execution ran on. They are
different layers. A prepared inspect has a program and no claim.

## Current boundaries

**Composition evidence.** Every licensed support-matrix cell completes the inspect, preview, execute, claim and consume sequence in the Rust compiler path on synthetic fixtures. That shows the pieces compose; it does not carry over a parent estimator's coverage evidence.

**Reusable studies.** `analyze` performs preparation followed by explicit
estimation. Completed analysis outcomes retain their applicable immutable study;
independently loaded outcomes have no live study or dataset. Read the outcome's
inspection and applicable operations for its actual retained capabilities.
Custom-validator findings remain caller-declared evidence. Scientific refusals
and unsupported exports do not silently rerun discovery or estimation.

Retargeting uses the frozen scores belonging to a completed outcome. Prepare
new data through `ant.rebind`, execute that branch, and use its completed
`outcome.study.retarget(weights, depends_on=...)`; earlier completed outcomes
retain their own scores.
A nonconstant-weight retarget exports under a `RowWeights` target whose
`target_weights` identity binds the exact weight bits, their row count, the data
snapshot, the score table (`score_reuse`) and `depends_on`. The weights travel in
the artifact, so a consumer re-derives the identity and confirms the population
the answer is about; `inspect().target_weights_id` shows it. Constant weights of
any scale keep the original target. The binding belongs to the retargeted
result, not to the plan: the plan is still an AllObserved study and re-estimates
on new data, while `population.replay_retarget(completed, artifact)` independently consumes
the artifact and reuses the completed source's exact scores. A different snapshot
or score identity returns the scientific refusal `row_weights_bound_to_snapshot`.
