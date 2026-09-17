# Python workflow reference

Start with the [Python quickstart](python-workflow.md) for a complete example.
Use this page when you need to configure or integrate an analysis.

## Prepare explicitly when useful

```python
study = ant.prepare(data, graph=graph, query=query, seed=19)
report = study.inspect()        # includes cached identification; does not estimate
result = study.estimate()      # uses retained data and preparation's seed/threads
```

### Defaults and latency

`analyze` and `prepare` use the same defaults. Set `bootstrap`, `refute`, or
`n_draws` explicitly when you need a particular budget. `latency=` selects a
budget tier only for values you omit; it never overrides an explicit value.

For integrations, the shared default table is available through
`antecedent._native.omitted_defaults()`. The Rust study builder owns it, and
`antecedent.estimation.PreparedAnalysis.prepare(...)` uses it too.

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
its frozen scores. `analyze_many` and `PreparedBatch.prepare` estimate each
query's own population.

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

## Read identification details

`identify(...)` returns an `Identification` whose `statement`, `verdict`,
`qualified_verdict`, `assumption_statements`, and `derivation_statements` are
human-readable state on the object (`identification.to_dict()` includes them).
`qualified_verdict` keeps the restriction an identified verdict holds under;
`statement` uses it. That is readable identification data, not a notebook
renderer.

## Reuse data and refresh a study

| Operation | Data used | Changes the study's retained data? |
|---|---|---|
| `study.estimate()` | Retained data | No |
| `study.estimate(other_data)` | Supplied data | No |
| `study.refresh(new_data)` | Supplied data | Yes, after successful execution |
| `result.inspect()` | That result's execution | No |
| `result.export()` | That result's execution | No |

A failed refresh leaves the study usable. Refresh does not mutate an earlier
result or its exported bytes. A study is a live, in-process resource; retain it
while repeated estimation is useful and release references when done.

## Software and agent consumers

```python
report = result.inspect().to_dict()
# json.dumps(report, allow_nan=False) is supported.

try:
    next_result = study.refresh(new_data)
except ant.CausalError as error:
    refusal = error.report.to_dict()
    # Original exception type and descriptive message are preserved.
    # The failed study operation also retains error.study for recovery.
```

Refusals contain the operation, code, message, query and available review hint /
pending edges. Unknown scientific evidence stays unavailable. Existing Python
argument-validation `ValueError` and `TypeError` exceptions also receive reports;
their original exception types remain unchanged.

Reports retain identification mass, uncertainty sources and targets (including
omitted components), assumption obligations, and support evidence. A licensed
matrix cell does not establish empirical support or validate its assumptions.

## Portable executions

```python
encoded = result.export()
loaded = ant.load(encoded)
loaded.acceptance.verified
loaded.answer
loaded.inspect().to_dict()
assert loaded.export() == encoded
```

`load` runs the Rust semantic consumer. Malformed contracts fail closed;
missing or unrecognized contracts yield explicitly unavailable semantic views
that can still be forwarded. Acceptance verifies the artifact's semantic
contract, not the truth of causal assumptions. Loading does not reconstruct a
live study or recover the source dataset. The original decoded body is available
as `loaded.artifact.payload`.

For prior transfer, `Bayesian(prior_from=...)` still expects the posterior-only
payload from `result.study.export_artifact()`, taken before the study changes.
The full `result.export()` archive is intended for execution inspection and
forwarding. See [artifact examples](artifacts.md#exporting-prepared-results).

`program_id` is the compiled program identity (`contract["program"]`).
`claim_id` is the execution claim identity (`contract["claim"]["claim_id"]`).
`data_snapshot_id` is the identity of the data the execution ran on. They are
different layers. A prepared inspect has a program and no claim.

## Current boundaries

**Composition evidence.** Every licensed support-matrix cell has a first-class
inspect/contract coordinate and completes inspect → preview → execute →
claim → consume on the Rust compiler path
(`compiler.e2e_licensed_cells`). That is composition-seam evidence on
synthetic licensed-coordinate fixtures (`internal_cross_check`). Parent
estimator evidence is not inherited.

**Reusable studies.** `analyze` is `prepare(...).estimate()`. Every licensed
Python product-matrix route retains a study and a contracted export. Custom
validator results travel as caller-attested evidence. A result with no
execution (prepared-only, cancelled, or a body-only load) refuses `.study` /
`.export()` with `not_executed` or `cancelled_no_claim`. It never silently
reruns discovery.

Retargeting remains available from frozen scores of the execution it follows:
after `estimate(new_data)`, `retarget` reweights the scores fitted on `new_data`.
A nonconstant-weight retarget exports under a `RowWeights` target whose
`target_weights` identity binds the exact weight bits, their row count, the data
snapshot, the score table (`score_reuse`) and `depends_on`. The weights travel in
the artifact, so a consumer re-derives the identity and confirms the population
the answer is about; `inspect().target_weights_id` shows it. Constant weights of
any scale keep the original target. The binding belongs to the retargeted
result, not to the plan: the plan is still an AllObserved study and re-estimates
on new data, while `reexecute_retarget(artifact)` re-runs the carried weights and
raises `row_weights_bound_to_snapshot` on a different snapshot.
