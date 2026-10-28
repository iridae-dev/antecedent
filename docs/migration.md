# Migrate from 2.4 to 3.0

Antecedent 3.0 separates preparing an analysis from executing it. A prepared study is immutable: you branch it with `rebind` (new observations) or `reconfigure` (new declarations) and execute each branch with `estimate`. Every execution returns an `Outcome` that holds the answer or a typed scientific refusal. Seeds, threads, cancellation and resource limits belong to one `Execution` object, and statistical budgets belong to an inference declaration.

Work through the sections in order. Each gives the 2.4 form, the 3.0 form and what the change buys you. Python blocks are executed against a small known-truth data set; Rust has its own section; stored artifacts have one compatibility table.

## Update in this order

1. Move loose `seed=`, `threads=`, `cancel=`, memory and progress arguments into `ant.Execution(...)`.
2. Move `bootstrap=`, `latency=` and posterior-draw arguments into `ant.Frequentist(...)` or `ant.Bayesian(...)`.
3. Replace mutable study methods with `ant.prepare`, `ant.estimate`, `ant.rebind` and `ant.reconfigure`, and keep the returned `Outcome`.
4. Decide where a scientific refusal should stay a returned value and where your application wants an exception.
5. Replace `.artifact()` conversions with `.export()` and pass the producer's `expectation` to `ant.load`.
6. Regenerate stored artifacts named in the [compatibility table](#artifact-compatibility).

## Configure execution and inference

| 2.4 | 3.0 | What changes |
| --- | --- | --- |
| `analyze(..., seed=s, threads=t, cancel=token)` and the same on `prepare(...)` | `execution=ant.Execution(seed=s, threads=t, cancel=token)` | One reusable physical policy, separate from the scientific question. |
| `analyze(..., bootstrap=n)` / `prepare(..., bootstrap=n)` | `inference=ant.Frequentist(bootstrap=n)` | The statistical budget is part of the declared inference. |
| `analyze(..., latency="interactive")` | `inference=ant.Frequentist(profile="interactive")` or `ant.Bayesian(profile=...)` | The profile selects the same presets. Explicit budgets and validation declarations still win. |
| Seed fields on sampling, optimizer, uncertainty and transport declarations | `Execution(seed=...)` | Removing a seed from a declaration does not change its statistical budget. |
| A mutable study's latest result | The `Outcome` that `ant.estimate(study)` returns | Each call returns an independent outcome and leaves the study unchanged. |

The default seed is now 1; earlier operations that defaulted to 0 reproduce their draws with `Execution(seed=0)`. A new `Execution` replaces the old one, so a cancelled token is never inherited silently.

## Prepare, branch and execute

| 2.4 | 3.0 |
| --- | --- |
| `PreparedAnalysis.prepare(...)` | `ant.prepare(...)`; the mutable class is private. |
| `study.estimate(new_data)` / `study.refresh(new_data)` | `ant.estimate(ant.rebind(study, data=new_data))`. `ant.estimate(study)` takes no data. |
| Replace the query through a mutable method | `ant.reconfigure(study, query=...)` for declarations; `ant.rebind` for observations. |
| `study.reprepare(data)` after a refused design change | `ant.prepare(data, ...)` with the new design. `ant.rebind` refuses a changed frozen design and names this remedy. |
| `study.inspect_plan()`, `study.last_result` | `study.inspect()` before execution; retain each `Outcome` and inspect it. `outcome.study` is the completed preparation. |
| `result.refute(data, ...)` / `study.refute(data, ...)` | `result.refute(suite=...)` or `study.refute(suite=...)`; neither takes replacement data. |
| Discovery run implicitly by `analyze(..., discovery=PC(...))` | `accepted = ant.discovery.discover(data, method=PC(...), accept_discovered=True)`, then `ant.prepare(data, graph=accepted, ...)`. |
| `identify.estimate(stage, data)` | `stage.prepare(data, inference=..., execution=...)`, then `ant.estimate(...)`. |

```python
import antecedent as ant

data = ant.data.table(frame)
study = ant.prepare(
    data, graph=graph, query=query,
    inference=ant.Frequentist(bootstrap=25),
    execution=ant.Execution(seed=19, threads=4),
)
result = ant.estimate(study)
updated = ant.rebind(study, data=new_data)
alternative = ant.reconfigure(study, query=other_query)
updated_result = ant.estimate(updated)
```

This block continues an application with its own `frame`, `new_data`, `graph` and queries. Branching does not execute the branch or change earlier results, so `result` still answers the original data after `updated_result` is computed. `study.rebind(...)`, `study.reconfigure(...)` and `study.estimate()` call the same functions. Read `updated.reuse` to see which work a branch retained and which it invalidated.

Array and frame inputs enter owned storage, so later mutation of your source cannot change a prepared study. Keep one `ant.data.table(...)` handle to reuse across compatible preparations.

A profile and an explicit budget combine as you declare them:

```python
import antecedent as ant

profile_study = ant.prepare(
    frame, graph=graph, query=query,
    inference=ant.Frequentist(profile="interactive", bootstrap=25),
    execution=ant.Execution(seed=19, threads=2),
)
profile_outcome = ant.estimate(profile_study)
```

Here the interactive preset would otherwise omit the bootstrap; the declared 25 replicates are kept. `AcceptedGraph.prepare(...)` and `.analyze(...)` add `profile="interactive"` only when your inference declares no profile, and never rediscover structure.

## Handle outcomes and refusals

Execution returns an `Outcome`. Read `answer`, `inspect()`, `uncertainty`, `support`, `assumptions`, `diagnostics`, `provenance` and `identities` on it. The specialized result is `outcome.value`, so a 2.4 read such as `result.estimate` becomes `outcome.value.estimate`, and calibration standing is `outcome.inspect().calibration`.

| Situation | 3.0 behavior |
| --- | --- |
| A valid question the declarations cannot answer (the effect is not identified, a population is not estimable, an implementation is unavailable) | Returned: `outcome.refusal` carries a reason code and the next meaningful action, and `outcome.answer.kind` is `"refused"`. It is not a missing point estimate. |
| Invalid arguments, wrong call shapes, an unsupported data layout | Raised as a typed error. |
| Cancellation, resource exhaustion, implementation failure, a callback that raises | Raised. Observer exceptions keep their original object and traceback. |

A refusal is scientific information, so keep it when your application can act on it. When you prefer an exception, `require_success(outcome)` re-raises the original typed error.

```python
from antecedent.errors import CausalValueError, EffectNotIdentified
from antecedent.outcomes import require_success

confounded = ant.Admg.from_edges(
    ["z", "treatment", "outcome"],
    [("z", "treatment"), ("z", "outcome"), ("treatment", "outcome")],
    [("treatment", "outcome")],
)
refused = ant.analyze(frame, graph=confounded, query=query)
assert not refused.accepted and refused.answer.kind == "refused"

try:
    require_success(refused)
except EffectNotIdentified as error:
    refusal_error = error

try:
    ant.reconfigure(study, data=new_data)
except CausalValueError as error:
    wrong_call = error
```

The graph declares an unmeasured common cause of `treatment` and `outcome`, so the effect is not identified and `analyze` returns the refusal. Changing observations through `reconfigure` is a call error and raises; use `rebind`. Opt in to a description of a refusal with `on_refusal="describe"`: the causal refusal stays the primary outcome and `outcome.associational` and `outcome.evidence_plan` sit beside it. The associational answer does not acquire a causal reading. `antecedent.fallback.described(outcome)` reads all three.

## Export and load

| 2.4 | 3.0 |
| --- | --- |
| `result.artifact()`, `ledger.artifact(...)`, `prepared.artifact(...)`, `.artifact` and `.data` fields | `.export(...)` returns a typed `ExportedArtifact`: `bytes(artifact)`, `artifact.expectation`, `artifact.metadata`. |
| `study.export_artifact(payload="query")` / `study.export_artifact()` | `outcome.export(quantity="query")` / `outcome.export(quantity="record")` |
| `return_posterior_artifact=True`, `result.posterior.artifact` | `result.export(quantity="posterior")` after normal execution. |
| `cyclic_result.artifact`, `.report` | `bytes(result.export())`, `result.to_dict()` |
| `advanced.consume_exact(artifact).estimate()` | `ant.load(artifact)`; a consumed artifact is a checked read model and is not re-estimated. |

```python
artifact = result.export()
loaded = ant.load(bytes(artifact), expected=artifact.expectation)
print(loaded.acceptance)
```

Keep `artifact.expectation` on the producing side and pass it as `expected=`; an expectation read from the incoming container is not independent evidence of what you expected. Loading runs the format's independent native consumer. A loaded value has no live study or data (`loaded.study is None`); read `loaded.acceptance` for verification, identity binding, replay and resume standing. An outcome with several possible exports needs an explicit `quantity=`.

Policy-program artifacts keep their encoding and need the original ordered variable names at load: `LoadContext(supplied_sources={"names": names})`.

Select a posterior after execution, without another fit:

```python
import antecedent as ant

posterior_result = ant.analyze(
    frame, graph=graph, query=query,
    inference=ant.Bayesian(n_draws=32, backend="conjugate"),
    execution=ant.Execution(seed=19), refute=False,
)
posterior_artifact = posterior_result.export(quantity="posterior")
posterior_record = ant.load(posterior_artifact, expected=posterior_artifact.expectation)
ate_record = posterior_record.value.quantity("ate")
```

The sealed record holds summaries and any retained draws, with no live study or training data. It binds the encoding digest and grants no replay.

## Retarget a completed study to another population

Population retargeting uses a completed study with compatible retained scores and explicit weight dependencies. It does not rebind observations or fit another score model, and it is a scientific operation, not a branch:

```python
import antecedent as ant
import numpy as np

population_result = ant.analyze(
    frame,
    graph=[("z", "treatment"), ("z", "outcome"), ("treatment", "outcome")],
    query=ant.AverageEffect("treatment", "outcome"), estimator="aipw",
    inference=ant.Frequentist(bootstrap=0), refute=False,
)
population_study = population_result.study
targeted = population_study.retarget(
    np.exp(np.asarray(frame["z"]) / 3), depends_on=["z"],
)
targeted_artifact = targeted.export()
targeted_loaded = ant.load(targeted_artifact, expected=targeted_artifact.expectation)
```

The graph declares `z` as a common cause. Antecedent checks score availability, the binding to the original rows and the declared dependencies; nonconstant weights must be functions of the certified adjustment set and cannot depend on treatment descendants. The original answer and completed study stay unchanged, `targeted.inspect().target_weights_id` identifies the weights, and the loaded targeted result keeps that identity without reconstructing a study or data.

## Run shared-fit batches

`prepare_many` accepts average-effect queries, or a homogeneous family of discrete joint response cells. Preparation identifies the family without fitting. Execution returns one structured outcome: member outcomes you select explicitly, their family inference, and the completed preparation.

| 2.4 | 3.0 |
| --- | --- |
| `PreparedBatch.prepare(...)` / `.prepare_cells(...)`, then `.estimate(data)` | `ant.estimation.prepare_many(data, queries=..., ...)`, then `ant.estimate(batch)` |
| Retarget the scores of a batch's latest run | Keep the completed `BatchResult` and call its `retarget(...)`. |

```python
import antecedent as ant
from antecedent.outcomes import require_success

batch_frame = {**frame, "other_outcome": -frame["treatment"] + 2 * frame["z"]}
batch_graph = [*graph, ("treatment", "other_outcome"), ("z", "other_outcome")]
batch_study = ant.estimation.prepare_many(
    batch_frame, graph=batch_graph,
    queries=[query, ant.AverageEffect("treatment", "other_outcome")],
    inference=ant.Frequentist(bootstrap=0), refute=False,
    execution=ant.Execution(seed=19, threads=2),
)
batch_result = require_success(ant.estimate(batch_study))
selected_member = batch_result.member(0)
batch_branch = ant.rebind(
    batch_result.study,
    data={**batch_frame, "outcome": batch_frame["outcome"] + frame["treatment"]},
)
changed_batch = require_success(ant.estimate(batch_branch))
member_artifact = selected_member.export()
loaded_member = ant.load(member_artifact, expected=member_artifact.expectation)
```

Select a member before you pass it on or export it; a batch does not invent a joint distribution or pick a quantity. A branch of the whole batch recomputes the family covariance and simultaneous inference, while a standalone branch of one member drops family inference.

## Cyclic and feedback models

Cyclic preparations use the same `Execution`, `rebind` and `reconfigure`. System handles take training rows through `rebind(rows)` and their declarations through `reconfigure(worlds=..., noise=..., initial=..., model=..., policy=...)`. Changing only the response worlds reuses a compatible fit; changing observations, mechanism parameters, priors or fitting policy refits.

```python
import antecedent as ant
from antecedent.cyclic import LinearCyclicModel, World
from antecedent.intervention import Set

model = LinearCyclicModel(("A", "B"), (0.0, 0.4, 0.5, 0.0), (1.0, 0.0, 0.0, 1.0))
feedback = model.worlds(
    [World(1, Set("A", 1.0))], component_id="feedback",
    source_id="specified-model", certifying_evidence=["declared-equations"],
)
branch = ant.reconfigure(feedback, worlds=[World(2, Set("A", 2.0))])
feedback_result = branch.interventions()
assert feedback_result.accepted
assert feedback_result.value[2].mean("B") == 1.0
```

This fully specified system has `B = 0.5 A`, so fixing `A` at 2 gives a mean of 1 for `B`. The answer is conditional on the specified model and its declared equilibrium assumptions. Cyclic results keep their native owner: `result.to_dict()` and `bytes(result.export())` are the explicit materialization boundaries, replacing `.report` and `.artifact`.

## Application-owned execution policy and transport

Cancellation tokens, memory limits and progress callbacks come from your application, so these blocks are fragments that need your own `data`, `query` and `cancellation_token`:

```python
execution = ant.Execution(
    seed=19,
    limits=ant.ResourceLimits(memory_bytes=256_000_000),
    cancel=cancellation_token,
)
study = ant.prepare(data, query=query, execution=execution)
```

Transport follows the same split. `TransportControls` and `controls=` are removed; `Execution` carries physical policy, while formula-operation, recursion-depth and support-enumeration bounds stay declarations on the query:

```python
query = transport.advanced.ExactTransportQuery(
    identification, catalog, {"x": 1.0},
    evaluation=transport.TransportEvaluation(
        max_operations=10_000_000, max_depth=256, max_support_rows=1_000_000,
    ),
)
```

| 2.4 | 3.0 |
| --- | --- |
| `advanced.prepare_exact(identification, catalog, data, at=at, max_operations=n)` | `ant.prepare(data, query=ExactTransportQuery(identification, catalog, at, evaluation=TransportEvaluation(max_operations=n)))` |
| `advanced.prepare_statistical(..., bootstrap=b, seed=s, memory_bytes=m, cancel=t)` | `ant.prepare(data, query=StatisticalTransportQuery(..., bootstrap=b), execution=ant.Execution(seed=s, limits=ant.ResourceLimits(memory_bytes=m), cancel=t))` |
| `advanced.prepare_response_grid(..., at=[...])` | `ant.prepare(data, query=TransportResponseGridQuery(..., at=[...]))` |
| `TransportInference(bootstrap=199, seed=8)` | `TransportInference(bootstrap=199)` with `execution=ant.Execution(seed=8)` |
| `advanced.prepare_trial(...)` | `ant.prepare(trial_data, query=trial_query, provider=transport.TrialAipw(...), inference=transport.TransportInference(...), execution=...)` |
| Loose `cancel=`, `memory_bytes=`, `threads=`, `seed=` on `identify_*`, `consume_*`, planning and pilot entries | `execution=ant.Execution(...)` |

Exhausting an evaluation bound is a typed evaluator failure and never publishes a silently truncated answer. `Execution(on_stage=...)` and `Execution(threads=...)` are refused with `option_not_applicable` on transport engines, because they stream no stages and run no worker pool; they are never dropped silently. The sampling seed of pilot and covariance entries was 0; pass `Execution(seed=0)` to reproduce an old draw.

The treatment-model nuisance score posterior is now an explicit fitting operation on the prepared study (`nuisance_scores.treatment_score_posterior`). Preparation and inspection do not fit it, and its seed comes from `Execution`.

## Rust

| 2.4 | 3.0 |
| --- | --- |
| `PreparedStudy::refresh`, `refresh_series`, `refresh_panel`, `refresh_multi_env` | `rebind_data`, `rebind_series`, `rebind_panel`, `rebind_multi_env`, each returning a new prepared study; a failed rebind leaves the old handle intact. |
| Physical plans built as struct literals | Obtain plans from the compiler. `PreparedStudy::compiled_linear_program()` inspects the retained scientific target, identification and operators without fitting, and preparation consumes that exact product instead of identifying again. |
| `FlagshipWorkflow::decision()` returning a value | It returns a result. `DecisionAuthority::withdraw_source` records trust loss with a reason; withdrawal and supersession invalidate dependent decision access, recalculation and export. |
| Version 1 decision export | `export_decision()` writes a version 2 envelope with the replay payload, identity, ledger result identity and source revisions. `load_decision` needs the caller's current `DecisionAuthority`; an embedded old ledger cannot authorize an artifact. |
| `replay_decision()` as the only read | It returns historical numerical moments without evaluating a current decision, for both envelope versions. `export_historical_replay()` serves independent numerical consumers. Historical replay never renews withdrawn permission. |
| `ContrastTerm::coefficient` as an integer | An `f64`. Use it directly in a shared-covariance quadratic form; do not cast it or subtract marginal interval endpoints. |
| Fixed difference and interaction contrasts only | `InterventionContrast::linear` takes an explicit finite combination of worlds with one nonzero coefficient each and an explicit reference. Adding a term changes the question; it does not identify the added world. |
| Convenience evaluation of a dose-grid posterior | `Study::prepare_posterior_row(request, ctx)` compiles without fitting or drawing; `estimate(ctx)` runs it and `retarget(grid, functional, ctx)` reuses the coefficient posterior for compatible grid and functional changes. Changed data, priors, draws or seeds need a new preparation. The convenience calls delegate to the same compiled owner. |
| Panel joint responses without a retained program | `Study::prepare_joint_panel`; retarget worlds on the prepared program to reuse a compatible fit. |
| Context-free bridge propagation | Contract-bound affine propagation and support mapping call `BridgeContract::check_licensing()` first. |

Python's `dose_grid_posterior` and `consume_dose_grid_posterior_artifact` take an optional `execution=Execution(threads=..., limits=ResourceLimits(memory_bytes=...), cancel=...)`; omitting it keeps the default policy. Resource and cancellation refusals raise the existing typed errors.

`mean_source`, `law_source` and `project_*_source` carry scientific permission into decision and search use; `numerical_*` accessors expose values without that permission. Frozen-fit replay keeps aligned draw slots and failures. Retained and returned results hold their memory reservations until their last owner is dropped; these are allocation quotes, not process memory measurements.

## Behaviour changes that can alter an answer or turn it into a refusal

These corrections change a number you got before, or replace a silently wrong answer with a typed refusal. Rerun affected studies and read the refusal's next attempt.

| Area | Before | Now |
| --- | --- | --- |
| Table weights on static routes | A weighted table was answered unweighted on conditional-effect, mediation, counterfactual, curve, derivative, intervention-response, dose, regression-discontinuity, temporal-design, synthetic-control and survival routes. | Refused (`option_not_applicable`, `sampling_weights.route_unsupported`), as are weights declared on an estimator the route does not run. Declare the weights on an average-effect estimator, or drop them. A multi-step sustained temporal window (frequentist) now reads weights. See [sampling weights](guides/sampling-weights.md). |
| Treatment with undeclared extra levels | Linear and GLM adjustment and instrumental fits read `0/1/2` as a dose (a slope, not the contrast of two levels). | Refused as `treatment_level_undeclared`. Rust regressions can declare `OtherLevels::ExcludeUndeclared`; Python `LinearAdjustment`, `GlmAdjustment` and the instrumental estimators have no such argument, so recode the treatment. See [treatment levels](guides/treatment-levels.md). |
| Declared dependence | The equivalence-class average-effect route used the plain HC1 error; a study-level `sampling_design="clustered_panel"` or `"dependent_time_series"` with the bootstrap off was recorded but not read. | The route applies a declared cluster or Newey-West error. With the bootstrap off, the design is honoured only where the estimator's standard error reads it (cluster, multiway or panel-HAC; Newey-West or panel-HAC); otherwise, and for matching and instrumental estimators, it is refused. |
| Declared dependence on routes that cannot read it | The natural-direct mediation bootstrap, the randomized-survival subject bootstrap, the sharp regression-discontinuity estimator (with or without a bootstrap), the fuzzy local-polynomial ratio, the policy value, the continuous-dose response, the static structure searches and the Bayesian static routes returned the independent-rows number (a row bootstrap or an independent-rows error or posterior) under a declared `clustered_panel` or `dependent_time_series`. | Refused with `option_not_applicable` and a next attempt. Declare independent subject trajectories if the rows are independent, or use an average-effect estimator with a cluster or Newey-West error. A policy value or sharp-discontinuity table that carries observation weights is also refused (`sampling_weights.route_unsupported`). |
| Conditional effect of a multi-level treatment | A conditional effect between two levels of a `0/1/2` treatment returned the dose slope (2 against a contrast of 1). | Refused as `treatment_level_undeclared`, naming the observed levels; recode the treatment to the two contrasted levels. |
| Multi-treatment derivatives | A Jacobian or directional derivative over several treatments could be answered from pairwise sets. | Identified jointly on Dag, Admg, Cpdag and Pag; not identified when no common joint adjustment set exists. Joint identification through the general ID algorithm is not wired. |
| Arrow strength | `ArrowStrength.strength` and `FittedGcm.arrow_strengths()[i].strength` were `coefficient**2 * Var(parent)`, a variance in the child's squared units that exceeds the child's variance when parents are correlated. | `strength` is the cited causal strength, the divergence of the joint from the joint with the edge cut: `0.5 * ln(1 + coefficient**2 * Var(parent) / sigma**2)` nats with `sigma**2` the child's noise variance (non-negative, scale-free). The old number is the new `variance_contribution`; rename any reader that wants it. A child with no noise variance beyond its parents is refused. No artifact stores either value. |
| Numerics | See the list below. | Estimates, standard errors or intervals can move. |

Numeric corrections: propensity stratification cuts at quantiles of the propensity; matching averages every tied donor (scalar and multivariate), so row order no longer matters; the trimmed AIPW influence and standard error use all rows; DML scores use the recorded folds; sandwich covariances are computed by QR; the Normal-Inverse-Gamma variance scale comes from residuals; Gaussian KL, the normal, Student-t and gamma tails and the Gaussian BIC keep full precision; a sharp regression-discontinuity window of four rows or fewer withholds its analytic error (NaN); the Anderson-Rubin set and the confounding-sensitivity threshold are scale-free; the Fisher-z interval keeps its endpoints at high correlation; discrete permutation tests count permuted statistics that tie the observed one; the temporal window certificate does not certify when a lag of two or more steps has autoregressive ancestry; regression-based sensitivity refuters refuse small-integer dose treatments in temporal designs.

Evidence labels were also tightened: claims whose fixture only asserted that a seeded dataset behaves as constructed (for example mechanism-change detection, the kernel two-sample shift test, the NOTEARS chain and static PAG operations) are recorded as software coverage rather than independent known truth. The conformance pages reflect the current labels.

## Artifact compatibility

The container format is unchanged, and readers that migrate container sources 0.1 through 0.5 still do. Identity values changed in two groups: those that took effect in 3.0, and those that took effect in 2.4.0 and matter only for artifacts written by 2.3 or earlier. Nothing is accepted silently: a consumer recomputes what it can and refuses a mismatch.

| Artifact or value | Applies to artifacts from | Change | What to do |
| --- | --- | --- | --- |
| Score-table, candidate-selection and checked-AIPW row ids (`row_index`, `estimate_rows`, `screen_rows`, `rows`) | 2.4.x | Eight-byte ids; there is no four-byte alias. The `score_reuse.row_index` digest hashes them. | Recompute the digest from the eight-byte ids. A retained four-byte digest of a nonempty index is refused. `antecedent.release_migration.consume_previous_release_artifact` reads a previous-release artifact (row ids, retained score-row digest, carried target digest, recalculation receipt) and refuses a stale digest with the remedy in the message. |
| Scientific-target digest | 2.4.x | Canonical payload `antecedent.scientific_target.v1` at payload format 3. Worlds and world collections are content-based; caller-chosen ids and the reference flag do not enter. | Recompute from the target. A carried digest that is not that value is refused. |
| Scientific target wire | 2.4.x | Version 2 adds the general linear form. Readers accept version 1 fixed forms and revalidate them. | A version 1 payload cannot carry a linear form. Unknown wire versions refuse. |
| Decision export | 2.4.x | Version 2 envelope; `load_decision` needs current authority. | Re-export from the live workflow. Older joint-law artifacts still replay historically. |
| Reduced retention (summary-only, disk-backed) | 2.4.x | Row reconstruction, refit from retained rows, exact row eviction, in-place column insert and in-place weight or mask update are refused. The refusal reads `{mode} cannot serve {operation}`. | Reload the rows or the disk artifact and run the operation on resident state. |
| Previous route records | 2.4.x | A record is carried onto a current method, kept as a refusal, or retired. | `antecedent.release_migration.consume_previous_route_records` reads a previous-release route artifact and returns one outcome or error per record. |
| Refusal reasons | 2.4.x | None renamed. | Nothing. |
| Decision contract and result, external claim, composition bundle, inverse-query and robust decision results | 2.3 and earlier | New identity domains. | Regenerate: rebuild the contract, evaluate it against the unchanged source distribution, re-bind the external claim, reassemble the bundle, then export again and retain the new identities. |
| Joint-law and decision-source artifacts | 2.3 and earlier | None; byte-compatible. | Nothing. |
| Design ranking request and signal fingerprints | 2.3 and earlier | Fingerprints derived differently. The old artifact decodes and its utilities and ranks can be read, but a consumer refuses it. | Rank the same candidates against the same contract and sources and export again. Values are unchanged; identities differ. |
| Design ranking verification labels and layout | 2.3 and earlier | A stored `exact_request_verified` label is read as `externally_attested` unless you supply the candidate's receipt (`consume_wire_with_receipts`). A newer layout version is refused before decoding. Overlap is decided from the evidence ledger; an undecided lineage refuses as `evidence_overlap_unknown`. | Supply the receipt, or register observations and prior sources in the ledger and use `evaluate_evsi_with_evidence`. |
| Recovery, mixed-source and study-plan artifacts | 2.3 and earlier | Identities use canonical digests. Rule spellings `x10.recovery.v1`, `b2.recovery_chain.v1`, `x9.rules.v1` and `x6.ranking.v1` are no longer accepted; the current spellings are `binary_observation_recovery.v1`, `ordered_response_recovery_chain.v1`, `mixed_source_proof_rules.v1` and `study_plan_ranking.v1`. | Rerun the request with the current producer and export again. Do not edit a spelling and reseal it: the consumer re-decides and refuses. Coverage records keyed by an old derivation identity no longer match, and whole-method calibration stays unmeasured until measured again. |
| Composite recalculation decision stage | 2.3 and earlier | The stage identity includes declared standing inputs. A workflow that declares none keeps its identities. | Nothing is refused. The first recalculation recomputes the decision stage once. |

Existing version-one posterior, dose-grid and cyclic artifacts keep their numerical replay and digest checks. A stored artifact never carries a current claim by being loaded: historical replay does not grant current decision authority.

For the Rust and Python naming of each of these, see the [naming dictionary](api_naming.md); for the artifact families and their loaders, see [artifacts](artifacts.md).
