# Prepared recalculation capability contract

`antecedent.recalc_capabilities.capability_matrix()` exposes seventy immutable cells: six required native analysis families, one separately scoped external callback family, and ten operations. Each cell names retained objects, complete identity requirements mapped to the existing shared planner's `Stage` inputs, compatible operations, actual Python/Rust adapter entry points, and a typed refusal. `capability(family, operation)` selects one cell. This inventory does not create an estimator, widen a causal license or certify a cache hit.

`require_adapter(family, operation, state)` selects an existing adapter from inspected native-backed retained state, including live checked adjusted fits and DML/DR-Learner state. An empty session, readable receipt/result, arbitrary callback, fabricated predictor wrapper, or caller-set `Capabilities`/`ResumeContext` flag supplies no reusable state. Missing family adapters raise `RecalcUnavailable` with registered reason `route_not_supported`, detail `recalc.capability_adapter_unavailable`, and the missing score/fit stage. Selection does not replace the adapter's request validation, instrumented execution or compatibility checks.

| Family | Utility | Contrast/functional | Target weights/law | Action grid | New rows/outcomes | Graph/query/regime | Provider/prior | Learner/folds/RNG | Inference settings | Fresh process |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Adjusted regression | live fit | supported mean contrasts | same-row fixed weights | supported contrasts; coding changes refit | supplied data/refit; point prediction | recheck identification/refit | unavailable | fit settings/RNG invalidate | covariance settings refit; no new guarantee | supplied data/refit; fit absent refuses |
| Doubly robust effects | live/scores | live/refit | live/scores, same-row weights only | live/refit, licensed cells only | live/refit; predictor point prediction | live/re-identify | unavailable | live/refit, declared folds/RNG only | unavailable | verified scores or predictor |
| Static response/transport | live law | supported means/contrast | checked source laws | supported grid | supplied data/factor refresh | checked identification | exact finite providers | unavailable | unavailable | verified artifact or supplied data |
| External mean callback | no standalone adapter; combined decision reuses means | unavailable | unavailable | changed declared grid revalidates and invokes provider | declared inputs invoke affected provider | changed program rebinds | provider replacement requires actual compatible invocation | declared parameters/environment/RNG invoke provider | unavailable; no inference upgrade | supplied compatible provider and actual replay |
| Bayesian analysis | retained draws | retained aligned effect summaries | unavailable | unavailable | supplied data/refit | checked identification/refit | verified native source capsule | model/config/seed refit | draw configuration refit; unmeasured | supplied raw data and verified source replay |
| Temporal analysis | retained point law | supported two-step response/effect | supplied target initial law | supported binary sequences | whole units/refit | checked source ID/refit | actual observed source factors | source identities/seed invalidate | dependent interval closed | complete raw-unit artifact replay |
| Design-specific effects | live estimate | checked model summary | unavailable | unavailable | supplied data/refit | recheck identification | unavailable | unavailable | unmeasured diagnostics | checked supplied-data replay |

“Unavailable” refers to the family adapter in the shared selective workflow. Existing prepared `.estimate` paths and separately licensed analyses remain available at their own coordinates; they do not establish retained-fit reuse through this planner. Bayesian and temporal adapters now execute their declared bounded coordinates below; wider inference and update routes retain separate gates. The doubly robust family now includes cross-fit/cell-AIPW and the checked DML-AIPW/DR-Learner adapter described below. These minimum operations do not license arbitrary learners, score quantities or target populations.

`recalc_adjusted.AdjustedSession.execute` runs linear/GLM adjustment and vector/categorical treatment contrasts through the same `RecalcPlan` and verified receipt. The existing identification crate checks a sufficient joint backdoor certificate for the declared common adjustment set. The existing joint OLS/dummy regression engines and shared GLM solve/Fisher covariance primitives retain immutable coefficients, covariance and design. Supported contrast, same-row fixed target weights and utility changes reuse the fit; changed data/outcomes, roles/graph/adjustment, coding/reference, link, covariance, fit options or RNG invalidate the appropriate stages. Target weights may depend only on checked adjustment roles. The covariance is conditional on that declared target row law; it does not add sampling variance for an independently estimated target population.

Numeric mean contrasts stay within each treatment's observed range, with exact observed binary states; categorical contrasts require populated declared levels and reference coding. These are parametric response-model operations under declared model validity. A vector's per-coordinate support does not establish joint or conditional positivity, and the adapter supplies no outcome joint law. Unsupported actions, sparse levels, unidentifiable adjustment and incompatible schemas produce typed refusals. Full-rerun checks include both independent finite algebra and the original scalar study engines.

`AdjustedSession.predict` validates feature names/order against the actual native fit and applies the same treatment-support boundary. Its point-prediction result measures zero model fits; prediction is a separate scoped consumer of the retained model, since the shared planner has no prediction stage. It supplies no prediction interval or residual/sampling state. Reuse shares immutable fitted data instead of copying the full design, and failed refits leave the previous successful session usable.

`receipt.totals.model_fits` counts successful numerical model solves separately from nuisance `fold_fits`; both receipt identity and independent artifact checks bind it. Historical zero-model receipts keep their original wire and digest. An exported adjusted receipt contains no fitted model or training data. `AdjustedSession.resume` rejects portable-fit/score flags as substitutes; explicit supplied-data mode plus an actual raw request refits in a fresh process. A missing fit refuses prediction. The Python bridge bounds requests to 100,000 rows, 256 columns and 1,000,000 values, and GLM fitting to 1,000 iterations; the native executor also checks its cancellation and workspace budget before fitting.

`recalc_dr.DrSession.execute` retains checked DML AIPW scores or DR-Learner marginal scores and its distinct final CATE model through the existing prepared-study engine and shared planner. The minimum accepts untrimmed, all-observed IID binary treatment with linear/ridge outcome and final models and linear/ridge/logistic treatment learners. Partially linear DML, trimming, automatic learner search and other learners refuse. Complete-case row IDs, graph/query, data, learner configuration, folds and seed bind retained state. Utility-only changes recompute the decision; compatible same-row target weights reuse scores and may depend only on checked adjustment roles. Weights align to the retained complete-case rows. Changed data, causal roles or nuisance configuration recheck or refit.

`DrSession.predict` consumes the retained CATE model with named, ordered features and measures zero fits. Live prediction stays within each feature's observed range; this per-coordinate boundary does not establish joint or conditional positivity. DML scores do not provide a predictor. Predictions and analytic IID score diagnostics remain point-only/unmeasured and supply no validated confidence or prediction interval.

Receipts measure successful nuisance fits, final CATE fits and actual score-table builds independently. Initial preparation and estimation build two score tables while sharing physical nuisance inputs within a bounded execution cache; they do not fit nuisances twice. Cache keys include fold units, kind and seed, and retention shares the existing value budget with at most eight prepared bindings. Cancellation, incompatible requests and failed refits preserve the prior successful session. The Python bridge bounds requests to 100,000 rows, 256 columns, 1,000,000 values and 20 folds; native refitting checks cancellation and its workspace budget.

`DrSession.export_scores` produces the existing independently verified frozen-score object for same-row retargeting, while `export_predictor` produces the existing parent analysis artifact for CATE point prediction. Export retains the producing execution identity. Neither object substitutes for the other. Receipt-only resume has no executable state; an explicit supplied raw-data request refits in a fresh process. Fresh-process tests execute both DML and CATE frozen-score retargeting without fits, and independently decode and predict from the CATE artifact.

The live score adapters are `RecalcSession.execute`, `CrossfitSession.execute`, `CellSession.execute` and `DrSession.execute`. The first two delegate to Rust `analysis::recalc_receipt::execute_with_receipt`; the cell and DR adapters use `analysis::recalc_cell::execute_cell_with_receipt` and `analysis::recalc_dr::execute_dr_with_receipt`, respectively. Full supplied data and native-derived identities are required. Utility-only changes reuse the law and recompute the decision; compatible same-row target weights reuse frozen scores. Supported changed graphs/queries re-identify; changed data, outcomes, folds, RNG or cell contrast refit invalidated scores. Off-grid actions, arbitrary new learners and incompatible targets still refuse at the executing adapter.

The portable score route is `resume_from_scores` followed by `ScoreResumeSession.retarget`, backed by Rust's verified frozen-score consumer. Scores retain row identities/order, snapshot, score quantity, folds, treatment grid and support; they support the declared same-row weighting and utility operation. They contain no fitted predictor, new outcomes, training factory or raw data. Changed data, graph, query, regime or folds cannot be resumed from scores.

`FittedEffectModel.load/export/predict` independently verifies the parent analysis artifact and retained predictor. Its Rust map is `antecedent_estimate::FittedEffect::predict`; consumption uses `antecedent_io::decode_analysis_result_artifact`. The model supports point CATE prediction on new feature rows matching its variable/basis schema, provider/fit identity and parent claim. It contains no training data or refit factory, and prediction does not recreate residuals or sampling covariance. It is a portable prediction route, not a complete shared `RecalcPlan` family adapter. A predictor cannot serve as an AIPW score retarget object.

Readable posterior artifacts, aligned draws and executable providers are distinct retained kinds. `recalc_bayesian.BayesianSession` executes the original checked Gaussian or quadratic-basis ATE engine and retains its actual aligned posterior rows. Compatible summaries and utility changes perform no new solves or draws. Changed prior, likelihood, basis, data or draw configuration invalidates the required numerical state. `posterior_draws` counts rows actually emitted by the numerical engine, alongside its successful model solves; zero additions preserve historical receipt identities. Posterior summaries remain unmeasured diagnostics and grant no new posterior-decision or interval guarantee.

`BayesianSession.export_prior_source()` exports a bounded capsule containing the actual source raw data, graph/query/configuration and native posterior artifact. A target verifies it by independently reexecuting the source and comparing the complete posterior. Same verified full source/likelihood content refuses before fitting; a compatible live verified capsule can be retained without replaying it on each summary. This does not detect partial evidence overlap without unit identifiers, license arbitrary external priors, or treat conditioning on new data as a universally valid update. The separately checked joint transport candidate remains behind `calibration-internal`; its calibrated public route stays closed.

`recalc_temporal.TemporalSession` executes source observational ID and empirical whole-history factors for the five-variable binary DAG S0,A1,L2,A2,Y. Only selection into the unconfounded root S0 is admitted; remaining mechanisms are invariant. The native source checker identifies joint (S0,Y) under do(A1,A2), actual observational leaves are bound to the real source regime, and positive source S0 margins permit mixing conditional responses over the supplied target S0 law. Actual source proof/program and target-law origins are preserved; no experimental source evidence is invented. Compatible utility and target initial-law changes reuse source state. New sequence/data refits affected factors; unsupported horizons, selections, latent structure and histories refuse. Complete-unit artifacts independently rerun actual source ID and factors in a fresh process. Whole-unit replicate identity is preserved by the original checked-source interval receipt. `TemporalSession.dependent_interval` returns `MeasuredInference` for the guarded balanced complete-history 95% protocol: response uses studentized intervals; paired effects admit studentized, percentile and basic intervals under their separate method records. Its independent consumer replays the original checked source identification and factors, rather than substituting the direct-panel artifact. Failed response percentile/basic methods are superseded by studentized response intervals and retired; their measurements remain historical evidence. Passing paired-effect percentile/basic methods remain supported. Unsupported protocols refuse.

Every operation declares the full graph, scientific query/quantity/scale/horizon, regime, identification/evidence standing, source and target populations, data/schema/row identities, sampling/dependence design, action support, learner/provider/basis/fold/RNG/inference settings and utility identity. All eight bounded external branches additionally name snapshots and datum IDs, source/target/query/regime, exact provider request, prior construction and evidence consumption, factor/proof leaves, draw alignment and inference standing. These semantic fields must be preserved or invalidated as appropriate; the inventory does not assert that a currently unavailable variation has already been bound by an executor.

Family-specific identity requirements include linear/GLM links, treatment levels and coefficient covariance; AIPW/DML/CATE score coordinates and positivity; transport factor/source proof maps and search completeness; Bayesian likelihood/prior/basis/noise and aligned draws; temporal unit/time/lag/history/initial-state/horizon and whole-unit resampling; IV instruments/exclusion/strength, RD cutoff/bandwidth/support, and front-door mediator positivity. Adjacent regression code or a neighboring inference record supplies no missing causal or uncertainty license.

Tests execute native score retention and portable-score resume with fit counts, independent reruns and a constant-effect SCM oracle, verify actual adapter paths, and reject readable/flag-only state across all seventy cells. Portable predictor tests verify prediction without turning the model into score or covariance state. No statistical calibration is performed by this inventory or its tests.

`recalc_static.StaticResponseSession` executes checked finite ADMG response grids and selects supported means and contrasts. This adapter requires a checked ADMG with bidirected structure and at least two strictly increasing support points; unsupported DAG requests receive `recalc.static_graph_unsupported`. Conservative workspace checks precede provider construction, and each CPT is capped at one million Cartesian cells before allocation. Retained CPTs bind the complete-case row identities and each factor's actual columns. Supported action changes reuse compatible factors and compiled functional programs; changes to support, graph, query or source premises require checking the affected work. `MultiSourceSession` executes the existing bounded multi-source transport engine against an evidence catalog and exact discrete source laws. Each cached factor binds its population, regime, snapshot, full source-law identity and proof leaf. Missing evidence and incomplete search remain scientific refusals. An absent exact provider is a missing-provider failure and cannot be skipped as a resampling support failure. Neither adapter synthesizes a joint law from incompatible sources.

Static receipts expose actual `factor_builds`, `program_compilations`, `provider_bindings`, `factor_evaluations`, `integrations` and `provider_calls`. The independent receipt consumer checks their stage ownership and binds them into the receipt identity. Historical receipts omit zero additions and retain their digest. Original response/transport artifacts preserve their existing independent consumer contracts. Receipt-only state contains no data or executable provider; explicit supplied-data resume runs the checked engine. Static results provide finite-law point means and contrasts, with no new interval license.

`recalc_design.DesignSession` runs the existing checked IV, sharp RD and linear front-door engines. Instrument/exclusion, running variable/cutoff/bandwidth and mediator declarations bind their design-specific causal preparation. Changed data refits; compatible utility changes reuse the immutable estimate. Successful least-squares observations count actual model solves, including IV and Anderson–Rubin auxiliary work, and `law_summaries` identifies extraction of a point from the checked estimate. This is distinct from score reweighting. Weak-IV requests retain the attempted point and actual Anderson–Rubin diagnostics while refusing an unavailable decision. Analytic covariance diagnostics remain unmeasured.

`recalc_design.consume_design_result` requires supplied raw data and independently executes checked preparation and estimation. It compares the original artifact's full contract and scientific body, preserving the producing seed and refusing changed data or model declarations even if they happen to give the same point estimate. The returned session contains fresh executed state and a new receipt. Existing historical readers remain unchanged; missing raw data supplies no RD checked-operation license.

`StaticResponseSession.response` exposes the actual retained checked response with no new fitting or factor work. `ProgramBinding.from_response` derives the scientific binding from its original identification; `native_claim` requires opaque producing execution authority and verifies the full projection, canonical declared premises, snapshot and RNG. Caller-created result views or labelled distributions cannot mint native trust. Actual aligned native posterior rows survive when their producer supplies them; a mean-only response remains mean-only. Historical labelled distributions remain readable and unverified at composition.

## Bounded foreign mean callback execution

`recalc_external.ExternalCallbackSession` invokes a real `CallbackProvider` against
an exact `ProgramBinding` and external response contract. Named input columns,
model parameters, provider implementation/version/environment, replay policy and
seed are explicit dependencies. Compatible deterministic or seeded output can be
retained; stateful output is recomputed. Unknown policy is refused. Planning does
not invoke a callback.

Returned values pass the original external response binder and remain externally
attested. Actual provider entries are counted as `external_invocations` on their
original branch; failed entries retain attempt evidence without replacing the last
successful scientific output. A side-effect retry needs its declared idempotency
policy. A known idempotency key is bound to its first full request and cannot be
reused for changed inputs.

Cancellation is cooperative at invocation boundaries and through the supplied
token; arbitrary Python code is not forcibly interrupted. The bounded portable
output carries the original attested claim and request/provider identity, rather
than a serialized callable. A fresh session needs a real compatible provider and
successful numerical replay before it can issue live attested output. Stateful or
side-effecting portable replay is refused. Nested law/posterior callbacks remain a
separate expansion package in the roadmap.

The original binary cross-fit AIPW session also retains its last successful prepared state through failed refresh, target weighting and decision evaluation. Unchanged or compatible law/utility reuse shares that state; a changed-data refresh prepares a separate candidate and publishes it only after all stages and receipt validation succeed. Its `fold_fits` unit is a successful nuisance fold task (the outcome task fits both arm models), so this count differs from the lower-level linear solves performed inside IRLS.

An actually executed `CompositeSession` can pass its uniquely selected action to
`rank_studies(ConditionalStudyPolicy(...))`. Every terminal action needs an explicit
bounded candidate table. The original structural ranking orders the selected table
by the supplied sufficiency declaration, cost, sample budget and identity. The
projection preserves that declaration; it does not establish sufficiency or EVSI.

`ConditionalStudyRanking.consume(bytes, replayed_session)` checks the full actual
native response, both issued foreign claims, source order, complete conditional
rule and original shared receipt, then reruns the original ranking. A fresh process
must first execute the raw native request and supplied callbacks. Imported bytes
alone cannot supply that scientific state. Mutating any selected source can change
the terminal action and therefore the candidate table; unchanged ranking performs
no provider calls or native fitting.

`antecedent.execution_attempt.observe_native_attempts(lambda: session.execute(request))`
invokes the original callable once and returns its unchanged value or original exception
alongside actual component attempts, completions, failures and unfinished work. `unwrap()`
returns that value or re-raises the same exception with its original traceback. Nested
observers see each actual source operation once, including original joined worker calls
that propagate the observation token. Complete scientific reuse and preflight cancellation
enter no component and report zero work; in-component failure/cancellation is unsuccessful
work. The original family retains its own transaction, cancellation and callback policies.

These are diagnostic observations of 18 fixed original component kinds. They do not
monitor unrelated threads, remote internals or uninstrumented Python, retry callbacks,
supply resumable state or issue a scientific receipt. A completed identification invocation
can still report an unidentified answer; a numerical solve and its enclosing model are
separate levels and must not be summed into a fit count. Cached failure reads are
separate from new numerical solves. Original source support, attestation and uncertainty
standing remain unchanged.
Tests execute native score retention and portable-score resume with fit counts, independent reruns and a constant-effect SCM oracle, verify actual adapter paths, and reject readable/flag-only state across all sixty cells. Portable predictor tests verify prediction without turning the model into score or covariance state. No statistical calibration is performed by this inventory or its tests.

## Cell-AIPW scores: frozen scores and portable resume

`antecedent.recalc_cell` (`python/tests/test_recalc_cell.py`) extends the same plan and receipt to the cell-saturated AIPW route over discrete joint binary treatments. `CellSession` mirrors `RecalcSession`: the first `execute` fits the cell models (5 folds x (1 multinomial propensity + 4 cell outcome regressions) = 25 `fold_fits` in the tested fixture); a utility-only change reuses the law and recomputes the decision with `fold_fits == 0`, and a compatible same-row target-weight change reweights the frozen cell scores. The count comes from the estimator's own instrument, so a zero is measured, not asserted. A changed outcome, fold seed, graph, adjustment set or data refits. `CrossfitSession` does the same for cross-fit scores.

`CellSession.export_frozen_scores()` (and `CrossfitSession.export_frozen_scores()`) return `FrozenScores`: a checksummed `frozen_scores_v1` artifact holding the scores and the producing workflow's stage digests, with no data and no model. Its `identity` is 64 hex characters; retain it independently of the bytes. `FrozenScores.load(data, expected_identity=...)` and `resume_from_scores(data, variables=, edges=, utility=, expected_identity=)` refuse a consistently resealed artifact of another run (`frozen_scores.identity_mismatch`) and raise `CausalSerializationError` for corrupt, truncated or unsupported-version bytes. Because the artifact does not hold the original workflow, resume needs `variables=` (column names in order), `edges=` and `utility=`.

```python
session = recalc_cell.CellSession()
session.execute(request, seed=7)
frozen = session.export_frozen_scores()

# Possibly in a fresh interpreter, from the bytes and the retained identity alone:
resumed = recalc_cell.resume_from_scores(
    frozen.export(), variables=VARIABLES, edges=EDGES, utility=utility,
    expected_identity=frozen.identity,
)
out = resumed.retarget(TargetWeights(weights, ("z",)), row_ids=resumed.row_ids)
out.receipt.totals.as_tuple()      # zero fits; only the law and decision are recomputed
```

`ScoreResumeSession.retarget` is the one licensed operation: reweight the frozen scores by non-negative row weights in the frozen row order (a `TargetWeights` may name `depends_on` variables inside the adjustment set) and recompute the law and decision. An unchanged request reproduces the first run's `ate`, `std_error` and `net_benefit`. In a fresh process the receipt shows `score_artifact` as `reused` and every other derived stage recomputed (for example `recomputed(fresh_process)`), since a fresh process never reuses a derived stage but the portable scores.

Anything that needs the data or a fit is refused before any work and leaves the session unchanged: a changed outcome, folds, graph, row design, data snapshot or treatment grid, and `execute` over new data. These raise `ScoreResumeUnavailable` (`recalc.unavailable_data`, `missing == "data"`, `reason_code == "score_table_unavailable"`); weights over other row ids or of another length, or a declared derived stage, raise `ScoreResumeRefusal`. Both are structured refusals (see [Refusal and partial knowledge](refusal-and-partial-knowledge.md#structured-refusals-and-their-remedies)). A `ResumeReceipt` is a verified record, not an exportable artifact: the `recalc_receipt_v1` format refuses a derived stage reused in a fresh process, so `ResumeReceipt.export()` raises.

Runnable example: [`recalc_cell_resume.py`](../examples/python/recalc_cell_resume.py).
