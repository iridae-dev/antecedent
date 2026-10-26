# Final resource invariant review

Reviewer handoff record. [Integrated verification](validation.md) records the executed results; references below to pending root validation describe the handoff stage.

Owner: foundations reviewer. Scope: planner/plan.rs, coordinated lowering installation, two focused plan-view regressions and proven-unused route-input fields. No Cargo/build/formatters run by reviewer. Root owns serialized verification.

## Finding and disposition

`PhysicalExecutionPlan::quote_route_memory` and helper `count` were unused, but the method was **not obsolete**. Source/history tracing showed it was introduced uncalled. `analysis/lowering/mod.rs::install` only admitted a design-shaped requote (rows, encoded width, expansion, worker scratch), then attached the statistical program without accounting for its declared folds/models/resamples/posterior storage. `RouteMemoryProfile::cross_fitting` / `posterior_draws` had no other facade callers. Runtime bootstrap gather, retained-cache, learner and numerical leases enforce real local allocation constraints; they do not repair an incomplete aggregate plan quote or make inspection accurately report statistical retention. Deleting this method would discard a promised invariant rather than remove a redundant abstraction.

Wired the quote at actual lowered installation, after its statistical/physical contracts are available and before publishing the candidate plan. Installation now stages a clone and commits only after design scan construction and aggregate statistical admission succeed. This keeps rejected installation transactional. The quote is still a comparison/estimate, not a live reservation. Existing runtime leases and cleanup are untouched.

## Corrected live-set semantics before wiring

Blindly enabling the legacy method would multiply training matrices by **all folds × all learners**, even where execution consumes one fold at a time. Reviewed actual callers:

- `antecedent-estimate/src/crossfit_plan.rs::CrossfitPlan::run` fits and consumes folds sequentially. AIPW uses this path; each propensity/outcome learner shares the current fold's training/validation design.
- `antecedent-estimate/src/association.rs` predictive evaluation loops over folds sequentially.
- `antecedent-learn/src/crossfit.rs` portable DML/DR cross-fitting uses `ctx.map_indexed(folds, ...)`, bounded by the available context worker count; the fold result retains predictions, not all training designs.
- Bootstrap gather admission (`estimate/util.rs::admit_gather_scratch`) charges one gather workspace per active worker. Sequential cross-fitting inside a bootstrap therefore has at most the replicate/worker window of live fold designs.

The corrected facade declaration keeps prediction retention and total resampling outputs while replacing the legacy all-fold/all-learner matrix multiplicity with the actual live window. It charges at most a complete train+validation row partition per live fold (one full design), independent of learner count; this also avoids assuming balanced held-out sizes for cluster folds. DML/DR use `min(folds, workers)`; sequential families use one fold, or `min(replicates, workers)` during bootstrap. AIPW includes the two arm-specific outcome channels beside propensity. Posterior coefficient storage covers each declared model rather than one model regardless of the program. Arithmetic remains checked and resource refusal stays typed.

These figures are plan estimates of the declared design/model storage, not measured RSS and not permission to skip executor admission. `copy_count` now describes simultaneous fold design copies; increasing the number of sequential waves alone does not increase resident designs. Statistical passes still count every fold/replicate for alternative costing.

## Specialized admission paths retained

The existing specialized linear operation sizes/adopts its own preparation and geometry through `analysis/compiled_linear.rs::lower` and `analysis/linear_resources.rs`; it holds a real retained admission lease and keeps that quote. The generic statistical quote does not overwrite it.

Streamed reductions remain on their own explicitly admitted `RouteChoice` and runtime reduction budgets. `analysis/streamed_route.rs` admits only supported frequentist static mean reductions, with no graph posterior, tier closure, refuter or custom validator, and no bootstrap for weighting/AIPW. Quoting a dense all-row resident design for a streamed reduction would falsely refuse its intended bounded-memory path. The existing specialized skip remains, with local reduction/gather/learner leases unchanged. No unsupported estimator is silently made streamable.

## Regression coverage added

`crates/antecedent/tests/plan_view_stages.rs`:

1. Bayesian association with 32 vs 512 draws must increase retained-score/draw and total memory estimates. A budget one byte below the complete statistical quote refuses **compile** before physical scan/copy/model-fit and returns no reservation. The complete retained journey then prepares, inspects, estimates, refreshes and exports/independently consumes the result. The plan quote survives the prepared view and result accounting as `Estimate`; actual admitted bytes remain `AdmissionReservation`; artifact native facts exactly match the original result's observed facts. Repeated estimates on unchanged data retain the same scientific terms.
2. Portable DML with five vs ten folds on two workers keeps exactly two live design copies and the same memory quote. Predictive association on the same two-worker context charges one sequential fold, not five retained fold designs. This specifically catches the incorrect legacy all-fold/all-learner multiplicity.

Requested verification: root runs the `plan_view_stages` integration target, existing route resource/streaming/lowering conformance tests and lifecycle/artifact suites after formatting. Reviewer does not claim unrun outcomes.

## Small ownership cleanup

Removed only unused `class: GraphClass` fields from `RouteInput::GcmClass` and `GcmTemporalClass`. All consumers read `Study.graph` or use `..`; no consumer used those duplicate fields. Source graph-class dispatch remains unchanged. `class` remains on cells that actually consume it. This is an internal redundant-state removal with no public API/wire/identity change.

Other compiler-warning files (conditional_admg test import and finite_discrete must_use result) remain assigned to root/facade owner; this reviewer did not touch them.

## Compatibility and measurement implications

No scientific formula, canonical input encoding or artifact schema/version changed. Four explicitly reviewed physical-plan report fixtures changed as detailed below. The statistical quote intentionally becomes larger where previously omitted retention existed; insufficient budgets now produce an earlier resource refusal. Physical plan/accounting estimates can change and must not be represented as new measured memory evidence. Existing serialized artifacts remain readable; observed native facts remain distinct from recomputed plan estimates. Final performance/release owner must use fresh measurement for any peak-memory claim.

## Final validation correction

The first integration run reached the new posterior regression and exposed a fixture error: unweighted Gaussian association uses exact Normal-Inverse-Gamma marginals. Both its lowered declaration and executed posterior intentionally report zero sampled draws, so configured n_draws does not imply retained sampled storage. Do not make this conjugate method allocate or quote fake samples. Corrected only `plan_view_stages.rs` to declare nonuniform sampling weights, selecting the actual Bayesian-bootstrap path; added explicit lowered posterior draw-count assertions (32/512), preserving draw-sensitive memory growth, pre-allocation budget rejection and complete retained estimate/refresh/artifact checks. Root owns rerun; no Cargo/fmt/golden writes by reviewer.

A separate source audit exposed configured DML/DR/AIPW refresh updating estimator spec without re-lowering/requoting the retained plan. This is now fixed through guarded `reconfigure_retained`, matching retention key and compiled-product checks, relowering, requoting and complete resealing inside the candidate transaction. Ordinary `attach` alone previously repeated identification. The final focused regressions passed exact fresh-contract/physical-plan comparisons, zero re-identification and budget/cancellation rollback; see [facade](facade.md) and [integrated verification](validation.md).

## Backend capability and report golden review

Integration rendered four report files (Bayesian effect and unlicensed executing route, Markdown/HTML). Reviewer made no golden edits. Diff review confines changes to physical-plan keys; integration independently removed complete Physical plan sections and found all other bytes identical. Scientific estimates, uncertainty, claims, disclosures and identities remain byte-identical.

Bayesian direct capability is real: `report.rs::bayesian_effect` chooses BayesianGcomp; `execute/checked_bayesian_dag_ate.rs::estimate_point` calls `fitter.prepare`; `estimate/src/bayesian/gcomp.rs:256` calls `prepare::direct_adjustment_design`. That kernel borrows Float64 source slices and gathers selected rows directly into final design and outcome allocations. This is the established direct backend used by linear adjustment; it still materializes its final matrix and does not claim zero copies. The previously materialized-only generic quote failed to expose this implementation. Expected golden changes: selected backend materialized_design→direct_prepared_numeric_views; selected cost7680→4608, corresponding rejection explanations; retained scores/draws0→2304 (96×3×8), output4120→4888, total33080→36152.

A separate genuine defect in the newly activated generic quote was fixed: numerical source eligibility (`DesignShape.direct`) alone does not establish that every estimator implements direct gathering. DML uses `prepare_propensity_problem_with_levels`, gathering each selected covariate into an intermediate Vec before copying it into design; BayesianConditional first builds a base design then replaces it using selected modifier/interaction arrays and CompiledDesign::linear_adjustment. Restricted direct capability to source-traced LinearAdjustmentAte, GlmAdjustment and BayesianGcomp. Other quoted routes retain materialized backend and its expansion charge. Added DML backend assertion to existing live-fold-window regression, retaining live-copy/concurrency assertions. No numerical kernels changed. Planner and test frozen for integration formatting/rerun.

Unlicensed executing route remains materialized. Expected physical diff: direct/lazy alternative quoted cost1200→2000 and work800→1600; streamed quoted cost1225→2825 and work800→2400; materialized quoted/selected cost2000→2800 and work1200→2000. Resident-byte quotes unchanged. Learner-fold bytes0→3200, live copy count0→1; retained scores0→1920; total8010→13130. These are formerly omitted declared fold/prediction storage and work, not scientific or empirical memory changes.

## Strict Clippy bounded core cleanup

Integration's first strict workspace Clippy run found four core issues. Reviewer fixed only assigned core files, no Cargo/rustfmt/commits. `identity.rs::ReusableProduct::required_axes` now shares the identical axis slice for LearnerBundle|PolicyNuisance; the enum variants, discriminants, product-class identity/namespaces, order and axis bytes are unchanged. `route_memory.rs::sum_checked` uses u64::checked_add directly with the same checked-overflow refusal.

`physical_choice.rs::BackendSupport` replaces four booleans with a private closed four-capability bit set keyed by exhaustive PhysicalAlternative matching. Copy, const public constructors/setters/query, Eq and public API remain; with(false) clears only the named capability. A manual Hash implementation feeds the exact previous four boolean fields in the same declared order, preserving caller hash input despite the private representation change. No serialization/identity byte format changes.

`RouteChoice::select` now maps its fixed four-element raw quote array to a typed four-element Result array, destructures it and constructs [direct?,streamed?,materialized?,lazy?]. The avoidable Option::expect panic is removed by construction, with no heap or alternate fallback. Checked arithmetic, alternatives/tie-break order, resource refusal and quote outputs are unchanged. `git diff --check` on the three assigned files passed; root owns formatting/strict Clippy and existing physical-choice/dependency-key regression verification. No blanket allowances, public API additions or golden changes.
