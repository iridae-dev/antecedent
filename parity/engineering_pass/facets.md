Reviewer handoff record. [Integrated verification](validation.md) records executed results; pending root language below describes the handoff stage.

# Engineering pass: calibration facet repair

Owner: foundations reviewer (surface.list only). Date: 2026-10-25.
Initial command: `python3 scripts/calibration_facets.py check`; evidence `/private/tmp/engineering-facets-check.log` (13 diagnostic findings).

This is a dependency-boundary review, not statistical recalibration. No stored scientific numbers, measured SHAs, waiver scopes or checker implementation changed. Changing a surface classification does not itself attest a drifted record. Root still owns calibration status/replay/remeasure dispositions.

## Reviewed classifications

- `analysis/evidence_plan.rs` is **design**, matching the existing repair/planning/design-ranking facades. It publicly re-exports the design crate's plan API and actually executes `failure_routes`, repair search, candidate generation and optional decision-valued ranking in explicit `evidence_plan` / `candidate_templates` methods. An allow entry alone would be wrong for a public re-export and an implementation of the facet. Existing ordinary estimation does not execute these APIs.
- `analysis/execute/class_attribution.rs` and `class_counterfactual.rs` are **mechanism**, rather than broad allowances. These files execute attribution over completion/noise-block members and supplied-DAG/ID* counterfactual member procedures, with uncertainty/publication. `execute/dispatch.rs` selects these paths only for their counterfactual/attribution routes. Existing mechanism keys cover `Counterfactual`, `AnomalyAttribution`, `ChangeAttribution`; Attribute/Simulate/Discover keys remain. A numerical implementation belongs in its own facet, and changing it owes those measurements.

## Reviewed cross-facet references

- `analysis/fallback.rs`: design references are in explicitly requested evidence gathering after an identification-class refusal. Successful ordinary analysis returns before the descriptive/evidence path; candidate generation, structural identification rechecks and planning do not move the original successful estimate. The original refusal is retained. Explicit-name allowance, including new public evidence-plan request/result types when the classified facade introduces them.
- `analysis/execute/dbn_route_posterior.rs` `evaluate`: a locally defined closure for slope-source posterior/replicate projection, not `antecedent_design::evaluate`. This is an over-inclusive lexical-name match; exact name allowed without narrowing the core file's statistical dependency.
- `planner/lowered.rs` `seal`: a local tuple binding from retained identification products, not a design sealing call. Extended existing exact-name allowance (retaining `IdentifiedSet`).
- `analysis/joint_panel/experiment/future_regime.rs`: `DecisionProblem` and `JointPanelResponseFunctional` drive explicit future-study regime mappings/ranking under `PanelExperimentResult`; all corresponding records carry design via existing `*JointPanel*` / `*joint_panel*` keys. Kept core numerical path and exact design names.
- `analysis/joint_panel/streamed.rs`: `functional_weights` from `ports` and `JointPanelResponseFunctional` are called only by explicit joint-panel stream projection methods. These are real executable dependencies, covered by the same existing JointPanel keys; not mere DTO references. Exact allowance retains conservative core ownership of stream projection.
- `analysis/propagation_chain.rs`: shared dependence constructors/checkers are actually consumed by the explicit chain workflow, which has its own compiled `PropagationChain` route and refuses unmeasured interval claims. Extended exact evidence allowance to `DependenceContract`, `DependenceError`, `DependenceSource`, `SharedFactorSets`; retained previous `SharedFactorKind`/`dependence` names. Added the actual compiled route's `PropagationChain*` query key for future measurements rather than suggest ordinary effects execute chain dependence checks. No current record has this query, and no measured evidence is fabricated.
- `analysis/propagation_chain_family_tests.rs`: `SharedFactorKind` is family regression scaffolding. Explicit test-file name allowance; source remains covered rather than exempted from the surface.
- `analysis/observation_query.rs`: shared composed calibration basis and resolver are used by observation **parameter intervals**, after deriving a `ComposedCoordinate` and requesting spread; the basis explicitly uses `COMPOSED_QUERY = "ComposedQuantity"`. Those records already carry `joint_law_composed`; ordinary observation point calculation never resolves this key. Exact allowance covers `COMPOSED_LEVEL`, `Resolution`, `composed_basis`, `joint_law_composed_basis`, `resolve`.
- `analysis/functional_sensitivity.rs`: `antecedent_attribution::ChangeAttributionResult` is a type annotation on a closure extracting an existing component's contribution for comparison with a refit, not an engine/fit dispatch. The closure reads public result fields. Exact type/module names allowed; core ownership of sensitivity remains.

## Stale allowance

Removed `observation_uncertainty` from the mechanism allowance of `io-core/observation_query_artifact.rs`; the file no longer references that module spelling. All other verified mechanism constructor/function/type references retained. No wildcards or blanket allowances added.

## Validation

Recheck in progress: `/private/tmp/engineering-facets-recheck.log`. Final outcome and any subsequent reviewed cascade additions will be appended. No Cargo/build commands executed by reviewer.

## Follow-through after owning-file classifications

The second full check surfaced the expected exact-name edges caused by newly assigning implementation files to their owners. Reviewed each before extension:

- `analysis/mod.rs` only declares the `evidence_plan` public module; added that name to its existing design export allowance.
- `analysis/execute/mod.rs` only declares the two class implementation modules; added their exact module names to its existing mechanism export allowance.
- Fallback imports `EvidencePlanRequest`, `EvidencePlanResult`, `STUDY_POPULATION`, `plan_over_routes`, `variable_names` from the newly classified plan module and names `evidence_plan`. These are the same opt-in post-refusal planning path reviewed above; extended its exact names.
- `evidence_plan::variable_names` is a newly facet-owned function whose generic spelling matches local parameters/bindings and artifact struct-field initializers in eleven unrelated files. `rg` inspection of **every occurrence** confirms no import or call to the design function: admg_conditional_transport, counterfactual_transport, learned_continuous, mixed_source, mz_transport, recalc_static, recovery, recovery_chain, smoothed_dose, counterfactual_id/artifact and cross_world/artifact. Added one exact `variable_names` allowance for each, leaving their existing statistical ownership intact. This is the checker's deliberate lexical overapproximation, not a new dependency.

The checker rejected an initially duplicated `planner/lowered.rs` allow row immediately; combined the local `seal` name into the existing `IdentifiedSet` row. No duplicate entry retained.

Final full recheck is running with output `/private/tmp/engineering-facets-final.log`.

**Final result: PASS, exit 0.** `calibration surface: 447 paths in 222 facets; boundaries hold; 14 replay waiver(s) valid; list blob 5a08ab5a487e`. Six existing inert-waiver notes remain informational; no waiver edited/deleted by reviewer. Output retained in `/private/tmp/engineering-facets-final.log`. Surface file frozen and ready for root's incremental commit.

## Final facade implementation coverage

After facade owner froze production and regression edits, a new full check started against the completed family-owned retained operations (`/private/tmp/engineering-facets-final-source.log`). Explicitly verified `load_surface().facet_of(...)` resolves both new files, `analysis/execute/checked_family_routes.rs` and `checked_static_cross_world.rs`, to **core** via the existing `core crates/antecedent/src/` rule. They are neither unlisted nor exempted. Checked-family invariants shared in `checked_route.rs` also remain core. Numerical module ownership continues to distinguish mechanism class implementations from shared cross-world retention/dispatch invariants; no new narrow facet is invented for the shared files.

**Final completed-source result: PASS, exit 0.** Full check after facade production/regression freeze again reports `447 paths in 222 facets; boundaries hold; 14 replay waiver(s) valid; list blob 5a08ab5a487e`. Evidence: `/private/tmp/engineering-facets-final-source.log`. The six existing inert-waiver notes are informational and unchanged. No source/waiver/record checker weakening was needed after the concrete family operations landed.

## Final configured-refresh lexical collision

Final full source check found one new lexical collision only: `analysis/lowering/mod.rs` token `seal`. Reviewed every occurrence (local binding and comparison in reconfigure_retained). It is produced exclusively by `analysis::identification_seal::seal_static(&cache.identification,&cache.estimand)` and compared with the retained lowered static proof digest before re-lowering numerical options. It never names/calls design rollout or sequential-evaluation artifact seal functions. The lowering module remains core under `core crates/antecedent/src/`; relevant implementation edits still invalidate all core-dependent records. Added only `allow design crates/antecedent/src/analysis/lowering/mod.rs seal` with explanatory comment. No owner shift, facet-key change, waiver SHA or record/attestation edit. Root owns rerunning the full checker; reviewer ran no heavy check/Cargo/formatter/commit.

## Frozen context-source validation

Integration’s full check after all caller-context and private phase refactors
passed: 447 paths, 222 facets, boundaries hold, 14 valid replay waivers, surface
list blob `303eecbe28af`. Six existing inert-waiver notes remain unchanged. No
waiver SHA or measurement record changed. The earlier root rerun obligation is
fulfilled by this check; see [integrated verification](validation.md).
