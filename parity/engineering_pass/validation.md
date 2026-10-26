# Engineering validation record

Integration owns every command below. No reviewer ran a competing build.
Default ignored measurement/acceptance tests remain ignored; no new ignore or
weakening of an assertion is used to close this pass. Full reference-host
calibration/performance phases remain required separately.

## Completed lower-crate tests

All commands use `--locked`, `CARGO_BUILD_JOBS=4` and four test threads.

| Command | Result |
| --- | --- |
| `cargo test -p antecedent-core -p antecedent-data -p antecedent-graph -p antecedent-state -p antecedent-prob -p antecedent-learn -p antecedent-attribution -p antecedent-io-core --lib` | 1,786 passed; no failures or ignored tests. |
| `cargo test -p antecedent-identify -p antecedent-model -p antecedent-validate --lib` | 908 passed; 7 existing ignored measurement/exhaustive tests. |
| `cargo test -p antecedent-estimate -p antecedent-learn-burn --lib` | 1,320 passed; 48 existing ignored measurement/acceptance tests. |

These cover canonical content identity, graph surgery consumers, actual admission
and lease cleanup, bounded spill/corruption/replacement, portable predictors,
posterior/solver boundaries, fit-cache retention, type-7 sensitivity quantiles,
and deterministic neural recovery. The later custom Deserialize implementation
and artifact boundary correction require final integration verification below;
the earlier learner result alone is not claimed to validate later source.

## Initial broad facade run

`CARGO_BUILD_JOBS=4 cargo nextest run --locked -p antecedent --profile ci
--test-threads 4 --no-tests fail` ran all 4,283 selected ordinary tests:
4,250 passed, 33 failed; 551 existing ignored tests skipped. Duration 958.270s.
The 48,000-draw panel workflow passed in 777.904s, including its actual fit,
retained reuse, diagnostics and subsequent workflows.

This run deliberately exposed integration defects rather than providing a green
completion result: 24 failures concerned one-shot static cross-world admission,
configuration test setup, cancelled refresh publication and typed smoothed-dose
artifact refusal. Those fixes and their regressions are subject to the final
rerun below. Nine failures concern preexisting missing measured-license
attestation; [their exact disposition](license_disposition.md) preserves all
assertions and measured records. Initial source licensing was audited without
claiming an initial numerical run.

## Broad integration and final focused rerun

The next broad run added `-p antecedent-io -p antecedent-contract -p antecedent-expr
-p antecedent-learn` to the facade command and used
`-E 'not test(=joint_panel_workflow::actual_bayesian_panel_fit_preserves_mixing_diagnostics_and_shared_draw_axes)'`.
That one already-passed 777.904s workflow was not repeated after unrelated
fixture/resource/refresh changes. No measurement assertion was filtered out.
The run executed **5,422 tests across 191 binaries: 5,406 passed, 16 failed,
568 skipped** (567 existing ignored tests plus that one prevalidated test).
Duration 578.844s.

Eleven failures require already-missing measured-license attestation, including
two additional independent I/O consumption assertions. Their exact records and
unreached later assertions are in the license disposition. Five software checks
required correction: an identifiable executing PAG fixture, inference-domain
configuration assertions, a truly sampled posterior fixture, explicit report
physical-plan updates, and separately exercising compile-time and runtime
bootstrap admission. The follow-up also exposed actual stale configured-refresh
plans and duplicate resolved options; those production defects are fixed.

The final focused command retains the same five packages with `--lib` and
integration targets `plan_view_stages`, `temporal_point_uncertainty`, `report`,
`graph_posterior_cross_world_routes`, `path_specific_admg_route`,
`path_specific_pag_route`, `prepared_analysis`, `prepared_refresh_fits`,
`prepared_rebind_custom_validators`, and `prepared_retained_refresh`.
It executed **1,394 tests: 1,393 passed, one failed, 10 existing ignored tests
skipped** in 53.272s. The only failure is the unchanged I/O measured-transport
attestation assertion. All selected software assertions passed, including full
fresh contract/plan/quote equivalence, post-install and budget rollback, no
re-identification, exact typed artifact refusals and report comparisons.

The report producer was run deliberately with `ANTECEDENT_REGENERATE_REPORTS=1`
only for rendering and diff review. Subsequent comparisons ran without that
variable. Exactly four report fixtures changed; stripping their Physical plan
sections leaves every remaining byte identical to the pre-pass fixtures. The
resource record explains actual implementation capability and exact estimate
deltas. No numerical/reference, artifact or scientific-identity golden changed.

## Final follow-up verification

`cargo test --locked -p antecedent-core -p antecedent-data -p antecedent-state
-p antecedent-learn-burn -p antecedent-contract --lib -- --test-threads=4`
passed all **993 tests**, with no ignored tests or failures, after the bounded
Clippy changes to capability representation, memory arithmetic, IPC iteration,
weighted sufficient-statistic updates, neural fallback and bridge validation.

`cargo test --locked -p antecedent-learn -p antecedent-discovery --lib`
passed **408 tests**, with 10 existing ignored cases. The subsequent identify,
model and attribution library run passed **773 tests**, with 3 existing ignored
cases, after the private search/certification helpers and slot-borrowing cleanup.

The estimate and I/O-core library follow-up passed **1,809 tests**, with 48
existing ignored cases. Design passed **168 tests** with none ignored. These runs
predate the final context corrections and are not claimed to execute their new
regressions. The later targeted `serial_dependence::tests` run passed **23 tests**,
including actual Bayesian-fit policy and warm-entry bypass cases. Validation’s
ordered-calculation follow-up passed **229 tests**, with 4 existing ignored cases,
before its additive typed-resource variant.

`python3 scripts/check_conformance_references.py` passed all eight independent
reference generators in check mode. No reference fixture was rewritten.
The Python surface (461 entries), generated host schema, architecture gate,
retained-route gate and changed-file Ruff checks also passed. The retained gate
self-test separately confirmed zero baseline findings and detection of all 34
injected violations; all 15 dependency-facet self-tests passed. Architecture
negative controls and native unit-test registration self-tests passed.

Source formatting uses `cargo fmt -p` for affected crates, as required by
CONTRIBUTING. Unrelated preexisting formatting differences in other files are
restored after each invocation; they are not represented as a full-workspace
format check. Q13 still owns the completed release-wide formatting gate.

## Caller-context follow-up

The current graph/design library run passed **215 graph tests** and **168 design
tests**, with none ignored. Identification passed 460 tests and caught one new
regression-test defect: comparing complete Debug output included randomized
hash-cons lookup-map order. The replacement compares canonical marked graph
digests, aligned indexers, every public scientific result field, ordered arena
nodes/derivations and interned tables; cancellation and tiny-memory assertions
remain. Its corrected execution passed in the final search library run below.

`cargo test --locked -p antecedent-design --test suite_01
design_ranking_artifact:: -- --test-threads=4` passed **17 tests**, none ignored,
including original-law ownership, exactly-once provider preparation, contextual
retention and cancellation/mismatched-report refusals. The current validation
library run passed **229 tests**, with 4 existing ignored cases, after its typed
resource variant.

The final retained-route negative controls pass with zero baseline findings and
all **34 injected violations** detected; compiler-bypass controls detect all
**10 reopenings**. Current architecture and test-target negative controls pass
5 cases each; registry self-tests pass all 14. The registry contains 1,529
compiled test paths. Python facade inventory passes 462 entries (342 bound,
120 internal), and generated host/family/route/acceptance records are current.
GCM conformance and acceptance manifests still pass 19 clauses/60 tests/3
benchmarks and 430 bound cases respectively. These source checks do not claim
execution of measured scientific licenses.

## Strict library and dependency validation

`PYO3_PYTHON=python/.venv/bin/python CARGO_BUILD_JOBS=4 cargo clippy --locked
--workspace --lib -- -D warnings` passed on the integrated source, including native
bindings. Meaningful private validation/publication/accounting helpers replace
large mixed responsibilities; narrowly scoped compatibility/ABI allowances carry
explicit reasons. No blanket warning suppression was added. The lint-allowance
gate and `git diff --check` pass.

The frozen-source calibration facet check passes **447 paths in 222 facets**,
with boundaries holding and **14 valid replay waivers**, surface list blob
`303eecbe28af`. Its six preexisting inert-waiver notes are unchanged. No waiver,
coverage attestation or measured result is rewritten by this pass.

## Fresh public API verification

The public API gate generated fresh rustdoc JSON for every published library
crate. Review found exactly **24 additive snapshot lines in five crates**, with
no existing public line removed or changed. The scoped snapshot update was then
checked against the same fresh output across every published crate: **PASS**.
[The API record](public_api.md) lists each addition, caller and compatibility
choice. The Python facade surface passes again against these snapshots.

## Expanded context/lifecycle integration

The nine-package nextest run (`antecedent`, I/O, contract, expression, learner,
design, identify, validate and graph) executed **7,510 tests across 213 binaries**:
**7,495 passed, 15 failed, 656 skipped**, in **1,521.666s** after a 7m08s build.
The skips comprise 655 existing ignored tests and the previously validated
48,000-draw panel workflow, subsequently passed in the separate final-source rerun below.
No measured-license assertion was filtered out.

Fourteen failures require missing current measured evidence, including three
additional unchanged design assertions; their source/hash/record dispositions
are recorded in [license disposition](license_disposition.md). The historical
reliability fixture itself verifies successfully before its strict promotion
premise fails. Subsequent assertions are explicitly not claimed executed.

One new software regression exposed actual static response/joint delegation
context loss and untyped memory denial. Adjacent review found Auto’s efficient
context and swallowed interruptions plus conditional/MAG funding classification.
These source fixes and new tests landed after this run’s compilation; this run
is not represented as verification of those later edits. Its current-source
follow-up is recorded below when executed.

The expanded run passed actual warm temporal context tests (203.575s), full
frequentist temporal-class route conformance (411.002s), full Bayesian route
conformance (829.500s), the independent multi-completion PAG numerical matrix
(863.590s), both temporal block-bootstrap checks (236.640s/236.100s), and actual
frequentist panel fit/reuse with independent subprocess replay (0.464s). Sample
sizes, oracle tolerances and scientific/reference fixtures were preserved.
These timings describe this software run, not reference-host performance.

After the fixes, strict workspace library Clippy passed again with `-D warnings`
in 1m22s. The following current-source tests, fresh affected API comparison and
rebuilt native/Python checks remain required before closure.

## Final delegated-search regression results

`cargo test --locked -p antecedent-identify -p antecedent --lib --
--test-threads=3` passed **1,194 tests**: 727 facade and 467 identify. Their 12
existing ignored cases remain ignored. This includes the previously failing
unsealed cell-AIPW checked compiler, direct/delegated joint search, both conditional
and MAG funding cases, and first/later Auto interruption tests. Funded scientific
claims, independently expected adjustment sets, explicit caps, cancellation,
Memory and lease return all execute. A test-only unavailable comparison method
was replaced with complete derived estimand/claim/diagnostic equality; the
Resource/Cancelled expectations and known adjustment assertions are unchanged.

The facet check passes again after these final production fixes: 447 paths,
222 facets, 14 valid waivers, list blob `303eecbe28af`. Test execution used three
threads while that one-worker source audit ran, with one Cargo process.

The fresh affected-crate API check rebuilt rustdoc for `antecedent` and
`antecedent-identify` after the final source changes and passes without further
snapshot edits. The full earlier API comparison covers the unchanged remaining
crates. Final registry checking passes **1,529 paths**; retained-route baseline
has zero findings, and the lint-allowance gate passes again.

## Current-source lifecycle and compiled obligations

The final facade nextest selection compiled all nine suite aggregators plus
retained-refresh, decision compilation, physical plan/report and graph-posterior
attribution/exposure targets. Selection covered response/AIPW, Auto/generalized,
refresh/retention, discovered transport, attribution and cross-world cases. It
executed **380 tests across 15 binaries: all 380 passed**, with 1,700 cases outside
the targeted selection or already ignored. Duration **152.039s**. Both longer
temporal response identified-set checks passed (87.779s and 132.718s). The broad
run above covers the other ordinary workflows; no previously failed software
assertion is hidden by this selection.

`CARGO_BUILD_JOBS=1 python3 scripts/check_route_obligations.py` passed after
compiling/listing its cited Rust tests. It ran while only two fixed test binaries
remained active, with one compiler job and no competing compiler process. The
previous transient compilation failures are resolved; they are not route gaps.

At completed-build barriers, integration reclaimed only generated
`target/debug/incremental` caches to retain disk headroom for native validation.
Compiled libraries/binaries, fresh rustdoc, scientific fixtures and historical
measurement artifacts remain intact.

## Full final-source panel and native binding unit execution

The previously excluded 48,000-draw Bayesian panel workflow was rerun on the final
source with exact test selection and one test thread: **PASS in 788.343s**. Its
fit, mixing diagnostics, retained reuse and independent artifact consumer all
execute. This completes the one explicit numerical-workflow exclusion in the
broad run; no sample budget, tolerance or fixture changed.

Native CPython 3.13 unit tests passed **25/25**, no ignored cases, with four harness
threads. The run uses the local interpreter’s actual base-library path for
embedding. It verifies concurrent original exception identity, actual native pool
ownership/restoration, repeated observer reuse, expired weak adapters, cleanup
beyond stash capacity, reentrant exception formatting and suppressed-interrupt
priority. The earlier module-global fixture collision is resolved with private
per-caller globals; its original identity/concurrency assertions remain.

`gate_checked_execution.py` passes statically for **723 cells and 804 citations**.
A separate observed-results audit resolves its **175 unique path/assertion pairs**
against actual PASS records: 17 final library, 41 final lifecycle, and 117 expanded
integration assertions. The three repeated function names are disambiguated by
actual module/binary, with no unmatched assertion. This is a combination of the
explicitly named runs, not a claim to have invoked the complete shell row runner
or rerun the broad snapshot after every final static-resource edit.

The full panel ran from its already-compiled binary while native compilation
used two jobs. No two compiler invocations competed. Generated compiler caches
were reclaimed only at completed-build barriers. The Python extension is now
being rebuilt in place with Maturin’s development profile and actual CPU-wheel
features; this creates no production performance or release-wheel claim.

## Final scoped exception attribution

The last binding review exposed two further attribution defects: nested exception
formatting could select the inner original, and another live run's tag could
consume its saved exception. Attribution now receives the current `Scope`,
snapshots its owned ids after formatting and selects only the last live owned
tag. The parser-only global attribution call is removed. Unscoped errors retain
no exception; unretained outer descriptions neutralize nested private tags,
including when nested formatting takes the last stash slot. The capacity bound,
non-eviction rule and signal/first-BaseException precedence remain intact.

After these corrections, the complete native CPython unit suite passed **28/28**,
no ignored or filtered tests, in 0.06s after a 25.78s build. Actual concurrent runs
prove foreign-tag isolation and exact owner identity. Reentrant formatting returns
nested tagged text and preserves the outer original, with both ids cleaned.
The capacity-boundary predicate and exact refusal-description transformation
are tested separately; a real run verifies transformed nested text cannot gain
an incorrect cause, without globally saturating the shared stash during parallel
tests. The previous 25-test result predates these additions and is not used as
proof of them.

## Rebuilt Python execution and closure checks

The final strict `cargo clippy --locked --workspace --lib -- -D warnings` passed
again after scoped attribution, in **2m36s**. No later production source changed.
The final CPython 3.13 extension was rebuilt with `maturin develop --locked
--profile dev --skip-install` in **23.14s**, using the installed virtualenv and
CPU-wheel features. The child environment unsets an unrelated `CONDA_PREFIX`;
`--skip-install` avoids a dependency-sync network request and still compiles and
copies the native extension. No dependencies or assertions are skipped. Import
preflight confirms the repository's CPython 3.13 binary is newer than every native
Rust source and SIGINT delivery is enabled. This debug build establishes software
behavior, not release-wheel performance.

From `python/`, `.venv/bin/python -m pytest -q` with the following **28 files**
passed **462 tests**, no failures or skips, in **29.63s**:

```text
tests/test_native_interrupts_and_callbacks.py
tests/test_python_callbacks.py
tests/test_cyclic.py
tests/test_discovery_ci_callback.py
tests/test_panel_discover_ci_callback.py
tests/test_provider_callbacks.py
tests/test_adaptive_inference.py
tests/test_adaptive_procedure.py
tests/test_host_contract.py
tests/test_staged.py
tests/test_prepared_analysis.py
tests/test_checked_functional_effect_lifecycle.py
tests/test_checked_distribution_linear_lifecycle.py
tests/test_graph_posterior_binding.py
tests/test_graph_posterior_analyze.py
tests/test_discovered_transport.py
tests/test_evidence_plan.py
tests/test_design_ranking.py
tests/test_design_ranking_for_study.py
tests/test_joint_panel.py
tests/test_joint_panel_experiment.py
tests/test_joint_panel_reference_decomposition.py
tests/test_causal_artifacts.py
tests/test_posterior_artifact.py
tests/test_support_artifact.py
tests/test_licensed_artifacts.py
tests/test_missing_imputation.py
tests/test_missing_imputation_routes.py
```

This includes actual SIGINT, concurrent callback causes, BaseException/report
priority, parallel progress ownership, static/temporal/panel discovery, local
cyclic execution, retained prepare/inspect/estimate/refresh, posterior routes,
decisions/evidence planning and independent artifact consumers. Seven warnings
are the expected debug-build warning and existing historical-field guidance.
The first run passed 461 tests and caught a new temporal CI fixture returning one
answer for a three-query batch. Its callback now returns one result per query;
all original graph, routing and serial-policy assertions remain. The complete
selection was rerun successfully; no failing case was filtered out.

Changed Python files pass both Ruff checking and formatting. Final architecture,
compiled-target, v3 registry (**1,529 paths**), Python facade (**462 entries**),
generated host-schema, retained-route, lint-allowance and documentation-link
checks all pass. Gate negative controls remain as recorded above. API snapshots,
route obligations, checked-execution citations, conformance generators and the
final published-source facet check also pass as recorded; no native-private
change alters their published Rust inputs.

The broad numerical runs retain **14 measured-evidence-related failures** with
explicit Q08/Q09 owners and exact record/hash/assertion dispositions. There is no
claim that the whole workspace suite or release-wide Q13 is green. Their shipped
zero-attestation predicate is unchanged, and no measurement, oracle, waiver SHA
or numerical/artifact golden is repinned. The engineering software defects are
fixed and their affected lifecycle/resource/artifact assertions execute; separate
Python UX, documentation, test-organization and scientific passes remain open.
