# Python API implementation checkpoint

Status: incomplete. No completion percentage is asserted. Calibration and attestation are excluded; all Python API and UX requirements remain in scope.

## Committed

- `324b66742` — Own Python table inputs and scope reusable execution controls.
- `f799e3343` — Separate prepared compilation from statistical materialization.
- `5e949703d` — Share admitted native buffers across Python snapshot views.

## Verified focused evidence

These results establish their specific assertions, not full API acceptance.

- Owned-data Rust library: 217 passed (`/private/tmp/python-api-owned-buffer-data-tests.log`).
- Deferred preparation/provider/grid Rust slice: 7 passed (`/private/tmp/python-api-deferred-grid-core-tests.log`).
- Independent bundle replay receipt regressions: 2 passed (`/private/tmp/python-api-bundle-core-tests.log`).
- Recent Python ownership/lifecycle/temporal/adaptive slice: 106 passed, 1 failed due to a JSON list/tuple test comparison; fix written, not yet rerun (`/private/tmp/python-api-fourth-python-tests.log`).
- Canonical load plus immutable preparation slice: 6 passed (`/private/tmp/python-api-canonical-load-tests.log`).
- Broader outcome integration: 302 passed, 57 failed; fixes written, not yet rerun (`/private/tmp/python-api-domain-outcome-tests.log`).
- Decision/composition integration: 119 passed, 2 failed; one new invalid-operation test fixed, one existing measured-evidence predicate preserved (`/private/tmp/python-api-decision-outcome-tests.log`).

## Current checkpoint — 26 October 2026

- Native compile check passed (`/private/tmp/python-api-checkpoint-native-check-v2.log`).
- Fresh extension build passed in 4m28s (`/private/tmp/python-api-checkpoint-build.log`).
- Ownership, immutable lifecycle, execution controls, canonical load, transport, tiered identification, temporal and adaptive focused batch: **100 passed** (`/private/tmp/python-api-checkpoint-foundation-tests.log`).
- Combined outcome, cyclic, composition, recalculation and decision batch: **482 passed, 2 failed** (`/private/tmp/python-api-checkpoint-integrated-tests.log`). Both failures are the existing Q08/Q09 measured-evidence predicates: `test_a_measured_interval_needs_a_declared_sampling_and_an_attesting_record` and `test_every_published_route_is_licensed_inside_a_measured_coverage_record`. They remain failures and their scientific predicates are unchanged. No new failures remain in these batches.
- Latest antecedent-data unit tests: **217 passed** (`/private/tmp/python-api-checkpoint-data-tests.log`).
- Focused root source/test Ruff check passed (`/private/tmp/python-api-checkpoint-root-ruff.log`).

These are separate batches with overlapping cases, not an aggregate unique test count or full-suite acceptance. Earlier failing results above are retained as checkpoint history. Agent source is frozen; feature expansion stopped while these failures were resolved. Incremental snapshot commit `5e949703d` has the requested DCO identity and no co-author trailer. ROADMAP.md remains unstaged and preserved.

## Sorted worktree checkpoint — 26 October 2026

The accumulated implementation is now split into these additional DCO commits:

- `b9507a535`: distinguish verified references from independently replayed bundle nodes.
- `72761e795`: account for reliability workspaces and preserve typed admission failures.
- `9240db329`: reuse keyed causal proofs across pure data and execution branches; rebuild physical and row-bound state.
- `7005fe0a6`: defer statistical transport fitting and share raw/fitted native storage across branches.
- `20ab32a5c`: atomically migrate the coupled Python Outcome/export contract, bindings, stubs and callers.

Latest evidence:

- Fresh extension built after the explicit-data report correction (`/private/tmp/python-api-report-native-build.log`).
- Expanded sorted integration batch: **643 passed, 2 failed** (`/private/tmp/python-api-sorted-final-tests.log`, JUnit `/private/tmp/python-api-sorted-final-tests.xml`). Both failures are the same two Q08/Q09 predicates listed above. They remain actual failures; no filtering or predicate changes were used.
- Full mypy: **240 source files, no issues** (`/private/tmp/python-api-sorted-final-mypy.log`).
- Ruff package/tests pass; **736 files formatted** (`/private/tmp/python-api-sorted-final-ruff.log`, `/private/tmp/python-api-sorted-final-format.log`).
- Deferred Rust slice: **8 passed** after correcting proof reuse and strengthening the changed-row-count/effect-size regression (`/private/tmp/python-api-deferred-current-v3.log`).
- Reliability resource regressions: **3 passed**, registered in consolidated suite_02 (`/private/tmp/python-api-resource-core-v2.log`).
- Bundle receipt regressions: **2 passed** (`/private/tmp/python-api-bundle-receipt-current.log`).
- Statistical seed authority and independent learned-trial replay/atomic refresh: **1 passed each** (`/private/tmp/python-api-transport-seed-current.log`, `/private/tmp/python-api-trial-replay-current.log`).
- Inventory negative controls: **12 passed**; integrity **0 findings**, **6,787 unresolved capability/surface mappings**. Many are aliases, DTOs and methods, but each still needs an actual reviewed consumer and relevant execution assertions. These counts cannot establish parity.

The first full-suite attempt stopped at three import/collection errors (now fixed). The next exploratory attempt was deliberately interrupted after 40 passes while sorting; neither is full-suite acceptance. Full coverage, remaining Rust/API gates and example execution remain outstanding.

Report goldens changed only after fixing retention of the actual materialized execution. Reviewed remaining diffs were physical accounting: the temporal report already declared/executed 199 bootstrap replicates, and the Bayesian report already requested 64 posterior draws. Fresh quotes now account for that retained/output/work storage. Scientific claim/program identities, values, intervals, priors and algorithms match the previous reports exactly.

Core proof reuse is keyed and guarded. Linear compiler identification and graph-posterior publication program derivation still rebuild; do not claim universal zero-identification reuse. Scenario empirical preparation, restricted empirical counting, smoothed posterior context propagation, provider/policy preparation, common refusal preparation and selected ledger result persistence/cause retention remain concrete implementation gaps.

## Work still required

- Complete and publish the canonical public API; remove superseded interfaces.
- Finish cyclic/feedback end-to-end lifecycle, fit sharing, execution controls, typed selection and downstream composition.
- Finish remaining family lifecycle, ownership, resource and independent-consumer gaps.
- Complete capability-to-journey inventory with executing evidence.
- Synchronize stubs, registries, host records, README, examples and migration guide.
- Complete fresh-extension Python validation, Rust checks, lint/type/API gates, coverage floor and local before/after measurements.

The TODO item stays open. Existing measured-evidence failures remain assigned to Q08/Q09; their predicates are not suppressed or altered. ROADMAP.md is preserved.
