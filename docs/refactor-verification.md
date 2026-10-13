# 2.4.0 refactor verification

This is a branch validation record, not a release or a calibration attestation.
The refactor audits, repairs and local verification described below are
complete at code commit `ef2cf46e`. The branch is not fully green: legitimate
calibration remains a precondition for the measured-only acceptance checks and
attestation gates. No calibration measurements or replay measurements have been
run for this pass.

## Numerical findings and regression evidence

| Area | Finding and repair | Evidence and limit |
| --- | --- | --- |
| Gram inversion | An overflowed product of finite equilibration scales could destroy a representable inverse entry. A guarded division order now handles extreme scales and rejects non-finite inverse entries. | Four extreme-scale regressions pass, including a mixed-scale subnormal result. Ordinary finite scale products retain their former arithmetic order. |
| Monte Carlo attribution | Changing one mechanism's draw count shifted the sequential noise consumed by unchanged nodes. Each node now restarts its own keyed stream across coalitions. | A zero-effect mechanism swap regression passes. This changes seeded Monte Carlo outputs and establishes this coupling invariant; it does not establish general coverage. |
| Structure MCMC | The reported add/delete Hastings factors were wrong. Both candidate orientations contribute to deletion, making the actual proposal symmetric. | Enumerated two-node proposal masses satisfy detailed balance. The sampler's transition code and seeded chain are unchanged. |
| Graph separation | The refactored ADMG Bayes-ball path agrees with ancestral moralization on the checked graph family. | The exhaustive four-node ordered-graph comparison passes for 98,304 cases. This is finite regression evidence, not a proof for every graph. |
| Counterfactual streams | A later world's stream should not depend on how many draws an earlier world consumes. | The draw-count independence regression passes. |
| Design ranking | Bounded parallel execution should preserve indexed replicate streams, reduction order and reported work. | Serial and four-worker results agree by bits for scores, standard errors, ranks, budgets and stopping behavior on the checked fixtures. |
| Matching cache | A compact feature hash cannot establish feature equality. Reuse now checks retained feature bits. | A forced key collision rebuilds the index and matches fresh nearest-neighbor results. No naturally occurring collision is claimed. |
| Exact factor cache | Rebinding to different content must not reuse a factor under an unchanged external label. | A warmed, rebound provider agrees with a fresh provider, and the original provider retains its original result. |
| Prepared refresh | Replacing observations must refresh numerical state and data identities while retained clones preserve their own snapshot. | The warmed study agrees by bits with a fresh fit; an independent changed-slope oracle and old-clone contract/estimate isolation checks pass. |
| AIPW recalculation | A request could change estimator settings while the retained study kept its old fitter. Refresh now installs the requested settings and rebuilds numerical preparation while retaining identification. | All nine recalculation-attempt integration tests pass, including exact warm/fresh law, score-table and identity checks, observed work and failed-refresh state isolation. Unsupported score-table configurations are refused before reuse. |
| Nuisance work counters | Counting only completed cache leases hid penalized or capacity-bypassed fits and completed work before a fallback. Receipts now observe completed outer-fold fits directly. | The 701-test estimator unit suite passes. Cache hits add no work; a later-fold failure followed by successful ridge fallback records exactly 12 completed fits. Inner penalty selection fits and failed fit attempts are explicitly outside this counter's unit. |
| External callbacks | The request digest omitted the planner branch, permitting identity reuse across different branches. | All nine external-callback integration tests pass, including branch-bound replay and retry idempotency. |
| Calibration attestation cache | An empty live assessment incorrectly revived the previous snapshot's attesting IDs. The generator now accepts only the live assessment, including an empty result. | Two controls reproduced stale-cache revival before the repair and pass afterward; three controls pass overall. All 792 measurement rows, original SHAs, method reasons and identity aliases remain unchanged; the 792 stale attesting IDs are removed. |

The ID/IDC review found no verified mathematical regression. Subsequent strict
lint repairs preserve traversal order, graph surgery, intervention values,
instrumentation boundaries and error propagation. Existing identification and
graph-surgery tests passed in the completed Rust suite.

## Cache ownership and invalidation

The review covered immutable graph and expression caches, per-run search state,
domain replacement, factor-provider content and axis keys, event-state cache
invalidation, nuisance fit keys, matching geometry, prepared-study refresh and
recalculation stage dependencies. A reuse receipt describes declared reuse and
observed work; it is not independent proof that a numerical cache is correct.
Warm-versus-fresh comparisons check actual outputs in the repaired paths.

The later identification-cache split into DBN, graph-posterior and temporal-class
modules was reviewed independently as code motion. The review found unchanged
cache keys, graph traversal and weight order, cancellation, projection horizons
and retained `Arc` ownership. After that split, all 404 facade library tests
passed (five ignored), the public API comparison passed, and strict all-target
facade Clippy passed after removing an unnecessary copy of a node slice.

Arrow borrowing depends on immutable exported buffers for the entire retained
lifetime. The owner preserves allocation lifetime but cannot prevent writes
through external aliases, including a NumPy array backing an Arrow array.
Cached content identities and fits rely on this premise. Ordinary NumPy mapping
ingestion copies; borrowed Arrow input requires new buffers and `refresh` to
replace observations. See [the borrowing contract](hot_paths.md).

## Claim integrity and licensing

Support cells remain closed unless their owning registry licenses them and their
cited evidence executes. Point-only evidence does not license interval coverage
or posterior guarantees. This pass corrected disclosures, source references,
independent temporal truth pins and deliberately broken gate controls. It did
not promote support cells, lower coverage floors, narrow numerical tolerances,
rewrite calibration measurements or create replay waivers.

The temporal reference corrections use structural population truths rather than
conditional regression coefficients: the mediation components are 1.5 and 2.0,
with total 3.5; the equal-mass marginal response is `3.25 + 2x`, with derivative 2. At
`x = 2`, its log-outcome derivative is `8/29` and elasticity is `16/29`. The existing
finite-sample tolerances remain unchanged.

The calibration-surface map follows extracted implementation paths and shared
test helpers conservatively. Boundary validation and widening against `45fed371`
pass. Interval-coordinate and interval-surface checks report only stale
calibration attestations; their other source/ownership checks have no remaining
failures. Existing measurements are invalidated by relevant code drift, as
intended. Passing build or fixture tests cannot restore that attestation.

Evidence discovery now follows enabled file-scope Rust `#[path]` modules and
their actual qualified helper calls. Comment/string decoys, disabled functions,
unrelated nested modules and ignored tests remain rejected. Closed generated
refusal helpers are credited only after checking the actual macro/error
constructor and reason-code path. Cargo collection and execution enable only
the selected target's declared `required-features`; this permits cited ordinary
fixtures to compile without running ignored calibration tests. New positive and
deliberately broken controls exercise these rules.

## Engineering status

Before the attestation-cache repair, the Cargo-selected Rust suite passed 6,878
tests across 480 executables, with 490 ignored and no failed or filtered tests.
This used the CI test feature set and left ignored measurements ignored.
Workspace doctests also pass. That full run preceded the later
identification-cache module split; the independent review and affected checks
described above cover that split.

Before the attestation-cache repair, the native Python suite passed after the
estimator and cache repairs: 4,711 tests passed and 54 were skipped, with 90.80%
coverage against the 85% floor. After the repair, the native extension was
rebuilt and the complete CI coverage command produced 4,680 passes, 31 failures
and 54 skips, with 90.62% coverage. Every failure was independently reviewed:
all refuse measured-only acceptance because `coverage_record_not_attesting`;
none indicate a separate numerical regression. These assertions remain intact.
The skips include private calibration acceptance hooks. The current configured
lint and typing gate passes.

The current affected IO-core suite passes all 330 tests, with zero failures or
ignores, and strict all-target IO-core Clippy passes. The current facade suite
has 403 passes, five ignores and one failure: the measured temporal artifact
acceptance test refuses for the same stale-attestation reason.

Benchmark smoke (`--test`, without a measurement campaign), fuzz-target
compilation, dependency/license checks, provenance, metadata and named-test
collection pass. No fuzz campaign was run. The optional learner suite passed 58
tests and the Gaussian-process model suite passed 79. Strict workspace and
Gaussian-process Clippy pass; the later default facade check also passes.

Checked execution passed all 776 estimator citations. Composition passed its
40 rows, including 29 Rust filters and two Python nodes; graphless support
passed all 63 citations. Evidence reachability passes for all 206 fixtures.
The full support matrix passed: 4,320 cells, 717 licensed, 1,809 not applicable,
and 1,794 explicitly refused; none lack reasons or remain allowed without a
license. These large cited-gate passes preceded the attestation-cache repair.
The current transport gate resolves 469 routes and passes all 26 fixture rows;
its eight failed route rows are measured-only acceptance refusals from stale
attestation. Promotion execution completed with 23 distinct failed citation IDs,
all independently traced to the same stale-record refusal. Its old summary
counted three rows sharing a failed Rust test as one entry; a reproduced
two-row control now verifies the corrected diagnostic accounting. No execution
criterion changed. Every official broken-input self-test subcommand passes.
The umbrella driver originally stopped at the promotion rule-usage lint;
after its AST repair, the complete promotion self-test passed and the remaining
official commands passed individually. One release-candidate control initially
failed on an unwritable default UV cache; the corrected-cache rerun passed.
The grouped-count regression was also rerun after its final repair. These
results are a composition of the actual command runs, not a claim that the
original fail-fast umbrella invocation passed.

The publishing dry run completed: two crates were packaged and verified; the
other twenty received the script's workspace check fallback because their 2.4.0
registry dependencies are unpublished. This checks packaging readiness within
that limit and does not establish that all twenty can be verified against the
registry. Nothing was uploaded.

All 22 API comparisons pass after review of the typed IO API changes, borrowed
static-table iteration and additive completed-fold work telemetry. The later
cache split has its own passing facade API comparison. Final-source CodeQL
passes with no unexcluded findings: all 1,732 Rust files extracted without
errors, all 630 executable Python source files extracted, and all five workflow
YAML files covered. The native `.pyi` type stub is the sole Python baseline file
outside the installed extractor's `.py` suffix configuration; typing checks
cover that declaration file. The last snapshot has manifest SHA-256
`da27ff7011990e359c4c91e197c66973885f55ad802932a9e967b4026205d1d0`,
with no code drift at completion. Existing query exclusions remain unchanged.
This is evidence from the configured queries, not a proof of security.

The inventory and composition/decision exit checks originally reported 15
promotion records backed by stale calibration measurements. Their story and
refusal tests passed before the attestation-cache repair. The repaired snapshot
also closes measured-only runtime factories until their records are current;
their positive acceptance checks are calibration-dependent. Calibration is
deliberately deferred until the branch's numerical surface is stable; these
attestations have not been fabricated or waived.

The branch is not fully green. The calibration-dependent positive acceptance
checks and attestation/exit gates remain required for a future release; this
pass does not substitute passing source, build or ordinary fixture checks for
them.

Current source/ownership checks, calibration-surface boundaries, generated
inventories, metadata and document links pass. The strict documentation build
passes. Nonblocking source-reachability warnings remain in the promotion
checker; its static contract passes, and actual citation failures are the
calibration-dependent cases listed above.

A source-only wiring audit found no calibration bootstrap cycle: ignored
measurement tests execute raw/candidate estimators, score `CoverageTally` and
emit records; they do not need an already-authorized `MeasuredInference` result.
The cleared attesting IDs therefore close measured factories without preventing
the later measurement campaign. This verifies the source path and compilation
readiness, not a completed calibration run.

Local validation does not execute the remote Linux, Windows or wheel CI matrix.
Calibration and release-candidate checks remain outside this pass. No branch has
been pushed or released. Work is committed in separate logical increments with
Charles Hinshaw's DCO sign-off and without co-author trailers.
