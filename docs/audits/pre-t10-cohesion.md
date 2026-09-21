# Pre-T10 cohesion implementation record

## Delivered surfaces

- Structured learner specifications are shared across DML, DR-Learner and
  transport. Native validation and scientific identities bind the parameters.
- `PreparedAnalysis[ResultT]` uses an internal transport lifecycle adapter.
  `transport.prepare` has typed scalar, grid and trial overloads, with separate
  provider, inference and physical-control objects.
- Transport uncertainty, contrasts, grid points and located support failures
  have typed Python projections. Mapping access remains a migration bridge;
  dictionaries are produced explicitly by `to_dict()`.
- DR-Learner and the native causal forest retain portable fitted-effect state.
  Loading verifies the parent claim; prediction uses the retained feature order
  and intercept map and never refits. CPU parametric/tree codecs are supported;
  neural export remains unsupported.
- Trial AIPW uses shared OOF roles, known randomization, explicit nested versus
  independent sampling, held-out losses, overlap diagnostics and joint outer
  refits. Portable claims replay structural identification and the score.
- Recursive learned transport fits a coherent finite categorical chain per
  dataset/world, shares laws across factors and grid points, and retains actual
  empirical conditioning support. It remains model-based plug-in estimation.
- Grid failure values live in core; durable records and container assembly live
  in I/O. v2 grids defer child encoding to export and bind point-evidence digests.
  v1 verification and round-trip compatibility remain tested.
- Exact grid coordinates retain compiled plans. Statistical execution reuses its
  retained original point laws; bootstrap draws still refit contributing datasets.
- Nuisance cache entries share immutable predictions and initialize per key.
  Fold assignments and original row identities are part of reuse validity;
  failed/cancelled fits cannot publish usable predictions.

## Evidence collected

The release-built Python suite passed **2,113 tests**, with one explicit long
calibration skip. The affected Rust library run passed **891 tests**, with 25
existing ignored tests. After the final retained-law reuse and role-capability changes, the affected
facade and estimator suites were rerun (154 and 363 passing tests).

Additional checks passed: Python typing/stub/public-surface checks, Ruff lint and
formatting, Rust formatting, causal-artifact regressions, transport-stage registry,
ordinary support matrix, provenance schema and metadata consistency. Builds were
checked without optional ML, with each CPU provider, with the full CPU bundle,
and with the optional neural-provider feature. The neural check is compilation,
not a hardware performance or numerical acceptance claim.

Regression evidence includes portable/native prediction agreement, malformed
model rejection, failed refresh, concurrent cache initialization, cancellation,
fold-identity invalidation, v1/v2 grid tampering, independent sample sizes,
complementary-source finite truth and separate nuisance-misspecification cases.
These tests do not establish empirical interval coverage.

## Performance and T10 boundaries

`benchmarks/cohesion/` contains reproducible release workloads and measurements.
Python allocation peaks and process high-water RSS are separate from native
allocator counts and incremental live bytes. The native benchmark pins one
execution thread and compares fresh fitting with retained nuisance reuse while
checking scientific values. Allocation instrumentation affects latency.

No branch-wide speedup is claimed: that needs a controlled before/after run on the
same host, build and workloads. Foreign allocations are outside the Rust allocator
counter. Full calibration and coverage-registry rebinding remain T10 work; all new
interval constructions remain nominal and visibly uncalibrated.

See the [migration guide](../migrations/2.0-pre-t10-cohesion.md) and executable
`examples/python/cohesive_ml_transport.py` for the public workflow.
