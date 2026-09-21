# T6 review and corrections

This review covers the finite categorical plug-in provider, joint IID bootstrap,
common retained lifecycle, Python facade, and statistical artifact consumer.
It does not remeasure the hours-long calibration suites. The recorded T6
coverage measurements retain their original commits and scope.

## Corrected findings

- **Intervention-world collision:** resampling keys omitted intervention values.
  Two arms with the same regime and snapshot could reuse the wrong indices;
  unequal arm sizes could then discard most draws. Keys now contain population,
  snapshot, regime and canonical concrete assignments.
- **Cancelled or resource-limited bootstrap:** cancellation could break the loop
  and publish an interval from a partial run. Cancellation now fails the whole
  execution, including after the final progress callback. Resource/numerical
  failures are not counted as statistical support failures. Checked budgets
  cover fitting, all requested evaluations and retained replicate buffers.
- **Missing joint grid path:** grid points now share a single fitted joint per
  sampled dataset and outer replicate. Any failed point invalidates that draw
  for the whole grid. Original replicate IDs survive export; contrasts match
  IDs, retain covariance and report the union of failures. Intervals remain
  pointwise, not simultaneous.
- **Insufficient execution identity:** fitted probabilities alone omitted sample
  size and row content. Snapshot identity now includes sample summaries and a
  content digest; inference identity includes the joint-cell bound and sampling
  bindings. The retained seed governs later execution and refresh regardless of
  a caller's context seed. The grid family is also part of inference identity
  because its joint failure filtering affects the retained draws.
- **Unverified artifact uncertainty:** loading previously trusted interval
  metadata and checked only three identity layers, lost mean intervals, and
  returned a differently identified prepared handle. Version 2 verifies all
  identity layers, origin/sample-size consistency, replicate IDs and counts,
  probability vectors, percentile arithmetic, means and four reasoning slots.
  It preserves the original frozen seed and result. Fitted joints and content
  digests are retained; raw participant rows are not embedded. Loaded artifacts
  require explicit sample refresh before re-estimation. No re-bootstrap, fitting
  or data fetching occurs during load. These checks establish internal
  consistency, not authenticity of the supplied samples or replay of resampling.
- **Numeric coordinate mismatch:** Python float-valued requests could not read
  integer-coded empirical tables, breaking target-observational execution.
  Exact-law lookup now matches exactly represented integral numeric levels
  locally, preserving stored values and global value equality.
- **Historical exact-law provenance:** adding origin metadata had changed the
  identity encoding of pre-T6 exact artifacts. Consumption now retains their
  historical encoding through export and refresh. Empirical-origin laws are
  rejected by the supplied-exact prepared modality.
- **Input semantics:** invalid confidence levels, unequal/empty native columns,
  malformed intervention worlds, and missing observations now fail explicitly.
  Complete-case deletion had no missingness assumption in the evidence contract
  and could estimate a selected population silently. The current estimator
  therefore requires complete observations. Weighted sampling cannot silently
  use unweighted counts. The convenience-target check no longer rejects an
  unrelated source sample. Estimated laws cannot masquerade as supplied laws.
- **Facade/provenance:** sample mappings are immutable; native parsing supports
  them. Results expose factor-level support, all reasoning slots, retained
  replicate metadata and an explicit `not_bound_to_this_execution` calibration
  status when no execution-specific record is bound. Editing Python display
  fields cannot alter the exported native claim.
- **Stage evidence gate:** a T6 license cited a Rust integration test, but the
  gate attempted to collect it as pytest. Evidence resolution now dispatches to
  the Rust or Python test collector and rejects unsupported file types.
- **Engineering:** frequency counting indexes each finite domain once, removing
  a repeated level scan from the row loop. Shared grid refits replace independent
  per-point fitting. Metadata previews no longer fit frequency tables.

The percentile construction and reuse of paired sample indices were checked
against the [SciPy bootstrap reference](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html).
The implementation retains the existing declared failure-fraction policy:
withhold intervals with fewer than two successful draws or more than 50% failed
draws. Successful-draw filtering is recorded, not a guarantee of nominal
coverage; the existing weak-overlap calibration boundary remains relevant.

## Compatibility and calibration scope

Historical trial-IPW behavior and exact-law artifact identities are preserved. The earlier T6
statistical artifact version lacked sufficient provenance for the corrected
checks and must be regenerated; it is never silently upgraded into a checked
version-2 claim. Known supplied laws remain fixed across bootstrap draws.
Unknown, linked and clustered dependence retain identification and withhold
IID intervals.

The recorded 95% mean-interval evidence covers the named binary front-door,
shared-joint and source/target-imbalance designs and their recorded sample-size
grids. It does not establish coverage for every finite table, confidence level,
bootstrap count, atom interval or contrast. In particular, a nominal percentile
interval and an execution-specific calibration binding are distinct claims.
No new coverage measurements or unconditional coverage licenses are asserted by
this review.

## Consuming checks

- Native empirical/provider and estimator unit tests.
- Native lifecycle, sample-size identity, frozen-seed, cancellation-after-first-
  replicate, and artifact mutation tests.
- Existing numerical witnesses that fixing target data or separately resampling
  shared factors gives the wrong uncertainty; trial-IPW scope regression.
- Python invalid coverage, missingness, immutable data, multi-world resampling,
  atomic refresh, native display authority, full load/export round trips, joint
  grid and paired contrast tests.
- Workspace checking, Rust Clippy, Python Ruff/mypy, full Python regression,
  transport/support-matrix and artifact compatibility gates.

Validation completed during review: the full Rust workspace library run passed
2,042 tests (41 ignored), and the full Python regression passed 2,063 tests
(1 skipped). The final rebuilt extension passed all 73 focused transport checks after
subsequent fixes, including invalid intervention values and the grid lifecycle.
Four exact-lifecycle tests also passed, including historical identity encoding
and empirical-origin rejection.
The support matrix passed with 3,402 cells and 463 licensed cells; all 15
transport stage contracts passed. The artifact gate passed 61 tests. No
hours-long calibration suite was rerun.
