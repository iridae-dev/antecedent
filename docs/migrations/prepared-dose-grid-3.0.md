# Prepared Bayesian dose-grid execution in 3.0

`Study::prepare_posterior_row(request, ctx)` compiles and validates a randomized
dose-grid posterior without fitting or drawing. The resulting prepared handle
owns the checked request and execution engine; the request builder can be dropped.
`estimate(ctx)` runs that retained program, and `retarget(grid, functional, ctx)`
reuses the coefficient posterior for compatible grid and functional changes.
Changed data, priors, draws or seeds require a new preparation; a changed kernel
policy refits. Execution uses caller admission and cancellation policies.

Rust convenience evaluation and artifact consumption delegate to the same
compiled owner. Python `dose_grid_posterior` and
`consume_dose_grid_posterior_artifact` accept an optional `execution=`
(`Execution(threads=..., limits=ResourceLimits(memory_bytes=...), cancel=...)`). Resource and cancellation refusals use the existing typed
Python errors. Omitting those arguments preserves the default execution policy.

Artifact encoding is separate from compilation and numerical execution. Existing
version-one artifacts retain their numerical replay and digest checks, including
independent consumption. The stored mathematical model and draw pairing are
unchanged. The row remains explicitly unlicensed with unmeasured calibration;
compiler integration adds no identification or scientific permission.

This change does not close the aggregate resource-accounting or release
attestation backlog. The reviewed cross-family repair scope and verification are
recorded in [repair acceptance](compiler-integration-acceptance-3.0.md).

Prepared numerical outputs and returned posterior artifacts retain admission leases
until their last owner is dropped. Cancellation does not release live outputs.
Frozen coefficient data and its lease share one immutable owner, so replacing a
cached numerical policy cannot uncharge a fit still used by another execution.
Retargeting replaces the grid reservation instead of retaining earlier grids.
Artifact resource guards are local execution state: serialization omits them,
independent consumption reserves its own replayed output, and equality compares
scientific values rather than admission ownership. These byte counts are typed
allocation quotes/reservations, not process resident memory measurements.

`CompiledPosteriorRow::artifact(ctx)` is the public IO execution entry point and
returns the guarded artifact owner. The intermediate conversion to a bare
`PosteriorRowResultWire` is now internal, so a public compiled workflow cannot
silently discard result admission ownership. Numerical component execution
through `CompiledDoseGridPosterior::execute` also returns a guarded output.

`BayesianPosteriorRow::export()` keeps the resource policy used to produce or
consume the row and admits CBOR encoding scratch under that shared account.
`export_with_context(ctx)` permits an explicit current policy. Cancellation and
resource exhaustion remain typed refusals. Counting serialization precedes CBOR
buffer allocation, and the numerical wire bytes remain unchanged. Returned byte
vectors become caller-owned after export; the row continues funding its own
retained numerical/artifact state.

Retained resource policy excludes caller progress callbacks, preserving the row's existing unwind-safety auto traits.
