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
`consume_dose_grid_posterior_artifact` accept optional `threads`, `max_bytes` and
`cancel` arguments. Resource and cancellation refusals use the existing typed
Python errors. Omitting those arguments preserves the default execution policy.

Artifact encoding is separate from compilation and numerical execution. Existing
version-one artifacts retain their numerical replay and digest checks, including
independent consumption. The stored mathematical model and draw pairing are
unchanged. The row remains explicitly unlicensed with unmeasured calibration;
compiler integration adds no identification or scientific permission.

This change does not close the aggregate resource-accounting or release
attestation backlog. Full cross-family acceptance remains a separate check.
