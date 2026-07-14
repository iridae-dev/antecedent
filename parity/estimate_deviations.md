# Estimate deviations

Intentional waivers relative to `parity/dowhy.toml` for classical effect
estimation surfaces. Shipped deferrals (do-samplers, conditional effects,
linear temporal mediation) are tracked as `done` under gcm / context inventories;
they are not listed here.

## Verification

Estimate capabilities in `parity/dowhy.toml` that are `status = "done"` are
backed by conformance under `conformance/estimate/` (including `refuters/`),
plus unit/integration harnesses. Linear, partial-linear, nonparametric, and
Reisz sensitivity are implemented; Reisz is tracked under
`dowhy.refute.sensitivity` notes.
