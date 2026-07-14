# Phase 10 deviations

Intentional deferrals from DESIGN.md §32 (Phase 10) as reconciled against
`parity/phase10.toml` and `parity/gcm.toml`.

## None blocking Phase 10 exit

Phase 10 attribution surfaces required by DESIGN.md §17 / DoWhy-GCM are
implemented. Kernel two-sample / change-point mechanism tests beyond the
shipped LR / mean-diff / classifier proxies remain available as future
extensions without blocking exit criteria.

## Verification

Every `phase = 10` row in `parity/phase10.toml` with `status = "done"` is backed
by unit tests, `conformance/phase10/`, and/or criterion benches mapped in
`scripts/gate_phase10.sh`.
