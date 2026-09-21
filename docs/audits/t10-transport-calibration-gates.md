# T10 transport calibration gate repair

The four empirical-transport coverage records now resolve their DGP names to
functions actually called by their tests. `xy_shared_joint` and
`xy_rare_treatment` fix the existing generator's scenario parameters;
`standardize_imbalance` and `frontdoor_binary_scm` are the renamed generators.
Seeds, sampling, truth, interval construction and acceptance thresholds are
unchanged.

The two `allow mechanism ... assignments` entries in
`scripts/calibration_surface.list` document a reviewed identifier collision.
The private helpers in `analysis/exact.rs` and `analysis/transport_grid.rs`
encode `antecedent_expr::Assignment` into sorted `(u32, ValueWire)` records;
neither executes a fitted-mechanism path. Both files remain in the core facet.

The attestation self-test now explicitly requires both `estimator.transport`
and `estimator.transport_empirical` for `transport.empirical_table_plugin`.
Other non-mechanism estimators retain the exact-one-estimator-facet check.
No facet obligation, support route or coverage threshold was removed.

## Fresh measurement

All four records were re-measured in a clean source snapshot at
`7bf07a1653693589c35ac175169855a8d2f68715`, retained on
`codex/t10-calibration-measurement`. The snapshot includes the in-progress T10
changes without committing them on the original working branch. Publish a ref
reaching this measurement commit alongside the registry update: attestation on
another machine needs the real source commit, not just its hash in the records.

The existing calibration runner selected the four
`v20_transport_statistical_calibration` groups and ran all three sample-size
points, with four concurrent jobs, the release profile and Rust 1.97.1. Each
point attempted 400 replicates. The unchanged runner retained its automatic
2,000-replicate precision-recheck rule; no recheck was triggered. The 12 jobs
passed. Every emitted field, including every grid point, reproduced the old
measurement bit for bit (`calibration_facets.compare_replay`).

| DGP | Sample-size grid | Observed coverage at each point | Role |
| --- | --- | --- | --- |
| `xy_shared_joint` | 100, 200, 400 | 0.9475, 0.9300, 0.9325 | Gated |
| `standardize_imbalance` | 200, 400, 800 target rows | 0.9500, 0.9550, 0.9375 | Gated |
| `frontdoor_binary_scm` | 150, 300, 600 | 0.9375, 0.9450, 0.9475 | Gated |
| `xy_rare_treatment` | 100, 200, 400 | 0.6810126582, 0.8525, 0.9025 | Named boundary |

The rare-treatment point at n=100 had 395 usable replicates and five skips;
all other points had 400 usable replicates. Its undercoverage remains an
explicit boundary, not a successful nominal-coverage claim. These measurements
cover the empirical-table route only, not learned categorical or trial AIPW
inference.

Only the four `calibration_sha` fields changed in the coverage registry and its
generated Rust mirror. The other 587 records were preserved without restamping
or deleting them. No replay waiver was added. Local raw logs are archived under
`target/t10-transport-calibration-7bf07a16/` and in the measurement worktree's
`target/calibration-records/`.

## Verification and remaining T10 work

- `gate_parity_schema.sh`: passes.
- Schema mutation self-test: all 22 cases pass.
- Calibration surface completeness and boundary check: passes.
- Calibration attestation self-test: passes.
- Transport calibration: all 12 grid jobs pass; four records are attested.
- Rust formatting and whitespace checks: pass.

The complete `gate_calibration_attestation.sh` still fails honestly on the
**587 other records** whose statistical surfaces changed since their recorded
measurements. Their broader re-measurement was explicitly left outside this
scoped repair. This change does not claim T10 or release acceptance.
