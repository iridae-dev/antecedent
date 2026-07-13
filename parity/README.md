# Parity manifests

Pinned reference baselines and assignable capability inventories for DoWhy and
Tigramite. See DESIGN.md §26 and ADR 0009.

- [dowhy.toml](dowhy.toml) — DoWhy v0.14 pin and inventory
- [tigramite.toml](tigramite.toml) — Tigramite 5.2.1.25 pin and inventory
- [phase4_deviations.md](phase4_deviations.md) — Phase 4 kept deferrals
- [phase5_deviations.md](phase5_deviations.md) — Phase 5 kept deferrals

Status values: `pending`, `in_progress`, `done`, `intentional_deviation`.

Do not mark a capability `done` without conformance fixtures under
`conformance/` and a recorded reference-output generation command.

## Phase 4 / 5 exit (2026-07-21)

Verified locally:

- `cargo test --workspace --exclude causal` — pass
- `scripts/gate_phase4_reuse.sh` — pass
- `cargo test -p causal-stats --lib ci::calibration` — pass
- `cargo test -p causal-analysis --test phase4_conformance` — pass
- `cargo test -p causal-discovery` (incl. PCMCI+ Exact) — pass

Kept deferrals only: do-samplers→P7, conditional/mediation→P9, PAG/LPCMCI→P8,
J/RPCMCI→P9, native GPDC (no torch), clean-room PCMCI+ pin. See the phase
deviation docs and matching `intentional_deviation` rows.
