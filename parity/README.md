# Parity manifests

Pinned reference baselines and assignable capability inventories for DoWhy and
Tigramite. See DESIGN.md §26 and ADR 0009.

- [dowhy.toml](dowhy.toml) — DoWhy v0.14 pin and inventory
- [tigramite.toml](tigramite.toml) — Tigramite 5.2.1.25 pin and inventory
- [bayesian.toml](bayesian.toml) — Bayesian core inventory
- [gcm.toml](gcm.toml) — GCM / counterfactual inventory
- [pag.toml](pag.toml) — PAG / LPCMCI inventory
- [context.toml](context.toml) — Context / regime / effects inventory
- [attribution.toml](attribution.toml) — Attribution inventory
- [design_state.toml](design_state.toml) — Design / incremental-state inventory
- [release.toml](release.toml) — Release-prep / parity-closure inventory

Status values: `pending`, `in_progress`, `done`.

Unfinished DESIGN chapters stay `pending` and must appear on `TODO.md`. Permanent
product contracts (for example native GPDC, no DoWhy secondary package mirror)
are documented in DESIGN.md and marked `done`. There is no waiver / deviation
status.

Do not mark a capability `done` without conformance fixtures under
`conformance/` **or** a named calibration/unit harness recorded in the
corresponding feature gate script, plus a recorded reference-output generation
command where black-box comparison applies.

## Exit gates

```bash
bash scripts/gate_estimate_ci.sh
bash scripts/gate_bayesian.sh
bash scripts/gate_gcm.sh
bash scripts/gate_pag.sh
bash scripts/gate_context.sh
bash scripts/gate_attribution.sh
bash scripts/gate_design_state.sh
bash scripts/gate_release.sh
```

PAG: inventory (`pag.toml`), LPCMCI / latent-projection / envelope /
DAG-only-reject conformance; FCI/RFCI and full ID/IDC remain `pending`
(`TODO.md`).

Context: J/RPCMCI, effects, conditional ATE gated by `gate_context.sh`.
RPCMCI uses caller-supplied regime labels.

Release: artifact format 0.1 freeze, wheel matrix, conformance docs, hot-path
baselines, security review (`release.toml`, ADR 0017). Package version remains
0.1.0. Remaining DESIGN chapters are `pending` + `TODO.md`.
