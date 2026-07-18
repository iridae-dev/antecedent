# Parity manifests

Domain capability inventories and pinned external baseline oracles. See DESIGN.md
§26 and ADR 0009.

**Inventories** (status: `pending` | `in_progress` | `done`):

- [estimate.toml](estimate.toml) — identify / estimate / refute
- [discovery.toml](discovery.toml) — temporal discovery / CI / graphs / effects
- [bayesian.toml](bayesian.toml) — Bayesian core
- [gcm.toml](gcm.toml) — GCM / counterfactual
- [pag.toml](pag.toml) — PAG / LPCMCI
- [context.toml](context.toml) — Context / regime / effects
- [attribution.toml](attribution.toml) — Attribution
- [design_state.toml](design_state.toml) — Design / incremental-state
- [release.toml](release.toml) — Release-prep / parity-closure

**Baseline pins** (oracle metadata only — not inventories):

- [baselines/dowhy.toml](baselines/dowhy.toml)
- [baselines/tigramite.toml](baselines/tigramite.toml)

Recorded black-box outputs live under `conformance/**/expected.json` in a
`reference` block (`project`, pin, command, `outputs`). Runtime and CI never
install upstream packages. Regeneration is out-of-repo; keep the frozen
`reference.command` as the audit trail.

Unfinished DESIGN chapters stay `pending` and must appear on `TODO.md`. Permanent
product contracts are documented in DESIGN.md and marked `done`. There is no
waiver / deviation status.

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
bash scripts/gate_upstream_names.sh
bash scripts/gate_calibration.sh
bash scripts/gate_release.sh
```

`gate_calibration.sh` is the DESIGN.md §28.3 statistical calibration suite
(SE coverage, CI Type I / permutation uniformity, discovery null FPR). It is
not part of every-PR unit CI; run locally before release, or via the weekly
GitHub Actions workflow [`.github/workflows/calibration.yml`](../.github/workflows/calibration.yml)
(`schedule` + `workflow_dispatch`).

PAG: inventory (`pag.toml`), LPCMCI / latent-projection / envelope /
DAG-only-reject conformance; FCI/RFCI and full ID/IDC remain `pending`
(`TODO.md`).

Context: J/RPCMCI, effects, conditional ATE gated by `gate_context.sh`.
RPCMCI uses caller-supplied regime labels.

Release: artifact format 0.1 freeze, wheel matrix, conformance docs, hot-path
baselines, security review (`release.toml`, ADR 0017). Package version remains
0.1.0. Remaining DESIGN chapters are `pending` + `TODO.md`.
