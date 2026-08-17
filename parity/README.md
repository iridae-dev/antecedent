# Parity manifests

Domain capability inventories and pinned external baseline oracles. See ADR 0009
and [docs/development.md](../docs/development.md).

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
- [response.toml](response.toml) — 0.5.0 response / observation / transport / interference

Cross-language names: [docs/api_naming.md](../docs/api_naming.md).

**Baseline pins** (oracle metadata only — not inventories):

- [baselines/dowhy.toml](baselines/dowhy.toml)
- [baselines/tigramite.toml](baselines/tigramite.toml)
- [baselines/bpbounds.toml](baselines/bpbounds.toml)
- [baselines/causaleffect.toml](baselines/causaleffect.toml)
- [baselines/riskRegression.toml](baselines/riskRegression.toml)

## Capability row schema

Each `[[capabilities]]` row uses:

| Field | Required | Values |
|-------|----------|--------|
| `id` | yes | Dotted capability id, unique within the manifest |
| `status` | yes | `pending` \| `in_progress` \| `done` |
| `group` | see below | Grouping key |
| `description` | see below | One-line capability description |
| `owner` | see below | Owning domain |
| `notes` | no | Free-form evidence / gate pointers |
| `python_facade` | no | `full` \| `thin` |

**`group` / `description` / `owner`** are required in
[estimate.toml](estimate.toml), [discovery.toml](discovery.toml), and
[context.toml](context.toml), and unused elsewhere.

The required-key contract is enforced by
[`scripts/gate_parity_schema.sh`](../scripts/gate_parity_schema.sh), which every
feature gate and `gate_release.sh` invoke. It exists because the gates' inline
`caps()` parser reads keys with a regex default — an absent `status` yields
`None`, matches no honesty check, and would otherwise pass forever without the
row ever being marked. The schema gate also pins that regex against a real TOML
parse, and enrolls new manifests automatically: any `parity/*.toml` carrying
`[[capabilities]]` rows that is not listed in the gate fails it.

**`python_facade`:** When a capability is Rust-done but the Python surface is
incomplete or `_native`-only without a typed facade, set `python_facade =
"thin"`. Use `"full"` when the public `antecedent` package exposes the capability
end-to-end (analyze kwargs, typed wrappers, or dedicated helpers). Older rows
may still embed `python_facade=full` inside `notes`; prefer the dedicated key
for new / updated rows.

Recorded black-box outputs live under `conformance/**/expected.json` in a
`reference` block (`project`, pin, command, `outputs`). Runtime and CI never
install upstream packages. Regeneration is out-of-repo; keep the frozen
`reference.command` as the audit trail.

Permanent product contracts are marked `done` with an inline note. There is no
waiver / deviation status. Required 1.0 chapters are closed in the inventories
below.

Do not mark a capability `done` without conformance fixtures under
`conformance/` **or** a named calibration/unit harness recorded in the
corresponding feature gate script, plus a recorded reference-output generation
command where black-box comparison applies.

## Exit gates

```bash
bash scripts/gate_parity_schema.sh
bash scripts/gate_estimate_ci.sh
bash scripts/gate_bayesian.sh
bash scripts/gate_gcm.sh
bash scripts/gate_pag.sh
bash scripts/gate_context.sh
bash scripts/gate_attribution.sh
bash scripts/gate_design_state.sh
bash scripts/gate_upstream_names.sh
bash scripts/gate_calibration.sh
bash scripts/gate_response_calibration.sh
bash scripts/gate_release.sh
```

`gate_calibration.sh` is the statistical calibration suite (SE coverage, CI
Type I / permutation uniformity, discovery null FPR). It is not part of
every-PR unit CI; run locally before release, or via the weekly GitHub Actions
workflow [`.github/workflows/calibration.yml`](../.github/workflows/calibration.yml)
(`schedule` + `workflow_dispatch`).

PAG: inventory (`pag.toml`), LPCMCI / latent-projection / envelope /
DAG-only-reject conformance; static FCI/RFCI `done`. Permanent: PAG-native
ID is generalized adjustment only; full ID/IDC needs MAG/ADMG completion.

Context: J/RPCMCI, effects, conditional ATE gated by `gate_context.sh`.
RPCMCI uses caller-supplied regime labels; unsupervised regime search is OOS.

Release: artifact format freeze, wheel matrix, conformance docs, hot-path
baselines, security review (`release.toml`, ADR 0017). Package version is
whatever `[workspace.package].version` in `Cargo.toml` says — do not restate it
here; a hardcoded copy in this file sat five minor releases behind for months.
