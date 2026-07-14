# Parity manifests

Pinned reference baselines and assignable capability inventories for DoWhy and
Tigramite. See DESIGN.md §26 and ADR 0009. Bayesian core inventory is tracked
separately (not a DoWhy/Tigramite surface).

- [dowhy.toml](dowhy.toml) — DoWhy v0.14 pin and inventory
- [tigramite.toml](tigramite.toml) — Tigramite 5.2.1.25 pin and inventory
- [bayesian.toml](bayesian.toml) — Phase 6 Bayesian core inventory
- [gcm.toml](gcm.toml) — Phase 7 GCM / counterfactual inventory
- [phase4_deviations.md](phase4_deviations.md) — Phase 4 kept deferrals
- [phase5_deviations.md](phase5_deviations.md) — Phase 5 kept deferrals
- [phase6_deviations.md](phase6_deviations.md) — Phase 6 kept deferrals
- [phase7_deviations.md](phase7_deviations.md) — Phase 7 kept deferrals (Shapley → P10)

Status values: `pending`, `in_progress`, `done`, `intentional_deviation`.

Do not mark a capability `done` without conformance fixtures under
`conformance/` **or** a named calibration/unit harness recorded in
`scripts/gate_phase45_parity.sh` / `scripts/gate_phase6.sh` /
`scripts/gate_phase7.sh`, plus a recorded
reference-output generation command where black-box comparison applies.

## Phase 4 / 5 coverage class

| Surface | Parity class | Reference |
|---------|--------------|-----------|
| DoWhy linear Gaussian ATE (P1) | StableFloat | Analytic + black-box DoWhy 0.14 when fixture regenerated |
| Phase 4 estimators | Approximate (tolerance in `expected.json`) | Clean-room SCMs in `conformance/phase4/*` |
| Phase 4 refuters / sensitivity | Exact validator-id set | `conformance/phase4/refuters` + smoke harness |
| PCMCI lag-1 (P2) | Exact parents | True parents + black-box Tigramite 5.2.1.30 when available |
| PCMCI+ lag-0 | Exact parents (subset) | Clean-room only (`intentional_deviation`) |
| Phase 5 CI suite | Calibration / dependence-ordering | `ci::calibration` (+ native GPDC deviation) |

## Exit gates

```bash
bash scripts/gate_phase45_parity.sh
bash scripts/gate_phase6.sh
bash scripts/gate_phase7.sh
```

Verified locally (2026-07-21): inventory evidence map, Phase 4 conformance,
DoWhy StableFloat vs refreshed black-box estimate ≈ 2.0, PCMCI Exact parents
vs recorded Tigramite recovered set, PCMCI+ Exact clean-room, CI calibration,
Phase 4 reuse gate.

Phase 6: Bayesian inventory evidence map, conjugate/Laplace/g-comp/envelope/PPC
fixtures, Laplace + posterior-functional criterion benches.

Phase 7: GCM inventory (`gcm.toml`), fit/intervene/anomaly/CF ITE conformance,
do-samplers, overlay + CF batch criterion benches; Shapley / distribution-change
attribution deferred to Phase 10 (`phase7_deviations.md`).

Kept deferrals only: conditional/mediation→P9, PAG/LPCMCI→P8,
J/RPCMCI→P9, native GPDC (no torch), clean-room PCMCI+ pin; Phase 6: Bayesian
discovery, Stan/PyMC, hierarchical/BVAR/GP, MCMC diagnostics; Phase 7→10:
Shapley and distribution/mechanism-change attribution.
