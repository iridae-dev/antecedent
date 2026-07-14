# Parity manifests

Pinned reference baselines and assignable capability inventories for DoWhy and
Tigramite. See DESIGN.md §26 and ADR 0009. Bayesian core inventory is tracked
separately (not a DoWhy/Tigramite surface).

- [dowhy.toml](dowhy.toml) — DoWhy v0.14 pin and inventory
- [tigramite.toml](tigramite.toml) — Tigramite 5.2.1.25 pin and inventory
- [bayesian.toml](bayesian.toml) — Phase 6 Bayesian core inventory
- [gcm.toml](gcm.toml) — Phase 7 GCM / counterfactual inventory
- [pag.toml](pag.toml) — Phase 8 PAG / LPCMCI inventory
- [phase4_deviations.md](phase4_deviations.md) — Phase 4 kept deferrals
- [phase5_deviations.md](phase5_deviations.md) — Phase 5 kept deferrals
- [phase6_deviations.md](phase6_deviations.md) — Phase 6 kept deferrals
- [phase7_deviations.md](phase7_deviations.md) — Phase 7 kept deferrals (Shapley → P10)
- [phase8_deviations.md](phase8_deviations.md) — Phase 8 kept deferrals (FCI/ID → later)
- [phase9.toml](phase9.toml) — Phase 9 context / regime / effects inventory
- [phase9_deviations.md](phase9_deviations.md) — Phase 9 kept deferrals (Shapley → P10)

Status values: `pending`, `in_progress`, `done`, `intentional_deviation`.

Do not mark a capability `done` without conformance fixtures under
`conformance/` **or** a named calibration/unit harness recorded in
`scripts/gate_phase45_parity.sh` / `scripts/gate_phase6.sh` /
`scripts/gate_phase7.sh` / `scripts/gate_phase8.sh`, plus a recorded
reference-output generation command where black-box comparison applies.

## Exit gates

```bash
bash scripts/gate_phase45_parity.sh
bash scripts/gate_phase6.sh
bash scripts/gate_phase7.sh
bash scripts/gate_phase8.sh
bash scripts/gate_phase9.sh
```

Phase 8: PAG inventory (`pag.toml`), LPCMCI / latent-projection / envelope /
DAG-only-reject conformance, m-separation + PAG orientation sparse/stress
benches; FCI/RFCI and full ID/IDC deferred (`phase8_deviations.md`).

Phase 9: J/RPCMCI, effects, conditional ATE gated by `gate_phase9.sh`.
Kept deferrals: Shapley/mechanism-change→P10, design/state→P11, FCI/RFCI, native GPDC (no torch);
Phase 6: Bayesian discovery, Stan/PyMC, hierarchical/BVAR/GP; Phase 7→10:
Shapley attribution; Phase 8→later: FCI/RFCI, full ID/IDC.
