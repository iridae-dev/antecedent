# Context / regime / effects deviations

Intentional waivers relative to `parity/context.toml`, `parity/tigramite.toml`,
and `parity/dowhy.toml` for contextual, regime, mediation, and conditional-effect
surfaces. Attribution and design/state surfaces are tracked separately.

## 1. Full nonparametric natural-effect ID

Linear temporal mediation (identify + estimate; NDE/NIE under linear SEM) is
`done`. Exotic nonparametric path-specific variants beyond that surface remain
waived. Tracked as `intentional_deviation` on `context.mediation.nonparametric`.

## 2. RPCMCI regime assignment mode

RPCMCI accepts an external `RegimeAssignment` (e.g. `two_regime_half_split`)
and fits one graph per regime. Full Tigramite-style unsupervised regime search
is not claimed; callers supply typed regime labels. Scope note on
`tigramite.discovery.rpcmci` / `context.rpcmci` (status `done` with that scope).

## 3. FCI / RFCI

Still waived (see `pag_deviations.md`).

## Verification

`tigramite.discovery.jpcmci_plus`, `tigramite.discovery.rpcmci`,
`tigramite.effects`, and `dowhy.estimate.conditional` are `done` with evidence
mapped in `scripts/gate_context.sh`.
