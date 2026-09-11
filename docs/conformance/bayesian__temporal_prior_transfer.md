# Temporal prior transfer

**Suite path:** `conformance/bayesian/temporal_prior_transfer`

**Suite path:** `conformance/bayesian/temporal_prior_transfer`

Same-design and mapped prior transfer ride licensed Bayesian temporal cells
on the staged path (`prepare` → estimate). They are not a matrix axis.

Each pin names a **source cell**, a **target cell**, and the existing
`PriorCatalog.filter_compatible` filter. Incompatible catalogs fail closed.
Conflict-sensitive weights stay diagnostic — they do not upgrade identification.

The stationary linear SCM is

```text
X_s = sin(0.04 s)
Y_s = 0.9 X_{s-1} + ε_s
```

with template edge `X@lag1 -> Y@lag0`. Pulse / single-step Sustained at `-1`,
horizon `1`. Temporal `ResponseCurve` uses the same unfolded linear design.

| Pin | Source cell | Target cell | Filter |
|---|---|---|---|
| `same_design_pulse` | PulseEffect × TemporalDag × explicit × Bayesian × none | PulseEffect × TemporalDag × explicit × Bayesian × none | `filter_compatible` / `require_usable` (Partial if durable coef names are absent; identical-subspace hydrate) |
| `same_design_sustained` | SustainedEffect × TemporalDag × explicit × Bayesian × none | SustainedEffect × TemporalDag × explicit × Bayesian × none | `filter_compatible` / `require_usable` |
| `same_design_response_curve` | PulseEffect × TemporalDag × explicit × Bayesian × none | ResponseCurve × TemporalDag × explicit × Bayesian × none | `filter_compatible` + declared identical mapping |
| `mapped_effect_pulse` | PulseEffect × TemporalDag × explicit × Bayesian × none | PulseEffect × TemporalDag × explicit × Bayesian × none (extra W) | `require_usable` + EffectFunctional |
| `incompatible_catalog` | PulseEffect (wrong outcome) | PulseEffect × TemporalDag × explicit × Bayesian × none | `filter_compatible` → `estimand_mismatch`, refuse |
| `sequence_refuses_transfer` | — | InterventionResponse multi-step Sequence × TemporalDag × explicit × Bayesian × none | same filter does not apply to per-mechanism sequential priors; refuse, no new filter |

Mapped Sustained and ResponseCurve reuse the Pulse source artifact and the
same EffectFunctional filter.

## Expected summary

Top-level keys: `atol, compatibility_filter, fixture_id, grid, horizon_steps, horizons, incompatible, n, n_draws, notes, outcome, seed, sequence_refuses, source_cells, target_cells, treatment, treatment_lag, true_effect` (18 fields).
