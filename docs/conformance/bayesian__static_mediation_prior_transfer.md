# Static mediation prior transfer

**Suite path:** `conformance/bayesian/static_mediation_prior_transfer`

EffectFunctional maps ATE/Δ onto the outcome-mechanism NDE slope only
(`filter_compatible` is not a class prior). Other mechanisms keep
isotropic `prior_scale`. A shared `PriorSet` or an artifact without a
mapping is refused.

| Pin | Source | Target | Filter |
|---|---|---|---|
| `mapped_nde` | AverageEffect × Dag × explicit × Bayesian × none | MediationEffect × Dag × explicit × Bayesian × none | `EffectFunctional` ATE/Δ onto outcome mechanism |
| `missing_mapping` | AverageEffect artifact | MediationEffect × Dag | typed refuse |

## Expected summary

Top-level keys: `compatibility_filter, n_draws, schema_version, seed, source_cell, target_cells` (6 fields).
