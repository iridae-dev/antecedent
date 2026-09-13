# Static mediation prior transfer

**Suite path:** `conformance/bayesian/static_mediation_prior_transfer`

Mapped coefficient artifacts hydrate independently onto each linear
mechanism (`filter_compatible` is not a class prior). A shared
`PriorSet` or an artifact without a mapping is refused.

| Pin | Source | Target | Filter |
|---|---|---|---|
| `mapped_nde` | AverageEffect × Dag × explicit × Bayesian × none | MediationEffect × Dag × explicit × Bayesian × none | `EffectFunctional` per mechanism |
| `missing_mapping` | AverageEffect artifact | MediationEffect × Dag | typed refuse |
