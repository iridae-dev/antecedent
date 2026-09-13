# Static effect prior transfer

**Suite path:** `conformance/bayesian/static_effect_prior_transfer`

Same-design and mapped prior transfer ride licensed Bayesian
`AverageEffect` / `ConditionalEffect` on explicit Dag and Cpdag/Pag class
envelopes via the staged path. They are not a matrix axis.

Incompatible catalogs fail closed (`estimand_mismatch`). Mechanism /
coefficient priors are not a class prior and must not upgrade identification.

| Pin | Source | Target | Filter |
|---|---|---|---|
| `same_design_ate` | AverageEffect × Dag × explicit × Bayesian × none | same | `filter_compatible` + identical subspace |
| `mapped_cate` | AverageEffect × Dag × explicit × Bayesian × none | ConditionalEffect × Dag × explicit × Bayesian × none | `EffectFunctional` |
| `class_envelope_ate` | AverageEffect × Cpdag × explicit × Bayesian × none | same per completion | identical subspace; not a class prior |
| `incompatible_catalog` | AverageEffect (wrong outcome) | AverageEffect × Dag | `estimand_mismatch`, refuse |
