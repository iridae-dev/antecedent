# Static response prior transfer

**Suite path:** `conformance/bayesian/static_response_prior_transfer`

Response-specific coefficient mappings (dose / intervention coordinate to
outcome-regression coefficients) ride licensed Bayesian `ResponseCurve` and
`InterventionResponse` on explicit Dag and Cpdag/Pag class envelopes.

A catalog without a declared mapping is refused. Incompatible catalogs must
not fall back to isotropic priors.

| Pin | Source | Target | Filter |
|---|---|---|---|
| `same_design_curve` | ResponseCurve × Dag × explicit × Bayesian × none | same | identical subspace |
| `class_envelope_curve` | ResponseCurve × Cpdag × explicit × Bayesian × none | per completion | identical subspace; not a class prior |
| `missing_mapping` | AverageEffect artifact | ResponseCurve | typed refuse, no isotropic fallback |
