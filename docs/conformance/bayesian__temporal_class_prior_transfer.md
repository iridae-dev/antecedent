# Temporal class prior transfer

**Suite path:** `conformance/bayesian/temporal_class_prior_transfer`

Same-design and mapped mechanism-prior transfer onto licensed 1.7 Bayesian
cells on `TemporalCpdag`. The prior binds separately against each
completion's design through the existing `PriorCatalog.filter_compatible`
filter. It is a mechanism prior, not a class prior: it cannot upgrade
identification or invent class probabilities, and a conflict diagnostic does
not change identification status.

The stationary linear law is the two-completion class envelope law
(`z` alternates 0/1; `t = 0.3 + 0.4 z + 0.05 sin(0.017 i)`;
`y = 1 + 2 t_{i-1} + 0.5 z_{i-1}`) with template edges `z@1 -> y@0`,
`t@1 -> y@0`, and the undirected mark `z@1 — t@1`.

| Pin | Source cell | Target cell | Filter |
|---|---|---|---|
| `same_design_pulse` | PulseEffect × TemporalCpdag × explicit × Bayesian × none | same | `filter_compatible` / identical-subspace hydrate per completion |
| `mapped_coefficient_sustained` | same Pulse source | SustainedEffect × TemporalCpdag × explicit × Bayesian × none | `filter_compatible` + `IdenticalCoefficientSubspace` |
| `mapped_coefficient_response_curve` | same Pulse source | ResponseCurve × TemporalCpdag × explicit × Bayesian × none | `filter_compatible` + `IdenticalCoefficientSubspace` |
| `mapped_effect_functional` | same Pulse source | PulseEffect × TemporalCpdag × explicit × Bayesian × none | `EffectFunctional` |
| `mapped_mediation` | same Pulse source | TemporalMediationEffect × TemporalCpdag × explicit × Bayesian × none | `filter_compatible` per mechanism |
| `sequence_refuses_transfer` | — | InterventionResponse multi-step Sequence × TemporalCpdag × explicit × Bayesian × none | refuse: sequential mechanisms keep isotropic per-mechanism priors |

Consumed by `temporal_class_prior_transfer_conflict_does_not_flip_identification`,
`temporal_class_sequence_refuses_prior_transfer`, and the mapped-transfer pins
in `crates/antecedent/tests/temporal_class_bayesian_envelope.rs`, plus
`python/tests/test_temporal_class_bayesian_envelope.py`.

## Expected summary

Top-level keys: `compatibility_filter, conflict_does_not_flip_identification, fixture_id, n, n_draws, notes, outcome, seed, sequence_refuses, source_cells, target_cells, treatment` (12 fields).
