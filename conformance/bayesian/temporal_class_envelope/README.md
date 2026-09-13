# Temporal class Bayesian envelope

Two-completion `TemporalCpdag` pulse under the 1.4 class-envelope law.

Without a caller-supplied class prior the result is an identified set plus
atom posteriors. Enumeration weights are not mixed. With
`ClassPrior.from_ordered([0.3, 0.7])` the blended posterior uses
`aggregate_effect_envelope` and retains unidentified mass when present.

The multi-step section pins a two-lag window against last-step collapse.
