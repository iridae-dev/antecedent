# graph_effect_envelope

**Suite path:** `conformance/bayesian/graph_effect_envelope`

Weighted graph ensemble with known unidentified fraction. Published effect
moments and draws are E[τ | identified] (identified-atom BMA). Unidentified
mass is retained as a separate, non-renormalized axis.
`renormalize_identified_only` refuses to publish a 100% mixture after
dropping that mass.

## Expected summary

Top-level keys: `effect_means, expected_mixture_mean, identified_weights, tolerance_class, unidentified_mass` (5 fields).
