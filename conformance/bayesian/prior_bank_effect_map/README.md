# prior_bank_effect_map

Cross-design effect-functional prior transfer. Source A (confounder Z) yields a
known ATE posterior; target B adds an extra covariate W. With
`EffectFunctional { source_quantity = "ate" }`, the target posterior mean is
pulled toward A's effect relative to a weakly informative baseline, and the
assumption record includes `external_effect_prior`.
