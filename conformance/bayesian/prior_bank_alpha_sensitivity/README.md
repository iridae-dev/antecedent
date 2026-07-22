# prior_bank_alpha_sensitivity

External prior-bank α-multiplier sensitivity grid on the ATE Bayesian facade
(`refute=Full` + `prior_from_composed`). Multiplier `0` is baseline-only; `1`
uses full applied α. Effect mean at `m=1` must sit closer to the banked
treatment coefficient than at `m=0`.

## Expected summary

Top-level keys: `n, n_draws, source_treatment_mean, source_coef_variance, alpha,
alpha_multipliers, require_finite_effect_means, m1_closer_to_source_than_m0,
notes`.
