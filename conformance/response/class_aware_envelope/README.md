# Class-aware Cpdag/Pag response envelope

The licensed cell is the generalized-adjustment envelope already used for ATE,
evaluated pointwise and mass-weighted. It is not MAG/PAG response ID.

## InterventionResponse (discrete ATE table)

Same contingency table as `pag_ate_envelope` / `cpdag_ate_envelope`. Completions
that adjust for `Z` identify `E[Y|do(T=t)] = 0.20 + 0.40 t`. The unadjusted
completion identifies `0.14` and `0.66`. Equal weights therefore pin ATE
contrasts `0.46` (CPDAG) and `0.44` (PAG).

Intervention g-computation recovers those levels up to the additive-GAM
plug-in. The recorded `do(0)` / `do(1)` values are output pins of that
estimator, and `do(1) − do(0)` is checked against the ATE contrast.

Kennedy-DR is singular on this binary treatment, so it is not the discrete
pin.

## ResponseCurve (continuous linear law)

`y = 0.10 + 0.40 t + 0.20 z` with continuous `t` so Kennedy-DR is defined.
The two-point grid is `[0, 1]`. Means are output pins; the contrast is
checked against AverageEffect on the same graph and table, with a declared
tolerance. The mixing rule is the claim, not bitwise equality with linear
ATE.

Consumers: `crates/antecedent/tests/class_aware_response_numeric_pins.rs`
and `python/tests/test_class_aware_response_numeric_pins.py`.

## Bayesian

Same envelope, `response.bayesian` per completion. Draws are not mixed;
the licensed number is the identified-mass mean. The two-point intervention
contrast is pinned against the CPDAG / PAG Bayesian ATE fixtures.
`estimate.envelope.response_posterior_not_mixed` discloses the omitted
posterior mix. cheap/full stay n/a.
