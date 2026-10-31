# Bernoulli Horvitz–Thompson variance estimates

The unadjusted Bernoulli trial route now reports the same conservative observed-arm
variance estimator as the route with a fixed pre-assignment CUPED adjustment:

`sum(observed_outcome[i]^2 / observed_arm_probability[i]^2) / n^2`.

Its expectation over independent assignments exceeds the true design variance by
`sum((Y1[i] - Y0[i])^2) / n^2`. This is a guarantee in randomization expectation;
an individual realized estimate need not exceed the unknown true variance.
Fixed CUPED uses the adjusted potential outcomes in this identity.

The previous unadjusted expression omitted the cross-potential-outcome term and
was not a conservative estimator. For example, if both potential outcomes are 10
and assignment probability is one half, the true variance of the HT effect estimate is
`400 / n`, whereas the old expression reports `200 / n`.

Point estimates and the separately gated score-based standard errors and
intervals keep their existing calculation. A conservative variance estimate alone
does not authorize an interval.

New point-only trial artifacts use
`bernoulli_ht_conservative_variance_no_interval`. Historical artifacts carrying
`bernoulli_ht_design_variance_no_interval` remain readable with their original
numeric value and label. That historical value must not be interpreted as the
corrected conservative estimate; current encoding rejects the stale label. An
interval-bearing artifact cannot use this historical point-only exception.

The direct Python interference estimator uses a separate kernel and retains its
existing label. This correction concerns the retained Bernoulli trial route.
