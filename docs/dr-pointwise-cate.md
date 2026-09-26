# DR-Learner pointwise CATE profiles

`DrLearner::fit_pointwise_profiles` provides a deliberately narrow pointwise CATE route for an
unpenalized linear final stage. Each profile lists every adjustment variable in the exact order
stored by the prepared problem. It is accepted only when the complete covariate tuple appears
exactly among retained observations in both treatment arms and every matching row passes the
configured propensity clip. Profiles outside this empirical joint support are refused; the API
does not extrapolate or infer support from coordinate-wise ranges.

The estimand is the best linear projection of the CATE onto the adjustment variables (Semenova and Chernozhukov), evaluated at the profile; it equals the true pointwise CATE only under a correctly specified linear or saturated final stage. The returned estimate is the linear prediction of the cross-fitted doubly robust score. Its
standard error is the HC0 sandwich evaluated at that profile from the same score fit. The
`lower_95` and `upper_95` fields are pointwise normal-approximation bounds. These bounds are
**uncalibrated**: no coverage record or calibration claim is attached to this API. They do not
provide simultaneous coverage across profiles. The existing rowwise `EffectEstimate::cate`,
`cate_se`, and forest leaf dispersion remain separate outputs and must not be substituted for
the profile result.

The exact-support restriction is useful for categorical or otherwise repeated covariate
profiles. Continuous profiles with no exact repeated tuple are refused. A future support rule
for continuous profiles requires separately specified joint-support semantics and validation.
