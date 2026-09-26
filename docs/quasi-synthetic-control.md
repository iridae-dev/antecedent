# Synthetic control point utility

`antecedent.quasi.SyntheticControl` and `estimate_synthetic_control` provide a
native Rust synthetic-control calculation for one treated unit in a balanced
panel. The pre-period fit chooses nonnegative donor weights that sum to one by
projected gradient descent on squared outcome error. The reported effect is
the treated unit's mean post-period outcome minus the weighted donors' mean.

The function requires one treated unit, at least three donor units, two or more
pre-periods, an observed intervention period, and identical periods for every
unit. It returns donor weights, pre-fit RMSE, effective donor count, and
leave-one-donor-out placebo effects. Its placebo tail fraction compares
post/pre-RMSE ratios; it assumes exchangeable units, is not calibrated, and is
not a confidence interval or licensed hypothesis test. The causal effect is
point-only and the result is marked `unlicensed_point_utility`.

Identification depends on no anticipation, stable treatment after adoption,
no concurrent treated-unit-specific shock, no interference, and a convex
combination of observed donors being a valid untreated counterfactual. The
fit RMSE is reported without an acceptance threshold; a small pre-fit error
does not verify those assumptions.

This utility does not implement augmented synthetic control, synthetic DiD,
covariate adjustment, or calibrated placebo inference. It does not add a
support-matrix license.
