# Synthetic control point utility

`antecedent.quasi.SyntheticControl` runs through graphless `analyze` and
`PreparedAnalysis.prepare`, with `estimate_synthetic_control` retained as a
direct utility. Both use the same native Rust calculation for one treated unit
in a balanced panel. The pre-period fit chooses nonnegative donor weights that sum to one by
projected gradient descent on squared outcome error. The reported effect is
the treated unit's mean post-period outcome minus the weighted donors' mean.

The design requires one treated unit, at least three donor units, two or more
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

The retained result is available as `result.synthetic_control`. The prepared
study freezes unit and period row order; refreshing with changed design labels
is refused. The result and point-only uncertainty semantics round-trip through
the contracted artifact. Synthetic DiD has its own retained result section;
augmented synthetic control, covariate adjustment, and calibrated placebo
inference remain unavailable. Neither route adds a support-matrix license.
