# Synthetic DiD and augmented panel DiD utilities

These native-backed Python workflows provide point estimates outside the
support matrix. Neither returns an interval or a calibrated inference claim.

`estimate_synthetic_did` requires a balanced panel with one treated unit, at
least two donors, two pre-periods and one post-period. It estimates simplex
weights over donor units and pre-treatment periods, then computes the weighted
difference-in-differences contrast. The caller must justify no anticipation,
no interference, and that the convex unit/time weights represent the untreated
counterfactual trend. Pre-fit RMSE is diagnostic only; there is no automatic
fit threshold or placebo-based interval.

`SyntheticDifferenceInDifferences` also works with `analyze(data, query=query)`
and `PreparedAnalysis.prepare(data, query=query)`. The retained route executes
the same Rust kernel and reports `result.synthetic_did`, including donor and
pre-period weights. Unit and period row order is frozen when prepared; the
artifact preserves both weight vectors and refuses a fabricated standard error
or interval. This route remains `unlicensed_point_utility`.

`estimate_augmented_panel_did` requires one row per subject, pre/post outcomes,
a binary treatment indicator, propensity scores, and predictions of the
untreated outcome change. It uses an ATT augmentation: the treated mean change
minus the predicted treated counterfactual change, corrected by propensity
odds-weighted control residuals. Propensities must be strictly inside (0, 1).
Nuisance models are supplied by the caller rather than fitted by the utility;
the result records whether the caller declares predictions cross-fitted but
does not verify fold ownership. Identification still requires conditional
parallel trends or a correct untreated-change model, and a correct propensity
model or outcome model. Effective weighted control sample size is reported as
a support diagnostic, not a license or inference guarantee.
