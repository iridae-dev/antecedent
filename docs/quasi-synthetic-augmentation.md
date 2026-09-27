# Synthetic DiD and augmented panel DiD

These native-backed Python workflows provide point estimates outside the
support matrix. Neither returns an interval or a calibrated inference claim.

`SyntheticDifferenceInDifferences` runs through `analyze(data, query=query)`
and `PreparedAnalysis.prepare(data, query=query)`, reporting
`result.synthetic_did`. It requires a balanced panel with one treated unit, at
least two donors, two pre-periods and one post-period. It estimates simplex
weights over donor units and pre-treatment periods, then computes the weighted
difference-in-differences contrast. The caller must justify no anticipation,
no interference, and that the convex unit/time weights represent the untreated
counterfactual trend. Pre-fit RMSE is diagnostic only; there is no automatic
fit threshold or placebo-based interval.

The retained route reports `result.synthetic_did`, including donor and
pre-period weights. Unit and period row order is frozen when prepared; the
artifact preserves both weight vectors and refuses a fabricated standard error
or interval. This route remains `unlicensed_point_utility`.

`AugmentedPanelDiD`, through `analyze(data, query=query).panel_did`, requires
one row per subject, pre/post outcomes,
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

`AugmentedPanelDiD` also enters the ordinary retained flow:

```python
import antecedent
from antecedent.quasi import AugmentedPanelDiD

query = AugmentedPanelDiD("pre", "post", "subject", "treated", "propensity", "untreated_change")
prepared = antecedent.prepare(data, query=query)
result = prepared.estimate()
assert result.panel_did.uncertainty == "point_only_no_standard_error"
```

The prepared query freezes unique subject IDs, optional higher-level cluster
IDs, treatment assignments, and the names of the supplied nuisance columns.
Refreshing with a different subject or cluster order is refused.
`result.panel_did` reports the propensity range and effective weighted control
count; the artifact records no standard error and no interval for this design.
`predictions_cross_fitted=True` is a caller declaration, not a verification of
out-of-fold training. This point route remains outside the licensed support
matrix until its identification and inference evidence is sufficient.
