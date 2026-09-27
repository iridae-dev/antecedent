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
the contracted artifact. Synthetic DiD has its own retained result section.

When the treated unit was selected uniformly from the observed units before
outcomes were seen, set `uniform_unit_randomization=True` on `SyntheticControl`.
The retained native path then refits each possible treated-unit assignment
against all other units. It reports every absolute post-period gap and their
exact two-sided tail fraction as `randomization_p_value`. This is a Fisher
sharp-null test under the declared assignment mechanism, with no interval for
the effect estimate. It is distinct from the descriptive donor-only placebo
rank. The artifact binds the assignment declaration and rejects a p-value
that disagrees with the saved assignment statistics. The exact enumeration is
currently limited to 32 candidate units; synthetic DiD refuses this option.
The route remains off the support-matrix axis.

Set `augmentation_ridge` to a finite positive penalty to correct the simplex
gap with a donor-trained outcome model. The native route fits a centered ridge
model from each donor's pre-period outcomes to its mean post-period outcome,
then subtracts the treated-versus-weighted-donor prediction difference. It
reports `unadjusted_effect`, `outcome_model_correction`, and the adjusted
`estimate`. This can correct the simplex fit when the treated unit lies outside
the donor convex hull, under the additional declared assumption that the donor
outcome model transports to the treated unit. The displayed placebo rank still
describes the unadjusted simplex fit. With declared uniform one-unit assignment,
the exact Fisher test refits both the donor weights and ridge correction for
every candidate unit and tests the adjusted effect under the sharp null. The
adjusted effect has no interval and remains `unlicensed_point_utility` off the
support-matrix axis. The ridge penalty is fixed by the caller; it is not tuned
from the panel. Augmentation cannot be combined with synthetic DiD. The penalty and correction round-trip through the retained
artifact, which rejects inconsistent saved effect fields.

```python
from antecedent import analyze
from antecedent.quasi import SyntheticControl

result = analyze(
    panel,
    query=SyntheticControl(
        "outcome", "unit", "period", "treated", 5,
        augmentation_ridge=1.0,
    ),
)
adjusted_effect = result.synthetic_control.estimate
```

Additional unit-level covariate adjustment remains unavailable.
