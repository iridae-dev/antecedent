# Shared mathematics and evidence review — 2026-09-13

> Historical evidence or review record. Findings and version references describe
> the work recorded here. For 1.10 behavior, use the [support matrix](support-matrix.md)
> and [1.10 release notes](release-notes/v1.10.0.md).

This pass follows the 1.7 temporal-class review and examines shared numerical
inference, discovery statistics, regression/spline machinery, and validation.
The implementation changes preserve existing licensed capabilities. They are
separate from the concurrent 1.7 cleanup.

## Corrections

| Finding | Implemented correction | Regression evidence |
| --- | --- | --- |
| Rank transforms declared distinct values tied whenever their difference was below `1e-15`. | Only equal observations share a midrank. Measurement units no longer change the ordering. | `[3, 1, 1, 2] × 1e-20` produces `[4, 1.5, 1.5, 3]`, rather than four identical ranks. |
| QR's rank tolerance included an absolute floor of one, and column units affected numerical rank. | Equilibrate design columns, assess rank relative to the largest QR pivot, and transform coefficients back to their original units. | The same full-rank regression recovers its coefficients with the entire design or just its predictor scaled by `1e-30`, `1`, and `1e30`. |
| Spline knot construction treated small ranges as constant; derivative recurrences zeroed small positive knot spacings; endpoint evaluation subtracted an absolute epsilon. | Distinguish zero spacing from small spacing and evaluate endpoints on the final span. Preserve caller-supplied non-clamped knot vectors. | Basis values and chain-rule derivatives agree across scales `1e-20`, `1`, and `1e20`, including boundaries; a cardinal cubic boundary has its analytic basis values. |
| Logistic likelihood values and scores lost representable tail probabilities, while curvature was floored at `1e-12`. | Stable softplus, complementary probabilities, scores, and actual curvature use the same Bernoulli likelihood. | At predictors `±40`, log probability, score, and curvature retain the analytic `exp(-40)` tail. Existing finite-difference likelihood tests remain in place. |
| Gaussian likelihood helpers silently substituted `1e-12` for smaller requested variances. | Use the requested positive variance and its precision in both values and derivatives. Numerically unrepresentable precision is reported explicitly. | A variance of `1e-20` yields the analytic likelihood, score, and Hessian. |
| Bayesian CI computed independence mass by subtracting a near-one dependence probability. | Compute the logistic transform of the negative log Bayes factor directly. | The existing dependent-data test now checks the strictly positive, representable posterior tail against the Bayes factor. |
| Prior-sensitivity comparisons used an absolute denominator floor that could turn a failed check into a pass after a unit change. | Normalize by the actual effect scale, handle an all-zero grid explicitly, and mark nonfinite/empty results uninformative. | An effect grid `[1, 2]` has relative range `0.5` and the same verdict at scales from `1e-100` to `1e100`. |
| Predictive mixture checks averaged two-sided probabilities, losing tail direction; their variance used subtraction of large second moments. | Retain both inclusive tails and mix them before forming the two-sided probability. Compute spread from centered within/between-component terms using `hypot`, with normalized weights. | Opposing point-mass components give a central mixture check; SD remains `sqrt(5)` around location `1e12`, for weights from `1e-200` to `1e300`. |

The rank, QR, spline-scale, likelihood, sensitivity, and predictive-mixture
counterexamples were observed failing before their fixes. Tests are
in the corresponding source modules under `antecedent-stats`, `antecedent-prob`,
and `antecedent-validate`; their names start with `review_`, except for the
additional assertion on the existing Bayesian-CI test.

## Interpretation

Posterior predictive tail probabilities remain model checks, not frequentist
calibration guarantees. Graph weights remain caller/model inputs; this pass does
not infer graph probabilities from enumeration. Rank and regression tests check
particular invariances, not a proof of every discovery or identification result.
No support-matrix entries were narrowed by this review.

`PredictiveCheckReport` now retains `location_tails` and `dispersion_tails` so
mixture checks can be computed correctly. Existing scalar report fields remain;
downstream Rust code constructing reports with struct literals must supply the
two new arrays.

## Validation

- Current shared-crate unit suites (`antecedent-prob`, `antecedent-stats`,
  `antecedent-validate`): **399 passed, 12 ignored**.
- Full Rust workspace testing exposed a repeated-knot endpoint regression during
  the review. After correcting it, all three affected integration suites passed:
  `class_aware_response_numeric_pins` (3), `joint_class_response` (5), and
  `v15_numeric_pins` (70 passed, 1 ignored). The full workspace was not rerun
  monolithically after this final endpoint fix; its remaining tests passed.
- Optimized native extension rebuilt with
  `maturin develop --skip-install --release`; full Python suite:
  **1,385 passed**.
- Clippy with warnings denied passed for the three changed crates and their
  targets. Formatting checks for all changed Rust files and `git diff --check`
  passed.
- The workspace-wide Clippy run also encountered three warnings in the concurrent
  temporal cleanup: `ClassObservationKind`'s large enum variant in
  `analysis/execute/temporal_path.rs`, and missing documentation backticks in
  `strategy_table/dispatch.rs`. Those files were left to the concurrent work.

Counts reflect the shared working tree during concurrent development. Ignored
tests, broader calibration studies, and cross-platform builds were not exercised
by this pass.

## Follow-up corrections (same day)

A second pass on the temporal-class surface and the shared kernels found four
remaining defects. Licensed cells were kept; the implementations were corrected
so those cells earn their existing contracts.

| Finding | Implemented correction | Regression evidence |
| --- | --- | --- |
| Class-prior response / Sequence / mediation published `conditional_on_identified` whenever a prior bound, including when `unidentified_mass > 0`. The mean renormalized over identified atoms. | `temporal_class_response_mean` withholds the summary unless the class is fully identified and evaluable. | Mixed-ID TemporalPag curve with a `ClassPrior` retains unidentified mass and omits the conditional surface. |
| GAM auto-λ used uncentered `Bβ` in RSS and `tr(S₁)` in GCV, while the fit applies the centered smoother with `tr(S₁) − 1`. | GCV scores centered predictions and `edf = tr(S₁) − 1`. | Selected λ is the argmin of that centered score. |
| Weighted Pearson (and Bayesian residual \|r\|) treated `sqrt(cxx cyy) ≤ ε` as undefined, so a unit change could hide a defined coefficient. | Degeneracy is a non-positive variance, not an absolute product floor. | `[1, 1+1e-9, 1+2e-9]` keeps the same r at scale `1e-4`. |
| Multivariate leading ρ came from a ridged CCA path; Wilks Λ used unregularized whitening. | ρ₁ is the leading Gram eigenvalue from the Wilks whitening. | For `px=1`, `ρ = sqrt(1 − Λ)` holds to working precision. |
