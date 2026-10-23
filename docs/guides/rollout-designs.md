# Rollout designs: staggered DiD, synthetic control with intervals, RD in time

These routes extend the difference-in-differences, synthetic-control and local-ratio queries
to real rollouts. Each one runs as a compiled route. Its uncertainty comes from refitting the
whole procedure, and its intervals are **unlicensed with calibration `unmeasured`**: no
coverage record measures them yet. Results are carried on the `quasi_rollout` payload
(`QuasiRolloutEstimate` in Rust, `result.quasi_rollout` in Python). The first effect in the
payload is the scalar summary. The others are group-time, event-time, cohort, calendar,
pre-trend, per-period or per-unit effects, depending on the design.

Each design records its own assignment, pre-trend and target-population assumptions in the
identification's assumption set. A generic time-series regression is never substituted for
them.

## Heterogeneity-robust staggered DiD

`PanelDidQuery::heterogeneity_robust` (Python `StaggeredRollout`) estimates the group-time
effect `ATT(g, t)` for each adoption cohort without using already-treated units as controls.
Three estimators are available:

- `callaway_santanna`: 2x2 comparisons against never-treated or not-yet-treated units. With
  baseline covariates it uses outcome regression, inverse probability weighting or the
  doubly robust 2x2.
- `sun_abraham`: interaction-weighted cohort-by-event-time contrasts against one control
  cohort. It needs a balanced panel and refuses an unbalanced one, naming the next attempt.
- `imputation`: unit and period effects are fitted on untreated rows only, and the untreated
  outcome of each treated row is imputed from them.

Effects are aggregated four ways:

| Aggregation | Summary |
| --- | --- |
| `simple` | Cohort-size-weighted average over post-adoption cells |
| `dynamic` | Event-time profile; the summary is the mean over post event times |
| `group` | Per-cohort average |
| `calendar` | Per-period average |

Anticipation of `δ` periods moves the base period to `g − 1 − δ`. Pre-adoption contrasts are
reported as falsification leads, with a sup-t band; they never enter an aggregation. The
static two-way fixed-effects coefficient is reported beside the summary as a diagnostic.

On panels and resampling:

- **Unbalanced panels** are read as they are. `missing=balanced_panel` keeps only complete
  units, using the shared unbalanced-panel preparation. `missing=available_case` keeps every
  complete unit-period.
- **Resampling** is declared with `with_resampling(cluster_bootstrap | cluster_multiplier,
  replicates)`. It redraws or reweights whole clusters, or whole subjects under
  `independent_subject_trajectories`, and refits every cell, nuisance and aggregation in each
  replicate. A `dependent_time_series` declaration is refused.

`PanelDidQuery::covariate_adjusted_two_period` (Python `CovariateAdjustedDiD`) is the
two-period form, for parallel trends that hold conditional on baseline covariates.

The augmented panel with supplied nuisance predictions accepts the same resample. Its
interval holds the supplied predictions fixed, so their own estimation error is not
propagated.

The Bayesian panel DiD refuses the rollout designs and names the frequentist route.

## Synthetic control with intervals

- **One treated unit:** `with_conformal_intervals()` gives conformal intervals (Chernozhukov,
  Wüthrich and Zhu). It inverts a permutation test over time, with the donor weights refit
  under each hypothesized effect, and reports one interval per post period plus a
  constant-effect interval. It needs at least 19 pre-periods for a 5% test. The test is
  exact when the residuals are exchangeable over time and approximate under weak dependence.
- **Several treated units:** `with_staggered_adoptions(adoptions, replicates)` fits each
  adopting unit against the never-treated donors at its own adoption period, using simplex
  weights or synthetic-DiD unit and time weights. The overall effect is the
  post-period-weighted average across units. Its interval comes from the placebo variance:
  the adoptions are reassigned to never-treated units and the procedure is refit.
- **Bayesian:** with staggered adoptions, `bayesian.synthetic_control` averages the per-unit
  simplex posteriors. It refuses the conformal declaration.

## Local ratio design layer and RD in time

`LocalPolynomialRatioQuery::options` controls three things:

- **Bandwidth:** `bandwidth_rule = mse_optimal` selects the plug-in MSE-optimal bandwidth,
  capped at the declared bandwidth. The robust bias-corrected interval is reported at that
  bandwidth. The plug-in is not the full rdrobust selector.
- **Diagnostics:** a McCrary density test and covariate-balance jumps. These are diagnostics
  only.
- **Running variable in calendar time:** `temporal_replicates` declares that the running
  variable is time. The interval then comes from a within-side moving-block bootstrap that
  refits the whole ratio.

The Bayesian local ratio reads a selected bandwidth and refuses a calendar-time running
variable.
