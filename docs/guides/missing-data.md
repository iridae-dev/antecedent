# Decide what happens to missing values

Missing values are never handled by accident. Each route says what it does with them, a
policy is declared with `missing=` (Rust: `StudyBuilder::missing`), and every result records
what happened to the rows.

```python
result = ant.analyze(data, query=query, missing="complete_case")
result.value.missing_data.policy         # "complete_case"
result.value.missing_data.assumption     # the assumption the policy needs
result.value.missing_data.accounting     # rows total / kept / dropped, per variable, per reason
```

## Defaults and policies

| Route | Without `missing=` | Policies it executes |
| --- | --- | --- |
| Average effect | complete case, reported | `error`, `complete_case`, `multiple_imputation` |
| Conditional, response, distribution, static mediation | complete case, reported | `error`, `complete_case`, `multiple_imputation` |
| Randomized two-arm ITT (`complete_case` and `inverse_probability` for every design except CUPED, ANCOVA and receipt-adjusted) | refuses at the first missing outcome | `error`, `complete_case`, `inverse_probability`, `multiple_imputation` |
| Panel difference-in-differences | refuses at the first missing outcome | `error`, `balanced_panel`, `available_case` |
| Lagged temporal effect and temporal mediation (one series) | refuses at the first missing time point | `error`, `complete_case`, `multiple_imputation` |
| NOTEARS discovery | refuses at the first missing cell | `error`, `complete_case`, `pairwise_available` |

A refusal is `missing_policy_required` and names the first missing cell (column and row) and
the policies the route can execute. A policy the route or design cannot execute is refused
when the study is built as `missing_policy_not_applicable`, naming the supported policies and
the next attempt; it is never accepted and ignored. Routes not listed read their rows in their
own way and accept no declaration.

### Route cells

Support is declared once per route cell (query kind, data modality, inference mode), and this table is checked against that declaration. A cell that is not listed executes no policy.

| Query | Modality | Inference | Without `missing=` | Policies it executes |
| --- | --- | --- | --- | --- |
| `AverageEffect` | tabular | frequentist | `complete_case` | `error`, `complete_case`, `multiple_imputation` |
| `AverageEffect` | tabular | bayesian | `complete_case` | `error`, `complete_case`, `multiple_imputation` |
| `ConditionalEffect` | tabular | frequentist | `complete_case` | `error`, `complete_case`, `multiple_imputation` |
| `ConditionalEffect` | tabular | bayesian | `complete_case` | `error`, `complete_case`, `multiple_imputation` |
| `Response` | tabular | frequentist | `complete_case` | `error`, `complete_case`, `multiple_imputation` |
| `Response` | tabular | bayesian | `complete_case` | `error`, `complete_case`, `multiple_imputation` |
| `Distribution` | tabular | frequentist | `complete_case` | `error`, `complete_case`, `multiple_imputation` |
| `Distribution` | tabular | bayesian | `complete_case` | `error`, `complete_case`, `multiple_imputation` |
| `Mediation` | tabular | frequentist | `complete_case` | `error`, `complete_case`, `multiple_imputation` |
| `Mediation` | tabular | bayesian | `complete_case` | `error`, `complete_case`, `multiple_imputation` |
| `RandomizedEffect` | tabular | frequentist | `error` | `error`, `complete_case`, `inverse_probability`, `multiple_imputation` |
| `RandomizedEffect` | tabular | bayesian | `error` | `error`, `complete_case`, `inverse_probability`, `multiple_imputation` |
| `PanelDid` | tabular | frequentist | `error` | `error`, `balanced_panel`, `available_case` |
| `PanelDid` | tabular | bayesian | `error` | `error`, `balanced_panel`, `available_case` |
| `TemporalEffect` | series | frequentist | `error` | `error`, `complete_case`, `multiple_imputation` |
| `TemporalEffect` | series | bayesian | `error` | `error`, `complete_case`, `multiple_imputation` |
| `Mediation` | series | frequentist | `error` | `error`, `complete_case`, `multiple_imputation` |
| `Mediation` | series | bayesian | `error` | `error`, `complete_case`, `multiple_imputation` |

Multiple imputation pools, per route cell:

| Route | Frequentist | Bayesian |
| --- | --- | --- |
| Average effect | `linear.adjustment.ate`, `glm.adjustment`, `aipw`, `propensity.weighting`, `propensity.matching`, `propensity.stratification`, `distance.matching` | `bayesian.gcomp` |
| Conditional effect | `conditional.linear.adjustment`, `aipw` | `conditional.bayesian` |
| Response | `response.intervention_gcomp`, `response.kennedy_dr`, `response.gam_derivative`, `response.riesz_ade` | `response.bayesian` |
| Distribution | `functional.distribution` | `functional.distribution` |
| Static mediation | `mediation.linear` | `mediation.linear` |
| Randomized ITT | `randomized.ht_itt`, `randomized.neyman_itt`, `randomized.ancova_itt`, `randomized.fixed_cuped_ht_itt`, `randomized.switchback_ht_itt`, `randomized.wald_cace_late` | `bayesian.randomized_finite_population`, `bayesian.randomized_superpopulation`, `bayesian.switchback_effect`, `bayesian.complier_effect` |
| Temporal effect | `temporal.linear.adjustment` | `bayesian.temporal.gcomp` |
| Temporal mediation | `temporal.mediation` | `temporal.mediation.bayesian` |

Another estimator chosen with it is refused at build (next attempt: one of those, or
`complete_case`).

**Matching and stratification.** `propensity.matching`, `distance.matching` and
`propensity.stratification` run whole inside every completed table: the propensity model is
refit, the donors are searched and the strata are cut again on that table, so the imputation
reaches the estimate through the same procedure a complete table would. The within-imputation
variance is the estimator's own published one. For nearest-neighbour matching it is the
Abadie–Imbens donor-reuse variance, because the nonparametric bootstrap is invalid for a fixed
number of matches and is never substituted. For stratification it is the analytic variance, or,
when `bootstrap` replicates are requested, the bootstrap variance, which resamples rows and
refits the propensity and the strata (the variance of the whole procedure). Rubin's rules then
add the between-imputation variance; no complete-data degrees of freedom is declared for these
estimators, so the reference distribution uses the Rubin degrees of freedom
`(m - 1) / lambda^2`, the Barnard–Rubin value for an unbounded complete-data degrees of freedom.

**Graph-posterior mixtures.** A Bayesian effect study over a posterior of directed acyclic
graphs (`graph_posterior=`) is pooled by stacking, not by Rubin's rules: the whole mixture runs
on every completed table (graph identification, the per-graph fits and the weighted mixture),
and the equal-weight stack of the tables' effect draws is the pooled posterior, so every table's
draws carry the declared graph weights and the pooling rule is reported as
`stacked_graph_posterior`. The declared graph posterior is held fixed: it is not re-learned on
each completed table, so it is the posterior of the structure you supplied, and a posterior that
was itself learned from the incomplete rows is a separate declaration. The claim says so
(`missing_imputation_structure:graph_posterior_held_fixed:weights_preserved`). Constructions with
no justified pooling rule are refused with their own detail: a frequentist study
(`multiple_imputation.graph_posterior_frequentist`), a series posterior
(`multiple_imputation.graph_posterior_series`), a query answered atom by atom, such as a response,
mediation, counterfactual, transport or attribution (`multiple_imputation.graph_posterior_query`),
and a class, partial-ancestral or latent-confounded atom kind whose per-table answer is not one
effect posterior (`multiple_imputation.graph_posterior_atoms`).

**What a pooled claim says, kept apart.** The claim names the missingness mechanism as a declared
assumption the data cannot test (`missing_imputation_mechanism:missing_at_random:declared:untestable`),
the sampling variance of the analysis within a completed table
(`missing_imputation_sampling_variance`), the between-imputation variance that the missing cells
add (`missing_imputation_between_variance`), and the pooling rule with the total
(`missing_imputation_pool`). None of the three is read as another: a small between-imputation
variance does not test the missing-at-random assumption.

## Row accounting

Every result of the routes above carries `missing_data.accounting`:

- `rows_total`, `rows_kept` (rows whose analysed values entered the estimate),
  `rows_dropped` and `rows_in_denominator` (what the estimate's denominators count);
- `dropped_by_variable`: rows missing each analysed variable; `dropped_by_reason`:
  `missing_value`, `analysis_mask` or `unit_incomplete`;
- `dropped_rows`: zero-based positions in the submitted table, and `dropped_digest`, a BLAKE3
  digest of exactly those positions. With `participant=` the rows are also named in
  `dropped_labels` with `dropped_labels_digest`;
- `groups`: total, kept and fully observed rows per arm or treatment group;
  `units_total` / `units_dropped` for panels.

The policy enters the data-snapshot identity only when it differs from the route's default, so
a study that keeps the default has the identity it always had. The accounting is derived from
the data and the policy and is not part of the identity.

## What each policy assumes

**Average-effect routes, `complete_case`.** Rows with a missing treatment, outcome or
adjustment value are dropped and the estimate speaks for the retained rows. It speaks for the
full population when missingness is independent of the outcome given the treatment and the
adjustment covariates. `error` refuses the table instead.

**Randomized trials, `complete_case`.** The responders of each arm are averaged (difference
in responder means with Neyman variance over responders). This needs outcomes missing
completely at random within arm. Every assigned participant is still counted in the arm sizes
and the assignment vector is never narrowed.

**Randomized trials, `inverse_probability`.** Declare the baseline covariates the response may
depend on (`MissingPolicy.inverse_probability(["baseline"])`). Within each arm the probability
of responding is fitted by logistic regression on those covariates, responders are weighted by
its inverse, and the arm mean is the Hajek ratio. The assumption is outcomes missing at random
given arm and the declared covariates (the chance of responding depends on the outcome only
through them), positivity (fitted response probabilities above 0.02, else the fit refuses) and
a correct logistic model. Every assigned participant stays in its arm's denominator. The
variance is the arm-wise influence-function variance including the score of the response
model. Covariates must be complete. Bernoulli and complete randomization use the Hajek
contrast above. Cluster, stratified, factorial, multi-arm and switchback designs keep their own
estimator and variance (cluster totals, stratum weights, cell contrasts, sequence scores) and
run it on every assigned unit's inverse-probability pseudo-outcome (the weighted outcome, zero
for a non-responder), with the observation model fitted within each assignment group. Their
variance is computed on pseudo-outcomes that carry the influence of the fitted response model (stacked M-estimation), so estimating the response probabilities is accounted for; across strata or clusters the shared coefficients are treated as independent between blocks.
CUPED, ANCOVA and receipt-adjusted fits read further per-unit inputs and refuse.

**Panel difference-in-differences, `balanced_panel`.** Units with a missing outcome in any
period are dropped whole; single unit-periods are never deleted, so no unit is relabelled and
the panel stays balanced. It is valid when a unit's chance of being complete is independent of
its outcome changes within cohort. `complete_case` is refused for panels because it would
unbalance them.

**Panel difference-in-differences, `available_case`.** For staggered designs each event-time
or group-time contrast uses the units observed in its own period and the cohort baseline,
with its own unit and cluster counts; unit and period identities are preserved. For a
repeated cross-section, rows are independent subjects and missing rows are dropped. For a
two-period panel it equals `balanced_panel`. The augmented panel's row-aligned nuisance
columns cannot be restricted and refuse both.

**Temporal, `complete_case`.** Time stays on its original grid. A lag window that reads a
missing time point is dropped and every other window keeps its own time and its own lags, so
no lag crosses a gap and nothing is concatenated. The accounting counts windows, names the
dropped windows by series time and counts missing time points per variable. The point
estimate is the regression on intact windows; it is valid when missingness is independent of
the outcome innovation given the lagged regressors. The circular-block bootstrap
draws its blocks on the original time axis: two windows share a block only when they are
adjacent in time, a replicate keeps the intact windows its blocks cover, and no block joins
windows across a gap. Its Bartlett kernel-bias inflation is not applied (it is defined for
consecutive scores), and no interval is published. The refuters refit the perturbed series
through the same gap-aware design (the perturbations keep the gaps in place), and the bootstrap
refuter resamples on the window axis. The multi-step sustained contrast, the one-modifier
conditional effect and the frequentist temporal mediation contrasts run the same way: every
mechanism is fitted on the intact windows at their original times, and their block bootstraps
draw blocks on the original window axis. The Bayesian single-window, sustained-contrast and mediation posteriors are fitted on the
intact windows and tempered by the segment-wise Bartlett long-run variance of their estimating
scores: autocovariances are summed within runs of consecutive windows only, so no pair that
straddles a gap enters. That is a different construction from the REML autoregressive factor
used on a series without gaps (the autoregression and its small-sample scale are defined for
consecutive rows), and their predictive checks and prior-sensitivity refuters, which read
consecutive rows, must be off. The sustained and mediation refuters refit through the intact
windows, and the frequentist mixtures over graph classes and the dynamic-Bayesian-network
mediation estimator fit every atom on its own intact windows. The posterior mixtures over
graph classes and over dynamic-Bayesian-network atoms do the same: each atom is fitted on its
own intact windows, a Bayesian atom is tempered segment-wise, and a frequentist mixture's shared
bootstrap resamples blocks of the original window axis, so every atom is refit on the same
resampled windows and no block joins windows across a gap. A mixture reports the windows that
any of its atoms lost (a window counts as dropped when some atom reads a gap in it) and the
missing time points of the variables it reads; the Bayesian dynamic-Bayesian-network mixture
declines refuters, whose predictive checks and prior sensitivity read consecutive rows, and
every mixture refuses a discovery-estimation split on a series with gaps.

**Average effect, `multiple_imputation`.** Declare it with
`MissingPolicy.multiple_imputation(model=..., imputations=..., iterations=..., seed=...,
assumption="missing_at_random", auxiliary=(...))` (Rust:
`MissingDeclaration::new(MissingPolicy::MultipleImputation(ImputationSpec::new()...))`).
Every row is kept. The imputation model reads every float column of the table, or, when
auxiliary variables are declared, the treatment, the outcome and those variables; an analysed
variable outside the model that is itself incomplete is refused. Two proper imputation models
are available:

- `chained_equations` (default): each incomplete variable is drawn in turn from its
  conditional model on every other variable (normal linear regression for a continuous
  variable, logistic regression for a 0/1 one), fitted on a Bayesian bootstrap of the observed
  rows, so every completed table draws the model's parameters before it draws the cells;
  `iterations` sweeps are run before a table is taken.
- `joint_normal`: multivariate normal data augmentation; each sweep draws the covariance
  from its inverse-Wishart posterior under the Jeffreys prior and the mean given it, then every
  missing cell from its conditional normal law. An incomplete binary variable is refused (next
  attempt: `chained_equations`).

The route's own estimator runs on every completed table. Frequentist fits are pooled by
Rubin's rules: the point is the mean of the `m` estimates, the variance is
`T = Ū + (1 + 1/m) B` (within plus inflated between-imputation variance), and the reference is
a Student t with the Barnard-Rubin degrees of freedom when the estimator has a finite
complete-data degrees of freedom (`n − p` for linear adjustment), Rubin's large-sample
degrees of freedom otherwise. Bayesian fits are stacked: the posterior draws of every
completed table form one equal-weight mixture, and the between-imputation variance of the
per-table posterior means is reported as its own axis. `result.value.missing_data.imputation`
carries the specification, the unit the parameter draws resampled, the cells imputed per
variable, every table's estimate and the variance components; the claim carries the
specification and the pooled components among its evidence, and the specification enters the
data-snapshot identity.

Imputation table `j` opens its own keyed random stream from `seed`, so the result does not
depend on threads or scheduling, and every completed table is leased from the execution's
admission pool before it is allocated (a pool that cannot fund one table is refused before
anything is drawn). The declared sampling design governs the unit of the parameter draw:
rows for independent rows or subjects, whole clusters for a clustered panel (refused on the
static routes, which bind no cluster labels; next attempt: supply cluster labels or
`complete_case`; a clustered randomized design takes its clusters from its assignment units),
and contiguous blocks for a dependent time series, which the temporal routes impute by the
lagged model described below. The assumption is missing at random given the variables of the imputation
model, and a correctly specified imputation model. Declaring `missing_not_at_random` is
refused with a sensitivity analysis (delta-adjusted or selection-model imputation over a grid
of departures) named as the next attempt; that analysis is
[`mnar_sensitivity.mnar_ate_sensitivity`](../sensitivity-and-robustness.md#missing-outcomes-not-at-random-mnar_ate_sensitivity),
which re-runs an arm-mean contrast over a declared bounded grid and reports assumption ranges,
tipping points and decision changes.

**Routes with several coordinates, `multiple_imputation`.** A conditional effect's cells, a
response curve, an interventional distribution, a mediation decomposition, a randomized
design's arms and a series' horizons are pooled jointly, not one coordinate at a time. Every
completed table's fit contributes its coordinates with the covariance the route published
for them: the joint influence covariance where the estimator retains one (conditional grids,
response curves), else each coordinate's own published variance on the diagonal
(`vector.covariance_basis` says which: `joint_covariance`, `marginal_variances` or
`posterior_draws`). Rubin's rules then give the within-, between- and total covariance, each
coordinate's Student-t degrees of freedom and fraction of missing information, the average
relative increase in variance and the pooled Wald test that every coordinate is zero (Li,
Raghunathan and Rubin's `D1`). A Bayesian route stacks the joint draws of every table, so the
dependence between coordinates is kept. A quantity with no published variance (a static
mediation's total, direct and mediated components, which a bootstrap reports only for the
requested contrast) is pooled by its mean over the tables and reported with its
between-imputation variance in `vector.extras`; it never enters the covariance. The route
reads the variance from its own resampled or analytic uncertainty, so a route whose fit
publishes none (a distribution or static mediation without bootstrap replicates) refuses with
the request for it as the next attempt.

A randomized design adds the arm each row was assigned to the imputation model (the table does
not hold it, and an outcome imputed without it would be drawn as if assignment did nothing);
the arm enters as a main effect, so an arm-specific outcome relation the model omits biases the
imputed effect toward zero. A clustered design draws its parameters by whole assignment-unit
clusters. The design's own estimator and its declared assignment probabilities run unchanged
on every completed table.

A single series is imputed by a lagged model rather than by exchangeable rows: each incomplete
variable is drawn given the other variables at the same step and every variable at the
`lags` neighbouring steps on each side (the deepest lag the query reads), read from
the current completed columns, with the parameters drawn by a Bayesian bootstrap of
contiguous blocks (the cube root of the length, at least one neighbourhood). The assumption
is missingness at random given the neighbouring steps and a stationary series whose
dependence is no longer than the conditioned neighbourhood; the dependent time-series design
is the only design a series accepts, and the model is `chained_equations`. The multi-horizon
Bayesian mediation reports a separate posterior per horizon and defines no joint posterior to
stack, so it is refused (next attempt: one horizon, or `complete_case`).

Every pooled result carries `assumptions_digest`, the digest of the imputation model, tables,
sweeps, seed, missingness assumption, resampling unit, declared sampling design, lagged
neighbourhood and every variable's kind and imputed-cell count. The same digest enters the
data-snapshot identity beside the specification, so two analyses that differ in any of them
are different claims.

**NOTEARS, `complete_case`.** Rows with a missing value in any discovery variable are dropped
and reported in `result.row_accounting`. The graph equals the one learned on the complete rows
and speaks for the full data only when missingness is completely at random.

**NOTEARS, `pairwise_available`.** The least-squares score depends on the data only through the
sample covariance, so each pair of variables uses the rows observed on both, the pairwise
covariance (correlation when standardizing) is repaired to positive semidefinite by clipping
negative eigenvalues, and the solver scores that matrix exactly. No row is deleted. The
accounting lists the rows behind every pair and the number of eigenvalues clipped. It needs
missingness completely at random and at least three rows on every pair.

## Recover a law when one response indicator causes another

Two binary variables `X1`, `X2` may each be observed only when a response indicator `R1`, `R2` is 1, and the second response may depend on the first (`R1 -> R2`, optionally with `X1 -> R2`). `antecedent.recovery_chain` asks whether the joint law `P(X1, X2)` is recoverable from the observed pattern law under the graph you declare, and returns it when it is. This is graph-licensed recovery, not complete-case analysis, missing-at-random adjustment or inverse-probability weighting; in the worked fixture of the tests, the complete-case table differs from the recovered one.

The graph is over `x1`, `x2`, their response indicators `r1`, `r2` and deterministic proxies `p1`, `p2` (`p_i = x_i` when `r_i = 1`, missing when `r_i = 0`), with exactly one response edge. The observed margin is a `PatternLaw`: the 2x2 mass of both observed, the mass of only the first and only the second observed, and the mass of neither.

```python
from antecedent.recovery_chain import (
    ChainRoles, PatternLaw, RecoveryChainQuery, decide_chain, recover_chain,
)

graph = {
    "nodes": ["x1", "x2", "r1", "r2", "p1", "p2"],
    "edges": [("x1", "x2"), ("x1", "p1"), ("r1", "p1"), ("x2", "p2"), ("r2", "p2"),
              ("r1", "r2"), ("x1", "r2")],
}
query = RecoveryChainQuery(ChainRoles("x1", "r1", "p1"), ChainRoles("x2", "r2", "p2"))

plan = decide_chain(graph, query)                  # the checked plan, with no data
law = PatternLaw(both=both, only_first=only_first, only_second=only_second, neither=neither)
result = recover_chain(graph, query, law)          # `both`, `only_*` and `neither` are your observed masses
result.law                                         # the recovered 2x2 law P(X1, X2), summing to 1
```

The pattern-law masses (`both` as a 2x2 nested list, the other three as in the class) are supplied by you from your data and are not shown here. In the executing test they come from an enumerated structural model, and `recover_chain` reproduces the enumerated truth to `1e-12` for either head response and with or without the `X1 -> R2` edge. The answer is point-only (`result.to_dict()["interval"]` reports `status` `point_only`) and does not depend on node order or labels. `result.export()` and `consume_recovery_chain_artifact(...)` recompute the recovered law from the artifact.

Two refusals are different scientific information. A self-censoring edge (`X_i -> R_i`) with every other edge in the class is nonrecoverable, and the refusal carries an exactly verified two-model witness. A graph outside this class (anything other than one response chain) is `route_not_supported`: that says the row does not cover it, not that the law is nonrecoverable. A zero-mass cell or an invalid law is refused as well.

## Claims

A claim over a result whose policy differs from the route's default carries the policy, the
assumption it needs and the digest of the dropped rows among its evidence references; a
multiple-imputation claim also names the specification, the pooled variance components and the
imputation's assumptions digest. Claims of default policies keep exactly the references they
had.

## Unbalanced panels

`antecedent_data::PanelLayout::assess(subjects, periods, observed)` reports a long panel
before any design reads it: the calendar grid, each unit's first and last observed period,
its internal gaps as calendar runs, late entry and attrition, the cells expected and present,
an attrition table per period (units at risk, observed, entering, leaving) and, given a
baseline column, the standardized difference between units that later leave and units that
stay. Duplicate unit-periods are refused. `PanelBalanceReport::prepare` keeps whole complete
units (`BalancedUnits`), every complete unit-period with its identities
(`AvailableUnitPeriods`), or each unit's maximal runs of consecutive periods
(`ContiguousRuns`), so no lag or difference spans a gap or crosses a unit, and produces the row
accounting above. `require_balance` refuses, for designs that need a balanced panel
(two-period panels, synthetic control, a balanced event study, an augmented panel), naming the
units that break balance and the next attempt; `require_lag` refuses a lag that would read
across a gap. No difference-in-differences estimator is run by this layer.

Two designs also read an unbalanced panel directly, without this layer relabelling anything.
A staggered rollout with `sun_abraham` fits the saturated cohort-by-period regression on the
observed unit-periods, reporting per-cohort support and the unit-periods it dropped; its
`observation` declaration (`ignorable_given_cohort`, recorded as an untestable assumption, or
`informative`, which is refused) states the observation process. A `SyntheticControl` with
`observed_window_support=True` fits over the treated unit's observed window with the donors
observed throughout it. See [Panel designs](designs/panel-designs.md).

## Intervals

A handled missingness changes the variance construction, so no interval is published for it:
responder-weighted and complete-case randomized effects, restricted panels and temporal fits
on series with gaps report a point estimate and a standard error where one exists, and the
interval stays closed until a coverage record measures the construction. Multiple
imputation publishes the pooled standard error and its degrees of freedom, and its interval
stays closed the same way: the whole impute-fit-pool procedure is one uncertainty path, so a
coverage record can measure it as one construction.
