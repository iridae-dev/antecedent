# Decide what happens to missing values

Use this guide when a column an analysis reads has missing cells (`NaN`, `None`, `NaT` or an
invalid Arrow cell). The table below states what each analysis does with them. Declare a
policy with `missing=`. In Rust, call `StudyBuilder::missing`. Every result records what
happened to the rows. No analysis deletes, weights or restricts rows on its own where that
would change the design.

```python
import antecedent as ant

result = ant.analyze(
    data,
    graph=[("z", "t"), ("z", "y"), ("t", "y")],
    query=ant.AverageEffect("t", "y"),
    missing="complete_case",             # a policy name, or an antecedent.missing.MissingPolicy
)
result.value.missing_data.policy         # "complete_case"
result.value.missing_data.declared       # True: declared, not the route's default
result.value.missing_data.assumption     # the assumption the policy needs
result.value.missing_data.accounting     # rows total / kept / dropped, per variable, per reason
```

To choose a policy:

1. Find your analysis in [the table below](#defaults-and-policies). The table lists the default
   and the policies that analysis executes.
2. Read [what each policy assumes](#what-each-policy-assumes) and pick the one whose
   missingness assumption you can defend. Three mechanisms recur. Missing completely at random:
   missingness is independent of every variable. Missing at random: missingness is independent
   of the unobserved values given the observed variables the policy conditions on. Missing not
   at random: missingness depends on the unobserved value itself. The data cannot distinguish
   the last two, so the assumption is declared, never tested.
3. Read [row accounting](#row-accounting) on the result to see which rows the estimate speaks
   for, and [intervals](#intervals) for what uncertainty is published under the policy.

The policy changes which rows enter and what is assumed about the rest. The estimand and its
identifying assumptions stay those of the query. [`examples/python/observational_study.py`](https://github.com/iridae-dev/antecedent/blob/v3.1.0/examples/python/observational_study.py)
declares `missing="complete_case"` and reads the accounting;
[`examples/python/missing_outcome_sensitivity.py`](https://github.com/iridae-dev/antecedent/blob/v3.1.0/examples/python/missing_outcome_sensitivity.py)
bounds an effect when outcomes are missing not at random.

## Defaults and policies

| Route | Without `missing=` | Policies it executes |
| --- | --- | --- |
| Average effect | complete case, reported | `error`, `complete_case`, `multiple_imputation` |
| Conditional, response, distribution, static mediation | complete case, reported | `error`, `complete_case`, `multiple_imputation` |
| Randomized two-arm ITT (`complete_case` and `inverse_probability` for every design except CUPED, ANCOVA and receipt-adjusted) | refuses at the first missing outcome | `error`, `complete_case`, `inverse_probability`, `multiple_imputation` |
| Panel difference-in-differences | refuses at the first missing outcome | `error`, `balanced_panel`, `available_case` |
| Anomaly attribution (rows scored against fitted mechanisms) | refuses at the first missing cell | `error`, `complete_case` |
| Temporal design, arm designs with the mean functional (one row per unit and period) | refuses at the first missing outcome | `error`, `complete_case`, `available_case`, `balanced_panel` |
| Lagged temporal effect and temporal mediation (one series) | refuses at the first missing time point | `error`, `complete_case`, `multiple_imputation` |
| NOTEARS discovery | refuses at the first missing cell | `error`, `complete_case`, `pairwise_available` |

A refusal is `missing_policy_required`. It names the first missing cell (column and row) and
the policies that analysis can execute. A policy the analysis or design cannot execute is
refused when the study is built as `missing_policy_not_applicable`. The refusal names the
supported policies and the next attempt. It is never accepted and ignored. Both codes are
listed under
[other registered refusal codes](../reference/refusal-codes.md#other-registered-refusal-codes). Analyses not listed read their rows in their
own way. They accept no declaration.

### Route cells

Each row is one query, one data kind and one inference mode. A combination that is not listed executes no policy.

<!-- generated:missing-policy-routes:begin -->
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
| `AnomalyAttribution` | tabular | frequentist | `error` | `error`, `complete_case` |
| `AnomalyAttribution` | tabular | bayesian | `error` | `error`, `complete_case` |
| `TemporalDesign` | tabular | frequentist | `error` | `error`, `complete_case`, `available_case`, `balanced_panel` |
| `TemporalDesign` | tabular | bayesian | `error` | `error`, `complete_case`, `available_case`, `balanced_panel` |
| `PanelDid` | tabular | frequentist | `error` | `error`, `balanced_panel`, `available_case` |
| `PanelDid` | tabular | bayesian | `error` | `error`, `balanced_panel`, `available_case` |
| `TemporalEffect` | series | frequentist | `error` | `error`, `complete_case`, `multiple_imputation` |
| `TemporalEffect` | series | bayesian | `error` | `error`, `complete_case`, `multiple_imputation` |
| `Mediation` | series | frequentist | `error` | `error`, `complete_case`, `multiple_imputation` |
| `Mediation` | series | bayesian | `error` | `error`, `complete_case`, `multiple_imputation` |
<!-- generated:missing-policy-routes:end -->

Multiple imputation pools, per route cell:

| Route | Frequentist | Bayesian |
| --- | --- | --- |
| Average effect | `linear.adjustment.ate`, `glm.adjustment`, `aipw`, `propensity.weighting`, `propensity.matching`, `propensity.stratification`, `distance.matching`, `rd.sharp` | `bayesian.gcomp`, `rd.bayesian_local_linear` |
| Conditional effect | `conditional.linear.adjustment`, `aipw` | `conditional.bayesian` |
| Response | `response.intervention_gcomp`, `response.kennedy_dr`, `response.gam_derivative`, `response.riesz_ade` | `response.bayesian` |
| Distribution | `functional.distribution` | `functional.distribution` |
| Static mediation | `mediation.linear` | `mediation.linear` |
| Randomized ITT | `randomized.ht_itt`, `randomized.neyman_itt`, `randomized.ancova_itt`, `randomized.fixed_cuped_ht_itt`, `randomized.switchback_ht_itt`, `randomized.wald_cace_late` | `bayesian.randomized_finite_population`, `bayesian.randomized_superpopulation`, `bayesian.switchback_effect`, `bayesian.complier_effect` |
| Temporal effect | `temporal.linear.adjustment` | `bayesian.temporal.gcomp` |
| Temporal mediation | `temporal.mediation` | `temporal.mediation.bayesian` |

Another estimator chosen with multiple imputation is refused at build. The next attempt is
one of the estimators in the table, or `complete_case`.

**Matching and stratification.** These estimators each run inside every completed table:

- `propensity.matching`
- `distance.matching`
- `propensity.stratification`

The propensity model is refit. The donors are searched again. The strata are cut again.
The imputation reaches the estimate through the same procedure a complete table would.
The within-imputation variance is the estimator's own published one. For nearest-neighbour
matching it is the Abadie–Imbens donor-reuse variance. The nonparametric bootstrap is invalid
for a fixed number of matches. It is never substituted. For stratification the
within-imputation variance is the analytic variance. When you request `bootstrap` replicates,
it is the bootstrap variance. That bootstrap resamples rows and refits the propensity and the
strata. It is the variance of the whole procedure. Rubin's rules then add the
between-imputation variance. No complete-data degrees of freedom is declared for these
estimators. The reference distribution uses the Rubin degrees of freedom
`(m - 1) / lambda^2`. That is the Barnard–Rubin value for an unbounded complete-data degrees
of freedom.

**Graph-posterior mixtures.** A Bayesian effect study over a posterior of directed acyclic
graphs (`graph_posterior=`) is pooled by stacking. It is not pooled by Rubin's rules. The
whole mixture runs on every completed table: graph identification, the per-graph fits and the
weighted mixture. The equal-weight stack of the tables' effect draws is the pooled posterior.
Every table's draws carry the declared graph weights. The pooling rule is reported as
`stacked_graph_posterior`. The declared graph posterior is held fixed. It is not re-learned on
each completed table. It is the posterior of the structure you supplied. A posterior that
was itself learned from the incomplete rows is a separate declaration. The claim says so
(`missing_imputation_structure:graph_posterior_held_fixed:weights_preserved`).

To carry graph-selection uncertainty, re-run the search on every completed table. In Rust,
call `antecedent::estimate_with_graph_selection_per_imputation`. Pass `data`, `spec`,
`&selection`, `ctx` and `build`. The call imputes the table. It searches each completed
table for its own DAG posterior (`GraphSelectionPerImputation`). It runs the Bayesian effect
study that `build` returns on that table and posterior. It stacks the effect posteriors with
equal weight. The pooling is `stacked_graph_posterior_rerun`. The claim entry is
`missing_imputation_structure:graph_selection_rerun_per_imputation:weights_preserved`. Call it
at the top level. A study cannot build and run another study. The table must be float
columns, searched in column order. The study must run `bayesian.gcomp`.

In Python, call `antecedent.missing.multiple_imputation_with_graph_selection`. Pass `data`,
`treatment`, `outcome`, `imputations`, `n_draws` and `prior_scale`. The table is float
columns with `NaN` for missing cells. You also pass the treatment, the outcome and the
Bayesian g-computation settings. The call searches the DAG posterior over the columns in
order on every completed table. The search is exact for at most six columns. It uses order
MCMC for more columns. The adjustment is whatever each sampled graph identifies. The seed
is `execution.seed`. You get the stacked effect (`effect`) and its posterior standard
deviation (`posterior_sd`). You also get the tables' posterior means (`points`) and the
`within`, `between` and `total` variances. An interval is not reported. `result.interval()`
refuses. The combined procedure has no independent construction check of its own interval.
Graph selection, imputation and effect estimation must be evaluated together. No confidence
or credible interval follows from the point support. The result retains the declared
missing-at-random assumption (`assumption`). Refusals raise
`GraphSelectionImputationRefusal`. Constructions with no justified pooling rule are refused
with their own detail:

- a frequentist study (`multiple_imputation.graph_posterior_frequentist`)
- a series posterior (`multiple_imputation.graph_posterior_series`)
- a query answered atom by atom, such as a response, mediation, counterfactual, transport or attribution (`multiple_imputation.graph_posterior_query`)
- a class, partial-ancestral or latent-confounded atom kind whose per-table answer is not one effect posterior (`multiple_imputation.graph_posterior_atoms`)

**Static mediation.** A static mediation decomposition publishes no analytic variance. Its
within-imputation variance is a bootstrap run inside each completed table. The whole
decomposition is refit on every resample. The replicate count is the imputation's own setting,
`ImputationSpec.bootstrap_replicates`. In Python, pass `bootstrap_replicates=200` to
`MissingPolicy.multiple_imputation`. That count is used in place of the study's global
`bootstrap_replicates` for those tables. With neither set the analysis is refused for want of
a variance to pool. A multi-horizon Bayesian mediation is refused with
`multiple_imputation.multi_horizon_posterior_not_joint`. Its result keeps per-horizon summaries
(`joint_posterior = false`, no draws). There is no joint draw set to stack. One horizon
stacks as usual.

**What a pooled claim says, kept apart.** The claim names four entries:

- the missingness mechanism as a declared assumption the data cannot test (`missing_imputation_mechanism:missing_at_random:declared:untestable`)
- the sampling variance of the analysis within a completed table (`missing_imputation_sampling_variance`)
- the between-imputation variance that the missing cells add (`missing_imputation_between_variance`)
- the pooling rule with the total (`missing_imputation_pool`)

Do not read the sampling variance, the between-imputation variance or the pooling rule as one
another. A small between-imputation variance does not test the missing-at-random assumption.

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

The policy enters the data-snapshot identity only when it differs from the analysis default.
Declaring that default leaves the identity unchanged. The accounting is derived from the data
and the policy. It is not part of the identity.

## What each policy assumes

**Average-effect analyses, `complete_case`.** Rows with a missing treatment, outcome or
adjustment value are dropped. The estimate speaks for the retained rows. It speaks for the
full population when missingness is independent of the outcome given the treatment and the
adjustment covariates. `error` refuses the table instead.

**Randomized trials, `complete_case`.** The responders of each arm are averaged. That is the
difference in responder means with Neyman variance over responders. This needs outcomes missing
completely at random within arm. Every assigned participant is still counted in the arm sizes.
The assignment vector is never narrowed.

**Randomized trials, `inverse_probability`.** Declare the baseline covariates the response may
depend on. Call `MissingPolicy.inverse_probability` with those names, for example
`["baseline"]`. Within each arm the probability of responding is fitted by logistic regression
on those covariates. Responders are weighted by the inverse of that probability. The arm mean
is the Hajek ratio. The assumption has three parts:

- Outcomes are missing at random given arm and the declared covariates. The chance of responding depends on the outcome only through them.
- Positivity. Fitted response probabilities must be above 0.02. Otherwise the fit refuses.
- The logistic model is correct.

Every assigned participant stays in its arm's denominator. The variance is the arm-wise
influence-function variance including the score of the response model. Covariates must be
complete. Bernoulli and complete randomization use the Hajek contrast above. Cluster,
stratified, factorial, multi-arm and switchback designs keep their own estimator and variance.
Those are cluster totals, stratum weights, cell contrasts and sequence scores. The estimator
runs on every assigned unit's inverse-probability pseudo-outcome. The pseudo-outcome is the
weighted outcome, and zero for a non-responder. The observation model is fitted within each
assignment group. Their variance is computed on pseudo-outcomes that carry the influence of
the fitted response model (stacked M-estimation). Estimating the response probabilities is
accounted for. Across strata or clusters the shared coefficients are treated as independent
between blocks. CUPED, ANCOVA and receipt-adjusted fits read further per-unit inputs. They
refuse.

Under Bayesian inference, `inverse_probability` has graphless support for two-arm
Bernoulli and complete randomization. Set `estimator="bayesian.randomized_superpopulation"`.
The arm prior is a proper Gaussian Normal-Inverse-Gamma. Every assigned unit contributes its
score-adjusted `R Y / pi` pseudo-outcome to the Gaussian working likelihood. The observation
model is fitted and held fixed. Its score correction enters the pseudo-outcome spread. Its
coefficients do not receive a posterior. This is a modular pseudo-posterior conditional
on the fitted observation model. Its support covers pointwise 95% posterior quantiles under
the declared MAR, positivity and correct logistic-model assumptions. That support does not rest
on a repeated-sampling coverage record. Each interval reports its calibration status.

The prior shrinks each arm's pseudo-mean. The posterior mean is not exactly a Hajek
responder mean. With saturated response strata the unshrunk pseudo-mean equals the
post-stratified responder mean. Retained results include source outcomes, response indicators,
covariates, assignment and the prior. You can replay the observation fit and the posterior
from them. Other Bayesian IPW designs refuse. The finite-population estimand refuses.
Beta-Bernoulli arm models refuse. They do not run outside this support.

**Panel difference-in-differences, `balanced_panel`.** Units with a missing outcome in any
period are dropped whole. Single unit-periods are never deleted. No unit is relabelled. The
panel stays balanced. It is valid when a unit's chance of being complete is independent of
its outcome changes within cohort. `complete_case` is refused for panels. It would unbalance
them.

**Panel difference-in-differences, `available_case`.** For staggered designs each event-time
or group-time contrast uses the units observed in its own period and the cohort baseline.
Each contrast has its own unit and cluster counts. Unit and period identities are preserved.
For a repeated cross-section, rows are independent subjects. Missing rows are dropped. For a
two-period panel it equals `balanced_panel`. The augmented panel's row-aligned nuisance
columns cannot be restricted. Both policies refuse.

**Temporal, `complete_case`.** Time stays on its original grid. A lag window that reads a
missing time point is dropped. Every other window keeps its own time and its own lags. No lag
crosses a gap. Nothing is concatenated. The accounting counts windows. It names the dropped
windows by series time. It counts missing time points per variable. The point estimate is the
regression on intact windows. It is valid when missingness is independent of the outcome
innovation given the lagged regressors. The circular-block bootstrap draws its blocks on the
original time axis. Two windows share a block only when they are adjacent in time. A replicate
keeps the intact windows its blocks cover. No block joins windows across a gap. Its Bartlett
kernel-bias inflation is not applied. That inflation is defined for consecutive scores. No
interval is published. The refuters refit the perturbed series through the same gap-aware
design. The perturbations keep the gaps in place. The bootstrap refuter resamples on the
window axis. The multi-step sustained contrast, the one-modifier conditional effect and the
frequentist temporal mediation contrasts run the same way. Every mechanism is fitted on the
intact windows at their original times. Their block bootstraps draw blocks on the original
window axis. The Bayesian single-window, sustained-contrast and mediation posteriors are
fitted on the intact windows. They are tempered by the segment-wise Bartlett long-run variance
of their estimating scores. Autocovariances are summed within runs of consecutive windows
only. No pair that straddles a gap enters. That is a different construction from the REML
autoregressive factor used on a series without gaps. The autoregression and its small-sample
scale are defined for consecutive rows. Their predictive checks and prior-sensitivity
refuters read consecutive rows. Those refuters must be off. The sustained and mediation
refuters refit through the intact windows. The frequentist mixtures over graph classes and
the dynamic-Bayesian-network mediation estimator fit every atom on its own intact windows.
The posterior mixtures over graph classes and over dynamic-Bayesian-network atoms do the
same. Each atom is fitted on its own intact windows. A Bayesian atom is tempered
segment-wise. A frequentist mixture's shared bootstrap resamples blocks of the original
window axis. Every atom is refit on the same resampled windows. No block joins windows
across a gap. A mixture reports the windows that any of its atoms lost. A window counts as
dropped when some atom reads a gap in it. The mixture also reports the missing time points
of the variables it reads. The Bayesian dynamic-Bayesian-network mixture declines refuters.
Those predictive checks and prior sensitivity read consecutive rows. Every mixture refuses a
discovery-estimation split on a series with gaps.

**Average effect, `multiple_imputation`.** Call `MissingPolicy.multiple_imputation`. Pass
`model`, `imputations`, `iterations`, `seed`, `assumption="missing_at_random"` and
`auxiliary`. The defaults are `chained_equations`, 20 imputations, 10 iterations and seed 0.
In Rust, call `MissingDeclaration::new` with `MissingPolicy::MultipleImputation` and
`ImputationSpec::new`. Every row is kept. The imputation model reads every float column of
the table. When auxiliary variables are declared, it reads the treatment, the outcome and
those variables. An analysed variable outside the model that is itself incomplete is refused.
Two proper imputation models are available:

- `chained_equations` (default): each incomplete variable is drawn in turn from its conditional model on every other variable. A continuous variable uses normal linear regression. A 0/1 variable uses logistic regression. The model is fitted on a Bayesian bootstrap of the observed rows. Every completed table draws the model's parameters before it draws the cells. `iterations` sweeps are run before a table is taken.
- `joint_normal`: multivariate normal data augmentation. Each sweep draws the covariance from its inverse-Wishart posterior under the Jeffreys prior and the mean given it. It then draws every missing cell from its conditional normal law. An incomplete binary variable is refused. The next attempt is `chained_equations`.

Your estimator runs on every completed table. Frequentist fits are pooled by Rubin's rules.
The point is the mean of the `m` estimates. The variance is `T = Ū + (1 + 1/m) B` (within
plus inflated between-imputation variance). The reference is a Student t with the
Barnard-Rubin degrees of freedom when the estimator has a finite complete-data degrees of
freedom (`n − p` for linear adjustment). Otherwise it uses Rubin's large-sample degrees of
freedom. Bayesian fits are stacked. The posterior draws of every completed table form one
equal-weight mixture. The between-imputation variance of the per-table posterior means is
reported as its own axis. `result.value.missing_data.imputation` carries the specification,
the unit the parameter draws resampled, the cells imputed per variable, every table's
estimate and the variance components. The claim carries the specification and the pooled
components among its evidence. The specification enters the data-snapshot identity.

Imputation table `j` opens its own keyed random stream from `seed`. The result does not
depend on threads or scheduling. Every completed table is leased from the execution's
admission pool before it is allocated. A pool that cannot fund one table is refused before
anything is drawn. The declared sampling design governs the unit of the parameter draw.
Independent rows or subjects draw by row. A clustered panel draws whole clusters. Static
analyses bind no cluster labels and refuse. The next attempt is to supply cluster labels or
`complete_case`. A clustered randomized design takes its clusters from its assignment units.
A dependent time series draws contiguous blocks. The temporal analyses impute that series by
the lagged model described below. The assumption is missing at random given the variables of
the imputation model. The imputation model must be correctly specified. Declaring
`missing_not_at_random` is refused. The next attempt is a sensitivity analysis: delta-adjusted
or selection-model imputation over a grid of departures. That analysis is
[`mnar_sensitivity.mnar_ate_sensitivity`](../sensitivity-and-robustness.md#missing-outcomes-not-at-random-mnar_ate_sensitivity).
It re-runs an arm-mean contrast over a declared bounded grid. It reports assumption ranges,
tipping points and decision changes.

**Analyses with several coordinates, `multiple_imputation`.** These coordinates are pooled
jointly. They are not pooled one coordinate at a time:

- a conditional effect's cells
- a response curve
- an interventional distribution
- a mediation decomposition
- a randomized design's arms
- a series' horizons

Every completed table's fit contributes its coordinates with the covariance that analysis
published for them. Where the estimator retains a joint influence covariance, that covariance
is used. Conditional grids and response curves are that case. Otherwise each coordinate's own
published variance sits on the diagonal. `vector.covariance_basis` says which:
`joint_covariance`, `marginal_variances` or `posterior_draws`. Rubin's rules then give the
within-, between- and total covariance. They also give each coordinate's Student-t degrees of
freedom and fraction of missing information, the average relative increase in variance, and
the pooled Wald test that every coordinate is zero (Li, Raghunathan and Rubin's `D1`). A
Bayesian analysis stacks the joint draws of every table. The dependence between coordinates
is kept. A quantity with no published variance is pooled by its mean over the tables. It is
reported with its between-imputation variance in `vector.extras`. It never enters the
covariance. A static mediation's total, direct and mediated components are that case. A
bootstrap reports a variance only for the requested contrast. The analysis reads the variance
from its own resampled or analytic uncertainty. A fit that publishes none refuses. A
distribution or a static mediation without bootstrap replicates is that case. The next
attempt is the request for that variance.

A randomized design adds the arm each row was assigned to the imputation model. The table
does not hold the arm. An outcome imputed without it would be drawn as if assignment did
nothing. The arm enters as a main effect. An arm-specific outcome relation the model omits
biases the imputed effect toward zero. A clustered design draws its parameters by whole
assignment-unit clusters. The design's own estimator and its declared assignment probabilities
run unchanged on every completed table.

A single series is imputed by a lagged model. It is not imputed by exchangeable rows. Each
incomplete variable is drawn given the other variables at the same step and every variable at
the `lags` neighbouring steps on each side. That neighbourhood is the deepest lag the query
reads. The neighbours are read from the current completed columns. The parameters are drawn
by a Bayesian bootstrap of contiguous blocks. The block length is the cube root of the
length, and at least one neighbourhood. The assumption is missingness at random given the
neighbouring steps. The series must be stationary. Its dependence must be no longer than the
conditioned neighbourhood. The dependent time-series design is the only design a series
accepts. The model is `chained_equations`. The multi-horizon Bayesian mediation reports a
separate posterior per horizon. It defines no joint posterior to stack. It is refused. The
next attempt is one horizon, or `complete_case`.

Every pooled result carries `assumptions_digest`. That digest covers the imputation model,
tables, sweeps, seed, missingness assumption, resampling unit, declared sampling design,
lagged neighbourhood and every variable's kind and imputed-cell count. The same digest enters
the data-snapshot identity beside the specification. Two analyses that differ in any of them
are different claims.

**NOTEARS, `complete_case`.** Rows with a missing value in any discovery variable are dropped.
They are reported in `result.row_accounting`. The graph equals the one learned on the complete
rows. It speaks for the full data only when missingness is completely at random.

**NOTEARS, `pairwise_available`.** The least-squares score depends on the data only through the
sample covariance. Each pair of variables uses the rows observed on both. The pairwise
covariance is repaired to positive semidefinite by clipping negative eigenvalues. When you
standardize, that covariance is a correlation. The solver scores that matrix exactly. No row
is deleted. The accounting lists the rows behind every pair and the number of eigenvalues
clipped. It needs missingness completely at random. It needs at least three rows on every
pair.

## Recover a law when one response indicator causes another

Two binary variables `X1` and `X2` may each be observed only when a response indicator is 1.
`R1` is the indicator for `X1`. `R2` is the indicator for `X2`. The second response may
depend on the first (`R1 -> R2`, optionally with `X1 -> R2`). Call
`antecedent.recovery_chain`. It asks whether the joint law `P(X1, X2)` is recoverable from
the observed pattern law under the graph you declare. It returns that law when it is. This
is graph-supported recovery. It is not complete-case analysis, missing-at-random adjustment
or inverse-probability weighting. In the worked fixture of the tests, the complete-case table
differs from the recovered one.

The graph is over `x1`, `x2`, their response indicators `r1` and `r2`, and deterministic
proxies `p1` and `p2`. `p_i = x_i` when `r_i = 1`. `p_i` is missing when `r_i = 0`. The
graph has exactly one response edge. The observed margin is a `PatternLaw`. It holds the 2x2
mass of both observed, the mass of only the first observed, the mass of only the second
observed, and the mass of neither.

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

The pattern-law masses are supplied by you from your data. They are not shown here. `both`
is a 2x2 nested list. The other three masses are as in the class. In the executing test they
come from an enumerated structural model. `recover_chain` reproduces the enumerated truth to
`1e-12` for either head response, with or without the `X1 -> R2` edge. The answer is
point-only. `result.to_dict()["interval"]` reports `status` `point_only`. The answer is an
identity applied to the pattern law you supply. It carries no sampling uncertainty for masses
estimated from a finite sample. It does not depend on node order or labels. `result.export()`
recomputes the recovered law from the artifact. So does
`consume_recovery_chain_artifact(...)`.

Two refusals are different scientific information. A self-censoring edge (`X_i -> R_i`) with
every other edge in the class is nonrecoverable. The refusal carries an exactly verified
two-model witness (`refusal.witness`). The witness is two models that agree on every observed
cell and differ on the target. The witness models are degenerate. The refusal claims nothing
under faithful or generic parameters. A graph outside this class is `route_not_supported`.
Anything other than one response chain is outside the class. That code says the call does not
cover the graph. It does not say the law is nonrecoverable. A zero-mass cell or an invalid
law is refused as well. Both raise `RecoveryChainRefusal`, a [structured refusal](../reference/refusal-codes.md#structured-refusals-and-their-remedies) with `code`, `detail` and `remedy`.

## Claims

A claim over a result whose policy differs from the analysis default carries three evidence
references: the policy, the assumption it needs and the digest of the dropped rows. A
multiple-imputation claim also names the specification, the pooled variance components and the
imputation's assumptions digest. A claim under the analysis default carries none of these
references.

## Unbalanced panels

In Rust, call `antecedent_data::PanelLayout::assess`. Pass `subjects`, `periods` and
`observed`. You get a long panel before any design reads it. The report contains the calendar
grid, each unit's first and last observed period, its internal gaps as calendar runs, late
entry and attrition, and the cells expected and present. It also contains an attrition table
per period: units at risk, observed, entering and leaving. Given a baseline column, it
contains the standardized difference between units that later leave and units that stay.
Duplicate unit-periods are refused. Call `PanelBalanceReport::prepare`. It keeps one of three
layouts:

- whole complete units (`BalancedUnits`)
- every complete unit-period with its identities (`AvailableUnitPeriods`)
- each unit's maximal runs of consecutive periods (`ContiguousRuns`)

No lag or difference spans a gap or crosses a unit. The call produces the row accounting
above. `require_balance` refuses for designs that need a balanced panel. Those designs are
two-period panels, synthetic control, a balanced event study and an augmented panel. The
refusal names the units that break balance and the next attempt. `require_lag` refuses a lag
that would read across a gap. No difference-in-differences estimator is run by this layer.

Two designs also read an unbalanced panel directly. This layer does not relabel anything.
A staggered rollout with `sun_abraham` fits the saturated cohort-by-period regression on the
observed unit-periods. It reports per-cohort support and the unit-periods it dropped. Its
`observation` declaration states the observation process. `ignorable_given_cohort` is recorded
as an untestable assumption. `informative` is refused. A `SyntheticControl` with
`observed_window_support=True` fits over the treated unit's observed window. The donors must
be observed throughout that window. See [Panel designs](designs/panel-designs.md).

## Intervals

A handled missingness changes the variance construction. No interval is published for it.
Responder-weighted and complete-case randomized effects, restricted panels and temporal fits
on series with gaps report a point estimate and a standard error where one exists. The
interval is not reported. Multiple imputation publishes the pooled standard error and its
degrees of freedom. The whole impute-fit-pool procedure is one uncertainty path. Any interval
would have to cover it as one construction.
