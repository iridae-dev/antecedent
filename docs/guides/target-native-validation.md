# Refute the quantity a result reports and bound an untestable assumption

A refutation speaks to one quantity. The `Refute` suites (`cheap`, `placebo`,
`full`) perturb the data and re-estimate a scalar average effect; they say
nothing about a conditional effect, a response derivative, a temporal effect
vector, a mediation decomposition, the mean outcome under several
interventions, or a joint contrast. This page describes the refuters that test
the quantity a result actually reports, and the sensitivity analyses that state
how far an untestable assumption can be relaxed before that quantity changes.

Every number here is a diagnostic. Pass thresholds are conventions with no
measured error rate, and none of these checks licenses an interval.

## The target a refutation speaks to

A **refutation target** names the quantity: its family (scalar average effect,
conditional effect, response derivative, temporal effect, mediation,
world-valued, joint contrast), its quadrant (static or temporal data,
frequentist or Bayesian inference), one label per coordinate, and the identity
of the compiled program that defines it. Two targets are the same only when
these identities agree, so a refutation computed under another program, for
another family or on other coordinates is another target.

A scalar average-effect refutation does not certify any other target. A claim
on a derivative, a conditional effect, a mediation decomposition or a vector of
world values that carries one is refused, naming the next attempt: run the
target-native suite on the claim's own target. Reports from the target-native
suite are bound into a certificate whose identity covers the target and every
verdict; a report on another target is refused.

## The catalogue

For every family and quadrant, `target_refutation_plan` lists the refuters:

| Refuter | What it does | Expected |
| --- | --- | --- |
| `target.placebo_treatment` | permute the treatment (a circular shift on a series) | the whole vector at zero (world values: no contrast between worlds) |
| `target.placebo_outcome` | permute the outcome (circular shift on a series) | zero |
| `target.random_common_cause` | add an independent common cause to the adjustment | unchanged |
| `target.subset_stability` | refit on random subsets (contiguous windows on a series) | unchanged |
| `target.bootstrap_stability` | refit the whole procedure on resamples | unchanged |
| `target.negative_control_exposure` / `_outcome` | swap in a declared negative control | zero |
| `target.pre_trend`, `target.anticipation` | temporal quadrants: treatment leads, pre-onset outcomes | zero |
| `target.sequential_ignorability` | mediation: range of mediator-outcome error correlation | break-even correlation above a bar |
| `target.finite_difference_consistency`, `target.local_linearity` | derivatives: the reported derivative against the fitted response | consistent; locally linear at the step |
| `target.prior_sensitivity`, `target.posterior_predictive` | Bayesian quadrants: alternative priors; replicated data | small posterior shift; observed target inside the predictive law |

Replicate refuters judge the **whole vector** with one joint discrepancy: the
squared Mahalanobis distance of the expected vector from the replicate mean,
in units of the replicate covariance (one replicate's spread, not the Monte
Carlo error of the mean). On one coordinate it is the squared standardised
shift; on several it keeps the correlation instead of running marginal checks.
The check passes when the discrepancy stays within the tolerance set by `alpha`
(its tail probability under a chi-square law on the target dimension is at
least `alpha`). This is a fixed yardstick on a stability or placebo
diagnostic, **not a hypothesis test with a nominal rejection rate**: for
bootstrap and subset stability the replicate mean minus the original estimates
the resampling bias, and with an unbiased procedure the discrepancy is about
`k / replicates` rather than chi-square with `k` degrees of freedom, so a clean
case rejects far below `alpha`. Each replicate report also carries `standardised_bias` (replicate mean minus
the expected vector, per coordinate, in replicate standard deviations) and
`standardised_bias_norm` (the Mahalanobis length of the same shift, the square
root of the discrepancy), the effect size behind the verdict. Subset stability
is inherently weakly powered against one influential observation: under
circular windows a mean-type statistic has a replicate mean equal to the
original whatever the outlier, while the replicate spread that standardises the
shift grows with it. Routes that read neither row weights nor an analysis mask
refit lagged resamples on contiguous sub-windows (a third of the series), never
on a block resample whose joins would pair values observed apart; attribution
and weight-reading routes keep the block resample with join-crossing design
rows hidden or carried as row weights. Its rejection rate on clean data and its power
against planted violations are measured separately; nothing here is a
calibration claim. Each refit's random stream is keyed by refuter and replicate,
so a report does not depend on the thread count, and the replicate buffer is
leased from the admission pool before it is allocated.

## Refuting a prepared study's own target

```python
import antecedent as ant
from antecedent import validation

study = ant.prepare(data, graph=graph, query=query, refute="none")
checks = validation.TargetRefutationSpec(suite="full", replicates=30)
outcome = validation.refute_target(study, checks=checks)
report = outcome.value     # typed TargetRefutation; inspect outcome.refusal first
report.target.family      # e.g. "response_derivative"
report.reports            # one TargetCheck per refuter that ran
report.refused            # refusers with reasons and next actions

# Reuse the exact original native result rather than execute it again.
original = ant.estimate(study)
validated = validation.refute_target(original, checks=checks)
# Different observations are an explicit immutable branch.
updated = ant.rebind(study, data=new_data)
updated_validation = validation.refute_target(updated, checks=checks)
```

Every refit re-executes a compiled program in full. Most refuters perturb the
data and run the study's own prepared program; where a refuter changes the
scientific problem, the study is compiled again with that one change and run.
Prepare with `refute="none"`: a scalar suite configured on the study would run
on every refit.

- **Random common cause.** A simulated cause is appended to the data and
  declared in the graph as a parent of the treatment and the outcome (on a
  temporal graph, at every lag the treatment appears at); the augmented study is
  compiled, prepared and run, so the program adjusts for it. With
  `common_cause_strength > 0` the cause also enters the outcome (and a
  non-binary treatment), and the refuter checks that adjusting for a declared
  confounder recovers the target. DAGs and temporal DAGs only: a graph posterior
  or a class with unresolved marks has no single structure to declare it in.
- **Negative controls.** Declared ones are used as given. Otherwise the graph
  supplies them: an exposure outside the adjustment set that is not a descendant
  of the treatment and is d-separated from the outcome given the adjustment set
  (on a temporal graph: none of its nodes reaches the outcome), and an outcome
  the treatment cannot reach and is d-separated from. On series and panels the
  default negative-control outcome is the lagged placebo, the outcome moved back
  past the graph's longest lag. A refusal is recorded only when the graph
  supplies no candidate. Spread comes from the design's own resample: rows,
  circular blocks of a series, or whole units of a panel.
- **Series bootstrap.** A circular-block bootstrap with the dependence-aware
  block length, carried as row multiplicities: each design row keeps its own
  lag window, so no lag is read across a block join. A series that carries
  sampling weights keeps them: the multiplicities multiply each row's own
  weight, and a route that reads no row weights is given the gathered
  circular-block series instead, in which every row carries the weight, the
  missing values and the analysis mask of the row it was gathered from (a row
  whose lag window crosses a block join gets weight zero). Independent
  environments (`PreparedTargetData::MultiEnv`) are perturbed one environment at
  a time by the same rules and re-run together; a panel resamples whole units,
  each with its own weights.
- **Placebo timing and pre-period outcome on one series.** `target.pre_trend`
  puts the treatment `pre_trend_lead` periods later in the treatment's role;
  its lags in the design are the actual treatment path, so the coefficient is
  the effect of a future treatment on the present outcome given the past.
  `target.anticipation` puts the outcome from before every treatment period the
  target reads in the outcome's role. Both assume the treatment does not respond
  to the outcome within the shift; rows without a shifted value get weight zero.
- **Derivative diagnostics.** The study's mean-curve query on the same data and
  graph is compiled and run at the five points the checks read (the derivative's
  own estimator when it serves a curve, otherwise the compiler's choice), and
  the reported derivative is checked against it. First derivatives on the
  identity scale; an average derivative is read at seven nodes of its weighting
  law, and a directional derivative or a Jacobian entry along a direction is
  read off the study's own intervention-response query at the shifted points.
- **Bayesian prior sensitivity.** The study is compiled under each alternative
  isotropic prior scale (`prior_scale_factors` times its own) and the posterior
  of the whole target is read back. An explicit coefficient prior set has no
  single scale: every Gaussian coefficient standard deviation of the set is
  multiplied by the factor around its own means (variances by the factor
  squared), and the study is compiled under each scaled set. A prior hydrated
  from a posterior artifact, or composed from external sources, exists only once
  the design is bound, so the same factor is applied where the route resolves it
  (`BayesianConfig::prior_variance_multiplier`, the squared factor on every
  Gaussian coefficient variance of the prior in force).
- **Posterior-predictive check.** Under the reference linear-Gaussian outcome
  model of the treatment and adjustment set, each posterior draw's noiseless
  outcome gives the target of the draw and its noisy replicate gives a
  replicated target, both through the study's program. Temporal data uses the
  native lagged reference model and preserves its series or unit structure;
  unsupported reference models remain individual refuter refusals.

`validation.refute_target` uses the exact native observations retained by the
preparation: independent rows, one series, a panel, independent environments,
or events already aligned to their declared grid. It never flattens trajectories
or converts a multidimensional target into an average effect. Native sampling
weights, active-row masks and the declared dependence structure follow each
perturbation. A route that cannot preserve them returns an individual refusal.

`TargetRefutationSpec` contains scientific settings; `Execution` contains the
seed, threads, resource limits, cache policy, cancellation and observers. The
operation uses the study's execution declaration unless one is supplied:

```python
checks = validation.TargetRefutationSpec(
    suite="full",
    replicates=30,
    pre_trend_lead=2,
    gathered_resamples=True,
    prior_scale_factors=(0.25, 4.0),
    prior_shift_tolerance=0.5,
    derivative_step=None,
    derivative_tolerance=0.1,
    linearity_tolerance=0.25,
)
outcome = validation.refute_target(
    original, checks=checks, execution=ant.Execution(seed=19, threads=2)
)
```

Other resolved defaults remain `alpha=0.05`, `subset_fraction=0.8`,
`mediation_rho_bar=0.1` and `common_cause_strength=0.0`. Negative-control
exposure/outcome and mediator names default to `None`; the graph supplies
compatible controls where possible. The default pre-trend lead is one period
and gathered resampling is off. A class posterior's common-cause perturbation
requires an explicit `within_class="uniform_over_members"` assumption; its
mass allocation is never silently chosen.

Successful outcomes contain a typed `TargetRefutation`. Its answer is partial
when some refuters refuse, with every reason and next attempt retained. If no
refuter can run, the outcome is a scientific refusal with the individual causes
and next actions. Failed diagnostic checks remain reports, not execution
exceptions. `passed` judges the informative reports that ran; a `True` value
does not remove partial standing or certify refused checks. It is `None` when
there are no informative reports. Bounds
remain sets, and a distribution must be queried through an explicitly declared
compatible response rather than silently becoming a mean.

`original_executions` records original target executions performed by this
operation: one for a fresh preparation, zero for the retained original result.
It does not count fits inside refuters. Native program and target identities
and the certificate remain available on the report. Inspection and notebook
representation read bounded retained metadata and execute nothing.

An independently loaded artifact has no live study or observations and cannot
be refitted. Supply a preparation or retained live outcome. Resource failure,
cancellation and callback failure remain typed exceptions; diagnostic operating
characteristics remain unmeasured.

## Sensitivity ranges

Each analysis returns an `AssumptionRange`: the assumption it relaxes, the
range of the sensitivity parameter it probed and the target (or its test) at
each value, and the break-even value at which the conclusion is explained away
or loses significance. The target identity is bound into the range's identity,
so a range computed for one target cannot be cited by a claim on another.

| Function | Assumption relaxed | Parameter | Break-even |
| --- | --- | --- | --- |
| `validation.evalue` | no unmeasured confounding, on the risk-ratio scale (VanderWeele and Ding 2017) | confounder risk ratio with treatment and outcome | the E-value of the estimate and of the confidence limit nearest the null |
| `validation.omitted_variable_bias`, `omitted_variable_bias_from_data` | no unmeasured confounding in a linear adjustment (Cinelli and Hazlett 2020) | partial R² with treatment and outcome | the robustness values `RV_q` and `RV_{q,α}`; benchmarks bound a confounder `k` times as strong as an observed covariate |
| `validation.rosenbaum_bounds`, `matched_sets_sensitivity` | random assignment within matched sets (Rosenbaum 2002) | hidden bias `Γ` | the largest `Γ` at which the upper-bound p-value stays at most `α` |
| `validation.mediation_sensitivity` | sequential ignorability of the mediator (Imai, Keele and Yamamoto 2010) | mediator-outcome error correlation `ρ` | the residual correlation, at which the indirect effect vanishes |

E-values convert odds ratios, hazard ratios and standardized mean differences
to the risk-ratio scale by the published approximations: an odds or hazard
ratio is itself the risk ratio for a rare outcome, a common outcome uses the
square root of the odds ratio or `(1 − 0.5^√HR) / (1 − 0.5^√(1/HR))`, and a
standardized mean difference `d` with standard error `s` uses `exp(0.91·d)`
with limits `exp(0.91·d ± 1.78·s)`.

The signed-rank bound is computed exactly up to 300 non-zero pairs and by the
normal approximation with the tie-corrected variance above. Matched sets with
several controls use the separable approximation of Gastwirth, Krieger and
Rosenbaum (2000). `validation.amplify(Γ, Λ)` returns the outcome effect `Δ` an
unobserved covariate with treatment effect `Λ` needs to produce `Γ`
(Rosenbaum and Silber 2009).

### Vector targets and posteriors

A claim about a conditional effect (one coordinate per stratum), a temporal
effect (one per horizon) or a joint contrast is explained away as soon as any
coordinate is, so its range is read at the **weakest coordinate**: the break-even
value is the smallest across coordinates and each tested value reports the
weakest coordinate's adjusted effect.

| Function | Reads | Weakest-coordinate rule |
| --- | --- | --- |
| `validation.evalue_vector` | one estimate and interval per label | smallest E-value; tested target is the smallest adjusted risk ratio oriented away from the null |
| `validation.omitted_variable_bias_vector` | one coefficient, standard error and degrees of freedom per label | smallest `RV_q` and `RV_{q,α}`; tested target is the smallest share of an estimate left |
| `validation.rosenbaum_bounds_vector` | matched-pair differences per label | smallest `Γ*` (absent when a coordinate does not reject at `Γ = 1`); tested value is the largest upper-bound p-value |
| `validation.evalue_posterior` | posterior draws per label | each draw is bias-adjusted at every strength; `test` is the joint posterior probability that every coordinate keeps its reported side, `break_even` the smallest E-value of a posterior median |
| `validation.omitted_variable_bias_posterior` | coefficient draws and the per-draw bias scale `sd(Y|D,X)/sd(D|X)` | each draw moves toward the null by `scale * sqrt(R²yz R²dz / (1 - R²dz))`; the break-even is the equal partial R² at which the posterior median (and the credible limit) reaches zero |

The confounding structure stays an assumption of the analysis; the posterior
variants carry uncertainty about the effect through it and do not make the
confounding assumption testable. A range is bound to its target, so a scalar
range cannot be cited for a vector claim.

### Mediation components and marginal structural model parameters

| Function | Reads | Assumption relaxed |
| --- | --- | --- |
| `validation.mediation_sensitivity_vector` | one mediator and several outcomes (or horizons); `component="indirect"` or `"direct"` | sequential ignorability of the mediator: the error correlation `ρ` of Imai, Keele and Yamamoto (2010). The total effect does not depend on `ρ`, so the direct effect is the total minus the indirect effect at `ρ`; the indirect effect vanishes at the residual correlation `ρ̃` and the direct effect at `w / sqrt(1 + w²)`, `w = (ρ̃ - β₁/A) / sqrt(1 - ρ̃²)`, `A = β₂σ₁/σ₂`. The break-even is the correlation of smallest magnitude at which some outcome's component reaches zero |
| `validation.mediation_sensitivity_posterior` | coordinate-major draws of the mediator coefficient, the two residual standard deviations, their correlation and the total effect | the same, draw by draw; `test` is the joint posterior probability that every outcome keeps its side, `break_even` the correlation at which the weakest posterior median reaches zero and `break_even_significance` the same for the credible limit |
| `validation.msm_parameter_sensitivity` | an outcome, the exposure terms of a marginal structural model and its subject weights | sequential exchangeability (no unmeasured time-varying confounding): the model is fitted by weighted least squares with the weights held fixed, each exposure parameter has a robust (HC1) standard error and the effective degrees of freedom `(Σw)²/Σw² - p`, and the partial-R² range is read at the weakest parameter |
| `model="mediation"` or `model="marginal_structural"` on `evalue_vector`, `evalue_posterior`, `omitted_variable_bias_vector`, `omitted_variable_bias_posterior` | the E-value or partial-R² analysis of a mediation component or an MSM parameter | no unmeasured mediator-outcome confounding (treatment unconfounded), or sequential exchangeability; the range records that assumption in its identity |

| `validation.mediation_path_sensitivity` | one mediator and several outcomes; `path="treatment_mediator"`, `"mediator_outcome"`, `"treatment_outcome"` (the direct effect given the mediator) or `"indirect"` | no unmeasured confounding of that path (each bound to its own assumption); for `"indirect"` both paths are confounded at the tested strength, each coefficient moving toward zero by its own worst-case bias, the share left is the product of the two paths' shares and the break-even is the equal partial R² at which that product falls to `1 - q` (at `q = 1`, the smaller of the two robustness values) |
| `validation.msm_parameter_sensitivity_refit` | as `msm_parameter_sensitivity`, with the stabilized-weight models of each time point | as above, with every weight model refitted in each unit-bootstrap replicate, so the standard error carries the cost of estimating the weights |

Each single path is the Cinelli and Hazlett analysis of that regression
coefficient (`T → M` in the mediator regression, `M → Y` and `T → Y` in the
outcome regression). The weight-refit standard error is the spread of the
bootstrap replicates; the robust error that holds estimated weights fixed is
reported beside it (`fixed_weight_standard_error`) and is conservative for an
estimated denominator model, so the plain `msm_parameter_sensitivity` is a
conservative analysis and the refit one the one that matches repeated sampling.

The marginal sensitivity model (`msm_sensitivity`) and the decision
sensitivity grid are unchanged; they remain the analyses for bounded
inverse-probability weights and for decisions.

## Related

- [Probe sensitivity to assumptions you cannot test](../sensitivity-and-robustness.md): decision-level assumption ranges, switch points and structural verdicts.
- [Preflight diagnostics](preflight-diagnostics.md) and [tier diagnostics](tier-diagnostics.md): checks before and after estimation.
