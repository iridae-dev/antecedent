# Target-native validation and sensitivity ranges

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

Replicate refuters test the **whole vector** with one joint statistic: the
squared Mahalanobis distance of the expected vector from the replicate mean
under the replicate covariance, referred to a chi-square law on the target
dimension. On one coordinate this is the squared z-score of the scalar
refuters; on several it keeps their correlation instead of running marginal
tests. Each refit's random stream is keyed by refuter and replicate, so a
report does not depend on the thread count, and the replicate buffer is leased
from the admission pool before it is allocated.

## Refuting a prepared study's own target

```python
import antecedent as ant

study = ant.prepare(data, graph=graph, query=query, refute="none")
outcome = study.refute_target(data, suite="full", replicates=30)
outcome.target.family      # e.g. "response_derivative"
outcome.reports            # one TargetCheck per refuter that ran
outcome.refused            # refuters this study cannot construct, with the next attempt
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
  lag window, so no lag is read across a block join. A route without a weighted
  construction (anything but the frequentist single-window temporal linear
  adjustment today) records the refuter as refused with the route's reason.
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
  the reported point derivative is checked against it. First derivatives on the
  identity scale; an average derivative, a directional derivative or a Jacobian
  is refused with the next attempt.
- **Bayesian prior sensitivity.** The study is compiled under each alternative
  isotropic prior scale (`prior_scale_factors` times its own) and the posterior
  of the whole target is read back. An explicit coefficient prior set has no
  single scale and is refused with the next attempt.
- **Posterior-predictive check.** Under the reference linear-Gaussian outcome
  model of the treatment and adjustment set, each posterior draw's noiseless
  outcome gives the target of the draw and its noisy replicate gives a
  replicated target, both through the study's program. Independent rows only.

From Python, `refute_target` perturbs independent rows; series and panels are
refuted from Rust through `PreparedTargetData::Series` and `::Panel`.

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

The marginal sensitivity model (`msm_sensitivity`) and the decision
sensitivity grid are unchanged; they remain the analyses for bounded
inverse-probability weights and for decisions.
