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

Every refit perturbs the data and re-executes the prepared compiled program in
full, then reads the whole target vector back. Prepare with `refute="none"`:
a scalar suite configured on the study would run on every refit. Refuters the
generic perturbation cannot construct are refused with the next attempt, not
skipped:

- **Random common cause.** A prepared program has a fixed schema and graph;
  declare the simulated cause in the graph and prepare that study.
- **Negative controls.** Declare them (`negative_control_exposure=`,
  `negative_control_outcome=`); on a single series a negative control has no
  resampling law for its null spread.
- **Series bootstrap.** Rebuilding lags on a block-resampled raw series pairs
  values across block joins; use the route's block bootstrap of its lagged
  design, or subset stability.
- **Pre-trend and anticipation on one series.** A single series has no
  treatment onset, and under outcome-to-treatment feedback a lead is not a
  placebo; panel event-study leads carry these checks.
- **Derivative diagnostics.** A prepared response reports the derivative, not
  the fitted response; call `validation.derivative_consistency` and
  `validation.local_linearity` on the fitted response.
- **Bayesian prior and predictive checks.** Prepare under each alternative
  prior, or simulate replicated data from the posterior, and pass the draws of
  the target to `validation.prior_sensitivity_target` and
  `validation.posterior_predictive_target`.

From Python, `refute_target` perturbs independent rows; a series is refuted
from Rust through `PreparedTargetData::Series`.

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
