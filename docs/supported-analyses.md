# Choose a question and an estimator

Use this page to go from the question you want answered to the query that states it, the
estimators that can compute it, and the guide that shows the call. You need a table and either
a causal graph you can defend or a design (an experiment, a panel, a threshold) that identifies
the effect without one. If you have not run an analysis yet, start with
[your first analysis](start/first-analysis.md).

## How to choose

1. **State the question as a query.** Find your question in the tables below. The query
   (`ant.AverageEffect`, `ant.ConditionalEffect`, `ant.ResponseCurve`, ...) fixes the
   estimand: what is estimated, in which population, under which intervention.
2. **Declare the structure.** Pass `graph=` (a `Dag`, `Cpdag`, `Pag`, `Admg`, a temporal graph,
   or a list of edges). Preparation then finds an identification strategy and an adjustment set,
   or refuses with `effect_not_identified` when the graph does not identify the query. Design
   queries in the last table carry their own identifying assumptions and take no graph.
3. **Pick an estimator, or leave it to the engine.** `estimator=None` selects the engine's
   default for the identified estimand. Pass an id (`estimator="aipw"`) or a typed
   configuration from `antecedent.estimators` (`estimator=Aipw(...)`) to choose one.
   Bayesian estimators need `inference=ant.Bayesian(...)`.
4. **Check support before relying on the result.** Prepare first and read what preparation
   established, then estimate:

```python
import antecedent as ant
from antecedent.estimators import Aipw

study = ant.prepare(
    data,
    graph=[("z", "t"), ("z", "y"), ("t", "y")],
    query=ant.AverageEffect("t", "y"),
    estimator=Aipw(),
)
print(study.inspect())         # identification, adjustment set, support, before any fit
result = ant.estimate(study)   # an Outcome: read result.refusal, then result.answer
```

**Supported** means that a combination has a provenance record and an executed independent scientific check, so it runs. A combination without that evidence is `not_supported` and a release build refuses it; a development build runs it and reports `would_refuse` (see [the evidence model](guides/scientific-verification.md#the-evidence-model)). Supported does not establish that your graph is correct or your data are adequate, and it does not mean the cell's interval coverage was measured: each reported interval carries a calibration status, and the [support matrix](support-matrix.md) counts the supported cells that have no coverage measurement for their estimator. Use [support lookup](support-lookup.md) to filter exact graph-based cells, or [the graphless matrix](graphless-support-matrix.md) for design-family studies.

## Effects of a treatment, identified from a graph

`T` is the treatment, `Y` the outcome and `Y(a)` the outcome a unit would have if `T` were set
to `a`. The average effect of a binary treatment is `E[Y(1)] - E[Y(0)]`; the
[glossary](concepts/glossary.md) defines identification and support.

| Your question | Query | Estimator options | Guide |
|---|---|---|---|
| What is the average effect of a binary treatment? | `AverageEffect(treatment, outcome)` | [Average-effect estimators](#average-effect-estimators) below | [Your first analysis](start/first-analysis.md), [an observational study](tutorials/observational-study.md) |
| The treatment has labels or more than two levels | `AverageEffect(..., control_level=, active_level=)`; `multi_level_treatment`, `categorical_treatment`, `vector_treatment` | The propensity estimators contrast any two declared levels | [Several treatment levels](guides/treatment-levels.md) |
| How does the effect vary with a measured modifier? | `ConditionalEffect(treatment, outcome, modifier)` | `conditional.linear.adjustment`; Bayesian `conditional.bayesian`, `bayesian.basis.gcomp` | [Heterogeneous effects and policies](guides/heterogeneous-effects-and-policies.md) |
| How does the mean outcome change along a continuous dose, or how steep is that curve? | `ResponseCurve`, `PointDerivative`, `Elasticity`, `SemiElasticity`, `AverageDerivative`, `DirectionalDerivative`, `ResponseJacobian` | `response.kennedy_dr`, `response.riesz_ade`, `response.gam_derivative`; Bayesian `response.bayesian` | [Responses, derivatives and distributions](causal-responses.md) |
| What is the mean outcome under a set, shifted or random intervention, or a joint intervention on several treatments? | `InterventionResponse` | `response.intervention_gcomp`, `cell.aipw`; Bayesian `response.bayesian` | [Responses, derivatives and distributions](causal-responses.md#intervention-responses), [joint binary interventions](guides/joint-cells.md) |
| What is the outcome distribution, a quantile or an exceedance probability under each arm? | `AverageEffect(..., outcome_functional=Quantile(tau))` or `ExceedanceGrid(...)`; `InterventionalDistribution` | `aipw` for quantile contrasts; `functional.distribution` | [Read a CDF and its supported bands](causal-responses.md#read-a-cdf-and-its-supported-bands) |
| How much of the effect runs through a mediator? | `MediationEffect`, `PathSpecificEffect`, `NestedCounterfactual` (linear models on a graph); `nonlinear_mediation.mediation_effects` (declared premises, no graph) | `mediation.linear` | [Mediation and latent classes](guides/mediation-and-latent-classes.md) |
| What would this unit's outcome have been under another treatment? | `Counterfactual(treatment, outcome, control_level=, active_level=)` | `gcm.fit` | [Fit and query a structural causal model](guides/gcm-estimands.md), [cross-world quantities](cross-world-execution.md) |
| Which upstream variables explain an anomaly or a change in distribution? | `AnomalyAttribution`, `ChangeAttribution` on an explicit `Dag` | `gcm.fit`; Bayesian `gcm.fit.bayesian`, `gcm.attribution.bayesian` | [`root_cause_attribution.py`](https://github.com/iridae-dev/antecedent/blob/v3.1.0/examples/python/root_cause_attribution.py), [capabilities: attribution](capabilities.md#attribution-and-diagnostics) |
| What does a pulse or a sustained intervention do over later time steps? | `PulseEffect`, `SustainedEffect`, `TemporalConditionalPulse`, `TemporalConditionalSustained`, `TemporalMediationEffect` on a temporal graph | `temporal.linear.adjustment`, `temporal.glm_link_contrast`, `temporal.lagged_instrument`, `temporal.sequential.gcomp`; Bayesian `bayesian.temporal.gcomp` and its variants | [Set the temporal history window](python-options.md#set-the-temporal-history-window), [one feedback system, three questions](tutorials/feedback-three-questions.md) |
| Does a treatment spill over to neighbours or other units in a cluster? | `InterferenceQuery` | `interference.ht_hajek`, `interference.cluster_neyman`, `interference.saturation_exact`, `interference.observational_ipw` | [Interference and network exposure](guides/designs/interference-and-network-exposure.md) |
| What is the effect in another population? | `antecedent.transport.Transport` | `transport.trial_ipw`; Bayesian `transport.trial_bayesian_bootstrap` | [Carry an effect to another population](guides/transport/carry-an-effect.md) |

## Average-effect estimators

For `AverageEffect` identified by back-door adjustment, every estimator below targets the same
estimand; they differ in what they model and in what their uncertainty conditions on. Each
class is in `antecedent.estimators`, and its `estimator_id` is the string you can pass instead.

| Estimator | Id | What it models | Choose it when | Guide |
|---|---|---|---|---|
| `LinearAdjustment` | `linear.adjustment.ate` | A linear outcome regression on `T` and the adjustment set (OLS, ridge, lasso or Huber fit) | The outcome is close to linear in the adjustment variables | [Your first analysis](start/first-analysis.md) |
| `GlmAdjustment` | `glm.adjustment` | A GLM outcome model (binomial, Poisson, negative binomial, Gaussian) | A binary or count outcome | [Python workflow reference](python-options.md) |
| `PropensityWeighting` | `propensity.weighting` | The propensity (the probability of treatment given the adjustment set), with Hajek-normalized inverse weights | You can model treatment assignment better than the outcome | [Weighting example](https://github.com/iridae-dev/antecedent/blob/v3.1.0/examples/python/propensity_weighting.py), [when overlap fails](tutorials/when-overlap-fails.md) |
| `PropensityMatching`, `DistanceMatching` | `propensity.matching`, `distance.matching` | Nearest-neighbour matches on the propensity or on a covariate distance | You want matched comparisons | [Python workflow reference](python-options.md) |
| `PropensityStratification` | `propensity.stratification` | Propensity strata with a within-stratum outcome model | You want a stratified summary | [Python workflow reference](python-options.md) |
| `Aipw` | `aipw` | Both the propensity and the outcome model; the point is consistent if either is correct | You want a doubly robust point estimate | [Penalized propensity](guides/penalized-aipw.md), [clusters](guides/clustered-dml.md) |
| `DML`, `DRLearner`, `CausalForest` | `dml`, `dr.learner`, `causal.forest` | Cross-fitted flexible nuisance learners; DR-Learner and the forest also fit effect heterogeneity | The nuisance functions are not well described by a parametric model | [Reuse DML scores](guides/dml-scores.md), [external learners](guides/external-learners-and-forecasts.md) |
| Bayesian g-computation | `bayesian.gcomp`, `bayesian.basis.gcomp`, `bayesian.robust_ate` | A posterior over the outcome model, with `inference=ant.Bayesian(...)` | You want a posterior rather than a confidence interval | [Choose a Bayesian likelihood](python-options.md#choose-a-bayesian-likelihood) |

When the graph identifies the effect by another strategy, the estimator follows it:

| Identification | Estimator options | Guide |
|---|---|---|
| Front-door (a mediator that intercepts every directed path from `T` to `Y`) | `frontdoor.functional`, `FrontdoorLinearTwoStage` (`frontdoor.linear_two_stage`) | [Capabilities: identification](capabilities.md#identification) |
| Instrumental variable | `IvWald` (`iv.wald`), `Iv2Sls` (`iv.2sls`); Bayesian `iv.bayesian_joint_linear`. The Wald ratio is the average effect only under a constant linear effect; otherwise, under monotonicity, it is the effect among compliers | [Capabilities: identification](capabilities.md#identification) |
| A class of graphs (`Cpdag`, `Pag`) whose edge directions remain uncertain | Check whether the answer is one effect, a set of effects or bounds before choosing an estimator | [Graph uncertainty](graph_uncertainty.md) |

An estimator that does not support the query, graph or inference is refused
(`route_not_supported` or `estimator_inference_mismatch`) rather than replaced. Row-level
declarations apply across estimators: [missing values](guides/missing-data.md),
[sampling weights](guides/sampling-weights.md) and [clusters](guides/clustered-dml.md).

## Effects identified by a design, without a graph

These queries carry their own identifying assumption (randomization, parallel trends, a
threshold, a known censoring law) and take no graph. A matching row of the
[graphless support matrix](graphless-support-matrix.md) supports that claim but is not a
coverage record, and a design with no matching row is not supported.

| Your question | Query | Guide | Example |
|---|---|---|---|
| What did a randomized or factorial experiment show? | `experiment.RandomizedEffect`, `ComplierEffect`, `TreatmentOnTreated`, `SwitchbackEffect`; `factorial` | [Randomized and factorial experiments](guides/designs/randomized-experiments.md) | [`experiment_randomized_effect.py`](https://github.com/iridae-dev/antecedent/blob/v3.1.0/examples/python/experiment_randomized_effect.py) |
| What did a policy change do to a panel of units? | `quasi.PanelDifferenceInDifferences`, `StaggeredAdoption`, `SyntheticControl`, `SyntheticDifferenceInDifferences`, `AugmentedPanelDiD` | [Panel designs](guides/designs/panel-designs.md) | [`quasi_difference_in_differences.py`](https://github.com/iridae-dev/antecedent/blob/v3.1.0/examples/python/quasi_difference_in_differences.py) |
| What happens at an eligibility threshold? | `quasi.SharpRegressionDiscontinuity`, `FuzzyRegressionDiscontinuity`, `RegressionKink`; `IvBounds` for a binary instrument | [Discontinuity and kink designs](guides/designs/discontinuity-and-kink.md) | [`discontinuity_and_iv.py`](https://github.com/iridae-dev/antecedent/blob/v3.1.0/examples/python/discontinuity_and_iv.py) |
| What would a treatment rule be worth? | `policy.PolicyValue`, `MultiActionPolicyValue`, `ConditionalDoseResponse`; `observational_policy.policy_value` | [Heterogeneous effects and policies](guides/heterogeneous-effects-and-policies.md) | [`policy_value_evaluation.py`](https://github.com/iridae-dev/antecedent/blob/v3.1.0/examples/python/policy_value_evaluation.py) |
| How does treatment change survival time? | `survival.SurvivalOutcome`, `CompetingRisksOutcome` | [Survival outcomes](guides/designs/survival-outcomes.md) | [`survival_rmst.py`](https://github.com/iridae-dev/antecedent/blob/v3.1.0/examples/python/survival_rmst.py) |
| What is the value of a sequence of treatment decisions? | `regimes.LongitudinalRegime`, `SequentialContrast` | [Sequential treatment regimes](guides/designs/sequential-regimes.md) | [`sequential_regimes.py`](https://github.com/iridae-dev/antecedent/blob/v3.1.0/examples/python/sequential_regimes.py) |
| What is the conditional odds ratio in matched case-control sets? | `matched.matched_case_control_odds_ratio` | [Matched case-control odds ratio](guides/matched-case-control.md) | |

## When the answer is in doubt

| Your situation | Where to go |
|---|---|
| The graph does not identify the effect | [When the effect is not identified](guides/fallback.md): a labelled noncausal description and a plan for the evidence that would identify it, beside the refusal |
| Edge directions or the graph itself are uncertain | [Graph uncertainty](graph_uncertainty.md) and [check a graph against data](guides/graph-falsification.md) |
| You want to stress the reported quantity | [Refute the reported quantity](guides/target-native-validation.md) |
| The conclusion rests on an assumption the data cannot test | [Probe sensitivity to assumptions](sensitivity-and-robustness.md); the range is an assumption range, not a confidence interval |
| The table may make a fit refuse | [Check a table before estimating](guides/preflight-diagnostics.md) and [data quality diagnostics](guides/data-diagnostics.md) |
| You can reuse earlier evidence | [Earlier posteriors as priors](priors.md) |

## Support is checked per combination

These are starting points, not permission to combine arbitrary options. The [full support matrix](support-matrix.md#supported-cells) is the authoritative list. Its [refusal reasons](support-matrix.md#refusal-reasons) explain excluded combinations. `n/a` means the combination does not define a valid question.

For an analysis you have configured, `ant.prepare(...)` checks identification without estimating. Use `study.inspect()` to read what is known before calling `ant.estimate(study)`. Preparation itself can refuse an unsupported request; see [handling refusals](start/first-analysis.md#when-an-analysis-is-refused).

Specialized decision, ranking, recalculation and measured-inference adapters
have their own checked contracts; they do not add arbitrary combinations to the
analysis grid. Start with the [decision contract](decisions.md) and consult the
[measured interval scopes](population-time-uncertainty.md).
