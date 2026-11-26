# Plan power, sample size and minimum detectable effect

Plan a randomized experiment before any data exist. You can ask how many units you need for a target power. You can ask how much power a given size gives. You can ask for the smallest effect a given size can detect. You declare the design. You declare the assumptions its operating characteristics rest on. You declare a criterion and a goal. Antecedent solves the plan analytically. When you ask, it simulates the plan end to end with the estimator you will analyse the data with.

The planner is `antecedent.design.planning`. In Rust, the planner is `antecedent::analysis::planning`. The plan's identity includes every assumption. Changing an assumption changes the plan's identity.

```python
from antecedent.design import planning

plan = planning.plan_experiment(
    planning.Cluster(mean_cluster_size=20, icc=0.05),
    planning.Assumptions.continuous(sd=1.0, effect=0.3, attrition=0.1),
    planning.Power(0.8),
    planning.SampleSize(),
    simulation=planning.Simulation(replicates=2000, seed=7),
)
plan.units                    # clusters to randomize
plan.arm_units                # clusters per arm, control first
plan.analytic.design_effect   # 1 + ((1 + CV²) m − 1) ρ at the analysed cluster size
plan.analytic.value           # analytic power at the planned size
plan.simulation.value         # simulated power, with plan.simulation.monte_carlo_se
```

The Rust entry is the same request:

```rust
use antecedent::analysis::planning::{
    PlannedDesign, PlanningAssumptions, PlanningCriterion, PlanningGoal, PlanningRequest,
    plan_experiment,
};

let request = PlanningRequest::new(
    PlannedDesign::Complete,
    PlanningAssumptions::continuous(1.0).with_effect(0.5),
    PlanningCriterion::Power { target: 0.8 },
    PlanningGoal::SampleSize { max_units: 100_000 },
)
.with_simulation(2_000, 7);
let planned = plan_experiment(&request, &ctx)?;   // 128 units, 64 per arm
```

In Rust, call `compile_planning`. You get the plan without solving it. In Rust, call `PlanningPlan::execute`. That call solves and simulates. In Rust, call `Study::compile_for_planning`. That call attaches the same request to a randomized study whose declared randomization is the planned design. You can then plan the next trial of that analysis.

## Three directions

| goal | solves | reported |
| --- | --- | --- |
| `SampleSize(max_units)` | the smallest size whose criterion reaches its target | `units`, and the criterion there (at or above the target) |
| `Evaluate(units)` | nothing | the criterion at `units` |
| `MinimumDetectableEffect(units)` | the smallest positive effect detected with the target power | `effect`, and the power there |

A sample size for a target minimum detectable effect is `SampleSize` with that effect declared.
The size counts the design's randomization unit (`plan.unit`): units, clusters, or switch blocks
per sequence. Every solve is inverse-consistent. Re-evaluating the planned size reaches the
target. One unit fewer does not.

## Designs and the analysis each is planned for

| design | analysis | reference law |
| --- | --- | --- |
| `Bernoulli()` | difference in means, or ANCOVA with declared covariates | binomial mixture over realized arm sizes of the noncentral t |
| `Complete()` | difference in means, or ANCOVA | noncentral t, `n − 2 − p` df |
| `Stratified(strata)` | block-weighted difference in means | noncentral t, `n − 2H` df |
| `Cluster(m, icc, cv)` | cluster-total contrast | noncentral t, `k − 2` df, design effect `1 + ((1 + CV²) m − 1) ρ` |
| `MultiArm(K, …)` | each arm against control (Bonferroni, Holm or Dunnett), or the one-way F test | equicorrelated multivariate t through the shared control; noncentral F |
| `Factorial(contrast)` | two-by-two cell contrasts (main effect or interaction) | noncentral t, `n − 4` df |
| `Switchback(S, L, …)` | sequence-clustered Horvitz–Thompson contrast | noncentral t, `S − 1` df |
| `GroupSequential(fractions, spending)` | z statistic at each look against an alpha-spending boundary | recursive integration of the canonical joint law |

A binary outcome is tested by its Wald statistic against the normal law. The arm
variances are those under the alternative. Notes on the less familiar rows:

- **Multi-arm.** The comparison statistics share the control. Their correlation is
  `n_t / (n_t + n_c)` under equal variances. Conditional on the shared control and the shared
  variance estimate, they are independent. The number of rejections of a single-step
  procedure (Bonferroni or Dunnett) or a step-down procedure (Holm) has an exact distribution.
  That distribution is dynamic programming over the ordered critical values. `family_power`
  chooses what power means. `"per_comparison"` is power per comparison. `"any"` is disjunctive
  power. `"all"` is conjunctive power. The Dunnett critical value is solved from the same joint law.
- **Cluster with unequal sizes.** The cluster-total analysis depends on the outcome level.
  With `cluster_size_cv > 0`, a continuous outcome must declare `control_mean`. The variance adds
  `μ² CV²` per cluster to the design-effect term.
- **Switchback.** Assignment is randomized per block of `L` periods. Within-block first-order
  autocorrelation `φ` inflates a block's variance by `1 + 2 Σ (1 − k/L) φ^k`. With independent
  block assignments the cross-block correlation does not enter. The Horvitz–Thompson analysis
  depends on the outcome level. You must declare `control_mean`. Carryover into the next period
  requires `washout_periods ≥ 1`. The effect is then the global (always-on against always-off)
  effect.
- **Group sequential.** Boundaries come from the alpha-spending recursion used by adaptive
  inference (O'Brien–Fleming-type or Pocock-type). The plan reports the boundary. It reports the
  stopping probability at each look under the effect and under no effect. It reports the expected
  enrolment under both. When it solved a size, it reports the fixed-design size for the same power.

Covariate adjustment reduces the outcome variance by `1 − R²`. Declare it with
`covariate_r_squared`. You can also declare it with `covariates`. It is planned for Bernoulli
and complete randomization, whose analysis adjusts. Attrition reduces the analysed units to
`(1 − attrition)` of those enrolled.

## Frequentist and Bayesian criteria

| criterion | value | computed by |
| --- | --- | --- |
| `Power(target)` | rejection probability of the planned test at the assumed effect | the reference law above |
| `Assurance(design_prior, Significance())` | power averaged over the design prior | Gauss–Hermite quadrature of the analytic power |
| `Assurance(design_prior, PosteriorSuccess(analysis_prior, threshold, probability))` | probability that the posterior puts `probability` above `threshold` | conjugate normal closed form |
| `PosteriorPrecision(analysis_prior, max_posterior_sd)` | posterior standard deviation | conjugate normal closed form; `credible_half_width` converts a half-width target |
| `CorrectDecision(design_prior, analysis_prior, threshold, probability)` | probability that the posterior decision about `effect > threshold` is right | one-dimensional quadrature of the conjugate decision rule |

The posterior forms use the planned estimator's normal law with a conjugate normal analysis
prior. An assurance target above what the design approaches as it grows (the design-prior
probability of an effect beyond the success threshold) is refused with that limit stated.
Posterior criteria are planned for a single contrast. A multi-arm family plans power or
significance assurance. A sequential boundary plans power or significance assurance.

## Simulation

`Simulation(replicates, seed)` draws the trial the assumptions declare. It runs the estimator
named in `plan.analysis`. It applies the planned decision. It replicates. That estimator is one
of these:

- `complete_unit_itt`
- `fit_ancova`
- `blocked_unit_itt`
- `complete_cluster_itt`
- `factorial_2x2`
- `switchback_itt`
- per-look estimates for a sequential design

A replicate whose estimator returns no estimate counts as no rejection. The result reports the
simulated criterion and its Monte Carlo standard error. It reports the mean and spread of the
estimates. It reports familywise and stopping frequencies. It reports `agreement_z`, the gap to
the analytic value in standard errors.

Each replicate opens its own stream keyed by the planning operation and the replicate index.
Replicates reduce in index order. The simulation is the same at every thread count. The
replicate records and one data scratch per worker are quoted with checked arithmetic. They are
leased from the execution context's admission pool before anything is allocated. If the pool
cannot hold that store, the call refuses with `CausalResourceError`. Running the plan refuses
the store at the planned size the same way.

## Refusals

| request | reason code | next attempt |
| --- | --- | --- |
| `Interference()` without an exposure model | `required_option_missing` | `declare_exposure_model`: declare interference contained in randomized clusters (`ContainedInClusters`), planned as the cluster design |
| `OutcomeAdaptive()` without a rule | `adaptive_contract_incomplete` | `prespecify_stopping_rule`: plan a `GroupSequential` boundary |
| `OutcomeAdaptive(rule=…)` | `route_not_supported` | `simulate_adaptive_procedure`: the adaptive procedure contract and its simulation |
| switchback carryover without washout | `effect_not_identified` | `declare_washout` |
| level-dependent analysis without `control_mean` | `required_option_missing` | `declare_control_mean` |
| covariate adjustment on a design whose analysis does not adjust | `option_not_applicable` | `covariate_adjusted_design` |
| a target not reached within `max_units` | `evidence_cannot_reach_level` | `relax_target` |
| `Study::compile_for_planning` (Rust) on a study whose randomization is not the planned design | `route_not_supported` | `standalone_planning` |

The reason codes and their remedies are listed in [refusal codes](../reference/refusal-codes.md#other-registered-refusal-codes).

## Feeding design ranking

A plan's sample size is the sample of a ranking candidate. In Python, call
`plan.sampling_plan(cost_per_individual)`. Call `plan.experiment_plan(targets, cost_per_individual)`
for the experiment plan. In Rust, call `PlanningResult::sampling_candidate`. Call
`experiment_candidate` for the experiment candidate. Call `future_data` for a
`NormalMeanFutureData` whose mean has the planned estimator variance. Pass the candidates to
`rank_designs` unchanged. You can also pass them to `rank_designs_bound`. See
[design ranking](../design-ranking.md).

## What a plan claims

A plan is a point statement under the declared assumptions. The reference laws are exact for
the stated models up to the quadrature and series tolerances. Noncentral t uses Lenth's
algorithm. Above a noncentrality of 37.5 it uses the Johnson–Welch normal form. A Wald test of a
binary outcome is a normal or large-sample approximation. An HC0 ANCOVA variance is the same
kind of approximation. A sandwich over few sequences is the same kind of approximation. The
simulation shows their finite-sample error. Every plan reports the calibration status of its
procedures in `plan.calibration`. A plan carries no coverage claim.

## Related

- [Analyze a randomized or factorial experiment](designs/randomized-experiments.md) for the analysis you are planning for.
- [Adaptive and group-sequential inference](../adaptive-inference.md) when the plan stops early or adapts.
- [Design ranking](../design-ranking.md) to compare planned designs by information value and cost.

For a connected HCI investigation spanning planning, follow-up selection and a final claim audit, see [from day zero to publication](../tutorials/research-from-day-zero.md); its runnable script, [`examples/python/research_from_day_zero.py`](https://github.com/iridae-dev/antecedent/blob/v3.1.0/examples/python/research_from_day_zero.py), plans participant recruitment with this planner.
