# Plan power, sample size and minimum detectable effect

Plan a randomized experiment before any data exist: how many units you need for a target power, how much power a given size gives, or the smallest effect a given size can detect. You declare the design, the assumptions its operating characteristics rest on, a criterion and a goal. Antecedent solves the plan analytically and, when you ask, simulates it end to end with the estimator you will analyse the data with.

`antecedent.design.planning` (Rust: `antecedent::analysis::planning`) holds the planner. The request is compiled to a lowered program that retains every assumption in its identity, so changing an assumption changes the plan's identity.

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

`compile_planning` returns the plan without solving it, and `PlanningPlan::execute` solves and simulates. `Study::compile_for_planning` attaches the same request to a randomized study whose declared randomization is the planned design, so you can plan the next trial of that analysis.

## Three directions

| goal | solves | reported |
| --- | --- | --- |
| `SampleSize(max_units)` | the smallest size whose criterion reaches its target | `units`, and the criterion there (at or above the target) |
| `Evaluate(units)` | nothing | the criterion at `units` |
| `MinimumDetectableEffect(units)` | the smallest positive effect detected with the target power | `effect`, and the power there |

A sample size for a target minimum detectable effect is `SampleSize` with that effect declared.
The size counts the design's randomization unit (`plan.unit`): units, clusters, or switch blocks
per sequence. Every solve is inverse-consistent: re-evaluating the planned size reaches the
target, and one unit fewer does not.

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

A binary outcome is tested by its Wald statistic against the normal law, with the arm
variances under the alternative. Notes on the less familiar rows:

- **Multi-arm.** The comparison statistics share the control, so their correlation is
  `n_t / (n_t + n_c)` under equal variances. Conditional on the shared control and the shared
  variance estimate they are independent, and the number of rejections of a single-step
  (Bonferroni, Dunnett) or step-down (Holm) procedure has an exact distribution by dynamic
  programming over the ordered critical values. `family_power` chooses what power means:
  `"per_comparison"`, `"any"` (disjunctive) or `"all"` (conjunctive). The Dunnett critical value
  is solved from the same joint law.
- **Cluster with unequal sizes.** The cluster-total analysis depends on the outcome level, so
  with `cluster_size_cv > 0` a continuous outcome must declare `control_mean`; the variance adds
  `μ² CV²` per cluster to the design-effect term.
- **Switchback.** Assignment is randomized per block of `L` periods. Within-block first-order
  autocorrelation `φ` inflates a block's variance by `1 + 2 Σ (1 − k/L) φ^k`; with independent
  block assignments the cross-block correlation does not enter. The Horvitz–Thompson analysis
  depends on the outcome level (`control_mean` is required). Carryover into the next period
  requires `washout_periods ≥ 1`; the effect is then the global (always-on against always-off)
  effect.
- **Group sequential.** Boundaries come from the alpha-spending recursion used by the adaptive
  inference routes (O'Brien–Fleming-type or Pocock-type). The plan reports the boundary, the
  stopping probability at each look under the effect and under no effect, the expected enrolment
  under both, and, when it solved a size, the fixed-design size for the same power.

Covariate adjustment (`covariate_r_squared`, `covariates`) reduces the outcome variance by
`1 − R²` and is planned for Bernoulli and complete randomization, whose analysis adjusts.
Attrition reduces the analysed units to `(1 − attrition)` of those enrolled.

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
Posterior criteria are planned for a single contrast; a multi-arm family or a sequential
boundary plans power or significance assurance.

## Simulation

`Simulation(replicates, seed)` draws the trial the assumptions declare, runs the estimator
named in `plan.analysis` (`complete_unit_itt`, `fit_ancova`, `blocked_unit_itt`,
`complete_cluster_itt`, `factorial_2x2`, `switchback_itt`, and per-look estimates for a
sequential design), applies the planned decision, and replicates. A replicate whose estimator
returns no estimate counts as no rejection. The result reports the simulated criterion, its
Monte Carlo standard error, the mean and spread of the estimates, familywise and stopping
frequencies, and `agreement_z`, the gap to the analytic value in standard errors.

Each replicate opens its own stream keyed by the planning operation and the replicate index,
and replicates reduce in index order, so the simulation is the same at every thread count. The
replicate records and one data scratch per worker are quoted with checked arithmetic and leased
from the execution context's admission pool before anything is allocated; compile refuses a
store the pool cannot hold, and execution refuses the planned-size store the same way
(`CausalResourceError`).

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
| `Study.compile_for_planning` on a study whose randomization is not the planned design | `route_not_supported` | `standalone_planning` |

## Feeding design ranking

A plan's sample size is the sample of a ranking candidate:
`plan.sampling_plan(cost_per_individual)` and `plan.experiment_plan(targets, cost_per_individual)`
in Python, and `PlanningResult::sampling_candidate`, `experiment_candidate` and `future_data`
(a `NormalMeanFutureData` whose mean has the planned estimator variance) in Rust. The candidates
go to `rank_designs` / `rank_designs_bound` unchanged; see [design ranking](../design-ranking.md).

## What a plan claims

A plan is a point statement under the declared assumptions. The reference laws are exact for
the stated models up to the quadrature and series tolerances (noncentral t by Lenth's
algorithm, with the Johnson–Welch normal form above a noncentrality of 37.5); a Wald test of a
binary outcome, an HC0 ANCOVA variance and a sandwich over few sequences are normal or
large-sample approximations whose finite-sample error the simulation shows. The calibration of
the simulation procedures has not been measured: every plan reports `calibration =
"unmeasured"` and carries no coverage claim.

## Related

- [Analyze a randomized or factorial experiment](designs/randomized-experiments.md) for the analysis you are planning for.
- [Adaptive and group-sequential inference](../adaptive-inference.md) when the plan stops early or adapts.
- [Design ranking](../design-ranking.md) to compare planned designs by information value and cost.
