//! Coverage designs for the whole-procedure intervals of the difference-in-differences routes
//! beyond the retained two-period score interval: the heterogeneity-robust staggered rollout
//! (three estimators, both resampling schemes, both frequentist comparison groups), the
//! covariate-adjusted two-period route (three nuisance methods), the augmented panel with
//! supplied nuisance predictions, and the two Bayesian routes (the rollout's Bayesian
//! bootstrap credible interval and the two-period Normal-Inverse-Gamma posterior).
//!
//! Every design draws a panel whose parallel-trends assumption holds exactly and whose target
//! is known by construction, runs the public route, and scores the interval the route publishes
//! for its scalar summary at the reported 95% level:
//!
//! * rollout: every unit draws its cohort (never treated, adopting at 2 or at 4) with equal
//!   probability, over five periods with effect `3 * (period - cohort)`. The resample redraws
//!   whole units with their cohort, so the interval targets the population aggregate, whose
//!   cohort shares are equal: the six post-adoption cells `0, 3, 6, 9` and `0, 3` average to
//!   `3.5`. Fixing the cohort sizes instead would remove the cohort-share variation the
//!   resample includes, and the interval would cover a fixed-share target too often.
//! * covariate-adjusted: the effect on the treated is `2`, the untreated change is
//!   `1 + x` and the treated share follows `logistic(x)`, so the conditional-trends, logistic
//!   propensity and linear change models all hold. The covariate is uniform with unit
//!   variance, so the propensity stays inside `(0.15, 0.85)`: the overlap the route declares
//!   is bounded away from one, which an unbounded covariate would not give.
//! * augmented panel: the supplied propensity and untreated-change prediction are the true
//!   functions of the covariate, so the interval's blind spot (the predictions' own estimation
//!   error) is absent by construction; the effect on the treated is `1.5`.
//! * Bayesian two-period: homoskedastic normal changes, the difference in mean changes is `2`.
//!
//! The frequentist rollout, covariate-adjusted and augmented routes publish their resampled
//! interval on a payload and no scalar interval on the effect, so the runtime's calibration key
//! names no interval method for them. Their records take the runtime's key for everything it
//! reports (graph class, estimator, identification, functional) and declare the interval
//! as the pointwise interval from a whole-cluster resampled standard error, the way the
//! weighted panel design declares its cluster interval. The Bayesian routes bind the runtime's
//! posterior-quantile key. A design route has no graph and runs no validation suite, so every
//! record is keyed on the design axes: the query's name, structure `design`, validation
//! `not_applicable`. A replicate that publishes no interval is a skip.
//!
//! The designs are `#[ignore]`d measurement designs run by `scripts/measure_calibration.sh`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

mod common;

use std::sync::Arc;

use antecedent::{BayesianConfig, InferenceMode, PanelDidQuery, Study, StudyResult};
use antecedent_core::{
    CausalRng, DidAggregation, DidComparisonGroup, DidNuisanceMethod, DidResamplingScheme,
    ExecutionContext, HeterogeneityRobustDid, RolloutDidEstimator, VariableId,
};
use antecedent_data::TabularData;
use antecedent_kernels::{sigmoid, standard_normal};
use common::calibration::{
    Construction, CoverageTally, REPORTED_LEVEL, RecordKey, ScopeFacts, grid_n, grid_seed,
    map_replicates, n_sim,
};
use common::calibration_bind::constructions;

/// Interval method of the resampled routes' pointwise interval.
const RESAMPLED: &str = "bootstrap_se";
/// Interval method of the Bayesian routes.
const POSTERIOR: &str = "posterior_quantile";
/// Whole-cluster resamples per replicate of the frequentist routes.
const RESAMPLES: u32 = 199;
/// Posterior draws per replicate of the Bayesian routes: the count a posterior support row is
/// stated from.
const DRAWS: usize = 2_000;
/// Query name of every record.
const QUERY: &str = "PanelDid";
/// A design route runs no validation suite.
const NO_VALIDATION: &str = "not_applicable";
/// Simple aggregate of the rollout design.
const ROLLOUT_TRUTH: f64 = 3.5;
/// Effect on the treated of the covariate-adjusted design.
const ADJUSTED_TRUTH: f64 = 2.0;
/// Effect on the treated of the augmented panel design.
const AUGMENTED_TRUTH: f64 = 1.5;
/// Difference in mean changes of the two-period Bayesian design.
const TWO_PERIOD_TRUTH: f64 = 2.0;
const OUTCOME: VariableId = VariableId::from_raw(0);

/// One replicate's scored interval and the construction the runtime reported for it.
struct Scored {
    interval: (f64, f64),
    truth: f64,
    construction: Construction,
    scope: ScopeFacts,
    validation: &'static str,
}

/// Score `interval` against `truth` under the runtime's primary key for `result`, with the
/// interval fields a resampled route does not report declared as the pointwise interval of a
/// whole-cluster resample.
fn resampled(
    study: &Study,
    result: &StudyResult,
    interval: [f64; 2],
    truth: f64,
    replicates_ok: Option<u32>,
) -> Option<Scored> {
    let (mut construction, mut scope, _) =
        constructions(&study.inspect().ok()?, result).into_iter().next()?;
    construction.query = QUERY.into();
    construction.structure = "design".into();
    construction.interval_method = RESAMPLED.into();
    construction.se_kind = String::new();
    construction.dependence = "cluster".into();
    construction.reported_level = REPORTED_LEVEL;
    scope.replicates_ok = replicates_ok;
    Some(Scored {
        interval: (interval[0], interval[1]),
        truth,
        construction,
        scope,
        validation: NO_VALIDATION,
    })
}

/// Score `interval` against `truth` under the runtime's posterior-quantile key for `result`.
fn posterior(
    study: &Study,
    result: &StudyResult,
    interval: (f64, f64),
    truth: f64,
) -> Option<Scored> {
    let (mut construction, scope, _) = constructions(&study.inspect().ok()?, result)
        .into_iter()
        .find(|(construction, _, _)| construction.interval_method == POSTERIOR)?;
    construction.query = QUERY.into();
    construction.structure = "design".into();
    Some(Scored { interval, truth, construction, scope, validation: NO_VALIDATION })
}

/// Measure one design: every replicate is bound to the record's construction and scored.
fn measure(
    test: &'static str,
    dgp: &'static str,
    interval: &'static str,
    replicate: impl Fn(u64) -> Option<Scored> + Sync,
) {
    let mut tally = CoverageTally::for_record(RecordKey { test, dgp, interval }, REPORTED_LEVEL);
    for row in map_replicates(n_sim(), replicate) {
        // A replicate that publishes no interval is a skip (a miss), never a free pass.
        let Some(scored) = row else {
            tally.skip();
            continue;
        };
        tally.bind_validation(&scored.construction, scored.scope, scored.validation);
        tally.record(Some(scored.interval), scored.truth);
    }
    tally.assert();
}

// ------------------------------------------------------------------ staggered rollout

/// `units` units over five periods, each drawing its cohort (never treated, adopting at 2 or
/// at 4) with equal probability; the unit effect is `3 * (period - cohort)` from adoption on.
fn staggered(
    units: usize,
    rng: &mut CausalRng,
) -> (TabularData, Vec<Arc<str>>, Vec<i64>, Vec<i64>) {
    const COHORTS: [i64; 3] = [0, 2, 4];
    let count = units;
    let (mut y, mut units, mut periods, mut cohorts) = (vec![], vec![], vec![], vec![]);
    for i in 0..count {
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "a uniform draw on [0, 1) scaled by 3 is a nonnegative cohort index below 3"
        )]
        let cohort = COHORTS[((rng.next_f64() * 3.0) as usize).min(2)];
        let label: Arc<str> = Arc::from(format!("c{cohort}_{i:05}"));
        let alpha = standard_normal(rng);
        for period in 1..=5_i64 {
            let effect =
                if cohort > 0 && period >= cohort { 3.0 * (period - cohort) as f64 } else { 0.0 };
            y.push(alpha + 0.5 * period as f64 + effect + standard_normal(rng));
            units.push(Arc::clone(&label));
            periods.push(period);
            cohorts.push(cohort);
        }
    }
    let data = TabularData::from_f64_columns([("y", y.as_slice())]).expect("rollout outcome");
    (data, units, periods, cohorts)
}

fn rollout_spec(
    estimator: RolloutDidEstimator,
    comparison: DidComparisonGroup,
) -> HeterogeneityRobustDid {
    HeterogeneityRobustDid {
        estimator,
        comparison,
        aggregation: DidAggregation::Simple,
        ..HeterogeneityRobustDid::default()
    }
}

fn rollout_query(
    units: Vec<Arc<str>>,
    periods: Vec<i64>,
    cohorts: Vec<i64>,
    spec: HeterogeneityRobustDid,
) -> PanelDidQuery {
    PanelDidQuery::heterogeneity_robust(OUTCOME, units.clone(), units, periods, cohorts, spec)
}

fn rollout_interval(
    spec: HeterogeneityRobustDid,
    scheme: DidResamplingScheme,
    seed_base: u64,
    rep: u64,
) -> Option<Scored> {
    let seed = grid_seed(seed_base + rep);
    let mut rng = CausalRng::from_seed(seed);
    let (data, units, periods, cohorts) = staggered(3 * grid_n(60), &mut rng);
    let query = rollout_query(units, periods, cohorts, spec).with_resampling(scheme, RESAMPLES);
    let study = Study::tabular(data).query(query).build().ok()?;
    let result = study.run(&ExecutionContext::for_tests(seed)).ok()?;
    let rollout = result.quasi_rollout()?;
    let interval = rollout.overall()?.interval?;
    resampled(&study, &result, interval, ROLLOUT_TRUTH, Some(rollout.replicates_ok))
}

/// Callaway-Sant'Anna group-time contrasts against not-yet-treated units, simple aggregate,
/// multiplier refit.
#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn rollout_callaway_santanna_not_yet_treated_multiplier_interval() {
    measure(
        "rollout_callaway_santanna_not_yet_treated_multiplier_interval",
        "staggered_three_cohort_heterogeneous_effects_known_simple_aggregate",
        RESAMPLED,
        |rep| {
            rollout_interval(
                rollout_spec(
                    RolloutDidEstimator::CallawaySantAnna,
                    DidComparisonGroup::NotYetTreated,
                ),
                DidResamplingScheme::ClusterMultiplier,
                810_000,
                rep,
            )
        },
    );
}

/// Callaway-Sant'Anna group-time contrasts against never-treated units, simple aggregate,
/// whole-cluster bootstrap refit.
#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn rollout_callaway_santanna_never_treated_bootstrap_interval() {
    measure(
        "rollout_callaway_santanna_never_treated_bootstrap_interval",
        "staggered_three_cohort_heterogeneous_effects_known_simple_aggregate",
        RESAMPLED,
        |rep| {
            rollout_interval(
                rollout_spec(
                    RolloutDidEstimator::CallawaySantAnna,
                    DidComparisonGroup::NeverTreated,
                ),
                DidResamplingScheme::ClusterBootstrap,
                820_000,
                rep,
            )
        },
    );
}

/// Sun-Abraham interaction-weighted contrasts against never-treated units, simple aggregate,
/// multiplier refit.
#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn rollout_sun_abraham_multiplier_interval() {
    measure(
        "rollout_sun_abraham_multiplier_interval",
        "staggered_three_cohort_heterogeneous_effects_known_simple_aggregate",
        RESAMPLED,
        |rep| {
            rollout_interval(
                rollout_spec(RolloutDidEstimator::SunAbraham, DidComparisonGroup::NeverTreated),
                DidResamplingScheme::ClusterMultiplier,
                830_000,
                rep,
            )
        },
    );
}

/// Untreated two-way fixed-effects imputation, simple aggregate, multiplier refit.
#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn rollout_imputation_multiplier_interval() {
    measure(
        "rollout_imputation_multiplier_interval",
        "staggered_three_cohort_heterogeneous_effects_known_simple_aggregate",
        RESAMPLED,
        |rep| {
            rollout_interval(
                rollout_spec(RolloutDidEstimator::Imputation, DidComparisonGroup::NotYetTreated),
                DidResamplingScheme::ClusterMultiplier,
                840_000,
                rep,
            )
        },
    );
}

// ------------------------------------------------------ covariate-adjusted two-period

fn adjusted_interval(nuisance: DidNuisanceMethod, seed_base: u64, rep: u64) -> Option<Scored> {
    let seed = grid_seed(seed_base + rep);
    let mut rng = CausalRng::from_seed(seed);
    let n = grid_n(400);
    let (mut y, mut x, mut treated, mut post, mut units) = (vec![], vec![], vec![], vec![], vec![]);
    for i in 0..n {
        // Uniform on (-sqrt 3, sqrt 3): unit variance and a propensity bounded away from one.
        let covariate = 3.0_f64.sqrt() * (2.0 * rng.next_f64() - 1.0);
        let d = rng.next_f64() < sigmoid(covariate);
        let alpha = standard_normal(&mut rng) + covariate;
        let label: Arc<str> = Arc::from(format!("s{i:05}"));
        for after in [false, true] {
            let trend = if after { 1.0 + covariate } else { 0.0 };
            let effect = if d && after { ADJUSTED_TRUTH } else { 0.0 };
            y.push(alpha + trend + effect + 0.5 * standard_normal(&mut rng));
            x.push(covariate);
            treated.push(d);
            post.push(after);
            units.push(Arc::clone(&label));
        }
    }
    let data = TabularData::from_f64_columns([("y", y.as_slice()), ("x", x.as_slice())]).ok()?;
    let query = PanelDidQuery::covariate_adjusted_two_period(
        OUTCOME,
        treated,
        post,
        units.clone(),
        units,
        vec![VariableId::from_raw(1)],
        nuisance,
    )
    .with_resampling(DidResamplingScheme::ClusterMultiplier, RESAMPLES);
    let study = Study::tabular(data).query(query).build().ok()?;
    let result = study.run(&ExecutionContext::for_tests(seed)).ok()?;
    let rollout = result.quasi_rollout()?;
    let interval = rollout.overall()?.interval?;
    resampled(&study, &result, interval, ADJUSTED_TRUTH, Some(rollout.replicates_ok))
}

/// Outcome regression of the untreated change on the covariate, multiplier refit of the
/// nuisance fit and the 2x2.
#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn covariate_adjusted_outcome_regression_multiplier_interval() {
    measure(
        "covariate_adjusted_outcome_regression_multiplier_interval",
        "two_period_conditional_trends_logistic_assignment_known_effect_on_treated",
        RESAMPLED,
        |rep| adjusted_interval(DidNuisanceMethod::OutcomeRegression, 850_000, rep),
    );
}

/// Inverse-probability weighting of the untreated change by the logistic cohort propensity,
/// multiplier refit.
#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn covariate_adjusted_inverse_probability_multiplier_interval() {
    measure(
        "covariate_adjusted_inverse_probability_multiplier_interval",
        "two_period_conditional_trends_logistic_assignment_known_effect_on_treated",
        RESAMPLED,
        |rep| adjusted_interval(DidNuisanceMethod::InverseProbabilityWeighting, 860_000, rep),
    );
}

/// The doubly robust combination of both nuisance fits, multiplier refit.
#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn covariate_adjusted_doubly_robust_multiplier_interval() {
    measure(
        "covariate_adjusted_doubly_robust_multiplier_interval",
        "two_period_conditional_trends_logistic_assignment_known_effect_on_treated",
        RESAMPLED,
        |rep| adjusted_interval(DidNuisanceMethod::DoublyRobust, 870_000, rep),
    );
}

// ----------------------------------------------------------------- augmented panel

/// Supplied nuisance predictions equal to the true propensity `0.25 + 0.5 x` and the true
/// untreated change `x`; the whole-cluster bootstrap holds them fixed.
fn augmented_interval(rep: u64) -> Option<Scored> {
    let seed = grid_seed(880_000 + rep);
    let mut rng = CausalRng::from_seed(seed);
    let n = grid_n(400);
    let (mut pre, mut post, mut p, mut m, mut treated, mut ids) =
        (vec![], vec![], vec![], vec![], vec![], vec![]);
    for i in 0..n {
        let x = rng.next_f64();
        let propensity = 0.25 + 0.5 * x;
        let d = rng.next_f64() < propensity;
        let base = standard_normal(&mut rng);
        pre.push(base);
        post.push(
            base + x + if d { AUGMENTED_TRUTH } else { 0.0 } + 0.2 * standard_normal(&mut rng),
        );
        p.push(propensity);
        m.push(x);
        treated.push(d);
        ids.push(Arc::<str>::from(format!("s{i:05}")));
    }
    let data = TabularData::from_f64_columns([
        ("pre", pre.as_slice()),
        ("post", post.as_slice()),
        ("p", p.as_slice()),
        ("m0", m.as_slice()),
    ])
    .ok()?;
    let query = PanelDidQuery::augmented_panel(
        VariableId::from_raw(1),
        VariableId::from_raw(0),
        VariableId::from_raw(2),
        VariableId::from_raw(3),
        treated,
        ids.clone(),
        ids,
        true,
    )
    .with_resampling(DidResamplingScheme::ClusterBootstrap, RESAMPLES);
    let study = Study::tabular(data).query(query).build().ok()?;
    let result = study.run(&ExecutionContext::for_tests(seed)).ok()?;
    let interval = result.panel_did()?.interval_95?;
    resampled(&study, &result, interval, AUGMENTED_TRUTH, None)
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn augmented_panel_true_nuisance_bootstrap_interval() {
    measure(
        "augmented_panel_true_nuisance_bootstrap_interval",
        "two_period_panel_true_propensity_and_untreated_change_supplied_known_effect_on_treated",
        RESAMPLED,
        augmented_interval,
    );
}

// ---------------------------------------------------------------- Bayesian routes

/// Bayesian bootstrap credible interval of the rollout's simple aggregate: flat-Dirichlet
/// whole-unit weights, the whole group-time procedure refit per draw.
fn bayesian_rollout_interval(rep: u64) -> Option<Scored> {
    let seed = grid_seed(890_000 + rep);
    let mut rng = CausalRng::from_seed(seed);
    let (data, units, periods, cohorts) = staggered(3 * grid_n(60), &mut rng);
    let query = rollout_query(
        units,
        periods,
        cohorts,
        rollout_spec(RolloutDidEstimator::CallawaySantAnna, DidComparisonGroup::NotYetTreated),
    );
    let study = Study::tabular(data)
        .query(query)
        .inference(InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(DRAWS)))
        .build()
        .ok()?;
    let result = study.run(&ExecutionContext::for_tests(seed)).ok()?;
    let [lower, upper] = result.quasi_rollout()?.overall()?.interval?;
    posterior(&study, &result, (lower, upper), ROLLOUT_TRUTH)
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn bayesian_rollout_bootstrap_credible_interval() {
    measure(
        "bayesian_rollout_bootstrap_credible_interval",
        "staggered_three_cohort_heterogeneous_effects_known_simple_aggregate",
        POSTERIOR,
        bayesian_rollout_interval,
    );
}

/// Normal-Inverse-Gamma posterior of the difference in mean changes of independent units:
/// the equal-tailed 95% interval of the effect column.
fn bayesian_two_period_interval(rep: u64) -> Option<Scored> {
    let seed = grid_seed(900_000 + rep);
    let mut rng = CausalRng::from_seed(seed);
    let per_arm = grid_n(30);
    let (mut subjects, mut treated, mut post, mut outcome) = (vec![], vec![], vec![], vec![]);
    for unit in 0..2 * per_arm {
        let active = unit >= per_arm;
        let before = 1.0 + standard_normal(&mut rng);
        let change =
            0.7 + if active { TWO_PERIOD_TRUTH } else { 0.0 } + 0.5 * standard_normal(&mut rng);
        for after in [false, true] {
            subjects.push(Arc::<str>::from(format!("s{unit}")));
            treated.push(active);
            post.push(after);
            outcome.push(if after { before + change } else { before });
        }
    }
    let data = TabularData::from_f64_columns([("y", outcome.as_slice())]).ok()?;
    let clusters = subjects.clone();
    let query = PanelDidQuery::new(OUTCOME, treated, post, subjects, clusters);
    let study = Study::tabular(data)
        .query(query)
        .inference(InferenceMode::Bayesian(
            BayesianConfig::conjugate().prior_scale(100.0).n_draws(DRAWS),
        ))
        .build()
        .ok()?;
    let result = study.run(&ExecutionContext::for_tests(seed)).ok()?;
    let posterior_fit = result.posterior.as_ref()?;
    let column = posterior_fit.effect_column()?;
    let interval = (posterior_fit.summaries.q025[column], posterior_fit.summaries.q975[column]);
    posterior(&study, &result, interval, TWO_PERIOD_TRUTH)
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn bayesian_two_period_posterior_interval() {
    measure(
        "bayesian_two_period_posterior_interval",
        "two_period_independent_units_homoskedastic_normal_changes_known_difference",
        POSTERIOR,
        bayesian_two_period_interval,
    );
}
