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
//! * rollout profiles: the same rollout under the dynamic, group and calendar aggregations.
//!   Each scores its scalar summary, every component's pointwise interval and the block's
//!   simultaneous band; the group-time and pre-trend bands are scored under the simple
//!   aggregate. A band is scored as one joint event, every member inside its band, recorded
//!   as the unit interval against `0.5` (covered) or `2` (missed).
//! * unbalanced rollout: every unit-period of the rollout is absent with probability `0.15`,
//!   independently of everything, so the cell effects and the simple aggregate are unchanged.
//! * augmented panel with estimated predictions: both nuisance functions are fitted by least
//!   squares on an independent sample of the same size and supplied as predictions.
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
use antecedent_estimate::rollout_resampling::{DesignEffect, DesignEffectKind};
use antecedent_kernels::{sigmoid, standard_normal};
use common::calibration::{
    Construction, CoverageTally, REPORTED_LEVEL, RecordKey, ScopeFacts, grid_n, grid_seed,
    map_replicates, n_sim,
};
use common::calibration_bind::constructions;

/// Interval method of the resampled routes' pointwise interval.
const RESAMPLED: &str = "bootstrap_se";
/// Interval method of a resampled sup-t simultaneous band.
const BAND: &str = "simultaneous_band";
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

// ------------------------------------------------- rollout aggregations and bands

/// One scored statement of a rollout profile.
#[derive(Clone, Copy)]
enum Target {
    /// Pointwise interval of the effect of this kind at this key (event time, cohort or
    /// period; unused for the scalar summary).
    Effect(&'static str, DesignEffectKind, i64),
    /// Simultaneous band over every effect of these kinds, at least this many of them.
    Band(&'static str, &'static [DesignEffectKind], usize),
}

impl Target {
    const fn label(self) -> &'static str {
        match self {
            Self::Effect(label, _, _) | Self::Band(label, _, _) => label,
        }
    }

    const fn interval(self) -> &'static str {
        match self {
            Self::Effect(..) => RESAMPLED,
            Self::Band(..) => BAND,
        }
    }
}

/// Known value of one reported effect of the rollout design. The cell effect is
/// `3 * (period - cohort)` and the population cohort shares are equal, so event time `e` is
/// `3e`, cohort 2 averages `0, 3, 6, 9` to `4.5` and cohort 4 averages `0, 3` to `1.5`, and
/// calendar periods 2 to 5 are `0`, `3`, `(6 + 0) / 2` and `(9 + 3) / 2`. Pre-adoption
/// contrasts are zero under the exact parallel trends.
fn rollout_effect_truth(effect: &DesignEffect, overall: f64) -> Option<f64> {
    match effect.kind {
        DesignEffectKind::Overall => Some(overall),
        DesignEffectKind::EventTime => {
            effect.event_time.filter(|event| *event >= 0).map(|event| 3.0 * event as f64)
        }
        DesignEffectKind::Group => match effect.group? {
            2 => Some(4.5),
            4 => Some(1.5),
            _ => None,
        },
        DesignEffectKind::Calendar => match effect.period? {
            2 => Some(0.0),
            3 | 4 => Some(3.0),
            5 => Some(6.0),
            _ => None,
        },
        DesignEffectKind::GroupTime => {
            let event = effect.period? - effect.group?;
            (event >= 0).then(|| 3.0 * event as f64)
        }
        DesignEffectKind::PreTrend => Some(0.0),
        _ => None,
    }
}

/// Whether `effect` is the effect of `kind` at `key`.
fn at_key(effect: &DesignEffect, kind: DesignEffectKind, key: i64) -> bool {
    effect.kind == kind
        && match kind {
            DesignEffectKind::Overall => true,
            DesignEffectKind::Group => effect.group == Some(key),
            DesignEffectKind::Calendar => effect.period == Some(key),
            _ => effect.event_time == Some(key),
        }
}

/// One replicate of the rollout under `spec` (multiplier refit), scored on every target. A
/// target whose interval or band is not published is `None`.
fn rollout_targets(
    spec: HeterogeneityRobustDid,
    seed_base: u64,
    rep: u64,
    overall: f64,
    targets: &[Target],
) -> Option<Vec<Option<Scored>>> {
    let seed = grid_seed(seed_base + rep);
    let mut rng = CausalRng::from_seed(seed);
    let (data, units, periods, cohorts) = staggered(3 * grid_n(60), &mut rng);
    let query = rollout_query(units, periods, cohorts, spec)
        .with_resampling(DidResamplingScheme::ClusterMultiplier, RESAMPLES);
    let study = Study::tabular(data).query(query).build().ok()?;
    let result = study.run(&ExecutionContext::for_tests(seed)).ok()?;
    let rollout = result.quasi_rollout()?;
    let base = resampled(&study, &result, [0.0, 1.0], 0.5, Some(rollout.replicates_ok))?;
    let scored = targets
        .iter()
        .map(|target| match *target {
            Target::Effect(_, kind, key) => {
                let effect = rollout.effects.iter().find(|effect| at_key(effect, kind, key))?;
                let [lower, upper] = effect.interval?;
                Some(Scored {
                    interval: (lower, upper),
                    truth: rollout_effect_truth(effect, overall)?,
                    construction: base.construction.clone(),
                    scope: base.scope,
                    validation: NO_VALIDATION,
                })
            }
            Target::Band(_, kinds, at_least) => {
                let members: Vec<&DesignEffect> =
                    rollout.effects.iter().filter(|effect| kinds.contains(&effect.kind)).collect();
                if members.len() < at_least {
                    return None;
                }
                // Joint event: every member's known value inside its band.
                let mut covered = true;
                for effect in members {
                    let [lower, upper] = effect.band?;
                    let truth = rollout_effect_truth(effect, overall)?;
                    covered &= lower <= truth && truth <= upper;
                }
                let mut construction = base.construction.clone();
                construction.interval_method = BAND.into();
                construction.se_kind = "monte_carlo".into();
                Some(Scored {
                    interval: (0.0, 1.0),
                    truth: if covered { 0.5 } else { 2.0 },
                    construction,
                    scope: base.scope,
                    validation: NO_VALIDATION,
                })
            }
        })
        .collect();
    Some(scored)
}

/// Measure every target of one rollout design on the same replicates, one labelled record
/// per target. A replicate that does not publish a target is a skip of that target.
fn measure_targets(
    test: &'static str,
    spec: HeterogeneityRobustDid,
    seed_base: u64,
    overall: f64,
    targets: &[Target],
) {
    let dgp = "staggered_three_cohort_heterogeneous_effects_known_cell_effects";
    let mut tallies: Vec<CoverageTally> = targets
        .iter()
        .map(|target| {
            let key = RecordKey { test, dgp, interval: target.interval() };
            CoverageTally::for_record(key, REPORTED_LEVEL).labelled(target.label())
        })
        .collect();
    let rows = map_replicates(n_sim(), |rep| {
        rollout_targets(spec.clone(), seed_base, rep, overall, targets)
    });
    for row in rows {
        let row = row.unwrap_or_default();
        for (index, tally) in tallies.iter_mut().enumerate() {
            match row.get(index).and_then(Option::as_ref) {
                Some(scored) => {
                    tally.bind_validation(&scored.construction, scored.scope, scored.validation);
                    tally.record(Some(scored.interval), scored.truth);
                }
                None => tally.skip(),
            }
        }
    }
    for tally in &tallies {
        tally.assert();
    }
}

fn aggregated_spec(aggregation: DidAggregation) -> HeterogeneityRobustDid {
    HeterogeneityRobustDid {
        aggregation,
        ..rollout_spec(RolloutDidEstimator::CallawaySantAnna, DidComparisonGroup::NotYetTreated)
    }
}

/// Dynamic aggregation: event times 0 to 3 are `0, 3, 6, 9` and the summary is their mean,
/// `4.5`. The band covers the summary and the four event times jointly.
#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn rollout_dynamic_aggregation_intervals_and_band() {
    measure_targets(
        "rollout_dynamic_aggregation_intervals_and_band",
        aggregated_spec(DidAggregation::Dynamic),
        910_000,
        4.5,
        &[
            Target::Effect("summary", DesignEffectKind::Overall, 0),
            Target::Effect("event_time_0", DesignEffectKind::EventTime, 0),
            Target::Effect("event_time_1", DesignEffectKind::EventTime, 1),
            Target::Effect("event_time_2", DesignEffectKind::EventTime, 2),
            Target::Effect("event_time_3", DesignEffectKind::EventTime, 3),
            Target::Band("band", &[DesignEffectKind::Overall, DesignEffectKind::EventTime], 5),
        ],
    );
}

/// Group aggregation: cohort 2 is `4.5`, cohort 4 is `1.5`, and the summary weights them by
/// the equal cohort shares, `3`. The band covers the three jointly.
#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn rollout_group_aggregation_intervals_and_band() {
    measure_targets(
        "rollout_group_aggregation_intervals_and_band",
        aggregated_spec(DidAggregation::Group),
        920_000,
        3.0,
        &[
            Target::Effect("summary", DesignEffectKind::Overall, 0),
            Target::Effect("cohort_2", DesignEffectKind::Group, 2),
            Target::Effect("cohort_4", DesignEffectKind::Group, 4),
            Target::Band("band", &[DesignEffectKind::Overall, DesignEffectKind::Group], 3),
        ],
    );
}

/// Calendar aggregation: periods 2 to 5 are `0, 3, 3, 6` and the summary is their mean, `3`.
/// The band covers the summary and the four periods jointly.
#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn rollout_calendar_aggregation_intervals_and_band() {
    measure_targets(
        "rollout_calendar_aggregation_intervals_and_band",
        aggregated_spec(DidAggregation::Calendar),
        930_000,
        3.0,
        &[
            Target::Effect("summary", DesignEffectKind::Overall, 0),
            Target::Effect("period_2", DesignEffectKind::Calendar, 2),
            Target::Effect("period_3", DesignEffectKind::Calendar, 3),
            Target::Effect("period_4", DesignEffectKind::Calendar, 4),
            Target::Effect("period_5", DesignEffectKind::Calendar, 5),
            Target::Band("band", &[DesignEffectKind::Overall, DesignEffectKind::Calendar], 5),
        ],
    );
}

/// The bands of the simple aggregate's other blocks: the six post-adoption group-time cells
/// (`3 * (period - cohort)`) and the pre-adoption contrasts (zero).
#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn rollout_group_time_and_pre_trend_simultaneous_bands() {
    measure_targets(
        "rollout_group_time_and_pre_trend_simultaneous_bands",
        aggregated_spec(DidAggregation::Simple),
        940_000,
        ROLLOUT_TRUTH,
        &[
            Target::Band("group_time", &[DesignEffectKind::GroupTime], 6),
            Target::Band("pre_trend", &[DesignEffectKind::PreTrend], 1),
        ],
    );
}

// -------------------------------------------------------------- unbalanced rollout

/// Chance that a unit-period of the unbalanced rollout is absent.
const ABSENT: f64 = 0.15;

/// The rollout of [`staggered`] with every unit-period absent with probability [`ABSENT`],
/// independently of the unit, its cohort and its outcomes.
fn staggered_unbalanced(
    units: usize,
    rng: &mut CausalRng,
) -> (TabularData, Vec<Arc<str>>, Vec<i64>, Vec<i64>) {
    const COHORTS: [i64; 3] = [0, 2, 4];
    let count = units;
    let (mut y, mut units, mut periods, mut cohorts) = (vec![], vec![], vec![], vec![]);
    for i in 0..count {
        let cohort = COHORTS[((rng.next_f64() * 3.0) as usize).min(2)];
        let label: Arc<str> = Arc::from(format!("c{cohort}_{i:05}"));
        let alpha = standard_normal(rng);
        for period in 1..=5_i64 {
            let effect =
                if cohort > 0 && period >= cohort { 3.0 * (period - cohort) as f64 } else { 0.0 };
            let value = alpha + 0.5 * period as f64 + effect + standard_normal(rng);
            if rng.next_f64() < ABSENT {
                continue;
            }
            y.push(value);
            units.push(Arc::clone(&label));
            periods.push(period);
            cohorts.push(cohort);
        }
    }
    let data = TabularData::from_f64_columns([("y", y.as_slice())]).expect("rollout outcome");
    (data, units, periods, cohorts)
}

/// The simple aggregate's interval on the unbalanced rollout. Absence is unrelated to the
/// outcomes and the cell effects are constants, so the aggregate is still `3.5`.
fn unbalanced_interval(
    estimator: RolloutDidEstimator,
    comparison: DidComparisonGroup,
    seed_base: u64,
    rep: u64,
) -> Option<Scored> {
    let seed = grid_seed(seed_base + rep);
    let mut rng = CausalRng::from_seed(seed);
    let (data, units, periods, cohorts) = staggered_unbalanced(3 * grid_n(60), &mut rng);
    let query = rollout_query(units, periods, cohorts, rollout_spec(estimator, comparison))
        .with_resampling(DidResamplingScheme::ClusterMultiplier, RESAMPLES);
    let study = Study::tabular(data).query(query).build().ok()?;
    let result = study.run(&ExecutionContext::for_tests(seed)).ok()?;
    let rollout = result.quasi_rollout()?;
    // The design is only measured when the panel was read as unbalanced.
    if rollout.diagnostic("balanced_panel") != Some(0.0) {
        return None;
    }
    let interval = rollout.overall()?.interval?;
    resampled(&study, &result, interval, ROLLOUT_TRUTH, Some(rollout.replicates_ok))
}

/// Callaway-Sant'Anna against not-yet-treated units: each 2x2 uses the units observed at
/// both of its periods.
#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn rollout_unbalanced_callaway_santanna_multiplier_interval() {
    measure(
        "rollout_unbalanced_callaway_santanna_multiplier_interval",
        "staggered_three_cohort_unit_periods_absent_at_random_known_simple_aggregate",
        RESAMPLED,
        |rep| {
            unbalanced_interval(
                RolloutDidEstimator::CallawaySantAnna,
                DidComparisonGroup::NotYetTreated,
                950_000,
                rep,
            )
        },
    );
}

/// Sun-Abraham on the observed unit-periods: the saturated regression of the unbalanced panel.
#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn rollout_unbalanced_sun_abraham_multiplier_interval() {
    measure(
        "rollout_unbalanced_sun_abraham_multiplier_interval",
        "staggered_three_cohort_unit_periods_absent_at_random_known_simple_aggregate",
        RESAMPLED,
        |rep| {
            unbalanced_interval(
                RolloutDidEstimator::SunAbraham,
                DidComparisonGroup::NeverTreated,
                960_000,
                rep,
            )
        },
    );
}

/// Imputation from every observed untreated unit-period.
#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn rollout_unbalanced_imputation_multiplier_interval() {
    measure(
        "rollout_unbalanced_imputation_multiplier_interval",
        "staggered_three_cohort_unit_periods_absent_at_random_known_simple_aggregate",
        RESAMPLED,
        |rep| {
            unbalanced_interval(
                RolloutDidEstimator::Imputation,
                DidComparisonGroup::NotYetTreated,
                970_000,
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

/// One sample of the augmented panel design: covariate, treatment, pre and post outcomes.
fn augmented_sample(n: usize, rng: &mut CausalRng) -> (Vec<f64>, Vec<bool>, Vec<f64>, Vec<f64>) {
    let (mut x, mut treated, mut pre, mut post) = (vec![], vec![], vec![], vec![]);
    for _ in 0..n {
        let covariate = rng.next_f64();
        let d = rng.next_f64() < 0.25 + 0.5 * covariate;
        let base = standard_normal(rng);
        pre.push(base);
        post.push(
            base + covariate + if d { AUGMENTED_TRUTH } else { 0.0 } + 0.2 * standard_normal(rng),
        );
        x.push(covariate);
        treated.push(d);
    }
    (x, treated, pre, post)
}

/// Least-squares intercept and slope of `y` on `x`.
fn least_squares_line(x: &[f64], y: &[f64]) -> (f64, f64) {
    let n = x.len() as f64;
    let (mean_x, mean_y) = (x.iter().sum::<f64>() / n, y.iter().sum::<f64>() / n);
    let cross: f64 = x.iter().zip(y).map(|(a, b)| (a - mean_x) * (b - mean_y)).sum();
    let square: f64 = x.iter().map(|a| (a - mean_x).powi(2)).sum();
    let slope = cross / square;
    (mean_y - slope * mean_x, slope)
}

/// Supplied predictions estimated on an independent sample of the same size: the propensity
/// is the least-squares line of treatment on the covariate (the true propensity is linear)
/// and the untreated change the least-squares line of the comparison units' change. Both
/// models are correct and the effect is constant, so the target stays the effect on the
/// treated, `1.5`; the bootstrap holds the estimated predictions fixed.
fn augmented_estimated_interval(rep: u64) -> Option<Scored> {
    let seed = grid_seed(980_000 + rep);
    let mut rng = CausalRng::from_seed(seed);
    let n = grid_n(400);
    let (x, treated, pre, post) = augmented_sample(n, &mut rng);
    let (fit_x, fit_treated, fit_pre, fit_post) = augmented_sample(n, &mut rng);
    let indicator: Vec<f64> = fit_treated.iter().map(|d| if *d { 1.0 } else { 0.0 }).collect();
    let (p0, p1) = least_squares_line(&fit_x, &indicator);
    let (control_x, control_change): (Vec<f64>, Vec<f64>) =
        (0..n).filter(|i| !fit_treated[*i]).map(|i| (fit_x[i], fit_post[i] - fit_pre[i])).unzip();
    let (m0, m1) = least_squares_line(&control_x, &control_change);
    // The fitted line is kept a probability; the true propensity lies in (0.25, 0.75).
    let p: Vec<f64> = x.iter().map(|x| (p0 + p1 * x).clamp(0.01, 0.99)).collect();
    let m: Vec<f64> = x.iter().map(|x| m0 + m1 * x).collect();
    let ids: Vec<Arc<str>> = (0..n).map(|i| Arc::<str>::from(format!("s{i:05}"))).collect();
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
fn augmented_panel_estimated_nuisance_bootstrap_interval() {
    measure(
        "augmented_panel_estimated_nuisance_bootstrap_interval",
        "two_period_panel_propensity_and_untreated_change_fitted_on_independent_sample_known_effect_on_treated",
        RESAMPLED,
        augmented_estimated_interval,
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
