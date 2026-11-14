//! Coverage of the further statements of the difference-in-differences routes: the rollout
//! under the dynamic, group and calendar aggregations (summary, each component and the
//! simultaneous band, a band scored as one joint event), the group-time and pre-trend bands,
//! the rollout with unit-periods missing completely at random, and the augmented panel with
//! nuisance predictions fitted on an independent sample.
//!
//! Builders mirror the family's first harness file, parametrized where a statement needs it.
//! Records are keyed on the design axes (structure `design`, validation `not_applicable`). A
//! replicate that publishes nothing to score is a skip (a miss).
//!
//! Ignored tests run via `scripts/measure_calibration.sh`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

mod common;

use antecedent::{PanelDidQuery, Study, StudyResult};
use antecedent_core::{
    CausalRng, DidAggregation, DidComparisonGroup, DidResamplingScheme, ExecutionContext,
    HeterogeneityRobustDid, RolloutDidEstimator, VariableId,
};
use antecedent_data::TabularData;
use antecedent_estimate::rollout_resampling::{DesignEffect, DesignEffectKind};
use antecedent_kernels::standard_normal;
use common::calibration::{
    Construction, CoverageTally, REPORTED_LEVEL, RecordKey, ScopeFacts, grid_n, grid_seed,
    map_replicates, n_sim,
};
use common::calibration_bind::constructions;
use std::sync::Arc;

/// Interval method of the resampled routes' pointwise interval.
const RESAMPLED: &str = "bootstrap_se";

/// Interval method of a resampled sup-t simultaneous band.
const BAND: &str = "simultaneous_band";

/// Whole-cluster resamples per replicate of the frequentist routes.
const RESAMPLES: u32 = 199;

/// Query name of every record.
const QUERY: &str = "PanelDid";

/// A design route runs no validation suite.
const NO_VALIDATION: &str = "not_applicable";

/// Simple aggregate of the rollout design.
const ROLLOUT_TRUTH: f64 = 3.5;

/// Effect on the treated of the augmented panel design.
const AUGMENTED_TRUTH: f64 = 1.5;

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

// ----------------------------------------------------------------- augmented panel

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
