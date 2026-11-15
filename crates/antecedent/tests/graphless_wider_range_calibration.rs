//! Coverage designs for the parts of the design-route support conditions that are stated wider
//! than the base harnesses measure.
//!
//! Each design repeats a base design (same data-generating family, same truth, same route and
//! settings) with one thing moved to the range the row states and the base grid does not reach.
//! The designs were fixed before any run and are not tuned.
//!
//! * Cluster Bayesian bootstrap. The Student-t reference law is stated for nine to forty-nine
//!   clusters per arm; the base design measures 9, 12 and 24 and the statements harness 25, 37
//!   and 49. This file measures 36, 43 and 49 per arm as one more look at the upper part of the
//!   range, and 50, 75 and 100 per arm, where the route publishes the Dirichlet law instead.
//! * Synthetic control donors. The rows require ten donors and the base designs use exactly
//!   ten. The Bayesian interval, the three predictive-path intervals and the three conformal
//!   sets are repeated with 20 and with 40 donors (the treated unit's path stays the simplex
//!   combination of donors 3, 4 and 5).
//! * Synthetic control post periods. The donor-weight posterior states any post-period count
//!   and was measured with five; it is repeated with one and with three.
//! * Conformal pre-periods that are not a multiple of twenty. The permutation count `n` is the
//!   pre-periods plus one (a period) or plus the post periods (the constant effect), and the
//!   test's size is `floor(0.05 n) / n`, exactly 5% only when twenty divides `n`. The designs
//!   use 47, 63 and 103 pre-periods (`n` = 48, 64 and 104 for a period, 52, 68 and 108 for the
//!   constant effect), where the size is below 5% and the set conservative, so each record is a
//!   one-sided floor (`assert_one_sided`) and over-coverage passes.
//! * Upper ranges of rows stated with a minimum only. Each is one design whose largest point
//!   is four times the largest the base grid measures: the two-period posterior (60 to 240
//!   subjects per group), the augmented panel (800 to 3,200 subjects), the staggered rollout
//!   (360 to 1,440 units), the association slope (400 to 1,600 rows), the binary-instrument
//!   bounds (2,000 to 8,000 rows) and the synthetic control (80 to 320 pre-periods, conformal
//!   159 to 639). The grid of such a design runs from half of that largest point.
//!
//! Every record is keyed on the design axes the way the base harnesses key theirs (structure
//! `design`, validation `not_applicable`). A replicate that publishes no interval, or states no
//! support row where the base design counts only replicates that state one, is a skip, never a
//! free pass.
//!
//! The designs are `#[ignore]`d measurement designs run by `scripts/measure_calibration.sh`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "simulation dimensions are small counts"
)]

mod common;

use std::sync::Arc;

use antecedent::{
    AssociationFunctional, AssociationQuery, BayesianConfig, CausalQuery, InferenceMode,
    InterferenceSpec, IvBoundsQuery, PanelDidQuery, PreparedStudy, RefuteSuite, Study,
    StudyBuilder, StudyResult,
};
use antecedent_core::{
    AssignmentDesign, CausalRng, DeclaredSamplingDesign, DidAggregation, DidComparisonGroup,
    DidResamplingScheme, ExecutionContext, ExposureLevel, ExposureMapping, HeterogeneityRobustDid,
    InterferenceFunctional, InterferenceQuery, RolloutDidEstimator, SyntheticControlQuery,
    VariableId,
};
use antecedent_data::{NetworkData, NetworkEdge, TabularData, TimeSeriesData};
use antecedent_estimate::rollout_resampling::DesignEffectKind;
use antecedent_graph::Dag;
use antecedent_kernels::standard_normal;
use common::calibration::{
    Construction, CoverageSide, CoverageTally, REPORTED_LEVEL, RecordKey, SampleGrid, ScopeFacts,
    Z95, ar1_noise, gaussian, grid_n, grid_seed, map_replicates, n_sim, normal_interval,
    stream_seed, uniform, unit_uniform,
};
use common::calibration_bind::{bind_reported, constructions};

/// One replicate's construction, scope facts and validation mode as the bind helpers take them.
type Reported = (Construction, ScopeFacts, &'static str);

/// A design route runs no validation suite.
const NO_VALIDATION: &str = "not_applicable";

/// Sample sizes at one half, three quarters and all of `base`: the grid of a design whose
/// largest point is four times the largest the base grid measures, so every point is beyond
/// the base grid's.
const UPPER: SampleGrid = SampleGrid { name: "upper_range", factors: [(1, 2), (3, 4), (1, 1)] };

fn v(raw: u32) -> VariableId {
    VariableId::from_raw(raw)
}

// ============================================================ cluster Bayesian bootstrap

const POSTERIOR_QUANTILE: &str = "posterior_quantile";
const CLUSTER_DRAWS: usize = 2_000;
const CLUSTER_TOTAL_EFFECT: f64 = 3.0;

/// 36, 43 and 49 clusters per arm: the upper part of the range in which the route publishes
/// its Student-t reference law.
const STUDENT_ARMS: SampleGrid =
    SampleGrid { name: "cluster_arms_36_to_49", factors: [(36, 43), (1, 1), (49, 43)] };

/// 50, 75 and 100 clusters per arm: the route publishes the Dirichlet law from fifty clusters
/// in both arms.
const DIRICHLET_ARMS: SampleGrid =
    SampleGrid { name: "cluster_arms_50_to_100", factors: [(2, 3), (1, 1), (4, 3)] };

/// `treated` units chosen uniformly without replacement.
fn complete_assignment(n: usize, treated: usize, seed: u64) -> Vec<bool> {
    let keys: Vec<f64> = (0..n).map(|i| unit_uniform(stream_seed(seed, i as u64))).collect();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| keys[a].total_cmp(&keys[b]));
    let mut assigned = vec![false; n];
    for &unit in &order[..treated] {
        assigned[unit] = true;
    }
    assigned
}

/// One scored replicate: the published interval, the known truth and the runtime's construction.
struct Replicate {
    interval: (f64, f64),
    truth: f64,
    reported: Vec<Reported>,
}

/// Score every replicate against its truth under one posterior-quantile record; a replicate
/// that publishes nothing is a skip.
fn measure_replicates(test: &'static str, dgp: &'static str, runs: &[Option<Replicate>]) {
    let mut tally = CoverageTally::for_record(
        RecordKey { test, dgp, interval: POSTERIOR_QUANTILE },
        REPORTED_LEVEL,
    );
    for run in runs {
        let Some(replicate) = run else {
            tally.skip();
            continue;
        };
        bind_reported(&mut tally, &replicate.reported);
        tally.record(Some(replicate.interval), replicate.truth);
    }
    tally.assert();
}

/// Two-unit clusters, complete cluster randomization of `clusters` clusters (half treated), a
/// shared cluster effect and a constant total effect of 3, under the Bayesian bootstrap.
fn cluster_replicate(clusters: usize, seed: u64) -> Option<Replicate> {
    let treated = complete_assignment(clusters, clusters / 2, stream_seed(seed, 0xC1));
    let mut noise = gaussian(seed);
    let (mut outcome, mut assignment) = (Vec::new(), Vec::new());
    for &arm in &treated {
        let shared = noise();
        for _ in 0..2 {
            outcome.push(shared + 0.5 * noise() + CLUSTER_TOTAL_EFFECT * f64::from(u8::from(arm)));
            assignment.push(arm);
        }
    }
    let units = 2 * clusters;
    let edges: Vec<NetworkEdge> = (0..clusters as u64)
        .flat_map(|c| {
            [NetworkEdge::new(2 * c, 2 * c + 1, 1.0), NetworkEdge::new(2 * c + 1, 2 * c, 1.0)]
        })
        .collect();
    let data = TabularData::from_f64_columns([("y", outcome.as_slice())]).ok()?;
    let query = InterferenceQuery::new(
        AssignmentDesign::ClusterRandomization {
            clusters: Arc::from((0..units).map(|u| (u / 2) as u32).collect::<Vec<_>>()),
            treated_clusters: clusters / 2,
        },
        ExposureMapping::NeighborFraction,
        InterferenceFunctional::ExposureContrast {
            outcome: v(0),
            from: ExposureLevel { own: 0.0, neighbors: 0.0 },
            to: ExposureLevel { own: 1.0, neighbors: 1.0 },
        },
    );
    let network = NetworkData::try_new(data.clone(), edges).ok()?;
    let builder: StudyBuilder = Study::tabular(data.clone())
        .graph(Dag::with_variables(1))
        .query(CausalQuery::Interference(query))
        .interference(InterferenceSpec { network, assignment: Arc::from(assignment) })
        .inference(InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(CLUSTER_DRAWS)))
        .refute(RefuteSuite::None);
    let ctx = ExecutionContext::for_tests(seed);
    let prepared = builder.build().ok()?.prepare(&ctx).ok()?;
    let result = prepared.estimate(&data, &ctx).ok()?;
    let contract = prepared.contract_for_result(&result).expect("the execution contract");
    let reported = constructions(&contract, &result);
    let posterior = result.posterior.as_ref()?;
    let column = posterior.effect_column()?;
    let (lower, upper) = (posterior.summaries.q025[column], posterior.summaries.q975[column]);
    (lower.is_finite() && upper.is_finite() && lower <= upper).then_some(Replicate {
        interval: (lower, upper),
        truth: CLUSTER_TOTAL_EFFECT,
        reported,
    })
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn cluster_bayesian_bootstrap_student_reference_law_upper_arms_credible_interval() {
    let clusters = 2 * STUDENT_ARMS.n(43);
    let runs =
        map_replicates(n_sim(), |rep| cluster_replicate(clusters, grid_seed(0x6E01_0000 + rep)));
    measure_replicates(
        "cluster_bayesian_bootstrap_student_reference_law_upper_arms_credible_interval",
        "paired_cluster_randomization_constant_total_effect_36_to_49_clusters_per_arm",
        &runs,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn cluster_bayesian_bootstrap_dirichlet_law_from_fifty_arms_credible_interval() {
    let clusters = 2 * DIRICHLET_ARMS.n(75);
    let runs =
        map_replicates(n_sim(), |rep| cluster_replicate(clusters, grid_seed(0x6E02_0000 + rep)));
    measure_replicates(
        "cluster_bayesian_bootstrap_dirichlet_law_from_fifty_arms_credible_interval",
        "paired_cluster_randomization_constant_total_effect_50_to_100_clusters_per_arm",
        &runs,
    );
}

// ============================================================ synthetic control

/// Query name of every synthetic-control record.
const SC_QUERY: &str = "SyntheticControl";
/// Interval method of the conformal sets.
const CONFORMAL: &str = "conformal";
const NOISE_SD: f64 = 0.5;
const SC_DRAWS: usize = 2_000;
const SC_DGP: &str = "convex_donor_panel_iid_noise";
/// Pre-intervention periods of the base designs at their base grid point.
const BASE_PRE_PERIODS: usize = 40;
/// Donors that form the treated unit's untreated path, with their simplex weights.
const WEIGHTS: [(usize, f64); 3] = [(3, 0.5), (4, 0.3), (5, 0.2)];
/// The interval scored when a run publishes no set: it covers nothing, so it is a miss.
const NO_SET: [f64; 2] = [f64::NAN; 2];

/// Donors and post periods of one panel.
#[derive(Clone, Copy)]
struct Panel {
    donors: usize,
    post: usize,
}

/// The base designs' panel: ten donors, five post periods.
const BASE_PANEL: Panel = Panel { donors: 10, post: 5 };

impl Panel {
    /// The known effect on the treated unit at the `k`-th post period.
    fn effect_at(k: usize) -> f64 {
        2.0 + 0.5 * k as f64
    }

    /// The known average effect over the post periods.
    fn average_effect(self) -> f64 {
        (0..self.post).map(Self::effect_at).sum::<f64>() / self.post as f64
    }
}

/// A simulated panel: outcome column, unit labels and period labels, with the treated unit
/// last in each period. `constant` adds the average effect to every post period instead of the
/// rising one.
fn simulate(
    panel: Panel,
    pre: usize,
    seed: u64,
    constant: bool,
) -> (TabularData, Vec<Arc<str>>, Vec<i64>) {
    let mut z = gaussian(seed);
    let (mut y, mut units, mut periods) = (Vec::new(), Vec::new(), Vec::new());
    for t in 1..=pre + panel.post {
        let donors: Vec<f64> = (0..panel.donors).map(|j| j as f64 + z()).collect();
        for (j, value) in donors.iter().enumerate() {
            y.push(*value);
            units.push(Arc::<str>::from(format!("d{j:02}")));
            periods.push(i64::try_from(t).expect("the index fits in i64"));
        }
        let mut treated: f64 =
            WEIGHTS.iter().map(|(j, w)| w * donors[*j]).sum::<f64>() + NOISE_SD * z();
        if t > pre {
            treated +=
                if constant { panel.average_effect() } else { Panel::effect_at(t - pre - 1) };
        }
        y.push(treated);
        units.push(Arc::<str>::from("treated"));
        periods.push(i64::try_from(t).expect("the index fits in i64"));
    }
    let data = TabularData::from_f64_columns([("y", y.as_slice())])
        .expect("the simulated outcome is finite and aligned");
    (data, units, periods)
}

fn sc_query(units: Vec<Arc<str>>, periods: Vec<i64>, pre: usize) -> SyntheticControlQuery {
    SyntheticControlQuery::new(
        VariableId::from_raw(0),
        units,
        periods,
        Arc::<str>::from("treated"),
        i64::try_from(pre).expect("the index fits in i64") + 1,
    )
}

/// One replicate: the scored interval and the record's construction.
struct Published {
    interval: (f64, f64),
    construction: Construction,
    scope: ScopeFacts,
}

/// How a record's coverage is asserted.
#[derive(Clone, Copy)]
enum Assertion {
    /// The two-sided band: the construction's size is exactly nominal.
    TwoSided,
    /// A floor with no ceiling: the construction is conservative at this design.
    Conservative(&'static str),
}

/// Score every replicate's interval against `truth` under its own construction; a refusal or
/// an absent interval is a skip, never a free pass.
fn measure_sc(
    test: &'static str,
    interval: &'static str,
    truth: f64,
    assertion: Assertion,
    replicate: impl Fn(u64) -> Option<Published> + Sync,
) {
    let mut tally =
        CoverageTally::for_record(RecordKey { test, dgp: SC_DGP, interval }, REPORTED_LEVEL);
    for run in map_replicates(n_sim(), replicate) {
        let Some(run) = run else {
            tally.skip();
            continue;
        };
        tally.bind_validation(&run.construction, run.scope, NO_VALIDATION);
        tally.record(Some(run.interval), truth);
    }
    match assertion {
        Assertion::TwoSided => tally.assert(),
        Assertion::Conservative(target) => tally.assert_one_sided(CoverageSide::Upper, target),
    }
}

/// Which Bayesian interval a design scores.
#[derive(Clone, Copy)]
enum Bayesian {
    /// The credible interval of the average effect of the donor-weight posterior.
    Posterior,
    /// The predictive path's interval of the average effect.
    Average,
    /// The predictive path's interval of the first post period.
    First,
    /// The predictive path's interval of the last post period.
    Last,
}

impl Bayesian {
    fn conformal(self) -> bool {
        !matches!(self, Self::Posterior)
    }

    fn truth(self, panel: Panel) -> f64 {
        match self {
            Self::Posterior | Self::Average => panel.average_effect(),
            Self::First => Panel::effect_at(0),
            Self::Last => Panel::effect_at(panel.post - 1),
        }
    }
}

/// Run the Bayesian synthetic control on one panel and read the scored interval; `None` when
/// the route refuses or publishes no interval for it.
fn bayesian_replicate(panel: Panel, pre: usize, scored: Bayesian, seed: u64) -> Option<Published> {
    let (data, units, periods) = simulate(panel, pre, seed, false);
    let mut query = sc_query(units, periods, pre);
    if scored.conformal() {
        query = query.with_conformal_intervals();
    }
    let study = Study::tabular(data)
        .query(query)
        .inference(InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(SC_DRAWS)))
        .build()
        .ok()?;
    let result = study.run(&ExecutionContext::for_tests(seed)).ok()?;
    let reported = constructions(&study.inspect().ok()?, &result);
    let intervention = i64::try_from(pre).expect("the index fits in i64") + 1;
    let interval = if let Bayesian::Posterior = scored {
        let fit = result.posterior.as_ref()?;
        let column = fit.effect_column()?;
        [fit.summaries.q025[column], fit.summaries.q975[column]]
    } else {
        let rollout = result.quasi_rollout()?;
        assert_eq!(rollout.design.as_ref(), "posterior_predictive_synthetic_control");
        let period = |k: usize| {
            let period = intervention + i64::try_from(k).expect("the index fits in i64");
            rollout.effects.iter().find(|effect| {
                effect.kind == DesignEffectKind::Period && effect.period == Some(period)
            })
        };
        let effect = match scored {
            Bayesian::Posterior => unreachable!("the posterior case is handled above"),
            Bayesian::Average => {
                rollout.effects.iter().find(|effect| effect.kind == DesignEffectKind::Overall)?
            }
            Bayesian::First => period(0)?,
            Bayesian::Last => period(panel.post - 1)?,
        };
        effect.interval?
    };
    let (mut construction, scope, _) = reported
        .into_iter()
        .find(|(construction, _, _)| construction.interval_method == POSTERIOR_QUANTILE)?;
    construction.query = SC_QUERY.into();
    construction.structure = "design".into();
    Some(Published { interval: (interval[0], interval[1]), construction, scope })
}

fn measure_bayesian(
    test: &'static str,
    panel: Panel,
    pre: usize,
    scored: Bayesian,
    seed_base: u64,
) {
    measure_sc(test, POSTERIOR_QUANTILE, scored.truth(panel), Assertion::TwoSided, |rep| {
        bayesian_replicate(panel, pre, scored, grid_seed(seed_base + rep))
    });
}

/// Which conformal set a design scores.
#[derive(Clone, Copy)]
enum ConformalSet {
    /// The set of the first post period, on the rising-effect panel.
    First,
    /// The set of the last post period, on the rising-effect panel.
    Last,
    /// The constant-effect set, on the constant-effect panel.
    Constant,
}

impl ConformalSet {
    fn truth(self, panel: Panel) -> f64 {
        match self {
            Self::First => Panel::effect_at(0),
            Self::Last => Panel::effect_at(panel.post - 1),
            Self::Constant => panel.average_effect(),
        }
    }

    /// Pre-periods that make the permutation count `20 * twenties`.
    fn aligned_pre_periods(self, panel: Panel, twenties: usize) -> usize {
        match self {
            Self::First | Self::Last => 20 * twenties - 1,
            Self::Constant => 20 * twenties - panel.post,
        }
    }
}

/// Run the conformal synthetic control on one panel and read the scored set; `None` when the
/// route refuses. A run that publishes no set for the scored quantity is scored as a miss.
fn conformal_replicate(
    panel: Panel,
    pre: usize,
    set: ConformalSet,
    seed: u64,
) -> Option<Published> {
    let (data, units, periods) = simulate(panel, pre, seed, matches!(set, ConformalSet::Constant));
    let intervention = i64::try_from(pre).expect("the index fits in i64") + 1;
    let query = sc_query(units, periods, pre).with_conformal_intervals();
    let study = Study::tabular(data).query(query).build().ok()?;
    let result = study.run(&ExecutionContext::for_tests(seed)).ok()?;
    let rollout = result.quasi_rollout()?;
    let period = |k: usize| {
        let period = intervention + i64::try_from(k).expect("the index fits in i64");
        rollout
            .effects
            .iter()
            .find(|effect| effect.kind == DesignEffectKind::Period && effect.period == Some(period))
    };
    let interval = match set {
        ConformalSet::Constant => rollout.overall()?.interval.unwrap_or(NO_SET),
        ConformalSet::First => period(0)?.interval.unwrap_or(NO_SET),
        ConformalSet::Last => period(panel.post - 1)?.interval.unwrap_or(NO_SET),
    };
    let (mut construction, scope, _) =
        constructions(&study.inspect().ok()?, &result).into_iter().next()?;
    construction.query = SC_QUERY.into();
    construction.structure = "design".into();
    construction.interval_method = CONFORMAL.into();
    construction.se_kind = String::new();
    construction.dependence = "iid".into();
    construction.reported_level = REPORTED_LEVEL;
    Some(Published { interval: (interval[0], interval[1]), construction, scope })
}

fn measure_conformal(
    test: &'static str,
    panel: Panel,
    pre: usize,
    set: ConformalSet,
    assertion: Assertion,
    seed_base: u64,
) {
    measure_sc(test, CONFORMAL, set.truth(panel), assertion, |rep| {
        conformal_replicate(panel, pre, set, grid_seed(seed_base + rep))
    });
}

/// Pre-periods of the base Bayesian designs at the current grid point.
fn base_pre() -> usize {
    grid_n(BASE_PRE_PERIODS)
}

/// Pre-periods of the base conformal designs at the current grid point: the permutation count
/// is a multiple of twenty.
fn base_twenties() -> usize {
    SampleGrid::STANDARD.n(4)
}

// ------------------------------------------------------------------ more donors

/// Twenty donors, then forty: the rows require ten.
const TWENTY_DONORS: Panel = Panel { donors: 20, ..BASE_PANEL };
const FORTY_DONORS: Panel = Panel { donors: 40, ..BASE_PANEL };

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn bayesian_synthetic_control_posterior_interval_twenty_donors() {
    measure_bayesian(
        "bayesian_synthetic_control_posterior_interval_twenty_donors",
        TWENTY_DONORS,
        base_pre(),
        Bayesian::Posterior,
        0x6E10_0000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn bayesian_synthetic_control_predictive_path_average_interval_twenty_donors() {
    measure_bayesian(
        "bayesian_synthetic_control_predictive_path_average_interval_twenty_donors",
        TWENTY_DONORS,
        base_pre(),
        Bayesian::Average,
        0x6E11_0000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn bayesian_synthetic_control_predictive_path_first_period_interval_twenty_donors() {
    measure_bayesian(
        "bayesian_synthetic_control_predictive_path_first_period_interval_twenty_donors",
        TWENTY_DONORS,
        base_pre(),
        Bayesian::First,
        0x6E12_0000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn bayesian_synthetic_control_predictive_path_last_period_interval_twenty_donors() {
    measure_bayesian(
        "bayesian_synthetic_control_predictive_path_last_period_interval_twenty_donors",
        TWENTY_DONORS,
        base_pre(),
        Bayesian::Last,
        0x6E13_0000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn conformal_synthetic_control_first_period_set_twenty_donors() {
    let set = ConformalSet::First;
    measure_conformal(
        "conformal_synthetic_control_first_period_set_twenty_donors",
        TWENTY_DONORS,
        set.aligned_pre_periods(TWENTY_DONORS, base_twenties()),
        set,
        Assertion::TwoSided,
        0x6E14_0000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn conformal_synthetic_control_last_period_set_twenty_donors() {
    let set = ConformalSet::Last;
    measure_conformal(
        "conformal_synthetic_control_last_period_set_twenty_donors",
        TWENTY_DONORS,
        set.aligned_pre_periods(TWENTY_DONORS, base_twenties()),
        set,
        Assertion::TwoSided,
        0x6E15_0000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn conformal_synthetic_control_constant_effect_set_twenty_donors() {
    let set = ConformalSet::Constant;
    measure_conformal(
        "conformal_synthetic_control_constant_effect_set_twenty_donors",
        TWENTY_DONORS,
        set.aligned_pre_periods(TWENTY_DONORS, base_twenties()),
        set,
        Assertion::TwoSided,
        0x6E16_0000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn bayesian_synthetic_control_posterior_interval_forty_donors() {
    measure_bayesian(
        "bayesian_synthetic_control_posterior_interval_forty_donors",
        FORTY_DONORS,
        base_pre(),
        Bayesian::Posterior,
        0x6E17_0000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn bayesian_synthetic_control_predictive_path_average_interval_forty_donors() {
    measure_bayesian(
        "bayesian_synthetic_control_predictive_path_average_interval_forty_donors",
        FORTY_DONORS,
        base_pre(),
        Bayesian::Average,
        0x6E18_0000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn bayesian_synthetic_control_predictive_path_first_period_interval_forty_donors() {
    measure_bayesian(
        "bayesian_synthetic_control_predictive_path_first_period_interval_forty_donors",
        FORTY_DONORS,
        base_pre(),
        Bayesian::First,
        0x6E19_0000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn bayesian_synthetic_control_predictive_path_last_period_interval_forty_donors() {
    measure_bayesian(
        "bayesian_synthetic_control_predictive_path_last_period_interval_forty_donors",
        FORTY_DONORS,
        base_pre(),
        Bayesian::Last,
        0x6E1A_0000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn conformal_synthetic_control_first_period_set_forty_donors() {
    let set = ConformalSet::First;
    measure_conformal(
        "conformal_synthetic_control_first_period_set_forty_donors",
        FORTY_DONORS,
        set.aligned_pre_periods(FORTY_DONORS, base_twenties()),
        set,
        Assertion::TwoSided,
        0x6E1B_0000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn conformal_synthetic_control_last_period_set_forty_donors() {
    let set = ConformalSet::Last;
    measure_conformal(
        "conformal_synthetic_control_last_period_set_forty_donors",
        FORTY_DONORS,
        set.aligned_pre_periods(FORTY_DONORS, base_twenties()),
        set,
        Assertion::TwoSided,
        0x6E1C_0000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn conformal_synthetic_control_constant_effect_set_forty_donors() {
    let set = ConformalSet::Constant;
    measure_conformal(
        "conformal_synthetic_control_constant_effect_set_forty_donors",
        FORTY_DONORS,
        set.aligned_pre_periods(FORTY_DONORS, base_twenties()),
        set,
        Assertion::TwoSided,
        0x6E1D_0000,
    );
}

// ------------------------------------------------------------------ post periods

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn bayesian_synthetic_control_posterior_interval_one_post_period() {
    measure_bayesian(
        "bayesian_synthetic_control_posterior_interval_one_post_period",
        Panel { post: 1, ..BASE_PANEL },
        base_pre(),
        Bayesian::Posterior,
        0x6E20_0000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn bayesian_synthetic_control_posterior_interval_three_post_periods() {
    measure_bayesian(
        "bayesian_synthetic_control_posterior_interval_three_post_periods",
        Panel { post: 3, ..BASE_PANEL },
        base_pre(),
        Bayesian::Posterior,
        0x6E21_0000,
    );
}

// ------------------------------------------------------------------ unaligned pre-periods

/// 47, 63 and 103 pre-periods: no permutation count is a multiple of twenty.
const UNALIGNED_PRE: SampleGrid =
    SampleGrid { name: "unaligned_pre_periods", factors: [(47, 63), (1, 1), (103, 63)] };

const CONSERVATIVE_SET: &str =
    "the known effect, conservatively at a permutation count that is not a multiple of twenty";

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn conformal_synthetic_control_first_period_set_unaligned_pre_periods() {
    measure_conformal(
        "conformal_synthetic_control_first_period_set_unaligned_pre_periods",
        BASE_PANEL,
        UNALIGNED_PRE.n(63),
        ConformalSet::First,
        Assertion::Conservative(CONSERVATIVE_SET),
        0x6E30_0000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn conformal_synthetic_control_last_period_set_unaligned_pre_periods() {
    measure_conformal(
        "conformal_synthetic_control_last_period_set_unaligned_pre_periods",
        BASE_PANEL,
        UNALIGNED_PRE.n(63),
        ConformalSet::Last,
        Assertion::Conservative(CONSERVATIVE_SET),
        0x6E31_0000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn conformal_synthetic_control_constant_effect_set_unaligned_pre_periods() {
    measure_conformal(
        "conformal_synthetic_control_constant_effect_set_unaligned_pre_periods",
        BASE_PANEL,
        UNALIGNED_PRE.n(63),
        ConformalSet::Constant,
        Assertion::Conservative(CONSERVATIVE_SET),
        0x6E32_0000,
    );
}

// ------------------------------------------------------------------ upper pre-period range

/// Bayesian designs: 120, 160 and 320 pre-periods, four times the base grid's largest 80 at the
/// top point.
fn upper_pre() -> usize {
    SampleGrid::SHORT_SERIES.n(160)
}

/// Conformal designs: permutation counts of 240, 400 and 640 (multiples of twenty), four times
/// the base grid's largest 160 at the top point.
const UPPER_TWENTIES: SampleGrid =
    SampleGrid { name: "upper_twenties", factors: [(3, 8), (5, 8), (1, 1)] };

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn bayesian_synthetic_control_posterior_interval_upper_pre_periods() {
    measure_bayesian(
        "bayesian_synthetic_control_posterior_interval_upper_pre_periods",
        BASE_PANEL,
        upper_pre(),
        Bayesian::Posterior,
        0x6E40_0000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn bayesian_synthetic_control_predictive_path_average_interval_upper_pre_periods() {
    measure_bayesian(
        "bayesian_synthetic_control_predictive_path_average_interval_upper_pre_periods",
        BASE_PANEL,
        upper_pre(),
        Bayesian::Average,
        0x6E41_0000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn bayesian_synthetic_control_predictive_path_first_period_interval_upper_pre_periods() {
    measure_bayesian(
        "bayesian_synthetic_control_predictive_path_first_period_interval_upper_pre_periods",
        BASE_PANEL,
        upper_pre(),
        Bayesian::First,
        0x6E42_0000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn bayesian_synthetic_control_predictive_path_last_period_interval_upper_pre_periods() {
    measure_bayesian(
        "bayesian_synthetic_control_predictive_path_last_period_interval_upper_pre_periods",
        BASE_PANEL,
        upper_pre(),
        Bayesian::Last,
        0x6E43_0000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn conformal_synthetic_control_first_period_set_upper_pre_periods() {
    let set = ConformalSet::First;
    measure_conformal(
        "conformal_synthetic_control_first_period_set_upper_pre_periods",
        BASE_PANEL,
        set.aligned_pre_periods(BASE_PANEL, UPPER_TWENTIES.n(32)),
        set,
        Assertion::TwoSided,
        0x6E44_0000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn conformal_synthetic_control_last_period_set_upper_pre_periods() {
    let set = ConformalSet::Last;
    measure_conformal(
        "conformal_synthetic_control_last_period_set_upper_pre_periods",
        BASE_PANEL,
        set.aligned_pre_periods(BASE_PANEL, UPPER_TWENTIES.n(32)),
        set,
        Assertion::TwoSided,
        0x6E45_0000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn conformal_synthetic_control_constant_effect_set_upper_pre_periods() {
    let set = ConformalSet::Constant;
    measure_conformal(
        "conformal_synthetic_control_constant_effect_set_upper_pre_periods",
        BASE_PANEL,
        set.aligned_pre_periods(BASE_PANEL, UPPER_TWENTIES.n(32)),
        set,
        Assertion::TwoSided,
        0x6E46_0000,
    );
}

// ============================================================ panel difference-in-differences

/// Query name of every panel record.
const PANEL_QUERY: &str = "PanelDid";
/// Interval method of the resampled routes' pointwise interval.
const RESAMPLED: &str = "bootstrap_se";
/// Whole-cluster resamples per replicate of the frequentist routes.
const RESAMPLES: u32 = 199;
const PANEL_DRAWS: usize = 2_000;
const ROLLOUT_TRUTH: f64 = 3.5;
const AUGMENTED_TRUTH: f64 = 1.5;
const TWO_PERIOD_TRUTH: f64 = 2.0;
const OUTCOME: VariableId = VariableId::from_raw(0);

/// One replicate's scored interval and the construction the runtime reported for it.
struct Scored {
    interval: (f64, f64),
    truth: f64,
    construction: Construction,
    scope: ScopeFacts,
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
    construction.query = PANEL_QUERY.into();
    construction.structure = "design".into();
    construction.interval_method = RESAMPLED.into();
    construction.se_kind = String::new();
    construction.dependence = "cluster".into();
    construction.reported_level = REPORTED_LEVEL;
    scope.replicates_ok = replicates_ok;
    Some(Scored { interval: (interval[0], interval[1]), truth, construction, scope })
}

/// Score `interval` against `truth` under the runtime's posterior-quantile key for `result`.
fn posterior_scored(
    study: &Study,
    result: &StudyResult,
    interval: (f64, f64),
    truth: f64,
) -> Option<Scored> {
    let (mut construction, scope, _) = constructions(&study.inspect().ok()?, result)
        .into_iter()
        .find(|(construction, _, _)| construction.interval_method == POSTERIOR_QUANTILE)?;
    construction.query = PANEL_QUERY.into();
    construction.structure = "design".into();
    Some(Scored { interval, truth, construction, scope })
}

/// Measure one panel design: every replicate is bound to the record's construction and scored.
fn measure_panel(
    test: &'static str,
    dgp: &'static str,
    interval: &'static str,
    replicate: impl Fn(u64) -> Option<Scored> + Sync,
) {
    let mut tally = CoverageTally::for_record(RecordKey { test, dgp, interval }, REPORTED_LEVEL);
    for row in map_replicates(n_sim(), replicate) {
        let Some(scored) = row else {
            tally.skip();
            continue;
        };
        tally.bind_validation(&scored.construction, scored.scope, NO_VALIDATION);
        tally.record(Some(scored.interval), scored.truth);
    }
    tally.assert();
}

// ------------------------------------------------------------------ two-period posterior

/// Subjects per group at the top point: 120, 180 and 240, four times the base grid's largest 60
/// at the top.
const TWO_PERIOD_TOP_PER_ARM: usize = 240;

/// Normal-Inverse-Gamma posterior of the difference in mean changes of independent units:
/// the equal-tailed 95% interval of the effect column.
fn two_period_interval(rep: u64) -> Option<Scored> {
    let seed = grid_seed(0x6E50_0000 + rep);
    let mut rng = CausalRng::from_seed(seed);
    let per_arm = UPPER.n(TWO_PERIOD_TOP_PER_ARM);
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
            BayesianConfig::conjugate().prior_scale(100.0).n_draws(PANEL_DRAWS),
        ))
        .build()
        .ok()?;
    let result = study.run(&ExecutionContext::for_tests(seed)).ok()?;
    let fit = result.posterior.as_ref()?;
    let column = fit.effect_column()?;
    let interval = (fit.summaries.q025[column], fit.summaries.q975[column]);
    posterior_scored(&study, &result, interval, TWO_PERIOD_TRUTH)
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn bayesian_two_period_posterior_interval_upper_group_sizes() {
    measure_panel(
        "bayesian_two_period_posterior_interval_upper_group_sizes",
        "two_period_independent_units_homoskedastic_normal_changes_known_difference",
        POSTERIOR_QUANTILE,
        two_period_interval,
    );
}

// ------------------------------------------------------------------ augmented panel

/// Subjects at the top point: 1,600, 2,400 and 3,200, four times the base grid's largest 800.
const AUGMENTED_TOP_SUBJECTS: usize = 3_200;

/// Supplied nuisance predictions equal to the true propensity `0.25 + 0.5 x` and the true
/// untreated change `x`; the whole-cluster bootstrap holds them fixed.
fn augmented_interval(rep: u64) -> Option<Scored> {
    let seed = grid_seed(0x6E51_0000 + rep);
    let mut rng = CausalRng::from_seed(seed);
    let n = UPPER.n(AUGMENTED_TOP_SUBJECTS);
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
fn augmented_panel_true_nuisance_bootstrap_interval_upper_subjects() {
    measure_panel(
        "augmented_panel_true_nuisance_bootstrap_interval_upper_subjects",
        "two_period_panel_true_propensity_and_untreated_change_supplied_known_effect_on_treated",
        RESAMPLED,
        augmented_interval,
    );
}

// ------------------------------------------------------------------ staggered rollout

/// Units per cohort-share at the top point: 3 x 480 = 1,440 units, four times the base grid's
/// largest 360; the grid is 720, 1,080 and 1,440 units.
const ROLLOUT_TOP_THIRD: usize = 480;

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
    let (data, units, periods, cohorts) = staggered(3 * UPPER.n(ROLLOUT_TOP_THIRD), &mut rng);
    let query = rollout_query(units, periods, cohorts, spec).with_resampling(scheme, RESAMPLES);
    let study = Study::tabular(data).query(query).build().ok()?;
    let result = study.run(&ExecutionContext::for_tests(seed)).ok()?;
    let rollout = result.quasi_rollout()?;
    let interval = rollout.overall()?.interval?;
    resampled(&study, &result, interval, ROLLOUT_TRUTH, Some(rollout.replicates_ok))
}

const ROLLOUT_DGP: &str = "staggered_three_cohort_heterogeneous_effects_known_simple_aggregate";

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn rollout_callaway_santanna_not_yet_treated_multiplier_interval_upper_units() {
    measure_panel(
        "rollout_callaway_santanna_not_yet_treated_multiplier_interval_upper_units",
        ROLLOUT_DGP,
        RESAMPLED,
        |rep| {
            rollout_interval(
                rollout_spec(
                    RolloutDidEstimator::CallawaySantAnna,
                    DidComparisonGroup::NotYetTreated,
                ),
                DidResamplingScheme::ClusterMultiplier,
                0x6E52_0000,
                rep,
            )
        },
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn rollout_callaway_santanna_never_treated_bootstrap_interval_upper_units() {
    measure_panel(
        "rollout_callaway_santanna_never_treated_bootstrap_interval_upper_units",
        ROLLOUT_DGP,
        RESAMPLED,
        |rep| {
            rollout_interval(
                rollout_spec(
                    RolloutDidEstimator::CallawaySantAnna,
                    DidComparisonGroup::NeverTreated,
                ),
                DidResamplingScheme::ClusterBootstrap,
                0x6E53_0000,
                rep,
            )
        },
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn rollout_sun_abraham_multiplier_interval_upper_units() {
    measure_panel(
        "rollout_sun_abraham_multiplier_interval_upper_units",
        ROLLOUT_DGP,
        RESAMPLED,
        |rep| {
            rollout_interval(
                rollout_spec(RolloutDidEstimator::SunAbraham, DidComparisonGroup::NeverTreated),
                DidResamplingScheme::ClusterMultiplier,
                0x6E54_0000,
                rep,
            )
        },
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn rollout_imputation_multiplier_interval_upper_units() {
    measure_panel(
        "rollout_imputation_multiplier_interval_upper_units",
        ROLLOUT_DGP,
        RESAMPLED,
        |rep| {
            rollout_interval(
                rollout_spec(RolloutDidEstimator::Imputation, DidComparisonGroup::NotYetTreated),
                DidResamplingScheme::ClusterMultiplier,
                0x6E55_0000,
                rep,
            )
        },
    );
}

/// Bayesian bootstrap credible interval of the rollout's simple aggregate: flat-Dirichlet
/// whole-unit weights, the whole group-time procedure refit per draw.
fn bayesian_rollout_interval(rep: u64) -> Option<Scored> {
    let seed = grid_seed(0x6E56_0000 + rep);
    let mut rng = CausalRng::from_seed(seed);
    let (data, units, periods, cohorts) = staggered(3 * UPPER.n(ROLLOUT_TOP_THIRD), &mut rng);
    let query = rollout_query(
        units,
        periods,
        cohorts,
        rollout_spec(RolloutDidEstimator::CallawaySantAnna, DidComparisonGroup::NotYetTreated),
    );
    let study = Study::tabular(data)
        .query(query)
        .inference(InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(PANEL_DRAWS)))
        .build()
        .ok()?;
    let result = study.run(&ExecutionContext::for_tests(seed)).ok()?;
    let [lower, upper] = result.quasi_rollout()?.overall()?.interval?;
    posterior_scored(&study, &result, (lower, upper), ROLLOUT_TRUTH)
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn bayesian_rollout_bootstrap_credible_interval_upper_units() {
    measure_panel(
        "bayesian_rollout_bootstrap_credible_interval_upper_units",
        ROLLOUT_DGP,
        POSTERIOR_QUANTILE,
        bayesian_rollout_interval,
    );
}

// ============================================================ bounds and association

/// What a design states about the interval it scores.
struct Stated<'a> {
    interval_method: &'a str,
    level: f64,
    se_kind: &'a str,
    dependence: &'a str,
    functional: &'a str,
    /// Bootstrap replicates that fitted, when the interval is a resampling one.
    replicates: Option<u32>,
    /// Posterior draws behind the interval, when it is drawn.
    draws: Option<u32>,
    /// The posterior a drawn interval comes from, when the runtime key names another.
    posterior: Option<&'a str>,
}

/// The construction the runtime keys the result under, on the design axes, with the interval
/// fields set to the interval scored.
fn reported_as(prepared: &PreparedStudy, result: &StudyResult, stated: &Stated<'_>) -> Reported {
    let contract = prepared.contract_for_result(result).expect("execution contract");
    let (base, scope, _) = constructions(&contract, result)
        .into_iter()
        .next()
        .expect("the runtime reports a calibration basis");
    let posterior = stated.posterior.map_or(base.posterior.clone(), Into::into);
    let construction = Construction {
        structure: "design".into(),
        posterior,
        interval_method: stated.interval_method.into(),
        se_kind: stated.se_kind.into(),
        dependence: stated.dependence.into(),
        functional: stated.functional.into(),
        reported_level: stated.level,
        ..base
    };
    let scope =
        ScopeFacts { replicates_ok: stated.replicates, posterior_draws: stated.draws, ..scope };
    (construction, scope, NO_VALIDATION)
}

// ------------------------------------------------------------------ binary-instrument bounds

/// Rows at the top point: 4,000, 6,000 and 8,000, four times the base grid's largest 2,000.
const IV_TOP_ROWS: usize = 8_000;
const IV_REPLICATES: u32 = 399;
const IV_DRAWS: usize = 400;
const IV_FUNCTIONAL: &str = "ate_identified_set";
const IV_FREQUENTIST_INTERVAL: &str = "imbens_manski_row_bootstrap";
const IV_BAYESIAN_INTERVAL: &str = "posterior_endpoint_quantiles";

/// Population shares of the always-takers, never-takers and compliers.
const TYPE_SHARE: [f64; 3] = [0.2, 0.3, 0.5];
/// `P(Y(1) = 1)` within each type.
const OUTCOME_UNDER_TREATMENT: [f64; 3] = [0.7, 0.4, 0.8];
/// `P(Y(0) = 1)` within each type.
const OUTCOME_UNDER_CONTROL: [f64; 3] = [0.5, 0.3, 0.4];

/// Treatment taken by a type (0 always, 1 never, 2 complier) under instrument value `z`.
fn treatment_of(kind: usize, z: usize) -> usize {
    match kind {
        0 => 1,
        1 => 0,
        _ => z,
    }
}

/// The exact observed law `P(Y=y, D=d | Z=z)` of the mixture; cells within an arm are ordered
/// `(0,0), (1,0), (0,1), (1,1)` as `(y, d)`.
fn response_type_mixture_population_law() -> [[f64; 4]; 2] {
    let mut cells = [[0.0; 4]; 2];
    for z in 0..2 {
        for kind in 0..3 {
            let d = treatment_of(kind, z);
            let p1 =
                if d == 1 { OUTCOME_UNDER_TREATMENT[kind] } else { OUTCOME_UNDER_CONTROL[kind] };
            cells[z][1 + 2 * d] += TYPE_SHARE[kind] * p1;
            cells[z][2 * d] += TYPE_SHARE[kind] * (1.0 - p1);
        }
    }
    cells
}

/// Lower Balke-Pearl bound of `E[Y(1) - Y(0)]` from the observed law, in closed form.
fn balke_pearl_lower(cells: &[[f64; 4]; 2]) -> f64 {
    let p = |y: usize, d: usize, z: usize| cells[z][y + 2 * d];
    [
        p(1, 1, 1) + p(0, 0, 0) - 1.0,
        p(1, 1, 0) + p(0, 0, 1) - 1.0,
        p(1, 1, 0) - p(1, 1, 1) - p(1, 0, 1) - p(0, 1, 0) - p(1, 0, 0),
        p(1, 1, 1) - p(1, 1, 0) - p(1, 0, 0) - p(0, 1, 1) - p(1, 0, 1),
        -p(0, 1, 1) - p(1, 0, 1),
        -p(0, 1, 0) - p(1, 0, 0),
        p(0, 0, 1) - p(0, 1, 1) - p(1, 0, 1) - p(0, 1, 0) - p(0, 0, 0),
        p(0, 0, 0) - p(0, 1, 0) - p(1, 0, 0) - p(0, 1, 1) - p(0, 0, 1),
    ]
    .into_iter()
    .fold(f64::NEG_INFINITY, f64::max)
}

/// The true sharp lower and upper bounds of the average effect on the population law. The
/// upper bound is the negated lower bound of the law with the outcome relabelled.
fn true_effect_bounds() -> (f64, f64) {
    let law = response_type_mixture_population_law();
    let mut flipped = [[0.0; 4]; 2];
    for z in 0..2 {
        for d in 0..2 {
            for y in 0..2 {
                flipped[z][(1 - y) + 2 * d] = law[z][y + 2 * d];
            }
        }
    }
    (balke_pearl_lower(&law), -balke_pearl_lower(&flipped))
}

/// Instrument, treatment and outcome columns of `n` units drawn from the mixture.
fn response_type_mixture_sample(n: usize, seed: u64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut next = uniform(seed, 0x1B01);
    let (mut z, mut d, mut y) =
        (Vec::with_capacity(n), Vec::with_capacity(n), Vec::with_capacity(n));
    for _ in 0..n {
        let instrument = usize::from(next() >= 0.5);
        let draw = next();
        let kind = if draw < TYPE_SHARE[0] {
            0
        } else if draw < TYPE_SHARE[0] + TYPE_SHARE[1] {
            1
        } else {
            2
        };
        let treated = treatment_of(kind, instrument);
        let p1 =
            if treated == 1 { OUTCOME_UNDER_TREATMENT[kind] } else { OUTCOME_UNDER_CONTROL[kind] };
        z.push(instrument as f64);
        d.push(treated as f64);
        y.push(f64::from(next() < p1));
    }
    (z, d, y)
}

/// The interval one simulated study publishes for the identified set, with its construction;
/// `None` when the route refuses the sample's law, publishes no interval, or does not state its
/// support row: a fit outside the row is not evidence for it.
fn iv_interval(n: usize, seed: u64, bayesian: bool) -> Option<([f64; 2], Reported)> {
    let (z, d, y) = response_type_mixture_sample(n, seed);
    let data = TabularData::from_f64_columns([
        ("y", y.as_slice()),
        ("d", d.as_slice()),
        ("z", z.as_slice()),
    ])
    .expect("the simulated columns are finite and aligned");
    let query = CausalQuery::IvBounds(IvBoundsQuery::new(v(0), v(1), v(2)));
    let builder = Study::tabular(data.clone()).query(query).refute(RefuteSuite::None);
    let builder = if bayesian {
        builder.inference(InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(IV_DRAWS)))
    } else {
        builder.bootstrap_replicates(IV_REPLICATES)
    };
    let study = builder.build().expect("the bounds study is valid");
    let ctx = ExecutionContext::for_tests(seed);
    let prepared = study.prepare(&ctx).expect("the bounds study prepares");
    let result = prepared.estimate(&data, &ctx).ok()?;
    let interval = result.iv_bounds().filter(|_| result.states_section_row())?.interval?;
    let wanted = if bayesian { IV_BAYESIAN_INTERVAL } else { IV_FREQUENTIST_INTERVAL };
    assert_eq!(interval.method.as_str(), wanted, "the route formed another interval");
    assert!((interval.level - REPORTED_LEVEL).abs() < 1e-12, "the route reports another level");
    let resamples = u32::try_from(interval.replicates).ok();
    let reported = reported_as(
        &prepared,
        &result,
        &Stated {
            interval_method: wanted,
            level: interval.level,
            se_kind: "",
            dependence: "iid",
            functional: IV_FUNCTIONAL,
            replicates: if bayesian { None } else { resamples },
            draws: if bayesian { resamples } else { None },
            posterior: bayesian.then_some("bayesian_bootstrap.row_weights"),
        },
    );
    Some(([interval.lower, interval.upper], reported))
}

#[derive(Clone, Copy)]
enum Edge {
    Lower,
    Upper,
}

fn measure_iv(test: &'static str, bayesian: bool, edge: Edge, seed_base: u64) {
    let n = UPPER.n(IV_TOP_ROWS);
    let (lower, upper) = true_effect_bounds();
    let (truth, side, target) = match edge {
        Edge::Lower => (lower, CoverageSide::Lower, "true sharp lower bound of the average effect"),
        Edge::Upper => (upper, CoverageSide::Upper, "true sharp upper bound of the average effect"),
    };
    let interval = if bayesian { IV_BAYESIAN_INTERVAL } else { IV_FREQUENTIST_INTERVAL };
    let mut tally = CoverageTally::for_record(
        RecordKey { test, dgp: "response_type_mixture_population_law", interval },
        REPORTED_LEVEL,
    );
    let runs = map_replicates(n_sim(), |rep| iv_interval(n, grid_seed(seed_base + rep), bayesian));
    for run in runs {
        let Some((band, reported)) = run else {
            tally.skip();
            continue;
        };
        bind_reported(&mut tally, &[reported]);
        tally.record(Some((band[0], band[1])), truth);
    }
    tally.assert_one_sided(side, target);
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn iv_bounds_imbens_manski_interval_covers_the_lower_bound_upper_rows() {
    measure_iv(
        "iv_bounds_imbens_manski_interval_covers_the_lower_bound_upper_rows",
        false,
        Edge::Lower,
        0x6E60_0000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn iv_bounds_imbens_manski_interval_covers_the_upper_bound_upper_rows() {
    measure_iv(
        "iv_bounds_imbens_manski_interval_covers_the_upper_bound_upper_rows",
        false,
        Edge::Upper,
        0x6E61_0000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn bayesian_iv_bounds_endpoint_interval_covers_the_lower_bound_upper_rows() {
    measure_iv(
        "bayesian_iv_bounds_endpoint_interval_covers_the_lower_bound_upper_rows",
        true,
        Edge::Lower,
        0x6E62_0000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn bayesian_iv_bounds_endpoint_interval_covers_the_upper_bound_upper_rows() {
    measure_iv(
        "bayesian_iv_bounds_endpoint_interval_covers_the_upper_bound_upper_rows",
        true,
        Edge::Upper,
        0x6E63_0000,
    );
}

// ------------------------------------------------------------------ association

/// Rows at the top point: 800, 1,200 and 1,600, four times the base grid's largest 400.
const ASSOCIATION_TOP_ROWS: usize = 1_600;
const REPLICATES: u32 = 399;
const DRAWS: usize = 400;
const SLOPE: f64 = 2.0;
const SERIES_SLOPE: f64 = 0.5;
const SERIES_PERSISTENCE: f64 = 0.5;
const ASSOCIATION_FUNCTIONAL: &str = "linear_regression_slope";

/// Rows `y = 1 + 2 x + e`, `x ~ N(0, 1)`, with error standard deviation `0.5 + 0.5 |x|`.
fn heteroskedastic_line_known_slope(n: usize, seed: u64) -> (Vec<f64>, Vec<f64>) {
    let mut draw_x = gaussian(stream_seed(seed, 3));
    let mut draw_e = gaussian(stream_seed(seed, 4));
    let x: Vec<f64> = (0..n).map(|_| draw_x()).collect();
    let y = x.iter().map(|x| 1.0 + SLOPE * x + (0.5 + 0.5 * x.abs()) * draw_e()).collect();
    (x, y)
}

/// Rows `y = 1 + 2 x + e`, `x ~ N(0, 1)`, `e ~ N(0, 1)`.
fn homoskedastic_line_known_slope(n: usize, seed: u64) -> (Vec<f64>, Vec<f64>) {
    let mut draw_x = gaussian(stream_seed(seed, 5));
    let mut draw_e = gaussian(stream_seed(seed, 6));
    let x: Vec<f64> = (0..n).map(|_| draw_x()).collect();
    let y = x.iter().map(|x| 1.0 + SLOPE * x + draw_e()).collect();
    (x, y)
}

/// A series `y_t = 1 + 0.5 x_t + u_t` with the predictor `x` and the error `u` independent
/// stationary AR(1) processes of persistence 0.5.
fn autoregressive_line_known_slope(n: usize, seed: u64) -> (Vec<f64>, Vec<f64>) {
    let x = ar1_noise(n, SERIES_PERSISTENCE, 1.0, stream_seed(seed, 1));
    let u = ar1_noise(n, SERIES_PERSISTENCE, 1.0, stream_seed(seed, 2));
    let y = x.iter().zip(&u).map(|(x, u)| 1.0 + SERIES_SLOPE * x + u).collect();
    (x, y)
}

/// Which interval of the slope term a design scores.
#[derive(Clone, Copy)]
enum Source {
    /// `estimate +- z * standard_error`, at the level the facade reports.
    StandardError,
    /// The resampling spread the estimator publishes.
    Spread,
    /// The equal-tailed posterior interval.
    Posterior,
}

#[derive(Clone, Copy)]
struct Design {
    temporal: bool,
    bayesian: bool,
    homoskedastic: bool,
    replicates: u32,
    source: Source,
    interval: &'static str,
    dependence: &'static str,
}

fn association_study(builder: StudyBuilder, design: Design) -> Study {
    let query =
        AssociationQuery::new(v(1), [v(0)]).with_functional(AssociationFunctional::Coefficients);
    let builder = builder.query(CausalQuery::Association(query)).refute(RefuteSuite::None);
    let builder = if design.bayesian {
        builder.inference(InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(DRAWS)))
    } else if design.replicates > 0 {
        builder.bootstrap_replicates(design.replicates)
    } else {
        builder
    };
    let builder = if design.temporal && !design.bayesian && design.replicates > 0 {
        builder.sampling_design(DeclaredSamplingDesign::DependentTimeSeries)
    } else {
        builder
    };
    builder.build().expect("the association study is valid")
}

/// The interval one simulated study publishes for the slope, with its construction; `None` when
/// the result states none or does not state its support row.
fn association_interval(n: usize, seed: u64, design: Design) -> Option<([f64; 2], Reported)> {
    let ctx = ExecutionContext::for_tests(seed);
    let (prepared, result) = if design.temporal {
        let (x, y) = autoregressive_line_known_slope(n, seed);
        let data = TimeSeriesData::from_f64_columns(
            [("x", x.as_slice()), ("y", y.as_slice())],
            1_000_000_000,
        )
        .expect("the simulated series is finite and aligned");
        let prepared = association_study(Study::series(data.clone()), design)
            .prepare(&ctx)
            .expect("the association study prepares");
        let result = prepared.estimate_series(&data, &ctx).expect("the association runs");
        (prepared, result)
    } else {
        let (x, y) = if design.homoskedastic {
            homoskedastic_line_known_slope(n, seed)
        } else {
            heteroskedastic_line_known_slope(n, seed)
        };
        let data = TabularData::from_f64_columns([("x", x.as_slice()), ("y", y.as_slice())])
            .expect("the simulated rows are finite and aligned");
        let prepared = association_study(Study::tabular(data.clone()), design)
            .prepare(&ctx)
            .expect("the association study prepares");
        let result = prepared.estimate(&data, &ctx).expect("the association runs");
        (prepared, result)
    };
    let payload = result.association().filter(|_| result.states_section_row())?;
    assert_eq!(payload.estimate.terms.len(), 2, "an intercept and one slope");
    let term = &payload.estimate.terms[1];
    let band = match design.source {
        Source::StandardError => normal_interval(term.estimate, term.standard_error, Z95),
        Source::Spread => term.spread,
        Source::Posterior => term.posterior.map(|summary| (summary.lower, summary.upper)),
    }?;
    let reported = reported_as(
        &prepared,
        &result,
        &Stated {
            interval_method: design.interval,
            level: REPORTED_LEVEL,
            // A posterior interval has no standard-error construction of its own.
            se_kind: match design.source {
                Source::Posterior => "",
                _ => term.standard_error_method.unwrap_or(""),
            },
            dependence: design.dependence,
            functional: ASSOCIATION_FUNCTIONAL,
            replicates: payload.estimate.resampling.map(|record| record.replicates_ok),
            draws: payload
                .estimate
                .posterior
                .filter(|record| record.draws > 0)
                .and_then(|record| u32::try_from(record.draws).ok()),
            // The series posterior is the block Bayesian bootstrap, not the conjugate model.
            posterior: (design.bayesian && design.temporal)
                .then_some("bayesian_bootstrap.contiguous_block_weights"),
        },
    );
    Some(([band.0, band.1], reported))
}

fn measure_association(
    test: &'static str,
    dgp: &'static str,
    design: Design,
    truth: f64,
    seed_base: u64,
) {
    let n = UPPER.n(ASSOCIATION_TOP_ROWS);
    let mut tally = CoverageTally::for_record(
        RecordKey { test, dgp, interval: design.interval },
        REPORTED_LEVEL,
    );
    let runs =
        map_replicates(n_sim(), |rep| association_interval(n, grid_seed(seed_base + rep), design));
    for run in runs {
        let Some((band, reported)) = run else {
            tally.skip();
            continue;
        };
        bind_reported(&mut tally, &[reported]);
        tally.record(Some((band[0], band[1])), truth);
    }
    tally.assert();
}

const STATIC: Design = Design {
    temporal: false,
    bayesian: false,
    homoskedastic: false,
    replicates: 0,
    source: Source::StandardError,
    interval: "analytic_se",
    dependence: "iid",
};

const TEMPORAL: Design = Design { temporal: true, dependence: "serial_hac", ..STATIC };

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn association_static_sandwich_slope_interval_upper_rows() {
    measure_association(
        "association_static_sandwich_slope_interval_upper_rows",
        "heteroskedastic_line_known_slope",
        STATIC,
        SLOPE,
        0x6E70_0000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn association_static_bootstrap_spread_slope_interval_upper_rows() {
    measure_association(
        "association_static_bootstrap_spread_slope_interval_upper_rows",
        "heteroskedastic_line_known_slope",
        Design {
            replicates: REPLICATES,
            source: Source::Spread,
            interval: "studentized_spread",
            ..STATIC
        },
        SLOPE,
        0x6E71_0000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn association_temporal_newey_west_slope_interval_upper_rows() {
    measure_association(
        "association_temporal_newey_west_slope_interval_upper_rows",
        "autoregressive_line_known_slope",
        TEMPORAL,
        SERIES_SLOPE,
        0x6E72_0000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn association_temporal_block_bootstrap_spread_slope_interval_upper_rows() {
    measure_association(
        "association_temporal_block_bootstrap_spread_slope_interval_upper_rows",
        "autoregressive_line_known_slope",
        Design {
            replicates: REPLICATES,
            source: Source::Spread,
            interval: "fixed_b_percentile_spread",
            dependence: "circular_block:row_block",
            ..TEMPORAL
        },
        SERIES_SLOPE,
        0x6E73_0000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn bayesian_association_static_slope_credible_interval_upper_rows() {
    measure_association(
        "bayesian_association_static_slope_credible_interval_upper_rows",
        "homoskedastic_line_known_slope",
        Design {
            bayesian: true,
            homoskedastic: true,
            source: Source::Posterior,
            interval: "posterior_equal_tailed",
            ..STATIC
        },
        SLOPE,
        0x6E74_0000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn bayesian_association_temporal_slope_credible_interval_upper_rows() {
    measure_association(
        "bayesian_association_temporal_slope_credible_interval_upper_rows",
        "autoregressive_line_known_slope",
        Design {
            bayesian: true,
            source: Source::Posterior,
            interval: "posterior_equal_tailed",
            dependence: "contiguous_block:row_block",
            ..TEMPORAL
        },
        SERIES_SLOPE,
        0x6E75_0000,
    );
}
