//! Support rows of the difference-in-differences rollout designs and the synthetic-control
//! designs with intervals, through the public study facade: the staggered heterogeneity-robust
//! rollout and its Bayesian bootstrap, the covariate-adjusted two-period panel, the augmented
//! panel with supplied predictions, the two-period posterior, the conformal sets and the
//! placebo-variance interval of a synthetic control, and its donor-weight posterior with and
//! without the predictive path.
//!
//! Each route's test runs a fit that meets its row and checks that the result states support,
//! that the statement survives the contracted artifact, that a forged statement (a count
//! below the row, another procedure) is refused by the loader, and that the same route below
//! its row runs and states no support.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0
#![allow(
    clippy::too_many_lines,
    clippy::cast_possible_wrap,
    reason = "a route's supported fit, its forgeries and its unsupported neighbour read as one sequence; period indices are small"
)]

use std::sync::Arc;

use antecedent::prelude::ExecutionContext;
use antecedent::{BayesianConfig, InferenceMode, PanelDidQuery, PreparedStudy, Study, StudyResult};
use antecedent_core::{
    DidAggregation, DidComparisonGroup, DidNuisanceMethod, DidResamplingScheme,
    HeterogeneityRobustDid, RolloutDidEstimator, SyntheticControlQuery, VariableId,
};
use antecedent_data::TabularData;
use antecedent_io::analysis_result_artifact::AnalysisResultWire;

#[path = "common/retained_graphless.rs"]
mod retained_graphless;

/// Code of the diagnostic by which a design posterior states its support row.
const POSTERIOR_ROW: &str = antecedent_io::graphless_support::DESIGN_POSTERIOR_ROW_CODE;
/// What the loader says of a rollout section whose support claim no row admits.
const NO_ROW: &str = "claims no support";
/// What the loader says of a stated count that is not the one its query and payload give.
const RECOUNTED: &str = "states support on a count";
/// What the loader says of a design posterior whose stated row does not admit it.
const NO_POSTERIOR_ROW: &str = "states a graphless support row that does not admit it";
const OUTCOME: VariableId = VariableId::from_raw(0);

struct Rng(u64);

impl Rng {
    fn uniform(&mut self) -> f64 {
        (antecedent_core::splitmix64(&mut self.0) >> 11) as f64 / (1_u64 << 53) as f64
    }

    fn normal(&mut self) -> f64 {
        let u = self.uniform().max(1e-300);
        let v = self.uniform();
        (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
    }
}

/// Bayesian inference with `draws` posterior draws, at `prior_scale` when one is declared.
fn bayes(prior_scale: Option<f64>, draws: usize) -> Option<InferenceMode> {
    let config = BayesianConfig::conjugate().n_draws(draws);
    Some(InferenceMode::Bayesian(match prior_scale {
        Some(scale) => config.prior_scale(scale),
        None => config,
    }))
}

fn fit_panel(
    data: &TabularData,
    query: PanelDidQuery,
    inference: Option<InferenceMode>,
    seed: u64,
) -> (PreparedStudy, StudyResult, ExecutionContext) {
    let ctx = ExecutionContext::for_tests(seed);
    let study = match inference {
        Some(mode) => Study::tabular(data.clone()).query(query).inference(mode).build(),
        None => Study::tabular(data.clone()).query(query).build(),
    }
    .unwrap();
    let prepared = study.prepare(&ctx).unwrap();
    let result = prepared.estimate(data, &ctx).unwrap();
    (prepared, result, ctx)
}

fn fit_synthetic(
    data: &TabularData,
    query: SyntheticControlQuery,
    inference: Option<InferenceMode>,
    seed: u64,
) -> (PreparedStudy, StudyResult, ExecutionContext) {
    let ctx = ExecutionContext::for_tests(seed);
    let study = match inference {
        Some(mode) => Study::tabular(data.clone()).query(query).inference(mode).build(),
        None => Study::tabular(data.clone()).query(query).build(),
    }
    .unwrap();
    let prepared = study.prepare(&ctx).unwrap();
    let result = prepared.estimate(data, &ctx).unwrap();
    (prepared, result, ctx)
}

/// The decoded contracted artifact of a result: its variable names and body.
fn artifact(
    prepared: &PreparedStudy,
    result: &StudyResult,
    ctx: &ExecutionContext,
    label: &str,
) -> (Vec<String>, AnalysisResultWire) {
    let bytes = prepared.encode_contracted_result(result, label, ctx).unwrap();
    let (_, header, body) = antecedent_io::decode_analysis_result_artifact(&bytes).unwrap();
    (header.variable_names, body)
}

/// The loader's refusal of a forged body.
fn refusal(names: &[String], forged: &AnalysisResultWire) -> String {
    antecedent_io::encode_analysis_result_artifact(forged, names.to_vec(), "forged-support")
        .unwrap_err()
        .to_string()
}

/// The loader's refusal after one named diagnostic of the rollout section is overwritten.
fn forged_section(names: &[String], body: &AnalysisResultWire, name: &str, value: f64) -> String {
    let mut forged = body.clone();
    let section = forged.quasi_rollout.as_mut().expect("the rollout section is exported");
    let entry = section
        .diagnostics
        .iter_mut()
        .find(|(key, _)| key == name)
        .unwrap_or_else(|| panic!("the section reports {name}"));
    entry.1 = value;
    refusal(names, &forged)
}

fn section_supported(result: &StudyResult) -> bool {
    result.quasi_rollout().expect("a rollout payload").supported
}

fn states_posterior_row(result: &StudyResult) -> bool {
    result.diagnostics.iter().any(|diagnostic| diagnostic.code.as_ref() == POSTERIOR_ROW)
}

/// The loader's refusal after one field of the stated posterior row is overwritten.
fn forged_posterior_row(
    names: &[String],
    body: &AnalysisResultWire,
    name: &str,
    value: &str,
) -> String {
    let mut forged = body.clone();
    let stated = forged
        .diagnostics
        .iter_mut()
        .find(|diagnostic| diagnostic.code == POSTERIOR_ROW)
        .expect("the artifact states a row");
    let entry = stated
        .fields
        .iter_mut()
        .find(|(key, _)| key == name)
        .unwrap_or_else(|| panic!("the stated row has no {name}"));
    entry.1 = value.to_owned();
    refusal(names, &forged)
}

// ------------------------------------------------------------------ fixtures

/// `per_cohort` units in each of the cohorts never treated, adopting at 2 and adopting at 4,
/// over five periods; the effect is `3 (period - cohort)` from adoption on, so the simple
/// aggregate is 3.5.
fn staggered(seed: u64, per_cohort: usize) -> (TabularData, Vec<Arc<str>>, Vec<i64>, Vec<i64>) {
    let mut rng = Rng(seed);
    let (mut y, mut units, mut periods, mut cohorts) = (vec![], vec![], vec![], vec![]);
    for cohort in [0_i64, 2, 4] {
        for i in 0..per_cohort {
            let label: Arc<str> = Arc::from(format!("c{cohort}_{i:04}"));
            let alpha = rng.normal();
            for period in 1..=5_i64 {
                let effect = if cohort > 0 && period >= cohort {
                    3.0 * (period - cohort) as f64
                } else {
                    0.0
                };
                y.push(alpha + 0.5 * period as f64 + effect + rng.normal());
                units.push(Arc::clone(&label));
                periods.push(period);
                cohorts.push(cohort);
            }
        }
    }
    (TabularData::from_f64_columns([("y", y.as_slice())]).unwrap(), units, periods, cohorts)
}

fn staggered_query(
    panel: &(TabularData, Vec<Arc<str>>, Vec<i64>, Vec<i64>),
    estimator: RolloutDidEstimator,
    comparison: DidComparisonGroup,
    aggregation: DidAggregation,
) -> PanelDidQuery {
    let (_, units, periods, cohorts) = panel;
    PanelDidQuery::heterogeneity_robust(
        OUTCOME,
        units.clone(),
        units.clone(),
        periods.clone(),
        cohorts.clone(),
        HeterogeneityRobustDid {
            estimator,
            comparison,
            aggregation,
            ..HeterogeneityRobustDid::default()
        },
    )
}

/// `n` units over two periods with a covariate on an even grid over `[-1.2, 1.2]`, a treated
/// share logistic in it, the untreated change `1 + x` and an effect on the treated of 2.
fn adjusted(seed: u64, n: usize, nuisance: DidNuisanceMethod) -> (TabularData, PanelDidQuery) {
    let mut rng = Rng(seed);
    let (mut y, mut x, mut treated, mut post, mut units) = (vec![], vec![], vec![], vec![], vec![]);
    for i in 0..n {
        let covariate = -1.2 + 2.4 * (i as f64 + 0.5) / n as f64;
        let d = rng.uniform() < 1.0 / (1.0 + (-covariate).exp());
        let alpha = rng.normal() + covariate;
        let label: Arc<str> = Arc::from(format!("s{i:05}"));
        for after in [false, true] {
            let trend = if after { 1.0 + covariate } else { 0.0 };
            let effect = if d && after { 2.0 } else { 0.0 };
            y.push(alpha + trend + effect + 0.5 * rng.normal());
            x.push(covariate);
            treated.push(d);
            post.push(after);
            units.push(Arc::clone(&label));
        }
    }
    let data = TabularData::from_f64_columns([("y", y.as_slice()), ("x", x.as_slice())]).unwrap();
    let query = PanelDidQuery::covariate_adjusted_two_period(
        OUTCOME,
        treated,
        post,
        units.clone(),
        units,
        vec![VariableId::from_raw(1)],
        nuisance,
    );
    (data, query)
}

/// `n` subjects with the true propensity `0.3 + 0.4 x` and the true untreated change `x`
/// supplied; the effect on the treated is 1.5.
fn augmented(seed: u64, n: usize) -> (TabularData, PanelDidQuery) {
    let mut rng = Rng(seed);
    let (mut pre, mut post, mut p, mut m, mut treated, mut ids) =
        (vec![], vec![], vec![], vec![], vec![], vec![]);
    for i in 0..n {
        let x = (i as f64 + 0.5) / n as f64;
        let propensity = 0.3 + 0.4 * x;
        let d = rng.uniform() < propensity;
        let base = rng.normal();
        pre.push(base);
        post.push(base + x + if d { 1.5 } else { 0.0 } + 0.2 * rng.normal());
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
    .unwrap();
    let query = PanelDidQuery::augmented_panel(
        VariableId::from_raw(1),
        VariableId::from_raw(0),
        VariableId::from_raw(2),
        VariableId::from_raw(3),
        treated,
        ids.clone(),
        ids,
        true,
    );
    (data, query)
}

/// `per_arm` subjects in each group over two periods with Gaussian changes; the difference
/// in mean changes is 2.
fn two_period(seed: u64, per_arm: usize) -> (TabularData, PanelDidQuery) {
    let mut rng = Rng(seed);
    let (mut subjects, mut treated, mut post, mut outcome) = (vec![], vec![], vec![], vec![]);
    for unit in 0..2 * per_arm {
        let active = unit >= per_arm;
        let before = 1.0 + rng.normal();
        let change = 0.7 + if active { 2.0 } else { 0.0 } + 0.5 * rng.normal();
        for after in [false, true] {
            subjects.push(Arc::<str>::from(format!("s{unit:04}")));
            treated.push(active);
            post.push(after);
            outcome.push(if after { before + change } else { before });
        }
    }
    let data = TabularData::from_f64_columns([("y", outcome.as_slice())]).unwrap();
    let query = PanelDidQuery::new(OUTCOME, treated, post, subjects.clone(), subjects);
    (data, query)
}

/// Ten donors and a treated unit that is a fixed convex combination of three of them plus
/// noise, over `pre` pre-periods and five post periods with an effect of 3 on each.
fn one_treated(seed: u64, pre: usize) -> (TabularData, SyntheticControlQuery) {
    let mut rng = Rng(seed);
    let (mut y, mut units, mut periods) = (vec![], vec![], vec![]);
    for t in 1..=pre + 5 {
        let donors: Vec<f64> = (0..10).map(|j| f64::from(j) + rng.normal()).collect();
        for (j, value) in donors.iter().enumerate() {
            y.push(*value);
            units.push(Arc::<str>::from(format!("d{j:02}")));
            periods.push(t as i64);
        }
        let effect = if t > pre { 3.0 } else { 0.0 };
        y.push(0.5 * donors[3] + 0.3 * donors[4] + 0.2 * donors[5] + 0.5 * rng.normal() + effect);
        units.push(Arc::<str>::from("treated"));
        periods.push(t as i64);
    }
    let data = TabularData::from_f64_columns([("y", y.as_slice())]).unwrap();
    let query = SyntheticControlQuery::new(
        OUTCOME,
        units,
        periods,
        Arc::<str>::from("treated"),
        pre as i64 + 1,
    );
    (data, query)
}

/// `donors` never-treated units and three units adopting at periods 21, 24 and 27 of 30, all
/// drawn from one law (a level, a common random walk, noise); the effect is 2 from adoption.
fn staggered_synthetic(seed: u64, donors: usize) -> (TabularData, SyntheticControlQuery) {
    let mut rng = Rng(seed);
    let mut factor = Vec::new();
    let mut level = 0.0;
    for _ in 0..30 {
        level += rng.normal();
        factor.push(level);
    }
    let adoptions: Vec<(Arc<str>, i64)> =
        vec![(Arc::from("t0"), 21), (Arc::from("t1"), 24), (Arc::from("t2"), 27)];
    let mut labels: Vec<(Arc<str>, Option<i64>)> =
        (0..donors).map(|j| (Arc::from(format!("c{j:03}")), None)).collect();
    labels.extend(adoptions.iter().map(|(label, adoption)| (Arc::clone(label), Some(*adoption))));
    let (mut y, mut units, mut periods) = (vec![], vec![], vec![]);
    for (label, adoption) in &labels {
        let unit_level = 3.0 * rng.uniform();
        for (t, common) in factor.iter().enumerate() {
            let period = t as i64 + 1;
            let effect = if adoption.is_some_and(|a| period >= a) { 2.0 } else { 0.0 };
            y.push(unit_level + common + effect + 0.5 * rng.normal());
            units.push(Arc::clone(label));
            periods.push(period);
        }
    }
    let data = TabularData::from_f64_columns([("y", y.as_slice())]).unwrap();
    let query = SyntheticControlQuery::new(OUTCOME, units, periods, Arc::<str>::from("t0"), 21)
        .with_staggered_adoptions(adoptions, 199);
    (data, query)
}

// ------------------------------------------------------------------ rollout intervals

#[test]
fn supported_staggered_rollout_interval_round_trips_and_refuses_forgery() {
    let label = "staggered rollout interval";
    let panel = staggered(11, 50);
    // Every measured estimator, comparison and scheme states its row on the simple aggregate.
    for (estimator, comparison, scheme) in [
        (
            RolloutDidEstimator::CallawaySantAnna,
            DidComparisonGroup::NeverTreated,
            DidResamplingScheme::ClusterBootstrap,
        ),
        (
            RolloutDidEstimator::SunAbraham,
            DidComparisonGroup::NeverTreated,
            DidResamplingScheme::ClusterMultiplier,
        ),
        (
            RolloutDidEstimator::Imputation,
            DidComparisonGroup::NotYetTreated,
            DidResamplingScheme::ClusterMultiplier,
        ),
    ] {
        let query = staggered_query(&panel, estimator, comparison, DidAggregation::Simple)
            .with_resampling(scheme, 199);
        let (_, result, _) = fit_panel(&panel.0, query, None, 3);
        assert!(section_supported(&result), "{estimator:?}");
    }
    let declared = |aggregation| {
        staggered_query(
            &panel,
            RolloutDidEstimator::CallawaySantAnna,
            DidComparisonGroup::NotYetTreated,
            aggregation,
        )
    };
    let query = declared(DidAggregation::Simple)
        .with_resampling(DidResamplingScheme::ClusterMultiplier, 199);
    let (prepared, result, ctx) = fit_panel(&panel.0, query, None, 3);
    assert!(section_supported(&result), "{label}");
    let rollout = result.quasi_rollout().unwrap();
    let overall = rollout.overall().unwrap();
    assert!((overall.estimate - 3.5).abs() < 0.6, "{label}: {}", overall.estimate);
    // The interval is centred and at least as wide as the normal interval of its error.
    let [lower, upper] = overall.interval.unwrap();
    let se = overall.standard_error.unwrap();
    assert!(((upper - overall.estimate) - (overall.estimate - lower)).abs() < 1e-9);
    assert!(upper - overall.estimate >= 1.959_963_984_540_054 * se);
    retained_graphless::assert_retained_graphless_program(label, &prepared, &result, &ctx);

    let (names, body) = artifact(&prepared, &result, &ctx, label);
    assert!(body.quasi_rollout.as_ref().unwrap().supported, "{label}");
    // Too few never-treated units, too few clusters, and a refit count below the row.
    assert!(forged_section(&names, &body, "never_treated_units", 20.0).contains(NO_ROW));
    assert!(forged_section(&names, &body, "clusters", 100.0).contains(NO_ROW));
    // A count restated above the query's still meets the row; the loader recounts it.
    assert!(forged_section(&names, &body, "never_treated_units", 4000.0).contains(RECOUNTED));
    assert!(forged_section(&names, &body, "treated_units", 4000.0).contains(RECOUNTED));
    let mut forged = body.clone();
    let section = forged.quasi_rollout.as_mut().unwrap();
    (section.replicates_ok, section.replicates_failed) = (100, 99);
    assert!(refusal(&names, &forged).contains(NO_ROW), "{label}: forged refits");
    // Another scheme than the one the query declared.
    let mut forged = body.clone();
    forged.quasi_rollout.as_mut().unwrap().uncertainty =
        "whole_cluster_bootstrap_refit_iqr_se_pointwise_and_sup_t_band".into();
    assert!(refusal(&names, &forged).contains(NO_ROW), "{label}: forged scheme");

    // Below the row, another aggregation and no resample run and state no support.
    let small = staggered(12, 20);
    let query = staggered_query(
        &small,
        RolloutDidEstimator::CallawaySantAnna,
        DidComparisonGroup::NotYetTreated,
        DidAggregation::Simple,
    )
    .with_resampling(DidResamplingScheme::ClusterMultiplier, 199);
    assert!(!section_supported(&fit_panel(&small.0, query, None, 3).1));
    let dynamic = declared(DidAggregation::Dynamic)
        .with_resampling(DidResamplingScheme::ClusterMultiplier, 199);
    assert!(!section_supported(&fit_panel(&panel.0, dynamic, None, 3).1));
    assert!(!section_supported(&fit_panel(&panel.0, declared(DidAggregation::Simple), None, 3).1));
}

#[test]
fn supported_staggered_rollout_posterior_round_trips_and_refuses_forgery() {
    let label = "staggered rollout posterior";
    let panel = staggered(21, 50);
    let query = || {
        staggered_query(
            &panel,
            RolloutDidEstimator::CallawaySantAnna,
            DidComparisonGroup::NotYetTreated,
            DidAggregation::Simple,
        )
    };
    let (prepared, result, ctx) = fit_panel(&panel.0, query(), bayes(None, 2_000), 4);
    assert!(states_posterior_row(&result), "{label}");
    let overall = result.quasi_rollout().unwrap().overall().unwrap().estimate;
    assert!((overall - 3.5).abs() < 0.6, "{label}: {overall}");
    retained_graphless::assert_retained_graphless_program(label, &prepared, &result, &ctx);

    let (names, body) = artifact(&prepared, &result, &ctx, label);
    assert!(body.diagnostics.iter().any(|diagnostic| diagnostic.code == POSTERIOR_ROW));
    assert!(forged_posterior_row(&names, &body, "blocks", "100").contains(NO_POSTERIOR_ROW));
    assert!(forged_posterior_row(&names, &body, "treated", "10").contains(NO_POSTERIOR_ROW));
    let other = "bayesian_bootstrap_sun_abraham_never_treated";
    assert!(forged_posterior_row(&names, &body, "method", other).contains(NO_POSTERIOR_ROW));

    // Fewer draws, and another comparison group, run and state no row.
    assert!(!states_posterior_row(&fit_panel(&panel.0, query(), bayes(None, 800), 4).1));
    let never = staggered_query(
        &panel,
        RolloutDidEstimator::CallawaySantAnna,
        DidComparisonGroup::NeverTreated,
        DidAggregation::Simple,
    );
    assert!(!states_posterior_row(&fit_panel(&panel.0, never, bayes(None, 2_000), 4).1));
}

#[test]
fn supported_covariate_adjusted_interval_round_trips_and_refuses_forgery() {
    let label = "covariate-adjusted two-period interval";
    for nuisance in
        [DidNuisanceMethod::OutcomeRegression, DidNuisanceMethod::InverseProbabilityWeighting]
    {
        let (data, query) = adjusted(31, 260, nuisance);
        let query = query.with_resampling(DidResamplingScheme::ClusterMultiplier, 199);
        assert!(section_supported(&fit_panel(&data, query, None, 5).1), "{nuisance:?}");
    }
    let (data, query) = adjusted(31, 260, DidNuisanceMethod::DoublyRobust);
    let multiplier = query.clone().with_resampling(DidResamplingScheme::ClusterMultiplier, 199);
    let (prepared, result, ctx) = fit_panel(&data, multiplier, None, 5);
    assert!(section_supported(&result), "{label}");
    let rollout = result.quasi_rollout().unwrap();
    let estimate = rollout.overall().unwrap().estimate;
    assert!((estimate - 2.0).abs() < 0.5, "{label}: {estimate}");
    let overlap = rollout
        .diagnostics
        .iter()
        .find(|(name, _)| name.as_ref() == "comparison_propensity_max")
        .map(|(_, value)| *value)
        .unwrap();
    assert!(overlap > 0.5 && overlap <= 0.9, "{label}: fitted overlap {overlap}");
    retained_graphless::assert_retained_graphless_program(label, &prepared, &result, &ctx);

    let (names, body) = artifact(&prepared, &result, &ctx, label);
    assert!(body.quasi_rollout.as_ref().unwrap().supported, "{label}");
    // Weaker overlap, and fewer treated units, than the row.
    assert!(forged_section(&names, &body, "comparison_propensity_max", 0.97).contains(NO_ROW));
    assert!(forged_section(&names, &body, "treated_units", 40.0).contains(NO_ROW));

    // The bootstrap scheme was not measured, and a smaller panel is below the row.
    let bootstrap = query.with_resampling(DidResamplingScheme::ClusterBootstrap, 199);
    assert!(!section_supported(&fit_panel(&data, bootstrap, None, 5).1));
    let (small, query) = adjusted(32, 120, DidNuisanceMethod::DoublyRobust);
    let query = query.with_resampling(DidResamplingScheme::ClusterMultiplier, 199);
    assert!(!section_supported(&fit_panel(&small, query, None, 5).1));
}

#[test]
fn supported_augmented_panel_interval_round_trips_and_refuses_forgery() {
    let label = "augmented panel bootstrap interval";
    let (data, query) = augmented(41, 260);
    let bootstrap = query.clone().with_resampling(DidResamplingScheme::ClusterBootstrap, 199);
    let (prepared, result, ctx) = fit_panel(&data, bootstrap, None, 6);
    // The row is stated in a coded diagnostic; the matrix support status stays absent.
    assert!(result.states_section_row() && result.support_status().is_none(), "{label}");
    let fit = result.panel_did().unwrap();
    assert!((fit.effect - 1.5).abs() < 0.2, "{label}: {}", fit.effect);
    assert!(fit.interval_95.is_some(), "{label}");
    retained_graphless::assert_retained_graphless_program(label, &prepared, &result, &ctx);

    let (names, body) = artifact(&prepared, &result, &ctx, label);
    let stored = body.panel_did.as_ref().unwrap();
    assert_eq!(stored.graphless_support_status, None, "{label}");
    let augmented_row = antecedent_io::graphless_support::AUGMENTED_DID_ROW_CODE;
    let claim = body
        .diagnostics
        .iter()
        .find(|diagnostic| diagnostic.code == augmented_row)
        .cloned()
        .expect("the exported body states the row");
    // A supplied propensity outside the row's range cannot carry the statement.
    let mut forged = body.clone();
    let recorded = forged.panel_did.as_mut().unwrap().augmented.as_mut().unwrap();
    recorded.0 = 0.1;
    assert!(refusal(&names, &forged).contains("states a graphless support row"), "{label}");
    // The section's own status is never a statement on an augmented panel.
    let mut flagged = body.clone();
    flagged.panel_did.as_mut().unwrap().graphless_support_status = Some("supported".to_owned());
    assert!(refusal(&names, &flagged).contains("invalid panel DiD payload"), "{label}");
    // The row was measured under the declaration that the predictions were not fitted on the
    // analysed rows: a section without it cannot carry the statement, and a fit without it
    // states none.
    let mut undeclared = body.clone();
    undeclared.panel_did.as_mut().unwrap().augmented.as_mut().unwrap().3 = false;
    antecedent_io::encode_analysis_result_artifact(&undeclared, names.clone(), "undeclared")
        .expect_err("a section without the declaration cannot state the row");
    let mut in_sample = query.clone().with_resampling(DidResamplingScheme::ClusterBootstrap, 199);
    in_sample.augmented.as_mut().unwrap().predictions_cross_fitted = false;
    assert!(!fit_panel(&data, in_sample, None, 6).1.states_section_row(), "{label}");

    // The multiplier scheme, no resample, and a smaller panel run and state no support.
    let multiplier = query.clone().with_resampling(DidResamplingScheme::ClusterMultiplier, 199);
    assert!(!fit_panel(&data, multiplier, None, 6).1.states_section_row());
    assert!(!fit_panel(&data, query, None, 6).1.states_section_row());
    let (small, query) = augmented(42, 120);
    let query = query.with_resampling(DidResamplingScheme::ClusterBootstrap, 199);
    let (prepared, result, ctx) = fit_panel(&small, query, None, 6);
    assert!(!result.states_section_row() && result.support_status().is_none(), "{label}");
    // The supported fit's statement carried onto the smaller panel is refused.
    let (names, mut stated) = artifact(&prepared, &result, &ctx, label);
    antecedent_io::encode_analysis_result_artifact(&stated, names.clone(), "no-statement")
        .expect("the unsupported result round trips");
    stated.diagnostics.push(claim);
    assert!(refusal(&names, &stated).contains("states a graphless support row"), "{label}");
}

#[test]
fn supported_two_period_posterior_round_trips_and_refuses_forgery() {
    let label = "two-period posterior";
    let (data, query) = two_period(51, 30);
    let (prepared, result, ctx) = fit_panel(&data, query.clone(), bayes(Some(100.0), 2_000), 7);
    assert!(states_posterior_row(&result), "{label}");
    let posterior = result.posterior.as_ref().unwrap();
    let column = posterior.effect_column().unwrap();
    assert!((posterior.summaries.mean[column] - 2.0).abs() < 0.5, "{label}");
    retained_graphless::assert_retained_graphless_program(label, &prepared, &result, &ctx);

    let (names, body) = artifact(&prepared, &result, &ctx, label);
    assert!(body.diagnostics.iter().any(|diagnostic| diagnostic.code == POSTERIOR_ROW));
    assert!(forged_posterior_row(&names, &body, "treated", "5").contains(NO_POSTERIOR_ROW));
    let other = "normal_inverse_gamma_change_score_posterior_prior_scale_10";
    assert!(forged_posterior_row(&names, &body, "method", other).contains(NO_POSTERIOR_ROW));

    // Another prior scale, fewer draws and fewer subjects run and state no row.
    assert!(!states_posterior_row(&fit_panel(&data, query.clone(), bayes(Some(10.0), 2_000), 7).1));
    assert!(!states_posterior_row(&fit_panel(&data, query, bayes(Some(100.0), 500), 7).1));
    let (small, query) = two_period(52, 12);
    assert!(!states_posterior_row(&fit_panel(&small, query, bayes(Some(100.0), 2_000), 7).1));
}

// ------------------------------------------------------------------ synthetic control

#[test]
fn supported_conformal_sets_round_trip_and_refuse_forgery() {
    let label = "conformal synthetic control";
    let (data, query) = one_treated(61, 39);
    let (prepared, result, ctx) = fit_synthetic(&data, query.with_conformal_intervals(), None, 8);
    assert!(section_supported(&result), "{label}");
    let rollout = result.quasi_rollout().unwrap();
    assert!(rollout.effects.iter().all(|effect| effect.interval.is_some()), "{label}");
    assert!((rollout.overall().unwrap().estimate - 3.0).abs() < 0.8, "{label}");
    retained_graphless::assert_retained_graphless_program(label, &prepared, &result, &ctx);

    let (names, body) = artifact(&prepared, &result, &ctx, label);
    assert!(body.quasi_rollout.as_ref().unwrap().supported, "{label}");
    // Fewer pre-periods or donors than the row, and a set reported unbounded.
    assert!(forged_section(&names, &body, "pre_periods", 20.0).contains(NO_ROW));
    assert!(forged_section(&names, &body, "donors", 4.0).contains(NO_ROW));
    assert!(forged_section(&names, &body, "unbounded_intervals", 1.0).contains(NO_ROW));
    // Counts restated above the query's still meet the row; the loader recounts them.
    assert!(forged_section(&names, &body, "pre_periods", 159.0).contains(RECOUNTED));
    assert!(forged_section(&names, &body, "donors", 40.0).contains(RECOUNTED));

    // A shorter pre-period runs and states no support.
    let (short, query) = one_treated(62, 25);
    assert!(!section_supported(
        &fit_synthetic(&short, query.with_conformal_intervals(), None, 8).1
    ));
}

#[test]
fn supported_placebo_interval_round_trips_and_refuses_forgery() {
    let label = "multi-unit placebo interval";
    let (data, query) = staggered_synthetic(71, 16);
    let (prepared, result, ctx) = fit_synthetic(&data, query, None, 9);
    assert!(section_supported(&result), "{label}");
    let overall = result.quasi_rollout().unwrap().overall().unwrap();
    assert!((overall.estimate - 2.0).abs() < 0.8, "{label}: {}", overall.estimate);
    assert!(overall.interval.is_some(), "{label}");
    retained_graphless::assert_retained_graphless_program(label, &prepared, &result, &ctx);

    let (names, body) = artifact(&prepared, &result, &ctx, label);
    assert!(body.quasi_rollout.as_ref().unwrap().supported, "{label}");
    // Fewer donors, fewer adopting units and fewer pre-adoption periods than the row.
    assert!(forged_section(&names, &body, "never_treated_donors", 6.0).contains(NO_ROW));
    assert!(forged_section(&names, &body, "adopting_units", 2.0).contains(NO_ROW));
    assert!(forged_section(&names, &body, "fewest_pre_periods", 10.0).contains(NO_ROW));
    assert!(forged_section(&names, &body, "never_treated_donors", 60.0).contains(RECOUNTED));
    assert!(forged_section(&names, &body, "fewest_pre_periods", 25.0).contains(RECOUNTED));

    // Fewer donors run and state no support.
    let (small, query) = staggered_synthetic(72, 8);
    assert!(!section_supported(&fit_synthetic(&small, query, None, 9).1));
}

#[test]
fn supported_synthetic_posterior_round_trips_and_refuses_forgery() {
    let label = "synthetic control posterior";
    let (data, query) = one_treated(81, 30);
    let (prepared, result, ctx) = fit_synthetic(&data, query.clone(), bayes(None, 2_000), 10);
    assert!(states_posterior_row(&result), "{label}");
    let posterior = result.posterior.as_ref().unwrap();
    let column = posterior.effect_column().unwrap();
    assert!((posterior.summaries.mean[column] - 3.0).abs() < 0.8, "{label}");
    retained_graphless::assert_retained_graphless_program(label, &prepared, &result, &ctx);

    let (names, body) = artifact(&prepared, &result, &ctx, label);
    assert!(body.diagnostics.iter().any(|diagnostic| diagnostic.code == POSTERIOR_ROW));
    assert!(forged_posterior_row(&names, &body, "rows", "12").contains(NO_POSTERIOR_ROW));
    assert!(forged_posterior_row(&names, &body, "blocks", "3").contains(NO_POSTERIOR_ROW));

    // Fewer draws, and a shorter pre-period, run and state no row; the supported fit's
    // statement beside the smaller posterior is refused, because the loader counts the draws
    // in the stored posterior.
    let (few_prepared, few, few_ctx) = fit_synthetic(&data, query, bayes(None, 800), 10);
    assert!(!states_posterior_row(&few));
    let (_, few_body) = artifact(&few_prepared, &few, &few_ctx, label);
    let mut thinned = body.clone();
    thinned.posterior_artifact = few_body.posterior_artifact;
    antecedent_io::encode_analysis_result_artifact(&thinned, names.clone(), "thinned-posterior")
        .expect_err("a row stated on fewer draws than it was measured with is refused");
    let (short, query) = one_treated(82, 20);
    assert!(!states_posterior_row(&fit_synthetic(&short, query, bayes(None, 2_000), 10).1));
}

#[test]
fn supported_predictive_path_round_trips_and_refuses_forgery() {
    let label = "synthetic control predictive path";
    let (data, query) = one_treated(91, 30);
    let query = query.with_conformal_intervals();
    let (prepared, result, ctx) = fit_synthetic(&data, query, bayes(None, 2_000), 11);
    assert!(states_posterior_row(&result), "{label}");
    let rollout = result.quasi_rollout().unwrap();
    assert_eq!(rollout.design.as_ref(), "posterior_predictive_synthetic_control");
    // A design posterior states its row in the diagnostic, never on the section.
    assert!(!rollout.supported, "{label}");
    assert!((rollout.overall().unwrap().estimate - 3.0).abs() < 0.8, "{label}");
    retained_graphless::assert_retained_graphless_program(label, &prepared, &result, &ctx);

    let (names, body) = artifact(&prepared, &result, &ctx, label);
    let stated = body.diagnostics.iter().find(|diagnostic| diagnostic.code == POSTERIOR_ROW);
    let claim = stated
        .and_then(|row| row.fields.iter().find(|(key, _)| key == "inference_claim"))
        .map(|(_, value)| value.as_str());
    assert_eq!(claim, Some("per_period_and_average_95_posterior_predictive_intervals"));
    assert!(forged_posterior_row(&names, &body, "rows", "12").contains(NO_POSTERIOR_ROW));
    assert!(
        forged_posterior_row(&names, &body, "all_reported_intervals", "false")
            .contains(NO_POSTERIOR_ROW)
    );

    // A shorter pre-period runs and states no row.
    let (short, query) = one_treated(92, 20);
    let query = query.with_conformal_intervals();
    assert!(!states_posterior_row(&fit_synthetic(&short, query, bayes(None, 2_000), 11).1));
}
