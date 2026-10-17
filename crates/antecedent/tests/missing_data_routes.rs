//! Declared missing-data handling on randomized, panel, temporal and adjustment routes:
//! refusal by default, known-truth execution under each policy, and row accounting.
#![allow(
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::too_many_lines,
    reason = "integration test with deterministic fixtures and exact accounting counts"
)]

use std::sync::Arc;

use antecedent::{
    BayesianConfig, InferenceMode, RandomizedEffectQuery, RefuteSuite, Study, StudyResult,
};
use antecedent_core::{
    CausalQuery, ExecutionContext, Lag, MediationContrast, MediationQuery, PanelDidQuery,
    TemporalEffectQuery, TemporalPolicy, VariableId,
};
use antecedent_data::{MissingDeclaration, MissingPolicy, TabularData, TimeSeriesData};
use antecedent_graph::{TemporalDag, ensure_lagged};

/// `SplitMix64` uniform stream.
struct Stream(u64);

impl Stream {
    fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
    }

    fn normal(&mut self) -> f64 {
        let (u, v) = (self.next().max(1e-300), self.next());
        (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
    }
}

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(0x4D15_5100)
}

fn declare(policy: MissingPolicy) -> MissingDeclaration {
    MissingDeclaration::new(policy)
}

// ---------------------------------------------------------------- randomized trial

const TRUE_EFFECT: f64 = 0.3;

struct Trial {
    outcome: Vec<f64>,
    baseline: Vec<f64>,
    assignment: Vec<bool>,
    responded: Vec<bool>,
}

/// Bernoulli(1/2) trial whose responding falls with a baseline that also moves the outcome.
fn trial(n: usize, seed: u64) -> Trial {
    let mut rng = Stream(seed);
    let mut t = Trial { outcome: vec![], baseline: vec![], assignment: vec![], responded: vec![] };
    for i in 0..n {
        let arm = i % 2 == 1;
        let x = rng.normal().clamp(-2.0, 2.0);
        let y = 1.0 + TRUE_EFFECT * f64::from(u8::from(arm)) + 2.0 * x + rng.normal();
        // Baseline X ~ N(0,1) clipped to [-2, 2] (so no fitted response probability can reach
        // the positivity floor), outcome slope 2 and response logit 0.8 - b_a X with b = 0.8
        // (control) and 1.6 (treated). By the probit approximation of the logistic,
        // E[X | responded, arm] = -c/sqrt(1+c^2) * phi(m)/Phi(m), c = b/1.7, m = 0.47/sqrt(1+c^2):
        // -0.233 (control) and -0.406 (treated), so the complete-case effect is biased by
        // 2 * (-0.406 + 0.233) = -0.35, far outside any sampling error at the sizes used here.
        let eta = 0.8 - (0.8 + 0.8 * f64::from(u8::from(arm))) * x;
        t.responded.push(rng.next() < 1.0 / (1.0 + (-eta).exp()));
        t.outcome.push(y);
        t.baseline.push(x);
        t.assignment.push(arm);
    }
    t
}

impl Trial {
    fn table(&self) -> TabularData {
        let observed: Vec<f64> = self
            .outcome
            .iter()
            .zip(&self.responded)
            .map(|(y, seen)| if *seen { *y } else { f64::NAN })
            .collect();
        TabularData::from_f64_columns([
            ("outcome", observed.as_slice()),
            ("baseline", self.baseline.as_slice()),
        ])
        .unwrap()
    }

    fn query(&self) -> RandomizedEffectQuery {
        let n = self.outcome.len();
        let units: Vec<Arc<str>> = (0..n).map(|i| Arc::from(format!("p{i}"))).collect();
        RandomizedEffectQuery::bernoulli_itt(
            VariableId::from_raw(0),
            self.assignment.clone(),
            vec![0.5; n],
            units.clone(),
            units,
            ("control", "treated"),
        )
    }
}

fn run_trial(
    t: &Trial,
    declaration: Option<MissingDeclaration>,
) -> Result<StudyResult, antecedent::CausalError> {
    let data = t.table();
    let mut builder = Study::tabular(data.clone()).query(t.query());
    if let Some(declaration) = declaration {
        builder = builder.missing(declaration);
    }
    let prepared = builder.build()?.prepare(&ctx())?;
    prepared.estimate(&data, &ctx())
}

#[test]
fn a_randomized_trial_with_missing_outcomes_refuses_without_a_policy() {
    let t = trial(400, 1);
    let first = t.responded.iter().position(|seen| !seen).unwrap();
    let error = run_trial(&t, None).unwrap_err();
    let text = error.to_string();
    assert_eq!(error.reason_code(), Some("missing_policy_required"), "{text}");
    assert!(text.contains(&format!("column `outcome` at row {first}")), "{text}");
    assert!(text.contains("complete_case, inverse_probability"), "{text}");
    let explicit = run_trial(&t, Some(declare(MissingPolicy::ErrorOnMissing))).unwrap_err();
    assert_eq!(explicit.reason_code(), Some("missing_policy_required"));
}

#[test]
fn complete_case_averages_responders_and_keeps_every_assignment() {
    let t = trial(2_000, 2);
    let result = run_trial(&t, Some(declare(MissingPolicy::CompleteCase))).unwrap();
    let fit = result.randomized_effect().unwrap();
    let mean = |arm: bool| {
        let v: Vec<f64> = (0..t.outcome.len())
            .filter(|&i| t.assignment[i] == arm && t.responded[i])
            .map(|i| t.outcome[i])
            .collect();
        v.iter().sum::<f64>() / v.len() as f64
    };
    assert!((fit.effect - (mean(true) - mean(false))).abs() < 1e-12);
    assert_eq!(fit.interval_95, None, "no interval is published for a handled missingness");
    assert_eq!(fit.treatment_units, 1_000);
    assert_eq!(fit.control_units, 1_000);

    let report = result.missing_data.as_ref().unwrap();
    assert!(report.declared);
    assert_eq!(report.policy(), MissingPolicy::CompleteCase);
    assert!(report.assumption.contains("missing completely at random within arm"));
    let accounting = &report.accounting;
    let responders = t.responded.iter().filter(|r| **r).count();
    assert_eq!((accounting.rows_total, accounting.rows_kept), (2_000, responders));
    assert_eq!(accounting.rows_in_denominator, responders);
    let dropped: Vec<u64> = (0..2_000).filter(|&i| !t.responded[i]).map(|i| i as u64).collect();
    assert_eq!(accounting.dropped_rows.as_ref(), dropped.as_slice());
    let arms: Vec<_> = accounting.groups.iter().map(|g| (g.label.to_string(), g.total)).collect();
    assert_eq!(arms, [("control".to_owned(), 1_000), ("treated".to_owned(), 1_000)]);
}

#[test]
fn inverse_probability_weighting_recovers_the_effect_that_complete_case_misses() {
    let t = trial(30_000, 3);
    let weighted = run_trial(
        &t,
        Some(
            declare(MissingPolicy::InverseProbabilityOfObservation)
                .with_covariates(vec![VariableId::from_raw(1)]),
        ),
    )
    .unwrap();
    let fit = weighted.randomized_effect().unwrap();
    let se = fit.variance_upper_bound.sqrt();
    assert!((fit.effect - TRUE_EFFECT).abs() < 4.0 * se, "weighted {} se {se}", fit.effect);
    assert_eq!(fit.interval_95, None);
    assert_eq!(fit.uncertainty.as_ref(), "responder_weighted_influence_variance_no_interval");

    let naive = run_trial(&t, Some(declare(MissingPolicy::CompleteCase))).unwrap();
    assert!((naive.randomized_effect().unwrap().effect - TRUE_EFFECT).abs() > 0.2);

    // Every assigned participant stays in its arm's denominator; the assignment is unchanged.
    let report = weighted.missing_data.as_ref().unwrap();
    assert!(report.assumption.contains("missing at random given arm"));
    assert_eq!(&*report.covariates, [Arc::<str>::from("baseline")]);
    let accounting = &report.accounting;
    assert_eq!(accounting.rows_in_denominator, 30_000);
    assert_eq!(accounting.rows_total - accounting.rows_kept, accounting.rows_dropped());
    assert_eq!(accounting.groups.iter().map(|g| g.total).sum::<usize>(), 30_000);
    assert_eq!(fit.control_units + fit.treatment_units, 30_000);
}

#[test]
fn a_complete_trial_is_unchanged_by_the_policy_and_still_reports_zero_dropped() {
    let mut t = trial(1_000, 4);
    t.responded.iter_mut().for_each(|seen| *seen = true);
    let plain = run_trial(&t, None).unwrap();
    let declared = run_trial(
        &t,
        Some(
            declare(MissingPolicy::InverseProbabilityOfObservation)
                .with_covariates(vec![VariableId::from_raw(1)]),
        ),
    )
    .unwrap();
    assert_eq!(
        plain.randomized_effect().unwrap().effect,
        declared.randomized_effect().unwrap().effect
    );
    let report = plain.missing_data.as_ref().unwrap();
    assert!(!report.declared);
    assert_eq!(report.accounting.rows_dropped(), 0);
}

#[test]
fn incomplete_observation_covariates_and_foreign_policies_are_refused() {
    let mut t = trial(600, 5);
    t.baseline[3] = f64::NAN;
    let error = run_trial(
        &t,
        Some(
            declare(MissingPolicy::InverseProbabilityOfObservation)
                .with_covariates(vec![VariableId::from_raw(1)]),
        ),
    )
    .unwrap_err();
    assert!(error.to_string().contains("missing_data.covariates_incomplete"), "{error}");

    let t = trial(600, 6);
    let build =
        Study::tabular(t.table()).query(t.query()).missing(declare(MissingPolicy::BalancedPanel));
    let error = build.build().unwrap_err();
    assert_eq!(error.reason_code(), Some("missing_policy_not_applicable"), "{error}");
}

// ---------------------------------------------------------------- panel difference-in-differences

/// Two-period panel, exactly additive: unit level + period level + `EFFECT` for treated post.
const DID_EFFECT: f64 = 0.5;

fn panel(units: usize, missing: &[(usize, bool)]) -> (TabularData, PanelDidQuery) {
    let (mut y, mut treated, mut post, mut subjects, mut clusters) =
        (vec![], vec![], vec![], vec![], vec![]);
    for unit in 0..units {
        let is_treated = unit % 2 == 0;
        for after in [false, true] {
            let level = 0.1 * (unit % 7) as f64 + 0.3 * f64::from(u8::from(after));
            let value = level + if is_treated && after { DID_EFFECT } else { 0.0 };
            y.push(if missing.contains(&(unit, after)) { f64::NAN } else { value });
            treated.push(is_treated);
            post.push(after);
            subjects.push(Arc::<str>::from(format!("u{unit}")));
            clusters.push(Arc::<str>::from(format!("c{unit}")));
        }
    }
    let data = TabularData::from_f64_columns([("y", y.as_slice())]).unwrap();
    let query = PanelDidQuery::new(VariableId::from_raw(0), treated, post, subjects, clusters);
    (data, query)
}

fn run_panel(
    data: &TabularData,
    query: PanelDidQuery,
    declaration: Option<MissingDeclaration>,
) -> Result<StudyResult, antecedent::CausalError> {
    let mut builder = Study::tabular(data.clone()).query(query);
    if let Some(declaration) = declaration {
        builder = builder.missing(declaration);
    }
    builder.build()?.prepare(&ctx())?.estimate(data, &ctx())
}

#[test]
fn panel_did_refuses_missing_cells_and_never_deletes_single_unit_periods() {
    let (data, query) = panel(60, &[(5, true)]);
    let error = run_panel(&data, query.clone(), None).unwrap_err();
    assert_eq!(error.reason_code(), Some("missing_policy_required"), "{error}");
    assert!(error.to_string().contains("column `y` at row 11"), "{error}");
    assert!(error.to_string().contains("balanced_panel, available_case"), "{error}");
    let built = Study::tabular(data.clone())
        .query(query)
        .missing(declare(MissingPolicy::CompleteCase))
        .build()
        .unwrap_err();
    assert_eq!(built.reason_code(), Some("missing_policy_not_applicable"), "{built}");
    assert!(built.to_string().contains("unbalance the panel"), "{built}");
}

#[test]
fn balanced_panel_drops_whole_units_and_recovers_the_effect() {
    let missing = [(5, true), (8, false), (31, true)];
    let (data, query) = panel(60, &missing);
    let result = run_panel(&data, query, Some(declare(MissingPolicy::BalancedPanel))).unwrap();
    let fit = result.panel_did().unwrap();
    assert!((fit.effect - DID_EFFECT).abs() < 1e-12, "{}", fit.effect);
    assert_eq!(fit.treated_subjects + fit.comparison_subjects, 57);
    assert_eq!(fit.interval_95, None, "a restricted panel publishes no interval");
    let accounting = &result.missing_data.as_ref().unwrap().accounting;
    assert_eq!((accounting.rows_total, accounting.rows_kept), (120, 114));
    let units = accounting.units.unwrap();
    assert_eq!((units.total, units.dropped), (60, 3));
    // Both rows of each affected unit are dropped, with the unit-incomplete reason for the
    // row that was itself observed.
    assert_eq!(accounting.dropped_rows.as_ref(), [10, 11, 16, 17, 62, 63]);
    assert_eq!(accounting.reasons.len(), 2);
}

#[test]
fn staggered_available_case_uses_each_contrasts_observed_units_without_relabelling() {
    // Three periods, cohort 2 adopts at period 2; additive structure so every observed pair
    // identifies the same effect, and pooling mismatched periods would not.
    let units = 90;
    let (mut y, mut subjects, mut clusters, mut periods, mut cohorts) =
        (vec![], vec![], vec![], vec![], vec![]);
    for unit in 0..units {
        let cohort: i64 = if unit % 3 == 0 { 0 } else { 2 };
        for period in 1..=3_i64 {
            let level = 0.2 * f64::from(unit % 5) + 0.4 * period as f64;
            let value = level + if cohort == 2 && period >= 2 { DID_EFFECT } else { 0.0 };
            // Unit 4 lacks period 3; unit 10 lacks the baseline; unit 21 lacks period 2.
            let hole = matches!((unit, period), (4, 3) | (10, 1) | (21, 2));
            y.push(if hole { f64::NAN } else { value });
            subjects.push(Arc::<str>::from(format!("u{unit}")));
            clusters.push(Arc::<str>::from(format!("u{unit}")));
            periods.push(period);
            cohorts.push(cohort);
        }
    }
    let data = TabularData::from_f64_columns([("y", y.as_slice())]).unwrap();
    let query = PanelDidQuery::staggered_event_study(
        VariableId::from_raw(0),
        subjects,
        clusters,
        periods,
        cohorts,
    );
    let error = run_panel(&data, query.clone(), None).unwrap_err();
    assert_eq!(error.reason_code(), Some("missing_policy_required"));
    let result =
        run_panel(&data, query.clone(), Some(declare(MissingPolicy::AvailableCase))).unwrap();
    let fit = result.panel_did().unwrap();
    assert!(!fit.event_time_effects.is_empty());
    for effect in fit.event_time_effects.iter().filter(|e| e.event_time >= 0) {
        assert!((effect.effect - DID_EFFECT).abs() < 1e-12, "{effect:?}");
    }
    for effect in fit.event_time_effects.iter().filter(|e| e.event_time < -1) {
        assert!(effect.effect.abs() < 1e-12, "{effect:?}");
    }
    // Unit 10 lacks the baseline so it is absent from both contrasts; the period-3 contrast
    // also lost unit 4 and the period-2 contrast unit 21: 90 - 2 = 88 units each, where a
    // balanced restriction would keep only 87.
    let counts: Vec<_> = fit
        .event_time_effects
        .iter()
        .map(|e| (e.period, e.treated_subjects + e.comparison_subjects))
        .collect();
    assert!(counts.contains(&(3, 88)) && counts.contains(&(2, 88)), "{counts:?}");
    assert!(fit.event_time_intervals_95.iter().all(Option::is_none));
    let accounting = &result.missing_data.as_ref().unwrap().accounting;
    assert_eq!((accounting.rows_total, accounting.rows_kept), (270, 267));
    assert_eq!(accounting.units.unwrap().dropped, 0, "no unit is relabelled or removed");
    // Balanced restriction removes the three incomplete units instead.
    let balanced = run_panel(&data, query, Some(declare(MissingPolicy::BalancedPanel))).unwrap();
    assert_eq!(balanced.missing_data.as_ref().unwrap().accounting.units.unwrap().dropped, 3);
    assert!((balanced.panel_did().unwrap().effect - DID_EFFECT).abs() < 1e-12);
}

// ---------------------------------------------------------------- temporal

fn temporal_graph() -> TemporalDag {
    let mut graph = TemporalDag::empty();
    let x1 = ensure_lagged(&mut graph, VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
    let y0 = ensure_lagged(&mut graph, VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    graph.insert_directed(x1, y0).unwrap();
    graph
}

fn pulse() -> CausalQuery {
    CausalQuery::TemporalEffect(
        TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0)
            .with_horizon_steps(1)
            .with_policy(TemporalPolicy::pulse(-1)),
    )
}

/// `y_t = 2 x_{t-1} + 0.3 sin(7t)`, with `x` missing at the given times.
fn series(n: usize, gaps: &[usize]) -> (TimeSeriesData, Vec<f64>, Vec<f64>) {
    let x: Vec<f64> =
        (0..n).map(|t| (0.37 * t as f64).sin() + 0.3 * (1.9 * t as f64).cos()).collect();
    let y: Vec<f64> = (0..n)
        .map(|t| if t == 0 { 0.0 } else { 2.0 * x[t - 1] + 0.3 * (7.0 * t as f64).sin() })
        .collect();
    let observed: Vec<f64> =
        x.iter().enumerate().map(|(t, v)| if gaps.contains(&t) { f64::NAN } else { *v }).collect();
    let data =
        TimeSeriesData::from_f64_columns([("x", observed.as_slice()), ("y", y.as_slice())], 1)
            .unwrap();
    (data, x, y)
}

fn run_series(
    data: &TimeSeriesData,
    declaration: Option<MissingDeclaration>,
    bootstrap: u32,
) -> Result<StudyResult, antecedent::CausalError> {
    let mut builder = Study::series(data.clone())
        .query(pulse())
        .graph(temporal_graph())
        .refute(RefuteSuite::None)
        .bootstrap_replicates(bootstrap);
    if let Some(declaration) = declaration {
        builder = builder.missing(declaration);
    }
    builder.build()?.prepare(&ctx())?.estimate_retained(&ctx())
}

#[test]
fn a_series_with_gaps_refuses_without_a_policy_and_without_a_point_estimate_path() {
    let (data, _, _) = series(300, &[100, 101, 202]);
    // No declaration: the error this fit has always raised for an incomplete series.
    let error = run_series(&data, None, 0).unwrap_err();
    assert!(
        matches!(
            error,
            antecedent::CausalError::Data(antecedent_data::DataError::IncompleteSeries {
                id: Some(_),
                ..
            })
        ),
        "{error:?}"
    );
    assert!(error.to_string().contains("missing=\"complete_case\""), "{error}");
    // A declared `error` is the explicit refusal naming the first missing cell.
    let declared = run_series(&data, Some(declare(MissingPolicy::ErrorOnMissing)), 0).unwrap_err();
    assert_eq!(declared.reason_code(), Some("missing_policy_required"), "{declared}");
    assert!(declared.to_string().contains("column `x` at row 100"), "{declared}");
}

#[test]
fn gaps_drop_only_the_windows_that_read_them_and_never_compress_time() {
    let gaps = [100, 101, 202];
    let n = 300;
    let (data, x, y) = series(n, &gaps);
    let result = run_series(&data, Some(declare(MissingPolicy::CompleteCase)), 0).unwrap();

    // Window t regresses y_t on x_{t-1}: it is lost when x_{t-1} is missing.
    let kept: Vec<usize> = (1..n).filter(|t| !gaps.contains(&(t - 1))).collect();
    let (mx, my) = (
        kept.iter().map(|&t| x[t - 1]).sum::<f64>() / kept.len() as f64,
        kept.iter().map(|&t| y[t]).sum::<f64>() / kept.len() as f64,
    );
    let slope = kept.iter().map(|&t| (x[t - 1] - mx) * (y[t] - my)).sum::<f64>()
        / kept.iter().map(|&t| (x[t - 1] - mx).powi(2)).sum::<f64>();
    let ate = result.estimate.as_effect().unwrap().ate;
    assert!((ate - slope).abs() < 1e-9, "ate {ate} vs independent slope {slope}");

    // Compressing the series (deleting the missing rows and re-lagging) pairs y_t with the
    // wrong x and gives a different number; the gap-preserving estimate does not.
    let compressed: Vec<usize> = (0..n).filter(|t| !gaps.contains(t)).collect();
    let pairs: Vec<(f64, f64)> = compressed.windows(2).map(|w| (x[w[0]], y[w[1]])).collect();
    let (cx, cy) = (
        pairs.iter().map(|p| p.0).sum::<f64>() / pairs.len() as f64,
        pairs.iter().map(|p| p.1).sum::<f64>() / pairs.len() as f64,
    );
    let wrong = pairs.iter().map(|p| (p.0 - cx) * (p.1 - cy)).sum::<f64>()
        / pairs.iter().map(|p| (p.0 - cx).powi(2)).sum::<f64>();
    assert!((wrong - slope).abs() > 1e-6);

    let report = result.missing_data.as_ref().unwrap();
    assert!(report.assumption.contains("original grid"));
    let accounting = &report.accounting;
    assert_eq!(accounting.rows_total, n - 1);
    assert_eq!(accounting.dropped_rows.as_ref(), [101, 102, 203]);
    assert_eq!(accounting.rows_kept, n - 1 - 3);
    let by_variable: Vec<_> =
        accounting.missing_by_variable.iter().map(|(v, c)| (v.to_string(), *c)).collect();
    assert_eq!(by_variable[0], ("x".to_owned(), 3));
    assert!(
        result.interval.as_ref().is_none_or(|b| b.method == antecedent_core::IntervalMethod::None)
    );
}

// ---------------------------------------------------------------- adjustment routes

#[test]
fn average_effect_keeps_complete_case_and_reports_what_it_dropped() {
    let n = 800;
    let mut rng = Stream(77);
    let z: Vec<f64> = (0..n).map(|_| rng.normal()).collect();
    let t: Vec<f64> =
        z.iter().map(|z| f64::from(u8::from(rng.next() < 1.0 / (1.0 + (-z).exp())))).collect();
    let y: Vec<f64> = (0..n).map(|i| 1.0 * t[i] + 0.7 * z[i] + rng.normal()).collect();
    let mut z_obs = z.clone();
    let holes: Vec<usize> = (0..n).filter(|i| i % 10 == 3).collect();
    for &i in &holes {
        z_obs[i] = f64::NAN;
    }
    let data = TabularData::from_f64_columns([
        ("t", t.as_slice()),
        ("y", y.as_slice()),
        ("z", z_obs.as_slice()),
    ])
    .unwrap();
    let query = antecedent_core::AverageEffectQuery::binary_ate(
        VariableId::from_raw(0),
        VariableId::from_raw(1),
    );
    let graph = {
        let mut g = antecedent_graph::Dag::empty();
        let ids: Vec<_> = (0..3)
            .map(|i| {
                g.add_node(antecedent_graph::NodeRef::Static(VariableId::from_raw(i))).unwrap()
            })
            .collect();
        g.insert_directed(ids[2], ids[0]).unwrap();
        g.insert_directed(ids[2], ids[1]).unwrap();
        g.insert_directed(ids[0], ids[1]).unwrap();
        g
    };
    let run = |declaration: Option<MissingDeclaration>| {
        let mut builder = Study::tabular(data.clone()).query(query.clone()).graph(graph.clone());
        if let Some(declaration) = declaration {
            builder = builder.missing(declaration);
        }
        builder.build()?.prepare(&ctx())?.estimate(&data, &ctx())
    };
    let default = run(None).unwrap();
    let explicit = run(Some(declare(MissingPolicy::CompleteCase))).unwrap();
    assert_eq!(
        default.estimate.as_effect().unwrap().ate,
        explicit.estimate.as_effect().unwrap().ate
    );
    let report = default.missing_data.as_ref().unwrap();
    assert!(!report.declared && explicit.missing_data.as_ref().unwrap().declared);
    assert_eq!(report.accounting.rows_total, n);
    assert_eq!(report.accounting.rows_dropped(), holes.len());
    let missing_z = report
        .accounting
        .missing_by_variable
        .iter()
        .find(|(v, _)| v.as_ref() == "z")
        .map(|(_, c)| *c);
    assert_eq!(missing_z, Some(holes.len()));
    // `error` is a declared refusal of the same table.
    let refused = run(Some(declare(MissingPolicy::ErrorOnMissing))).unwrap_err();
    assert_eq!(refused.reason_code(), Some("missing_policy_required"), "{refused}");
}

#[test]
fn the_claim_carries_the_policy_and_assumption_only_when_the_policy_is_not_the_default() {
    let t = trial(1_000, 8);
    let data = t.table();
    let claim_evidence = |declaration: MissingDeclaration| {
        let prepared = Study::tabular(data.clone())
            .query(t.query())
            .missing(declaration)
            .build()
            .unwrap()
            .prepare(&ctx())
            .unwrap();
        let result = prepared.estimate(&data, &ctx()).unwrap();
        let (_, claim) = prepared.execution_contract(&result, &ctx()).unwrap();
        claim.evidence.iter().map(ToString::to_string).collect::<Vec<_>>()
    };
    let evidence = claim_evidence(declare(MissingPolicy::CompleteCase));
    assert!(evidence.iter().any(|e| e == "missing_policy:complete_case"), "{evidence:?}");
    assert!(
        evidence.iter().any(|e| e.starts_with("missing_assumption:outcomes missing completely")),
        "{evidence:?}"
    );
    assert!(evidence.iter().any(|e| e.starts_with("missing_dropped:")));

    // A complete-data trial under the route default declares nothing and keeps three references.
    let mut full = trial(400, 9);
    full.responded.iter_mut().for_each(|r| *r = true);
    let data = full.table();
    let prepared =
        Study::tabular(data.clone()).query(full.query()).build().unwrap().prepare(&ctx()).unwrap();
    let result = prepared.estimate(&data, &ctx()).unwrap();
    let (_, claim) = prepared.execution_contract(&result, &ctx()).unwrap();
    assert_eq!(claim.evidence.len(), 3);
}

#[test]
fn the_block_bootstrap_on_a_series_with_gaps_runs_on_the_original_time_axis() {
    let (data, _, _) = series(400, &[120, 121, 122, 250]);
    let declaration = declare(MissingPolicy::CompleteCase);
    let result = run_series(&data, Some(declaration), 150).unwrap();
    let estimate = result.estimate.as_effect().unwrap();
    let se = estimate.se_bootstrap().expect("a bootstrap standard error");
    assert!(se.is_finite() && se > 0.0, "{se}");
    // The point estimate is the one the point-only fit gives; the bootstrap adds uncertainty only.
    let point = run_series(&data, Some(declare(MissingPolicy::CompleteCase)), 0).unwrap();
    assert_eq!(estimate.ate, point.estimate.as_effect().unwrap().ate);
}

#[test]
fn refuters_on_a_series_with_gaps_refit_through_the_intact_windows() {
    let (data, x, y) = series(400, &[120, 121, 250]);
    let result = Study::series(data)
        .query(pulse())
        .graph(temporal_graph())
        .refute(RefuteSuite::Cheap)
        .bootstrap_replicates(0)
        .missing(declare(MissingPolicy::CompleteCase))
        .build()
        .unwrap()
        .prepare(&ctx())
        .unwrap()
        .estimate_retained(&ctx())
        .unwrap();
    assert!(!result.refutations.is_empty(), "the cheap suite ran");
    // The point estimate is still the regression on the intact windows.
    let kept: Vec<usize> = (1..400).filter(|t| ![120, 121, 250].contains(&(t - 1))).collect();
    let (mx, my) = (
        kept.iter().map(|&t| x[t - 1]).sum::<f64>() / kept.len() as f64,
        kept.iter().map(|&t| y[t]).sum::<f64>() / kept.len() as f64,
    );
    let slope = kept.iter().map(|&t| (x[t - 1] - mx) * (y[t] - my)).sum::<f64>()
        / kept.iter().map(|&t| (x[t - 1] - mx).powi(2)).sum::<f64>();
    assert!((result.estimate.as_effect().unwrap().ate - slope).abs() < 1e-9);
}

/// `y_t = 2 t_{t-1} + 0.8 t_{t-2} + small deterministic term`, with `t` missing at `gaps`.
fn sustained_series(n: usize, gaps: &[usize]) -> (TimeSeriesData, Vec<f64>, Vec<f64>) {
    let t: Vec<f64> =
        (0..n).map(|i| (0.31 * i as f64).sin() + 0.4 * (1.7 * i as f64).cos()).collect();
    let y: Vec<f64> = (0..n)
        .map(|i| {
            if i < 2 {
                0.0
            } else {
                2.0 * t[i - 1] + 0.8 * t[i - 2] + 0.05 * (5.3 * i as f64).sin()
            }
        })
        .collect();
    let observed: Vec<f64> =
        t.iter().enumerate().map(|(i, v)| if gaps.contains(&i) { f64::NAN } else { *v }).collect();
    let data =
        TimeSeriesData::from_f64_columns([("t", observed.as_slice()), ("y", y.as_slice())], 1)
            .unwrap();
    (data, t, y)
}

#[test]
fn sustained_windows_on_a_series_with_gaps_match_the_intact_window_regression() {
    let gaps = [90, 91, 200];
    let n = 360;
    let (data, t, y) = sustained_series(n, &gaps);
    let mut graph = TemporalDag::empty();
    let t1 = ensure_lagged(&mut graph, VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
    let t2 = ensure_lagged(&mut graph, VariableId::from_raw(0), Lag::from_raw(2)).unwrap();
    let y0 = ensure_lagged(&mut graph, VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    graph.insert_directed(t1, y0).unwrap();
    graph.insert_directed(t2, y0).unwrap();
    let query = CausalQuery::TemporalEffect(
        TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0)
            .with_policy(TemporalPolicy::sustained(-2, -1)),
    );
    let run = |bootstrap: u32| {
        Study::series(data.clone())
            .query(query.clone())
            .graph(graph.clone())
            .refute(RefuteSuite::None)
            .bootstrap_replicates(bootstrap)
            .missing(declare(MissingPolicy::CompleteCase))
            .build()?
            .prepare(&ctx())?
            .estimate_retained(&ctx())
    };
    let result = run(0).unwrap();

    // Independent computation: OLS of y_i on (1, t_{i-1}, t_{i-2}) over the windows whose
    // regressors are all observed; the sustained contrast is the sum of the two slopes.
    let rows: Vec<usize> =
        (2..n).filter(|i| !gaps.contains(&(i - 1)) && !gaps.contains(&(i - 2))).collect();
    let design: Vec<[f64; 3]> = rows.iter().map(|&i| [1.0, t[i - 1], t[i - 2]]).collect();
    let mut gram = [[0.0; 3]; 3];
    let mut moment = [0.0; 3];
    for (row, &i) in design.iter().zip(&rows) {
        for a in 0..3 {
            moment[a] += row[a] * y[i];
            for b in 0..3 {
                gram[a][b] += row[a] * row[b];
            }
        }
    }
    let beta = solve3(gram, moment);
    let expected = beta[1] + beta[2];
    let ate = result.estimate.as_effect().unwrap().ate;
    assert!((ate - expected).abs() < 1e-6, "ate {ate} vs intact-window OLS {expected}");
    assert!((ate - 2.8).abs() < 0.05, "ate {ate}");

    // Windows that read a gap are dropped, each by its own time; no window spans one.
    let accounting = &result.missing_data.as_ref().unwrap().accounting;
    let dropped: Vec<usize> = (2..n).filter(|i| !rows.contains(i)).collect();
    assert_eq!(
        accounting.dropped_rows.iter().map(|r| usize::try_from(*r).unwrap()).collect::<Vec<_>>(),
        dropped
    );
    // The block bootstrap runs on the original window axis.
    let boot = run(120).unwrap();
    let se = boot.estimate.as_effect().unwrap().se_bootstrap().expect("bootstrap SE");
    assert!(se.is_finite() && se > 0.0, "{se}");
    assert_eq!(boot.estimate.as_effect().unwrap().ate, ate);
}

fn solve3(mut a: [[f64; 3]; 3], mut b: [f64; 3]) -> [f64; 3] {
    for col in 0..3 {
        let pivot = (col..3).max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs())).unwrap();
        a.swap(col, pivot);
        b.swap(col, pivot);
        for row in col + 1..3 {
            let f = a[row][col] / a[col][col];
            for k in col..3 {
                a[row][k] -= f * a[col][k];
            }
            b[row] -= f * b[col];
        }
    }
    let mut x = [0.0; 3];
    for row in (0..3).rev() {
        let tail: f64 = (row + 1..3).map(|k| a[row][k] * x[k]).sum();
        x[row] = (b[row] - tail) / a[row][row];
    }
    x
}

#[test]
fn the_conditional_marginal_effect_on_a_series_with_gaps_uses_the_intact_windows() {
    let n = 120;
    let gap = 30;
    let x: Vec<f64> = (0..n).map(|i| ((i as f64) * 0.37).sin()).collect();
    let w: Vec<f64> = (0..n).map(|i| if i % 2 == 0 { 0.0 } else { 1.0 }).collect();
    let y: Vec<f64> = (0..n)
        .map(|i| if i == 0 { 0.0 } else { 1.5 * x[i - 1] + 0.2 * w[i] + 0.8 * x[i - 1] * w[i] })
        .collect();
    let observed: Vec<f64> =
        x.iter().enumerate().map(|(i, v)| if i == gap { f64::NAN } else { *v }).collect();
    let data = TimeSeriesData::from_f64_columns(
        [("x", observed.as_slice()), ("y", y.as_slice()), ("w", w.as_slice())],
        1,
    )
    .unwrap();
    let mut graph = TemporalDag::empty();
    let x1 = ensure_lagged(&mut graph, VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
    let y0 = ensure_lagged(&mut graph, VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    let w0 = ensure_lagged(&mut graph, VariableId::from_raw(2), Lag::CONTEMPORANEOUS).unwrap();
    graph.insert_directed(x1, y0).unwrap();
    graph.insert_directed(w0, y0).unwrap();
    let query = CausalQuery::TemporalEffect(
        TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0)
            .with_horizon_steps(1)
            .with_max_history_lag(Some(1))
            .with_effect_modifiers([VariableId::from_raw(2)])
            .with_policy(TemporalPolicy::pulse(-1)),
    );
    let result = Study::series(data)
        .query(query)
        .graph(graph)
        .refute(RefuteSuite::None)
        .missing(declare(MissingPolicy::CompleteCase))
        .build()
        .unwrap()
        .prepare(&ctx())
        .unwrap()
        .estimate_retained(&ctx())
        .unwrap();
    // y is exactly linear in x_{t-1}, w_t and their product, so any set of intact windows
    // gives slope 1.5 + 0.8 mean(w) with the mean over the windows kept.
    let kept: Vec<usize> = (1..n).filter(|t| t - 1 != gap).collect();
    let wbar = kept.iter().map(|&t| w[t]).sum::<f64>() / kept.len() as f64;
    let ate = result.estimate.as_effect().unwrap().ate;
    assert!((ate - (1.5 + 0.8 * wbar)).abs() < 1e-8, "ate {ate} vs {}", 1.5 + 0.8 * wbar);
    assert_eq!(result.missing_data.as_ref().unwrap().accounting.dropped_rows.as_ref(), [31]);
}

#[test]
fn the_bayesian_single_window_on_a_series_with_gaps_uses_segment_wise_tempering() {
    let gaps = [100, 101, 202, 303];
    let n = 420;
    let (data, x, y) = series(n, &gaps);
    let result = Study::series(data)
        .query(pulse())
        .graph(temporal_graph())
        .refute(RefuteSuite::None)
        .inference(InferenceMode::Bayesian(
            BayesianConfig::conjugate().prior_scale(1_000.0).n_draws(400),
        ))
        .missing(declare(MissingPolicy::CompleteCase))
        .build()
        .unwrap()
        .prepare(&ctx())
        .unwrap()
        .estimate_retained(&ctx())
        .unwrap();
    // A diffuse prior leaves the posterior mean at the intact-window regression slope.
    let kept: Vec<usize> = (1..n).filter(|t| !gaps.contains(&(t - 1))).collect();
    let (mx, my) = (
        kept.iter().map(|&t| x[t - 1]).sum::<f64>() / kept.len() as f64,
        kept.iter().map(|&t| y[t]).sum::<f64>() / kept.len() as f64,
    );
    let slope = kept.iter().map(|&t| (x[t - 1] - mx) * (y[t] - my)).sum::<f64>()
        / kept.iter().map(|&t| (x[t - 1] - mx).powi(2)).sum::<f64>();
    let posterior = result.posterior.as_ref().expect("a Bayesian posterior");
    // Column 0 of the summaries is not the effect: read the effect column.
    let col = posterior.effect_column().expect("an effect column");
    assert!(
        (posterior.summaries.mean[col] - slope).abs() < 0.02,
        "{} vs {slope}",
        posterior.summaries.mean[col]
    );
    assert!(posterior.summaries.sd[col] > 0.0);
    assert_eq!(
        result.missing_data.as_ref().unwrap().accounting.dropped_rows.as_ref(),
        [101, 102, 203, 304]
    );
    // Predictive checks read consecutive rows, so a Bayesian fit with gaps declines them.
    let refused = Study::series(series(n, &gaps).0)
        .query(pulse())
        .graph(temporal_graph())
        .refute(RefuteSuite::Cheap)
        .inference(InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(50)))
        .missing(declare(MissingPolicy::CompleteCase))
        .build()
        .unwrap()
        .prepare(&ctx())
        .unwrap()
        .estimate_retained(&ctx())
        .unwrap_err();
    assert_eq!(refused.reason_code(), Some("missing_policy_not_applicable"), "{refused}");
}

fn sustained_graph() -> TemporalDag {
    let mut graph = TemporalDag::empty();
    let t1 = ensure_lagged(&mut graph, VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
    let t2 = ensure_lagged(&mut graph, VariableId::from_raw(0), Lag::from_raw(2)).unwrap();
    let y0 = ensure_lagged(&mut graph, VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    graph.insert_directed(t1, y0).unwrap();
    graph.insert_directed(t2, y0).unwrap();
    graph
}

fn sustained_query() -> CausalQuery {
    CausalQuery::TemporalEffect(
        TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0)
            .with_policy(TemporalPolicy::sustained(-2, -1)),
    )
}

#[test]
fn the_bayesian_sustained_contrast_on_a_series_with_gaps_recovers_the_known_effect() {
    let gaps = [90, 91, 200];
    let (data, _, _) = sustained_series(360, &gaps);
    let result = Study::series(data)
        .query(sustained_query())
        .graph(sustained_graph())
        .refute(RefuteSuite::None)
        .inference(InferenceMode::Bayesian(
            BayesianConfig::conjugate().prior_scale(1_000.0).n_draws(600),
        ))
        .missing(declare(MissingPolicy::CompleteCase))
        .build()
        .unwrap()
        .prepare(&ctx())
        .unwrap()
        .estimate_retained(&ctx())
        .unwrap();
    // The data are nearly deterministic: the sustained contrast is 2 + 0.8.
    let ate = result.estimate.as_effect().unwrap().ate;
    assert!((ate - 2.8).abs() < 0.1, "posterior contrast {ate}");
    assert!(result.posterior.is_some());
    let accounting = &result.missing_data.as_ref().unwrap().accounting;
    assert!(accounting.dropped_rows.iter().any(|row| *row == 91));
}

#[test]
fn sustained_refuters_on_a_series_with_gaps_refit_the_intact_windows() {
    let gaps = [90, 91, 200];
    let (data, _, _) = sustained_series(360, &gaps);
    let run = |inference: InferenceMode| {
        Study::series(data.clone())
            .query(sustained_query())
            .graph(sustained_graph())
            .refute(RefuteSuite::Full)
            .inference(inference)
            .bootstrap_replicates(60)
            .missing(declare(MissingPolicy::CompleteCase))
            .build()?
            .prepare(&ctx())?
            .estimate_retained(&ctx())
    };
    let result = run(InferenceMode::Frequentist).unwrap();
    assert!(!result.refutations.is_empty(), "the sustained refuters ran");
    assert!((result.estimate.as_effect().unwrap().ate - 2.8).abs() < 0.05);
    // Predictive checks read consecutive rows, so a Bayesian fit with gaps declines them.
    let refused =
        run(InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(50))).unwrap_err();
    assert_eq!(refused.reason_code(), Some("missing_policy_not_applicable"), "{refused}");
}

/// `m_i = 0.8 t_{i-1} + 0.6 z_{i-1} + noise`, `y_i = 1 + 0.25 t_{i-1} + 0.55 m_i + 5 z_{i-1}`,
/// with the mediator missing at `gaps`. The mediated effect is `0.8 * 0.55`.
fn mediation_series(gaps: &[usize]) -> TimeSeriesData {
    let n = 241;
    let z: Vec<f64> = (0..n).map(|i| if i % 4 < 2 { 1.0 } else { -1.0 }).collect();
    let t: Vec<f64> =
        (0..n).map(|i| z[i] + if i % 4 == 0 || i % 4 == 2 { 1.0 } else { -1.0 }).collect();
    let mut m = vec![0.0; n];
    let mut y = vec![1.0; n];
    for i in 1..n {
        let noise = (i % 5) as f64 - 2.0;
        m[i] = 0.8 * t[i - 1] + 0.6 * z[i - 1] + 0.15 * noise;
        y[i] = 1.0 + 0.25 * t[i - 1] + 0.55 * m[i] + 5.0 * z[i - 1];
    }
    let observed: Vec<f64> =
        m.iter().enumerate().map(|(i, v)| if gaps.contains(&i) { f64::NAN } else { *v }).collect();
    TimeSeriesData::from_f64_columns(
        [("t", t.as_slice()), ("m", observed.as_slice()), ("y", y.as_slice()), ("z", z.as_slice())],
        1,
    )
    .unwrap()
}

fn mediation_graph() -> TemporalDag {
    let mut graph = TemporalDag::empty();
    let t0 = ensure_lagged(&mut graph, VariableId::from_raw(0), Lag::CONTEMPORANEOUS).unwrap();
    let t1 = ensure_lagged(&mut graph, VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
    let m0 = ensure_lagged(&mut graph, VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    let y0 = ensure_lagged(&mut graph, VariableId::from_raw(2), Lag::CONTEMPORANEOUS).unwrap();
    let z0 = ensure_lagged(&mut graph, VariableId::from_raw(3), Lag::CONTEMPORANEOUS).unwrap();
    let z1 = ensure_lagged(&mut graph, VariableId::from_raw(3), Lag::from_raw(1)).unwrap();
    for (from, to) in [(z0, t0), (z1, y0), (z1, m0), (t1, y0), (t1, m0), (m0, y0)] {
        graph.insert_directed(from, to).unwrap();
    }
    graph
}

fn run_mediation(
    gaps: &[usize],
    inference: InferenceMode,
    suite: RefuteSuite,
    bootstrap: u32,
    declared: bool,
) -> Result<StudyResult, antecedent::CausalError> {
    let query = MediationQuery::binary(
        VariableId::from_raw(0),
        VariableId::from_raw(2),
        [VariableId::from_raw(1)],
        MediationContrast::Mediated,
    )
    .with_horizons(vec![1])
    .unwrap();
    let mut builder = Study::series(mediation_series(gaps))
        .graph(mediation_graph())
        .query(CausalQuery::Mediation(query))
        .inference(inference)
        .refute(suite)
        .bootstrap_replicates(bootstrap);
    if declared {
        builder = builder.missing(declare(MissingPolicy::CompleteCase));
    }
    builder.build()?.prepare(&ctx())?.estimate_retained(&ctx())
}

#[test]
fn mediation_on_a_series_with_gaps_recovers_the_known_mediated_effect_under_both_inferences() {
    const MEDIATED_TRUTH: f64 = 0.8 * 0.55;
    let gaps = [60, 61, 150];
    // Undeclared, the gap refuses as it always has.
    assert!(run_mediation(&gaps, InferenceMode::Frequentist, RefuteSuite::None, 0, false).is_err());
    let frequentist =
        run_mediation(&gaps, InferenceMode::Frequentist, RefuteSuite::None, 0, true).unwrap();
    let ate = frequentist.estimate.as_effect().unwrap().ate;
    // The mediator equation carries noise 0.15 * (i % 5 - 2), sd 0.15 * sqrt(2) = 0.212; the
    // outcome equation is exact, so the product is 0.55 * a_hat. With about 235 intact windows
    // and a treatment residual sd near 1, SE(a_hat) = 0.212 / sqrt(235) = 0.0138, so
    // SE(mediated) = 0.55 * 0.0138 = 0.0076. The tolerance is 3.3 of those SEs.
    assert!((ate - MEDIATED_TRUTH).abs() < 0.025, "frequentist mediated effect {ate}");
    let dropped = &frequentist.missing_data.as_ref().unwrap().accounting.dropped_rows;
    assert!([60, 61, 150].iter().all(|time| dropped.contains(time)), "{dropped:?}");

    let bayesian = run_mediation(
        &gaps,
        InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(2048).prior_scale(1_000.0)),
        RefuteSuite::None,
        0,
        true,
    )
    .unwrap();
    let ate = bayesian.estimate.as_effect().unwrap().ate;
    assert!((ate - MEDIATED_TRUTH).abs() < 0.08, "Bayesian mediated effect {ate}");
    assert!(bayesian.posterior.is_some());
}

#[test]
fn mediation_refuters_on_a_series_with_gaps_refit_the_intact_windows() {
    let gaps = [60, 61, 150];
    let result =
        run_mediation(&gaps, InferenceMode::Frequentist, RefuteSuite::Full, 60, true).unwrap();
    assert!(!result.refutations.is_empty(), "the mediation refuters ran");
    let refused = run_mediation(
        &gaps,
        InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(50)),
        RefuteSuite::Cheap,
        0,
        true,
    )
    .unwrap_err();
    assert_eq!(refused.reason_code(), Some("missing_policy_not_applicable"), "{refused}");
}
