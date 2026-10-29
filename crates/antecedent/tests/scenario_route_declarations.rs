//! Declared dependence, table weights and extra treatment levels are read or refused per route.
//!
//! Each route is first run where it works, against an answer written out by hand, and the same
//! route is then given the declaration it cannot read: a clustered panel or dependent time
//! series (study-level), observation weights on the table, or a discrete treatment with a level
//! beyond the two contrasted. It is refused with `option_not_applicable` (or
//! `treatment_level_undeclared`) and a next attempt; none returns the independent-rows or
//! dose-slope number as if the declaration had been read.
//!
//! - mediation and survival resample subjects, so a declared dependence refuses their bootstrap;
//! - the sharp and fuzzy discontinuity estimators sandwich rows one by one;
//! - the policy value is an influence-function interval over independent evaluation subjects and
//!   reads no weights either;
//! - the Bayesian static routes condition on independent rows;
//! - the conditional effect refuses a three-level treatment instead of fitting its slope.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::{
    BayesianConfig, CausalError, EstimatorId, IdentifierId, InferenceMode, LocalPolynomialRatioQuery,
    RefuteSuite, Study, StudyResult,
};
use antecedent_core::{
    AverageEffectQuery, CausalQuery, ChangeAttributionQuery, ConditionalEffectQuery,
    DeclaredSamplingDesign, ExecutionContext, Intervention, MediationContrast, MediationQuery,
    ObservationAssumption, PolicyValueQuery, PopulationSelector, SurvivalFunctional,
    SurvivalQuery, Value, VariableId,
};
use antecedent_data::TabularData;
use antecedent_graph::{Dag, DenseNodeId};

fn v(raw: u32) -> VariableId {
    VariableId::from_raw(raw)
}

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(5)
}

const DEPENDENT: [DeclaredSamplingDesign; 2] =
    [DeclaredSamplingDesign::ClusteredPanel, DeclaredSamplingDesign::DependentTimeSeries];

fn assert_refused(what: &str, error: Option<CausalError>, phrase: &str) {
    let error = error.unwrap_or_else(|| panic!("{what}: published a number for a declaration it ignores"));
    assert_eq!(error.reason_code(), Some("option_not_applicable"), "{what}: {error}");
    let text = error.to_string();
    assert!(text.contains(phrase), "{what}: {text}");
    assert!(text.contains("Next attempt"), "{what}: {text}");
}

fn edges(graph: &mut Dag, list: &[(u32, u32)]) {
    for &(from, to) in list {
        graph.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).unwrap();
    }
}

// ---------------------------------------------------------------------------------------------
// Mediation: y = 3 a + 4 m + z, m = 2 a + noise, so the natural direct effect of a is 3.
// ---------------------------------------------------------------------------------------------

const N: usize = 300;

fn mediation_data() -> TabularData {
    let at = |i: usize| i as f64;
    let z: Vec<f64> = (0..N).map(|i| (at(i) * 0.37).sin()).collect();
    let a: Vec<f64> = (0..N).map(|i| z[i] + (at(i) * 0.71).cos()).collect();
    let m: Vec<f64> = (0..N).map(|i| 2.0 * a[i] + (at(i) * 1.13).cos()).collect();
    let y: Vec<f64> =
        (0..N).map(|i| 3.0 * a[i] + 4.0 * m[i] + z[i] + 0.1 * (at(i) * 0.31).sin()).collect();
    TabularData::from_f64_columns([
        ("a", a.as_slice()),
        ("m", m.as_slice()),
        ("y", y.as_slice()),
        ("z", z.as_slice()),
    ])
    .unwrap()
}

fn mediation_dag() -> Dag {
    let mut graph = Dag::with_variables(4);
    edges(&mut graph, &[(3, 0), (3, 2), (0, 1), (1, 2), (0, 2)]);
    graph
}

fn mediation(replicates: u32, design: Option<DeclaredSamplingDesign>) -> Result<StudyResult, CausalError> {
    let mut query =
        MediationQuery::binary(v(0), v(2), Arc::from([v(1)]), MediationContrast::NaturalDirect);
    query.control = Intervention::set(v(0), Value::f64(0.0));
    query.active = Intervention::set(v(0), Value::f64(1.0));
    let builder = Study::tabular(mediation_data())
        .graph(mediation_dag())
        .refute(RefuteSuite::None)
        .bootstrap_replicates(replicates)
        .query(CausalQuery::Mediation(query));
    let builder = match design {
        Some(design) => builder.sampling_design(design),
        None => builder,
    };
    builder.build().and_then(|study| study.run(&ctx()))
}

#[test]
fn the_mediation_bootstrap_refits_independent_subjects_and_refuses_a_declared_dependence() {
    let effect = mediation(20, Some(DeclaredSamplingDesign::IndependentSubjectTrajectories))
        .unwrap();
    let effect = effect.estimate().as_effect().unwrap();
    assert!((effect.ate - 3.0).abs() < 1e-3, "{}", effect.ate);
    assert!(effect.uncertainty.bootstrap.se.is_some_and(|se| se > 0.0));
    for design in DEPENDENT {
        assert_refused(
            &format!("mediation {design:?}"),
            mediation(20, Some(design)).err(),
            "resamples independent subjects",
        );
    }
    // Without a bootstrap the route publishes the point value only: there is no error for the
    // declaration to be ignored by.
    for design in DEPENDENT {
        let point = mediation(0, Some(design)).unwrap();
        let point = point.estimate().as_effect().unwrap();
        assert!((point.ate - 3.0).abs() < 1e-3, "{design:?}: {}", point.ate);
        assert!(point.uncertainty.analytic.se.is_nan());
    }
}

// ---------------------------------------------------------------------------------------------
// Fuzzy and sharp regression discontinuity: a jump of 3 at the cutoff.
// ---------------------------------------------------------------------------------------------

fn fuzzy_table() -> TabularData {
    let (mut x, mut treatment, mut outcome) = (Vec::new(), Vec::new(), Vec::new());
    for step in 1..80 {
        for sign in [-1.0, 1.0] {
            let score = sign * f64::from(step) / 80.0;
            for replicate in 0..4 {
                let dose = if sign < 0.0 {
                    f64::from(u8::from(replicate == 0))
                } else {
                    f64::from(u8::from(replicate != 3))
                };
                x.push(score);
                treatment.push(dose);
                outcome.push(
                    1.0 + 2.0 * score
                        + 0.5 * score * score
                        + 3.0 * dose
                        + [-0.02, -0.01, 0.01, 0.02][replicate],
                );
            }
        }
    }
    TabularData::from_f64_columns([
        ("x", x.as_slice()),
        ("t", treatment.as_slice()),
        ("y", outcome.as_slice()),
    ])
    .unwrap()
}

fn fuzzy(design: Option<DeclaredSamplingDesign>) -> Result<StudyResult, CausalError> {
    let query = LocalPolynomialRatioQuery {
        outcome: v(2),
        treatment: v(1),
        running: v(0),
        cutoff: 0.0,
        bandwidth: 1.0,
        kink: false,
        options: None,
    };
    let builder = Study::tabular(fuzzy_table())
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .query(CausalQuery::LocalPolynomialRatio(query));
    let builder = match design {
        Some(design) => builder.sampling_design(design),
        None => builder,
    };
    builder.build().and_then(|study| study.run(&ctx()))
}

#[test]
fn the_fuzzy_ratio_recovers_the_jump_and_refuses_a_declared_clustered_or_serial_design() {
    let ate = |result: StudyResult| result.estimate().as_effect().unwrap().ate;
    // Dose 3, outcome jump 3 over a first-stage jump of 0.75 - 0.25 = 0.5 per unit of dose.
    assert!((ate(fuzzy(None).unwrap()) - 3.0).abs() < 1e-9);
    assert!(
        (ate(fuzzy(Some(DeclaredSamplingDesign::IndependentSubjectTrajectories)).unwrap()) - 3.0)
            .abs()
            < 0.05
    );
    assert_refused(
        "fuzzy ratio clustered",
        fuzzy(Some(DeclaredSamplingDesign::ClusteredPanel)).err(),
        "clustered_panel",
    );
    assert_refused(
        "fuzzy ratio serial",
        fuzzy(Some(DeclaredSamplingDesign::DependentTimeSeries)).err(),
        "dependent_time_series",
    );
}

fn sharp(replicates: u32, design: Option<DeclaredSamplingDesign>) -> Result<StudyResult, CausalError> {
    sharp_weighted(replicates, design, None)
}

fn sharp_weighted(
    replicates: u32,
    design: Option<DeclaredSamplingDesign>,
    weights: Option<Vec<f64>>,
) -> Result<StudyResult, CausalError> {
    let n = 600usize;
    let r: Vec<f64> = (0..n).map(|i| -1.0 + 2.0 * (i as f64 + 0.5) / n as f64).collect();
    let t: Vec<f64> = r.iter().map(|&x| f64::from(u8::from(x >= 0.0))).collect();
    let y: Vec<f64> = (0..n)
        .map(|i| 3.0 * t[i] + 0.5 * r[i] + 0.05 * ((i * 37 % 17) as f64 / 17.0 - 0.5))
        .collect();
    let mut data = TabularData::from_f64_columns([
        ("t", t.as_slice()),
        ("y", y.as_slice()),
        ("r", r.as_slice()),
    ])
    .unwrap();
    if let Some(weights) = weights {
        data = data.with_weights(Arc::from(weights)).unwrap();
    }
    let mut graph = Dag::with_variables(3);
    edges(&mut graph, &[(2, 0), (0, 1), (2, 1)]);
    let builder = Study::tabular(data)
        .graph(graph)
        .query(AverageEffectQuery::binary_ate(v(0), v(1)))
        .identifier(IdentifierId::RdSharp)
        .estimator(EstimatorId::RdSharp)
        .rd_config(v(2), 0.0, 1.0)
        .bootstrap_replicates(replicates);
    let builder = match design {
        Some(design) => builder.sampling_design(design),
        None => builder,
    };
    builder.build().and_then(|study| study.run(&ctx()))
}

#[test]
fn the_sharp_discontinuity_recovers_the_jump_and_refuses_a_declared_dependence_with_or_without_a_bootstrap()
{
    for replicates in [0, 20] {
        let result = sharp(replicates, None).unwrap();
        let effect = result.estimate().as_effect().unwrap();
        assert!((effect.ate - 3.0).abs() < 1e-3, "{}", effect.ate);
        for design in DEPENDENT {
            assert_refused(
                &format!("sharp rd {design:?} replicates {replicates}"),
                sharp(replicates, Some(design)).err(),
                "regression-discontinuity",
            );
        }
    }
}

#[test]
fn the_sharp_discontinuity_refuses_observation_weights_instead_of_answering_unweighted() {
    assert!(sharp_weighted(0, None, None).is_ok());
    for weights in [(0..600).map(|i| 1.0 + (i % 4) as f64).collect::<Vec<_>>(), vec![1.0; 600]] {
        let error = sharp_weighted(0, None, Some(weights)).err().expect("weights were ignored");
        assert_eq!(error.reason_code(), Some("option_not_applicable"), "{error}");
        let text = error.to_string();
        assert!(text.contains("sampling_weights.route_unsupported"), "{text}");
        assert!(text.contains("Next attempt"), "{text}");
    }
}

// ---------------------------------------------------------------------------------------------
// Randomized survival: two arms with identical durations and events have identical curves.
// ---------------------------------------------------------------------------------------------

fn survival(replicates: u32, design: Option<DeclaredSamplingDesign>) -> Result<StudyResult, CausalError> {
    // 60 subjects per arm with the same durations and events in both arms.
    let base_duration: Vec<f64> = (0..60).map(|i| 1.0 + ((i * 7) % 11) as f64 / 3.0).collect();
    let base_event: Vec<f64> = (0..60).map(|i| f64::from(u8::from(i % 4 != 0))).collect();
    let (mut duration, mut event, mut treatment) = (vec![], vec![], vec![]);
    for arm in [0.0, 1.0] {
        for k in 0..base_duration.len() {
            duration.push(base_duration[k]);
            event.push(base_event[k]);
            treatment.push(arm);
        }
    }
    let data = TabularData::from_f64_columns([
        ("duration", duration.as_slice()),
        ("event", event.as_slice()),
        ("treatment", treatment.as_slice()),
    ])
    .unwrap();
    let query = CausalQuery::Survival(SurvivalQuery {
        duration: v(0),
        event: v(1),
        treatment: v(2),
        tau: 3.0,
        delayed_entry: None,
        known_censoring: None,
        observation_assumption: ObservationAssumption::IndependentGiven(Arc::from([])),
        functional: SurvivalFunctional::SurvivalAndRmst,
    });
    let builder = Study::tabular(data).query(query).bootstrap_replicates(replicates);
    let builder = match design {
        Some(design) => builder.sampling_design(design),
        None => builder,
    };
    builder.build().and_then(|study| study.run(&ctx()))
}

#[test]
fn the_survival_subject_bootstrap_refuses_a_declared_clustered_or_serial_design() {
    let result = survival(199, None).unwrap();
    let curves = result.survival().unwrap();
    assert_eq!(curves.control, curves.treated, "identical arms have identical curves");
    let [low, high] = curves.rmst_difference_interval.expect("the requested bootstrap interval");
    assert!(low <= 0.0 && 0.0 <= high, "an identical-arm contrast covers zero: {low}, {high}");
    survival(199, Some(DeclaredSamplingDesign::IndependentSubjectTrajectories)).unwrap();
    for design in DEPENDENT {
        assert_refused(
            &format!("survival {design:?}"),
            survival(199, Some(design)).err(),
            "resamples independent subjects",
        );
    }
    // Without a bootstrap the contrast is a point value with no interval to understate.
    for design in DEPENDENT {
        assert!(survival(0, Some(design)).unwrap().survival().unwrap().rmst_difference_interval.is_none());
    }
}

// ---------------------------------------------------------------------------------------------
// Policy value: the doubly robust value of acting on every third row.
// ---------------------------------------------------------------------------------------------

const POLICY_ROWS: usize = 300;

fn policy_study(
    weights: Option<Vec<f64>>,
    design: Option<DeclaredSamplingDesign>,
) -> (Result<StudyResult, CausalError>, f64) {
    let n = POLICY_ROWS;
    let assignment: Vec<bool> = (0..n).map(|i| i % 2 == 1).collect();
    let actions: Vec<bool> = (0..n).map(|i| i % 3 == 0).collect();
    let residual = |i: usize| (i % 5) as f64 / 10.0;
    let outcomes: Vec<f64> =
        (0..n).map(|i| 1.0 + 2.0 * f64::from(assignment[i]) + residual(i)).collect();
    // Doubly robust score with mu0 = 1, mu1 = 3 (exact) and assignment probability one half:
    // the model value of the policy's action plus the inverse-probability residual where the
    // observed arm equals the policy's action.
    let truth = (0..n)
        .map(|i| {
            let model = if actions[i] { 3.0 } else { 1.0 };
            let correction =
                if actions[i] == assignment[i] { (outcomes[i] - model) / 0.5 } else { 0.0 };
            model + correction
        })
        .sum::<f64>()
        / n as f64;
    let mut data = TabularData::from_f64_columns([("outcome", outcomes.as_slice())]).unwrap();
    if let Some(weights) = weights {
        data = data.with_weights(Arc::from(weights)).unwrap();
    }
    let query = PolicyValueQuery {
        outcome: v(0),
        assignment: assignment.into(),
        propensity: Arc::from([0.5]),
        actions: actions.into(),
        reference: vec![false; n].into(),
        mu0: vec![1.0; n].into(),
        mu1: vec![3.0; n].into(),
        costs: Arc::from([0.0]),
        reference_costs: Arc::from([0.0]),
        evaluation_subject_ids: (0..n)
            .map(|i| Arc::<str>::from(format!("evaluation-{i}")))
            .collect::<Vec<_>>()
            .into(),
        disjoint_training_subjects: false,
        crossfit_fold_ownership_valid: true,
        global_constraints_present: false,
        multi_action: None,
        uplift_bins: Arc::from([]),
        uplift_bin_count: 0,
        uplift_training_subject_ids: Arc::from([]),
        regret: None,
    };
    let builder = Study::tabular(data).query(CausalQuery::PolicyValue(query));
    let builder = match design {
        Some(design) => builder.sampling_design(design),
        None => builder,
    };
    (builder.build().and_then(|study| study.run(&ctx())), truth)
}

#[test]
fn the_policy_value_matches_its_hand_score_and_refuses_weights_and_declared_dependence() {
    let (result, truth) = policy_study(None, None);
    let value = result.unwrap();
    let value = value.policy_value().unwrap();
    assert!((value.policy_value - truth).abs() < 1e-9, "{} vs {truth}", value.policy_value);
    let (independent, _) =
        policy_study(None, Some(DeclaredSamplingDesign::IndependentSubjectTrajectories));
    assert!(independent.is_ok());
    for design in DEPENDENT {
        assert_refused(
            &format!("policy value {design:?}"),
            policy_study(None, Some(design)).0.err(),
            "independent evaluation subjects",
        );
    }
    // Observation weights on the table: uneven, and the constant one.
    for weights in [
        (0..POLICY_ROWS).map(|i| 1.0 + (i % 4) as f64).collect::<Vec<_>>(),
        vec![1.0; POLICY_ROWS],
    ] {
        let error = policy_study(Some(weights), None).0.err().expect("weights were ignored");
        assert_eq!(error.reason_code(), Some("option_not_applicable"), "{error}");
        let text = error.to_string();
        assert!(text.contains("sampling_weights.route_unsupported"), "{text}");
        assert!(text.contains("Next attempt"), "{text}");
    }
}

// ---------------------------------------------------------------------------------------------
// The Bayesian static routes condition on independent rows.
// ---------------------------------------------------------------------------------------------

#[test]
fn a_bayesian_static_route_refuses_a_declared_clustered_or_serial_design() {
    let run = |design: Option<DeclaredSamplingDesign>| {
        let builder = Study::tabular(mediation_data())
            .graph(mediation_dag())
            .refute(RefuteSuite::None)
            .inference(InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(200)))
            .query(AverageEffectQuery::binary_ate(v(0), v(2)));
        let builder = match design {
            Some(design) => builder.sampling_design(design),
            None => builder,
        };
        builder.build().and_then(|study| study.run(&ctx()))
    };
    run(None).unwrap();
    run(Some(DeclaredSamplingDesign::IndependentSubjectTrajectories)).unwrap();
    for design in DEPENDENT {
        assert_refused(&format!("bayesian {design:?}"), run(Some(design)).err(), "rows are independent");
    }
}

// ---------------------------------------------------------------------------------------------
// Conditional effect of a three-level treatment: the contrast of level 1 against 0 is 1.
// ---------------------------------------------------------------------------------------------

fn conditional_levels(levels: usize) -> Result<StudyResult, CausalError> {
    let n = 120usize;
    let t: Vec<f64> = (0..n).map(|i| (i % levels) as f64).collect();
    let z: Vec<f64> = (0..n).map(|i| ((i / 3) % 2) as f64).collect();
    let y: Vec<f64> = (0..n).map(|i| [0.0, 1.0, 4.0][i % levels] + 0.5 * z[i]).collect();
    let data = TabularData::from_f64_columns([
        ("t", t.as_slice()),
        ("y", y.as_slice()),
        ("z", z.as_slice()),
    ])
    .unwrap();
    let mut graph = Dag::with_variables(3);
    edges(&mut graph, &[(2, 0), (2, 1), (0, 1)]);
    let query = ConditionalEffectQuery::try_new(
        AverageEffectQuery::with_levels(v(0), v(1), 0.0, 1.0).with_effect_modifiers(vec![v(2)]),
    )
    .unwrap();
    Study::tabular(data)
        .graph(graph)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .query(CausalQuery::ConditionalEffect(query))
        .build()
        .and_then(|study| study.run(&ctx()))
}

#[test]
fn a_three_level_treatment_is_refused_by_the_conditional_effect_instead_of_read_as_a_dose() {
    let two = conditional_levels(2).unwrap();
    assert!((two.estimate().as_effect().unwrap().ate - 1.0).abs() < 1e-9);
    let error = conditional_levels(3)
        .err()
        .expect("a three-level treatment was read as a dose (slope 2 against a contrast of 1)");
    assert_eq!(error.reason_code(), Some("treatment_level_undeclared"), "{error}");
    let text = error.to_string();
    assert!(text.contains("observed treatment levels: 0 (40 rows), 1 (40 rows), 2 (40 rows)"), "{text}");
    assert!(text.contains("Next attempt"), "{text}");
}

// ---------------------------------------------------------------------------------------------
// Change attribution: an intercept shift of 3 between two periods.
// ---------------------------------------------------------------------------------------------

#[test]
fn the_attribution_bootstrap_refuses_a_declared_clustered_panel() {
    let n = 200usize;
    let x: Vec<f64> = (0..n).map(|i| (((i % (n / 2)) * 29) % 31) as f64 / 31.0).collect();
    let y: Vec<f64> = (0..n)
        .map(|i| 1.0 + if i < n / 2 { 0.0 } else { 3.0 } + 2.0 * x[i])
        .collect();
    let data = TabularData::from_f64_columns([("x", x.as_slice()), ("y", y.as_slice())]).unwrap();
    let mut graph = Dag::with_variables(2);
    edges(&mut graph, &[(0, 1)]);
    let run = |design: DeclaredSamplingDesign| {
        Study::tabular(data.clone())
            .graph(graph.clone())
            .query(CausalQuery::ChangeAttribution(ChangeAttributionQuery::new(
                v(1),
                PopulationSelector::TimeRange { start: 0, end: n / 2 },
                PopulationSelector::TimeRange { start: n / 2, end: n },
            )))
            .inference(InferenceMode::Frequentist)
            .refute(RefuteSuite::None)
            .bootstrap_replicates(20)
            .sampling_design(design)
            .build()
            .and_then(|study| study.run(&ctx()))
    };
    let independent = run(DeclaredSamplingDesign::IndependentSubjectTrajectories).unwrap();
    let total = independent.estimate().as_effect().unwrap().ate;
    assert!((total - 3.0).abs() < 1e-6, "total change {total}");
    assert_refused(
        "change attribution clustered",
        run(DeclaredSamplingDesign::ClusteredPanel).err(),
        "resamples independent subjects",
    );
}

// ---------------------------------------------------------------------------------------------
// Missing values and a three-level treatment on the routes that read neither.
// ---------------------------------------------------------------------------------------------

/// The columns of `data` as owned vectors, with row `row` of column `column` set to NaN.
fn with_nan(columns: &[(&str, Vec<f64>)], column: usize, row: usize) -> TabularData {
    let mut owned: Vec<(&str, Vec<f64>)> = columns.to_vec();
    owned[column].1[row] = f64::NAN;
    TabularData::from_f64_columns(owned.iter().map(|(n, c)| (*n, c.as_slice()))).unwrap()
}

#[test]
fn a_continuous_dose_route_refuses_declared_dependence_and_missing_values_instead_of_using_them() {
    use antecedent_core::ContinuousDoseResponseQuery;
    let doses = [-1.0, -0.5, 0.0, 0.5, 1.0];
    let (mut outcome, mut dose) = (Vec::new(), Vec::new());
    for intercept in [1.0, 10.0] {
        for value in doses {
            outcome.push(intercept + 2.0 * value);
            dose.push(value);
        }
    }
    let columns = vec![("y", outcome), ("dose", dose), ("density", vec![0.5; 10])];
    let groups: Vec<Arc<str>> = ["control", "treated"]
        .into_iter()
        .flat_map(|group| std::iter::repeat_n(Arc::<str>::from(group), 5))
        .collect();
    let query = ContinuousDoseResponseQuery {
        outcome: v(0),
        dose: v(1),
        dose_density: v(2),
        baseline_groups: groups.into(),
        target_doses: Arc::from([0.0, 0.5]),
        bandwidth: 0.6,
        min_local_support: 3,
        density_provenance: Arc::from("known"),
        fixed_policy: None,
    };
    let run = |data: TabularData| {
        Study::tabular(data)
            .refute(RefuteSuite::None)
            .bootstrap_replicates(0)
            .query(CausalQuery::ContinuousDoseResponse(query.clone()))
            .build()
            .and_then(|study| study.run(&ctx()))
    };
    let complete =
        TabularData::from_f64_columns(columns.iter().map(|(n, c)| (*n, c.as_slice()))).unwrap();
    run(complete.clone()).unwrap();
    // Its variance and interval are over independent rows, so a declared dependence is refused.
    let declared = |design: DeclaredSamplingDesign| {
        Study::tabular(complete.clone())
            .refute(RefuteSuite::None)
            .bootstrap_replicates(0)
            .sampling_design(design)
            .query(CausalQuery::ContinuousDoseResponse(query.clone()))
            .build()
            .and_then(|study| study.run(&ctx()))
    };
    declared(DeclaredSamplingDesign::IndependentSubjectTrajectories).unwrap();
    for design in DEPENDENT {
        assert_refused(
            &format!("continuous dose {design:?}"),
            declared(design).err(),
            "independent evaluation subjects",
        );
    }
    for column in 0..3 {
        let error = run(with_nan(&columns, column, 3))
            .err()
            .unwrap_or_else(|| panic!("column {column}: a row with a missing value was used"));
        let text = error.to_string();
        assert!(text.contains("finite"), "column {column}: {text}");
    }
}

#[test]
fn a_randomized_survival_contrast_refuses_a_three_level_treatment_instead_of_reading_a_dose() {
    let n = 90usize;
    let duration: Vec<f64> =
        (0..n).map(|i| 1.0 + ((i * 7) % 11) as f64 / 3.0 + (i % 3) as f64 * 0.2).collect();
    let event: Vec<f64> = (0..n).map(|i| f64::from(u8::from(i % 4 != 0))).collect();
    let treatment: Vec<f64> = (0..n).map(|i| (i % 3) as f64).collect();
    let data = TabularData::from_f64_columns([
        ("duration", duration.as_slice()),
        ("event", event.as_slice()),
        ("treatment", treatment.as_slice()),
    ])
    .unwrap();
    let query = CausalQuery::Survival(SurvivalQuery {
        duration: v(0),
        event: v(1),
        treatment: v(2),
        tau: 3.0,
        delayed_entry: None,
        known_censoring: None,
        observation_assumption: ObservationAssumption::IndependentGiven(Arc::from([])),
        functional: SurvivalFunctional::SurvivalAndRmst,
    });
    let error = Study::tabular(data)
        .query(query)
        .build()
        .and_then(|study| study.run(&ctx()))
        .err()
        .expect("a three-level treatment was read as a binary arm");
    assert!(error.to_string().contains("treatment must be binary zero or one"), "{error}");
}

// ---------------------------------------------------------------------------------------------
// Randomized designs: the variance is over the declared assignment units, one per row.
// ---------------------------------------------------------------------------------------------

#[test]
fn multi_arm_and_factorial_designs_refuse_outcome_rows_nested_in_one_assignment_unit() {
    use antecedent::RandomizedEffectQuery;
    use antecedent_core::RandomizationDesign;
    let units = |ids: &[usize]| -> Vec<Arc<str>> {
        ids.iter().map(|id| Arc::<str>::from(format!("unit-{id}"))).collect()
    };
    let rows = |n: usize| units(&(0..n).collect::<Vec<_>>());
    // Twelve rows from six assignment units of two rows each, arms 0, 1, 2 by unit.
    let n = 12usize;
    let nested: Vec<usize> = (0..n).map(|i| i / 2).collect();
    let arm: Vec<usize> = nested.iter().map(|unit| unit % 3).collect();
    let multi = |assignment_units: Vec<Arc<str>>| {
        RandomizedEffectQuery::with_design(
            RandomizationDesign::MultiArm {
                assignment: arm.clone().into(),
                probabilities: vec![vec![1.0 / 3.0; 3]; n].into(),
                arms: ["control", "low", "high"].map(Arc::<str>::from).into(),
            },
            v(0),
            arm.iter().map(|&a| a != 0).collect::<Vec<_>>(),
            vec![1.0 / 3.0; n],
            assignment_units,
            rows(n),
            ("control", "low"),
        )
    };
    let outcome: Vec<f64> = (0..n).map(|i| [0.0, 2.0, 5.0][arm[i]] + 0.1 * (i % 2) as f64).collect();
    let data = TabularData::from_f64_columns([("outcome", outcome.as_slice())]).unwrap();
    let run = |query: RandomizedEffectQuery| {
        Study::tabular(data.clone()).query(query).build().and_then(|study| study.run(&ctx()))
    };
    // One assignment unit per row runs, and the low-versus-control contrast is the arm means'
    // difference (2, up to the half-step alternation that is balanced within every arm).
    let fit = run(multi(rows(n))).unwrap();
    let effect = fit.randomized_effect().unwrap().effect;
    assert!((effect - 2.0).abs() < 1e-12, "{effect}");
    let error = run(multi(units(&nested))).err().expect("nested rows were treated as independent");
    assert!(error.to_string().contains("distinct unit identities"), "{error}");

    // 2x2 factorial with two rows per assignment unit.
    let outcome = [0.0, 2.0, 2.0, 4.0, 1.0, 3.0, 5.0, 7.0];
    let data = TabularData::from_f64_columns([("outcome", outcome.as_slice())]).unwrap();
    let factorial = |assignment_units: Vec<Arc<str>>| {
        RandomizedEffectQuery::with_design(
            RandomizationDesign::Factorial2x2 {
                second_factor_assignment: [false, false, false, false, true, true, true, true]
                    .into(),
                cell_counts: [2; 4],
                second_factor_arms: (Arc::from("off"), Arc::from("on")),
            },
            v(0),
            [false, false, true, true, false, false, true, true],
            [0.5; 8],
            assignment_units,
            rows(8),
            ("control", "treated"),
        )
    };
    let run = |query: RandomizedEffectQuery| {
        Study::tabular(data.clone()).query(query).build().and_then(|study| study.run(&ctx()))
    };
    let fit = run(factorial(rows(8))).unwrap();
    assert!((fit.randomized_effect().unwrap().effect - 3.0).abs() < 1e-12);
    let error = run(factorial(units(&[0, 0, 1, 1, 2, 2, 3, 3])))
        .err()
        .expect("nested rows were treated as independent");
    assert!(error.to_string().contains("one assignment unit per row"), "{error}");
}

#[test]
fn the_quantile_effect_refuses_a_third_level_and_a_declared_dependence_instead_of_reading_them() {
    use antecedent_core::OutcomeFunctional;
    let run = |levels: usize, design: Option<DeclaredSamplingDesign>, replicates: u32| {
        let n = 200usize;
        let t: Vec<f64> = (0..n).map(|i| (i % levels) as f64).collect();
        let z: Vec<f64> = (0..n).map(|i| ((i / 3) % 2) as f64).collect();
        // A shift of 1 at level 1 and 4 at level 2; the within-arm spread is the same pattern.
        let y: Vec<f64> = (0..n)
            .map(|i| [0.0, 1.0, 4.0][i % levels] + 0.5 * z[i] + ((i * 7) % 11) as f64 / 11.0)
            .collect();
        let data = TabularData::from_f64_columns([
            ("t", t.as_slice()),
            ("y", y.as_slice()),
            ("z", z.as_slice()),
        ])
        .unwrap();
        let mut graph = Dag::with_variables(3);
        edges(&mut graph, &[(2, 0), (2, 1), (0, 1)]);
        let query = AverageEffectQuery::with_levels(v(0), v(1), 0.0, 1.0)
            .with_outcome_functional(OutcomeFunctional::quantile(0.5));
        let builder = Study::tabular(data)
            .graph(graph)
            .refute(RefuteSuite::None)
            .estimator(antecedent_estimate::AipwAte::new().with_bootstrap_replicates(replicates))
            .query(query);
        let builder = match design {
            Some(design) => builder.sampling_design(design),
            None => builder,
        };
        builder.build().and_then(|study| study.run(&ctx()))
    };
    // The median contrast of level 1 against level 0 is the shift 1 (the spread is shared).
    let two = run(2, None, 99).unwrap();
    let effect = two.estimate().as_effect().unwrap();
    assert!((effect.ate - 1.0).abs() < 5e-3, "{}", effect.ate);
    let error = run(3, None, 99).err().expect("a three-level treatment was read as a dose");
    assert_eq!(error.reason_code(), Some("treatment_level_undeclared"), "{error}");
    for design in DEPENDENT {
        let text = run(2, Some(design), 99)
            .err()
            .unwrap_or_else(|| panic!("{design:?}: a row bootstrap under a declared dependence"))
            .to_string();
        assert!(text.contains("does not allow row_iid resampling"), "{design:?}: {text}");
        let text = run(2, Some(design), 0)
            .err()
            .unwrap_or_else(|| panic!("{design:?}: an independent-rows error under a dependence"))
            .to_string();
        assert!(text.contains("Next attempt"), "{design:?}: {text}");
    }
}

#[test]
fn a_static_structure_search_refuses_a_declared_clustered_or_serial_design_instead_of_testing_rows_as_independent()
{
    use antecedent::DiscoverySpec;
    use antecedent::discovery::{BayesianDiscoverParams, StaticDiscoverParams};
    use antecedent::discovery_defaults::resolve_ci;
    use antecedent_core::CausalRng;
    use antecedent_discovery::BackgroundKnowledge;
    // z -> t -> y with uniform noise.
    let mut rng = CausalRng::from_seed(7);
    let mut uniform = || 2.0 * rng.next_f64() - 1.0;
    let (mut z, mut t, mut y) = (Vec::new(), Vec::new(), Vec::new());
    for _ in 0..300 {
        let zi = uniform();
        let ti = 0.8 * zi + uniform();
        let yi = 1.5 * ti + uniform();
        z.push(zi);
        t.push(ti);
        y.push(yi);
    }
    let data = TabularData::from_f64_columns([
        ("z", z.as_slice()),
        ("t", t.as_slice()),
        ("y", y.as_slice()),
    ])
    .unwrap();
    let params = StaticDiscoverParams {
        alpha: 0.05,
        max_cond_size: 2,
        fdr: None,
        ci: resolve_ci("parcorr", None).unwrap(),
        screen_pc: false,
        max_subset: None,
        knowledge: BackgroundKnowledge::default(),
    };
    let specs = [
        ("pc", DiscoverySpec::pc(params, "parcorr")),
        ("exact_dag_posterior", DiscoverySpec::exact_dag_posterior(BayesianDiscoverParams::default())),
    ];
    for (name, spec) in specs {
        let run = |design: Option<DeclaredSamplingDesign>| {
            let builder = Study::tabular(data.clone()).discovery(spec.clone(), vec![v(0), v(1), v(2)]);
            let builder = match design {
                Some(design) => builder.sampling_design(design),
                None => builder,
            };
            builder.build().and_then(|study| antecedent::CompiledExecution::run(&study, &ctx()).map(|_| ()))
        };
        run(None).unwrap_or_else(|error| panic!("{name}: {error}"));
        run(Some(DeclaredSamplingDesign::IndependentSubjectTrajectories))
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        for design in DEPENDENT {
            assert_refused(
                &format!("{name} {design:?}"),
                run(Some(design)).err(),
                "tests and scores rows as independent draws",
            );
        }
    }
}

#[test]
fn a_gaussian_glm_adjustment_fits_the_complete_rows_when_an_outcome_is_missing() {
    use antecedent_estimate::GlmAdjustmentAte;
    use antecedent_stats::GlmFamily;
    let columns: Vec<(&str, Vec<f64>)> = {
        let data = mediation_data();
        ["a", "m", "y", "z"]
            .iter()
            .enumerate()
            .map(|(index, name)| {
                let keep = vec![true; N];
                (*name, data.float64_selected(v(index as u32), &keep).unwrap().to_vec())
            })
            .collect()
    };
    let fit = |data: TabularData| {
        Study::tabular(data)
            .graph(mediation_dag())
            .query(AverageEffectQuery::binary_ate(v(0), v(2)))
            .estimator(
                GlmAdjustmentAte::new()
                    .with_family(GlmFamily::GaussianIdentity)
                    .with_bootstrap_replicates(0),
            )
            .refute(RefuteSuite::None)
            .build()
            .and_then(|study| study.run(&ctx()))
    };
    let missing_row = 17;
    let mut with_gap = columns.clone();
    with_gap[2].1[missing_row] = f64::NAN;
    let dropped: Vec<(&str, Vec<f64>)> = columns
        .iter()
        .map(|(name, values)| {
            let kept = values
                .iter()
                .enumerate()
                .filter(|(row, _)| *row != missing_row)
                .map(|(_, value)| *value)
                .collect();
            (*name, kept)
        })
        .collect();
    let table = |columns: &[(&str, Vec<f64>)]| {
        TabularData::from_f64_columns(columns.iter().map(|(n, c)| (*n, c.as_slice()))).unwrap()
    };
    let gap = fit(table(&with_gap)).unwrap();
    let rest = fit(table(&dropped)).unwrap();
    let (gap, rest) = (gap.estimate().as_effect().unwrap(), rest.estimate().as_effect().unwrap());
    assert!((gap.ate - rest.ate).abs() < 1e-9, "{} vs {}", gap.ate, rest.ate);
    assert!(
        (gap.uncertainty.analytic.se - rest.uncertainty.analytic.se).abs() < 1e-9,
        "{} vs {}",
        gap.uncertainty.analytic.se,
        rest.uncertainty.analytic.se
    );
}

#[test]
fn a_policy_value_refuses_a_missing_outcome_instead_of_dropping_the_row() {
    let n = POLICY_ROWS;
    let assignment: Vec<bool> = (0..n).map(|i| i % 2 == 1).collect();
    let mut outcomes: Vec<f64> = (0..n).map(|i| 1.0 + 2.0 * f64::from(assignment[i]) + (i % 5) as f64 / 10.0).collect();
    outcomes[5] = f64::NAN;
    let data = TabularData::from_f64_columns([("outcome", outcomes.as_slice())]).unwrap();
    let query = PolicyValueQuery {
        outcome: v(0), assignment: assignment.into(), propensity: Arc::from([0.5]), actions: (0..n).map(|i| i % 3 == 0).collect::<Vec<_>>().into(),
        reference: vec![false; n].into(), mu0: vec![1.0; n].into(), mu1: vec![3.0; n].into(), costs: Arc::from([0.0]), reference_costs: Arc::from([0.0]),
        evaluation_subject_ids: (0..n).map(|i| Arc::<str>::from(format!("e{i}"))).collect::<Vec<_>>().into(),
        disjoint_training_subjects: false, crossfit_fold_ownership_valid: true, global_constraints_present: false, multi_action: None,
        uplift_bins: Arc::from([]), uplift_bin_count: 0, uplift_training_subject_ids: Arc::from([]), regret: None };
    // The per-row predictions, propensities and actions are aligned with the outcome rows, so a
    // row cannot be dropped without shifting them: the missing outcome is refused.
    let error = Study::tabular(data)
        .query(CausalQuery::PolicyValue(query))
        .build()
        .and_then(|study| study.run(&ctx()))
        .err()
        .expect("a row with a missing outcome was scored");
    assert!(error.to_string().contains("outcomes, predictions, propensities, or costs are invalid"), "{error}");
}

/// The augmented two-period panel reads each subject's change between two columns. Its only
/// uncertainty is a whole-subject resample, so a declared dependent series (which would need a
/// block or lag structure over time) is refused with the designs that are honoured; the point-only
/// form publishes no standard error and so claims no dependence correction.
#[test]
fn the_augmented_panel_resample_refuses_a_declared_dependent_series() {
    use antecedent::PanelDidQuery;
    use antecedent_core::DidResamplingScheme;
    use std::sync::Arc;
    let pre = [10.0, 12.0, 13.0, 9.0, 15.0, 11.0, 10.5, 12.5];
    let post = [15.0, 17.0, 18.0, 11.0, 17.0, 13.0, 15.5, 14.5];
    let propensity = [0.5; 8];
    let prediction = [2.0; 8];
    let data = TabularData::from_f64_columns([
        ("pre", pre.as_slice()),
        ("post", post.as_slice()),
        ("p", propensity.as_slice()),
        ("m0", prediction.as_slice()),
    ])
    .unwrap();
    let ids = (0..8).map(|i| Arc::<str>::from(format!("s{i}"))).collect::<Vec<_>>();
    let run = |resample: bool, design: Option<DeclaredSamplingDesign>| {
        let mut query = PanelDidQuery::augmented_panel(
            v(1),
            v(0),
            v(2),
            v(3),
            [true, true, true, true, false, false, false, false],
            ids.clone(),
            ids.clone(),
            true,
        );
        if resample {
            query = query.with_resampling(DidResamplingScheme::ClusterBootstrap, 49);
        }
        let builder = Study::tabular(data.clone()).query(query);
        let builder = match design {
            Some(design) => builder.sampling_design(design),
            None => builder,
        };
        builder.build().and_then(|study| study.run(&ctx()))
    };
    // The augmented effect of the first four subjects against the rest, with the supplied
    // propensity and untreated-change columns, is 1.5.
    let point = run(false, None).unwrap();
    let fit = point.panel_did().unwrap();
    assert!((fit.effect - 1.5).abs() < 1e-12, "{}", fit.effect);
    assert_eq!(fit.uncertainty.as_ref(), "point_only_no_standard_error");
    for design in
        [None, Some(DeclaredSamplingDesign::IndependentSubjectTrajectories), Some(DeclaredSamplingDesign::ClusteredPanel)]
    {
        let resampled = run(true, design).unwrap();
        let fit = resampled.panel_did().unwrap();
        assert!((fit.effect - 1.5).abs() < 1e-12, "{design:?}: {}", fit.effect);
        assert!(fit.uncertainty.contains("whole_cluster_resample"), "{design:?}");
    }
    assert_refused(
        "augmented panel resample serial",
        run(true, Some(DeclaredSamplingDesign::DependentTimeSeries)).err(),
        "dependent_time_series",
    );
}
