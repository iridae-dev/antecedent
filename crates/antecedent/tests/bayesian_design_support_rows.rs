//! The graphless support rows of the Bayesian design posteriors, through the study facade.
//!
//! A Bayesian design posterior has no design section of its own, so a fit that matches its row
//! states it in the diagnostic `estimate.design_posterior.graphless_support`, whose fields
//! carry the row key and the observed support. Each test runs one route on a design that
//! meets its row, checks the retained program, the statement and a known value, exports the
//! result, forges every threshold below the row and another method (all refused by the
//! artifact check), and runs the same route below the row, where it states no support.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    reason = "simulation dimensions are small counts"
)]

#[path = "common/retained_graphless.rs"]
mod retained_graphless;

use std::sync::Arc;

use antecedent::{
    BayesianConfig, DesignPosteriorModel, EstimatorId, InferenceMode, InterferenceSpec,
    RandomizedEffectQuery, RefuteSuite, Study, StudyBuilder, StudyResult,
};
use antecedent_core::{
    AssignmentDesign, CausalQuery, ContinuousDoseResponseQuery, ExecutionContext, ExposureLevel,
    ExposureMapping, FixedGroupDosePolicy, InterferenceFunctional, InterferenceQuery,
    ObservationAssumption, RandomizationDesign, SurvivalFunctional, SurvivalQuery, VariableId,
};
use antecedent_data::{NetworkData, NetworkEdge, TabularData};
use antecedent_estimate::dose_basis_regression::ResidualPrior;
use antecedent_estimate::randomized_posterior::{ArmBetaPrior, ArmNormalPrior, ArmOutcomeModel};
use antecedent_estimate::switchback_posterior::{
    CarryoverDeclaration, PeriodAdjustment, SequenceDependence, SwitchbackModel, SwitchbackPrior,
};
use antecedent_graph::Dag;

const ROW_CODE: &str = "estimate.design_posterior.graphless_support";
const NO_RECORD_CODE: &str = "estimate.quasi.bayesian.unsupported";
const DRAWS: usize = 2_000;

/// A fixed-seed generator for the simulated designs.
struct Stream(u64);

impl Stream {
    fn uniform(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((z ^ (z >> 31)) >> 11) as f64 / (1_u64 << 53) as f64
    }

    fn normal(&mut self) -> f64 {
        let radius = (-2.0 * (1.0 - self.uniform()).ln()).sqrt();
        radius * (std::f64::consts::TAU * self.uniform()).cos()
    }

    /// `treated` of `n` units chosen uniformly without replacement.
    fn complete(&mut self, n: usize, treated: usize) -> Vec<bool> {
        let keys: Vec<f64> = (0..n).map(|_| self.uniform()).collect();
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| keys[a].total_cmp(&keys[b]));
        let mut assigned = vec![false; n];
        for &unit in &order[..treated] {
            assigned[unit] = true;
        }
        assigned
    }
}

fn v(raw: u32) -> VariableId {
    VariableId::from_raw(raw)
}

fn labels(n: usize, prefix: &str) -> Vec<Arc<str>> {
    (0..n).map(|i| Arc::<str>::from(format!("{prefix}-{i}"))).collect()
}

fn bayesian(model: Option<DesignPosteriorModel>, draws: usize) -> InferenceMode {
    let config = BayesianConfig::conjugate().n_draws(draws);
    InferenceMode::Bayesian(match model {
        Some(model) => config.design_model(model),
        None => config,
    })
}

fn has(result: &StudyResult, code: &str) -> bool {
    result.diagnostics.iter().any(|diagnostic| diagnostic.code.as_ref() == code)
}

/// Mean and equal-tailed 95% interval of the posterior's effect column.
fn effect(result: &StudyResult) -> (f64, f64, f64) {
    let posterior = result.posterior.as_ref().expect("a posterior");
    let column = posterior.effect_column().unwrap_or(0);
    let summaries = &posterior.summaries;
    (summaries.mean[column], summaries.q025[column], summaries.q975[column])
}

/// Run a design that meets its row: the retained program, the stated row, the exported
/// statement, every forged field and another method refused, and a result that states less
/// than its row accepted.
fn assert_supported_row(
    label: &str,
    builder: StudyBuilder,
    data: &TabularData,
    method: &str,
    forged: &[(&str, &str)],
) -> StudyResult {
    let ctx = ExecutionContext::for_tests(0xB1A5);
    let prepared = builder.build().unwrap().prepare(&ctx).unwrap();
    let result = prepared.estimate(data, &ctx).unwrap();
    retained_graphless::assert_retained_graphless_program(label, &prepared, &result, &ctx);
    let row = result
        .diagnostics
        .iter()
        .find(|diagnostic| diagnostic.code.as_ref() == ROW_CODE)
        .unwrap_or_else(|| panic!("{label}: the fit states its support row"));
    assert!(
        row.fields
            .iter()
            .any(|(name, value)| name.as_ref() == "method" && value.as_ref() == method),
        "{label}: {:?}",
        row.fields
    );
    assert!(!has(&result, NO_RECORD_CODE), "{label}: a supported fit keeps no unsupported note");

    let bytes = prepared.encode_contracted_result(&result, label, &ctx).unwrap();
    let (_, header, body) = antecedent_io::decode_analysis_result_artifact(&bytes).unwrap();
    let index = body
        .diagnostics
        .iter()
        .position(|diagnostic| diagnostic.code == ROW_CODE)
        .unwrap_or_else(|| panic!("{label}: the statement is exported"));
    let refused = |forged: &antecedent_io::analysis_result_artifact::AnalysisResultWire| {
        antecedent_io::encode_analysis_result_artifact(
            forged,
            header.variable_names.clone(),
            "forged-support",
        )
        .unwrap_err()
        .to_string()
    };
    for (name, value) in forged.iter().chain(&[("method", "another_method")]) {
        let mut wire = body.clone();
        let slot = wire.diagnostics[index]
            .fields
            .iter_mut()
            .find(|slot| slot.0 == *name)
            .unwrap_or_else(|| panic!("{label}: field {name}"));
        (*value).clone_into(&mut slot.1);
        assert!(refused(&wire).contains("does not admit"), "{label}: forged {name}={value}");
    }
    let mut unclaimed = body;
    unclaimed.diagnostics.remove(index);
    antecedent_io::encode_analysis_result_artifact(
        &unclaimed,
        header.variable_names.clone(),
        "unclaimed-support",
    )
    .unwrap();
    result
}

/// Run a design outside its row: the route executes and states no support.
fn assert_no_row(label: &str, builder: StudyBuilder, data: &TabularData) -> StudyResult {
    let ctx = ExecutionContext::for_tests(0xB1A5);
    let prepared = builder.build().unwrap().prepare(&ctx).unwrap();
    let result = prepared.estimate(data, &ctx).unwrap();
    assert!(!has(&result, ROW_CODE), "{label}: no support row is stated");
    result
}

// ------------------------------------------------------------------ randomized arm contrasts

fn weak_gaussian_arms() -> DesignPosteriorModel {
    DesignPosteriorModel::ArmOutcome {
        model: ArmOutcomeModel::Gaussian(ArmNormalPrior {
            mean: 0.0,
            pseudo_units: 0.1,
            variance_shape: 1.0,
            variance_scale: 1.0,
        }),
    }
}

fn uniform_bernoulli_arms() -> DesignPosteriorModel {
    DesignPosteriorModel::ArmOutcome {
        model: ArmOutcomeModel::Bernoulli(ArmBetaPrior { alpha: 1.0, beta: 1.0 }),
    }
}

/// A completely randomized trial of `n` units with half treated. Gaussian outcomes have unit
/// effect `1.4 + 0.2 x`; binary outcomes have success probabilities 0.35 and 0.55.
fn complete_trial(n: usize, binary: bool, seed: u64) -> (TabularData, RandomizedEffectQuery, f64) {
    let mut stream = Stream(seed);
    let assigned = stream.complete(n, n / 2);
    let covariate: Vec<f64> = (0..n).map(|i| (0.17 * i as f64).sin()).collect();
    let outcome: Vec<f64> = (0..n)
        .map(|i| {
            if binary {
                let success = if assigned[i] { 0.55 } else { 0.35 };
                f64::from(u8::from(stream.uniform() < success))
            } else {
                let baseline = 2.0 + 1.2 * covariate[i] + 0.5 * stream.normal();
                baseline + if assigned[i] { 1.4 + 0.2 * covariate[i] } else { 0.0 }
            }
        })
        .collect();
    let truth =
        if binary { 0.2 } else { covariate.iter().map(|x| 1.4 + 0.2 * x).sum::<f64>() / n as f64 };
    let data = TabularData::from_f64_columns([("outcome", outcome.as_slice())]).unwrap();
    let query = RandomizedEffectQuery::with_design(
        RandomizationDesign::Complete { treated_units: n / 2 },
        v(0),
        assigned,
        vec![0.5; n],
        labels(n, "unit"),
        labels(n, "row"),
        ("control", "treated"),
    );
    (data, query, truth)
}

fn arm_study(
    data: &TabularData,
    query: RandomizedEffectQuery,
    model: DesignPosteriorModel,
    superpopulation: bool,
    draws: usize,
) -> StudyBuilder {
    let builder = Study::tabular(data.clone()).query(query).inference(bayesian(Some(model), draws));
    if superpopulation {
        builder.estimator(EstimatorId::BayesianRandomizedSuperpopulation)
    } else {
        builder
    }
}

/// The finite-population posterior of a complete two-arm trial with Gaussian arms.
#[test]
fn supported_finite_population_posterior_round_trips_and_refuses_forgery() {
    let (data, query, truth) = complete_trial(300, false, 11);
    let result = assert_supported_row(
        "finite_population_gaussian_arms",
        arm_study(&data, query.clone(), weak_gaussian_arms(), false, DRAWS),
        &data,
        "gaussian_arm_finite_population_posterior",
        &[("rows", "149"), ("treated", "74"), ("control", "74")],
    );
    let (mean, lower, upper) = effect(&result);
    assert!((mean - truth).abs() < 0.4, "{mean} vs {truth}");
    assert!(lower < mean && mean < upper);

    // Fewer draws than the row was measured with, a smaller trial, and binary arms under the
    // finite-population estimand each run and state no support.
    let few = assert_no_row(
        "finite_population_few_draws",
        arm_study(&data, query.clone(), weak_gaussian_arms(), false, 1_000),
        &data,
    );
    assert!(has(&few, NO_RECORD_CODE));
    let (small, small_query, _) = complete_trial(100, false, 11);
    assert_no_row(
        "finite_population_small_trial",
        arm_study(&small, small_query, weak_gaussian_arms(), false, DRAWS),
        &small,
    );
    let (binary, binary_query, _) = complete_trial(300, true, 11);
    assert_no_row(
        "finite_population_binary_arms",
        arm_study(&binary, binary_query, uniform_bernoulli_arms(), false, DRAWS),
        &binary,
    );
}

/// The superpopulation posterior of a complete two-arm trial, with Gaussian and binary arms.
#[test]
fn supported_superpopulation_posterior_round_trips_and_refuses_forgery() {
    for (binary, model, method) in [
        (false, weak_gaussian_arms(), "gaussian_arm_superpopulation_posterior"),
        (true, uniform_bernoulli_arms(), "bernoulli_arm_superpopulation_posterior"),
    ] {
        let (data, query, truth) = complete_trial(300, binary, 13);
        let result = assert_supported_row(
            method,
            arm_study(&data, query, model.clone(), true, DRAWS),
            &data,
            method,
            &[("rows", "149"), ("treated", "74"), ("control", "74")],
        );
        let (mean, lower, upper) = effect(&result);
        assert!((mean - truth).abs() < 0.4, "{method}: {mean} vs {truth}");
        assert!(lower < mean && mean < upper);
        let (small, small_query, _) = complete_trial(100, binary, 13);
        let unsupported =
            assert_no_row(method, arm_study(&small, small_query, model, true, DRAWS), &small);
        assert!(has(&unsupported, NO_RECORD_CODE));
    }
}

// ------------------------------------------------------------------------------- switchback

const PERIODS: usize = 12;
const RHO: f64 = 0.4;

fn switchback_study(sequences: usize, periods: usize, seed: u64) -> (TabularData, StudyBuilder) {
    let mut stream = Stream(seed);
    let rows = sequences * periods;
    let (mut outcome, mut assigned, mut ids, mut slots) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for s in 0..sequences {
        let shared = stream.normal();
        for t in 0..periods {
            let treated = stream.uniform() < 0.5;
            let error = RHO.sqrt() * shared + (1.0 - RHO).sqrt() * stream.normal();
            outcome.push(0.5 + 0.7 * f64::from(u8::from(treated)) + error);
            assigned.push(treated);
            ids.push(Arc::<str>::from(format!("s{s}")));
            slots.push(Arc::<str>::from(format!("p{t}")));
        }
    }
    let data = TabularData::from_f64_columns([("outcome", outcome.as_slice())]).unwrap();
    let query = RandomizedEffectQuery::with_design(
        RandomizationDesign::Switchback { periods: slots.into() },
        v(0),
        assigned,
        vec![0.5; rows],
        ids,
        labels(rows, "row"),
        ("off", "on"),
    );
    let model = DesignPosteriorModel::Switchback {
        model: SwitchbackModel {
            carryover: CarryoverDeclaration::None,
            dependence: SequenceDependence::CompoundSymmetric { rho: RHO },
            periods: PeriodAdjustment::None,
            prior: SwitchbackPrior {
                effect_precision: 0.5,
                nuisance_precision: 0.01,
                variance_shape: 2.0,
                variance_scale: 1.0,
            },
        },
    };
    let builder = Study::tabular(data.clone()).query(query).inference(bayesian(Some(model), DRAWS));
    (data, builder)
}

/// The switchback posterior without carryover under declared compound-symmetric errors.
#[test]
fn supported_switchback_posterior_round_trips_and_refuses_forgery() {
    let (data, builder) = switchback_study(30, PERIODS, 17);
    let result = assert_supported_row(
        "switchback_direct_effect",
        builder,
        &data,
        "compound_symmetric_gaussian_sequence_posterior",
        &[
            ("blocks", "22"),
            ("rows", "275"),
            ("min_action_rows", "109"),
            ("min_probability", "0.4"),
            ("balanced_sequences", "false"),
        ],
    );
    let (mean, lower, upper) = effect(&result);
    assert!((mean - 0.7).abs() < 0.4, "{mean}");
    assert!(lower < mean && mean < upper);

    // Fewer sequences than the row, and sequences shorter than the measured length.
    let (few, few_builder) = switchback_study(12, PERIODS, 17);
    assert!(has(&assert_no_row("switchback_few_sequences", few_builder, &few), NO_RECORD_CODE));
    let (short, short_builder) = switchback_study(60, 6, 17);
    assert_no_row("switchback_short_sequences", short_builder, &short);
}

// ---------------------------------------------------------------------------- complier effect

fn complier_study(n: usize, compliance: f64, seed: u64) -> (TabularData, StudyBuilder) {
    let mut stream = Stream(seed);
    let assigned: Vec<bool> = (0..n).map(|_| stream.uniform() < 0.5).collect();
    let complier: Vec<bool> = (0..n).map(|_| stream.uniform() < compliance).collect();
    let received: Vec<bool> = (0..n).map(|i| assigned[i] && complier[i]).collect();
    let outcome: Vec<f64> = (0..n)
        .map(|i| {
            1.0 + 0.8 * f64::from(u8::from(complier[i]))
                + 2.0 * f64::from(u8::from(received[i]))
                + stream.normal()
        })
        .collect();
    let data = TabularData::from_f64_columns([("y", outcome.as_slice())]).unwrap();
    let query = RandomizedEffectQuery::bernoulli_itt(
        v(0),
        assigned,
        vec![0.5; n],
        labels(n, "account"),
        labels(n, "row"),
        ("control", "encouraged"),
    )
    .with_received_treatment(received);
    let builder = Study::tabular(data.clone()).query(query).inference(bayesian(None, DRAWS));
    (data, builder)
}

/// The Bayesian-bootstrap posterior of the complier effect under Bernoulli encouragement.
#[test]
fn supported_complier_posterior_round_trips_and_refuses_forgery() {
    let (data, builder) = complier_study(600, 0.7, 19);
    let result = assert_supported_row(
        "complier_effect",
        builder,
        &data,
        "bayesian_bootstrap_wald_ratio",
        &[("rows", "299"), ("min_expected_arm_units", "149.0"), ("min_probability", "0.4")],
    );
    let (mean, lower, upper) = effect(&result);
    assert!((mean - 2.0).abs() < 0.6, "{mean}");
    assert!(lower < mean && mean < upper);

    // A smaller trial, and a first stage weaker than the row was measured near.
    let (small, small_builder) = complier_study(200, 0.7, 19);
    assert_no_row("complier_small_trial", small_builder, &small);
    let (weak, weak_builder) = complier_study(600, 0.4, 19);
    assert_no_row("complier_weak_first_stage", weak_builder, &weak);
}

// ----------------------------------------------------------------------- piecewise survival

const HORIZON: f64 = 3.0;

fn survival_study(n: usize, seed: u64) -> (TabularData, StudyBuilder) {
    let mut stream = Stream(seed);
    let (mut duration, mut event, mut treatment) = (Vec::new(), Vec::new(), Vec::new());
    for _ in 0..n {
        let treated = stream.uniform() < 0.5;
        let hazard = if treated { 0.25 } else { 0.4 };
        let failure = -(1.0 - stream.uniform()).ln() / hazard;
        let censor = (-(1.0 - stream.uniform()).ln() / 0.15).min(HORIZON);
        duration.push(failure.min(censor).max(1e-9));
        event.push(f64::from(u8::from(failure <= censor)));
        treatment.push(f64::from(u8::from(treated)));
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
        tau: HORIZON,
        delayed_entry: None,
        known_censoring: None,
        observation_assumption: ObservationAssumption::IndependentGiven(Arc::from([])),
        functional: SurvivalFunctional::SurvivalAndRmst,
    });
    let model = DesignPosteriorModel::PiecewiseHazard {
        cuts: Arc::from([1.0, 2.0, 3.0].as_slice()),
        prior_shape: 1.0,
        prior_rate: 1.0,
    };
    let builder = Study::tabular(data.clone()).query(query).inference(bayesian(Some(model), DRAWS));
    (data, builder)
}

/// The piecewise-exponential posterior of the survival and restricted-mean differences.
#[test]
fn supported_piecewise_survival_posterior_round_trips_and_refuses_forgery() {
    let (data, builder) = survival_study(400, 23);
    let result = assert_supported_row(
        "piecewise_survival",
        builder,
        &data,
        "piecewise_exponential_gamma_posterior",
        &[("rows", "199"), ("treated", "79"), ("control", "79"), ("reported_intervals", "1")],
    );
    let row = result.posterior_row().expect("the posterior row");
    let truth = (-0.75_f64).exp() - (-1.2_f64).exp();
    assert!((row.mean[0] - truth).abs() < 0.15, "{} vs {truth}", row.mean[0]);
    assert!(row.lower[0] < row.mean[0] && row.mean[0] < row.upper[0]);

    let (small, small_builder) = survival_study(120, 23);
    assert!(has(&assert_no_row("survival_small_trial", small_builder, &small), NO_RECORD_CODE));
}

// ---------------------------------------------------------------------- continuous dose policy

const CURVES: [[f64; 3]; 2] = [[1.0, 0.5, 0.25], [2.0, -0.3, 0.4]];
const POLICY: [f64; 2] = [1.5, 2.5];
const REFERENCE: [f64; 2] = [2.5, 1.5];

/// Average of a group's quadratic curve over the Epanechnikov window of half-width one.
fn window_average(group: usize, target: f64) -> f64 {
    let [a, b, c] = CURVES[group];
    a + b * target + c * (target * target + 0.2)
}

fn dose_study(n: usize, seed: u64) -> (TabularData, StudyBuilder) {
    let mut stream = Stream(seed);
    let (mut dose, mut outcome, mut groups) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..n {
        let group = i % 2;
        let d = 4.0 * stream.uniform();
        let [a, b, c] = CURVES[group];
        dose.push(d);
        outcome.push(a + b * d + c * d * d + stream.normal());
        groups.push(Arc::<str>::from(if group == 0 { "A" } else { "B" }));
    }
    let density = vec![1.0; n];
    let data = TabularData::from_f64_columns([
        ("outcome", outcome.as_slice()),
        ("dose", dose.as_slice()),
        ("density", density.as_slice()),
    ])
    .unwrap();
    let rule = |doses: [f64; 2]| -> Arc<[(Arc<str>, f64)]> {
        [(Arc::<str>::from("A"), doses[0]), (Arc::<str>::from("B"), doses[1])].into_iter().collect()
    };
    let query = CausalQuery::ContinuousDoseResponse(ContinuousDoseResponseQuery {
        outcome: v(0),
        dose: v(1),
        dose_density: v(2),
        baseline_groups: groups.into(),
        target_doses: Arc::from([]),
        bandwidth: 1.0,
        min_local_support: 20,
        density_provenance: Arc::from("known"),
        fixed_policy: Some(FixedGroupDosePolicy {
            policy_doses: rule(POLICY),
            reference_doses: rule(REFERENCE),
        }),
    });
    let model = DesignPosteriorModel::DosePolynomial {
        support: (0.0, 4.0),
        coefficient_scales: Arc::from([10.0, 10.0, 10.0].as_slice()),
        residual: ResidualPrior::InverseGamma { shape: 1.0, scale: 1.0 },
    };
    let builder = Study::tabular(data.clone()).query(query).inference(bayesian(Some(model), DRAWS));
    (data, builder)
}

/// The quadratic conjugate posterior of the fixed group-to-dose policy value.
#[test]
fn supported_dose_policy_posterior_round_trips_and_refuses_forgery() {
    let (data, builder) = dose_study(400, 29);
    let result = assert_supported_row(
        "dose_policy",
        builder,
        &data,
        "quadratic_group_dose_conjugate_posterior",
        &[
            ("rows", "199"),
            ("min_group_rows", "99"),
            ("min_local_rows", "34"),
            ("reported_intervals", "2"),
        ],
    );
    let row = result.posterior_row().expect("the posterior row");
    let truth = 0.5 * window_average(0, POLICY[0]) + 0.5 * window_average(1, POLICY[1]);
    assert!((row.mean[0] - truth).abs() < 0.4, "{} vs {truth}", row.mean[0]);
    assert!(row.lower[0] < row.mean[0] && row.mean[0] < row.upper[0]);

    let (small, small_builder) = dose_study(120, 29);
    assert!(has(&assert_no_row("dose_policy_small_trial", small_builder, &small), NO_RECORD_CODE));
}

// ------------------------------------------------------------------ cluster reference posterior

fn cluster_study(clusters: usize, seed: u64) -> (TabularData, StudyBuilder) {
    let mut stream = Stream(seed);
    let treated = stream.complete(clusters, clusters / 2);
    let (mut outcome, mut assignment) = (Vec::new(), Vec::new());
    for &arm in &treated {
        let shared = stream.normal();
        for _ in 0..2 {
            outcome.push(shared + 0.5 * stream.normal() + 3.0 * f64::from(u8::from(arm)));
            assignment.push(arm);
        }
    }
    let units = 2 * clusters;
    let edges: Vec<NetworkEdge> = (0..clusters as u64)
        .flat_map(|c| {
            [NetworkEdge::new(2 * c, 2 * c + 1, 1.0), NetworkEdge::new(2 * c + 1, 2 * c, 1.0)]
        })
        .collect();
    let data = TabularData::from_f64_columns([("y", outcome.as_slice())]).unwrap();
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
    let network = NetworkData::try_new(data.clone(), edges).unwrap();
    let builder = Study::tabular(data.clone())
        .graph(Dag::with_variables(1))
        .query(CausalQuery::Interference(query))
        .interference(InterferenceSpec { network, assignment: Arc::from(assignment) })
        .inference(bayesian(None, DRAWS))
        .refute(RefuteSuite::None);
    (data, builder)
}

/// The Student-t reference posterior of the cluster total effect with few clusters per arm.
#[test]
fn supported_cluster_reference_posterior_round_trips_and_refuses_forgery() {
    let (data, builder) = cluster_study(24, 31);
    let result = assert_supported_row(
        "cluster_reference_posterior",
        builder,
        &data,
        "cluster_student_reference_posterior",
        &[("treated", "8"), ("control", "8"), ("assignment_unit", "unit")],
    );
    let (mean, lower, upper) = effect(&result);
    assert!((mean - 3.0).abs() < 1.5, "{mean}");
    assert!(lower < mean && mean < upper);

    // Eight clusters per arm are below the row; fifty per arm run the Dirichlet law, which
    // has no row.
    let (few, few_builder) = cluster_study(16, 31);
    assert_no_row("cluster_reference_few_clusters", few_builder, &few);
    let (many, many_builder) = cluster_study(100, 31);
    assert_no_row("cluster_dirichlet_law", many_builder, &many);
}
