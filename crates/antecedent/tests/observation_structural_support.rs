//! Support decision and independent known-truth checks for the structural observation families:
//! joint and dependent measured observation, the fitted dependent-error table, the latent soft
//! intervention, the event process and the declared-shift MNAR candidate.
//!
//! Each family's plan view reports the unit its route runs under, and that decision equals the
//! status the build embeds for the unit. The fitted joint-error table is checked against an
//! independent two-dimensional Beta quadrature of the posterior of the answered probability:
//! the posterior rows are independent Dirichlet laws, so the answer depends only on the joint
//! "both positive" cell of each construct row, whose marginals are Beta.

use antecedent::Study;
use antecedent::analysis::events::{Arrival, EventHistory, EventWindow, HawkesParameters};
use antecedent::analysis::fitted_dependent_measurement::FittedDependentMeasurementRequest;
use antecedent::analysis::latent_soft_intervention::LatentSoftInterventionRequest;
use antecedent::analysis::observation_query::{
    DependentMeasuredObservationRequest, DependentMeasurementWire,
};
use antecedent::planner::PlanView;
use antecedent::{
    DependenceInput, MechanismCompositionRequest, MechanismInput, MechanismSource,
    PreparedMechanismComposition, StudyBuilder,
};
use antecedent_contract::{
    MechanismDeclaration, MechanismOriginKind, QuantityRole, ScientificQuantity,
};
use antecedent_core::mnar_fixed_shift::{FixedShiftSource, SamplingPremise, SourceRow};
use antecedent_core::{CausalRng, ExecutionContext};
use antecedent_model::compose::{Domain, TabularRow};

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(1)
}

/// The status the build embeds for the capability unit `id`, read from the snapshot directly.
fn embedded_status(id: &str) -> String {
    let snapshot: serde_json::Value =
        serde_json::from_str(antecedent_io_core::graphless_support::UNIT_STATUS)
            .expect("the embedded unit status decodes");
    snapshot["capabilities"]
        .as_array()
        .expect("capabilities list")
        .iter()
        .find(|unit| unit["id"] == id)
        .unwrap_or_else(|| panic!("{id} is a listed capability unit"))["status"]
        .as_str()
        .expect("status text")
        .to_owned()
}

/// The plan view reports the unit a route runs under and the status the build embeds for it.
fn assert_unit_support(view: &PlanView, unit: &str) {
    let support = view.unit_support().expect("the route is bound to a unit");
    assert_eq!(support.unit(), unit);
    assert_eq!(support.status().as_str(), embedded_status(unit));
    assert_eq!(support.supported(), embedded_status(unit) == "supported");
    if support.supported() {
        assert!(support.missing_evidence().is_empty());
    }
}

fn quantity(variable: &str) -> ScientificQuantity {
    ScientificQuantity {
        variable_id: variable.into(),
        variable_name: variable.into(),
        role: if variable == "Y" { QuantityRole::Outcome } else { QuantityRole::Covariate },
        units: "category".into(),
        population_id: "pop".into(),
        regime_id: "observational".into(),
        horizon: 0,
        functional_id: "realization".into(),
        conditioning: Vec::new(),
        transform_id: "identity".into(),
    }
}

fn table(variable: &str, parent: Option<&str>, probabilities: &[[f64; 2]]) -> MechanismInput {
    let levels = vec!["0".to_owned(), "1".to_owned()];
    let parents = parent.map_or_else(Vec::new, |name| vec![name]);
    MechanismInput {
        declaration: MechanismDeclaration {
            output: quantity(variable),
            parents: parents.iter().map(|name| quantity(name)).collect(),
            conditioning_variable_ids: parents.iter().map(|name| (*name).to_owned()).collect(),
            origin: MechanismOriginKind::Declared,
            artifact_id: format!("declared:{variable}"),
            evidence_id: format!("assumed:{variable}"),
            latent_causes: Vec::new(),
        },
        source: MechanismSource::DeclaredTable {
            levels: levels.clone(),
            rows: probabilities
                .iter()
                .enumerate()
                .map(|(index, row)| TabularRow {
                    parent_levels: parent
                        .map(|name| (name.to_owned(), index.to_string()))
                        .into_iter()
                        .collect(),
                    probabilities: vec![("0".to_owned(), row[0]), ("1".to_owned(), row[1])],
                })
                .collect(),
        },
        output_support: Domain::Levels(levels.clone()),
        parent_domains: parent
            .into_iter()
            .map(|name| (name.to_owned(), Domain::Levels(levels.clone())))
            .collect(),
        uncertainty: None,
        latent_loadings: Vec::new(),
    }
}

/// Construct `C` with prevalence 0.1 and outcome `Y` with `P(Y=1 | C) = 0.1, 0.7`.
fn construct_model() -> PreparedMechanismComposition {
    let mechanisms =
        vec![table("C", None, &[[0.9, 0.1]]), table("Y", Some("C"), &[[0.9, 0.1], [0.3, 0.7]])];
    let edges = vec![("C".to_owned(), "Y".to_owned())];
    StudyBuilder::mechanism_composition(
        MechanismCompositionRequest {
            target_population_id: "pop".into(),
            target_regime_id: "observational".into(),
            mechanisms,
            edges,
            admissions: Vec::new(),
            dependence: DependenceInput::Unknown,
            admit_declared_mechanisms: true,
        },
        &ctx(),
    )
    .expect("the construct model composes")
}

fn joint_error_law(evidence: &str, bits: Vec<u64>) -> DependentMeasurementWire {
    DependentMeasurementWire {
        construct: "C".into(),
        population_id: "pop".into(),
        regime_id: "observational".into(),
        evidence_id: evidence.into(),
        indicator_ids: vec!["assay_a".into(), "assay_b".into()],
        indicator_levels: vec![vec!["0".into(), "1".into()], vec!["0".into(), "1".into()]],
        construct_levels: vec!["0".into(), "1".into()],
        probability_bits: bits,
    }
}

fn both_positive(error_law: DependentMeasurementWire) -> DependentMeasuredObservationRequest {
    DependentMeasuredObservationRequest {
        outcome: "Y".into(),
        observed: Vec::new(),
        measured: vec![("assay_a".into(), "1".into()), ("assay_b".into(), "1".into())],
        interventions: Vec::new(),
        error_law,
    }
}

/// `P(Y=1 | both indicators positive)` when `a = P(11 | C=0)` and `b = P(11 | C=1)`: the
/// construct posterior is `0.1 b / (0.9 a + 0.1 b)` and `Y` mixes `0.1` and `0.7` over it.
fn answer(a: f64, b: f64) -> f64 {
    let construct = 0.1 * b / (0.9 * a + 0.1 * b);
    0.1 * (1.0 - construct) + 0.7 * construct
}

/// `ln Gamma(x)` by the Lanczos approximation (g = 7, n = 9), accurate to about 1e-14.
fn ln_gamma(x: f64) -> f64 {
    const COEFFICIENTS: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    let x = x - 1.0;
    let t = x + 7.5;
    let sum = COEFFICIENTS[0]
        + COEFFICIENTS[1..].iter().enumerate().map(|(i, c)| c / (x + 1.0 + i as f64)).sum::<f64>();
    0.5 * (std::f64::consts::TAU).ln() + (x + 0.5) * t.ln() - t + sum.ln()
}

/// Midpoint-grid weights of the `Beta(alpha, beta)` density, normalised on the grid.
fn beta_weights(alpha: f64, beta: f64, points: usize) -> Vec<f64> {
    let log_norm = ln_gamma(alpha + beta) - ln_gamma(alpha) - ln_gamma(beta);
    let mut weights: Vec<f64> = (0..points)
        .map(|i| {
            let x = (i as f64 + 0.5) / points as f64;
            (log_norm + (alpha - 1.0) * x.ln() + (beta - 1.0) * (1.0 - x).ln()).exp()
        })
        .collect();
    let total: f64 = weights.iter().sum();
    weights.iter_mut().for_each(|w| *w /= total);
    weights
}

#[test]
fn the_gamma_quadrature_helper_reproduces_known_beta_moments() {
    // Beta(11, 93): mean 11/104 and variance 11 * 93 / (104^2 * 105).
    let weights = beta_weights(11.0, 93.0, 4000);
    let mean: f64 = weights.iter().enumerate().map(|(i, w)| w * (i as f64 + 0.5) / 4000.0).sum();
    let second: f64 =
        weights.iter().enumerate().map(|(i, w)| w * ((i as f64 + 0.5) / 4000.0).powi(2)).sum();
    assert!((mean - 11.0 / 104.0).abs() < 1e-6, "{mean}");
    assert!(((second - mean * mean) - 11.0 * 93.0 / (104.0_f64.powi(2) * 105.0)).abs() < 1e-6);
}

#[test]
fn fitted_joint_error_table_draws_match_the_dirichlet_posterior_quadrature() {
    // Validation counts in construct-row order; cells in `[00, 01, 10, 11]` order. With the
    // symmetric Dirichlet(1) prior each row's posterior is Dirichlet(count + 1), so the "11"
    // cell of row 0 is Beta(10 + 1, 100 + 4 - 11) and of row 1 is Beta(85 + 1, 100 + 4 - 86).
    let counts = vec![70_u64, 10, 10, 10, 5, 5, 5, 85];
    let prepared = construct_model();
    let draws = 1000_u32;
    let request = FittedDependentMeasurementRequest {
        observation: both_positive(joint_error_law("labelled-joint-validation", Vec::new())),
        counts,
        prior_alpha: 1.0,
        draws,
        seed: 20_260,
    };
    let plan = prepared.compile_fitted_dependent_measurement(request).expect("fitted compiler");
    assert_eq!(plan.plan_view().route().as_str(), "FittedDependentMeasuredObservation");
    let report = plan.execute(&ctx()).expect("posterior draws");
    let index = report.levels.iter().position(|level| level == "1").expect("outcome level 1");
    let values: Vec<f64> = report.posterior_draws.iter().map(|row| row[index]).collect();
    assert_eq!(values.len(), draws as usize);

    // Independent reference: E and Var of answer(a, b) over independent Beta marginals.
    let points = 1200;
    let (wa, wb) = (beta_weights(11.0, 93.0, points), beta_weights(86.0, 18.0, points));
    let (mut first, mut second) = (0.0, 0.0);
    for (i, pa) in wa.iter().enumerate() {
        let a = (i as f64 + 0.5) / points as f64;
        for (j, pb) in wb.iter().enumerate() {
            let value = answer(a, (j as f64 + 0.5) / points as f64);
            first += pa * pb * value;
            second += pa * pb * value * value;
        }
    }
    let variance = second - first * first;
    let n = f64::from(draws);
    let mean = values.iter().sum::<f64>() / n;
    let spread = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0);
    // Five Monte Carlo standard errors of the mean; the spread within 12 percent, about five
    // standard errors of a sample variance of this many draws.
    assert!(
        (mean - first).abs() <= 5.0 * (variance / n).sqrt(),
        "draw mean {mean} against quadrature {first} (sd {})",
        variance.sqrt()
    );
    assert!((spread / variance - 1.0).abs() <= 0.12, "draw variance {spread} against {variance}");

    // The plug-in is the answer at the posterior-mean cells, 11/104 and 86/104.
    assert!((report.plug_in[index] - answer(11.0 / 104.0, 86.0 / 104.0)).abs() < 1e-12);
}

#[test]
fn the_fitted_dependent_table_reports_the_unit_its_route_runs_under() {
    let prepared = construct_model();
    let plan = prepared
        .compile_fitted_dependent_measurement(FittedDependentMeasurementRequest {
            observation: both_positive(joint_error_law("labelled-joint-validation", Vec::new())),
            counts: vec![70, 10, 10, 10, 5, 5, 5, 85],
            prior_alpha: 1.0,
            draws: 4,
            seed: 7,
        })
        .expect("fitted compiler");
    assert_unit_support(&plan.plan_view(), "observation_query.finite_law_conditioning");
}

#[test]
fn the_dependent_observation_reports_the_unit_its_route_runs_under() {
    let bits = [0.7_f64, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.85].map(f64::to_bits).to_vec();
    let plan = construct_model()
        .compile_dependent_measured_observation(&both_positive(joint_error_law(
            "joint-assay-validation",
            bits,
        )))
        .expect("dependent compiler");
    assert_unit_support(&plan.plan_view(), "observation_query.finite_law_conditioning");
    let claim = plan.execute(&ctx()).expect("the supported unit runs");
    let posterior = (0.1 * 0.85) / (0.9 * 0.1 + 0.1 * 0.85);
    let level_one = claim
        .conditional_levels()
        .and_then(|levels| levels.into_iter().find(|(level, _)| level == "1"))
        .expect("level 1")
        .1;
    assert!((level_one - (0.1 * (1.0 - posterior) + 0.7 * posterior)).abs() < 1e-12);
}

fn standard_normal(rng: &mut CausalRng) -> f64 {
    let radial = (-2.0 * (1.0 - rng.next_f64()).ln()).sqrt();
    radial * (std::f64::consts::TAU * rng.next_f64()).cos()
}

#[test]
fn the_latent_soft_intervention_reports_the_unit_its_route_runs_under() {
    let mut rng = CausalRng::from_seed(0x4c53_5550_5052);
    let mut rows = |shift: f64| -> Vec<Vec<f64>> {
        (0..1_200)
            .map(|_| {
                let x = standard_normal(&mut rng);
                let a = 0.4 * x + 0.45 * standard_normal(&mut rng) + shift;
                let b = -0.3 * x + 0.45 * standard_normal(&mut rng);
                let y = a + b + 0.6 * standard_normal(&mut rng);
                vec![x, a, b, y]
            })
            .collect()
    };
    let (baseline, action) = (rows(0.0), rows(3.0));
    let request = LatentSoftInterventionRequest {
        variables: ["x", "path_a", "path_b", "outcome"].map(String::from).to_vec(),
        edges: vec![(0, 1), (0, 2), (1, 3), (2, 3)],
        context: vec![0],
        candidates: vec![1, 2, 3],
        baseline,
        action,
        population: "independent_gaussian_trial".into(),
        randomized: true,
        version_exogenous_given_context: true,
        seed: 73,
        refits: 0,
        search_operations: 41_000,
    };
    let prepared =
        Study::prepare_latent_soft_intervention(request, &ctx()).expect("the study compiles");
    assert_unit_support(&prepared.plan_view(), "estimate.latent_soft_intervention");
}

#[test]
fn the_event_process_reports_the_unit_its_route_runs_under() {
    let history = EventHistory {
        catalog: vec!["mention".into()],
        time_unit: "hour".into(),
        origin: "origin".into(),
        source: "news".into(),
        empty_origin: 0.0,
        window: EventWindow { start: 0.0, end: 2.0 },
        left: vec![],
        events: vec![Arrival { time: 0.5, kind: 0 }, Arrival { time: 1.5, kind: 0 }],
        gaps: vec![],
    };
    let parameters =
        HawkesParameters { baseline: vec![2.0], branching: vec![0.0], decay: vec![1.0] };
    let prepared = Study::prepare_event_likelihood(history, parameters, &ctx()).expect("compiles");
    assert_unit_support(&prepared.plan_view(), "forecast.event_process");
    // A Poisson process of rate 2 on [0, 2] with two events: 2 ln 2 - 4.
    let value = prepared.execute(&ctx()).expect("the supported unit runs");
    assert!((value - (2.0 * 2.0_f64.ln() - 4.0)).abs() < 1e-13);
}

#[test]
fn the_declared_shift_candidate_reports_the_unit_its_route_runs_under() {
    let rows: Vec<SourceRow> = (0..40_u64)
        .map(|index| SourceRow {
            subject_id: index,
            treated: index >= 20,
            outcome: (index % 5 != 0).then(|| (index as f64 * 0.37).sin() + (index % 7) as f64),
        })
        .collect();
    let source = FixedShiftSource::new(
        "support_population".into(),
        0,
        1,
        SamplingPremise::IndependentRandomizedArms,
        0.5,
        rows,
    )
    .expect("a valid declared-shift source");
    let prepared = Study::prepare_mnar_fixed_shift(source, &ctx()).expect("the study compiles");
    assert_unit_support(&prepared.plan_view(), "mnar_fixed_shift.compiled_candidate");
}
