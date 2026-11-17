//! Source-fixed repeated-sampling design for a fitted dependent-indicator table.
//!
//! The mechanism law is declared and fixed. Independent labelled validation
//! samples identify the two rows of a complete joint indicator-error table.
//! Each replicate scores the route's own equal-tail posterior interval for
//! `P(Y=1 | assay_a=1, assay_b=1)` against its analytic truth. This test is
//! deliberately ignored until the new coverage family is registered and run;
//! the runtime continues to report `not_measured`.

mod common;

use antecedent::analysis::fitted_dependent_measurement::FittedDependentMeasurementRequest;
use antecedent::analysis::observation_query::{
    DependentMeasuredObservationRequest, DependentMeasurementWire,
};
use antecedent::{
    DependenceInput, MechanismCompositionRequest, MechanismInput, MechanismSource,
    StudyBuilder,
};
use antecedent_contract::{
    MechanismDeclaration, MechanismOriginKind, QuantityRole, ScientificQuantity,
};
use antecedent_core::ExecutionContext;
use antecedent_model::compose::{Domain, TabularRow};
use common::calibration::{
    Construction, CoverageTally, REPORTED_LEVEL, RecordKey, ScopeFacts, grid_n,
    grid_seed, map_replicates, n_sim, uniform,
};

const TEST: &str = "fitted_dependent_measurement_joint_error_l95";
const DGP: &str = "binary_construct_two_dependent_indicators_labelled_validation";
const INTERVAL: &str = "fitted_joint_error_dirichlet_equal_tailed";
const DRAWS: u32 = 256;
const TRUE_ROWS: [[f64; 4]; 2] = [
    [0.7, 0.1, 0.1, 0.1],
    [0.05, 0.05, 0.05, 0.85],
];

fn quantity(variable: &str) -> ScientificQuantity {
    ScientificQuantity {
        variable_id: variable.into(), variable_name: variable.into(),
        role: if variable == "Y" { QuantityRole::Outcome } else { QuantityRole::Covariate },
        units: "category".into(), population_id: "pop".into(),
        regime_id: "observational".into(), horizon: 0,
        functional_id: "realization".into(), conditioning: Vec::new(),
        transform_id: "identity".into(),
    }
}

fn table(variable: &str, parent: Option<&str>, probabilities: &[[f64; 2]]) -> MechanismInput {
    let levels = vec!["0".to_owned(), "1".to_owned()];
    let parents = parent.map_or_else(Vec::new, |name| vec![name]);
    MechanismInput {
        declaration: MechanismDeclaration {
            output: quantity(variable), parents: parents.iter().map(|name| quantity(name)).collect(),
            conditioning_variable_ids: parents.iter().map(|name| (*name).to_owned()).collect(),
            origin: MechanismOriginKind::Declared,
            artifact_id: format!("declared:{variable}"),
            evidence_id: format!("assumed:{variable}"), latent_causes: Vec::new(),
        },
        source: MechanismSource::DeclaredTable {
            levels: levels.clone(),
            rows: probabilities.iter().enumerate().map(|(index, row)| TabularRow {
                parent_levels: parent.map_or_else(Vec::new, |name| {
                    vec![(name.to_owned(), levels[index].clone())]
                }),
                probabilities: vec![("0".into(), row[0]), ("1".into(), row[1])],
            }).collect(),
        },
        output_support: Domain::Levels(levels.clone()),
        parent_domains: parents.iter().map(|name| ((*name).to_owned(), Domain::Levels(levels.clone()))).collect(),
        uncertainty: None, latent_loadings: Vec::new(),
    }
}

fn observation() -> DependentMeasuredObservationRequest {
    DependentMeasuredObservationRequest {
        outcome: "Y".into(), observed: Vec::new(),
        measured: vec![("assay_a".into(), "1".into()), ("assay_b".into(), "1".into())],
        interventions: Vec::new(),
        error_law: DependentMeasurementWire {
            construct: "C".into(), population_id: "pop".into(),
            regime_id: "observational".into(),
            evidence_id: "labelled-joint-validation".into(),
            indicator_ids: vec!["assay_a".into(), "assay_b".into()],
            indicator_levels: vec![vec!["0".into(), "1".into()], vec!["0".into(), "1".into()]],
            construct_levels: vec!["0".into(), "1".into()],
            probability_bits: Vec::new(),
        },
    }
}

fn replicate(seed: u64, row_n: usize) -> Option<(f64, f64)> {
    let mut next = uniform(seed, 0xF17E_D001);
    let mut counts = vec![0_u64; 8];
    for (row, probabilities) in TRUE_ROWS.iter().enumerate() {
        for _ in 0..row_n {
            let draw = next();
            let cell = probabilities.iter().scan(0.0, |sum, probability| {
                *sum += probability;
                Some(*sum)
            }).position(|upper| draw < upper).unwrap_or(3);
            counts[row * 4 + cell] += 1;
        }
    }
    let mechanisms = vec![
        table("C", None, &[[0.9, 0.1]]),
        table("Y", Some("C"), &[[0.9, 0.1], [0.3, 0.7]]),
    ];
    let request = MechanismCompositionRequest {
        target_population_id: "pop".into(), target_regime_id: "observational".into(),
        mechanisms, edges: vec![("C".into(), "Y".into())],
        admissions: Vec::new(), dependence: DependenceInput::Unknown,
        admit_declared_mechanisms: true,
    };
    let ctx = ExecutionContext::for_tests(seed);
    let prepared = StudyBuilder::mechanism_composition(request, &ctx).ok()?;
    let plan = prepared.compile_fitted_dependent_measurement(FittedDependentMeasurementRequest {
        observation: observation(), counts, prior_alpha: 1.0,
        draws: DRAWS, seed: seed ^ 0xA55A_1234,
    }).ok()?;
    let report = plan.execute(&ctx).ok()?;
    let interval = report.nominal_interval("1", REPORTED_LEVEL).ok()?;
    assert_eq!(interval.calibration, "not_measured");
    Some((interval.lower, interval.upper))
}

#[test]
#[ignore = "source-only design; new coverage family has not been earned"]
fn fitted_dependent_measurement_joint_error_l95() {
    let row_n = grid_n(300);
    let truth_c1 = (0.1 * TRUE_ROWS[1][3])
        / (0.9 * TRUE_ROWS[0][3] + 0.1 * TRUE_ROWS[1][3]);
    let truth = 0.1 * (1.0 - truth_c1) + 0.7 * truth_c1;
    let construction = Construction {
        query: "conditional_outcome_probability_given_joint_measurement".into(),
        graph_class: "declared_finite_mechanisms".into(),
        structure: "fixed".into(), modality: "tabular".into(),
        inference: "bayesian".into(),
        estimator: "fitted_dependent_measurement_v1".into(),
        interval_method: INTERVAL.into(), se_kind: "posterior_draw_quantile".into(),
        dependence: "joint_indicator_errors".into(),
        posterior: "symmetric_dirichlet_rows_alpha_1".into(),
        functional: "P_Y_1_given_assay_a_1_assay_b_1".into(),
        identification: "model_conditional".into(),
        reported_level: REPORTED_LEVEL,
    };
    let mut tally = CoverageTally::for_record(RecordKey {
        test: TEST, dgp: DGP, interval: INTERVAL,
    }, REPORTED_LEVEL);
    for interval in map_replicates(n_sim(), |rep| {
        replicate(grid_seed(0xD0C0_0001 + rep), row_n)
    }) {
        if let Some(interval) = interval {
            tally.bind_validation(&construction, ScopeFacts {
                row_count: (2 * row_n) as u64, replicates_ok: None,
                posterior_draws: Some(DRAWS), unidentified_mass: 0.0,
            }, "not_applicable");
            tally.record(Some(interval), truth);
        } else {
            tally.skip();
        }
    }
    tally.assert();
}
