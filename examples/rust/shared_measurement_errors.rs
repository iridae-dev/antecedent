// SPDX-License-Identifier: MIT OR Apache-2.0
//! Account for two measurements that can be wrong together.
//!
//! Joint labelled validation counts let two positive readings share measurement error.

use antecedent::analysis::fitted_dependent_measurement::FittedDependentMeasurementRequest;
use antecedent::analysis::observation_query::{
    DependentMeasuredObservationRequest, DependentMeasurementWire,
};
use antecedent::{
    DependenceInput, MechanismCompositionRequest, MechanismInput, MechanismSource,
    PreparedMechanismComposition, StudyBuilder,
};
use antecedent_contract::{
    MechanismDeclaration, MechanismOriginKind, QuantityRole, ScientificQuantity,
};
use antecedent_core::ExecutionContext;
use antecedent_model::compose::{Domain, TabularRow};

const TOL: f64 = 1e-12;

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(1)
}

fn near(a: f64, b: f64) {
    assert!((a - b).abs() <= TOL, "{a} vs {b}");
}

fn pair(variable: &str, level: &str) -> (String, String) {
    (variable.to_owned(), level.to_owned())
}

fn quantity(variable: &str) -> ScientificQuantity {
    ScientificQuantity {
        variable_id: variable.into(),
        variable_name: format!("label of {variable}"),
        role: match variable {
            "S" => QuantityRole::Treatment,
            "Y" => QuantityRole::Outcome,
            _ => QuantityRole::Covariate,
        },
        units: "category".into(),
        population_id: "pop".into(),
        regime_id: "observational".into(),
        horizon: 0,
        functional_id: "realization".into(),
        conditioning: Vec::new(),
        transform_id: "identity".into(),
    }
}

fn table_input(variable: &str, parents: &[&str], rows: &[(&[&str], [f64; 2])]) -> MechanismInput {
    let levels = vec!["0".to_owned(), "1".to_owned()];
    MechanismInput {
        declaration: MechanismDeclaration {
            output: quantity(variable),
            parents: parents.iter().map(|p| quantity(p)).collect(),
            conditioning_variable_ids: parents.iter().map(|p| (*p).to_owned()).collect(),
            origin: MechanismOriginKind::Declared,
            artifact_id: format!("declared:{variable}"),
            evidence_id: format!("assumed:{variable}"),
            latent_causes: Vec::new(),
        },
        source: MechanismSource::DeclaredTable {
            levels: levels.clone(),
            rows: rows
                .iter()
                .map(|(config, p)| TabularRow {
                    parent_levels: parents
                        .iter()
                        .zip(config.iter())
                        .map(|(parent, l)| ((*parent).to_owned(), (*l).to_owned()))
                        .collect(),
                    probabilities: vec![("0".to_owned(), p[0]), ("1".to_owned(), p[1])],
                })
                .collect(),
        },
        output_support: Domain::Levels(levels.clone()),
        parent_domains: parents
            .iter()
            .map(|p| ((*p).to_owned(), Domain::Levels(levels.clone())))
            .collect(),
        uncertainty: None,
        latent_loadings: Vec::new(),
    }
}

fn request_of(mechanisms: Vec<MechanismInput>) -> MechanismCompositionRequest {
    let mut edges = Vec::new();
    for m in &mechanisms {
        for p in &m.declaration.parents {
            edges.push((p.variable_id.clone(), m.declaration.output.variable_id.clone()));
        }
    }
    MechanismCompositionRequest {
        target_population_id: "pop".into(),
        target_regime_id: "observational".into(),
        mechanisms,
        edges,
        admissions: Vec::new(),
        dependence: DependenceInput::Unknown,
        admit_declared_mechanisms: true,
    }
}

fn construct_model() -> PreparedMechanismComposition {
    StudyBuilder::mechanism_composition(
        request_of(vec![
            table_input("C", &[], &[(&[], [0.9, 0.1])]),
            table_input("Y", &["C"], &[(&["0"], [0.9, 0.1]), (&["1"], [0.3, 0.7])]),
        ]),
        &ctx(),
    )
    .unwrap()
}

fn main() {
    let prepared = construct_model();
    let observation = DependentMeasuredObservationRequest {
        outcome: "Y".into(),
        observed: Vec::new(),
        measured: vec![pair("assay_a", "1"), pair("assay_b", "1")],
        interventions: Vec::new(),
        error_law: DependentMeasurementWire {
            construct: "C".into(),
            population_id: "pop".into(),
            regime_id: "observational".into(),
            evidence_id: "labelled-joint-validation".into(),
            indicator_ids: vec!["assay_a".into(), "assay_b".into()],
            indicator_levels: vec![vec!["0".into(), "1".into()], vec!["0".into(), "1".into()]],
            construct_levels: vec!["0".into(), "1".into()],
            probability_bits: Vec::new(),
        },
    };
    let request = FittedDependentMeasurementRequest {
        observation,
        counts: vec![70, 10, 10, 10, 5, 5, 5, 85],
        prior_alpha: 1.0,
        draws: 16,
        seed: 91,
    };
    let plan = prepared
        .compile_fitted_dependent_measurement(request)
        .expect("fitted measurement compiler");
    assert_eq!(plan.plan_view().route().as_str(), "FittedDependentMeasuredObservation");
    let report = plan.execute(&ctx()).expect("posterior measurement draws");
    let construct_posterior = (0.1 * 86.0) / (0.9 * 11.0 + 0.1 * 86.0);
    near(report.plug_in[1], 0.1 * (1.0 - construct_posterior) + 0.7 * construct_posterior);
    assert_eq!(report.posterior_draws.len(), 16);
    assert!(report.posterior_draws.iter().all(|row| row.len() == 2));
    assert_eq!(report.calibration, "not_measured");
    let interval = report.nominal_interval("1", 0.95).expect("uncalibrated posterior interval");
    assert!(interval.lower < interval.upper);
    assert_eq!(interval.calibration, "not_measured");
    let artifact = plan.export(&report, &ctx()).expect("fitted artifact");
    assert_eq!(
        plan.consume(&artifact, plan.lowered().identity(), &ctx()).expect("fitted replay"),
        report
    );
    println!(
        "Predicted probability of the outcome after two positive readings: {}",
        report.plug_in[1]
    );
}
