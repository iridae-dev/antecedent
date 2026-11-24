//! Per-route calibration standing: each interval route keys its slot by its own construction,
//! and its status is exactly what the coverage records give that basis. A route never sets
//! a status itself, so an independent consumer re-derives the same slot from the basis the
//! slot carries, and only a record calibrates.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

mod common;

use antecedent::analysis::fitted_dependent_measurement::FittedDependentMeasurementRequest;
use antecedent::analysis::multi_level_treatment::AdjustmentColumn;
use antecedent::analysis::observation_query::{
    DependentMeasuredObservationRequest, DependentMeasurementWire,
};
use antecedent::analysis::observational_policy::{
    ObservationalPolicyRequest, ObservationalPolicySpecWire, PolicyGraphDeclaration,
    compile_observational_policy,
};
use antecedent::{
    DependenceInput, MechanismCompositionRequest, MechanismInput, MechanismSource, StudyBuilder,
};
use antecedent_contract::{
    MechanismDeclaration, MechanismOriginKind, QuantityRole, ScientificQuantity,
};
use antecedent_core::ExecutionContext;
use antecedent_io::calibration::{CalibrationBasisWire, calibration_slot, rederive_calibration};
use antecedent_io::contract_section::{CalibrationSlotWire, CalibrationStatus};
use antecedent_model::compose::{Domain, TabularRow};
use common::calibration::uniform;

/// The slot is the route's basis read against the records, and nothing more.
fn assert_route_slot(slot: &CalibrationSlotWire, basis: &CalibrationBasisWire) {
    assert_eq!(slot.basis.as_ref(), Some(basis), "the slot carries the route's basis");
    let recomputed = calibration_slot(basis);
    assert_eq!(slot.status, recomputed.status);
    assert_eq!(slot.record_id, recomputed.record_id);
    let consumed = rederive_calibration(slot);
    assert_eq!(consumed.status, slot.status);
    assert_eq!(consumed.record_id, slot.record_id);
    if slot.status == CalibrationStatus::Calibrated {
        assert!(slot.record_id.is_some(), "a calibrated slot names its record");
    } else {
        assert!(slot.reason.is_some(), "an uncalibrated slot says why");
    }
}

// -- observational policy -------------------------------------------------------------

fn policy_request(seed: u64, n: usize, evaluation: &str) -> ObservationalPolicyRequest {
    let mut next = uniform(seed, 0x57A0_D001);
    let (mut outcome, mut treatment, mut x, mut w, mut p0, mut p1) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for _ in 0..n {
        let xi = if next() < 0.5 { -1.0 } else { 1.0 };
        let wi = if next() < 0.5 { -1.0 } else { 1.0 };
        let propensity = 0.35 + 0.10 * xi + 0.05 * wi;
        let action = f64::from(u8::from(next() < propensity));
        let noise = 0.1 * (2.0 * next() - 1.0);
        outcome.push(1.0 + 0.3 * xi + 0.1 * wi + action * (2.0 + 0.3 * xi) + noise);
        treatment.push(action);
        x.push(xi);
        w.push(wi);
        p0.push(1.0 - propensity);
        p1.push(propensity);
    }
    let spec = ObservationalPolicySpecWire {
        support: Some(vec![0.0, 1.0]),
        objective: Some("maximize".into()),
        evaluation: Some(evaluation.into()),
        evaluation_fraction: (evaluation == "held_out").then_some(0.4),
        folds: Some(3),
        known_propensity_columns: Some(vec![p0, p1]),
        causal_graph: Some(PolicyGraphDeclaration {
            nodes: vec!["x".into(), "w".into(), "a".into(), "y".into()],
            edges: vec![
                ("x".into(), "a".into()),
                ("w".into(), "a".into()),
                ("x".into(), "y".into()),
                ("w".into(), "y".into()),
                ("a".into(), "y".into()),
            ],
            treatment: "a".into(),
            outcome: "y".into(),
            information_set: vec!["x".into()],
        }),
        seed,
        ..ObservationalPolicySpecWire::default()
    };
    ObservationalPolicyRequest {
        outcome,
        treatment,
        adjustment: vec![
            AdjustmentColumn { name: "x".into(), values: x },
            AdjustmentColumn { name: "w".into(), values: w },
        ],
        spec,
    }
}

#[test]
fn held_out_policy_slot_is_its_basis_read_against_the_records() {
    let ctx = ExecutionContext::for_tests(41);
    let plan = compile_observational_policy(policy_request(41, 600, "held_out"), &ctx)
        .expect("a graph-checked held-out policy");
    let candidate = plan.held_out_value_candidate(&ctx).expect("a held-out candidate");
    assert_eq!(candidate.basis.key.query, "held_out_learned_policy_value");
    assert_eq!(candidate.basis.key.interval_method, "held_out_policy_score_wald_v1");
    assert!(
        candidate.basis.key.functional.starts_with("held_out_policy_value.known_row_assignment.")
    );
    assert_eq!(candidate.basis.scope.row_count, candidate.report.evaluation_rows);
    assert_route_slot(&candidate.slot, &candidate.basis);
    assert_eq!(candidate.interval.calibration, candidate.slot.status.as_str());
}

#[test]
fn cross_fitted_policy_slot_is_its_basis_read_against_the_records() {
    let ctx = ExecutionContext::for_tests(43);
    let plan = compile_observational_policy(policy_request(43, 600, "cross_fitted_learning"), &ctx)
        .expect("a graph-checked cross-fitted policy");
    let candidate = plan.cross_fitted_value_candidate(&ctx).expect("a cross-fitted candidate");
    assert_eq!(candidate.basis.key.interval_method, "cross_fitted_policy_score_wald_v1");
    assert_route_slot(&candidate.slot, &candidate.basis);
    assert_eq!(candidate.interval.calibration, candidate.slot.status.as_str());
}

// -- fitted dependent measurement -----------------------------------------------------

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
                    parent_levels: parent.map_or_else(Vec::new, |name| {
                        vec![(name.to_owned(), levels[index].clone())]
                    }),
                    probabilities: vec![("0".into(), row[0]), ("1".into(), row[1])],
                })
                .collect(),
        },
        output_support: Domain::Levels(levels.clone()),
        parent_domains: parents
            .iter()
            .map(|name| ((*name).to_owned(), Domain::Levels(levels.clone())))
            .collect(),
        uncertainty: None,
        latent_loadings: Vec::new(),
    }
}

fn observation() -> DependentMeasuredObservationRequest {
    DependentMeasuredObservationRequest {
        outcome: "Y".into(),
        observed: Vec::new(),
        measured: vec![("assay_a".into(), "1".into()), ("assay_b".into(), "1".into())],
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
    }
}

#[test]
fn fitted_dependent_measurement_slot_is_its_basis_read_against_the_records() {
    let ctx = ExecutionContext::for_tests(7);
    let request = MechanismCompositionRequest {
        target_population_id: "pop".into(),
        target_regime_id: "observational".into(),
        mechanisms: vec![
            table("C", None, &[[0.9, 0.1]]),
            table("Y", Some("C"), &[[0.9, 0.1], [0.3, 0.7]]),
        ],
        edges: vec![("C".into(), "Y".into())],
        admissions: Vec::new(),
        dependence: DependenceInput::Unknown,
        admit_declared_mechanisms: true,
    };
    let prepared = StudyBuilder::mechanism_composition(request, &ctx).expect("a composition");
    let compile = |alpha: f64| {
        prepared
            .compile_fitted_dependent_measurement(FittedDependentMeasurementRequest {
                observation: observation(),
                counts: vec![70, 10, 10, 10, 5, 5, 5, 85],
                prior_alpha: alpha,
                draws: 256,
                seed: 7,
            })
            .expect("a fitted dependent measurement")
    };
    let plan = compile(1.0);
    let report = plan.execute(&ctx).expect("an execution");
    let (interval, slot) =
        plan.nominal_interval_with_calibration(&report, "1", 0.95).expect("an interval");
    let basis = plan.calibration_basis("1", 0.95).expect("a basis");
    assert_eq!(basis.key.estimator, "fitted_dependent_measurement_v1");
    assert_eq!(basis.key.identification, "model_conditional");
    assert_eq!(basis.scope.row_count, 200);
    assert_route_slot(&slot, &basis);
    assert_eq!(interval.calibration, slot.status.as_str());
    // Another prior is another construction: no record crosses it.
    let other = compile(2.0).calibration_basis("1", 0.95).expect("a basis");
    assert_ne!(other.key, basis.key);
}
