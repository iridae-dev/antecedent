//! A retained fitted effect model carried to a cohort of another population under a
//! declared selection-diagram premise: actual DR-learner producer, hand-calculated cohort
//! means, the typed receipt and identity, the existing decision source for the same cohort,
//! fresh re-derivation, and the named refusal of every unmet premise.
//!
//! Oracle. The fixture is noiseless: `Y(0) = 1 + z`, `Y(1) = effect + 1 + 1.5 z`, with every
//! (treatment, z) cell occurring 40 times, so the linear final stage recovers
//! `tau(z) = effect + 0.5 z` exactly whatever the nuisance fits are, and the fitted rows cover
//! `z in [-1, 1]`. For `effect = 2` the cohort `z = (-1, 0.5, 1)` with weights
//! `(0.2, 0.3, 0.5)` has row contrasts `(1.5, 2.25, 2.5)` and weighted mean
//! `0.2 * 1.5 + 0.3 * 2.25 + 0.5 * 2.5 = 2.225`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::analysis::cohort_decision::{CohortBinding, CohortColumn};
use antecedent::analysis::transported_cohort::{
    COHORT_TRANSPORT_ROUTE, COHORT_TRANSPORT_SUPPORT, DETAIL_COHORT_NOT_NUMERIC,
    DETAIL_IDENTITY_MISMATCH, SelectionShiftDeclaration, TransportedInference,
    transport_cohort_effect_from_result, verify_cohort_transport,
};
use antecedent::{RefuteSuite, Study};
use antecedent_core::{AverageEffectQuery, ExecutionContext, SupportStatus, VariableId};
use antecedent_data::TabularData;
use antecedent_estimate::DrLearner;
use antecedent_graph::{Dag, DenseNodeId};
use antecedent_learn::{LearnerSpec, LinearSpec};

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(97)
}

fn close(actual: f64, expected: f64, tolerance: f64) {
    assert!((actual - expected).abs() < tolerance, "actual={actual}, expected={expected}");
}

fn columns(effect: f64) -> Vec<(String, Vec<f64>)> {
    let (mut t, mut z, mut y) = (Vec::new(), Vec::new(), Vec::new());
    for _ in 0..40 {
        for treatment in [0.0, 1.0] {
            for confounder in [-1.0, 1.0] {
                t.push(treatment);
                z.push(confounder);
                y.push(1.0 + confounder + treatment * (effect + 0.5 * confounder));
            }
        }
    }
    vec![("t".into(), t), ("y".into(), y), ("z".into(), z)]
}

/// The exported analysis result of a DR-learner fit.
fn fit(effect: f64) -> Vec<u8> {
    let columns = columns(effect);
    let borrowed: Vec<(&str, &[f64])> =
        columns.iter().map(|(name, values)| (name.as_str(), values.as_slice())).collect();
    let data = TabularData::from_f64_columns(borrowed).unwrap();
    let mut graph = Dag::with_variables(3);
    for (from, to) in [(2, 0), (2, 1), (0, 1)] {
        graph.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).unwrap();
    }
    let study = Study::tabular(data.clone())
        .graph(graph)
        .query(AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1)))
        .estimator(
            DrLearner::new().with_outcome(LearnerSpec::Linear(LinearSpec::default())).with_folds(4),
        )
        .refute(RefuteSuite::None)
        .build()
        .unwrap();
    let prepared = study.prepare(&ctx()).unwrap();
    let result = prepared.estimate(&data, &ctx()).unwrap();
    prepared.encode_contracted_result(&result, "dr-fit", &ctx()).unwrap()
}

fn cohort_of(population: &str, z: &[f64], w: &[f64]) -> CohortBinding {
    CohortBinding::new(
        "clinic-b-2026-q1",
        population,
        (0..z.len()).map(|i| format!("patient-{i}")).collect(),
        vec!["z".into()],
        vec![CohortColumn::Numeric(z.to_vec())],
        w.to_vec(),
    )
    .unwrap()
}

fn cohort() -> CohortBinding {
    cohort_of("clinic_b", &[-1.0, 0.5, 1.0], &[0.2, 0.3, 0.5])
}

fn declaration() -> SelectionShiftDeclaration {
    SelectionShiftDeclaration::new("clinic_a", "clinic_b", "S", &["z"], &["z"])
}

#[test]
fn transported_mean_matches_the_hand_calculation_with_a_typed_receipt() {
    let bytes = fit(2.0);
    let transported = transport_cohort_effect_from_result(
        &bytes,
        None,
        "clinic_a",
        &cohort(),
        Some(&declaration()),
        &ctx(),
    )
    .unwrap();
    close(transported.mean_contrast(), 2.225, 1e-8);
    let rows = transported.row_contrasts();
    close(rows[0], 1.5, 1e-8);
    close(rows[1], 2.25, 1e-8);
    close(rows[2], 2.5, 1e-8);

    let receipt = transported.receipt();
    assert_eq!(receipt.route, COHORT_TRANSPORT_ROUTE);
    assert_eq!(receipt.support, COHORT_TRANSPORT_SUPPORT);
    assert_eq!(receipt.inference, TransportedInference::PointOnly);
    assert_eq!(receipt.identity.len(), 64);
    assert_eq!(receipt.identity, transported.identity());
    assert_eq!(receipt.source_population, "clinic_a");
    assert_eq!(receipt.target_population, "clinic_b");
    assert_eq!(receipt.selection_node, "S");
    assert_eq!(receipt.selection_children, vec!["z".to_owned()]);
    assert_eq!(receipt.effect_modifiers, vec!["z".to_owned()]);
    // The receipt carries the transport route's own cohort digest (bound into the identity).
    assert_eq!(receipt.cohort_digest.len(), 64);
    assert_eq!(receipt.parent_claim, transported.model().parent_claim());
    assert_eq!(receipt.overlap.rows_checked, 3);
    assert_eq!(receipt.overlap.rows_outside, 0);
    assert_eq!(receipt.overlap.source_rows, 160);
    assert!(receipt.overlap.fully_covered());
    assert!(receipt.assumptions.iter().any(|a| a.contains("not a density or positivity")));
}

#[test]
fn the_decision_source_is_the_existing_surface_with_the_same_mean() {
    let bytes = fit(2.0);
    let transported = transport_cohort_effect_from_result(
        &bytes,
        None,
        "clinic_a",
        &cohort(),
        Some(&declaration()),
        &ctx(),
    )
    .unwrap();
    let source = transported.decision_source("mmHg", &ctx()).unwrap();
    close(source.mean_contrast(), transported.mean_contrast(), 1e-12);
    assert_eq!(source.support_status(), SupportStatus::Extrapolative);
    assert!(source.shift().is_some());
    assert_eq!(source.outcome_units(), "mmHg");
    // Units are never inferred.
    let error = transported.decision_source(" ", &ctx()).unwrap_err();
    assert_eq!(error.detail, "transported_effect.outcome_units_required");
    assert_eq!(error.code, "invalid_argument");
}

#[test]
fn a_fresh_consumer_rederives_and_a_changed_input_moves_the_identity() {
    let bytes = fit(2.0);
    let first = transport_cohort_effect_from_result(
        &bytes,
        None,
        "clinic_a",
        &cohort(),
        Some(&declaration()),
        &ctx(),
    )
    .unwrap();
    let retained = first.identity().to_owned();
    let again = verify_cohort_transport(
        &bytes,
        Some(first.model().parent_claim()),
        "clinic_a",
        &cohort(),
        Some(&declaration()),
        &retained,
        &ctx(),
    )
    .unwrap();
    assert_eq!(again.mean_contrast().to_bits(), first.mean_contrast().to_bits());

    // A wrong retained identity is refused, naming both.
    let error = verify_cohort_transport(
        &bytes,
        None,
        "clinic_a",
        &cohort(),
        Some(&declaration()),
        &"0".repeat(64),
        &ctx(),
    )
    .unwrap_err();
    assert_eq!(error.code, "transport_not_certified");
    assert_eq!(error.detail, DETAIL_IDENTITY_MISMATCH);

    // A changed weight at an unchanged shape is a different value and identity.
    let reweighted = cohort_of("clinic_b", &[-1.0, 0.5, 1.0], &[0.5, 0.25, 0.25]);
    let moved = transport_cohort_effect_from_result(
        &bytes,
        None,
        "clinic_a",
        &reweighted,
        Some(&declaration()),
        &ctx(),
    )
    .unwrap();
    close(moved.mean_contrast(), 0.5 * 1.5 + 0.25 * 2.25 + 0.25 * 2.5, 1e-8);
    assert_ne!(moved.identity(), retained);
    // A different model is a different identity at the same cohort.
    let other = transport_cohort_effect_from_result(
        &fit(3.0),
        None,
        "clinic_a",
        &cohort(),
        Some(&declaration()),
        &ctx(),
    )
    .unwrap();
    close(other.mean_contrast(), 3.225, 1e-8);
    assert_ne!(other.identity(), retained);
    // Another declared source population is a different declaration, hence identity.
    let renamed = SelectionShiftDeclaration::new("clinic_a", "clinic_b", "S2", &["z"], &["z"]);
    let node = transport_cohort_effect_from_result(
        &bytes,
        None,
        "clinic_a",
        &cohort(),
        Some(&renamed),
        &ctx(),
    )
    .unwrap();
    close(node.mean_contrast(), first.mean_contrast(), 1e-12);
    assert_ne!(node.identity(), retained);
}

#[test]
fn every_unmet_premise_is_named() {
    let bytes = fit(2.0);
    let run = |source: &str, cohort: &CohortBinding, shift: Option<&SelectionShiftDeclaration>| {
        transport_cohort_effect_from_result(&bytes, None, source, cohort, shift, &ctx())
    };

    let error = run("clinic_a", &cohort(), None).unwrap_err();
    assert_eq!(error.code, "transport_missing_evidence");
    assert_eq!(error.detail, "transported_effect.shift_declaration_required");
    assert!(!error.remedy.is_empty());

    // A selection arrow into the outcome mechanism.
    let arrow = SelectionShiftDeclaration::new("clinic_a", "clinic_b", "S", &["z", "y"], &["z"]);
    let error = run("clinic_a", &cohort(), Some(&arrow)).unwrap_err();
    assert_eq!(error.code, "transport_not_certified");
    assert_eq!(error.detail, "transported_effect.selection_enters_mechanism");
    assert_eq!(error.subject, "y");

    // Modifiers that are not exactly the model's features.
    let modifiers =
        SelectionShiftDeclaration::new("clinic_a", "clinic_b", "S", &["z"], &["z", "age"]);
    let error = run("clinic_a", &cohort(), Some(&modifiers)).unwrap_err();
    assert_eq!(error.code, "transport_not_certified");
    assert_eq!(error.detail, "transported_effect.modifier_set_mismatch");

    // A declaration for other populations.
    let wrong = SelectionShiftDeclaration::new("clinic_x", "clinic_b", "S", &["z"], &["z"]);
    let error = run("clinic_a", &cohort(), Some(&wrong)).unwrap_err();
    assert_eq!(error.detail, "transported_effect.shift_declaration_mismatch");

    // The cohort is declared to be the model's own population.
    let same = cohort_of("clinic_a", &[0.0], &[1.0]);
    let error = run("clinic_a", &same, Some(&declaration())).unwrap_err();
    assert_eq!(error.code, "quantity_semantics_mismatch");
    assert_eq!(error.detail, "transported_effect.population_not_different");

    // Overlap: a disjoint feature and a row outside the fitted box.
    let disjoint = cohort_of("clinic_b", &[3.0, 4.0], &[0.5, 0.5]);
    let error = run("clinic_a", &disjoint, Some(&declaration())).unwrap_err();
    assert_eq!(error.code, "transport_support_failure");
    assert_eq!(error.detail, "transported_effect.no_overlap");
    let beyond = cohort_of("clinic_b", &[0.0, 3.0], &[0.5, 0.5]);
    let error = run("clinic_a", &beyond, Some(&declaration())).unwrap_err();
    assert_eq!(error.code, "transport_support_failure");
    assert_eq!(error.detail, "transported_effect.out_of_support");
    assert!(error.subject.contains("patient-1"), "{}", error.subject);

    // A feature the model does not have, and a categorical feature.
    let schema = CohortBinding::new(
        "s",
        "clinic_b",
        vec!["a".into()],
        vec!["age".into()],
        vec![CohortColumn::Numeric(vec![1.0])],
        vec![1.0],
    )
    .unwrap();
    let error = run("clinic_a", &schema, Some(&declaration())).unwrap_err();
    assert_eq!(error.code, "schema_mismatch");
    assert_eq!(error.detail, "transported_effect.feature_schema_mismatch");
    let levels = CohortBinding::new(
        "s",
        "clinic_b",
        vec!["a".into()],
        vec!["z".into()],
        vec![CohortColumn::Categorical(vec!["high".into()])],
        vec![1.0],
    )
    .unwrap();
    let error = run("clinic_a", &levels, Some(&declaration())).unwrap_err();
    assert_eq!(error.detail, DETAIL_COHORT_NOT_NUMERIC);

    // A claim the consumer did not retain.
    let error = transport_cohort_effect_from_result(
        &bytes,
        Some("not-the-claim"),
        "clinic_a",
        &cohort(),
        Some(&declaration()),
        &ctx(),
    )
    .unwrap_err();
    assert_eq!(error.detail, "cohort_decision.wrong_parent_identity");
}

#[test]
fn interval_and_law_requests_are_refused_with_the_licensing_evidence() {
    let bytes = fit(2.0);
    let transported = transport_cohort_effect_from_result(
        &bytes,
        None,
        "clinic_a",
        &cohort(),
        Some(&declaration()),
        &ctx(),
    )
    .unwrap();
    let interval = transported.interval_refusal();
    assert_eq!(interval.code, "cell_not_licensed");
    assert_eq!(interval.detail, "transported_effect.interval_not_licensed");
    assert!(interval.remedy.contains("coverage"));
    let law = transported.law_refusal();
    assert_eq!(law.code, "decision_contract_unsatisfied");
    assert_eq!(law.detail, "transported_effect.law_not_supplied");
}
