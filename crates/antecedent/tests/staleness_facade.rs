//! Staleness diagnostics of a prepared study: every recommendation branch, agreement of the
//! named keys with what selective recalculation invalidates when the same data is applied,
//! agreement of a refusal with what `refresh` refuses, and consumption of the exported
//! baseline without the producer.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::analysis::recalc_adjusted::{
    AdjustedContrast, AdjustedModel, AdjustedRequest, AdjustedSession,
    execute_adjusted_with_receipt,
};
use antecedent::analysis::recalc_receipt::UtilitySpec;
use antecedent::analysis::staleness::{
    FindingKind, Recommendation, StalenessPolicy, StalenessReport, assess_artifact_staleness,
    staleness_policy_from_json, staleness_policy_json, staleness_report_json,
};
use antecedent::{EstimatorId, PreparedStudy, RefuteSuite, Study};
use antecedent_core::recalc::{
    RecalcCapabilities, RecalcPlan, RetargetSupport, Stage, StageIdentity,
};
use antecedent_core::{AverageEffectQuery, CausalRng, ExecutionContext, VariableId};
use antecedent_data::TabularData;
use antecedent_estimate::vector_treatment::VectorCovariance;
use antecedent_graph::{Dag, DenseNodeId};
use antecedent_kernels::{sigmoid, standard_normal};

/// `(t, y, z)` columns: `z` standard normal (shifted by `z_shift`), `t` logistic in `z` with
/// slope `slope`, `y = 1 + 2t + z + noise`.
fn columns(n: usize, seed: u64, z_shift: f64, slope: f64) -> Vec<(String, Vec<f64>)> {
    let mut rng = CausalRng::from_seed(seed);
    let (mut t, mut y, mut z) = (Vec::new(), Vec::new(), Vec::new());
    for _ in 0..n {
        let zi = standard_normal(&mut rng) + z_shift;
        let ti = if rng.next_f64() < sigmoid(slope * zi) { 1.0 } else { 0.0 };
        z.push(zi);
        t.push(ti);
        y.push(1.0 + 2.0 * ti + zi + standard_normal(&mut rng));
    }
    vec![("t".into(), t), ("y".into(), y), ("z".into(), z)]
}

fn table(columns: &[(String, Vec<f64>)]) -> TabularData {
    TabularData::from_f64_columns(columns.iter().map(|(n, v)| (n.as_str(), v.as_slice()))).unwrap()
}

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(17)
}

fn policy() -> StalenessPolicy {
    StalenessPolicy { subsample: 60, permutations: 99, ..StalenessPolicy::default() }
}

fn prepare(data: TabularData) -> PreparedStudy {
    let mut dag = Dag::with_variables(3);
    dag.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(0)).unwrap();
    dag.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(1)).unwrap();
    dag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    Study::tabular(data)
        .graph(dag)
        .query(AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1)))
        .estimator(EstimatorId::LinearAdjustmentAte)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .prepare(&ctx())
        .unwrap()
}

/// The adjusted-regression recalculation request of the same rows and roles.
fn adjusted(columns: &[(String, Vec<f64>)]) -> AdjustedRequest {
    AdjustedRequest {
        columns: columns.to_vec(),
        edges: vec![(2, 0), (2, 1), (0, 1)],
        treatments: vec![0],
        outcome: 1,
        adjustment: vec![2],
        model: AdjustedModel::Linear { covariance: VectorCovariance::ModelBased },
        contrast: AdjustedContrast::Numeric { active: vec![1.0], control: vec![0.0] },
        target: None,
        utility: UtilitySpec { benefit_per_unit: 1.0, cost: 0.5 },
    }
}

/// What the adjusted recalculation session invalidates when `new` is applied after `old`, with
/// `evidence_changed` declaring a changed evidence input on top of the data.
fn recalculated(
    old: &[(String, Vec<f64>)],
    new: &[(String, Vec<f64>)],
    evidence_changed: bool,
) -> Vec<Stage> {
    let mut session = AdjustedSession::new();
    execute_adjusted_with_receipt(&mut session, &adjusted(old), &ctx()).unwrap();
    if !evidence_changed {
        return session.plan(&adjusted(new), &ctx()).recomputed();
    }
    let mut requested = adjusted(new).identities(&ctx());
    requested.set(Stage::Evidence, StageIdentity::of("evidence", &[b"premise no longer holds"]));
    RecalcPlan::plan(
        session.identities(),
        &requested,
        &RecalcCapabilities::in_process(RetargetSupport::Licensed),
    )
    .recomputed()
}

fn exceeded(report: &StalenessReport, kind: FindingKind) -> bool {
    report.findings().any(|f| f.kind == kind)
}

#[test]
fn the_prepared_plan_records_the_identified_adjustment_set() {
    let prepared = prepare(table(&columns(600, 1, 0.0, 0.5)));
    let plan = prepared.staleness_plan();
    assert_eq!(plan.treatment.as_deref(), Some("t"));
    assert_eq!(plan.outcome.as_deref(), Some("y"));
    assert_eq!(plan.adjustment, vec!["z".to_string()]);
    assert!(!plan.bayesian);
    assert_eq!(plan.plan_identity.len(), 64);
}

#[test]
fn the_retained_data_is_reused() {
    let old = columns(600, 2, 0.0, 0.5);
    let prepared = prepare(table(&old));
    let report = prepared.staleness(&table(&old), &policy(), &ctx()).unwrap();
    assert_eq!(report.recommendation, Recommendation::Reuse);
    assert!(report.changed_keys.is_empty() && report.invalidated.is_empty());
}

#[test]
fn a_refresh_names_exactly_what_recalculation_invalidates_and_refresh_then_runs() {
    let old = columns(1000, 3, 0.0, 0.5);
    let new = columns(900, 4, 0.4, 0.5);
    let mut prepared = prepare(table(&old));
    let report = prepared.staleness(&table(&new), &policy(), &ctx()).unwrap();
    assert!(exceeded(&report, FindingKind::CovariateShift), "{report:#?}");
    assert_eq!(report.recommendation, Recommendation::Refresh, "{report:#?}");
    assert_eq!(report.changed_keys, vec![Stage::DataSnapshot, Stage::RowDesign]);
    assert_eq!(report.invalidated, recalculated(&old, &new, false));
    assert!(!report.invalidated.contains(&Stage::Identification));
    prepared.refresh(table(&new), &ctx()).expect("a refresh recommendation refreshes");
}

#[test]
fn a_reidentify_names_exactly_what_recalculation_invalidates_with_the_changed_evidence() {
    let old = columns(1000, 5, 0.0, 0.5);
    let new = columns(1000, 6, 0.0, 9.0);
    let prepared = prepare(table(&old));
    let report = prepared.staleness(&table(&new), &policy(), &ctx()).unwrap();
    assert!(exceeded(&report, FindingKind::PositivityChange), "{report:#?}");
    assert_eq!(report.recommendation, Recommendation::Reidentify);
    assert_eq!(report.changed_keys, vec![Stage::Evidence, Stage::DataSnapshot]);
    assert_eq!(report.invalidated, recalculated(&old, &new, true));
    assert!(report.invalidated.contains(&Stage::Identification));
}

#[test]
fn a_refusal_agrees_with_what_refresh_refuses() {
    let old = columns(500, 7, 0.0, 0.5);
    let mut wider = columns(500, 8, 0.0, 0.5);
    wider.push(("w".into(), vec![1.0; 500]));
    let mut prepared = prepare(table(&old));
    let report = prepared.staleness(&table(&wider), &policy(), &ctx()).unwrap();
    assert_eq!(report.recommendation, Recommendation::Refuse);
    assert!(exceeded(&report, FindingKind::SchemaChange));
    assert!(prepared.refresh(table(&wider), &ctx()).is_err(), "refresh refuses the same data");
}

#[test]
fn an_exported_baseline_is_assessed_without_the_producer() {
    let old = columns(600, 9, 0.0, 0.5);
    let new = table(&columns(600, 10, 0.5, 0.5));
    let prepared = prepare(table(&old));
    let direct = prepared.staleness(&new, &policy(), &ctx()).unwrap();
    let bytes = prepared.staleness_baseline(&policy(), &ctx()).unwrap().export().unwrap();
    drop(prepared);
    let consumed = assess_artifact_staleness(&bytes, &new, &policy(), &ctx()).unwrap();
    assert_eq!(consumed.identity, direct.identity);
    assert_eq!(consumed.recommendation, direct.recommendation);
    assert!(assess_artifact_staleness(&bytes[..bytes.len() / 2], &new, &policy(), &ctx()).is_err());
}

#[test]
fn reports_and_policies_render_as_json() {
    let old = columns(400, 11, 0.0, 0.5);
    let prepared = prepare(table(&old));
    let report = prepared.staleness(&table(&old), &policy(), &ctx()).unwrap();
    let rendered = staleness_report_json(&report);
    assert_eq!(rendered["recommendation"], "reuse");
    assert_eq!(rendered["identity"], report.identity.as_str());
    assert!(rendered["evaluations"].as_array().is_some_and(|e| !e.is_empty()));
    let round = staleness_policy_from_json(
        &staleness_policy_json(&policy())
            .to_string()
            .replace(",\"identity\"", ",\"ignored_identity\""),
    );
    assert!(round.is_err(), "an unknown policy field is refused");
    let mut object = staleness_policy_json(&policy());
    object.as_object_mut().unwrap().remove("identity");
    let parsed = staleness_policy_from_json(&object.to_string()).unwrap();
    assert_eq!(parsed.identity(), policy().identity());
}
