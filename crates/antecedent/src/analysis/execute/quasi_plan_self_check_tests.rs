//! Retained quasi-experimental operations refuse a compiled plan that names a
//! different query, plan id, identifier, or estimator than the study they seal.
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{
    CausalQuery, ExecutionContext, LocalPolynomialRatioQuery, PanelDidQuery, SyntheticControlQuery,
    VariableId,
};
use antecedent_data::TabularData;

use super::{
    CheckedLocalPolynomialRatioOperation, CheckedPanelDidOperation,
    CheckedSyntheticControlOperation, PhysicalExecutionPlan,
};
use crate::Study;

/// The compiled plan with, in turn, a different query, plan id, identifier,
/// and estimator. `other` is a valid query of the same kind that differs from
/// the sealed one.
fn forged_plans(
    physical: &PhysicalExecutionPlan,
    other: impl Into<CausalQuery>,
) -> [PhysicalExecutionPlan; 4] {
    let mut query = physical.clone();
    query.logical.query = other.into();
    assert_ne!(query.logical.query, physical.logical.query, "the forged query must differ");
    let mut plan_id = physical.clone();
    plan_id.logical.record.plan_id = Arc::from("randomized.trial");
    let mut identifier = physical.clone();
    identifier.logical.record.identifier = Some(Arc::from("randomized.design"));
    let mut estimator = physical.clone();
    estimator.logical.record.estimator = Some(Arc::from("randomized.ht_itt"));
    [query, plan_id, identifier, estimator]
}

fn panel_did() -> (TabularData, Study) {
    let mut ids = Vec::new();
    let mut treated = Vec::new();
    let mut post = Vec::new();
    let mut clusters = Vec::new();
    let mut outcome = Vec::new();
    for subject in 0..8_u32 {
        for after in [false, true] {
            ids.push(Arc::<str>::from(format!("s{subject}")));
            clusters.push(Arc::<str>::from(format!("c{subject}")));
            treated.push(subject < 4);
            post.push(after);
            outcome.push(f64::from(subject) + if after { f64::from(subject % 3) } else { 0.0 });
        }
    }
    let data = TabularData::from_f64_columns([("outcome", outcome.as_slice())]).unwrap();
    let query = PanelDidQuery::new(VariableId::from_raw(0), treated, post, ids, clusters);
    let study = Study::tabular(data.clone()).query(query).build().unwrap();
    (data, study)
}

#[test]
fn panel_did_refuses_a_forged_plan_or_structure_source() {
    let ctx = ExecutionContext::for_tests(1);
    let (data, study) = panel_did();
    let physical = study.plan(&ctx).unwrap();
    assert!(CheckedPanelDidOperation::checked(&study, &data, &physical).is_ok());
    let CausalQuery::PanelDid(mut other) = study.query.clone() else { unreachable!() };
    other.clusters = other.clusters.iter().map(|_| Arc::<str>::from("c0")).collect();
    for forged in forged_plans(&physical, other) {
        assert!(CheckedPanelDidOperation::checked(&study, &data, &forged).is_err());
    }
    let mut explicit = study.clone();
    explicit.structure_source = crate::support::StructureSource::Explicit;
    assert!(CheckedPanelDidOperation::checked(&explicit, &data, &physical).is_err());
}

#[test]
fn synthetic_control_refuses_a_forged_plan() {
    let ctx = ExecutionContext::for_tests(2);
    let mut outcome = Vec::new();
    let mut units = Vec::new();
    let mut periods = Vec::new();
    for (unit, offset) in [("a", 0.0), ("b", 1.0), ("c", 2.0), ("treated", 1.5)] {
        for period in 1..=6_i32 {
            units.push(Arc::<str>::from(unit));
            periods.push(i64::from(period));
            outcome.push(offset + f64::from(period % 3));
        }
    }
    let data = TabularData::from_f64_columns([("outcome", outcome.as_slice())]).unwrap();
    let query = SyntheticControlQuery::new(
        VariableId::from_raw(0),
        units,
        periods,
        Arc::<str>::from("treated"),
        5,
    );
    let study = Study::tabular(data.clone()).query(query).build().unwrap();
    let physical = study.plan(&ctx).unwrap();
    assert!(CheckedSyntheticControlOperation::checked(&study, &data, &physical).is_ok());
    let CausalQuery::SyntheticControl(mut other) = study.query.clone() else { unreachable!() };
    other.treated_unit = Arc::from("a");
    for forged in forged_plans(&physical, other) {
        assert!(CheckedSyntheticControlOperation::checked(&study, &data, &forged).is_err());
    }
}

#[test]
fn local_polynomial_ratio_refuses_a_forged_plan() {
    let ctx = ExecutionContext::for_tests(3);
    let mut running = Vec::new();
    let mut treatment = Vec::new();
    let mut outcome = Vec::new();
    for step in 1..40_i32 {
        for sign in [-1.0, 1.0] {
            let score = sign * f64::from(step) / 40.0;
            let dose = 1.0 + 0.2 * score + 0.8 * f64::max(score, 0.0);
            running.push(score);
            treatment.push(dose);
            outcome.push(1.0 + score + 3.0 * dose);
        }
    }
    let data = TabularData::from_f64_columns([
        ("x", running.as_slice()),
        ("t", treatment.as_slice()),
        ("y", outcome.as_slice()),
    ])
    .unwrap();
    let query = LocalPolynomialRatioQuery {
        outcome: VariableId::from_raw(2),
        treatment: VariableId::from_raw(1),
        running: VariableId::from_raw(0),
        cutoff: 0.0,
        bandwidth: 1.0,
        kink: true,
    };
    let study = Study::tabular(data.clone()).query(query).build().unwrap();
    let physical = study.plan(&ctx).unwrap();
    assert!(CheckedLocalPolynomialRatioOperation::checked(&study, &data, &physical).is_ok());
    let CausalQuery::LocalPolynomialRatio(mut other) = study.query.clone() else { unreachable!() };
    other.bandwidth = 0.5;
    for forged in forged_plans(&physical, other) {
        assert!(CheckedLocalPolynomialRatioOperation::checked(&study, &data, &forged).is_err());
    }
}
