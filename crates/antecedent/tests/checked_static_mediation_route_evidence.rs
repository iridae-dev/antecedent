//! Builder-independent evidence for the licensed frequentist static mediation procedure.
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::{AcceptedGraph, RefuteSuite, Study};
use antecedent_core::{
    CausalQuery, ExecutionContext, Intervention, MediationContrast, MediationQuery, Value,
    VariableId,
};
use antecedent_data::TabularData;
use antecedent_graph::{Dag, DenseNodeId};

fn v(raw: u32) -> VariableId {
    VariableId::from_raw(raw)
}

fn graph() -> Dag {
    let mut graph = Dag::with_variables(4);
    for (a, b) in [(3, 0), (3, 1), (3, 2), (0, 1), (0, 2), (1, 2)] {
        graph.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
    }
    graph
}

#[test]
fn static_mediation_preserves_declared_nonbinary_levels_and_default_validation() {
    let base = data(0.0);
    let context = ExecutionContext::for_tests(204);
    for (contrast, unit_truth) in [
        (MediationContrast::Total, 0.7),
        (MediationContrast::Direct, 0.4),
        (MediationContrast::Mediated, 0.3),
        (MediationContrast::NaturalDirect, 0.4),
        (MediationContrast::NaturalIndirect, 0.3),
    ] {
        let mut query = MediationQuery::binary(v(0), v(2), [v(1)], contrast);
        query.control = Intervention::set(v(0), Value::f64(0.2));
        query.active = Intervention::set(v(0), Value::f64(0.8));
        let builder = Study::tabular(base.clone())
            .graph(graph())
            .query(CausalQuery::Mediation(query.clone()))
            .bootstrap_replicates(0)
            .build()
            .unwrap();
        let mut prepared = builder.prepare(&context).unwrap();
        drop(builder);
        let plan = prepared.checked_static_mediation_info().unwrap();
        assert_eq!(plan.query, query);
        assert_eq!(plan.validation, RefuteSuite::PlaceboAndRcc);
        let result = prepared.estimate(&base, &context).unwrap();
        assert!((result.estimate.ate - 0.6 * unit_truth).abs() < 1e-10);
        assert!(!result.refutations.is_empty());
        let refreshed = prepared.refresh(data(0.15), &context).unwrap();
        let direct_delta = if matches!(
            contrast,
            MediationContrast::Total | MediationContrast::Direct | MediationContrast::NaturalDirect
        ) {
            0.15
        } else {
            0.0
        };
        assert!((refreshed.estimate.ate - 0.6 * (unit_truth + direct_delta)).abs() < 1e-10);
        assert_eq!(prepared.checked_static_mediation_info().unwrap().query, query);
    }
}

fn data(direct_delta: f64) -> TabularData {
    let (mut t, mut m, mut y, mut x) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    // Balanced four-factor design gives independent errors and exact known
    // structural coefficients in each equation, without fitting as an oracle.
    for _ in 0..20 {
        for mask in 0..16 {
            let sign = |bit| if mask & (1 << bit) == 0 { -1.0 } else { 1.0 };
            let z = sign(0);
            let treatment = 0.5 * z + sign(1);
            let mediator = 0.6 * treatment + 0.2 * z + 0.3 * sign(2);
            let outcome =
                (0.4 + direct_delta) * treatment + 0.5 * mediator + 0.3 * z + 0.2 * sign(3);
            x.push(z);
            t.push(treatment);
            m.push(mediator);
            y.push(outcome);
        }
    }
    TabularData::from_f64_columns([
        ("t", t.as_slice()),
        ("m", m.as_slice()),
        ("y", y.as_slice()),
        ("x", x.as_slice()),
    ])
    .unwrap()
}

#[test]
fn static_mediation_plan_executes_every_contrast_and_licensed_validation_suite() {
    let base = data(0.0);
    let context = ExecutionContext::for_tests(203);
    for accepted in [false, true] {
        for suite in [RefuteSuite::None, RefuteSuite::Cheap, RefuteSuite::Full] {
            for (contrast, truth) in [
                (MediationContrast::Total, 0.7),
                (MediationContrast::Direct, 0.4),
                (MediationContrast::Mediated, 0.3),
                (MediationContrast::NaturalDirect, 0.4),
                (MediationContrast::NaturalIndirect, 0.3),
            ] {
                let query = MediationQuery::binary(v(0), v(2), Arc::from([v(1)]), contrast);
                let builder = Study::tabular(base.clone());
                let builder = if accepted {
                    builder.graph(AcceptedGraph::dag(graph()))
                } else {
                    builder.graph(graph())
                };
                let builder = builder
                    .query(CausalQuery::Mediation(query.clone()))
                    .refute(suite)
                    .bootstrap_replicates(0)
                    .build()
                    .unwrap();
                let one_shot = builder.run(&context).unwrap();
                assert!((one_shot.estimate.ate - truth).abs() < 1e-10);
                assert!(
                    one_shot.diagnostics.iter().any(|d| d.code.as_ref() == "exec.identify.cached")
                );
                let mut prepared = builder.prepare(&context).unwrap();
                drop(builder);
                let plan = prepared.checked_static_mediation_info().expect("sealed mediation plan");
                assert_eq!(plan.query, query);
                assert_eq!(plan.identifier.as_str(), "path_specific.natural");
                assert_eq!(plan.estimator.as_str(), "mediation.linear");
                assert_eq!(plan.validation, suite);
                assert_eq!(plan.bootstrap_replicates, 0);
                assert_eq!(plan.graph_edges.len(), 6);
                assert_eq!(plan.functional_roots.0, plan.functional_roots.1);
                let result = prepared.estimate(&base, &context).unwrap();
                assert!(
                    (result.estimate.ate - truth).abs() < 1e-10,
                    "{accepted} {suite:?} {contrast:?}: {} vs {truth}",
                    result.estimate.ate
                );
                assert!(
                    result.diagnostics.iter().any(|d| d.code.as_ref() == "exec.identify.cached")
                );
                if suite == RefuteSuite::None {
                    assert!(result.refutations.is_empty());
                } else {
                    assert!(
                        !result.refutations.is_empty(),
                        "{suite:?} must execute its retained refuter"
                    );
                }
                let refreshed = prepared.refresh(data(0.15), &context).unwrap();
                let refreshed_truth = if matches!(
                    contrast,
                    MediationContrast::Total
                        | MediationContrast::Direct
                        | MediationContrast::NaturalDirect
                ) {
                    truth + 0.15
                } else {
                    truth
                };
                assert!((refreshed.estimate.ate - refreshed_truth).abs() < 1e-10);
                assert_eq!(prepared.checked_static_mediation_info().unwrap().query, query);
                let artifact = prepared
                    .encode_contracted_result(&refreshed, "checked-mediation", &context)
                    .unwrap();
                let consumed = antecedent_io::consume_analysis_result(&artifact).unwrap();
                assert!(consumed.acceptance.unresolved.iter().any(|reason| {
                    reason.as_ref() == "dependencies.checked_mediation_operation"
                }));
                assert!(!consumed.acceptance.accepts_as_verified_program());
            }
        }
    }
}
