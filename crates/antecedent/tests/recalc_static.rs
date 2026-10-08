//! Actual-work and independent finite-law oracles for static recalculation.
//! SPDX-License-Identifier: MIT OR Apache-2.0
#[path = "../../antecedent-estimate/tests/common/mod.rs"]
mod common;
use antecedent::analysis::recalc_receipt::{RecalcRunError, UtilitySpec};
use antecedent::analysis::recalc_static::{
    MzRecalcRequest, MzRecalcSession, StaticResponseRequest, StaticResponseSession,
    execute_mz_with_receipt, execute_static_response_with_receipt,
};
use antecedent::consume_mz_transport_artifact;
use antecedent_core::{ExecutionContext, Value, VariableId};
use antecedent_expr::execution_counts::{StaticWorkCounts, count_static_work};
use antecedent_expr::{Assignment, ExactEvaluationLimits};
use antecedent_graph::{Admg, DenseNodeId};
use antecedent_identify::MZ_TRANSPORT_DEFAULT_LIMITS;
use antecedent_identify::execution_counts::count_checked_identifications;
use antecedent_io::consume_analysis_result;
use antecedent_io::mz_transport_artifact::MzTransportConsumeLimits;

fn static_oracle() -> serde_json::Value {
    serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../conformance/recalculation/checked_static_design/expected.json"
    )))
    .unwrap()
}
fn utility() -> UtilitySpec {
    UtilitySpec { benefit_per_unit: 1.0, cost: 0.0 }
}
fn response() -> StaticResponseRequest {
    let pin: serde_json::Value = serde_json::from_str(include_str!(
        "../../../conformance/estimate/admg_frontdoor_functional/expected.json"
    ))
    .unwrap();
    let names = ["t", "m", "y"];
    let mut columns: Vec<(String, Vec<f64>)> =
        names.iter().map(|n| ((*n).into(), Vec::new())).collect();
    for cell in pin["contingency_table"].as_array().unwrap() {
        let count = usize::try_from(cell["count"].as_u64().unwrap()).unwrap();
        for (name, values) in &mut columns {
            values.extend(std::iter::repeat_n(cell[name.as_str()].as_f64().unwrap(), count));
        }
    }
    let mut graph = Admg::with_variables(3);
    graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    graph.insert_directed(DenseNodeId::from_raw(1), DenseNodeId::from_raw(2)).unwrap();
    graph.insert_bidirected(DenseNodeId::from_raw(0), DenseNodeId::from_raw(2)).unwrap();
    StaticResponseRequest {
        columns,
        graph,
        treatment: VariableId::from_raw(0),
        outcome: VariableId::from_raw(2),
        support: vec![0.0, 1.0],
        actions: vec![0.0, 1.0],
        baseline: 0,
        active: 1,
        utility: utility(),
    }
}
fn mz() -> MzRecalcRequest {
    let (catalog, data) = common::mz_fixture::evidence();
    MzRecalcRequest {
        variable_names: vec!["z1".into(), "x".into(), "z2".into(), "y".into()],
        graph: common::mz_fixture::graph(),
        query: common::mz_fixture::query(common::mz_fixture::sources()),
        catalog,
        data,
        requests: [false, true]
            .map(|x| {
                Assignment::from_pairs([(
                    common::z_scm::vid(common::mz_fixture::X),
                    Value::Bool(x),
                )])
            })
            .into(),
        search: MZ_TRANSPORT_DEFAULT_LIMITS,
        limits: ExactEvaluationLimits::default(),
        baseline: 0,
        active: 1,
        utility: utility(),
    }
}
#[test]
fn frontdoor_actual_components_and_coordinate_reuse() {
    let ctx = ExecutionContext::for_tests(7);
    let mut request = response();
    let mut session = StaticResponseSession::new();
    let mut invalid = request.clone();
    invalid.actions[0] = 2.0;
    let ((refusal, checks), work) = count_static_work(|| {
        count_checked_identifications(|| {
            execute_static_response_with_receipt(&mut StaticResponseSession::new(), &invalid, &ctx)
        })
    });
    assert!(matches!(refusal, Err(RecalcRunError::Request("recalc.static_action_out_of_support"))));
    assert_eq!(checks, 0);
    assert_eq!(work, StaticWorkCounts::default());
    let ((out, checks), work) = count_static_work(|| {
        count_checked_identifications(|| {
            execute_static_response_with_receipt(&mut session, &request, &ctx).unwrap()
        })
    });
    assert!(
        (out.contrast - static_oracle()["frontdoor_discrete_contrast"].as_f64().unwrap()).abs()
            < 1e-12
    );
    assert_eq!(out.receipt.totals().identifications, checks);
    assert_eq!(out.receipt.totals().factor_builds, work.factor_builds);
    assert_eq!(out.receipt.totals().program_compilations, work.program_compilations);
    assert_eq!(out.receipt.totals().provider_bindings, work.provider_bindings);
    assert_eq!(out.receipt.totals().factor_evaluations, work.factor_evaluations);
    assert_eq!(out.receipt.totals().integrations, work.integrations);
    assert!(work.factor_builds > 0 && work.program_compilations > 0 && work.integrations > 0);
    consume_analysis_result(&session.export_result(&ctx).unwrap()).unwrap();
    let ids = session.factor_identities();
    let same = execute_static_response_with_receipt(&mut session, &request, &ctx).unwrap();
    assert_eq!(same.receipt.totals().total(), 0);
    request.utility.cost = 1.0;
    let utility = execute_static_response_with_receipt(&mut session, &request, &ctx).unwrap();
    assert_eq!(utility.receipt.totals().factor_builds, 0);
    assert_eq!(utility.receipt.totals().integrations, 0);
    request.actions.reverse();
    let reversed = execute_static_response_with_receipt(&mut session, &request, &ctx).unwrap();
    assert!((reversed.contrast + 0.282).abs() < 1e-12);
    assert_eq!(reversed.receipt.totals().law_summaries, 1);
    assert_eq!(reversed.receipt.totals().program_compilations, 0);
    assert_eq!(session.factor_identities(), ids);
    request.actions[0] = 2.0;
    assert!(matches!(
        execute_static_response_with_receipt(&mut session, &request, &ctx),
        Err(RecalcRunError::Request("recalc.static_action_out_of_support"))
    ));
    request.actions[0] = 1.0;
    assert_eq!(
        execute_static_response_with_receipt(&mut session, &request, &ctx)
            .unwrap()
            .receipt
            .totals()
            .total(),
        0
    );
}
#[test]
fn complementary_sources_actual_counts_and_independent_artifact() {
    let ctx = ExecutionContext::for_tests(3);
    let mut request = mz();
    let mut session = MzRecalcSession::new();
    let ((out, checks), work) = count_static_work(|| {
        count_checked_identifications(|| {
            execute_mz_with_receipt(&mut session, &request, &ctx).unwrap()
        })
    });
    for (index, &point) in out.means.iter().enumerate() {
        let x = u8::try_from(index).unwrap();
        assert!(
            (point
                - common::mz_fixture::target_scm()
                    .risk(&[(common::mz_fixture::X, x)], common::mz_fixture::Y))
            .abs()
                < 1e-12
        );
    }
    let counts = out.receipt.totals();
    assert_eq!(counts.identifications, checks);
    assert_eq!(counts.provider_bindings, work.provider_bindings);
    assert_eq!(counts.program_compilations, work.program_compilations);
    assert_eq!(counts.factor_evaluations, work.factor_evaluations);
    assert_eq!(counts.provider_calls, work.provider_calls);
    assert_eq!(counts.integrations, work.integrations);
    assert!(checks > 0 && work.factor_evaluations > 0);
    let bytes = session.export_result(&ExecutionContext::for_tests(999)).unwrap();
    let consumed =
        consume_mz_transport_artifact(&bytes, MzTransportConsumeLimits::default(), &ctx).unwrap();
    assert_eq!(consumed.distributions().len(), 2);
    assert_eq!(
        execute_mz_with_receipt(&mut session, &request, &ctx).unwrap().receipt.totals().total(),
        0
    );
    request.baseline = 1;
    request.active = 0;
    let swapped = execute_mz_with_receipt(&mut session, &request, &ctx).unwrap();
    assert_eq!(swapped.receipt.totals().law_summaries, 1);
    assert_eq!(swapped.receipt.totals().factor_evaluations, 0);
    assert!((swapped.contrast + out.contrast).abs() < 1e-12);
}
#[test]
fn empirical_refresh_reuses_only_unchanged_factor_inputs() {
    let ctx = ExecutionContext::for_tests(4);
    let mut request = response();
    let mut session = StaticResponseSession::new();
    let first = execute_static_response_with_receipt(&mut session, &request, &ctx).unwrap();
    let before = session.factor_identities();
    for y in &mut request.columns[2].1 {
        *y = 1.0 - *y;
    }
    let ((changed, checks), work) = count_static_work(|| {
        count_checked_identifications(|| {
            execute_static_response_with_receipt(&mut session, &request, &ctx).unwrap()
        })
    });
    assert_eq!(checks, 0);
    assert_eq!(work.program_compilations, 0);
    assert!(work.factor_builds > 0);
    assert!(work.factor_builds < first.receipt.totals().factor_builds);
    assert!((changed.contrast + first.contrast).abs() < 1e-12);
    let after = session.factor_identities();
    assert!(before.iter().all(|id| after.contains(id)));
    let fresh =
        execute_static_response_with_receipt(&mut StaticResponseSession::new(), &request, &ctx)
            .unwrap();
    assert_eq!(fresh.means, changed.means);
    request.columns[2].1[0] = f64::NAN;
    let missing = execute_static_response_with_receipt(&mut session, &request, &ctx).unwrap();
    assert!(missing.receipt.totals().factor_builds >= first.receipt.totals().factor_builds); // changed joint complete-case rows invalidate every factor
}
#[test]
fn changed_transport_law_reuses_other_population_factors() {
    let ctx = ExecutionContext::for_tests(5);
    let mut request = mz();
    let mut session = MzRecalcSession::new();
    execute_mz_with_receipt(&mut session, &request, &ctx).unwrap();
    let mut laws = request.data.laws().to_vec();
    let old = &laws[1];
    let mut probabilities = old.probabilities().to_vec();
    // Strictly positive mixture alters this law while preserving descriptor/world ownership.
    let uniform = 1.0 / f64::from(u32::try_from(probabilities.len()).unwrap());
    for p in &mut probabilities {
        *p = 0.9 * *p + 0.1 * uniform;
    }
    laws[1] = antecedent_expr::ExactDiscreteLaw::try_new(
        old.population(),
        old.regime(),
        old.interventions().to_vec(),
        old.axes().to_vec(),
        probabilities,
        old.snapshot_identity(),
        old.tolerance(),
    )
    .unwrap();
    request.data = antecedent_expr::ExactTransportData::try_new(laws, 4096).unwrap();
    let ((changed, checks), work) = count_static_work(|| {
        count_checked_identifications(|| {
            execute_mz_with_receipt(&mut session, &request, &ctx).unwrap()
        })
    });
    assert_eq!(checks, 0);
    assert_eq!(changed.receipt.totals().identifications, 0);
    let fresh = execute_mz_with_receipt(&mut MzRecalcSession::new(), &request, &ctx).unwrap();
    assert_eq!(changed.means, fresh.means);
    assert!(work.factor_evaluations < fresh.receipt.totals().factor_evaluations);
    assert!(work.factor_evaluations > 0);
    consume_mz_transport_artifact(
        &session.export_result(&ctx).unwrap(),
        MzTransportConsumeLimits::default(),
        &ctx,
    )
    .unwrap();
}
#[test]
fn refusal_is_transactional_and_receipts_do_not_grant_live_factors() {
    let ctx = ExecutionContext::for_tests(9);
    let mut request = response();
    let mut session = StaticResponseSession::new();
    execute_static_response_with_receipt(&mut session, &request, &ctx).unwrap();
    let mut budget = ExecutionContext::for_tests(9);
    budget.memory.hard_limit_bytes = Some(1);
    request.columns[2].1[0] = 1.0;
    let (failed, work) =
        count_static_work(|| execute_static_response_with_receipt(&mut session, &request, &budget));
    assert!(failed.is_err());
    assert_eq!(work.factor_builds, 0);
    request = response();
    assert_eq!(
        execute_static_response_with_receipt(&mut session, &request, &ctx)
            .unwrap()
            .receipt
            .totals()
            .total(),
        0
    );
    let mut fresh = StaticResponseSession::resume(
        session.identities().clone(),
        antecedent_core::recalc::ResumeContext::default(),
    );
    assert!(matches!(
        execute_static_response_with_receipt(&mut fresh, &request, &ctx),
        Err(RecalcRunError::Refused(_))
    ));
    let mut refit = StaticResponseSession::resume(
        session.identities().clone(),
        antecedent_core::recalc::ResumeContext { supplied_data: true, ..Default::default() },
    );
    assert!(
        execute_static_response_with_receipt(&mut refit, &request, &ctx)
            .unwrap()
            .receipt
            .totals()
            .factor_builds
            > 0
    );
}
#[test]
fn verma_checked_scalar_functionals_match_enumerated_latent_scm() {
    let mut columns: Vec<(String, Vec<f64>)> =
        (1..=4).map(|i| (format!("x{i}"), Vec::new())).collect();
    for u in 0..=1 {
        for x1 in 0..=1 {
            for x2 in 0..=1 {
                for x3 in 0..=1 {
                    for x4 in 0..=1 {
                        let probabilities = [
                            0.3,
                            0.5,
                            0.1 + 0.2 * f64::from(x1) + 0.4 * f64::from(u),
                            0.25 + 0.5 * f64::from(x2),
                            0.15 + 0.35 * f64::from(x3) + 0.3 * f64::from(u),
                        ];
                        let bits = [u, x1, x2, x3, x4];
                        let p = probabilities
                            .iter()
                            .zip(bits)
                            .map(|(&p, b)| if b == 1 { p } else { 1.0 - p })
                            .product::<f64>();
                        let count = format!("{:.0}", p * 16_000.0).parse::<usize>().unwrap();
                        for ((_, column), value) in columns.iter_mut().zip([x1, x2, x3, x4]) {
                            column.extend(std::iter::repeat_n(f64::from(value), count));
                        }
                    }
                }
            }
        }
    }
    assert_eq!(columns[0].1.len(), 16_000);
    let mut graph = Admg::with_variables(4);
    for (a, b) in [(0, 1), (1, 2), (2, 3)] {
        graph.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
    }
    graph.insert_bidirected(DenseNodeId::from_raw(1), DenseNodeId::from_raw(3)).unwrap();
    let request = StaticResponseRequest {
        columns,
        graph,
        treatment: VariableId::from_raw(1),
        outcome: VariableId::from_raw(3),
        support: vec![0.0, 1.0],
        actions: vec![0.0, 1.0],
        baseline: 0,
        active: 1,
        utility: utility(),
    };
    let ctx = ExecutionContext::for_tests(11);
    let mut session = StaticResponseSession::new();
    let result = execute_static_response_with_receipt(&mut session, &request, &ctx).unwrap();
    assert!(
        (result.means[0] - static_oracle()["verma_do_means"][0].as_f64().unwrap()).abs() < 1e-12
    );
    assert!(
        (result.means[1] - static_oracle()["verma_do_means"][1].as_f64().unwrap()).abs() < 1e-12
    );
    assert!((result.contrast - static_oracle()["verma_contrast"].as_f64().unwrap()).abs() < 1e-12);
    consume_analysis_result(&session.export_result(&ctx).unwrap()).unwrap();
}
#[test]
fn bounded_search_missing_provider_and_shared_evidence_keep_point_only_contract() {
    let ctx = ExecutionContext::for_tests(13);
    let mut request = mz();
    let mut session = MzRecalcSession::new();
    execute_mz_with_receipt(&mut session, &request, &ctx).unwrap();
    let original = request.clone();
    request.search.operations = 1;
    assert!(matches!(
        execute_mz_with_receipt(&mut session, &request, &ctx),
        Err(RecalcRunError::Execution(_))
    ));
    request = original.clone();
    let laws = request.data.laws()[..3].to_vec();
    request.data = antecedent_expr::ExactTransportData::try_new(laws, 4096).unwrap();
    assert!(matches!(
        execute_mz_with_receipt(&mut session, &request, &ctx),
        Err(RecalcRunError::Execution(antecedent::CausalError::Serialization(
            antecedent_io::IoError::Refused { code: "transport_missing_provider", .. }
        )))
    ));
    assert_eq!(
        execute_mz_with_receipt(&mut session, &original, &ctx).unwrap().receipt.totals().total(),
        0
    );
    let (catalog, data) = common::mz_fixture::with_shared_b_trial(
        &original.catalog,
        &common::mz_fixture::empirical(&original.data, 5_000.0),
        common::mz_fixture::SharedTable::Identical,
    );
    request = original;
    request.catalog = catalog;
    request.data = data;
    let shared = execute_mz_with_receipt(&mut session, &request, &ctx).unwrap();
    assert!(shared.receipt.totals().identifications > 0);
    let bytes = session.export_result(&ctx).unwrap();
    let wire =
        antecedent_io::mz_transport_artifact::MzTransportArtifactWire::decode(&bytes).unwrap();
    assert_eq!(wire.uncertainty.status, antecedent_io::mz_transport_artifact::MZ_WITHHELD);
    consume_mz_transport_artifact(&bytes, MzTransportConsumeLimits::default(), &ctx).unwrap();
}
#[test]
fn graph_and_query_substitution_recheck_identification() {
    let ctx = ExecutionContext::for_tests(17);
    let mut request = response();
    let original = request.clone();
    let mut session = StaticResponseSession::new();
    execute_static_response_with_receipt(&mut session, &request, &ctx).unwrap();
    request.graph.insert_bidirected(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    assert!(execute_static_response_with_receipt(&mut session, &request, &ctx).is_err());
    assert_eq!(
        execute_static_response_with_receipt(&mut session, &original, &ctx)
            .unwrap()
            .receipt
            .totals()
            .total(),
        0
    );
    request = original;
    request.outcome = VariableId::from_raw(1);
    let changed = execute_static_response_with_receipt(&mut session, &request, &ctx).unwrap();
    assert!(changed.receipt.totals().identifications > 0);
    assert!((changed.contrast - 0.6).abs() < 1e-12);
}
#[test]
fn generic_refresh_exports_actual_refreshed_factor_snapshot() {
    let ctx = ExecutionContext::for_tests(23);
    let mut request = response();
    let data = |columns: &[(String, Vec<f64>)]| {
        antecedent_data::TabularData::from_f64_columns(
            columns.iter().map(|(n, v)| (n.as_str(), v.as_slice())).collect::<Vec<_>>(),
        )
        .unwrap()
    };
    let query =
        antecedent_core::ResponseQuery::new(antecedent_core::ResponseFunctional::MeanCurve {
            outcome: request.outcome,
            treatment: antecedent_core::ContinuousDomain::new(
                request.treatment,
                antecedent_core::GridSpec::Values(request.support.clone().into()),
            ),
        });
    let mut prepared = antecedent::Study::tabular(data(&request.columns))
        .graph(request.graph.clone())
        .query(query)
        .identifier(antecedent::IdentifierId::GeneralId)
        .estimator(antecedent::EstimatorId::FunctionalEffect)
        .refute(antecedent::RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .prepare(&ctx)
        .unwrap();
    for y in &mut request.columns[2].1 {
        *y = 1.0 - *y;
    }
    let result = prepared.refresh(data(&request.columns), &ctx).unwrap();
    let antecedent_core::ResponseIdentification::PointIdentified(
        antecedent_core::ResponseValue::Surface { mean, .. },
    ) = &result.response.as_ref().unwrap().estimate
    else {
        panic!("point response required")
    };
    assert!((mean[1] - mean[0] + 0.282).abs() < 1e-12);
    let bytes = prepared.encode_contracted_result(&result, "changed-factor-refresh", &ctx).unwrap();
    consume_analysis_result(&bytes).unwrap();
}

#[test]
fn dense_factor_workspace_respects_budget_before_provider_construction() {
    let mut request = response();
    request.columns = (0..4)
        .map(|column| {
            (
                format!("v{column}"),
                (0..64)
                    .map(|row| if column == 0 { f64::from(row % 2) } else { f64::from(row) })
                    .collect(),
            )
        })
        .collect();
    request.graph = Admg::with_variables(4);
    for (a, b) in [(0, 1), (3, 1), (1, 2)] {
        request.graph.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
    }
    request.graph.insert_bidirected(DenseNodeId::from_raw(0), DenseNodeId::from_raw(2)).unwrap();
    let mut ctx = ExecutionContext::for_tests(31);
    ctx.memory.hard_limit_bytes = Some(65_536);
    let (result, work) = count_static_work(|| {
        execute_static_response_with_receipt(&mut StaticResponseSession::new(), &request, &ctx)
    });
    assert!(matches!(result, Err(RecalcRunError::Request("recalc.memory_budget_exceeded"))));
    assert_eq!(work.factor_builds, 0);
    assert_eq!(work.provider_bindings, 0);
}
#[test]
fn adjacent_dag_and_unsorted_grid_refuse_before_work() {
    let mut request = response();
    let ctx = ExecutionContext::for_tests(37);
    request.support.reverse();
    let (result, work) = count_static_work(|| {
        execute_static_response_with_receipt(&mut StaticResponseSession::new(), &request, &ctx)
    });
    assert!(matches!(result, Err(RecalcRunError::Request("recalc.static_invalid_query"))));
    assert_eq!(work.factor_builds, 0);
    request = response();
    request.graph = Admg::with_variables(3);
    let (result, work) = count_static_work(|| {
        execute_static_response_with_receipt(&mut StaticResponseSession::new(), &request, &ctx)
    });
    assert!(matches!(result, Err(RecalcRunError::Request("recalc.static_graph_unsupported"))));
    assert_eq!(work.provider_bindings, 0);
}
