//! Independent planted autoregressions exercise compiled regime discovery.
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::discovery::{
    BackgroundKnowledge, DiscoverParams, MultiDatasetConstraints, RegimeCountPolicy,
    RpcmciDiscoveryResult, RpcmciLearning,
};
use antecedent::discovery_defaults::resolve_ci;
use antecedent::prelude::{ExecutionContext, TimeSeriesData, VariableId};
use antecedent::{CompiledExecution, DiscoveredStructure, DiscoverySpec, Study};
use antecedent_core::CausalRng;
use antecedent_graph::{Endpoint, NodeRef, TemporalCpdag};

fn params() -> DiscoverParams {
    DiscoverParams {
        max_lag: 1,
        alpha: 0.01,
        fdr: None,
        ci: resolve_ci("parcorr", None).unwrap(),
        multi_dataset: MultiDatasetConstraints::default(),
        max_cond_size: 2,
        knowledge: BackgroundKnowledge::default(),
    }
}

fn variable(index: u32) -> VariableId {
    VariableId::from_raw(index)
}

fn known_series(rows: usize, split: Option<usize>, seed: u64) -> TimeSeriesData {
    let mut random = CausalRng::from_seed(seed);
    let mut noise = || (0..12).map(|_| random.next_f64()).sum::<f64>() - 6.0;
    let mut x = vec![0.0; rows];
    let mut y = vec![0.0; rows];
    for t in 1..rows {
        if split.is_some_and(|cut| t >= cut) {
            y[t] = 2.5 + noise();
            x[t] = -2.0 + 0.9 * (y[t - 1] - 2.5) + 0.18 * noise();
        } else {
            x[t] = noise();
            y[t] = 0.95 * x[t - 1] + 0.18 * noise();
        }
    }
    TimeSeriesData::from_f64_columns([("x", x.as_slice()), ("y", y.as_slice())], 1).unwrap()
}

fn run(data: TimeSeriesData, count: RegimeCountPolicy) -> CompiledExecution {
    let options = RpcmciLearning { starts: 4, switch_penalty: 18.0, ..RpcmciLearning::new(count) };
    let spec = DiscoverySpec::rpcmci_learn(params(), Some(48), options, "parcorr").unwrap();
    let study =
        Study::series(data).discovery(spec, vec![variable(0), variable(1)]).build().unwrap();
    CompiledExecution::run(&study, &ExecutionContext::production(71, 1)).unwrap()
}

fn result(execution: &CompiledExecution) -> &RpcmciDiscoveryResult {
    let DiscoveredStructure::Regimes(result) = &execution.result().discovery().unwrap().structure
    else {
        panic!("expected per-regime graphs")
    };
    result
}

fn link(graph: &TemporalCpdag, x: VariableId, y: VariableId) -> bool {
    let has_node = |node: NodeRef, var, lag| matches!(node, NodeRef::Lagged { variable, lag: which } if variable == var && which.raw() == lag);
    graph.edges().into_iter().any(|edge| {
        let a = graph.nodes()[edge.a.as_usize()];
        let b = graph.nodes()[edge.b.as_usize()];
        (has_node(a, x, 1)
            && has_node(b, y, 0)
            && edge.at_a == Endpoint::Tail
            && edge.at_b == Endpoint::Arrow)
            || (has_node(b, x, 1)
                && has_node(a, y, 0)
                && edge.at_b == Endpoint::Tail
                && edge.at_a == Endpoint::Arrow)
    })
}

#[test]
fn independent_changing_graphs_recover_labels_and_replay() {
    let policy = RegimeCountPolicy::Fixed(2);
    let split = 195;
    let data = known_series(450, Some(split), 1279);
    let first = run(data, policy);
    let fit = result(&first);
    let labels = &fit.assignments.regimes;
    let accuracy = labels
        .iter()
        .enumerate()
        .filter(|&(t, label)| label.raw() as usize == usize::from(t >= split))
        .count();
    assert!(accuracy.max(labels.len() - accuracy) >= labels.len() * 94 / 100);
    let early = fit.graphs.get(labels[87]).unwrap();
    let late = fit.graphs.get(labels[340]).unwrap();
    assert!(link(early, variable(0), variable(1)));
    assert!(!link(early, variable(1), variable(0)));
    assert!(link(late, variable(1), variable(0)));
    assert!(!link(late, variable(0), variable(1)));
    let diag = fit.learning.as_ref().unwrap();
    assert_eq!(diag.seed, 71);
    assert_eq!(diag.selected_count, 2);
    assert!(diag.assignment_agreement.is_some_and(|score| (0.5..=1.0).contains(&score)));
    assert!(diag.graph_agreement.iter().all(Option::is_some));
    let second = run(known_series(450, Some(split), 1279), policy);
    assert_eq!(first.lowered_identity(), second.lowered_identity());
    assert_eq!(fit.assignments, result(&second).assignments);
    assert_eq!(fit.graphs.graphs, result(&second).graphs.graphs);
    assert_eq!(fit.learning, result(&second).learning);
    let selected =
        run(known_series(450, Some(split), 1279), RegimeCountPolicy::Bic { max_regimes: 2 });
    assert_eq!(result(&selected).learning.as_ref().unwrap().selected_count, 2);
    assert_ne!(first.lowered_identity(), selected.lowered_identity());
}

#[test]
fn stationary_series_declares_one_regime_and_short_counts_are_infeasible() {
    let fit = run(known_series(132, None, 493), RegimeCountPolicy::Bic { max_regimes: 4 });
    let selected = result(&fit);
    let diag = selected.learning.as_ref().unwrap();
    assert_eq!(diag.selected_count, 1);
    assert_eq!(diag.infeasible_counts, [3, 4]);
    assert!(diag.converged);
    assert!(selected.assignments.regimes.iter().all(|label| label.raw() == 0));
    assert!(link(&selected.graphs.graphs[0].1, variable(0), variable(1)));
    assert_eq!(diag.graph_agreement, [None]);
}

#[test]
fn portable_shared_series_replays_and_accepts_explicit_regimes() {
    use antecedent::GraphRef;
    use antecedent::analysis::composition::{
        BundleBuilder, SuppliedSources, consume, export_bundle,
    };
    use antecedent::analysis::rpcmci_portable::{LearnedRpcmciRequest, consume_learned_rpcmci};
    use antecedent_core::execution::AdmissionPool;
    use serde::Deserialize;

    #[derive(Deserialize)]
    struct Fixture {
        split: usize,
        seed: u64,
        names: Vec<String>,
        columns: Vec<Vec<f64>>,
    }
    antecedent::evidence_unit!("capability", "discovery.learned_rpcmci_replay");
    let data: Fixture = serde_json::from_str(include_str!(
        "../../../conformance/discovery/rpcmci_shared_portable.json"
    ))
    .unwrap();
    let ctx = ExecutionContext::production(data.seed, 1);
    let request = LearnedRpcmciRequest {
        names: data.names,
        columns: data.columns,
        interval_ns: 1,
        count: 2,
        select_count: false,
        starts: 4,
        max_iterations: 12,
        switch_penalty: 18.0,
        min_regime_len: Some(48),
        max_lag: 1,
        alpha: 0.01,
        fdr: false,
        max_cond_size: 2,
    };
    let fitted = request.clone().run(&ctx).unwrap();
    let report = fitted.report();
    assert_eq!(report.seed, data.seed);
    assert_eq!(report.selected_count, 2);
    let agreement = report
        .assignments
        .iter()
        .enumerate()
        .filter(|&(i, &v)| (i >= data.split) == (v != 0))
        .count();
    assert!(
        agreement.max(report.assignments.len() - agreement) * 100 >= 94 * report.assignments.len()
    );
    let early = report.assignments[48];
    let late = report.assignments[330];
    assert_ne!(early, late);
    let accepted_early = fitted.accept_regime(early).unwrap();
    let accepted_late = fitted.accept_regime(late).unwrap();
    let GraphRef::TemporalCpdag(first) = accepted_early.graph_ref() else { panic!("first graph") };
    let GraphRef::TemporalCpdag(second) = accepted_late.graph_ref() else { panic!("second graph") };
    assert!(link(first, variable(0), variable(1)));
    assert!(!link(first, variable(1), variable(0)));
    assert!(link(second, variable(1), variable(0)));
    assert!(!link(second, variable(0), variable(1)));
    assert!(fitted.accept_regime(42).is_err());
    assert!(fitted.regime_scope(42).is_err());
    for regime in [early, late] {
        let scope = fitted.regime_scope(regime).unwrap();
        assert_eq!(scope.artifact_identity, fitted.identity());
        assert_eq!(scope.data_identity, report.data_identity);
        let graph_index = report.graphs.iter().position(|(id, _)| *id == regime).unwrap();
        assert_eq!(scope.graph_identity, report.graphs[graph_index].1);
        assert_eq!(scope.graph_agreement, report.graph_agreement[graph_index]);
        assert_eq!(scope.assignment_agreement, report.assignment_agreement);
        assert_eq!(scope.lookback, 2);
        let expected_rows: Vec<_> = (0..report.assignments.len())
            .filter(|&row| report.assignments[row] == regime)
            .collect();
        let expected_windows: Vec<_> = (2..report.assignments.len())
            .filter(|&row| (0..=2).all(|lag| report.assignments[row - lag] == regime))
            .collect();
        assert_eq!(scope.assigned_rows, expected_rows);
        assert_eq!(scope.window_end_rows, expected_windows);
        assert!(scope.window_end_rows.len() < scope.assigned_rows.len());
    }

    let bytes = fitted.to_bytes().unwrap();
    let replay =
        consume_learned_rpcmci(&bytes, fitted.identity(), &ExecutionContext::for_tests(0)).unwrap();
    assert_eq!(replay.report(), fitted.report());
    assert_eq!(replay.regime_scope(early).unwrap(), fitted.regime_scope(early).unwrap());
    assert_eq!(replay.result().graphs.graphs, fitted.result().graphs.graphs);
    let limited = ExecutionContext::for_tests(0).with_admission(AdmissionPool::fixed(1, 1024));
    assert!(consume_learned_rpcmci(&bytes, fitted.identity(), &limited).is_err());
    let mut changed = request;
    changed.columns[0][0] += 0.25;
    let changed = changed.run(&ctx).unwrap();
    assert_ne!(changed.report().data_identity, report.data_identity);
    assert!(consume_learned_rpcmci(&changed.to_bytes().unwrap(), fitted.identity(), &ctx).is_err());

    let mut builder = BundleBuilder::new();
    builder.add_detected_artifact(Some("learned-regimes"), &bytes).unwrap();
    let bundle = builder.build().unwrap();
    let data = export_bundle(&bundle, "learned-rpcmci").unwrap();
    let verified = consume(&data, bundle.identity(), &SuppliedSources::default()).unwrap();
    assert!(verified.all_replayed());
    assert_eq!(verified.value("learned-regimes", "regime_count"), Some(2.0));
}

#[test]
fn learned_duration_search_charges_admission_before_allocating_candidates() {
    use antecedent_core::execution::AdmissionPool;
    use antecedent_discovery::{DiscoveryError, DiscoveryWorkspace, Rpcmci};
    let data = known_series(450, Some(195), 1279);
    let ctx = ExecutionContext::for_tests(71).with_admission(AdmissionPool::fixed(1, 1024));
    let result = Rpcmci::default().learn(
        &data,
        &[variable(0), variable(1)],
        RpcmciLearning::new(RegimeCountPolicy::Fixed(2)),
        &mut DiscoveryWorkspace::default(),
        &ctx,
    );
    assert!(matches!(result, Err(DiscoveryError::Resource(_))));
}
