//! Builder independent evidence for frequentist temporal class effect envelopes.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#[path = "fixtures_common/mod.rs"]
mod common;

use antecedent::analysis::RetainedOperation;
use antecedent::{AcceptedGraph, InferenceMode, RefuteSuite, Study};
use antecedent_core::{
    CausalQuery, ExecutionContext, TemporalEffectQuery, TemporalPolicy, VariableId,
};
use antecedent_data::TimeSeriesData;
use antecedent_graph::{TemporalCpdag, TemporalPag};
use common::fixtures::{
    self, B1, chain_pag, chain_pag_series, confounded_cpdag, confounded_series,
};

/// The structural effect must lie in the identified envelope up to estimation error: the
/// envelope's near end is within 0.011 of the truth in every case (observed), the far end is
/// 0.26 to 0.49 beyond it, so 0.03 admits the sampling error and rejects a displaced envelope.
const SLACK: f64 = 0.03;

enum SourceGraph {
    Cpdag(TemporalCpdag),
    Pag(TemporalPag),
}

fn graphs() -> [SourceGraph; 2] {
    [SourceGraph::Cpdag(confounded_cpdag()), SourceGraph::Pag(chain_pag())]
}

fn series(graph: &SourceGraph) -> TimeSeriesData {
    match graph {
        SourceGraph::Cpdag(_) => confounded_series(640, 0.0, 0.0, 977),
        SourceGraph::Pag(_) => chain_pag_series(640, 0.0, 977),
    }
}

fn accepted(graph: &SourceGraph) -> AcceptedGraph {
    match graph {
        SourceGraph::Cpdag(graph) => AcceptedGraph::from(graph.clone()),
        SourceGraph::Pag(graph) => AcceptedGraph::from(graph.clone()),
    }
}

fn explicit(graph: &SourceGraph) -> AcceptedGraph {
    match graph {
        SourceGraph::Cpdag(graph) => graph.clone().into(),
        SourceGraph::Pag(graph) => graph.clone().into(),
    }
}

fn graph_class(graph: &SourceGraph) -> &'static str {
    match graph {
        SourceGraph::Cpdag(_) => "TemporalCpdag",
        SourceGraph::Pag(_) => "TemporalPag",
    }
}

fn query(multi_step: bool) -> TemporalEffectQuery {
    let treatment = VariableId::from_raw(0);
    let outcome = VariableId::from_raw(1);
    if multi_step {
        TemporalEffectQuery::sustained(treatment, outcome, -2, 1.0)
            .with_policy(TemporalPolicy::sustained(-2, -1))
            .with_horizon_steps(1)
    } else {
        TemporalEffectQuery::pulse(treatment, outcome, 1.0)
            .with_policy(TemporalPolicy::pulse(-1))
            .with_horizon_steps(1)
    }
}

fn truth(graph: &SourceGraph, multi_step: bool) -> f64 {
    match graph {
        SourceGraph::Cpdag(_) if multi_step => B1,
        SourceGraph::Cpdag(_) => {
            let [_, adjusted] = fixtures::cpdag_completion_truths(0.0);
            adjusted
        }
        // The generating directed completion of the seven atom chain PAG has
        // treatment effect B1 for Pulse and this terminal outcome horizon.
        SourceGraph::Pag(_) => B1,
    }
}

#[test]
#[allow(
    clippy::too_many_lines,
    reason = "a linear test script over every graph, suite and effect coordinate"
)]
fn all_frequentist_temporal_class_effect_coordinates_execute_from_retained_source_proofs() {
    let ctx = ExecutionContext::for_tests(977);
    let suites = [RefuteSuite::None, RefuteSuite::Cheap, RefuteSuite::Full];
    let graphs = graphs();
    let all_data: Vec<_> = graphs.iter().map(|graph| series(graph)).collect();
    let mut cases = Vec::new();
    for index in 0..graphs.len() {
        for multi_step in [false, true] {
            for source in ["explicit", "accepted"] {
                for suite in suites {
                    cases.push((index, multi_step, source, suite));
                }
            }
        }
    }
    in_parallel(&cases, |&(index, multi_step, source, suite)| {
        let (graph, data) = (&graphs[index], &all_data[index]);
        let query = query(multi_step);
        let structure = if source == "accepted" { accepted(&graph) } else { explicit(&graph) };
        let builder = Study::series(data.clone())
            .graph(structure)
            .query(CausalQuery::TemporalEffect(query.clone()))
            .inference(InferenceMode::Frequentist)
            .refute(suite)
            .bootstrap_replicates(16)
            .build()
            .unwrap();
        let mut prepared = builder.prepare(&ctx).unwrap();
        drop(builder);
        let RetainedOperation::TemporalClassEffect(plan) = prepared.retained_operation() else {
            panic!("class effect must retain its complete execution plan")
        };
        assert_eq!(plan.query, query);
        assert_eq!(plan.graph_class.as_str(), graph_class(&graph));
        assert_eq!(plan.identifier.as_str(), "generalized.adjustment");
        assert_eq!(
            plan.estimator.as_str(),
            if multi_step { "temporal.sequential.gcomp" } else { "temporal.linear.adjustment" }
        );
        assert_eq!(plan.validation, suite);
        assert_eq!(plan.bootstrap_replicates, 16);
        assert!(plan.completion_count > 0);
        assert!(plan.identified_mass > 0.0);

        let result = prepared.estimate_series(&data, &ctx).unwrap();
        assert!(result.estimate().as_effect().unwrap().ate.is_finite());
        let true_effect = truth(&graph, multi_step);
        let mixture = result
            .structural_response
            .as_ref()
            .expect("completion-level effects remain in the result");
        let envelope = mixture
            .identified_set
            .as_ref()
            .expect("identified completion values define an envelope");
        assert!(
            envelope.lower[0] - SLACK <= true_effect && true_effect <= envelope.upper[0] + SLACK,
            "class={}, source={source}, multi_step={multi_step}, suite={suite:?}, truth {true_effect} outside [{}, {}]",
            graph_class(&graph),
            envelope.lower[0],
            envelope.upper[0]
        );
        if matches!(graph, SourceGraph::Pag(_)) {
            assert!(
                mixture.unidentified_mass + mixture.unevaluable_mass > 0.0,
                "TemporalPag must preserve the unresolved or unevaluable completion mass"
            );
        }
        let standard_error = result
            .estimate()
            .as_effect()
            .unwrap()
            .se_bootstrap()
            .expect("shared circular-block procedure must execute");
        assert!(standard_error.is_finite() && standard_error >= 0.0);

        let refreshed_data = series(&graph);
        let refreshed = prepared.refresh_series(refreshed_data, &ctx).unwrap();
        let refreshed_truth = true_effect;
        let refreshed_envelope = refreshed
            .structural_response
            .as_ref()
            .and_then(|mixture| mixture.identified_set.as_ref())
            .expect("refreshed completion values define an envelope");
        assert!(
            refreshed_envelope.lower[0] - SLACK <= refreshed_truth
                && refreshed_truth <= refreshed_envelope.upper[0] + SLACK
        );
        let RetainedOperation::TemporalClassEffect(rebound) = prepared.retained_operation() else {
            panic!("class effect must retain its plan after refresh")
        };
        assert_eq!(rebound.query, query);
        let artifact = prepared
            .encode_contracted_result(&refreshed, "checked-temporal-class-effect", &ctx)
            .unwrap();
        let consumed = antecedent_io::consume_analysis_result(&artifact).unwrap();
        assert!(consumed.acceptance.unresolved.iter().any(|dependency| {
            dependency.as_ref() == "dependencies.checked_temporal_class_effect_operation"
        }));
        assert!(!consumed.acceptance.accepts_as_verified_program());
    });
}

/// Runs the independent coordinate cases on scoped threads (one per available core) and
/// re-raises the first panic with its original payload, so a failure reads as it would in a
/// sequential loop. Each case builds its own study and shares only the read-only fixtures.
fn in_parallel<T: Sync>(cases: &[T], run: impl Fn(&T) + Sync) {
    let next = std::sync::atomic::AtomicUsize::new(0);
    let workers = std::thread::available_parallelism().map_or(1, usize::from).min(cases.len());
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..workers)
            .map(|_| {
                scope.spawn(|| {
                    while let Some(case) =
                        cases.get(next.fetch_add(1, std::sync::atomic::Ordering::Relaxed))
                    {
                        run(case);
                    }
                })
            })
            .collect();
        let mut first = None;
        for handle in handles {
            if let Err(payload) = handle.join() {
                first.get_or_insert(payload);
            }
        }
        if let Some(payload) = first {
            std::panic::resume_unwind(payload);
        }
    });
}

#[test]
fn temporal_class_preparation_honors_context_after_a_warm_process_envelope() {
    struct CancelIdentification(antecedent_core::CancellationToken);
    impl antecedent_core::ProgressSink for CancelIdentification {
        fn report(&self, _fraction: f64, stage: &str) {
            if stage == "identify.compute" {
                self.0.cancel();
            }
        }
    }
    for graph in graphs() {
        for multi_step in [false, true] {
            let query = query(multi_step);
            let identifier = antecedent_identify::GeneralizedAdjustmentIdentifier::new();
            // Warm the actual context-free lower-crate memo with exactly these premises.
            match &graph {
                SourceGraph::Cpdag(graph) => {
                    identifier.identify_temporal_cpdag_envelope(graph, &query).unwrap();
                }
                SourceGraph::Pag(graph) => {
                    identifier.identify_temporal_pag_envelope(graph, &query).unwrap();
                }
            }
            let data = series(&graph);
            let study = Study::series(data.clone())
                .graph(explicit(&graph))
                .query(CausalQuery::TemporalEffect(query))
                .refute(RefuteSuite::None)
                .bootstrap_replicates(0)
                .build()
                .unwrap();
            let mut normal = ExecutionContext::for_tests(979);
            normal.cache_policy = antecedent_core::CachePolicy::enabled(Some(4_000_000));
            let prepared = study.prepare(&normal).unwrap();
            let expected = prepared.estimate_series(&data, &normal).unwrap();
            let mut disabled = ExecutionContext::for_tests(979);
            disabled.cache_policy = antecedent_core::CachePolicy::disabled();
            let uncached = study.prepare(&disabled).unwrap();
            assert_eq!(
                uncached.contract().unwrap().identities,
                prepared.contract().unwrap().identities
            );
            assert_eq!(
                uncached.estimate_series(&data, &disabled).unwrap().effect().to_bits(),
                expected.effect().to_bits()
            );
            let limited = ExecutionContext::for_tests(979).with_memory_limit(Some(1));
            assert!(matches!(
                study.prepare(&limited),
                Err(antecedent::CausalError::Resource { .. })
            ));
            let mut cancelled = ExecutionContext::for_tests(979);
            cancelled.progress =
                Some(std::sync::Arc::new(CancelIdentification(cancelled.cancellation.clone())));
            assert!(matches!(
                study.prepare(&cancelled),
                Err(antecedent::CausalError::Cancelled { .. })
            ));
        }
    }
}
