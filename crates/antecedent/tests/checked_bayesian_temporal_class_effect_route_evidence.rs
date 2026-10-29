//! Builder independent evidence for Bayesian temporal class effect envelopes.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::too_many_lines, reason = "one loop covers every licensed coordinate")]

#[path = "fixtures_common/mod.rs"]
mod common;

use antecedent::analysis::RetainedOperation;
use antecedent::{AcceptedGraph, BayesianConfig, ClassPrior, InferenceMode, RefuteSuite, Study};
use antecedent_core::{
    CausalQuery, ExecutionContext, TemporalEffectQuery, TemporalPolicy, VariableId,
};
use antecedent_data::TimeSeriesData;
use antecedent_graph::{TemporalCpdag, TemporalPag};
use common::fixtures::{
    self, B1, chain_pag, chain_pag_series, confounded_cpdag, confounded_series,
};

const DRAWS: usize = 96;

enum SourceGraph {
    Cpdag(TemporalCpdag),
    Pag(TemporalPag),
}

fn graphs() -> [SourceGraph; 2] {
    [SourceGraph::Cpdag(confounded_cpdag()), SourceGraph::Pag(chain_pag())]
}

fn series(graph: &SourceGraph, seed: u64) -> TimeSeriesData {
    match graph {
        SourceGraph::Cpdag(_) => confounded_series(640, 0.0, 0.0, seed),
        SourceGraph::Pag(_) => chain_pag_series(640, 0.0, seed),
    }
}

fn structure(graph: &SourceGraph, accepted: bool) -> AcceptedGraph {
    match (graph, accepted) {
        (SourceGraph::Cpdag(graph), true) => AcceptedGraph::from(graph.clone()),
        (SourceGraph::Cpdag(graph), false) => graph.clone().into(),
        (SourceGraph::Pag(graph), true) => AcceptedGraph::from(graph.clone()),
        (SourceGraph::Pag(graph), false) => graph.clone().into(),
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
        SourceGraph::Pag(_) => B1,
    }
}

fn inference() -> InferenceMode {
    InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(DRAWS).prior_scale(100.0))
}

/// Range of the completion-conditional posterior means retained on the atoms.
fn envelope_bounds(result: &antecedent::StudyResult) -> (f64, f64) {
    let mixture = result
        .structural_response
        .as_ref()
        .expect("completion-level posteriors remain in the result");
    let means = mixture
        .atoms
        .iter()
        .filter_map(|atom| {
            let posterior = atom.posterior.as_ref()?;
            let column = posterior.effect_column()?;
            posterior.summaries.mean.get(column).copied()
        })
        .collect::<Vec<_>>();
    assert!(!means.is_empty(), "identified completions retain posterior means");
    (
        means.iter().copied().fold(f64::INFINITY, f64::min),
        means.iter().copied().fold(f64::NEG_INFINITY, f64::max),
    )
}

#[test]
fn all_bayesian_temporal_class_effect_coordinates_execute_from_retained_source_proofs() {
    let ctx = ExecutionContext::for_tests(9_771);
    let suites = [RefuteSuite::None, RefuteSuite::Cheap, RefuteSuite::Full];
    let graphs = graphs();
    let all_data: Vec<_> = graphs.iter().map(|graph| series(graph, 977)).collect();
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
        let builder = Study::series(data.clone())
            .graph(structure(&graph, source == "accepted"))
            .query(CausalQuery::TemporalEffect(query.clone()))
            .inference(inference())
            .refute(suite)
            .build()
            .unwrap();
        let one_shot = builder.run(&ctx).unwrap();
        let mut prepared = builder.prepare(&ctx).unwrap();
        drop(builder);

        let RetainedOperation::BayesianTemporalClassEffect(plan) = prepared.retained_operation()
        else {
            panic!("Bayesian class effect must retain its complete execution plan")
        };
        assert_eq!(plan.query, query);
        assert_eq!(plan.graph_class.as_str(), graph_class(&graph));
        assert_eq!(plan.identifier.as_str(), "generalized.adjustment");
        assert_eq!(
            plan.estimator.as_str(),
            if multi_step { "temporal.sequential.gcomp" } else { "bayesian.temporal.gcomp" }
        );
        assert_eq!(plan.validation, suite);
        assert_eq!(plan.posterior_draws, DRAWS);
        assert!(!plan.class_prior_supplied);
        assert!(plan.completion_count > 0);
        assert!(plan.identified_mass > 0.0);
        assert!(!matches!(
            prepared.retained_operation(),
            RetainedOperation::BayesianTemporalDagEffect(..)
        ));
        assert!(!matches!(
            prepared.retained_operation(),
            RetainedOperation::TemporalClassEffect(..)
        ));

        let result = prepared.estimate_series(&data, &ctx).unwrap();
        let (lower, upper) = envelope_bounds(&result);
        let true_effect = truth(&graph, multi_step);
        assert!(
            lower - 0.1 <= true_effect && true_effect <= upper + 0.1,
            "class={}, source={source}, multi_step={multi_step}, suite={suite:?}, truth {true_effect} outside [{lower}, {upper}]",
            graph_class(&graph),
        );
        if matches!(graph, SourceGraph::Cpdag(_)) {
            // The two completions are the adjusted law (truth B1) and the unadjusted one, whose
            // pulse coefficient is the confounded probability limit; the envelope ends sit on them.
            let [unadjusted, _] = fixtures::cpdag_completion_truths(0.0);
            assert!((lower - true_effect).abs() < 0.03, "lower {lower} vs adjusted {true_effect}");
            assert!((upper - unadjusted).abs() < 0.03, "upper {upper} vs unadjusted {unadjusted}");
        }
        let mixture = result.structural_response.as_ref().unwrap();
        assert!(
            mixture.atoms.iter().any(|atom| atom.posterior.is_some()),
            "completion posteriors must ride the structural atoms"
        );
        if matches!(graph, SourceGraph::Pag(_)) {
            assert!(
                mixture.unidentified_mass + mixture.unevaluable_mass > 0.0,
                "TemporalPag must preserve the unresolved or unevaluable completion mass"
            );
        }
        // The one-shot facade executes the same sealed program.
        let (one_lower, one_upper) = envelope_bounds(&one_shot);
        assert!(
            (one_lower - lower).abs() < 1e-9 && (one_upper - upper).abs() < 1e-9,
            "one-shot envelope [{one_lower}, {one_upper}] differs from the prepared plan [{lower}, {upper}]"
        );
        assert!(one_shot.diagnostics.iter().any(|d| d.code.as_ref() == "exec.identify.cached"));

        let refreshed = prepared.refresh_series(series(&graph, 978), &ctx).unwrap();
        let (refreshed_lower, refreshed_upper) = envelope_bounds(&refreshed);
        assert!(refreshed_lower - 0.1 <= true_effect && true_effect <= refreshed_upper + 0.1);
        let RetainedOperation::BayesianTemporalClassEffect(retained) =
            prepared.retained_operation()
        else {
            panic!("Bayesian class effect retains its plan")
        };
        assert_eq!(retained.query, query);
        let narrowed = TimeSeriesData::from_f64_columns(
            [("x", &[0.0, 1.0, 0.0, 1.0][..]), ("y", &[0.0, 1.0, 1.0, 0.0][..])],
            1,
        )
        .unwrap();
        assert!(
            prepared.refresh_series(narrowed, &ctx).is_err(),
            "schema-changed data must be refused"
        );
        let RetainedOperation::BayesianTemporalClassEffect(retained) =
            prepared.retained_operation()
        else {
            panic!("Bayesian class effect retains its plan")
        };
        assert_eq!(retained.query, query);

        let artifact = prepared
            .encode_contracted_result(&refreshed, "checked-bayesian-temporal-class-effect", &ctx)
            .unwrap();
        let consumed = antecedent_io::consume_analysis_result(&artifact).unwrap();
        assert!(
            consumed.acceptance.unresolved.iter().any(|dependency| {
                dependency.as_ref()
                    == "dependencies.checked_bayesian_temporal_class_effect_operation"
            }),
            "unresolved={:?}",
            consumed.acceptance.unresolved
        );
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
fn bayesian_temporal_class_effect_retains_a_caller_supplied_class_mass() {
    let ctx = ExecutionContext::for_tests(9_772);
    let graph = SourceGraph::Cpdag(confounded_cpdag());
    let data = series(&graph, 979);
    let query = query(false);
    let builder = Study::series(data.clone())
        .graph(structure(&graph, true))
        .query(CausalQuery::TemporalEffect(query.clone()))
        .inference(inference())
        .class_prior(ClassPrior::from_ordered([0.25, 0.75]).unwrap())
        .refute(RefuteSuite::None)
        .build()
        .unwrap();
    let prepared = builder.prepare(&ctx).unwrap();
    drop(builder);
    let RetainedOperation::BayesianTemporalClassEffect(plan) = prepared.retained_operation() else {
        panic!("sealed plan")
    };
    assert!(plan.class_prior_supplied);
    assert_eq!(plan.completion_count, 2);
    let result = prepared.estimate_series(&data, &ctx).unwrap();
    let [unadjusted, adjusted] = fixtures::cpdag_completion_truths(0.0);
    let mixed_truth = 0.25 * unadjusted + 0.75 * adjusted;
    assert!(
        (result.estimate().as_effect().unwrap().ate - mixed_truth).abs() < 0.05,
        "class-mass mixture {} far from {mixed_truth}",
        result.estimate().as_effect().unwrap().ate
    );
    assert!(result.posterior.is_some(), "a class mass licenses one mixed posterior");
}
