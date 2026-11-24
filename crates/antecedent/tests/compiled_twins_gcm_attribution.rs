//! The direct attribution functions of `antecedent::gcm` and the compiled `Attribute` route
//! give the same numbers.
//!
//! The route fits the standard mechanism program, which selects the mechanisms `fit_gcm`
//! selects, and hands each target to the same attribution function with the same model, data,
//! options and execution context. Every pair below runs both sides under one seed and compares
//! every number of the two results on `to_bits()`; the coalition-cache statistics and the
//! evaluation counts are bookkeeping, not results, and are not compared.
//!
//! * `distribution_change` and a `DistributionChange` target with the sampled estimator: the
//!   change measure maps onto the difference measure and the draws and seed pass unchanged.
//! * `distribution_change_robust` and a `DistributionChange` target with the robust estimator.
//! * `root_cause_rank` and a `RootCauses` target: the route ranks its change attribution with
//!   no graph samples and no contribution draws, as the direct call below does.
//! * `path_decompose` and a `Paths` target that names no node the paths must pass through.
//! * `feature_relevance` and a `FeatureRelevance` target.
//! * `score_anomalies_with` and an `Anomaly` target. `score_anomalies`, the form without an
//!   execution context, is the same computation: abduction of a complete table draws nothing
//!   and every row is an exact Shapley game, so no seed enters the scores.
//! * `unit_change` and a `UnitChange` target.
//! * `structure_change` and a `StructureChange` target: the comparison model is the compiled
//!   DAG of the declared comparison edges over the same variables, built here as the route
//!   builds it.
//! * `intrinsic_influence` and an `IntrinsicInfluence` target: the query is assembled from the
//!   target field by field, independent noise is the declared independence, and with no
//!   mechanism-fit refits requested the refit law is the row bootstrap.
//! * `detect_mechanism_changes` and a `MechanismChange` target with the likelihood-ratio test
//!   on a static table under frequentist inference, which reads no block length and no
//!   posterior.
//! * `ChangeAttribution::run` and the sampled `DistributionChange` target: the builder's query
//!   has the components, allocation and component cap of `ChangeAttributionQuery::new`.
//! * `ChangeAttribution::run_structure` and the `StructureChange` target, the builder set to
//!   structure components.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    deprecated,
    reason = "these tests call the direct attribution functions the compiled route is compared with"
)]
#![allow(
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::too_many_lines,
    reason = "bit-identical equality is the property under test; columns come from small indices"
)]

use std::sync::Arc;

use antecedent::gcm::{
    AnomalyScores, ChangeAttribution, ChangeAttributionResult, CompiledCausalModel,
    ComponentStatus, DifferenceMeasure, DistributionChangeOptions, FeatureRelevance, FitResampling,
    IntrinsicAllocation, IntrinsicInfluenceQuery, IntrinsicInfluenceResult,
    MechanismChangeDetection, MechanismChangeMethod, NoiseIndependence, RobustChangeOptions,
    RootCauseRank, StructureChangeOptions, UnitChangeResult, detect_mechanism_changes,
    distribution_change, distribution_change_robust, feature_relevance, fit_gcm,
    intrinsic_influence, path_decompose, root_cause_rank, score_anomalies, score_anomalies_with,
    structure_change, unit_change,
};
use antecedent::{AttributionOutcome, CompiledExecution, InferenceMode, RefuteSuite, Study};
use antecedent_attribution::FeatureRelevanceOptions;
use antecedent_core::{
    AnomalyAttributionQuery, AttributeQuery, AttributeTarget, AttributionComponents, CausalQuery,
    ChangeAttributionQuery, ChangeEstimator, ChangeMeasure, ExecutionContext, InfluenceAllocation,
    IntrinsicInfluenceTarget, MechanismChangeQuery, MechanismChangeTest, NoiseDeclaration,
    PopulationSelector, UnitChangeQuery, VariableId,
};
use antecedent_data::TabularData;
use antecedent_graph::{Dag, DenseNodeId};

const ROWS: usize = 300;
const HALF: usize = ROWS / 2;
const SEED: u64 = 0x5EED_0A77;

fn v(raw: u32) -> VariableId {
    VariableId::from_raw(raw)
}

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(SEED)
}

fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|x| x.to_bits()).collect()
}

fn opt_bits(value: Option<f64>) -> Option<u64> {
    value.map(f64::to_bits)
}

fn on_large_stack(name: &str, body: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .name(name.into())
        .stack_size(32 * 1024 * 1024)
        .spawn(body)
        .unwrap()
        .join()
        .unwrap();
}

fn dag(nodes: u32, edges: &[(u32, u32)]) -> Dag {
    let mut graph = Dag::with_variables(nodes);
    for &(from, to) in edges {
        graph.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).unwrap();
    }
    graph
}

/// `x -> y` over two periods of `HALF` rows; the intercept of `y` moves from 1 to 3 in the
/// second. Variables `0 = x`, `1 = y`.
fn two_period() -> (TabularData, Dag) {
    let x: Vec<f64> =
        (0..ROWS).map(|i| (i % 17) as f64 * 0.2 + 0.3 * (i as f64 * 0.53).sin()).collect();
    let y: Vec<f64> = (0..ROWS)
        .map(|i| {
            let intercept = if i < HALF { 1.0 } else { 3.0 };
            intercept + 2.0 * x[i] + 0.3 * (i as f64 * 0.37).sin()
        })
        .collect();
    let data = TabularData::from_f64_columns([("x", x.as_slice()), ("y", y.as_slice())]).unwrap();
    (data, dag(2, &[(0, 1)]))
}

/// `z -> t -> y`, `z -> y`, each with its own bounded wiggle. Variables `0 = z`, `1 = t`,
/// `2 = y`.
fn chain() -> (TabularData, Dag) {
    let z: Vec<f64> =
        (0..ROWS).map(|i| (i % 17) as f64 * 0.2 - 1.6 + 0.1 * (i as f64 * 1.3).cos()).collect();
    let t: Vec<f64> = (0..ROWS).map(|i| 0.8 * z[i] + 0.5 * (i as f64 * 0.37).sin()).collect();
    let y: Vec<f64> =
        (0..ROWS).map(|i| 1.5 * t[i] + 0.6 * z[i] + 0.5 * (i as f64 * 0.91).cos()).collect();
    let data = TabularData::from_f64_columns([
        ("z", z.as_slice()),
        ("t", t.as_slice()),
        ("y", y.as_slice()),
    ])
    .unwrap();
    (data, dag(3, &[(0, 1), (1, 2), (0, 2)]))
}

/// The change of `y` from the first period to the second.
fn change() -> ChangeAttributionQuery {
    ChangeAttributionQuery::new(
        v(1),
        PopulationSelector::TimeRange { start: 0, end: HALF },
        PopulationSelector::TimeRange { start: HALF, end: ROWS },
    )
}

/// The compiled model of a comparison structure over `nodes` variables, as the route builds it.
fn comparison_model(nodes: u32, edges: &[(VariableId, VariableId)]) -> CompiledCausalModel {
    let mut graph = Dag::with_variables(nodes);
    for (from, to) in edges {
        graph
            .insert_directed(DenseNodeId::from_raw(from.raw()), DenseNodeId::from_raw(to.raw()))
            .unwrap();
    }
    CompiledCausalModel::compile(graph).unwrap()
}

/// Run `target` through the compiled `Attribute` route and return its outcome.
fn compiled(data: &TabularData, graph: &Dag, target: AttributeTarget) -> AttributionOutcome {
    let study = Study::tabular(data.clone())
        .graph(graph.clone())
        .query(CausalQuery::Attribute(AttributeQuery::new(target)))
        .inference(InferenceMode::Frequentist)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    let plan = study.compile(&ctx()).unwrap();
    let lowered = plan.lowered().expect("compile attaches a lowered program");
    assert_eq!(lowered.route(), "GcmAttributeDag");
    let execution = CompiledExecution::run(&study, &ctx()).unwrap();
    execution
        .result()
        .attribution()
        .map(|payload| payload.outcome.clone())
        .expect("an attribution returns its payload")
}

fn compiled_change(outcome: AttributionOutcome) -> ChangeAttributionResult {
    match outcome {
        AttributionOutcome::Change(result) => result,
        other => panic!("a change target returns a change attribution: {other:?}"),
    }
}

fn assert_change_eq(compiled: &ChangeAttributionResult, direct: &ChangeAttributionResult) {
    assert!(direct.total_change.is_finite(), "the direct attribution computed a total");
    assert_eq!(compiled.outcome, direct.outcome);
    assert_eq!(compiled.total_change.to_bits(), direct.total_change.to_bits());
    assert_eq!(opt_bits(compiled.observed_change), opt_bits(direct.observed_change));
    let contributions = |result: &ChangeAttributionResult| {
        result
            .contributions
            .iter()
            .map(|c| {
                (
                    c.component,
                    c.contribution.to_bits(),
                    opt_bits(c.stderr),
                    opt_bits(c.ci_low),
                    opt_bits(c.ci_high),
                )
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(contributions(compiled), contributions(direct));
    let interactions = |result: &ChangeAttributionResult| {
        result.interactions.iter().map(|i| (i.a, i.b, i.value.to_bits())).collect::<Vec<_>>()
    };
    assert_eq!(interactions(compiled), interactions(direct));
    let paths = |result: &ChangeAttributionResult| {
        result
            .path_breakdown
            .iter()
            .map(|p| (Arc::clone(&p.path), p.contribution.to_bits()))
            .collect::<Vec<_>>()
    };
    assert_eq!(paths(compiled), paths(direct));
    assert_eq!(compiled.unidentified, direct.unidentified);
    assert_eq!(
        compiled.graph_sensitivity.as_deref().map(bits),
        direct.graph_sensitivity.as_deref().map(bits)
    );
    assert_eq!(opt_bits(compiled.monte_carlo_stderr), opt_bits(direct.monte_carlo_stderr));
    assert_eq!(
        compiled.component_mc_stderr.as_deref().map(bits),
        direct.component_mc_stderr.as_deref().map(bits)
    );
    let fit = |result: &ChangeAttributionResult| {
        result.fit_uncertainty.as_ref().map(|f| {
            (
                f.replicates,
                f.failures,
                f.level.to_bits(),
                f.lower.as_deref().map(bits),
                f.upper.as_deref().map(bits),
                f.total_interval.map(|(low, high)| (low.to_bits(), high.to_bits())),
            )
        })
    };
    assert_eq!(fit(compiled), fit(direct));
}

fn rank_bits(ranks: &[RootCauseRank]) -> Vec<(antecedent_core::ComponentId, u64, Option<u64>)> {
    ranks.iter().map(|r| (r.component, r.score.to_bits(), opt_bits(r.graph_std))).collect()
}

fn feature_bits(features: &[FeatureRelevance]) -> Vec<(VariableId, VariableId, u64)> {
    features.iter().map(|f| (f.feature, f.outcome, f.score.to_bits())).collect()
}

fn assert_anomalies_eq(compiled: &[AnomalyScores], direct: &[AnomalyScores]) {
    assert_eq!(compiled.len(), direct.len());
    assert!(!direct.is_empty(), "one score set per target");
    for (compiled, direct) in compiled.iter().zip(direct) {
        assert_eq!(compiled.target, direct.target);
        assert_eq!(compiled.rows, direct.rows);
        assert_eq!(bits(&compiled.scores), bits(&direct.scores));
        assert_eq!(bits(&compiled.baseline_scores), bits(&direct.baseline_scores));
        assert_eq!(bits(&compiled.abs_contribution_sum), bits(&direct.abs_contribution_sum));
        assert_eq!(compiled.noise_components, direct.noise_components);
        assert_eq!(bits(&compiled.noise_contributions), bits(&direct.noise_contributions));
    }
}

fn assert_unit_change_eq(compiled: &UnitChangeResult, direct: &UnitChangeResult) {
    assert_eq!(compiled.outcome, direct.outcome);
    assert_eq!(compiled.unit_rows, direct.unit_rows);
    assert_eq!(compiled.components, direct.components);
    assert_eq!(bits(&compiled.contributions), bits(&direct.contributions));
    assert_eq!(bits(&compiled.mean_contributions), bits(&direct.mean_contributions));
    assert_eq!(opt_bits(compiled.monte_carlo_stderr), opt_bits(direct.monte_carlo_stderr));
    assert_eq!(
        compiled.component_mc_stderr.as_deref().map(bits),
        direct.component_mc_stderr.as_deref().map(bits)
    );
}

/// A status as a comparable key: the bits of every number of a quantified component, the
/// whole status otherwise.
fn status_key(status: &ComponentStatus) -> String {
    match status {
        ComponentStatus::Quantified { stderr, ci_low, ci_high, level, df } => {
            format!("quantified {:?}", bits(&[*stderr, *ci_low, *ci_high, *level, *df]))
        }
        other => format!("{other:?}"),
    }
}

fn assert_influence_eq(compiled: &IntrinsicInfluenceResult, direct: &IntrinsicInfluenceResult) {
    assert!(direct.total_variance.is_finite(), "the direct attribution computed a variance");
    assert_eq!(compiled.target, direct.target);
    let sources = |result: &IntrinsicInfluenceResult| {
        result
            .sources
            .iter()
            .map(|s| {
                (
                    s.source,
                    s.contribution.to_bits(),
                    s.uncertainty
                        .iter()
                        .map(|u| (u.component, status_key(&u.status)))
                        .collect::<Vec<_>>(),
                    status_key(&s.whole_procedure),
                )
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(sources(compiled), sources(direct));
    assert_eq!(compiled.total_variance.to_bits(), direct.total_variance.to_bits());
    assert_eq!(compiled.total_variance_stderr.to_bits(), direct.total_variance_stderr.to_bits());
    assert_eq!(compiled.allocation_residual.to_bits(), direct.allocation_residual.to_bits());
    assert_eq!(
        compiled.allocation_residual_stderr.to_bits(),
        direct.allocation_residual_stderr.to_bits()
    );
    assert_eq!(compiled.level.to_bits(), direct.level.to_bits());
    assert_eq!(format!("{:?}", compiled.method), format!("{:?}", direct.method));
    let draws = |result: &IntrinsicInfluenceResult| {
        result.fit_draws.iter().map(|draw| bits(draw.as_slice())).collect::<Vec<_>>()
    };
    assert_eq!(draws(compiled), draws(direct));
}

type DetectionKey =
    (VariableId, bool, u64, u64, Option<u64>, Arc<str>, Option<usize>, Option<u64>, Option<u64>);

fn detection_bits(detections: &[MechanismChangeDetection]) -> Vec<DetectionKey> {
    detections
        .iter()
        .map(|d| {
            (
                d.variable,
                d.changed,
                d.statistic.to_bits(),
                d.p_value.to_bits(),
                opt_bits(d.adjusted_p_value),
                Arc::clone(&d.method),
                d.n_permutations,
                opt_bits(d.p_value_floor),
                opt_bits(d.posterior_probability),
            )
        })
        .collect()
}

#[test]
fn distribution_change_equals_the_compiled_sampled_change_route() {
    on_large_stack("twin-distribution-change", || {
        let (data, graph) = two_period();
        let fitted = fit_gcm(graph.clone(), &data).unwrap();
        let options = DistributionChangeOptions {
            measure: DifferenceMeasure::MeanDiff,
            n_samples: 400,
            seed: 3,
        };
        let direct =
            distribution_change(&fitted.model, &data, &change(), &options, &ctx()).unwrap();
        let route = compiled_change(compiled(
            &data,
            &graph,
            AttributeTarget::DistributionChange {
                change: change(),
                estimator: ChangeEstimator::Sampled {
                    measure: ChangeMeasure::MeanDifference,
                    n_samples: 400,
                    seed: 3,
                },
            },
        ));
        assert_change_eq(&route, &direct);
    });
}

#[test]
fn distribution_change_robust_equals_the_compiled_robust_change_route() {
    on_large_stack("twin-robust-change", || {
        let (data, graph) = two_period();
        let fitted = fit_gcm(graph.clone(), &data).unwrap();
        let options = RobustChangeOptions { max_rows: 1000, n_samples: 200, seed: 5 };
        let direct =
            distribution_change_robust(&fitted.model, &data, &change(), &options, &ctx()).unwrap();
        let route = compiled_change(compiled(
            &data,
            &graph,
            AttributeTarget::DistributionChange {
                change: change(),
                estimator: ChangeEstimator::Robust { max_rows: 1000, n_samples: 200, seed: 5 },
            },
        ));
        assert_change_eq(&route, &direct);
    });
}

#[test]
fn root_cause_rank_equals_the_compiled_root_causes_route() {
    on_large_stack("twin-root-causes", || {
        let (data, graph) = two_period();
        let fitted = fit_gcm(graph.clone(), &data).unwrap();
        let options = DistributionChangeOptions {
            measure: DifferenceMeasure::MeanDiff,
            n_samples: 400,
            seed: 3,
        };
        let direct_attribution =
            distribution_change(&fitted.model, &data, &change(), &options, &ctx()).unwrap();
        let direct_ranks = root_cause_rank(&direct_attribution, None, None, &ctx()).unwrap();
        let outcome = compiled(
            &data,
            &graph,
            AttributeTarget::RootCauses {
                change: change(),
                estimator: ChangeEstimator::Sampled {
                    measure: ChangeMeasure::MeanDifference,
                    n_samples: 400,
                    seed: 3,
                },
            },
        );
        let (attribution, ranks) = match outcome {
            AttributionOutcome::RootCauses { attribution, ranks } => (attribution, ranks),
            other => panic!("a root-cause target returns its attribution and ranks: {other:?}"),
        };
        assert_change_eq(&attribution, &direct_attribution);
        assert!(!direct_ranks.is_empty(), "every changed mechanism is ranked");
        assert_eq!(rank_bits(&ranks), rank_bits(&direct_ranks));
    });
}

#[test]
fn path_decompose_equals_the_compiled_paths_route() {
    on_large_stack("twin-paths", || {
        let (data, graph) = chain();
        let fitted = fit_gcm(graph.clone(), &data).unwrap();
        let direct = path_decompose(&fitted.model, &[v(0)], v(2), 16, 4, &ctx()).unwrap();
        let route = compiled_change(compiled(
            &data,
            &graph,
            AttributeTarget::Paths {
                sources: Arc::from([v(0)]),
                outcome: v(2),
                max_paths: 16,
                max_len: 4,
                through: Arc::from([]),
            },
        ));
        assert!(!direct.path_breakdown.is_empty(), "z reaches y along two paths");
        assert_change_eq(&route, &direct);
    });
}

#[test]
fn feature_relevance_equals_the_compiled_feature_relevance_route() {
    on_large_stack("twin-feature-relevance", || {
        let (data, graph) = chain();
        let fitted = fit_gcm(graph.clone(), &data).unwrap();
        let options = FeatureRelevanceOptions { delta: 1.0, n_samples: 200, max_features: 4 };
        let direct =
            feature_relevance(&fitted.model, &data, v(2), &[v(0), v(1)], &options, &ctx()).unwrap();
        let outcome = compiled(
            &data,
            &graph,
            AttributeTarget::FeatureRelevance {
                outcome: v(2),
                features: Arc::from([v(0), v(1)]),
                delta: 1.0,
                n_samples: 200,
                max_features: 4,
            },
        );
        let route = match outcome {
            AttributionOutcome::FeatureRelevance(route) => route,
            other => panic!("a feature-relevance target returns its scores: {other:?}"),
        };
        assert_eq!(direct.len(), 2, "one score per feature");
        assert_eq!(feature_bits(&route), feature_bits(&direct));
    });
}

#[test]
fn score_anomalies_equals_the_compiled_anomaly_route() {
    on_large_stack("twin-anomaly", || {
        let (data, graph) = chain();
        let fitted = fit_gcm(graph.clone(), &data).unwrap();
        let query = AnomalyAttributionQuery::new([v(2)], 40)
            .with_unit_rows((0..40).collect::<Vec<usize>>());
        let direct = score_anomalies_with(&fitted.model, &data, &query, &ctx()).unwrap();
        let without_context = score_anomalies(&fitted.model, &data, &query).unwrap();
        let outcome = compiled(&data, &graph, AttributeTarget::Anomaly(query));
        let route = match outcome {
            AttributionOutcome::Anomaly(route) => route,
            other => panic!("an anomaly target returns its scores: {other:?}"),
        };
        assert_anomalies_eq(&route, &direct);
        assert_anomalies_eq(&route, &without_context);
    });
}

#[test]
fn unit_change_equals_the_compiled_unit_change_route() {
    on_large_stack("twin-unit-change", || {
        let (data, graph) = two_period();
        let fitted = fit_gcm(graph.clone(), &data).unwrap();
        let query = UnitChangeQuery::new(v(1), 20).with_unit_rows((0..20).collect::<Vec<usize>>());
        let direct = unit_change(&fitted.model, &data, &query, &ctx()).unwrap();
        let outcome = compiled(&data, &graph, AttributeTarget::UnitChange(query));
        let route = match outcome {
            AttributionOutcome::UnitChange(route) => route,
            other => panic!("a unit-change target returns its attribution: {other:?}"),
        };
        assert_unit_change_eq(&route, &direct);
    });
}

#[test]
fn structure_change_equals_the_compiled_structure_change_route() {
    on_large_stack("twin-structure-change", || {
        let (data, graph) = two_period();
        let fitted = fit_gcm(graph.clone(), &data).unwrap();
        let comparison: Arc<[(VariableId, VariableId)]> = Arc::from([]);
        let query = change().with_components(AttributionComponents::Structure);
        let options = StructureChangeOptions {
            measure: DifferenceMeasure::MeanDiff,
            n_samples: 200,
            seed: 5,
        };
        let direct = structure_change(
            &fitted.model,
            &comparison_model(2, &comparison),
            &data,
            &query,
            &options,
            &ctx(),
        )
        .unwrap();
        let route = compiled_change(compiled(
            &data,
            &graph,
            AttributeTarget::StructureChange {
                change: query,
                comparison,
                measure: ChangeMeasure::MeanDifference,
                n_samples: 200,
                seed: 5,
            },
        ));
        assert_change_eq(&route, &direct);
    });
}

#[test]
fn intrinsic_influence_equals_the_compiled_intrinsic_influence_route() {
    on_large_stack("twin-intrinsic-influence", || {
        let (data, graph) = chain();
        let fitted = fit_gcm(graph.clone(), &data).unwrap();
        let query = IntrinsicInfluenceQuery::new(v(2))
            .with_allocation(IntrinsicAllocation::Exact)
            .with_simulation(40, 10, 3)
            .with_fit_bootstrap(0)
            .with_fit_resampling(FitResampling::RowBootstrap)
            .with_level(0.95)
            .with_max_sources(8)
            .with_seed(11);
        let direct = intrinsic_influence(
            &fitted.model,
            NoiseIndependence::Declared,
            Some(&data),
            &query,
            &ctx(),
        )
        .unwrap();
        let outcome = compiled(
            &data,
            &graph,
            AttributeTarget::IntrinsicInfluence(IntrinsicInfluenceTarget {
                target: v(2),
                allocation: InfluenceAllocation::Exact,
                outer_draws: 40,
                inner_draws: 10,
                replicates: 3,
                fit_bootstrap_replicates: 0,
                level: 0.95,
                max_sources: 8,
                seed: 11,
                noise: NoiseDeclaration::Independent,
            }),
        );
        let route = match outcome {
            AttributionOutcome::IntrinsicInfluence(route) => route,
            other => panic!("an intrinsic-influence target returns its attribution: {other:?}"),
        };
        assert_eq!(direct.sources.len(), 3, "one share per noise source");
        assert_influence_eq(&route, &direct);
    });
}

#[test]
fn detect_mechanism_changes_equals_the_compiled_mechanism_change_route() {
    on_large_stack("twin-mechanism-change", || {
        let (data, graph) = two_period();
        let fitted = fit_gcm(graph.clone(), &data).unwrap();
        let query = MechanismChangeQuery::new(
            [v(0), v(1)],
            PopulationSelector::TimeRange { start: 0, end: HALF },
            PopulationSelector::TimeRange { start: HALF, end: ROWS },
            0.05,
            4,
        );
        let direct = detect_mechanism_changes(
            &fitted.model,
            &data,
            &query,
            MechanismChangeMethod::LikelihoodRatio,
            &ctx(),
        )
        .unwrap();
        let outcome = compiled(
            &data,
            &graph,
            AttributeTarget::MechanismChange { query, test: MechanismChangeTest::LikelihoodRatio },
        );
        let route = match outcome {
            AttributionOutcome::MechanismChange(route) => route,
            other => panic!("a mechanism-change target returns its detections: {other:?}"),
        };
        assert_eq!(direct.len(), 2, "one detection per tested mechanism");
        assert_eq!(detection_bits(&route), detection_bits(&direct));
    });
}

#[test]
fn change_attribution_run_equals_the_compiled_sampled_change_route() {
    on_large_stack("twin-change-builder", || {
        let (data, graph) = two_period();
        let fitted = fit_gcm(graph.clone(), &data).unwrap();
        let direct = ChangeAttribution::new()
            .outcome(v(1))
            .baseline(PopulationSelector::TimeRange { start: 0, end: HALF })
            .comparison(PopulationSelector::TimeRange { start: HALF, end: ROWS })
            .with_sampling(DifferenceMeasure::MeanDiff, 400, 3)
            .run(&fitted.model, &data, &ctx())
            .unwrap();
        let route = compiled_change(compiled(
            &data,
            &graph,
            AttributeTarget::DistributionChange {
                change: change(),
                estimator: ChangeEstimator::Sampled {
                    measure: ChangeMeasure::MeanDifference,
                    n_samples: 400,
                    seed: 3,
                },
            },
        ));
        assert_change_eq(&route, &direct);
    });
}

#[test]
fn change_attribution_run_structure_equals_the_compiled_structure_change_route() {
    on_large_stack("twin-structure-builder", || {
        let (data, graph) = two_period();
        let fitted = fit_gcm(graph.clone(), &data).unwrap();
        let comparison: Arc<[(VariableId, VariableId)]> = Arc::from([]);
        let direct = ChangeAttribution::new()
            .outcome(v(1))
            .baseline(PopulationSelector::TimeRange { start: 0, end: HALF })
            .comparison(PopulationSelector::TimeRange { start: HALF, end: ROWS })
            .components(AttributionComponents::Structure)
            .with_sampling(DifferenceMeasure::MeanDiff, 200, 5)
            .run_structure(&fitted.model, &comparison_model(2, &comparison), &data, &ctx())
            .unwrap();
        let route = compiled_change(compiled(
            &data,
            &graph,
            AttributeTarget::StructureChange {
                change: change().with_components(AttributionComponents::Structure),
                comparison,
                measure: ChangeMeasure::MeanDifference,
                n_samples: 200,
                seed: 5,
            },
        ));
        assert_change_eq(&route, &direct);
    });
}
