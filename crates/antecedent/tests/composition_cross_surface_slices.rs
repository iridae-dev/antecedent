//! Cross-surface slice cases whose producers live on the study facade.
//!
//! Each case builds through a public API, drops that producer, and reads the
//! artifact back. Scientific inputs and hand values come from
//! `conformance/cross_surface/slice_cases.json`.
//!
//! The line is `y = 1 + 2t`, so every Gaussian cell's contrast is 2. The ledger
//! pools `y = (1, 3)` with variances `(1, 4)` to `7/5` and `4/5`. The temporal
//! point is `0.3 * 0.35 + 0.7 * 0.55 = 0.49` once even and odd unit shocks cancel.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#[path = "../../antecedent-estimate/tests/support/recovery_scm.rs"]
mod recovery_scm;

use std::sync::Arc;

use antecedent::analysis::evidence_ledger::{ObservationId, SourceDeclaration, SourceKind};
use antecedent::{
    BayesianConfig, Dag, InferenceMode, IntervalEstimand, PublishedScalarUncertainty, RefuteSuite,
    Study, StudyBuilder, StudyResult, TemporalDag, TemporalEffectQuery, TemporalPolicy,
    consume_evidence_ledger_artifact, consume_observation_recovery_artifact,
    temporal_dependent_interval,
};
use antecedent_core::{
    AverageEffectQuery, CausalSchemaBuilder, ExecutionContext, IntervalMethod, Lag, VariableId,
};
use antecedent_data::{TabularData, TimeSeriesData};
use antecedent_estimate::temporal_dependent_interval::{
    DependentIntervalConfig, SequenceHistory, UnitHistories,
    IntervalMethod as TemporalIntervalMethod,
};
use antecedent_estimate::temporal_initial_state::{
    InitialStateLaw, InitialStatePopulation, InitialStateSpec,
};
use antecedent_expr::{Assignment, ExactEvaluationLimits};
use antecedent_graph::{Admg, DenseNodeId, NodeRef};
use antecedent_identify::{RecoveredEffectQuery, RecoveryLimits};
use antecedent_io::measured_inference::MeasuredInference;
use antecedent_io::recovery_artifact::RecoveryConsumeLimits;
use antecedent_state::retention::{RetainedOperation, RetentionMode};
use antecedent_test_support::{conformance_dir, read_fixture_bytes};
use recovery_scm::{MModel, POPULATION, SNAPSHOT, v};

const EDGES: [(u32, u32); 3] = [(2, 0), (2, 1), (0, 1)];

fn cases() -> serde_json::Value {
    let bytes = read_fixture_bytes(
        &conformance_dir("cross_surface"),
        "slice_cases.json",
        "slice_cases.json is a committed cross-surface fixture",
    );
    serde_json::from_slice(&bytes).expect("slice_cases.json")
}

fn f64s(value: &serde_json::Value) -> Vec<f64> {
    value.as_array().expect("numbers").iter().map(|item| item.as_f64().expect("f64")).collect()
}

fn close(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "{actual} differs from {expected} by more than {tolerance}"
    );
}

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(1)
}

struct GaussianCell {
    horizon: bool,
    bayesian: bool,
}

impl GaussianCell {
    const fn label(&self) -> &'static str {
        match (self.horizon, self.bayesian) {
            (false, false) => "gaussian-static-frequentist",
            (false, true) => "gaussian-static-bayesian",
            (true, false) => "gaussian-horizon-frequentist",
            (true, true) => "gaussian-horizon-bayesian",
        }
    }

    const fn tolerance(&self) -> f64 {
        match (self.horizon, self.bayesian) {
            (false, false) => 1e-12,
            (true, false) => 1e-8,
            (false, true) => 1e-6,
            (true, true) => 1e-4,
        }
    }
}

fn static_study(treatment: &[f64], outcome: &[f64]) -> antecedent::StudyBuilder {
    let schema = CausalSchemaBuilder::new()
        .continuous("t")
        .treatment()
        .continuous("y")
        .outcome()
        .build()
        .unwrap();
    let data = TabularData::try_from_schema_f64(schema.clone(), [("t", treatment), ("y", outcome)])
        .unwrap();
    let graph = Dag::from_named_edges(&schema, &[("t", "y")]).unwrap();
    let query = AverageEffectQuery::binary_ate(
        schema.id_of("t").unwrap(),
        schema.id_of("y").unwrap(),
    );
    Study::tabular(data).graph(graph).query(query)
}

fn horizon_study(treatment: &[f64], outcome: &[f64]) -> antecedent::StudyBuilder {
    let series = TimeSeriesData::from_f64_columns(
        [("t", treatment), ("y", outcome)],
        1,
    )
    .unwrap();
    let treatment_id = VariableId::from_raw(0);
    let outcome_id = VariableId::from_raw(1);
    let mut graph = TemporalDag::empty();
    let lagged =
        antecedent_graph::ensure_lagged(&mut graph, treatment_id, Lag::from_raw(1)).unwrap();
    let now =
        antecedent_graph::ensure_lagged(&mut graph, outcome_id, Lag::CONTEMPORANEOUS).unwrap();
    graph.insert_directed(lagged, now).unwrap();
    let query = TemporalEffectQuery::pulse(treatment_id, outcome_id, 1.0)
        .with_policy(TemporalPolicy::pulse(-1))
        .with_horizon_steps(1);
    Study::series(series).graph(graph).temporal_query(query)
}

fn finish(
    builder: antecedent::StudyBuilder,
    cell: &GaussianCell,
    draws: usize,
    scale: f64,
) -> Study {
    let builder = if cell.bayesian {
        builder.inference(InferenceMode::Bayesian(
            BayesianConfig::conjugate().n_draws(draws).prior_scale(scale),
        ))
    } else {
        builder
    };
    builder.refute(RefuteSuite::None).bootstrap_replicates(0).build().unwrap()
}

fn reload_gaussian(cell: &GaussianCell, fixture: &serde_json::Value) {
    let gaussian = &fixture["gaussian"];
    let builder = if cell.horizon {
        horizon_study(&f64s(&gaussian["horizon_treatment"]), &f64s(&gaussian["horizon_outcome"]))
    } else {
        static_study(&f64s(&gaussian["treatment"]), &f64s(&gaussian["outcome"]))
    };
    let draws = usize::try_from(gaussian["posterior_draws"].as_u64().unwrap()).unwrap();
    let prepared = finish(builder, cell, draws, gaussian["prior_scale"].as_f64().unwrap())
        .prepare(&ctx())
        .unwrap();
    let result = prepared.estimate_retained(&ctx()).unwrap();
    let effect = result.estimate.as_effect().expect("effect");
    let truth = gaussian["effect"].as_f64().unwrap();
    assert!(
        (effect.ate - truth).abs() <= cell.tolerance(),
        "{} ate {} differs from {} by more than {}",
        cell.label(),
        effect.ate,
        truth,
        cell.tolerance()
    );
    assert_interval(&result, cell.bayesian);
    if cell.bayesian {
        let posterior = result.posterior.as_ref().expect("posterior");
        let column = posterior.effect_column().expect("effect column");
        close(posterior.summaries.mean[column], truth, 0.05);
    } else {
        assert!(result.posterior.is_none());
    }
    let bytes = prepared.encode_contracted_result(&result, cell.label(), &ctx()).unwrap();
    drop(prepared);
    let consumed = antecedent_io::consume_analysis_result(&bytes).unwrap();
    assert!(consumed.acceptance.accepts_as_verified_program());
    close(consumed.body.estimate.expect("contrast"), truth, cell.tolerance());
}

fn assert_interval(result: &StudyResult, bayesian: bool) {
    let method = result.primary_interval_binding(bayesian).method;
    if bayesian {
        assert_eq!(method, IntervalMethod::PosteriorQuantile);
    } else {
        let published = PublishedScalarUncertainty::select(result.estimate.as_effect().unwrap());
        assert_ne!(published.method, IntervalMethod::PosteriorQuantile);
    }
}

/// Static and horizon, frequentist and Bayesian, each reloaded from its own artifact.
#[test]
fn four_way_gaussian_cells_reload_from_independent_artifacts() {
    let fixture = cases();
    for cell in [
        GaussianCell { horizon: false, bayesian: false },
        GaussianCell { horizon: false, bayesian: true },
        GaussianCell { horizon: true, bayesian: false },
        GaussianCell { horizon: true, bayesian: true },
    ] {
        reload_gaussian(&cell, &fixture);
    }
}

fn observation_rows(start: u32, end: u32) -> Vec<ObservationId> {
    (start..end).map(|index| ObservationId::from_key("cohort", &index.to_le_bytes())).collect()
}

fn range_pair(value: &serde_json::Value) -> (u32, u32) {
    let pair = value.as_array().unwrap();
    (
        u32::try_from(pair[0].as_u64().unwrap()).unwrap(),
        u32::try_from(pair[1].as_u64().unwrap()).unwrap(),
    )
}

/// Two disjoint studies pool to 7/5 and 4/5, then a consumer recomputes that pool.
#[test]
fn mixed_evidence_pools_and_reloads_from_the_ledger() {
    let mixed = cases()["mixed_evidence"].clone();
    let estimates = f64s(&mixed["estimates"]);
    let variances = f64s(&mixed["variances"]);
    let ranges = mixed["observation_ranges"].as_array().unwrap();
    let mut session = StudyBuilder::evidence_ledger();
    let mut summaries = Vec::new();
    for (index, range) in ranges.iter().enumerate() {
        let (start, end) = range_pair(range);
        let source = session
            .add_source(
                SourceDeclaration::new(format!("study_{index}"), SourceKind::ObservationSet)
                    .with_observations(observation_rows(start, end)),
            )
            .unwrap();
        summaries.push(antecedent::StudySummary {
            source,
            estimate: estimates[index],
            variance: variances[index],
        });
    }
    let pooled = session
        .pool_summaries(
            antecedent::PoolingRequest { label: "pool".into(), summaries, replaces: None },
            antecedent::PoolingDeclaration::Independent,
            &ctx(),
        )
        .unwrap();
    let antecedent::SynthesisValue::FixedEffect(fit) = &pooled.value else {
        panic!("independent pooling is a fixed-effect pool");
    };
    close(fit.estimate, mixed["pooled"].as_f64().unwrap(), 1e-12);
    close(fit.variance, mixed["pooled_variance"].as_f64().unwrap(), 1e-12);
    let declared: u64 = ranges
        .iter()
        .map(|range| {
            let (start, end) = range_pair(range);
            u64::from(end - start)
        })
        .sum();
    assert_eq!(pooled.mass.distinct_observations, declared);
    let export = session.export("mixed-evidence").unwrap();
    drop(session);
    let consumed = consume_evidence_ledger_artifact(&export.bytes, Some(&export.identity)).unwrap();
    let values = &consumed.synthesis_of(pooled.result).unwrap().values;
    close(values[0], mixed["pooled"].as_f64().unwrap(), 1e-12);
    close(values[1], mixed["pooled_variance"].as_f64().unwrap(), 1e-12);
}

/// Citing a result twice does not count its observations again.
#[test]
fn reused_evidence_keeps_one_observation_count() {
    let mixed = cases()["mixed_evidence"].clone();
    let estimates = f64s(&mixed["estimates"]);
    let variances = f64s(&mixed["variances"]);
    let ranges = mixed["observation_ranges"].as_array().unwrap();
    let mut session = StudyBuilder::evidence_ledger();
    let mut summaries = Vec::new();
    for (index, range) in ranges.iter().enumerate() {
        let (start, end) = range_pair(range);
        let source = session
            .add_source(
                SourceDeclaration::new(format!("study_{index}"), SourceKind::ObservationSet)
                    .with_observations(observation_rows(start, end)),
            )
            .unwrap();
        summaries.push(antecedent::StudySummary {
            source,
            estimate: estimates[index],
            variance: variances[index],
        });
    }
    let pooled = session
        .pool_summaries(
            antecedent::PoolingRequest { label: "pool".into(), summaries, replaces: None },
            antecedent::PoolingDeclaration::Independent,
            &ctx(),
        )
        .unwrap();
    let first = session.reuse_result("cited_once", pooled.result).unwrap();
    let second = session.reuse_result("cited_again", pooled.result).unwrap();
    assert_ne!(first, second);
    for id in [pooled.result, first, second] {
        assert_eq!(session.result(id).unwrap().mass, pooled.mass);
    }
    let export = session.export("reused-evidence").unwrap();
    drop(session);
    let consumed = consume_evidence_ledger_artifact(&export.bytes, Some(&export.identity)).unwrap();
    let counted = pooled.mass.distinct_observations;
    for id in [pooled.result, first, second] {
        let record = consumed.ledger().result(id).unwrap();
        assert_eq!(record.mass.distinct_observations, counted);
    }
}

fn one_unit(unit: usize) -> UnitHistories {
    let mut histories = Vec::new();
    let mut time = 0_u64;
    let shock = if unit % 2 == 0 { -0.2 } else { 0.2 };
    for s0 in 0..2_u32 {
        for a1 in 0..2_u32 {
            for l2 in 0..2_u32 {
                for a2 in 0..2_u32 {
                    let state = f64::from(s0);
                    let covariate = f64::from(l2);
                    let actions = f64::from(a1) + f64::from(a2);
                    let mean = 0.3
                        + 0.1 * covariate
                        + 0.1 * state
                        + 0.2 * state * covariate
                        + 0.05 * actions;
                    histories.push(SequenceHistory {
                        time_id: time,
                        s0,
                        a1,
                        l2,
                        a2,
                        y: mean + shock,
                    });
                    time += 1;
                }
            }
        }
    }
    UnitHistories { unit_id: u64::try_from(unit).unwrap(), histories }
}

/// The measured whole-unit response reloads at the hand marginal 0.49.
#[test]
fn temporal_dependence_reloads_from_the_measured_artifact() {
    let temporal = cases()["temporal"].clone();
    let units = usize::try_from(temporal["units"].as_u64().unwrap()).unwrap();
    let panel: Vec<_> = (0..units).map(one_unit).collect();
    let states = temporal["target_law"]
        .as_array()
        .unwrap()
        .iter()
        .map(|pair| {
            let pair = pair.as_array().unwrap();
            (u32::try_from(pair[0].as_u64().unwrap()).unwrap(), pair[1].as_f64().unwrap())
        })
        .collect();
    let law = InitialStateLaw::new(
        InitialStatePopulation::Target,
        temporal["target_snapshot"].as_str().unwrap(),
        states,
    )
    .unwrap();
    let sequence = temporal["sequence"].as_array().unwrap();
    let sequence = [
        u32::try_from(sequence[0].as_u64().unwrap()).unwrap(),
        u32::try_from(sequence[1].as_u64().unwrap()).unwrap(),
    ];
    let config = DependentIntervalConfig {
        replicates: usize::try_from(temporal["replicates"].as_u64().unwrap()).unwrap(),
        seed: temporal["seed"].as_u64().unwrap(),
        level: 0.95,
        method: TemporalIntervalMethod::Studentized,
        min_units: usize::try_from(temporal["min_units"].as_u64().unwrap()).unwrap(),
        max_failed_fraction: 0.05,
    };
    let produced = temporal_dependent_interval(
        temporal["snapshot"].as_str().unwrap(),
        Some(panel),
        sequence,
        IntervalEstimand::Marginalized(InitialStateSpec::Law(law)),
        &config,
    )
    .unwrap();
    let expected = produced.expectation().clone();
    let bytes = produced.export().to_vec();
    let point = temporal["point"].as_f64().unwrap();
    let scalar = produced.report().scalars.iter().find(|item| item.name == "response").unwrap();
    close(scalar.point, point, 1e-12);
    assert_eq!(produced.report().calibration, "unmeasured");
    drop(produced);
    let consumed = MeasuredInference::consume(&bytes, &expected, &ctx()).unwrap();
    let scalar = consumed.report().scalars.iter().find(|item| item.name == "response").unwrap();
    close(scalar.point, point, 1e-12);
}

fn recovery_model() -> MModel {
    MModel::new(2, 1, &EDGES, &[vec![2], vec![0]], 17).unwrap()
}

fn recovery_effect(model: &MModel) -> RecoveredEffectQuery {
    let mut graph = Admg::empty();
    for node in 0..model.k + model.m {
        graph.add_node(NodeRef::Static(v(node))).unwrap();
    }
    for (from, to) in EDGES {
        graph.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).unwrap();
    }
    RecoveredEffectQuery { graph, outcomes: Arc::from([v(1)]), treatments: Arc::from([v(0)]) }
}

/// The recovered observation law matches the structural truth after the producer is gone.
#[test]
fn recovered_feedback_reloads_the_observation_law() {
    let recovery = cases()["recovery"].clone();
    assert_eq!(POPULATION, recovery["population"].as_str().unwrap());
    assert_eq!(SNAPSHOT, recovery["rust_snapshot"].as_str().unwrap());
    let model = recovery_model();
    let prepared = StudyBuilder::observation_recovery(
        model.graph.clone(),
        &model.query(),
        model.catalog(),
        Some(recovery_effect(&model)),
        RecoveryLimits::default(),
        antecedent::RecoveryEvaluation {
            observed: model.observed_law(),
            requests: [0.0, 1.0]
                .into_iter()
                .map(|level| Assignment::from_pairs([(v(0), antecedent_core::Value::f64(level))]))
                .collect(),
            limits: ExactEvaluationLimits::default(),
        },
        &ctx(),
    )
    .unwrap();
    let result = prepared.estimate(&ctx()).unwrap();
    let names: Vec<String> = (0..model.nodes()).map(|node| format!("n{node}")).collect();
    let bytes = result.export_named(&prepared, &names).unwrap();
    let live: Vec<f64> = result.recovered().law().probabilities().to_vec();
    drop(prepared);
    drop(result);
    let consumed = consume_observation_recovery_artifact(
        &bytes,
        RecoveryConsumeLimits::default(),
        &ctx(),
    )
    .unwrap();
    assert!(consumed.recovered().law().snapshot_identity().starts_with("recovered:"));
    for (actual, expected) in consumed.recovered().law().probabilities().iter().zip(&live) {
        close(*actual, *expected, 1e-12);
    }
    for (actual, expected) in consumed.recovered().law().probabilities().iter().zip(model.truth()) {
        close(*actual, expected, 1e-12);
    }
}

/// Summary-only retention, named by the fixture, serves a reuse summary and nothing else.
#[test]
fn reduced_retention_serves_only_the_summary() {
    let cases = cases();
    let name = cases["retention"]["reduced"].as_str().unwrap();
    let mode = RetentionMode::ALL.into_iter().find(|mode| mode.as_str() == name).expect(name);
    assert!(mode.serves(RetainedOperation::ReuseSummary));
    assert!(mode.require(RetainedOperation::ReuseSummary).is_ok());
    assert!(mode.require(RetainedOperation::RowReconstruction).is_err());
}
