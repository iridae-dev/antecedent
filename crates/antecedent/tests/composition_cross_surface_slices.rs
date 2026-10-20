//! Cross-surface slice cases whose producers live on the study facade.
//!
//! Each case builds through a public API, drops that producer, and reads the
//! artifact back. Scientific inputs and hand values come from
//! `conformance/cross_surface/slice_cases.json`.
//!
//! The line is `y = 1 + 2t`, so every Gaussian cell's contrast is 2. The ledger
//! pools `y = (1, 3)` with variances `(1, 4)` to `7/5` and `4/5`; once the second study
//! grows to share 10 observations with the first and declares covariance `0.5`, the
//! generalized-least-squares pool is `5/4` with variance `15/16`. The temporal point is
//! `0.3 * 0.35 + 0.7 * 0.55 = 0.49` once even and odd unit shocks cancel. Shocks of `0` and
//! `0.4` add their mean `0.2` and leave the unit scores two-valued, so the whole-unit standard
//! error is `0.2 * sqrt(100 / 99) / 10`.
//!
//! The finite-discrete adjustment has no producer on the study facade, so it is covered in
//! `antecedent-io/tests/composition_cross_surface.rs` from the law artifact.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#[path = "../../antecedent-estimate/tests/support/recovery_scm.rs"]
mod recovery_scm;

use std::sync::Arc;

use antecedent::analysis::evidence_ledger::{
    EvidenceLedgerSession, ObservationId, ResultStatus, Revalidation, SourceDeclaration, SourceId,
    SourceKind, SynthesisOutcome,
};
use antecedent::{
    BayesianConfig, Dag, InferenceMode, IntervalEstimand, PoolingDeclaration, PoolingRequest,
    PublishedScalarUncertainty, RefuteSuite, SourceRevision, Study, StudyBuilder, StudyResult,
    StudySummary, SynthesisValue, TemporalDag, TemporalEffectQuery, TemporalPolicy,
    consume_evidence_ledger_artifact, consume_observation_recovery_artifact,
    temporal_dependent_interval, temporal_dependent_interval_candidate,
};
use antecedent_core::{
    AverageEffectQuery, CausalSchemaBuilder, ExecutionContext, IntervalMethod, Lag, VariableId,
};
use antecedent_data::{TabularData, TimeSeriesData};
use antecedent_estimate::temporal_dependent_interval::{
    DependentIntervalConfig, IntervalMethod as TemporalIntervalMethod, SequenceHistory,
    UnitHistories,
};
use antecedent_estimate::temporal_initial_state::{
    InitialStateLaw, InitialStatePopulation, InitialStateSpec,
};
use antecedent_expr::{Assignment, ExactEvaluationLimits};
use antecedent_graph::{Admg, DenseNodeId, NodeRef};
use antecedent_identify::{RecoveredEffectQuery, RecoveryLimits};
use antecedent_io::measured_inference::MeasuredInference;
use antecedent_io::recovery_artifact::RecoveryConsumeLimits;
use antecedent_io::temporal_interval_artifact::{
    TemporalIntervalArtifactWire, TemporalIntervalConsumeLimits,
};
use antecedent_state::retention::{RetainedOperation, RetentionMode};
use antecedent_state::{OlsRefresh, RefreshRow};
use antecedent_test_support::slice_cases::{self, Gaussian, MixedEvidence, Retention, Temporal};
use antecedent_test_support::{assert_all_close, assert_close};
use recovery_scm::{MModel, POPULATION, SNAPSHOT, v};

const EDGES: [(u32, u32); 3] = [(2, 0), (2, 1), (0, 1)];
/// Exact arithmetic on a noiseless line, or the same pooled value recomputed by a consumer.
const EXACT: f64 = 1e-12;
/// A fixed-effect fit through the horizon route solves a slightly larger system.
const HORIZON_FIT: f64 = 1e-8;
const MONTE_CARLO_STANDARD_ERRORS: f64 = 5.0;

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(1)
}

/// A count as a real. Every count in the fixtures is far below `u32::MAX`.
fn count(value: usize) -> f64 {
    f64::from(u32::try_from(value).expect("a fixture count fits in u32"))
}

#[derive(Clone, Copy, Debug)]
enum GaussianCell {
    StaticFrequentist,
    StaticBayesian,
    HorizonFrequentist,
    HorizonBayesian,
}

impl GaussianCell {
    const ALL: [Self; 4] = [
        Self::StaticFrequentist,
        Self::StaticBayesian,
        Self::HorizonFrequentist,
        Self::HorizonBayesian,
    ];

    const fn is_horizon(self) -> bool {
        matches!(self, Self::HorizonFrequentist | Self::HorizonBayesian)
    }

    const fn is_bayesian(self) -> bool {
        matches!(self, Self::StaticBayesian | Self::HorizonBayesian)
    }

    const fn label(self) -> &'static str {
        match self {
            Self::StaticFrequentist => "gaussian-static-frequentist",
            Self::StaticBayesian => "gaussian-static-bayesian",
            Self::HorizonFrequentist => "gaussian-horizon-frequentist",
            Self::HorizonBayesian => "gaussian-horizon-bayesian",
        }
    }

    /// The checked-operation reference the independent consumer leaves unresolved. Only the static
    /// frequentist line has an independent replay; the body of every other cell is readable and
    /// forwardable, and is not claimed as a verified program.
    const fn unresolved_reference(self) -> &'static [&'static str] {
        match self {
            Self::StaticFrequentist => &[],
            Self::StaticBayesian => &["dependencies.checked_bayesian_dag_ate_operation"],
            Self::HorizonFrequentist => &["dependencies.checked_temporal_dag_effect_operation"],
            Self::HorizonBayesian => {
                &["dependencies.checked_bayesian_temporal_dag_effect_operation"]
            }
        }
    }

    /// A frequentist fit is exact on the noiseless line. A Bayesian point estimate is the mean of
    /// iid conjugate draws, so it carries Monte Carlo error: five standard errors of that mean.
    fn tolerance(self, result: &StudyResult) -> f64 {
        match self {
            Self::StaticFrequentist => EXACT,
            Self::HorizonFrequentist => HORIZON_FIT,
            Self::StaticBayesian | Self::HorizonBayesian => {
                let posterior = result.posterior.as_ref().expect("posterior");
                let column = posterior.effect_column().expect("effect column");
                MONTE_CARLO_STANDARD_ERRORS * posterior.summaries.sd[column]
                    / count(posterior.summaries.n_draws).sqrt()
            }
        }
    }
}

fn static_study(treatment: &[f64], outcome: &[f64]) -> StudyBuilder {
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
    let query =
        AverageEffectQuery::binary_ate(schema.id_of("t").unwrap(), schema.id_of("y").unwrap());
    Study::tabular(data).graph(graph).query(query)
}

fn horizon_study(treatment: &[f64], outcome: &[f64]) -> StudyBuilder {
    let series = TimeSeriesData::from_f64_columns([("t", treatment), ("y", outcome)], 1).unwrap();
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

fn finish(builder: StudyBuilder, cell: GaussianCell, gaussian: &Gaussian) -> Study {
    let builder = if cell.is_bayesian() {
        builder.inference(InferenceMode::Bayesian(
            BayesianConfig::conjugate()
                .n_draws(gaussian.posterior_draws)
                .prior_scale(gaussian.prior_scale),
        ))
    } else {
        builder
    };
    builder.refute(RefuteSuite::None).bootstrap_replicates(0).build().unwrap()
}

fn reload_gaussian(cell: GaussianCell, gaussian: &Gaussian) {
    let builder = if cell.is_horizon() {
        horizon_study(&gaussian.horizon_treatment, &gaussian.horizon_outcome)
    } else {
        static_study(&gaussian.treatment, &gaussian.outcome)
    };
    let prepared = finish(builder, cell, gaussian).prepare(&ctx()).unwrap();
    let result = prepared.estimate_retained(&ctx()).unwrap();
    let effect = result.estimate.as_effect().expect("effect");
    let truth = gaussian.effect;
    let tolerance = cell.tolerance(&result);
    assert_close(effect.ate, truth, tolerance);
    assert_interval(&result, cell.is_bayesian());
    if cell.is_bayesian() {
        let posterior = result.posterior.as_ref().expect("posterior");
        let column = posterior.effect_column().expect("effect column");
        assert_close(posterior.summaries.mean[column], truth, 0.05);
    } else {
        assert!(result.posterior.is_none());
    }
    let bytes = prepared.encode_contracted_result(&result, cell.label(), &ctx()).unwrap();
    drop(prepared);
    let consumed = antecedent_io::consume_analysis_result(&bytes).unwrap();
    let acceptance = &consumed.acceptance;
    let unresolved: Vec<&str> = acceptance.unresolved.iter().map(AsRef::as_ref).collect();
    assert_eq!(unresolved, cell.unresolved_reference(), "{} {acceptance:?}", cell.label());
    assert_eq!(acceptance.accepts_as_verified_program(), unresolved.is_empty());
    assert!(
        acceptance.supported_operations.iter().any(|operation| operation.as_ref() == "read_body"),
        "{} {acceptance:?}",
        cell.label()
    );
    assert_close(consumed.body.estimate.expect("contrast"), truth, tolerance);
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
    let fixture = slice_cases::load();
    for cell in GaussianCell::ALL {
        reload_gaussian(cell, &fixture.gaussian);
    }
}

fn observation_rows(start: u32, end: u32) -> Vec<ObservationId> {
    (start..end).map(|index| ObservationId::from_key("cohort", &index.to_le_bytes())).collect()
}

fn observation_source(label: &str, (start, end): (u32, u32)) -> SourceDeclaration {
    SourceDeclaration::new(label, SourceKind::ObservationSet)
        .with_observations(observation_rows(start, end))
}

fn study_summaries(sources: &[SourceId], mixed: &MixedEvidence) -> Vec<StudySummary> {
    sources
        .iter()
        .zip(&mixed.estimates)
        .zip(&mixed.variances)
        .map(|((source, estimate), variance)| StudySummary {
            source: *source,
            estimate: *estimate,
            variance: *variance,
        })
        .collect()
}

/// Two disjoint studies from the fixture, pooled by inverse variance.
struct PooledSession {
    session: EvidenceLedgerSession,
    sources: Vec<SourceId>,
    pooled: SynthesisOutcome,
}

fn pooled_session(mixed: &MixedEvidence) -> PooledSession {
    let mut session = StudyBuilder::evidence_ledger();
    let sources: Vec<SourceId> = mixed
        .observation_ranges
        .iter()
        .enumerate()
        .map(|(index, range)| {
            session.add_source(observation_source(&format!("study_{index}"), *range)).unwrap()
        })
        .collect();
    let pooled = session
        .pool_summaries(
            PoolingRequest {
                label: "pool".into(),
                summaries: study_summaries(&sources, mixed),
                replaces: None,
            },
            PoolingDeclaration::Independent,
            &ctx(),
        )
        .unwrap();
    PooledSession { session, sources, pooled }
}

/// Two disjoint studies pool to 7/5 and 4/5, then a consumer recomputes that pool.
#[test]
fn mixed_evidence_pools_and_reloads_from_the_ledger() {
    let mixed = slice_cases::load().mixed_evidence;
    let PooledSession { session, pooled, .. } = pooled_session(&mixed);
    let SynthesisValue::FixedEffect(fit) = &pooled.value else {
        panic!("independent pooling is a fixed-effect pool");
    };
    assert_close(fit.estimate, mixed.pooled, EXACT);
    assert_close(fit.variance, mixed.pooled_variance, EXACT);
    let declared: u64 =
        mixed.observation_ranges.iter().map(|(start, end)| u64::from(end - start)).sum();
    assert_eq!(pooled.mass.distinct_observations, declared);
    let export = session.export("mixed-evidence").unwrap();
    drop(session);
    let consumed = consume_evidence_ledger_artifact(&export.bytes, Some(&export.identity)).unwrap();
    let values = &consumed.synthesis_of(pooled.result).unwrap().values;
    assert_close(values[0], mixed.pooled, EXACT);
    assert_close(values[1], mixed.pooled_variance, EXACT);
}

/// Citing a result twice does not count its observations again.
#[test]
fn reused_evidence_keeps_one_observation_count() {
    let mixed = slice_cases::load().mixed_evidence;
    let PooledSession { mut session, pooled, .. } = pooled_session(&mixed);
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

/// The second study grows into the first. Independent pooling is a double count; a declared
/// covariance pools them by generalized least squares and replaces the disjoint pool, and a
/// consumer replays both the value and the replacement.
#[test]
fn overlapping_studies_pool_under_a_declared_covariance_and_replace_the_disjoint_pool() {
    let mixed = slice_cases::load().mixed_evidence;
    let overlap = &mixed.overlap;
    let PooledSession { mut session, sources, pooled } = pooled_session(&mixed);
    let disjoint = pooled.result;
    let report = session
        .revise_source(SourceRevision::Update {
            target: sources[1],
            replacement: observation_source("study_1_grown", overlap.grown_range),
        })
        .unwrap();
    let grown = report.created_source.expect("the update creates the grown source");
    assert_eq!(session.standing(disjoint), Some(Revalidation::Recompute));
    let account = session.account_observations(&[sources[0], grown]).unwrap();
    assert_eq!(account.mass.distinct_observations, overlap.distinct_observations);

    let summaries = study_summaries(&[sources[0], grown], &mixed);
    let request = |replaces| PoolingRequest {
        label: "overlap_pool".into(),
        summaries: summaries.clone(),
        replaces,
    };
    let refusal = session
        .pool_summaries(request(Some(disjoint)), PoolingDeclaration::Independent, &ctx())
        .unwrap_err();
    assert_eq!(refusal.code, "evidence_double_counted", "{refusal}");
    assert_eq!(refusal.detail, "evidence_ledger.shared_observations", "{refusal}");

    let replacing = session
        .pool_summaries(
            request(Some(disjoint)),
            PoolingDeclaration::DeclaredCovariance { covariance: overlap.covariance.clone() },
            &ctx(),
        )
        .unwrap();
    let SynthesisValue::FixedEffect(fit) = &replacing.value else {
        panic!("a declared covariance pools by generalized least squares");
    };
    assert_close(fit.estimate, overlap.pooled, EXACT);
    assert_close(fit.variance, overlap.pooled_variance, EXACT);
    assert_eq!(replacing.mass.declared_slots, overlap.declared_slots);
    assert_eq!(replacing.mass.distinct_observations, overlap.distinct_observations);
    assert_eq!(replacing.mass.repeated_slots, overlap.shared);
    let record = session.result(replacing.result).unwrap();
    assert!(record.dependence_declared);
    assert_eq!(record.replaces, Some(disjoint));
    assert!(matches!(
        &session.result(disjoint).unwrap().status,
        ResultStatus::Replaced { by, .. } if *by == replacing.result
    ));
    assert!(session.synthesis(disjoint).is_err());

    let export = session.export("overlapping-evidence").unwrap();
    drop(session);
    let consumed = consume_evidence_ledger_artifact(&export.bytes, Some(&export.identity)).unwrap();
    let values = &consumed.synthesis_of(replacing.result).unwrap().values;
    assert_close(values[0], overlap.pooled, EXACT);
    assert_close(values[1], overlap.pooled_variance, EXACT);
    assert!(matches!(
        &consumed.ledger().result(disjoint).unwrap().status,
        ResultStatus::Replaced { by, .. } if *by == replacing.result
    ));
}

fn one_unit(unit: usize, shocks: &[f64]) -> UnitHistories {
    let mut histories = Vec::new();
    let mut time = 0_u64;
    let shock = shocks[unit % shocks.len()];
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

fn temporal_panel(temporal: &Temporal, shocks: &[f64]) -> Vec<UnitHistories> {
    (0..temporal.units).map(|unit| one_unit(unit, shocks)).collect()
}

fn temporal_config(temporal: &Temporal) -> DependentIntervalConfig {
    DependentIntervalConfig {
        replicates: temporal.replicates,
        seed: temporal.seed,
        level: temporal.level,
        method: TemporalIntervalMethod::Studentized,
        min_units: temporal.min_units,
        max_failed_fraction: temporal.max_failed_fraction,
    }
}

fn temporal_estimand(temporal: &Temporal) -> IntervalEstimand {
    let law = InitialStateLaw::new(
        InitialStatePopulation::Target,
        temporal.target_snapshot.as_str(),
        temporal.target_law.clone(),
    )
    .unwrap();
    IntervalEstimand::Marginalized(InitialStateSpec::Law(law))
}

/// What a consumer reads back from the candidate artifact once its producer is gone.
struct ReloadedInterval {
    point: f64,
    lower: f64,
    upper: f64,
    standard_error: f64,
}

/// Build the candidate over `panel`, drop it, and consume its artifact.
fn reload_candidate(temporal: &Temporal, panel: &[UnitHistories], point: f64) -> ReloadedInterval {
    let config = temporal_config(temporal);
    let sequence = [temporal.sequence.0, temporal.sequence.1];
    // The candidate artifact replays the whole resampling from the embedded panel.
    let candidate = temporal_dependent_interval_candidate(
        temporal.snapshot.as_str(),
        Some(panel.to_vec()),
        sequence,
        temporal_estimand(temporal),
        &config,
        &ctx(),
    )
    .unwrap();
    assert_close(candidate.result.point, point, EXACT);
    assert_eq!(candidate.result.calibration, "unmeasured");
    let bytes = candidate.export().unwrap();
    drop(candidate);
    let (fresh, interval) = TemporalIntervalArtifactWire::consume(
        &bytes,
        &TemporalIntervalConsumeLimits::default(),
        &ctx(),
    )
    .unwrap();
    assert_close(interval.point, point, EXACT);
    assert_eq!(fresh.result.replicates.len(), config.replicates);
    let receipt = fresh.result.studentization.as_ref().expect("a studentized interval");
    ReloadedInterval {
        point: fresh.result.point,
        lower: fresh.result.lower,
        upper: fresh.result.upper,
        standard_error: receipt.standard_error,
    }
}

/// The measured whole-unit response reloads at the hand marginal 0.49.
#[test]
fn temporal_dependence_reloads_from_the_measured_artifact() {
    let temporal = slice_cases::load().temporal;
    let panel = temporal_panel(&temporal, &temporal.unit_shocks);
    reload_candidate(&temporal, &panel, temporal.point);

    // The measured route additionally needs a current attesting coverage record. Until the
    // coverage records are re-measured it refuses for exactly that reason and no other.
    let sequence = [temporal.sequence.0, temporal.sequence.1];
    match temporal_dependent_interval(
        temporal.snapshot.as_str(),
        Some(panel),
        sequence,
        temporal_estimand(&temporal),
        &temporal_config(&temporal),
    ) {
        Ok(produced) => {
            let expected = produced.expectation().clone();
            let bytes = produced.export().to_vec();
            drop(produced);
            let consumed = MeasuredInference::consume(&bytes, &expected, &ctx()).unwrap();
            let scalar =
                consumed.report().scalars.iter().find(|item| item.name == "response").unwrap();
            assert_close(scalar.point, temporal.point, EXACT);
        }
        Err(error) => {
            assert!(error.to_string().contains("coverage_record_not_attesting"), "{error}");
        }
    }
}

/// The sample standard error of the mean of every outcome row, as if the rows were independent.
fn independent_rows_standard_error(panel: &[UnitHistories]) -> f64 {
    let outcomes: Vec<f64> =
        panel.iter().flat_map(|unit| unit.histories.iter().map(|history| history.y)).collect();
    let rows = count(outcomes.len());
    let mean = outcomes.iter().sum::<f64>() / rows;
    let variance = outcomes.iter().map(|y| (y - mean).powi(2)).sum::<f64>() / (rows - 1.0);
    (variance / rows).sqrt()
}

/// Shocks of 0 and 0.4 do not cancel: they move the point by their mean and make the unit scores
/// two-valued. Every row of a unit carries its shock, so treating the rows as independent would
/// understate the width the whole-unit interval reports.
#[test]
fn uneven_unit_shocks_widen_the_whole_unit_interval_beyond_independent_rows() {
    let temporal = slice_cases::load().temporal;
    let uneven = &temporal.uneven;
    let panel = temporal_panel(&temporal, &uneven.unit_shocks);
    let reloaded = reload_candidate(&temporal, &panel, uneven.point);

    let mean_shock = (0..temporal.units)
        .map(|unit| uneven.unit_shocks[unit % uneven.unit_shocks.len()])
        .sum::<f64>()
        / count(temporal.units);
    assert!(mean_shock.abs() > 0.1, "the shocks do not cancel");
    assert_close(reloaded.point - temporal.point, mean_shock, EXACT);
    assert_close(reloaded.standard_error, uneven.unit_score_standard_error, EXACT);

    let whole_unit_width = reloaded.upper - reloaded.lower;
    let rows_width = 2.0 * temporal.row_interval_z * independent_rows_standard_error(&panel);
    assert!(
        whole_unit_width > rows_width,
        "whole-unit width {whole_unit_width} is not wider than independent-rows width {rows_width}"
    );
    assert!(reloaded.lower < reloaded.point && reloaded.point < reloaded.upper);
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
    let recovery = slice_cases::load().recovery;
    assert_eq!(POPULATION, recovery.population);
    assert_eq!(SNAPSHOT, recovery.rust_snapshot);
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
    let consumed =
        consume_observation_recovery_artifact(&bytes, RecoveryConsumeLimits::default(), &ctx())
            .unwrap();
    assert!(consumed.recovered().law().snapshot_identity().starts_with("recovered:"));
    assert_all_close(consumed.recovered().law().probabilities(), &live, EXACT);
    assert_all_close(consumed.recovered().law().probabilities(), &model.truth(), EXACT);
}

fn retained_operation(name: &str) -> RetainedOperation {
    match name {
        "row_reconstruction" => RetainedOperation::RowReconstruction,
        "refit_from_retained_rows" => RetainedOperation::RefitFromRetainedRows,
        "exact_row_eviction" => RetainedOperation::ExactRowEviction,
        "in_place_weight_or_mask_update" => RetainedOperation::InPlaceWeightOrMaskUpdate,
        other => panic!("{other} is not a retained operation named by the fixture"),
    }
}

/// The least-squares summary of the fixture line `y = 1 + 2t`, with design columns `1` and `t`.
fn line_summary(gaussian: &Gaussian) -> OlsRefresh {
    let rows = gaussian
        .treatment
        .iter()
        .zip(&gaussian.outcome)
        .enumerate()
        .map(|(id, (arm, response))| {
            RefreshRow::new(u64::try_from(id).unwrap(), vec![1.0, *arm], *response)
        })
        .collect();
    OlsRefresh::from_rows("line", 2, rows).unwrap()
}

/// Summary-only retention serves the same summary as full retention and refuses, with its
/// reason, every operation that needs the rows. There is no artifact format for the summary, so
/// the summary is read from the live fit.
#[test]
fn reduced_retention_serves_only_the_summary() {
    let cases = slice_cases::load();
    let Retention { reduced, unserved } = &cases.retention;
    let mode = RetentionMode::ALL
        .into_iter()
        .find(|mode| mode.as_str() == reduced)
        .unwrap_or_else(|| panic!("{reduced} is not a retention mode"));
    assert!(mode.serves(RetainedOperation::ReuseSummary));
    assert!(mode.require(RetainedOperation::ReuseSummary).is_ok());

    let full = line_summary(&cases.gaussian);
    let mut summary_only = line_summary(&cases.gaussian).into_summary_only();
    assert_eq!(full.retention(), RetentionMode::FullReusable);
    assert_eq!(summary_only.retention(), mode);
    assert!(!full.rows().is_empty());
    assert!(summary_only.rows().is_empty());

    // The served quantity is the same and equals the line.
    let served = summary_only.snapshot();
    assert_eq!(served, full.snapshot());
    let beta = served.beta.as_ref().expect("the line has a nonsingular design");
    assert_all_close(beta, &[cases.gaussian.intercept, cases.gaussian.effect], 1e-9);

    // What the rows would have served is refused, naming the mode and the operation.
    for name in unserved {
        let operation = retained_operation(name);
        assert!(mode.no_longer_serves().contains(&operation), "{name}");
        assert!(full.fresh_snapshot().is_ok());
        let reason = format!("{mode} cannot serve {operation}");
        let error = match operation {
            RetainedOperation::RefitFromRetainedRows => summary_only.fresh_snapshot().unwrap_err(),
            RetainedOperation::ExactRowEviction => summary_only.evict_row(0).unwrap_err(),
            RetainedOperation::InPlaceWeightOrMaskUpdate => {
                summary_only.set_mask(0, true).unwrap_err()
            }
            RetainedOperation::RowReconstruction => {
                assert!(mode.require(operation).is_err());
                continue;
            }
            other => panic!("{other} has no summary operation to refuse"),
        };
        assert!(error.to_string().contains(&reason), "{error}");
    }
    // A refused operation leaves the served summary as it was.
    assert_eq!(summary_only.snapshot(), served);
}
