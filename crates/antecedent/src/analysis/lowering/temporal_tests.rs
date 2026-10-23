//! Temporal family lowering: every cell lowers completely, and the program reads
//! the study's target, estimator, inference and resampling.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeSet;
use std::sync::Arc;

use antecedent_contract::ClaimKind;
use antecedent_core::{
    CausalQuery, ContinuousDomain, DeclaredSamplingDesign, ExecutionContext, GraphFamily, GridSpec,
    InferenceSelection, Lag, MediationContrast, MediationQuery, NestedCounterfactualQuery,
    ResponseFunctional, ResponseQuery, SemanticDigest, TargetForm, TemporalEffectQuery,
    TemporalPolicy, TemporalResponseSpec, VariableId,
};
use antecedent_data::{MultiEnvironmentData, PanelData, PanelUnit, TableView, TimeSeriesData};
use antecedent_graph::{TemporalCpdag, TemporalDag, ensure_lagged};

use super::super::registry::REGISTERED;
use crate::analysis::execute::CheckedTemporalMediationOperation;
use crate::planner::PhysicalExecutionPlan;
use crate::planner::lowered::{FunctionalKind, LoweringDepth, ProblemTarget, UncertaintyReading};
use crate::planner::statistical::{Realization, ResampleUnit, ResamplingMethod};
use crate::{BayesianConfig, EstimatorId, InferenceMode, RefuteSuite, Study, StudyBuilder};

const X: VariableId = VariableId::from_raw(0);
const M: VariableId = VariableId::from_raw(1);
const XY_OUTCOME: VariableId = VariableId::from_raw(1);
const XMY_OUTCOME: VariableId = VariableId::from_raw(2);
const ROWS: usize = 96;

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(41)
}

/// `y_t = 0.8 x_{t-1} + small periodic noise`.
fn xy_series(phase: f64) -> TimeSeriesData {
    let mut x = vec![0.0; ROWS];
    let mut y = vec![0.0; ROWS];
    let mut clock = 0.0_f64;
    for t in 0..ROWS {
        x[t] = (0.37 * clock + phase).sin();
        if t > 0 {
            y[t] = 0.8 * x[t - 1] + 0.1 * (1.3 * clock).cos();
        }
        clock += 1.0;
    }
    TimeSeriesData::from_f64_columns([("x", x.as_slice()), ("y", y.as_slice())], 1).unwrap()
}

/// `m_t = 0.5 x_{t-1} + shock`, `y_t = 1.5 x_{t-1} + 4 m_t`.
fn xmy_series() -> TimeSeriesData {
    let mut x = vec![0.0; ROWS];
    let mut m = vec![0.0; ROWS];
    let mut y = vec![0.0; ROWS];
    let mut clock = 0.0_f64;
    for t in 0..ROWS {
        x[t] = (0.37 * clock).sin();
        if t > 0 {
            let shock = if t % 2 == 0 { 0.3 } else { -0.3 };
            m[t] = 0.5 * x[t - 1] + shock;
            y[t] = 1.5 * x[t - 1] + 4.0 * m[t];
        }
        clock += 1.0;
    }
    TimeSeriesData::from_f64_columns(
        [("x", x.as_slice()), ("m", m.as_slice()), ("y", y.as_slice())],
        1,
    )
    .unwrap()
}

fn xy_dag() -> TemporalDag {
    let mut graph = TemporalDag::empty();
    let x1 = ensure_lagged(&mut graph, X, Lag::from_raw(1)).unwrap();
    let y0 = ensure_lagged(&mut graph, XY_OUTCOME, Lag::CONTEMPORANEOUS).unwrap();
    graph.insert_directed(x1, y0).unwrap();
    graph
}

fn xmy_dag() -> TemporalDag {
    let mut graph = TemporalDag::empty();
    let x1 = ensure_lagged(&mut graph, X, Lag::from_raw(1)).unwrap();
    let m0 = ensure_lagged(&mut graph, M, Lag::CONTEMPORANEOUS).unwrap();
    let y0 = ensure_lagged(&mut graph, XMY_OUTCOME, Lag::CONTEMPORANEOUS).unwrap();
    graph.insert_directed(x1, m0).unwrap();
    graph.insert_directed(x1, y0).unwrap();
    graph.insert_directed(m0, y0).unwrap();
    graph
}

fn panel() -> PanelData {
    PanelData::try_new(Arc::<[PanelUnit]>::from(vec![
        PanelUnit { unit_id: 0, series: xy_series(0.1) },
        PanelUnit { unit_id: 1, series: xy_series(0.4) },
        PanelUnit { unit_id: 2, series: xy_series(0.7) },
    ]))
    .unwrap()
}

fn environments() -> MultiEnvironmentData {
    MultiEnvironmentData::try_new(Arc::<[TimeSeriesData]>::from(vec![
        xy_series(0.2),
        xy_series(0.9),
    ]))
    .unwrap()
}

fn effect_query() -> TemporalEffectQuery {
    TemporalEffectQuery::pulse(X, XY_OUTCOME, 1.0)
        .with_policy(TemporalPolicy::pulse(-1))
        .with_horizon_steps(1)
}

fn response_query() -> CausalQuery {
    CausalQuery::Response(
        ResponseQuery::new(ResponseFunctional::MeanCurve {
            outcome: XY_OUTCOME,
            treatment: ContinuousDomain::new(X, GridSpec::Values(vec![0.0, 1.0].into())),
        })
        .with_temporal(
            TemporalResponseSpec::new(vec![1_u32], TemporalPolicy::pulse(-1), Some(1)).unwrap(),
        ),
    )
}

fn mediation_query() -> MediationQuery {
    MediationQuery::binary(X, XMY_OUTCOME, [M], MediationContrast::Mediated)
}

fn nested_query() -> CausalQuery {
    CausalQuery::NestedCounterfactual(NestedCounterfactualQuery::new(X, M, XMY_OUTCOME).unwrap())
}

fn point(builder: StudyBuilder) -> StudyBuilder {
    builder.refute(RefuteSuite::None).bootstrap_replicates(0)
}

/// One buildable study per lowered cell of this family.
fn cells() -> Vec<(&'static str, StudyBuilder)> {
    let class = || TemporalCpdag::from_temporal_dag(&xy_dag());
    let mediation_class = || TemporalCpdag::from_temporal_dag(&xmy_dag());
    vec![
        (
            "TemporalEffectDag",
            Study::series(xy_series(0.0)).graph(xy_dag()).temporal_query(effect_query()),
        ),
        (
            "TemporalEffectClass",
            Study::series(xy_series(0.0)).graph(class()).temporal_query(effect_query()),
        ),
        (
            "PanelTemporalEffectDag",
            Study::panel(panel()).graph(xy_dag()).temporal_query(effect_query()),
        ),
        (
            "PanelTemporalEffectClass",
            Study::panel(panel()).graph(class()).temporal_query(effect_query()),
        ),
        (
            "MultiEnvTemporalEffect",
            Study::series_multi(environments()).graph(xy_dag()).temporal_query(effect_query()),
        ),
        (
            "TemporalResponseDag",
            Study::series(xy_series(0.0)).graph(xy_dag()).query(response_query()),
        ),
        (
            "TemporalResponseClass",
            Study::series(xy_series(0.0)).graph(class()).query(response_query()),
        ),
        ("PanelTemporalResponseDag", Study::panel(panel()).graph(xy_dag()).query(response_query())),
        (
            "PanelTemporalResponseClass",
            Study::panel(panel()).graph(class()).query(response_query()),
        ),
        (
            "TemporalMediationDag",
            Study::series(xmy_series())
                .graph(xmy_dag())
                .query(CausalQuery::Mediation(mediation_query())),
        ),
        (
            "TemporalMediationCpdag",
            Study::series(xmy_series())
                .graph(mediation_class())
                .query(CausalQuery::Mediation(mediation_query())),
        ),
        (
            "TemporalCrossWorldDag",
            Study::series(xmy_series()).graph(xmy_dag()).query(nested_query()),
        ),
        (
            "TemporalCrossWorldClass",
            Study::series(xmy_series()).graph(mediation_class()).query(nested_query()),
        ),
    ]
    .into_iter()
    .map(|(name, builder)| (name, point(builder)))
    .collect()
}

fn compile(study: &Study) -> PhysicalExecutionPlan {
    study.compile(&ctx()).unwrap()
}

fn identity(builder: StudyBuilder) -> SemanticDigest {
    compile(&builder.build().unwrap()).lowered().unwrap().identity()
}

fn series_effect() -> StudyBuilder {
    point(Study::series(xy_series(0.0)).graph(xy_dag()).temporal_query(effect_query()))
}

#[test]
fn every_temporal_cell_lowers_completely() {
    let lowered: BTreeSet<&str> = cells().iter().map(|(name, _)| *name).collect();
    let registered: BTreeSet<&str> = REGISTERED
        .iter()
        .filter(|(cell, family)| {
            *family == "temporal" && *cell != super::super::RouteCell::TemporalMediationPag
        })
        .map(|(cell, _)| cell.as_str())
        .collect();
    assert_eq!(lowered, registered, "every temporal cell has a fixture here");
    for (name, builder) in cells() {
        let study = builder.build().unwrap_or_else(|error| panic!("{name}: {error}"));
        let plan = study.compile(&ctx()).unwrap_or_else(|error| panic!("{name}: {error}"));
        let program = plan.lowered().expect("compile attaches a lowered program");
        crate::planner::plan_view::checks::assert_view_agrees(name, &plan);
        assert_eq!(program.route(), name);
        assert_eq!(program.depth(), LoweringDepth::Complete, "{name}");
        assert!(program.claim().kind().is_some(), "{name} states its claim");
        assert_eq!(
            compile(&study).lowered().unwrap().identity(),
            program.identity(),
            "{name}: the identity is stable across compiles"
        );
    }
}

#[test]
fn series_effect_program_reads_the_study_configuration() {
    let study = series_effect().build().unwrap();
    let plan = compile(&study);
    let program = plan.lowered().unwrap();
    assert!(program.identification_retained());
    let ProblemTarget::Target(target) = program.scientific().target() else {
        panic!("a pulse contrast lowers to its world-indexed target");
    };
    let contrast = super::target_contrast(target).expect("a temporal contrast");
    let frame = super::contrast_frame(contrast).unwrap();
    assert_eq!(frame.horizon_steps(), Some(1));
    let history = frame.history().unwrap();
    assert!(history.is_planner_default(), "no declared cap: the planner chose");
    assert!(history.resolved_cap().is_some(), "the planner's choice is pinned");
    assert_eq!(super::contrast_outcome(contrast).unwrap().outcome(), XY_OUTCOME);
    assert_eq!(target.assumptions().graph(), GraphFamily::TemporalDag);

    let statistical = program.statistical();
    let recorded =
        plan.logical.record.estimator.as_deref().and_then(|name| name.parse::<EstimatorId>().ok());
    assert_eq!(statistical.estimator(), recorded);
    assert!(statistical.estimator().is_some());
    assert!(matches!(statistical.realization(), Realization::Frequentist(_)));
    assert!(statistical.resampling().is_none());
    assert!(statistical.folds().is_none());
    assert_eq!(statistical.models()[0].output(), Some(XY_OUTCOME));
    assert!(statistical.models()[0].inputs().contains(&X));
    assert_eq!(program.claim().kind(), Some(ClaimKind::Point));
    assert_eq!(program.claim().uncertainty(), UncertaintyReading::Sampling);

    let base = program.identity();
    assert_eq!(identity(series_effect()), base, "stable across compiles");

    let bootstrapped = series_effect().bootstrap_replicates(50).build().unwrap();
    let bootstrapped_plan = compile(&bootstrapped);
    let resampling = bootstrapped_plan.lowered().unwrap().statistical().resampling().unwrap();
    assert_eq!(resampling.method(), ResamplingMethod::BlockBootstrap);
    assert_eq!(resampling.unit(), ResampleUnit::CalendarBlock);
    assert_eq!(resampling.replicates(), 50);
    assert_ne!(bootstrapped_plan.lowered().unwrap().identity(), base);

    let declared = point(
        Study::series(xy_series(0.0))
            .graph(xy_dag())
            .temporal_query(effect_query().with_max_history_lag(Some(1))),
    );
    let declared_study = declared.build().unwrap();
    let declared_plan = compile(&declared_study);
    let ProblemTarget::Target(declared_target) =
        declared_plan.lowered().unwrap().scientific().target()
    else {
        panic!("a declared cap still lowers the contrast");
    };
    let declared_history = super::target_contrast(declared_target)
        .and_then(super::contrast_frame)
        .and_then(antecedent_core::TimeFrame::history)
        .unwrap();
    assert_eq!(declared_history.max_lag_steps(), Some(1));
    assert_ne!(declared_plan.lowered().unwrap().identity(), base);

    let bayesian = series_effect()
        .inference(InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(64)))
        .build()
        .unwrap();
    let bayesian_plan = compile(&bayesian);
    let bayesian_program = bayesian_plan.lowered().unwrap();
    assert!(matches!(bayesian_program.statistical().realization(), Realization::Bayesian(_)));
    assert_eq!(
        bayesian_program
            .statistical()
            .posterior()
            .map(crate::planner::statistical::PosteriorPlan::draws),
        Some(64)
    );
    assert_eq!(bayesian_program.claim().uncertainty(), UncertaintyReading::Posterior);
    assert_ne!(bayesian_program.identity(), base);
}

#[test]
fn prepare_and_estimate_stamp_the_compiled_identity() {
    let study = series_effect().build().unwrap();
    let compiled = compile(&study).lowered().unwrap().identity();
    let prepared = study.prepare(&ctx()).unwrap();
    assert_eq!(prepared.plan().lowered_program_identity(), Some(compiled));
    let estimated = prepared.estimate_retained(&ctx()).unwrap();
    assert_eq!(estimated.lowered_program_identity(), Some(compiled));
}

#[test]
fn pooled_resampling_draws_the_unit_the_design_declares() {
    let unit_of = |builder: StudyBuilder| {
        let plan = compile(&builder.bootstrap_replicates(20).build().unwrap());
        let program = plan.lowered().unwrap();
        (program.statistical().resampling().unwrap().unit(), program.identity())
    };
    let panel_builder =
        || point(Study::panel(panel()).graph(xy_dag()).temporal_query(effect_query()));
    let (panel_unit, panel_identity) = unit_of(panel_builder());
    assert_eq!(panel_unit, ResampleUnit::Subject, "a panel resamples whole units");
    let (declared_unit, declared_identity) =
        unit_of(panel_builder().sampling_design(DeclaredSamplingDesign::ClusteredPanel));
    assert_eq!(declared_unit, ResampleUnit::Cluster, "the declared design wins");
    assert_ne!(panel_identity, declared_identity);
    let (environment_unit, _) = unit_of(point(
        Study::series_multi(environments()).graph(xy_dag()).temporal_query(effect_query()),
    ));
    assert_eq!(environment_unit, ResampleUnit::Cluster, "environments are the clusters");
}

#[test]
fn response_lowers_to_its_dose_time_surface() {
    let study = point(Study::series(xy_series(0.0)).graph(xy_dag()).query(response_query()))
        .build()
        .unwrap();
    let plan = compile(&study);
    let program = plan.lowered().unwrap();
    let ProblemTarget::Target(target) = program.scientific().target() else {
        panic!("a mean curve lowers to its dose-time surface");
    };
    let TargetForm::DoseTime(surface) = target.form() else {
        panic!("a mean curve is a dose-time surface");
    };
    assert_eq!(surface.dose(), X);
    assert_eq!(surface.horizons(), &[1]);
    assert_eq!(surface.level_bits().len(), 2);
    assert_eq!(program.claim().kind(), Some(ClaimKind::Response));
}

#[test]
fn mediation_lowers_to_the_theorem_target_the_checked_operation_freezes() {
    let query = mediation_query();
    let study = point(
        Study::series(xmy_series()).graph(xmy_dag()).query(CausalQuery::Mediation(query.clone())),
    )
    .build()
    .unwrap();
    let plan = compile(&study);
    let program = plan.lowered().unwrap();
    let ProblemTarget::Target(target) = program.scientific().target() else {
        panic!("one-horizon mediation lowers to its theorem target");
    };
    assert!(matches!(target.form(), TargetForm::Mediation(_)));
    let cache = crate::analysis::prepared::identify_temporal_mediation_horizons(
        &xmy_dag(),
        &query,
        EstimatorId::TemporalMediation,
    )
    .unwrap();
    let frozen = CheckedTemporalMediationOperation::theorem_targets(
        &query,
        xmy_series().schema(),
        cache.by_horizon.iter().map(|entry| (entry.horizon, &entry.identification, &entry.indexer)),
        InferenceSelection::Frequentist,
    )
    .unwrap();
    assert_eq!(frozen[0].identity(), target.identity());
    let roles: Vec<_> =
        program.statistical().models().iter().map(|model| model.role().as_str()).collect();
    assert_eq!(roles, ["outcome", "mediator"]);
}

#[test]
fn multi_horizon_mediation_states_one_target_per_horizon() {
    let query = mediation_query().with_horizons(vec![1, 2]).unwrap();
    let study = point(
        Study::series(xmy_series()).graph(xmy_dag()).query(CausalQuery::Mediation(query.clone())),
    )
    .build()
    .unwrap();
    let plan = compile(&study);
    let program = plan.lowered().unwrap();
    let ProblemTarget::Targets(targets) = program.scientific().target() else {
        panic!("a two-horizon mediation lowers to one theorem target per horizon");
    };
    let cache = crate::analysis::prepared::identify_temporal_mediation_horizons(
        &xmy_dag(),
        &query,
        EstimatorId::TemporalMediation,
    )
    .unwrap();
    let frozen = CheckedTemporalMediationOperation::theorem_targets(
        &query,
        xmy_series().schema(),
        cache.by_horizon.iter().map(|entry| (entry.horizon, &entry.identification, &entry.indexer)),
        InferenceSelection::Frequentist,
    )
    .unwrap();
    let lowered: Vec<_> = targets.iter().map(antecedent_core::ScientificTarget::identity).collect();
    let expected: Vec<_> = frozen.iter().map(antecedent_core::ScientificTarget::identity).collect();
    assert_eq!(lowered, expected);
    assert_eq!(lowered.len(), 2);
    assert_ne!(lowered[0], lowered[1], "each horizon is its own target");
    let single = point(
        Study::series(xmy_series())
            .graph(xmy_dag())
            .query(CausalQuery::Mediation(mediation_query())),
    )
    .build()
    .unwrap();
    assert_ne!(
        compile(&single).lowered().unwrap().identity(),
        program.identity(),
        "a horizon family is not its first horizon"
    );
}

#[test]
fn cross_world_identifies_on_each_execution() {
    let study =
        point(Study::series(xmy_series()).graph(xmy_dag()).query(nested_query())).build().unwrap();
    let plan = compile(&study);
    let program = plan.lowered().unwrap();
    assert!(!program.identification_retained());
    assert!(matches!(
        program.scientific().target(),
        ProblemTarget::QueryFunctional {
            kind: FunctionalKind::PathSpecific,
            outcome: Some(outcome),
            ..
        }
            if *outcome == XMY_OUTCOME
    ));
}

#[test]
fn class_effect_is_stated_under_the_class_and_reads_its_envelope() {
    let study = point(
        Study::series(xy_series(0.0))
            .graph(TemporalCpdag::from_temporal_dag(&xy_dag()))
            .temporal_query(effect_query()),
    )
    .build()
    .unwrap();
    let plan = compile(&study);
    let program = plan.lowered().unwrap();
    let ProblemTarget::Target(target) = program.scientific().target() else {
        panic!("a class pulse contrast lowers to its world-indexed target");
    };
    assert_eq!(target.assumptions().graph(), GraphFamily::TemporalCpdag);
    let envelope = matches!(program.statistical().realization(), Realization::Envelope(_));
    let bounds = program.claim().kind() == Some(ClaimKind::Bounds);
    assert_eq!(envelope, bounds, "an identified set is read as an envelope");
}
