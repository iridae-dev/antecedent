//! Temporal family lowering: every cell lowers completely, and the program reads
//! the study's target, estimator, inference and resampling.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeSet;
use std::sync::Arc;

use antecedent_contract::ClaimKind;
use antecedent_core::{
    CausalQuery, ContinuousDomain, DeclaredSamplingDesign, DerivativeScale, ExecutionContext,
    GraphFamily, GridSpec, InferenceSelection, Lag, MediationContrast, MediationQuery,
    NestedCounterfactualQuery, ResponseFunctional, ResponseQuery, SemanticDigest, TargetForm,
    TemporalEffectQuery, TemporalPolicy, TemporalResponseSpec, VariableId,
};
use antecedent_data::{MultiEnvironmentData, PanelData, PanelUnit, TableView, TimeSeriesData};
use antecedent_graph::{TemporalAdmg, TemporalCpdag, TemporalDag, TemporalPag, ensure_lagged};

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

/// `x(t-1) -> y(t)` as a temporal ADMG template.
fn xy_admg() -> TemporalAdmg {
    let mut graph = TemporalAdmg::empty();
    let x1 = graph.add_lagged(X, Lag::from_raw(1)).unwrap();
    let y0 = graph.add_lagged(XY_OUTCOME, Lag::CONTEMPORANEOUS).unwrap();
    graph.insert_directed(x1, y0).unwrap();
    graph
}

/// `x(t-1) -> m(t) -> y(t)` and `x(t-1) -> y(t)` as a temporal ADMG template.
fn xmy_admg() -> TemporalAdmg {
    let mut graph = TemporalAdmg::empty();
    let x1 = graph.add_lagged(X, Lag::from_raw(1)).unwrap();
    let m0 = graph.add_lagged(M, Lag::CONTEMPORANEOUS).unwrap();
    let y0 = graph.add_lagged(XMY_OUTCOME, Lag::CONTEMPORANEOUS).unwrap();
    graph.insert_directed(x1, m0).unwrap();
    graph.insert_directed(x1, y0).unwrap();
    graph.insert_directed(m0, y0).unwrap();
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

/// `x(t-1) -> m(t) -> y(t)` and `x(t-1) -> y(t)` as a temporal PAG template (every mark
/// definite).
fn xmy_pag() -> TemporalPag {
    let mut graph = TemporalPag::empty();
    let x1 = graph.add_lagged(X, Lag::from_raw(1)).unwrap();
    let m0 = graph.add_lagged(M, Lag::CONTEMPORANEOUS).unwrap();
    let y0 = graph.add_lagged(XMY_OUTCOME, Lag::CONTEMPORANEOUS).unwrap();
    graph.insert_directed(x1, m0).unwrap();
    graph.insert_directed(x1, y0).unwrap();
    graph.insert_directed(m0, y0).unwrap();
    graph
}

const PAG_C: VariableId = VariableId::from_raw(0);
const PAG_X: VariableId = VariableId::from_raw(1);
const PAG_W: VariableId = VariableId::from_raw(2);
const PAG_M: VariableId = VariableId::from_raw(3);
const PAG_Y: VariableId = VariableId::from_raw(4);

/// Five deterministic columns: `c`, `x`, `w`, `m` and `y`.
fn pag_mediation_series() -> TimeSeriesData {
    let column = |frequency: f64, phase: f64| -> Vec<f64> {
        (0..ROWS).map(|t| (frequency * t as f64 + phase).sin()).collect()
    };
    let (c, x, w, m, y) = (
        column(0.37, 0.0),
        column(0.53, 0.4),
        column(0.71, 1.1),
        column(0.29, 2.3),
        column(0.43, 0.9),
    );
    TimeSeriesData::from_f64_columns(
        [
            ("c", c.as_slice()),
            ("x", x.as_slice()),
            ("w", w.as_slice()),
            ("m", m.as_slice()),
            ("y", y.as_slice()),
        ],
        1,
    )
    .unwrap()
}

/// `c(t-1) -> x`, `x -> m`, `w -> m`, `m -> y`, `x -> y` within a period. A PAG leaves latent
/// confounding of two adjacent variables open, so the indirect effect identifies in every
/// model the PAG represents only when a visible cause separates the treatment from its
/// mediator (`c` into `x`, `w` into `m`).
fn pag_mediation_template() -> TemporalPag {
    let mut graph = TemporalPag::empty();
    let c1 = graph.add_lagged(PAG_C, Lag::from_raw(1)).unwrap();
    let x0 = graph.add_lagged(PAG_X, Lag::CONTEMPORANEOUS).unwrap();
    let w0 = graph.add_lagged(PAG_W, Lag::CONTEMPORANEOUS).unwrap();
    let m0 = graph.add_lagged(PAG_M, Lag::CONTEMPORANEOUS).unwrap();
    let y0 = graph.add_lagged(PAG_Y, Lag::CONTEMPORANEOUS).unwrap();
    for (from, to) in [(c1, x0), (x0, m0), (w0, m0), (m0, y0), (x0, y0)] {
        graph.insert_directed(from, to).unwrap();
    }
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

/// The local slope of `y` in `x` one step ahead.
fn derivative_query() -> CausalQuery {
    CausalQuery::Response(
        ResponseQuery::new(ResponseFunctional::PointDerivative {
            outcome: XY_OUTCOME,
            treatment: X,
            at: 0.0,
            order: 1,
            scale: DerivativeScale::Identity,
        })
        .with_temporal(
            TemporalResponseSpec::new(vec![1_u32], TemporalPolicy::pulse(-1), None).unwrap(),
        ),
    )
}

fn mediation_query() -> MediationQuery {
    MediationQuery::binary(X, XMY_OUTCOME, [M], MediationContrast::Mediated)
}

/// The pulse origin sits at the treated period and the outcome at `origin + horizon - 1`, so a
/// treatment whose only edge into the mediator is lagged by one reaches it at horizon two.
fn window_mediation_query() -> MediationQuery {
    mediation_query().with_horizons(vec![2]).unwrap()
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
            "TemporalEffectAdmg",
            Study::series(xy_series(0.0)).graph(xy_admg()).temporal_query(effect_query()),
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
        (
            "TemporalResponseAdmg",
            Study::series(xy_series(0.0)).graph(xy_admg()).query(derivative_query()),
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
            "TemporalMediationAdmg",
            Study::series(xmy_series())
                .graph(xmy_admg())
                .query(CausalQuery::Mediation(mediation_query())),
        ),
        (
            "TemporalMediationPag",
            Study::series(xmy_series())
                .graph(xmy_pag())
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
        .filter(|(_, family)| *family == "temporal")
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
fn every_temporal_cell_runs_the_shared_history_operators() {
    use antecedent_core::OperatorEffect;
    for (name, builder) in cells() {
        let study = builder.build().unwrap_or_else(|error| panic!("{name}: {error}"));
        let plan = compile(&study);
        let program = plan.lowered().unwrap();
        let operators = program.statistical().operators();
        let count =
            |effect| operators.iter().filter(|operator| operator.effect() == effect).count();
        // The linear route's read, fit and intervention, plus the shared history scan
        // (condition), the sequential regression (fit), the action timing and policy
        // evaluation (intervene), the transition (scan) and the trajectory reduction.
        assert_eq!(count(OperatorEffect::Read), 1, "{name}");
        assert_eq!(count(OperatorEffect::Condition), 1, "{name}: one history construction");
        assert_eq!(count(OperatorEffect::Fit), 2, "{name}: nuisance fit and sequential regression");
        assert_eq!(count(OperatorEffect::Intervene), 3, "{name}: world, timing and policy");
        assert_eq!(count(OperatorEffect::Scan), 1, "{name}: one transition scan");
        assert_eq!(count(OperatorEffect::ReuseFrozenFit), 1, "{name}: one trajectory reduction");
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
fn cross_world_identifies_once_at_compile_and_retains_the_probe() {
    let study =
        point(Study::series(xmy_series()).graph(xmy_dag()).query(nested_query())).build().unwrap();
    let plan = compile(&study);
    let program = plan.lowered().unwrap();
    assert!(program.identification_retained());
    let retained = program.retained_identification().unwrap();
    // The probe rides the program as an execution proof (identification and indexer), not as a
    // preparation cache: execution reads it back instead of identifying again.
    assert!(
        matches!(
            retained.execution_proof(),
            Some(crate::planner::lowered::RetainedExecutionProof::CrossWorld {
                identification,
                ..
            }) if !identification.estimands.is_empty()
        ),
        "the one-step probe is identified at compile"
    );
    assert!(retained.caches().is_empty());
    assert!(retained.static_seal().is_none());
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
fn window_mediation_identifies_once_at_compile_and_retains_the_components() {
    use crate::planner::lowered::RetainedExecutionProof;
    for (name, builder) in [
        (
            "TemporalMediationAdmg",
            Study::series(xmy_series())
                .graph(xmy_admg())
                .query(CausalQuery::Mediation(window_mediation_query())),
        ),
        (
            "TemporalMediationPag",
            Study::series(pag_mediation_series()).graph(pag_mediation_template()).query(
                CausalQuery::Mediation(MediationQuery::binary(
                    PAG_X,
                    PAG_Y,
                    [PAG_M],
                    MediationContrast::Mediated,
                )),
            ),
        ),
    ] {
        let study = point(builder).build().unwrap();
        let plan = compile(&study);
        let program = plan.lowered().expect("compile attaches a lowered program");
        assert!(program.identification_retained(), "{name}: the window identifies at compile");
        let retained = program.retained_identification().unwrap();
        // The three components (total, direct, indirect) ride the program.
        match (name, retained.execution_proof()) {
            ("TemporalMediationAdmg", Some(RetainedExecutionProof::Admg(components))) => {
                assert_eq!(components.len(), 3, "{name}");
                assert!(components.iter().flatten().count() >= 1, "{name}");
            }
            ("TemporalMediationPag", Some(RetainedExecutionProof::Pag { envelopes })) => {
                assert_eq!(envelopes.len(), 3, "{name}");
                assert!(envelopes.iter().flatten().count() >= 1, "{name}");
            }
            other => panic!("{name}: no window proof retained, got {other:?}"),
        }
        // Execution finds it for the study it was compiled for.
        assert!(
            crate::analysis::lowering::plan_retained_execution_proof(&study, &plan).is_some(),
            "{name}: the retention key matches the compiled study"
        );
        // A study that asks another contrast is another study and finds nothing.
        let other =
            point(Study::series(xmy_series()).graph(xmy_admg()).query(CausalQuery::Mediation(
                MediationQuery::binary(X, XMY_OUTCOME, [M], MediationContrast::NaturalDirect),
            )))
            .build()
            .unwrap();
        assert!(
            crate::analysis::lowering::plan_retained_execution_proof(&other, &plan).is_none(),
            "{name}: a retained proof is installed only into the study it was identified for"
        );
    }
}

#[test]
fn the_cross_world_run_reads_the_retained_probe_and_identifies_nothing_more() {
    use antecedent_core::execution_attempt::{Operation, observe_execution};
    let study =
        point(Study::series(xmy_series()).graph(xmy_dag()).query(nested_query())).build().unwrap();
    let plan = compile(&study);
    assert!(matches!(
        crate::analysis::lowering::plan_retained_execution_proof(&study, &plan),
        Some(crate::planner::lowered::RetainedExecutionProof::CrossWorld { .. })
    ));
    // A run is a compile and an execution: the execution adds no identification of its own.
    let compiled = observe_execution(|| study.compile(&ctx()));
    compiled.result.as_ref().expect("compile");
    let ran = observe_execution(|| study.run(&ctx()));
    ran.result.as_ref().expect("the cross-world run executes from the retained probe");
    assert_eq!(
        ran.report.counts(Operation::Identification).attempted,
        compiled.report.counts(Operation::Identification).attempted,
        "execution reads the probe compile identified"
    );
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
