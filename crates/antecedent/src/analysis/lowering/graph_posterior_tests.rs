//! Graph-posterior lowering: every cell lowers a complete mixture program whose
//! layers read the study's configuration, and preparation and estimation carry
//! the identity compile stamped.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeSet;
use std::sync::Arc;

use antecedent_core::{
    AverageEffectQuery, CausalQuery, ConditionalEffectQuery, ExecutionContext, Intervention,
    InterventionalDistributionQuery, MediationContrast, MediationQuery, ResponseFunctional,
    ResponseQuery, SemanticDigest, TargetForm, TemporalEffectQuery, TemporalPolicy,
    TemporalResponseSpec, Value, VariableId,
};
use antecedent_data::{TabularData, TimeSeriesData};
use antecedent_discovery::{GraphPosterior, GraphPosteriorAtomKind, SamplingSummary, set_edge};
use antecedent_prob::InferenceDiagnostics;

use crate::analysis::lowering::{GraphPosteriorCell, RouteCell};
use crate::planner::lowered::{
    Composition, LoweredProgram, LoweringDepth, ProblemTarget, UncertaintyReading,
};
use crate::planner::statistical::{
    PosteriorAlgorithm, Realization, ResampleUnit, ResamplingMethod,
};
use crate::{BayesianConfig, EstimatorId, IdentifierId, InferenceMode, RefuteSuite, Study};

const DRAWS: usize = 16;

/// Graph-posterior estimation recurses deeply; run each body on a large stack.
fn on_large_stack(body: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new().stack_size(64 << 20).spawn(body).unwrap().join().unwrap();
}

fn summary(weights: &[f64], label: &str) -> SamplingSummary {
    SamplingSummary {
        ess: 1.0 / weights.iter().map(|weight| weight * weight).sum::<f64>(),
        diagnostics: InferenceDiagnostics::analytic(label),
        rejected_invalid: 0,
    }
}

fn bayesian() -> InferenceMode {
    InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(DRAWS).prior_scale(100.0))
}

// --- Tabular fixtures: `y = 2t + 2z + eps`, `z` confounds `t`. ----------------

fn tabular_data() -> TabularData {
    let mut treatment = Vec::new();
    let mut outcome = Vec::new();
    let mut confounder = Vec::new();
    for _ in 0..20 {
        for (z, t, count) in [(0.0, 0.0, 6), (0.0, 1.0, 2), (1.0, 0.0, 2), (1.0, 1.0, 6)] {
            for row in 0..count {
                let epsilon = if row % 2 == 0 { -0.2 } else { 0.2 };
                treatment.push(t);
                confounder.push(z);
                outcome.push(2.0 * t + 2.0 * z + epsilon);
            }
        }
    }
    TabularData::from_f64_columns([
        ("t", treatment.as_slice()),
        ("y", outcome.as_slice()),
        ("z", confounder.as_slice()),
    ])
    .unwrap()
}

/// DAG atoms over `[t, y, z]`: `t -> y`, the `z`-confounded graph, and, when
/// `reverse`, an unidentified `y -> t` atom carrying weight 0.2.
fn tabular_posterior(reverse: bool) -> GraphPosterior {
    let direct = set_edge(0, 3, 0, 1, true);
    let adjusted = set_edge(set_edge(set_edge(0, 3, 0, 1, true), 3, 2, 0, true), 3, 2, 1, true);
    let (weights, atoms) = if reverse {
        (vec![0.5, 0.3, 0.2], vec![direct, adjusted, set_edge(0, 3, 1, 0, true)])
    } else {
        (vec![0.6, 0.4], vec![direct, adjusted])
    };
    let mut marginals = vec![0.0; 9];
    for (weight, atom) in weights.iter().zip(&atoms) {
        for from in 0..3 {
            for to in 0..3 {
                if from != to && set_edge(*atom, 3, from, to, true) == *atom {
                    marginals[from * 3 + to] += weight;
                }
            }
        }
    }
    let sampling = summary(&weights, "graph_posterior_lowering");
    GraphPosterior::new(3, weights, atoms, marginals.clone(), marginals, sampling)
        .unwrap()
        .with_algorithm("graph_posterior_lowering")
}

fn average_effect() -> AverageEffectQuery {
    AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1))
}

fn tabular_study(
    query: CausalQuery,
    posterior: GraphPosterior,
    inference: InferenceMode,
    bootstrap: u32,
) -> Study {
    Study::tabular(tabular_data())
        .graph_posterior(posterior)
        .query(query)
        .inference(inference)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(bootstrap)
        .build()
        .unwrap()
}

// --- Series fixtures. ------------------------------------------------------

fn lag_bit(n: usize, from: usize, to: usize) -> u64 {
    1u64 << (from * n + to)
}

fn uniform(state: &mut u64) -> f64 {
    *state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
    ((*state >> 33) as f64 / (1u64 << 31) as f64) * 2.0 - 1.0
}

/// `defect_t = 0.9 pressure_{t-1}` on uniform white-noise pressure.
fn pressure_series() -> TimeSeriesData {
    let n = 200;
    let mut state = 17_u64;
    let mut pressure = vec![0.0; n];
    let mut defect = vec![0.0; n];
    for t in 0..n {
        pressure[t] = uniform(&mut state);
        if t > 0 {
            defect[t] = 0.9 * pressure[t - 1];
        }
    }
    TimeSeriesData::from_f64_columns(
        [("pressure", pressure.as_slice()), ("defect", defect.as_slice())],
        1,
    )
    .unwrap()
}

/// One DBN atom: `pressure@1 -> defect`.
fn pressure_posterior() -> GraphPosterior {
    GraphPosterior::from_atoms(
        2,
        vec![1.0],
        vec![0_u64],
        vec![0.0; 4],
        vec![0.0; 4],
        summary(&[1.0], "graph_posterior_lowering"),
    )
    .unwrap()
    .with_lagged_marginals(1, vec![0.0; 4])
    .unwrap()
    .with_lag_masks(vec![lag_bit(2, 0, 1)])
    .unwrap()
}

/// `m_s = 0.8 t_{s-1} + e`, `y_s = 0.55 m_s + 0.3 t_{s-1} + e`.
fn mediation_series() -> TimeSeriesData {
    let n = 240;
    let mut state = 29_u64;
    let mut t = vec![0.0; n];
    let mut m = vec![0.0; n];
    let mut y = vec![0.0; n];
    for s in 0..n {
        t[s] = uniform(&mut state);
        if s > 0 {
            m[s] = 0.8 * t[s - 1] + 0.4 * uniform(&mut state);
            y[s] = 0.55 * m[s] + 0.3 * t[s - 1] + 0.4 * uniform(&mut state);
        }
    }
    TimeSeriesData::from_f64_columns(
        [("t", t.as_slice()), ("m", m.as_slice()), ("y", y.as_slice())],
        1,
    )
    .unwrap()
}

/// One DBN atom over `[t, m, y]`: `m -> y`, `t@1 -> m`, `t@1 -> y`.
fn mediation_posterior() -> GraphPosterior {
    let n = 3;
    GraphPosterior::from_atoms(
        n,
        vec![1.0],
        vec![set_edge(0, n, 1, 2, true)],
        vec![0.0; n * n],
        vec![0.0; n * n],
        summary(&[1.0], "graph_posterior_lowering"),
    )
    .unwrap()
    .with_atom_kind(GraphPosteriorAtomKind::Dag)
    .with_lagged_marginals(1, vec![0.0; n * n])
    .unwrap()
    .with_lag_masks(vec![lag_bit(n, 0, 1) | lag_bit(n, 0, 2)])
    .unwrap()
}

/// `y = 1.5 x_{t-1} + 4 m`, `m = 0.5 x_{t-1} + shock`.
fn cross_series() -> TimeSeriesData {
    let n = 200;
    let mut x = vec![0.0; n];
    let mut m = vec![0.0; n];
    let mut y = vec![0.0; n];
    for t in 0..n {
        x[t] = ((t as f64) * 0.37).sin();
        if t > 0 {
            let shock = if t % 2 == 0 { 0.3 } else { -0.3 };
            m[t] = 0.5 * x[t - 1] + shock;
            y[t] = 1.5 * x[t - 1] + 4.0 * m[t];
        }
    }
    TimeSeriesData::from_f64_columns(
        [("x", x.as_slice()), ("m", m.as_slice()), ("y", y.as_slice())],
        1,
    )
    .unwrap()
}

/// One temporal CPDAG atom over `[x, m, y]`: `m - y`, `x@1 -> m`, `x@1 -> y`.
fn cross_posterior() -> GraphPosterior {
    let n = 3;
    GraphPosterior::from_atoms(
        n,
        vec![1.0],
        vec![set_edge(0, n, 1, 2, true)],
        vec![0.0; n * n],
        vec![0.0; n * n],
        summary(&[1.0], "graph_posterior_lowering"),
    )
    .unwrap()
    .with_atom_kind(GraphPosteriorAtomKind::Cpdag)
    .with_lagged_marginals(1, vec![0.0; n * n])
    .unwrap()
    .with_lag_masks(vec![lag_bit(n, 0, 1) | lag_bit(n, 0, 2)])
    .unwrap()
}

fn series_study(
    data: TimeSeriesData,
    posterior: GraphPosterior,
    query: CausalQuery,
    bootstrap: u32,
) -> Study {
    Study::series(data)
        .graph_posterior(posterior)
        .query(query)
        .inference(InferenceMode::Frequentist)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(bootstrap)
        .build()
        .unwrap()
}

/// Twelve rows over `[t, y, z, trial, sel, prop]`: eight trial rows, four target rows, and
/// constant supplied selection and treatment probabilities.
fn transport_data() -> TabularData {
    let t = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let y = [3.1, 1.0, 2.9, 1.2, 3.3, 0.8, 3.0, 1.1, 0.0, 1.0, 0.5, 0.7];
    let z = [1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0];
    let trial = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0];
    TabularData::from_f64_columns([
        ("t", &t[..]),
        ("y", &y[..]),
        ("z", &z[..]),
        ("trial", &trial[..]),
        ("sel", &[0.6; 12][..]),
        ("prop", &[0.5; 12][..]),
    ])
    .unwrap()
}

/// DAG atoms over the six transport columns with the covariate `z` as the selection target:
/// `t -> y` (no population-varying ancestor of `y`), `t -> y <- z` (standardized over `z`), and
/// a cyclic `t <-> y` atom of weight 0.2 that is not a DAG.
fn transport_posterior() -> GraphPosterior {
    let direct = set_edge(0, 6, 0, 1, true);
    let standardized = set_edge(set_edge(0, 6, 0, 1, true), 6, 2, 1, true);
    let cyclic = set_edge(set_edge(0, 6, 0, 1, true), 6, 1, 0, true);
    let weights = vec![0.5, 0.3, 0.2];
    let sampling = summary(&weights, "graph_posterior_transport_lowering");
    GraphPosterior::new(
        6,
        weights,
        vec![direct, standardized, cyclic],
        vec![0.0; 36],
        vec![0.0; 36],
        sampling,
    )
    .unwrap()
    .with_algorithm("graph_posterior_transport_lowering")
}

fn transport_study(inference: InferenceMode) -> Study {
    let response = ResponseQuery::new(ResponseFunctional::MeanCurve {
        outcome: VariableId::from_raw(1),
        treatment: antecedent_core::ContinuousDomain::new(
            VariableId::from_raw(0),
            antecedent_core::GridSpec::Values(Arc::from([0.0, 1.0])),
        ),
    });
    let query =
        antecedent_core::TransportQuery::new(response, "trial", "target", [VariableId::from_raw(0)]);
    Study::tabular(transport_data())
        .graph_posterior(transport_posterior())
        .query(CausalQuery::Transport(query))
        .selection_targets(Arc::from([VariableId::from_raw(2)]))
        .transport_trial(crate::TransportTrialSpec {
            trial: VariableId::from_raw(3),
            selection_probability: VariableId::from_raw(4),
            treatment_probability: VariableId::from_raw(5),
        })
        .inference(inference)
        .refute(RefuteSuite::None)
        .build()
        .unwrap()
}

/// Eight units in four disjoint pairs under the cluster total-effect design, two covariates.
fn interference_study(inference: InferenceMode) -> Study {
    let data = TabularData::from_f64_columns([
        ("y", &[4.0, 4.0, 6.0, 6.0, 1.0, 1.0, 3.0, 3.0][..]),
        ("z", &[0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0][..]),
    ])
    .unwrap();
    let edges = (0..4_u64)
        .flat_map(|cluster| {
            let first = cluster * 2;
            [
                antecedent_data::NetworkEdge::new(first, first + 1, 1.0),
                antecedent_data::NetworkEdge::new(first + 1, first, 1.0),
            ]
        })
        .collect::<Vec<_>>();
    let network = antecedent_data::NetworkData::try_new(data.clone(), edges).unwrap();
    let query = antecedent_core::InterferenceQuery::new(
        antecedent_core::AssignmentDesign::ClusterRandomization {
            clusters: Arc::from([0, 0, 1, 1, 2, 2, 3, 3]),
            treated_clusters: 2,
        },
        antecedent_core::ExposureMapping::NeighborFraction,
        antecedent_core::InterferenceFunctional::ExposureContrast {
            outcome: VariableId::from_raw(0),
            from: antecedent_core::ExposureLevel { own: 0.0, neighbors: 0.0 },
            to: antecedent_core::ExposureLevel { own: 1.0, neighbors: 1.0 },
        },
    );
    let weights = vec![0.5, 0.3, 0.2];
    let sampling = summary(&weights, "graph_posterior_interference_lowering");
    let posterior = GraphPosterior::new(
        2,
        weights,
        vec![
            set_edge(0, 2, 0, 1, true),
            0,
            set_edge(set_edge(0, 2, 0, 1, true), 2, 1, 0, true),
        ],
        vec![0.0; 4],
        vec![0.0; 4],
        sampling,
    )
    .unwrap()
    .with_algorithm("graph_posterior_interference_lowering");
    Study::tabular(data)
        .graph_posterior(posterior)
        .query(CausalQuery::Interference(query))
        .interference(crate::InterferenceSpec {
            network,
            assignment: Arc::from([true, true, true, true, false, false, false, false]),
        })
        .inference(inference)
        .refute(RefuteSuite::None)
        .build()
        .unwrap()
}

fn pulse() -> TemporalEffectQuery {
    TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0)
        .with_policy(TemporalPolicy::pulse(-1))
        .with_horizon_steps(1)
        .with_max_history_lag(Some(1))
}

/// One small study per graph-posterior cell.
fn representatives() -> Vec<(GraphPosteriorCell, Study)> {
    let t = VariableId::from_raw(0);
    let y = VariableId::from_raw(1);
    let z = VariableId::from_raw(2);
    let conditional =
        ConditionalEffectQuery::try_new(average_effect().with_effect_modifiers([z])).unwrap();
    let response = ResponseQuery::new(ResponseFunctional::InterventionResponse {
        outcome: y,
        interventions: Arc::from([Intervention::set(t, Value::f64(1.0))]),
    });
    let temporal_response = ResponseQuery::new(ResponseFunctional::InterventionResponse {
        outcome: y,
        interventions: Arc::from([Intervention::set(t, Value::f64(1.0))]),
    })
    .with_temporal(TemporalResponseSpec::new([1], TemporalPolicy::pulse(-1), Some(1)).unwrap());
    let mediation = MediationQuery::binary(
        VariableId::from_raw(0),
        VariableId::from_raw(2),
        [VariableId::from_raw(1)],
        MediationContrast::Mediated,
    )
    .with_horizons(vec![1])
    .unwrap();
    let distribution = InterventionalDistributionQuery::new(
        VariableId::from_raw(2),
        [Intervention::set(VariableId::from_raw(0), Value::f64(1.0))],
    );
    vec![
        (
            GraphPosteriorCell::AverageEffect,
            tabular_study(
                CausalQuery::AverageEffect(average_effect()),
                tabular_posterior(true),
                InferenceMode::Frequentist,
                0,
            ),
        ),
        (
            GraphPosteriorCell::ConditionalEffect,
            tabular_study(
                CausalQuery::ConditionalEffect(conditional),
                tabular_posterior(false),
                InferenceMode::Frequentist,
                0,
            ),
        ),
        (
            GraphPosteriorCell::StaticResponse,
            tabular_study(
                CausalQuery::Response(response),
                tabular_posterior(false),
                InferenceMode::Frequentist,
                0,
            ),
        ),
        (
            GraphPosteriorCell::TemporalEffect,
            series_study(
                pressure_series(),
                pressure_posterior(),
                CausalQuery::TemporalEffect(pulse()),
                8,
            ),
        ),
        (
            GraphPosteriorCell::TemporalResponse,
            series_study(
                pressure_series(),
                pressure_posterior(),
                CausalQuery::Response(temporal_response),
                8,
            ),
        ),
        (
            GraphPosteriorCell::TemporalMediation,
            series_study(
                mediation_series(),
                mediation_posterior(),
                CausalQuery::Mediation(mediation),
                8,
            ),
        ),
        (
            GraphPosteriorCell::CrossWorld,
            series_study(
                cross_series(),
                cross_posterior(),
                CausalQuery::Distribution(distribution),
                0,
            ),
        ),
        (GraphPosteriorCell::Transport, transport_study(InferenceMode::Frequentist)),
        (GraphPosteriorCell::Interference, interference_study(InferenceMode::Frequentist)),
    ]
}

fn compiled(study: &Study, ctx: &ExecutionContext) -> LoweredProgram {
    study.compile(ctx).unwrap().lowered().expect("compile attaches a lowered program").clone()
}

fn identity(study: &Study, ctx: &ExecutionContext) -> SemanticDigest {
    compiled(study, ctx).identity()
}

#[test]
fn every_graph_posterior_cell_lowers_a_complete_program_that_execution_stamps() {
    on_large_stack(|| {
        let ctx = ExecutionContext::for_tests(31);
        let mut covered = BTreeSet::new();
        for (cell, study) in representatives() {
            let name = RouteCell::GraphPosterior(cell).as_str();
            let plan = study.compile(&ctx).unwrap();
            let lowered = plan.lowered().expect("compile attaches a lowered program");
            let view = crate::planner::plan_view::checks::assert_view_agrees(name, &plan);
            assert!(view.mixture().is_some(), "{name}: a graph-posterior view states its mixture");
            let with_study = study.plan_view(&ctx).unwrap();
            assert_eq!(with_study.identity(), view.identity(), "{name}: one program");
            let mixture = with_study.mixture().expect("a graph-posterior view states its mixture");
            let posterior = study.graph_posterior.as_ref().expect("a graph-posterior study");
            assert_eq!(mixture.atom_kind(), Some(posterior.atom_kind.as_str()), "{name}");
            assert!(!mixture.atoms().is_empty(), "{name}: the mixture names its atoms");
            if mixture.identification_retained() {
                let identified = mixture.identified_mass().unwrap();
                let unidentified = mixture.unidentified_mass().unwrap();
                assert!(
                    (identified + unidentified - mixture.total_weight()).abs() < 1e-9,
                    "{name}: the mixture keeps every atom's mass"
                );
            } else {
                assert!(
                    (mixture.total_weight() - posterior.weights.iter().sum::<f64>()).abs() < 1e-9,
                    "{name}: the mixture carries the posterior's whole weight"
                );
            }
            assert_eq!(lowered.route(), name);
            assert_eq!(lowered.depth(), LoweringDepth::Complete, "{name} lowers completely");
            match (cell, lowered.scientific().target()) {
                (
                    GraphPosteriorCell::CrossWorld | GraphPosteriorCell::Interference,
                    ProblemTarget::QueryFunctional { .. },
                )
                | (_, ProblemTarget::Target(_) | ProblemTarget::Targets(_)) => {}
                (_, other) => panic!("{name} lowers no world-indexed target: {other:?}"),
            }
            assert_eq!(
                lowered.identification_retained(),
                cell != GraphPosteriorCell::CrossWorld,
                "{name}: only the cross-world probe identifies per execution"
            );
            let record = &plan.logical.record;
            assert_eq!(
                lowered.statistical().estimator(),
                record.estimator.as_deref().and_then(|id| id.parse::<EstimatorId>().ok()),
                "{name}: the program names the recorded estimator"
            );
            let mixture =
                lowered.statistical().posterior().expect("an atom-mixture posterior plan");
            assert_eq!(mixture.algorithm(), PosteriorAlgorithm::AtomMixture, "{name}");
            assert!(
                matches!(lowered.evidence().composition(), Composition::AtomMixture { atoms }
                    if atoms == study.graph_posterior.as_ref().unwrap().n_graphs),
                "{name}: layer four mixes every atom"
            );
            if matches!(
                cell,
                GraphPosteriorCell::TemporalEffect
                    | GraphPosteriorCell::TemporalResponse
                    | GraphPosteriorCell::TemporalMediation
            ) {
                let resampling = lowered.statistical().resampling().expect("bootstrap requested");
                assert_eq!(resampling.method(), ResamplingMethod::BlockBootstrap, "{name}");
                assert_eq!(resampling.unit(), ResampleUnit::CalendarBlock, "{name}");
                assert_eq!(resampling.replicates(), 8, "{name}");
            }
            let stamped = lowered.identity();
            assert_eq!(identity(&study, &ctx), stamped, "{name}: identity is stable");
            let prepared = study.prepare(&ctx).unwrap();
            assert_eq!(prepared.plan().lowered_program_identity(), Some(stamped), "{name}");
            let estimated = prepared.estimate_retained(&ctx).unwrap();
            assert_eq!(estimated.lowered_program_identity(), Some(stamped), "{name}");
            covered.insert(name);
        }
        let all: BTreeSet<_> = GraphPosteriorCell::ALL
            .iter()
            .map(|cell| RouteCell::GraphPosterior(*cell).as_str())
            .collect();
        assert_eq!(covered, all, "every graph-posterior cell has a complete lowering");
    });
}

#[test]
fn average_effect_program_reads_the_study_configuration() {
    on_large_stack(|| {
        let ctx = ExecutionContext::for_tests(37);
        let query = CausalQuery::AverageEffect(average_effect());

        // Unidentified posterior mass: the answer is an identified set over atoms.
        let partial =
            tabular_study(query.clone(), tabular_posterior(true), InferenceMode::Frequentist, 0);
        let lowered = compiled(&partial, &ctx);
        assert!(matches!(lowered.statistical().realization(), Realization::Envelope(_)));
        assert_eq!(lowered.claim().uncertainty(), UncertaintyReading::Envelope);
        assert_eq!(lowered.statistical().posterior().unwrap().draws(), 3);
        let ProblemTarget::Target(target) = lowered.scientific().target() else {
            panic!("an average effect lowers its contrast");
        };
        assert!(matches!(target.form(), TargetForm::Contrast(_)));

        // Every atom identified: a frequentist score per atom, no resampling.
        let frequentist =
            tabular_study(query.clone(), tabular_posterior(false), InferenceMode::Frequentist, 0);
        let lowered = compiled(&frequentist, &ctx);
        let statistical = lowered.statistical();
        assert!(matches!(statistical.realization(), Realization::Frequentist(_)));
        assert_eq!(statistical.identifier(), Some(IdentifierId::BackdoorAdjustment));
        assert_eq!(statistical.estimator(), Some(EstimatorId::LinearAdjustmentAte));
        assert!(statistical.resampling().is_none());
        assert_eq!(statistical.posterior().unwrap().draws(), 2);
        assert_eq!(lowered.claim().uncertainty(), UncertaintyReading::Sampling);
        let base = lowered.identity();
        assert_ne!(base, compiled(&partial, &ctx).identity(), "a different posterior");

        // The bootstrap request is the resampling plan, over rows.
        let resampled =
            tabular_study(query.clone(), tabular_posterior(false), InferenceMode::Frequentist, 20);
        let lowered = compiled(&resampled, &ctx);
        let plan = lowered.statistical().resampling().expect("bootstrap requested");
        assert_eq!(plan.method(), ResamplingMethod::Bootstrap);
        assert_eq!(plan.replicates(), 20);
        assert_eq!(plan.unit(), ResampleUnit::Row);
        assert_ne!(lowered.identity(), base, "resampling is part of the program");

        // Bayesian inference: ordinary posterior per atom, draws from the config.
        let posterior = tabular_study(query, tabular_posterior(false), bayesian(), 0);
        let lowered = compiled(&posterior, &ctx);
        let statistical = lowered.statistical();
        assert!(matches!(statistical.realization(), Realization::Bayesian(_)));
        assert_eq!(statistical.estimator(), Some(EstimatorId::BayesianGcomp));
        let plan = statistical.posterior().unwrap();
        assert_eq!(plan.algorithm(), PosteriorAlgorithm::AtomMixture);
        assert_eq!(plan.draws(), DRAWS);
        assert_eq!(lowered.claim().uncertainty(), UncertaintyReading::Posterior);
        assert_ne!(lowered.identity(), base, "inference is part of the program");

        // A different active level is a different target.
        let mut shifted = average_effect();
        shifted.active = Intervention::set(VariableId::from_raw(0), Value::f64(2.0));
        let moved = tabular_study(
            CausalQuery::AverageEffect(shifted),
            tabular_posterior(false),
            InferenceMode::Frequentist,
            0,
        );
        assert_ne!(identity(&moved, &ctx), base, "the target is part of the program");
        assert_eq!(identity(&frequentist, &ctx), base, "identity is stable across compiles");
    });
}

#[derive(Default)]
struct Stages(std::sync::Mutex<Vec<String>>);

impl antecedent_core::ProgressSink for Stages {
    fn report(&self, _fraction: f64, stage: &str) {
        self.0.lock().unwrap().push(stage.to_owned());
    }
}

#[test]
fn a_refused_response_coordinate_is_refused_before_any_atom_is_identified() {
    let t = VariableId::from_raw(0);
    let z = VariableId::from_raw(2);
    let query = ResponseQuery::new(ResponseFunctional::InterventionResponse {
        outcome: VariableId::from_raw(1),
        interventions: Arc::from([
            Intervention::set(t, Value::f64(1.0)),
            Intervention::set(z, Value::f64(0.0)),
        ]),
    });
    let study = tabular_study(
        CausalQuery::Response(query),
        tabular_posterior(false),
        InferenceMode::Frequentist,
        0,
    );
    let stages = Arc::new(Stages::default());
    let mut ctx = ExecutionContext::for_tests(41);
    ctx.progress = Some(Arc::clone(&stages) as Arc<dyn antecedent_core::ProgressSink>);
    let refusal = study.compile(&ctx).unwrap_err();
    assert!(refusal.to_string().contains("intervention"), "{refusal}");
    assert!(
        stages.0.lock().unwrap().iter().all(|stage| stage != "identify.compute"),
        "no posterior atom is identified before the refusal"
    );
}

#[test]
fn compile_identifies_the_atoms_once_and_preparation_installs_them() {
    on_large_stack(|| {
        let stages = Arc::new(Stages::default());
        let mut ctx = ExecutionContext::for_tests(43);
        ctx.progress = Some(Arc::clone(&stages) as Arc<dyn antecedent_core::ProgressSink>);
        let study = tabular_study(
            CausalQuery::AverageEffect(average_effect()),
            tabular_posterior(true),
            InferenceMode::Frequentist,
            0,
        );
        let identified =
            || stages.0.lock().unwrap().iter().filter(|stage| *stage == "identify.compute").count();
        let prepared = study.prepare(&ctx).unwrap();
        assert_eq!(identified(), 1, "compile identifies the atoms; preparation installs them");
        assert!(prepared.plan().lowered().unwrap().identification_retained());
        let _ = prepared.estimate_retained(&ctx).unwrap();
        assert_eq!(identified(), 1, "an estimate reuses the installed identification");
    });
}
