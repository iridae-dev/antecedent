//! Every licensed support-matrix cell is first-class on inspect and completes
//! inspect → preview → execute → claim → consume.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_lossless,
    clippy::cast_precision_loss,
    clippy::many_single_char_names,
    clippy::cast_possible_truncation,
    clippy::match_same_arms,
    clippy::needless_range_loop,
    clippy::too_many_lines
)]

use std::sync::Arc;

use antecedent::{
    AcceptedGraph, BayesianConfig, CellStatus, EstimatorId, IdentifierId, InferenceMode,
    InterferenceSpec, RefuteSuite, Study, StudyBuilder, TransportTrialSpec, cell_coordinate,
    licensed_support_cells,
};
use antecedent_core::{
    AnomalyAttributionQuery, AssignmentDesign, AverageEffectQuery, CausalQuery,
    ChangeAttributionQuery, ConditionalEffectQuery, ContinuousDomain, CounterfactualQuery,
    DerivativeScale, DerivativeWeighting, ExecutionContext, ExposureLevel, ExposureMapping,
    GridSpec, InterferenceFunctional, InterferenceQuery, Intervention,
    InterventionalDistributionQuery, Lag, MediationContrast, MediationQuery,
    PathSpecificEffectQuery, PopulationSelector, ResponseFunctional, ResponseQuery,
    SlotAvailability, TemporalEffectQuery, TemporalPolicy, TemporalResponseSpec, TransformIntent,
    TransportQuery, Value, VariableId,
};
use antecedent_data::{
    NetworkData, NetworkEdge, SamplingRegularity, TableView, TabularData, TimeIndex, TimeSeriesData,
};
use antecedent_discovery::GraphPosterior;
use antecedent_estimate::ContinuousResponseOptions;
use antecedent_graph::{
    Admg, Cpdag, Dag, DenseNodeId, Endpoint, MarkedEdge, MiddleMark, Pag, TemporalCpdag,
    TemporalDag, TieredBackground, WithinTier, ensure_lagged,
};
use antecedent_io::consume_analysis_result;
use antecedent_prob::InferenceDiagnostics;

mod common;

// The pinned temporal-PAG law and structure, in one owner.
use common::fixtures::{pinned_pag as identified_pag, pinned_pag_series as identified_pag_series};

fn vid(raw: u32) -> VariableId {
    VariableId::from_raw(raw)
}

fn nid(raw: u32) -> DenseNodeId {
    DenseNodeId::from_raw(raw)
}

fn refute_of(validation: &str) -> RefuteSuite {
    match validation {
        "none" => RefuteSuite::None,
        "cheap" => RefuteSuite::Cheap,
        "full" => RefuteSuite::Full,
        other => panic!("unexpected validation axis {other}"),
    }
}

fn inference_of(axis: &str) -> InferenceMode {
    match axis {
        "Frequentist" => InferenceMode::Frequentist,
        "Bayesian" => InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(8)),
        other => panic!("unexpected inference axis {other}"),
    }
}

fn binary_treatment_table(cols: &[(&str, usize)]) -> TabularData {
    let n = 48usize;
    let owned: Vec<(String, Vec<f64>)> = cols
        .iter()
        .enumerate()
        .map(|(j, (name, _))| {
            let values: Vec<f64> = (0..n)
                .map(|i| {
                    if j == 0 {
                        (i % 2) as f64
                    } else if *name == "y" {
                        0.3 + 0.8 * (i % 2) as f64 + 0.2 * ((i + 5) as f64 * 0.13).sin()
                    } else {
                        ((i / 2 + 3 * j) % 5) as f64 * 0.25 + 0.1 * ((i + j) as f64 * 0.2).sin()
                    }
                })
                .collect();
            ((*name).to_string(), values)
        })
        .collect();
    let refs: Vec<(&str, &[f64])> = owned.iter().map(|(n, v)| (n.as_str(), v.as_slice())).collect();
    TabularData::from_f64_columns(refs).unwrap()
}

fn discrete_joint_table(names: &[&str]) -> TabularData {
    let mut columns: Vec<Vec<f64>> = vec![Vec::new(); names.len()];
    for bits in 0..(1 << names.len()) {
        let count = 8 + (bits % 5);
        for _ in 0..count {
            for (j, column) in columns.iter_mut().enumerate() {
                column.push(((bits >> j) & 1) as f64);
            }
        }
    }
    let refs: Vec<(&str, &[f64])> =
        names.iter().zip(columns.iter()).map(|(n, v)| (*n, v.as_slice())).collect();
    TabularData::from_f64_columns(refs).unwrap()
}

fn static_table(cols: &[(&str, usize)]) -> TabularData {
    let n = 48usize;
    let owned: Vec<(String, Vec<f64>)> = cols
        .iter()
        .enumerate()
        .map(|(j, (name, _))| {
            let values: Vec<f64> =
                (0..n).map(|i| ((i + 3 * j) as f64 * 0.17).sin() + 0.25 * (i % 4) as f64).collect();
            ((*name).to_string(), values)
        })
        .collect();
    let refs: Vec<(&str, &[f64])> = owned.iter().map(|(n, v)| (n.as_str(), v.as_slice())).collect();
    TabularData::from_f64_columns(refs).unwrap()
}

fn series_from(names: &[&str]) -> TimeSeriesData {
    let n = 48usize;
    let owned: Vec<(String, Vec<f64>)> = names
        .iter()
        .enumerate()
        .map(|(j, name)| {
            let mut values = vec![0.0; n];
            for t in 1..n {
                values[t] = 0.4 * values[t - 1] + 0.2 * ((t + 5 * j) as f64 * 0.11).sin();
            }
            ((*name).to_string(), values)
        })
        .collect();
    let refs: Vec<(&str, &[f64])> = owned.iter().map(|(n, v)| (n.as_str(), v.as_slice())).collect();
    let data = TabularData::from_f64_columns(refs).unwrap();
    TimeSeriesData::try_new(
        data.storage().clone(),
        TimeIndex { regularity: SamplingRegularity::Regular { interval_ns: 1 }, length: n },
    )
    .unwrap()
}

fn temporal_mediation_series() -> TimeSeriesData {
    let n = 320usize;
    let mut t = vec![0.0; n];
    let mut m = vec![0.0; n];
    let mut y = vec![0.0; n];
    for i in 0..n {
        t[i] = (0.071 * i as f64).sin() + 0.35 * (0.137 * i as f64).cos();
    }
    for i in 1..n {
        m[i] = 0.8 * t[i - 1] + 0.12 * (0.43 * i as f64).sin();
        y[i] = 0.25 * t[i - 1] + 0.55 * m[i] + 0.09 * (0.29 * i as f64).cos();
    }
    let data = TabularData::from_f64_columns([
        ("t", t.as_slice()),
        ("m", m.as_slice()),
        ("y", y.as_slice()),
    ])
    .unwrap();
    TimeSeriesData::try_new(
        data.storage().clone(),
        TimeIndex { regularity: SamplingRegularity::Regular { interval_ns: 1 }, length: n },
    )
    .unwrap()
}

fn chain_dag(n: u32) -> Dag {
    let mut g = Dag::with_variables(n);
    for i in 0..n.saturating_sub(1) {
        g.insert_directed(nid(i), nid(i + 1)).unwrap();
    }
    g
}

fn mediation_dag() -> Dag {
    let mut g = Dag::with_variables(3);
    g.insert_directed(nid(0), nid(1)).unwrap();
    g.insert_directed(nid(1), nid(2)).unwrap();
    g.insert_directed(nid(0), nid(2)).unwrap();
    g
}

fn dag_for(cell: &antecedent::SupportCell, n: u32) -> Dag {
    match cell.query {
        "MediationEffect" | "PathSpecificEffect" => mediation_dag(),
        "AverageDerivative"
        | "PointDerivative"
        | "Elasticity"
        | "SemiElasticity"
        | "DirectionalDerivative"
        | "ResponseJacobian" => {
            let mut g = Dag::with_variables(n);
            if n >= 3 {
                g.insert_directed(nid(0), nid(2)).unwrap();
            }
            if n >= 4 {
                g.insert_directed(nid(1), nid(3)).unwrap();
            }
            g
        }
        "ConditionalEffect"
        | "AverageEffect"
        | "ResponseCurve"
        | "InterventionResponse"
        | "InterventionalDistribution"
        | "AnomalyAttribution"
        | "ChangeAttribution" => {
            if n >= 3 {
                let mut g = Dag::with_variables(n);
                g.insert_directed(nid(2), nid(0)).unwrap();
                g.insert_directed(nid(2), nid(1)).unwrap();
                g.insert_directed(nid(0), nid(1)).unwrap();
                g
            } else {
                chain_dag(n)
            }
        }
        "SustainedEffect" | "PulseEffect" | "TemporalMediationEffect" => chain_dag(n),
        "Counterfactual" => chain_dag(n.max(3)),
        "TransportQuery" => {
            let mut g = Dag::with_variables(n.max(5));
            g.insert_directed(nid(0), nid(1)).unwrap();
            g
        }
        "InterferenceQuery" => chain_dag(n.max(2)),
        other => panic!("unprofiled licensed query {other}"),
    }
}

fn confounder_admg() -> Admg {
    let mut g = Admg::with_variables(3);
    g.insert_directed(nid(0), nid(1)).unwrap();
    g.insert_bidirected(nid(0), nid(2)).unwrap();
    g
}

fn pag_envelope_pin() -> serde_json::Value {
    serde_json::from_str(include_str!(
        "../../../conformance/estimate/pag_ate_envelope_identified/expected.json"
    ))
    .unwrap()
}

fn pag_envelope_fixture() -> (TabularData, Pag) {
    let pin = pag_envelope_pin();
    let columns: Vec<&str> =
        pin["columns"].as_array().unwrap().iter().map(|value| value.as_str().unwrap()).collect();
    let mut values: Vec<Vec<f64>> = vec![Vec::new(); columns.len()];
    for cell in pin["contingency_table"].as_array().unwrap() {
        let count = usize::try_from(cell["count"].as_u64().unwrap()).unwrap();
        for (index, name) in columns.iter().enumerate() {
            values[index].extend(std::iter::repeat_n(cell[*name].as_f64().unwrap(), count));
        }
    }
    let pairs: Vec<(&str, &[f64])> =
        columns.iter().zip(values.iter()).map(|(name, col)| (*name, col.as_slice())).collect();
    let data = TabularData::from_f64_columns(pairs).unwrap();
    let index = |name: &str| {
        u32::try_from(columns.iter().position(|column| *column == name).unwrap()).unwrap()
    };
    let endpoint = |mark: &str| match mark {
        "tail" => Endpoint::Tail,
        "arrow" => Endpoint::Arrow,
        "circle" => Endpoint::Circle,
        other => panic!("unknown endpoint {other}"),
    };
    let mut pag = Pag::with_variables(u32::try_from(columns.len()).unwrap());
    for edge in pin["graph"]["marked_edges"].as_array().unwrap() {
        pag.insert_marked(MarkedEdge {
            a: DenseNodeId::from_raw(index(edge[0].as_str().unwrap())),
            b: DenseNodeId::from_raw(index(edge[1].as_str().unwrap())),
            at_a: endpoint(edge[2].as_str().unwrap()),
            at_b: endpoint(edge[3].as_str().unwrap()),
            middle: MiddleMark::Empty,
        })
        .unwrap();
    }
    (data, pag)
}

fn uses_pag_envelope(cell: &antecedent::SupportCell) -> bool {
    cell.graph_class == "Pag" && matches!(cell.query, "AverageEffect" | "ConditionalEffect")
}

fn uses_identified_temporal_pag(cell: &antecedent::SupportCell) -> bool {
    cell.graph_class == "TemporalPag"
}

fn identified_pag_pin() -> serde_json::Value {
    serde_json::from_str(include_str!(
        "../../../conformance/estimate/temporal_class_envelope/identified_pag.json"
    ))
    .unwrap()
}

fn uses_pag_response_curve(cell: &antecedent::SupportCell) -> bool {
    cell.graph_class == "Pag" && cell.query == "ResponseCurve"
}

fn pag_response_curve_fixture() -> (TabularData, Pag) {
    let pin: serde_json::Value = serde_json::from_str(include_str!(
        "../../../conformance/response/class_aware_envelope/pag_identified.json"
    ))
    .unwrap();
    let spec = &pin["curve"];
    let n = usize::try_from(spec["law"]["n"].as_u64().unwrap()).unwrap();
    let wave = |i: usize, freq: f64| (i as f64 * freq).sin();
    let z: Vec<f64> = (0..n).map(|i| wave(i, 0.37)).collect();
    let r: Vec<f64> = (0..n).map(|i| wave(i, 0.53)).collect();
    let t: Vec<f64> = (0..n).map(|i| 0.5 + 0.5 * z[i] + 0.4 * r[i] + 0.5 * wave(i, 0.61)).collect();
    let y: Vec<f64> = (0..n)
        .map(|i| {
            0.1 + spec["true_contrast"].as_f64().unwrap() * t[i] + 0.2 * z[i] + 0.02 * wave(i, 0.29)
        })
        .collect();
    let data = TabularData::from_f64_columns([
        ("t", t.as_slice()),
        ("y", y.as_slice()),
        ("z", z.as_slice()),
        ("r", r.as_slice()),
    ])
    .unwrap();
    let columns = ["t", "y", "z", "r"];
    let mut pag = Pag::with_variables(4);
    let node = |name: &str| {
        DenseNodeId::from_raw(
            u32::try_from(columns.iter().position(|c| *c == name).unwrap()).unwrap(),
        )
    };
    for edge in spec["graph"]["marked_edges"].as_array().unwrap() {
        pag.insert_marked(MarkedEdge {
            a: node(edge[0].as_str().unwrap()),
            b: node(edge[1].as_str().unwrap()),
            at_a: match edge[2].as_str().unwrap() {
                "tail" => Endpoint::Tail,
                "arrow" => Endpoint::Arrow,
                "circle" => Endpoint::Circle,
                other => panic!("unknown endpoint {other}"),
            },
            at_b: match edge[3].as_str().unwrap() {
                "tail" => Endpoint::Tail,
                "arrow" => Endpoint::Arrow,
                "circle" => Endpoint::Circle,
                other => panic!("unknown endpoint {other}"),
            },
            middle: MiddleMark::Empty,
        })
        .unwrap();
    }
    (data, pag)
}

fn pag_from_dag(dag: &Dag) -> Pag {
    let mut g = Pag::with_variables(u32::try_from(dag.node_count()).unwrap());
    for edge in dag.edges() {
        if edge.at_b == antecedent_graph::Endpoint::Arrow {
            g.insert_directed(edge.a, edge.b).unwrap();
        }
    }
    g
}

fn transport_admg() -> Admg {
    let mut g = Admg::with_variables(5);
    g.insert_directed(nid(0), nid(1)).unwrap();
    g
}

fn temporal_dag(n_vars: u32) -> TemporalDag {
    let mut g = TemporalDag::empty();
    if n_vars >= 3 {
        let t1 = ensure_lagged(&mut g, vid(0), Lag::from_raw(1)).unwrap();
        let m0 = ensure_lagged(&mut g, vid(1), Lag::CONTEMPORANEOUS).unwrap();
        let y0 = ensure_lagged(&mut g, vid(2), Lag::CONTEMPORANEOUS).unwrap();
        g.insert_directed(t1, m0).unwrap();
        g.insert_directed(t1, y0).unwrap();
        g.insert_directed(m0, y0).unwrap();
        return g;
    }
    if n_vars >= 2 {
        let x1 = ensure_lagged(&mut g, vid(0), Lag::from_raw(1)).unwrap();
        let y0 = ensure_lagged(&mut g, vid(1), Lag::CONTEMPORANEOUS).unwrap();
        g.insert_directed(x1, y0).unwrap();
    }
    g
}

fn static_gp(n_vars: usize) -> GraphPosterior {
    let cell = n_vars * n_vars;
    let mut contemporaneous = vec![0.0; cell];
    if n_vars >= 2 {
        contemporaneous[1] = 1.0;
    }
    if n_vars >= 3 {
        contemporaneous[2 * n_vars] = 1.0;
        contemporaneous[2 * n_vars + 1] = 1.0;
    }
    GraphPosterior::new(
        n_vars,
        vec![1.0],
        vec![0],
        contemporaneous,
        vec![0.0; cell],
        1.0,
        InferenceDiagnostics::analytic("licensed-compiler-static-gp"),
        0,
    )
    .unwrap()
}

fn temporal_gp(n_vars: usize) -> GraphPosterior {
    let cell = n_vars * n_vars;
    if n_vars >= 3 {
        let mut lagged = vec![0.0; cell];
        lagged[1] = 1.0;
        lagged[2] = 1.0;
        return GraphPosterior::new(
            n_vars,
            vec![1.0],
            vec![8],
            vec![0.0; cell],
            vec![0.0; cell],
            1.0,
            InferenceDiagnostics::analytic("licensed-compiler-temporal-mediation-gp"),
            0,
        )
        .unwrap()
        .with_lagged_marginals(1, lagged)
        .unwrap()
        .with_lag_masks(vec![6])
        .unwrap();
    }
    let mut lagged = vec![0.0; cell];
    if n_vars >= 2 {
        lagged[1] = 1.0;
    }
    static_gp(n_vars).with_lagged_marginals(1, lagged).unwrap().with_lag_masks(vec![2]).unwrap()
}

fn temporal_spec() -> TemporalResponseSpec {
    TemporalResponseSpec::new([1], TemporalPolicy::pulse(-1), Some(1)).unwrap()
}

fn mean_curve(treatment: u32, outcome: u32, temporal: bool) -> CausalQuery {
    let q = ResponseQuery::new(ResponseFunctional::MeanCurve {
        outcome: vid(outcome),
        treatment: ContinuousDomain::new(vid(treatment), GridSpec::Values(Arc::from([0.0, 1.0]))),
    });
    CausalQuery::Response(if temporal { q.with_temporal(temporal_spec()) } else { q })
}

fn intervention_response(treatments: &[u32], outcome: u32, temporal: bool) -> CausalQuery {
    let interventions: Arc<[Intervention]> =
        treatments.iter().map(|&t| Intervention::set(vid(t), Value::f64(1.0))).collect();
    let q = ResponseQuery::new(ResponseFunctional::InterventionResponse {
        outcome: vid(outcome),
        interventions,
    });
    CausalQuery::Response(if temporal { q.with_temporal(temporal_spec()) } else { q })
}

fn query_for(cell: &antecedent::SupportCell) -> CausalQuery {
    let temporal_graph =
        matches!(cell.graph_class, "TemporalDag" | "TemporalCpdag" | "TemporalPag");
    match cell.query {
        "AverageEffect" => {
            CausalQuery::AverageEffect(AverageEffectQuery::binary_ate(vid(0), vid(1)))
        }
        "ConditionalEffect" => {
            let modifier = if cell.graph_class == "Pag" { vid(5) } else { vid(2) };
            CausalQuery::ConditionalEffect(
                ConditionalEffectQuery::try_new(
                    AverageEffectQuery::binary_ate(vid(0), vid(1))
                        .with_effect_modifiers([modifier]),
                )
                .unwrap(),
            )
        }
        "Counterfactual" => CausalQuery::Counterfactual(
            CounterfactualQuery::new(
                vid(1),
                Arc::from([Intervention::set(vid(0), Value::f64(1.0))]),
            )
            .with_control_level(0.0),
        ),
        "InterventionalDistribution" => {
            CausalQuery::Distribution(InterventionalDistributionQuery::new(
                vid(1),
                [Intervention::set(vid(0), Value::f64(1.0))],
            ))
        }
        "MediationEffect" | "TemporalMediationEffect" => {
            let contrast = if cell.query == "TemporalMediationEffect" {
                MediationContrast::Mediated
            } else {
                MediationContrast::NaturalDirect
            };
            CausalQuery::Mediation(MediationQuery::binary(
                vid(0),
                vid(2),
                Arc::from([vid(1)]),
                contrast,
            ))
        }
        "PathSpecificEffect" => CausalQuery::PathSpecific(
            PathSpecificEffectQuery::binary(vid(0), vid(2)).with_path_nodes([vid(1)]),
        ),
        "PulseEffect" => CausalQuery::TemporalEffect(
            TemporalEffectQuery::pulse(vid(0), vid(1), 1.0)
                .with_policy(TemporalPolicy::pulse(-1))
                .with_horizon_steps(1)
                .with_max_history_lag(Some(1)),
        ),
        "SustainedEffect" => CausalQuery::TemporalEffect(
            TemporalEffectQuery::pulse(vid(0), vid(1), 1.0)
                .with_policy(TemporalPolicy::sustained(-1, -1))
                .with_horizon_steps(1)
                .with_max_history_lag(Some(1)),
        ),
        "ResponseCurve" => mean_curve(0, 1, temporal_graph),
        "InterventionResponse" if cell.graph_class == "CoDetermined" => {
            intervention_response(&[1, 2], 3, false)
        }
        "InterventionResponse" => intervention_response(&[0], 1, temporal_graph),
        "AverageDerivative" => {
            CausalQuery::Response(ResponseQuery::new(ResponseFunctional::AverageDerivative {
                outcome: vid(2),
                treatment: vid(0),
                weighting: DerivativeWeighting::Observed,
            }))
        }
        "PointDerivative" => {
            CausalQuery::Response(ResponseQuery::new(ResponseFunctional::PointDerivative {
                outcome: vid(2),
                treatment: vid(0),
                at: 0.5,
                order: 1,
                scale: DerivativeScale::Identity,
            }))
        }
        "Elasticity" => {
            CausalQuery::Response(ResponseQuery::new(ResponseFunctional::PointDerivative {
                outcome: vid(2),
                treatment: vid(0),
                at: 0.5,
                order: 1,
                scale: DerivativeScale::LogLog,
            }))
        }
        "SemiElasticity" => {
            CausalQuery::Response(ResponseQuery::new(ResponseFunctional::PointDerivative {
                outcome: vid(2),
                treatment: vid(0),
                at: 0.5,
                order: 1,
                scale: DerivativeScale::LogTreatment,
            }))
        }
        "DirectionalDerivative" => {
            CausalQuery::Response(ResponseQuery::new(ResponseFunctional::DirectionalDerivative {
                outcomes: Arc::from([vid(2), vid(3)]),
                treatments: Arc::from([vid(0), vid(1)]),
                at: Arc::from([0.5, 0.0]),
                direction: Arc::from([1.0, 1.0]),
            }))
        }
        "ResponseJacobian" => {
            CausalQuery::Response(ResponseQuery::new(ResponseFunctional::Jacobian {
                outcomes: Arc::from([vid(2), vid(3)]),
                treatments: Arc::from([vid(0), vid(1)]),
                at: Arc::from([0.5, 0.0]),
                scale: DerivativeScale::Identity,
            }))
        }
        "TransportQuery" => CausalQuery::Transport(TransportQuery::new(
            ResponseQuery::new(ResponseFunctional::MeanCurve {
                outcome: vid(1),
                treatment: ContinuousDomain::new(vid(0), GridSpec::Values(Arc::from([0.0, 1.0]))),
            }),
            "trial",
            "target",
            [vid(0)],
        )),
        "InterferenceQuery" => CausalQuery::Interference(InterferenceQuery::new(
            AssignmentDesign::Bernoulli { probabilities: Arc::from([0.5]) },
            ExposureMapping::NeighborCount,
            InterferenceFunctional::ExposureContrast {
                outcome: vid(0),
                from: ExposureLevel { own: 0.0, neighbors: 1.0 },
                to: ExposureLevel { own: 1.0, neighbors: 0.0 },
            },
        )),
        "AnomalyAttribution" => {
            CausalQuery::AnomalyAttribution(AnomalyAttributionQuery::new([vid(1)], 48))
        }
        "ChangeAttribution" => CausalQuery::ChangeAttribution(ChangeAttributionQuery::new(
            vid(1),
            PopulationSelector::TimeRange { start: 0, end: 24 },
            PopulationSelector::TimeRange { start: 24, end: 48 },
        )),
        other => panic!("unhandled licensed query {other}"),
    }
}

fn needs_series(cell: &antecedent::SupportCell) -> bool {
    matches!(cell.query, "PulseEffect" | "SustainedEffect" | "TemporalMediationEffect")
        || matches!(cell.graph_class, "TemporalDag" | "TemporalCpdag" | "TemporalPag")
}

fn n_vars(cell: &antecedent::SupportCell) -> u32 {
    match cell.query {
        "PathSpecificEffect" | "MediationEffect" | "TemporalMediationEffect" => 3,
        "AverageDerivative"
        | "PointDerivative"
        | "Elasticity"
        | "SemiElasticity"
        | "DirectionalDerivative"
        | "ResponseJacobian" => 4,
        "TransportQuery" => 5,
        "InterferenceQuery" => 2,
        "InterventionResponse" if cell.graph_class == "CoDetermined" => 4,
        "AverageEffect" if matches!(cell.graph_class, "CoDetermined" | "Unknown") => 4,
        "ConditionalEffect" => 3,
        _ if needs_series(cell) && cell.query == "TemporalMediationEffect" => 3,
        _ if needs_series(cell) => 2,
        _ => 3,
    }
}

enum CellData {
    Tabular(TabularData),
    Series(TimeSeriesData),
}

struct CellSetup {
    expected: String,
    builder: StudyBuilder,
    data: CellData,
}

fn cell_setup(cell: &antecedent::SupportCell) -> Result<CellSetup, String> {
    let expected = cell_coordinate(*cell);
    let query = query_for(cell);
    let n = n_vars(cell);
    let series = needs_series(cell);
    let inference = inference_of(cell.inference);
    let refute = refute_of(cell.validation);

    let (mut builder, data) = if uses_identified_temporal_pag(cell) {
        let pin = identified_pag_pin();
        let series = identified_pag_series(&pin);
        (Study::series(series.clone()), CellData::Series(series))
    } else if series {
        let names: Vec<&str> = match n {
            2 => vec!["x", "y"],
            3 => vec!["t", "m", "y"],
            _ => vec!["x", "y", "z", "w"][..n as usize].to_vec(),
        };
        let series = if cell.query == "TemporalMediationEffect" {
            temporal_mediation_series()
        } else {
            series_from(&names)
        };
        (Study::series(series.clone()), CellData::Series(series))
    } else {
        let table = if uses_pag_response_curve(cell) {
            pag_response_curve_fixture().0
        } else if uses_pag_envelope(cell) {
            pag_envelope_fixture().0
        } else if cell.graph_class == "CoDetermined" && cell.query == "InterventionResponse" {
            let n = 48usize;
            let z: Vec<f64> = (0..n).map(|i| (i % 2) as f64).collect();
            let t1: Vec<f64> = (0..n).map(|i| (i / 2 % 2) as f64).collect();
            let t2: Vec<f64> = (0..n).map(|i| (i / 4 % 2) as f64).collect();
            let y: Vec<f64> = (0..n)
                .map(|i| 0.4 * t1[i] + 0.3 * t2[i] + 0.2 * z[i] + 0.05 * (i as f64 * 0.1).sin())
                .collect();
            TabularData::from_f64_columns([
                ("z", z.as_slice()),
                ("t1", t1.as_slice()),
                ("t2", t2.as_slice()),
                ("y", y.as_slice()),
            ])
            .unwrap()
        } else if cell.graph_class == "CoDetermined" {
            binary_treatment_table(&[("t", 0), ("y", 1), ("z", 2), ("u", 3)])
        } else if cell.graph_class == "Unknown" {
            binary_treatment_table(&[("t", 0), ("y", 1), ("era", 2), ("m", 3)])
        } else if cell.query == "TransportQuery" {
            let n = 48usize;
            let a: Vec<f64> = (0..n).map(|i| (i % 2) as f64).collect();
            let trial: Vec<f64> = (0..n).map(|i| if i < n / 2 { 1.0 } else { 0.0 }).collect();
            let s: Vec<f64> = (0..n).map(|_| 0.4).collect();
            let e: Vec<f64> = (0..n).map(|_| 0.5).collect();
            let y: Vec<f64> = (0..n).map(|i| 0.2 + 0.8 * a[i] + 0.1 * trial[i]).collect();
            TabularData::from_f64_columns([
                ("a", a.as_slice()),
                ("y", y.as_slice()),
                ("trial", trial.as_slice()),
                ("s", s.as_slice()),
                ("e", e.as_slice()),
            ])
            .unwrap()
        } else if cell.query == "InterferenceQuery" {
            static_table(&[("y", 0), ("x", 1)])
        } else if cell.query == "InterventionalDistribution"
            || (cell.query == "AverageEffect" && cell.graph_class == "Admg")
        {
            discrete_joint_table(&["t", "y", "z"][..n.min(3) as usize])
        } else if matches!(
            cell.query,
            "AverageDerivative"
                | "PointDerivative"
                | "Elasticity"
                | "SemiElasticity"
                | "DirectionalDerivative"
                | "ResponseJacobian"
        ) {
            static_table(&[("a", 0), ("b", 1), ("y", 2), ("v", 3)])
        } else if matches!(cell.query, "MediationEffect" | "PathSpecificEffect") {
            discrete_joint_table(&["t", "m", "y"])
        } else if matches!(cell.query, "AverageEffect" | "ConditionalEffect" | "MediationEffect") {
            binary_treatment_table(&[("t", 0), ("y", 1), ("z", 2)])
        } else {
            static_table(&[("t", 0), ("y", 1), ("z", 2)])
        };
        (Study::tabular(table.clone()), CellData::Tabular(table))
    };

    builder = builder.query(query).refute(refute).inference(inference).bootstrap_replicates(0);

    builder = match cell.structure {
        "graph_posterior" => builder.graph_posterior(if series {
            temporal_gp(n as usize)
        } else {
            static_gp(n as usize)
        }),
        "explicit" | "accepted" => {
            let accepted = cell.structure == "accepted";
            match cell.graph_class {
                "Dag" => {
                    let g = dag_for(cell, n);
                    if accepted { builder.graph(AcceptedGraph::from(g)) } else { builder.graph(g) }
                }
                "Admg" => {
                    let g = if cell.query == "TransportQuery" {
                        transport_admg()
                    } else {
                        confounder_admg()
                    };
                    if accepted { builder.graph(AcceptedGraph::from(g)) } else { builder.graph(g) }
                }
                "Cpdag" => {
                    let g = Cpdag::from_dag(&dag_for(cell, n));
                    if accepted { builder.graph(AcceptedGraph::from(g)) } else { builder.graph(g) }
                }
                "Pag" => {
                    let g = if uses_pag_response_curve(cell) {
                        pag_response_curve_fixture().1
                    } else if uses_pag_envelope(cell) {
                        pag_envelope_fixture().1
                    } else {
                        pag_from_dag(&dag_for(cell, n))
                    };
                    if accepted { builder.graph(AcceptedGraph::from(g)) } else { builder.graph(g) }
                }
                "TemporalDag" => {
                    let g = temporal_dag(n);
                    if accepted { builder.graph(AcceptedGraph::from(g)) } else { builder.graph(g) }
                }
                "TemporalCpdag" => {
                    let g = TemporalCpdag::from_temporal_dag(&temporal_dag(n));
                    if accepted { builder.graph(AcceptedGraph::from(g)) } else { builder.graph(g) }
                }
                "TemporalPag" => {
                    let g = identified_pag(&identified_pag_pin());
                    if accepted { builder.graph(AcceptedGraph::from(g)) } else { builder.graph(g) }
                }
                "CoDetermined" => {
                    let background = if cell.query == "InterventionResponse" {
                        TieredBackground::from_named(
                            static_table(&[("z", 0), ("t1", 1), ("t2", 2), ("y", 3)]).schema(),
                            &[vec!["z"], vec!["t1", "t2"], vec!["y"]],
                            WithinTier::CoDetermined,
                        )
                    } else {
                        TieredBackground::from_named(
                            static_table(&[("t", 0), ("y", 1), ("z", 2), ("u", 3)]).schema(),
                            &[vec!["z", "u"], vec!["t"], vec!["y"]],
                            WithinTier::CoDetermined,
                        )
                    }
                    .map_err(|e| format!("{expected}: {e}"))?;
                    builder.tiered_background(background).map_err(|e| format!("{expected}: {e}"))?
                }
                "Unknown" => {
                    let background = TieredBackground::from_named(
                        static_table(&[("t", 0), ("y", 1), ("era", 2), ("m", 3)]).schema(),
                        &[vec!["era"], vec!["t", "m"], vec!["y"]],
                        WithinTier::Unknown,
                    )
                    .map_err(|e| format!("{expected}: {e}"))?;
                    builder.tiered_background(background).map_err(|e| format!("{expected}: {e}"))?
                }
                other => return Err(format!("{expected}: unhandled graph class {other}")),
            }
        }
        other => return Err(format!("{expected}: unhandled structure {other}")),
    };

    if cell.query == "PathSpecificEffect" {
        builder = builder
            .identifier(IdentifierId::PathSpecificNatural)
            .estimator(EstimatorId::FunctionalEffect);
    }
    if cell.query == "InterventionalDistribution" {
        builder = builder
            .identifier(IdentifierId::GeneralId)
            .estimator(EstimatorId::FunctionalDistribution);
    }
    if cell.query == "AverageEffect" && cell.graph_class == "Admg" {
        builder =
            builder.identifier(IdentifierId::GeneralId).estimator(EstimatorId::FunctionalEffect);
    }
    if matches!(
        cell.query,
        "PointDerivative"
            | "Elasticity"
            | "SemiElasticity"
            | "DirectionalDerivative"
            | "ResponseJacobian"
            | "AverageDerivative"
    ) {
        builder = builder.response_options(ContinuousResponseOptions {
            bandwidth: Some(0.35),
            ..ContinuousResponseOptions::default()
        });
    }

    if cell.query == "TransportQuery" {
        builder = builder.selection_targets(Arc::from([])).transport_trial(TransportTrialSpec {
            trial: vid(2),
            selection_probability: vid(3),
            treatment_probability: vid(4),
        });
    }
    if cell.query == "InterferenceQuery" {
        let units = static_table(&[("y", 0)]);
        let n_units = 48usize;
        let edges: Vec<NetworkEdge> = (0..n_units)
            .map(|i| NetworkEdge { from: i as u32, to: ((i + 1) % n_units) as u32, weight: 1.0 })
            .collect();
        let network = NetworkData::try_new(units, edges).map_err(|e| format!("{expected}: {e}"))?;
        let assignment: Arc<[bool]> = (0..n_units).map(|i| i % 2 == 0).collect();
        builder = builder.interference(InterferenceSpec { network, assignment });
    }

    Ok(CellSetup { expected, builder, data })
}

fn inspect_setup(setup: &CellSetup) -> Result<antecedent::CausalContract, String> {
    let expected = &setup.expected;
    let inspected =
        setup.builder.clone().inspect().map_err(|e| format!("{expected}: inspect {e}"))?;
    if inspected.support_status != Some(CellStatus::Licensed) {
        return Err(format!("{expected}: support_status={:?}", inspected.support_status));
    }
    match &inspected.reasoning.identification {
        SlotAvailability::Unavailable { .. } => {}
        other => return Err(format!("{expected}: identification was {other:?}")),
    }
    match &inspected.reasoning.support {
        SlotAvailability::Available(slot) => {
            let got = slot.matrix_coordinate.as_deref().unwrap_or("");
            if got != expected {
                return Err(format!("{expected}: inspect published {got}"));
            }
        }
        other => return Err(format!("{expected}: support slot {other:?}")),
    }
    Ok(inspected)
}

fn inspect_cell(cell: antecedent::SupportCell) -> Result<antecedent::CausalContract, String> {
    inspect_setup(&cell_setup(&cell)?)
}

/// The identifier and estimator a licensed cell's execution ran, as recorded on its
/// logical plan (`-` when the plan names none).
struct Route {
    coordinate: String,
    identifier: String,
    estimator: String,
}

fn consume_setup(setup: CellSetup) -> Result<Route, String> {
    let expected = setup.expected.clone();
    inspect_setup(&setup)?;
    let ctx = ExecutionContext::for_tests(1);
    let built = setup.builder.build().map_err(|e| format!("{expected}: build {e}"))?;
    let prepared = built.prepare(&ctx).map_err(|e| format!("{expected}: prepare {e}"))?;
    let contract = prepared.contract().map_err(|e| format!("{expected}: contract {e}"))?;
    let preview = prepared
        .preview_transform(TransformIntent::CompatibleDataReplace)
        .map_err(|e| format!("{expected}: preview {e}"))?;
    if preview.refused {
        return Err(format!("{expected}: compatible-data preview refused"));
    }
    if !contract.identities.program.is_some_and(|program| preview.binds_program(program)) {
        return Err(format!("{expected}: preview unbound from program"));
    }
    let result = match &setup.data {
        CellData::Tabular(data) => prepared.estimate(data, &ctx),
        CellData::Series(data) => prepared.estimate_series(data, &ctx),
    }
    .map_err(|e| format!("{expected}: execute {e}"))?;
    let claim = result.claim(&contract, &ctx).map_err(|e| format!("{expected}: claim {e}"))?;
    if claim.identities.program != contract.identities.program
        || claim.identities.target != contract.identities.target
    {
        return Err(format!("{expected}: claim identities drifted"));
    }
    if Some(claim.claim_id) == contract.identities.program {
        return Err(format!("{expected}: claim identity collapsed onto program identity"));
    }
    // A partially identified answer must carry its bounds. `mixture` is the
    // kind a claim takes when it withholds the scalar and publishes no
    // identified set, which leaves a consumer with neither.
    if claim.kind == antecedent_core::ClaimKind::Mixture {
        return Err(format!("{expected}: partial claim carries no identified set"));
    }
    let bytes = prepared
        .encode_contracted_result(&result, &expected, &ctx)
        .map_err(|e| format!("{expected}: encode {e}"))?;
    let consumed =
        consume_analysis_result(&bytes).map_err(|e| format!("{expected}: consume {e}"))?;
    if !consumed.acceptance.accepts_as_verified_program() {
        return Err(format!("{expected}: consume did not accept a verified program"));
    }
    let plan = &result.logical_plan;
    Ok(Route {
        coordinate: expected,
        identifier: plan.identifier.as_deref().unwrap_or("-").to_owned(),
        estimator: plan.estimator.as_deref().unwrap_or("-").to_owned(),
    })
}

const ROUTES: &str = "parity/licensed_routes.toml";

/// `parity/licensed_routes.toml` as the executions produce it.
fn render_routes(routes: &[Route]) -> String {
    let mut out = String::from(concat!(
        "# Generated by crates/antecedent/tests/v110_licensed_compiler.rs\n",
        "# (every_licensed_cell_completes_the_compiler_path): the identifier and estimator\n",
        "# each licensed cell's execution recorded on its logical plan. The test fails when\n",
        "# this file drifts; regenerate with\n",
        "#   UPDATE_LICENSED_ROUTES=1 cargo test -p antecedent --test v110_licensed_compiler \\\n",
        "#     every_licensed_cell_completes_the_compiler_path\n",
        "# scripts/generate_support_matrix_docs.py joins these component ids to the parity\n",
        "# rows whose external oracles exercise them.\n",
    ));
    for route in routes {
        out.push_str(&format!(
            "\n[[route]]\ncoordinate = \"{}\"\nidentifier = \"{}\"\nestimator = \"{}\"\n",
            route.coordinate, route.identifier, route.estimator
        ));
    }
    out
}

#[test]
fn every_licensed_cell_inspects_as_a_first_class_contract() {
    let mut failures = Vec::new();
    let mut n = 0usize;
    for cell in licensed_support_cells() {
        n += 1;
        if let Err(err) = inspect_cell(cell) {
            failures.push(err);
        }
    }
    assert_eq!(n, 341, "licensed inventory drifted");
    assert!(
        failures.is_empty(),
        "{} licensed cells are not first-class on inspect:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
fn every_licensed_query_has_a_fixture_profile() {
    let mut queries = std::collections::BTreeSet::new();
    for cell in licensed_support_cells() {
        queries.insert(cell.query);
    }
    for query in queries {
        let cell = antecedent::SupportCell {
            query,
            graph_class: "Dag",
            structure: "explicit",
            inference: "Frequentist",
            validation: "none",
        };
        let _ = dag_for(&cell, 4);
    }
}

#[test]
fn every_licensed_cell_completes_the_compiler_path() {
    let mut failures = Vec::new();
    let mut routes = Vec::new();
    let mut n = 0usize;
    for cell in licensed_support_cells() {
        n += 1;
        match cell_setup(&cell).and_then(consume_setup) {
            Ok(route) => routes.push(route),
            Err(err) => failures.push(err),
        }
    }
    assert_eq!(n, 341, "licensed inventory drifted");
    assert!(
        failures.is_empty(),
        "{} licensed cells did not finish inspect→preview→execute→claim→consume:\n{}",
        failures.len(),
        failures.join("\n")
    );
    let rendered = render_routes(&routes);
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..").join(ROUTES);
    if std::env::var("UPDATE_LICENSED_ROUTES").ok().as_deref() == Some("1") {
        std::fs::write(&path, &rendered).expect("write licensed routes");
    }
    let committed = std::fs::read_to_string(&path).unwrap_or_default();
    assert!(
        committed == rendered,
        "{ROUTES} does not match the identifiers and estimators the licensed cells ran; \
         regenerate with UPDATE_LICENSED_ROUTES=1"
    );
}
