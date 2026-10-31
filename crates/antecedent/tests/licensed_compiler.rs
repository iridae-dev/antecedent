//! Every licensed support-matrix cell is first-class on inspect and completes
//! inspect → preview → execute → claim → consume.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_possible_truncation,
    clippy::float_cmp,
    reason = "test scaffolding compares exact constants and indexes with small literals"
)]

use std::sync::Arc;

use antecedent::support::{classify_estimator, licensed_estimators, licensed_route_estimator};
use antecedent::{
    AcceptedGraph, BayesianConfig, CellStatus, EstimatorId, IdentifierId, InferenceMode,
    InterferenceSpec, RefuteSuite, Study, StudyBuilder, TransportTrialSpec, cell_coordinate,
    licensed_support_cells,
};
use antecedent_contract::{SlotAvailability, TransformIntent};
use antecedent_core::{
    AnomalyAttributionQuery, AssignmentDesign, AverageEffectQuery, CausalQuery,
    ChangeAttributionQuery, ConditionalEffectQuery, ContinuousDomain, CounterfactualQuery,
    DerivativeScale, DerivativeWeighting, ExecutionContext, ExposureLevel, ExposureMapping,
    GridSpec, InterferenceFunctional, InterferenceQuery, Intervention,
    InterventionalDistributionQuery, Lag, MediationContrast, MediationQuery,
    NestedCounterfactualQuery, PathSpecificEffectQuery, PopulationSelector, ResponseFunctional,
    ResponseQuery, TemporalEffectQuery, TemporalPolicy, TemporalResponseSpec, TransportQuery,
    Value, VariableId,
};
use antecedent_data::{
    NetworkData, NetworkEdge, SamplingRegularity, TableView, TabularData, TimeIndex, TimeSeriesData,
};
use antecedent_discovery::adjacency_mask::{adjacency_mask_from_admg, set_edge};
use antecedent_discovery::{
    AdjacencyMask, GraphPosterior, GraphPosteriorAtomKind, SamplingSummary,
};
use antecedent_estimate::ContinuousResponseOptions;
use antecedent_graph::{
    Admg, Cpdag, Dag, DenseNodeId, Endpoint, MarkedEdge, MiddleMark, Pag, TemporalAdmg,
    TemporalCpdag, TemporalDag, TemporalPag, TieredBackground, WithinTier, ensure_lagged,
};
use antecedent_io::consume_analysis_result;
use antecedent_prob::InferenceDiagnostics;

#[path = "fixtures_common/mod.rs"]
mod common;

// The pinned temporal-PAG law and structure, in one owner.
use common::fixtures::{pinned_pag as identified_pag, pinned_pag_series as identified_pag_series};

/// Whether the prepared handle's plan view names the retained sealed operation `name`.
fn retains(prepared: &antecedent::PreparedStudy, name: &str) -> bool {
    prepared.plan_view().is_ok_and(|view| view.operation() == Some(name))
}

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
                column.push(f64::from((bits >> j) & 1));
            }
        }
    }
    let refs: Vec<(&str, &[f64])> =
        names.iter().zip(columns.iter()).map(|(n, v)| (*n, v.as_slice())).collect();
    TabularData::from_f64_columns(refs).unwrap()
}

/// [`discrete_joint_table`] with the second column following the first, so a permuted first
/// column changes the table.
fn dependent_joint_table(names: &[&str]) -> TabularData {
    let mut columns: Vec<Vec<f64>> = vec![Vec::new(); names.len()];
    for bits in 0..(1 << names.len()) {
        let agree = names.len() < 2 || (bits & 1) == ((bits >> 1) & 1);
        let count = 6 + if agree { 9 } else { 0 } + (bits % 3);
        for _ in 0..count {
            for (j, column) in columns.iter_mut().enumerate() {
                column.push(f64::from((bits >> j) & 1));
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

#[allow(
    clippy::needless_range_loop,
    reason = "each step reads t[i - 1] and m[i] alongside the index-dependent signal term"
)]
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

fn dag_for(cell: antecedent::SupportCell, n: u32) -> Dag {
    match cell.query.as_str() {
        "MediationEffect" | "PathSpecificEffect" | "NestedCounterfactualEffect" => mediation_dag(),
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
        "SustainedEffect"
        | "PulseEffect"
        | "TemporalConditionalPulse"
        | "TemporalConditionalSustained"
        | "TemporalMediationEffect" => chain_dag(n),
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

/// Four-variable ADMG of the derivative queries: both treatments cause both outcomes and a latent
/// common cause of the two outcomes stays in the graph, so the back-door set is empty.
fn derivative_admg() -> Admg {
    let mut g = Admg::with_variables(4);
    for (from, to) in [(0, 2), (1, 2), (0, 3), (1, 3)] {
        g.insert_directed(nid(from), nid(to)).unwrap();
    }
    g.insert_bidirected(nid(2), nid(3)).unwrap();
    g
}

/// Five-variable PAG of the derivative queries: `r -> a`, `r -> b`, and both treatments cause both
/// outcomes. `r` is a parent of each treatment and not adjacent to an outcome, so every
/// treatment-to-outcome edge is visible and the back-door set is empty.
fn derivative_pag() -> Pag {
    let mut g = Pag::with_variables(5);
    for (from, to) in [(4, 0), (4, 1), (0, 2), (1, 2), (0, 3), (1, 3)] {
        g.insert_directed(nid(from), nid(to)).unwrap();
    }
    g
}

fn confounder_admg() -> Admg {
    let mut g = Admg::with_variables(3);
    g.insert_directed(nid(0), nid(1)).unwrap();
    g.insert_bidirected(nid(0), nid(2)).unwrap();
    g
}

/// `t -> m -> y` and `t -> y`, no latent cause: every component of the mediation is identified.
fn mediation_admg() -> Admg {
    let mut g = Admg::with_variables(3);
    for (from, to) in [(0, 1), (0, 2), (1, 2)] {
        g.insert_directed(nid(from), nid(to)).unwrap();
    }
    g
}

/// Columns `t, m, y, c, w`: `c -> t` and `w -> m` make every edge out of `t` and into `y` visible.
fn visible_mediation_static_pag() -> Pag {
    let mut g = Pag::with_variables(5);
    for (from, to) in [(3, 0), (4, 1), (0, 1), (0, 2), (1, 2)] {
        g.insert_directed(nid(from), nid(to)).unwrap();
    }
    g
}

fn is_attribution(cell: antecedent::SupportCell) -> bool {
    matches!(cell.query.as_str(), "AnomalyAttribution" | "ChangeAttribution")
}

/// `0 o-> 1` among `n` variables: one MAG completion makes the target a root, which identifies
/// its attribution, and the other is the invisible directed edge whose mechanism a hedge leaves
/// unidentified, so the class attribution is partly identified.
fn attribution_pag(n: u32) -> Pag {
    let mut g = Pag::with_variables(n);
    g.insert_circle_arrow(nid(0), nid(1)).unwrap();
    g
}

/// `x(t-1) o-> y(t)`: the same two completions on a one-lag template.
fn attribution_temporal_pag() -> TemporalPag {
    let mut g = TemporalPag::empty();
    let cause = g.add_lagged(vid(0), Lag::from_raw(1)).unwrap();
    let effect = g.add_lagged(vid(1), Lag::CONTEMPORANEOUS).unwrap();
    g.insert_circle_arrow(cause, effect).unwrap();
    g
}

/// A lag-template posterior whose single atom is `x o-> y` with the lagged edge `x(t-1) -> y(t)`.
fn attribution_temporal_pag_gp(n_vars: usize) -> GraphPosterior {
    let adjacency = set_edge(0, n_vars, 0, 1, true);
    GraphPosterior::from_atoms(
        n_vars,
        vec![1.0],
        vec![adjacency.clone()],
        vec![0.0; n_vars * n_vars],
        vec![0.0; n_vars * n_vars],
        SamplingSummary {
            ess: 1.0,
            diagnostics: InferenceDiagnostics::analytic("licensed-compiler-attribution-pag-gp"),
            rejected_invalid: 0,
        },
    )
    .unwrap()
    .with_mark_atoms(vec![adjacency])
    .unwrap()
    .with_lagged_marginals(1, vec![0.0; n_vars * n_vars])
    .unwrap()
    .with_lag_masks(vec![1_u64 << 1])
    .unwrap()
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

fn uses_pag_envelope(cell: antecedent::SupportCell) -> bool {
    cell.graph_class.as_str() == "Pag"
        && matches!(cell.query.as_str(), "AverageEffect" | "ConditionalEffect")
        && cell.structure.as_str() != "graph_posterior"
}

fn uses_identified_temporal_pag(cell: antecedent::SupportCell) -> bool {
    cell.graph_class.as_str() == "TemporalPag"
        && cell.structure.as_str() != "graph_posterior"
        && !is_attribution(cell)
        && !matches!(
            cell.query.as_str(),
            "TemporalConditionalPulse"
                | "TemporalConditionalSustained"
                | "AverageDerivative"
                | "PointDerivative"
                | "Elasticity"
                | "SemiElasticity"
                | "DirectionalDerivative"
                | "ResponseJacobian"
                | "Counterfactual"
                | "InterventionalDistribution"
                | "NestedCounterfactualEffect"
                | "PathSpecificEffect"
        )
}

/// Directed mediation PAG with a lag-1 parent of treatment, so `t → y` is visible
/// and the back-door set of the licensed point queries stays empty.
fn visible_mediation_pag() -> TemporalPag {
    let mut g = TemporalPag::empty();
    let t1 = g.add_lagged(vid(0), Lag::from_raw(1)).unwrap();
    let m0 = g.add_lagged(vid(1), Lag::CONTEMPORANEOUS).unwrap();
    let y0 = g.add_lagged(vid(2), Lag::CONTEMPORANEOUS).unwrap();
    let r1 = g.add_lagged(vid(3), Lag::from_raw(1)).unwrap();
    g.insert_directed(r1, t1).unwrap();
    g.insert_directed(t1, m0).unwrap();
    g.insert_directed(t1, y0).unwrap();
    g.insert_directed(m0, y0).unwrap();
    g
}

fn identified_pag_pin() -> serde_json::Value {
    serde_json::from_str(include_str!(
        "../../../conformance/estimate/temporal_class_envelope/identified_pag.json"
    ))
    .unwrap()
}

// A confounder triangle read as a MAG leaves `t -> y` invisible (nothing points
// into `t` from outside `y`'s neighbourhood), so PAG responses are refused on
// it; these cells need the fixture whose treatment edge has a visibility witness.
fn uses_pag_response_curve(cell: antecedent::SupportCell) -> bool {
    cell.graph_class.as_str() == "Pag"
        && matches!(cell.query.as_str(), "ResponseCurve" | "InterventionResponse")
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
    GraphPosterior::from_atoms(
        n_vars,
        vec![1.0],
        vec![0_u64],
        contemporaneous.clone(),
        contemporaneous,
        SamplingSummary {
            ess: 1.0,
            diagnostics: InferenceDiagnostics::analytic("licensed-compiler-static-gp"),
            rejected_invalid: 0,
        },
    )
    .unwrap()
}

fn temporal_gp(n_vars: usize) -> GraphPosterior {
    let cell = n_vars * n_vars;
    if n_vars >= 3 {
        let mut lagged = vec![0.0; cell];
        lagged[1] = 1.0;
        lagged[2] = 1.0;
        return GraphPosterior::from_atoms(
            n_vars,
            vec![1.0],
            vec![8_u64],
            vec![0.0; cell],
            vec![0.0; cell],
            SamplingSummary {
                ess: 1.0,
                diagnostics: InferenceDiagnostics::analytic(
                    "licensed-compiler-temporal-mediation-gp",
                ),
                rejected_invalid: 0,
            },
        )
        .unwrap()
        .with_lagged_marginals(1, lagged)
        .unwrap()
        .with_lag_masks(vec![6_u64])
        .unwrap();
    }
    let mut lagged = vec![0.0; cell];
    if n_vars >= 2 {
        lagged[1] = 1.0;
    }
    static_gp(n_vars).with_lagged_marginals(1, lagged).unwrap().with_lag_masks(vec![2_u64]).unwrap()
}

/// Reconstructible TemporalCpdag/Pag atom.
///
/// `TemporalPag` MAG adjustment needs a contemporaneous arrowhead into the
/// treatment so the unfolded `t_{t-1} → y_t` edge is visible. `2→0` and `2→1`
/// plus lag `0→1` is that backdoor template.
/// Masks of [`temporal_dag`], plus a contemporaneous witness into treatment
/// for a `TemporalPag` atom. That witness is what makes the lagged treatment
/// edge visible without putting a confounder on the outcome.
fn point_family_gp(n_vars: usize, pag: bool) -> GraphPosterior {
    let cell = n_vars * n_vars;
    let mut lag_mask = 0u64;
    let mut adjacency = AdjacencyMask::EMPTY;
    if n_vars >= 2 {
        lag_mask |= 1u64 << 1;
    }
    if n_vars >= 3 {
        lag_mask |= 1u64 << 2;
        adjacency = set_edge(0, n_vars, 1, 2, true);
    }
    if pag && n_vars >= 4 {
        adjacency = set_edge(adjacency, n_vars, 3, 0, true);
    }
    GraphPosterior::from_atoms(
        n_vars,
        vec![1.0],
        vec![adjacency],
        vec![0.0; cell],
        vec![0.0; cell],
        SamplingSummary {
            ess: 1.0,
            diagnostics: InferenceDiagnostics::analytic("licensed-compiler-temporal-class-point"),
            rejected_invalid: 0,
        },
    )
    .unwrap()
    .with_lagged_marginals(1, vec![0.0; cell])
    .unwrap()
    .with_lag_masks(vec![lag_mask])
    .unwrap()
}

/// One latent-confounded atom `z -> t -> y`, `z <-> y` (no bow, so an ADMG posterior can encode
/// it): conditioning on the modifier `z` closes `t <- z <-> y`, so the conditional effect is
/// identified by IDC on the atom.
fn conditional_admg_gp() -> GraphPosterior {
    let mut g = Admg::with_variables(3);
    g.insert_directed(nid(2), nid(0)).unwrap();
    g.insert_directed(nid(0), nid(1)).unwrap();
    g.insert_bidirected(nid(2), nid(1)).unwrap();
    GraphPosterior::from_atoms(
        3,
        vec![1.0],
        vec![adjacency_mask_from_admg(&g).unwrap()],
        vec![0.0; 9],
        vec![0.0; 9],
        SamplingSummary {
            ess: 1.0,
            diagnostics: InferenceDiagnostics::analytic("licensed-compiler-conditional-admg-gp"),
            rejected_invalid: 0,
        },
    )
    .unwrap()
}

/// One front-door atom `t -> y -> z`, `t <-> z` over `(t, y, z)`: the curve of `t` on `z` is
/// identified through the mediator `y` and its placebo (a permuted treatment) has spread.
fn frontdoor_curve_gp() -> GraphPosterior {
    let mut g = Admg::with_variables(3);
    g.insert_directed(nid(0), nid(1)).unwrap();
    g.insert_directed(nid(1), nid(2)).unwrap();
    g.insert_bidirected(nid(0), nid(2)).unwrap();
    GraphPosterior::from_atoms(
        3,
        vec![1.0],
        vec![adjacency_mask_from_admg(&g).unwrap()],
        vec![0.0; 9],
        vec![0.0; 9],
        SamplingSummary {
            ess: 1.0,
            diagnostics: InferenceDiagnostics::analytic("licensed-compiler-frontdoor-curve-gp"),
            rejected_invalid: 0,
        },
    )
    .unwrap()
}

fn temporal_class_gp(n_vars: usize) -> GraphPosterior {
    let cell = n_vars * n_vars;
    let mut lagged = vec![0.0; cell];
    let mut adjacency = AdjacencyMask::EMPTY;
    let mut lag_mask = 0u64;
    if n_vars >= 2 {
        lagged[1] = 1.0;
        lag_mask |= 1u64 << 1;
    }
    if n_vars >= 3 {
        lagged[2 * n_vars + 1] = 1.0;
        adjacency = set_edge(set_edge(0, n_vars, 2, 0, true), n_vars, 2, 1, true);
    }
    GraphPosterior::from_atoms(
        n_vars,
        vec![1.0],
        vec![adjacency],
        vec![0.0; cell],
        vec![0.0; cell],
        SamplingSummary {
            ess: 1.0,
            diagnostics: InferenceDiagnostics::analytic("licensed-compiler-temporal-class-gp"),
            rejected_invalid: 0,
        },
    )
    .unwrap()
    .with_lagged_marginals(1, lagged)
    .unwrap()
    .with_lag_masks(vec![lag_mask])
    .unwrap()
}

fn temporal_spec() -> TemporalResponseSpec {
    TemporalResponseSpec::new([1], TemporalPolicy::pulse(-1), Some(1)).unwrap()
}

fn temporalize(query: ResponseQuery, temporal: bool) -> CausalQuery {
    CausalQuery::Response(if temporal { query.with_temporal(temporal_spec()) } else { query })
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

#[allow(
    clippy::too_many_lines,
    reason = "one arm per support-matrix cell maps the cell to its literal query"
)]
fn query_for(cell: antecedent::SupportCell) -> CausalQuery {
    let temporal_graph =
        matches!(cell.graph_class.as_str(), "TemporalDag" | "TemporalCpdag" | "TemporalPag");
    match cell.query.as_str() {
        "AverageEffect" => {
            CausalQuery::AverageEffect(AverageEffectQuery::binary_ate(vid(0), vid(1)))
        }
        "ConditionalEffect" => {
            let modifier = if cell.graph_class.as_str() == "Pag"
                && cell.structure.as_str() != "graph_posterior"
            {
                vid(5)
            } else {
                vid(2)
            };
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
        "NestedCounterfactualEffect" => CausalQuery::NestedCounterfactual(
            NestedCounterfactualQuery::new(vid(0), vid(1), vid(2)).unwrap(),
        ),
        "InterventionalDistribution" => {
            CausalQuery::Distribution(InterventionalDistributionQuery::new(
                vid(1),
                [Intervention::set(vid(0), Value::f64(1.0))],
            ))
        }
        "MediationEffect" | "TemporalMediationEffect" => {
            let contrast = if cell.query.as_str() == "TemporalMediationEffect" {
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
        "TemporalConditionalPulse" => CausalQuery::TemporalEffect(
            TemporalEffectQuery::pulse(vid(0), vid(1), 1.0)
                .with_policy(TemporalPolicy::pulse(-1))
                .with_horizon_steps(1)
                .with_max_history_lag(Some(1))
                .with_effect_modifiers([vid(2)]),
        ),
        "TemporalConditionalSustained" => CausalQuery::TemporalEffect(
            TemporalEffectQuery::pulse(vid(0), vid(1), 1.0)
                .with_policy(TemporalPolicy::sustained(-1, -1))
                .with_horizon_steps(1)
                .with_max_history_lag(Some(1))
                .with_effect_modifiers([vid(2)]),
        ),
        // The refuters of a graph-posterior curve over mixed atoms permute the treatment; a curve
        // whose outcome is a child of the treatment alone has a placebo with no spread, so the
        // validated cells read the front-door outcome `z` through the mediator `y`.
        "ResponseCurve"
            if cell.structure.as_str() == "graph_posterior"
                && matches!(cell.graph_class.as_str(), "Admg" | "TemporalAdmg")
                && cell.validation.as_str() != "none" =>
        {
            mean_curve(0, 2, temporal_graph)
        }
        "ResponseCurve" => mean_curve(0, 1, temporal_graph),
        "InterventionResponse" if cell.graph_class.as_str() == "CoDetermined" => {
            intervention_response(&[1, 2], 3, false)
        }
        "InterventionResponse" => intervention_response(&[0], 1, temporal_graph),
        "AverageDerivative" => temporalize(
            ResponseQuery::new(ResponseFunctional::AverageDerivative {
                outcome: vid(2),
                treatment: vid(0),
                weighting: DerivativeWeighting::Observed,
            }),
            temporal_graph,
        ),
        "PointDerivative" => temporalize(
            ResponseQuery::new(ResponseFunctional::PointDerivative {
                outcome: vid(2),
                treatment: vid(0),
                at: if temporal_graph { 2.0 } else { 0.5 },
                order: 1,
                scale: DerivativeScale::Identity,
            }),
            temporal_graph,
        ),
        "Elasticity" => temporalize(
            ResponseQuery::new(ResponseFunctional::PointDerivative {
                outcome: vid(2),
                treatment: vid(0),
                at: if temporal_graph { 2.0 } else { 0.5 },
                order: 1,
                scale: DerivativeScale::LogLog,
            }),
            temporal_graph,
        ),
        "SemiElasticity" => temporalize(
            ResponseQuery::new(ResponseFunctional::PointDerivative {
                outcome: vid(2),
                treatment: vid(0),
                at: if temporal_graph { 2.0 } else { 0.5 },
                order: 1,
                scale: DerivativeScale::LogTreatment,
            }),
            temporal_graph,
        ),
        "DirectionalDerivative" => temporalize(
            ResponseQuery::new(ResponseFunctional::DirectionalDerivative {
                outcomes: Arc::from([vid(2), vid(3)]),
                treatments: Arc::from([vid(0), vid(1)]),
                at: if temporal_graph { Arc::from([2.0, 1.0]) } else { Arc::from([0.5, 0.0]) },
                direction: Arc::from([1.0, 1.0]),
            }),
            temporal_graph,
        ),
        "ResponseJacobian" => temporalize(
            ResponseQuery::new(ResponseFunctional::Jacobian {
                outcomes: Arc::from([vid(2), vid(3)]),
                treatments: Arc::from([vid(0), vid(1)]),
                at: if temporal_graph { Arc::from([2.0, 1.0]) } else { Arc::from([0.5, 0.0]) },
                scale: DerivativeScale::Identity,
            }),
            temporal_graph,
        ),
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

fn needs_series(cell: antecedent::SupportCell) -> bool {
    matches!(cell.query.as_str(), "PulseEffect" | "SustainedEffect" | "TemporalMediationEffect")
        || matches!(cell.graph_class.as_str(), "TemporalDag" | "TemporalCpdag" | "TemporalPag")
}

fn point_family_query(cell: antecedent::SupportCell) -> bool {
    matches!(
        cell.query.as_str(),
        "TemporalConditionalPulse"
            | "TemporalConditionalSustained"
            | "AverageDerivative"
            | "PointDerivative"
            | "Elasticity"
            | "SemiElasticity"
            | "DirectionalDerivative"
            | "ResponseJacobian"
            | "Counterfactual"
            | "InterventionalDistribution"
            | "NestedCounterfactualEffect"
            | "PathSpecificEffect"
    )
}

#[allow(
    clippy::match_same_arms,
    reason = "one arm per support-matrix query kind, in matrix order, so each reads against the row it mirrors"
)]
fn n_vars(cell: antecedent::SupportCell) -> u32 {
    if cell.graph_class.as_str() == "TemporalPag"
        && cell.structure.as_str() == "graph_posterior"
        && point_family_query(cell)
    {
        // A fourth variable is the visibility witness into treatment. The
        // query still names the first three columns.
        return 4;
    }
    match cell.query.as_str() {
        "PathSpecificEffect"
        | "MediationEffect"
        | "TemporalMediationEffect"
        | "NestedCounterfactualEffect" => 3,
        "AverageDerivative"
        | "PointDerivative"
        | "Elasticity"
        | "SemiElasticity"
        | "DirectionalDerivative"
        | "ResponseJacobian" => 4,
        "TransportQuery" => 5,
        "InterferenceQuery" => 2,
        "InterventionResponse" if cell.graph_class.as_str() == "CoDetermined" => 4,
        "AverageEffect" if matches!(cell.graph_class.as_str(), "CoDetermined" | "Unknown") => 4,
        "ConditionalEffect" | "TemporalConditionalPulse" | "TemporalConditionalSustained" => 3,
        _ if needs_series(cell) && cell.query.as_str() == "TemporalMediationEffect" => 3,
        _ if needs_series(cell) && cell.graph_class.as_str() == "TemporalPag" => 3,
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

/// The path-specific, interventional-distribution and mediation cells over a graph posterior read
/// the identified atoms of the mixture fixture: each class has atoms the identifier decides, on
/// the count table the fixture enumerates, so the mixture is the posterior-weighted mean of
/// identified functionals.
fn mixture_setup(cell: antecedent::SupportCell) -> Option<Result<CellSetup, String>> {
    let family = match cell.query.as_str() {
        "PathSpecificEffect" => "path_specific",
        "InterventionalDistribution" => "distribution",
        "MediationEffect" => "mediation",
        _ => return None,
    };
    if cell.structure.as_str() != "graph_posterior" {
        return None;
    }
    let class = match cell.graph_class.as_str() {
        "Dag" => "dag",
        "Admg" => "admg",
        "TemporalAdmg" => "temporal_admg",
        "Cpdag" => "cpdag",
        "Pag" => "pag",
        _ => return None,
    };
    let pin: serde_json::Value = serde_json::from_str(include_str!(
        "../../../conformance/estimate/functional_graph_posterior_mixtures/expected.json"
    ))
    .expect("mixture fixture");
    let fixture_class = if class == "temporal_admg" { "admg" } else { class };
    let entry = &pin["posteriors"][family][fixture_class];
    let table = &pin["tables"][entry["table"].as_str().unwrap()];
    let names: Vec<String> = table["columns"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| c.as_str().unwrap().to_owned())
        .collect();
    let index_of =
        |name: &str| u32::try_from(names.iter().position(|n| n == name).unwrap()).unwrap();
    let scale: u64 = if entry["table"] == "linear_mediation" || cell.validation.as_str() == "none" {
        1
    } else {
        40
    };
    let mut values: Vec<Vec<f64>> = vec![Vec::new(); names.len()];
    for row in table["contingency_table"].as_array().unwrap() {
        let count = usize::try_from(row["count"].as_u64().unwrap() * scale).unwrap();
        for (j, name) in names.iter().enumerate() {
            values[j].extend(std::iter::repeat_n(row[name.as_str()].as_f64().unwrap(), count));
        }
    }
    let rows = values[0].len();
    let order: Vec<usize> = (0..rows).map(|row| (row * 97) % rows).collect();
    let shuffled: Vec<Vec<f64>> =
        values.iter().map(|column| order.iter().map(|&row| column[row]).collect()).collect();
    let data = TabularData::from_f64_columns(
        names.iter().zip(&shuffled).map(|(n, v)| (n.as_str(), v.as_slice())),
    )
    .unwrap();
    let edges = |list: &serde_json::Value| -> Vec<(u32, u32)> {
        list.as_array()
            .map(|list| {
                list.iter()
                    .map(|e| (index_of(e[0].as_str().unwrap()), index_of(e[1].as_str().unwrap())))
                    .collect()
            })
            .unwrap_or_default()
    };
    let n = names.len();
    let atoms = entry["atoms"].as_array().unwrap();
    let masks: Vec<u64> = atoms
        .iter()
        .map(|atom| {
            let graph = &atom["graph"];
            match class {
                "dag" | "pag" => edges(&graph["directed"]).iter().fold(0_u64, |m, &(a, b)| {
                    antecedent_discovery::set_edge(m, n, a as usize, b as usize, true)
                }),
                "admg" | "temporal_admg" => {
                    let mut g = Admg::with_variables(n as u32);
                    for (a, b) in edges(&graph["directed"]) {
                        g.insert_directed(nid(a), nid(b)).unwrap();
                    }
                    for (a, b) in edges(&graph["bidirected"]) {
                        g.insert_bidirected(nid(a), nid(b)).unwrap();
                    }
                    antecedent_discovery::adjacency_mask_from_admg(&g).unwrap()
                }
                _ => {
                    let mut g = Cpdag::with_variables(n as u32);
                    for (a, b) in edges(&graph["directed"]) {
                        g.insert_directed(nid(a), nid(b)).unwrap();
                    }
                    for (a, b) in edges(&graph["undirected"]) {
                        g.insert_undirected(nid(a), nid(b)).unwrap();
                    }
                    antecedent_discovery::adjacency_mask_from_cpdag(&g).unwrap()
                }
            }
        })
        .collect();
    let weights: Vec<f64> = atoms
        .iter()
        .map(|atom| {
            let (num, den) = atom["weight"].as_str().unwrap().split_once('/').unwrap();
            num.parse::<f64>().unwrap() / den.parse::<f64>().unwrap()
        })
        .collect();
    let kind = match class {
        "dag" => GraphPosteriorAtomKind::Dag,
        "admg" => GraphPosteriorAtomKind::Admg,
        "temporal_admg" => GraphPosteriorAtomKind::TemporalAdmg,
        "cpdag" => GraphPosteriorAtomKind::Cpdag,
        _ => GraphPosteriorAtomKind::Pag,
    };
    let mut posterior = GraphPosterior::new(
        n,
        weights,
        masks.clone(),
        vec![0.0; n * n],
        vec![0.0; n * n],
        SamplingSummary {
            ess: 1.0,
            diagnostics: InferenceDiagnostics::analytic("licensed-compiler-mixture"),
            rejected_invalid: 0,
        },
    )
    .unwrap()
    .with_atom_kind(kind);
    if kind == GraphPosteriorAtomKind::Pag {
        posterior = posterior.with_mark_masks(vec![0; masks.len()]).unwrap();
    }
    let v = |name: &str| vid(index_of(name));
    let q = &pin["queries"][family];
    let query = match family {
        "path_specific" => CausalQuery::PathSpecific(
            PathSpecificEffectQuery::binary(
                v(q["treatment"].as_str().unwrap()),
                v(q["outcome"].as_str().unwrap()),
            )
            .with_path_nodes(
                q["path_nodes"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|n| v(n.as_str().unwrap()))
                    .collect::<Vec<_>>(),
            )
            .with_max_len(4),
        ),
        "distribution" => CausalQuery::Distribution(InterventionalDistributionQuery::new(
            v(q["outcome"].as_str().unwrap()),
            [Intervention::set(v("t"), Value::f64(q["do"]["t"].as_f64().unwrap()))],
        )),
        _ => CausalQuery::Mediation(MediationQuery::binary(
            v(q["treatment"].as_str().unwrap()),
            v(q["outcome"].as_str().unwrap()),
            q["mediators"]
                .as_array()
                .unwrap()
                .iter()
                .map(|n| v(n.as_str().unwrap()))
                .collect::<Vec<_>>(),
            MediationContrast::NaturalDirect,
        )),
    };
    let builder = Study::tabular(data.clone())
        .graph_posterior(posterior)
        .query(query)
        .inference(inference_of(cell.inference.as_str()))
        .refute(refute_of(cell.validation.as_str()))
        .bootstrap_replicates(0);
    Some(Ok(CellSetup { expected: cell_coordinate(cell), builder, data: CellData::Tabular(data) }))
}

/// The counterfactual and nested-effect cells on mixed, class and graph-posterior structures read
/// the enumerated count tables of their conformance fixtures: a table with few levels, and a
/// graph that describes exactly its columns.
fn cross_world_setup(cell: antecedent::SupportCell) -> Option<Result<CellSetup, String>> {
    let class = cell.graph_class.as_str();
    let counterfactual = cell.query.as_str() == "Counterfactual"
        && (matches!(class, "Admg" | "TemporalAdmg" | "Pag")
            && cell.structure.as_str() != "graph_posterior"
            || cell.structure.as_str() == "graph_posterior"
                && matches!(class, "Admg" | "TemporalAdmg" | "Pag" | "Cpdag" | "Dag"));
    let nested = cell.query.as_str() == "NestedCounterfactualEffect"
        && (matches!(class, "Admg" | "TemporalAdmg" | "Pag" | "Cpdag")
            || (cell.structure.as_str() == "graph_posterior" && class == "Dag"));
    if !(counterfactual || nested) {
        return None;
    }
    let pin: serde_json::Value = serde_json::from_str(include_str!(
        "../../../conformance/estimate/cross_world_class_truth/expected.json"
    ))
    .expect("class fixture");
    let table = |name: &str| {
        let spec = &pin["tables"][name];
        let names: Vec<String> = spec["columns"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| c.as_str().unwrap().to_owned())
            .collect();
        let mut data: Vec<Vec<f64>> = vec![Vec::new(); names.len()];
        for cell in spec["counts"].as_array().unwrap() {
            let levels: Vec<f64> =
                cell["levels"].as_array().unwrap().iter().map(|l| l.as_f64().unwrap()).collect();
            for _ in 0..cell["count"].as_u64().unwrap() {
                for (column, level) in data.iter_mut().zip(&levels) {
                    column.push(*level);
                }
            }
        }
        TabularData::from_f64_columns(
            names.iter().zip(&data).map(|(n, c)| (n.as_str(), c.as_slice())),
        )
        .unwrap()
    };
    let expected = cell_coordinate(cell);
    let inference = inference_of(cell.inference.as_str());
    let refute = refute_of(cell.validation.as_str());
    let accepted = cell.structure.as_str() == "accepted";
    let admg = |directed: &[(u32, u32)], bidirected: &[(u32, u32)]| {
        let mut g = Admg::with_variables(3);
        for &(a, b) in directed {
            g.insert_directed(nid(a), nid(b)).unwrap();
        }
        for &(a, b) in bidirected {
            g.insert_bidirected(nid(a), nid(b)).unwrap();
        }
        g
    };
    let counterfactual_query = |treatment: u32, outcome: u32| {
        CausalQuery::Counterfactual(CounterfactualQuery::new(
            vid(outcome),
            [Intervention::set(vid(treatment), Value::f64(1.0))],
        ))
    };
    let nested_query = |t: u32, m: u32, y: u32| {
        CausalQuery::NestedCounterfactual(
            NestedCounterfactualQuery::new(vid(t), vid(m), vid(y)).unwrap(),
        )
    };
    let static_pag = |n: u32, directed: &[(u32, u32)]| {
        let mut g = Pag::with_variables(n);
        for &(a, b) in directed {
            g.insert_directed(nid(a), nid(b)).unwrap();
        }
        g
    };
    let line_data = || {
        let x: Vec<f64> = (0..80).map(|i| f64::from(i) / 20.0 - 2.0).collect();
        let y: Vec<f64> = x.iter().map(|x| 1.0 + 2.0 * x).collect();
        TabularData::from_f64_columns([("x", x.as_slice()), ("y", y.as_slice())]).unwrap()
    };
    let gaussian_data = || {
        let z: Vec<f64> = (0..240_u32).map(|i| (f64::from(i) * 1.37).sin()).collect();
        let t: Vec<f64> =
            (0..240_u32).zip(&z).map(|(i, z)| 0.6 * z + (f64::from(i) * 0.91).cos()).collect();
        let y: Vec<f64> = (0..240_u32)
            .zip(t.iter().zip(&z))
            .map(|(i, (t, z))| 1.0 + 2.0 * t + z + 0.4 * (f64::from(i) * 2.3).sin())
            .collect();
        TabularData::from_f64_columns([
            ("z", z.as_slice()),
            ("t", t.as_slice()),
            ("y", y.as_slice()),
        ])
        .unwrap()
    };
    let (data, query) = match (cell.query.as_str(), class, cell.structure.as_str()) {
        ("Counterfactual", "Cpdag", "graph_posterior") => (line_data(), counterfactual_query(0, 1)),
        ("Counterfactual", "Dag", "graph_posterior") => (
            gaussian_data(),
            CausalQuery::Counterfactual(
                CounterfactualQuery::new(vid(2), [Intervention::set(vid(1), Value::f64(1.5))])
                    .with_control_level(0.0),
            ),
        ),
        ("Counterfactual", "Pag", _) => (table("side_path_chain"), counterfactual_query(1, 3)),
        ("Counterfactual", _, _) => (table("front_door"), counterfactual_query(0, 2)),
        ("NestedCounterfactualEffect", "Pag", _) => {
            (table("mediated_two_parents"), nested_query(2, 3, 4))
        }
        ("NestedCounterfactualEffect", "Cpdag", "graph_posterior") => {
            (table("front_door"), nested_query(0, 1, 2))
        }
        ("NestedCounterfactualEffect", "Cpdag", _) => {
            (table("two_mediators"), nested_query(0, 1, 3))
        }
        _ => (table("front_door"), nested_query(0, 1, 2)),
    };
    let builder = Study::tabular(data.clone())
        .query(query)
        .refute(refute)
        .inference(inference)
        .bootstrap_replicates(0);
    let a = [(0, 1), (0, 2), (1, 2)];
    let b = [(0, 2), (1, 2)];
    let builder = if cell.structure.as_str() == "graph_posterior" {
        let kind = match class {
            "Dag" => GraphPosteriorAtomKind::Dag,
            "Admg" => GraphPosteriorAtomKind::Admg,
            "TemporalAdmg" => GraphPosteriorAtomKind::TemporalAdmg,
            "Cpdag" => GraphPosteriorAtomKind::Cpdag,
            _ => GraphPosteriorAtomKind::Pag,
        };
        let edge_mask = |n: usize, edges: &[(usize, usize)]| {
            edges.iter().fold(0_u64, |m, &(x, z)| antecedent_discovery::set_edge(m, n, x, z, true))
        };
        let is_counterfactual = cell.query.as_str() == "Counterfactual";
        let (n, weights, masks): (usize, Vec<f64>, Vec<u64>) = match class {
            "Pag" if is_counterfactual => (
                4,
                vec![0.8, 0.2],
                vec![
                    edge_mask(4, &[(0, 1), (0, 2), (2, 3), (1, 3)]),
                    edge_mask(4, &[(1, 0), (0, 2), (2, 3), (1, 3)]),
                ],
            ),
            "Admg" | "TemporalAdmg" if is_counterfactual => (
                3,
                vec![0.5, 0.3, 0.2],
                vec![
                    antecedent_discovery::adjacency_mask_from_admg(&admg(
                        &[(0, 1), (1, 2)],
                        &[(0, 2)],
                    ))
                    .unwrap(),
                    antecedent_discovery::adjacency_mask_from_admg(&admg(&a, &[])).unwrap(),
                    edge_mask(3, &[(0, 1), (1, 2), (2, 0)]),
                ],
            ),
            "Cpdag" if is_counterfactual => {
                let mut line = Cpdag::with_variables(2);
                line.insert_undirected(nid(0), nid(1)).unwrap();
                (
                    2,
                    vec![1.0],
                    vec![antecedent_discovery::adjacency_mask_from_cpdag(&line).unwrap()],
                )
            }
            "Dag" if is_counterfactual => (
                3,
                vec![0.5, 0.3, 0.2],
                vec![
                    edge_mask(3, &a.map(|(x, z)| (x as usize, z as usize))),
                    edge_mask(3, &b.map(|(x, z)| (x as usize, z as usize))),
                    edge_mask(3, &[(0, 1), (1, 2), (2, 0)]),
                ],
            ),
            "Pag" => (
                5,
                vec![0.8, 0.2],
                vec![
                    edge_mask(5, &[(0, 2), (1, 3), (2, 3), (2, 4), (3, 4)]),
                    edge_mask(5, &[(2, 0), (1, 3), (2, 3), (2, 4), (3, 4)]),
                ],
            ),
            "Admg" | "TemporalAdmg" => (
                3,
                vec![0.5, 0.3, 0.2],
                vec![
                    antecedent_discovery::adjacency_mask_from_admg(&admg(&a, &[])).unwrap(),
                    antecedent_discovery::adjacency_mask_from_admg(&admg(&b, &[])).unwrap(),
                    edge_mask(3, &[(0, 1), (1, 2), (2, 0)]),
                ],
            ),
            "Cpdag" => {
                let mut da = Dag::with_variables(3);
                for (x, z) in a {
                    da.insert_directed(nid(x), nid(z)).unwrap();
                }
                let mut db = Dag::with_variables(3);
                for (x, z) in b {
                    db.insert_directed(nid(x), nid(z)).unwrap();
                }
                (
                    3,
                    vec![0.5, 0.3, 0.2],
                    vec![
                        antecedent_discovery::adjacency_mask_from_cpdag(&Cpdag::from_dag(&da))
                            .unwrap(),
                        antecedent_discovery::adjacency_mask_from_cpdag(&Cpdag::from_dag(&db))
                            .unwrap(),
                        edge_mask(3, &[(0, 1), (1, 2), (2, 0)]),
                    ],
                )
            }
            _ => (
                3,
                vec![0.5, 0.3, 0.2],
                vec![
                    edge_mask(3, &a.map(|(x, z)| (x as usize, z as usize))),
                    edge_mask(3, &b.map(|(x, z)| (x as usize, z as usize))),
                    edge_mask(3, &[(0, 1), (1, 2), (2, 0)]),
                ],
            ),
        };
        let gp = GraphPosterior::new(
            n,
            weights,
            masks,
            vec![0.0; n * n],
            vec![0.0; n * n],
            SamplingSummary {
                ess: 1.0,
                diagnostics: InferenceDiagnostics::analytic("cross_world_compiler"),
                rejected_invalid: 0,
            },
        )
        .unwrap()
        .with_atom_kind(kind);
        let gp = if class == "Pag" {
            let atoms = gp.n_graphs();
            gp.with_mark_masks(vec![0; atoms]).unwrap()
        } else {
            gp
        };
        builder.graph_posterior(gp)
    } else {
        match (class, cell.query.as_str()) {
            ("Admg", "Counterfactual") => {
                bind_graph(builder, admg(&[(0, 1), (1, 2)], &[(0, 2)]), accepted)
            }
            ("Admg", _) => bind_graph(builder, admg(&a, &[]), accepted),
            ("TemporalAdmg", "Counterfactual") => bind_graph(
                builder,
                TemporalAdmg::from_static_contemporaneous(&admg(&[(0, 1), (1, 2)], &[(0, 2)]))
                    .unwrap(),
                accepted,
            ),
            ("TemporalAdmg", _) => bind_graph(
                builder,
                TemporalAdmg::from_static_contemporaneous(&admg(&a, &[])).unwrap(),
                accepted,
            ),
            ("Pag", "Counterfactual") => {
                bind_graph(builder, static_pag(4, &[(0, 1), (0, 2), (2, 3), (1, 3)]), accepted)
            }
            ("Pag", _) => bind_graph(
                builder,
                static_pag(5, &[(0, 2), (1, 3), (2, 3), (2, 4), (3, 4)]),
                accepted,
            ),
            ("Cpdag", _) => {
                let mut g = Cpdag::with_variables(4);
                for (x, z) in [(0, 1), (0, 2), (1, 3), (2, 3)] {
                    g.insert_directed(nid(x), nid(z)).unwrap();
                }
                bind_graph(builder, g, accepted)
            }
            _ => bind_graph(
                builder,
                {
                    let mut d = Dag::with_variables(3);
                    for (x, z) in a {
                        d.insert_directed(nid(x), nid(z)).unwrap();
                    }
                    d
                },
                accepted,
            ),
        }
    };
    Some(Ok(CellSetup { expected, builder, data: CellData::Tabular(data) }))
}

fn bind_graph<G>(builder: StudyBuilder, graph: G, accepted: bool) -> StudyBuilder
where
    G: antecedent::IntoGraphInput + Into<AcceptedGraph>,
{
    if accepted { builder.graph(graph.into()) } else { builder.graph(graph) }
}

/// Cells that read the unrolled window of a stationary template (the temporal ADMG contrasts and
/// mediation, and the temporal PAG mediation) run on the exact-law series of the temporal window
/// fixture: the same series, templates and atoms the known-truth tests use.
#[allow(
    clippy::too_many_lines,
    reason = "one setup builds the series, the template or the posterior of every window cell"
)]
fn temporal_window_setup(cell: antecedent::SupportCell) -> Option<Result<CellSetup, String>> {
    use antecedent_discovery::{LagMask, TemporalAdmgAtom, with_lagged_edge};
    let class = cell.graph_class.as_str();
    let name = cell.query.as_str();
    let contrast = class == "TemporalAdmg"
        && matches!(
            name,
            "PulseEffect"
                | "SustainedEffect"
                | "TemporalConditionalPulse"
                | "TemporalConditionalSustained"
        );
    let mediation =
        name == "TemporalMediationEffect" && matches!(class, "TemporalAdmg" | "TemporalPag");
    if !(contrast || mediation) {
        return None;
    }
    let pin: serde_json::Value = serde_json::from_str(include_str!(
        "../../../conformance/estimate/temporal_window_truth/expected.json"
    ))
    .expect("temporal window fixture");
    let expected = cell_coordinate(cell);
    let series = |section: &str, names: &[&str]| -> TimeSeriesData {
        let states: Vec<u32> = pin[section]["series"]
            .as_str()
            .unwrap()
            .chars()
            .map(|c| "0123456789abcdefghijklmnopqrstuv".find(c).unwrap() as u32)
            .collect();
        let columns: Vec<Vec<f64>> = (0..names.len())
            .map(|bit| states.iter().map(|state| f64::from((state >> bit) & 1)).collect())
            .collect();
        TimeSeriesData::from_f64_columns(
            names.iter().zip(&columns).map(|(n, c)| (*n, c.as_slice())),
            1,
        )
        .unwrap()
    };
    let lag = Lag::from_raw(1);
    let now = Lag::CONTEMPORANEOUS;
    let lagged = |entries: &[(usize, usize)], n: usize| {
        entries.iter().fold(LagMask::EMPTY, |word, &(from, to)| {
            with_lagged_edge(word, n, 1, from, to, true).unwrap()
        })
    };
    let contemporaneous = |edges: &[(usize, usize)], n: usize| {
        edges
            .iter()
            .fold(AdjacencyMask::EMPTY, |mask, &(from, to)| set_edge(mask, n, from, to, true))
    };
    let posterior = |weights: Vec<f64>,
                     atoms: Vec<TemporalAdmgAtom>,
                     n: usize,
                     kind: GraphPosteriorAtomKind| {
        let gp = GraphPosterior::from_atoms(
            n,
            weights,
            atoms.iter().map(|atom| atom.contemporaneous.clone()),
            vec![0.0; n * n],
            vec![0.0; n * n],
            SamplingSummary {
                ess: 1.0,
                diagnostics: InferenceDiagnostics::analytic("licensed-compiler-temporal-window"),
                rejected_invalid: 0,
            },
        )
        .unwrap()
        .with_atom_kind(kind)
        .with_lagged_marginals(1, vec![0.0; n * n])
        .unwrap()
        .with_lag_masks(atoms.iter().map(|atom| atom.lagged_directed.clone()).collect::<Vec<_>>())
        .unwrap();
        if matches!(kind, GraphPosteriorAtomKind::TemporalAdmg) {
            gp.with_lag_bidirected_masks(
                atoms.iter().map(|atom| atom.lagged_bidirected.clone()).collect::<Vec<_>>(),
            )
            .unwrap()
        } else {
            gp
        }
    };
    let (data, causal_query) = if contrast {
        let base = TemporalEffectQuery::pulse(vid(0), vid(1), 1.0).with_horizon_steps(2);
        let query = match name {
            "PulseEffect" => base,
            "SustainedEffect" => base.with_policy(TemporalPolicy::sustained(0, 1)),
            "TemporalConditionalPulse" => base.with_effect_modifiers([vid(2)]),
            _ => base.with_policy(TemporalPolicy::sustained(0, 1)).with_effect_modifiers([vid(2)]),
        };
        (series("contrast", &["x", "y", "w"]), CausalQuery::TemporalEffect(query))
    } else {
        (
            series("mediation", &["c", "x", "w", "m", "y"]),
            CausalQuery::Mediation(MediationQuery::binary(
                vid(1),
                vid(4),
                [vid(3)],
                MediationContrast::NaturalIndirect,
            )),
        )
    };
    let builder = Study::series(data.clone())
        .query(causal_query)
        .refute(refute_of(cell.validation.as_str()))
        .inference(inference_of(cell.inference.as_str()))
        .bootstrap_replicates(0);
    let accepted = cell.structure.as_str() == "accepted";
    let builder = match cell.structure.as_str() {
        "graph_posterior" => {
            let gp = if contrast {
                let generating = TemporalAdmgAtom {
                    contemporaneous: contemporaneous(&[(0, 1), (2, 1)], 3),
                    lagged_directed: lagged(&[(1, 1)], 3),
                    lagged_bidirected: LagMask::EMPTY,
                };
                let no_effect = TemporalAdmgAtom {
                    contemporaneous: contemporaneous(&[(2, 1)], 3),
                    lagged_directed: lagged(&[(1, 1)], 3),
                    lagged_bidirected: LagMask::EMPTY,
                };
                let bow = TemporalAdmgAtom {
                    contemporaneous: contemporaneous(&[(0, 1), (2, 1)], 3),
                    lagged_directed: lagged(&[(1, 1), (0, 1)], 3),
                    lagged_bidirected: lagged(&[(0, 1)], 3),
                };
                posterior(
                    vec![0.5, 0.3, 0.2],
                    vec![generating, no_effect, bow],
                    3,
                    GraphPosteriorAtomKind::TemporalAdmg,
                )
            } else if class == "TemporalPag" {
                let edges = [(1, 3), (2, 3), (3, 4), (1, 4)];
                let visible = TemporalAdmgAtom {
                    contemporaneous: contemporaneous(&edges, 5),
                    lagged_directed: lagged(&[(0, 1)], 5),
                    lagged_bidirected: LagMask::EMPTY,
                };
                let invisible = TemporalAdmgAtom {
                    contemporaneous: contemporaneous(&edges, 5),
                    lagged_directed: LagMask::EMPTY,
                    lagged_bidirected: LagMask::EMPTY,
                };
                posterior(vec![0.7, 0.3], vec![visible, invisible], 5, GraphPosteriorAtomKind::Pag)
            } else {
                let edges = [(1, 3), (2, 3), (3, 4), (1, 4)];
                let generating = TemporalAdmgAtom {
                    contemporaneous: contemporaneous(&edges, 5),
                    lagged_directed: lagged(&[(0, 1)], 5),
                    lagged_bidirected: LagMask::EMPTY,
                };
                let inert = TemporalAdmgAtom {
                    contemporaneous: contemporaneous(&[(1, 3), (2, 3), (1, 4)], 5),
                    lagged_directed: lagged(&[(0, 1)], 5),
                    lagged_bidirected: LagMask::EMPTY,
                };
                let latent = TemporalAdmgAtom {
                    lagged_bidirected: lagged(&[(2, 3), (2, 4)], 5),
                    ..generating.clone()
                };
                posterior(
                    vec![0.5, 0.3, 0.2],
                    vec![generating, inert, latent],
                    5,
                    GraphPosteriorAtomKind::TemporalAdmg,
                )
            };
            builder.graph_posterior(gp)
        }
        "explicit" | "accepted" => {
            if contrast {
                let mut g = TemporalAdmg::empty();
                let y1 = g.add_lagged(vid(1), lag).unwrap();
                let x0 = g.add_lagged(vid(0), now).unwrap();
                let w0 = g.add_lagged(vid(2), now).unwrap();
                let y0 = g.add_lagged(vid(1), now).unwrap();
                for (from, to) in [(y1, y0), (w0, y0), (x0, y0)] {
                    g.insert_directed(from, to).unwrap();
                }
                bind_graph(builder, g, accepted)
            } else if class == "TemporalAdmg" {
                let mut g = TemporalAdmg::empty();
                let c1 = g.add_lagged(vid(0), lag).unwrap();
                let x0 = g.add_lagged(vid(1), now).unwrap();
                let w0 = g.add_lagged(vid(2), now).unwrap();
                let m0 = g.add_lagged(vid(3), now).unwrap();
                let y0 = g.add_lagged(vid(4), now).unwrap();
                for (from, to) in [(c1, x0), (x0, m0), (w0, m0), (m0, y0), (x0, y0)] {
                    g.insert_directed(from, to).unwrap();
                }
                bind_graph(builder, g, accepted)
            } else {
                let mut g = TemporalPag::empty();
                let c1 = g.add_lagged(vid(0), lag).unwrap();
                let x0 = g.add_lagged(vid(1), now).unwrap();
                let w0 = g.add_lagged(vid(2), now).unwrap();
                let m0 = g.add_lagged(vid(3), now).unwrap();
                let y0 = g.add_lagged(vid(4), now).unwrap();
                for (from, to) in [(c1, x0), (x0, m0), (w0, m0), (m0, y0), (x0, y0)] {
                    g.insert_directed(from, to).unwrap();
                }
                bind_graph(builder, g, accepted)
            }
        }
        other => return Some(Err(format!("{expected}: unhandled structure {other}"))),
    };
    Some(Ok(CellSetup { expected, builder, data: CellData::Series(data) }))
}

#[allow(
    clippy::too_many_lines,
    reason = "one arm per support-matrix cell maps the cell to its builder setup"
)]
fn cell_setup(cell: antecedent::SupportCell) -> Result<CellSetup, String> {
    if cell.graph_class.as_str() == "TemporalAdmg"
        && matches!(cell.structure.as_str(), "explicit" | "accepted")
    {
        let case = match cell.query.as_str() {
            "PointDerivative" => Some("point"),
            "Elasticity" => Some("elasticity"),
            "SemiElasticity" => Some("semi_treatment"),
            "AverageDerivative" => Some("average"),
            "DirectionalDerivative" => Some("directional"),
            "ResponseJacobian" => Some("jacobian"),
            _ => None,
        };
        if let Some(case) = case {
            let pin = temporal_admg_compiler_fixture::fixture();
            let at = temporal_admg_compiler_fixture::number(&pin, "at");
            let functional = temporal_admg_compiler_fixture::cases(&pin, at)
                .into_iter()
                .find(|(name, _, _)| *name == case)
                .expect("known temporal derivative fixture")
                .1;
            let data = temporal_admg_compiler_fixture::series();
            let builder = temporal_admg_compiler_fixture::study(
                &data,
                &functional,
                cell.structure.as_str() == "accepted",
                inference_of(cell.inference.as_str()),
                refute_of(cell.validation.as_str()),
            );
            return Ok(CellSetup {
                expected: cell_coordinate(cell),
                builder,
                data: CellData::Series(data),
            });
        }
    }
    if let Some(setup) = mixture_setup(cell) {
        return setup;
    }
    if let Some(setup) = cross_world_setup(cell) {
        return setup;
    }
    if let Some(setup) = temporal_window_setup(cell) {
        return setup;
    }
    let expected = cell_coordinate(cell);
    let query = query_for(cell);
    let n = n_vars(cell);
    let series = needs_series(cell);
    let inference = inference_of(cell.inference.as_str());
    let refute = refute_of(cell.validation.as_str());

    let (mut builder, data) = if uses_identified_temporal_pag(cell) {
        let pin = identified_pag_pin();
        let series = identified_pag_series(&pin);
        (Study::series(series.clone()), CellData::Series(series))
    } else if series {
        let names: Vec<&str> = if cell.graph_class.as_str() == "TemporalPag"
            && cell.structure.as_str() != "graph_posterior"
            && !uses_identified_temporal_pag(cell)
        {
            vec!["t", "m", "y", "r"]
        } else {
            match n {
                2 => vec!["x", "y"],
                3 => vec!["t", "m", "y"],
                _ => vec!["x", "y", "z", "w"][..n as usize].to_vec(),
            }
        };
        let series = if cell.query.as_str() == "TemporalMediationEffect" {
            temporal_mediation_series()
        } else {
            series_from(&names)
        };
        (Study::series(series.clone()), CellData::Series(series))
    } else {
        let table = if cell.query.as_str() == "NestedCounterfactualEffect" {
            let x: Vec<f64> = (0..300).map(|i| (f64::from(i) * 0.37).sin()).collect();
            let m: Vec<f64> =
                x.iter().enumerate().map(|(i, x)| 0.8 * x + (i as f64 * 0.91).cos()).collect();
            let y: Vec<f64> = x
                .iter()
                .zip(&m)
                .enumerate()
                .map(|(i, (x, m))| 1.7 * x + 4.0 * m + (i as f64 * 0.23).sin() * 0.01)
                .collect();
            TabularData::from_f64_columns([
                ("x", x.as_slice()),
                ("m", m.as_slice()),
                ("y", y.as_slice()),
            ])
            .unwrap()
        } else if uses_pag_response_curve(cell) {
            pag_response_curve_fixture().0
        } else if uses_pag_envelope(cell) {
            pag_envelope_fixture().0
        } else if cell.graph_class.as_str() == "CoDetermined"
            && cell.query.as_str() == "InterventionResponse"
        {
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
        } else if cell.graph_class.as_str() == "CoDetermined" {
            binary_treatment_table(&[("t", 0), ("y", 1), ("z", 2), ("u", 3)])
        } else if cell.graph_class.as_str() == "Unknown" {
            binary_treatment_table(&[("t", 0), ("y", 1), ("era", 2), ("m", 3)])
        } else if cell.query.as_str() == "TransportQuery" {
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
        } else if cell.query.as_str() == "InterferenceQuery" {
            static_table(&[("y", 0), ("x", 1)])
        } else if matches!(
            cell.query.as_str(),
            "MediationEffect" | "PathSpecificEffect" | "InterventionalDistribution"
        ) && cell.graph_class.as_str() == "Pag"
            && cell.structure.as_str() != "graph_posterior"
        {
            // The treatment and the mediator each need a parent for their edges to be visible.
            discrete_joint_table(&["t", "m", "y", "c", "w"])
        } else if cell.query.as_str() == "InterventionalDistribution"
            || (matches!(
                cell.query.as_str(),
                "AverageEffect" | "ResponseCurve" | "InterventionResponse"
            ) && matches!(cell.graph_class.as_str(), "Admg" | "TemporalAdmg"))
            || (cell.query.as_str() == "ConditionalEffect"
                && matches!(cell.graph_class.as_str(), "Admg" | "TemporalAdmg"))
        {
            if cell.query.as_str() == "ResponseCurve"
                && cell.structure.as_str() == "graph_posterior"
            {
                // A treatment independent of the outcome in every cell leaves a permuted
                // treatment's refit identical to the data: the placebo has no spread.
                dependent_joint_table(&["t", "y", "z"][..n.min(3) as usize])
            } else {
                discrete_joint_table(&["t", "y", "z"][..n.min(3) as usize])
            }
        } else if matches!(
            cell.query.as_str(),
            "AverageDerivative"
                | "PointDerivative"
                | "Elasticity"
                | "SemiElasticity"
                | "DirectionalDerivative"
                | "ResponseJacobian"
        ) {
            if cell.graph_class.as_str() == "Pag" && cell.structure.as_str() != "graph_posterior" {
                static_table(&[("a", 0), ("b", 1), ("y", 2), ("v", 3), ("r", 4)])
            } else {
                static_table(&[("a", 0), ("b", 1), ("y", 2), ("v", 3)])
            }
        } else if cell.query.as_str() == "MediationEffect"
            && cell.graph_class.as_str() == "Pag"
            && cell.structure.as_str() != "graph_posterior"
        {
            // The treatment and the mediator each need a parent for their edges to be visible.
            discrete_joint_table(&["t", "m", "y", "c", "w"])
        } else if matches!(cell.query.as_str(), "MediationEffect" | "PathSpecificEffect") {
            discrete_joint_table(&["t", "m", "y"])
        } else if matches!(
            cell.query.as_str(),
            "AverageEffect" | "ConditionalEffect" | "MediationEffect"
        ) {
            binary_treatment_table(&[("t", 0), ("y", 1), ("z", 2)])
        } else {
            static_table(&[("t", 0), ("y", 1), ("z", 2)])
        };
        (Study::tabular(table.clone()), CellData::Tabular(table))
    };

    builder = builder.query(query).refute(refute).inference(inference).bootstrap_replicates(0);

    builder = match cell.structure.as_str() {
        "graph_posterior" => {
            let gp = if series {
                if cell.query.as_str() == "TemporalMediationEffect"
                    && matches!(cell.graph_class.as_str(), "TemporalCpdag" | "TemporalPag")
                {
                    temporal_gp(n as usize)
                } else if cell.graph_class.as_str() == "TemporalPag" && is_attribution(cell) {
                    attribution_temporal_pag_gp(n as usize)
                } else if matches!(cell.graph_class.as_str(), "TemporalCpdag" | "TemporalPag") {
                    if point_family_query(cell) {
                        point_family_gp(n as usize, cell.graph_class.as_str() == "TemporalPag")
                    } else {
                        temporal_class_gp(n as usize)
                    }
                } else if cell.graph_class.as_str() == "TemporalDag"
                    && matches!(
                        cell.query.as_str(),
                        "AverageDerivative"
                            | "PointDerivative"
                            | "Elasticity"
                            | "SemiElasticity"
                            | "DirectionalDerivative"
                            | "ResponseJacobian"
                    )
                {
                    point_family_gp(n as usize, false)
                } else {
                    temporal_gp(n as usize)
                }
            } else if cell.query.as_str() == "ConditionalEffect"
                && matches!(cell.graph_class.as_str(), "Admg" | "TemporalAdmg")
            {
                conditional_admg_gp()
            } else if cell.query.as_str() == "ResponseCurve"
                && matches!(cell.graph_class.as_str(), "Admg" | "TemporalAdmg")
                && cell.validation.as_str() != "none"
            {
                frontdoor_curve_gp()
            } else {
                static_gp(n as usize)
            };
            let gp = match cell.graph_class.as_str() {
                "Cpdag" | "TemporalCpdag" => gp.with_atom_kind(GraphPosteriorAtomKind::Cpdag),
                "Pag" | "TemporalPag" => gp.with_atom_kind(GraphPosteriorAtomKind::Pag),
                "Admg" => gp.with_atom_kind(GraphPosteriorAtomKind::Admg),
                "TemporalAdmg" => gp.with_atom_kind(GraphPosteriorAtomKind::TemporalAdmg),
                _ => gp,
            };
            builder.graph_posterior(gp)
        }
        "explicit" | "accepted" => {
            let accepted = cell.structure.as_str() == "accepted";
            match cell.graph_class.as_str() {
                "Dag" => {
                    let g = dag_for(cell, n);
                    if accepted { builder.graph(AcceptedGraph::from(g)) } else { builder.graph(g) }
                }
                "Admg" => {
                    let g = match cell.query.as_str() {
                        "TransportQuery" => transport_admg(),
                        "InterferenceQuery" => Admg::with_variables(2),
                        "MediationEffect" | "PathSpecificEffect" | "InterventionalDistribution" => {
                            mediation_admg()
                        }
                        _ if point_family_query(cell) => derivative_admg(),
                        _ => confounder_admg(),
                    };
                    if accepted { builder.graph(AcceptedGraph::from(g)) } else { builder.graph(g) }
                }
                "TemporalAdmg" => {
                    let g = TemporalAdmg::from_static_contemporaneous(&match cell.query.as_str() {
                        "TransportQuery" => transport_admg(),
                        "InterferenceQuery" => Admg::with_variables(2),
                        "MediationEffect" | "PathSpecificEffect" | "InterventionalDistribution" => {
                            mediation_admg()
                        }
                        "AverageDerivative"
                        | "PointDerivative"
                        | "Elasticity"
                        | "SemiElasticity"
                        | "DirectionalDerivative"
                        | "ResponseJacobian" => derivative_admg(),
                        _ => confounder_admg(),
                    })
                    .map_err(|e| format!("{expected}: {e}"))?;
                    if accepted { builder.graph(AcceptedGraph::from(g)) } else { builder.graph(g) }
                }
                "Cpdag" => {
                    let g = Cpdag::from_dag(&dag_for(cell, n));
                    if accepted { builder.graph(AcceptedGraph::from(g)) } else { builder.graph(g) }
                }
                "Pag" => {
                    let g = if matches!(
                        cell.query.as_str(),
                        "MediationEffect" | "PathSpecificEffect" | "InterventionalDistribution"
                    ) && cell.structure.as_str() != "graph_posterior"
                    {
                        visible_mediation_static_pag()
                    } else if uses_pag_response_curve(cell) {
                        pag_response_curve_fixture().1
                    } else if uses_pag_envelope(cell) {
                        pag_envelope_fixture().1
                    } else if cell.structure.as_str() != "graph_posterior"
                        && matches!(
                            cell.query.as_str(),
                            "AverageDerivative"
                                | "PointDerivative"
                                | "Elasticity"
                                | "SemiElasticity"
                                | "DirectionalDerivative"
                                | "ResponseJacobian"
                        )
                    {
                        derivative_pag()
                    } else if is_attribution(cell) && cell.structure.as_str() != "graph_posterior" {
                        attribution_pag(n)
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
                    let g = if is_attribution(cell) {
                        attribution_temporal_pag()
                    } else if uses_identified_temporal_pag(cell) {
                        identified_pag(&identified_pag_pin())
                    } else {
                        visible_mediation_pag()
                    };
                    if accepted { builder.graph(AcceptedGraph::from(g)) } else { builder.graph(g) }
                }
                "CoDetermined" => {
                    let background = if cell.query.as_str() == "InterventionResponse" {
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

    if cell.query.as_str() == "PathSpecificEffect"
        && !matches!(cell.graph_class.as_str(), "TemporalDag" | "TemporalCpdag" | "TemporalPag")
    {
        builder = builder
            .identifier(IdentifierId::PathSpecificNatural)
            .estimator(EstimatorId::FunctionalEffect);
    }
    if cell.query.as_str() == "InterventionalDistribution"
        && !matches!(cell.graph_class.as_str(), "TemporalDag" | "TemporalCpdag" | "TemporalPag")
    {
        builder = builder
            .identifier(IdentifierId::GeneralId)
            .estimator(EstimatorId::FunctionalDistribution);
    }
    if (matches!(cell.query.as_str(), "AverageEffect" | "ResponseCurve" | "InterventionResponse")
        && matches!(cell.graph_class.as_str(), "Admg" | "TemporalAdmg"))
        || (cell.query.as_str() == "ConditionalEffect" && cell.graph_class.as_str() == "Admg")
    {
        builder =
            builder.identifier(IdentifierId::GeneralId).estimator(EstimatorId::FunctionalEffect);
    }
    if matches!(
        cell.query.as_str(),
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

    if cell.query.as_str() == "TransportQuery" {
        builder = builder.selection_targets(Arc::from([])).transport_trial(TransportTrialSpec {
            trial: vid(2),
            selection_probability: vid(3),
            treatment_probability: vid(4),
        });
        if cell.graph_class.as_str() == "TemporalAdmg" {
            builder = builder.estimator(if cell.inference.as_str() == "Bayesian" {
                EstimatorId::TransportTrialBayesianBootstrap
            } else {
                EstimatorId::TransportTrialIpw
            });
        }
    }
    if cell.query.as_str() == "InterferenceQuery" {
        let units = static_table(&[("y", 0)]);
        let n_units = 48usize;
        let edges: Vec<NetworkEdge> = (0..n_units)
            .map(|i| {
                NetworkEdge::new(
                    u64::try_from(i).expect("row id"),
                    u64::try_from((i + 1) % n_units).expect("row id"),
                    1.0,
                )
            })
            .collect();
        let network = NetworkData::try_new(units, edges).map_err(|e| format!("{expected}: {e}"))?;
        // The repeated 1100 pattern supplies all own/neighbor exposure pairs.
        // Alternation makes the two regressors perfectly collinear.
        let assignment: Arc<[bool]> = (0..n_units).map(|i| i % 4 < 2).collect();
        builder = builder.interference(InterferenceSpec { network, assignment });
    }

    Ok(CellSetup { expected, builder, data })
}

/// Optional estimators often need a different, but still same-coordinate,
/// design than the shared support fixture. Keep the registry cell unchanged
/// while supplying the data and identifier the named estimator expects.
#[allow(
    clippy::too_many_lines,
    clippy::match_same_arms,
    reason = "one arm per optional-estimator cell maps the cell to its builder setup, in matrix order"
)]
fn optional_estimator_setup(
    cell: antecedent::SupportCell,
    estimator: EstimatorId,
) -> Result<CellSetup, String> {
    let name = estimator.as_str();
    let custom = match name {
        "frontdoor.linear_two_stage" => Some("frontdoor"),
        "iv.2sls" | "iv.wald" | "iv.bayesian_joint_linear" => Some("iv"),
        "rd.sharp" | "rd.bayesian_local_linear" => Some("rd"),
        "glm.adjustment" => Some("glm"),
        "cell.aipw" if cell.query.as_str() == "InterventionResponse" => Some("cell_aipw"),
        _ => None,
    };
    let Some(kind) = custom else {
        let mut setup = cell_setup(cell)?;
        setup.builder = setup.builder.estimator(estimator);
        // The optional registry alternatives are routed through their named
        // identifier families; the default shared fixture otherwise infers
        // the primary compiler route and silently rewrites several choices.
        let identifier = match (cell.query.as_str(), cell.graph_class.as_str(), name) {
            ("AverageEffect", "CoDetermined", "cell.aipw") => return Ok(setup),
            ("InterventionResponse", _, "cell.aipw") => IdentifierId::ResponseBackdoor,
            ("ConditionalEffect", _, "bayesian.basis.gcomp") => IdentifierId::BackdoorAdjustment,
            (
                "AverageEffect",
                _,
                "bayesian.basis.gcomp" | "bayesian.robust_ate" | "bayesian.gcomp",
            ) => IdentifierId::BackdoorAdjustment,
            (
                "AverageEffect",
                _,
                "aipw" | "propensity.matching" | "propensity.weighting" | "glm.adjustment",
            ) => IdentifierId::BackdoorAdjustment,
            _ => return Ok(setup),
        };
        setup.builder = setup.builder.identifier(identifier);
        return Ok(setup);
    };

    let expected = cell_coordinate(cell);
    let accepted = cell.structure.as_str() == "accepted";
    let n = 120usize;
    let (data, query, graph, identifier, rd) = match kind {
        "frontdoor" => {
            let t: Vec<f64> = (0..n).map(|i| (i % 2) as f64).collect();
            let m: Vec<f64> = (0..n).map(|i| 0.3 * t[i] + ((i * 7 % 13) as f64) / 13.0).collect();
            let y: Vec<f64> = (0..n).map(|i| 0.7 * m[i] + ((i * 11 % 17) as f64) / 17.0).collect();
            let data = TabularData::from_f64_columns([
                ("t", t.as_slice()),
                ("m", m.as_slice()),
                ("y", y.as_slice()),
            ])
            .map_err(|e| e.to_string())?;
            let mut graph = Dag::with_variables(3);
            graph.insert_directed(nid(0), nid(1)).unwrap();
            graph.insert_directed(nid(1), nid(2)).unwrap();
            (
                data,
                CausalQuery::AverageEffect(AverageEffectQuery::binary_ate(vid(0), vid(2))),
                graph,
                IdentifierId::Frontdoor,
                None,
            )
        }
        "iv" => {
            let z: Vec<f64> = (0..n).map(|i| (i % 2) as f64).collect();
            let t: Vec<f64> = (0..n).map(|i| f64::from(z[i] > 0.5 && i % 4 != 1)).collect();
            let y: Vec<f64> = (0..n).map(|i| 0.5 * t[i] + ((i * 7 % 19) as f64) / 19.0).collect();
            let data = TabularData::from_f64_columns([
                ("t", t.as_slice()),
                ("y", y.as_slice()),
                ("z", z.as_slice()),
            ])
            .map_err(|e| e.to_string())?;
            let mut graph = Dag::with_variables(3);
            graph.insert_directed(nid(2), nid(0)).unwrap();
            graph.insert_directed(nid(0), nid(1)).unwrap();
            (
                data,
                CausalQuery::AverageEffect(AverageEffectQuery::binary_ate(vid(0), vid(1))),
                graph,
                IdentifierId::Iv,
                None,
            )
        }
        "rd" => {
            let r: Vec<f64> = (0..n).map(|i| (i as f64 - 60.0) / 30.0).collect();
            let t: Vec<f64> = r.iter().map(|x| f64::from(*x >= 0.0)).collect();
            let y: Vec<f64> =
                (0..n).map(|i| 1.0 + 1.5 * t[i] + 0.4 * r[i] + ((i % 7) as f64) * 0.01).collect();
            let data = TabularData::from_f64_columns([
                ("t", t.as_slice()),
                ("y", y.as_slice()),
                ("r", r.as_slice()),
            ])
            .map_err(|e| e.to_string())?;
            let mut graph = Dag::with_variables(3);
            graph.insert_directed(nid(2), nid(0)).unwrap();
            graph.insert_directed(nid(0), nid(1)).unwrap();
            graph.insert_directed(nid(2), nid(1)).unwrap();
            (
                data,
                CausalQuery::AverageEffect(AverageEffectQuery::binary_ate(vid(0), vid(1))),
                graph,
                IdentifierId::RdSharp,
                Some((vid(2), 0.0, 1.5)),
            )
        }
        "glm" => {
            let t: Vec<f64> = (0..n).map(|i| (i % 2) as f64).collect();
            let z: Vec<f64> = (0..n).map(|i| ((i / 2) % 2) as f64).collect();
            let y: Vec<f64> =
                (0..n).map(|i| f64::from(u8::from((i % 5 < 2) ^ (t[i] > 0.5)))).collect();
            let data = TabularData::from_f64_columns([
                ("t", t.as_slice()),
                ("y", y.as_slice()),
                ("z", z.as_slice()),
            ])
            .map_err(|e| e.to_string())?;
            (
                data,
                CausalQuery::AverageEffect(AverageEffectQuery::binary_ate(vid(0), vid(1))),
                dag_for(cell, 3),
                IdentifierId::BackdoorAdjustment,
                None,
            )
        }
        "cell_aipw" => {
            let t1: Vec<f64> = (0..n).map(|i| (i % 2) as f64).collect();
            let t2: Vec<f64> = (0..n).map(|i| ((i / 2) % 2) as f64).collect();
            let z: Vec<f64> = (0..n).map(|i| ((i / 4) % 2) as f64).collect();
            let y: Vec<f64> = (0..n)
                .map(|i| 0.4 * t1[i] + 0.3 * t2[i] + 0.2 * z[i] + ((i * 7 % 13) as f64) * 0.01)
                .collect();
            let data = TabularData::from_f64_columns([
                ("t1", t1.as_slice()),
                ("t2", t2.as_slice()),
                ("y", y.as_slice()),
                ("z", z.as_slice()),
            ])
            .map_err(|e| e.to_string())?;
            let query = intervention_response(&[0, 1], 2, false);
            let mut graph = Dag::with_variables(4);
            graph.insert_directed(nid(3), nid(0)).unwrap();
            graph.insert_directed(nid(3), nid(1)).unwrap();
            graph.insert_directed(nid(3), nid(2)).unwrap();
            graph.insert_directed(nid(0), nid(2)).unwrap();
            graph.insert_directed(nid(1), nid(2)).unwrap();
            (data, query, graph, IdentifierId::ResponseBackdoor, None)
        }
        _ => unreachable!(),
    };
    let mut builder = Study::tabular(data.clone())
        .query(query)
        .inference(inference_of(cell.inference.as_str()))
        .refute(refute_of(cell.validation.as_str()))
        .bootstrap_replicates(0)
        .identifier(identifier)
        .estimator(estimator);
    builder =
        if accepted { builder.graph(AcceptedGraph::from(graph)) } else { builder.graph(graph) };
    if let Some((running, cutoff, bandwidth)) = rd {
        builder = builder.rd_config(running, cutoff, bandwidth);
    }
    Ok(CellSetup { expected, builder, data: CellData::Tabular(data) })
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
    inspect_setup(&cell_setup(cell)?)
}

/// The identifier and estimator a licensed cell's execution ran, as recorded on its
/// logical plan (`-` when the plan names none).
struct Route {
    coordinate: String,
    identifier: String,
    estimator: String,
    lowering: String,
}

fn consume_setup(setup: CellSetup) -> Result<Route, String> {
    let expected = setup.expected.clone();
    inspect_setup(&setup)?;
    let ctx = ExecutionContext::for_tests(1);
    let built = setup.builder.build().map_err(|e| format!("{expected}: build {e}"))?;
    let prepared = built.prepare(&ctx).map_err(|e| format!("{expected}: prepare {e}"))?;
    let lowered = prepared
        .plan()
        .lowered()
        .ok_or_else(|| format!("{expected}: compile attached no lowered program"))?;
    let lowering = lowered.route().to_owned();
    if lowered.depth() != antecedent::planner::lowered::LoweringDepth::Complete {
        return Err(format!("{expected}: {lowering} did not lower every layer"));
    }
    let lowered_identity = lowered.identity();
    let view = prepared.plan_view().map_err(|e| format!("{expected}: plan view {e}"))?;
    if view.identity() != lowered_identity || view.route().as_str() != lowering {
        return Err(format!("{expected}: the plan view does not read the lowered program"));
    }
    if view.stage() != antecedent::planner::PlanStage::Prepared || view.operation().is_none() {
        return Err(format!("{expected}: the prepared plan view names no retained operation"));
    }
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
    if result.lowered_program_identity() != Some(lowered_identity) {
        return Err(format!("{expected}: execution did not dispatch the plan's lowered program"));
    }
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
    if claim.kind == antecedent_contract::ClaimKind::Mixture {
        return Err(format!("{expected}: partial claim carries no identified set"));
    }
    let bytes = prepared
        .encode_contracted_result(&result, &expected, &ctx)
        .map_err(|e| format!("{expected}: encode {e}"))?;
    let consumed =
        consume_analysis_result(&bytes).map_err(|e| format!("{expected}: consume {e}"))?;
    if !consumed.acceptance.accepts_as_verified_program()
        && !sealed_checked_operation_dependency_only(&consumed)
        && !functional_effect_replay_dependency_only(&consumed)
        && !linear_replay_dependency_only(&consumed)
        && !bayesian_distribution_replay_dependency_only(&consumed)
        && !class_distribution_replay_dependency_only(&consumed)
        && !graph_posterior_distribution_replay_dependency_only(&consumed)
    {
        return Err(format!(
            "{expected}: consume did not accept a verified program: {:?}",
            consumed.acceptance.unresolved
        ));
    }
    let plan = &result.logical_plan;
    Ok(Route {
        coordinate: expected,
        identifier: plan.identifier.as_deref().unwrap_or("-").to_owned(),
        estimator: plan.estimator.as_deref().unwrap_or("-").to_owned(),
        lowering,
    })
}

const ROUTES: &str = "parity/licensed_routes.toml";

/// `parity/licensed_routes.toml` as the executions produce it.
fn render_routes(routes: &[Route]) -> String {
    let mut out = String::from(concat!(
        "# Generated by crates/antecedent/tests/licensed_compiler.rs\n",
        "# (every_licensed_cell_completes_the_compiler_path): the identifier and estimator\n",
        "# each licensed cell's execution recorded on its logical plan, and the route cell\n",
        "# its lowered program was compiled for. The test fails when\n",
        "# this file drifts; regenerate with\n",
        "#   UPDATE_LICENSED_ROUTES=1 cargo test -p antecedent --test licensed_compiler \\\n",
        "#     every_licensed_cell_completes_the_compiler_path\n",
        "# scripts/generate_support_matrix_docs.py joins these component ids to the parity\n",
        "# rows whose external oracles exercise them.\n",
    ));
    for route in routes {
        out.push_str(&format!(
            "\n[[route]]\ncoordinate = \"{}\"\nidentifier = \"{}\"\nestimator = \"{}\"\nlowering = \"{}\"\n",
            route.coordinate, route.identifier, route.estimator, route.lowering
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
    assert_eq!(n, 2696, "licensed inventory drifted");
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
        queries.insert(cell.query.as_str());
    }
    for query in queries {
        let cell =
            antecedent::SupportCell::from_labels(query, "Dag", "explicit", "Frequentist", "none")
                .expect("fixture support axis");
        let _ = dag_for(cell, 4);
    }
}

/// `cases.iter().map(run)` on scoped threads, one per available core, results in input order.
/// The first panic is re-raised with its original payload.
fn parallel_map<T: Sync, R: Send>(cases: &[T], run: impl Fn(&T) -> R + Sync) -> Vec<R> {
    let next = std::sync::atomic::AtomicUsize::new(0);
    let slots: Vec<std::sync::Mutex<Option<R>>> =
        cases.iter().map(|_| std::sync::Mutex::new(None)).collect();
    let workers = std::thread::available_parallelism().map_or(1, usize::from).min(cases.len());
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..workers)
            .map(|_| {
                scope.spawn(|| {
                    loop {
                        let index = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        let Some(case) = cases.get(index) else { break };
                        *slots[index].lock().unwrap() = Some(run(case));
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
    slots.into_iter().map(|slot| slot.into_inner().unwrap().expect("every case ran")).collect()
}

#[test]
fn every_licensed_cell_completes_the_compiler_path() {
    let mut failures = Vec::new();
    let mut routes = Vec::new();
    // The cells are independent executions: each builds its own setup and consumes it. They run
    // on scoped threads, and the outcomes are read back in inventory order, so the rendered
    // route file and the failure list are exactly what a sequential loop produces.
    let cells: Vec<_> = licensed_support_cells().collect();
    let n = cells.len();
    for outcome in parallel_map(&cells, |&cell| cell_setup(cell).and_then(consume_setup)) {
        match outcome {
            Ok(route) => routes.push(route),
            Err(err) => failures.push(err),
        }
    }
    if let Ok(scratch) = std::env::var("SCRATCH_ROUTES") {
        std::fs::write(scratch, render_routes(&routes)).unwrap();
    }
    assert_eq!(n, 2696, "licensed inventory drifted");
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

/// Diagnostic sweep for retained checked execution after every source owner is
/// dropped. This stays opt-in because the full 669-cell execution is expensive.
#[test]
#[ignore = "opt-in registry closure diagnostic; run with --ignored --exact"]
#[allow(
    clippy::too_many_lines,
    reason = "one scenario kept in one body: the fixture, the run and the pinned assertions for diagnostic registry closure after builder and study drop"
)]
fn diagnostic_registry_closure_after_builder_and_study_drop() {
    let mut count = 0usize;
    let mut plan_gaps = Vec::new();
    let mut lifecycle_gaps = Vec::new();
    let mut not_inspectable = Vec::new();
    let mut dependency_refusals = Vec::new();

    for cell in licensed_support_cells() {
        count += 1;
        let coordinate = cell_coordinate(cell);
        let probe = (|| -> Result<(String, String), String> {
            let setup = cell_setup(cell)?;
            let CellSetup { expected: _, builder, data } = setup;
            let ctx = ExecutionContext::for_tests(1);
            let study = builder.clone().build().map_err(|e| format!("build: {e}"))?;
            let mut prepared = study.prepare(&ctx).map_err(|e| format!("prepare: {e}"))?;
            let estimator =
                prepared.plan().logical.record.estimator.as_deref().unwrap_or("-").to_owned();
            let checked_status = match estimator.as_str() {
                "functional.distribution" => {
                    Some(("distribution", retains(&prepared, "distribution")))
                }
                "linear.adjustment.ate" => {
                    Some(("linear_adjustment", retains(&prepared, "linear_adjustment")))
                }
                "aipw" => Some(("aipw", retains(&prepared, "aipw"))),
                "bayesian.gcomp" => Some(("bayesian_gcomp", retains(&prepared, "bayesian_gcomp"))),
                _ => None,
            };
            if let Some((inspector, present)) = checked_status {
                if !present {
                    plan_gaps
                        .push(format!("{coordinate}: {inspector} absent (estimator={estimator})"));
                }
            } else {
                not_inspectable.push(format!(
                    "{coordinate}: no retained checked operation (estimator={estimator})"
                ));
            }

            drop(builder);
            drop(study);

            let result = match &data {
                CellData::Tabular(table) => prepared.estimate(table, &ctx),
                CellData::Series(series) => prepared.estimate_series(series, &ctx),
            }
            .map_err(|e| format!("execute after drop: {e}"))?;
            let refreshed = match &data {
                CellData::Tabular(table) => rebind_estimate(&mut prepared, table.clone(), &ctx),
                CellData::Series(series) => prepared.refresh_series(series.clone(), &ctx),
            }
            .map_err(|e| format!("compatible refresh after drop: {e}"))?;
            let bytes = prepared
                .encode_contracted_result(&refreshed, &format!("closure-{coordinate}"), &ctx)
                .map_err(|e| format!("export after refresh: {e}"))?;
            let consumed = consume_analysis_result(&bytes).map_err(|e| format!("consume: {e}"))?;
            let missing_distribution_laws = consumed
                .acceptance
                .unresolved
                .iter()
                .any(|item| item.as_ref() == "dependencies.distribution_factor_laws");
            if estimator == "functional.distribution" && missing_distribution_laws {
                dependency_refusals
                    .push(format!("{coordinate}: dependencies.distribution_factor_laws"));
            } else if sealed_checked_operation_dependency_only(&consumed)
                || functional_effect_replay_dependency_only(&consumed)
            {
                dependency_refusals
                    .push(format!("{coordinate}: {:?}", consumed.acceptance.unresolved));
            } else if linear_replay_dependency_only(&consumed) {
                dependency_refusals
                    .push(format!("{coordinate}: dependencies.linear_fit_sufficient_statistics"));
            } else if bayesian_distribution_replay_dependency_only(&consumed) {
                dependency_refusals.push(format!(
                    "{coordinate}: dependencies.distribution_posterior_factor_draws"
                ));
            } else if !consumed.acceptance.accepts_as_verified_program() {
                return Err(format!(
                    "consumer refused verified program: {:?}",
                    consumed.acceptance.unresolved
                ));
            }
            let _ = result;
            Ok((estimator, "execute/refresh/export/consume passed".into()))
        })();

        if let Err(error) = probe {
            lifecycle_gaps.push(format!("{coordinate}: {error}"));
        }
    }

    println!("registry closure diagnostic inspected {count} support cells");
    println!("checked-plan gaps ({}):\n{}", plan_gaps.len(), plan_gaps.join("\n"));
    println!("lifecycle gaps ({}):\n{}", lifecycle_gaps.len(), lifecycle_gaps.join("\n"));
    println!(
        "known distribution dependency refusals ({}):\n{}",
        dependency_refusals.len(),
        dependency_refusals.join("\n")
    );
    println!(
        "coordinates without public checked-plan inspectors ({}):\n{}",
        not_inspectable.len(),
        not_inspectable.join("\n")
    );
    assert_eq!(count, 2696, "licensed support inventory drifted");
}

/// Probe for the separately licensed alternate estimators. It keeps
/// every exact `base::estimator=...` coordinate visible and classifies public
/// plan inspection separately from execution and artifact acceptance.
#[test]
#[allow(
    clippy::too_many_lines,
    reason = "one scenario kept in one body: the fixture, the run and the pinned assertions for diagnostic optional estimator closure after builder and study drop"
)]
fn diagnostic_optional_estimator_closure_after_builder_and_study_drop() {
    let mut routes = Vec::new();
    for cell in licensed_support_cells() {
        let Some(selected) = licensed_route_estimator(cell) else {
            continue;
        };
        for wire in licensed_estimators(cell) {
            let estimator = wire.parse::<EstimatorId>().unwrap_or_else(|e| {
                panic!("{}: invalid licensed estimator {wire}: {e}", cell_coordinate(cell))
            });
            if estimator != selected {
                routes.push((cell, estimator));
            }
        }
    }
    routes.sort_by_key(|(cell, estimator)| {
        format!("{}::estimator={}", cell_coordinate(*cell), estimator.as_str())
    });

    let mut checked_plan_gaps = Vec::new();
    let mut uninspectable = Vec::new();
    let mut execution_gaps = Vec::new();
    let mut selection_gaps = Vec::new();
    let mut refresh_gaps = Vec::new();
    let mut dependency_refusals = Vec::new();
    let mut inspected = 0usize;

    for (cell, estimator) in routes {
        inspected += 1;
        let base = cell_coordinate(cell);
        let coordinate = format!("{base}::estimator={}", estimator.as_str());
        if classify_estimator(cell, estimator) != CellStatus::Licensed {
            execution_gaps
                .push(format!("{coordinate}: concrete estimator no longer licenses on base cell"));
            continue;
        }

        let probe = (|| -> Result<(), String> {
            let CellSetup { builder, data, .. } = optional_estimator_setup(cell, estimator)?;
            let ctx = ExecutionContext::for_tests(1);
            let study = builder.clone().build().map_err(|e| format!("build: {e}"))?;
            let mut prepared = study.prepare(&ctx).map_err(|e| format!("prepare: {e}"))?;
            let retained = prepared.plan().logical.record.estimator.as_deref();
            if retained != Some(estimator.as_str()) {
                return Err(format!("selected estimator drifted at prepare: {retained:?}"));
            }

            let checked_status = match estimator.as_str() {
                "functional.distribution" => {
                    Some(("distribution", retains(&prepared, "distribution")))
                }
                "linear.adjustment.ate" => {
                    Some(("linear_adjustment", retains(&prepared, "linear_adjustment")))
                }
                "aipw" => Some(("aipw", retains(&prepared, "aipw"))),
                "frontdoor.linear_two_stage" => {
                    Some(("frontdoor_linear", retains(&prepared, "frontdoor_linear")))
                }
                "iv.2sls" | "iv.wald" => Some(("iv", retains(&prepared, "iv"))),
                "bayesian.gcomp" => Some(("bayesian_gcomp", retains(&prepared, "bayesian_gcomp"))),
                _ => None,
            };
            match checked_status {
                Some((_, true)) => {}
                Some((inspector, false)) => {
                    checked_plan_gaps.push(format!("{coordinate}: {inspector} absent"))
                }
                None => uninspectable.push(format!("{coordinate}: no retained checked operation")),
            }

            drop(builder);
            drop(study);
            let result = match &data {
                CellData::Tabular(table) => prepared.estimate(table, &ctx),
                CellData::Series(series) => prepared.estimate_series(series, &ctx),
            }
            .map_err(|e| format!("execute after builder/Study drop: {e}"))?;
            let refreshed = match &data {
                CellData::Tabular(table) => rebind_estimate(&mut prepared, table.clone(), &ctx),
                CellData::Series(series) => prepared.refresh_series(series.clone(), &ctx),
            }
            .map_err(|e| format!("compatible refresh: {e}"))?;
            if result.effect().to_bits() != refreshed.effect().to_bits() {
                return Err(format!(
                    "identical-data refresh changed effect {} -> {}",
                    result.effect(),
                    refreshed.effect()
                ));
            }
            let bytes = prepared
                .encode_contracted_result(&refreshed, &coordinate, &ctx)
                .map_err(|e| format!("export: {e}"))?;
            let consumed = consume_analysis_result(&bytes).map_err(|e| format!("consume: {e}"))?;
            if !consumed.acceptance.accepts_as_verified_program() {
                if consumed
                    .acceptance
                    .unresolved
                    .iter()
                    .any(|item| item.as_ref() == "dependencies.distribution_factor_laws")
                {
                    dependency_refusals
                        .push(format!("{coordinate}: dependencies.distribution_factor_laws"));
                } else if sealed_checked_operation_dependency_only(&consumed)
                    || functional_effect_replay_dependency_only(&consumed)
                {
                    dependency_refusals
                        .push(format!("{coordinate}: {:?}", consumed.acceptance.unresolved));
                } else if linear_replay_dependency_only(&consumed) {
                    dependency_refusals.push(format!(
                        "{coordinate}: dependencies.linear_fit_sufficient_statistics"
                    ));
                } else if bayesian_distribution_replay_dependency_only(&consumed) {
                    dependency_refusals.push(format!(
                        "{coordinate}: dependencies.distribution_posterior_factor_draws"
                    ));
                } else {
                    return Err(format!("consumer refusal: {:?}", consumed.acceptance.unresolved));
                }
            } else if consumed.body.estimate.is_some_and(|estimate| {
                (estimate - refreshed.effect()).abs() > 1e-12 * refreshed.effect().abs().max(1.0)
            }) {
                return Err(format!(
                    "independently consumed estimate {:?} differs from refreshed effect {}",
                    consumed.body.estimate,
                    refreshed.effect()
                ));
            }
            Ok(())
        })();

        if let Err(error) = probe {
            if error.starts_with("compatible refresh:") {
                refresh_gaps.push(format!("{coordinate}: {error}"));
            } else if error.starts_with("selected estimator drifted") {
                selection_gaps.push(format!("{coordinate}: {error}"));
            } else {
                execution_gaps.push(format!("{coordinate}: {error}"));
            }
        }
    }

    println!("optional estimator diagnostic inspected {inspected} coordinates");
    println!("checked-plan gaps ({}):\n{}", checked_plan_gaps.len(), checked_plan_gaps.join("\n"));
    println!("no public plan inspector ({}):\n{}", uninspectable.len(), uninspectable.join("\n"));
    println!("selection/plan drift ({}):\n{}", selection_gaps.len(), selection_gaps.join("\n"));
    println!(
        "execution or artifact gaps ({}):\n{}",
        execution_gaps.len(),
        execution_gaps.join("\n")
    );
    println!("refresh gaps ({}):\n{}", refresh_gaps.len(), refresh_gaps.join("\n"));
    println!(
        "known dependency refusals ({}):\n{}",
        dependency_refusals.len(),
        dependency_refusals.join("\n")
    );
    assert_eq!(inspected, 136, "optional estimator registry drifted");
}

/// The sealed outcome. Every licensed route executes on a retained checked
/// operation; a consumer that holds only the exported bytes cannot resolve that
/// operation, so it names it as `dependencies.checked_<family>_operation` (the
/// counterfactual GCM route names its fitted mechanisms) and withholds
/// verified-program acceptance while it still recognizes the contract section
/// and reads the body. Any other unresolved reason, or an unrecognized or
/// unreadable contract, is a genuine refusal.
fn sealed_checked_operation_dependency_only(
    consumed: &antecedent_io::AnalysisResultConsumption,
) -> bool {
    let acceptance = &consumed.acceptance;
    acceptance.recognized
        && consumed.contract.is_some()
        && !acceptance.unresolved.is_empty()
        && acceptance.unresolved.iter().all(|reason| {
            reason.as_ref() == "dependencies.fitted_counterfactual_mechanisms"
                || reason
                    .as_ref()
                    .strip_prefix("dependencies.checked_")
                    .is_some_and(|family| family.ends_with("_operation"))
        })
}

/// The functional-effect programs export a readable claim without the replay
/// input an independent consumer would need: posterior row weights for a
/// Bayesian scalar or response, or a response grid for a frequentist ADMG
/// intervention response. The route evidence pins each key as the outcome.
fn functional_effect_replay_dependency_only(
    consumed: &antecedent_io::AnalysisResultConsumption,
) -> bool {
    let Some(contract) = consumed.contract.as_ref() else {
        return false;
    };
    let functional_effect = contract.estimator.as_deref() == Some("functional.effect")
        || contract.program.as_ref().is_some_and(|program| {
            program.commitments.resolved_estimator.as_deref() == Some("functional.effect")
        });
    let bayesian = contract
        .inference_binding
        .as_ref()
        .is_some_and(|binding| binding.inference == "bayesian" && binding.bayesian.is_some());
    consumed.acceptance.recognized
        && functional_effect
        && consumed.acceptance.unresolved.len() == 1
        && match consumed.acceptance.unresolved[0].as_ref() {
            "dependencies.functional_effect_posterior_draws"
            | "dependencies.functional_effect_response_posterior_draws" => bayesian,
            "program.functional_effect_response_grid" => {
                !bayesian
                    && matches!(contract.graph_class.as_str(), "Admg" | "TemporalAdmg")
                    && matches!(contract.target.query, antecedent_io::CausalQueryWire::Response(_))
            }
            _ => false,
        }
}

fn linear_replay_dependency_only(consumed: &antecedent_io::AnalysisResultConsumption) -> bool {
    consumed.acceptance.unresolved.len() == 1
        && consumed.acceptance.unresolved[0].as_ref()
            == "dependencies.linear_fit_sufficient_statistics"
        && consumed.contract.as_ref().is_some_and(|contract| {
            (contract.estimator.as_deref() == Some("linear.adjustment.ate")
                || contract.program.as_ref().is_some_and(|program| {
                    program.commitments.resolved_estimator.as_deref()
                        == Some("linear.adjustment.ate")
                }))
                && contract
                    .program
                    .as_ref()
                    .is_some_and(|program| program.checked_linear_adjustment_lowering.is_some())
        })
}

/// A distribution over the completions of a CPDAG or the ADMGs of a PAG publishes the class
/// envelope, not one member's factor laws, so an independent replay lacks those laws (and, under
/// the Bayesian bootstrap, the posterior factor draws); the program itself is verified.
fn class_distribution_replay_dependency_only(
    consumed: &antecedent_io::AnalysisResultConsumption,
) -> bool {
    !consumed.acceptance.unresolved.is_empty()
        && consumed.acceptance.unresolved.iter().all(|item| {
            matches!(
                item.as_ref(),
                "dependencies.distribution_factor_laws"
                    | "dependencies.distribution_posterior_factor_draws"
            )
        })
        && consumed.contract.as_ref().is_some_and(|contract| {
            contract.estimator.as_deref() == Some("functional.distribution")
                && matches!(contract.graph_class.as_str(), "Cpdag" | "Pag")
                && matches!(contract.target.query, antecedent_io::CausalQueryWire::Distribution(_))
        })
}

/// A distribution over the atoms of a graph posterior is a posterior-weighted mixture: the
/// contract carries one sealed program per atom, verified against the mixture, but no single
/// fitted factor-law set an independent replay could rebuild the mixture from (and, under the
/// Bayesian bootstrap, no posterior factor draws).
fn graph_posterior_distribution_replay_dependency_only(
    consumed: &antecedent_io::AnalysisResultConsumption,
) -> bool {
    !consumed.acceptance.unresolved.is_empty()
        && consumed.acceptance.unresolved.iter().all(|item| {
            matches!(
                item.as_ref(),
                "dependencies.distribution_factor_laws"
                    | "dependencies.distribution_posterior_factor_draws"
            )
        })
        && consumed.contract.as_ref().is_some_and(|contract| {
            (contract.estimator.as_deref() == Some("functional.distribution")
                || contract.program.as_ref().is_some_and(|program| {
                    program.commitments.resolved_estimator.as_deref()
                        == Some("functional.distribution")
                }))
                && contract.structure_source == "graph_posterior"
                && matches!(contract.target.query, antecedent_io::CausalQueryWire::Distribution(_))
                && contract
                    .program
                    .as_ref()
                    .is_some_and(|program| program.graph_posterior_programs.is_some())
        })
}

fn bayesian_distribution_replay_dependency_only(
    consumed: &antecedent_io::AnalysisResultConsumption,
) -> bool {
    consumed.acceptance.unresolved.len() == 1
        && consumed.acceptance.unresolved[0].as_ref()
            == "dependencies.distribution_posterior_factor_draws"
        && consumed.contract.as_ref().is_some_and(|contract| {
            contract.estimator.as_deref() == Some("functional.distribution")
                && matches!(contract.target.query, antecedent_io::CausalQueryWire::Distribution(_))
                && contract.inference_binding.as_ref().is_some_and(|binding| {
                    binding.inference == "bayesian" && binding.bayesian.is_some()
                })
                && contract.identification_product.is_some()
                && contract
                    .data_snapshot
                    .as_ref()
                    .is_some_and(|snapshot| snapshot.distribution_factor_laws.is_some())
        })
}

// ---------------------------------------------------------------------------
// Independent numerical and uncertainty acceptance, and replay.
//
// The sweep above proves that every licensed cell completes the compiler path.
// These tests prove that the numbers a compiled route publishes are the numbers
// an independent computation gives on the same data: the point value and the
// published analytic standard error are recomputed here from the raw columns
// by ordinary least squares and instrumental-variable algebra that shares no
// code with the estimators.
// ---------------------------------------------------------------------------

use antecedent_estimate::AnalyticSeKind;

/// What one execution of a cell published.
struct Observed {
    effect: f64,
    analytic_se: f64,
    se_kind: Option<AnalyticSeKind>,
    estimator: String,
    /// First-stage F statistic and Anderson-Rubin set `(lower, upper, level)`, when published.
    first_stage_f: Option<f64>,
    anderson_rubin: Option<(f64, f64, f64)>,
}

fn observe(setup: CellSetup) -> Result<Observed, String> {
    let expected = setup.expected.clone();
    let ctx = ExecutionContext::for_tests(1);
    let built = setup.builder.build().map_err(|e| format!("{expected}: build {e}"))?;
    let prepared = built.prepare(&ctx).map_err(|e| format!("{expected}: prepare {e}"))?;
    let result = match &setup.data {
        CellData::Tabular(data) => prepared.estimate(data, &ctx),
        CellData::Series(data) => prepared.estimate_series(data, &ctx),
    }
    .map_err(|e| format!("{expected}: execute {e}"))?;
    let (analytic_se, se_kind) = result
        .estimate()
        .as_effect()
        .map_or((f64::NAN, None), |e| (e.uncertainty.analytic.se, e.uncertainty.analytic.kind));
    Ok(Observed {
        effect: result.effect(),
        analytic_se,
        se_kind,
        estimator: result.logical_plan.estimator.as_deref().unwrap_or("-").to_owned(),
        first_stage_f: result
            .estimate()
            .as_effect()
            .and_then(|e| e.diagnostics.first_stage.as_ref())
            .map(|d| d.f_statistic),
        anderson_rubin: result
            .estimate()
            .as_effect()
            .and_then(|e| e.diagnostics.first_stage.as_ref())
            .and_then(|d| d.anderson_rubin),
    })
}

/// The first Frequentist explicit-structure average-effect cell that licenses `estimator`.
fn cell_licensing(estimator: &str) -> antecedent::SupportCell {
    licensed_support_cells()
        .into_iter()
        .find(|cell| {
            cell.inference.as_str() == "Frequentist"
                && cell.structure.as_str() == "explicit"
                && cell.query.as_str() == "AverageEffect"
                && licensed_estimators(*cell).iter().any(|name| *name == estimator)
        })
        .unwrap_or_else(|| panic!("no licensed average-effect cell names {estimator}"))
}

fn invert(mut a: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let p = a.len();
    let mut inverse: Vec<Vec<f64>> =
        (0..p).map(|i| (0..p).map(|j| f64::from(u8::from(i == j))).collect()).collect();
    for column in 0..p {
        let pivot = (column..p)
            .max_by(|&i, &j| a[i][column].abs().total_cmp(&a[j][column].abs()))
            .expect("a pivot row");
        a.swap(column, pivot);
        inverse.swap(column, pivot);
        let divisor = a[column][column];
        assert!(divisor.abs() > 1e-12, "the reference design is singular");
        for j in 0..p {
            a[column][j] /= divisor;
            inverse[column][j] /= divisor;
        }
        let (row_a, row_i) = (a[column].clone(), inverse[column].clone());
        for i in 0..p {
            if i != column {
                let factor = a[i][column];
                for j in 0..p {
                    a[i][j] -= factor * row_a[j];
                    inverse[i][j] -= factor * row_i[j];
                }
            }
        }
    }
    inverse
}

/// Ordinary least squares from first principles.
struct Regression {
    beta: Vec<f64>,
    bread: Vec<Vec<f64>>,
    design: Vec<Vec<f64>>,
    residuals: Vec<f64>,
    leverage: Vec<f64>,
}

fn regress(design: Vec<Vec<f64>>, y: &[f64]) -> Regression {
    let p = design[0].len();
    let mut xtx = vec![vec![0.0; p]; p];
    let mut xty = vec![0.0; p];
    for (row, outcome) in design.iter().zip(y) {
        for i in 0..p {
            xty[i] += row[i] * outcome;
            for j in 0..p {
                xtx[i][j] += row[i] * row[j];
            }
        }
    }
    let bread = invert(xtx);
    let beta: Vec<f64> =
        (0..p).map(|i| (0..p).map(|j| bread[i][j] * xty[j]).sum::<f64>()).collect();
    let residuals: Vec<f64> = design
        .iter()
        .zip(y)
        .map(|(row, outcome)| outcome - row.iter().zip(&beta).map(|(x, b)| x * b).sum::<f64>())
        .collect();
    let leverage: Vec<f64> = design
        .iter()
        .map(|row| {
            (0..p).map(|i| (0..p).map(|j| row[i] * bread[i][j] * row[j]).sum::<f64>()).sum::<f64>()
        })
        .collect();
    Regression { beta, bread, design, residuals, leverage }
}

impl Regression {
    fn homoskedastic_se(&self, k: usize) -> f64 {
        let n = self.residuals.len() as f64;
        let p = self.beta.len() as f64;
        let sigma2 = self.residuals.iter().map(|e| e * e).sum::<f64>() / (n - p);
        (sigma2 * self.bread[k][k]).sqrt()
    }

    /// Sandwich standard error with weight `scale(leverage)` on each squared residual.
    fn sandwich_se(&self, k: usize, scale: impl Fn(f64) -> f64) -> f64 {
        let p = self.beta.len();
        let mut meat = vec![vec![0.0; p]; p];
        for ((row, e), h) in self.design.iter().zip(&self.residuals).zip(&self.leverage) {
            let weight = e * e * scale(*h);
            for i in 0..p {
                for j in 0..p {
                    meat[i][j] += weight * row[i] * row[j];
                }
            }
        }
        let mut variance = 0.0;
        for i in 0..p {
            for j in 0..p {
                variance += self.bread[k][i] * meat[i][j] * self.bread[j][k];
            }
        }
        variance.sqrt()
    }

    fn published_se(&self, k: usize, kind: Option<AnalyticSeKind>) -> f64 {
        let n = self.residuals.len() as f64;
        let p = self.beta.len() as f64;
        match kind {
            None | Some(AnalyticSeKind::Homoskedastic) => self.homoskedastic_se(k),
            Some(AnalyticSeKind::Hc0) => self.sandwich_se(k, |_| 1.0),
            Some(AnalyticSeKind::Hc1) => self.sandwich_se(k, |_| n / (n - p)),
            Some(AnalyticSeKind::Hc2) => self.sandwich_se(k, |h| 1.0 / (1.0 - h)),
            Some(AnalyticSeKind::Hc3) => self.sandwich_se(k, |h| 1.0 / ((1.0 - h) * (1.0 - h))),
            Some(other) => panic!("no independent reference for the {other:?} standard error"),
        }
    }
}

/// The columns `binary_treatment_table(&[("t", 0), ("y", 1), ("z", 2)])` holds.
fn binary_treatment_columns() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = 48usize;
    let t = (0..n).map(|i| (i % 2) as f64).collect();
    let y =
        (0..n).map(|i| 0.3 + 0.8 * (i % 2) as f64 + 0.2 * ((i + 5) as f64 * 0.13).sin()).collect();
    let z = (0..n)
        .map(|i| ((i / 2 + 6) % 5) as f64 * 0.25 + 0.1 * ((i + 2) as f64 * 0.2).sin())
        .collect();
    (t, y, z)
}

#[test]
fn the_linear_adjustment_route_matches_an_independent_regression_and_its_published_se() {
    let cell = antecedent::SupportCell::from_labels(
        "AverageEffect",
        "Dag",
        "explicit",
        "Frequentist",
        "none",
    )
    .expect("fixture support axis");
    let observed = observe(cell_setup(cell).expect("setup")).expect("executes");
    assert_eq!(observed.estimator, "linear.adjustment.ate");
    // The graph adjusts for the confounder z: the effect is the coefficient on t
    // in y ~ 1 + t + z.
    let (t, y, z) = binary_treatment_columns();
    let design: Vec<Vec<f64>> = (0..t.len()).map(|i| vec![1.0, t[i], z[i]]).collect();
    let reference = regress(design, &y);
    assert!(
        (observed.effect - reference.beta[1]).abs() < 1e-8,
        "route {} vs regression {}",
        observed.effect,
        reference.beta[1]
    );
    // The truth is 0.8 plus a smooth component that z only partly absorbs.
    assert!((reference.beta[1] - 0.8).abs() < 0.2);
    // The published standard error is the one its recorded formula defines.
    let expected_se = reference.published_se(1, observed.se_kind);
    assert!(
        (observed.analytic_se - expected_se).abs() < 1e-6 * expected_se,
        "{:?}: published {} vs independent {expected_se}",
        observed.se_kind,
        observed.analytic_se
    );
    // The adjusted and unadjusted contrasts differ, so the check can tell them apart.
    let naive = regress((0..t.len()).map(|i| vec![1.0, t[i]]).collect(), &y);
    assert!((naive.beta[1] - reference.beta[1]).abs() > 1e-6);
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

#[test]
fn the_wald_and_two_stage_iv_routes_match_the_instrumental_variable_algebra() {
    let n = 120usize;
    let z: Vec<f64> = (0..n).map(|i| (i % 2) as f64).collect();
    let t: Vec<f64> = (0..n).map(|i| f64::from(z[i] > 0.5 && i % 4 != 1)).collect();
    let y: Vec<f64> = (0..n).map(|i| 0.5 * t[i] + ((i * 7 % 19) as f64) / 19.0).collect();
    let (mz, mt, my) = (mean(&z), mean(&t), mean(&y));
    let szy: f64 = (0..n).map(|i| (z[i] - mz) * (y[i] - my)).sum();
    let szt: f64 = (0..n).map(|i| (z[i] - mz) * (t[i] - mt)).sum();
    let wald = szy / szt;
    let (syy, syt, stt): (f64, f64, f64) = (
        (0..n).map(|i| (y[i] - my) * (y[i] - my)).sum(),
        (0..n).map(|i| (y[i] - my) * (t[i] - mt)).sum(),
        (0..n).map(|i| (t[i] - mt) * (t[i] - mt)).sum(),
    );
    let szz: f64 = (0..n).map(|i| (z[i] - mz) * (z[i] - mz)).sum();
    // First stage `t ~ 1 + z`: the F statistic of the instrument.
    let slope = szt / szz;
    let first_stage_rss: f64 = (0..n).map(|i| (t[i] - mt - slope * (z[i] - mz)).powi(2)).sum();
    let df = n as f64 - 2.0;
    let first_stage_f = slope * slope * szz / (first_stage_rss / df);
    // Anderson-Rubin set at 95%: beta0 is kept when the F test of `z` in `y - beta0 t ~ 1 + z`
    // does not reject, `df * szu^2 / szz / (suu - szu^2 / szz) <= c` with `c` the 0.95 quantile
    // of F(1, 118) (the square of the 0.975 quantile of t(118)). That is the quadratic
    // `(df + c) (szy - b szt)^2 - c szz (syy - 2 b syt + b^2 stt) <= 0` in `b`.
    let critical = 1.980_272_249_272_974_f64.powi(2);
    let q_a = (df + critical) * szt * szt - critical * szz * stt;
    let q_b = -2.0 * (df + critical) * szy * szt + 2.0 * critical * szz * syt;
    let q_c = (df + critical) * szy * szy - critical * szz * syy;
    assert!(q_a > 0.0, "the fixture needs a strong instrument: a bounded set");
    let root = (q_b * q_b - 4.0 * q_a * q_c).sqrt();
    let (ar_lower, ar_upper) = ((-q_b - root) / (2.0 * q_a), (-q_b + root) / (2.0 * q_a));
    assert!(ar_lower < wald && wald < ar_upper, "the set contains the Wald ratio");
    for estimator in ["iv.wald", "iv.2sls"] {
        let id = estimator.parse::<EstimatorId>().expect("estimator id");
        let cell = cell_licensing(estimator);
        let observed = observe(optional_estimator_setup(cell, id).expect("setup")).expect("runs");
        assert_eq!(observed.estimator, estimator);
        assert!(
            (observed.effect - wald).abs() < 1e-8,
            "{estimator}: route {} vs IV algebra {wald}",
            observed.effect
        );
        // Licensed instrumental-variable uncertainty is the weak-instrument-robust
        // Anderson-Rubin set on the first-stage diagnostics; a Wald or bootstrap standard
        // error is withheld, so the point carries none.
        assert!(!observed.analytic_se.is_finite(), "{estimator}: {}", observed.analytic_se);
        // The published set is the independently inverted Anderson-Rubin test.
        let (lower, upper, level) = observed.anderson_rubin.expect("an Anderson-Rubin set");
        assert!((level - 0.95).abs() < 1e-12, "{estimator}: level {level}");
        assert!(
            (lower - ar_lower).abs() < 1e-5 * (1.0 + ar_lower.abs())
                && (upper - ar_upper).abs() < 1e-5 * (1.0 + ar_upper.abs()),
            "{estimator}: published [{lower}, {upper}] vs independent [{ar_lower}, {ar_upper}]"
        );
        let f = observed.first_stage_f.expect("a first-stage F");
        assert!(
            (f - first_stage_f).abs() < 1e-6 * first_stage_f,
            "{estimator}: first-stage F {f} vs {first_stage_f}"
        );
    }
}

#[test]
fn the_front_door_route_matches_the_product_of_its_two_stage_coefficients() {
    let n = 120usize;
    let t: Vec<f64> = (0..n).map(|i| (i % 2) as f64).collect();
    let m: Vec<f64> = (0..n).map(|i| 0.3 * t[i] + ((i * 7 % 13) as f64) / 13.0).collect();
    let y: Vec<f64> = (0..n).map(|i| 0.7 * m[i] + ((i * 11 % 17) as f64) / 17.0).collect();
    let first = regress((0..n).map(|i| vec![1.0, t[i]]).collect(), &m);
    let second = regress((0..n).map(|i| vec![1.0, m[i], t[i]]).collect(), &y);
    let product = first.beta[1] * second.beta[1];
    let estimator = "frontdoor.linear_two_stage";
    let id = estimator.parse::<EstimatorId>().expect("estimator id");
    let cell = cell_licensing(estimator);
    let observed = observe(optional_estimator_setup(cell, id).expect("setup")).expect("runs");
    assert_eq!(observed.estimator, estimator);
    assert!(
        (observed.effect - product).abs() < 1e-8,
        "route {} vs two-stage product {product}",
        observed.effect
    );
    // The mediated path is not the direct coefficient: the check discriminates.
    assert!((product - second.beta[2]).abs() > 1e-4);
}

#[test]
fn the_sharp_regression_discontinuity_route_recovers_the_jump_a_local_fit_measures() {
    let n = 120usize;
    let r: Vec<f64> = (0..n).map(|i| (i as f64 - 60.0) / 30.0).collect();
    let y: Vec<f64> = (0..n)
        .map(|i| 1.0 + 1.5 * f64::from(r[i] >= 0.0) + 0.4 * r[i] + ((i % 7) as f64) * 0.01)
        .collect();
    let side = |keep: &dyn Fn(f64) -> bool| {
        let rows: Vec<usize> = (0..n).filter(|&i| keep(r[i])).collect();
        regress(
            rows.iter().map(|&i| vec![1.0, r[i]]).collect(),
            &rows.iter().map(|&i| y[i]).collect::<Vec<_>>(),
        )
    };
    let right = side(&|x| (0.0..=1.5).contains(&x));
    let left = side(&|x| (-1.5..0.0).contains(&x));
    let local_jump = right.beta[0] - left.beta[0];
    let estimator = "rd.sharp";
    let id = estimator.parse::<EstimatorId>().expect("estimator id");
    let cell = cell_licensing(estimator);
    let observed = observe(optional_estimator_setup(cell, id).expect("setup")).expect("runs");
    assert_eq!(observed.estimator, estimator);
    assert!((observed.effect - 1.5).abs() < 0.1, "{} vs truth 1.5", observed.effect);
    assert!(
        (observed.effect - local_jump).abs() < 0.06,
        "route {} vs local linear jump {local_jump}",
        observed.effect
    );
}

/// What one full compiler path published and what a consumer read back.
struct Replayed {
    effect: f64,
    wire_estimate: Option<f64>,
    scalar_effect: bool,
    lowered: Option<antecedent_core::SemanticDigest>,
}

fn replay_cell(cell: antecedent::SupportCell) -> Result<Replayed, String> {
    let setup = cell_setup(cell)?;
    let expected = setup.expected.clone();
    let ctx = ExecutionContext::for_tests(1);
    let built = setup.builder.build().map_err(|e| format!("{expected}: build {e}"))?;
    let prepared = built.prepare(&ctx).map_err(|e| format!("{expected}: prepare {e}"))?;
    let result = match &setup.data {
        CellData::Tabular(data) => prepared.estimate(data, &ctx),
        CellData::Series(data) => prepared.estimate_series(data, &ctx),
    }
    .map_err(|e| format!("{expected}: execute {e}"))?;
    let bytes = prepared
        .encode_contracted_result(&result, &expected, &ctx)
        .map_err(|e| format!("{expected}: encode {e}"))?;
    let consumed =
        consume_analysis_result(&bytes).map_err(|e| format!("{expected}: consume {e}"))?;
    Ok(Replayed {
        effect: result.effect(),
        wire_estimate: consumed.body.estimate,
        scalar_effect: result.estimate().as_effect().is_some(),
        lowered: result.lowered_program_identity(),
    })
}

/// A consumer that reads only the exported bytes recovers the number the route
/// published, and an independent rebuild and execution of the same cell
/// reproduces it bit for bit under the same lowered program. Every ninth
/// licensed cell is replayed, which covers every query family on both
/// inference axes.
#[test]
fn sampled_licensed_cells_replay_to_the_published_number() {
    let mut failures = Vec::new();
    let mut replayed = 0usize;
    for (index, cell) in licensed_support_cells().into_iter().enumerate() {
        if index % 9 != 0 {
            continue;
        }
        let coordinate = cell_coordinate(cell);
        let (first, second) = match (replay_cell(cell), replay_cell(cell)) {
            (Ok(first), Ok(second)) => (first, second),
            (Err(error), _) | (_, Err(error)) => {
                failures.push(error);
                continue;
            }
        };
        replayed += 1;
        if first.effect.to_bits() != second.effect.to_bits() {
            failures
                .push(format!("{coordinate}: reruns differ {} vs {}", first.effect, second.effect));
        }
        if first.lowered.is_none() || first.lowered != second.lowered {
            failures.push(format!("{coordinate}: the lowered program identity did not replay"));
        }
        if let Some(wire) = first.wire_estimate {
            if first.scalar_effect && first.effect.is_finite() {
                let allowed = 1e-9 * first.effect.abs().max(1.0);
                if (wire - first.effect).abs() > allowed {
                    failures.push(format!(
                        "{coordinate}: consumer read {wire}, route published {}",
                        first.effect
                    ));
                }
            }
        }
    }
    assert!(replayed >= 60, "the sample drifted: {replayed} cells");
    assert!(failures.is_empty(), "{} replay failures:\n{}", failures.len(), failures.join("\n"));
}

/// Bind `data` on a new branch, materialize and estimate it, then adopt the branch as the handle.
/// A failure leaves the handle unchanged, as the immutable lifecycle does.
fn rebind_estimate(
    prepared: &mut antecedent::PreparedStudy,
    data: antecedent_data::TabularData,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<antecedent::StudyResult, antecedent::CausalError> {
    let branch = prepared.rebind_data(data, ctx)?.materialize(ctx)?;
    let result = branch.estimate_retained(ctx)?;
    *prepared = branch;
    Ok(result)
}

#[allow(dead_code)]
mod temporal_admg_compiler_fixture {
    //! Known-truth evidence for derivatives on a supplied or accepted temporal ADMG at one horizon,
    //! at validation none, cheap and full.
    //!
    //! The series is the confounded lag-one law of
    //! `conformance/response/temporal_admg_derivative_states/expected.json`: the observed confounder
    //! `z(t-1)` separates the treatment `x(t-1)` from the latent cause of `w(t-1)` and `y(t)`, so the
    //! structural slope is the coefficient of `x(t-1)` in the regression of `y(t)` on
    //! `(1, x(t-1), z(t-1), w(t-1))`, which the fixture's reference solves in exact rational
    //! arithmetic. Every derivative coordinate is built under both structure routes, both inference
    //! modes and the three validation suites; each run executes from the prepared study with the
    //! builder dropped. The published value is compared with the fixture (not the confounded slope),
    //! the refuter states (support, the unit-step difference of the identified levels, the
    //! Pulse-witness suite, Bayesian predictive checks and prior sensitivity) are checked against it,
    //! and the planted violation the fixture declares (an evaluation point beyond the observed
    //! treatment range) must be detected.
    //!
    //! SPDX-License-Identifier: MIT OR Apache-2.0

    #![allow(
        clippy::cast_precision_loss,
        clippy::too_many_lines,
        reason = "test scaffolding builds small fixtures and one scenario is kept in one body"
    )]

    use antecedent::{AcceptedGraph, InferenceMode, RefuteSuite, Study, StudyResult};
    use antecedent_core::{
        CausalQuery, DerivativeScale, DerivativeWeighting, Lag, ResponseFunctional as F,
        ResponseIdentification, ResponseQuery, ResponseValue,
        TemporalPolicy, TemporalResponseSpec, VariableId,
    };
    use antecedent_data::TimeSeriesData;
    use antecedent_graph::TemporalAdmg;
    use serde_json::Value;
    use std::sync::Arc;

    const N: usize = 600;
    const X: u32 = 0;
    const Y: u32 = 1;
    const Z: u32 = 2;
    const W: u32 = 3;

    pub(super) fn fixture() -> Value {
        serde_json::from_str(include_str!(
            "../../../conformance/response/temporal_admg_derivative_states/expected.json"
        ))
        .expect("temporal_admg_derivative_states fixture parses")
    }

    pub(super) fn number(document: &Value, key: &str) -> f64 {
        document[key].as_f64().unwrap_or_else(|| panic!("fixture lacks number `{key}`"))
    }

    fn v(i: u32) -> VariableId {
        VariableId::from_raw(i)
    }

    /// `w = sin(1.7 t + .3) + .8 l`, `z = cos(.41 t) + .5 w`, `x = 2 + .6 z + sin(.71 t) + .2 cos(.13
    /// t)`, `y(t) = 5 + 2 x(t-1) + 3 z(t-1) + .7 l(t-1) + .5 e` with the latent `l = cos(.77 t + 2)`
    /// and `e` the sum of twelve uniforms of one 64-bit LCG stream minus six, one draw per `t >= 1`.
    /// Returns the columns `x, y, z, w` as a series.
    pub(super) fn series() -> TimeSeriesData {
        let mut state = 20_240_607_u64;
        let mut gauss = || {
            let mut sum = 0.0;
            for _ in 0..12 {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                sum += (state >> 11) as f64 / 9_007_199_254_740_992.0;
            }
            sum - 6.0
        };
        let mut x = vec![0.0; N];
        let mut y = vec![0.0; N];
        let mut z = vec![0.0; N];
        let mut w = vec![0.0; N];
        let mut latent = vec![0.0; N];
        for t in 0..N {
            let clock = t as f64;
            latent[t] = (clock * 0.77 + 2.0).cos();
            w[t] = (clock * 1.7 + 0.3).sin() + 0.8 * latent[t];
            z[t] = (clock * 0.41).cos() + 0.5 * w[t];
            x[t] = 2.0 + 0.6 * z[t] + (clock * 0.71).sin() + 0.2 * (clock * 0.13).cos();
        }
        for t in 1..N {
            y[t] = 5.0 + 2.0 * x[t - 1] + 3.0 * z[t - 1] + 0.7 * latent[t - 1] + 0.5 * gauss();
        }
        TimeSeriesData::from_f64_columns(
            [("x", x.as_slice()), ("y", y.as_slice()), ("z", z.as_slice()), ("w", w.as_slice())],
            1,
        )
        .unwrap()
    }

    /// `z(t-1) -> x(t-1)`, `z(t-1) -> y(t)`, `x(t-1) -> y(t)`, `w(t-1) -> z(t-1)` and a latent common
    /// cause of `w(t-1)` and `y(t)`.
    fn template() -> TemporalAdmg {
        let mut graph = TemporalAdmg::empty();
        let x1 = graph.add_lagged(v(X), Lag::from_raw(1)).unwrap();
        let z1 = graph.add_lagged(v(Z), Lag::from_raw(1)).unwrap();
        let w1 = graph.add_lagged(v(W), Lag::from_raw(1)).unwrap();
        let y0 = graph.add_lagged(v(Y), Lag::CONTEMPORANEOUS).unwrap();
        graph.insert_directed(z1, x1).unwrap();
        graph.insert_directed(z1, y0).unwrap();
        graph.insert_directed(x1, y0).unwrap();
        graph.insert_directed(w1, z1).unwrap();
        graph.insert_bidirected(w1, y0).unwrap();
        graph
    }

    fn point(scale: DerivativeScale, at: f64) -> F {
        F::PointDerivative { outcome: v(Y), treatment: v(X), at, order: 1, scale }
    }

    pub(super) fn study(
        data: &TimeSeriesData,
        functional: &F,
        accepted: bool,
        inference: InferenceMode,
        suite: RefuteSuite,
    ) -> antecedent::StudyBuilder {
        let query = ResponseQuery::new(functional.clone()).with_temporal(
            TemporalResponseSpec::new(vec![1_u32], TemporalPolicy::pulse(-1), None).unwrap(),
        );
        let builder = Study::series(data.clone()).query(CausalQuery::Response(query));
        let builder = if accepted {
            builder.graph(AcceptedGraph::from(template()))
        } else {
            builder.graph(template())
        };
        builder.inference(inference).refute(suite).bootstrap_replicates(12)
    }

    fn published(result: &StudyResult, label: &str) -> Vec<f64> {
        let ResponseIdentification::PointIdentified(value) = &result.response().unwrap().estimate
        else {
            panic!("{label}: the response is not point identified");
        };
        match value {
            ResponseValue::Scalar(x) => vec![*x],
            ResponseValue::Jacobian { values, .. } | ResponseValue::Vector(values) => {
                values.to_vec()
            }
            other => panic!("{label}: unexpected payload {other:?}"),
        }
    }

    fn report<'a>(
        result: &'a StudyResult,
        name: &str,
    ) -> Option<&'a antecedent_validate::RefutationReport> {
        result.refutations.iter().find(|report| report.refuter.as_ref() == name)
    }

    /// Every functional with the reference values the fixture implies.
    pub(super) fn cases(pin: &Value, at: f64) -> Vec<(&'static str, F, Vec<f64>)> {
        let slope = number(pin, "slope_y");
        vec![
            ("point", point(DerivativeScale::Identity, at), vec![slope]),
            ("elasticity", point(DerivativeScale::LogLog, at), vec![number(pin, "elasticity")]),
            (
                "semi_treatment",
                point(DerivativeScale::LogTreatment, at),
                vec![number(pin, "semi_treatment")],
            ),
            (
                "semi_outcome",
                point(DerivativeScale::LogOutcome, at),
                vec![number(pin, "semi_outcome")],
            ),
            (
                "average",
                F::AverageDerivative {
                    outcome: v(Y),
                    treatment: v(X),
                    weighting: DerivativeWeighting::Observed,
                },
                vec![slope],
            ),
            (
                "jacobian",
                F::Jacobian {
                    outcomes: Arc::from([v(Y)]),
                    treatments: Arc::from([v(X)]),
                    at: Arc::from([at]),
                    scale: DerivativeScale::Identity,
                },
                vec![slope],
            ),
            (
                "directional",
                F::DirectionalDerivative {
                    outcomes: Arc::from([v(Y)]),
                    treatments: Arc::from([v(X)]),
                    at: Arc::from([at]),
                    direction: Arc::from([1.0]),
                },
                vec![slope],
            ),
        ]
    }
}
