//! Every licensed support-matrix cell is first-class on inspect.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::too_many_lines)]

use std::sync::Arc;

use antecedent::{
    AcceptedGraph, BayesianConfig, CellStatus, InferenceMode, InterferenceSpec, RefuteSuite, Study,
    TransportTrialSpec, cell_coordinate, licensed_support_cells,
};
use antecedent_core::{
    AnomalyAttributionQuery, AssignmentDesign, AverageEffectQuery, CausalQuery, ChangeAttributionQuery,
    ConditionalEffectQuery, ContinuousDomain, CounterfactualQuery, DerivativeScale,
    DerivativeWeighting, ExposureLevel, ExposureMapping, GridSpec, InterferenceFunctional,
    InterferenceQuery, Intervention, InterventionalDistributionQuery, Lag, MediationContrast,
    MediationQuery, PathSpecificEffectQuery, PopulationSelector, ResponseFunctional, ResponseQuery,
    SlotAvailability, TemporalEffectQuery, TemporalPolicy, TemporalResponseSpec, TransportQuery,
    Value, VariableId,
};
use antecedent_data::{
    NetworkData, NetworkEdge, SamplingRegularity, TableView, TabularData, TimeIndex, TimeSeriesData,
};
use antecedent_discovery::GraphPosterior;
use antecedent_graph::{
    Admg, Cpdag, Dag, DenseNodeId, Pag, TemporalCpdag, TemporalDag, TemporalPag, TieredBackground,
    WithinTier, ensure_lagged,
};
use antecedent_prob::InferenceDiagnostics;

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

fn static_table(cols: &[(&str, usize)]) -> TabularData {
    let n = 48usize;
    let owned: Vec<(String, Vec<f64>)> = cols
        .iter()
        .enumerate()
        .map(|(j, (name, _))| {
            let values: Vec<f64> = (0..n)
                .map(|i| ((i + 3 * j) as f64 * 0.17).sin() + 0.25 * (i % 4) as f64)
                .collect();
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

fn chain_dag(n: u32) -> Dag {
    let mut g = Dag::with_variables(n);
    for i in 0..n.saturating_sub(1) {
        g.insert_directed(nid(i), nid(i + 1)).unwrap();
    }
    g
}

fn confounder_dag() -> Dag {
    let mut g = Dag::with_variables(3);
    g.insert_directed(nid(2), nid(0)).unwrap();
    g.insert_directed(nid(2), nid(1)).unwrap();
    g.insert_directed(nid(0), nid(1)).unwrap();
    g
}

fn confounder_admg() -> Admg {
    let mut g = Admg::with_variables(3);
    g.insert_directed(nid(0), nid(1)).unwrap();
    g.insert_bidirected(nid(0), nid(2)).unwrap();
    g
}

fn confounder_pag() -> Pag {
    let mut g = Pag::with_variables(3);
    g.insert_directed(nid(2), nid(0)).unwrap();
    g.insert_directed(nid(2), nid(1)).unwrap();
    g.insert_directed(nid(0), nid(1)).unwrap();
    g
}

fn transport_admg() -> Admg {
    let mut g = Admg::with_variables(5);
    g.insert_directed(nid(0), nid(1)).unwrap();
    g
}

fn temporal_dag(n_vars: u32) -> TemporalDag {
    let mut g = TemporalDag::empty();
    if n_vars >= 2 {
        let x1 = ensure_lagged(&mut g, vid(0), Lag::from_raw(1)).unwrap();
        let y0 = ensure_lagged(&mut g, vid(1), Lag::CONTEMPORANEOUS).unwrap();
        g.insert_directed(x1, y0).unwrap();
    }
    if n_vars >= 3 {
        let m1 = ensure_lagged(&mut g, vid(1), Lag::from_raw(1)).unwrap();
        let y0 = ensure_lagged(&mut g, vid(2), Lag::CONTEMPORANEOUS).unwrap();
        g.insert_directed(m1, y0).unwrap();
    }
    g
}

fn temporal_pag(n_vars: u32) -> TemporalPag {
    let mut g = TemporalPag::empty();
    if n_vars >= 2 {
        let x1 = g.add_lagged(vid(0), Lag::from_raw(1)).unwrap();
        let y0 = g.add_lagged(vid(1), Lag::CONTEMPORANEOUS).unwrap();
        g.insert_directed(x1, y0).unwrap();
    }
    if n_vars >= 3 {
        let m1 = g.add_lagged(vid(1), Lag::from_raw(1)).unwrap();
        let y0 = g.add_lagged(vid(2), Lag::CONTEMPORANEOUS).unwrap();
        g.insert_directed(m1, y0).unwrap();
    }
    g
}

fn static_gp(n_vars: usize) -> GraphPosterior {
    let cell = n_vars * n_vars;
    GraphPosterior::new(
        n_vars,
        vec![1.0],
        vec![0],
        vec![0.0; cell],
        vec![0.0; cell],
        1.0,
        InferenceDiagnostics::analytic("licensed-compiler-static-gp"),
        0,
    )
    .unwrap()
}

fn temporal_gp(n_vars: usize) -> GraphPosterior {
    let cell = n_vars * n_vars;
    let mut lagged = vec![0.0; cell];
    if n_vars >= 2 {
        lagged[1] = 1.0;
    }
    static_gp(n_vars)
        .with_lagged_marginals(1, lagged)
        .unwrap()
        .with_lag_masks(vec![2])
        .unwrap()
}

fn temporal_spec() -> TemporalResponseSpec {
    TemporalResponseSpec::new([1], TemporalPolicy::pulse(-1), None).unwrap()
}

fn mean_curve(treatment: u32, outcome: u32, temporal: bool) -> CausalQuery {
    let q = ResponseQuery::new(ResponseFunctional::MeanCurve {
        outcome: vid(outcome),
        treatment: ContinuousDomain::new(vid(treatment), GridSpec::Values(Arc::from([0.0, 1.0]))),
    });
    CausalQuery::Response(if temporal { q.with_temporal(temporal_spec()) } else { q })
}

fn intervention_response(treatments: &[u32], outcome: u32, temporal: bool) -> CausalQuery {
    let interventions: Arc<[Intervention]> = treatments
        .iter()
        .map(|&t| Intervention::set(vid(t), Value::f64(1.0)))
        .collect();
    let q = ResponseQuery::new(ResponseFunctional::InterventionResponse {
        outcome: vid(outcome),
        interventions,
    });
    CausalQuery::Response(if temporal { q.with_temporal(temporal_spec()) } else { q })
}

fn query_for(cell: &antecedent::SupportCell) -> CausalQuery {
    let temporal_graph = matches!(
        cell.graph_class,
        "TemporalDag" | "TemporalCpdag" | "TemporalPag"
    );
    match cell.query {
        "AverageEffect" => CausalQuery::AverageEffect(AverageEffectQuery::binary_ate(vid(0), vid(1))),
        "ConditionalEffect" => CausalQuery::ConditionalEffect(
            ConditionalEffectQuery::try_new(
                AverageEffectQuery::binary_ate(vid(0), vid(1)).with_effect_modifiers([vid(2)]),
            )
            .unwrap(),
        ),
        "Counterfactual" => CausalQuery::Counterfactual(
            CounterfactualQuery::new(vid(1), Arc::from([Intervention::set(vid(0), Value::f64(1.0))]))
                .with_control_level(0.0),
        ),
        "InterventionalDistribution" => CausalQuery::Distribution(InterventionalDistributionQuery::new(
            vid(1),
            [Intervention::set(vid(0), Value::f64(1.0))],
        )),
        "MediationEffect" | "TemporalMediationEffect" => {
            let (t, m, y) = if temporal_graph { (0, 1, 2) } else { (0, 1, 2) };
            CausalQuery::Mediation(MediationQuery::binary(
                vid(t),
                vid(y),
                Arc::from([vid(m)]),
                MediationContrast::NaturalDirect,
            ))
        }
        "PathSpecificEffect" => CausalQuery::PathSpecific(
            PathSpecificEffectQuery::binary(vid(0), vid(2)).with_path_nodes([vid(1)]),
        ),
        "PulseEffect" => CausalQuery::TemporalEffect(
            TemporalEffectQuery::pulse(vid(0), vid(1), 1.0)
                .with_policy(TemporalPolicy::pulse(-1))
                .with_horizon_steps(1),
        ),
        "SustainedEffect" => CausalQuery::TemporalEffect(
            TemporalEffectQuery::pulse(vid(0), vid(1), 1.0)
                .with_policy(TemporalPolicy::sustained(-1, -1))
                .with_horizon_steps(1),
        ),
        "ResponseCurve" => mean_curve(0, 1, temporal_graph),
        "InterventionResponse" if cell.graph_class == "CoDetermined" => {
            intervention_response(&[1, 2], 3, false)
        }
        "InterventionResponse" => intervention_response(&[0], 1, temporal_graph),
        "AverageDerivative" => CausalQuery::Response(ResponseQuery::new(
            ResponseFunctional::AverageDerivative {
                outcome: vid(2),
                treatment: vid(0),
                weighting: DerivativeWeighting::Observed,
            },
        )),
        "PointDerivative" => CausalQuery::Response(ResponseQuery::new(
            ResponseFunctional::PointDerivative {
                outcome: vid(2),
                treatment: vid(0),
                at: 0.5,
                order: 1,
                scale: DerivativeScale::Identity,
            },
        )),
        "Elasticity" => CausalQuery::Response(ResponseQuery::new(
            ResponseFunctional::PointDerivative {
                outcome: vid(2),
                treatment: vid(0),
                at: 0.5,
                order: 1,
                scale: DerivativeScale::LogLog,
            },
        )),
        "SemiElasticity" => CausalQuery::Response(ResponseQuery::new(
            ResponseFunctional::PointDerivative {
                outcome: vid(2),
                treatment: vid(0),
                at: 0.5,
                order: 1,
                scale: DerivativeScale::LogTreatment,
            },
        )),
        "DirectionalDerivative" => CausalQuery::Response(ResponseQuery::new(
            ResponseFunctional::DirectionalDerivative {
                outcomes: Arc::from([vid(2), vid(3)]),
                treatments: Arc::from([vid(0), vid(1)]),
                at: Arc::from([0.5, 0.0]),
                direction: Arc::from([1.0, 1.0]),
            },
        )),
        "ResponseJacobian" => CausalQuery::Response(ResponseQuery::new(
            ResponseFunctional::Jacobian {
                outcomes: Arc::from([vid(2), vid(3)]),
                treatments: Arc::from([vid(0), vid(1)]),
                at: Arc::from([0.5, 0.0]),
                scale: DerivativeScale::Identity,
            },
        )),
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
            CausalQuery::AnomalyAttribution(AnomalyAttributionQuery::new([vid(1)], 16))
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
    matches!(
        cell.query,
        "PulseEffect" | "SustainedEffect" | "TemporalMediationEffect"
    ) || matches!(cell.graph_class, "TemporalDag" | "TemporalCpdag" | "TemporalPag")
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
        "InterferenceQuery" => 1,
        "InterventionResponse" if cell.graph_class == "CoDetermined" => 4,
        "AverageEffect" if matches!(cell.graph_class, "CoDetermined" | "Unknown") => 4,
        "ConditionalEffect" => 3,
        _ if needs_series(cell) && cell.query == "TemporalMediationEffect" => 3,
        _ if needs_series(cell) => 2,
        _ => 3,
    }
}

fn inspect_cell(cell: antecedent::SupportCell) -> Result<antecedent::CausalContract, String> {
    let expected = cell_coordinate(cell);
    let query = query_for(&cell);
    let n = n_vars(&cell);
    let series = needs_series(&cell);
    let inference = inference_of(cell.inference);
    let refute = refute_of(cell.validation);

    let mut builder = if series {
        let names: Vec<&str> = match n {
            2 => vec!["x", "y"],
            3 => vec!["t", "m", "y"],
            _ => vec!["x", "y", "z", "w"][..n as usize].to_vec(),
        };
        Study::series(series_from(&names))
    } else if cell.graph_class == "CoDetermined" && cell.query == "InterventionResponse" {
        Study::tabular(static_table(&[("z", 0), ("t1", 1), ("t2", 2), ("y", 3)]))
    } else if cell.graph_class == "CoDetermined" {
        Study::tabular(static_table(&[("t", 0), ("y", 1), ("z", 2), ("u", 3)]))
    } else if cell.graph_class == "Unknown" {
        Study::tabular(static_table(&[("t", 0), ("y", 1), ("era", 2), ("m", 3)]))
    } else if cell.query == "TransportQuery" {
        Study::tabular(static_table(&[
            ("a", 0),
            ("y", 1),
            ("trial", 2),
            ("s", 3),
            ("e", 4),
        ]))
    } else if cell.query == "InterferenceQuery" {
        Study::tabular(static_table(&[("y", 0)]))
    } else if matches!(
        cell.query,
        "AverageDerivative"
            | "PointDerivative"
            | "Elasticity"
            | "SemiElasticity"
            | "DirectionalDerivative"
            | "ResponseJacobian"
    ) {
        Study::tabular(static_table(&[("a", 0), ("b", 1), ("y", 2), ("v", 3)]))
    } else if matches!(cell.query, "MediationEffect" | "PathSpecificEffect") {
        Study::tabular(static_table(&[("t", 0), ("m", 1), ("y", 2)]))
    } else {
        Study::tabular(static_table(&[("t", 0), ("y", 1), ("z", 2)]))
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
                    let g = chain_dag(n);
                    if accepted {
                        builder.graph(AcceptedGraph::from(g))
                    } else {
                        builder.graph(g)
                    }
                }
                "Admg" => {
                    let g = if cell.query == "TransportQuery" {
                        transport_admg()
                    } else {
                        confounder_admg()
                    };
                    if accepted {
                        builder.graph(AcceptedGraph::from(g))
                    } else {
                        builder.graph(g)
                    }
                }
                "Cpdag" => {
                    let g = Cpdag::from_dag(&confounder_dag());
                    if accepted {
                        builder.graph(AcceptedGraph::from(g))
                    } else {
                        builder.graph(g)
                    }
                }
                "Pag" => {
                    let g = confounder_pag();
                    if accepted {
                        builder.graph(AcceptedGraph::from(g))
                    } else {
                        builder.graph(g)
                    }
                }
                "TemporalDag" => {
                    let g = temporal_dag(n);
                    if accepted {
                        builder.graph(AcceptedGraph::from(g))
                    } else {
                        builder.graph(g)
                    }
                }
                "TemporalCpdag" => {
                    let g = TemporalCpdag::from_temporal_dag(&temporal_dag(n));
                    if accepted {
                        builder.graph(AcceptedGraph::from(g))
                    } else {
                        builder.graph(g)
                    }
                }
                "TemporalPag" => {
                    let g = temporal_pag(n);
                    if accepted {
                        builder.graph(AcceptedGraph::from(g))
                    } else {
                        builder.graph(g)
                    }
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

    if cell.query == "TransportQuery" {
        builder = builder.selection_targets(Arc::from([])).transport_trial(TransportTrialSpec {
            trial: vid(2),
            selection_probability: vid(3),
            treatment_probability: vid(4),
        });
    }
    if cell.query == "InterferenceQuery" {
        let units = static_table(&[("y", 0)]);
        let network = NetworkData::try_new(
            units,
            [
                NetworkEdge { from: 0, to: 1, weight: 1.0 },
                NetworkEdge { from: 1, to: 0, weight: 1.0 },
            ],
        )
        .map_err(|e| format!("{expected}: {e}"))?;
        builder = builder.interference(InterferenceSpec {
            network,
            assignment: Arc::from(vec![true, false, true, false].repeat(12)),
        });
    }

    let inspected = builder.inspect().map_err(|e| format!("{expected}: inspect {e}"))?;
    if inspected.support_status != Some(CellStatus::Licensed) {
        return Err(format!(
            "{expected}: support_status={:?}",
            inspected.support_status
        ));
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
