//! Placeholder graph classes for discovery and posterior studies.

use super::{
    AcceptedGraph, Admg, CausalError, Cpdag, Dag, DataInput, DenseNodeId, Pag, TemporalCpdag,
    TemporalDag, TemporalPag, data_schema,
};

/// Placeholder structure for [`Study::graph`][crate::analysis::execute::Study] when a graph
/// posterior drives the analysis instead.
///
/// This carries only the shape (variable count / modality) needed for logical-plan
/// bookkeeping (row counts, data classification, …). It is never consulted for
/// identification — identification runs per-graph, against the real posterior atoms,
/// inside `execute()`. `n_vars` comes from the supplied [`GraphPosterior`], not from
/// re-inspecting `data`, so it always matches the ensemble the caller discovered.
pub(in crate::analysis) fn stub_accepted_graph_for(
    data: &DataInput,
    n_vars: usize,
    atom_kind: antecedent_discovery::GraphPosteriorAtomKind,
) -> Result<AcceptedGraph, CausalError> {
    match data {
        DataInput::Tabular(_) => {
            let n = u32::try_from(n_vars).map_err(|_| CausalError::Compile {
                message: "too many variables for graph-posterior stub graph".into(),
            })?;
            match atom_kind {
                antecedent_discovery::GraphPosteriorAtomKind::Dag => {
                    Ok(AcceptedGraph::dag(Dag::with_variables(n)))
                }
                antecedent_discovery::GraphPosteriorAtomKind::Cpdag => {
                    AcceptedGraph::cpdag(Cpdag::with_variables(n))
                }
                antecedent_discovery::GraphPosteriorAtomKind::Pag => {
                    Ok(AcceptedGraph::pag(Pag::with_variables(n)))
                }
                antecedent_discovery::GraphPosteriorAtomKind::Admg => {
                    let mut admg = Admg::with_variables(n);
                    if n >= 2 {
                        admg.insert_bidirected(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1))
                            .map_err(CausalError::rejected)?;
                    }
                    Ok(AcceptedGraph::admg(admg))
                }
                antecedent_discovery::GraphPosteriorAtomKind::TemporalAdmg => {
                    let mut admg = Admg::with_variables(n);
                    if n >= 2 {
                        admg.insert_bidirected(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1))
                            .map_err(CausalError::rejected)?;
                    }
                    let temporal =
                        antecedent_graph::TemporalAdmg::from_static_contemporaneous(&admg)
                            .map_err(CausalError::rejected)?;
                    Ok(AcceptedGraph::temporal_admg(temporal))
                }
                _ => Err(CausalError::Unsupported {
                    message: "graph-posterior stub supports only static Dag, Cpdag, Pag, or Admg atoms",
                }),
            }
        }
        DataInput::Temporal(_) | DataInput::Event(_) => match atom_kind {
            antecedent_discovery::GraphPosteriorAtomKind::Dag => {
                Ok(AcceptedGraph::temporal_dag(TemporalDag::empty()))
            }
            antecedent_discovery::GraphPosteriorAtomKind::Cpdag => {
                AcceptedGraph::temporal_cpdag(TemporalCpdag::empty())
            }
            antecedent_discovery::GraphPosteriorAtomKind::Pag => {
                Ok(AcceptedGraph::temporal_pag(TemporalPag::empty()))
            }
            // Lagged temporal-ADMG atoms: the posterior carries the templates; the stub only
            // names their class.
            antecedent_discovery::GraphPosteriorAtomKind::TemporalAdmg => {
                Ok(AcceptedGraph::temporal_admg(antecedent_graph::TemporalAdmg::empty()))
            }
            antecedent_discovery::GraphPosteriorAtomKind::Admg => Err(CausalError::Unsupported {
                message: "Admg graph posteriors require tabular data; a series posterior over \
                          latent-confounded templates is a lagged TemporalAdmg posterior",
            }),
            _ => Err(CausalError::Unsupported {
                message: "graph-posterior stub supports only TemporalDag, TemporalCpdag, \
                          TemporalPag, or lagged TemporalAdmg atoms on temporal/event data",
            }),
        },
        DataInput::MultiEnv(_) | DataInput::Panel(_) => Err(CausalError::Unsupported {
            message: "graph-posterior analysis supports tabular or temporal/event data only",
        }),
    }
}

/// The empty structure of the class a discovery study answers with.
///
/// A discovery study supplies no graph: this edgeless graph of the returned
/// class (CPDAG and PAG stay classes) states the accepted graph class its
/// contract reports. It is never consulted for identification; the discovered
/// structure reaches a downstream study only through its review gate.
///
/// # Errors
///
/// A schema wider than a node id holds.
pub(in crate::analysis) fn discovery_stub_graph(
    data: &DataInput,
    class: antecedent_core::DiscoveredClass,
) -> Result<AcceptedGraph, CausalError> {
    use antecedent_core::DiscoveredClass as C;
    let width = u32::try_from(data_schema(data).len()).map_err(|_| CausalError::Compile {
        message: "too many variables for a discovery stub graph".into(),
    })?;
    match class {
        C::Dag | C::DagPosterior | C::DagEdgeMarginals => {
            Ok(AcceptedGraph::dag(Dag::with_variables(width)))
        }
        // A class posterior is over CPDAGs; a stability table has no class of its own and is
        // given the class of the search it re-runs by `discovery_stub_class` before this call.
        C::Cpdag | C::CpdagPosterior | C::StructureStability => {
            AcceptedGraph::cpdag(Cpdag::with_variables(width))
        }
        C::Pag | C::PagPosterior => Ok(AcceptedGraph::pag(Pag::with_variables(width))),
        C::TemporalDag | C::TemporalDagPosterior => {
            Ok(AcceptedGraph::temporal_dag(TemporalDag::empty()))
        }
        C::TemporalCpdag | C::RegimeTemporalCpdags => {
            AcceptedGraph::temporal_cpdag(TemporalCpdag::empty())
        }
        C::TemporalPag => Ok(AcceptedGraph::temporal_pag(TemporalPag::empty())),
    }
}

/// The class the stub graph of a discovery study states: the class the search returns, or,
/// for a stability search, the class of the search it re-runs.
pub(in crate::analysis) fn discovery_stub_class(
    method: &antecedent_core::DiscoveryMethod,
) -> antecedent_core::DiscoveredClass {
    match method {
        antecedent_core::DiscoveryMethod::BootstrapStability { base, .. } => base.output(),
        // A class posterior over lagged templates is answered on the temporal side.
        antecedent_core::DiscoveryMethod::ClassPosterior { base } if base.is_temporal() => {
            base.output()
        }
        other => other.output(),
    }
}
