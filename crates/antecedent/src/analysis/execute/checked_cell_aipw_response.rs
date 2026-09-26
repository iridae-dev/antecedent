//! Sealed checked execution product for joint-cell AIPW responses on a static
//! DAG or on a CoDetermined tier closure.
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{
    CausalQuery, ExecutionContext, Intervention, ObservationSpec, ResponseFunctional,
    ResponseQuery, TargetPopulation, VariableId,
};
use antecedent_data::TabularData;
use antecedent_estimate::{CellSaturatedAipw, ScoreTable};
use antecedent_expr::{EstimandMethod, IdentifiedEstimand};
use antecedent_graph::{Admg, Dag, TieredBackground, WithinTier};
use antecedent_identify::{IdentificationResult, IdentificationStatus};

use crate::accepted::AcceptedGraph;
use crate::{CausalError, EstimatorId, IdentifierId, RefuteSuite};

/// The structure a joint cell was identified on.
#[derive(Clone, Debug)]
pub(crate) enum CellAipwStructure {
    /// A supplied or accepted static DAG.
    Dag(Dag),
    /// A CoDetermined tier background and the ancestral ADMG it materializes.
    TierClosure { admg: Admg, background: TieredBackground },
}

/// Node count, sorted directed edges, and sorted bidirected edges.
type StructureSignature = (usize, Arc<[(u32, u32)]>, Arc<[(u32, u32)]>);

/// Whether preparation received a graph explicitly or through acceptance.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DagResponseOrigin {
    /// The DAG was supplied directly by the analysis caller.
    Explicit,
    /// The DAG was accepted from a discovery or review step.
    Accepted,
}

/// Complete checked execution contract for `cell.aipw` on a DAG response cell.
///
/// The constructor validates the exact response query, target, method, estimator,
/// graph and validation suite. The operation then owns those values and runs the
/// numerical cell AIPW procedure without consulting a retained Study builder.
#[derive(Clone, Debug)]
pub(crate) struct CheckedCellAipwResponseOperation {
    structure: CellAipwStructure,
    structure_signature: StructureSignature,
    query: ResponseQuery,
    outcome: VariableId,
    treatments: Arc<[VariableId]>,
    requested_arm: u32,
    identification: IdentificationResult,
    estimand: IdentifiedEstimand,
    identifier: IdentifierId,
    estimator: CellSaturatedAipw,
    validation: RefuteSuite,
    origin: DagResponseOrigin,
}

impl CheckedCellAipwResponseOperation {
    /// Validate and seal the licensed joint-cell response route on a static
    /// DAG or a CoDetermined tier closure.
    #[expect(
        clippy::float_cmp,
        reason = "joint binary intervention levels are exact discrete contract values"
    )]
    pub(crate) fn checked(
        structure: &CellAipwStructure,
        query: &ResponseQuery,
        identification: &IdentificationResult,
        estimand: &IdentifiedEstimand,
        identifier: IdentifierId,
        estimator: EstimatorId,
        fold_seed: u64,
        validation: RefuteSuite,
        origin: DagResponseOrigin,
    ) -> Result<Self, CausalError> {
        query.validate().map_err(|error| CausalError::Compile { message: error.to_string() })?;
        let ResponseFunctional::InterventionResponse { outcome, interventions } = &query.functional
        else {
            return Err(CausalError::Unsupported {
                message: "checked cell AIPW requires InterventionResponse",
            });
        };
        if interventions.len() < 2
            || interventions.len() > antecedent_estimate::cell_aipw::MAX_JOINT_BINARY
        {
            return Err(CausalError::Unsupported {
                message: "checked cell AIPW requires two or three joint binary interventions",
            });
        }
        let mut treatments = Vec::with_capacity(interventions.len());
        let mut requested_arm = 0_u32;
        for (index, intervention) in interventions.iter().enumerate() {
            let Intervention::Set { variable, value } = intervention else {
                return Err(CausalError::Unsupported {
                    message: "checked cell AIPW supports hard binary Set interventions only",
                });
            };
            let Some(level) = value.as_f64() else {
                return Err(CausalError::Unsupported {
                    message: "checked cell AIPW requires numeric binary Set levels",
                });
            };
            if level != 0.0 && level != 1.0 {
                return Err(CausalError::Unsupported {
                    message: "checked cell AIPW requires Set levels 0 or 1",
                });
            }
            if treatments.contains(variable) {
                return Err(CausalError::Unsupported {
                    message: "checked cell AIPW requires distinct intervention coordinates",
                });
            }
            requested_arm |= u32::from(level == 1.0) << index;
            treatments.push(*variable);
        }
        let valid_suite =
            matches!(validation, RefuteSuite::None | RefuteSuite::Cheap | RefuteSuite::Full);
        let (expected_identifier, structure_ok) = match structure {
            CellAipwStructure::Dag(graph) => (
                IdentifierId::ResponseBackdoor,
                graph_contains(graph, *outcome, &treatments, &estimand.adjustment_set),
            ),
            CellAipwStructure::TierClosure { admg, background } => (
                IdentifierId::GeneralizedAdjustment,
                background.within_tier == WithinTier::CoDetermined
                    && origin == DagResponseOrigin::Explicit
                    && admg_contains(admg, *outcome, &treatments, &estimand.adjustment_set),
            ),
        };
        if query.temporal.is_some()
            || query.observation != ObservationSpec::Complete
            || query.target_population != TargetPopulation::AllObserved
            || identifier != expected_identifier
            || estimator != EstimatorId::CellAipw
            || !valid_suite
            || identification.query != CausalQuery::Response(query.clone())
            || identification.status != IdentificationStatus::NonparametricallyIdentified
            || identification.estimands.is_empty()
            || identification.estimands.first().is_none_or(|candidate| {
                candidate.method != estimand.method
                    || candidate.adjustment_set != estimand.adjustment_set
                    || candidate.instruments != estimand.instruments
                    || candidate.mediators != estimand.mediators
                    || candidate.functional != estimand.functional
                    || candidate.rd_design != estimand.rd_design
            })
            || estimand.method_kind().ok() != Some(EstimandMethod::BackdoorAdjustment)
            || identification
                .estimands
                .iter()
                .any(|candidate| candidate.adjustment_set != estimand.adjustment_set)
            || !structure_ok
        {
            return Err(CausalError::Compile {
                message: "cell AIPW query, graph, target, identification, procedure, or validation suite do not form one checked route".into(),
            });
        }
        let graph_replay = match structure {
            CellAipwStructure::Dag(graph) => crate::strategy_table::identify_static_query(
                identifier,
                graph,
                &CausalQuery::Response(query.clone()),
            )?,
            CellAipwStructure::TierClosure { admg, background } => {
                antecedent_identify::identify_tiered_joint_on(background, admg, query)?
            }
        };
        if graph_replay.status != identification.status
            || graph_replay.estimands.len() != identification.estimands.len()
            || graph_replay
                .estimands
                .iter()
                .zip(&identification.estimands)
                .any(|(expected, supplied)| !same_estimand(expected, supplied))
        {
            return Err(CausalError::Compile {
                message: "cell AIPW identification does not replay on the retained DAG and query"
                    .into(),
            });
        }
        Ok(Self {
            structure: structure.clone(),
            structure_signature: structure_signature(structure),
            query: query.clone(),
            outcome: *outcome,
            treatments: treatments.into(),
            requested_arm,
            identification: identification.clone(),
            estimand: estimand.clone(),
            identifier,
            estimator: CellSaturatedAipw::new().with_fold_seed(fold_seed),
            validation,
            origin,
        })
    }

    pub(crate) fn query(&self) -> &ResponseQuery {
        &self.query
    }

    pub(crate) fn origin(&self) -> DagResponseOrigin {
        self.origin
    }

    pub(crate) fn validation(&self) -> RefuteSuite {
        self.validation
    }

    pub(crate) fn procedure(&self) -> (IdentifierId, EstimatorId) {
        (self.identifier, EstimatorId::CellAipw)
    }

    pub(crate) fn target(&self) -> (&IdentificationResult, &IdentifiedEstimand) {
        (&self.identification, &self.estimand)
    }

    pub(crate) fn requested_arm(&self) -> u32 {
        self.requested_arm
    }

    /// The tier closure this cell was sealed on, when it is not a plain DAG route.
    pub(crate) fn within_tier(&self) -> Option<WithinTier> {
        match &self.structure {
            CellAipwStructure::Dag(_) => None,
            CellAipwStructure::TierClosure { background, .. } => Some(background.within_tier),
        }
    }

    /// Verify that a study still carries the structure fixed at preparation:
    /// the same DAG, or the same tier background and its materialized closure.
    pub(crate) fn matches_structure(
        &self,
        graph: &AcceptedGraph,
        tiered: Option<&TieredBackground>,
    ) -> bool {
        let current = match (&self.structure, graph.as_dag(), graph.as_admg(), tiered) {
            (CellAipwStructure::Dag(_), Some(dag), None, None) => {
                structure_signature(&CellAipwStructure::Dag(dag.clone()))
            }
            (
                CellAipwStructure::TierClosure { background, .. },
                None,
                Some(admg),
                Some(current_background),
            ) if current_background == background => {
                structure_signature(&CellAipwStructure::TierClosure {
                    admg: admg.clone(),
                    background: background.clone(),
                })
            }
            _ => return false,
        };
        current == self.structure_signature
            && structure_signature(&self.structure) == self.structure_signature
    }

    /// Validate replacement rows against the frozen coordinates, then return
    /// this same procedure and query for refresh execution.
    pub(crate) fn refresh(
        &self,
        graph: &AcceptedGraph,
        tiered: Option<&TieredBackground>,
        data: &TabularData,
    ) -> Result<Self, CausalError> {
        if !self.matches_structure(graph, tiered) {
            return Err(CausalError::Compile {
                message:
                    "refreshed cell AIPW structure differs from the checked graph or tier closure"
                        .into(),
            });
        }
        let mut variables = self.treatments.to_vec();
        variables.push(self.outcome);
        variables.extend(self.estimand.adjustment_set.iter().copied());
        let mask = data.complete_case_mask(&variables).map_err(CausalError::from)?;
        if !mask.iter().any(|keep| *keep) {
            return Err(CausalError::Compile {
                message: "refreshed cell AIPW data has no complete-case rows".into(),
            });
        }
        Ok(self.clone())
    }

    /// Execute the frozen cross-fitted cell AIPW procedure.
    pub(crate) fn execute(
        &self,
        data: &TabularData,
        ctx: &ExecutionContext,
    ) -> Result<ScoreTable, CausalError> {
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled {
                stage: crate::analysis::stage::STAGE_ESTIMATE_POINT,
            });
        }
        self.estimator
            .fit_scores(
                data,
                &self.treatments,
                self.outcome,
                &self.estimand.adjustment_set,
                &self.query.outcome_functional,
                None,
            )
            .map_err(CausalError::from)
    }
}

fn structure_signature(structure: &CellAipwStructure) -> StructureSignature {
    match structure {
        CellAipwStructure::Dag(graph) => {
            let mut edges =
                graph.edges().map(|edge| (edge.a.raw(), edge.b.raw())).collect::<Vec<_>>();
            edges.sort_unstable();
            (graph.node_count(), edges.into(), Arc::from([]))
        }
        CellAipwStructure::TierClosure { admg, .. } => {
            let count = admg.node_count();
            let mut directed = Vec::new();
            let mut bidirected = Vec::new();
            for raw in 0..u32::try_from(count).unwrap_or(u32::MAX) {
                let node = antecedent_graph::DenseNodeId::from_raw(raw);
                directed.extend(admg.children(node).iter().map(|child| (raw, child.raw())));
                bidirected.extend(
                    admg.bidirected_neighbors(node)
                        .iter()
                        .map(|other| (raw.min(other.raw()), raw.max(other.raw()))),
                );
            }
            directed.sort_unstable();
            directed.dedup();
            bidirected.sort_unstable();
            bidirected.dedup();
            (count, directed.into(), bidirected.into())
        }
    }
}

/// Every role variable is a node of the closure ADMG and the roles are disjoint.
fn admg_contains(
    admg: &Admg,
    outcome: VariableId,
    treatments: &[VariableId],
    adjustment: &[VariableId],
) -> bool {
    let count = admg.node_count();
    std::iter::once(outcome)
        .chain(treatments.iter().copied())
        .chain(adjustment.iter().copied())
        .all(|variable| usize::try_from(variable.raw()).is_ok_and(|raw| raw < count))
        && treatments.iter().all(|treatment| treatment != &outcome)
        && treatments.iter().all(|treatment| !adjustment.contains(treatment))
        && !adjustment.contains(&outcome)
}

fn same_estimand(left: &IdentifiedEstimand, right: &IdentifiedEstimand) -> bool {
    left.method == right.method
        && left.adjustment_set == right.adjustment_set
        && left.instruments == right.instruments
        && left.mediators == right.mediators
        && left.functional == right.functional
        && left.rd_design == right.rd_design
}

fn graph_contains(
    graph: &Dag,
    outcome: VariableId,
    treatments: &[VariableId],
    adjustment: &[VariableId],
) -> bool {
    let count = graph.node_count();
    std::iter::once(outcome)
        .chain(treatments.iter().copied())
        .chain(adjustment.iter().copied())
        .all(|variable| usize::try_from(variable.raw()).is_ok_and(|raw| raw < count))
        && treatments.iter().all(|treatment| treatment != &outcome)
        && treatments.iter().all(|treatment| !adjustment.contains(treatment))
        && !adjustment.contains(&outcome)
        && graph.nodes().iter().all(|node| match node {
            antecedent_graph::NodeRef::Static(variable) => {
                usize::try_from(variable.raw()).is_ok_and(|raw| raw < count)
            }
            _ => false,
        })
        && graph.edges().all(|edge| edge.a.as_usize() < count && edge.b.as_usize() < count)
        && graph.edges().all(|edge| edge.at_a != antecedent_graph::Endpoint::Arrow)
}

#[cfg(test)]
mod tests {
    use super::*;
    use antecedent_core::{CausalQuery, Intervention, ResponseFunctional, Value};
    use antecedent_data::{TableView, TabularData};
    use antecedent_graph::DenseNodeId;

    fn fixture() -> (Dag, ResponseQuery, TabularData, IdentificationResult) {
        let n = 1200;
        let mut t1 = Vec::with_capacity(n);
        let mut t2 = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);
        for row in 0..n {
            let first = (row % 2) as f64;
            let second = ((row / 2) % 2) as f64;
            t1.push(first);
            t2.push(second);
            y.push(
                1.5 + 2.0 * first
                    + 3.0 * second
                    + 4.0 * first * second
                    + 0.05 * (row as f64 * 0.31).sin(),
            );
        }
        let data = TabularData::from_f64_columns([
            ("t1", t1.as_slice()),
            ("t2", t2.as_slice()),
            ("y", y.as_slice()),
        ])
        .unwrap();
        let t1_id = data.schema().id_of("t1").unwrap();
        let t2_id = data.schema().id_of("t2").unwrap();
        let y_id = data.schema().id_of("y").unwrap();
        let query = ResponseQuery::new(ResponseFunctional::InterventionResponse {
            outcome: y_id,
            interventions: Arc::from([
                Intervention::set(t1_id, Value::f64(1.0)),
                Intervention::set(t2_id, Value::f64(1.0)),
            ]),
        });
        let mut graph = Dag::with_variables(3);
        graph
            .insert_directed(DenseNodeId::from_raw(t1_id.raw()), DenseNodeId::from_raw(y_id.raw()))
            .unwrap();
        graph
            .insert_directed(DenseNodeId::from_raw(t2_id.raw()), DenseNodeId::from_raw(y_id.raw()))
            .unwrap();
        let identifier = crate::strategy_table::IdentifierId::ResponseBackdoor;
        let identification = crate::strategy_table::identify_static_query(
            identifier,
            &graph,
            &CausalQuery::Response(query.clone()),
        )
        .unwrap();
        (graph, query, data, identification)
    }

    #[test]
    fn sealed_joint_cell_route_executes_known_truth_and_refreshes_same_target() {
        let (graph, query, data, identification) = fixture();
        let estimand = identification.estimands.first().unwrap().clone();
        let operation = CheckedCellAipwResponseOperation::checked(
            &CellAipwStructure::Dag(graph.clone()),
            &query,
            &identification,
            &estimand,
            IdentifierId::ResponseBackdoor,
            EstimatorId::CellAipw,
            817,
            RefuteSuite::Full,
            DagResponseOrigin::Accepted,
        )
        .unwrap();
        assert_eq!(operation.origin(), DagResponseOrigin::Accepted);
        assert_eq!(operation.validation(), RefuteSuite::Full);
        assert_eq!(operation.procedure(), (IdentifierId::ResponseBackdoor, EstimatorId::CellAipw));
        assert_eq!(operation.query(), &query);
        assert_eq!(operation.requested_arm(), 3);
        let frozen_target = operation.target().1.adjustment_set.clone();
        let scores = operation.execute(&data, &ExecutionContext::for_tests(817)).unwrap();
        let (summary, _, _) = antecedent_estimate::summarize_functional(&scores, None).unwrap();
        let cell = scores.columns.iter().position(|column| column.arm == 3).unwrap();
        assert!((summary.means[cell] - 10.5).abs() < 0.2);

        let refreshed =
            operation.refresh(&AcceptedGraph::from(graph.clone()), None, &data).unwrap();
        assert_eq!(refreshed.target().1.adjustment_set, frozen_target);
        assert!(
            (antecedent_estimate::summarize_functional(
                &refreshed.execute(&data, &ExecutionContext::for_tests(817)).unwrap(),
                None,
            )
            .unwrap()
            .0
            .means[cell]
                - summary.means[cell])
                .abs()
                < 1e-12
        );
    }

    #[test]
    fn sealed_joint_cell_route_refuses_a_changed_graph_and_bad_interventions() {
        let (graph, query, _, identification) = fixture();
        let estimand = identification.estimands.first().unwrap().clone();
        let operation = CheckedCellAipwResponseOperation::checked(
            &CellAipwStructure::Dag(graph.clone()),
            &query,
            &identification,
            &estimand,
            IdentifierId::ResponseBackdoor,
            EstimatorId::CellAipw,
            1,
            RefuteSuite::None,
            DagResponseOrigin::Explicit,
        )
        .unwrap();
        let mut wrong_graph = graph.clone();
        wrong_graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        assert!(!operation.matches_structure(&AcceptedGraph::from(wrong_graph), None));
        assert!(operation.matches_structure(&AcceptedGraph::from(graph.clone()), None));

        let treatment = match &query.functional {
            ResponseFunctional::InterventionResponse { interventions, .. } => {
                match interventions[0] {
                    Intervention::Set { variable, .. } => variable,
                    _ => unreachable!(),
                }
            }
            _ => unreachable!(),
        };
        let bad_query = ResponseQuery::new(ResponseFunctional::InterventionResponse {
            outcome: match query.functional {
                ResponseFunctional::InterventionResponse { outcome, .. } => outcome,
                _ => unreachable!(),
            },
            interventions: Arc::from([
                Intervention::set(treatment, Value::f64(0.5)),
                Intervention::set(VariableId::from_raw(1), Value::f64(1.0)),
            ]),
        });
        let bad_identification = crate::strategy_table::identify_static_query(
            IdentifierId::ResponseBackdoor,
            &graph,
            &CausalQuery::Response(bad_query.clone()),
        )
        .unwrap();
        let err = CheckedCellAipwResponseOperation::checked(
            &CellAipwStructure::Dag(graph.clone()),
            &bad_query,
            &bad_identification,
            bad_identification.estimands.first().unwrap(),
            IdentifierId::ResponseBackdoor,
            EstimatorId::CellAipw,
            1,
            RefuteSuite::None,
            DagResponseOrigin::Explicit,
        )
        .unwrap_err();
        assert!(err.to_string().contains("Set levels 0 or 1"));
    }
}
