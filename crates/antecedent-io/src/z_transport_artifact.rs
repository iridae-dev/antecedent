//! Independent point-result artifacts for single-source z transport.
use crate::{
    IoError, admg_from_wire, admg_to_wire,
    exact_law_wire::ExactLawWire,
    expr_wire::{ExprArenaWire, expr_arena_from_wire, expr_arena_to_wire},
    query_wire::ValueWire,
    transport_catalog_wire::EvidenceCatalogWire,
    wire::AdmgWire,
};
use antecedent_core::{ExecutionContext, InterventionAssignment, VariableId};
use antecedent_expr::{Assignment, ExactDistribution, ExactEvaluationLimits, ExactTransportData};
use antecedent_graph::SelectionDiagram;
use antecedent_identify::{
    ZTransportDerivation, ZTransportDerivationRecord, ZTransportQuery, bind_z_transport_catalog,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Versioned z-transport point execution, including every premise needed for replay.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ZTransportArtifactWire {
    /// Independent format version.
    pub version: u32,
    /// Required feature marker.
    pub required_features: Vec<String>,
    /// Causal graph.
    pub graph: AdmgWire,
    /// Selection targets.
    pub selections: Vec<u32>,
    /// The theorem query.
    pub query: ZTransportQueryWire,
    /// Checked proof premises and expression arena.
    pub proof: ZTransportDerivationRecord,
    /// Expression nodes produced by the checked derivation.
    pub expression: ExprArenaWire,
    /// Evidence catalog binding factor authority.
    pub catalog: EvidenceCatalogWire,
    /// Joint source and target laws with provider snapshots.
    pub laws: Vec<ExactLawWire>,
    /// Required source support budget.
    pub max_support_rows: usize,
    /// Point result recomputed by independent consumers.
    pub result: ZTransportPointWire,
    /// This format never reports an interval.
    pub interval_status: String,
}

/// Portable graph/query coordinates and exact intervention values.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ZTransportQueryWire {
    /// Outcomes.
    pub outcomes: Vec<u32>,
    /// Treatment variables.
    pub treatments: Vec<u32>,
    /// Controllable variables.
    pub controllable: Vec<u32>,
    /// Concrete experiment assignment.
    pub experiment_assignment: Vec<(u32, ValueWire)>,
    /// Source population.
    pub source: String,
    /// Target population.
    pub target: String,
    /// Target effect request.
    pub request: Vec<(u32, ValueWire)>,
    /// Exact evaluator limits.
    pub operation_limit: usize,
    /// Maximum expression depth.
    pub depth_limit: usize,
}

/// Complete point distribution in canonical atom order.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ZTransportPointWire {
    /// Outcome coordinates.
    pub outcomes: Vec<u32>,
    /// Complete outcome assignments.
    pub atoms: Vec<Vec<ValueWire>>,
    /// Point probabilities.
    pub probabilities: Vec<f64>,
}

impl ZTransportArtifactWire {
    /// Validate and construct an artifact from checked premises and a point result.
    pub fn checked(
        diagram: &SelectionDiagram,
        functional: &antecedent_identify::BoundZTransportFunctional,
        data: &ExactTransportData,
        request: &Assignment,
        limits: ExactEvaluationLimits,
        result: &ExactDistribution,
    ) -> Result<Self, IoError> {
        let edge_wire = admg_to_wire(diagram.causal_graph())?;
        let query = functional.derivation().query();
        let wire = Self {
            version: 1,
            required_features: vec!["checked_z_transport_point_v1".into()],
            graph: edge_wire,
            selections: diagram.selection_targets().iter().map(|v| v.raw()).collect(),
            query: ZTransportQueryWire {
                outcomes: query.outcomes.iter().map(|v| v.raw()).collect(),
                treatments: query.treatments.iter().map(|v| v.raw()).collect(),
                controllable: query.controllable.iter().map(|v| v.raw()).collect(),
                experiment_assignment: query
                    .experiment_assignment
                    .iter()
                    .map(|a| (a.variable.raw(), ValueWire::from_value(&a.value)))
                    .collect(),
                source: query.source.to_string(),
                target: query.target.to_string(),
                request: request
                    .entries()
                    .iter()
                    .map(|(v, x)| (v.raw(), ValueWire::from_value(x)))
                    .collect(),
                operation_limit: limits.operations,
                depth_limit: limits.depth,
            },
            proof: functional.derivation().to_record(),
            expression: expr_arena_to_wire(functional.derivation().arena())?,
            catalog: EvidenceCatalogWire::from_catalog(functional.catalog()),
            laws: data.laws().iter().map(ExactLawWire::from_law).collect(),
            max_support_rows: data.max_support_rows(),
            result: ZTransportPointWire {
                outcomes: result.outcomes.iter().map(|v| v.raw()).collect(),
                atoms: result
                    .atoms
                    .iter()
                    .map(|atom| atom.iter().map(ValueWire::from_value).collect())
                    .collect(),
                probabilities: result.probabilities.to_vec(),
            },
            interval_status: "no_interval_reported".into(),
        };
        wire.validate_shape()?;
        Ok(wire)
    }

    fn validate_shape(&self) -> Result<(), IoError> {
        if self.version != 1
            || self.required_features != ["checked_z_transport_point_v1"]
            || self.interval_status != "no_interval_reported"
        {
            return Err(IoError::Convert("unsupported z-transport artifact semantics".into()));
        }
        if self.result.atoms.len() != self.result.probabilities.len()
            || self.result.outcomes != self.query.outcomes
        {
            return Err(IoError::Convert("invalid z-transport point result shape".into()));
        }
        Ok(())
    }

    /// Encode the versioned artifact as CBOR.
    pub fn export(&self) -> Result<Vec<u8>, IoError> {
        crate::to_cbor(self)
    }

    /// Rebuild checked proof and provider objects without executing the point formula.
    pub fn reconstruct(
        bytes: &[u8],
    ) -> Result<
        (
            SelectionDiagram,
            antecedent_identify::BoundZTransportFunctional,
            ExactTransportData,
            Assignment,
            ExactEvaluationLimits,
            Self,
        ),
        IoError,
    > {
        let wire: Self = crate::from_cbor(bytes)?;
        wire.validate_shape()?;
        let graph = admg_from_wire(&wire.graph)?;
        let diagram = SelectionDiagram::try_new(
            graph,
            Arc::<[VariableId]>::from(
                wire.selections.iter().copied().map(VariableId::from_raw).collect::<Vec<_>>(),
            ),
        )
        .map_err(|e| IoError::Convert(e.to_string()))?;
        let q = &wire.query;
        let query = ZTransportQuery {
            outcomes: q
                .outcomes
                .iter()
                .copied()
                .map(VariableId::from_raw)
                .collect::<Vec<_>>()
                .into(),
            treatments: q
                .treatments
                .iter()
                .copied()
                .map(VariableId::from_raw)
                .collect::<Vec<_>>()
                .into(),
            controllable: q
                .controllable
                .iter()
                .copied()
                .map(VariableId::from_raw)
                .collect::<Vec<_>>()
                .into(),
            experiment_assignment: q
                .experiment_assignment
                .iter()
                .map(|(v, x)| InterventionAssignment {
                    variable: VariableId::from_raw(*v),
                    value: x.to_value(),
                })
                .collect::<Vec<_>>()
                .into(),
            source: Arc::from(q.source.as_str()),
            target: Arc::from(q.target.as_str()),
        };
        let arena = expr_arena_from_wire(&wire.expression)?;
        let proof = ZTransportDerivation::from_record_checked(&diagram, &query, &wire.proof, arena)
            .map_err(|e| IoError::Convert(e.to_string()))?;
        let catalog = wire.catalog.to_catalog()?;
        let functional = bind_z_transport_catalog(&diagram, &query, &proof, &catalog)
            .map_err(|e| IoError::Convert(e.to_string()))?;
        let laws = wire.laws.iter().map(ExactLawWire::to_law).collect::<Result<Vec<_>, _>>()?;
        let data = ExactTransportData::try_new(laws, wire.max_support_rows)
            .map_err(|e| IoError::Convert(e.to_string()))?;
        let request = Assignment::from_pairs(
            q.request.iter().map(|(v, x)| (VariableId::from_raw(*v), x.to_value())),
        );
        let limits = ExactEvaluationLimits { operations: q.operation_limit, depth: q.depth_limit };
        Ok((diagram, functional, data, request, limits, wire))
    }

    /// Decode, recheck the proof and evidence bindings, then recompute the point result.
    /// No external provider is accessed.
    pub fn consume(
        bytes: &[u8],
        ctx: &ExecutionContext,
    ) -> Result<(SelectionDiagram, ExactDistribution), IoError> {
        let (diagram, functional, data, request, limits, wire) = Self::reconstruct(bytes)?;
        let plan =
            antecedent_estimate::prepare_exact_z_transport(&functional, data, request, limits, ctx)
                .map_err(|e| IoError::Convert(e.to_string()))?;
        let result = plan.evaluate(ctx).map_err(|e| IoError::Convert(e.to_string()))?;
        let recomputed = ZTransportPointWire {
            outcomes: result.outcomes.iter().map(|v| v.raw()).collect(),
            atoms: result
                .atoms
                .iter()
                .map(|a| a.iter().map(ValueWire::from_value).collect())
                .collect(),
            probabilities: result.probabilities.to_vec(),
        };
        if recomputed != wire.result {
            return Err(IoError::Convert("z-transport point result mismatch".into()));
        }
        Ok((diagram, result))
    }
}
