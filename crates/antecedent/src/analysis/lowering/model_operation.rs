//! Lower a fitted-model operation of a prepared study into the common six-layer program.
//!
//! The operation reads the mechanisms the study's mechanism program fits on the study's own
//! graph and table. Its program is keyed by the study's own lowered program, the mechanism
//! program's label and the operation's canonical encoding, so two operations, or one operation
//! on two studies, have different identities while the study's own program is unchanged. Edge
//! strengths and the do-contrast lower to the attribution family (`FittedModelAttribution`,
//! unit `gcm.attribute.mechanism_refit`); the do-samplers, the evaluation and the nested
//! counterfactual to the simulation family (`FittedModelSimulation`, unit
//! `gcm.simulate.mechanism_refit`).
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_contract::DataClassification;
use antecedent_core::{
    ByteEstimates, CanonicalEncode, CanonicalWriter, ModelOperation, PhysicalEffect,
    PhysicalOperator, SemanticDigest,
};
use antecedent_data::{TableView, TabularData};
use antecedent_graph::Dag;

use crate::analysis::supplied_identity::Canonical;
use crate::error::CausalError;
use crate::planner::lowered::{
    ClaimDemand, Composition, DesignShape, EvidenceDecisionProgram, EvidenceRole,
    EvidenceSourceRef, FunctionalKind, IdentificationLayer, LoweredProgram,
    ModelOperationLoweredParts, NonGraphicalIdentification, PhysicalLayer, ProblemTarget,
    ProgramIdentityScope, ScientificProblem, UncertaintyReading,
};
use crate::planner::statistical::{Realization, StatisticalProgram};

/// Domain tag of an operation's question digest.
const QUERY_TAG: &str = "antecedent.model_operation.query";
/// Domain tag of the fitted model's digest.
const MODEL_TAG: &str = "antecedent.model_operation.model.v2";

fn error(message: impl std::fmt::Display) -> CausalError {
    crate::compile_reason!("invalid_argument", "model_operation.lowering: {message}")
}

/// What the operation is lowered under: the study it reads and the mechanism program it fits.
pub(crate) struct OperationSource<'a> {
    /// The study's own lowered program, when it has one.
    pub(crate) study_program: Option<SemanticDigest>,
    /// Label of the mechanism program fitted on the study's graph.
    pub(crate) mechanism_program: &'a str,
    /// The study's graph.
    pub(crate) graph: &'a Dag,
    /// The study's table.
    pub(crate) data: &'a TabularData,
}

fn model_digest(source: &OperationSource<'_>) -> [u8; 32] {
    let mut writer = CanonicalWriter::new(MODEL_TAG);
    writer.text(source.mechanism_program);
    writer.word(source.graph.node_count() as u64);
    let edges = source.graph.directed_edge_pairs_sorted();
    writer.word(edges.len() as u64);
    for (from, to) in edges {
        writer.word(u64::from(from)).word(u64::from(to));
    }
    // The edges index dense nodes; each dense node's label names the variable it fits.
    Canonical::encode(source.graph.nodes(), &mut writer);
    writer.finish()
}

fn query_digest(source: &OperationSource<'_>, operation: &ModelOperation) -> [u8; 32] {
    let mut writer = CanonicalWriter::new(QUERY_TAG);
    match source.study_program {
        Some(program) => {
            writer.word(1).bytes(program.as_bytes());
        }
        None => {
            writer.word(0);
        }
    }
    writer.text(source.mechanism_program);
    operation.encode(&mut writer);
    writer.finish()
}

/// Lower `operation` on the study `source` describes.
///
/// # Errors
///
/// A table whose size overflows the read quote.
pub(crate) fn lower(
    source: &OperationSource<'_>,
    operation: &ModelOperation,
) -> Result<LoweredProgram, CausalError> {
    let attribution = operation.is_attribution();
    let model = model_digest(source);
    let scientific = ScientificProblem {
        runtime_query: None,
        identity_scope: ProgramIdentityScope::Semantic,
        target: ProblemTarget::QueryFunctional {
            kind: if attribution {
                FunctionalKind::Attribution
            } else {
                FunctionalKind::Simulation
            },
            outcome: None,
            query: query_digest(source, operation),
        },
        plan_id: Arc::from(format!("gcm.model_operation.{}", operation.as_str())),
        data: DataClassification::Tabular,
        graph_class: "Dag",
        query_variables: Arc::from(operation.variables()),
        source_digest: None,
        support_coordinate: None,
        supported: false,
        support_status: None,
    };
    let identification = IdentificationLayer::NonGraphical(Arc::new(
        NonGraphicalIdentification::SpecifiedModelConditional { model },
    ));
    let design = DesignShape::not_design_based(
        "mechanisms fitted on the study's declared DAG; the answer is a statement about them",
    );
    let statistical = StatisticalProgram {
        identifier: None,
        estimator: None,
        realization: Realization::ExactPoint {
            evaluator: Arc::from(format!("gcm.model_operation.{}", operation.as_str())),
        },
        models: Arc::from(Vec::new()),
        folds: None,
        resampling: None,
        posterior: None,
        design,
        operators: Arc::from(Vec::new()),
    };
    let evidence = EvidenceDecisionProgram {
        sources: Arc::from(vec![EvidenceSourceRef {
            role: EvidenceRole::Observations,
            digest: None,
        }]),
        composition: Composition::Single,
        decision: None,
        planning: None,
        chain: None,
    };
    let output = (source.data.row_count() as u64)
        .checked_mul(source.graph.node_count() as u64)
        .and_then(|cells| cells.checked_mul(8))
        .ok_or_else(|| error("table size overflow"))?;
    let read = PhysicalOperator::declare(PhysicalEffect::Read)
        .estimates(ByteEstimates::absent().with_output(output))
        .build()
        .map_err(error)?;
    let physical = PhysicalLayer { operators: Arc::from(vec![read]), design };
    let claim = ClaimDemand { uncertainty: UncertaintyReading::None, kind: None };
    Ok(ModelOperationLoweredParts {
        attribution,
        scientific,
        identification,
        statistical,
        evidence,
        physical,
        claim,
    }
    .finish())
}
