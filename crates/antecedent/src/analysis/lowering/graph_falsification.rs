//! Lower the permutation falsification of a study's DAG against its retained table into the
//! common six-layer program.
//!
//! The question is a test of the declared structure, not an estimate: the graph's local Markov
//! implications are tested on the rows and their violation count compared with that of random
//! relabellings of the graph's nodes. Nothing is fitted, so the identification layer is the
//! declared graph itself, held as the specified model the verdict is conditional on.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_contract::DataClassification;
use antecedent_core::{
    ByteEstimates, CanonicalWriter, PhysicalEffect, PhysicalOperator, VariableId,
};
use antecedent_data::{TableView, TabularData};
use antecedent_graph::Dag;
use antecedent_model::GraphFalsificationConfig;

use crate::error::CausalError;
use crate::planner::lowered::{
    ClaimDemand, Composition, DesignShape, EvidenceDecisionProgram, EvidenceRole,
    EvidenceSourceRef, FunctionalKind, GraphFalsificationLoweredParts, IdentificationLayer,
    LoweredProgram, NonGraphicalIdentification, PhysicalLayer, ProblemTarget, ProgramIdentityScope,
    ScientificProblem, UncertaintyReading,
};
use crate::planner::statistical::{Realization, StatisticalProgram};

/// Domain tag of the declared graph's digest.
const GRAPH_TAG: &str = "antecedent.graph_falsification.graph";
/// Domain tag of the falsification question's digest.
const QUERY_TAG: &str = "antecedent.graph_falsification.query";
/// Plan id and evaluator of the route.
const PLAN_ID: &str = "gcm.graph_falsification";

fn error(message: impl std::fmt::Display) -> CausalError {
    crate::compile_reason!("invalid_argument", "graph_falsification.lowering: {message}")
}

/// Digest of the declared DAG: its node count and sorted directed edges.
fn graph_digest(graph: &Dag) -> [u8; 32] {
    let mut writer = CanonicalWriter::new(GRAPH_TAG);
    writer.word(graph.node_count() as u64);
    let edges = graph.directed_edge_pairs_sorted();
    writer.word(edges.len() as u64);
    for (from, to) in edges {
        writer.word(u64::from(from)).word(u64::from(to));
    }
    writer.finish()
}

/// Digest of the question: the declared graph and every control of the test.
fn query_digest(graph: [u8; 32], config: &GraphFalsificationConfig) -> [u8; 32] {
    let mut writer = CanonicalWriter::new(QUERY_TAG);
    writer.bytes(&graph);
    writer
        .text(&config.ci_test)
        .real(config.alpha)
        .real(config.significance)
        .word(config.n_permutations as u64)
        .word(config.min_permutations as u64)
        .word(config.max_ci_tests as u64)
        .word(u64::from(config.ci_replicates));
    match config.max_equivalent_share {
        Some(share) => {
            writer.word(1).real(share);
        }
        None => {
            writer.word(0);
        }
    }
    match config.seed {
        Some(seed) => {
            writer.word(1).word(seed);
        }
        None => {
            writer.word(0);
        }
    }
    writer.finish()
}

/// Lower the falsification of `graph` against `data` under `config`.
///
/// # Errors
///
/// A table whose size overflows the read quote.
pub(crate) fn lower(
    graph: &Dag,
    data: &TabularData,
    config: &GraphFalsificationConfig,
) -> Result<LoweredProgram, CausalError> {
    let graph_id = graph_digest(graph);
    let query = query_digest(graph_id, config);
    let nodes = u32::try_from(graph.node_count()).map_err(|_| error("too many nodes"))?;
    let variables: Vec<VariableId> = (0..nodes).map(VariableId::from_raw).collect();
    let scientific = ScientificProblem {
        runtime_query: None,
        identity_scope: ProgramIdentityScope::Semantic,
        target: ProblemTarget::QueryFunctional {
            kind: FunctionalKind::StructureSearch,
            outcome: None,
            query,
        },
        plan_id: Arc::from(PLAN_ID),
        data: DataClassification::Tabular,
        graph_class: "Dag",
        query_variables: Arc::from(variables),
        source_digest: None,
        support_coordinate: None,
        supported: false,
        support_status: None,
    };
    let identification = IdentificationLayer::NonGraphical(Arc::new(
        NonGraphicalIdentification::SpecifiedModelConditional { model: graph_id },
    ));
    let design = DesignShape::not_design_based(
        "declared DAG; its local Markov implications tested against node relabellings",
    );
    let statistical = StatisticalProgram {
        identifier: None,
        estimator: None,
        realization: Realization::ExactPoint { evaluator: Arc::from(PLAN_ID) },
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
    let output = (data.row_count() as u64)
        .checked_mul(graph.node_count() as u64)
        .and_then(|cells| cells.checked_mul(8))
        .ok_or_else(|| error("table size overflow"))?;
    let read = PhysicalOperator::declare(PhysicalEffect::Read)
        .estimates(ByteEstimates::absent().with_output(output))
        .build()
        .map_err(error)?;
    let physical = PhysicalLayer { operators: Arc::from(vec![read]), design };
    let claim = ClaimDemand { uncertainty: UncertaintyReading::None, kind: None };
    Ok(GraphFalsificationLoweredParts {
        scientific,
        identification,
        statistical,
        evidence,
        physical,
        claim,
    }
    .finish())
}
