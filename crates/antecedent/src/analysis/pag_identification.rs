//! Compile and independently replay direct identification from an oracle PAG.

use std::sync::Arc;

use antecedent_contract::DataClassification;
use antecedent_core::{
    ByteEstimates, ExecutionContext, PhysicalEffect, PhysicalOperator, SemanticDigest, VariableId,
};
pub use antecedent_io::pag_identification::{
    PagIdError, PagIdentificationArtifact, PagIdentificationProblem, PagIdentificationProof,
    PagReplayError, PagResourceStop,
};

use crate::planner::PlanView;
use crate::planner::lowered::{
    ClaimDemand, Composition, DesignShape, EvidenceDecisionProgram, FunctionalKind,
    IdentificationLayer, LoweredProgram, PhysicalLayer, ProblemTarget, ProgramCell,
    ProgramIdentityScope, ScientificProblem, StandaloneCompositionLoweredParts, UncertaintyReading,
};
use crate::planner::statistical::{Realization, StatisticalProgram};
use crate::support::unit::{SupportKey, UnitId, require_program, require_program_unscoped};

const COMPILE: &str = "antecedent.identify.compile_pag_identification";
const CONSUME: &str = "antecedent.identify.consume_pag_identification";
const UNIT: UnitId = UnitId::new("identify.pag_direct");

/// An oracle PAG query whose identification work has yet to execute.
#[derive(Clone, Debug)]
pub struct CompiledPagIdentification {
    problem: PagIdentificationProblem,
    lowered: Arc<LoweredProgram>,
}

/// Lower marginal IDP or conditional CIDP from the declared causal premises.
///
/// Compiling checks cancellation and refuses re-entry from a lowered dispatch. The
/// lowered layers do not depend on `ctx`.
///
/// # Errors
/// The context is cancelled or already inside a lowered dispatch, the graph or query
/// binding is invalid, or compiler support is unavailable.
pub fn compile_pag_identification(
    problem: PagIdentificationProblem,
    ctx: &ExecutionContext,
) -> Result<CompiledPagIdentification, PagReplayError> {
    if ctx.cancellation.is_cancelled() {
        return Err(PagReplayError::Identification(PagIdError::Cancelled));
    }
    if crate::analysis::lowering::refuse_reentry(ctx).is_err() {
        return Err(PagReplayError::Invalid(
            "a lowered dispatch executes its retained program; it does not compile another identification",
        ));
    }
    lower_pag(problem)
}

fn lower_pag(
    problem: PagIdentificationProblem,
) -> Result<CompiledPagIdentification, PagReplayError> {
    require_program_unscoped(&SupportKey::Unit(UNIT), COMPILE)?;
    let identity = problem.identity()?;
    let conditional = !problem.given.is_empty();
    let method = if conditional { "pag.cidp_2022" } else { "pag.idp_2022" };
    let design = DesignShape::quoted(0, u64::from(problem.graph.node_count), true);
    let scientific = ScientificProblem {
        runtime_query: None,
        identity_scope: ProgramIdentityScope::Semantic,
        target: ProblemTarget::QueryFunctional {
            kind: if conditional {
                FunctionalKind::ConditionalObservation
            } else {
                FunctionalKind::InterventionalDistribution
            },
            outcome: problem.outcomes.first().copied().map(VariableId::from_raw),
            query: *identity.as_bytes(),
        },
        plan_id: Arc::from(method),
        data: DataClassification::Tabular,
        graph_class: "oracle_pag_no_selection",
        query_variables: problem
            .actions
            .iter()
            .chain(&problem.outcomes)
            .chain(&problem.given)
            .copied()
            .map(VariableId::from_raw)
            .collect::<Vec<_>>()
            .into(),
        source_digest: Some(*identity.as_bytes()),
        support_coordinate: None,
        supported: false,
        support_status: None,
    };
    let identification = IdentificationLayer::PerExecution {
        reason: "run the complete PAG algorithm under caller resources",
    };
    let statistical = StatisticalProgram {
        identifier: None,
        estimator: None,
        realization: Realization::ExactPoint { evaluator: Arc::from(method) },
        models: Arc::from([]),
        folds: None,
        resampling: None,
        posterior: None,
        design,
        operators: Arc::from([]),
    };
    let evidence = EvidenceDecisionProgram {
        sources: Arc::from([]),
        composition: Composition::Chained { stages: 1 },
        decision: None,
        planning: None,
        chain: None,
    };
    let physical = PhysicalLayer {
        operators: Arc::from([
            PhysicalOperator::declare(PhysicalEffect::Read)
                .estimates(ByteEstimates::absent().with_output(
                    u64::from(problem.graph.node_count) * 64
                        + problem.graph.edges.len() as u64 * 32,
                ))
                .build()
                .map_err(|_| PagReplayError::Invalid("graph read lowering"))?,
            PhysicalOperator::declare(PhysicalEffect::Condition)
                .build()
                .map_err(|_| PagReplayError::Invalid("identification lowering"))?,
        ]),
        design,
    };
    let lowered = Arc::new(
        StandaloneCompositionLoweredParts {
            cell: ProgramCell::DirectPagIdentification,
            scientific,
            identification,
            statistical,
            evidence,
            physical,
            claim: ClaimDemand { uncertainty: UncertaintyReading::None, kind: None },
        }
        .finish(),
    );
    Ok(CompiledPagIdentification { problem, lowered })
}

impl CompiledPagIdentification {
    /// The causal program retaining the oracle graph and query premises.
    #[must_use]
    pub fn lowered(&self) -> &LoweredProgram {
        &self.lowered
    }

    /// Inspect the compiled procedure without running identification.
    #[must_use]
    pub fn plan_view(&self) -> PlanView {
        crate::analysis::program_handle::CompiledProgramHandle::plan_view(self)
    }

    /// Run the direct PAG identification and retain its functional or obstruction.
    ///
    /// # Errors
    /// Invalid graph, resource stop, cancellation or unsupported compiler route.
    pub fn execute(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<PagIdentificationArtifact, PagReplayError> {
        let scoped = ctx.enter_lowered_dispatch(self.lowered.identity());
        require_program(&scoped, &SupportKey::Unit(UNIT), COMPILE)?;
        PagIdentificationArtifact::execute(self.problem.clone(), &scoped)
    }

    /// Run identification and retain a deterministic work or depth stop.
    ///
    /// # Errors
    /// Invalid graph, unsupported route, cancellation or resource admission failure.
    pub fn execute_retaining_stop(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<PagIdentificationArtifact, PagReplayError> {
        let scoped = ctx.enter_lowered_dispatch(self.lowered.identity());
        require_program(&scoped, &SupportKey::Unit(UNIT), COMPILE)?;
        PagIdentificationArtifact::execute_retaining_stop(self.problem.clone(), &scoped)
    }
}

impl crate::analysis::program_handle::CompiledProgramHandle for CompiledPagIdentification {
    fn program(&self) -> &Arc<LoweredProgram> {
        &self.lowered
    }
}

/// Independently verify the graph, route and recorded PAG conclusion.
///
/// # Errors
/// A different result or program identity, changed premises or resource stop.
pub fn consume_pag_identification(
    bytes: &[u8],
    expected_result: &str,
    expected_program: SemanticDigest,
    ctx: &ExecutionContext,
) -> Result<PagIdentificationArtifact, PagReplayError> {
    let scoped = ctx.enter_lowered_dispatch(expected_program);
    require_program(&scoped, &SupportKey::Unit(UNIT), CONSUME)?;
    let result = PagIdentificationArtifact::consume(bytes, Some(expected_result), &scoped)?;
    let reconstructed = lower_pag(result.problem().clone())?;
    if reconstructed.lowered().identity() != expected_program {
        return Err(PagReplayError::Invalid("unexpected retained PAG program"));
    }
    Ok(result)
}
