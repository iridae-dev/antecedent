//! Lower the declared-shift MNAR candidate into the common six-layer program.
//! The candidate is model-conditional; its interval is released by the measured consumer.

use std::sync::Arc;

use antecedent_contract::DataClassification;
use antecedent_core::mnar_fixed_shift::{FixedShiftProgram, FixedShiftSource};
use antecedent_core::{ByteEstimates, PhysicalEffect, PhysicalOperator, VariableId};

use crate::error::CausalError;
use crate::planner::lowered::{
    ClaimDemand, Composition, DesignShape, EvidenceDecisionProgram, EvidenceRole,
    EvidenceSourceRef, FunctionalKind, IdentificationLayer, LoweredProgram,
    MnarFixedShiftLoweredParts, NonGraphicalIdentification, PhysicalLayer, ProblemTarget,
    ProgramIdentityScope, ScientificProblem, UncertaintyReading,
};
use crate::planner::statistical::{Realization, StatisticalProgram};

fn error(message: impl std::fmt::Display) -> CausalError {
    CausalError::Compile { message: format!("mnar_fixed_shift.lowering: {message}") }
}

/// Reconstruct the bounded source and lower its native operator graph. The declared
/// randomization and missingness assumptions remain conditional, not verified facts.
pub(crate) fn lower(source: &FixedShiftSource) -> Result<LoweredProgram, CausalError> {
    let program = FixedShiftProgram::compile(source).map_err(error)?;
    let source_id = source.source_digest();
    let program_id = *program.identity();
    let variables: Arc<[VariableId]> = Arc::from(vec![
        VariableId::from_raw(source.treatment()),
        VariableId::from_raw(source.outcome()),
    ]);
    let scientific = ScientificProblem {
        runtime_query: None,
        identity_scope: ProgramIdentityScope::Semantic,
        target: ProblemTarget::QueryFunctional {
            kind: FunctionalKind::AssignmentContrast,
            outcome: Some(VariableId::from_raw(source.outcome())),
            query: program_id,
        },
        plan_id: Arc::from("mnar.fixed_treated_shift"),
        data: DataClassification::Tabular,
        graph_class: "none",
        query_variables: variables,
        source_digest: Some(source_id),
        support_coordinate: None,
        licensed: false,
        support_status: None,
    };
    let identification = IdentificationLayer::NonGraphical(Arc::new(
        NonGraphicalIdentification::SpecifiedModelConditional { model: program_id },
    ));
    let design =
        DesignShape::not_design_based("declared independent arms; no fitted adjustment design");
    let statistical = StatisticalProgram {
        identifier: None,
        estimator: None,
        realization: Realization::ExactPoint { evaluator: Arc::from("mnar.fixed_treated_shift") },
        models: Arc::from(Vec::new()),
        folds: None,
        resampling: None,
        posterior: None,
        design,
        operators: Arc::from(program.operators().to_vec()),
    };
    let evidence = EvidenceDecisionProgram {
        sources: Arc::from(vec![EvidenceSourceRef {
            role: EvidenceRole::Trial,
            digest: Some(source_id),
        }]),
        composition: Composition::Single,
        decision: None,
        planning: None,
        chain: None,
    };
    let output = (source.rows().len() as u64)
        .checked_mul(32)
        .ok_or_else(|| error("source size overflow"))?;
    let read = PhysicalOperator::declare(PhysicalEffect::Read)
        .estimates(ByteEstimates::absent().with_output(output))
        .build()
        .map_err(error)?;
    let physical = PhysicalLayer { operators: Arc::from(vec![read]), design };
    let claim = ClaimDemand { uncertainty: UncertaintyReading::None, kind: None };
    Ok(MnarFixedShiftLoweredParts {
        scientific,
        identification,
        statistical,
        evidence,
        physical,
        claim,
    }
    .finish())
}
