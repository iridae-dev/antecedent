//! Six-layer lowering of the bounded categorical posterior candidate.
//!
//! The program states the claim it would publish, a posterior, but carries no scientific
//! support: whether an execution is calibrated is decided at run time by coverage records.
use std::sync::Arc;

use antecedent_contract::DataClassification;
use antecedent_core::categorical_candidate_contract::{PRIOR_NAME, admission_bytes};
use antecedent_core::{ByteEstimates, PhysicalEffect, PhysicalOperator, Prior};

use crate::error::CausalError;
use crate::planner::lowered::{
    CategoricalPosteriorLoweredParts, ClaimDemand, Composition, DesignShape,
    EvidenceDecisionProgram, EvidenceRole, EvidenceSourceRef, FunctionalKind, IdentificationLayer,
    LoweredProgram, PhysicalLayer, ProblemTarget, ProgramIdentityScope, ScientificProblem,
    UncertaintyReading,
};
use crate::planner::statistical::{
    ModelRole, ModelSlot, PosteriorAlgorithm, PosteriorPlan, Realization, StatisticalProgram,
};

const CELL: &str = "compiled_categorical_posterior_conjugate_candidate";

fn operator(effect: PhysicalEffect, bytes: Option<u64>) -> Result<PhysicalOperator, CausalError> {
    let mut op = PhysicalOperator::declare(effect);
    if let Some(point) = effect.required_cancellation_point() {
        op = op.cancellation_point(point);
    }
    if let Some(bytes) = bytes {
        op = op.estimates(ByteEstimates::absent().with_output(bytes));
    }
    op.build().map_err(|error| CausalError::Compile { message: error.to_string() })
}

/// Lower a candidate of `rows` rows and `levels` declared levels. The design has an intercept
/// and one dummy column per non-reference level, so its width equals the level count.
pub(crate) fn lower(
    source: [u8; 32],
    query: [u8; 32],
    rows: usize,
    levels: usize,
) -> Result<LoweredProgram, CausalError> {
    let quote = |count: usize| u64::try_from(count).unwrap_or(u64::MAX);
    let design = DesignShape::quoted(quote(rows), quote(levels), false);
    let scientific = ScientificProblem {
        runtime_query: None,
        identity_scope: ProgramIdentityScope::Semantic,
        target: ProblemTarget::QueryFunctional {
            kind: FunctionalKind::Association,
            outcome: None,
            query,
        },
        plan_id: Arc::from(CELL),
        data: DataClassification::Tabular,
        graph_class: "not_applicable",
        query_variables: Arc::from([]),
        source_digest: Some(source),
        support_coordinate: None,
        supported: false,
        support_status: None,
    };
    // An observed-law coefficient is not a causal identification result.
    let identification = IdentificationLayer::NotClaimed { refusal: None };
    let statistical = StatisticalProgram {
        identifier: None,
        estimator: None,
        realization: Realization::CategoricalPosteriorCandidateV1,
        models: Arc::from([ModelSlot {
            role: ModelRole::Outcome,
            family: Arc::from("native.categorical_normal_inverse_gamma"),
            inputs: Arc::from([]),
            output: None,
        }]),
        folds: None,
        resampling: None,
        posterior: Some(PosteriorPlan {
            algorithm: PosteriorAlgorithm::Conjugate,
            draws: 0,
            prior: Prior::new(PRIOR_NAME)
                .map_err(|error| CausalError::Compile { message: error.to_string() })?,
        }),
        design,
        operators: Arc::from(Vec::new()),
    };
    let evidence = EvidenceDecisionProgram {
        sources: Arc::from([EvidenceSourceRef {
            role: EvidenceRole::Observations,
            digest: Some(source),
        }]),
        composition: Composition::Single,
        decision: None,
        planning: None,
        chain: None,
    };
    let physical = PhysicalLayer {
        operators: Arc::from([
            operator(PhysicalEffect::Read, None)?,
            operator(PhysicalEffect::Fit, Some(admission_bytes(rows, levels)))?,
            operator(PhysicalEffect::Condition, None)?,
        ]),
        design,
    };
    let claim = ClaimDemand { uncertainty: UncertaintyReading::Posterior, kind: None };
    Ok(CategoricalPosteriorLoweredParts {
        scientific,
        identification,
        statistical,
        evidence,
        physical,
        claim,
    }
    .finish())
}
