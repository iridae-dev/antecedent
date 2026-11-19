//! Lower an independently declared, bounded latent soft-intervention study.

use std::sync::{Arc, OnceLock};

use antecedent_contract::{ClaimKind, DataClassification};
use antecedent_core::{
    ByteEstimates, EvidenceClaims, FrozenFitReuse, Operator, OperatorDeclaration, OperatorEffect,
    OperatorGraph, OperatorPorts, OperatorRole, PhysicalEffect, PhysicalOperator, SemanticDigest,
    VariableId,
};

use crate::analysis::latent_soft_intervention::{LatentSoftInterventionRequest, METHOD, invalid};
use crate::error::CausalError;
use crate::planner::lowered::{
    ClaimDemand, Composition, DesignShape, EvidenceDecisionProgram, EvidenceRole,
    EvidenceSourceRef, FunctionalKind, IdentificationLayer, LatentSoftInterventionLoweredParts,
    LoweredProgram, NonGraphicalIdentification, PhysicalLayer, ProblemTarget, ProgramIdentityScope,
    ScientificProblem, UncertaintyReading,
};
use crate::planner::statistical::{Realization, StatisticalProgram};
use crate::support::CellStatus;

pub(crate) fn release_supported() -> bool {
    #[derive(serde::Deserialize)]
    struct Unit {
        id: String,
        status: String,
    }

    #[derive(serde::Deserialize)]
    struct Snapshot {
        capabilities: Vec<Unit>,
    }

    static APPROVED: OnceLock<bool> = OnceLock::new();
    let snapshot = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../antecedent-io-core/data/unit_support_status.json",
    ));
    *APPROVED.get_or_init(|| {
        serde_json::from_str::<Snapshot>(snapshot).is_ok_and(|snapshot| {
            snapshot.capabilities.iter().any(|unit| unit.id == METHOD && unit.status == "supported")
        })
    })
}

pub(crate) fn lower(
    request: &LatentSoftInterventionRequest,
    source: SemanticDigest,
    source_bytes: u64,
) -> Result<LoweredProgram, CausalError> {
    let variables =
        (0..request.variables.len()).map(|i| VariableId::from_raw(i as u32)).collect::<Vec<_>>();
    let status = if release_supported() { CellStatus::Supported } else { CellStatus::NotSupported };
    let scientific = ScientificProblem {
        runtime_query: None,
        identity_scope: ProgramIdentityScope::Semantic,
        target: ProblemTarget::QueryFunctional {
            kind: FunctionalKind::InterventionalDistribution,
            outcome: None,
            query: *source.as_bytes(),
        },
        plan_id: Arc::from(METHOD),
        data: DataClassification::Tabular,
        graph_class: "Dag",
        query_variables: variables.into(),
        source_digest: Some(*source.as_bytes()),
        support_coordinate: None,
        supported: status == CellStatus::Supported,
        support_status: Some(status),
    };
    let identification = IdentificationLayer::NonGraphical(Arc::new(
        NonGraphicalIdentification::SpecifiedModelConditional { model: *source.as_bytes() },
    ));
    let design = DesignShape::not_design_based(
        "randomized recorded action; exogenous latent version given declared roots",
    );
    let read_rows = Operator::new(
        OperatorDeclaration::new(
            OperatorEffect::Read,
            OperatorGraph::Execution,
            OperatorPorts::new([OperatorRole::Population], [OperatorRole::Outcome]),
        )
        .with_evidence_claims(EvidenceClaims::observation_read()),
    )
    .map_err(invalid)?;
    let fit_shifts = Operator::new(
        OperatorDeclaration::new(
            OperatorEffect::Fit,
            OperatorGraph::Execution,
            OperatorPorts::new([OperatorRole::Outcome], [OperatorRole::FitHandle]),
        )
        .with_evidence_claims(EvidenceClaims::nuisance_fit()),
    )
    .map_err(invalid)?;
    let frozen = Operator::new(
        OperatorDeclaration::new(
            OperatorEffect::ReuseFrozenFit,
            OperatorGraph::Execution,
            OperatorPorts::new([OperatorRole::FitHandle], [OperatorRole::FitHandle]),
        )
        .with_frozen_fit_reuse(FrozenFitReuse::Allowed),
    )
    .map_err(invalid)?;
    let condition_on_context = Operator::new(OperatorDeclaration::new(
        OperatorEffect::Condition,
        OperatorGraph::Causal,
        OperatorPorts::new([OperatorRole::FitHandle], [OperatorRole::Conditioning]),
    ))
    .map_err(invalid)?;
    let statistical = StatisticalProgram {
        identifier: None,
        estimator: None,
        // The response law is evaluated at the retained fit. Whole-fit refits are
        // diagnostics and do not turn this into a parameter sampling-law claim.
        realization: Realization::ExactPoint {
            evaluator: Arc::from("latent_soft_intervention_v1"),
        },
        models: Arc::from([]),
        folds: None,
        resampling: None,
        posterior: None,
        design,
        operators: Arc::from([read_rows, fit_shifts, frozen, condition_on_context]),
    };
    let evidence = EvidenceDecisionProgram {
        sources: Arc::from([EvidenceSourceRef {
            role: EvidenceRole::Trial,
            digest: Some(*source.as_bytes()),
        }]),
        composition: Composition::Single,
        decision: None,
        planning: None,
        chain: None,
    };
    let read = PhysicalOperator::declare(PhysicalEffect::Read)
        .estimates(ByteEstimates::absent().with_output(source_bytes))
        .build()
        .map_err(invalid)?;
    let width = request.variables.len() as u64;
    let rows = (request.baseline.len() + request.action.len()) as u64;
    let fit = PhysicalOperator::declare(PhysicalEffect::Fit)
        .estimates(
            ByteEstimates::absent()
                .with_scratch(rows * (512 + 64 * width))
                .with_retained((1 + u64::from(request.refits)) * (4096 + width * width * 64)),
        )
        .build()
        .map_err(invalid)?;
    let condition =
        PhysicalOperator::declare(PhysicalEffect::Condition).build().map_err(invalid)?;
    let physical = PhysicalLayer { operators: Arc::from([read, fit, condition]), design };
    let claim =
        ClaimDemand { uncertainty: UncertaintyReading::None, kind: Some(ClaimKind::Response) };
    Ok(LatentSoftInterventionLoweredParts {
        scientific,
        identification,
        statistical,
        evidence,
        physical,
        claim,
    }
    .finish())
}
