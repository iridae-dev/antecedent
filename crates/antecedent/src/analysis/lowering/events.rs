//! Compiler lowering for timestamped event forecasting and dynamic composition.

use std::sync::Arc;

use antecedent_contract::{ClaimKind, DataClassification};
use antecedent_core::{
    ByteEstimates, CancellationPoint, EvidenceClaims, Operator, OperatorDeclaration,
    OperatorEffect, OperatorGraph, OperatorPorts, OperatorRole, PhysicalEffect, PhysicalOperator,
    SemanticDigest,
};

use antecedent_design::DecisionContract;

use crate::analysis::events::release_supported;
use crate::error::CausalError;
use crate::planner::lowered::{
    CarrierDemand, ClaimDemand, Composition, DecisionProgram, DesignShape,
    EventProcessLoweredParts, EventProcessRoute, EvidenceDecisionProgram, EvidenceRole,
    EvidenceSourceRef, FunctionalKind, IdentificationLayer, LoweredProgram,
    NonGraphicalIdentification, PhysicalLayer, ProblemTarget, ProgramIdentityScope,
    ScientificProblem, UncertaintyReading,
};
use crate::planner::statistical::{Realization, StatisticalProgram};
use crate::support::CellStatus;

fn invalid(error: impl std::fmt::Display) -> CausalError {
    crate::compile_reason!(
        "invalid_argument",
        "{}: {error}",
        antecedent_core::reason_details::events::INVALID_REQUEST
    )
}

/// Operations that read an intervened or composed world rather than the observed stream.
const INTERVENED: &[&str] = &[
    "paired_event_worlds",
    "future_event_study",
    "excitation_question",
    "release_law",
    "release_question",
    "release_study",
];

/// Operations that estimate parameters before the program reads its result.
const FITTED: &[&str] = &[
    "fit_and_forecast",
    "refit_and_forecast",
    "assess_forecasts",
    "marked_fit",
    "excitation_challenge",
    "marked_excitation_challenge",
    "release_comparison",
    "marked_release_comparison",
    "release_question",
];

/// Operations whose result is an exact evaluation with no predictive law.
const POINT: &[&str] = &[
    "likelihood",
    "assess_forecasts",
    "future_event_study",
    "count_interval",
    "marked_likelihood",
    "marked_fit",
    "excitation_offspring",
    "marked_offspring",
    "excitation_challenge",
    "marked_excitation_challenge",
    "release_comparison",
    "marked_release_comparison",
    "excitation_question",
    "shock_query",
    "release_study",
];

/// Operations that scan paired worlds or schedules after conditioning.
const SCANNED: &[&str] = &["paired_event_worlds", "future_event_study", "release_study"];

#[allow(clippy::too_many_lines, reason = "one lowering that assembles every program section")]
pub(crate) fn lower(
    route: EventProcessRoute,
    source: SemanticDigest,
    source_bytes: u64,
    operation: &'static str,
    capability: &'static str,
    parent: Option<SemanticDigest>,
    contract: Option<&DecisionContract>,
) -> Result<LoweredProgram, CausalError> {
    let supported = release_supported(capability);
    let design = DesignShape::not_design_based(
        "complete timestamped history and model-conditional exogenous arrivals",
    );
    let scientific = ScientificProblem {
        runtime_query: None,
        identity_scope: ProgramIdentityScope::Semantic,
        target: ProblemTarget::QueryFunctional {
            kind: if INTERVENED.contains(&operation) {
                FunctionalKind::InterventionalDistribution
            } else {
                FunctionalKind::ConditionalObservation
            },
            outcome: None,
            query: *source.as_bytes(),
        },
        plan_id: Arc::from(format!("event_process.{operation}.v1")),
        data: DataClassification::Event,
        graph_class: "graphless",
        query_variables: Arc::from([]),
        source_digest: Some(*source.as_bytes()),
        support_coordinate: None,
        supported,
        support_status: Some(if supported {
            CellStatus::Supported
        } else {
            CellStatus::NotSupported
        }),
    };
    let read = Operator::new(
        OperatorDeclaration::new(
            OperatorEffect::Read,
            OperatorGraph::Execution,
            OperatorPorts::new([OperatorRole::Population], [OperatorRole::Outcome]),
        )
        .with_evidence_claims(EvidenceClaims::observation_read()),
    )
    .map_err(invalid)?;
    let mut operators = vec![read];
    let mut physical = vec![
        PhysicalOperator::declare(PhysicalEffect::Read)
            .estimates(ByteEstimates::absent().with_output(source_bytes))
            .build()
            .map_err(invalid)?,
    ];
    if FITTED.contains(&operation) {
        operators.push(
            Operator::new(
                OperatorDeclaration::new(
                    OperatorEffect::Fit,
                    OperatorGraph::Execution,
                    OperatorPorts::new([OperatorRole::Outcome], [OperatorRole::FitHandle]),
                )
                .with_evidence_claims(EvidenceClaims::nuisance_fit()),
            )
            .map_err(invalid)?,
        );
        physical.push(PhysicalOperator::declare(PhysicalEffect::Fit).build().map_err(invalid)?);
    }
    let condition = Operator::new(OperatorDeclaration::new(
        OperatorEffect::Condition,
        OperatorGraph::Causal,
        OperatorPorts::new([OperatorRole::Outcome], [OperatorRole::Conditioning]),
    ))
    .map_err(invalid)?;
    operators.push(condition);
    physical.push(PhysicalOperator::declare(PhysicalEffect::Condition).build().map_err(invalid)?);
    if SCANNED.contains(&operation) {
        operators.push(
            Operator::new(OperatorDeclaration::new(
                OperatorEffect::Scan,
                OperatorGraph::Causal,
                OperatorPorts::new([OperatorRole::Conditioning], [OperatorRole::WorldOrSchedule]),
            ))
            .map_err(invalid)?,
        );
        physical.push(
            PhysicalOperator::declare(PhysicalEffect::Scan)
                .cancellation_point(CancellationPoint::Scan)
                .build()
                .map_err(invalid)?,
        );
    }
    let point = POINT.contains(&operation);
    let statistical = StatisticalProgram {
        identifier: None,
        estimator: None,
        realization: if point {
            Realization::ExactPoint { evaluator: Arc::from(operation) }
        } else {
            Realization::DeclaredPredictive { evaluator: Arc::from(operation) }
        },
        models: Arc::from([]),
        folds: None,
        resampling: None,
        posterior: None,
        design,
        operators: operators.into(),
    };
    let mut sources = vec![EvidenceSourceRef {
        role: EvidenceRole::Observations,
        digest: Some(*source.as_bytes()),
    }];
    if let Some(parent) = parent {
        sources.push(EvidenceSourceRef {
            role: EvidenceRole::JointScenarios,
            digest: Some(*parent.as_bytes()),
        });
    }
    let decision = contract
        .map(|contract| -> Result<DecisionProgram, CausalError> {
            Ok(DecisionProgram {
                carrier: CarrierDemand::PredictiveLaw,
                utility: DecisionProgram::utility_form(None),
                actions: contract.actions.len(),
                constraints: contract.constraints.len(),
                chance_threshold: contract
                    .constraints
                    .iter()
                    .map(|c| c.min_probability)
                    .fold(0.0, f64::max),
                utility_unit: Some(Arc::from(contract.utility_units.as_str())),
                contract_identity: Some(Arc::from(contract.identity().map_err(invalid)?)),
                information_value: None,
                scenario_source: Some(Arc::from("shared_exogenous_event_paths_and_sde_noise_v1")),
            })
        })
        .transpose()?;
    let composition = if parent.is_some() {
        Composition::Chained { stages: if decision.is_some() { 3 } else { 2 } }
    } else {
        Composition::Single
    };
    let evidence = EvidenceDecisionProgram {
        sources: sources.into(),
        composition,
        decision,
        planning: None,
        chain: None,
    };
    Ok(EventProcessLoweredParts {
        route,
        scientific,
        identification: IdentificationLayer::NonGraphical(Arc::new(
            NonGraphicalIdentification::SpecifiedModelConditional { model: *source.as_bytes() },
        )),
        statistical,
        evidence,
        physical: PhysicalLayer { operators: physical.into(), design },
        claim: ClaimDemand {
            uncertainty: if point {
                UncertaintyReading::None
            } else {
                UncertaintyReading::Predictive
            },
            kind: Some(if INTERVENED.contains(&operation) {
                ClaimKind::Response
            } else {
                ClaimKind::Associational
            }),
        },
    }
    .finish())
}
