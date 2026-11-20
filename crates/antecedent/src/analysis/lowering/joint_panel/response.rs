//! Lower a fitted joint panel response: the reviewed multi-channel panel mechanism evaluated
//! over the request's program worlds, invariant or time-varying.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_contract::{ClaimKind, DataClassification};
use antecedent_core::{
    ByteEstimates, CanonicalWriter, FrozenFitReuse, LikelihoodModel, Operator, OperatorDeclaration,
    OperatorEffect, OperatorGraph, OperatorPorts, OperatorRole, OrdinaryBayesianLikelihood,
    PhysicalEffect, PhysicalOperator, Prior, ScoreFrequentist, ScoreModel, SemanticDigest,
    VariableId,
};
use antecedent_estimate::joint_panel::PanelInference;
use antecedent_identify::CompiledIdentificationProduct;

use super::design::{read_and_fit, refuse};
use crate::analysis::joint_panel::JointPanelRequest;
use crate::analysis::joint_panel::compiled::support_status;
use crate::error::CausalError;
use crate::planner::lowered::{
    ClaimDemand, Composition, DesignShape, EvidenceDecisionProgram, EvidenceRole,
    EvidenceSourceRef, IdentificationLayer, JointPanelLoweredParts, JointPanelRoute,
    LoweredProgram, NonGraphicalIdentification, PhysicalLayer, ProblemTarget, ProgramIdentityScope,
    RetainedIdentification, ScientificProblem, UncertaintyReading,
};
use crate::planner::statistical::{
    ModelRole, ModelSlot, PosteriorAlgorithm, PosteriorPlan, Realization, StatisticalProgram,
};
use crate::support::CellStatus;

/// What a panel response program fits instead of one quoted design matrix.
const DESIGN_REASON: &str =
    "a nonlinear multi-channel panel mechanism fitted by its marginal likelihood";

/// Lower the program of a prepared joint panel response.
///
/// The route is `JointPanelTimeVarying` when `request.temporal` declares a piecewise or smooth
/// mechanism and `JointPanelResponse` otherwise. The six layers:
///
/// 1. the request's scientific target over panel data;
/// 2. `identification` retained with the program when it carries an identified claim, and
///    otherwise identification conditional on the specified panel model;
/// 3. the realization of the request's inference: approximate or bootstrap draws are
///    frequentist, a posterior sampler is Bayesian, and no draws is an exact point;
/// 4. the panel source as observations;
/// 5. the read of `source_bytes`, the fit and the evaluation over the worlds;
/// 6. the response claim.
///
/// `source` is the identity of the adapted panel source and `options` the identity of the
/// numerical fit policy. A retarget, a data refit, a static-source proof, an unrolled-graph
/// attempt and a Gaussian restriction each lower again through this function.
///
/// # Errors
/// A program layer the planner refuses to declare.
pub(crate) fn lower(
    request: &JointPanelRequest,
    identification: &CompiledIdentificationProduct,
    source: SemanticDigest,
    options: SemanticDigest,
    source_bytes: u64,
) -> Result<LoweredProgram, CausalError> {
    let (route, plan_id) = if request.temporal.is_some() {
        (JointPanelRoute::TimeVarying, "joint_panel_time_varying")
    } else {
        (JointPanelRoute::Response, "joint_panel_response")
    };
    let status = support_status(route);
    let mechanism = mechanism_identity(request, options)?;
    let columns = &request.columns;
    let scientific = ScientificProblem {
        runtime_query: None,
        identity_scope: ProgramIdentityScope::Semantic,
        target: ProblemTarget::Target(request.target.clone()),
        plan_id: Arc::from(plan_id),
        data: DataClassification::Panel,
        graph_class: "joint_panel_model",
        query_variables: columns.channels.iter().copied().chain([columns.outcome]).collect(),
        source_digest: Some(*source.as_bytes()),
        support_coordinate: None,
        supported: status == CellStatus::Supported,
        support_status: Some(status),
    };
    let identification = if identification.is_identified_claim() {
        IdentificationLayer::Retained(Arc::new(RetainedIdentification::new(
            Some(identification.clone()),
            Vec::new(),
            Some(*source.as_bytes()),
        )?))
    } else {
        let mut writer = CanonicalWriter::new("antecedent.lowering.joint_panel.model.v1");
        writer.bytes(source.as_bytes()).bytes(mechanism.as_bytes());
        IdentificationLayer::NonGraphical(Arc::new(
            NonGraphicalIdentification::SpecifiedModelConditional { model: writer.finish() },
        ))
    };
    let shape = DesignShape::not_design_based(DESIGN_REASON);
    let (realization, posterior, uncertainty) = inference_realization(&request.options.inference)?;
    let [read_rows, fit] = read_and_fit()?;
    let statistical = StatisticalProgram {
        identifier: None,
        estimator: None,
        realization,
        models: Arc::from([ModelSlot {
            role: ModelRole::Mechanism,
            family: Arc::from(format!("joint_panel.mechanism.{mechanism}")),
            inputs: columns
                .channels
                .iter()
                .copied()
                .chain([columns.confounder, columns.season])
                .collect::<Arc<[VariableId]>>(),
            output: Some(columns.outcome),
        }]),
        folds: None,
        resampling: None,
        posterior,
        design: shape,
        operators: Arc::from([read_rows, fit, reuse_frozen_fit()?, evaluate_worlds()?]),
    };
    let evidence = EvidenceDecisionProgram {
        sources: Arc::from([EvidenceSourceRef {
            // A declared whole-program randomized assignment makes the rows a trial.
            role: if request.assignment.is_some() {
                EvidenceRole::Trial
            } else {
                EvidenceRole::Observations
            },
            digest: Some(*source.as_bytes()),
        }]),
        composition: Composition::Single,
        decision: None,
        planning: None,
        chain: None,
    };
    let physical =
        PhysicalLayer { operators: physical_operators(request, source_bytes)?, design: shape };
    let claim = ClaimDemand { uncertainty, kind: Some(ClaimKind::Response) };
    Ok(JointPanelLoweredParts {
        route,
        scientific,
        identification,
        statistical,
        evidence,
        physical,
        claim,
    }
    .finish())
}

/// Digest of the fitted mechanism beyond the source: the numerical fit policy, the mechanism
/// family and the declared time-varying selection procedure.
fn mechanism_identity(
    request: &JointPanelRequest,
    options: SemanticDigest,
) -> Result<SemanticDigest, CausalError> {
    let mut writer = CanonicalWriter::new("antecedent.lowering.joint_panel.mechanism.v1");
    writer
        .bytes(options.as_bytes())
        .bytes(&antecedent_io::to_cbor(&request.family).map_err(refuse)?)
        .word(request.lag_order as u64)
        .word(request.seasons as u64);
    match &request.temporal {
        Some(spec) => writer.word(1).bytes(&antecedent_io::to_cbor(spec).map_err(refuse)?),
        None => writer.word(0),
    };
    Ok(SemanticDigest::from_bytes(writer.finish()))
}

/// The statistical realization, posterior plan and uncertainty reading of `inference`.
fn inference_realization(
    inference: &PanelInference,
) -> Result<(Realization, Option<PosteriorPlan>, UncertaintyReading), CausalError> {
    let frequentist = |name: &str| -> Result<Realization, CausalError> {
        Ok(Realization::Frequentist(ScoreFrequentist::new(ScoreModel::new(name).map_err(refuse)?)))
    };
    Ok(match inference {
        PanelInference::Frequentist { draws: 0 }
        | PanelInference::FrequentistBootstrap { draws: 0 } => (
            Realization::ExactPoint { evaluator: Arc::from("joint_panel.fitted_mean_v1") },
            None,
            UncertaintyReading::None,
        ),
        PanelInference::Frequentist { .. } => (
            frequentist("joint_panel.local_gaussian_parameter_law")?,
            None,
            UncertaintyReading::Sampling,
        ),
        PanelInference::FrequentistBootstrap { .. } => (
            frequentist("joint_panel.hierarchical_parametric_bootstrap")?,
            None,
            UncertaintyReading::Sampling,
        ),
        PanelInference::Bayesian { sampler, .. } => {
            let prior = Prior::new("joint_panel.declared_panel_prior").map_err(refuse)?;
            (
                Realization::Bayesian(OrdinaryBayesianLikelihood::new(
                    LikelihoodModel::new("joint_panel.gaussian_ar1_panel_likelihood")
                        .map_err(refuse)?,
                    prior.clone(),
                )),
                Some(PosteriorPlan {
                    algorithm: PosteriorAlgorithm::Hmc,
                    draws: sampler.chains.saturating_mul(sampler.draws),
                    prior,
                }),
                UncertaintyReading::Posterior,
            )
        }
    })
}

/// Reuse of a retained fit when only the program worlds change.
fn reuse_frozen_fit() -> Result<Operator, CausalError> {
    Operator::new(
        OperatorDeclaration::new(
            OperatorEffect::ReuseFrozenFit,
            OperatorGraph::Execution,
            OperatorPorts::new([OperatorRole::FitHandle], [OperatorRole::FitHandle]),
        )
        .with_frozen_fit_reuse(FrozenFitReuse::Allowed),
    )
    .map_err(refuse)
}

/// Evaluation of the fitted mechanism under every declared program world.
fn evaluate_worlds() -> Result<Operator, CausalError> {
    Operator::new(OperatorDeclaration::new(
        OperatorEffect::Intervene,
        OperatorGraph::Causal,
        OperatorPorts::new([OperatorRole::FitHandle], [OperatorRole::WorldOrSchedule]),
    ))
    .map_err(refuse)
}

/// The read of `source_bytes`, the fit, and the evaluation over the worlds with its quoted
/// response bytes: every world period for the point fit and each requested draw slot.
fn physical_operators(
    request: &JointPanelRequest,
    source_bytes: u64,
) -> Result<Arc<[PhysicalOperator]>, CausalError> {
    let periods =
        request.worlds.iter().map(|world| world.horizon() as u64).fold(0, u64::saturating_add);
    let draws = match &request.options.inference {
        PanelInference::Frequentist { draws } | PanelInference::FrequentistBootstrap { draws } => {
            *draws as u64
        }
        PanelInference::Bayesian { sampler, .. } => {
            sampler.chains.saturating_mul(sampler.draws) as u64
        }
    };
    let responses = periods.saturating_mul(draws.saturating_add(1)).saturating_mul(8);
    Ok(Arc::from([
        PhysicalOperator::declare(PhysicalEffect::Read)
            .estimates(ByteEstimates::absent().with_output(source_bytes))
            .build()
            .map_err(refuse)?,
        PhysicalOperator::declare(PhysicalEffect::Fit).build().map_err(refuse)?,
        PhysicalOperator::declare(PhysicalEffect::Intervene)
            .estimates(ByteEstimates::absent().with_output(responses))
            .build()
            .map_err(refuse)?,
    ]))
}
