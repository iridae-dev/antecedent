//! Graph-selection, missing-row imputation, child effect fitting and utility.
//!
//! The graph-imputation compiler owns the whole upstream schedule and its
//! native replay. This outer compiler binds that retained program and an exact
//! binary average-effect decision before execution. Its posterior law is a
//! model-conditional causal-functional mixture, never a predictive outcome
//! law or an earned sampling confidence distribution.

use std::sync::Arc;

use antecedent_contract::dependence::SharedFactorKind;
use antecedent_contract::propagation_chain::{
    ChainReading, LawMeaning, StageKind, TableConstruction,
};
use antecedent_contract::{ClaimKind, DataClassification, QuantityRole, ScientificQuantity};
use antecedent_core::graph_imputation_contract::GraphImputationSource;
use antecedent_core::{
    ByteEstimates, CalibrationBasis, CanonicalWriter, ExecutionContext, PhysicalEffect,
    PhysicalOperator, SemanticDigest,
};
use antecedent_design::decision_artifact::DecisionContractArtifact;
use antecedent_design::decision_contract::{ActionKind, DecisionContract, DecisionContractError};
use antecedent_io::IoError;
use antecedent_io::measured_inference::graph_candidate::{
    GraphPairedWorldReading, TransportStandardization, replay_bernoulli_logit_predictive_worlds,
    replay_gaussian_target_predictive_worlds, replay_paired_worlds,
    replay_target_predictive_cluster_bootstrap_worlds, replay_transport_paired_worlds,
    replay_transport_paired_worlds_bootstrap,
};
use antecedent_io::propagation_chain_record::{
    ChainCoefficientWire, ChainJointOutputWire, ChainOpWire, ChainSourceWire, ChainStageWire,
    PropagationChainRecordWire,
};
use antecedent_io::quantity_wire::ScientificQuantityWire;
use serde::{Deserialize, Serialize};

use super::graph_imputation::{
    CompiledGraphImputation, GraphImputationError, GraphImputationExpectation,
    GraphImputationPoint, consume as consume_graph_imputation,
};
use super::graph_imputation_transport_decision::{TargetClusterBootstrap, TargetFrameBootstrap};
use super::propagation_chain_decision::{
    ChainDecisionError, ChainDecisionResult, compile_action_world_chain_decision,
    compile_chain_decision, consume_chain_decision,
};
use crate::planner::lowered::{
    CarrierDemand, ClaimDemand, Composition, DecisionProgram, DesignShape, EvidenceDecisionProgram,
    EvidenceRole, EvidenceSourceRef, FunctionalKind, IdentificationLayer, LoweredProgram,
    NonGraphicalIdentification, PhysicalLayer, ProblemTarget, ProgramCell, ProgramIdentityScope,
    ScientificProblem, StandaloneCompositionLoweredParts, UncertaintyReading,
};
use crate::planner::statistical::{
    Realization, ResampleUnit, ResamplingMethod, ResamplingPlan, StatisticalProgram,
};
use crate::planner::{PlanStage, PlanView};

const FEATURE: &str = "graph_imputation_effect_decision_v1";
const MAX_ARTIFACT_BYTES: usize = 80 * 1024 * 1024;

/// Refusal of an upstream mixture, downstream contract or portable replay.
#[derive(Debug, thiserror::Error)]
pub enum GraphImputationEffectDecisionError {
    /// A coordinate, carrier or artifact identity is invalid.
    #[error("graph imputation effect decision: {0}")]
    Invalid(&'static str),
    /// The graph-imputation compiler or native consumer refused.
    #[error("graph imputation effect producer: {0}")]
    Producer(#[from] GraphImputationError),
    /// Decision contract refused.
    #[error("graph imputation effect contract: {0}")]
    Contract(#[from] DecisionContractError),
    /// Chain or utility compiler refused.
    #[error("graph imputation effect chain: {0}")]
    Chain(#[from] ChainDecisionError),
    /// Artifact encoding or decoding refused.
    #[error("graph imputation effect artifact: {0}")]
    Io(#[from] IoError),
}

/// The retained graph-imputation producer and the exact effect decision.
pub struct CompiledGraphImputationEffectDecision {
    producer: CompiledGraphImputation,
    quantity: ScientificQuantity,
    decision: DecisionContract,
    lowered: Arc<LoweredProgram>,
}

/// Whole upstream effect mixture with downstream utility on its aligned draws.
pub struct GraphImputationEffectDecisionResult {
    /// Native graph-imputation fit and effect draw receipt.
    pub producer: GraphImputationPoint,
    /// Chain and multi-action utility over the effect posterior.
    pub decision: ChainDecisionResult,
    /// Full outer compiler identity.
    pub lowered_program_identity: SemanticDigest,
    /// No whole decision reliability record has been earned.
    pub calibration: &'static str,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct GraphImputationEffectDecisionWire {
    version: u32,
    required_feature: String,
    producer_source: Vec<u8>,
    producer_program: [u8; 32],
    producer_data: [u8; 32],
    producer_receipt: [u8; 32],
    quantity: ScientificQuantityWire,
    decision_identity: String,
    decision_artifact: Vec<u8>,
    chain_artifact: Vec<u8>,
    program_identity: [u8; 32],
    calibration: String,
}

fn check_decision(
    quantity: &ScientificQuantity,
    decision: &DecisionContract,
) -> Result<(), GraphImputationEffectDecisionError> {
    decision.validate()?;
    if quantity.population_id != decision.target_population
        || quantity.horizon != decision.horizon
        || decision.actions.is_empty()
        || decision.actions.iter().any(|action| {
            action.inputs.len() != 1 || action.inputs[0].require_same_coordinate(quantity).is_err()
        })
    {
        return Err(GraphImputationEffectDecisionError::Invalid(
            "every action must read the exact graph-mixture effect coordinate",
        ));
    }
    Ok(())
}

fn lower_outer(
    producer: SemanticDigest,
    source: SemanticDigest,
    quantity: &ScientificQuantity,
    decision: &DecisionContract,
) -> Result<Arc<LoweredProgram>, GraphImputationEffectDecisionError> {
    check_decision(quantity, decision)?;
    let quantity_bytes = antecedent_io::to_cbor(&ScientificQuantityWire::from(quantity))?;
    let decision_identity = decision.identity()?;
    let mut writer =
        CanonicalWriter::new("antecedent.graph_imputation_effect_decision.question.v1");
    writer
        .bytes(producer.as_bytes())
        .bytes(source.as_bytes())
        .bytes(&quantity_bytes)
        .text(&decision_identity);
    let question = writer.finish();
    let design = DesignShape::quoted(1, 1, true);
    let scientific = ScientificProblem {
        runtime_query: None,
        identity_scope: ProgramIdentityScope::Semantic,
        target: ProblemTarget::QueryFunctional {
            kind: FunctionalKind::PolicyValue,
            outcome: None,
            query: question,
        },
        plan_id: Arc::from("graph_imputation_effect_decision.v1"),
        data: DataClassification::Tabular,
        graph_class: "bounded_graph_imputation_effect_mixture",
        query_variables: Arc::from([]),
        source_digest: Some(*source.as_bytes()),
        support_coordinate: None,
        supported: false,
        support_status: None,
    };
    let identification = IdentificationLayer::NonGraphical(Arc::new(
        NonGraphicalIdentification::SharedEvidenceContract { contract: question },
    ));
    let statistical = StatisticalProgram {
        identifier: None,
        estimator: None,
        realization: Realization::GraphImputationV1,
        models: Arc::from([]),
        folds: None,
        resampling: None,
        posterior: None,
        design,
        operators: Arc::from([]),
    };
    let evidence = EvidenceDecisionProgram {
        sources: Arc::from([EvidenceSourceRef {
            role: EvidenceRole::ChainSource,
            digest: Some(*source.as_bytes()),
        }]),
        composition: Composition::Chained { stages: 2 },
        decision: Some(DecisionProgram {
            carrier: CarrierDemand::ParameterPosterior,
            utility: DecisionProgram::utility_form(None),
            actions: decision.actions.len(),
            constraints: decision.constraints.len(),
            chance_threshold: decision
                .constraints
                .iter()
                .map(|constraint| constraint.min_probability)
                .fold(0.0_f64, f64::max),
            utility_unit: Some(Arc::from(decision.utility_units.as_str())),
            contract_identity: Some(Arc::from(decision_identity.as_str())),
            information_value: None,
            scenario_source: Some(Arc::from("native_graph_imputation_effect_draws_v1")),
        }),
        planning: None,
        chain: None,
    };
    let physical = PhysicalLayer {
        operators: Arc::from([
            PhysicalOperator::declare(PhysicalEffect::Read)
                .estimates(ByteEstimates::absent().with_output(quantity_bytes.len() as u64))
                .build()
                .map_err(|_| {
                    GraphImputationEffectDecisionError::Invalid(
                        "invalid producer receipt read operator",
                    )
                })?,
            PhysicalOperator::declare(PhysicalEffect::Fit).build().map_err(|_| {
                GraphImputationEffectDecisionError::Invalid("invalid graph mixture operator")
            })?,
        ]),
        design,
    };
    Ok(Arc::new(
        StandaloneCompositionLoweredParts {
            cell: ProgramCell::GraphImputationEffectDecision,
            scientific,
            identification,
            statistical,
            evidence,
            physical,
            claim: ClaimDemand {
                uncertainty: UncertaintyReading::Posterior,
                kind: Some(ClaimKind::PolicyValue),
            },
        }
        .finish(),
    ))
}

/// Compose an already compiled graph-imputation producer with an effect
/// posterior decision. The upstream compiler's exact program identity is a
/// dependency of this outer program; no imputation or fitting runs here.
///
/// # Errors
/// An effect coordinate that differs from the producer, a decision mismatch,
/// or lowering refusal.
pub fn compile_graph_imputation_effect_decision(
    producer: CompiledGraphImputation,
    quantity: ScientificQuantity,
    decision: DecisionContract,
) -> Result<CompiledGraphImputationEffectDecision, GraphImputationEffectDecisionError> {
    if !producer.matches_effect_quantity(&quantity)? {
        return Err(GraphImputationEffectDecisionError::Invalid(
            "quantity differs from the compiled binary graph effect",
        ));
    }
    let lowered = lower_outer(
        producer.lowered().identity(),
        producer.source_identity(),
        &quantity,
        &decision,
    )?;
    Ok(CompiledGraphImputationEffectDecision { producer, quantity, decision, lowered })
}

fn effect_chain(
    point: &GraphImputationPoint,
    quantity: &ScientificQuantity,
) -> PropagationChainRecordWire {
    let receipt = SemanticDigest::from_bytes(point.receipt).to_hex();
    let alignment = format!("graph_imputation:{receipt}");
    let source = ChainSourceWire::draws(
        "graph.effect.posterior",
        LawMeaning::Posterior,
        &alignment,
        point.effect_draws(),
    )
    .with_factor(SharedFactorKind::Model, &receipt);
    let stage = ChainStageWire::new(
        "graph.effect",
        StageKind::DerivedQuantity,
        &["graph.effect.posterior"],
        ChainOpWire::affine(ChainCoefficientWire::fixed(1.0), ChainCoefficientWire::fixed(0.0)),
    );
    PropagationChainRecordWire::draft(
        TableConstruction::JointDraws,
        ChainReading::Posterior,
        "graph.effect",
        vec![source],
        vec![stage],
    )
    .with_output_quantity(ScientificQuantityWire::from(quantity))
}

impl CompiledGraphImputationEffectDecision {
    /// The six-layer outer program that binds the complete upstream schedule.
    #[must_use]
    pub fn lowered(&self) -> &LoweredProgram {
        &self.lowered
    }

    /// Inspect the retained program without fitting or scoring.
    #[must_use]
    pub fn plan_view(&self) -> PlanView {
        PlanView::of_program(Arc::clone(&self.lowered), PlanStage::Compiled)
    }

    /// Rerun graph selection and missing-row imputation through the producer,
    /// then compile utility over its retained effect posterior draws.
    ///
    /// # Errors
    /// An upstream child failure, mismatched effect or decision refusal.
    pub fn execute(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<GraphImputationEffectDecisionResult, GraphImputationEffectDecisionError> {
        let point = self.producer.execute(ctx)?;
        point.effect_law(self.quantity.clone())?;
        let decision = compile_chain_decision(
            effect_chain(&point, &self.quantity),
            self.decision.clone(),
            ctx,
        )?
        .execute(ctx)?;
        Ok(GraphImputationEffectDecisionResult {
            producer: point,
            decision,
            lowered_program_identity: self.lowered.identity(),
            calibration: "not_measured",
        })
    }

    /// Export the source-bound native producer and chain decision after replay.
    ///
    /// # Errors
    /// Changed graph mixture, utility, identity or artifact encoding.
    pub fn export(
        &self,
        result: &GraphImputationEffectDecisionResult,
        artifact_id: &str,
        ctx: &ExecutionContext,
    ) -> Result<Vec<u8>, GraphImputationEffectDecisionError> {
        let replay = self.producer.replay(result.producer.receipt, ctx)?;
        if result.lowered_program_identity != self.lowered.identity()
            || result.calibration != "not_measured"
            || replay.effect_draws() != result.producer.effect_draws()
        {
            return Err(GraphImputationEffectDecisionError::Invalid(
                "graph mixture differs on replay",
            ));
        }
        let graph =
            self.producer.export(&result.producer, ctx).map_err(GraphImputationError::from)?;
        let contract = DecisionContractArtifact::new(self.decision.clone())?;
        let chain = compile_chain_decision(
            effect_chain(&replay, &self.quantity),
            self.decision.clone(),
            ctx,
        )?;
        let live = chain.execute(ctx)?;
        if live.decision != result.decision.decision
            || live.chain.identity() != result.decision.chain.identity()
        {
            return Err(GraphImputationEffectDecisionError::Invalid("utility differs on replay"));
        }
        let expected = graph.expectation();
        let wire = GraphImputationEffectDecisionWire {
            version: 1,
            required_feature: FEATURE.into(),
            producer_source: graph.bytes().to_vec(),
            producer_program: expected.program,
            producer_data: expected.source,
            producer_receipt: expected.receipt,
            quantity: ScientificQuantityWire::from(&self.quantity),
            decision_identity: contract.identity().into(),
            decision_artifact: contract.to_bytes(artifact_id)?,
            chain_artifact: chain.export(&result.decision, artifact_id)?,
            program_identity: *self.lowered.identity().as_bytes(),
            calibration: "not_measured".into(),
        };
        let bytes = antecedent_io::to_cbor(&wire)?;
        if bytes.len() > MAX_ARTIFACT_BYTES {
            return Err(GraphImputationEffectDecisionError::Invalid(
                "combined artifact exceeds its byte bound",
            ));
        }
        Ok(bytes)
    }
}

/// Independently replay the original graph-imputation source and its child
/// fits, then recompile and compare the final effect utility. The outer
/// program identity must be retained separately from the artifact.
///
/// # Errors
/// A changed upstream schedule, model-conditional effect law, utility or
/// compiler identity refuses.
pub fn consume_graph_imputation_effect_decision(
    bytes: &[u8],
    expected_program: SemanticDigest,
    ctx: &ExecutionContext,
) -> Result<GraphImputationEffectDecisionResult, GraphImputationEffectDecisionError> {
    if bytes.len() > MAX_ARTIFACT_BYTES {
        return Err(GraphImputationEffectDecisionError::Invalid(
            "combined artifact exceeds its byte bound",
        ));
    }
    let wire: GraphImputationEffectDecisionWire = antecedent_io::from_cbor(bytes)?;
    if wire.version != 1
        || wire.required_feature != FEATURE
        || wire.program_identity != *expected_program.as_bytes()
        || wire.calibration != "not_measured"
    {
        return Err(GraphImputationEffectDecisionError::Invalid(
            "combined feature or identity differs",
        ));
    }
    let expectation = GraphImputationExpectation {
        program: wire.producer_program,
        source: wire.producer_data,
        receipt: wire.producer_receipt,
    };
    let point = consume_graph_imputation(&wire.producer_source, expectation, ctx)?;
    let quantity: ScientificQuantity =
        wire.quantity.try_into().map_err(GraphImputationEffectDecisionError::Invalid)?;
    point.effect_law(quantity.clone())?;
    let contract =
        DecisionContractArtifact::from_bytes(&wire.decision_artifact, &wire.decision_identity)?;
    let lowered = lower_outer(
        SemanticDigest::from_bytes(wire.producer_program),
        SemanticDigest::from_bytes(wire.producer_data),
        &quantity,
        contract.contract(),
    )?;
    if lowered.identity() != expected_program {
        return Err(GraphImputationEffectDecisionError::Invalid(
            "recompiled outer program differs",
        ));
    }
    let decision =
        compile_chain_decision(effect_chain(&point, &quantity), contract.contract().clone(), ctx)?
            .execute(ctx)?;
    let replay =
        consume_chain_decision(&wire.chain_artifact, Some(decision.lowered_program_identity), ctx)?;
    if replay.chain.identity() != decision.chain.identity() || replay.decision != decision.decision
    {
        return Err(GraphImputationEffectDecisionError::Invalid(
            "graph effect utility differs on replay",
        ));
    }
    Ok(GraphImputationEffectDecisionResult {
        producer: point,
        decision,
        lowered_program_identity: lowered.identity(),
        calibration: "not_measured",
    })
}

fn lower_paired_outer(
    producer: SemanticDigest,
    source: SemanticDigest,
    quantities: &[ScientificQuantity; 2],
    decision: &DecisionContract,
    target: Option<&TransportStandardization>,
    bootstrap: Option<TargetFrameBootstrap>,
    predictive: bool,
    gaussian: Option<GaussianPredictivePolicy>,
    cluster_bootstrap: Option<&TargetClusterBootstrap>,
) -> Result<Arc<LoweredProgram>, GraphImputationEffectDecisionError> {
    decision.validate()?;
    if bootstrap.is_some_and(|policy| target.is_none() || !(40..=256).contains(&policy.replicates))
    {
        return Err(GraphImputationEffectDecisionError::Invalid(
            "paired target bootstrap needs a target and 40 to 256 replicates",
        ));
    }
    if let Some(policy) = cluster_bootstrap {
        if !predictive || bootstrap.is_some() {
            return Err(GraphImputationEffectDecisionError::Invalid(
                "predictive cluster bootstrap needs a predictive fixed target",
            ));
        }
        let target = target.ok_or(GraphImputationEffectDecisionError::Invalid(
            "predictive cluster target missing",
        ))?;
        policy.groups(target.rows()).map_err(|_| {
            GraphImputationEffectDecisionError::Invalid("invalid predictive target clusters")
        })?;
    }
    let quantity_bytes = antecedent_io::to_cbor(
        &quantities.iter().map(ScientificQuantityWire::from).collect::<Vec<_>>(),
    )?;
    let decision_identity = decision.identity()?;
    let mut writer =
        CanonicalWriter::new("antecedent.graph_imputation_paired_world_decision.question.v1");
    writer
        .bytes(producer.as_bytes())
        .bytes(source.as_bytes())
        .bytes(&quantity_bytes)
        .text(&decision_identity);
    if let Some(policy) = gaussian {
        validate_gaussian_policy(policy)?;
        writer
            .text("graph_imputation_gaussian_target_predictive_world_decision_v1")
            .text("shared_graph_imputation_coefficient_covariate_correlated_gaussian_residuals_v1")
            .word(policy.residual_variance.to_bits())
            .word(policy.cross_world_correlation.to_bits())
            .word(policy.seed);
    } else if predictive {
        writer.text("graph_imputation_bernoulli_logit_target_predictive_world_decision_v1")
        .text("laplace_joint_normal_prior_independent_potential_outcome_innovations_given_graph_and_coefficients_v1");
    }
    if let Some(target) = target {
        writer.text("graph_imputation_transport_paired_world_decision_v2").bytes(&target.digest()?);
    }
    if let Some(policy) = bootstrap {
        writer
            .text("graph_imputation_transport_paired_world_bootstrap_decision_v2")
            .word(u64::from(policy.replicates))
            .word(policy.seed);
    }
    if let Some(policy) = cluster_bootstrap {
        writer
            .text(if gaussian.is_some() {
                "graph_imputation_gaussian_target_predictive_cluster_bootstrap_decision_v1"
            } else {
                "graph_imputation_bernoulli_target_predictive_cluster_bootstrap_decision_v1"
            })
            .bytes(&antecedent_io::to_cbor(policy)?);
    }
    let question = writer.finish();
    let design = DesignShape::quoted(target.map_or(1, |target| target.rows() as u64), 1, true);
    let scientific = ScientificProblem {
        runtime_query: None,
        identity_scope: ProgramIdentityScope::Semantic,
        target: ProblemTarget::QueryFunctional {
            kind: FunctionalKind::PolicyValue,
            outcome: None,
            query: question,
        },
        plan_id: Arc::from(if cluster_bootstrap.is_some() {
            if gaussian.is_some() {
                "graph_imputation_gaussian_target_predictive_cluster_bootstrap_decision.v1"
            } else {
                "graph_imputation_bernoulli_target_predictive_cluster_bootstrap_decision.v1"
            }
        } else if gaussian.is_some() {
            "graph_imputation_gaussian_target_predictive_world_decision.v1"
        } else if predictive {
            "graph_imputation_bernoulli_logit_target_predictive_world_decision.v1"
        } else if bootstrap.is_some() {
            "graph_imputation_transport_paired_world_bootstrap_decision.v2"
        } else if target.is_some() {
            "graph_imputation_transport_paired_world_decision.v2"
        } else {
            "graph_imputation_paired_world_decision.v1"
        }),
        data: DataClassification::Tabular,
        graph_class: if gaussian.is_some() {
            "bounded_graph_imputation_gaussian_target_predictive"
        } else if predictive {
            "bounded_graph_imputation_bernoulli_logit_target_predictive"
        } else {
            "bounded_graph_imputation_paired_world_mixture"
        },
        query_variables: Arc::from([]),
        source_digest: Some(*source.as_bytes()),
        support_coordinate: None,
        supported: false,
        support_status: None,
    };
    let identification = IdentificationLayer::NonGraphical(Arc::new(
        NonGraphicalIdentification::SharedEvidenceContract { contract: question },
    ));
    let statistical = StatisticalProgram {
        identifier: None,
        estimator: None,
        realization: Realization::GraphImputationV1,
        models: Arc::from([]),
        folds: None,
        resampling: if let Some(policy) = cluster_bootstrap {
            Some(ResamplingPlan {
                method: ResamplingMethod::Bootstrap,
                replicates: policy.replicates,
                unit: ResampleUnit::Cluster,
            })
        } else {
            bootstrap.map(|policy| ResamplingPlan {
                method: ResamplingMethod::Bootstrap,
                replicates: policy.replicates,
                unit: ResampleUnit::Row,
            })
        },
        posterior: None,
        design,
        operators: Arc::from([]),
    };
    let mut evidence_sources = vec![EvidenceSourceRef {
        role: EvidenceRole::ChainSource,
        digest: Some(*source.as_bytes()),
    }];
    if let Some(target) = target {
        evidence_sources.push(EvidenceSourceRef {
            role: EvidenceRole::TransportSource,
            digest: Some(target.digest()?),
        });
    }
    let evidence = EvidenceDecisionProgram {
        sources: Arc::from(evidence_sources),
        composition: Composition::Chained { stages: if target.is_some() { 3 } else { 2 } },
        decision: Some(DecisionProgram {
            carrier: if predictive {
                CarrierDemand::PredictiveLaw
            } else {
                CarrierDemand::ParameterPosterior
            },
            utility: DecisionProgram::utility_form(None),
            actions: decision.actions.len(),
            constraints: decision.constraints.len(),
            chance_threshold: decision
                .constraints
                .iter()
                .map(|constraint| constraint.min_probability)
                .fold(0.0_f64, f64::max),
            utility_unit: Some(Arc::from(decision.utility_units.as_str())),
            contract_identity: Some(Arc::from(decision_identity.as_str())),
            information_value: None,
            scenario_source: Some(Arc::from(if gaussian.is_some() {
                "native_graph_imputation_gaussian_target_predictive_world_draws_v1"
            } else if predictive {
                "native_graph_imputation_bernoulli_target_predictive_world_draws_v1"
            } else {
                "native_graph_imputation_paired_world_draws_v1"
            })),
        }),
        planning: None,
        chain: None,
    };
    let mut operators = vec![
        PhysicalOperator::declare(PhysicalEffect::Read)
            .estimates(ByteEstimates::absent().with_output(quantity_bytes.len() as u64))
            .build()
            .map_err(|_| {
                GraphImputationEffectDecisionError::Invalid(
                    "invalid producer receipt read operator",
                )
            })?,
        PhysicalOperator::declare(PhysicalEffect::Fit).build().map_err(|_| {
            GraphImputationEffectDecisionError::Invalid("invalid graph mixture operator")
        })?,
    ];
    if target.is_some() {
        operators.push(PhysicalOperator::declare(PhysicalEffect::Transport).build().map_err(
            |_| GraphImputationEffectDecisionError::Invalid("invalid target transport operator"),
        )?);
    }
    if let Some(policy) = bootstrap {
        for _ in 0..policy.replicates {
            operators.push(PhysicalOperator::declare(PhysicalEffect::Fit).build().map_err(
                |_| GraphImputationEffectDecisionError::Invalid("invalid bootstrap fit operator"),
            )?);
            operators.push(PhysicalOperator::declare(PhysicalEffect::Transport).build().map_err(
                |_| {
                    GraphImputationEffectDecisionError::Invalid(
                        "invalid bootstrap transport operator",
                    )
                },
            )?);
        }
    }
    if let Some(policy) = cluster_bootstrap {
        for _ in 0..policy.replicates {
            operators.push(PhysicalOperator::declare(PhysicalEffect::Fit).build().map_err(
                |_| GraphImputationEffectDecisionError::Invalid("invalid cluster fit operator"),
            )?);
            operators.push(PhysicalOperator::declare(PhysicalEffect::Transport).build().map_err(
                |_| {
                    GraphImputationEffectDecisionError::Invalid(
                        "invalid cluster transport operator",
                    )
                },
            )?);
        }
    }
    let physical = PhysicalLayer { operators: Arc::from(operators), design };
    Ok(Arc::new(
        StandaloneCompositionLoweredParts {
            cell: if cluster_bootstrap.is_some() {
                if gaussian.is_some() {
                    ProgramCell::GraphImputationGaussianPredictiveClusterBootstrapDecision
                } else {
                    ProgramCell::GraphImputationBernoulliPredictiveClusterBootstrapDecision
                }
            } else if gaussian.is_some() {
                ProgramCell::GraphImputationGaussianTargetPredictiveWorldDecision
            } else if predictive {
                ProgramCell::GraphImputationBernoulliPredictiveWorldDecision
            } else {
                ProgramCell::GraphImputationEffectDecision
            },
            scientific,
            identification,
            statistical,
            evidence,
            physical,
            claim: ClaimDemand {
                uncertainty: if predictive {
                    UncertaintyReading::Predictive
                } else {
                    UncertaintyReading::Posterior
                },
                kind: Some(ClaimKind::PolicyValue),
            },
        }
        .finish(),
    ))
}

/// Compiler-owned graph, imputation and shared-coefficient treatment-world decision.
pub struct CompiledGraphImputationWorldDecision {
    producer: CompiledGraphImputation,
    quantities: [ScientificQuantity; 2],
    decision: DecisionContract,
    lowered: Arc<LoweredProgram>,
    target: Option<TransportStandardization>,
    bootstrap: Option<TargetFrameBootstrap>,
    predictive: bool,
    gaussian: Option<GaussianPredictivePolicy>,
    cluster_bootstrap: Option<TargetClusterBootstrap>,
}

/// Declared individual residual law for two Gaussian action worlds.
#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct GaussianPredictivePolicy {
    /// Positive residual variance in each world.
    pub residual_variance: f64,
    /// Conditional correlation of the two residual innovations, from -1 to 1.
    pub cross_world_correlation: f64,
    /// Seed for the predictive residual draws.
    pub seed: u64,
}

fn validate_gaussian_policy(
    policy: GaussianPredictivePolicy,
) -> Result<(), GraphImputationEffectDecisionError> {
    if !policy.residual_variance.is_finite()
        || policy.residual_variance <= 0.0
        || !policy.cross_world_correlation.is_finite()
        || !(-1.0..=1.0).contains(&policy.cross_world_correlation)
    {
        return Err(GraphImputationEffectDecisionError::Invalid(
            "invalid Gaussian predictive residual law",
        ));
    }
    Ok(())
}

/// Model-conditional decision on paired potential-outcome mean worlds.
pub struct GraphImputationWorldDecisionResult {
    /// Shared graph, imputation and coefficient world pairs.
    pub producer: GraphPairedWorldReading,
    /// Utility on those paired parameter-world rows.
    pub decision: ChainDecisionResult,
    /// Identity of the enclosing compiler program.
    pub lowered_program_identity: SemanticDigest,
    /// Always `not_measured`.
    pub calibration: &'static str,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct GraphImputationWorldDecisionWire {
    version: u32,
    feature: String,
    source: Vec<u8>,
    producer_program: [u8; 32],
    source_identity: [u8; 32],
    receipt: [u8; 32],
    quantities: Vec<ScientificQuantityWire>,
    decision_identity: String,
    decision_artifact: Vec<u8>,
    chain_artifact: Vec<u8>,
    program_identity: [u8; 32],
    #[serde(default, skip_serializing_if = "Option::is_none")]
    target: Option<TransportStandardization>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    bootstrap: Option<TargetFrameBootstrap>,
    #[serde(default, skip_serializing_if = "is_false")]
    predictive: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    gaussian: Option<GaussianPredictivePolicy>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    cluster_bootstrap: Option<TargetClusterBootstrap>,
}

fn is_false(value: &bool) -> bool {
    !*value
}

fn validate_world_coordinates(
    source: &[u8],
    quantities: &[ScientificQuantity; 2],
    decision: &DecisionContract,
    predictive: bool,
    gaussian: bool,
) -> Result<(), GraphImputationEffectDecisionError> {
    decision.validate()?;
    let source = GraphImputationSource::decode(source)
        .map_err(GraphImputationEffectDecisionError::Invalid)?;
    let parts = source.parts();
    if predictive && !gaussian && parts.domains[parts.outcome as usize] != 1 {
        return Err(GraphImputationEffectDecisionError::Invalid(
            "Bernoulli predictive outcome must be binary",
        ));
    }
    if gaussian && parts.domains[parts.outcome as usize] != 0 {
        return Err(GraphImputationEffectDecisionError::Invalid(
            "Gaussian predictive outcome must be continuous",
        ));
    }
    let treatment = &parts.names[parts.treatment as usize];
    let outcome = &parts.names[parts.outcome as usize];
    for (i, quantity) in quantities.iter().enumerate() {
        if quantity.variable_id != *outcome
            || quantity.role != QuantityRole::Outcome
            || quantity.transform_id != "identity"
            || !quantity.conditioning.is_empty()
            || quantity.units != quantities[0].units
            || quantity.functional_id != if predictive { "outcome" } else { "mean" }
            || quantity.regime_id != format!("do({treatment}={i})")
            || quantity.population_id != decision.target_population
            || quantity.horizon != 0
            || decision.horizon != 0
        {
            return Err(GraphImputationEffectDecisionError::Invalid(
                "world coordinates differ from source treatment or outcome",
            ));
        }
    }
    if decision.actions.len() != 2
        || decision.actions.iter().any(|action| {
            !matches!(
                action.kind,
                ActionKind::Intervention | ActionKind::Regime | ActionKind::Policy
            )
        })
        || quantities.iter().any(|quantity| {
            !decision.actions.iter().any(|action| {
                action.inputs.len() == 1
                    && action.inputs[0].require_same_coordinate(quantity).is_ok()
            })
        })
    {
        return Err(GraphImputationEffectDecisionError::Invalid(
            "decision must name both distinct treatment worlds",
        ));
    }
    Ok(())
}

fn paired_world_chain(
    reading: &GraphPairedWorldReading,
    quantities: &[ScientificQuantity; 2],
    predictive: bool,
) -> PropagationChainRecordWire {
    let alignment = SemanticDigest::from_bytes(reading.receipt).to_hex();
    let mut sources = Vec::new();
    let mut stages = Vec::new();
    let mut outputs = Vec::new();
    for (i, quantity) in quantities.iter().enumerate() {
        let source = format!("graph.world.source.{i}");
        let stage = format!("graph.world.mean.{i}");
        let values: Vec<_> = reading.worlds.iter().map(|pair| pair[i]).collect();
        sources.push(
            ChainSourceWire::draws(
                &source,
                if predictive { LawMeaning::Predictive } else { LawMeaning::Posterior },
                &alignment,
                &values,
            )
            .with_factor(SharedFactorKind::Model, &alignment),
        );
        stages.push(ChainStageWire::new(
            &stage,
            StageKind::DerivedQuantity,
            &[&source],
            ChainOpWire::affine(ChainCoefficientWire::fixed(1.0), ChainCoefficientWire::fixed(0.0)),
        ));
        outputs
            .push(ChainJointOutputWire { stage, quantity: ScientificQuantityWire::from(quantity) });
    }
    PropagationChainRecordWire::draft(
        TableConstruction::JointDraws,
        if predictive { ChainReading::Predictive } else { ChainReading::Posterior },
        &outputs[0].stage,
        sources,
        stages,
    )
    .with_output_quantity(outputs[0].quantity.clone())
    .with_joint_outputs(outputs)
}

/// Compile the paired-world model and decision before fitting any child.
/// Both worlds are mean potential outcomes, not individual predictive draws.
///
/// # Errors
/// Mismatched outcome, treatment, regimes, actions or decision contract.
pub fn compile_graph_imputation_world_decision(
    producer: CompiledGraphImputation,
    quantities: [ScientificQuantity; 2],
    decision: DecisionContract,
) -> Result<CompiledGraphImputationWorldDecision, GraphImputationEffectDecisionError> {
    validate_world_coordinates(producer.source_bytes(), &quantities, &decision, false, false)?;
    let lowered = lower_paired_outer(
        producer.lowered().identity(),
        producer.source_identity(),
        &quantities,
        &decision,
        None,
        None,
        false,
        None,
        None,
    )?;
    Ok(CompiledGraphImputationWorldDecision {
        producer,
        quantities,
        decision,
        lowered,
        target: None,
        bootstrap: None,
        predictive: false,
        gaussian: None,
        cluster_bootstrap: None,
    })
}

/// Compile target-population Bernoulli predictive action worlds using a
/// Laplace joint coefficient approximation under the producer's normal prior.
/// The graph, imputation, target covariate row and coefficient draw are shared;
/// potential-outcome innovations are independent conditional on that state.
///
/// # Errors
/// Invalid target, nonbinary outcome or mismatched regimes and decision.
pub fn compile_graph_imputation_bernoulli_predictive_world_decision(
    producer: CompiledGraphImputation,
    target: TransportStandardization,
    quantities: [ScientificQuantity; 2],
    decision: DecisionContract,
) -> Result<CompiledGraphImputationWorldDecision, GraphImputationEffectDecisionError> {
    let source = GraphImputationSource::decode(producer.source_bytes())
        .map_err(GraphImputationEffectDecisionError::Invalid)?;
    target.validate(source.parts())?;
    if target.source_sample != producer.source_identity().to_hex()
        || quantities.iter().any(|quantity| quantity.population_id != target.target_population)
        || decision.target_population != target.target_population
    {
        return Err(GraphImputationEffectDecisionError::Invalid(
            "predictive target source or population differs",
        ));
    }
    validate_world_coordinates(producer.source_bytes(), &quantities, &decision, true, false)?;
    let lowered = lower_paired_outer(
        producer.lowered().identity(),
        producer.source_identity(),
        &quantities,
        &decision,
        Some(&target),
        None,
        true,
        None,
        None,
    )?;
    Ok(CompiledGraphImputationWorldDecision {
        producer,
        quantities,
        decision,
        lowered,
        target: Some(target),
        bootstrap: None,
        predictive: true,
        gaussian: None,
        cluster_bootstrap: None,
    })
}

/// Compile individual Gaussian target action worlds. Both actions share the
/// graph, completed table, target covariate and coefficient draw; the declared
/// residual variance and cross-world correlation determine predictive noise.
///
/// # Errors
/// Invalid source, target, residual law, outcome, coordinates or contract.
pub fn compile_graph_imputation_gaussian_target_predictive_world_decision(
    producer: CompiledGraphImputation,
    target: TransportStandardization,
    quantities: [ScientificQuantity; 2],
    decision: DecisionContract,
    policy: GaussianPredictivePolicy,
) -> Result<CompiledGraphImputationWorldDecision, GraphImputationEffectDecisionError> {
    validate_gaussian_policy(policy)?;
    let source = GraphImputationSource::decode(producer.source_bytes())
        .map_err(GraphImputationEffectDecisionError::Invalid)?;
    target.validate(source.parts())?;
    if target.source_sample != producer.source_identity().to_hex()
        || quantities.iter().any(|quantity| quantity.population_id != target.target_population)
        || decision.target_population != target.target_population
    {
        return Err(GraphImputationEffectDecisionError::Invalid(
            "predictive target source or population differs",
        ));
    }
    validate_world_coordinates(producer.source_bytes(), &quantities, &decision, true, true)?;
    let lowered = lower_paired_outer(
        producer.lowered().identity(),
        producer.source_identity(),
        &quantities,
        &decision,
        Some(&target),
        None,
        true,
        Some(policy),
        None,
    )?;
    Ok(CompiledGraphImputationWorldDecision {
        producer,
        quantities,
        decision,
        lowered,
        target: Some(target),
        bootstrap: None,
        predictive: true,
        gaussian: Some(policy),
        cluster_bootstrap: None,
    })
}

/// Add independent whole-target-cluster bootstrap to a compiled predictive
/// world route. Each replicate refits the graph-imputation target model and
/// retains one intact paired individual outcome draw.
///
/// # Errors
/// The route is not predictive or the declared cluster policy is invalid.
pub fn compile_graph_imputation_predictive_cluster_bootstrap_decision(
    fixed: CompiledGraphImputationWorldDecision,
    policy: TargetClusterBootstrap,
) -> Result<CompiledGraphImputationWorldDecision, GraphImputationEffectDecisionError> {
    if !fixed.predictive || fixed.cluster_bootstrap.is_some() {
        return Err(GraphImputationEffectDecisionError::Invalid(
            "cluster bootstrap requires a fixed predictive route",
        ));
    }
    let lowered = lower_paired_outer(
        fixed.producer.lowered().identity(),
        fixed.producer.source_identity(),
        &fixed.quantities,
        &fixed.decision,
        fixed.target.as_ref(),
        None,
        true,
        fixed.gaussian,
        Some(&policy),
    )?;
    Ok(CompiledGraphImputationWorldDecision { lowered, cluster_bootstrap: Some(policy), ..fixed })
}

/// Compile paired target mean worlds and an action decision from the complete
/// source and representative binary target frame. This carries parameter
/// uncertainty, not predictive outcome noise or earned decision reliability.
///
/// # Errors
/// Invalid target, source binding, world coordinate or decision contract.
pub fn compile_graph_imputation_transport_world_decision(
    producer: CompiledGraphImputation,
    target: TransportStandardization,
    quantities: [ScientificQuantity; 2],
    decision: DecisionContract,
) -> Result<CompiledGraphImputationWorldDecision, GraphImputationEffectDecisionError> {
    let source = GraphImputationSource::decode(producer.source_bytes())
        .map_err(GraphImputationEffectDecisionError::Invalid)?;
    target.validate(source.parts())?;
    if target.source_sample != producer.source_identity().to_hex()
        || quantities.iter().any(|quantity| quantity.population_id != target.target_population)
        || decision.target_population != target.target_population
    {
        return Err(GraphImputationEffectDecisionError::Invalid(
            "paired target source or population differs",
        ));
    }
    validate_world_coordinates(producer.source_bytes(), &quantities, &decision, false, false)?;
    let lowered = lower_paired_outer(
        producer.lowered().identity(),
        producer.source_identity(),
        &quantities,
        &decision,
        Some(&target),
        None,
        false,
        None,
        None,
    )?;
    Ok(CompiledGraphImputationWorldDecision {
        producer,
        quantities,
        decision,
        lowered,
        target: Some(target),
        bootstrap: None,
        predictive: false,
        gaussian: None,
        cluster_bootstrap: None,
    })
}

/// Compile IID target-frame resampling of paired target mean worlds.
/// Every replicate replays the complete graph and imputation fit and retains
/// one jointly sampled control/active coefficient pair.
///
/// # Errors
/// Invalid target, source, decision or bootstrap policy.
pub fn compile_graph_imputation_transport_world_bootstrap_decision(
    producer: CompiledGraphImputation,
    target: TransportStandardization,
    quantities: [ScientificQuantity; 2],
    decision: DecisionContract,
    policy: TargetFrameBootstrap,
) -> Result<CompiledGraphImputationWorldDecision, GraphImputationEffectDecisionError> {
    let fixed =
        compile_graph_imputation_transport_world_decision(producer, target, quantities, decision)?;
    let lowered = lower_paired_outer(
        fixed.producer.lowered().identity(),
        fixed.producer.source_identity(),
        &fixed.quantities,
        &fixed.decision,
        fixed.target.as_ref(),
        Some(policy),
        false,
        None,
        None,
    )?;
    Ok(CompiledGraphImputationWorldDecision { lowered, bootstrap: Some(policy), ..fixed })
}

/// Unmeasured coordinate for a prospective known-optimum action-choice study.
pub struct GraphImputationWorldReliabilityCandidate {
    /// Complete action-world execution, with `not_measured` standing.
    pub result: GraphImputationWorldDecisionResult,
    /// Exact source/design and action-contract study coordinate. External
    /// truth is needed to score choices; this is not a reliability record.
    pub action_choice_basis: SemanticDigest,
    /// Nearest-rank equal-tail 95% intervals for control and active means.
    pub world_intervals: [(f64, f64); 2],
    /// Separate prospective coverage coordinates for the two mean worlds.
    pub coverage_bases: [CalibrationBasis; 2],
}

/// Prospective, unmeasured predictive outcome and action-choice study inputs.
pub struct GraphImputationPredictiveCalibrationCandidate {
    /// Complete predictive execution.
    pub result: GraphImputationWorldDecisionResult,
    /// Nearest-rank 95% intervals for the two individual action outcomes.
    pub world_intervals: [(f64, f64); 2],
    /// Separate exact prospective coverage scopes for those outcomes.
    pub coverage_bases: [CalibrationBasis; 2],
    /// Exact prospective action-choice scope; no reliability is earned.
    pub action_choice_basis: SemanticDigest,
}

impl CompiledGraphImputationWorldDecision {
    /// Retained six-layer joint decision program.
    #[must_use]
    pub fn lowered(&self) -> &LoweredProgram {
        &self.lowered
    }
    /// Inspect the compiled program.
    #[must_use]
    pub fn plan_view(&self) -> PlanView {
        PlanView::of_program(Arc::clone(&self.lowered), PlanStage::Compiled)
    }
    /// Refit shared graph/imputation worlds and evaluate the action decision.
    ///
    /// # Errors
    /// Any graph, child, pairing or utility refusal.
    pub fn execute(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<GraphImputationWorldDecisionResult, GraphImputationEffectDecisionError> {
        let reading = if let Some(cluster) = &self.cluster_bootstrap {
            replay_target_predictive_cluster_bootstrap_worlds(
                self.producer.source_bytes(),
                self.target.as_ref().ok_or(GraphImputationEffectDecisionError::Invalid(
                    "predictive target missing",
                ))?,
                &cluster.cluster_ids,
                cluster.replicates,
                cluster.seed,
                self.gaussian.map(|p| (p.residual_variance, p.cross_world_correlation, p.seed)),
                ctx,
            )?
        } else if let Some(policy) = self.gaussian {
            replay_gaussian_target_predictive_worlds(
                self.producer.source_bytes(),
                self.target.as_ref().ok_or(GraphImputationEffectDecisionError::Invalid(
                    "predictive target missing",
                ))?,
                policy.residual_variance,
                policy.cross_world_correlation,
                policy.seed,
                ctx,
            )?
        } else if self.predictive {
            replay_bernoulli_logit_predictive_worlds(
                self.producer.source_bytes(),
                self.target.as_ref().ok_or(GraphImputationEffectDecisionError::Invalid(
                    "predictive target missing",
                ))?,
                ctx,
            )?
        } else if let (Some(target), Some(policy)) = (&self.target, self.bootstrap) {
            replay_transport_paired_worlds_bootstrap(
                self.producer.source_bytes(),
                target,
                policy.replicates,
                policy.seed,
                ctx,
            )?
        } else if let Some(target) = &self.target {
            replay_transport_paired_worlds(self.producer.source_bytes(), target, ctx)?
        } else {
            replay_paired_worlds(self.producer.source_bytes(), ctx)?
        };
        let chain = compile_action_world_chain_decision(
            paired_world_chain(&reading, &self.quantities, self.predictive),
            self.decision.clone(),
            ctx,
        )?;
        let decision = chain.execute(ctx)?;
        Ok(GraphImputationWorldDecisionResult {
            producer: reading,
            decision,
            lowered_program_identity: self.lowered.identity(),
            calibration: "not_measured",
        })
    }
    /// Return the unmeasured prospective calibration coordinate for the
    /// target predictive schedule. This creates no record.
    ///
    /// # Errors
    /// Called on a different route or with an invalid retained source.
    pub fn predictive_calibration_basis(
        &self,
    ) -> Result<CalibrationBasis, GraphImputationEffectDecisionError> {
        if !self.predictive {
            return Err(GraphImputationEffectDecisionError::Invalid(
                "predictive basis requires a target predictive route",
            ));
        }
        let source = GraphImputationSource::decode(self.producer.source_bytes())
            .map_err(GraphImputationEffectDecisionError::Invalid)?;
        let target = self
            .target
            .as_ref()
            .ok_or(GraphImputationEffectDecisionError::Invalid("predictive target missing"))?;
        let parts = source.parts();
        let mut study =
            CanonicalWriter::new("antecedent.graph_imputation.predictive_study_design.v1");
        study
            .text(parts.support.as_str())
            .word(parts.atoms.len() as u64)
            .words(&parts.domains.iter().map(|domain| u64::from(*domain)).collect::<Vec<_>>())
            .word(parts.rows() as u64)
            .word(target.rows() as u64)
            .word(parts.imputations as u64)
            .word(parts.iterations as u64)
            .word(parts.posterior_draws as u64)
            .word(parts.prior_scale.to_bits())
            .words(&target.selection_targets.iter().map(|id| u64::from(*id)).collect::<Vec<_>>())
            .words(&target.columns.iter().map(|(id, _)| u64::from(*id)).collect::<Vec<_>>())
            .text(if self.cluster_bootstrap.is_some() {
                "independent_source_units_target_clusters:source_outcome_mar:cluster_bootstrap_v1"
            } else {
                "independent_source_target_units:source_outcome_mar:fixed_empirical_target_frame"
            });
        if let Some(policy) = &self.cluster_bootstrap {
            let groups = policy.groups(target.rows()).map_err(|_| {
                GraphImputationEffectDecisionError::Invalid("invalid predictive target clusters")
            })?;
            study.word(groups.len() as u64).word(u64::from(policy.replicates));
            let mut sizes: Vec<u64> = groups.iter().map(|group| group.len() as u64).collect();
            sizes.sort_unstable();
            study.words(&sizes);
        }
        if let Some(policy) = self.gaussian {
            study
                .text("gaussian_joint_coefficients_correlated_residuals")
                .word(policy.residual_variance.to_bits())
                .word(policy.cross_world_correlation.to_bits());
        } else {
            study.text("bernoulli_logit_laplace_joint_normal_coefficients_independent_residuals");
        }
        let functional = format!(
            "{}_target_predictive_worlds:design={}",
            if self.gaussian.is_some() { "gaussian" } else { "binary" },
            SemanticDigest::from_bytes(study.finish()).to_hex()
        );
        Ok(CalibrationBasis::new([
            "PredictiveOutcome", "Dag", "graph_posterior", "multi_env", "Bayesian",
            if self.cluster_bootstrap.is_some() { if self.gaussian.is_some() { "graph_imputation_gaussian_target_predictive_cluster_bootstrap_decision_v1" }
                else { "graph_imputation_bernoulli_target_predictive_cluster_bootstrap_decision_v1" } }
                else if self.gaussian.is_some() { "graph_imputation_gaussian_target_predictive_world_decision_v1" }
                else { "graph_imputation_bernoulli_logit_target_predictive_world_decision_v1" },
            if self.gaussian.is_some() { "predictive_gaussian_worlds" } else { "predictive_binary_worlds" }, "", if self.cluster_bootstrap.is_some() { "independent_source_units_and_target_clusters" }
                else { "independent_source_target_units" },
            if self.gaussian.is_some() { "correlated_gaussian_individual_residuals_v1" }
                else { "laplace_joint_normal_prior_independent_potential_outcome_innovations_given_graph_and_coefficients_v1" }, &functional,
        ].map(Arc::from), 0.95, Arc::from("model_conditional"),
            (parts.rows() as u64).saturating_add(target.rows() as u64), None,
            Some(parts.posterior_draws.saturating_mul(parts.imputations)), 0.0).with_validation("none"))
    }
    /// Execute a predictive route and prepare exact prospective outcome and
    /// action-choice coordinates without asserting coverage or reliability.
    ///
    /// # Errors
    /// The route is not predictive, lacks forty finite draws, or execution refuses.
    pub fn predictive_calibration_candidate(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<GraphImputationPredictiveCalibrationCandidate, GraphImputationEffectDecisionError>
    {
        let result = self.execute(ctx)?;
        let base = self.predictive_calibration_basis()?;
        let mut intervals = [(0.0, 0.0); 2];
        for (world, interval) in intervals.iter_mut().enumerate() {
            let mut values: Vec<_> = result.producer.worlds.iter().map(|row| row[world]).collect();
            if values.len() < 40 || values.iter().any(|value| !value.is_finite()) {
                return Err(GraphImputationEffectDecisionError::Invalid(
                    "predictive interval needs forty finite draws",
                ));
            }
            values.sort_by(f64::total_cmp);
            let rank = |n: usize| (values.len() * n).div_ceil(1000).saturating_sub(1);
            *interval = (values[rank(25)], values[rank(975)]);
        }
        let coverage_bases = std::array::from_fn(|world| {
            let mut basis = base.clone();
            basis.replicates_ok = self.cluster_bootstrap.as_ref().map(|policy| policy.replicates);
            basis.functional = Arc::from(format!("{}:world={world}", base.functional));
            basis
        });
        let mut action =
            CanonicalWriter::new("antecedent.graph_imputation.predictive_action_choice_study.v1");
        action.text(base.functional.as_ref()).text(&self.decision.identity()?).bytes(
            &antecedent_io::to_cbor(
                &self.quantities.iter().map(ScientificQuantityWire::from).collect::<Vec<_>>(),
            )?,
        );
        Ok(GraphImputationPredictiveCalibrationCandidate {
            result,
            world_intervals: intervals,
            coverage_bases,
            action_choice_basis: SemanticDigest::from_bytes(action.finish()),
        })
    }
    /// Execute and bind an unmeasured action-choice study coordinate. True
    /// utility and the optimal action must be specified independently.
    ///
    /// # Errors
    /// Source, target, decision or encoding refusal.
    pub fn reliability_candidate(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<GraphImputationWorldReliabilityCandidate, GraphImputationEffectDecisionError> {
        if self.predictive {
            return Err(GraphImputationEffectDecisionError::Invalid(
                "mean-world reliability candidate does not apply to predictive outcomes",
            ));
        }
        let result = self.execute(ctx)?;
        let source = GraphImputationSource::decode(self.producer.source_bytes())
            .map_err(GraphImputationEffectDecisionError::Invalid)?;
        let parts = source.parts();
        let mut writer =
            CanonicalWriter::new("antecedent.graph_imputation_paired_world.action_choice_study.v1");
        writer
            .text(parts.support.as_str())
            .word(parts.atoms.len() as u64)
            .word(parts.treatment as u64)
            .word(parts.outcome as u64)
            .words(&parts.domains.iter().map(|domain| u64::from(*domain)).collect::<Vec<_>>())
            .word(parts.imputations as u64)
            .word(parts.iterations as u64)
            .word(parts.imputation_seed)
            .word(parts.execution_seed)
            .word(parts.posterior_draws as u64)
            .word(parts.prior_scale.to_bits())
            .word(parts.rows() as u64)
            .text(&self.decision.identity()?)
            .bytes(&antecedent_io::to_cbor(
                &self.quantities.iter().map(ScientificQuantityWire::from).collect::<Vec<_>>(),
            )?);
        if let Some(target) = &self.target {
            writer
                .text("binary_target_frame_s_admissible_v2")
                .word(target.rows() as u64)
                .words(&target.columns.iter().map(|(id, _)| u64::from(*id)).collect::<Vec<_>>())
                .words(
                    &target.selection_targets.iter().map(|id| u64::from(*id)).collect::<Vec<_>>(),
                );
        } else {
            writer.text("source_frame_v1");
        }
        if let Some(policy) = self.bootstrap {
            writer
                .text("iid_target_rows_paired_bootstrap_unbiased_pool_v2")
                .word(u64::from(policy.replicates))
                .word(policy.seed);
        }
        let action_choice_basis = SemanticDigest::from_bytes(writer.finish());
        let mut intervals = [(0.0, 0.0); 2];
        for (i, interval) in intervals.iter_mut().enumerate() {
            let mut ordered: Vec<f64> = result.producer.worlds.iter().map(|pair| pair[i]).collect();
            if ordered.len() < 40 || ordered.iter().any(|value| !value.is_finite()) {
                return Err(GraphImputationEffectDecisionError::Invalid(
                    "world interval needs finite paired draws",
                ));
            }
            ordered.sort_by(f64::total_cmp);
            let rank =
                |numerator: usize| (ordered.len() * numerator).div_ceil(1000).saturating_sub(1);
            *interval = (ordered[rank(25)], ordered[rank(975)]);
        }
        let feature = if self.bootstrap.is_some() {
            "graph_imputation_transport_paired_world_bootstrap_decision_v2"
        } else if self.target.is_some() {
            "graph_imputation_transport_paired_world_decision_v2"
        } else {
            "graph_imputation_paired_world_decision_v1"
        };
        let draws = self
            .bootstrap
            .map_or(parts.posterior_draws.saturating_mul(parts.imputations), |policy| {
                policy.replicates
            });
        let rows = (parts.rows() as u64)
            .saturating_add(self.target.as_ref().map_or(0, |target| target.rows() as u64));
        let coverage_bases = std::array::from_fn(|world| {
            let functional = format!(
                "paired_{}_mean_world={world}:study={}",
                if self.target.is_some() { "target" } else { "source" },
                action_choice_basis.to_hex()
            );
            let dependence = if self.target.is_some() {
                "independent_source_target_units"
            } else {
                "source_units"
            };
            CalibrationBasis::new(
                [
                    "MeanOutcome",
                    "Dag",
                    "graph_posterior",
                    "multi_env",
                    "Bayesian",
                    feature,
                    "nearest_rank_95",
                    "",
                    dependence,
                    "mar_graph_mixture_paired_mean_worlds",
                    &functional,
                ]
                .map(Arc::from),
                0.95,
                Arc::from("model_conditional"),
                rows,
                None,
                Some(draws),
                0.0,
            )
            .with_validation("none")
        });
        Ok(GraphImputationWorldReliabilityCandidate {
            result,
            action_choice_basis,
            world_intervals: intervals,
            coverage_bases,
        })
    }
    /// Recompute and seal the paired schedule and decision for replay.
    ///
    /// # Errors
    /// Changed result, nested refusal or oversized artifact.
    pub fn export(
        &self,
        result: &GraphImputationWorldDecisionResult,
        artifact_id: &str,
        ctx: &ExecutionContext,
    ) -> Result<Vec<u8>, GraphImputationEffectDecisionError> {
        let live = self.execute(ctx)?;
        if result.producer.receipt != live.producer.receipt
            || result.lowered_program_identity != self.lowered.identity()
            || result.calibration != "not_measured"
            || result.decision.decision != live.decision.decision
            || result.decision.chain.identity() != live.decision.chain.identity()
        {
            return Err(GraphImputationEffectDecisionError::Invalid(
                "paired world decision changed",
            ));
        }
        let contract = DecisionContractArtifact::new(self.decision.clone())?;
        let chain = compile_action_world_chain_decision(
            paired_world_chain(&live.producer, &self.quantities, self.predictive),
            self.decision.clone(),
            ctx,
        )?;
        let bytes = antecedent_io::to_cbor(&GraphImputationWorldDecisionWire {
            version: 1,
            feature: if self.cluster_bootstrap.is_some() {
                if self.gaussian.is_some() {
                    "graph_imputation_gaussian_target_predictive_cluster_bootstrap_decision_v1"
                } else {
                    "graph_imputation_bernoulli_target_predictive_cluster_bootstrap_decision_v1"
                }
            } else if self.gaussian.is_some() {
                "graph_imputation_gaussian_target_predictive_world_decision_v1"
            } else if self.predictive {
                "graph_imputation_bernoulli_logit_target_predictive_world_decision_v1"
            } else if self.bootstrap.is_some() {
                "graph_imputation_transport_paired_world_bootstrap_decision_v2"
            } else if self.target.is_some() {
                "graph_imputation_transport_paired_world_decision_v2"
            } else {
                "graph_imputation_paired_world_decision_v1"
            }
            .into(),
            source: self.producer.source_bytes().to_vec(),
            producer_program: *self.producer.lowered().identity().as_bytes(),
            source_identity: *self.producer.source_identity().as_bytes(),
            receipt: live.producer.receipt,
            quantities: self.quantities.iter().map(ScientificQuantityWire::from).collect(),
            decision_identity: contract.identity().into(),
            decision_artifact: contract.to_bytes(artifact_id)?,
            chain_artifact: chain.export(&live.decision, artifact_id)?,
            program_identity: *self.lowered.identity().as_bytes(),
            target: self.target.clone(),
            bootstrap: self.bootstrap,
            predictive: self.predictive,
            gaussian: self.gaussian,
            cluster_bootstrap: self.cluster_bootstrap.clone(),
        })?;
        if bytes.len() > MAX_ARTIFACT_BYTES {
            return Err(GraphImputationEffectDecisionError::Invalid(
                "paired world artifact exceeds its bound",
            ));
        }
        Ok(bytes)
    }
}

/// Independently reconstruct the source, paired world rows and action decision.
///
/// # Errors
/// Changed program, source, receipt, quantity, decision or paired replay.
pub fn consume_graph_imputation_world_decision(
    bytes: &[u8],
    expected: SemanticDigest,
    ctx: &ExecutionContext,
) -> Result<GraphImputationWorldDecisionResult, GraphImputationEffectDecisionError> {
    if bytes.len() > MAX_ARTIFACT_BYTES {
        return Err(GraphImputationEffectDecisionError::Invalid(
            "paired world artifact exceeds its bound",
        ));
    }
    let wire: GraphImputationWorldDecisionWire = antecedent_io::from_cbor(bytes)?;
    if wire.cluster_bootstrap.is_some() && (!wire.predictive || wire.bootstrap.is_some()) {
        return Err(GraphImputationEffectDecisionError::Invalid(
            "predictive cluster policy conflicts with stored route",
        ));
    }
    if wire.gaussian.is_some() && !wire.predictive {
        return Err(GraphImputationEffectDecisionError::Invalid(
            "Gaussian predictive policy lacks predictive route",
        ));
    }
    if wire.predictive && (wire.target.is_none() || wire.bootstrap.is_some()) {
        return Err(GraphImputationEffectDecisionError::Invalid(
            "predictive route requires a fixed target frame",
        ));
    }
    if wire.bootstrap.is_some() && wire.target.is_none() {
        return Err(GraphImputationEffectDecisionError::Invalid("paired bootstrap lacks target"));
    }
    let feature = if wire.cluster_bootstrap.is_some() {
        if wire.gaussian.is_some() {
            "graph_imputation_gaussian_target_predictive_cluster_bootstrap_decision_v1"
        } else {
            "graph_imputation_bernoulli_target_predictive_cluster_bootstrap_decision_v1"
        }
    } else if wire.gaussian.is_some() {
        "graph_imputation_gaussian_target_predictive_world_decision_v1"
    } else if wire.predictive {
        "graph_imputation_bernoulli_logit_target_predictive_world_decision_v1"
    } else if wire.bootstrap.is_some() {
        "graph_imputation_transport_paired_world_bootstrap_decision_v2"
    } else if wire.target.is_some() {
        "graph_imputation_transport_paired_world_decision_v2"
    } else {
        "graph_imputation_paired_world_decision_v1"
    };
    if wire.version != 1 || wire.feature != feature || wire.program_identity != *expected.as_bytes()
    {
        return Err(GraphImputationEffectDecisionError::Invalid(
            "paired world feature or identity differs",
        ));
    }
    let reading = if let Some(cluster) = &wire.cluster_bootstrap {
        replay_target_predictive_cluster_bootstrap_worlds(
            &wire.source,
            wire.target
                .as_ref()
                .ok_or(GraphImputationEffectDecisionError::Invalid("predictive target missing"))?,
            &cluster.cluster_ids,
            cluster.replicates,
            cluster.seed,
            wire.gaussian.map(|p| (p.residual_variance, p.cross_world_correlation, p.seed)),
            ctx,
        )?
    } else if let Some(policy) = wire.gaussian {
        validate_gaussian_policy(policy)?;
        replay_gaussian_target_predictive_worlds(
            &wire.source,
            wire.target
                .as_ref()
                .ok_or(GraphImputationEffectDecisionError::Invalid("predictive target missing"))?,
            policy.residual_variance,
            policy.cross_world_correlation,
            policy.seed,
            ctx,
        )?
    } else if wire.predictive {
        replay_bernoulli_logit_predictive_worlds(
            &wire.source,
            wire.target
                .as_ref()
                .ok_or(GraphImputationEffectDecisionError::Invalid("predictive target missing"))?,
            ctx,
        )?
    } else if let (Some(target), Some(policy)) = (&wire.target, wire.bootstrap) {
        replay_transport_paired_worlds_bootstrap(
            &wire.source,
            target,
            policy.replicates,
            policy.seed,
            ctx,
        )?
    } else if let Some(target) = &wire.target {
        replay_transport_paired_worlds(&wire.source, target, ctx)?
    } else {
        replay_paired_worlds(&wire.source, ctx)?
    };
    if reading.program != wire.producer_program
        || reading.source != wire.source_identity
        || reading.receipt != wire.receipt
    {
        return Err(GraphImputationEffectDecisionError::Invalid(
            "paired world source or receipt differs",
        ));
    }
    let quantities: Vec<ScientificQuantity> = wire
        .quantities
        .into_iter()
        .map(|q| q.try_into().map_err(GraphImputationEffectDecisionError::Invalid))
        .collect::<Result<_, _>>()?;
    let quantities: [ScientificQuantity; 2] = quantities.try_into().map_err(|_| {
        GraphImputationEffectDecisionError::Invalid("paired world coordinates missing")
    })?;
    let contract =
        DecisionContractArtifact::from_bytes(&wire.decision_artifact, &wire.decision_identity)?;
    validate_world_coordinates(
        &wire.source,
        &quantities,
        contract.contract(),
        wire.predictive,
        wire.gaussian.is_some(),
    )?;
    if let Some(target) = &wire.target {
        if target.source_sample != SemanticDigest::from_bytes(reading.source).to_hex()
            || quantities.iter().any(|quantity| quantity.population_id != target.target_population)
            || contract.contract().target_population != target.target_population
        {
            return Err(GraphImputationEffectDecisionError::Invalid(
                "stored paired target population differs",
            ));
        }
    }
    let lowered = lower_paired_outer(
        SemanticDigest::from_bytes(reading.program),
        SemanticDigest::from_bytes(reading.source),
        &quantities,
        contract.contract(),
        wire.target.as_ref(),
        wire.bootstrap,
        wire.predictive,
        wire.gaussian,
        wire.cluster_bootstrap.as_ref(),
    )?;
    if lowered.identity() != expected {
        return Err(GraphImputationEffectDecisionError::Invalid(
            "paired world outer program differs",
        ));
    }
    let decision = compile_action_world_chain_decision(
        paired_world_chain(&reading, &quantities, wire.predictive),
        contract.contract().clone(),
        ctx,
    )?
    .execute(ctx)?;
    let replay =
        consume_chain_decision(&wire.chain_artifact, Some(decision.lowered_program_identity), ctx)?;
    if replay.chain.identity() != decision.chain.identity() || replay.decision != decision.decision
    {
        return Err(GraphImputationEffectDecisionError::Invalid(
            "paired world decision replay differs",
        ));
    }
    Ok(GraphImputationWorldDecisionResult {
        producer: reading,
        decision,
        lowered_program_identity: expected,
        calibration: "not_measured",
    })
}

#[cfg(test)]
mod paired_world_tests {
    use super::super::graph_imputation::GraphImputationSpec;
    use super::*;
    use antecedent_core::{VariableId, graph_imputation_contract::GraphSupport};
    use antecedent_data::{ImputationSpec, TabularData};
    use antecedent_design::decision_contract::{
        DecisionAction, DecisionCriterion, StructuralPolicy, UtilityExpr,
    };

    #[test]
    #[ignore = "source-only Bernoulli graph predictive decision; reliability not earned"]
    fn bernoulli_predictive_worlds_replay_and_bind_model() {
        let t: Vec<_> = (0..96).map(|i| f64::from(u8::from(i % 2 == 0))).collect();
        let x: Vec<_> = (0..96).map(|i| f64::from(u8::from(i % 4 < 2))).collect();
        let y: Vec<_> = (0..96)
            .map(|i| {
                if i % 13 == 0 {
                    f64::NAN
                } else {
                    f64::from(u8::from((i * 7 + i / 3) % 11 < 4 + (t[i] as usize)))
                }
            })
            .collect();
        let ctx = ExecutionContext::for_tests(41);
        let table = TabularData::from_f64_columns([
            ("t", t.as_slice()),
            ("y", y.as_slice()),
            ("x", x.as_slice()),
        ])
        .unwrap();
        let producer = CompiledGraphImputation::compile(
            table,
            &GraphImputationSpec {
                treatment: VariableId::from_raw(0),
                outcome: VariableId::from_raw(1),
                support: GraphSupport::TreatmentRootPreTreatmentCovariates,
                imputation: ImputationSpec::new()
                    .with_imputations(2)
                    .with_iterations(2)
                    .with_seed(17),
                domains: Arc::from(vec![1, 1, 1]),
                posterior_draws: 100,
                prior_scale: 1.0,
                execution_seed: 19,
            },
            &ctx,
        )
        .unwrap();
        let quantities = [0, 1].map(|a| ScientificQuantity {
            variable_id: "y".into(),
            variable_name: "outcome".into(),
            role: QuantityRole::Outcome,
            units: "outcome".into(),
            population_id: "source".into(),
            regime_id: format!("do(t={a})"),
            horizon: 0,
            functional_id: "outcome".into(),
            conditioning: Vec::new(),
            transform_id: "identity".into(),
        });
        let decision = DecisionContract {
            actions: quantities
                .iter()
                .enumerate()
                .map(|(i, q)| DecisionAction {
                    id: format!("action-{i}"),
                    kind: ActionKind::Intervention,
                    inputs: vec![q.clone()],
                    utility: UtilityExpr::Input(0),
                })
                .collect(),
            utility_units: "outcome".into(),
            criterion: DecisionCriterion::PosteriorExpectedUtility,
            constraints: Vec::new(),
            target_population: "source".into(),
            horizon: 0,
            structural_policy: StructuralPolicy::RequireInvariantBestAction,
        };
        let target = TransportStandardization {
            columns: vec![(2, x.clone())],
            selection_targets: vec![2],
            source_population: "source".into(),
            target_population: "target".into(),
            source_sample: producer.source_identity().to_hex(),
            target_sample: "target-sample".into(),
            source_unit_ids: (0..96).map(|row| format!("source-{row}")).collect(),
            target_unit_ids: (0..96).map(|row| format!("target-{row}")).collect(),
        };
        let quantities = quantities.map(|mut q| {
            q.population_id = "target".into();
            q
        });
        let mut decision = decision;
        decision.target_population = "target".into();
        for (action, quantity) in decision.actions.iter_mut().zip(&quantities) {
            action.inputs = vec![quantity.clone()];
        }
        let plan = compile_graph_imputation_bernoulli_predictive_world_decision(
            producer, target, quantities, decision,
        )
        .unwrap();
        let result = plan.execute(&ctx).unwrap();
        assert!(result.producer.worlds.iter().flatten().all(|&value| value == 0.0 || value == 1.0));
        assert_eq!(result.calibration, "not_measured");
        let artifact = plan.export(&result, "bernoulli-predictive", &ctx).unwrap();
        let replay =
            consume_graph_imputation_world_decision(&artifact, plan.lowered().identity(), &ctx)
                .unwrap();
        assert_eq!(result.producer.worlds, replay.producer.worlds);
        assert_eq!(result.decision.decision, replay.decision.decision);
        let mut wire: GraphImputationWorldDecisionWire =
            antecedent_io::from_cbor(&artifact).unwrap();
        wire.predictive = false;
        assert!(
            consume_graph_imputation_world_decision(
                &antecedent_io::to_cbor(&wire).unwrap(),
                plan.lowered().identity(),
                &ctx
            )
            .is_err()
        );
        let fixed_identity = plan.lowered().identity();
        let cluster = compile_graph_imputation_predictive_cluster_bootstrap_decision(
            plan,
            TargetClusterBootstrap {
                replicates: 40,
                seed: 17,
                cluster_ids: (0..96).map(|row| format!("cluster-{}", row / 2)).collect(),
            },
        )
        .unwrap();
        assert_ne!(fixed_identity, cluster.lowered().identity());
        let candidate = cluster.predictive_calibration_candidate(&ctx).unwrap();
        assert_eq!(candidate.result.producer.worlds.len(), 40);
        assert_eq!(candidate.coverage_bases[0].replicates_ok, Some(40));
        let bytes = cluster.export(&candidate.result, "bernoulli-cluster", &ctx).unwrap();
        let replay =
            consume_graph_imputation_world_decision(&bytes, cluster.lowered().identity(), &ctx)
                .unwrap();
        assert_eq!(candidate.result.producer.worlds, replay.producer.worlds);
        assert_eq!(candidate.result.decision.decision, replay.decision.decision);
        let mut changed: GraphImputationWorldDecisionWire =
            antecedent_io::from_cbor(&bytes).unwrap();
        changed.cluster_bootstrap.as_mut().unwrap().cluster_ids[0].push('x');
        assert!(
            consume_graph_imputation_world_decision(
                &antecedent_io::to_cbor(&changed).unwrap(),
                cluster.lowered().identity(),
                &ctx
            )
            .is_err()
        );
    }

    #[test]
    #[ignore = "source-only paired graph-imputation world decision; reliability not earned"]
    fn paired_world_decision_replays() {
        let treatment: Vec<_> =
            (0..80).map(|i| f64::from(u8::from(i % 4 == 0 || i % 4 == 3))).collect();
        let covariate: Vec<_> = (0..80).map(|i| f64::from(u8::from(i % 2 == 0))).collect();
        let outcome: Vec<_> = (0..80)
            .map(|i| {
                if i % 13 == 0 {
                    f64::NAN
                } else {
                    0.2 * covariate[i] + 0.5 * treatment[i] + 0.1 * (i % 5) as f64
                }
            })
            .collect();
        let table = TabularData::from_f64_columns([
            ("t", treatment.as_slice()),
            ("y", outcome.as_slice()),
            ("x", covariate.as_slice()),
        ])
        .unwrap();
        let ctx = ExecutionContext::for_tests(19);
        let producer = CompiledGraphImputation::compile(
            table,
            &GraphImputationSpec {
                treatment: VariableId::from_raw(0),
                outcome: VariableId::from_raw(1),
                support: GraphSupport::TreatmentRootPreTreatmentCovariates,
                imputation: ImputationSpec::new()
                    .with_imputations(2)
                    .with_iterations(2)
                    .with_seed(17),
                domains: Arc::from(vec![1, 0, 1]),
                posterior_draws: 100,
                prior_scale: 1.0,
                execution_seed: 19,
            },
            &ctx,
        )
        .unwrap();
        let quantity = |a| ScientificQuantity {
            variable_id: "y".into(),
            variable_name: "outcome".into(),
            role: QuantityRole::Outcome,
            units: "score".into(),
            population_id: "source".into(),
            regime_id: format!("do(t={a})"),
            horizon: 0,
            functional_id: "mean".into(),
            conditioning: Vec::new(),
            transform_id: "identity".into(),
        };
        let quantities = [quantity(0), quantity(1)];
        let decision = DecisionContract {
            actions: quantities
                .iter()
                .enumerate()
                .map(|(i, q)| DecisionAction {
                    id: format!("action-{i}"),
                    kind: ActionKind::Intervention,
                    inputs: vec![q.clone()],
                    utility: UtilityExpr::Input(0),
                })
                .collect(),
            utility_units: "score".into(),
            criterion: DecisionCriterion::PosteriorExpectedUtility,
            constraints: Vec::new(),
            target_population: "source".into(),
            horizon: 0,
            structural_policy: StructuralPolicy::RequireInvariantBestAction,
        };
        let target_table = TabularData::from_f64_columns([
            ("t", treatment.as_slice()),
            ("y", outcome.as_slice()),
            ("x", covariate.as_slice()),
        ])
        .unwrap();
        let target_producer = CompiledGraphImputation::compile(
            target_table,
            &GraphImputationSpec {
                treatment: VariableId::from_raw(0),
                outcome: VariableId::from_raw(1),
                support: GraphSupport::TreatmentRootPreTreatmentCovariates,
                imputation: ImputationSpec::new()
                    .with_imputations(2)
                    .with_iterations(2)
                    .with_seed(17),
                domains: Arc::from(vec![1, 0, 1]),
                posterior_draws: 100,
                prior_scale: 1.0,
                execution_seed: 19,
            },
            &ctx,
        )
        .unwrap();
        let target = TransportStandardization {
            columns: vec![(2, (0..80).map(|row| f64::from(u8::from(row % 5 != 0))).collect())],
            selection_targets: vec![2],
            source_population: "source".into(),
            target_population: "target".into(),
            source_sample: target_producer.source_identity().to_hex(),
            target_sample: "target-sample".into(),
            source_unit_ids: (0..80).map(|row| format!("source-{row}")).collect(),
            target_unit_ids: (0..80).map(|row| format!("target-{row}")).collect(),
        };
        let target_quantities = quantities.clone().map(|mut quantity| {
            quantity.population_id = "target".into();
            quantity
        });
        let mut target_decision = decision.clone();
        target_decision.target_population = "target".into();
        for (action, quantity) in target_decision.actions.iter_mut().zip(&target_quantities) {
            action.inputs = vec![quantity.clone()];
        }
        let gaussian_table = TabularData::from_f64_columns([
            ("t", treatment.as_slice()),
            ("y", outcome.as_slice()),
            ("x", covariate.as_slice()),
        ])
        .unwrap();
        let gaussian_producer = CompiledGraphImputation::compile(
            gaussian_table,
            &GraphImputationSpec {
                treatment: VariableId::from_raw(0),
                outcome: VariableId::from_raw(1),
                support: GraphSupport::TreatmentRootPreTreatmentCovariates,
                imputation: ImputationSpec::new()
                    .with_imputations(2)
                    .with_iterations(2)
                    .with_seed(17),
                domains: Arc::from(vec![1, 0, 1]),
                posterior_draws: 100,
                prior_scale: 1.0,
                execution_seed: 19,
            },
            &ctx,
        )
        .unwrap();
        let gaussian_quantities = target_quantities.clone().map(|mut q| {
            q.functional_id = "outcome".into();
            q
        });
        let mut gaussian_decision = target_decision.clone();
        for (action, quantity) in gaussian_decision.actions.iter_mut().zip(&gaussian_quantities) {
            action.inputs = vec![quantity.clone()];
        }
        assert!(
            validate_gaussian_policy(GaussianPredictivePolicy {
                residual_variance: 0.0,
                cross_world_correlation: 0.0,
                seed: 5
            })
            .is_err()
        );
        assert!(
            validate_gaussian_policy(GaussianPredictivePolicy {
                residual_variance: 1.0,
                cross_world_correlation: 1.1,
                seed: 5
            })
            .is_err()
        );
        let gaussian_plan = compile_graph_imputation_gaussian_target_predictive_world_decision(
            gaussian_producer,
            target.clone(),
            gaussian_quantities,
            gaussian_decision,
            GaussianPredictivePolicy {
                residual_variance: 0.4,
                cross_world_correlation: 0.3,
                seed: 5,
            },
        )
        .unwrap();
        let gaussian_result = gaussian_plan.execute(&ctx).unwrap();
        assert_eq!(gaussian_result.calibration, "not_measured");
        assert!(gaussian_result.producer.worlds.iter().flatten().all(|value| value.is_finite()));
        assert_eq!(
            gaussian_plan.predictive_calibration_basis().unwrap().validation.as_ref(),
            "none"
        );
        let gaussian_bytes =
            gaussian_plan.export(&gaussian_result, "gaussian-predictive", &ctx).unwrap();
        let gaussian_replay = consume_graph_imputation_world_decision(
            &gaussian_bytes,
            gaussian_plan.lowered().identity(),
            &ctx,
        )
        .unwrap();
        assert_eq!(gaussian_result.producer.worlds, gaussian_replay.producer.worlds);
        assert_eq!(gaussian_result.decision.decision, gaussian_replay.decision.decision);
        let mut gaussian_wire: GraphImputationWorldDecisionWire =
            antecedent_io::from_cbor(&gaussian_bytes).unwrap();
        gaussian_wire.gaussian.as_mut().unwrap().cross_world_correlation = 0.7;
        assert!(
            consume_graph_imputation_world_decision(
                &antecedent_io::to_cbor(&gaussian_wire).unwrap(),
                gaussian_plan.lowered().identity(),
                &ctx
            )
            .is_err()
        );
        let fixed_identity = gaussian_plan.lowered().identity();
        let gaussian_cluster = compile_graph_imputation_predictive_cluster_bootstrap_decision(
            gaussian_plan,
            TargetClusterBootstrap {
                replicates: 40,
                seed: 17,
                cluster_ids: (0..80).map(|row| format!("cluster-{}", row / 2)).collect(),
            },
        )
        .unwrap();
        assert_ne!(fixed_identity, gaussian_cluster.lowered().identity());
        let cluster_result = gaussian_cluster.execute(&ctx).unwrap();
        assert_eq!(cluster_result.producer.worlds.len(), 40);
        let cluster_bytes =
            gaussian_cluster.export(&cluster_result, "gaussian-cluster", &ctx).unwrap();
        let cluster_replay = consume_graph_imputation_world_decision(
            &cluster_bytes,
            gaussian_cluster.lowered().identity(),
            &ctx,
        )
        .unwrap();
        assert_eq!(cluster_result.producer.worlds, cluster_replay.producer.worlds);
        let mut altered: GraphImputationWorldDecisionWire =
            antecedent_io::from_cbor(&cluster_bytes).unwrap();
        altered.cluster_bootstrap.as_mut().unwrap().seed ^= 1;
        assert!(
            consume_graph_imputation_world_decision(
                &antecedent_io::to_cbor(&altered).unwrap(),
                gaussian_cluster.lowered().identity(),
                &ctx
            )
            .is_err()
        );
        let bootstrap_table = TabularData::from_f64_columns([
            ("t", treatment.as_slice()),
            ("y", outcome.as_slice()),
            ("x", covariate.as_slice()),
        ])
        .unwrap();
        let bootstrap_producer = CompiledGraphImputation::compile(
            bootstrap_table,
            &GraphImputationSpec {
                treatment: VariableId::from_raw(0),
                outcome: VariableId::from_raw(1),
                support: GraphSupport::TreatmentRootPreTreatmentCovariates,
                imputation: ImputationSpec::new()
                    .with_imputations(2)
                    .with_iterations(2)
                    .with_seed(17),
                domains: Arc::from(vec![1, 0, 1]),
                posterior_draws: 100,
                prior_scale: 1.0,
                execution_seed: 19,
            },
            &ctx,
        )
        .unwrap();
        let bootstrap_plan = compile_graph_imputation_transport_world_bootstrap_decision(
            bootstrap_producer,
            target.clone(),
            target_quantities.clone(),
            target_decision.clone(),
            TargetFrameBootstrap { replicates: 40, seed: 37 },
        )
        .unwrap();
        let target_plan = compile_graph_imputation_transport_world_decision(
            target_producer,
            target,
            target_quantities.clone(),
            target_decision,
        )
        .unwrap();
        let plan = compile_graph_imputation_world_decision(producer, quantities.clone(), decision)
            .unwrap();
        assert_ne!(plan.lowered().identity(), target_plan.lowered().identity());
        assert_ne!(bootstrap_plan.lowered().identity(), target_plan.lowered().identity());
        let bootstrap_candidate = bootstrap_plan.reliability_candidate(&ctx).unwrap();
        assert_eq!(bootstrap_candidate.result.calibration, "not_measured");
        assert_eq!(bootstrap_candidate.result.producer.worlds.len(), 40);
        assert!(
            bootstrap_candidate
                .coverage_bases
                .iter()
                .all(|basis| basis.validation.as_ref() == "none")
        );
        assert!(
            bootstrap_candidate.world_intervals.iter().all(|interval| interval.0 <= interval.1)
        );
        let bootstrap_joint =
            paired_world_chain(&bootstrap_candidate.result.producer, &target_quantities, false);
        for (index, pair) in bootstrap_candidate.result.producer.worlds.iter().enumerate() {
            assert_eq!(bootstrap_joint.sources[0].values[index], pair[0].to_bits());
            assert_eq!(bootstrap_joint.sources[1].values[index], pair[1].to_bits());
        }
        let bootstrap_bytes =
            bootstrap_plan.export(&bootstrap_candidate.result, "paired-bootstrap", &ctx).unwrap();
        let bootstrap_replay = consume_graph_imputation_world_decision(
            &bootstrap_bytes,
            bootstrap_plan.lowered().identity(),
            &ctx,
        )
        .unwrap();
        assert_eq!(bootstrap_replay.producer.worlds, bootstrap_candidate.result.producer.worlds);
        assert_eq!(
            bootstrap_replay.decision.decision,
            bootstrap_candidate.result.decision.decision
        );
        let mut bootstrap_wire: GraphImputationWorldDecisionWire =
            antecedent_io::from_cbor(&bootstrap_bytes).unwrap();
        bootstrap_wire.bootstrap.as_mut().unwrap().seed ^= 1;
        let tampered_bootstrap = antecedent_io::to_cbor(&bootstrap_wire).unwrap();
        assert!(
            consume_graph_imputation_world_decision(
                &tampered_bootstrap,
                bootstrap_plan.lowered().identity(),
                &ctx,
            )
            .is_err()
        );
        let target_candidate = target_plan.reliability_candidate(&ctx).unwrap();
        let target_result = target_candidate.result;
        assert_eq!(target_result.calibration, "not_measured");
        let target_joint = paired_world_chain(&target_result.producer, &target_quantities, false);
        assert_eq!(target_joint.sources[0].alignment, target_joint.sources[1].alignment);
        assert_eq!(target_joint.sources[0].factors, target_joint.sources[1].factors);
        assert!(
            target_result
                .producer
                .worlds
                .iter()
                .all(|pair| pair[0].is_finite() && pair[1].is_finite())
        );
        let target_artifact =
            target_plan.export(&target_result, "target-paired-world", &ctx).unwrap();
        let target_replay = consume_graph_imputation_world_decision(
            &target_artifact,
            target_plan.lowered().identity(),
            &ctx,
        )
        .unwrap();
        assert_eq!(target_replay.producer.worlds, target_result.producer.worlds);
        assert_eq!(target_replay.decision.decision, target_result.decision.decision);
        let mut target_wire: GraphImputationWorldDecisionWire =
            antecedent_io::from_cbor(&target_artifact).unwrap();
        let first_value = target_wire.target.as_ref().unwrap().columns[0].1[0];
        target_wire.target.as_mut().unwrap().columns[0].1[0] = 1.0 - first_value;
        let target_tampered = antecedent_io::to_cbor(&target_wire).unwrap();
        assert!(
            consume_graph_imputation_world_decision(
                &target_tampered,
                target_plan.lowered().identity(),
                &ctx
            )
            .is_err()
        );
        let result = plan.execute(&ctx).unwrap();
        let joint = paired_world_chain(&result.producer, &quantities, false);
        assert_eq!(joint.sources[0].alignment, joint.sources[1].alignment);
        assert_eq!(joint.sources[0].factors, joint.sources[1].factors);
        for (i, pair) in result.producer.worlds.iter().enumerate() {
            assert_eq!(joint.sources[0].values[i], pair[0].to_bits());
            assert_eq!(joint.sources[1].values[i], pair[1].to_bits());
        }
        assert_eq!(result.calibration, "not_measured");
        let artifact = plan.export(&result, "paired-world", &ctx).unwrap();
        let replay =
            consume_graph_imputation_world_decision(&artifact, plan.lowered().identity(), &ctx)
                .unwrap();
        assert_eq!(result.producer.receipt, replay.producer.receipt);
        assert_eq!(result.producer.worlds, replay.producer.worlds);
        assert_eq!(result.decision.decision, replay.decision.decision);
        assert!(result.producer.worlds.iter().any(|pair| pair[0] != pair[1]));
        let mut wire: GraphImputationWorldDecisionWire =
            antecedent_io::from_cbor(&artifact).unwrap();
        wire.receipt[0] ^= 1;
        let tampered = antecedent_io::to_cbor(&wire).unwrap();
        assert!(
            consume_graph_imputation_world_decision(&tampered, plan.lowered().identity(), &ctx)
                .is_err()
        );
    }
}
