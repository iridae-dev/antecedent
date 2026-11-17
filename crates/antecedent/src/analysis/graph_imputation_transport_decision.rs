//! Graph selection, source imputation, graph-specific target standardization,
//! child fitting and effect utility in one source-bound compiler program.
//!
//! The target supplies complete binary pretreatment covariates from a declared
//! representative sample. A selected covariate omitted by one graph's effect
//! model has a declared effect-homogeneity premise in that graph. Each
//! completed source table gets its own graph
//! weights and target-to-source stratum weights before child fitting. The
//! resulting posterior is conditional on the empirical target covariate frame
//! and model assumptions; it does not propagate target-sample uncertainty.
//! No interval or decision reliability record is inherited from the source
//! graph-imputation route.

use std::sync::Arc;

use antecedent_contract::{ClaimKind, DataClassification, QuantityRole, ScientificQuantity};
use antecedent_contract::dependence::SharedFactorKind;
use antecedent_contract::propagation_chain::{
    ChainReading, LawMeaning, StageKind, TableConstruction,
};
use antecedent_core::graph_imputation_contract::GraphImputationSource;
use antecedent_core::{
    ByteEstimates, CanonicalWriter, ExecutionContext, PhysicalEffect,
    PhysicalOperator, SemanticDigest, VariableId,
};
use antecedent_data::{ImputationSpec, TabularData};
use antecedent_design::decision_artifact::DecisionContractArtifact;
use antecedent_design::decision_contract::{DecisionContract, DecisionContractError};
use antecedent_io::IoError;
use antecedent_io::measured_inference::graph_candidate::{
    GraphPointReading, TransportStandardization, replay_transport_point,
};
use antecedent_io::propagation_chain_record::{
    ChainCoefficientWire, ChainOpWire, ChainSourceWire, ChainStageWire,
    PropagationChainRecordWire,
};
use antecedent_io::quantity_wire::ScientificQuantityWire;
use serde::{Deserialize, Serialize};

use super::graph_imputation::{
    CompiledGraphImputation, GraphImputationError, GraphImputationSpec,
};
use super::propagation_chain_decision::{
    ChainDecisionError, ChainDecisionResult, compile_chain_decision,
    consume_chain_decision,
};
use crate::planner::lowered::{
    CarrierDemand, ClaimDemand, Composition, DecisionProgram, DesignShape,
    EvidenceDecisionProgram, EvidenceRole, EvidenceSourceRef, FunctionalKind,
    IdentificationLayer, LoweredProgram, NonGraphicalIdentification,
    PhysicalLayer, ProblemTarget, ProgramCell, ProgramIdentityScope,
    ScientificProblem, StandaloneCompositionLoweredParts, UncertaintyReading,
};
use crate::planner::statistical::{Realization, StatisticalProgram};
use crate::planner::{PlanStage, PlanView};

const FEATURE: &str = "graph_imputation_transport_decision_v1";
const MAX_ARTIFACT_BYTES: usize = 80 * 1024 * 1024;

/// Refusal of the bound source, target, decision or portable replay.
#[derive(Debug, thiserror::Error)]
pub enum GraphImputationTransportDecisionError {
    /// One coordinate or retained identity differs.
    #[error("graph imputation transport decision: {0}")]
    Invalid(&'static str),
    /// Upstream graph-imputation compilation refused.
    #[error("graph imputation producer: {0}")]
    Producer(#[from] GraphImputationError),
    /// Decision contract refused.
    #[error("graph imputation transport contract: {0}")]
    Contract(#[from] DecisionContractError),
    /// Law or decision compiler refused.
    #[error("graph imputation transport chain: {0}")]
    Chain(#[from] ChainDecisionError),
    /// Native replay or artifact codec refused.
    #[error("graph imputation transport artifact: {0}")]
    Io(#[from] IoError),
}

/// Retained original source, representative target and final utility program.
pub struct CompiledGraphImputationTransportDecision {
    producer: CompiledGraphImputation,
    target: TransportStandardization,
    quantity: ScientificQuantity,
    decision: DecisionContract,
    native_source: Vec<u8>,
    lowered: Arc<LoweredProgram>,
}

/// Target-standardized effect posterior and utility over its aligned draws.
pub struct GraphImputationTransportDecisionResult {
    /// Native graph-selection, imputation, target-standardization and fit result.
    pub producer: GraphPointReading,
    /// Compiled effect utility.
    pub decision: ChainDecisionResult,
    /// Complete outer six-layer identity.
    pub lowered_program_identity: SemanticDigest,
    /// No complete-schedule reliability record has been earned.
    pub calibration: &'static str,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct GraphImputationTransportDecisionWire {
    version: u32,
    required_feature: String,
    original_source: Vec<u8>,
    target: TransportStandardization,
    quantity: ScientificQuantityWire,
    decision_identity: String,
    decision_artifact: Vec<u8>,
    chain_artifact: Vec<u8>,
    producer_receipt: [u8; 32],
    program_identity: [u8; 32],
    calibration: String,
}

fn check_quantity(
    source: &GraphImputationSource, target: &TransportStandardization,
    quantity: &ScientificQuantity, decision: &DecisionContract,
) -> Result<(), GraphImputationTransportDecisionError> {
    let parts = source.parts();
    let treatment = &parts.names[parts.treatment as usize];
    let outcome = &parts.names[parts.outcome as usize];
    decision.validate()?;
    if quantity.role != QuantityRole::Outcome
        || quantity.variable_id != outcome.as_str()
        || quantity.population_id != target.target_population
        || quantity.population_id != decision.target_population
        || quantity.regime_id != format!("do({treatment}=1)-do({treatment}=0)")
        || quantity.functional_id != "average_effect"
        || quantity.horizon != 0 || quantity.horizon != decision.horizon
        || !quantity.conditioning.is_empty() || quantity.transform_id != "identity"
        || decision.actions.is_empty()
        || decision.actions.iter().any(|action| {
            action.inputs.len() != 1
                || action.inputs[0].require_same_coordinate(quantity).is_err()
        })
    {
        return Err(GraphImputationTransportDecisionError::Invalid(
            "utility must read the exact target binary average-effect coordinate",
        ));
    }
    Ok(())
}

fn lower_outer(
    source: &GraphImputationSource, target: &TransportStandardization,
    quantity: &ScientificQuantity, decision: &DecisionContract,
) -> Result<Arc<LoweredProgram>, GraphImputationTransportDecisionError> {
    target.validate(source.parts())?;
    check_quantity(source, target, quantity, decision)?;
    let target_digest = target.digest()?;
    let source_digest = SemanticDigest::from_bytes(source.parts().data_digest);
    let coordinate = antecedent_io::to_cbor(&ScientificQuantityWire::from(quantity))?;
    let contract = decision.identity()?;
    let mut writer = CanonicalWriter::new("antecedent.graph_imputation_transport_decision.question.v1");
    writer.bytes(&source.program()).bytes(&source.parts().contract_digest())
        .bytes(source_digest.as_bytes()).bytes(&target_digest)
        .bytes(&coordinate).text(&contract);
    let question = writer.finish();
    let rows = source.parts().rows().saturating_add(target.rows()) as u64;
    let design = DesignShape::quoted(rows, source.parts().names.len() as u64, true);
    let scientific = ScientificProblem {
        runtime_query: None, identity_scope: ProgramIdentityScope::Semantic,
        target: ProblemTarget::QueryFunctional {
            kind: FunctionalKind::PolicyValue,
            outcome: Some(VariableId::from_raw(source.parts().outcome)), query: question,
        },
        plan_id: Arc::from("graph_imputation_transport_decision.v1"),
        data: DataClassification::Tabular,
        graph_class: "bounded_graph_imputation_binary_target_standardization",
        query_variables: Arc::from([]), source_digest: Some(source_digest),
        support_coordinate: None, supported: false, support_status: None,
    };
    let identification = IdentificationLayer::NonGraphical(Arc::new(
        NonGraphicalIdentification::SharedEvidenceContract { contract: question },
    ));
    let statistical = StatisticalProgram {
        identifier: None, estimator: None,
        realization: Realization::GraphImputationV1,
        models: Arc::from([]), folds: None, resampling: None,
        posterior: None, design, operators: Arc::from([]),
    };
    let evidence = EvidenceDecisionProgram {
        sources: Arc::from([
            EvidenceSourceRef { role: EvidenceRole::Observations, digest: Some(source_digest) },
            EvidenceSourceRef { role: EvidenceRole::TransportSource,
                digest: Some(SemanticDigest::from_bytes(target_digest)) },
        ]),
        composition: Composition::Chained { stages: 5 },
        decision: Some(DecisionProgram {
            carrier: CarrierDemand::ParameterPosterior,
            utility: DecisionProgram::utility_form(None),
            actions: decision.actions.len(), constraints: decision.constraints.len(),
            chance_threshold: decision.constraints.iter()
                .map(|constraint| constraint.min_probability).fold(0.0_f64, f64::max),
            utility_unit: Some(Arc::from(decision.utility_units.as_str())),
            contract_identity: Some(Arc::from(contract.as_str())),
            information_value: None,
            scenario_source: Some(Arc::from("graph_imputation_target_effect_draws_v1")),
        }), planning: None, chain: None,
    };
    let physical = PhysicalLayer {
        operators: Arc::from([
            PhysicalOperator::declare(PhysicalEffect::Read)
                .estimates(ByteEstimates::absent().with_output(
                    (source.parts().rows() as u64).saturating_add(target.rows() as u64),
                )).build().map_err(|_| GraphImputationTransportDecisionError::Invalid("invalid source read operator"))?,
            PhysicalOperator::declare(PhysicalEffect::Fit)
                .build().map_err(|_| GraphImputationTransportDecisionError::Invalid("invalid graph fit operator"))?,
            PhysicalOperator::declare(PhysicalEffect::Transport)
                .build().map_err(|_| GraphImputationTransportDecisionError::Invalid("invalid target standardization operator"))?,
        ]), design,
    };
    Ok(Arc::new(StandaloneCompositionLoweredParts {
        cell: ProgramCell::GraphImputationTransportDecision,
        scientific, identification, statistical, evidence, physical,
        claim: ClaimDemand { uncertainty: UncertaintyReading::Posterior,
            kind: Some(ClaimKind::PolicyValue) },
    }.finish()))
}

/// Bind source graph support, imputation schedule, target rows and selection
/// premise, exact effect coordinate and utility before any fitting executes.
///
/// # Errors
/// Invalid target support, quantity, contract or six-layer lowering.
pub fn compile_graph_imputation_transport_decision(
    producer: CompiledGraphImputation, target: TransportStandardization,
    quantity: ScientificQuantity, decision: DecisionContract,
) -> Result<CompiledGraphImputationTransportDecision, GraphImputationTransportDecisionError> {
    let source = GraphImputationSource::decode(producer.source_bytes())
        .map_err(GraphImputationTransportDecisionError::Invalid)?;
    if source.program() != *producer.lowered().identity().as_bytes()
        || target.source_sample != producer.source_identity().to_hex()
    {
        return Err(GraphImputationTransportDecisionError::Invalid(
            "target source-sample identity differs from the retained producer",
        ));
    }
    let lowered = lower_outer(&source, &target, &quantity, &decision)?;
    let native_source = GraphImputationSource::new(
        source.parts().clone(), *lowered.identity().as_bytes(),
    ).map_err(GraphImputationTransportDecisionError::Invalid)?.encode();
    Ok(CompiledGraphImputationTransportDecision {
        producer, target, quantity, decision, native_source, lowered,
    })
}

fn effect_chain(
    reading: &GraphPointReading, quantity: &ScientificQuantity,
) -> PropagationChainRecordWire {
    let receipt = SemanticDigest::from_bytes(reading.receipt).to_hex();
    let source = ChainSourceWire::draws(
        "transport.effect.posterior", LawMeaning::Posterior,
        &format!("graph_imputation_transport:{receipt}"), &reading.draws,
    ).with_factor(SharedFactorKind::Model, &receipt);
    PropagationChainRecordWire::draft(
        TableConstruction::JointDraws, ChainReading::Posterior,
        "transport.effect", vec![source], vec![ChainStageWire::new(
            "transport.effect", StageKind::DerivedQuantity,
            &["transport.effect.posterior"],
            ChainOpWire::affine(
                ChainCoefficientWire::fixed(1.0), ChainCoefficientWire::fixed(0.0),
            ),
        )],
    ).with_output_quantity(ScientificQuantityWire::from(quantity))
}

impl CompiledGraphImputationTransportDecision {
    /// Inspect the complete six-layer schedule before executing it.
    #[must_use]
    pub fn lowered(&self) -> &LoweredProgram { &self.lowered }

    /// Inspect the retained plan without fitting.
    #[must_use]
    pub fn plan_view(&self) -> PlanView {
        PlanView::of_program(Arc::clone(&self.lowered), PlanStage::Compiled)
    }

    /// Reimpute, reselect graphs, fit target-standardized children and score
    /// the exact effect utility on the retained posterior draws.
    ///
    /// # Errors
    /// Source, target, fit or decision refusal.
    pub fn execute(
        &self, ctx: &ExecutionContext,
    ) -> Result<GraphImputationTransportDecisionResult, GraphImputationTransportDecisionError> {
        let source = GraphImputationSource::decode(&self.native_source)
            .map_err(GraphImputationTransportDecisionError::Invalid)?;
        if source.program() != *self.lowered.identity().as_bytes()
            || self.target.source_sample != self.producer.source_identity().to_hex()
        {
            return Err(GraphImputationTransportDecisionError::Invalid("retained native source changed"));
        }
        let reading = replay_transport_point(&self.native_source, &self.target, ctx)?;
        if reading.program != *self.lowered.identity().as_bytes()
            || reading.source != *self.producer.source_identity().as_bytes()
        {
            return Err(GraphImputationTransportDecisionError::Invalid("native replay belongs to another source or program"));
        }
        let decision = compile_chain_decision(
            effect_chain(&reading, &self.quantity), self.decision.clone(), ctx,
        )?.execute(ctx)?;
        Ok(GraphImputationTransportDecisionResult {
            producer: reading, decision,
            lowered_program_identity: self.lowered.identity(),
            calibration: "not_measured",
        })
    }

    /// Export the original source, target frame, final contract and live
    /// receipt after replaying the complete compiled schedule.
    ///
    /// # Errors
    /// A changed receipt, utility or artifact encoding.
    pub fn export(
        &self, result: &GraphImputationTransportDecisionResult,
        artifact_id: &str, ctx: &ExecutionContext,
    ) -> Result<Vec<u8>, GraphImputationTransportDecisionError> {
        let live = self.execute(ctx)?;
        if result.lowered_program_identity != self.lowered.identity()
            || result.calibration != "not_measured"
            || result.producer.receipt != live.producer.receipt
            || result.producer.draws != live.producer.draws
            || result.decision.decision != live.decision.decision
            || result.decision.chain.identity() != live.decision.chain.identity()
        {
            return Err(GraphImputationTransportDecisionError::Invalid("transport decision changed on replay"));
        }
        let contract = DecisionContractArtifact::new(self.decision.clone())?;
        let chain = compile_chain_decision(
            effect_chain(&live.producer, &self.quantity), self.decision.clone(), ctx,
        )?;
        let wire = GraphImputationTransportDecisionWire {
            version: 1, required_feature: FEATURE.into(),
            original_source: self.producer.source_bytes().to_vec(),
            target: self.target.clone(),
            quantity: ScientificQuantityWire::from(&self.quantity),
            decision_identity: contract.identity().into(),
            decision_artifact: contract.to_bytes(artifact_id)?,
            chain_artifact: chain.export(&live.decision, artifact_id)?,
            producer_receipt: live.producer.receipt,
            program_identity: *self.lowered.identity().as_bytes(),
            calibration: "not_measured".into(),
        };
        let bytes = antecedent_io::to_cbor(&wire)?;
        if bytes.len() > MAX_ARTIFACT_BYTES {
            return Err(GraphImputationTransportDecisionError::Invalid("combined artifact exceeds its byte bound"));
        }
        Ok(bytes)
    }
}

/// Reconstruct the source compiler from original rows, check its retained
/// operations and program, rebuild the outer compiler, and independently
/// replay imputation, graph selection, transport fitting and utility.
///
/// # Errors
/// Foreign feature or changed source, target, receipt or final decision.
pub fn consume_graph_imputation_transport_decision(
    bytes: &[u8], expected_program: SemanticDigest, ctx: &ExecutionContext,
) -> Result<GraphImputationTransportDecisionResult, GraphImputationTransportDecisionError> {
    if bytes.len() > MAX_ARTIFACT_BYTES {
        return Err(GraphImputationTransportDecisionError::Invalid("combined artifact exceeds its byte bound"));
    }
    let wire: GraphImputationTransportDecisionWire = antecedent_io::from_cbor(bytes)?;
    if wire.version != 1 || wire.required_feature != FEATURE
        || wire.program_identity != *expected_program.as_bytes()
        || wire.calibration != "not_measured"
    {
        return Err(GraphImputationTransportDecisionError::Invalid("combined feature or identity differs"));
    }
    let original = GraphImputationSource::decode(&wire.original_source)
        .map_err(GraphImputationTransportDecisionError::Invalid)?;
    let parts = original.parts();
    let table = TabularData::from_f64_columns(
        parts.names.iter().zip(&parts.columns)
            .map(|(name, column)| (name.as_str(), column.as_slice())),
    ).map_err(|_| GraphImputationTransportDecisionError::Invalid("stored source table is invalid"))?;
    let spec = GraphImputationSpec {
        treatment: VariableId::from_raw(parts.treatment),
        outcome: VariableId::from_raw(parts.outcome), support: parts.support,
        imputation: ImputationSpec::new()
            .with_imputations(parts.imputations).with_iterations(parts.iterations)
            .with_seed(parts.imputation_seed),
        domains: Arc::from(parts.domains.clone()),
        posterior_draws: parts.posterior_draws, prior_scale: parts.prior_scale,
        execution_seed: parts.execution_seed,
    };
    let producer = CompiledGraphImputation::compile(table, &spec, ctx)?;
    if producer.source_bytes() != wire.original_source.as_slice()
        || original.program() != *producer.lowered().identity().as_bytes()
    {
        return Err(GraphImputationTransportDecisionError::Invalid("original source compiler changed"));
    }
    let quantity: ScientificQuantity = wire.quantity.try_into()
        .map_err(GraphImputationTransportDecisionError::Invalid)?;
    let contract = DecisionContractArtifact::from_bytes(
        &wire.decision_artifact, &wire.decision_identity,
    )?;
    let plan = compile_graph_imputation_transport_decision(
        producer, wire.target, quantity, contract.contract().clone(),
    )?;
    if plan.lowered.identity() != expected_program {
        return Err(GraphImputationTransportDecisionError::Invalid("outer compiler changed"));
    }
    let live = plan.execute(ctx)?;
    if live.producer.receipt != wire.producer_receipt {
        return Err(GraphImputationTransportDecisionError::Invalid("target-standardized producer changed"));
    }
    let replay = consume_chain_decision(
        &wire.chain_artifact, Some(live.decision.lowered_program_identity), ctx,
    )?;
    if replay.chain.identity() != live.decision.chain.identity()
        || replay.decision != live.decision.decision
    {
        return Err(GraphImputationTransportDecisionError::Invalid("effect utility changed on replay"));
    }
    Ok(live)
}

#[cfg(test)]
mod tests {
    use super::*;
    use antecedent_contract::QuantityRole;
    use antecedent_core::graph_imputation_contract::GraphSupport;
    use antecedent_design::decision_contract::{
        ActionKind, DecisionAction, DecisionCriterion, StructuralPolicy, UtilityExpr,
    };

    #[test]
    #[ignore = "source-only full transport schedule assertion; reliability not earned"]
    fn missing_source_graph_mixture_transports_to_target_and_replays() {
        let mut treatment = Vec::new();
        let mut outcome = Vec::new();
        let mut covariate = Vec::new();
        for row in 0..80 {
            let x = f64::from(u8::from(row % 2 == 0));
            let a = f64::from(u8::from(row % 4 == 0 || row % 4 == 3));
            treatment.push(a);
            outcome.push(if row % 13 == 0 { f64::NAN }
                else { 0.2 * x + a * (0.5 + 0.5 * x) + 0.1 * (row % 5) as f64 });
            covariate.push(x);
        }
        let table = TabularData::from_f64_columns([
            ("t", treatment.as_slice()), ("y", outcome.as_slice()),
            ("x", covariate.as_slice()),
        ]).unwrap();
        let ctx = ExecutionContext::for_tests(19);
        let producer = CompiledGraphImputation::compile(table, &GraphImputationSpec {
            treatment: VariableId::from_raw(0), outcome: VariableId::from_raw(1),
            support: GraphSupport::PreTreatmentCovariates,
            imputation: ImputationSpec::new().with_imputations(2)
                .with_iterations(2).with_seed(17),
            domains: Arc::<[u8]>::from(vec![1, 0, 1]), posterior_draws: 100,
            prior_scale: 1.0, execution_seed: 19,
        }, &ctx).unwrap();
        let target = TransportStandardization {
            columns: vec![(2, (0..80).map(|row| f64::from(u8::from(row % 5 != 0))).collect())],
            selection_targets: vec![2],
            source_population: "source".into(), target_population: "target".into(),
            source_sample: producer.source_identity().to_hex(),
            target_sample: "independent-target-sample".into(),
            source_unit_ids: (0..80).map(|row| format!("source-{row}")).collect(),
            target_unit_ids: (0..80).map(|row| format!("target-{row}")).collect(),
        };
        let original = GraphImputationSource::decode(producer.source_bytes()).unwrap();
        let mut overlapping = target.clone();
        overlapping.target_unit_ids[0] = overlapping.source_unit_ids[0].clone();
        assert!(overlapping.validate(original.parts()).is_err());
        let quantity = ScientificQuantity {
            variable_id: "y".into(), variable_name: "target effect".into(),
            role: QuantityRole::Outcome, units: "score".into(),
            population_id: "target".into(),
            regime_id: "do(t=1)-do(t=0)".into(), horizon: 0,
            functional_id: "average_effect".into(), conditioning: Vec::new(),
            transform_id: "identity".into(),
        };
        let decision = DecisionContract {
            actions: vec![
                DecisionAction {
                    id: "treat".into(), kind: ActionKind::External,
                    inputs: vec![quantity.clone()],
                    utility: UtilityExpr::difference(
                        UtilityExpr::Input(0), UtilityExpr::Const(0.2)),
                },
                DecisionAction {
                    id: "hold".into(), kind: ActionKind::External,
                    inputs: vec![quantity.clone()], utility: UtilityExpr::Const(0.0),
                },
            ],
            utility_units: "score".into(),
            criterion: DecisionCriterion::PosteriorExpectedUtility,
            constraints: Vec::new(), target_population: "target".into(), horizon: 0,
            structural_policy: StructuralPolicy::RequireInvariantBestAction,
        };
        let plan = compile_graph_imputation_transport_decision(
            producer, target, quantity, decision,
        ).unwrap();
        assert_eq!(plan.plan_view().route().as_str(), "GraphImputationTransportDecision");
        let result = plan.execute(&ctx).unwrap();
        assert!(result.producer.mean.is_finite());
        assert_eq!(result.calibration, "not_measured");
        let bytes = plan.export(&result, "source-target-graph-decision", &ctx).unwrap();
        let replay = consume_graph_imputation_transport_decision(
            &bytes, result.lowered_program_identity, &ctx,
        ).unwrap();
        assert_eq!(replay.producer.receipt, result.producer.receipt);
        assert_eq!(replay.decision.decision, result.decision.decision);
    }
}
