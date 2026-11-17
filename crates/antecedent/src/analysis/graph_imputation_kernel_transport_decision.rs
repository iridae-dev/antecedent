//! Graph selection, missing-row imputation, mixed-covariate kernel transport,
//! fitted target effect and nonlinear utility under one compiler program.
//!
//! The source graph support still requires root treatment and pretreatment
//! covariates. A declared Gaussian kernel density ratio transports each
//! completed source table to an independent empirical target frame that may
//! include continuous and binary covariates. Each graph gets its own safe
//! adjustment set plus selected effect modifiers. Bandwidths and the overlap
//! floor are premises; target-frame uncertainty and final decision reliability
//! are not measured.

use std::sync::Arc;

use antecedent_contract::{ClaimKind, DataClassification, QuantityRole, ScientificQuantity};
use antecedent_core::graph_imputation_contract::GraphImputationSource;
use antecedent_core::{
    ByteEstimates, CalibrationBasis, CanonicalWriter, ExecutionContext,
    PhysicalEffect, PhysicalOperator, SemanticDigest, VariableId,
};
use antecedent_data::{ImputationSpec, TabularData};
use antecedent_design::decision_artifact::DecisionContractArtifact;
use antecedent_design::decision_contract::DecisionContract;
use antecedent_io::measured_inference::graph_candidate::{
    KernelTransportStandardization, replay_kernel_transport_point,
};
use antecedent_io::quantity_wire::ScientificQuantityWire;
use serde::{Deserialize, Serialize};

use super::graph_imputation::{CompiledGraphImputation, GraphImputationSpec};
use super::graph_imputation_transport_decision::{
    GraphImputationTransportDecisionError as Error,
    GraphImputationTransportDecisionResult, effect_chain,
};
use super::propagation_chain_decision::{compile_chain_decision, consume_chain_decision};
use crate::planner::lowered::{
    CarrierDemand, ClaimDemand, Composition, DecisionProgram, DesignShape,
    EvidenceDecisionProgram, EvidenceRole, EvidenceSourceRef, FunctionalKind,
    IdentificationLayer, LoweredProgram, NonGraphicalIdentification,
    PhysicalLayer, ProblemTarget, ProgramCell, ProgramIdentityScope,
    ScientificProblem, StandaloneCompositionLoweredParts, UncertaintyReading,
};
use crate::planner::statistical::{
    Realization, ResampleUnit, ResamplingMethod, ResamplingPlan, StatisticalProgram,
};
use crate::planner::{PlanStage, PlanView};

const FEATURE: &str = "graph_imputation_kernel_transport_decision_v2";
const BOOTSTRAP_FEATURE: &str = "graph_imputation_kernel_transport_target_bootstrap_decision_v2";
const MAX_ARTIFACT_BYTES: usize = 80 * 1024 * 1024;

/// Source graph-imputation plan, mixed target frame and final utility under
/// one retained six-layer compiler identity.
pub struct CompiledGraphImputationKernelTransportDecision {
    producer: CompiledGraphImputation,
    target: KernelTransportStandardization,
    quantity: ScientificQuantity,
    decision: DecisionContract,
    native_source: Vec<u8>,
    lowered: Arc<LoweredProgram>,
    target_bootstrap: Option<TargetFrameBootstrap>,
}

/// Prespecified iid empirical bootstrap of independent target units.
/// Each replicate refits the full graph-imputation transport schedule and
/// contributes one posterior draw. This is a source-only uncertainty model;
/// it has no earned population coverage or decision reliability.
#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TargetFrameBootstrap {
    /// Number of independent target-frame bootstrap replicates, from 40 to 256.
    pub replicates: u32,
    /// Deterministic seed for target-row resampling.
    pub seed: u64,
}

/// Unmeasured target effect interval candidate at the exact kernel policy.
pub struct GraphImputationKernelTransportCalibrationCandidate {
    /// Complete graph, imputation, kernel, fit and decision execution.
    pub result: GraphImputationTransportDecisionResult,
    /// Nearest-rank central 95% effect posterior interval.
    pub interval: (f64, f64),
    /// Prospective whole-route calibration coordinate.
    pub basis: CalibrationBasis,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct KernelTransportDecisionWire {
    version: u32,
    required_feature: String,
    original_source: Vec<u8>,
    target: KernelTransportStandardization,
    quantity: ScientificQuantityWire,
    decision_identity: String,
    decision_artifact: Vec<u8>,
    chain_artifact: Vec<u8>,
    producer_receipt: [u8; 32],
    program_identity: [u8; 32],
    calibration: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    target_bootstrap: Option<TargetFrameBootstrap>,
}

fn check_quantity(
    source: &GraphImputationSource, target: &KernelTransportStandardization,
    quantity: &ScientificQuantity, decision: &DecisionContract,
) -> Result<(), Error> {
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
        || decision.actions.iter().any(|action| {
            action.inputs.len() != 1
                || action.inputs[0].require_same_coordinate(quantity).is_err()
        })
    {
        return Err(Error::Invalid("utility must read the exact mixed-target average-effect coordinate"));
    }
    Ok(())
}

fn lower_outer(
    source: &GraphImputationSource, target: &KernelTransportStandardization,
    quantity: &ScientificQuantity, decision: &DecisionContract,
    target_bootstrap: Option<TargetFrameBootstrap>,
) -> Result<Arc<LoweredProgram>, Error> {
    target.validate(source.parts())?;
    if target_bootstrap.is_some_and(|policy| !(40..=256).contains(&policy.replicates)) {
        return Err(Error::Invalid("target bootstrap needs 40 to 256 replicates"));
    }
    check_quantity(source, target, quantity, decision)?;
    let target_digest = target.digest()?;
    let source_digest = SemanticDigest::from_bytes(source.parts().data_digest);
    let coordinate = antecedent_io::to_cbor(&ScientificQuantityWire::from(quantity))?;
    let contract = decision.identity()?;
    let mut writer = CanonicalWriter::new(
        "antecedent.graph_imputation_kernel_transport_decision.question.v1",
    );
    writer.bytes(&source.program()).bytes(&source.parts().contract_digest())
        .bytes(source_digest.as_bytes()).bytes(&target_digest)
        .bytes(&coordinate).text(&contract).text(FEATURE);
    if let Some(policy) = target_bootstrap {
        writer.text(BOOTSTRAP_FEATURE).word(u64::from(policy.replicates)).word(policy.seed);
    }
    let question = writer.finish();
    let rows = source.parts().rows().saturating_add(target.rows()) as u64;
    let design = DesignShape::quoted(rows, source.parts().names.len() as u64, true);
    let scientific = ScientificProblem {
        runtime_query: None, identity_scope: ProgramIdentityScope::Semantic,
        target: ProblemTarget::QueryFunctional {
            kind: FunctionalKind::PolicyValue,
            outcome: Some(VariableId::from_raw(source.parts().outcome)), query: question,
        },
        plan_id: Arc::from(if target_bootstrap.is_some() { BOOTSTRAP_FEATURE }
            else { FEATURE }),
        data: DataClassification::Tabular,
        graph_class: "bounded_graph_imputation_mixed_kernel_target",
        query_variables: Arc::from([]), source_digest: Some(*source_digest.as_bytes()),
        support_coordinate: None, supported: false, support_status: None,
    };
    let identification = IdentificationLayer::NonGraphical(Arc::new(
        NonGraphicalIdentification::SharedEvidenceContract { contract: question },
    ));
    let statistical = StatisticalProgram {
        identifier: None, estimator: None,
        realization: Realization::GraphImputationV1,
        models: Arc::from([]), folds: None,
        resampling: target_bootstrap.map(|policy| ResamplingPlan {
            method: ResamplingMethod::Bootstrap, replicates: policy.replicates,
            unit: ResampleUnit::Row,
        }),
        posterior: None, design, operators: Arc::from([]),
    };
    let evidence = EvidenceDecisionProgram {
        sources: Arc::from([
            EvidenceSourceRef { role: EvidenceRole::Observations, digest: Some(*source_digest.as_bytes()) },
            EvidenceSourceRef { role: EvidenceRole::TransportSource,
                digest: Some(target_digest) },
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
            scenario_source: Some(Arc::from("graph_imputation_kernel_target_effect_draws_v1")),
        }), planning: None, chain: None,
    };
    let mut operators = vec![PhysicalOperator::declare(PhysicalEffect::Read)
        .estimates(ByteEstimates::absent().with_output(rows.saturating_mul(8)))
        .build().map_err(|_| Error::Invalid("invalid mixed source read operator"))?];
    // The empirical-frame fit runs once, followed by one complete fit and
    // transport per bootstrap replicate. Keep the repeated work visible in
    // the physical schedule; these operations execute sequentially.
    let executions = 1 + target_bootstrap.map_or(0, |policy| policy.replicates);
    for _ in 0..executions {
        operators.push(PhysicalOperator::declare(PhysicalEffect::Fit)
            .build().map_err(|_| Error::Invalid("invalid mixed graph fit operator"))?);
        operators.push(PhysicalOperator::declare(PhysicalEffect::Transport)
            .build().map_err(|_| Error::Invalid("invalid kernel standardization operator"))?);
    }
    let physical = PhysicalLayer { operators: Arc::from(operators), design };
    Ok(Arc::new(StandaloneCompositionLoweredParts {
        cell: ProgramCell::GraphImputationKernelTransportDecision,
        scientific, identification, statistical, evidence, physical,
        claim: ClaimDemand { uncertainty: UncertaintyReading::Posterior,
            kind: Some(ClaimKind::PolicyValue) },
    }.finish()))
}

/// Bind source graph operations, raw and missing rows, mixed target frame,
/// kernel bandwidths, overlap policy, exact effect coordinate and utility
/// before fitting or target standardization.
///
/// # Errors
/// Invalid source binding, target support, contract or six-layer lowering.
pub fn compile_graph_imputation_kernel_transport_decision(
    producer: CompiledGraphImputation, target: KernelTransportStandardization,
    quantity: ScientificQuantity, decision: DecisionContract,
) -> Result<CompiledGraphImputationKernelTransportDecision, Error> {
    let source = GraphImputationSource::decode(producer.source_bytes())
        .map_err(Error::Invalid)?;
    if source.program() != *producer.lowered().identity().as_bytes()
        || target.source_sample != producer.source_identity().to_hex()
    {
        return Err(Error::Invalid("mixed target source-sample identity differs from retained producer"));
    }
    let lowered = lower_outer(&source, &target, &quantity, &decision, None)?;
    let native_source = GraphImputationSource::new(
        source.parts().clone(), *lowered.identity().as_bytes(),
    ).map_err(Error::Invalid)?.encode();
    Ok(CompiledGraphImputationKernelTransportDecision {
        producer, target, quantity, decision, native_source, lowered, target_bootstrap: None,
    })
}

/// Compile an independent iid target-frame bootstrap into the complete
/// graph-imputation, kernel transport and utility program. The empirical
/// target rows are resampled together, preserving their joint distribution.
///
/// # Errors
/// Invalid source, target, bootstrap policy or compiler binding.
pub fn compile_graph_imputation_kernel_transport_target_bootstrap_decision(
    producer: CompiledGraphImputation, target: KernelTransportStandardization,
    quantity: ScientificQuantity, decision: DecisionContract, policy: TargetFrameBootstrap,
) -> Result<CompiledGraphImputationKernelTransportDecision, Error> {
    let source = GraphImputationSource::decode(producer.source_bytes()).map_err(Error::Invalid)?;
    if source.program() != *producer.lowered().identity().as_bytes()
        || target.source_sample != producer.source_identity().to_hex()
    {
        return Err(Error::Invalid("mixed target source-sample identity differs from retained producer"));
    }
    let lowered = lower_outer(&source, &target, &quantity, &decision, Some(policy))?;
    let native_source = GraphImputationSource::new(
        source.parts().clone(), *lowered.identity().as_bytes(),
    ).map_err(Error::Invalid)?.encode();
    Ok(CompiledGraphImputationKernelTransportDecision {
        producer, target, quantity, decision, native_source, lowered,
        target_bootstrap: Some(policy),
    })
}

// SplitMix64 provides a stable, locally specified stream for portable replay.
fn bootstrap_word(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}

fn bootstrap_pool_index(state: &mut u64, len: usize) -> usize {
    let bound = len as u64;
    let threshold = bound.wrapping_neg() % bound;
    loop {
        let word = bootstrap_word(state);
        if word >= threshold { return (word % bound) as usize; }
    }
}

impl CompiledGraphImputationKernelTransportDecision {
    /// All six compiler layers of the mixed target effect decision.
    #[must_use]
    pub fn lowered(&self) -> &LoweredProgram { &self.lowered }

    /// Inspect the schedule before fitting or kernel standardization.
    #[must_use]
    pub fn plan_view(&self) -> PlanView {
        PlanView::of_program(Arc::clone(&self.lowered), PlanStage::Compiled)
    }

    /// Reimpute source rows, reselect graphs, fit kernel-standardized children
    /// and score nonlinear effect utility under this retained program.
    ///
    /// # Errors
    /// Source, overlap, fit or decision refusal.
    pub fn execute(&self, ctx: &ExecutionContext)
        -> Result<GraphImputationTransportDecisionResult, Error> {
        let source = GraphImputationSource::decode(&self.native_source)
            .map_err(Error::Invalid)?;
        if source.program() != *self.lowered.identity().as_bytes()
            || self.target.source_sample != self.producer.source_identity().to_hex()
        {
            return Err(Error::Invalid("retained mixed source changed"));
        }
        let mut reading = replay_kernel_transport_point(&self.native_source, &self.target, ctx)?;
        if let Some(policy) = self.target_bootstrap {
            // Account for the sampled column copy, row indices and pooled draws;
            // each nested replay separately admits its own fitting workspace.
            let bootstrap_bytes = (self.target.rows() as u64)
                .saturating_mul((self.target.columns.len() as u64).saturating_mul(2) + 1)
                .saturating_mul(8)
                .saturating_add(self.target.source_unit_ids.iter()
                    .chain(&self.target.target_unit_ids)
                    .map(|id| (id.len() as u64).saturating_add(24))
                    .fold(0_u64, u64::saturating_add))
                .saturating_add(u64::from(policy.replicates).saturating_mul(8));
            let _bootstrap_lease = ctx.acquire_admission(0, bootstrap_bytes)
                .map_err(|_| Error::Invalid("target bootstrap admission refused"))?;
            let mut state = policy.seed;
            // Domain-separated stream selects uniformly across all imputation/graph draws.
            let mut draw_state = antecedent_core::execution::mix64(policy.seed ^ 0x4452_4157_5f56_3201);
            let rows = self.target.rows();
            let mut draws = Vec::with_capacity(policy.replicates as usize);
            let mut receipt = CanonicalWriter::new(
                "antecedent.graph_imputation_kernel_transport.target_bootstrap.execution.v2",
            );
            receipt.bytes(self.lowered.identity().as_bytes())
                .bytes(&self.target.digest()?).word(u64::from(policy.replicates)).word(policy.seed);
            for replicate in 0..policy.replicates {
                if ctx.cancellation.is_cancelled() {
                    return Err(Error::Invalid("target bootstrap execution cancelled"));
                }
                let mut sampled = self.target.clone();
                let indices: Vec<usize> = (0..rows).map(|_| {
                    // Rejection avoids modulo bias for arbitrary frame sizes.
                    let bound = rows as u64;
                    let threshold = bound.wrapping_neg() % bound;
                    loop {
                        let word = bootstrap_word(&mut state);
                        if word >= threshold { break (word % bound) as usize; }
                    }
                }).collect();
                for ((_, values), (_, original)) in sampled.columns.iter_mut().zip(&self.target.columns) {
                    *values = indices.iter().map(|&index| original[index]).collect();
                }
                // Unit IDs identify bootstrap slots; sampled indices are sealed below.
                let run = replay_kernel_transport_point(&self.native_source, &sampled, ctx)?;
                if run.draws.is_empty() { return Err(Error::Invalid("target bootstrap has no posterior draws")); }
                let draw_index = bootstrap_pool_index(&mut draw_state, run.draws.len());
                let draw = run.draws[draw_index];
                receipt.word(u64::from(replicate));
                for index in indices { receipt.word(index as u64); }
                receipt.bytes(&run.receipt).word(draw_index as u64).word(draw.to_bits());
                draws.push(draw);
            }
            let mean = draws.iter().sum::<f64>() / draws.len() as f64;
            let variance = draws.iter().map(|value| (value - mean).powi(2)).sum::<f64>()
                / (draws.len() - 1) as f64;
            reading.mean = mean;
            reading.diagnostic_sd = variance.sqrt();
            reading.draws = draws;
            reading.receipt = receipt.finish();
        }
        if reading.program != *self.lowered.identity().as_bytes()
            || reading.source != *self.producer.source_identity().as_bytes()
        {
            return Err(Error::Invalid("mixed replay belongs to another source or program"));
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

    /// Complete-route target effect interval candidate under the prespecified
    /// kernel policy. It does not confer decision reliability.
    ///
    /// # Errors
    /// A source, overlap or posterior-draw refusal.
    pub fn calibration_candidate(&self, ctx: &ExecutionContext)
        -> Result<GraphImputationKernelTransportCalibrationCandidate, Error> {
        let result = self.execute(ctx)?;
        let mut ordered = result.producer.draws.clone();
        if ordered.len() < 40 || ordered.iter().any(|value| !value.is_finite()) {
            return Err(Error::Invalid("mixed target effect interval needs finite posterior draws"));
        }
        ordered.sort_by(f64::total_cmp);
        let rank = |numerator: usize| {
            (ordered.len() * numerator).div_ceil(1000).saturating_sub(1)
        };
        let interval = (ordered[rank(25)], ordered[rank(975)]);
        let source = GraphImputationSource::decode(&self.native_source)
            .map_err(Error::Invalid)?;
        let parts = source.parts();
        let mut policy = CanonicalWriter::new(
            "antecedent.graph_imputation_kernel_transport.calibration_policy.v1",
        );
        policy.word(self.target.bandwidths.len() as u64);
        for &(id, bandwidth) in &self.target.bandwidths {
            policy.word(id as u64).word(bandwidth.to_bits());
        }
        policy.word(self.target.min_kernel_overlap.to_bits());
        let policy = SemanticDigest::from_bytes(policy.finish()).to_hex();
        let mut functional = format!(
            "target_average_effect:graph_support={}:graphs={}:domains={}:selected={:?}:m={}:sweeps={}:draws={}:prior_bits={:016x}:kernel_policy={policy}",
            parts.support.as_str(), parts.atoms.len(),
            parts.domains.iter().map(u8::to_string).collect::<String>(),
            self.target.selection_targets, parts.imputations, parts.iterations,
            parts.posterior_draws, parts.prior_scale.to_bits(),
        );
        let (feature, model, draws) = if let Some(bootstrap) = self.target_bootstrap {
            functional.push_str(&format!(
                ":source_n={}:target_n={}:target_bootstrap=iid_rows_unbiased_pool_v2:replicates={}:seed={}",
                parts.rows(), self.target.rows(), bootstrap.replicates, bootstrap.seed,
            ));
            (BOOTSTRAP_FEATURE, "mar_graph_mixture_kernel_target_iid_bootstrap_v1",
                bootstrap.replicates)
        } else {
            functional.push_str(&format!(":source_n={}:target_n={}:s_admissibility=v2", parts.rows(), self.target.rows()));
            (FEATURE, "mar_graph_mixture_kernel_target_v2",
                parts.posterior_draws.saturating_mul(parts.imputations))
        };
        let basis = CalibrationBasis::new(
            [
                "AverageEffect", "Dag", "graph_posterior", "multi_env", "Bayesian",
                feature, "nearest_rank_95", "",
                "independent_source_target_units", model, &functional,
            ].map(Arc::from),
            0.95, Arc::from("model_conditional"),
            (parts.rows() as u64).saturating_add(self.target.rows() as u64),
            None, Some(draws), 0.0,
        ).with_validation("none");
        Ok(GraphImputationKernelTransportCalibrationCandidate { result, interval, basis })
    }

    /// Re-execute and seal the original source, mixed target, utility and
    /// child decision artifact for independent replay.
    ///
    /// # Errors
    /// Changed result or artifact encoding refusal.
    pub fn export(
        &self, result: &GraphImputationTransportDecisionResult,
        artifact_id: &str, ctx: &ExecutionContext,
    ) -> Result<Vec<u8>, Error> {
        let live = self.execute(ctx)?;
        if result.lowered_program_identity != self.lowered.identity()
            || result.calibration != "not_measured"
            || result.producer.receipt != live.producer.receipt
            || result.producer.draws != live.producer.draws
            || result.decision.decision != live.decision.decision
            || result.decision.chain.identity() != live.decision.chain.identity()
        {
            return Err(Error::Invalid("mixed target decision changed on replay"));
        }
        let contract = DecisionContractArtifact::new(self.decision.clone())?;
        let chain = compile_chain_decision(
            effect_chain(&live.producer, &self.quantity), self.decision.clone(), ctx,
        )?;
        let wire = KernelTransportDecisionWire {
            version: 1, required_feature: if self.target_bootstrap.is_some() { BOOTSTRAP_FEATURE } else { FEATURE }.into(),
            original_source: self.producer.source_bytes().to_vec(),
            target: self.target.clone(),
            quantity: ScientificQuantityWire::from(&self.quantity),
            decision_identity: contract.identity().into(),
            decision_artifact: contract.to_bytes(artifact_id)?,
            chain_artifact: chain.export(&live.decision, artifact_id)?,
            producer_receipt: live.producer.receipt,
            program_identity: *self.lowered.identity().as_bytes(),
            calibration: "not_measured".into(),
            target_bootstrap: self.target_bootstrap,
        };
        let bytes = antecedent_io::to_cbor(&wire)?;
        if bytes.len() > MAX_ARTIFACT_BYTES {
            return Err(Error::Invalid("mixed target artifact exceeds its byte bound"));
        }
        Ok(bytes)
    }
}

/// Rebuild the source compiler, validate the kernel target and reexecute
/// graph selection, imputation, target fitting and utility from portable bytes.
///
/// # Errors
/// Foreign feature or changed source, target, receipt or decision.
pub fn consume_graph_imputation_kernel_transport_decision(
    bytes: &[u8], expected_program: SemanticDigest, ctx: &ExecutionContext,
) -> Result<GraphImputationTransportDecisionResult, Error> {
    if bytes.len() > MAX_ARTIFACT_BYTES {
        return Err(Error::Invalid("mixed target artifact exceeds its byte bound"));
    }
    let wire: KernelTransportDecisionWire = antecedent_io::from_cbor(bytes)?;
    let expected_feature = if wire.target_bootstrap.is_some() { BOOTSTRAP_FEATURE } else { FEATURE };
    if wire.version != 1 || wire.required_feature != expected_feature
        || wire.program_identity != *expected_program.as_bytes()
        || wire.calibration != "not_measured"
    {
        return Err(Error::Invalid("mixed target feature or identity differs"));
    }
    let original = GraphImputationSource::decode(&wire.original_source)
        .map_err(Error::Invalid)?;
    let parts = original.parts();
    let table = TabularData::from_f64_columns(
        parts.names.iter().zip(&parts.columns)
            .map(|(name, column)| (name.as_str(), column.as_slice())),
    ).map_err(|_| Error::Invalid("stored mixed source table is invalid"))?;
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
        return Err(Error::Invalid("original mixed source compiler changed"));
    }
    let quantity: ScientificQuantity = wire.quantity.try_into().map_err(Error::Invalid)?;
    let contract = DecisionContractArtifact::from_bytes(
        &wire.decision_artifact, &wire.decision_identity,
    )?;
    let plan = if let Some(policy) = wire.target_bootstrap {
        compile_graph_imputation_kernel_transport_target_bootstrap_decision(
            producer, wire.target, quantity, contract.contract().clone(), policy,
        )?
    } else {
        compile_graph_imputation_kernel_transport_decision(
            producer, wire.target, quantity, contract.contract().clone(),
        )?
    };
    if plan.lowered.identity() != expected_program {
        return Err(Error::Invalid("mixed target outer compiler changed"));
    }
    let live = plan.execute(ctx)?;
    if live.producer.receipt != wire.producer_receipt {
        return Err(Error::Invalid("mixed target producer changed"));
    }
    let replay = consume_chain_decision(
        &wire.chain_artifact, Some(live.decision.lowered_program_identity), ctx,
    )?;
    if replay.chain.identity() != live.decision.chain.identity()
        || replay.decision != live.decision.decision
    {
        return Err(Error::Invalid("mixed effect utility changed on replay"));
    }
    Ok(live)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "source-only bootstrap selector fixture"]
    fn pooled_selector_reaches_later_imputations() {
        let mut state = antecedent_core::execution::mix64(7 ^ 0x4452_4157_5f56_3201);
        let indices: Vec<_> = (0..40).map(|_| bootstrap_pool_index(&mut state, 200)).collect();
        assert!(indices.iter().any(|&index| index < 100));
        assert!(indices.iter().any(|&index| index >= 100));
    }

    use antecedent_core::graph_imputation_contract::GraphSupport;
    use antecedent_design::decision_contract::{
        ActionKind, DecisionAction, DecisionCriterion, StructuralPolicy, UtilityExpr,
    };

    #[test]
    #[ignore = "source-only mixed target route; whole-route reliability is unearned"]
    fn continuous_target_modifier_replays_graph_imputation_and_utility() {
        let mut treatment = Vec::new();
        let mut outcome = Vec::new();
        let mut covariate = Vec::new();
        for row in 0..80 {
            let x = (row % 20) as f64 / 19.0;
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
        let ctx = ExecutionContext::for_tests(29);
        let producer = CompiledGraphImputation::compile(table, &GraphImputationSpec {
            treatment: VariableId::from_raw(0), outcome: VariableId::from_raw(1),
            support: GraphSupport::TreatmentRootPreTreatmentCovariates,
            imputation: ImputationSpec::new().with_imputations(2)
                .with_iterations(2).with_seed(23),
            domains: Arc::<[u8]>::from(vec![1, 0, 0]), posterior_draws: 100,
            prior_scale: 1.0, execution_seed: 29,
        }, &ctx).unwrap();
        let target = KernelTransportStandardization {
            columns: vec![(2, (0..80)
                .map(|row| 0.2 + 0.8 * (row % 20) as f64 / 19.0).collect())],
            selection_targets: vec![2],
            source_population: "source".into(), target_population: "target".into(),
            source_sample: producer.source_identity().to_hex(),
            target_sample: "independent-continuous-target".into(),
            source_unit_ids: (0..80).map(|row| format!("source-{row}")).collect(),
            target_unit_ids: (0..80).map(|row| format!("target-{row}")).collect(),
            bandwidths: vec![(2, 0.3)], min_kernel_overlap: 0.01,
        };
        let original = GraphImputationSource::decode(producer.source_bytes()).unwrap();
        assert!(target.validate(original.parts()).is_ok());
        let mut disjoint = target.clone();
        disjoint.target_unit_ids[0] = disjoint.source_unit_ids[0].clone();
        assert!(disjoint.validate(original.parts()).is_err());
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
        let parts = original.parts();
        let bootstrap_table = TabularData::from_f64_columns(
            parts.names.iter().zip(&parts.columns)
                .map(|(name, column)| (name.as_str(), column.as_slice())),
        ).unwrap();
        let bootstrap_producer = CompiledGraphImputation::compile(
            bootstrap_table, &GraphImputationSpec {
                treatment: VariableId::from_raw(parts.treatment),
                outcome: VariableId::from_raw(parts.outcome), support: parts.support,
                imputation: ImputationSpec::new().with_imputations(parts.imputations)
                    .with_iterations(parts.iterations).with_seed(parts.imputation_seed),
                domains: Arc::from(parts.domains.clone()), posterior_draws: parts.posterior_draws,
                prior_scale: parts.prior_scale, execution_seed: parts.execution_seed,
            }, &ctx,
        ).unwrap();
        let bootstrap = compile_graph_imputation_kernel_transport_target_bootstrap_decision(
            bootstrap_producer, target.clone(), quantity.clone(), decision.clone(),
            TargetFrameBootstrap { replicates: 40, seed: 7 },
        ).unwrap();
        let plan = compile_graph_imputation_kernel_transport_decision(
            producer, target, quantity, decision,
        ).unwrap();
        assert_eq!(plan.plan_view().route().as_str(),
            "GraphImputationKernelTransportDecision");
        let result = plan.execute(&ctx).unwrap();
        assert!(result.producer.mean.is_finite());
        assert_eq!(result.calibration, "not_measured");
        let bytes = plan.export(&result, "mixed-source-target-decision", &ctx).unwrap();
        let replay = consume_graph_imputation_kernel_transport_decision(
            &bytes, result.lowered_program_identity, &ctx,
        ).unwrap();
        assert_eq!(replay.producer.receipt, result.producer.receipt);
        assert_eq!(replay.decision.decision, result.decision.decision);
        assert_ne!(bootstrap.lowered().identity(), plan.lowered().identity());
        let sampled = bootstrap.execute(&ctx).unwrap();
        assert_eq!(sampled.producer.draws.len(), 40);
        assert_eq!(sampled.calibration, "not_measured");
        let candidate = bootstrap.calibration_candidate(&ctx).unwrap();
        assert!(candidate.interval.0 <= candidate.interval.1);
        assert_eq!(candidate.result.calibration, "not_measured");
        let bytes = bootstrap.export(&sampled, "target-bootstrap", &ctx).unwrap();
        let replay = consume_graph_imputation_kernel_transport_decision(
            &bytes, sampled.lowered_program_identity, &ctx,
        ).unwrap();
        assert_eq!(sampled.producer.draws, replay.producer.draws);
        assert_eq!(sampled.producer.receipt, replay.producer.receipt);
        assert_eq!(sampled.decision.decision, replay.decision.decision);
    }
}
