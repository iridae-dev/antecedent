//! Weighted complete-case target transport under a declared MAR mechanism.
//! The supplied observation probabilities and MAR are assumptions; target
//! sampling uncertainty and calibration are not inferred by this route.
use super::graph_imputation::{CompiledGraphImputation, GraphImputationError, GraphImputationSpec};
use super::graph_imputation_transport_decision::{
    POOLED_INTERVAL, REPLICATE_INTERVAL, effect_chain, imputation_pooled_interval_95,
    order_statistic_interval_95, transported_child_df,
};
use super::propagation_chain_decision::{
    ChainDecisionError, ChainDecisionResult, compile_chain_decision, consume_chain_decision,
};
use crate::planner::PlanView;
use crate::planner::lowered::{
    CarrierDemand, ClaimDemand, Composition, DecisionProgram, DesignShape, EvidenceDecisionProgram,
    EvidenceRole, EvidenceSourceRef, FunctionalKind, IdentificationLayer, LoweredProgram,
    NonGraphicalIdentification, PhysicalLayer, ProblemTarget, ProgramCell, ProgramIdentityScope,
    ScientificProblem, StandaloneCompositionLoweredParts, UncertaintyReading,
};
use crate::planner::statistical::{
    Realization, ResampleUnit, ResamplingMethod, ResamplingPlan, StatisticalProgram,
};
use antecedent_contract::{ClaimKind, DataClassification, QuantityRole, ScientificQuantity};
use antecedent_core::graph_imputation_contract::GraphImputationSource;
use antecedent_core::{
    CalibrationBasis, CanonicalWriter, ExecutionContext, PhysicalEffect, PhysicalOperator,
    SemanticDigest, VariableId,
};
use antecedent_data::{ImputationSpec, TabularData};
use antecedent_design::decision_artifact::DecisionContractArtifact;
use antecedent_design::decision_contract::{DecisionContract, DecisionContractError};
use antecedent_io::IoError;
use antecedent_io::measured_inference::graph_candidate::{
    GraphPointReading, MarTargetStandardization, replay_mar_transport_point, replicate_source,
};
use antecedent_io::quantity_wire::ScientificQuantityWire;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

const FEATURE: &str = "graph_imputation_mar_transport_decision_v2";
const MAX_BYTES: usize = 80 * 1024 * 1024;

/// Refusal of the MAR target, upstream producer, utility or replay.
#[derive(Debug, thiserror::Error)]
pub enum MarTransportError {
    /// Invalid target or identity.
    #[error("MAR target transport: {0}")]
    Invalid(&'static str),
    /// Producer refused.
    #[error("MAR target producer: {0}")]
    Producer(#[from] GraphImputationError),
    /// Utility contract refused.
    #[error("MAR target contract: {0}")]
    Contract(#[from] DecisionContractError),
    /// Effect decision refused.
    #[error("MAR target decision: {0}")]
    Chain(#[from] ChainDecisionError),
    /// Raw replay or encoding refused.
    #[error("MAR target artifact: {0}")]
    Io(#[from] IoError),
}

/// Compiled source, target mechanism and effect utility.
pub struct CompiledMarTransportDecision {
    producer: CompiledGraphImputation,
    target: MarTargetStandardization,
    quantity: ScientificQuantity,
    decision: DecisionContract,
    native_source: Vec<u8>,
    lowered: Arc<LoweredProgram>,
    bootstrap: Option<MarClusterBootstrap>,
}
/// Model-conditional decision with no earned calibration.
pub struct MarTransportResult {
    /// Native graph-imputation and weighted target result.
    pub producer: GraphPointReading,
    /// Decision over the target effect posterior.
    pub decision: ChainDecisionResult,
    /// Full compiler identity.
    pub lowered_program_identity: SemanticDigest,
    /// Always not measured.
    pub calibration: &'static str,
}
/// Independent-cluster resampling of the raw target frame. Each replicate
/// refits the source on its own imputation and posterior streams and retains
/// one draw; the interval is the replicate order-statistic interval.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MarClusterBootstrap {
    /// Number of whole-frame replicates (40 to 256).
    pub replicates: u32,
    /// Bootstrap seed.
    pub seed: u64,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Wire {
    version: u32,
    feature: String,
    source: Vec<u8>,
    target: MarTargetStandardization,
    quantity: ScientificQuantityWire,
    contract_identity: String,
    contract: Vec<u8>,
    chain: Vec<u8>,
    receipt: [u8; 32],
    program: [u8; 32],
    calibration: String,
    bootstrap: Option<MarClusterBootstrap>,
}

/// Compile a raw declared-MAR target before fitting the graph mixture.
///
/// # Errors
/// Invalid source, target support, estimand, utility or lowering.
#[allow(
    clippy::result_large_err,
    reason = "the structured refusal is returned by value on a cold path"
)]
pub fn compile_graph_imputation_mar_transport_decision(
    producer: CompiledGraphImputation,
    target: MarTargetStandardization,
    quantity: ScientificQuantity,
    decision: DecisionContract,
) -> Result<CompiledMarTransportDecision, MarTransportError> {
    compile_graph_imputation_mar_transport_with_cluster(producer, target, quantity, decision, None)
}

/// Compile with an optional independent-cluster bootstrap of the raw target.
///
/// # Errors
/// Invalid source, target, cluster design, estimand or utility.
#[allow(
    clippy::result_large_err,
    reason = "the structured refusal is returned by value on a cold path"
)]
#[allow(clippy::too_many_lines, reason = "one linear sequence of steps that reads top to bottom")]
pub fn compile_graph_imputation_mar_transport_with_cluster(
    producer: CompiledGraphImputation,
    target: MarTargetStandardization,
    quantity: ScientificQuantity,
    decision: DecisionContract,
    bootstrap: Option<MarClusterBootstrap>,
) -> Result<CompiledMarTransportDecision, MarTransportError> {
    let source = GraphImputationSource::decode(producer.source_bytes())
        .map_err(MarTransportError::Invalid)?;
    if source.program() != *producer.lowered().identity().as_bytes()
        || target.source_sample != producer.source_identity().to_hex()
    {
        return Err(MarTransportError::Invalid("target source identity differs from producer"));
    }
    target.validate(source.parts())?;
    if let Some(policy) = bootstrap {
        if !(40..=256).contains(&policy.replicates) {
            return Err(MarTransportError::Invalid("cluster bootstrap needs 40 to 256 replicates"));
        }
        cluster_groups(&target)?;
    }
    decision.validate()?;
    let parts = source.parts();
    if quantity.role != QuantityRole::Outcome
        || quantity.variable_id != parts.names[parts.outcome as usize]
        || quantity.population_id != target.target_population
        || quantity.population_id != decision.target_population
        || quantity.regime_id
            != format!(
                "do({}=1)-do({}=0)",
                parts.names[parts.treatment as usize], parts.names[parts.treatment as usize]
            )
        || quantity.functional_id != "average_effect"
        || quantity.horizon != 0
        || quantity.horizon != decision.horizon
        || !quantity.conditioning.is_empty()
        || quantity.transform_id != "identity"
        || decision.actions.is_empty()
        || decision
            .actions
            .iter()
            .any(|a| a.inputs.len() != 1 || a.inputs[0].require_same_coordinate(&quantity).is_err())
    {
        return Err(MarTransportError::Invalid(
            "decision must read the exact target average-effect coordinate",
        ));
    }
    let digest = target.digest()?;
    let contract = decision.identity()?;
    let mut writer = CanonicalWriter::new("antecedent.graph_imputation.mar_transport.question.v2");
    writer
        .bytes(&source.program())
        .bytes(&parts.contract_digest())
        .bytes(&parts.data_digest)
        .bytes(&digest)
        .bytes(&antecedent_io::to_cbor(&ScientificQuantityWire::from(&quantity))?)
        .text(&contract);
    if let Some(policy) = bootstrap {
        writer.text("independent_clusters_v1").word(u64::from(policy.replicates)).word(policy.seed);
    } else {
        writer.text("fixed_target_frame_v1");
    }
    let question = writer.finish();
    let design = DesignShape::quoted(
        (parts.rows() + target.frame.row_ids.len()) as u64,
        parts.names.len() as u64,
        true,
    );
    let scientific = ScientificProblem {
        runtime_query: None,
        identity_scope: ProgramIdentityScope::Semantic,
        target: ProblemTarget::QueryFunctional {
            kind: FunctionalKind::PolicyValue,
            outcome: Some(VariableId::from_raw(parts.outcome)),
            query: question,
        },
        plan_id: Arc::from(FEATURE),
        data: DataClassification::Tabular,
        graph_class: if target.kernel {
            if target.confounded { "mar_confounded_kernel_target" } else { "mar_kernel_target" }
        } else if target.confounded {
            "mar_confounded_binary_target"
        } else {
            "mar_binary_target"
        },
        query_variables: Arc::from([]),
        source_digest: Some(parts.data_digest),
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
        resampling: bootstrap.map(|policy| ResamplingPlan {
            method: ResamplingMethod::Bootstrap,
            replicates: policy.replicates,
            unit: ResampleUnit::Cluster,
        }),
        posterior: None,
        design,
        operators: Arc::from([]),
    };
    let evidence = EvidenceDecisionProgram {
        sources: Arc::from([
            EvidenceSourceRef { role: EvidenceRole::Observations, digest: Some(parts.data_digest) },
            EvidenceSourceRef { role: EvidenceRole::TransportSource, digest: Some(digest) },
        ]),
        composition: Composition::Chained { stages: 5 },
        decision: Some(DecisionProgram {
            carrier: CarrierDemand::ParameterPosterior,
            utility: DecisionProgram::utility_form(None),
            actions: decision.actions.len(),
            constraints: decision.constraints.len(),
            chance_threshold: decision
                .constraints
                .iter()
                .map(|c| c.min_probability)
                .fold(0.0_f64, f64::max),
            utility_unit: Some(Arc::from(decision.utility_units.as_str())),
            contract_identity: Some(Arc::from(contract.as_str())),
            information_value: None,
            scenario_source: Some(Arc::from("declared_mar_weighted_target_effect_v1")),
        }),
        planning: None,
        chain: None,
    };
    let mut operators = vec![
        PhysicalOperator::declare(PhysicalEffect::Read)
            .build()
            .map_err(|_| MarTransportError::Invalid("invalid read operator"))?,
    ];
    for _ in 0..=bootstrap.map_or(0, |policy| policy.replicates) {
        operators.push(
            PhysicalOperator::declare(PhysicalEffect::Fit)
                .build()
                .map_err(|_| MarTransportError::Invalid("invalid fit operator"))?,
        );
        operators.push(
            PhysicalOperator::declare(PhysicalEffect::Transport)
                .build()
                .map_err(|_| MarTransportError::Invalid("invalid transport operator"))?,
        );
    }
    let physical = PhysicalLayer { operators: Arc::from(operators), design };
    let lowered = Arc::new(
        StandaloneCompositionLoweredParts {
            cell: ProgramCell::GraphImputationMarTransportDecision,
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
    );
    let native_source = GraphImputationSource::new(parts.clone(), *lowered.identity().as_bytes())
        .map_err(MarTransportError::Invalid)?
        .encode();
    Ok(CompiledMarTransportDecision {
        producer,
        target,
        quantity,
        decision,
        native_source,
        lowered,
        bootstrap,
    })
}

impl crate::analysis::program_handle::CompiledProgramHandle for CompiledMarTransportDecision {
    fn program(&self) -> &Arc<LoweredProgram> {
        &self.lowered
    }
}

impl CompiledMarTransportDecision {
    /// Full six-layer compiler program.
    #[must_use]
    pub fn lowered(&self) -> &LoweredProgram {
        &self.lowered
    }
    /// Inspect without executing.
    #[must_use]
    pub fn plan_view(&self) -> PlanView {
        crate::analysis::program_handle::CompiledProgramHandle::plan_view(self)
    }
    /// Refit source graphs and integrate over the weighted complete target rows.
    ///
    /// # Errors
    /// Any source, support, overlap, fitting or decision refusal.
    #[allow(
        clippy::result_large_err,
        reason = "the structured refusal is returned by value on a cold path"
    )]
    pub fn execute(&self, ctx: &ExecutionContext) -> Result<MarTransportResult, MarTransportError> {
        super::graph_imputation::require_decision_support(ctx, self)?;
        let mut reading = replay_mar_transport_point(&self.native_source, &self.target, ctx)?;
        if let Some(policy) = self.bootstrap {
            self.run_cluster_bootstrap(&mut reading, policy, ctx)?;
        }
        if reading.program != *self.lowered.identity().as_bytes()
            || reading.source != *self.producer.source_identity().as_bytes()
        {
            return Err(MarTransportError::Invalid("native execution changed source or program"));
        }
        let decision = compile_chain_decision(
            effect_chain(&reading, &self.quantity),
            self.decision.clone(),
            ctx,
        )?
        .execute(ctx)?;
        Ok(MarTransportResult {
            producer: reading,
            decision,
            lowered_program_identity: self.lowered.identity(),
            calibration: "not_measured",
        })
    }
    /// Export the raw source and raw target after independent execution.
    ///
    /// # Errors
    /// Changed result, evaluation refusal or encoding failure.
    #[allow(
        clippy::result_large_err,
        reason = "the structured refusal is returned by value on a cold path"
    )]
    pub fn export(
        &self,
        result: &MarTransportResult,
        artifact_id: &str,
        ctx: &ExecutionContext,
    ) -> Result<Vec<u8>, MarTransportError> {
        let live = self.execute(ctx)?;
        if result.lowered_program_identity != self.lowered.identity()
            || result.calibration != "not_measured"
            || result.producer.receipt != live.producer.receipt
            || result.producer.draws != live.producer.draws
            || result.decision.decision != live.decision.decision
            || result.decision.chain.identity() != live.decision.chain.identity()
        {
            return Err(MarTransportError::Invalid("result differs from raw replay"));
        }
        let contract = DecisionContractArtifact::new(self.decision.clone())?;
        let chain = compile_chain_decision(
            effect_chain(&live.producer, &self.quantity),
            self.decision.clone(),
            ctx,
        )?;
        let bytes = antecedent_io::to_cbor(&Wire {
            version: 2,
            feature: FEATURE.into(),
            source: self.producer.source_bytes().to_vec(),
            target: self.target.clone(),
            quantity: ScientificQuantityWire::from(&self.quantity),
            contract_identity: contract.identity().into(),
            contract: contract.to_bytes(artifact_id)?,
            chain: chain.export(&live.decision, artifact_id)?,
            receipt: live.producer.receipt,
            program: *self.lowered.identity().as_bytes(),
            calibration: "not_measured".into(),
            bootstrap: self.bootstrap,
        })?;
        if bytes.len() > MAX_BYTES {
            return Err(MarTransportError::Invalid("artifact exceeds byte limit"));
        }
        Ok(bytes)
    }
}

/// Reconstruct the original source compiler and replay the raw MAR target.
///
/// # Errors
/// Changed source, target, program, effect, utility or unsupported feature.
#[allow(
    clippy::result_large_err,
    reason = "the structured refusal is returned by value on a cold path"
)]
pub fn consume_graph_imputation_mar_transport_decision(
    bytes: &[u8],
    expected: SemanticDigest,
    ctx: &ExecutionContext,
) -> Result<MarTransportResult, MarTransportError> {
    if bytes.len() > MAX_BYTES {
        return Err(MarTransportError::Invalid("artifact exceeds byte limit"));
    }
    let wire: Wire = antecedent_io::from_cbor(bytes)?;
    if wire.version != 2
        || wire.feature != FEATURE
        || wire.program != *expected.as_bytes()
        || wire.calibration != "not_measured"
    {
        return Err(MarTransportError::Invalid("artifact feature or program differs"));
    }
    let original =
        GraphImputationSource::decode(&wire.source).map_err(MarTransportError::Invalid)?;
    let parts = original.parts();
    let table = TabularData::from_f64_columns(
        parts
            .names
            .iter()
            .zip(&parts.columns)
            .map(|(name, column)| (name.as_str(), column.as_slice())),
    )
    .map_err(|_| MarTransportError::Invalid("stored source table is invalid"))?;
    let spec = GraphImputationSpec {
        treatment: VariableId::from_raw(parts.treatment),
        outcome: VariableId::from_raw(parts.outcome),
        support: parts.support,
        imputation: ImputationSpec::new()
            .with_imputations(parts.imputations)
            .with_iterations(parts.iterations)
            .with_seed(parts.imputation_seed),
        domains: Arc::from(parts.domains.clone()),
        posterior_draws: parts.posterior_draws,
        prior_scale: parts.prior_scale,
        execution_seed: parts.execution_seed,
    };
    let producer = CompiledGraphImputation::compile_with_candidates(
        table,
        &spec,
        parts.candidates.as_deref(),
        ctx,
    )?;
    if producer.source_bytes() != wire.source.as_slice()
        || original.program() != *producer.lowered().identity().as_bytes()
    {
        return Err(MarTransportError::Invalid("source compiler changed"));
    }
    let quantity: ScientificQuantity =
        wire.quantity.try_into().map_err(MarTransportError::Invalid)?;
    let contract = DecisionContractArtifact::from_bytes(&wire.contract, &wire.contract_identity)?;
    let plan = compile_graph_imputation_mar_transport_with_cluster(
        producer,
        wire.target,
        quantity,
        contract.contract().clone(),
        wire.bootstrap,
    )?;
    if plan.lowered.identity() != expected {
        return Err(MarTransportError::Invalid("compiler identity changed"));
    }
    let live = plan.execute(ctx)?;
    if live.producer.receipt != wire.receipt {
        return Err(MarTransportError::Invalid("raw target replay differs"));
    }
    let replay =
        consume_chain_decision(&wire.chain, Some(live.decision.lowered_program_identity), ctx)?;
    if replay.chain.identity() != live.decision.chain.identity()
        || replay.decision != live.decision.decision
    {
        return Err(MarTransportError::Invalid("utility replay differs"));
    }
    Ok(live)
}

#[cfg(test)]
mod tests {
    use super::*;
    use antecedent_io::measured_inference::graph_candidate::RawTargetFrame;

    fn target() -> MarTargetStandardization {
        MarTargetStandardization {
            observation_mechanism_id: "mar-v1".into(),
            sampling_design_id: "cluster-design-v1".into(),
            frame: RawTargetFrame {
                columns: vec![(2, vec![Some(0.0), None, Some(1.0)])],
                row_ids: vec!["a".into(), "b".into(), "c".into()],
                cluster_ids: vec!["a".into(), "b".into(), "c".into()],
                weights: vec![1.0, 2.0, 3.0],
                observation_probabilities: vec![0.5, 0.5, 0.5],
                always_observed: vec![],
            },
            selection_targets: vec![2],
            source_population: "source".into(),
            target_population: "target".into(),
            source_sample: "source-sample".into(),
            target_sample: "target-sample".into(),
            source_unit_ids: vec!["s1".into(), "s2".into()],
            bandwidths: vec![],
            min_kernel_overlap: 0.0,
            kernel: false,
            confounded: false,
        }
    }
    #[test]
    fn whole_clusters_are_resampled_together() {
        let mut target = target();
        target.frame.cluster_ids = vec!["a".into(), "b".into(), "a".into()];
        let groups = cluster_groups(&target).unwrap();
        let mut seed = 19;
        let rows = super::super::graph_imputation_transport_decision::bootstrap_cluster_rows(
            &mut seed, &groups,
        );
        assert_eq!(
            rows.iter().filter(|&&row| row == 0).count(),
            rows.iter().filter(|&&row| row == 2).count()
        );
        target.frame.cluster_ids = vec!["a".into(); 3];
        assert!(cluster_groups(&target).is_err());
    }
    #[test]
    fn raw_identity_binds_design_missingness_and_modes() {
        let original = target();
        assert_eq!(original.frame.complete_rows(), vec![(0, 2.0), (2, 6.0)]);
        let digest = original.digest().unwrap();
        let mut changed = original.clone();
        changed.frame.observation_probabilities[1] = 0.25;
        assert_ne!(digest, changed.digest().unwrap());
        changed = original.clone();
        changed.frame.weights[0] = 4.0;
        assert_ne!(digest, changed.digest().unwrap());
        changed = original.clone();
        changed.confounded = true;
        assert_ne!(digest, changed.digest().unwrap());
        changed = original.clone();
        changed.observation_mechanism_id = "mar-v2".into();
        assert_ne!(digest, changed.digest().unwrap());
        changed = original.clone();
        changed.sampling_design_id = "other-design".into();
        assert_ne!(digest, changed.digest().unwrap());
        changed = original.clone();
        changed.kernel = true;
        assert_ne!(digest, changed.digest().unwrap());
    }
}

#[allow(
    clippy::result_large_err,
    reason = "the structured refusal is returned by value on a cold path"
)]
fn cluster_groups(target: &MarTargetStandardization) -> Result<Vec<Vec<usize>>, MarTransportError> {
    let mut groups = std::collections::BTreeMap::<&str, Vec<usize>>::new();
    for (row, id) in target.frame.cluster_ids.iter().enumerate() {
        groups.entry(id).or_default().push(row);
    }
    let groups: Vec<Vec<usize>> = groups.into_values().collect();
    if groups.len() < 2
        || groups
            .len()
            .checked_mul(groups.iter().map(Vec::len).max().unwrap_or(0))
            .is_none_or(|n| n > 1_000_000)
    {
        return Err(MarTransportError::Invalid("cluster count or replicate size is unsupported"));
    }
    Ok(groups)
}

impl CompiledMarTransportDecision {
    #[allow(
        clippy::result_large_err,
        reason = "the structured refusal is returned by value on a cold path"
    )]
    #[allow(
        clippy::cast_possible_truncation,
        reason = "the replicate count is bounded by the bootstrap policy"
    )]
    #[allow(
        clippy::too_many_lines,
        reason = "one linear sequence of steps that reads top to bottom"
    )]
    fn run_cluster_bootstrap(
        &self,
        reading: &mut GraphPointReading,
        policy: MarClusterBootstrap,
        ctx: &ExecutionContext,
    ) -> Result<(), MarTransportError> {
        let groups = cluster_groups(&self.target)?;
        let max_rows = groups.len() * groups.iter().map(Vec::len).max().unwrap_or(0);
        let bytes = (max_rows as u64)
            .saturating_mul((self.target.frame.columns.len() as u64 + 5).saturating_mul(16))
            .saturating_add(u64::from(policy.replicates).saturating_mul(8));
        let _lease = ctx
            .acquire_admission(0, bytes)
            .map_err(|_| MarTransportError::Invalid("cluster bootstrap admission refused"))?;
        let mut state = policy.seed;
        let mut draw_state = antecedent_core::execution::mix64(policy.seed ^ 0x4d41_525f_4452_4157);
        let mut draws = Vec::with_capacity(policy.replicates as usize);
        let mut receipt =
            CanonicalWriter::new("antecedent.graph_imputation.mar_cluster_bootstrap.v1");
        receipt
            .bytes(self.lowered.identity().as_bytes())
            .bytes(&self.target.digest()?)
            .word(u64::from(policy.replicates))
            .word(policy.seed);
        for replicate in 0..policy.replicates {
            if ctx.cancellation.is_cancelled() {
                return Err(MarTransportError::Invalid("cluster bootstrap cancelled"));
            }
            let mut indices = Vec::new();
            let mut cluster_slots = Vec::new();
            for slot in 0..groups.len() {
                let selected = super::graph_imputation_transport_decision::bootstrap_pool_index(
                    &mut state,
                    groups.len(),
                );
                for &row in &groups[selected] {
                    indices.push(row);
                    cluster_slots.push(slot);
                }
            }
            let mut sampled = self.target.clone();
            for ((_, values), (_, original)) in
                sampled.frame.columns.iter_mut().zip(&self.target.frame.columns)
            {
                *values = indices.iter().map(|&i| original[i]).collect();
            }
            sampled.frame.weights = indices.iter().map(|&i| self.target.frame.weights[i]).collect();
            sampled.frame.observation_probabilities =
                indices.iter().map(|&i| self.target.frame.observation_probabilities[i]).collect();
            let mut nonce = 0_u64;
            let prefix = loop {
                let candidate = format!("mar-bootstrap:{nonce}:");
                if !sampled.source_unit_ids.iter().any(|id| id.starts_with(&candidate)) {
                    break candidate;
                }
                nonce = nonce
                    .checked_add(1)
                    .ok_or(MarTransportError::Invalid("bootstrap identity overflow"))?;
            };
            sampled.frame.row_ids = (0..indices.len()).map(|i| format!("{prefix}{i}")).collect();
            sampled.frame.cluster_ids =
                cluster_slots.into_iter().map(|slot| format!("{prefix}cluster:{slot}")).collect();
            // Each replicate redraws its completed tables and posterior on its
            // own streams. On the declared streams every replicate held the
            // same posterior draws, so the retained draws repeated one another
            // and understated the source and imputation spread.
            let replicate_bytes = replicate_source(&self.native_source, policy.seed, replicate)?;
            let run = replay_mar_transport_point(&replicate_bytes, &sampled, ctx)?;
            if run.draws.is_empty() {
                return Err(MarTransportError::Invalid(
                    "bootstrap replicate has no posterior draws",
                ));
            }
            let bound = run.draws.len() as u64;
            let threshold = bound.wrapping_neg() % bound;
            let index = loop {
                draw_state = antecedent_core::execution::mix64(draw_state);
                if draw_state >= threshold {
                    break (draw_state % bound) as usize;
                }
            };
            let draw = run.draws[index];
            let replicate_decision = compile_chain_decision(
                effect_chain(&run, &self.quantity),
                self.decision.clone(),
                ctx,
            )?
            .execute(ctx)?;
            receipt.word(u64::from(replicate)).word(indices.len() as u64);
            for index in indices {
                receipt.word(index as u64);
            }
            receipt
                .bytes(&run.receipt)
                .word(index as u64)
                .word(draw.to_bits())
                .bytes(replicate_decision.chain.identity().as_bytes())
                .bytes(replicate_decision.lowered_program_identity.as_bytes());
            draws.push(draw);
        }
        let mean = draws.iter().sum::<f64>() / draws.len() as f64;
        let variance = draws.iter().map(|value| (value - mean).powi(2)).sum::<f64>()
            / (draws.len() - 1) as f64;
        if !mean.is_finite() || !variance.is_finite() {
            return Err(MarTransportError::Invalid("bootstrap result is nonfinite"));
        }
        reading.mean = mean;
        reading.diagnostic_sd = variance.sqrt();
        reading.draws = draws;
        reading.receipt = receipt.finish();
        Ok(())
    }
}

/// A prospective, unmeasured interval and action-choice study coordinate.
pub struct MarTransportCalibrationCandidate {
    /// Full raw-to-decision execution.
    pub result: MarTransportResult,
    /// 95% effect interval: replicate order statistics under cluster
    /// resampling, Rubin's rules across the completed tables on a fixed frame.
    pub interval: (f64, f64),
    /// Exact unmeasured interval scope.
    pub interval_basis: CalibrationBasis,
    /// Exact unmeasured action-choice study scope.
    pub action_choice_basis: SemanticDigest,
}
impl CompiledMarTransportDecision {
    /// Construct prospective interval and action-choice scope without claiming coverage.
    /// The scope excludes realized target values and execution seeds.
    ///
    /// # Errors
    /// Execution refusal or fewer than forty finite draws.
    #[allow(
        clippy::result_large_err,
        reason = "the structured refusal is returned by value on a cold path"
    )]
    pub fn calibration_candidate(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<MarTransportCalibrationCandidate, MarTransportError> {
        let result = self.execute(ctx)?;
        let mut draws = result.producer.draws.clone();
        if draws.len() < 40 || draws.iter().any(|value| !value.is_finite()) {
            return Err(MarTransportError::Invalid("interval needs at least forty finite draws"));
        }
        draws.sort_by(f64::total_cmp);
        let source = GraphImputationSource::decode(&self.native_source)
            .map_err(MarTransportError::Invalid)?;
        let parts = source.parts();
        // Under cluster resampling each draw is one independent replicate, each
        // on its own completed tables. On a fixed frame the draws pool the
        // declared tables, which Rubin's rules combine.
        let interval = if self.bootstrap.is_some() {
            order_statistic_interval_95(&draws)
        } else {
            imputation_pooled_interval_95(
                &result.producer.draws,
                parts.imputations as usize,
                transported_child_df(parts, &self.target.selection_targets),
            )
            .ok_or(MarTransportError::Invalid("draws do not pool across completed tables"))?
        };
        let bandwidths = self
            .target
            .bandwidths
            .iter()
            .map(|(id, bandwidth)| format!("{id}:{:016x}", bandwidth.to_bits()))
            .collect::<Vec<_>>()
            .join(",");
        // Source and target row counts reach the record through the scope's
        // row count; the target cluster count is not part of the construction.
        let mut functional = format!(
            "target_average_effect:graph_support={}:graphs={}:domains={}:selected={:?}:imputations={}:iterations={}:posterior_draws={}:prior_bits={:016x}:mode={}:confounded={}:mar_conditioning={:?}:probability_policy=declared_complete_vector:observation_mechanism={:?}:sampling_design={:?}:weights=sampling_over_observation:bandwidths={}:overlap_bits={:016x}:target_sampling={}:replicates={}",
            parts.support.as_str(),
            parts.atoms.len(),
            parts.domains.iter().map(u8::to_string).collect::<String>(),
            self.target.selection_targets,
            parts.imputations,
            parts.iterations,
            parts.posterior_draws,
            parts.prior_scale.to_bits(),
            if self.target.kernel { "kernel" } else { "binary" },
            self.target.confounded,
            self.target.frame.always_observed,
            self.target.observation_mechanism_id,
            self.target.sampling_design_id,
            bandwidths,
            self.target.min_kernel_overlap.to_bits(),
            if self.bootstrap.is_some() { "independent_cluster_bootstrap" } else { "fixed_frame" },
            self.bootstrap.map_or(0, |policy| policy.replicates)
        );
        if let Some(candidates) = parts.declared_candidate_identity() {
            use std::fmt::Write as _;
            let _ = write!(functional, ":candidates={}", candidates.to_hex());
        }
        let n_draws = self
            .bootstrap
            .map_or(parts.posterior_draws.saturating_mul(parts.imputations), |p| p.replicates);
        let interval_basis = CalibrationBasis::new(
            [
                "AverageEffect",
                "Dag",
                "graph_posterior",
                "multi_env",
                "Bayesian",
                FEATURE,
                if self.bootstrap.is_some() { REPLICATE_INTERVAL } else { POOLED_INTERVAL },
                "",
                "independent_source_target_units",
                "declared_mar_weighted_target",
                &functional,
            ]
            .map(Arc::from),
            0.95,
            Arc::from("model_conditional"),
            (parts.rows() + self.target.frame.row_ids.len()) as u64,
            None,
            Some(n_draws),
            0.0,
        )
        .with_validation("none");
        let mut action =
            CanonicalWriter::new("antecedent.graph_imputation.mar_transport.action_choice.v1");
        action
            .text(FEATURE)
            .text(&functional)
            .text(&self.decision.identity()?)
            .bytes(&antecedent_io::to_cbor(&ScientificQuantityWire::from(&self.quantity))?);
        Ok(MarTransportCalibrationCandidate {
            result,
            interval,
            interval_basis,
            action_choice_basis: SemanticDigest::from_bytes(action.finish()),
        })
    }
}

#[cfg(test)]
mod mar_schedule_tests {
    use super::*;
    use antecedent_core::graph_imputation_contract::GraphSupport;
    use antecedent_design::decision_contract::{
        ActionKind, DecisionAction, DecisionCriterion, StructuralPolicy, UtilityExpr,
    };
    use antecedent_io::measured_inference::graph_candidate::RawTargetFrame;

    #[test]
    #[allow(
        clippy::too_many_lines,
        reason = "one linear sequence of steps that reads top to bottom"
    )]
    fn binary_kernel_and_confounding_replay_weighted_raw_frame() {
        for kernel in [false, true] {
            for confounded in [false, true] {
                let x: Vec<f64> = (0..80)
                    .map(|i| if kernel { f64::from(i % 8) / 8.0 } else { f64::from(i % 2) })
                    .collect();
                let t: Vec<f64> = (0..80).map(|i| f64::from((i / 2) % 2)).collect();
                let y: Vec<f64> = (0..80)
                    .map(|i| {
                        if i % 13 == 0 {
                            f64::NAN
                        } else {
                            t[i] * (0.5 + x[i]) + x[i] * 0.2 + (i % 5) as f64 * 0.1
                        }
                    })
                    .collect();
                let table = TabularData::from_f64_columns([
                    ("t", t.as_slice()),
                    ("y", y.as_slice()),
                    ("x", x.as_slice()),
                ])
                .unwrap();
                let ctx = ExecutionContext::for_tests(19);
                let producer = CompiledGraphImputation::compile(
                    table,
                    &GraphImputationSpec {
                        treatment: VariableId::from_raw(0),
                        outcome: VariableId::from_raw(1),
                        support: if confounded {
                            GraphSupport::PreTreatmentCovariates
                        } else {
                            GraphSupport::TreatmentRootPreTreatmentCovariates
                        },
                        imputation: ImputationSpec::new()
                            .with_imputations(2)
                            .with_iterations(2)
                            .with_seed(17),
                        domains: Arc::from(vec![1, 0, u8::from(!kernel)]),
                        posterior_draws: 100,
                        prior_scale: 1.0,
                        execution_seed: 19,
                    },
                    &ctx,
                )
                .unwrap();
                let target = MarTargetStandardization {
                    observation_mechanism_id: "mar-v1".into(),
                    sampling_design_id: "cluster-design-v1".into(),
                    frame: RawTargetFrame {
                        columns: vec![(
                            2,
                            (0..80).map(|i| if i % 7 == 0 { None } else { Some(x[i]) }).collect(),
                        )],
                        row_ids: (0..80).map(|i| format!("target-{i}")).collect(),
                        cluster_ids: (0..80).map(|i| format!("cluster-{}", i / 4)).collect(),
                        weights: (0..80).map(|i| if i % 2 == 0 { 2.0 } else { 1.0 }).collect(),
                        observation_probabilities: vec![0.8; 80],
                        always_observed: vec![],
                    },
                    selection_targets: vec![2],
                    source_population: "source".into(),
                    target_population: "target".into(),
                    source_sample: producer.source_identity().to_hex(),
                    target_sample: "target-sample".into(),
                    source_unit_ids: (0..80).map(|i| format!("source-{i}")).collect(),
                    bandwidths: if kernel { vec![(2, 0.5)] } else { vec![] },
                    min_kernel_overlap: if kernel { 0.001 } else { 0.0 },
                    kernel,
                    confounded,
                };
                let quantity = ScientificQuantity {
                    variable_id: "y".into(),
                    variable_name: "target effect".into(),
                    role: QuantityRole::Outcome,
                    units: "score".into(),
                    population_id: "target".into(),
                    regime_id: "do(t=1)-do(t=0)".into(),
                    horizon: 0,
                    functional_id: "average_effect".into(),
                    conditioning: vec![],
                    transform_id: "identity".into(),
                };
                let decision = DecisionContract {
                    actions: vec![
                        DecisionAction {
                            id: "treat".into(),
                            kind: ActionKind::External,
                            inputs: vec![quantity.clone()],
                            utility: UtilityExpr::Input(0),
                        },
                        DecisionAction {
                            id: "hold".into(),
                            kind: ActionKind::External,
                            inputs: vec![quantity.clone()],
                            utility: UtilityExpr::product(
                                UtilityExpr::Input(0),
                                UtilityExpr::Const(0.0),
                            ),
                        },
                    ],
                    utility_units: "score".into(),
                    criterion: DecisionCriterion::PosteriorExpectedUtility,
                    constraints: vec![],
                    target_population: "target".into(),
                    horizon: 0,
                    structural_policy: StructuralPolicy::RequireInvariantBestAction,
                };
                let plan = compile_graph_imputation_mar_transport_decision(
                    producer, target, quantity, decision,
                )
                .unwrap();
                let result = plan.execute(&ctx).unwrap();
                let artifact = plan.export(&result, "mar-test", &ctx).unwrap();
                let replay = consume_graph_imputation_mar_transport_decision(
                    &artifact,
                    plan.lowered().identity(),
                    &ctx,
                )
                .unwrap();
                assert_eq!(result.producer.receipt, replay.producer.receipt);
                assert_eq!(result.decision.decision, replay.decision.decision);
                assert_eq!(result.calibration, "not_measured");
                if !kernel && !confounded {
                    let original =
                        GraphImputationSource::decode(plan.producer.source_bytes()).unwrap();
                    let parts = original.parts();
                    let table = TabularData::from_f64_columns(
                        parts
                            .names
                            .iter()
                            .zip(&parts.columns)
                            .map(|(name, column)| (name.as_str(), column.as_slice())),
                    )
                    .unwrap();
                    let spec = GraphImputationSpec {
                        treatment: VariableId::from_raw(parts.treatment),
                        outcome: VariableId::from_raw(parts.outcome),
                        support: parts.support,
                        imputation: ImputationSpec::new()
                            .with_imputations(parts.imputations)
                            .with_iterations(parts.iterations)
                            .with_seed(parts.imputation_seed),
                        domains: Arc::from(parts.domains.clone()),
                        posterior_draws: parts.posterior_draws,
                        prior_scale: parts.prior_scale,
                        execution_seed: parts.execution_seed,
                    };
                    let producer = CompiledGraphImputation::compile(table, &spec, &ctx).unwrap();
                    let clustered = compile_graph_imputation_mar_transport_with_cluster(
                        producer,
                        plan.target.clone(),
                        plan.quantity.clone(),
                        plan.decision.clone(),
                        Some(MarClusterBootstrap { replicates: 40, seed: 7 }),
                    )
                    .unwrap();
                    assert_ne!(clustered.lowered().identity(), plan.lowered().identity());
                    let result = clustered.execute(&ctx).unwrap();
                    let bytes = clustered.export(&result, "mar-cluster", &ctx).unwrap();
                    let replay = consume_graph_imputation_mar_transport_decision(
                        &bytes,
                        clustered.lowered().identity(),
                        &ctx,
                    )
                    .unwrap();
                    assert_eq!(result.producer.receipt, replay.producer.receipt);
                    assert_eq!(result.decision.decision, replay.decision.decision);
                }
            }
        }
    }

    /// The binary, unconfounded declaration above; `edit` changes its target frame and
    /// contract before compiling.
    #[allow(clippy::too_many_lines, reason = "one declaration read top to bottom")]
    fn mar_question(
        edit: impl Fn(&mut MarTargetStandardization, &mut DecisionContract),
        bootstrap: Option<MarClusterBootstrap>,
    ) -> Option<[u8; 32]> {
        let x: Vec<f64> = (0..80).map(|i| f64::from(i % 2)).collect();
        let t: Vec<f64> = (0..80).map(|i| f64::from((i / 2) % 2)).collect();
        let y: Vec<f64> = (0..80)
            .map(|i| {
                if i % 13 == 0 {
                    f64::NAN
                } else {
                    t[i] * (0.5 + x[i]) + x[i] * 0.2 + (i % 5) as f64 * 0.1
                }
            })
            .collect();
        let table = TabularData::from_f64_columns([
            ("t", t.as_slice()),
            ("y", y.as_slice()),
            ("x", x.as_slice()),
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
        let mut target = MarTargetStandardization {
            observation_mechanism_id: "mar-v1".into(),
            sampling_design_id: "cluster-design-v1".into(),
            frame: RawTargetFrame {
                columns: vec![(
                    2,
                    (0..80).map(|i| if i % 7 == 0 { None } else { Some(x[i]) }).collect(),
                )],
                row_ids: (0..80).map(|i| format!("target-{i}")).collect(),
                cluster_ids: (0..80).map(|i| format!("cluster-{}", i / 4)).collect(),
                weights: (0..80).map(|i| if i % 2 == 0 { 2.0 } else { 1.0 }).collect(),
                observation_probabilities: vec![0.8; 80],
                always_observed: vec![],
            },
            selection_targets: vec![2],
            source_population: "source".into(),
            target_population: "target".into(),
            source_sample: producer.source_identity().to_hex(),
            target_sample: "target-sample".into(),
            source_unit_ids: (0..80).map(|i| format!("source-{i}")).collect(),
            bandwidths: vec![],
            min_kernel_overlap: 0.0,
            kernel: false,
            confounded: false,
        };
        let quantity = ScientificQuantity {
            variable_id: "y".into(),
            variable_name: "target effect".into(),
            role: QuantityRole::Outcome,
            units: "score".into(),
            population_id: "target".into(),
            regime_id: "do(t=1)-do(t=0)".into(),
            horizon: 0,
            functional_id: "average_effect".into(),
            conditioning: vec![],
            transform_id: "identity".into(),
        };
        let mut decision = DecisionContract {
            actions: vec![
                DecisionAction {
                    id: "treat".into(),
                    kind: ActionKind::External,
                    inputs: vec![quantity.clone()],
                    utility: UtilityExpr::Input(0),
                },
                DecisionAction {
                    id: "hold".into(),
                    kind: ActionKind::External,
                    inputs: vec![quantity.clone()],
                    utility: UtilityExpr::product(UtilityExpr::Input(0), UtilityExpr::Const(0.0)),
                },
            ],
            utility_units: "score".into(),
            criterion: DecisionCriterion::PosteriorExpectedUtility,
            constraints: vec![],
            target_population: "target".into(),
            horizon: 0,
            structural_policy: StructuralPolicy::RequireInvariantBestAction,
        };
        edit(&mut target, &mut decision);
        let plan = compile_graph_imputation_mar_transport_with_cluster(
            producer, target, quantity, decision, bootstrap,
        )
        .ok()?;
        match plan.lowered().scientific().target() {
            ProblemTarget::QueryFunctional { query, .. } => Some(*query),
            _ => None,
        }
    }

    #[test]
    fn site_graph_imputation_mar_transport_question_binds_target_contract_and_schedule() {
        let keep = |_: &mut MarTargetStandardization, _: &mut DecisionContract| {};
        crate::identity_laws::all_distinct(&[
            mar_question(keep, None),
            mar_question(
                |target: &mut MarTargetStandardization, _: &mut DecisionContract| {
                    target.frame.weights[0] = 3.0;
                },
                None,
            ),
            mar_question(
                |target: &mut MarTargetStandardization, _: &mut DecisionContract| {
                    target.observation_mechanism_id = "mar-v2".into();
                },
                None,
            ),
            mar_question(
                |_: &mut MarTargetStandardization, decision: &mut DecisionContract| {
                    decision.utility_units = "points".into();
                },
                None,
            ),
            // The fixed frame and the cluster bootstrap carry different tags.
            mar_question(keep, Some(MarClusterBootstrap { replicates: 40, seed: 7 })),
            mar_question(keep, Some(MarClusterBootstrap { replicates: 40, seed: 8 })),
            mar_question(keep, Some(MarClusterBootstrap { replicates: 41, seed: 7 })),
        ]);
    }
}
