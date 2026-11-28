//! Graph selection, missing-row imputation, mixed-covariate kernel transport,
//! fitted target effect and nonlinear utility under one compiler program.
//!
//! The source graph support still requires root treatment and pretreatment
//! covariates. A declared Gaussian kernel density ratio, tilted so that the
//! weighted source reproduces the frame's covariate means, transports each
//! completed source table to an independent empirical target frame that may
//! include continuous and binary covariates. Each graph gets its own safe
//! adjustment set plus selected effect modifiers. Bandwidths and the overlap
//! floor are premises; target-frame uncertainty and final decision reliability
//! are not measured.

use std::sync::Arc;

use antecedent_contract::{ClaimKind, DataClassification, QuantityRole, ScientificQuantity};
use antecedent_core::graph_imputation_contract::GraphImputationSource;
use antecedent_core::{
    ByteEstimates, CalibrationBasis, CanonicalWriter, ExecutionContext, PhysicalEffect,
    PhysicalOperator, SemanticDigest, VariableId,
};
use antecedent_data::{ImputationSpec, TabularData};
use antecedent_design::decision_artifact::DecisionContractArtifact;
use antecedent_design::decision_contract::DecisionContract;
use antecedent_io::measured_inference::graph_candidate::{
    KernelTransportStandardization, replay_confounded_kernel_transport_point,
    replay_kernel_transport_point, replicate_source,
};
use antecedent_io::quantity_wire::ScientificQuantityWire;
use serde::{Deserialize, Serialize};

use super::graph_imputation::{CompiledGraphImputation, GraphImputationSpec};
use super::graph_imputation_transport_decision::{
    GraphImputationTransportDecisionError as Error, GraphImputationTransportDecisionResult,
    POOLED_INTERVAL, REPLICATE_INTERVAL, TargetClusterBootstrap, bootstrap_cluster_rows,
    effect_chain, imputation_pooled_interval_95, order_statistic_interval_95, transported_child_df,
};
use super::propagation_chain_decision::{compile_chain_decision, consume_chain_decision};
use crate::analysis::program_handle::CompiledProgramHandle;
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
use crate::support::capability_calibration;
use crate::support::units_generated::GRAPH_IMPUTATION_COMPILED_DECISION;

const FEATURE: &str = "graph_imputation_kernel_transport_decision_v2";
const BOOTSTRAP_FEATURE: &str = "graph_imputation_kernel_transport_target_bootstrap_decision_v2";
const CONFOUNDED_FEATURE: &str = "graph_imputation_confounded_kernel_transport_decision_v1";
const CONFOUNDED_BOOTSTRAP_FEATURE: &str =
    "graph_imputation_confounded_kernel_transport_target_bootstrap_decision_v1";
const CLUSTER_FEATURE: &str =
    "graph_imputation_kernel_transport_target_cluster_bootstrap_decision_v1";
const CONFOUNDED_CLUSTER_FEATURE: &str =
    "graph_imputation_confounded_kernel_transport_target_cluster_bootstrap_decision_v1";
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
    cluster_bootstrap: Option<TargetClusterBootstrap>,
    confounded: bool,
}

/// Prespecified iid empirical bootstrap of independent target units.
/// Each replicate refits the full graph-imputation transport schedule on its
/// own imputation and posterior streams and contributes one posterior draw;
/// the interval is the replicate order-statistic interval of those draws.
/// This is a source-only uncertainty model;
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
    /// 95% interval from the target effect draws: replicate order statistics
    /// under target resampling, Rubin's rules across the completed tables on a
    /// fixed frame.
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
    #[serde(default, skip_serializing_if = "is_false")]
    confounded: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    cluster_bootstrap: Option<TargetClusterBootstrap>,
}

#[allow(
    clippy::trivially_copy_pass_by_ref,
    reason = "serde's `skip_serializing_if` calls the predicate with a reference"
)]
fn is_false(value: &bool) -> bool {
    !*value
}

#[allow(
    clippy::result_large_err,
    reason = "the structured refusal is returned by value on a cold path"
)]
fn check_quantity(
    source: &GraphImputationSource,
    target: &KernelTransportStandardization,
    quantity: &ScientificQuantity,
    decision: &DecisionContract,
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
        || quantity.horizon != 0
        || quantity.horizon != decision.horizon
        || !quantity.conditioning.is_empty()
        || quantity.transform_id != "identity"
        || decision.actions.iter().any(|action| {
            action.inputs.len() != 1 || action.inputs[0].require_same_coordinate(quantity).is_err()
        })
    {
        return Err(Error::Invalid(
            "utility must read the exact mixed-target average-effect coordinate",
        ));
    }
    Ok(())
}

#[allow(
    clippy::result_large_err,
    reason = "the structured refusal is returned by value on a cold path"
)]
#[allow(clippy::too_many_lines, reason = "one linear sequence of steps that reads top to bottom")]
fn lower_outer(
    source: &GraphImputationSource,
    target: &KernelTransportStandardization,
    quantity: &ScientificQuantity,
    decision: &DecisionContract,
    target_bootstrap: Option<TargetFrameBootstrap>,
    confounded: bool,
    cluster_bootstrap: Option<&TargetClusterBootstrap>,
) -> Result<Arc<LoweredProgram>, Error> {
    if confounded {
        target.validate_confounded(source.parts())?;
    } else {
        target.validate(source.parts())?;
    }
    if target_bootstrap.is_some_and(|policy| !(40..=256).contains(&policy.replicates)) {
        return Err(Error::Invalid("target bootstrap needs 40 to 256 replicates"));
    }
    if let Some(policy) = cluster_bootstrap {
        if target_bootstrap.is_some() {
            return Err(Error::Invalid("row and cluster bootstrap are exclusive"));
        }
        policy.groups(target.rows())?;
    }
    check_quantity(source, target, quantity, decision)?;
    let target_digest = target.digest()?;
    let source_digest = SemanticDigest::from_bytes(source.parts().data_digest);
    let coordinate = antecedent_io::to_cbor(&ScientificQuantityWire::from(quantity))?;
    let contract = decision.identity()?;
    let mut writer =
        CanonicalWriter::new("antecedent.graph_imputation_kernel_transport_decision.question.v1");
    writer
        .bytes(&source.program())
        .bytes(&source.parts().contract_digest())
        .bytes(source_digest.as_bytes())
        .bytes(&target_digest)
        .bytes(&coordinate)
        .text(&contract)
        .text(FEATURE);
    if confounded {
        writer.text(CONFOUNDED_FEATURE);
    }
    if let Some(policy) = target_bootstrap {
        writer
            .text(if confounded { CONFOUNDED_BOOTSTRAP_FEATURE } else { BOOTSTRAP_FEATURE })
            .word(u64::from(policy.replicates))
            .word(policy.seed);
    }
    if let Some(policy) = cluster_bootstrap {
        writer
            .text(if confounded { CONFOUNDED_CLUSTER_FEATURE } else { CLUSTER_FEATURE })
            .bytes(&antecedent_io::to_cbor(policy)?);
    }
    let question = writer.finish();
    let rows = source.parts().rows().saturating_add(target.rows()) as u64;
    let design = DesignShape::quoted(rows, source.parts().names.len() as u64, true);
    let scientific = ScientificProblem {
        runtime_query: None,
        identity_scope: ProgramIdentityScope::Semantic,
        target: ProblemTarget::QueryFunctional {
            kind: FunctionalKind::PolicyValue,
            outcome: Some(VariableId::from_raw(source.parts().outcome)),
            query: question,
        },
        plan_id: Arc::from(if cluster_bootstrap.is_some() {
            if confounded { CONFOUNDED_CLUSTER_FEATURE } else { CLUSTER_FEATURE }
        } else if confounded && target_bootstrap.is_some() {
            CONFOUNDED_BOOTSTRAP_FEATURE
        } else if confounded {
            CONFOUNDED_FEATURE
        } else if target_bootstrap.is_some() {
            BOOTSTRAP_FEATURE
        } else {
            FEATURE
        }),
        data: DataClassification::Tabular,
        graph_class: if confounded {
            "bounded_graph_imputation_confounded_mixed_kernel_target"
        } else {
            "bounded_graph_imputation_mixed_kernel_target"
        },
        query_variables: Arc::from([]),
        source_digest: Some(*source_digest.as_bytes()),
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
            target_bootstrap.map(|policy| ResamplingPlan {
                method: ResamplingMethod::Bootstrap,
                replicates: policy.replicates,
                unit: ResampleUnit::Row,
            })
        },
        posterior: None,
        design,
        operators: Arc::from([]),
    };
    let evidence = EvidenceDecisionProgram {
        sources: Arc::from([
            EvidenceSourceRef {
                role: EvidenceRole::Observations,
                digest: Some(*source_digest.as_bytes()),
            },
            EvidenceSourceRef { role: EvidenceRole::TransportSource, digest: Some(target_digest) },
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
                .map(|constraint| constraint.min_probability)
                .fold(0.0_f64, f64::max),
            utility_unit: Some(Arc::from(decision.utility_units.as_str())),
            contract_identity: Some(Arc::from(contract.as_str())),
            information_value: None,
            scenario_source: Some(Arc::from("graph_imputation_kernel_target_effect_draws_v1")),
        }),
        planning: None,
        chain: None,
    };
    let mut operators = vec![
        PhysicalOperator::declare(PhysicalEffect::Read)
            .estimates(ByteEstimates::absent().with_output(rows.saturating_mul(8)))
            .build()
            .map_err(|_| Error::Invalid("invalid mixed source read operator"))?,
    ];
    // The empirical-frame fit runs once, followed by one complete fit and
    // transport per bootstrap replicate. Keep the repeated work visible in
    // the physical schedule; these operations execute sequentially.
    let executions = 1 + cluster_bootstrap.map_or_else(
        || target_bootstrap.map_or(0, |policy| policy.replicates),
        |policy| policy.replicates,
    );
    for _ in 0..executions {
        operators.push(
            PhysicalOperator::declare(PhysicalEffect::Fit)
                .build()
                .map_err(|_| Error::Invalid("invalid mixed graph fit operator"))?,
        );
        operators.push(
            PhysicalOperator::declare(PhysicalEffect::Transport)
                .build()
                .map_err(|_| Error::Invalid("invalid kernel standardization operator"))?,
        );
    }
    let physical = PhysicalLayer { operators: Arc::from(operators), design };
    Ok(Arc::new(
        StandaloneCompositionLoweredParts {
            cell: if cluster_bootstrap.is_some() {
                if confounded {
                    ProgramCell::GraphImputationConfoundedKernelTransportClusterBootstrapDecision
                } else {
                    ProgramCell::GraphImputationKernelTransportClusterBootstrapDecision
                }
            } else {
                ProgramCell::GraphImputationKernelTransportDecision
            },
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

/// Bind source graph operations, raw and missing rows, mixed target frame,
/// kernel bandwidths, overlap policy, exact effect coordinate and utility
/// before fitting or target standardization.
///
/// # Errors
/// Invalid source binding, target support, contract or six-layer lowering.
#[allow(
    clippy::result_large_err,
    reason = "the structured refusal is returned by value on a cold path"
)]
pub fn compile_graph_imputation_kernel_transport_decision(
    producer: CompiledGraphImputation,
    target: KernelTransportStandardization,
    quantity: ScientificQuantity,
    decision: DecisionContract,
) -> Result<CompiledGraphImputationKernelTransportDecision, Error> {
    let source = GraphImputationSource::decode(producer.source_bytes()).map_err(Error::Invalid)?;
    if source.program() != *producer.lowered().identity().as_bytes()
        || target.source_sample != producer.source_identity().to_hex()
    {
        return Err(Error::Invalid(
            "mixed target source-sample identity differs from retained producer",
        ));
    }
    let lowered = lower_outer(&source, &target, &quantity, &decision, None, false, None)?;
    let native_source =
        GraphImputationSource::new(source.parts().clone(), *lowered.identity().as_bytes())
            .map_err(Error::Invalid)?
            .encode();
    Ok(CompiledGraphImputationKernelTransportDecision {
        producer,
        target,
        quantity,
        decision,
        native_source,
        lowered,
        target_bootstrap: None,
        cluster_bootstrap: None,
        confounded: false,
    })
}

/// Compile an independent iid target-frame bootstrap into the complete
/// graph-imputation, kernel transport and utility program. The empirical
/// target rows are resampled together, preserving their joint distribution.
///
/// # Errors
/// Invalid source, target, bootstrap policy or compiler binding.
#[allow(
    clippy::result_large_err,
    reason = "the structured refusal is returned by value on a cold path"
)]
pub fn compile_graph_imputation_kernel_transport_target_bootstrap_decision(
    producer: CompiledGraphImputation,
    target: KernelTransportStandardization,
    quantity: ScientificQuantity,
    decision: DecisionContract,
    policy: TargetFrameBootstrap,
) -> Result<CompiledGraphImputationKernelTransportDecision, Error> {
    let source = GraphImputationSource::decode(producer.source_bytes()).map_err(Error::Invalid)?;
    if source.program() != *producer.lowered().identity().as_bytes()
        || target.source_sample != producer.source_identity().to_hex()
    {
        return Err(Error::Invalid(
            "mixed target source-sample identity differs from retained producer",
        ));
    }
    let lowered = lower_outer(&source, &target, &quantity, &decision, Some(policy), false, None)?;
    let native_source =
        GraphImputationSource::new(source.parts().clone(), *lowered.identity().as_bytes())
            .map_err(Error::Invalid)?
            .encode();
    Ok(CompiledGraphImputationKernelTransportDecision {
        producer,
        target,
        quantity,
        decision,
        native_source,
        lowered,
        target_bootstrap: Some(policy),
        cluster_bootstrap: None,
        confounded: false,
    })
}

/// Compile confounded-pretreatment mixed kernel transport. When `bootstrap`
/// is supplied, independent IID target rows are resampled and the complete
/// graph and child fit is replayed for every replicate.
///
/// # Errors
/// Invalid source binding, augmented backdoor or selection proof, target,
/// overlap policy, bootstrap policy or decision.
#[allow(
    clippy::result_large_err,
    reason = "the structured refusal is returned by value on a cold path"
)]
pub fn compile_graph_imputation_confounded_kernel_transport_decision(
    producer: CompiledGraphImputation,
    target: KernelTransportStandardization,
    quantity: ScientificQuantity,
    decision: DecisionContract,
    bootstrap: Option<TargetFrameBootstrap>,
) -> Result<CompiledGraphImputationKernelTransportDecision, Error> {
    let source = GraphImputationSource::decode(producer.source_bytes()).map_err(Error::Invalid)?;
    if source.program() != *producer.lowered().identity().as_bytes()
        || target.source_sample != producer.source_identity().to_hex()
    {
        return Err(Error::Invalid("confounded kernel source identity differs"));
    }
    let lowered = lower_outer(&source, &target, &quantity, &decision, bootstrap, true, None)?;
    let native_source =
        GraphImputationSource::new(source.parts().clone(), *lowered.identity().as_bytes())
            .map_err(Error::Invalid)?
            .encode();
    Ok(CompiledGraphImputationKernelTransportDecision {
        producer,
        target,
        quantity,
        decision,
        native_source,
        lowered,
        target_bootstrap: bootstrap,
        cluster_bootstrap: None,
        confounded: true,
    })
}

/// Compile independent-cluster mixed-target resampling with either supported
/// pretreatment graph family. Each replicate reruns the complete schedule on
/// its own imputation and posterior streams, and the interval is the replicate
/// order-statistic interval of the retained draws.
///
/// # Errors
/// Invalid source, target, cluster policy, graph premise or decision.
#[allow(
    clippy::result_large_err,
    reason = "the structured refusal is returned by value on a cold path"
)]
pub fn compile_graph_imputation_kernel_transport_cluster_bootstrap_decision(
    producer: CompiledGraphImputation,
    target: KernelTransportStandardization,
    quantity: ScientificQuantity,
    decision: DecisionContract,
    policy: TargetClusterBootstrap,
    confounded: bool,
) -> Result<CompiledGraphImputationKernelTransportDecision, Error> {
    let source = GraphImputationSource::decode(producer.source_bytes()).map_err(Error::Invalid)?;
    if source.program() != *producer.lowered().identity().as_bytes()
        || target.source_sample != producer.source_identity().to_hex()
    {
        return Err(Error::Invalid("cluster mixed-target source identity differs"));
    }
    let lowered =
        lower_outer(&source, &target, &quantity, &decision, None, confounded, Some(&policy))?;
    let native_source =
        GraphImputationSource::new(source.parts().clone(), *lowered.identity().as_bytes())
            .map_err(Error::Invalid)?
            .encode();
    Ok(CompiledGraphImputationKernelTransportDecision {
        producer,
        target,
        quantity,
        decision,
        native_source,
        lowered,
        target_bootstrap: None,
        cluster_bootstrap: Some(policy),
        confounded,
    })
}

// SplitMix64 provides a stable, locally specified stream for portable replay.
fn bootstrap_word(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

#[allow(
    clippy::cast_possible_truncation,
    reason = "the remainder is below the usize bound it was reduced by"
)]
fn bootstrap_pool_index(state: &mut u64, len: usize) -> usize {
    let bound = len as u64;
    let threshold = bound.wrapping_neg() % bound;
    loop {
        let word = bootstrap_word(state);
        if word >= threshold {
            return (word % bound) as usize;
        }
    }
}

impl CompiledProgramHandle for CompiledGraphImputationKernelTransportDecision {
    fn program(&self) -> &Arc<LoweredProgram> {
        &self.lowered
    }
}

impl CompiledGraphImputationKernelTransportDecision {
    /// All six compiler layers of the mixed target effect decision.
    #[must_use]
    pub fn lowered(&self) -> &LoweredProgram {
        &self.lowered
    }

    /// Inspect the schedule before fitting or kernel standardization.
    #[must_use]
    pub fn plan_view(&self) -> PlanView {
        CompiledProgramHandle::plan_view(self)
    }

    /// Reimpute source rows, reselect graphs, fit kernel-standardized children
    /// and score nonlinear effect utility under this retained program.
    ///
    /// # Errors
    /// Source, overlap, fit or decision refusal.
    #[allow(
        clippy::result_large_err,
        reason = "the structured refusal is returned by value on a cold path"
    )]
    #[allow(
        clippy::too_many_lines,
        reason = "one linear sequence of steps that reads top to bottom"
    )]
    #[allow(
        clippy::cast_possible_truncation,
        reason = "the replicate count is bounded by the bootstrap policy and the drawn index is below the row count"
    )]
    pub fn execute(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<GraphImputationTransportDecisionResult, Error> {
        super::graph_imputation::require_decision_support(ctx, self)?;
        let source = GraphImputationSource::decode(&self.native_source).map_err(Error::Invalid)?;
        if source.program() != *self.lowered.identity().as_bytes()
            || self.target.source_sample != self.producer.source_identity().to_hex()
        {
            return Err(Error::Invalid("retained mixed source changed"));
        }
        let mut reading = if self.confounded {
            replay_confounded_kernel_transport_point(&self.native_source, &self.target, ctx)?
        } else {
            replay_kernel_transport_point(&self.native_source, &self.target, ctx)?
        };
        let cluster_groups = self
            .cluster_bootstrap
            .as_ref()
            .map(|policy| policy.groups(self.target.rows()))
            .transpose()?;
        let effective_bootstrap =
            self.target_bootstrap.or(self.cluster_bootstrap.as_ref().map(|policy| {
                TargetFrameBootstrap { replicates: policy.replicates, seed: policy.seed }
            }));
        if let Some(policy) = effective_bootstrap {
            // Account for the sampled column copy, row indices and pooled draws;
            // each nested replay separately admits its own fitting workspace.
            let max_rows = cluster_groups.as_ref().map_or(self.target.rows(), |groups| {
                groups.len() * groups.iter().map(Vec::len).max().unwrap_or(0)
            });
            let bootstrap_bytes = (max_rows as u64)
                .saturating_mul((self.target.columns.len() as u64).saturating_mul(2) + 1)
                .saturating_mul(8)
                .saturating_add(
                    self.target
                        .source_unit_ids
                        .iter()
                        .chain(&self.target.target_unit_ids)
                        .map(|id| (id.len() as u64).saturating_add(24))
                        .fold(0_u64, u64::saturating_add),
                )
                .saturating_add(u64::from(policy.replicates).saturating_mul(8))
                .saturating_add(if cluster_groups.is_some() {
                    (max_rows as u64).saturating_mul(128)
                } else {
                    0
                });
            let _bootstrap_lease = ctx
                .acquire_admission(0, bootstrap_bytes)
                .map_err(|_| Error::Invalid("target bootstrap admission refused"))?;
            let mut state = policy.seed;
            // Domain-separated stream selects uniformly across all imputation/graph draws.
            let mut draw_state =
                antecedent_core::execution::mix64(policy.seed ^ 0x4452_4157_5f56_3201);
            let rows = self.target.rows();
            let mut draws = Vec::with_capacity(policy.replicates as usize);
            let mut receipt = CanonicalWriter::new(if cluster_groups.is_some() {
                "antecedent.graph_imputation_kernel_transport.target_cluster_bootstrap.execution.v1"
            } else if self.confounded {
                "antecedent.graph_imputation_confounded_kernel_transport.target_bootstrap.execution.v1"
            } else {
                "antecedent.graph_imputation_kernel_transport.target_bootstrap.execution.v2"
            });
            receipt
                .bytes(self.lowered.identity().as_bytes())
                .bytes(&self.target.digest()?)
                .word(u64::from(policy.replicates))
                .word(policy.seed);
            for replicate in 0..policy.replicates {
                if ctx.cancellation.is_cancelled() {
                    return Err(Error::Invalid("target bootstrap execution cancelled"));
                }
                let mut sampled = self.target.clone();
                let indices: Vec<usize> = if let Some(groups) = &cluster_groups {
                    bootstrap_cluster_rows(&mut state, groups)
                } else {
                    (0..rows)
                        .map(|_| {
                            // Rejection avoids modulo bias for arbitrary frame sizes.
                            let bound = rows as u64;
                            let threshold = bound.wrapping_neg() % bound;
                            loop {
                                let word = bootstrap_word(&mut state);
                                if word >= threshold {
                                    break (word % bound) as usize;
                                }
                            }
                        })
                        .collect()
                };
                for ((_, values), (_, original)) in
                    sampled.columns.iter_mut().zip(&self.target.columns)
                {
                    *values = indices.iter().map(|&index| original[index]).collect();
                }
                if cluster_groups.is_some() {
                    let source_ids: std::collections::BTreeSet<&str> =
                        sampled.source_unit_ids.iter().map(String::as_str).collect();
                    let mut nonce = 0_u64;
                    let prefix = loop {
                        let candidate = format!("bootstrap-slot:{nonce}:");
                        if !source_ids.iter().any(|id| id.starts_with(&candidate)) {
                            break candidate;
                        }
                        nonce = nonce
                            .checked_add(1)
                            .ok_or(Error::Invalid("bootstrap slot identity overflow"))?;
                    };
                    sampled.target_unit_ids =
                        (0..indices.len()).map(|row| format!("{prefix}{row}")).collect();
                }
                // Each replicate redraws its completed tables and posterior on
                // its own streams. On the declared streams every replicate held
                // the same posterior draws, so the retained draws repeated one
                // another and understated the source and imputation spread.
                let replicate_bytes =
                    replicate_source(&self.native_source, policy.seed, replicate)?;
                let run = if self.confounded {
                    replay_confounded_kernel_transport_point(&replicate_bytes, &sampled, ctx)?
                } else {
                    replay_kernel_transport_point(&replicate_bytes, &sampled, ctx)?
                };
                if run.draws.is_empty() {
                    return Err(Error::Invalid("target bootstrap has no posterior draws"));
                }
                let draw_index = bootstrap_pool_index(&mut draw_state, run.draws.len());
                let draw = run.draws[draw_index];
                receipt.word(u64::from(replicate));
                for index in indices {
                    receipt.word(index as u64);
                }
                receipt.bytes(&run.receipt).word(draw_index as u64).word(draw.to_bits());
                if cluster_groups.is_some() {
                    let replicate_decision = compile_chain_decision(
                        effect_chain(&run, &self.quantity),
                        self.decision.clone(),
                        ctx,
                    )?
                    .execute(ctx)?;
                    receipt
                        .bytes(replicate_decision.chain.identity().as_bytes())
                        .bytes(replicate_decision.lowered_program_identity.as_bytes());
                }
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
            effect_chain(&reading, &self.quantity),
            self.decision.clone(),
            ctx,
        )?
        .execute(ctx)?;
        Ok(GraphImputationTransportDecisionResult {
            producer: reading,
            decision,
            lowered_program_identity: self.lowered.identity(),
            calibration: self.unit_calibration(),
        })
    }

    /// Complete-route target effect interval candidate under the prespecified
    /// kernel policy. It does not confer decision reliability.
    ///
    /// # Errors
    /// A source, overlap or posterior-draw refusal.
    #[allow(
        clippy::result_large_err,
        reason = "the structured refusal is returned by value on a cold path"
    )]
    #[allow(
        clippy::too_many_lines,
        reason = "one linear sequence of steps that reads top to bottom"
    )]
    pub fn calibration_candidate(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<GraphImputationKernelTransportCalibrationCandidate, Error> {
        let result = self.execute(ctx)?;
        let mut ordered = result.producer.draws.clone();
        if ordered.len() < 40 || ordered.iter().any(|value| !value.is_finite()) {
            return Err(Error::Invalid(
                "mixed target effect interval needs finite posterior draws",
            ));
        }
        ordered.sort_by(f64::total_cmp);
        let source = GraphImputationSource::decode(&self.native_source).map_err(Error::Invalid)?;
        let parts = source.parts();
        // Under target resampling each draw is one independent replicate, each
        // on its own completed tables. On a fixed frame the draws pool the
        // declared tables, which Rubin's rules combine.
        let replicated = self.cluster_bootstrap.is_some() || self.target_bootstrap.is_some();
        let interval = if replicated {
            order_statistic_interval_95(&ordered)
        } else {
            imputation_pooled_interval_95(
                &result.producer.draws,
                parts.imputations as usize,
                transported_child_df(parts, &self.target.selection_targets),
            )
            .ok_or(Error::Invalid("mixed target draws do not pool across completed tables"))?
        };
        let mut policy = CanonicalWriter::new(
            "antecedent.graph_imputation_kernel_transport.calibration_policy.v1",
        );
        policy.word(self.target.bandwidths.len() as u64);
        for &(id, bandwidth) in &self.target.bandwidths {
            policy.word(u64::from(id)).word(bandwidth.to_bits());
        }
        policy.word(self.target.min_kernel_overlap.to_bits());
        let policy = SemanticDigest::from_bytes(policy.finish()).to_hex();
        let mut functional = format!(
            "target_average_effect:graph_support={}:graphs={}:domains={}:selected={:?}:m={}:sweeps={}:draws={}:prior_bits={:016x}:kernel_policy={policy}",
            parts.support.as_str(),
            parts.atoms.len(),
            parts.domains.iter().map(u8::to_string).collect::<String>(),
            self.target.selection_targets,
            parts.imputations,
            parts.iterations,
            parts.posterior_draws,
            parts.prior_scale.to_bits(),
        );
        if let Some(candidates) = parts.declared_candidate_identity() {
            use std::fmt::Write as _;
            let _ = write!(functional, ":candidates={}", candidates.to_hex());
        }
        // Source and target row counts reach the record through the scope's
        // row count; cluster sizes and seeds are not part of the construction.
        let (feature, model, draws) = if let Some(bootstrap) = &self.cluster_bootstrap {
            functional = format!(
                "{functional}:target_bootstrap=independent_clusters:replicates={}",
                bootstrap.replicates
            );
            (
                if self.confounded { CONFOUNDED_CLUSTER_FEATURE } else { CLUSTER_FEATURE },
                "mar_graph_mixture_kernel_target_cluster_bootstrap_v1",
                bootstrap.replicates,
            )
        } else if let Some(bootstrap) = self.target_bootstrap {
            functional = format!(
                "{functional}:target_bootstrap=iid_rows_unbiased_pool_v2:replicates={}",
                bootstrap.replicates,
            );
            (
                if self.confounded { CONFOUNDED_BOOTSTRAP_FEATURE } else { BOOTSTRAP_FEATURE },
                if self.confounded {
                    "mar_confounded_graph_mixture_kernel_target_iid_bootstrap_v1"
                } else {
                    "mar_graph_mixture_kernel_target_iid_bootstrap_v1"
                },
                bootstrap.replicates,
            )
        } else if self.confounded {
            functional.push_str(":arm_overlap=v1");
            (
                CONFOUNDED_FEATURE,
                "mar_confounded_graph_mixture_kernel_target_v1",
                parts.posterior_draws.saturating_mul(parts.imputations),
            )
        } else {
            functional.push_str(":s_admissibility=v2");
            (
                FEATURE,
                "mar_graph_mixture_kernel_target_v2",
                parts.posterior_draws.saturating_mul(parts.imputations),
            )
        };
        let basis = CalibrationBasis::new(
            [
                "AverageEffect",
                "Dag",
                "graph_posterior",
                "multi_env",
                "Bayesian",
                feature,
                if replicated { REPLICATE_INTERVAL } else { POOLED_INTERVAL },
                "",
                if self.cluster_bootstrap.is_some() {
                    "independent_source_units_and_target_clusters"
                } else {
                    "independent_source_target_units"
                },
                model,
                &functional,
            ]
            .map(Arc::from),
            0.95,
            Arc::from("model_conditional"),
            (parts.rows() as u64).saturating_add(self.target.rows() as u64),
            None,
            Some(draws),
            0.0,
        )
        .with_validation("none");
        Ok(GraphImputationKernelTransportCalibrationCandidate { result, interval, basis })
    }

    /// Re-execute and seal the original source, mixed target, utility and
    /// child decision artifact for independent replay.
    ///
    /// # Errors
    /// Changed result or artifact encoding refusal.
    #[allow(
        clippy::result_large_err,
        reason = "the structured refusal is returned by value on a cold path"
    )]
    pub fn export(
        &self,
        result: &GraphImputationTransportDecisionResult,
        artifact_id: &str,
        ctx: &ExecutionContext,
    ) -> Result<Vec<u8>, Error> {
        let live = self.execute(ctx)?;
        if result.lowered_program_identity != self.lowered.identity()
            || result.calibration != self.unit_calibration()
            || result.producer.receipt != live.producer.receipt
            || result.producer.draws != live.producer.draws
            || result.decision.decision != live.decision.decision
            || result.decision.chain.identity() != live.decision.chain.identity()
        {
            return Err(Error::Invalid("mixed target decision changed on replay"));
        }
        let contract = DecisionContractArtifact::new(self.decision.clone())?;
        let chain = compile_chain_decision(
            effect_chain(&live.producer, &self.quantity),
            self.decision.clone(),
            ctx,
        )?;
        let wire = KernelTransportDecisionWire {
            version: 1,
            required_feature: if self.cluster_bootstrap.is_some() {
                if self.confounded { CONFOUNDED_CLUSTER_FEATURE } else { CLUSTER_FEATURE }
            } else if self.confounded && self.target_bootstrap.is_some() {
                CONFOUNDED_BOOTSTRAP_FEATURE
            } else if self.confounded {
                CONFOUNDED_FEATURE
            } else if self.target_bootstrap.is_some() {
                BOOTSTRAP_FEATURE
            } else {
                FEATURE
            }
            .into(),
            original_source: self.producer.source_bytes().to_vec(),
            target: self.target.clone(),
            quantity: ScientificQuantityWire::from(&self.quantity),
            decision_identity: contract.identity().into(),
            decision_artifact: contract.to_bytes(artifact_id)?,
            chain_artifact: chain.export(&live.decision, artifact_id)?,
            producer_receipt: live.producer.receipt,
            program_identity: *self.lowered.identity().as_bytes(),
            calibration: self.unit_calibration().into(),
            target_bootstrap: self.target_bootstrap,
            confounded: self.confounded,
            cluster_bootstrap: self.cluster_bootstrap.clone(),
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
#[allow(
    clippy::result_large_err,
    reason = "the structured refusal is returned by value on a cold path"
)]
#[allow(clippy::too_many_lines, reason = "one linear sequence of steps that reads top to bottom")]
pub fn consume_graph_imputation_kernel_transport_decision(
    bytes: &[u8],
    expected_program: SemanticDigest,
    ctx: &ExecutionContext,
) -> Result<GraphImputationTransportDecisionResult, Error> {
    if bytes.len() > MAX_ARTIFACT_BYTES {
        return Err(Error::Invalid("mixed target artifact exceeds its byte bound"));
    }
    let wire: KernelTransportDecisionWire = antecedent_io::from_cbor(bytes)?;
    if wire.cluster_bootstrap.is_some() && wire.target_bootstrap.is_some() {
        return Err(Error::Invalid("stored bootstrap policies conflict"));
    }
    let expected_feature = if wire.cluster_bootstrap.is_some() {
        if wire.confounded { CONFOUNDED_CLUSTER_FEATURE } else { CLUSTER_FEATURE }
    } else if wire.confounded && wire.target_bootstrap.is_some() {
        CONFOUNDED_BOOTSTRAP_FEATURE
    } else if wire.confounded {
        CONFOUNDED_FEATURE
    } else if wire.target_bootstrap.is_some() {
        BOOTSTRAP_FEATURE
    } else {
        FEATURE
    };
    if wire.version != 1
        || wire.required_feature != expected_feature
        || wire.program_identity != *expected_program.as_bytes()
        || wire.calibration != capability_calibration(GRAPH_IMPUTATION_COMPILED_DECISION.as_str())
    {
        return Err(Error::Invalid("mixed target feature or identity differs"));
    }
    let original = GraphImputationSource::decode(&wire.original_source).map_err(Error::Invalid)?;
    let parts = original.parts();
    let table = TabularData::from_f64_columns(
        parts
            .names
            .iter()
            .zip(&parts.columns)
            .map(|(name, column)| (name.as_str(), column.as_slice())),
    )
    .map_err(|_| Error::Invalid("stored mixed source table is invalid"))?;
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
    if producer.source_bytes() != wire.original_source.as_slice()
        || original.program() != *producer.lowered().identity().as_bytes()
    {
        return Err(Error::Invalid("original mixed source compiler changed"));
    }
    let quantity: ScientificQuantity = wire.quantity.try_into().map_err(Error::Invalid)?;
    let contract =
        DecisionContractArtifact::from_bytes(&wire.decision_artifact, &wire.decision_identity)?;
    let plan = if let Some(policy) = wire.cluster_bootstrap {
        compile_graph_imputation_kernel_transport_cluster_bootstrap_decision(
            producer,
            wire.target,
            quantity,
            contract.contract().clone(),
            policy,
            wire.confounded,
        )?
    } else if wire.confounded {
        compile_graph_imputation_confounded_kernel_transport_decision(
            producer,
            wire.target,
            quantity,
            contract.contract().clone(),
            wire.target_bootstrap,
        )?
    } else if let Some(policy) = wire.target_bootstrap {
        compile_graph_imputation_kernel_transport_target_bootstrap_decision(
            producer,
            wire.target,
            quantity,
            contract.contract().clone(),
            policy,
        )?
    } else {
        compile_graph_imputation_kernel_transport_decision(
            producer,
            wire.target,
            quantity,
            contract.contract().clone(),
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
        &wire.chain_artifact,
        Some(live.decision.lowered_program_identity),
        ctx,
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
    #[allow(
        clippy::too_many_lines,
        reason = "one linear sequence of steps that reads top to bottom"
    )]
    fn continuous_target_modifier_replays_graph_imputation_and_utility() {
        let mut treatment = Vec::new();
        let mut outcome = Vec::new();
        let mut covariate = Vec::new();
        for row in 0..80 {
            let x = f64::from(row % 20) / 19.0;
            let a = f64::from(u8::from(row % 4 == 0 || row % 4 == 3));
            treatment.push(a);
            outcome.push(if row % 13 == 0 {
                f64::NAN
            } else {
                0.2 * x + a * (0.5 + 0.5 * x) + 0.1 * f64::from(row % 5)
            });
            covariate.push(x);
        }
        let table = TabularData::from_f64_columns([
            ("t", treatment.as_slice()),
            ("y", outcome.as_slice()),
            ("x", covariate.as_slice()),
        ])
        .unwrap();
        let ctx = ExecutionContext::for_tests(29);
        let producer = CompiledGraphImputation::compile(
            table,
            &GraphImputationSpec {
                treatment: VariableId::from_raw(0),
                outcome: VariableId::from_raw(1),
                support: GraphSupport::TreatmentRootPreTreatmentCovariates,
                imputation: ImputationSpec::new()
                    .with_imputations(2)
                    .with_iterations(2)
                    .with_seed(23),
                domains: Arc::<[u8]>::from(vec![1, 0, 0]),
                posterior_draws: 100,
                prior_scale: 1.0,
                execution_seed: 29,
            },
            &ctx,
        )
        .unwrap();
        let target = KernelTransportStandardization {
            columns: vec![(2, (0..80).map(|row| 0.2 + 0.8 * f64::from(row % 20) / 19.0).collect())],
            selection_targets: vec![2],
            source_population: "source".into(),
            target_population: "target".into(),
            source_sample: producer.source_identity().to_hex(),
            target_sample: "independent-continuous-target".into(),
            source_unit_ids: (0..80).map(|row| format!("source-{row}")).collect(),
            target_unit_ids: (0..80).map(|row| format!("target-{row}")).collect(),
            bandwidths: vec![(2, 0.3)],
            min_kernel_overlap: 0.01,
        };
        let original = GraphImputationSource::decode(producer.source_bytes()).unwrap();
        assert!(target.validate(original.parts()).is_ok());
        let mut disjoint = target.clone();
        disjoint.target_unit_ids[0] = disjoint.source_unit_ids[0].clone();
        assert!(disjoint.validate(original.parts()).is_err());
        let quantity = ScientificQuantity {
            variable_id: "y".into(),
            variable_name: "target effect".into(),
            role: QuantityRole::Outcome,
            units: "score".into(),
            population_id: "target".into(),
            regime_id: "do(t=1)-do(t=0)".into(),
            horizon: 0,
            functional_id: "average_effect".into(),
            conditioning: Vec::new(),
            transform_id: "identity".into(),
        };
        let decision = DecisionContract {
            actions: vec![
                DecisionAction {
                    id: "treat".into(),
                    kind: ActionKind::External,
                    inputs: vec![quantity.clone()],
                    utility: UtilityExpr::difference(
                        UtilityExpr::Input(0),
                        UtilityExpr::Const(0.2),
                    ),
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
            constraints: Vec::new(),
            target_population: "target".into(),
            horizon: 0,
            structural_policy: StructuralPolicy::RequireInvariantBestAction,
        };
        let parts = original.parts();
        let bootstrap_table = TabularData::from_f64_columns(
            parts
                .names
                .iter()
                .zip(&parts.columns)
                .map(|(name, column)| (name.as_str(), column.as_slice())),
        )
        .unwrap();
        let bootstrap_producer = CompiledGraphImputation::compile(
            bootstrap_table,
            &GraphImputationSpec {
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
            },
            &ctx,
        )
        .unwrap();
        let bootstrap = compile_graph_imputation_kernel_transport_target_bootstrap_decision(
            bootstrap_producer,
            target.clone(),
            quantity.clone(),
            decision.clone(),
            TargetFrameBootstrap { replicates: 40, seed: 7 },
        )
        .unwrap();
        let cluster_table = TabularData::from_f64_columns(
            parts
                .names
                .iter()
                .zip(&parts.columns)
                .map(|(name, column)| (name.as_str(), column.as_slice())),
        )
        .unwrap();
        let cluster_producer = CompiledGraphImputation::compile(
            cluster_table,
            &GraphImputationSpec {
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
            },
            &ctx,
        )
        .unwrap();
        let cluster = compile_graph_imputation_kernel_transport_cluster_bootstrap_decision(
            cluster_producer,
            target.clone(),
            quantity.clone(),
            decision.clone(),
            TargetClusterBootstrap {
                replicates: 40,
                seed: 29,
                cluster_ids: (0..80).map(|row| format!("cluster-{}", row / 2)).collect(),
            },
            false,
        )
        .unwrap();
        let cluster_result = cluster.execute(&ctx).unwrap();
        assert_eq!(cluster_result.producer.draws.len(), 40);
        let cluster_bytes = cluster.export(&cluster_result, "kernel-cluster-target", &ctx).unwrap();
        let cluster_replay = consume_graph_imputation_kernel_transport_decision(
            &cluster_bytes,
            cluster.lowered().identity(),
            &ctx,
        )
        .unwrap();
        assert_eq!(cluster_result.producer.receipt, cluster_replay.producer.receipt);
        let mut cluster_wire: KernelTransportDecisionWire =
            antecedent_io::from_cbor(&cluster_bytes).unwrap();
        cluster_wire.cluster_bootstrap.as_mut().unwrap().cluster_ids[0].push('x');
        assert!(
            consume_graph_imputation_kernel_transport_decision(
                &antecedent_io::to_cbor(&cluster_wire).unwrap(),
                cluster.lowered().identity(),
                &ctx
            )
            .is_err()
        );
        let plan = compile_graph_imputation_kernel_transport_decision(
            producer, target, quantity, decision,
        )
        .unwrap();
        assert_eq!(plan.plan_view().route().as_str(), "GraphImputationKernelTransportDecision");
        let result = plan.execute(&ctx).unwrap();
        assert!(result.producer.mean.is_finite());
        assert_eq!(result.calibration, "not_measured");
        let bytes = plan.export(&result, "mixed-source-target-decision", &ctx).unwrap();
        let replay = consume_graph_imputation_kernel_transport_decision(
            &bytes,
            result.lowered_program_identity,
            &ctx,
        )
        .unwrap();
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
            &bytes,
            sampled.lowered_program_identity,
            &ctx,
        )
        .unwrap();
        assert_eq!(sampled.producer.draws, replay.producer.draws);
        assert_eq!(sampled.producer.receipt, replay.producer.receipt);
        assert_eq!(sampled.decision.decision, replay.decision.decision);
    }
    #[test]
    #[allow(
        clippy::too_many_lines,
        reason = "one linear sequence of steps that reads top to bottom"
    )]
    fn confounded_binary_continuous_target_replays_both_policies() {
        let binary: Vec<f64> = (0..80).map(|i| f64::from(u8::from(i % 2 == 1))).collect();
        let continuous: Vec<f64> = (0..80).map(|i| f64::from(i % 20) / 19.0).collect();
        let treatment: Vec<f64> = (0..80)
            .map(|i| f64::from(u8::from(if binary[i] == 0.0 { i % 10 < 2 } else { i % 10 < 8 })))
            .collect();
        let outcome: Vec<f64> = (0..80)
            .map(|i| {
                if i % 13 == 0 {
                    f64::NAN
                } else {
                    0.2 * binary[i]
                        + 0.1 * continuous[i]
                        + treatment[i] * (0.5 + 0.3 * continuous[i])
                        + 0.03 * (i % 7) as f64
                }
            })
            .collect();
        let ctx = ExecutionContext::for_tests(29);
        let producer = || {
            let table = TabularData::from_f64_columns([
                ("t", treatment.as_slice()),
                ("y", outcome.as_slice()),
                ("x", continuous.as_slice()),
                ("z", binary.as_slice()),
            ])
            .unwrap();
            CompiledGraphImputation::compile(
                table,
                &GraphImputationSpec {
                    treatment: VariableId::from_raw(0),
                    outcome: VariableId::from_raw(1),
                    support: GraphSupport::PreTreatmentCovariates,
                    imputation: ImputationSpec::new()
                        .with_imputations(2)
                        .with_iterations(2)
                        .with_seed(23),
                    domains: Arc::from(vec![1, 0, 0, 1]),
                    posterior_draws: 100,
                    prior_scale: 1.0,
                    execution_seed: 29,
                },
                &ctx,
            )
            .unwrap()
        };
        let fixed_producer = producer();
        let fixed_source = GraphImputationSource::decode(fixed_producer.source_bytes()).unwrap();
        assert!(
            fixed_source.parts().atoms.iter().any(|atom| !atom.adjustment.is_empty()),
            "the retained graph class must include a confounded adjustment"
        );
        let target = KernelTransportStandardization {
            columns: vec![
                (2, (0..80).map(|i| 0.1 + 0.8 * f64::from(i % 20) / 19.0).collect()),
                (3, (0..80).map(|i| f64::from(u8::from(i % 3 != 0))).collect()),
            ],
            selection_targets: vec![2, 3],
            source_population: "source".into(),
            target_population: "target".into(),
            source_sample: fixed_producer.source_identity().to_hex(),
            target_sample: "independent-target".into(),
            source_unit_ids: (0..80).map(|i| format!("source-{i}")).collect(),
            target_unit_ids: (0..80).map(|i| format!("target-{i}")).collect(),
            bandwidths: vec![(2, 0.3)],
            min_kernel_overlap: 0.01,
        };
        let quantity = ScientificQuantity {
            variable_id: "y".into(),
            variable_name: "effect".into(),
            role: QuantityRole::Outcome,
            units: "score".into(),
            population_id: "target".into(),
            regime_id: "do(t=1)-do(t=0)".into(),
            horizon: 0,
            functional_id: "average_effect".into(),
            conditioning: Vec::new(),
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
                    utility: UtilityExpr::product(UtilityExpr::Input(0), UtilityExpr::Const(0.0)),
                },
            ],
            utility_units: "score".into(),
            criterion: DecisionCriterion::PosteriorExpectedUtility,
            constraints: Vec::new(),
            target_population: "target".into(),
            horizon: 0,
            structural_policy: StructuralPolicy::RequireInvariantBestAction,
        };
        let all_control = vec![0.0; 80];
        let no_arm_table = TabularData::from_f64_columns([
            ("t", all_control.as_slice()),
            ("y", outcome.as_slice()),
            ("x", continuous.as_slice()),
            ("z", binary.as_slice()),
        ])
        .unwrap();
        let no_arm_producer = CompiledGraphImputation::compile(
            no_arm_table,
            &GraphImputationSpec {
                treatment: VariableId::from_raw(0),
                outcome: VariableId::from_raw(1),
                support: GraphSupport::PreTreatmentCovariates,
                imputation: ImputationSpec::new()
                    .with_imputations(2)
                    .with_iterations(2)
                    .with_seed(23),
                domains: Arc::from(vec![1, 0, 0, 1]),
                posterior_draws: 100,
                prior_scale: 1.0,
                execution_seed: 29,
            },
            &ctx,
        )
        .unwrap();
        let mut no_arm_target = target.clone();
        no_arm_target.source_sample = no_arm_producer.source_identity().to_hex();
        let no_arm = compile_graph_imputation_confounded_kernel_transport_decision(
            no_arm_producer,
            no_arm_target,
            quantity.clone(),
            decision.clone(),
            None,
        )
        .unwrap();
        assert!(no_arm.execute(&ctx).is_err());
        let fixed = compile_graph_imputation_confounded_kernel_transport_decision(
            fixed_producer,
            target.clone(),
            quantity.clone(),
            decision.clone(),
            None,
        )
        .unwrap();
        let cluster = compile_graph_imputation_kernel_transport_cluster_bootstrap_decision(
            producer(),
            target.clone(),
            quantity.clone(),
            decision.clone(),
            TargetClusterBootstrap {
                replicates: 40,
                seed: 19,
                cluster_ids: (0..80).map(|row| format!("cluster-{}", row / 2)).collect(),
            },
            true,
        )
        .unwrap();
        let sampled = compile_graph_imputation_confounded_kernel_transport_decision(
            producer(),
            target,
            quantity,
            decision,
            Some(TargetFrameBootstrap { replicates: 40, seed: 13 }),
        )
        .unwrap();
        assert_ne!(fixed.lowered().identity(), sampled.lowered().identity());
        for plan in [&fixed, &sampled, &cluster] {
            let candidate = plan.calibration_candidate(&ctx).unwrap();
            assert_eq!(candidate.result.calibration, "not_measured");
            assert_eq!(candidate.basis.validation.as_ref(), "none");
            let bytes = plan.export(&candidate.result, "confounded-mixed", &ctx).unwrap();
            let replay = consume_graph_imputation_kernel_transport_decision(
                &bytes,
                candidate.result.lowered_program_identity,
                &ctx,
            )
            .unwrap();
            assert_eq!(replay.producer.draws, candidate.result.producer.draws);
            assert_eq!(replay.decision.decision, candidate.result.decision.decision);
            let mut wire: KernelTransportDecisionWire = antecedent_io::from_cbor(&bytes).unwrap();
            wire.target.bandwidths[0].1 *= 2.0;
            let changed = antecedent_io::to_cbor(&wire).unwrap();
            assert!(
                consume_graph_imputation_kernel_transport_decision(
                    &changed,
                    candidate.result.lowered_program_identity,
                    &ctx,
                )
                .is_err()
            );
        }
    }

    #[test]
    #[allow(clippy::too_many_lines, reason = "one fixture and its comparisons read in order")]
    fn site_graph_imputation_kernel_transport_decision_question_binds_target_and_resampling() {
        let mut treatment = Vec::new();
        let mut outcome = Vec::new();
        let mut covariate = Vec::new();
        for row in 0..80 {
            let x = f64::from(row % 20) / 19.0;
            let a = f64::from(u8::from(row % 4 == 0 || row % 4 == 3));
            treatment.push(a);
            outcome.push(if row % 13 == 0 {
                f64::NAN
            } else {
                0.2 * x + a * (0.5 + 0.5 * x) + 0.1 * f64::from(row % 5)
            });
            covariate.push(x);
        }
        let table = TabularData::from_f64_columns([
            ("t", treatment.as_slice()),
            ("y", outcome.as_slice()),
            ("x", covariate.as_slice()),
        ])
        .unwrap();
        let ctx = ExecutionContext::for_tests(29);
        let producer = CompiledGraphImputation::compile(
            table,
            &GraphImputationSpec {
                treatment: VariableId::from_raw(0),
                outcome: VariableId::from_raw(1),
                support: GraphSupport::TreatmentRootPreTreatmentCovariates,
                imputation: ImputationSpec::new()
                    .with_imputations(2)
                    .with_iterations(2)
                    .with_seed(23),
                domains: Arc::<[u8]>::from(vec![1, 0, 0]),
                posterior_draws: 100,
                prior_scale: 1.0,
                execution_seed: 29,
            },
            &ctx,
        )
        .unwrap();
        let source = GraphImputationSource::decode(producer.source_bytes()).unwrap();
        let target = KernelTransportStandardization {
            columns: vec![(2, (0..80).map(|row| 0.2 + 0.8 * f64::from(row % 20) / 19.0).collect())],
            selection_targets: vec![2],
            source_population: "source".into(),
            target_population: "target".into(),
            source_sample: producer.source_identity().to_hex(),
            target_sample: "independent-continuous-target".into(),
            source_unit_ids: (0..80).map(|row| format!("source-{row}")).collect(),
            target_unit_ids: (0..80).map(|row| format!("target-{row}")).collect(),
            bandwidths: vec![(2, 0.3)],
            min_kernel_overlap: 0.01,
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
            conditioning: Vec::new(),
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
                    utility: UtilityExpr::product(UtilityExpr::Input(0), UtilityExpr::Const(0.0)),
                },
            ],
            utility_units: "score".into(),
            criterion: DecisionCriterion::PosteriorExpectedUtility,
            constraints: Vec::new(),
            target_population: "target".into(),
            horizon: 0,
            structural_policy: StructuralPolicy::RequireInvariantBestAction,
        };
        let question = |target: &KernelTransportStandardization,
                        decision: &DecisionContract,
                        bootstrap: Option<TargetFrameBootstrap>,
                        cluster: Option<&TargetClusterBootstrap>| {
            let lowered =
                lower_outer(&source, target, &quantity, decision, bootstrap, false, cluster)
                    .ok()?;
            match lowered.scientific().target() {
                ProblemTarget::QueryFunctional { query, .. } => Some(*query),
                _ => None,
            }
        };
        let clusters = |cluster_ids: Vec<String>| TargetClusterBootstrap {
            replicates: 40,
            seed: 29,
            cluster_ids,
        };
        let paired: Vec<String> = (0..80).map(|row| format!("cluster-{}", row / 2)).collect();
        let mut moved = paired.clone();
        moved.swap(0, 2);
        let mut bandwidth = target.clone();
        bandwidth.bandwidths = vec![(2, 0.4)];
        let mut units = decision.clone();
        units.utility_units = "points".into();
        let rows = |replicates: u32, seed: u64| Some(TargetFrameBootstrap { replicates, seed });
        crate::identity_laws::all_distinct(&[
            question(&target, &decision, None, None),
            question(&bandwidth, &decision, None, None),
            question(&target, &units, None, None),
            question(&target, &decision, rows(40, 7), None),
            question(&target, &decision, rows(40, 8), None),
            question(&target, &decision, None, Some(&clusters(paired))),
            question(&target, &decision, None, Some(&clusters(moved))),
        ]);
    }
}
