//! Graph selection, source imputation, graph-specific target standardization,
//! child fitting and effect utility in one source-bound compiler program.
//!
//! The target supplies complete binary pretreatment covariates from a declared
//! representative sample. A distinct support requires root treatment and
//! pretreatment covariates, so the transport fit can safely augment each
//! graph's minimal adjustment set with selected target covariates and model
//! treatment-effect heterogeneity. Each completed source table gets its own graph
//! weights and target-to-source stratum weights before child fitting. The
//! resulting posterior is conditional on the empirical target covariate frame
//! and model assumptions; it does not propagate target-sample uncertainty.
//! No interval or decision reliability record is inherited from the source
//! graph-imputation route.

use std::sync::Arc;

use antecedent_contract::dependence::SharedFactorKind;
use antecedent_contract::propagation_chain::{
    ChainReading, LawMeaning, StageKind, TableConstruction,
};
use antecedent_contract::{ClaimKind, DataClassification, QuantityRole, ScientificQuantity};
use antecedent_core::graph_imputation_contract::{GraphImputationParts, GraphImputationSource};
use antecedent_core::{
    ByteEstimates, CalibrationBasis, CanonicalWriter, ExecutionContext, PhysicalEffect,
    PhysicalOperator, SemanticDigest, VariableId,
};
use antecedent_data::{ImputationSpec, TabularData};
use antecedent_design::decision_artifact::DecisionContractArtifact;
use antecedent_design::decision_contract::{DecisionContract, DecisionContractError};
use antecedent_estimate::multiple_imputation::rubin_pool;
use antecedent_io::IoError;
use antecedent_io::measured_inference::graph_candidate::{
    GraphPointReading, TransportStandardization, replay_confounded_transport_point,
    replay_transport_point, replicate_source,
};
use antecedent_io::propagation_chain_record::{
    ChainCoefficientWire, ChainOpWire, ChainSourceWire, ChainStageWire, PropagationChainRecordWire,
};
use antecedent_io::quantity_wire::ScientificQuantityWire;
use serde::{Deserialize, Serialize};

use super::graph_imputation::{CompiledGraphImputation, GraphImputationError, GraphImputationSpec};
use super::propagation_chain_decision::{
    ChainDecisionError, ChainDecisionResult, compile_chain_decision, consume_chain_decision,
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

const FEATURE: &str = "graph_imputation_transport_decision_v2";
const CONFOUNDED_FEATURE: &str = "graph_imputation_confounded_transport_decision_v2";
const CONFOUNDED_BOOTSTRAP_FEATURE: &str =
    "graph_imputation_confounded_transport_target_bootstrap_decision_v2";
const BOOTSTRAP_FEATURE: &str = "graph_imputation_transport_target_bootstrap_decision_v2";
const CLUSTER_FEATURE: &str = "graph_imputation_transport_target_cluster_bootstrap_decision_v1";
const CONFOUNDED_CLUSTER_FEATURE: &str =
    "graph_imputation_confounded_transport_target_cluster_bootstrap_decision_v1";
const MAX_ARTIFACT_BYTES: usize = 80 * 1024 * 1024;
/// Interval method of a draw table with one draw per resampling replicate
/// ([`order_statistic_interval_95`]).
pub(crate) const REPLICATE_INTERVAL: &str = "replicate_order_statistic_95";

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
    target_bootstrap: Option<TargetFrameBootstrap>,
    cluster_bootstrap: Option<TargetClusterBootstrap>,
    confounded: bool,
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

/// An independent-cluster target resampling declaration. Cluster IDs are in
/// original target-row order and identify whole independent sampling units.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TargetClusterBootstrap {
    /// Number of full-schedule bootstrap replicates, from 40 to 256.
    pub replicates: u32,
    /// Deterministic resampling seed.
    pub seed: u64,
    /// Original target cluster identities, aligned to target rows.
    pub cluster_ids: Vec<String>,
}

impl TargetClusterBootstrap {
    #[allow(
        clippy::result_large_err,
        reason = "the refusal carries the full transport decision error"
    )]
    pub(crate) fn groups(
        &self,
        rows: usize,
    ) -> Result<Vec<Vec<usize>>, GraphImputationTransportDecisionError> {
        if !(40..=256).contains(&self.replicates)
            || self.cluster_ids.len() != rows
            || self.cluster_ids.iter().any(|id| id.trim().is_empty())
        {
            return Err(GraphImputationTransportDecisionError::Invalid(
                "invalid target cluster bootstrap declaration",
            ));
        }
        let mut groups = std::collections::BTreeMap::<&str, Vec<usize>>::new();
        for (row, id) in self.cluster_ids.iter().enumerate() {
            groups.entry(id.as_str()).or_default().push(row);
        }
        if groups.len() < 2
            || groups
                .len()
                .checked_mul(groups.values().map(Vec::len).max().unwrap_or(0))
                .is_none_or(|bound| bound > 1_000_000)
        {
            return Err(GraphImputationTransportDecisionError::Invalid(
                "target cluster count or worst-case replicate size is unsupported",
            ));
        }
        Ok(groups.into_values().collect())
    }
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

/// Unmeasured whole-route interval candidate and its exact runtime key.
/// It has no public coverage authority until a separate record is earned.
pub struct GraphImputationTransportCalibrationCandidate {
    /// Full compiler execution, including the final decision.
    pub result: GraphImputationTransportDecisionResult,
    /// 95% interval from the target effect draws: replicate order statistics
    /// under target resampling, Rubin's rules across the completed tables on a
    /// fixed frame.
    pub interval: (f64, f64),
    /// Runtime construction and scope for a prospective coverage record.
    pub basis: CalibrationBasis,
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    target_bootstrap: Option<TargetFrameBootstrap>,
    #[serde(default, skip_serializing_if = "is_false")]
    confounded: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    cluster_bootstrap: Option<TargetClusterBootstrap>,
}

#[allow(
    clippy::trivially_copy_pass_by_ref,
    reason = "serde skip_serializing_if passes a reference"
)]
fn is_false(value: &bool) -> bool {
    !*value
}

#[allow(clippy::result_large_err, reason = "the refusal carries the full transport decision error")]
fn check_quantity(
    source: &GraphImputationSource,
    target: &TransportStandardization,
    quantity: &ScientificQuantity,
    decision: &DecisionContract,
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
        || quantity.horizon != 0
        || quantity.horizon != decision.horizon
        || !quantity.conditioning.is_empty()
        || quantity.transform_id != "identity"
        || decision.actions.is_empty()
        || decision.actions.iter().any(|action| {
            action.inputs.len() != 1 || action.inputs[0].require_same_coordinate(quantity).is_err()
        })
    {
        return Err(GraphImputationTransportDecisionError::Invalid(
            "utility must read the exact target binary average-effect coordinate",
        ));
    }
    Ok(())
}

#[allow(clippy::result_large_err, reason = "the refusal carries the full transport decision error")]
#[allow(
    clippy::too_many_lines,
    reason = "one lowering from source checks through the six-layer program"
)]
fn lower_outer(
    source: &GraphImputationSource,
    target: &TransportStandardization,
    quantity: &ScientificQuantity,
    decision: &DecisionContract,
    target_bootstrap: Option<TargetFrameBootstrap>,
    confounded: bool,
    cluster_bootstrap: Option<&TargetClusterBootstrap>,
) -> Result<Arc<LoweredProgram>, GraphImputationTransportDecisionError> {
    if confounded {
        target.validate_confounded(source.parts())?;
    } else {
        target.validate(source.parts())?;
    }
    if target_bootstrap.is_some_and(|policy| !(40..=256).contains(&policy.replicates)) {
        return Err(GraphImputationTransportDecisionError::Invalid(
            "target bootstrap needs 40 to 256 replicates",
        ));
    }
    if let Some(policy) = cluster_bootstrap {
        if target_bootstrap.is_some() {
            return Err(GraphImputationTransportDecisionError::Invalid(
                "row and cluster bootstrap are exclusive",
            ));
        }
        policy.groups(target.rows())?;
    }
    check_quantity(source, target, quantity, decision)?;
    let target_digest = target.digest()?;
    let source_digest = SemanticDigest::from_bytes(source.parts().data_digest);
    let coordinate = antecedent_io::to_cbor(&ScientificQuantityWire::from(quantity))?;
    let contract = decision.identity()?;
    let mut writer =
        CanonicalWriter::new("antecedent.graph_imputation_transport_decision.question.v1");
    writer
        .bytes(&source.program())
        .bytes(&source.parts().contract_digest())
        .bytes(source_digest.as_bytes())
        .bytes(&target_digest)
        .bytes(&coordinate)
        .text(&contract);
    if confounded {
        writer.text(CONFOUNDED_FEATURE);
    } else {
        writer.text(FEATURE);
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
            "bounded_graph_imputation_confounded_binary_target"
        } else {
            "bounded_graph_imputation_binary_target_standardization"
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
            scenario_source: Some(Arc::from("graph_imputation_target_effect_draws_v1")),
        }),
        planning: None,
        chain: None,
    };
    let mut operators = vec![
        PhysicalOperator::declare(PhysicalEffect::Read)
            .estimates(ByteEstimates::absent().with_output(rows))
            .build()
            .map_err(|_| {
                GraphImputationTransportDecisionError::Invalid("invalid source read operator")
            })?,
    ];
    // The empirical fit and each bootstrap replicate run sequentially.
    let executions = 1 + cluster_bootstrap.map_or_else(
        || target_bootstrap.map_or(0, |policy| policy.replicates),
        |policy| policy.replicates,
    );
    for _ in 0..executions {
        operators.push(PhysicalOperator::declare(PhysicalEffect::Fit).build().map_err(|_| {
            GraphImputationTransportDecisionError::Invalid("invalid graph fit operator")
        })?);
        operators.push(PhysicalOperator::declare(PhysicalEffect::Transport).build().map_err(
            |_| {
                GraphImputationTransportDecisionError::Invalid(
                    "invalid target standardization operator",
                )
            },
        )?);
    }
    let physical = PhysicalLayer { operators: Arc::from(operators), design };
    Ok(Arc::new(
        StandaloneCompositionLoweredParts {
            cell: if cluster_bootstrap.is_some() {
                if confounded {
                    ProgramCell::GraphImputationConfoundedTransportClusterBootstrapDecision
                } else {
                    ProgramCell::GraphImputationTransportClusterBootstrapDecision
                }
            } else if confounded {
                ProgramCell::GraphImputationConfoundedTransportDecision
            } else {
                ProgramCell::GraphImputationTransportDecision
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

/// Bind source graph support, imputation schedule, target rows and selection
/// premise, exact effect coordinate and utility before any fitting executes.
///
/// # Errors
/// Invalid target support, quantity, contract or six-layer lowering.
#[allow(clippy::result_large_err, reason = "the refusal carries the full transport decision error")]
pub fn compile_graph_imputation_transport_decision(
    producer: CompiledGraphImputation,
    target: TransportStandardization,
    quantity: ScientificQuantity,
    decision: DecisionContract,
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
    let lowered = lower_outer(&source, &target, &quantity, &decision, None, false, None)?;
    let native_source =
        GraphImputationSource::new(source.parts().clone(), *lowered.identity().as_bytes())
            .map_err(GraphImputationTransportDecisionError::Invalid)?
            .encode();
    Ok(CompiledGraphImputationTransportDecision {
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
/// graph-imputation, binary-stratum transport and utility program. The empirical
/// target rows are resampled together, preserving their joint distribution.
///
/// # Errors
/// Invalid source, target, bootstrap policy or compiler binding.
#[allow(clippy::result_large_err, reason = "the refusal carries the full transport decision error")]
pub fn compile_graph_imputation_transport_target_bootstrap_decision(
    producer: CompiledGraphImputation,
    target: TransportStandardization,
    quantity: ScientificQuantity,
    decision: DecisionContract,
    policy: TargetFrameBootstrap,
) -> Result<CompiledGraphImputationTransportDecision, GraphImputationTransportDecisionError> {
    let source = GraphImputationSource::decode(producer.source_bytes())
        .map_err(GraphImputationTransportDecisionError::Invalid)?;
    if source.program() != *producer.lowered().identity().as_bytes()
        || target.source_sample != producer.source_identity().to_hex()
    {
        return Err(GraphImputationTransportDecisionError::Invalid(
            "target source-sample identity differs from retained producer",
        ));
    }
    let lowered = lower_outer(&source, &target, &quantity, &decision, Some(policy), false, None)?;
    let native_source =
        GraphImputationSource::new(source.parts().clone(), *lowered.identity().as_bytes())
            .map_err(GraphImputationTransportDecisionError::Invalid)?
            .encode();
    Ok(CompiledGraphImputationTransportDecision {
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

/// Compile the separate observational pretreatment-confounding transport
/// route. Every retained graph must pass the augmented backdoor check; each
/// completed target stratum must contain both treatment arms.
///
/// # Errors
/// Invalid source identity, unsafe graph, target or decision.
#[allow(clippy::result_large_err, reason = "the refusal carries the full transport decision error")]
pub fn compile_graph_imputation_confounded_transport_decision(
    producer: CompiledGraphImputation,
    target: TransportStandardization,
    quantity: ScientificQuantity,
    decision: DecisionContract,
) -> Result<CompiledGraphImputationTransportDecision, GraphImputationTransportDecisionError> {
    let source = GraphImputationSource::decode(producer.source_bytes())
        .map_err(GraphImputationTransportDecisionError::Invalid)?;
    if source.program() != *producer.lowered().identity().as_bytes()
        || target.source_sample != producer.source_identity().to_hex()
    {
        return Err(GraphImputationTransportDecisionError::Invalid(
            "confounded source identity differs",
        ));
    }
    let lowered = lower_outer(&source, &target, &quantity, &decision, None, true, None)?;
    let native_source =
        GraphImputationSource::new(source.parts().clone(), *lowered.identity().as_bytes())
            .map_err(GraphImputationTransportDecisionError::Invalid)?
            .encode();
    Ok(CompiledGraphImputationTransportDecision {
        producer,
        target,
        quantity,
        decision,
        native_source,
        lowered,
        target_bootstrap: None,
        cluster_bootstrap: None,
        confounded: true,
    })
}

/// Compile target-frame IID resampling with the confounded-pretreatment
/// graph route. Every resampled frame reruns imputation, graph selection,
/// augmented backdoor validation, stratum overlap and child fitting; the
/// resulting effect draws feed the final compiled utility decision.
///
/// # Errors
/// Invalid source binding, unsafe graph, target, bootstrap policy or utility.
#[allow(clippy::result_large_err, reason = "the refusal carries the full transport decision error")]
pub fn compile_graph_imputation_confounded_transport_target_bootstrap_decision(
    producer: CompiledGraphImputation,
    target: TransportStandardization,
    quantity: ScientificQuantity,
    decision: DecisionContract,
    policy: TargetFrameBootstrap,
) -> Result<CompiledGraphImputationTransportDecision, GraphImputationTransportDecisionError> {
    let source = GraphImputationSource::decode(producer.source_bytes())
        .map_err(GraphImputationTransportDecisionError::Invalid)?;
    if source.program() != *producer.lowered().identity().as_bytes()
        || target.source_sample != producer.source_identity().to_hex()
    {
        return Err(GraphImputationTransportDecisionError::Invalid(
            "confounded source identity differs",
        ));
    }
    let lowered = lower_outer(&source, &target, &quantity, &decision, Some(policy), true, None)?;
    let native_source =
        GraphImputationSource::new(source.parts().clone(), *lowered.identity().as_bytes())
            .map_err(GraphImputationTransportDecisionError::Invalid)?
            .encode();
    Ok(CompiledGraphImputationTransportDecision {
        producer,
        target,
        quantity,
        decision,
        native_source,
        lowered,
        target_bootstrap: Some(policy),
        cluster_bootstrap: None,
        confounded: true,
    })
}

/// Compile an independent-cluster binary target bootstrap. Every replicate
/// reruns graph discovery, imputation, target standardization and decision on
/// its own imputation and posterior streams, and the interval is the replicate
/// order-statistic interval of the retained draws. The confounded flag chooses
/// the separately checked pretreatment support.
///
/// # Errors
/// Invalid producer, frame, cluster policy or decision.
#[allow(clippy::result_large_err, reason = "the refusal carries the full transport decision error")]
pub fn compile_graph_imputation_transport_cluster_bootstrap_decision(
    producer: CompiledGraphImputation,
    target: TransportStandardization,
    quantity: ScientificQuantity,
    decision: DecisionContract,
    policy: TargetClusterBootstrap,
    confounded: bool,
) -> Result<CompiledGraphImputationTransportDecision, GraphImputationTransportDecisionError> {
    let source = GraphImputationSource::decode(producer.source_bytes())
        .map_err(GraphImputationTransportDecisionError::Invalid)?;
    if source.program() != *producer.lowered().identity().as_bytes()
        || target.source_sample != producer.source_identity().to_hex()
    {
        return Err(GraphImputationTransportDecisionError::Invalid(
            "cluster target source identity differs",
        ));
    }
    let lowered =
        lower_outer(&source, &target, &quantity, &decision, None, confounded, Some(&policy))?;
    let native_source =
        GraphImputationSource::new(source.parts().clone(), *lowered.identity().as_bytes())
            .map_err(GraphImputationTransportDecisionError::Invalid)?
            .encode();
    Ok(CompiledGraphImputationTransportDecision {
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
    reason = "the remainder is below a bound that came from a usize"
)]
pub(crate) fn bootstrap_pool_index(state: &mut u64, len: usize) -> usize {
    let bound = len as u64;
    let threshold = bound.wrapping_neg() % bound;
    loop {
        let word = bootstrap_word(state);
        if word >= threshold {
            return (word % bound) as usize;
        }
    }
}

/// Equal-tail 95% interval from `B` exchangeable draws sorted ascending: their
/// `floor(0.025 (B + 1))`-th and `ceil(0.975 (B + 1))`-th order statistics,
/// clamped to the sample. A further draw exchangeable with them falls between
/// those order statistics with probability `(upper - lower) / (B + 1)`, at
/// least 0.95 for every `B >= 39`. Nearest-rank quantiles (`ceil(pB)`) give
/// `38/41` at forty replicates and miss twice as often above as below,
/// whatever the draws' law.
pub(crate) fn order_statistic_interval_95(ordered: &[f64]) -> (f64, f64) {
    let slots = ordered.len() + 1;
    let lower = (slots * 25 / 1000).clamp(1, ordered.len());
    let upper = (slots * 975).div_ceil(1000).clamp(lower, ordered.len());
    (ordered[lower - 1], ordered[upper - 1])
}

/// Interval method of a fixed frame's draws pooled across completed tables
/// ([`imputation_pooled_interval_95`]).
pub(crate) const POOLED_INTERVAL: &str = "rubin_barnard_t_95";

/// 95% interval of a fixed frame's posterior draws pooled from `tables`
/// completed tables, held in their execution order as equal consecutive
/// blocks, by Rubin's rules: `Q ± t_v sqrt(W + (1 + 1/m) B)` from the table
/// posterior means and variances, with the Barnard-Rubin degrees of freedom at
/// the complete-data `df_complete` (large-sample when `None`).
///
/// A plain mixture of `m` tables' posteriors has variance `W + (m - 1) B / m`,
/// with `W` the mean within-table variance and `B` the variance of the table
/// means: it is the posterior only as `m` grows. At the two tables a study
/// commonly declares, the mixture carries `B / 2` of the `3 B / 2` Rubin's
/// total assigns a finite `m`, and with a tenth of the outcomes missing its
/// nearest-rank interval covers about 0.93. `B` then rests on one degree of
/// freedom, which the `t` reference carries and a quantile of the mixture
/// cannot. `None` unless the draws split into `tables >= 2` equal blocks of at
/// least two finite draws.
pub(crate) fn imputation_pooled_interval_95(
    draws: &[f64],
    tables: usize,
    df_complete: Option<f64>,
) -> Option<(f64, f64)> {
    if tables < 2 || draws.len() % tables != 0 || draws.len() / tables < 2 {
        return None;
    }
    let block = draws.len() / tables;
    let mut means = Vec::with_capacity(tables);
    let mut variances = Vec::with_capacity(tables);
    for table in draws.chunks_exact(block) {
        let mean = table.iter().sum::<f64>() / block as f64;
        means.push(mean);
        variances.push(
            table.iter().map(|value| (value - mean).powi(2)).sum::<f64>() / (block - 1) as f64,
        );
    }
    let pooled = rubin_pool(&means, &variances, df_complete).ok()?;
    let interval = pooled.interval(0.95);
    (interval.0.is_finite() && interval.1.is_finite()).then_some(interval)
}

/// Complete-data degrees of freedom of a transported child fit: the source
/// rows less the coefficients of the widest retained design, an intercept,
/// the treatment, and each adjusted or selected covariate with its centred
/// treatment interaction. `None` when no degree remains.
pub(crate) fn transported_child_df(
    parts: &GraphImputationParts,
    selections: &[u32],
) -> Option<f64> {
    let widest = parts
        .atoms
        .iter()
        .map(|atom| {
            let mut covariates: Vec<u32> =
                atom.adjustment.iter().chain(selections).copied().collect();
            covariates.sort_unstable();
            covariates.dedup();
            covariates.len()
        })
        .max()
        .unwrap_or(selections.len());
    let coefficients = 2 + 2 * widest;
    (parts.rows() > coefficients).then(|| (parts.rows() - coefficients) as f64)
}

pub(crate) fn bootstrap_cluster_rows(state: &mut u64, groups: &[Vec<usize>]) -> Vec<usize> {
    let mut rows = Vec::new();
    for _ in 0..groups.len() {
        rows.extend_from_slice(&groups[bootstrap_pool_index(state, groups.len())]);
    }
    rows
}

pub(crate) fn effect_chain(
    reading: &GraphPointReading,
    quantity: &ScientificQuantity,
) -> PropagationChainRecordWire {
    let receipt = SemanticDigest::from_bytes(reading.receipt).to_hex();
    let source = ChainSourceWire::draws(
        "transport.effect.posterior",
        LawMeaning::Posterior,
        &format!("graph_imputation_transport:{receipt}"),
        &reading.draws,
    )
    .with_factor(SharedFactorKind::Model, &receipt);
    PropagationChainRecordWire::draft(
        TableConstruction::JointDraws,
        ChainReading::Posterior,
        "transport.effect",
        vec![source],
        vec![ChainStageWire::new(
            "transport.effect",
            StageKind::DerivedQuantity,
            &["transport.effect.posterior"],
            ChainOpWire::affine(ChainCoefficientWire::fixed(1.0), ChainCoefficientWire::fixed(0.0)),
        )],
    )
    .with_output_quantity(ScientificQuantityWire::from(quantity))
}

impl CompiledGraphImputationTransportDecision {
    /// Inspect the complete six-layer schedule before executing it.
    #[must_use]
    pub fn lowered(&self) -> &LoweredProgram {
        &self.lowered
    }

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
    #[allow(
        clippy::result_large_err,
        reason = "the refusal carries the full transport decision error"
    )]
    #[allow(
        clippy::too_many_lines,
        reason = "one pass from reimputation through the decision utility"
    )]
    #[allow(
        clippy::cast_possible_truncation,
        reason = "the remainder is below a bound that came from a usize"
    )]
    pub fn execute(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<GraphImputationTransportDecisionResult, GraphImputationTransportDecisionError> {
        let source = GraphImputationSource::decode(&self.native_source)
            .map_err(GraphImputationTransportDecisionError::Invalid)?;
        if source.program() != *self.lowered.identity().as_bytes()
            || self.target.source_sample != self.producer.source_identity().to_hex()
        {
            return Err(GraphImputationTransportDecisionError::Invalid(
                "retained native source changed",
            ));
        }
        let mut reading = if self.confounded {
            replay_confounded_transport_point(&self.native_source, &self.target, ctx)?
        } else {
            replay_transport_point(&self.native_source, &self.target, ctx)?
        };
        let cluster_groups = self
            .cluster_bootstrap
            .as_ref()
            .map(|policy| policy.groups(self.target.rows()))
            .transpose()?;
        let effective_bootstrap = self.target_bootstrap.or_else(|| {
            self.cluster_bootstrap.as_ref().map(|policy| TargetFrameBootstrap {
                replicates: policy.replicates,
                seed: policy.seed,
            })
        });
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
            let _bootstrap_lease = ctx.acquire_admission(0, bootstrap_bytes).map_err(|_| {
                GraphImputationTransportDecisionError::Invalid("target bootstrap admission refused")
            })?;
            let mut state = policy.seed;
            // Domain-separated stream selects uniformly across all imputation/graph draws.
            let mut draw_state =
                antecedent_core::execution::mix64(policy.seed ^ 0x4452_4157_5f56_3201);
            let rows = self.target.rows();
            let mut draws = Vec::with_capacity(policy.replicates as usize);
            let mut receipt = CanonicalWriter::new(if cluster_groups.is_some() {
                "antecedent.graph_imputation_transport.target_cluster_bootstrap.execution.v1"
            } else if self.confounded {
                "antecedent.graph_imputation_confounded_transport.target_bootstrap.execution.v2"
            } else {
                "antecedent.graph_imputation_transport.target_bootstrap.execution.v2"
            });
            receipt
                .bytes(self.lowered.identity().as_bytes())
                .bytes(&self.target.digest()?)
                .word(u64::from(policy.replicates))
                .word(policy.seed);
            for replicate in 0..policy.replicates {
                if ctx.cancellation.is_cancelled() {
                    return Err(GraphImputationTransportDecisionError::Invalid(
                        "target bootstrap execution cancelled",
                    ));
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
                // Cluster resampling can change row count. Give each slot a
                // distinct deterministic identity, disjoint from source units.
                if cluster_groups.is_some() {
                    let source_ids: std::collections::BTreeSet<&str> =
                        sampled.source_unit_ids.iter().map(String::as_str).collect();
                    let mut nonce = 0_u64;
                    let prefix = loop {
                        let candidate = format!("bootstrap-slot:{nonce}:");
                        if !source_ids.iter().any(|id| id.starts_with(&candidate)) {
                            break candidate;
                        }
                        nonce = nonce.checked_add(1).ok_or(
                            GraphImputationTransportDecisionError::Invalid(
                                "bootstrap slot identity overflow",
                            ),
                        )?;
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
                    replay_confounded_transport_point(&replicate_bytes, &sampled, ctx)?
                } else {
                    replay_transport_point(&replicate_bytes, &sampled, ctx)?
                };
                if run.draws.is_empty() {
                    return Err(GraphImputationTransportDecisionError::Invalid(
                        "target bootstrap has no posterior draws",
                    ));
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
            return Err(GraphImputationTransportDecisionError::Invalid(
                "native replay belongs to another source or program",
            ));
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
            calibration: "not_measured",
        })
    }

    /// Run the complete source-to-decision schedule and expose its exact
    /// target-effect interval candidate for prospective whole-route coverage
    /// measurement. This does not change the decision's calibration standing.
    ///
    /// # Errors
    /// Any execution refusal or an invalid final draw table.
    #[allow(
        clippy::result_large_err,
        reason = "the refusal carries the full transport decision error"
    )]
    #[allow(
        clippy::too_many_lines,
        reason = "one pass from execution through the interval candidate and its calibration basis"
    )]
    pub fn calibration_candidate(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<GraphImputationTransportCalibrationCandidate, GraphImputationTransportDecisionError>
    {
        use std::fmt::Write as _;
        let result = self.execute(ctx)?;
        let mut ordered = result.producer.draws.clone();
        if ordered.len() < 40 || ordered.iter().any(|value| !value.is_finite()) {
            return Err(GraphImputationTransportDecisionError::Invalid(
                "target effect interval needs finite posterior draws",
            ));
        }
        ordered.sort_by(f64::total_cmp);
        let source = GraphImputationSource::decode(&self.native_source)
            .map_err(GraphImputationTransportDecisionError::Invalid)?;
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
            .ok_or(GraphImputationTransportDecisionError::Invalid(
                "target effect draws do not pool across completed tables",
            ))?
        };
        let mut functional = format!(
            "target_average_effect:graph_support={}:graphs={}:variables={}:domains={}:selected={:?}:m={}:sweeps={}:draws={}:prior_bits={:016x}:target_columns={}",
            parts.support.as_str(),
            parts.atoms.len(),
            parts.names.len(),
            parts.domains.iter().map(u8::to_string).collect::<String>(),
            self.target.selection_targets,
            parts.imputations,
            parts.iterations,
            parts.posterior_draws,
            parts.prior_scale.to_bits(),
            self.target.columns.len(),
        );
        let (feature, model, draws) = if let Some(bootstrap) = &self.cluster_bootstrap {
            let mut sizes: Vec<_> =
                bootstrap.groups(self.target.rows())?.iter().map(Vec::len).collect();
            sizes.sort_unstable();
            let _ = write!(
                functional,
                ":source_n={}:target_n={}:target_bootstrap=independent_clusters:replicates={}",
                parts.rows(),
                self.target.rows(),
                bootstrap.replicates
            );
            let _ = write!(functional, ":cluster_sizes={sizes:?}");
            (
                if self.confounded { CONFOUNDED_CLUSTER_FEATURE } else { CLUSTER_FEATURE },
                "mar_graph_mixture_binary_target_cluster_bootstrap_v1",
                bootstrap.replicates,
            )
        } else if let Some(bootstrap) = self.target_bootstrap {
            let _ = write!(
                functional,
                ":source_n={}:target_n={}:target_bootstrap=iid_rows_unbiased_pool_v2:replicates={}:seed={}",
                parts.rows(),
                self.target.rows(),
                bootstrap.replicates,
                bootstrap.seed,
            );
            (
                if self.confounded { CONFOUNDED_BOOTSTRAP_FEATURE } else { BOOTSTRAP_FEATURE },
                if self.confounded {
                    "mar_confounded_graph_mixture_binary_target_iid_bootstrap_v1"
                } else {
                    "mar_graph_mixture_binary_target_iid_bootstrap_v1"
                },
                bootstrap.replicates,
            )
        } else if self.confounded {
            let _ = write!(
                functional,
                ":source_n={}:target_n={}:s_admissibility=v2",
                parts.rows(),
                self.target.rows()
            );
            (
                CONFOUNDED_FEATURE,
                "mar_graph_mixture_confounded_target_standardization_v2",
                parts.posterior_draws.saturating_mul(parts.imputations),
            )
        } else {
            let _ = write!(
                functional,
                ":source_n={}:target_n={}:s_admissibility=v2",
                parts.rows(),
                self.target.rows()
            );
            (
                FEATURE,
                "mar_graph_mixture_target_standardization_v2",
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
        Ok(GraphImputationTransportCalibrationCandidate { result, interval, basis })
    }

    /// Export the original source, target frame, final contract and live
    /// receipt after replaying the complete compiled schedule.
    ///
    /// # Errors
    /// A changed receipt, utility or artifact encoding.
    #[allow(
        clippy::result_large_err,
        reason = "the refusal carries the full transport decision error"
    )]
    pub fn export(
        &self,
        result: &GraphImputationTransportDecisionResult,
        artifact_id: &str,
        ctx: &ExecutionContext,
    ) -> Result<Vec<u8>, GraphImputationTransportDecisionError> {
        let live = self.execute(ctx)?;
        if result.lowered_program_identity != self.lowered.identity()
            || result.calibration != "not_measured"
            || result.producer.receipt != live.producer.receipt
            || result.producer.draws != live.producer.draws
            || result.decision.decision != live.decision.decision
            || result.decision.chain.identity() != live.decision.chain.identity()
        {
            return Err(GraphImputationTransportDecisionError::Invalid(
                "transport decision changed on replay",
            ));
        }
        let contract = DecisionContractArtifact::new(self.decision.clone())?;
        let chain = compile_chain_decision(
            effect_chain(&live.producer, &self.quantity),
            self.decision.clone(),
            ctx,
        )?;
        let wire = GraphImputationTransportDecisionWire {
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
            calibration: "not_measured".into(),
            target_bootstrap: self.target_bootstrap,
            confounded: self.confounded,
            cluster_bootstrap: self.cluster_bootstrap.clone(),
        };
        let bytes = antecedent_io::to_cbor(&wire)?;
        if bytes.len() > MAX_ARTIFACT_BYTES {
            return Err(GraphImputationTransportDecisionError::Invalid(
                "combined artifact exceeds its byte bound",
            ));
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
#[allow(clippy::result_large_err, reason = "the refusal carries the full transport decision error")]
#[allow(
    clippy::too_many_lines,
    reason = "one replay from artifact checks through the final decision comparison"
)]
pub fn consume_graph_imputation_transport_decision(
    bytes: &[u8],
    expected_program: SemanticDigest,
    ctx: &ExecutionContext,
) -> Result<GraphImputationTransportDecisionResult, GraphImputationTransportDecisionError> {
    if bytes.len() > MAX_ARTIFACT_BYTES {
        return Err(GraphImputationTransportDecisionError::Invalid(
            "combined artifact exceeds its byte bound",
        ));
    }
    let wire: GraphImputationTransportDecisionWire = antecedent_io::from_cbor(bytes)?;
    if wire.cluster_bootstrap.is_some() && wire.target_bootstrap.is_some() {
        return Err(GraphImputationTransportDecisionError::Invalid(
            "stored bootstrap policies conflict",
        ));
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
        || wire.calibration != "not_measured"
    {
        return Err(GraphImputationTransportDecisionError::Invalid(
            "combined feature or identity differs",
        ));
    }
    let original = GraphImputationSource::decode(&wire.original_source)
        .map_err(GraphImputationTransportDecisionError::Invalid)?;
    let parts = original.parts();
    let table = TabularData::from_f64_columns(
        parts
            .names
            .iter()
            .zip(&parts.columns)
            .map(|(name, column)| (name.as_str(), column.as_slice())),
    )
    .map_err(|_| {
        GraphImputationTransportDecisionError::Invalid("stored source table is invalid")
    })?;
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
    let producer = CompiledGraphImputation::compile(table, &spec, ctx)?;
    if producer.source_bytes() != wire.original_source.as_slice()
        || original.program() != *producer.lowered().identity().as_bytes()
    {
        return Err(GraphImputationTransportDecisionError::Invalid(
            "original source compiler changed",
        ));
    }
    let quantity: ScientificQuantity =
        wire.quantity.try_into().map_err(GraphImputationTransportDecisionError::Invalid)?;
    let contract =
        DecisionContractArtifact::from_bytes(&wire.decision_artifact, &wire.decision_identity)?;
    let plan = if let Some(policy) = wire.cluster_bootstrap {
        compile_graph_imputation_transport_cluster_bootstrap_decision(
            producer,
            wire.target,
            quantity,
            contract.contract().clone(),
            policy,
            wire.confounded,
        )?
    } else if let (true, Some(policy)) = (wire.confounded, wire.target_bootstrap) {
        compile_graph_imputation_confounded_transport_target_bootstrap_decision(
            producer,
            wire.target,
            quantity,
            contract.contract().clone(),
            policy,
        )?
    } else if wire.confounded {
        compile_graph_imputation_confounded_transport_decision(
            producer,
            wire.target,
            quantity,
            contract.contract().clone(),
        )?
    } else if let Some(policy) = wire.target_bootstrap {
        compile_graph_imputation_transport_target_bootstrap_decision(
            producer,
            wire.target,
            quantity,
            contract.contract().clone(),
            policy,
        )?
    } else {
        compile_graph_imputation_transport_decision(
            producer,
            wire.target,
            quantity,
            contract.contract().clone(),
        )?
    };
    if plan.lowered.identity() != expected_program {
        return Err(GraphImputationTransportDecisionError::Invalid("outer compiler changed"));
    }
    let live = plan.execute(ctx)?;
    if live.producer.receipt != wire.producer_receipt {
        return Err(GraphImputationTransportDecisionError::Invalid(
            "target-standardized producer changed",
        ));
    }
    let replay = consume_chain_decision(
        &wire.chain_artifact,
        Some(live.decision.lowered_program_identity),
        ctx,
    )?;
    if replay.chain.identity() != live.decision.chain.identity()
        || replay.decision != live.decision.decision
    {
        return Err(GraphImputationTransportDecisionError::Invalid(
            "effect utility changed on replay",
        ));
    }
    Ok(live)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cluster_bootstrap_keeps_whole_clusters_and_is_reproducible() {
        let policy = TargetClusterBootstrap {
            replicates: 40,
            seed: 19,
            cluster_ids: vec!["a".into(), "b".into(), "a".into(), "b".into(), "c".into()],
        };
        let groups = policy.groups(5).unwrap();
        let mut first = policy.seed;
        let mut second = policy.seed;
        let rows = bootstrap_cluster_rows(&mut first, &groups);
        assert_eq!(rows, bootstrap_cluster_rows(&mut second, &groups));
        assert_eq!(
            rows.iter().filter(|&&row| row == 0).count(),
            rows.iter().filter(|&&row| row == 2).count()
        );
        assert_eq!(
            rows.iter().filter(|&&row| row == 1).count(),
            rows.iter().filter(|&&row| row == 3).count()
        );
        assert!(
            TargetClusterBootstrap { cluster_ids: vec!["same".into(); 5], ..policy }
                .groups(5)
                .is_err()
        );
    }

    #[test]
    #[allow(clippy::float_cmp, reason = "the interval returns sample values exactly")]
    fn order_statistic_interval_reaches_nominal_exchangeable_coverage() {
        for (draws, lower, upper) in [(40, 1, 40), (200, 5, 196), (256, 6, 251)] {
            let ordered: Vec<f64> = (1..=draws).map(f64::from).collect();
            assert_eq!(order_statistic_interval_95(&ordered), (f64::from(lower), f64::from(upper)));
            // A further exchangeable draw lands in one of `draws + 1` equally
            // likely gaps; `upper - lower` of them lie inside the interval.
            assert!(f64::from(upper - lower) / f64::from(draws + 1) >= 0.95);
        }
    }

    #[test]
    #[allow(clippy::float_cmp, reason = "refusals compare as None")]
    fn pooled_interval_combines_completed_tables_by_rubin_rules() {
        // Two tables of a hundred draws at -1 and +1, each spread 0.5 about its mean.
        let draws: Vec<f64> = (0..200)
            .map(|i| {
                let table = if i < 100 { -1.0 } else { 1.0 };
                table + if i % 2 == 0 { -0.5 } else { 0.5 }
            })
            .collect();
        let within = 25.0 / 99.0;
        let expected =
            rubin_pool(&[-1.0, 1.0], &[within, within], Some(60.0)).unwrap().interval(0.95);
        let (lower, upper) = imputation_pooled_interval_95(&draws, 2, Some(60.0)).unwrap();
        assert!((lower - expected.0).abs() < 1e-12 && (upper - expected.1).abs() < 1e-12);
        // The between-table variance 2 enters at (1 + 1/2) 2 = 3, not at the
        // plain mixture's 2 / 2 = 1, and on a t reference with few degrees.
        assert!(lower < -1.96 * (within + 3.0).sqrt() && upper > 1.96 * (within + 3.0).sqrt());
        // Identical tables leave the within-table variance on the complete-data t.
        let same: Vec<f64> = (0..200).map(|i| if i % 2 == 0 { -0.5 } else { 0.5 }).collect();
        let (lower, upper) = imputation_pooled_interval_95(&same, 2, Some(60.0)).unwrap();
        let half = antecedent_stats::student_t_ppf(0.975, 60.0 * 61.0 / 63.0) * within.sqrt();
        assert!((lower + half).abs() < 1e-9 && (upper - half).abs() < 1e-9);
        assert_eq!(imputation_pooled_interval_95(&draws[..199], 2, None), None);
        assert_eq!(imputation_pooled_interval_95(&draws, 1, None), None);
    }

    #[test]
    fn pooled_selector_reaches_later_imputations() {
        let mut state = antecedent_core::execution::mix64(7 ^ 0x4452_4157_5f56_3201);
        let indices: Vec<_> = (0..40).map(|_| bootstrap_pool_index(&mut state, 200)).collect();
        assert!(indices.iter().any(|&index| index < 100));
        assert!(indices.iter().any(|&index| index >= 100));
    }

    use antecedent_contract::QuantityRole;
    use antecedent_core::graph_imputation_contract::GraphSupport;
    use antecedent_design::decision_contract::{
        ActionKind, DecisionAction, DecisionCriterion, StructuralPolicy, UtilityExpr,
    };

    #[test]
    #[allow(clippy::too_many_lines, reason = "one scenario walked from compile through replay")]
    fn missing_source_graph_mixture_transports_to_target_and_replays() {
        let mut treatment = Vec::new();
        let mut outcome = Vec::new();
        let mut covariate = Vec::new();
        for row in 0..80 {
            let x = f64::from(u8::from(row % 2 == 0));
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
                domains: Arc::<[u8]>::from(vec![1, 0, 1]),
                posterior_draws: 100,
                prior_scale: 1.0,
                execution_seed: 19,
            },
            &ctx,
        )
        .unwrap();
        let source_effect = producer.execute(&ctx).unwrap().mean;
        let target = TransportStandardization {
            columns: vec![(2, (0..80).map(|row| f64::from(u8::from(row % 5 != 0))).collect())],
            selection_targets: vec![2],
            source_population: "source".into(),
            target_population: "target".into(),
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
        let bootstrap = compile_graph_imputation_transport_target_bootstrap_decision(
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
        let cluster = compile_graph_imputation_transport_cluster_bootstrap_decision(
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
        let cluster_bytes = cluster.export(&cluster_result, "cluster-target", &ctx).unwrap();
        let cluster_replay = consume_graph_imputation_transport_decision(
            &cluster_bytes,
            cluster.lowered().identity(),
            &ctx,
        )
        .unwrap();
        assert_eq!(cluster_result.producer.receipt, cluster_replay.producer.receipt);
        let mut cluster_wire: GraphImputationTransportDecisionWire =
            antecedent_io::from_cbor(&cluster_bytes).unwrap();
        cluster_wire.cluster_bootstrap.as_mut().unwrap().cluster_ids[0].push('x');
        assert!(
            consume_graph_imputation_transport_decision(
                &antecedent_io::to_cbor(&cluster_wire).unwrap(),
                cluster.lowered().identity(),
                &ctx
            )
            .is_err()
        );
        let confounded_table = TabularData::from_f64_columns(
            parts
                .names
                .iter()
                .zip(&parts.columns)
                .map(|(name, column)| (name.as_str(), column.as_slice())),
        )
        .unwrap();
        let confounded_producer = CompiledGraphImputation::compile(
            confounded_table,
            &GraphImputationSpec {
                treatment: VariableId::from_raw(parts.treatment),
                outcome: VariableId::from_raw(parts.outcome),
                support: GraphSupport::PreTreatmentCovariates,
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
        let mut confounded_target = target.clone();
        confounded_target.source_sample = confounded_producer.source_identity().to_hex();
        let confounded = compile_graph_imputation_confounded_transport_decision(
            confounded_producer,
            confounded_target,
            quantity.clone(),
            decision.clone(),
        )
        .unwrap();
        let confounded_bootstrap_table = TabularData::from_f64_columns(
            parts
                .names
                .iter()
                .zip(&parts.columns)
                .map(|(name, column)| (name.as_str(), column.as_slice())),
        )
        .unwrap();
        let confounded_bootstrap_producer = CompiledGraphImputation::compile(
            confounded_bootstrap_table,
            &GraphImputationSpec {
                treatment: VariableId::from_raw(parts.treatment),
                outcome: VariableId::from_raw(parts.outcome),
                support: GraphSupport::PreTreatmentCovariates,
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
        let mut confounded_bootstrap_target = target.clone();
        confounded_bootstrap_target.source_sample =
            confounded_bootstrap_producer.source_identity().to_hex();
        let confounded_bootstrap =
            compile_graph_imputation_confounded_transport_target_bootstrap_decision(
                confounded_bootstrap_producer,
                confounded_bootstrap_target,
                quantity.clone(),
                decision.clone(),
                TargetFrameBootstrap { replicates: 40, seed: 11 },
            )
            .unwrap();
        let confounded_cluster_table = TabularData::from_f64_columns(
            parts
                .names
                .iter()
                .zip(&parts.columns)
                .map(|(name, column)| (name.as_str(), column.as_slice())),
        )
        .unwrap();
        let confounded_cluster_producer = CompiledGraphImputation::compile(
            confounded_cluster_table,
            &GraphImputationSpec {
                treatment: VariableId::from_raw(parts.treatment),
                outcome: VariableId::from_raw(parts.outcome),
                support: GraphSupport::PreTreatmentCovariates,
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
        let mut confounded_cluster_target = target.clone();
        confounded_cluster_target.source_sample =
            confounded_cluster_producer.source_identity().to_hex();
        let confounded_cluster = compile_graph_imputation_transport_cluster_bootstrap_decision(
            confounded_cluster_producer,
            confounded_cluster_target,
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
        let confounded_cluster_result = confounded_cluster.execute(&ctx).unwrap();
        let confounded_cluster_bytes = confounded_cluster
            .export(&confounded_cluster_result, "confounded-cluster", &ctx)
            .unwrap();
        let confounded_cluster_replay = consume_graph_imputation_transport_decision(
            &confounded_cluster_bytes,
            confounded_cluster.lowered().identity(),
            &ctx,
        )
        .unwrap();
        assert_eq!(
            confounded_cluster_result.producer.receipt,
            confounded_cluster_replay.producer.receipt
        );
        let no_overlap_table = TabularData::from_f64_columns([
            ("t", covariate.as_slice()),
            ("y", outcome.as_slice()),
            ("x", covariate.as_slice()),
        ])
        .unwrap();
        let no_overlap_producer = CompiledGraphImputation::compile(
            no_overlap_table,
            &GraphImputationSpec {
                treatment: VariableId::from_raw(0),
                outcome: VariableId::from_raw(1),
                support: GraphSupport::PreTreatmentCovariates,
                imputation: ImputationSpec::new()
                    .with_imputations(2)
                    .with_iterations(2)
                    .with_seed(17),
                domains: Arc::<[u8]>::from(vec![1, 0, 1]),
                posterior_draws: 100,
                prior_scale: 1.0,
                execution_seed: 19,
            },
            &ctx,
        )
        .unwrap();
        let mut no_overlap_target = target.clone();
        no_overlap_target.source_sample = no_overlap_producer.source_identity().to_hex();
        let no_overlap = compile_graph_imputation_confounded_transport_decision(
            no_overlap_producer,
            no_overlap_target,
            quantity.clone(),
            decision.clone(),
        )
        .unwrap();
        assert!(no_overlap.execute(&ctx).is_err());
        let plan =
            compile_graph_imputation_transport_decision(producer, target, quantity, decision)
                .unwrap();
        assert_eq!(plan.plan_view().route().as_str(), "GraphImputationTransportDecision");
        let result = plan.execute(&ctx).unwrap();
        assert!(result.producer.mean.is_finite());
        assert!(
            result.producer.mean > source_effect + 0.05,
            "target standardization must respond to the x shift"
        );
        assert_eq!(result.calibration, "not_measured");
        let bytes = plan.export(&result, "source-target-graph-decision", &ctx).unwrap();
        let replay = consume_graph_imputation_transport_decision(
            &bytes,
            result.lowered_program_identity,
            &ctx,
        )
        .unwrap();
        assert_eq!(replay.producer.receipt, result.producer.receipt);
        assert_eq!(replay.decision.decision, result.decision.decision);
        assert_ne!(confounded.lowered().identity(), plan.lowered().identity());
        let confounded_candidate = confounded.calibration_candidate(&ctx).unwrap();
        assert_eq!(confounded_candidate.result.calibration, "not_measured");
        let bytes =
            confounded.export(&confounded_candidate.result, "confounded-target", &ctx).unwrap();
        let confounded_replay = consume_graph_imputation_transport_decision(
            &bytes,
            confounded_candidate.result.lowered_program_identity,
            &ctx,
        )
        .unwrap();
        assert_eq!(confounded_candidate.result.producer.draws, confounded_replay.producer.draws);
        assert_eq!(
            confounded_candidate.result.decision.decision,
            confounded_replay.decision.decision
        );
        assert_ne!(confounded_bootstrap.lowered().identity(), confounded.lowered().identity());
        assert_ne!(confounded_bootstrap.lowered().identity(), bootstrap.lowered().identity());
        let confounded_sampled = confounded_bootstrap.calibration_candidate(&ctx).unwrap();
        assert_eq!(confounded_sampled.result.producer.draws.len(), 40);
        assert_eq!(confounded_sampled.result.calibration, "not_measured");
        assert_eq!(confounded_sampled.basis.estimator.as_ref(), CONFOUNDED_BOOTSTRAP_FEATURE);
        let bytes = confounded_bootstrap
            .export(&confounded_sampled.result, "confounded-bootstrap", &ctx)
            .unwrap();
        let replay = consume_graph_imputation_transport_decision(
            &bytes,
            confounded_sampled.result.lowered_program_identity,
            &ctx,
        )
        .unwrap();
        assert_eq!(replay.producer.draws, confounded_sampled.result.producer.draws);
        assert_eq!(replay.producer.receipt, confounded_sampled.result.producer.receipt);
        assert_eq!(replay.decision.decision, confounded_sampled.result.decision.decision);
        let mut wire: GraphImputationTransportDecisionWire =
            antecedent_io::from_cbor(&bytes).unwrap();
        wire.target_bootstrap.as_mut().unwrap().seed ^= 1;
        let tampered = antecedent_io::to_cbor(&wire).unwrap();
        assert!(
            consume_graph_imputation_transport_decision(
                &tampered,
                confounded_sampled.result.lowered_program_identity,
                &ctx,
            )
            .is_err()
        );
        assert_ne!(bootstrap.lowered().identity(), plan.lowered().identity());
        let candidate = bootstrap.calibration_candidate(&ctx).unwrap();
        assert_eq!(candidate.result.producer.draws.len(), 40);
        assert_eq!(candidate.result.calibration, "not_measured");
        assert!(candidate.interval.0 <= candidate.interval.1);
        let bytes = bootstrap.export(&candidate.result, "binary-target-bootstrap", &ctx).unwrap();
        let replay = consume_graph_imputation_transport_decision(
            &bytes,
            candidate.result.lowered_program_identity,
            &ctx,
        )
        .unwrap();
        assert_eq!(candidate.result.producer.draws, replay.producer.draws);
        assert_eq!(candidate.result.producer.receipt, replay.producer.receipt);
        assert_eq!(candidate.result.decision.decision, replay.decision.decision);
        let mut wire: GraphImputationTransportDecisionWire =
            antecedent_io::from_cbor(&bytes).unwrap();
        wire.target_bootstrap.as_mut().unwrap().seed ^= 1;
        let changed = antecedent_io::to_cbor(&wire).unwrap();
        assert!(
            consume_graph_imputation_transport_decision(
                &changed,
                candidate.result.lowered_program_identity,
                &ctx,
            )
            .is_err()
        );
    }
}
