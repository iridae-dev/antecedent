//! Read-only inspection records and accessors for a prepared handle.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    CachedTemporalIdentification, CheckedAipwOperation, CheckedBayesianGcompOperation,
    CheckedFunctionalEffectResponseMember, CheckedIvOperation, CheckedLinearOperation,
    CheckedProgramBinding, ExecutionKind, PreparedExecution, PreparedStudy,
};
use crate::accepted::GraphClass;
use crate::analysis::builder::{DataInput, RefuteSuite};
use crate::analysis::checked_propensity::{
    CheckedPropensityProcedure, CheckedPropensityUncertainty,
};
use crate::analysis::execute::Study;
use crate::analysis::route_guards::inference_label;
use crate::analysis::{ConditionalProcedure, StaticClassIdentification};
use crate::error::CausalError;
use crate::inference::InferenceMode;
use crate::planner::PhysicalExecutionPlan;
use crate::strategy_table::{EstimatorId, IdentifierId};
use antecedent_core::{
    AverageEffectQuery, CausalQuery, CausalSchema, Intervention, MediationQuery, ResponseQuery,
    TargetPopulation, TemporalEffectQuery, VariableId,
};
use antecedent_data::TabularData;
use antecedent_estimate::ScoreTable;
use antecedent_expr::IdentifiedEstimand;
use antecedent_identify::IdentificationStatus;
use std::sync::Arc;

/// Read-only inspection of a retained checked derivative-response plan.
#[derive(Clone, Debug, PartialEq)]
pub struct CheckedDerivativeResponseInfo {
    /// Exact response functional and evaluation coordinates.
    pub query: ResponseQuery,
    /// Identifier fixed during preparation.
    pub identifier: crate::strategy_table::IdentifierId,
    /// Estimator fixed during preparation.
    pub estimator: crate::strategy_table::EstimatorId,
    /// Adjustment variables retained by the checked target.
    pub adjustment_set: Arc<[antecedent_core::VariableId]>,
    /// Maximum derivative coordinates the operation may materialize.
    pub max_derivative_cells: usize,
}

/// Read-only inspection of a retained static DAG response operation.
#[derive(Clone, Debug, PartialEq)]
pub struct CheckedStaticDagResponseInfo {
    /// Exact mean-curve or intervention query, including its member values.
    pub query: ResponseQuery,
    /// Identifier fixed during preparation.
    pub identifier: crate::strategy_table::IdentifierId,
    /// Estimator fixed during preparation.
    pub estimator: crate::strategy_table::EstimatorId,
    /// Validation suite fixed during preparation.
    pub validation: RefuteSuite,
    /// Ordered curve grid. Empty for intervention-response queries.
    pub grid_members: Arc<[f64]>,
    /// Inference procedure label fixed during preparation
    /// (`frequentist` or `bayesian:<backend>`).
    pub inference: Arc<str>,
}

/// Inspect the sealed joint-cell AIPW response route retained by preparation.
#[derive(Clone, Debug)]
pub struct CheckedCellAipwResponseInfo {
    /// The complete intervention-response query.
    pub query: ResponseQuery,
    /// The checked identifier selected at preparation.
    pub identifier: IdentifierId,
    /// The checked estimator selected at preparation.
    pub estimator: EstimatorId,
    /// Refutation suite carried by the operation.
    pub validation: RefuteSuite,
    /// Whether the graph was explicit or accepted.
    pub origin: crate::analysis::execute::DagResponseOrigin,
    /// Binary joint-cell index evaluated by this route.
    pub requested_arm: u32,
    /// Covariate roles retained for adjustment.
    pub adjustment_set: Arc<[VariableId]>,
    /// Tier interpretation when the cell was sealed on a tier closure rather
    /// than a plain DAG.
    pub within_tier: Option<antecedent_graph::WithinTier>,
}

/// Read-only target, graph, and procedure of a checked static mediation plan.
#[derive(Clone, Debug)]
pub struct CheckedStaticMediationInfo {
    /// Complete contrast, mediators, intervention levels, and population.
    pub query: MediationQuery,
    /// Prepare-time identifier.
    pub identifier: crate::strategy_table::IdentifierId,
    /// Prepare-time estimator.
    pub estimator: crate::strategy_table::EstimatorId,
    /// Identified adjustment roles used by the model.
    pub adjustment_set: Arc<[antecedent_core::VariableId]>,
    /// Source and executable functional root identifiers.
    pub functional_roots: (u32, u32),
    /// Canonical supplied DAG edges.
    pub graph_edges: Arc<[(u32, u32)]>,
    /// Bootstrap replicate budget, including zero for point-only execution.
    pub bootstrap_replicates: u32,
    /// Prepare-time validation suite.
    pub validation: RefuteSuite,
}

/// Read-only target and procedure retained by a checked GCM attribution plan.
#[derive(Clone, Debug)]
pub struct CheckedAttributionInfo {
    /// Exact anomaly or distribution-change target.
    pub query: CausalQuery,
    /// Prepare-time identifier and estimator.
    pub identifier: crate::strategy_table::IdentifierId,
    /// Prepare-time estimator.
    pub estimator: crate::strategy_table::EstimatorId,
    /// Canonical supplied DAG edges.
    pub graph_edges: Arc<[(u32, u32)]>,
    /// Identification status fixed at preparation.
    pub identification_status: antecedent_identify::IdentificationStatus,
}

/// Read-only inspection of a retained TemporalDag response operation.
#[derive(Clone, Debug, PartialEq)]
pub struct CheckedTemporalDagResponseInfo {
    /// Exact response functional, policy, and requested horizons.
    pub query: ResponseQuery,
    /// Identifier fixed during preparation.
    pub identifier: crate::strategy_table::IdentifierId,
    /// Estimator fixed during preparation.
    pub estimator: crate::strategy_table::EstimatorId,
    /// Interval procedure and replicate count.
    pub uncertainty: Arc<str>,
    /// Ordered response grid.
    pub grid_members: Arc<[f64]>,
    /// Ordered horizon members.
    pub horizons: Arc<[u32]>,
}

/// Read-only source graph, completion proof, and procedure of a retained
/// TemporalCpdag/TemporalPag response.
#[derive(Clone, Debug)]
pub struct CheckedTemporalClassResponseInfo {
    /// Exact response functional, policy, and requested horizons.
    pub query: ResponseQuery,
    /// Source class retained with the completion proof.
    pub graph_class: GraphClass,
    /// Identifier fixed during preparation.
    pub identifier: crate::strategy_table::IdentifierId,
    /// Estimator fixed during preparation.
    pub estimator: crate::strategy_table::EstimatorId,
    /// Inference mode label fixed during preparation.
    pub inference: Arc<str>,
    /// Validation suite fixed during preparation.
    pub validation: RefuteSuite,
    /// Circular-block bootstrap budget fixed during preparation.
    pub bootstrap_replicates: u32,
    /// Ordered requested horizons, each with a retained completion envelope.
    pub horizons: Arc<[u32]>,
    /// Number of completion atoms retained for the first requested horizon.
    pub completion_count: usize,
    /// Maximum number of class completions searched, when configured.
    pub completion_limit: Option<usize>,
    /// Identified enumeration mass retained for the first requested horizon.
    pub identified_mass: f64,
    /// Unidentified enumeration mass retained for the first requested horizon.
    pub unidentified_mass: f64,
    /// Source graph version sealed with the proof.
    pub graph_version: u64,
}

/// Frozen temporal response posterior: atom keys, weights, identification
/// flags, and the selected inference and validation. The retained fields are
/// those of [`CheckedGraphPosteriorResponseInfo`].
pub type CheckedTemporalGraphPosteriorResponseInfo = CheckedGraphPosteriorResponseInfo;

/// Read-only target, proof, and procedure of a prepared TemporalDag effect.
#[derive(Clone, Debug)]
pub struct CheckedTemporalDagEffectInfo {
    /// Exact temporal policy, intervention levels, horizon, and population.
    pub query: TemporalEffectQuery,
    /// Identifier fixed during preparation.
    pub identifier: crate::strategy_table::IdentifierId,
    /// Estimator fixed during preparation.
    pub estimator: crate::strategy_table::EstimatorId,
    /// Validation suite fixed during preparation.
    pub validation: RefuteSuite,
    /// Circular-block bootstrap budget fixed during preparation.
    pub bootstrap_replicates: u32,
    /// Graph-derived adjustment variables in the unfolded proof.
    pub adjustment_set: Arc<[antecedent_core::TemporalNodeKey]>,
    /// Canonical node and edge signature of the prepared temporal graph.
    pub graph_signature: Arc<str>,
}

/// Read-only target and procedure of a sealed Bayesian TemporalDag effect.
#[derive(Clone, Debug)]
pub struct CheckedBayesianTemporalDagEffectInfo {
    /// Exact pulse or one-step sustained target.
    pub query: TemporalEffectQuery,
    /// Identifier fixed during preparation.
    pub identifier: crate::strategy_table::IdentifierId,
    /// Estimator fixed during preparation.
    pub estimator: crate::strategy_table::EstimatorId,
    /// Validation suite fixed during preparation.
    pub validation: RefuteSuite,
    /// Requested posterior draws.
    pub posterior_draws: usize,
    /// Signature of the retained TemporalDag.
    pub graph_signature: Arc<str>,
}

/// Read-only target and procedure for a checked temporal mediation family.
#[derive(Clone, Debug)]
pub struct CheckedTemporalMediationInfo {
    /// Exact mediation target and requested horizon family.
    pub query: MediationQuery,
    /// Signature of the fixed TemporalDag.
    pub graph_signature: Arc<str>,
    /// Validation suite selected at preparation.
    pub validation: RefuteSuite,
    /// Horizon members retained in order.
    pub horizons: Arc<[u32]>,
    /// Estimator family fixed at preparation.
    pub estimator: crate::strategy_table::EstimatorId,
    /// Graph-derived adjustment node keys retained at each horizon.
    pub adjustment_sets: Arc<[(u32, Arc<[antecedent_core::TemporalNodeKey]>)]>,
}

/// Read-only target, structural proof, and procedure for a checked
/// TemporalCpdag class-envelope or temporal graph-posterior mediation family.
#[derive(Clone, Debug)]
pub struct CheckedTemporalClassMediationInfo {
    /// Exact mediation target and requested horizon family.
    pub query: MediationQuery,
    /// Class of the fixed source graph or of the posterior's atoms.
    pub graph_class: GraphClass,
    /// How the structure was supplied: explicit, accepted, or a graph posterior.
    pub structure_source: crate::support::StructureSource,
    /// Identifier recorded by the checked physical plan.
    pub identifier: Arc<str>,
    /// Estimator family fixed at preparation.
    pub estimator: crate::strategy_table::EstimatorId,
    /// Validation suite selected at preparation.
    pub validation: RefuteSuite,
    /// Circular-block bootstrap budget fixed at preparation.
    pub bootstrap_replicates: u32,
    /// Horizon members retained in order.
    pub horizons: Arc<[u32]>,
    /// Completion atoms (fixed class) or posterior atoms (graph posterior)
    /// retained by the proof.
    pub atom_count: usize,
    /// Enumeration or posterior mass that identified.
    pub identified_mass: f64,
    /// Enumeration or posterior mass that stays unidentified.
    pub unidentified_mass: f64,
    /// Signature of the retained source graph or frozen posterior.
    pub proof_signature: Arc<str>,
}

/// Read-only target and fixed design for checked interference execution.
#[derive(Clone, Debug)]
pub struct CheckedInterferenceInfo {
    /// Exposure contrast, assignment mechanism, and outcome.
    pub query: antecedent_core::InterferenceQuery,
    /// Identifier selected during preparation.
    pub identifier: crate::strategy_table::IdentifierId,
    /// Estimator selected during preparation.
    pub estimator: crate::strategy_table::EstimatorId,
    /// Number of units in the fixed network.
    pub unit_count: usize,
    /// Number of directed network edges.
    pub network_edge_count: usize,
}

/// Read-only source graph, completion proof, and procedure for a temporal
/// Cpdag/Pag pulse or sustained effect.
#[derive(Clone, Debug)]
pub struct CheckedTemporalClassEffectInfo {
    /// Exact policy, intervention levels, horizon, and target population.
    pub query: TemporalEffectQuery,
    /// Source class retained with the completion proof.
    pub graph_class: GraphClass,
    /// Identifier fixed during preparation.
    pub identifier: crate::strategy_table::IdentifierId,
    /// Estimator fixed during preparation.
    pub estimator: crate::strategy_table::EstimatorId,
    /// Validation suite fixed during preparation.
    pub validation: RefuteSuite,
    /// Circular-block bootstrap budget fixed during preparation.
    pub bootstrap_replicates: u32,
    /// Number of completion atoms retained by the checked proof.
    pub completion_count: usize,
    /// Maximum number of class completions searched, when configured.
    pub completion_limit: Option<usize>,
    /// Identified and unresolved enumeration mass retained by the proof.
    pub identified_mass: f64,
    /// Unidentified enumeration mass retained by the proof.
    pub unidentified_mass: f64,
    /// Number of completions whose enumeration was cut off.
    pub truncated_completions: usize,
    /// Source graph version sealed with the proof.
    pub graph_version: u64,
}

/// Read-only target and uncertainty binding for a prepared propensity route.
#[derive(Clone, Debug)]
pub struct CheckedPropensityInfo {
    /// The selected weighting or matching estimator.
    pub estimator: crate::strategy_table::EstimatorId,
    /// Adjustment variables in the checked target.
    pub adjustment_set: Arc<[antecedent_core::VariableId]>,
    /// Population targeted by the bound design.
    pub population: TargetPopulation,
    /// Source rows retained by the propensity preparation.
    pub source_rows: Arc<[u32]>,
    /// Named uncertainty procedure.
    pub uncertainty: Arc<str>,
    /// Bootstrap count only when the weighting procedure can use it.
    pub bootstrap_replicates: Option<u32>,
}

/// Read-only checked target and procedure for a frequentist static conditional effect.
#[derive(Clone, Debug)]
pub struct CheckedConditionalEffectInfo {
    /// Exact conditional query, including modifiers, arms, and outcome functional.
    pub query: antecedent_core::ConditionalEffectQuery,
    /// Source graph class retained with the identification proof.
    pub graph_class: GraphClass,
    /// Selected identification procedure.
    pub identifier: crate::strategy_table::IdentifierId,
    /// Selected estimator identity.
    pub estimator: crate::strategy_table::EstimatorId,
    /// Actual scalar or distribution score procedure.
    pub procedure: Arc<str>,
    /// Invariant adjustment variables retained by the checked proof.
    pub adjustment_set: Arc<[antecedent_core::VariableId]>,
    /// Number of class completions checked, or `None` for a fixed DAG.
    pub completion_count: Option<usize>,
    /// Identified completion mass for a class proof.
    pub identified_mass: Option<f64>,
    /// Unresolved completion mass for a class proof.
    pub unresolved_mass: Option<f64>,
    /// Prepared refutation suite.
    pub validation: RefuteSuite,
    /// Variables in the physical design, retaining their semantic IDs.
    pub design_roles: Arc<[antecedent_core::VariableId]>,
    /// Complete source row indices at preparation or latest refresh.
    pub source_rows: Arc<[u32]>,
}

/// Read-only checked target and procedure for a static CPDAG/PAG average effect.
#[derive(Clone, Debug)]
pub struct CheckedStaticClassEffectInfo {
    /// Exact average-effect target sealed at preparation.
    pub query: AverageEffectQuery,
    /// Source graph class retained with the completion envelope.
    pub graph_class: GraphClass,
    /// Selected identification procedure.
    pub identifier: crate::strategy_table::IdentifierId,
    /// Selected estimator identity.
    pub estimator: crate::strategy_table::EstimatorId,
    /// Inference settings (including any Bayesian prior and backend) retained by preparation.
    pub inference: InferenceMode,
    /// Prepared refutation suite.
    pub validation: RefuteSuite,
    /// Number of class completions in the retained envelope.
    pub completion_count: usize,
    /// Identified completion mass in the retained envelope.
    pub identified_mass: f64,
    /// Unresolved completion mass in the retained envelope.
    pub unresolved_mass: f64,
}

/// Read-only checked target and procedure for a Bayesian static class conditional effect.
#[derive(Clone, Debug)]
pub struct CheckedBayesianClassConditionalInfo {
    /// Exact conditional query, including modifiers and arms.
    pub query: antecedent_core::ConditionalEffectQuery,
    /// Source graph class retained with the completion proof.
    pub graph_class: GraphClass,
    /// Selected identification procedure.
    pub identifier: crate::strategy_table::IdentifierId,
    /// Selected estimator identity.
    pub estimator: crate::strategy_table::EstimatorId,
    /// Bayesian prior and inference settings retained by preparation.
    pub inference: InferenceMode,
    /// Prepared refutation suite.
    pub validation: RefuteSuite,
    /// Invariant adjustment variables shared by every completion.
    pub adjustment_set: Arc<[antecedent_core::VariableId]>,
    /// Number of class completions checked.
    pub completion_count: usize,
    /// Identified completion mass for the class proof.
    pub identified_mass: f64,
    /// Unresolved completion mass for the class proof.
    pub unresolved_mass: f64,
}

/// Read-only graph atoms, frozen weights and procedure retained for a
/// frequentist DAG graph-posterior effect.
#[derive(Clone, Debug, PartialEq)]
pub struct CheckedGraphPosteriorEffectInfo {
    /// Average-effect query every posterior atom was identified for.
    pub query: AverageEffectQuery,
    /// Whether the sealed click estimates a conditional effect over the atoms.
    pub conditional: bool,
    /// Static estimator used for every identified atom.
    pub estimator: crate::strategy_table::EstimatorId,
    /// Validation suite applied to the atom-wise estimates.
    pub validation: RefuteSuite,
    /// Opaque graph identity per posterior sample.
    pub graph_keys: Arc<[u64]>,
    /// Frozen posterior weight per graph sample.
    pub weights: Arc<[f64]>,
    /// Identification status per graph sample, preserving unknown graph mass.
    pub identified: Arc<[antecedent_prob::GraphIdentFlag]>,
    /// Procedure-specific bootstrap count retained by preparation.
    pub bootstrap_replicates: u32,
}

/// Read-only CPDAG/PAG posterior samples and completion envelopes retained for
/// a frequentist average effect.
#[derive(Clone, Debug, PartialEq)]
pub struct CheckedClassGraphPosteriorEffectInfo {
    /// Average-effect query every posterior atom was identified for.
    pub query: AverageEffectQuery,
    /// Whether the sealed click estimates a conditional effect over the atoms.
    pub conditional: bool,
    /// Inference label frozen at preparation (`frequentist` or `bayesian:<backend>`).
    pub inference: Arc<str>,
    /// Static estimator used inside each identified completion.
    pub estimator: crate::strategy_table::EstimatorId,
    /// Validation suite applied to the atom-wise estimates.
    pub validation: RefuteSuite,
    /// Posterior sample key, including repeated and unidentified samples.
    pub graph_keys: Arc<[u64]>,
    /// Frozen posterior sample weights.
    pub weights: Arc<[f64]>,
    /// Sample-level identification flags, preserving unidentified mass.
    pub identified: Arc<[antecedent_prob::GraphIdentFlag]>,
    /// Number of identified CPDAG/PAG completion envelopes retained.
    pub class_atom_count: usize,
    /// Procedure-specific bootstrap count.
    pub bootstrap_replicates: u32,
}

/// Read-only target and atom identities retained for Bayesian DAG graph-posterior ATE.
#[derive(Clone, Debug, PartialEq)]
pub struct CheckedBayesianGraphPosteriorAteInfo {
    /// Average-effect query every posterior atom was identified for.
    pub query: AverageEffectQuery,
    /// Whether the sealed click estimates a conditional effect over the atoms.
    pub conditional: bool,
    /// Bayesian estimator applied to each identified atom.
    pub estimator: crate::strategy_table::EstimatorId,
    /// Validation suite frozen at preparation.
    pub validation: RefuteSuite,
    /// Frozen graph identity per posterior sample.
    pub graph_keys: Arc<[u64]>,
    /// Posterior weight per graph sample.
    pub weights: Arc<[f64]>,
    /// Identification state per sample, retaining failed atom mass.
    pub identified: Arc<[antecedent_prob::GraphIdentFlag]>,
    /// Backend label for the frozen Bayesian configuration.
    pub inference: Arc<str>,
}

/// Frozen ADMG response posterior, including atom weights and identification flags.
#[derive(Clone, Debug, PartialEq)]
pub struct CheckedAdmgGraphPosteriorResponseInfo {
    /// Response target retained at preparation.
    pub query: antecedent_core::ResponseQuery,
    /// Frozen graph identity per posterior atom.
    pub graph_keys: Arc<[u64]>,
    /// Frozen posterior weight per atom.
    pub weights: Arc<[f64]>,
    /// Identified and unidentified atom status, preserving unresolved mass.
    pub identified: Arc<[antecedent_prob::GraphIdentFlag]>,
    /// Validation suite fixed at preparation.
    pub validation: RefuteSuite,
}

/// Frozen explicit/accepted CPDAG or PAG response route.
#[derive(Clone, Debug, PartialEq)]
pub struct CheckedStaticClassResponseInfo {
    /// Response target retained at preparation.
    pub query: antecedent_core::ResponseQuery,
    /// Graph class the completion envelope was enumerated for.
    pub graph_class: GraphClass,
    /// Identifier fixed at preparation.
    pub identifier: crate::strategy_table::IdentifierId,
    /// Estimator fixed at preparation.
    pub estimator: crate::strategy_table::EstimatorId,
    /// Inference procedure label (`frequentist` or `bayesian:<backend>`).
    pub inference: Arc<str>,
    /// Validation suite fixed at preparation.
    pub validation: RefuteSuite,
    /// Completions retained in the frozen envelope.
    pub completion_count: usize,
    /// Identified completion mass retained in the frozen envelope.
    pub identified_mass: f64,
    /// Unidentified completion mass retained in the frozen envelope.
    pub unidentified_mass: f64,
}

/// Frozen static or temporal graph-posterior response: atom keys, weights,
/// identification flags, and the selected procedure, inference and validation.
#[derive(Clone, Debug, PartialEq)]
pub struct CheckedGraphPosteriorResponseInfo {
    /// Response target retained at preparation.
    pub query: antecedent_core::ResponseQuery,
    /// Atom kind of the frozen posterior.
    pub atom_kind: antecedent_discovery::GraphPosteriorAtomKind,
    /// Identifier fixed at preparation.
    pub identifier: crate::strategy_table::IdentifierId,
    /// Estimator fixed at preparation.
    pub estimator: crate::strategy_table::EstimatorId,
    /// Inference procedure label (`frequentist` or `bayesian:<backend>`).
    pub inference: Arc<str>,
    /// Frozen graph identity per posterior atom.
    pub graph_keys: Arc<[u64]>,
    /// Frozen posterior weight per atom.
    pub weights: Arc<[f64]>,
    /// Identified and unidentified atom status, preserving unresolved mass.
    pub identified: Arc<[antecedent_prob::GraphIdentFlag]>,
    /// Validation suite fixed at preparation.
    pub validation: RefuteSuite,
}

/// Read-only source graph, completion proof, and Bayesian model settings for a
/// temporal Cpdag/Pag pulse or sustained effect.
#[derive(Clone, Debug)]
pub struct CheckedBayesianTemporalClassEffectInfo {
    /// Exact policy, intervention levels, horizon, and target population.
    pub query: TemporalEffectQuery,
    /// Source class retained with the completion proof.
    pub graph_class: GraphClass,
    /// Identifier fixed during preparation.
    pub identifier: crate::strategy_table::IdentifierId,
    /// Estimator fixed during preparation.
    pub estimator: crate::strategy_table::EstimatorId,
    /// Validation suite fixed during preparation.
    pub validation: RefuteSuite,
    /// Requested posterior draws.
    pub posterior_draws: usize,
    /// Whether a caller-supplied class mass was sealed with the operation.
    pub class_prior_supplied: bool,
    /// Number of completion atoms retained by the checked proof.
    pub completion_count: usize,
    /// Maximum number of class completions searched, when configured.
    pub completion_limit: Option<usize>,
    /// Identified enumeration mass retained by the proof.
    pub identified_mass: f64,
    /// Unidentified enumeration mass retained by the proof.
    pub unidentified_mass: f64,
    /// Number of completions whose enumeration was cut off.
    pub truncated_completions: usize,
    /// Source graph version sealed with the proof.
    pub graph_version: u64,
}

/// Read-only frozen samples, per-atom proof counts, and procedure retained for
/// a pulse or sustained effect over a temporal graph posterior.
#[derive(Clone, Debug, PartialEq)]
pub struct CheckedTemporalGraphPosteriorEffectInfo {
    /// Exact policy, intervention levels, horizon, and target population.
    pub query: TemporalEffectQuery,
    /// Atom kind of the frozen posterior (TemporalDag, TemporalCpdag, or TemporalPag).
    pub atom_kind: antecedent_discovery::GraphPosteriorAtomKind,
    /// Identifier fixed during preparation.
    pub identifier: crate::strategy_table::IdentifierId,
    /// Estimator fixed during preparation.
    pub estimator: crate::strategy_table::EstimatorId,
    /// Validation suite fixed during preparation.
    pub validation: RefuteSuite,
    /// Inference label frozen at preparation.
    pub inference: Arc<str>,
    /// Posterior sample key, including repeated and unidentified samples.
    pub graph_keys: Arc<[u64]>,
    /// Frozen posterior sample weights.
    pub weights: Arc<[f64]>,
    /// Sample-level identification flags, preserving unidentified mass.
    pub identified: Arc<[antecedent_prob::GraphIdentFlag]>,
    /// Number of identified atoms carrying a retained proof.
    pub identified_atom_count: usize,
    /// Circular-block bootstrap budget fixed during preparation.
    pub bootstrap_replicates: u32,
}

/// Inspection receipt for the checked Bayesian quadratic-basis ATE route.
#[derive(Clone, Debug, PartialEq)]
pub struct CheckedBayesianBasisAteInfo {
    /// Target query retained by preparation.
    pub query: AverageEffectQuery,
    /// Back-door adjustment variables used by the basis model.
    pub adjustment_set: Arc<[antecedent_core::VariableId]>,
    /// Number of posterior draws used by execution.
    pub posterior_draws: usize,
    /// Isotropic coefficient-prior scale retained by preparation.
    pub prior_scale: f64,
}

/// Target and fixed row/fold bindings for the checked robust Bayesian ATE.
#[derive(Clone, Debug, PartialEq)]
pub struct CheckedBayesianRobustAteInfo {
    /// Mean binary ATE retained by preparation.
    pub query: AverageEffectQuery,
    /// Number of bootstrap pushforward draws requested.
    pub posterior_draws: usize,
    /// Back-door adjustment variables bound to the nuisance design.
    pub adjustment_set: Arc<[antecedent_core::VariableId]>,
    /// Stable row identities bound to the prepared fold plan.
    pub row_ids: Arc<[u64]>,
    /// Treatment-stratified fold identity in row order.
    pub fold_ids: Arc<[u16]>,
}

/// Inspection receipt for a prepared Bayesian DAG conditional-effect route.
#[derive(Clone, Debug)]
pub struct CheckedBayesianConditionalInfo {
    query: antecedent_core::ConditionalEffectQuery,
    inference: InferenceMode,
    validation: RefuteSuite,
    modifier_roles: Arc<[antecedent_core::VariableId]>,
    identification_status: IdentificationStatus,
    estimand: IdentifiedEstimand,
}

impl CheckedBayesianConditionalInfo {
    /// Conditional causal target retained by preparation.
    #[must_use]
    pub fn query(&self) -> &antecedent_core::ConditionalEffectQuery {
        &self.query
    }
    /// Bayesian prior and inference settings retained by preparation.
    #[must_use]
    pub fn inference(&self) -> &InferenceMode {
        &self.inference
    }
    /// Validation suite retained by preparation.
    #[must_use]
    pub fn validation(&self) -> RefuteSuite {
        self.validation
    }
    /// Semantic modifier variables retained by preparation.
    #[must_use]
    pub fn modifier_roles(&self) -> &[antecedent_core::VariableId] {
        &self.modifier_roles
    }
    /// Identification status retained with the target.
    #[must_use]
    pub fn identification_status(&self) -> IdentificationStatus {
        self.identification_status
    }
    /// Identified target functional and adjustment roles.
    #[must_use]
    pub fn estimand(&self) -> &IdentifiedEstimand {
        &self.estimand
    }
}

/// Public inspection of the retained two-scenario Unknown-tier ATE operation.
#[derive(Clone, Debug)]
pub struct CheckedUnknownTieredAverageInfo {
    /// Frozen average-effect target.
    pub query: AverageEffectQuery,
    /// Number of declared tiers.
    pub tier_count: usize,
    /// Canonical identified scenarios retained by the checked operation.
    pub scenario_count: usize,
    /// Selected identifier.
    pub identifier: crate::strategy_table::IdentifierId,
    /// Selected estimator.
    pub estimator: crate::strategy_table::EstimatorId,
    /// The prepared physical plan identity.
    pub plan_id: Arc<str>,
}

impl PreparedStudy {
    pub(crate) fn rebind_checked_bayesian_inference(
        &self,
        config: crate::BayesianConfig,
    ) -> Result<Self, CausalError> {
        let execution = match &self.execution {
            PreparedExecution::BayesianGcomp(operation) => {
                PreparedExecution::BayesianGcomp(operation.with_inference_config(config.clone())?)
            }
            PreparedExecution::BayesianBasisAte(operation) => PreparedExecution::BayesianBasisAte(
                operation.with_inference_config(config.clone())?,
            ),
            _ => {
                return Err(CausalError::Unsupported {
                    message: "retained Bayesian inference rebind requires its checked Gaussian operation",
                });
            }
        };
        let mut next = self.clone();
        next.execution = execution;
        let mut analysis = next.analysis.clone();
        analysis.inference = InferenceMode::Bayesian(config);
        next.replace_study(analysis);
        Ok(next)
    }

    pub(crate) fn checked_program_binding(&self) -> CheckedProgramBinding<'_> {
        self.execution.program_binding()
    }

    /// Retained GLM target design for the checked DAG adjustment route.
    #[must_use]
    pub fn checked_glm_adjustment(&self) -> Option<&antecedent_estimate::PreparedGlmProblem> {
        if let PreparedExecution::CheckedGlmAdjustment(operation) = &self.execution {
            Some(&operation.preparation)
        } else {
            None
        }
    }

    /// Retained local boundary and design for the checked sharp RD route.
    #[must_use]
    pub fn checked_rd_preparation(&self) -> Option<&antecedent_estimate::CheckedRdPreparation> {
        if let PreparedExecution::CheckedRd(operation) = &self.execution {
            Some(&operation.preparation)
        } else {
            None
        }
    }

    pub(crate) fn has_sealed_glm_operation(&self) -> bool {
        matches!(&self.execution, PreparedExecution::CheckedGlmAdjustment(op) if op.sealed_for_direct_execution())
    }

    pub(crate) fn has_checked_rd_operation(&self) -> bool {
        self.execution.kind() == ExecutionKind::CheckedRd
    }

    /// Whether this prepared handle owns the complete operation for a family
    /// whose one-shot facade must execute through preparation as well.
    pub(crate) fn has_complete_program_operation(&self, query: &CausalQuery) -> bool {
        matches!(
            (&self.execution, query),
            (PreparedExecution::Distribution(_), CausalQuery::Distribution(_))
                | (PreparedExecution::PathSpecificEffect(_), CausalQuery::PathSpecific(_))
                | (
                    PreparedExecution::NestedCounterfactual(_),
                    CausalQuery::NestedCounterfactual(_)
                )
        )
    }

    /// Retained checked shared-exogenous natural direct effect operation.
    #[must_use]
    pub fn checked_nested_counterfactual_operation(
        &self,
    ) -> Option<&crate::gcm::NestedCounterfactualOperation> {
        self.execution.nested_counterfactual()
    }

    /// Retained static linear fit inputs for portable result replay.
    pub(crate) fn checked_linear_operation(&self) -> Option<&CheckedLinearOperation> {
        self.execution.linear_operation()
    }

    /// Checked Bayesian g-computation receipt for a static mean ATE, when this
    /// prepared handle uses that licensed route.
    #[must_use]
    pub fn checked_bayesian_gcomp_operation(&self) -> Option<&CheckedBayesianGcompOperation> {
        self.execution.bayesian_gcomp()
    }

    /// Checked Bayesian quadratic-basis ATE plan, when this handle owns one.
    #[must_use]
    pub fn checked_bayesian_basis_ate_info(&self) -> Option<CheckedBayesianBasisAteInfo> {
        match &self.execution {
            PreparedExecution::BayesianBasisAte(operation) => Some(operation.info()),
            _ => None,
        }
    }

    /// Inspect the retained target for the checked Bayesian basis CATE route.
    #[must_use]
    pub fn checked_bayesian_basis_cate_query(
        &self,
    ) -> Option<&antecedent_core::ConditionalEffectQuery> {
        match &self.execution {
            PreparedExecution::BayesianBasisCate(operation) => Some(operation.query()),
            _ => None,
        }
    }

    /// Inspect the retained target and row/fold plan for Bayesian robust ATE.
    #[must_use]
    pub fn checked_bayesian_robust_ate_info(&self) -> Option<CheckedBayesianRobustAteInfo> {
        match &self.execution {
            PreparedExecution::BayesianRobustAte(operation) => Some(CheckedBayesianRobustAteInfo {
                query: operation.query().clone(),
                posterior_draws: operation.posterior_draws(),
                adjustment_set: Arc::from(operation.adjustment_set()),
                row_ids: Arc::from(operation.row_ids()),
                fold_ids: Arc::from(operation.fold_ids()),
            }),
            _ => None,
        }
    }

    /// Validation procedure frozen with the checked Bayesian DAG effect route.
    #[must_use]
    pub fn checked_bayesian_gcomp_validation(&self) -> Option<RefuteSuite> {
        match &self.execution {
            PreparedExecution::BayesianGcomp(operation) => Some(operation.validation()),
            _ => None,
        }
    }

    /// Checked Bayesian conditional-effect plan, when this handle owns one.
    #[must_use]
    pub fn checked_bayesian_conditional_operation(&self) -> Option<CheckedBayesianConditionalInfo> {
        let operation = self.execution.bayesian_conditional()?;
        Some(CheckedBayesianConditionalInfo {
            query: operation.query().clone(),
            inference: operation.inference().clone(),
            validation: operation.validation(),
            modifier_roles: Arc::from(operation.modifier_roles()),
            identification_status: operation.identification().status,
            estimand: operation.estimand().clone(),
        })
    }
    /// Checked linear adjustment lowering retained by this prepared handle.
    ///
    /// `None` means this route carries no checked adjustment lowering;
    /// it does not imply that another estimator is unsupported.
    #[must_use]
    pub fn checked_linear_adjustment(
        &self,
    ) -> Option<&antecedent_estimate::CheckedLinearAdjustmentAte> {
        self.execution.checked_linear()
    }

    /// Checked AIPW lowering retained for a supported mean ATE.
    #[must_use]
    pub fn checked_aipw_ate(&self) -> Option<&antecedent_estimate::CheckedAipwPreparation> {
        self.execution.checked_aipw()
    }

    pub(crate) fn has_sealed_aipw_operation(&self) -> bool {
        self.execution
            .aipw_operation()
            .is_some_and(CheckedAipwOperation::sealed_for_direct_execution)
    }

    pub(crate) fn has_checked_counterfactual_operation(&self) -> bool {
        self.execution.kind() == ExecutionKind::Counterfactual
    }

    /// The retained graph and cross-world target for a checked counterfactual route.
    #[must_use]
    pub fn checked_counterfactual_operation(
        &self,
    ) -> Option<&crate::gcm::CheckedCounterfactualOperation> {
        if let PreparedExecution::Counterfactual(plan) = &self.execution {
            Some(plan.operation())
        } else {
            None
        }
    }

    pub(crate) fn has_checked_derivative_response_operation(&self) -> bool {
        self.execution.kind() == ExecutionKind::DerivativeResponse
    }

    pub(crate) fn has_checked_static_dag_response_operation(&self) -> bool {
        self.execution.kind() == ExecutionKind::StaticDagResponse
    }

    /// Inspect the sealed static DAG mediation target and procedure.
    #[must_use]
    pub fn checked_static_mediation_info(&self) -> Option<CheckedStaticMediationInfo> {
        let (query, estimand, program, bootstrap_replicates, validation, graph) =
            self.execution.static_mediation()?.inspect();
        let edges = graph.directed_edge_pairs_sorted();
        Some(CheckedStaticMediationInfo {
            query: query.clone(),
            identifier: crate::strategy_table::IdentifierId::PathSpecificNatural,
            estimator: crate::strategy_table::EstimatorId::StaticMediationLinear,
            adjustment_set: Arc::clone(&estimand.adjustment_set),
            functional_roots: (program.mapping().source.raw(), program.mapping().executable.raw()),
            graph_edges: edges.into(),
            bootstrap_replicates,
            validation,
        })
    }

    /// Inspect the retained Bayesian DAG mediation target, model, and validation.
    #[must_use]
    pub fn checked_bayesian_static_mediation_info(&self) -> Option<CheckedStaticMediationInfo> {
        let operation = self.execution.bayesian_static_mediation()?;
        let (query, estimand, program, validation) = operation.inspect();
        let edges = operation.graph().directed_edge_pairs_sorted();
        Some(CheckedStaticMediationInfo {
            query: query.clone(),
            identifier: crate::strategy_table::IdentifierId::PathSpecificNatural,
            estimator: crate::strategy_table::EstimatorId::StaticMediationLinear,
            adjustment_set: Arc::clone(&estimand.adjustment_set),
            functional_roots: (program.mapping().source.raw(), program.mapping().executable.raw()),
            graph_edges: edges.into(),
            bootstrap_replicates: 0,
            validation,
        })
    }

    /// Inspect the sealed frequentist attribution target and supplied DAG.
    #[must_use]
    pub fn checked_attribution_info(&self) -> Option<CheckedAttributionInfo> {
        let operation = self.execution.attribution()?;
        let edges = operation.graph().directed_edge_pairs_sorted();
        let (identifier, estimator) = operation.procedure();
        Some(CheckedAttributionInfo {
            query: operation.query(),
            identifier: identifier.parse().ok()?,
            estimator: estimator.parse().ok()?,
            graph_edges: edges.into(),
            identification_status: operation.identification().status,
        })
    }

    pub(crate) fn has_checked_temporal_dag_response_operation(&self) -> bool {
        self.execution.kind() == ExecutionKind::TemporalDagResponse
    }

    pub(crate) fn has_checked_temporal_class_response_operation(&self) -> bool {
        self.execution.kind() == ExecutionKind::TemporalClassResponse
    }

    pub(crate) fn has_checked_temporal_graph_posterior_response_operation(&self) -> bool {
        self.execution.kind() == ExecutionKind::TemporalGraphPosteriorResponse
    }

    pub(crate) fn has_checked_temporal_dag_effect_operation(&self) -> bool {
        self.execution.kind() == ExecutionKind::TemporalDagEffect
    }

    pub(crate) fn has_checked_propensity_operation(&self) -> bool {
        self.execution.kind() == ExecutionKind::CheckedPropensity
    }

    /// Whether preparation retained the frequentist DAG graph-posterior effect plan.
    #[must_use]
    pub fn has_checked_graph_posterior_effect_operation(&self) -> bool {
        self.execution.kind() == ExecutionKind::GraphPosteriorEffect
    }

    /// Whether preparation retained a frequentist CPDAG/PAG graph-posterior ATE plan.
    #[must_use]
    pub fn has_checked_class_graph_posterior_effect_operation(&self) -> bool {
        self.execution.kind() == ExecutionKind::ClassGraphPosteriorEffect
    }

    /// Returns whether the static CPDAG/PAG average effect is bound to a
    /// completion envelope retained at preparation time.
    #[must_use]
    pub fn has_checked_static_class_effect_operation(&self) -> bool {
        self.execution.kind() == ExecutionKind::StaticClassEffect
    }

    /// Inspect the retained static CPDAG/PAG average-effect target, procedure,
    /// inference settings, and completion envelope masses.
    #[must_use]
    pub fn checked_static_class_effect_info(&self) -> Option<CheckedStaticClassEffectInfo> {
        let operation = self.execution.static_class_effect()?;
        let graph_class = match operation.identification() {
            StaticClassIdentification::Cpdag(_) => GraphClass::Cpdag,
            StaticClassIdentification::Pag(_) => GraphClass::Pag,
        };
        let (completion_count, identified_mass, unresolved_mass) =
            operation.identification().envelope_summary();
        Some(CheckedStaticClassEffectInfo {
            query: operation.query().clone(),
            graph_class,
            identifier: IdentifierId::GeneralizedAdjustment,
            estimator: operation.procedure().id(),
            inference: operation.inference().clone(),
            validation: operation.validation(),
            completion_count,
            identified_mass,
            unresolved_mass,
        })
    }

    /// Inspect the retained Bayesian class conditional target, prior, and completion proof.
    #[must_use]
    pub fn checked_bayesian_class_conditional_info(
        &self,
    ) -> Option<CheckedBayesianClassConditionalInfo> {
        let operation = self.execution.bayesian_class_conditional()?;
        let (completion_count, identified_mass, unresolved_mass) =
            operation.proof().completion_mass();
        Some(CheckedBayesianClassConditionalInfo {
            query: operation.query().clone(),
            graph_class: operation.proof().graph_class(),
            identifier: operation.identifier(),
            estimator: EstimatorId::BayesianConditional,
            inference: operation.inference().clone(),
            validation: operation.validation(),
            adjustment_set: Arc::clone(&operation.estimand().adjustment_set),
            completion_count,
            identified_mass,
            unresolved_mass,
        })
    }

    /// Whether preparation retained an ADMG graph-posterior response operation.
    #[must_use]
    pub fn has_checked_admg_graph_posterior_response_operation(&self) -> bool {
        self.execution.kind() == ExecutionKind::AdmgGraphPosteriorResponse
    }

    /// Inspect the frozen graph-posterior atoms and estimator procedure.
    #[must_use]
    pub fn checked_graph_posterior_effect_info(&self) -> Option<CheckedGraphPosteriorEffectInfo> {
        let operation = self.execution.graph_posterior_effect()?;
        let (graph_keys, weights, identified) = operation.atom_weights();
        Some(CheckedGraphPosteriorEffectInfo {
            query: operation.query().clone(),
            conditional: operation.target().is_conditional(),
            estimator: operation.estimator(),
            validation: operation.validation(),
            graph_keys: Arc::from(graph_keys),
            weights: Arc::from(weights),
            identified: Arc::from(identified),
            bootstrap_replicates: operation.bootstrap_replicates(),
        })
    }

    /// Inspect the frozen sample masses and completion envelopes for a class
    /// graph-posterior average effect.
    #[must_use]
    pub fn checked_class_graph_posterior_effect_info(
        &self,
    ) -> Option<CheckedClassGraphPosteriorEffectInfo> {
        let operation = self.execution.class_graph_posterior_effect()?;
        let (graph_keys, weights, identified) = operation.sample_mass();
        Some(CheckedClassGraphPosteriorEffectInfo {
            query: operation.query().clone(),
            conditional: operation.target().is_conditional(),
            inference: inference_label(operation.inference()),
            estimator: operation.procedure().id(),
            validation: operation.validation(),
            graph_keys: Arc::from(graph_keys),
            weights: Arc::from(weights),
            identified: Arc::from(identified),
            class_atom_count: operation.class_atoms().len(),
            bootstrap_replicates: operation.bootstrap_replicates(),
        })
    }

    /// Inspect the retained Bayesian DAG posterior target, atoms, and validation suite.
    #[must_use]
    pub fn checked_bayesian_graph_posterior_ate_info(
        &self,
    ) -> Option<CheckedBayesianGraphPosteriorAteInfo> {
        let PreparedExecution::BayesianGraphPosteriorAte(operation) = &self.execution else {
            return None;
        };
        let inference = inference_label(operation.inference());
        Some(CheckedBayesianGraphPosteriorAteInfo {
            query: operation.query().clone(),
            conditional: operation.target().is_conditional(),
            estimator: operation.procedure().id(),
            validation: operation.validation(),
            graph_keys: Arc::clone(&operation.identification().graphs.graph_keys),
            weights: Arc::clone(&operation.identification().graphs.weights),
            identified: Arc::clone(&operation.identification().graphs.identified),
            inference,
        })
    }

    /// Inspect the retained ADMG graph-posterior response operation.
    #[must_use]
    pub fn checked_admg_graph_posterior_response_info(
        &self,
    ) -> Option<CheckedAdmgGraphPosteriorResponseInfo> {
        let operation = self.execution.admg_graph_posterior_response()?;
        Some(CheckedAdmgGraphPosteriorResponseInfo {
            query: operation.query().clone(),
            graph_keys: Arc::clone(&operation.identification().graphs.graph_keys),
            weights: Arc::clone(&operation.identification().graphs.weights),
            identified: Arc::clone(&operation.identification().graphs.identified),
            validation: operation.validation(),
        })
    }

    /// Whether preparation retained an explicit/accepted class response operation.
    #[must_use]
    pub fn has_checked_static_class_response_operation(&self) -> bool {
        self.execution.kind() == ExecutionKind::StaticClassResponse
    }

    /// Inspect the retained explicit/accepted CPDAG or PAG response operation.
    #[must_use]
    pub fn checked_static_class_response_info(&self) -> Option<CheckedStaticClassResponseInfo> {
        let operation = self.execution.static_class_response()?;
        let (identifier, estimator) = operation.procedure();
        let (completion_count, identified_mass, unidentified_mass) = operation.envelope_summary();
        Some(CheckedStaticClassResponseInfo {
            query: operation.query().clone(),
            graph_class: operation.graph_class(),
            identifier,
            estimator,
            inference: inference_label(operation.inference()),
            validation: operation.validation(),
            completion_count,
            identified_mass,
            unidentified_mass,
        })
    }

    /// Whether preparation retained a DAG, CPDAG, or PAG graph-posterior
    /// response operation.
    #[must_use]
    pub fn has_checked_graph_posterior_response_operation(&self) -> bool {
        self.execution.kind() == ExecutionKind::GraphPosteriorResponse
    }

    /// Inspect the retained DAG, CPDAG, or PAG graph-posterior response operation.
    #[must_use]
    pub fn checked_graph_posterior_response_info(
        &self,
    ) -> Option<CheckedGraphPosteriorResponseInfo> {
        let operation = self.execution.graph_posterior_response()?;
        let (identifier, estimator) = operation.procedure();
        Some(CheckedGraphPosteriorResponseInfo {
            query: operation.query().clone(),
            atom_kind: operation.posterior().atom_kind,
            identifier,
            estimator,
            inference: inference_label(operation.inference()),
            graph_keys: Arc::clone(&operation.identification().graphs.graph_keys),
            weights: Arc::clone(&operation.identification().graphs.weights),
            identified: Arc::clone(&operation.identification().graphs.identified),
            validation: operation.validation(),
        })
    }

    /// Read-only target, row binding, and uncertainty selected for propensity estimation.
    #[must_use]
    pub fn checked_propensity_info(&self) -> Option<CheckedPropensityInfo> {
        let operation = self.execution.propensity()?;
        let lowering = operation.lowering();
        let (estimator, uncertainty, bootstrap_replicates) =
            match (lowering.procedure, lowering.uncertainty) {
                (
                    CheckedPropensityProcedure::HajekWeighting,
                    CheckedPropensityUncertainty::HajekAnalyticAndBootstrap {
                        bootstrap_replicates,
                    },
                ) => (
                    crate::strategy_table::EstimatorId::PropensityWeighting,
                    Arc::from("hajek_analytic_and_optional_bootstrap"),
                    Some(bootstrap_replicates),
                ),
                (
                    CheckedPropensityProcedure::AbadieImbensMatching,
                    CheckedPropensityUncertainty::AbadieImbensAnalytic { .. },
                ) => (
                    crate::strategy_table::EstimatorId::PropensityMatching,
                    Arc::from("abadie_imbens_analytic"),
                    None,
                ),
                _ => return None,
            };
        Some(CheckedPropensityInfo {
            estimator,
            adjustment_set: Arc::clone(&lowering.adjustment),
            population: lowering.population.clone(),
            source_rows: Arc::clone(&lowering.source_rows),
            uncertainty,
            bootstrap_replicates,
        })
    }

    /// Inspect the retained conditional target, procedure, and bound design.
    #[must_use]
    pub fn checked_conditional_effect_info(&self) -> Option<CheckedConditionalEffectInfo> {
        let operation = self.execution.conditional()?;
        let (completion_count, identified_mass, unresolved_mass) =
            operation.class_proof.as_ref().map_or((None, None, None), |proof| {
                let (count, identified, unresolved) = proof.completion_mass();
                (Some(count), Some(identified), Some(unresolved))
            });
        Some(CheckedConditionalEffectInfo {
            query: operation.query.clone(),
            graph_class: operation.graph_class(),
            identifier: operation.identifier,
            estimator: operation.estimator,
            procedure: Arc::from(match operation.procedure {
                ConditionalProcedure::LinearInteraction => "linear_interaction_plugin",
                ConditionalProcedure::CrossfitAipwDistribution => {
                    "crossfit_aipw_distribution_scores"
                }
            }),
            adjustment_set: Arc::clone(&operation.estimand.adjustment_set),
            completion_count,
            identified_mass,
            unresolved_mass,
            validation: operation.refute,
            design_roles: Arc::clone(&operation.design_roles),
            source_rows: Arc::clone(&operation.source_rows),
        })
    }

    /// Read-only view of the checked static derivative-response route, when
    /// preparation retained one.
    #[must_use]
    pub fn checked_derivative_response_info(&self) -> Option<CheckedDerivativeResponseInfo> {
        let PreparedExecution::DerivativeResponse(operation) = &self.execution else {
            return None;
        };
        let (_, estimand) = operation.target();
        let (identifier, estimator) = operation.procedure();
        Some(CheckedDerivativeResponseInfo {
            query: operation.query().clone(),
            identifier,
            estimator,
            adjustment_set: Arc::clone(&estimand.adjustment_set),
            max_derivative_cells: operation.max_derivative_cells(),
        })
    }

    /// Read-only view of the retained checked static DAG response route.
    #[must_use]
    pub fn checked_static_dag_response_info(&self) -> Option<CheckedStaticDagResponseInfo> {
        let operation = self.execution.static_dag_response()?;
        let (identifier, estimator, validation) = operation.procedure();
        Some(CheckedStaticDagResponseInfo {
            query: operation.query().clone(),
            identifier,
            estimator,
            validation,
            grid_members: Arc::from(operation.curve_grid()),
            inference: inference_label(operation.inference()),
        })
    }

    /// Read-only view of the sealed Cell AIPW intervention-response route.
    #[must_use]
    pub fn checked_cell_aipw_response_info(&self) -> Option<CheckedCellAipwResponseInfo> {
        let PreparedExecution::CellAipwResponse(operation) = &self.execution else {
            return None;
        };
        let (_, estimand) = operation.target();
        let (identifier, estimator) = operation.procedure();
        Some(CheckedCellAipwResponseInfo {
            query: operation.query().clone(),
            identifier,
            estimator,
            validation: operation.validation(),
            origin: operation.origin(),
            requested_arm: operation.requested_arm(),
            adjustment_set: Arc::clone(&estimand.adjustment_set),
            within_tier: operation.within_tier(),
        })
    }

    /// Read-only view of the retained TemporalDag response route.
    #[must_use]
    pub fn checked_temporal_dag_response_info(&self) -> Option<CheckedTemporalDagResponseInfo> {
        let operation = self.execution.temporal_dag_response()?.operation();
        let (identifier, estimator, uncertainty, _) = operation.procedure();
        Some(CheckedTemporalDagResponseInfo {
            query: operation.query().clone(),
            identifier,
            estimator,
            uncertainty: Arc::from(uncertainty),
            grid_members: Arc::from(operation.grid()),
            horizons: Arc::from(operation.query().temporal.as_ref()?.horizons.as_ref()),
        })
    }

    /// Inspect the retained checked temporal mediation target before execution.
    ///
    /// # Panics
    ///
    /// Panics only if the sealed operation contains a temporal lag that cannot
    /// be represented as an `i32`; preparation rejects such lags.
    #[must_use]
    pub fn checked_temporal_mediation_info(&self) -> Option<CheckedTemporalMediationInfo> {
        let operation = self.execution.temporal_mediation()?;
        let (query, graph, validation, horizons) = operation.inspect();
        Some(CheckedTemporalMediationInfo {
            query: query.clone(),
            graph_signature: Arc::from(
                crate::analysis::checked_temporal_effect::temporal_dag_signature(graph),
            ),
            validation,
            horizons: Arc::from(horizons),
            estimator: operation.estimator(),
            adjustment_sets: Arc::from(
                operation
                    .horizon_contracts()
                    .map(|(horizon, _, _, _, design)| {
                        let keys = design
                            .iter()
                            .map(|column| antecedent_core::TemporalNodeKey {
                                variable: column.variable,
                                offset: -i32::try_from(column.lag.raw())
                                    .expect("checked temporal mediation lags fit i32"),
                            })
                            .collect::<Vec<_>>();
                        (horizon, Arc::from(keys))
                    })
                    .collect::<Vec<_>>(),
            ),
        })
    }

    /// Inspect the retained checked class-envelope or graph-posterior temporal
    /// mediation target, proof mass, and procedure before execution.
    #[must_use]
    pub fn checked_temporal_class_mediation_info(
        &self,
    ) -> Option<CheckedTemporalClassMediationInfo> {
        let operation = self.execution.temporal_class_mediation()?;
        let (atom_count, identified_mass, unidentified_mass) = operation.proof_mass();
        Some(CheckedTemporalClassMediationInfo {
            query: operation.query().clone(),
            graph_class: operation.graph_class(),
            structure_source: operation.structure_source(),
            identifier: operation.identifier(),
            estimator: operation.estimator(),
            validation: operation.validation(),
            bootstrap_replicates: operation.bootstrap_replicates(),
            horizons: Arc::clone(&operation.query().horizons),
            atom_count,
            identified_mass,
            unidentified_mass,
            proof_signature: operation.proof_signature(),
        })
    }

    pub(crate) fn has_checked_temporal_class_mediation_operation(&self) -> bool {
        self.execution.kind() == ExecutionKind::TemporalClassMediation
    }

    /// Inspect the retained fixed-network interference target and procedure.
    #[must_use]
    pub fn checked_interference_info(&self) -> Option<CheckedInterferenceInfo> {
        let operation = self.execution.interference()?;
        let (identifier, estimator) = operation.procedure();
        let (unit_count, network_edge_count) = operation.counts();
        Some(CheckedInterferenceInfo {
            query: operation.query().clone(),
            identifier,
            estimator,
            unit_count,
            network_edge_count,
        })
    }

    /// Inspect the sealed two-scenario Unknown-tier average-effect operation.
    #[must_use]
    pub fn checked_unknown_tiered_average_info(&self) -> Option<CheckedUnknownTieredAverageInfo> {
        let PreparedExecution::UnknownTieredAverage(operation) = &self.execution else {
            return None;
        };
        Some(CheckedUnknownTieredAverageInfo {
            query: operation.query.clone(),
            tier_count: operation.background.tiers.len(),
            scenario_count: operation.identification.estimands.len(),
            identifier: operation.identifier,
            estimator: operation.estimator,
            plan_id: Arc::clone(&operation.physical.record.plan_id),
        })
    }

    /// Inspect the retained Bayesian IV or sharp-RD model before execution.
    #[must_use]
    pub fn checked_bayesian_specialist_info(
        &self,
    ) -> Option<crate::analysis::execute::CheckedBayesianSpecialistInfo> {
        match &self.execution {
            PreparedExecution::BayesianSpecialist(operation) => Some(operation.info()),
            _ => None,
        }
    }

    /// Whether preparation retained the complete Bayesian IV or sharp-RD operation.
    #[must_use]
    pub fn has_checked_bayesian_specialist_operation(&self) -> bool {
        self.execution.kind() == ExecutionKind::BayesianSpecialist
    }

    /// Inspect the retained certified trial-to-target transport operation.
    #[must_use]
    pub fn checked_transport_trial_info(
        &self,
    ) -> Option<crate::analysis::execute::CheckedTransportTrialInfo> {
        match &self.execution {
            PreparedExecution::TransportTrial(operation) => Some(operation.info()),
            _ => None,
        }
    }

    /// Whether preparation retained the complete trial-to-target transport operation.
    #[must_use]
    pub fn has_checked_transport_trial_operation(&self) -> bool {
        self.execution.kind() == ExecutionKind::TransportTrial
    }

    /// Inspect the retained checked temporal effect before execution.
    #[must_use]
    pub fn checked_temporal_dag_effect_info(&self) -> Option<CheckedTemporalDagEffectInfo> {
        let operation = self.execution.temporal_dag_effect()?.operation();
        let (identifier, estimator, validation, bootstrap_replicates) = operation.procedure();
        let adjustment_set = operation
            .estimand()
            .adjustment_set
            .iter()
            .filter_map(|id| operation.indexer().key_of(id.raw()).ok())
            .collect::<Vec<_>>();
        Some(CheckedTemporalDagEffectInfo {
            query: operation.query().clone(),
            identifier,
            estimator,
            validation,
            bootstrap_replicates,
            adjustment_set: adjustment_set.into(),
            graph_signature: Arc::from(operation.graph_signature()),
        })
    }

    /// Inspect the retained Bayesian temporal DAG target and model settings.
    #[must_use]
    pub fn checked_bayesian_temporal_dag_effect_info(
        &self,
    ) -> Option<CheckedBayesianTemporalDagEffectInfo> {
        let operation = self.execution.bayesian_temporal_dag_effect()?;
        let target = operation.target();
        let estimator = EstimatorId::temporal_effect_procedure(target.query(), true);
        Some(CheckedBayesianTemporalDagEffectInfo {
            query: target.query().clone(),
            identifier: crate::strategy_table::IdentifierId::TemporalBackdoorUnfolded,
            estimator,
            validation: operation.validation(),
            posterior_draws: operation.config().n_draws,
            graph_signature: Arc::from(target.graph_signature()),
        })
    }

    /// Inspect the retained source graph and complete temporal class envelope.
    #[must_use]
    pub fn checked_temporal_class_effect_info(&self) -> Option<CheckedTemporalClassEffectInfo> {
        let operation = self.execution.temporal_class_effect()?.operation();
        let (identifier, estimator, bootstrap_replicates, _, validation, _) = operation.procedure();
        let envelope = &operation.bundle().envelope.envelope;
        Some(CheckedTemporalClassEffectInfo {
            query: operation.query().clone(),
            graph_class: operation.graph_class(),
            identifier,
            estimator,
            validation,
            bootstrap_replicates,
            completion_count: envelope.cases.len(),
            completion_limit: operation.max_completions(),
            identified_mass: envelope.identified_weight.0,
            unidentified_mass: envelope.unidentified_weight.0,
            truncated_completions: envelope.truncated_completions,
            graph_version: u64::from(operation.structure_version()),
        })
    }

    /// Whether preparation retained a Bayesian TemporalCpdag/TemporalPag effect plan.
    #[must_use]
    pub fn has_checked_bayesian_temporal_class_effect_operation(&self) -> bool {
        self.execution.kind() == ExecutionKind::BayesianTemporalClassEffect
    }

    /// Inspect the retained source class, completion proof, and Bayesian settings.
    #[must_use]
    pub fn checked_bayesian_temporal_class_effect_info(
        &self,
    ) -> Option<CheckedBayesianTemporalClassEffectInfo> {
        let operation = self.execution.bayesian_temporal_class_effect()?;
        let target = operation.target();
        let (identifier, _, _, _, _, _) = target.procedure();
        let envelope = &target.bundle().envelope.envelope;
        Some(CheckedBayesianTemporalClassEffectInfo {
            query: target.query().clone(),
            graph_class: target.graph_class(),
            identifier,
            estimator: operation.estimator(),
            validation: operation.validation(),
            posterior_draws: operation.config().n_draws,
            class_prior_supplied: operation.class_prior().is_some(),
            completion_count: envelope.cases.len(),
            completion_limit: target.max_completions(),
            identified_mass: envelope.identified_weight.0,
            unidentified_mass: envelope.unidentified_weight.0,
            truncated_completions: envelope.truncated_completions,
            graph_version: u64::from(target.structure_version()),
        })
    }

    /// Whether preparation retained a temporal graph-posterior effect plan.
    #[must_use]
    pub fn has_checked_temporal_graph_posterior_effect_operation(&self) -> bool {
        self.execution.kind() == ExecutionKind::TemporalGraphPosteriorEffect
    }

    /// Inspect the frozen temporal posterior samples, atom proofs, and procedure.
    #[must_use]
    pub fn checked_temporal_graph_posterior_effect_info(
        &self,
    ) -> Option<CheckedTemporalGraphPosteriorEffectInfo> {
        let operation = self.execution.temporal_graph_posterior_effect()?;
        let (graph_keys, weights, identified) = operation.sample_mass();
        let inference = inference_label(operation.inference());
        Some(CheckedTemporalGraphPosteriorEffectInfo {
            query: operation.query().clone(),
            atom_kind: operation.posterior().atom_kind,
            identifier: operation.identifier(),
            estimator: operation.estimator(),
            validation: operation.validation(),
            inference,
            graph_keys: Arc::from(graph_keys),
            weights: Arc::from(weights),
            identified: Arc::from(identified),
            identified_atom_count: operation.identified_atom_count(),
            bootstrap_replicates: operation.bootstrap_replicates(),
        })
    }

    /// Compatibility spelling emphasizing the temporal effect operation family.
    #[must_use]
    pub fn checked_temporal_effect_info(&self) -> Option<CheckedTemporalDagEffectInfo> {
        self.checked_temporal_dag_effect_info()
    }

    /// Compatibility spelling emphasizing the response operation family.
    #[must_use]
    pub fn checked_temporal_response_info(&self) -> Option<CheckedTemporalDagResponseInfo> {
        self.checked_temporal_dag_response_info()
    }

    /// Inspect the retained source graph and per-horizon temporal class
    /// response envelopes.
    #[must_use]
    pub fn checked_temporal_class_response_info(&self) -> Option<CheckedTemporalClassResponseInfo> {
        let operation = self.execution.temporal_class_response()?;
        let (identifier, estimator, bootstrap_replicates, validation) = operation.procedure();
        let (identified_mass, unidentified_mass) = operation.enumeration_mass();
        Some(CheckedTemporalClassResponseInfo {
            query: operation.query().clone(),
            graph_class: operation.graph_class(),
            identifier,
            estimator,
            inference: inference_label(operation.inference()),
            validation,
            bootstrap_replicates,
            horizons: Arc::from(operation.horizons()),
            completion_count: operation.completion_count(),
            completion_limit: operation.max_completions(),
            identified_mass,
            unidentified_mass,
            graph_version: u64::from(operation.structure_version()),
        })
    }

    /// Inspect the retained temporal graph-posterior response operation.
    #[must_use]
    pub fn checked_temporal_graph_posterior_response_info(
        &self,
    ) -> Option<CheckedTemporalGraphPosteriorResponseInfo> {
        let operation = self.execution.temporal_graph_posterior_response()?;
        let (graph_keys, weights, identified) = operation.atom_weights();
        Some(CheckedTemporalGraphPosteriorResponseInfo {
            query: operation.query().clone(),
            atom_kind: operation.posterior().atom_kind,
            graph_keys: Arc::from(graph_keys),
            weights: Arc::from(weights),
            identified: Arc::from(identified),
            identifier: operation.identifier(),
            estimator: operation.estimator(),
            inference: inference_label(operation.inference()),
            validation: operation.validation(),
        })
    }

    /// Checked linear two-stage front-door lowering retained by this handle.
    #[must_use]
    pub fn checked_frontdoor_linear(
        &self,
    ) -> Option<&antecedent_estimate::CheckedFrontDoorPreparation> {
        self.execution.frontdoor_linear().map(|operation| &operation.preparation)
    }

    /// Checked Wald or single-binary-instrument 2SLS lowering retained by this handle.
    #[must_use]
    pub fn checked_iv(&self) -> Option<&antecedent_estimate::CheckedIvPreparation> {
        self.execution.iv().map(CheckedIvOperation::preparation)
    }

    /// Checked functional program retained for a prepared distribution query.
    #[must_use]
    pub fn checked_distribution_program(&self) -> Option<&antecedent_expr::FunctionalProgram> {
        self.execution.distribution().map(|operation| operation.prepared.program())
    }

    /// Checked functional program retained for an ADMG general-ID effect.
    #[must_use]
    pub fn checked_functional_effect_program(&self) -> Option<&antecedent_expr::FunctionalProgram> {
        self.execution
            .functional_effect_operation()
            .map(|operation| operation.prepared.program())
            .or_else(|| {
                self.execution
                    .path_specific_effect_operation()
                    .map(|operation| operation.prepared.program())
            })
    }

    /// Ordered checked members for a prepared ADMG mean response curve or
    /// scalar intervention response.
    #[must_use]
    pub fn checked_functional_effect_response_members(
        &self,
    ) -> Option<&[CheckedFunctionalEffectResponseMember]> {
        self.execution.admg_response_curve().map(|operation| operation.members.as_ref())
    }

    /// Export the complete empirical factor laws for the retained effect program.
    pub(crate) fn functional_effect_factor_snapshot(
        &self,
        data: &TabularData,
    ) -> Result<
        Option<antecedent_estimate::functional_distribution::EmpiricalDistributionFactorSnapshot>,
        CausalError,
    > {
        self.ensure_schema_compatible(data)?;
        if let Some(operation) = self.execution.functional_effect_operation() {
            return operation
                .rebind(data)?
                .prepared
                .factor_snapshot()
                .map(Some)
                .map_err(CausalError::from);
        }
        if let Some(operation) = self.execution.path_specific_effect_operation() {
            return operation
                .rebind(data)?
                .prepared
                .factor_snapshot()
                .map(Some)
                .map_err(CausalError::from);
        }
        Ok(None)
    }

    /// Rebound empirical factor snapshots, one for each response member in grid order.
    pub(crate) fn functional_effect_response_factor_snapshots(
        &self,
        data: &TabularData,
    ) -> Result<Option<Arc<[antecedent_estimate::functional_distribution::EmpiricalDistributionFactorSnapshot]>>, CausalError>{
        let Some(operation) = self.execution.admg_response_curve() else { return Ok(None) };
        self.ensure_schema_compatible(data)?;
        // Compare the factors' actual producing snapshot, rather than the
        // study's retained data: generic refresh can replace only the latter.
        let rebound;
        let current = if operation.factor_snapshot_digest == data.storage().content_digest() {
            operation
        } else {
            rebound = operation.rebind(data)?;
            &rebound
        };
        let snapshots = current
            .members
            .iter()
            .map(|member| member.prepared.factor_snapshot().map_err(CausalError::from))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Some(Arc::from(snapshots)))
    }

    /// Export the complete provider laws for the retained distribution program,
    /// rebound to `data` after the normal semantic-schema check.
    pub(crate) fn distribution_factor_snapshot(
        &self,
        data: &TabularData,
    ) -> Result<
        Option<antecedent_estimate::functional_distribution::EmpiricalDistributionFactorSnapshot>,
        CausalError,
    > {
        let Some(operation) = self.execution.distribution() else {
            return Ok(None);
        };
        self.ensure_schema_compatible(data)?;
        let prepared = operation.rebind(data)?;
        prepared.factor_snapshot().map(Some).map_err(CausalError::from)
    }

    /// Borrow the caller's frozen query, with original variable ids and query kind.
    #[must_use]
    pub fn query(&self) -> &CausalQuery {
        &self.analysis.query
    }

    /// Crate-visible frozen study.
    pub(crate) fn study(&self) -> &Study {
        &self.analysis
    }

    #[cfg(test)]
    pub(crate) fn study_mut(&mut self) -> &mut Study {
        self.program_cache = std::sync::OnceLock::new();
        if self.execution.kind() == ExecutionKind::CheckedLinear {
            self.execution = PreparedExecution::OrdinaryDispatch;
        }
        &mut self.analysis
    }

    /// Replace the frozen study after a successful refresh.
    pub(super) fn replace_study(&mut self, study: Study) {
        self.program_cache = std::sync::OnceLock::new();
        self.analysis = study;
    }

    pub(crate) fn program_cache(
        &self,
    ) -> &std::sync::OnceLock<Arc<crate::analysis::contract::ProgramPayloads>> {
        &self.program_cache
    }

    /// Replace custom validators. Incoming names must match the prepare-time set.
    ///
    /// # Errors
    ///
    /// Name mismatch.
    pub fn rebind_custom_validators(
        &mut self,
        validators: Vec<std::sync::Arc<dyn antecedent_validate::CustomEffectValidator>>,
    ) -> Result<(), CausalError> {
        let expected: std::collections::BTreeSet<&str> =
            self.analysis.custom_validators.iter().map(|v| v.name()).collect();
        let incoming: std::collections::BTreeSet<&str> =
            validators.iter().map(|v| v.name()).collect();
        if expected != incoming {
            return Err(crate::unsupported_reason!(
                "attested_not_reverifiable",
                "rebind_validators names must match attested names"
            ));
        }
        self.execution.set_custom_validators(&validators);
        self.analysis.custom_validators = validators;
        Ok(())
    }

    /// Population bindings frozen at prepare, when the query names a predicate
    /// or a custom target distribution.
    #[must_use]
    pub fn population_registry(&self) -> Option<&antecedent_core::PopulationRegistry> {
        self.analysis.population_registry.as_ref()
    }

    /// Names of the caller custom validators frozen at prepare, in order.
    ///
    /// These are the names a claim attests and the names
    /// [`Self::rebind_custom_validators`] requires.
    #[must_use]
    pub fn custom_validator_names(&self) -> Vec<&str> {
        self.analysis.custom_validators.iter().map(|validator| validator.name()).collect()
    }

    /// Stream progressive stages from later estimates of this handle.
    ///
    /// `None` stops streaming. The sink is an in-process execution control: it
    /// is not part of the contract, program, or claim identity.
    pub fn set_stage_sink(
        &mut self,
        sink: Option<Arc<dyn crate::analysis::stage::StageResultSink>>,
    ) {
        self.execution.set_stage_sink(sink.clone());
        self.analysis.stage_sink = sink;
    }

    /// Whether estimates of this handle stream identify → estimate_point →
    /// uncertainty → validate stage payloads.
    ///
    /// Stages describe one identification and one scalar effect estimate. They
    /// stream from the single-estimand static executors: a supplied or DAG-coerced
    /// DAG average effect (Frequentist or Bayesian, excluding `rd.sharp` and the
    /// general-ID functional plug-in) and the Bayesian DAG conditional effect.
    #[must_use]
    pub fn streams_stages(&self) -> bool {
        let analysis = &self.analysis;
        if analysis.graph_posterior.is_some()
            || analysis.tiered.is_some()
            || !matches!(analysis.data, DataInput::Tabular(_))
        {
            return false;
        }
        let dag_like = match analysis.graph.class() {
            GraphClass::Dag => true,
            GraphClass::Admg => analysis
                .graph
                .as_admg()
                .is_some_and(|admg| !crate::analysis::execute::admg_has_bidirected(admg)),
            _ => false,
        };
        let estimator = self.plan.logical.record.estimator.as_deref();
        match &analysis.query {
            CausalQuery::AverageEffect(_) => {
                dag_like && !matches!(estimator, Some("rd.sharp" | "functional.effect"))
            }
            CausalQuery::ConditionalEffect(_) => {
                analysis.graph.class() == GraphClass::Dag
                    && matches!(analysis.inference, InferenceMode::Bayesian(_))
            }
            _ => false,
        }
    }

    /// Frozen horizon-specific identification and its exact unfolded variable namespace.
    ///
    /// A `TemporalDag` prepare caches this directly. A DBN posterior or a TemporalCpdag/Pag
    /// envelope caches a per-atom or per-completion result instead; this projects either
    /// onto the same shape (see `full_temporal_identification`), so an
    /// exported `analysis_result` artifact validates its identification against the same
    /// namespace the compiled contract already uses.
    #[must_use]
    pub fn temporal_identification(
        &self,
    ) -> Option<std::borrow::Cow<'_, CachedTemporalIdentification>> {
        crate::analysis::contract::full_temporal_identification(&self.analysis)
    }

    /// Borrow the frozen schema fingerprint.
    #[must_use]
    pub fn schema(&self) -> &CausalSchema {
        &self.schema
    }

    /// Matrix structure-source axis frozen at prepare.
    #[must_use]
    pub const fn structure_source(&self) -> crate::support::StructureSource {
        self.state.analysis.structure_source()
    }

    /// Evidence contract frozen at prepare. `None` when the query is off-axis.
    #[must_use]
    pub const fn support_status(&self) -> Option<crate::support::CellStatus> {
        self.state.analysis.support_status()
    }

    /// Borrow the ready physical plan retained from prepare.
    #[must_use]
    pub fn plan(&self) -> &PhysicalExecutionPlan {
        &self.plan
    }

    /// Borrow the prepare-time AIPW score table, when the cell exported one.
    #[must_use]
    pub fn score_table(&self) -> Option<&ScoreTable> {
        self.score_table.as_ref()
    }

    /// Shared batch design attached when this plan was prepared inside a batch.
    #[must_use]
    pub fn shared_design(&self) -> Option<&crate::analysis::batch::SharedBatchDesign> {
        self.analysis.shared_batch_design.as_deref()
    }

    /// Control intervention level frozen on a counterfactual ITE query.
    #[must_use]
    pub fn counterfactual_control_level(&self) -> f64 {
        match self.query() {
            CausalQuery::Counterfactual(q) => match &q.control {
                Intervention::Set { value, .. } => value.as_f64().unwrap_or(0.0),
                _ => 0.0,
            },
            CausalQuery::NestedCounterfactual(q) => q.control_value(),
            _ => 0.0,
        }
    }
}
