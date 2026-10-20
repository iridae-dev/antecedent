//! One decision from a cluster trial, a temporal period effect, a Bayesian cost
//! model, and an uncertain endpoint-to-margin bridge.
//!
//! The trial is the cluster-randomized Neyman route. The period effect is the
//! checked two-step temporal history, and its variance is the square of the
//! standard error that route's balanced-unit interval publishes. The cost model
//! is the conjugate Gaussian posterior. The margin is an affine scientific
//! bridge with declared mapping uncertainty and slope-source covariance. The
//! decision is the expectation of an affine utility of their joint linear
//! combination. The three inputs are estimated apart, so that independence is
//! the recorded dependence and the cross-terms are zero. The bridge dependence
//! stays on the margin.
//!
//! A new utility recomputes the decision and refits no trial, mechanism, or
//! posterior. A new prior refits the cost posterior. A new period refits the
//! temporal mechanisms. Withdrawing the trial withdraws the margin and the
//! decision and leaves the temporal and cost claims.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeMap;
use std::sync::Arc;

use antecedent_contract::bridge::{
    AssumptionKind, BridgeAssumption, BridgeContract, BridgeEndpoint, BridgeKind, BridgeMapping,
    BridgeScope, BridgeSource, QuantityClass, SourceMappingDependence, ValueKind,
};
use antecedent_contract::{
    ClaimDomains, ClaimEnvelope, ClaimKind, DomainStatus, ExternalRefusal, InputRole,
    PropagationOperation, QuantityRole, ReasoningView, ScientificQuantity, SlotAvailability,
};
use antecedent_core::recalc::{Stage, StageStatus};
use antecedent_core::{
    CanonicalWriter, ContractIdentities, ExecutionContext, RandomizationDesign,
    RandomizedEffectQuery, SemanticDigest, VariableId,
};
use antecedent_data::TabularData;
use antecedent_design::decision_contract::{
    ActionKind, DecisionAction, DecisionContract, DecisionCriterion, StructuralPolicy, UtilityExpr,
};
use antecedent_design::decision_eval::{DecisionEvalError, DecisionResult, MeanSource};
use antecedent_estimate::temporal_dependent_interval::{DependentIntervalConfig, IntervalMethod};

use crate::analysis::joint_law_propagation::{
    JointLawPropagationRefusal, LinearDependence, MomentInput, PropagationClaim,
    consume_joint_law_propagation_artifact, propagate_linear_moments,
};
use crate::analysis::recalc_bayesian::{
    BayesianLaw, BayesianModel, BayesianRequest, BayesianSession, PosteriorSummarySpec,
    execute_bayesian_with_receipt,
};
use crate::analysis::recalc_receipt::{RecalcRunError, UtilitySpec};
use crate::analysis::recalc_temporal::{
    TemporalFunctional, TemporalHistoryWire, TemporalRequest, TemporalRunError, TemporalSession,
    TemporalUnitWire, execute_temporal_with_receipt,
};
use crate::analysis::scientific_bridge::{PreparedScientificBridge, ScientificBridgeRefusal};
use super::{
    CompositionGraph, CompositionNode, CompositionVerification, CompositionVerificationRefusal,
    DependencyKind, NodeKind, ObservedEffect,
};

use crate::{BayesianConfig, Study};

const SLOPE_VARIANCE: f64 = 0.04;
const POPULATION: &str = "mixed-trial";
const REGIME: &str = "cluster-assignment";
const MARGIN_UNITS: &str = "utility";

/// Why the mixed workflow refused.
#[derive(Debug)]
pub enum FlagshipError {
    /// The cluster-randomized trial refused.
    Cluster(Box<crate::CausalError>),
    /// The temporal period effect refused.
    Temporal(Box<TemporalRunError>),
    /// The Bayesian cost model refused.
    Bayesian(Box<RecalcRunError>),
    /// The endpoint bridge refused to prepare or derive.
    Bridge(Box<ScientificBridgeRefusal>),
    /// The uncertain margin refused.
    Margin(Box<ExternalRefusal>),
    /// The joint combination refused.
    Joint(Box<JointLawPropagationRefusal>),
    /// The decision contract refused.
    Decision(Box<DecisionEvalError>),
    /// Claim withdrawal could not be classified.
    Verification(Box<CompositionVerificationRefusal>),
    /// The claim graph refused.
    Graph(String),
    /// A route did not return the object this workflow requires.
    Route(&'static str),
}

impl std::fmt::Display for FlagshipError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cluster(error) => write!(f, "{error}"),
            Self::Temporal(error) => write!(f, "{error}"),
            Self::Bayesian(error) => write!(f, "{error}"),
            Self::Bridge(error) => write!(f, "{error}"),
            Self::Margin(error) => write!(f, "{}: {}", error.code, error.detail),
            Self::Joint(error) => write!(f, "{error}"),
            Self::Decision(error) => write!(f, "decision evaluation refused: {error:?}"),
            Self::Verification(error) => write!(f, "{error}"),
            Self::Graph(detail) => write!(f, "{detail}"),
            Self::Route(detail) => write!(f, "{detail}"),
        }
    }
}

impl std::error::Error for FlagshipError {}

impl From<crate::CausalError> for FlagshipError {
    fn from(error: crate::CausalError) -> Self {
        Self::Cluster(Box::new(error))
    }
}

impl From<TemporalRunError> for FlagshipError {
    fn from(error: TemporalRunError) -> Self {
        Self::Temporal(Box::new(error))
    }
}

impl From<RecalcRunError> for FlagshipError {
    fn from(error: RecalcRunError) -> Self {
        Self::Bayesian(Box::new(error))
    }
}

impl From<ScientificBridgeRefusal> for FlagshipError {
    fn from(error: ScientificBridgeRefusal) -> Self {
        Self::Bridge(Box::new(error))
    }
}

impl From<ExternalRefusal> for FlagshipError {
    fn from(error: ExternalRefusal) -> Self {
        Self::Margin(Box::new(error))
    }
}

impl From<JointLawPropagationRefusal> for FlagshipError {
    fn from(error: JointLawPropagationRefusal) -> Self {
        Self::Joint(Box::new(error))
    }
}

impl From<DecisionEvalError> for FlagshipError {
    fn from(error: DecisionEvalError) -> Self {
        Self::Decision(Box::new(error))
    }
}

impl From<CompositionVerificationRefusal> for FlagshipError {
    fn from(error: CompositionVerificationRefusal) -> Self {
        Self::Verification(Box::new(error))
    }
}

/// The cluster-randomized contrast the decision keeps.
#[derive(Clone, Debug)]
struct ClusterFit {
    effect: f64,
    variance: f64,
    design: String,
    uncertainty: String,
}

/// The decision, with the joint combination and the bridge dependence attached.
#[derive(Clone, Debug)]
pub struct MixedDecision {
    /// Linear combination under the declared covariance.
    pub joint: PropagationClaim,
    /// Affine margin, including mapping uncertainty and source dependence.
    pub margin: antecedent_contract::bridge::PropagatedUncertainty,
    /// Recorded independence of the margin, the period effect, and the cost.
    pub dependence: LinearDependence,
    /// Point value the bridge derived before the delta method.
    pub bridge_value: f64,
    /// Variance the temporal interval published for the period effect.
    pub temporal_variance: f64,
    /// Variance the cost posterior published.
    pub cost_variance: f64,
    /// Affine expected-utility comparison of acting and declining.
    pub evaluation: DecisionResult,
}

/// Bytes of the joint combination and the identity retained apart from them.
#[derive(Clone, Debug)]
pub struct DecisionArtifact {
    /// `joint_law_propagation_v1` bytes.
    pub bytes: Vec<u8>,
    /// Identity a consumer passes back independently of `bytes`.
    pub identity: String,
}

/// A decision rebuilt by replaying an artifact.
#[derive(Clone, Debug)]
pub struct LoadedDecision {
    /// Replayed combination mean.
    pub mean: f64,
    /// Replayed combination variance.
    pub variance: f64,
    /// Decision evaluated again on the replayed mean.
    pub evaluation: DecisionResult,
}

/// Work the utility change measured on the temporal and cost routes.
#[derive(Clone, Copy, Debug)]
pub struct UtilityReplay {
    /// Temporal factor constructions during the utility change.
    pub temporal_factor_builds: u64,
    /// Temporal identifications during the utility change.
    pub temporal_identifications: u64,
    /// Cost-model solves during the utility change.
    pub bayesian_model_fits: u64,
    /// Posterior rows emitted during the utility change.
    pub bayesian_posterior_draws: u64,
    /// Cost-model identifications during the utility change.
    pub bayesian_identifications: u64,
}

/// What a prior change did to the cost model.
#[derive(Clone, Copy, Debug)]
pub struct PriorReplay {
    /// Posterior solves performed for the new prior.
    pub bayesian_model_fits: u64,
    /// Identifications charged to the cost model.
    pub bayesian_identifications: u64,
    /// The posterior stage was recomputed.
    pub score_recomputed: bool,
    /// The identification stage was reused.
    pub identification_reused: bool,
}

/// What a period change did to the temporal route.
#[derive(Clone, Copy, Debug)]
pub struct PeriodReplay {
    /// Factor constructions performed for the new period.
    pub temporal_factor_builds: u64,
    /// Identifications charged to the temporal route.
    pub temporal_identifications: u64,
    /// The mechanism stage was recomputed.
    pub score_recomputed: bool,
    /// The identification stage was reused.
    pub identification_reused: bool,
}

/// Retained trial, period effect, cost posterior, margin, and decision.
pub struct FlagshipWorkflow {
    cluster: ClusterFit,
    cluster_trials: u64,
    temporal: TemporalSession,
    temporal_request: TemporalRequest,
    temporal_effect: f64,
    temporal_variance: f64,
    temporal_means: usize,
    bayesian: BayesianSession,
    bayesian_request: BayesianRequest,
    cost_mean: f64,
    cost_sd: f64,
    cost_draws: usize,
    benefit: f64,
    decision: MixedDecision,
}

impl FlagshipWorkflow {
    /// Fit every route and the decision at benefit `1`.
    ///
    /// # Errors
    /// A route, bridge, joint combination, or decision refusal.
    pub fn execute(ctx: &ExecutionContext) -> Result<Self, FlagshipError> {
        let cluster = fit_cluster(ctx)?;
        let temporal_request = temporal_request(1.0);
        let mut temporal = TemporalSession::new();
        let temporal_out = execute_temporal_with_receipt(&mut temporal, &temporal_request, ctx)?;
        let bayesian_request = cost_request(1.0, 1.0e6);
        let mut bayesian = BayesianSession::new();
        let bayesian_out = execute_bayesian_with_receipt(&mut bayesian, &bayesian_request, ctx)?;
        let temporal_variance = published_temporal_variance(&temporal, ctx)?;
        let decision = compose(
            &cluster,
            temporal_out.recalc.law.ate,
            temporal_variance,
            bayesian_out.law.mean,
            bayesian_out.law.standard_deviation,
            1.0,
            ctx,
        )?;
        Ok(Self {
            cluster,
            cluster_trials: 1,
            temporal,
            temporal_request,
            temporal_effect: temporal_out.recalc.law.ate,
            temporal_variance,
            temporal_means: temporal_out.means.len(),
            bayesian,
            bayesian_request,
            cost_mean: bayesian_out.law.mean,
            cost_sd: bayesian_out.law.standard_deviation,
            cost_draws: bayesian_out.law.draws,
            benefit: 1.0,
            decision,
        })
    }

    /// The current decision.
    #[must_use = "the decision carries the joint combination and the bridge dependence"]
    pub const fn decision(&self) -> &MixedDecision {
        &self.decision
    }

    /// Cluster-randomized contrasts this workflow has fit.
    #[must_use = "the count is how many times the cluster trial ran"]
    pub const fn cluster_trials(&self) -> u64 {
        self.cluster_trials
    }

    /// Cluster contrast.
    #[must_use = "the contrast is the trial input of the margin"]
    pub const fn cluster_effect(&self) -> f64 {
        self.cluster.effect
    }

    /// Cluster variance bound.
    #[must_use = "the variance is the source moment of the uncertain margin"]
    pub const fn cluster_variance(&self) -> f64 {
        self.cluster.variance
    }

    /// Uncertainty label of the cluster route.
    #[must_use = "the label names the cluster estimator that ran"]
    pub fn cluster_uncertainty(&self) -> &str {
        &self.cluster.uncertainty
    }

    /// Assignment design label of the cluster route.
    #[must_use = "the design label distinguishes the cluster trial"]
    pub fn cluster_design(&self) -> &str {
        &self.cluster.design
    }

    /// Temporal period contrast.
    #[must_use = "the contrast is the temporal input of the decision"]
    pub const fn temporal_effect(&self) -> f64 {
        self.temporal_effect
    }

    /// Sequence means the temporal effect was built from.
    #[must_use = "two means are the active and control sequences of an effect"]
    pub const fn temporal_means(&self) -> usize {
        self.temporal_means
    }

    /// Posterior mean of the cost contrast.
    #[must_use = "the mean is the cost input of the decision"]
    pub const fn cost_mean(&self) -> f64 {
        self.cost_mean
    }

    /// Posterior rows retained for the cost contrast.
    #[must_use = "the row count is the posterior the cost model issued"]
    pub const fn cost_draws(&self) -> usize {
        self.cost_draws
    }

    /// Recompute the decision at `benefit` and refit no causal route.
    ///
    /// The temporal and cost sessions are asked again so their receipts measure
    /// the work. Only their utility stages move.
    ///
    /// # Errors
    /// A non-finite benefit, or a refusal from a reused session or the decision.
    pub fn utility_only(
        &mut self,
        benefit: f64,
        ctx: &ExecutionContext,
    ) -> Result<UtilityReplay, FlagshipError> {
        require_finite(benefit)?;
        self.temporal_request.benefit_per_unit = benefit;
        self.bayesian_request.utility.benefit_per_unit = benefit;
        let temporal_out =
            execute_temporal_with_receipt(&mut self.temporal, &self.temporal_request, ctx)?;
        let bayesian_out =
            execute_bayesian_with_receipt(&mut self.bayesian, &self.bayesian_request, ctx)?;
        self.install(
            temporal_out.recalc.law.ate,
            self.temporal_variance,
            &bayesian_out.law,
            benefit,
            ctx,
        )?;
        let temporal = temporal_out.recalc.receipt.totals();
        let cost = bayesian_out.receipt.totals();
        Ok(UtilityReplay {
            temporal_factor_builds: temporal.factor_builds,
            temporal_identifications: temporal.identifications,
            bayesian_model_fits: cost.model_fits,
            bayesian_posterior_draws: cost.posterior_draws,
            bayesian_identifications: cost.identifications,
        })
    }

    /// Replace the cost prior scale and refit the cost posterior.
    ///
    /// The cluster trial and the temporal mechanisms are not asked.
    ///
    /// # Errors
    /// A non-positive scale, or a refusal from the cost model or the decision.
    pub fn prior_change(
        &mut self,
        prior_scale: f64,
        ctx: &ExecutionContext,
    ) -> Result<PriorReplay, FlagshipError> {
        if !(prior_scale.is_finite() && prior_scale > 0.0) {
            return Err(FlagshipError::Route("a prior scale is a finite positive number"));
        }
        self.bayesian_request.inference.prior_scale = prior_scale;
        let bayesian_out =
            execute_bayesian_with_receipt(&mut self.bayesian, &self.bayesian_request, ctx)?;
        self.install(
            self.temporal_effect,
            self.temporal_variance,
            &bayesian_out.law,
            self.benefit,
            ctx,
        )?;
        Ok(PriorReplay {
            bayesian_model_fits: bayesian_out.receipt.totals().model_fits,
            bayesian_identifications: bayesian_out.receipt.totals().identifications,
            score_recomputed: recomputed(&bayesian_out.plan, Stage::ScoreArtifact),
            identification_reused: reused(&bayesian_out.plan, Stage::Identification),
        })
    }

    /// Replace the temporal observation period and refit the temporal mechanisms.
    ///
    /// The cluster trial and the cost posterior are not asked. The period must
    /// still contain the retained histories.
    ///
    /// # Errors
    /// A period that does not contain the histories, or a temporal or decision refusal.
    pub fn period_change(
        &mut self,
        period: (u64, u64),
        ctx: &ExecutionContext,
    ) -> Result<PeriodReplay, FlagshipError> {
        if !period_contains(&self.temporal_request, period) {
            return Err(FlagshipError::Route(
                "a replacement period must still contain the retained two-step histories",
            ));
        }
        self.temporal_request.period = period;
        let temporal_out =
            execute_temporal_with_receipt(&mut self.temporal, &self.temporal_request, ctx)?;
        let temporal_variance = published_temporal_variance(&self.temporal, ctx)?;
        self.install(
            temporal_out.recalc.law.ate,
            temporal_variance,
            &retained_cost(self),
            self.benefit,
            ctx,
        )?;
        Ok(PeriodReplay {
            temporal_factor_builds: temporal_out.recalc.receipt.totals().factor_builds,
            temporal_identifications: temporal_out.recalc.receipt.totals().identifications,
            score_recomputed: recomputed(&temporal_out.recalc.plan, Stage::ScoreArtifact),
            identification_reused: reused(&temporal_out.recalc.plan, Stage::Identification),
        })
    }

    /// Withdraw the cluster trial and classify every claim.
    ///
    /// The margin and the decision depend on the trial. The temporal effect and
    /// the cost model do not.
    ///
    /// # Errors
    /// The claim graph does not validate.
    pub fn withdraw_cluster(&self) -> Result<CompositionVerification, FlagshipError> {
        let graph = claim_graph(self)?;
        let mut effects = BTreeMap::new();
        effects.insert(
            "cluster_trial".to_owned(),
            ObservedEffect::Withdrawn { cause: "cluster_trial.withdrawn".to_owned() },
        );
        Ok(CompositionVerification::verify(Some(&graph), &graph, &effects, None)?)
    }

    /// Export the joint combination for an independent consumer.
    ///
    /// # Errors
    /// The combination artifact cannot be encoded.
    pub fn export_decision(&self) -> Result<DecisionArtifact, FlagshipError> {
        Ok(DecisionArtifact {
            bytes: self.decision.joint.export()?,
            identity: self.decision.joint.identity().to_owned(),
        })
    }

    fn install(
        &mut self,
        temporal_effect: f64,
        temporal_variance: f64,
        cost: &BayesianLaw,
        benefit: f64,
        ctx: &ExecutionContext,
    ) -> Result<(), FlagshipError> {
        self.decision = compose(
            &self.cluster,
            temporal_effect,
            temporal_variance,
            cost.mean,
            cost.standard_deviation,
            benefit,
            ctx,
        )?;
        self.temporal_effect = temporal_effect;
        self.temporal_variance = temporal_variance;
        self.cost_mean = cost.mean;
        self.cost_sd = cost.standard_deviation;
        self.cost_draws = cost.draws;
        self.benefit = benefit;
        Ok(())
    }
}

fn retained_cost(workflow: &FlagshipWorkflow) -> BayesianLaw {
    BayesianLaw {
        mean: workflow.cost_mean,
        standard_deviation: workflow.cost_sd,
        lower_quantile: 0.0,
        upper_quantile: 0.0,
        probability_below: 0.0,
        draws: workflow.cost_draws,
    }
}

fn require_finite(benefit: f64) -> Result<(), FlagshipError> {
    if benefit.is_finite() {
        Ok(())
    } else {
        Err(FlagshipError::Route("a benefit per unit is a finite number"))
    }
}

fn recomputed(plan: &antecedent_core::recalc::RecalcPlan, stage: Stage) -> bool {
    matches!(plan.status(stage), Some(StageStatus::Recomputed { .. }))
}

fn reused(plan: &antecedent_core::recalc::RecalcPlan, stage: Stage) -> bool {
    matches!(plan.status(stage), Some(StageStatus::Reused { .. }))
}

fn fit_cluster(ctx: &ExecutionContext) -> Result<ClusterFit, FlagshipError> {
    const N: usize = 40;
    let assignment: Vec<bool> = (0..N).map(|i| i / 2 < 10).collect();
    let outcomes: Vec<f64> = (0..N)
        .map(|i| {
            let index = u32::try_from(i).unwrap_or(0);
            1.0 + 2.0 * f64::from(u8::from(assignment[i])) + 0.5 * (f64::from(index) * 0.3).sin()
        })
        .collect();
    let units: Vec<Arc<str>> = (0..N).map(|i| Arc::<str>::from(format!("unit-{i}"))).collect();
    let clusters: Vec<Arc<str>> =
        (0..N).map(|i| Arc::<str>::from(format!("cluster-{}", i / 2))).collect();
    let data = TabularData::from_f64_columns([("outcome", outcomes.as_slice())])
        .map_err(crate::CausalError::from)?;
    let query = RandomizedEffectQuery::with_design(
        RandomizationDesign::Cluster { treated_clusters: 10 },
        VariableId::from_raw(0),
        assignment,
        vec![0.5; N],
        clusters,
        units,
        ("control", "treated"),
    );
    let result = Study::tabular(data).query(query).build()?.run(ctx)?;
    let estimate = result.randomized_effect().ok_or(FlagshipError::Route(
        "the cluster trial produced no randomized effect",
    ))?;
    if estimate.assignment_design.as_ref() != "cluster" {
        return Err(FlagshipError::Route("the trial did not take the cluster assignment route"));
    }
    if !(estimate.variance_upper_bound.is_finite() && estimate.variance_upper_bound > 0.0) {
        return Err(FlagshipError::Route("the cluster trial returned no positive variance"));
    }
    Ok(ClusterFit {
        effect: estimate.effect,
        variance: estimate.variance_upper_bound,
        design: estimate.assignment_design.to_string(),
        uncertainty: estimate.uncertainty.to_string(),
    })
}

fn temporal_request(benefit: f64) -> TemporalRequest {
    let mut units = Vec::new();
    for unit_id in 0..24u64 {
        units.push(TemporalUnitWire { unit_id, histories: complete_binary_histories(unit_id) });
    }
    TemporalRequest {
        edges: vec![
            (0, 1),
            (0, 2),
            (0, 3),
            (0, 4),
            (1, 2),
            (1, 3),
            (1, 4),
            (2, 3),
            (2, 4),
            (3, 4),
        ],
        bidirected: vec![],
        selection_targets: vec![0],
        horizon: 2,
        lag_alignment: [0, 1, 2, 2, 2],
        period: (0, 48),
        units,
        snapshot_id: "flagship-temporal-source".to_owned(),
        initial_state: [0.2, 0.8],
        initial_state_id: "flagship-target-law".to_owned(),
        functional: TemporalFunctional::Effect { active: [1, 1], control: [0, 0] },
        benefit_per_unit: benefit,
        cost: 0.5,
    }
}

fn complete_binary_histories(unit_id: u64) -> Vec<TemporalHistoryWire> {
    let mut histories = Vec::new();
    for s0 in 0..2u32 {
        for a1 in 0..2u32 {
            for l2 in 0..2u32 {
                for a2 in 0..2u32 {
                    let index = u64::try_from(histories.len()).unwrap_or(0);
                    histories.push(TemporalHistoryWire {
                        time_id: index.saturating_mul(3),
                        s0,
                        a1,
                        l2,
                        a2,
                        y: binary_outcome(unit_id, a1, a2),
                    });
                }
            }
        }
    }
    histories
}

fn binary_outcome(unit_id: u64, a1: u32, a2: u32) -> f64 {
    let treated = a1 == 1 && a2 == 1;
    let on = if treated { unit_id % 2 == 0 } else { unit_id % 3 == 0 };
    if on { 1.0 } else { 0.0 }
}

fn published_temporal_variance(
    session: &TemporalSession,
    ctx: &ExecutionContext,
) -> Result<f64, FlagshipError> {
    let interval = session.candidate_interval_replay(
        &DependentIntervalConfig {
            replicates: 20,
            seed: ctx.rng.master_seed(),
            level: 0.95,
            method: IntervalMethod::Studentized,
            min_units: 20,
            max_failed_fraction: 0.05,
        },
        ctx,
    )?;
    let standard_error = interval
        .studentization
        .as_ref()
        .map(|receipt| receipt.standard_error)
        .filter(|error| error.is_finite() && *error > 0.0)
        .ok_or(FlagshipError::Route(
            "the temporal interval published no positive standard error",
        ))?;
    Ok(standard_error * standard_error)
}

fn period_contains(request: &TemporalRequest, period: (u64, u64)) -> bool {
    period.0 < period.1
        && request.units.iter().flat_map(|unit| &unit.histories).all(|history| {
            history.time_id >= period.0
                && history.time_id.checked_add(2).is_some_and(|end| end < period.1)
        })
}

fn cost_request(benefit: f64, prior_scale: f64) -> BayesianRequest {
    let mut treatment = Vec::new();
    let mut cost = Vec::new();
    for _ in 0..16 {
        for action in [0.0, 1.0] {
            for noise in [-0.25, 0.25] {
                treatment.push(action);
                cost.push(0.5 + 1.5 * action + noise);
            }
        }
    }
    BayesianRequest {
        columns: vec![("treatment".to_owned(), treatment), ("cost".to_owned(), cost)],
        edges: vec![(0, 1)],
        treatment: VariableId::from_raw(0),
        outcome: VariableId::from_raw(1),
        model: BayesianModel::Gaussian,
        inference: BayesianConfig::conjugate().n_draws(128).prior_scale(prior_scale),
        summary: PosteriorSummarySpec::default(),
        utility: UtilitySpec { benefit_per_unit: benefit, cost: 0.0 },
    }
}

fn compose(
    cluster: &ClusterFit,
    temporal_effect: f64,
    temporal_variance: f64,
    cost_mean: f64,
    cost_sd: f64,
    benefit: f64,
    ctx: &ExecutionContext,
) -> Result<MixedDecision, FlagshipError> {
    let (margin, bridge_value) = margin_of(cluster)?;
    let cost_variance = cost_sd * cost_sd;
    if !(temporal_variance.is_finite() && temporal_variance > 0.0 && cost_variance.is_finite()) {
        return Err(FlagshipError::Route(
            "a joint input is missing the variance its route published",
        ));
    }
    let dependence = LinearDependence::DeclaredIndependent;
    let inputs = [
        moment("margin", "cluster-trial", margin.mean),
        moment("temporal_drift", "temporal-panel", temporal_effect),
        moment("bayesian_cost", "bayesian-cost", cost_mean),
    ];
    let covariance = block_covariance(margin.variance, temporal_variance, cost_variance);
    let joint = propagate_linear_moments(
        PropagationOperation::WeightedSum,
        &inputs,
        &[benefit, 1.0, -1.0],
        &covariance,
        &dependence,
        ctx,
    )?;
    let mean = joint.value("mean").ok_or(FlagshipError::Route(
        "the joint combination did not return a mean",
    ))?;
    Ok(MixedDecision {
        evaluation: evaluate_mean(mean)?,
        joint,
        margin,
        dependence,
        bridge_value,
        temporal_variance,
        cost_variance,
    })
}

fn moment(id: &str, evidence_id: &str, mean: f64) -> MomentInput {
    MomentInput {
        id: id.to_owned(),
        evidence_id: evidence_id.to_owned(),
        role: InputRole::Estimate,
        mean,
    }
}

const fn block_covariance(
    margin_variance: f64,
    temporal_variance: f64,
    cost_variance: f64,
) -> [f64; 9] {
    [
        margin_variance,
        0.0,
        0.0,
        0.0,
        temporal_variance,
        0.0,
        0.0,
        0.0,
        cost_variance,
    ]
}

fn margin_of(
    cluster: &ClusterFit,
) -> Result<(antecedent_contract::bridge::PropagatedUncertainty, f64), FlagshipError> {
    let contract = margin_contract(slope_covariance(cluster.variance));
    let quantity = contract.source.quantity.clone();
    let prepared = PreparedScientificBridge::prepare(contract)?;
    let claim = source_claim(cluster.effect);
    let derived = prepared.derive(&BridgeSource { claim: &claim, quantity: &quantity })?;
    let margin = prepared
        .contract()
        .propagate_affine_uncertainty(cluster.effect, cluster.variance)?;
    Ok((margin, derived.value()))
}

fn slope_covariance(source_variance: f64) -> f64 {
    let product = SLOPE_VARIANCE * source_variance;
    if product.is_finite() && product > 0.0 { 0.25 * product.sqrt() } else { 0.0 }
}

fn margin_contract(covariance: f64) -> BridgeContract {
    let source = scientific_quantity("trial:endpoint");
    let mut target = scientific_quantity("trial:margin");
    "trial:margin".clone_into(&mut target.variable_name);
    BridgeContract {
        kind: BridgeKind::ScientificDecisionUtility,
        source: BridgeEndpoint { quantity: source, class: QuantityClass::ScientificEndpoint },
        target: BridgeEndpoint { quantity: target, class: QuantityClass::DecisionUtility },
        source_value: ValueKind::Mean,
        target_value: ValueKind::Mean,
        evidence_scope: BridgeScope {
            population_id: POPULATION.to_owned(),
            regime_id: REGIME.to_owned(),
        },
        mapping: BridgeMapping::Affine { intercept: 0.0, slope: 1.0 },
        assumptions: vec![BridgeAssumption {
            id: "endpoint-to-margin".to_owned(),
            kind: AssumptionKind::UtilityMapping,
            statement: "the declared affine map carries the endpoint into the decision margin"
                .to_owned(),
            evidence: Some("margin-validation".to_owned()),
        }],
        transport: None,
        mapping_uncertainty: Some(antecedent_contract::bridge::MappingUncertainty {
            id: "margin-map".to_owned(),
            intercept_variance: 0.0,
            slope_variance: SLOPE_VARIANCE,
            intercept_slope_covariance: 0.0,
        }),
        source_dependence: Some(SourceMappingDependence::SlopeCovariance {
            covariance,
            evidence: "cluster-trial-rows".to_owned(),
        }),
        estimand_bridge: None,
    }
}

fn scientific_quantity(variable: &str) -> ScientificQuantity {
    ScientificQuantity {
        variable_id: variable.to_owned(),
        variable_name: variable.to_owned(),
        role: QuantityRole::Outcome,
        units: MARGIN_UNITS.to_owned(),
        population_id: POPULATION.to_owned(),
        regime_id: REGIME.to_owned(),
        horizon: 0,
        functional_id: "mean".to_owned(),
        conditioning: Vec::new(),
        transform_id: "identity".to_owned(),
    }
}

fn source_claim(value: f64) -> ClaimEnvelope {
    let digest = |byte: u8| SemanticDigest::from_bytes([byte; 32]);
    ClaimEnvelope::new(
        digest(1),
        ContractIdentities::new(
            digest(2),
            digest(3),
            Some(digest(4)),
            Some(digest(5)),
            digest(6),
            digest(7),
            digest(8),
        ),
        ClaimKind::Point,
        Some(value),
        Some(Arc::from(MARGIN_UNITS)),
        ReasoningView::new(
            SlotAvailability::unavailable("cluster-trial"),
            SlotAvailability::unavailable("cluster-trial"),
            SlotAvailability::unavailable("cluster-trial"),
            SlotAvailability::unavailable("cluster-trial"),
        ),
        ClaimDomains::new(
            DomainStatus::Identified,
            DomainStatus::Supported,
            DomainStatus::Evaluated,
        ),
        None,
        vec![Arc::<str>::from("cluster-trial")],
    )
}

fn evaluate_mean(mean: f64) -> Result<DecisionResult, FlagshipError> {
    let quantity = decision_quantity();
    let source = MeanSource {
        coordinates: vec![quantity.clone()],
        means: vec![mean],
        provider_id: "flagship".to_owned(),
        snapshot_id: "mixed-decision".to_owned(),
        causal_contract_id: "flagship-joint".to_owned(),
        rng_id: "none:joint-mean".to_owned(),
    };
    Ok(antecedent_design::decision_eval::evaluate_contract_on_means(
        &decision_contract(quantity),
        &source,
    )?)
}

fn decision_quantity() -> ScientificQuantity {
    ScientificQuantity {
        variable_id: "decision:net_benefit".to_owned(),
        variable_name: "decision:net_benefit".to_owned(),
        role: QuantityRole::Outcome,
        units: MARGIN_UNITS.to_owned(),
        population_id: POPULATION.to_owned(),
        regime_id: "decision".to_owned(),
        horizon: 0,
        functional_id: "mean".to_owned(),
        conditioning: Vec::new(),
        transform_id: "identity".to_owned(),
    }
}

fn decision_contract(quantity: ScientificQuantity) -> DecisionContract {
    DecisionContract {
        actions: vec![
            DecisionAction {
                id: "decline".to_owned(),
                kind: ActionKind::Intervention,
                inputs: vec![quantity.clone()],
                utility: UtilityExpr::Mul(
                    Box::new(UtilityExpr::Const(0.0)),
                    Box::new(UtilityExpr::Input(0)),
                ),
            },
            DecisionAction {
                id: "act".to_owned(),
                kind: ActionKind::Intervention,
                inputs: vec![quantity],
                utility: UtilityExpr::Input(0),
            },
        ],
        utility_units: MARGIN_UNITS.to_owned(),
        criterion: DecisionCriterion::PosteriorExpectedUtility,
        constraints: vec![],
        target_population: POPULATION.to_owned(),
        horizon: 0,
        structural_policy: StructuralPolicy::ReportOnly,
    }
}

fn claim_graph(workflow: &FlagshipWorkflow) -> Result<CompositionGraph, FlagshipError> {
    let mean = workflow.decision.joint.value("mean").ok_or(FlagshipError::Route(
        "the joint combination did not return a mean",
    ))?;
    let mut graph = CompositionGraph::new();
    let nodes = [
        ("cluster_trial", NodeKind::Claim, workflow.cluster.effect.to_bits()),
        ("temporal_drift", NodeKind::Claim, workflow.temporal_effect.to_bits()),
        ("bayesian_cost", NodeKind::Claim, workflow.cost_mean.to_bits()),
        ("margin_bridge", NodeKind::Bridge, workflow.decision.margin.mean.to_bits()),
        ("decision", NodeKind::Decision, mean.to_bits()),
    ];
    for (key, kind, bits) in nodes {
        graph.add_node(claim_node(key, kind, bits)).map_err(graph_error)?;
    }
    let edges = [
        ("margin_bridge", "cluster_trial", DependencyKind::Data),
        ("decision", "margin_bridge", DependencyKind::Evidence),
        ("decision", "temporal_drift", DependencyKind::Evidence),
        ("decision", "bayesian_cost", DependencyKind::Evidence),
    ];
    for (downstream, upstream, kind) in edges {
        graph.add_edge(downstream, upstream, kind).map_err(graph_error)?;
    }
    Ok(graph)
}

fn claim_node(key: &str, kind: NodeKind, bits: u64) -> CompositionNode {
    let mut writer = CanonicalWriter::new("antecedent.flagship_mixed.node");
    writer.text(key).word(bits);
    CompositionNode::new(key, kind, SemanticDigest::from_bytes(writer.finish()))
}

fn graph_error(error: super::GraphError) -> FlagshipError {
    FlagshipError::Graph(error.to_string())
}

/// Replay `bytes` and evaluate the decision on the replayed mean.
///
/// `expected_identity`, when set, is the identity retained independently of
/// the bytes.
///
/// # Errors
/// The artifact does not replay, or the decision contract refuses the mean.
pub fn load_decision(
    bytes: &[u8],
    expected_identity: Option<&str>,
    ctx: &ExecutionContext,
) -> Result<LoadedDecision, FlagshipError> {
    let consumed = consume_joint_law_propagation_artifact(bytes, expected_identity, ctx)?;
    let mean = consumed
        .value("mean")
        .ok_or(FlagshipError::Route("the replayed combination has no mean"))?;
    let variance = consumed
        .value("variance")
        .ok_or(FlagshipError::Route("the replayed combination has no variance"))?;
    Ok(LoadedDecision { mean, variance, evaluation: evaluate_mean(mean)? })
}
