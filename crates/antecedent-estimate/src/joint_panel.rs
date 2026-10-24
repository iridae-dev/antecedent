//! Finite-lag, explicitly configured multichannel nonlinear Gaussian panel mechanisms.
//!
//! This numerical component fits the reviewed mechanism, not a causal contrast. Global
//! parameters are shared across groups; Gaussian random intercepts are integrated out of
//! the stationary AR(1) likelihood and recovered conditionally. Every response evaluation
//! uses one complete parameter/random-effect draw across all programs. Identification,
//! scientific target binding, artifact authority and licensing belong to the caller's
//! checked compiler, never to the existence of a fit or a finite response.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

/// Aligned four-component reference-world decompositions of panel responses.
pub mod decomposition;
/// Scalar expressions derived from retained fitted panel mechanisms and schedules.
pub mod expression;
mod inference;
mod likelihood;
/// Specification-conditional split of decomposed temporal laws.
pub mod selection_uncertainty;
mod streaming;
pub use streaming::{
    PanelProjectedLaw, PanelStreamReceipt, project_panel_law, project_temporal_law,
    stream_panel_law, stream_temporal_law,
};
#[cfg(test)]
mod tests;
/// Model-conditional piecewise/smooth change attempts over this same panel mechanism.
pub mod time_varying;

use std::sync::Arc;

use antecedent_core::{CanonicalWriter, ExecutionContext, SemanticDigest};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// One stable channel identity; display names are not scientific identities.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PanelChannel {
    /// Stable source-variable identity.
    pub id: String,
}
/// The declared feature input scale; a source variable name never silently changes meaning.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PanelFeatureInputs {
    /// Per-channel constrained carryover followed by the shared Michaelis-Menten saturation.
    LaggedSaturatedChannels,
}
/// One prespecified programmable numerical feature with a fitted coefficient.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PanelInteractionSpec {
    /// Stable interaction identity.
    pub id: String,
    /// Explicit transformed input binding.
    pub inputs: PanelFeatureInputs,
    /// Existing portable expression syntax. Every variable reads that channel's
    /// lag-weighted, Michaelis-Menten saturated value, never raw exposure or a second lag.
    pub expression: antecedent_io_core::query_wire::ValueExprWire,
}
impl PanelInteractionSpec {
    /// Bind the existing programmable expression to lag-weighted, saturated channel features.
    ///
    /// # Errors
    /// Refuses expressions that cannot be encoded portably; source-axis/type checks occur at compilation.
    pub fn from_saturated_features(
        id: impl Into<String>,
        expression: &antecedent_core::ValueExpr,
    ) -> Result<Self, PanelError> {
        let expression = antecedent_io_core::query_wire::ValueExprWire::from_domain(expression)
            .map_err(|error| invalid(&error.to_string()))?;
        Ok(Self { id: id.into(), inputs: PanelFeatureInputs::LaggedSaturatedChannels, expression })
    }
}
/// One channel's effectiveness and constrained finite carryover.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PanelChannelParameters {
    /// Exact source channel identity.
    pub id: String,
    /// Saturated response coefficient.
    pub coefficient: f64,
    /// Current/lag-one/lag-two weights on the simplex.
    pub lag: [f64; 3],
}
/// One coefficient on an explicitly declared channel product.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PanelInteractionParameter {
    /// Exact declared interaction identity.
    pub id: String,
    /// Product coefficient.
    pub coefficient: f64,
}
/// Future assignments with an explicit channel key.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PanelChannelProgram {
    /// Exact source channel identity.
    pub id: String,
    /// Nonnegative assignments on the shared future time axis.
    pub values: Vec<f64>,
}
/// Actual pre-period assignments at a declared scenario origin.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PanelChannelHistory {
    /// Exact source channel identity.
    pub id: String,
    /// Offsets -2 and -1, oldest first.
    pub values: [f64; 2],
}

/// Explicit balanced or unbalanced geographic panel; each vector has contiguous cadence.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PanelData {
    /// Immutable ordered channel axis.
    pub channels: Vec<PanelChannel>,
    /// Prespecified programmable features on the explicitly bound saturated channel axis.
    pub interactions: Vec<PanelInteractionSpec>,
    /// Groups, in retained random-effect coordinate order.
    pub groups: Vec<PanelGroup>,
    /// Finite seasonal dummy levels; level zero is the reference.
    pub seasons: usize,
    /// Reviewed temporal mechanism is two; zero drops carryover and AR dependence for the static specialization.
    pub lag_order: usize,
}

/// One independent group, with serial dependence within its contiguous outcome periods.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PanelGroup {
    /// Stable geographic identity.
    pub id: String,
    /// Per-channel actual assignments at offsets -2 and -1, in the declared channel order.
    pub history: Vec<[f64; 2]>,
    /// Nonnegative assignments, in the declared channel order.
    pub channels: Vec<Vec<f64>>,
    /// Outcome at each period.
    pub outcome: Vec<f64>,
    /// One declared observable confounder at each period.
    pub confounder: Vec<f64>,
    /// Seasonal level at each period.
    pub season: Vec<usize>,
}

/// Complete mechanism parameters, with no omitted lag, interaction or dependence term.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PanelParameters {
    /// Global baseline intercept.
    pub intercept: f64,
    /// Channel coefficients and lag simplexes, in the exact declared channel order.
    pub channels: Vec<PanelChannelParameters>,
    /// Coefficients in the exact declared sparse-interaction order.
    pub interactions: Vec<PanelInteractionParameter>,
    /// Additive confounder coefficient.
    pub confounder: f64,
    /// Seasonal effects; first entry is exactly zero.
    pub season_effects: Vec<f64>,
    /// Shared strictly positive Michaelis-Menten half-saturation k.
    pub half_saturation: f64,
    /// Stationary within-group AR(1) correlation, strictly between -1 and 1; exactly zero for the static specialization.
    pub rho: f64,
    /// Positive within-group innovation standard deviation.
    pub innovation_sd: f64,
    /// Positive shared Gaussian random-intercept standard deviation.
    pub group_sd: f64,
}

/// A single global mechanism and its jointly retained group effects.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PanelDraw {
    /// Same parameter realization for every group and every intervention world.
    pub parameters: PanelParameters,
    /// Random intercepts in source group order, never silently filled for unknown groups.
    pub group_intercepts: Vec<f64>,
}

/// Bounded local likelihood optimizer; convergence is a numerical diagnostic.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PanelOptimizerOptions {
    /// Maximum accepted iterations.
    pub max_iterations: usize,
    /// Infinity norm of the marginal log-likelihood score required for convergence.
    pub gradient_tolerance: f64,
}
impl Default for PanelOptimizerOptions {
    fn default() -> Self {
        Self { max_iterations: 800, gradient_tolerance: 1e-5 }
    }
}

/// Explicit prior under the same marginal Gaussian panel likelihood.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PanelPrior {
    /// Zero-mean Normal standard deviation for baseline, channels, interaction, z and season effects.
    pub coefficient_sd: f64,
    /// Mean of log(k).
    pub log_half_saturation_mean: f64,
    /// Standard deviation of log(k).
    pub log_half_saturation_sd: f64,
    /// Per-channel Dirichlet concentrations; empty means uniform [1,1,1] for every channel.
    pub lag: Vec<[f64; 3]>,
    /// Half-Normal scale for the innovation standard deviation.
    pub innovation_scale: f64,
    /// Half-Normal scale for the random-intercept standard deviation.
    pub group_scale: f64,
}
impl Default for PanelPrior {
    fn default() -> Self {
        Self {
            coefficient_sd: 10.0,
            log_half_saturation_mean: 0.0,
            log_half_saturation_sd: 1.0,
            lag: Vec::new(),
            innovation_scale: 2.0,
            group_scale: 2.0,
        }
    }
}

/// Fixed random-walk policy and explicit convergence thresholds.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PanelSamplingOptions {
    /// Independent chains.
    pub chains: usize,
    /// Discarded iterations per chain.
    pub warmup: usize,
    /// Retained draws per chain.
    pub draws: usize,
    /// Proposal scale after local posterior curvature preconditioning when available.
    pub proposal_scale: f64,
    /// Maximum rank/folded split R-hat for a successful diagnostic.
    pub max_rhat: f64,
    /// Minimum bulk and tail ESS for a successful diagnostic.
    pub min_ess: f64,
}
impl Default for PanelSamplingOptions {
    fn default() -> Self {
        Self {
            chains: 4,
            warmup: 2000,
            draws: 2000,
            proposal_scale: 0.35,
            max_rhat: 1.05,
            min_ess: 100.0,
        }
    }
}

/// Statistical realization, distinct from any causal identification class.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum PanelInference {
    /// Local group-cluster sandwich parameter approximation, not a posterior or licensed interval.
    Frequentist {
        /// Requested approximate draws; invalid transformed slots remain None.
        draws: usize,
    },
    /// Hierarchical Gaussian/AR parametric bootstrap, repeating the complete
    /// fitting procedure. Temporal fits also repeat change and smoothing selection.
    /// This is model-based sampling uncertainty, not a posterior or a calibrated interval.
    FrequentistBootstrap {
        /// Requested complete-procedure replicate slots; failed fits remain None.
        draws: usize,
    },
    /// Same likelihood with explicit priors and conditional Gaussian group effects.
    Bayesian {
        /// Model prior.
        prior: PanelPrior,
        /// Bounded sampler and convergence policy.
        sampler: PanelSamplingOptions,
    },
}

/// Full numerical fit policy; changing any entry invalidates the fitted dependency.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PanelFitOptions {
    /// Local optimizer policy (also determines the Bayesian starting point).
    pub optimizer: PanelOptimizerOptions,
    /// Frequentist or Bayesian realization.
    pub inference: PanelInference,
    /// Optional complete starting parameter, validated against this design.
    pub initial: Option<PanelParameters>,
}

/// Explicit numerical and uncertainty facts, never statistical promotion.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PanelFitDiagnostics {
    /// Actual marginal Gaussian log likelihood at the retained point.
    pub log_likelihood: f64,
    /// Local optimizer's maximum absolute score.
    pub score_norm: f64,
    /// Accepted likelihood iterations.
    pub iterations: usize,
    /// Local score tolerance was met.
    pub optimizer_converged: bool,
    /// Bounded line search exhausted before meeting tolerance.
    pub line_search_exhausted: bool,
    /// Curvature was positive definite at the retained point.
    pub curvature_available: bool,
    /// Global parameter dimension.
    pub parameter_count: usize,
    /// Upper bound on the group-cluster covariance rank, or None for Bayesian inference.
    pub cluster_rank_bound: Option<usize>,
    /// Maximum posterior rank/folded R-hat, if sampled.
    pub max_rhat: Option<f64>,
    /// Minimum posterior bulk ESS, if sampled.
    pub min_bulk_ess: Option<f64>,
    /// Minimum posterior tail ESS, if sampled.
    pub min_tail_ess: Option<f64>,
    /// Posterior convergence thresholds and actual chain movement passed.
    pub posterior_converged: Option<bool>,
    /// Requested slots that could not produce a finite mechanism.
    pub failed_draws: usize,
    /// Full uncertainty meaning; parameter-fit and conditional group-effect laws are separate.
    pub uncertainty: String,
    /// The likelihood fit itself grants no causal permission.
    pub claim: String,
    /// Calibration remains unmeasured until a separate whole-procedure campaign.
    pub calibration: String,
}

/// Retained fit; all draws preserve one shared mechanism across response worlds.
#[derive(Clone, Debug)]
pub struct PanelFit {
    /// Point mechanism and conditional mean random effects.
    pub point: PanelDraw,
    /// Retained requested slots, preserving failures and chain/draw order.
    pub draws: Vec<Option<PanelDraw>>,
    /// Actual numerical and statistical scope facts.
    pub diagnostics: PanelFitDiagnostics,
    /// Identity of every source group, observed column, cadence and pre-period history.
    pub source_identity: SemanticDigest,
    /// Exact inference policy used.
    pub inference: PanelInference,
    /// Admitted storage for this fit and its draw slots.
    retention: Arc<antecedent_core::execution::AdmissionLease>,
    retained_volume: u64,
}
impl PanelFit {
    /// Conservatively admitted retained volume, not allocator bytes or RSS.
    #[must_use]
    pub fn retained_bytes(&self) -> u64 {
        self.retention.observe_current();
        self.retained_volume
    }
}

/// Explicit covariates, seasons and histories for one group's response scenario.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PanelScenario {
    /// Must identify an existing fitted group.
    pub group: String,
    /// Explicit per-channel histories keyed by channel identity.
    pub history: Vec<PanelChannelHistory>,
    /// Declared covariate at every future response time; never extrapolated implicitly.
    pub confounder: Vec<f64>,
    /// Declared seasonal level at every future response time.
    pub season: Vec<usize>,
}

/// Prespecified multichannel schedule, not an optimization/search certificate.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PanelProgram {
    /// Stable program/world identity supplied by the checked caller.
    pub id: String,
    /// Explicit schedules keyed by channel identity; all share one future time axis.
    pub channels: Vec<PanelChannelProgram>,
    /// Exactly one explicit scenario for every fitted group.
    pub scenarios: Vec<PanelScenario>,
}

/// One program's conditional mean response, uniformly averaged over the retained groups.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PanelResponse {
    /// Program identity.
    pub program: String,
    /// Mean response by future time. Outcome innovations are integrated out.
    pub means: Vec<f64>,
}

/// Typed data, numerical, resource and cancellation failures.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum PanelError {
    /// Shape, parameter or declared policy mismatch.
    #[error("joint panel invalid input: {0}")]
    Invalid(String),
    /// Failed numerical likelihood, curvature or sampling calculation.
    #[error("joint panel numerical failure: {0}")]
    Numerical(String),
    /// Shared admission refused an allocation.
    #[error("joint panel resource refusal: {0}")]
    Resource(String),
    /// Cancellation remains distinct from failed identification or convergence.
    #[error("joint panel cancelled")]
    Cancelled,
}

fn invalid(message: &str) -> PanelError {
    PanelError::Invalid(message.into())
}
pub(super) fn check_memory(ctx: &ExecutionContext, bytes: u64) -> Result<(), PanelError> {
    if ctx.memory_limit_bytes().is_some_and(|limit| bytes > limit) {
        return Err(PanelError::Resource(format!(
            "requires {bytes} admitted bytes above the declared hard memory limit"
        )));
    }
    Ok(())
}
fn check_cancelled(ctx: &ExecutionContext) -> Result<(), PanelError> {
    if ctx.cancellation.is_cancelled() { Err(PanelError::Cancelled) } else { Ok(()) }
}

impl PanelData {
    fn channel_variables(&self) -> Result<Vec<antecedent_core::VariableId>, PanelError> {
        self.channels
            .iter()
            .map(|channel| {
                channel
                    .id
                    .parse::<u32>()
                    .map(antecedent_core::VariableId::from_raw)
                    .map_err(|_| invalid("channel identities must be raw source VariableIds"))
            })
            .collect()
    }

    /// Check explicit group axes, two histories and contiguous within-group vectors.
    ///
    /// # Errors
    /// Refuses ambiguous cadence, missing/duplicate group IDs, nonfinite values and invalid dimensions.
    pub fn validate(&self) -> Result<(), PanelError> {
        if !(2..=1024).contains(&self.groups.len())
            || !(1..=24).contains(&self.seasons)
            || !matches!(self.lag_order, 0 | 2)
        {
            return Err(invalid(
                "requires 2..=1024 groups, 1..=24 seasons and lag order zero or two",
            ));
        }
        if self.channels.is_empty() {
            return Err(invalid("at least one explicitly declared channel is required"));
        }
        let mut channel_ids = std::collections::BTreeSet::new();
        for channel in &self.channels {
            if channel.id.trim().is_empty() || !channel_ids.insert(channel.id.as_str()) {
                return Err(invalid("empty or duplicate channel identity"));
            }
        }
        let mut interaction_ids = std::collections::BTreeSet::new();
        let mut expressions = Vec::new();
        let axis = self.channel_variables()?;
        for interaction in &self.interactions {
            if interaction.id.trim().is_empty()
                || !interaction_ids.insert(&interaction.id)
                || expressions.contains(&interaction.expression)
            {
                return Err(invalid("empty/duplicate programmable feature identity or expression"));
            }
            expressions.push(interaction.expression.clone());
            antecedent_model::program::NodeProgram::compile_columns(
                &axis,
                &interaction.expression.to_domain(),
            )
            .map_err(|error| invalid(&format!("invalid programmable feature: {error:?}")))?;
        }
        let mut ids = std::collections::BTreeSet::new();
        for group in &self.groups {
            let n = group.outcome.len();
            if group.id.trim().is_empty()
                || !ids.insert(&group.id)
                || !(3..=1_000_000).contains(&n)
                || group.channels.len() != self.channels.len()
                || group.history.len() != self.channels.len()
                || group.channels.iter().any(|values| values.len() != n)
                || group.confounder.len() != n
                || group.season.len() != n
                || group
                    .channels
                    .iter()
                    .flatten()
                    .chain(group.history.iter().flatten())
                    .any(|value| !value.is_finite() || *value < 0.0)
                || group.outcome.iter().chain(&group.confounder).any(|value| !value.is_finite())
                || group.season.iter().any(|&value| value >= self.seasons)
            {
                return Err(invalid("group IDs, finite values, cadence or vector dimensions"));
            }
        }
        Ok(())
    }

    /// Canonical source identity including order, histories and each statistical input.
    ///
    /// # Errors
    /// Returns validation errors rather than giving malformed data a scientific dependency identity.
    pub fn identity(&self) -> Result<SemanticDigest, PanelError> {
        self.validate()?;
        let mut writer = CanonicalWriter::new("antecedent.estimate.joint_panel.source.v2");
        writer.word(self.seasons as u64).word(self.lag_order as u64).word(self.groups.len() as u64);
        writer.word(self.channels.len() as u64);
        for channel in &self.channels {
            writer.text(&channel.id);
        }
        writer.word(self.interactions.len() as u64);
        for interaction in &self.interactions {
            writer.text(&interaction.id).word(0).bytes(
                &antecedent_io_core::policy_program_artifact::value_expression_identity(
                    &interaction.expression.to_domain(),
                )
                .map_err(|error| invalid(&error.to_string()))?,
            );
        }
        for group in &self.groups {
            writer.text(&group.id);
            for values in group
                .history
                .iter()
                .map(|values| &values[..])
                .chain(group.channels.iter().map(Vec::as_slice))
                .chain([group.outcome.as_slice(), group.confounder.as_slice()])
            {
                writer.word(values.len() as u64);
                for value in values {
                    writer.word(value.to_bits());
                }
            }
            writer.word(group.season.len() as u64);
            for &season in &group.season {
                writer.word(season as u64);
            }
        }
        Ok(SemanticDigest::from_bytes(writer.finish()))
    }
}

/// Validate fit policy and any initial mechanism without fitting or sampling.
///
/// # Errors
/// Refuses malformed source, optimizer/sampler/prior policy or initial parameters.
pub fn validate_fit_options(data: &PanelData, options: &PanelFitOptions) -> Result<(), PanelError> {
    data.validate()?;
    inference::validate_options(options)?;
    if let PanelInference::Bayesian { prior, .. } = &options.inference {
        if !prior.lag.is_empty() && prior.lag.len() != data.channels.len() {
            return Err(invalid("Dirichlet priors must follow the complete channel axis"));
        }
    }
    if options.optimizer.max_iterations == 0
        || options.optimizer.max_iterations > 100_000
        || !options.optimizer.gradient_tolerance.is_finite()
        || options.optimizer.gradient_tolerance <= 0.0
    {
        return Err(invalid("optimizer iteration bound or score tolerance"));
    }
    if let Some(initial) = &options.initial {
        likelihood::validate_parameters(data, initial)?;
    }
    Ok(())
}

/// Validate explicit intervention/scenario axes without fitting, simulation or output allocation.
///
/// # Errors
/// Refuses missing/foreign groups, duplicate worlds, nonfinite values and mismatched future axes.
pub fn validate_panel_programs(
    data: &PanelData,
    programs: &[PanelProgram],
) -> Result<(), PanelError> {
    data.validate()?;
    let mut ids = std::collections::BTreeSet::new();
    if programs.is_empty() {
        return Err(invalid("at least one prespecified program is required"));
    }
    for program in programs {
        let horizon = program.horizon();
        if program.id.trim().is_empty()
            || !ids.insert(&program.id)
            || horizon == 0
            || program.channels.len() != data.channels.len()
            || program.channels.iter().any(|channel| channel.values.len() != horizon)
            || program.scenarios.len() != data.groups.len()
            || program
                .channels
                .iter()
                .flat_map(|channel| &channel.values)
                .any(|value| !value.is_finite() || *value < 0.0)
            || !same_keys(
                data.channels.iter().map(|channel| channel.id.as_str()),
                program.channels.iter().map(|channel| channel.id.as_str()),
            )
        {
            return Err(invalid("program identities, horizon, assignments or group scenarios"));
        }
        let mut scenario_ids = std::collections::BTreeSet::new();
        for scenario in &program.scenarios {
            if !data.groups.iter().any(|group| group.id == scenario.group)
                || !scenario_ids.insert(&scenario.group)
                || scenario.confounder.len() != horizon
                || scenario.season.len() != horizon
                || scenario.season.iter().any(|&season| season >= data.seasons)
                || scenario.confounder.iter().any(|value| !value.is_finite())
                || !same_keys(
                    data.channels.iter().map(|channel| channel.id.as_str()),
                    scenario.history.iter().map(|channel| channel.id.as_str()),
                )
                || scenario
                    .history
                    .iter()
                    .flat_map(|channel| &channel.values)
                    .any(|value| !value.is_finite() || *value < 0.0)
            {
                return Err(invalid("scenario covariates, seasons, histories or group identities"));
            }
        }
    }
    Ok(())
}

impl PanelFitOptions {
    /// Canonical dependency identity for optimizer, inference, priors and initial values.
    ///
    /// # Errors
    /// Refuses invalid source/policy; execution seed and numerical kernel policy are separate caller dependencies.
    pub fn identity(&self, data: &PanelData) -> Result<SemanticDigest, PanelError> {
        validate_fit_options(data, self)?;
        let mut writer = CanonicalWriter::new("antecedent.estimate.joint_panel.fit_policy.v3");
        writer
            .word(self.optimizer.max_iterations as u64)
            .word(self.optimizer.gradient_tolerance.to_bits());
        match &self.inference {
            PanelInference::Frequentist { draws } => {
                writer.word(0).word(*draws as u64);
            }
            PanelInference::FrequentistBootstrap { draws } => {
                writer.word(2).word(*draws as u64);
            }
            PanelInference::Bayesian { prior, sampler } => {
                writer
                    .word(1)
                    .word(sampler.chains as u64)
                    .word(sampler.warmup as u64)
                    .word(sampler.draws as u64)
                    .word(sampler.proposal_scale.to_bits())
                    .word(sampler.max_rhat.to_bits())
                    .word(sampler.min_ess.to_bits());
                for value in [
                    prior.coefficient_sd,
                    prior.log_half_saturation_mean,
                    prior.log_half_saturation_sd,
                    prior.innovation_scale,
                    prior.group_scale,
                ]
                .iter()
                .chain(prior.lag.iter().flatten())
                {
                    writer.word(value.to_bits());
                }
            }
        }
        if let Some(initial) = &self.initial {
            writer.word(1);
            for value in likelihood::encode(data, initial)? {
                writer.word(value.to_bits());
            }
        } else {
            writer.word(0);
        }
        Ok(SemanticDigest::from_bytes(writer.finish()))
    }
}

/// Fit this numerical mechanism under caller-owned checked identification/target policy.
///
/// # Errors
/// Explicit input, numerical, shared resource or cancellation failures. Failed optimizer/
/// convergence diagnostics remain in the fit; they never become a licensed uncertainty claim.
pub fn fit_panel(
    data: &PanelData,
    options: &PanelFitOptions,
    ctx: &ExecutionContext,
) -> Result<PanelFit, PanelError> {
    ctx.try_map_indexed(1, |_, worker| inference::fit(data, options, worker))
        .map_err(map_execution_error)?
        .pop()
        .ok_or_else(|| invalid("missing admitted fit worker"))
}

/// Evaluate all declared programs from one complete mechanism/random-effect draw.
///
/// No fitting, identification or implicit future-covariate/initial-law selection occurs.
/// This returns model-conditional mean response, never predictive innovations or a license.
///
/// # Errors
/// Refuses unknown/missing group scenarios, changed parameter dimensions, malformed future
/// covariates/history, cancellation and failed shared admission.
pub fn evaluate_panel_programs(
    data: &PanelData,
    draw: &PanelDraw,
    programs: &[PanelProgram],
    ctx: &ExecutionContext,
) -> Result<Vec<PanelResponse>, PanelError> {
    ctx.try_map_indexed(1, |_, worker| evaluate_inner(data, draw, programs, worker))
        .map_err(map_execution_error)?
        .pop()
        .ok_or_else(|| invalid("missing admitted response worker"))
}
fn evaluate_inner(
    data: &PanelData,
    draw: &PanelDraw,
    programs: &[PanelProgram],
    ctx: &ExecutionContext,
) -> Result<Vec<PanelResponse>, PanelError> {
    validate_panel_programs(data, programs)?;
    let compiled = likelihood::CompiledPanel::new(data)?;
    let data = &compiled;
    let bytes = response_storage_bytes(programs)?
        .checked_add(response_workspace_bytes(data)?)
        .ok_or_else(|| invalid("response peak size overflow"))?;
    check_memory(ctx, bytes)?;
    let _working =
        ctx.acquire_admission(0, bytes).map_err(|error| PanelError::Resource(error.to_string()))?;
    evaluate_bound(data, draw, programs, ctx)
}
fn evaluate_bound(
    data: &likelihood::CompiledPanel<'_>,
    draw: &PanelDraw,
    programs: &[PanelProgram],
    ctx: &ExecutionContext,
) -> Result<Vec<PanelResponse>, PanelError> {
    likelihood::validate_parameters(data, &draw.parameters)?;
    if draw.group_intercepts.len() != data.groups.len()
        || draw.group_intercepts.iter().any(|value| !value.is_finite())
    {
        return Err(invalid("random-effect axis differs from source groups"));
    }
    check_cancelled(ctx)?;
    let mut output = Vec::with_capacity(programs.len());
    for program in programs {
        check_cancelled(ctx)?;
        let horizon = program.horizon();
        let mut means = vec![0.0; horizon];
        for scenario in &program.scenarios {
            let Some(group_index) = data.groups.iter().position(|group| group.id == scenario.group)
            else {
                return Err(invalid("program refers to a group with no fitted random effect"));
            };
            for (time, mean) in means.iter_mut().enumerate() {
                check_cancelled(ctx)?;
                *mean +=
                    likelihood::program_response(data, &draw.parameters, program, scenario, time)
                        + draw.group_intercepts[group_index];
            }
        }
        for mean in &mut means {
            *mean /= data.groups.len() as f64;
            if !mean.is_finite() {
                return Err(PanelError::Numerical("response overflow".into()));
            }
        }
        output.push(PanelResponse { program: program.id.clone(), means });
    }
    Ok(output)
}

impl PanelProgram {
    /// Declared common horizon; validation checks every schedule against it.
    #[must_use]
    pub fn horizon(&self) -> usize {
        self.channels.first().map_or(0, |channel| channel.values.len())
    }
}
fn same_keys<'a>(
    expected: impl Iterator<Item = &'a str>,
    actual: impl Iterator<Item = &'a str>,
) -> bool {
    let expected: std::collections::BTreeSet<_> = expected.collect();
    let actual: Vec<_> = actual.collect();
    actual.len() == expected.len()
        && actual.iter().copied().collect::<std::collections::BTreeSet<_>>() == expected
}

fn map_execution_error(
    error: antecedent_core::execution::MapExecutionError<PanelError>,
) -> PanelError {
    match error {
        antecedent_core::execution::MapExecutionError::Work(error) => error,
        antecedent_core::execution::MapExecutionError::Admission(
            antecedent_core::execution::AdmissionError::Cancelled,
        ) => PanelError::Cancelled,
        antecedent_core::execution::MapExecutionError::Admission(error) => {
            PanelError::Resource(error.to_string())
        }
    }
}

/// Apply already compiled linear response reductions in the retained world order.
///
/// This numerical reduction carries no support, identification or uncertainty license.
///
/// # Errors
/// Refuses missing world weights, mismatched horizons, nonfinite inputs or arithmetic overflow.
pub fn project_panel_responses(
    rows: &[PanelResponse],
    weights: &[Vec<f64>],
) -> Result<Vec<f64>, PanelError> {
    if rows.len() != weights.len() {
        return Err(invalid("response projection must cover the exact ordered world axis"));
    }
    rows.iter()
        .zip(weights)
        .map(|(row, weights)| {
            if row.means.len() != weights.len()
                || row.means.iter().chain(weights).any(|value| !value.is_finite())
            {
                return Err(invalid("response projection must cover each finite horizon exactly"));
            }
            let value =
                row.means.iter().zip(weights).map(|(mean, weight)| mean * weight).sum::<f64>();
            if value.is_finite() {
                Ok(value)
            } else {
                Err(PanelError::Numerical("response projection overflow".into()))
            }
        })
        .collect()
}

/// Aligned complete-mechanism response slots, retaining numerical failures in place.
pub type PanelAlignedResponses = Vec<Option<Vec<PanelResponse>>>;

/// A complete response execution with its original admitted output storage.
#[derive(Debug)]
pub struct PanelLawResponses {
    point: Vec<PanelResponse>,
    draws: PanelAlignedResponses,
    retention: Arc<antecedent_core::execution::AdmissionLease>,
    retained_volume: u64,
}
impl PanelLawResponses {
    /// Point responses in the declared world order.
    #[must_use]
    pub fn point(&self) -> &[PanelResponse] {
        &self.point
    }
    /// Aligned draw responses, preserving failed slots.
    #[must_use]
    pub fn draws(&self) -> &[Option<Vec<PanelResponse>>] {
        &self.draws
    }
    /// Actual admitted retained reservation, not allocator bytes or RSS.
    #[must_use]
    pub fn retained_bytes(&self) -> u64 {
        self.retention.observe_current();
        self.retained_volume
    }
    /// Transfer numerical buffers and their existing output lease to a retained consumer.
    #[must_use]
    pub fn into_parts(
        self,
    ) -> (Vec<PanelResponse>, PanelAlignedResponses, Arc<antecedent_core::execution::AdmissionLease>)
    {
        (self.point, self.draws, self.retention)
    }
}

/// Evaluate a complete fitted law with one shared compiled expression plan.
///
/// This numerical component preserves the caller's world bindings and failed slots;
/// scientific authorization remains with the checked compiler that supplied the law.
///
/// # Errors
/// Input mismatch, point numerical failure, shared resource refusal or cancellation.
pub fn evaluate_panel_law(
    data: &PanelData,
    point: &PanelDraw,
    draws: &[Option<PanelDraw>],
    programs: &[PanelProgram],
    ctx: &ExecutionContext,
) -> Result<PanelLawResponses, PanelError> {
    ctx.try_map_indexed(1, |_, worker| {
        evaluate_law(data, point, draws, programs, worker, evaluate_bound)
    })
    .map_err(map_execution_error)?
    .pop()
    .ok_or_else(|| invalid("missing admitted law worker"))
}

fn response_storage_bytes(programs: &[PanelProgram]) -> Result<u64, PanelError> {
    let bytes =
        programs.iter().try_fold(std::mem::size_of::<Vec<PanelResponse>>(), |sum, program| {
            program
                .horizon()
                .checked_mul(8)
                .and_then(|n| n.checked_add(program.id.len()))
                .and_then(|n| n.checked_add(std::mem::size_of::<PanelResponse>()))
                .and_then(|n| sum.checked_add(n))
                .ok_or_else(|| invalid("response storage size overflow"))
        })?;
    u64::try_from(bytes).map_err(|_| invalid("response storage size overflow"))
}
fn response_workspace_bytes(data: &likelihood::CompiledPanel<'_>) -> Result<u64, PanelError> {
    let bytes = data
        .groups
        .len()
        .checked_mul(128)
        .and_then(|n| n.checked_add(data.workspace_bytes().ok()?))
        .and_then(|n| n.checked_add(data.plan_bytes().ok()?))
        .ok_or_else(|| invalid("response workspace size overflow"))?;
    u64::try_from(bytes).map_err(|_| invalid("response workspace size overflow"))
}
fn evaluate_law<T>(
    data: &PanelData,
    point: &T,
    draws: &[Option<T>],
    programs: &[PanelProgram],
    ctx: &ExecutionContext,
    evaluate: impl Fn(
        &likelihood::CompiledPanel<'_>,
        &T,
        &[PanelProgram],
        &ExecutionContext,
    ) -> Result<Vec<PanelResponse>, PanelError>,
) -> Result<PanelLawResponses, PanelError> {
    validate_panel_programs(data, programs)?;
    check_cancelled(ctx)?;
    let compiled = likelihood::CompiledPanel::new(data)?;
    let slots = u64::try_from(draws.len()).map_err(|_| invalid("response slot count overflow"))?;
    let valid = u64::try_from(draws.iter().filter(|draw| draw.is_some()).count())
        .map_err(|_| invalid("response valid slot count overflow"))?;
    let storage = response_storage_bytes(programs)?
        .checked_mul(valid.checked_add(1).ok_or_else(|| invalid("response slot count overflow"))?)
        .and_then(|n| {
            n.checked_add(
                slots.checked_mul(std::mem::size_of::<Option<Vec<PanelResponse>>>() as u64)?,
            )
        })
        .ok_or_else(|| invalid("law response storage overflow"))?;
    let workspace = response_workspace_bytes(&compiled)?;
    check_memory(
        ctx,
        storage.checked_add(workspace).ok_or_else(|| invalid("law response peak overflow"))?,
    )?;
    let retention = Arc::new(
        ctx.acquire_admission(0, storage)
            .map_err(|error| PanelError::Resource(error.to_string()))?,
    );
    let _working = ctx
        .acquire_admission(0, workspace)
        .map_err(|error| PanelError::Resource(error.to_string()))?;
    let point = evaluate(&compiled, point, programs, ctx)?;
    let mut output = Vec::with_capacity(draws.len());
    for draw in draws {
        check_cancelled(ctx)?;
        output.push(match draw.as_ref().map(|draw| evaluate(&compiled, draw, programs, ctx)) {
            None | Some(Err(PanelError::Numerical(_))) => None,
            Some(Ok(row)) => Some(row),
            Some(Err(error)) => return Err(error),
        });
    }
    antecedent_core::execution_attempt::record_physical(
        antecedent_core::execution_attempt::PhysicalEvent::RetainedFull,
        1,
    );
    antecedent_core::execution_attempt::record_physical(
        antecedent_core::execution_attempt::PhysicalEvent::RetainedBytes,
        storage,
    );
    Ok(PanelLawResponses { point, draws: output, retention, retained_volume: storage })
}
