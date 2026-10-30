// SPDX-License-Identifier: MIT OR Apache-2.0
//! Sampling-law and posterior uncertainty for fitted continuous-time feedback
//! families: ODE drift fits under Gaussian measurement error ([`OdeFit`]),
//! stationary OU fits ([`OuFit`]) and shared-mechanism OU fits ([`OuMechanismFit`]).
//!
//! The bootstrap reruns the whole declared fitting procedure, from its template,
//! on every replicate. A nonparametric replicate draws whole independent units: a
//! trajectory, a stationary replicate row, or one observation of a regime. It never
//! draws rows inside a trajectory, and the declared sampling design must allow that
//! whole-unit rule. A parametric replicate simulates every unit from the fitted model
//! and keeps the design facts: initial states, times, schedules, clamps and regime
//! sizes. A failed or unconverged replicate keeps its slot and counts as failed mass.
//! Every replicate opens its stream from a semantic key, so a result does not depend
//! on the worker count or on how many replicates were requested in one call.
//!
//! The observed-information covariance is reported only when the estimate converged
//! and the negative Hessian is positive definite and conditioned; otherwise the
//! reason is returned instead. The posterior uses an explicit independent normal prior
//! and random-walk Metropolis preconditioned by the Laplace covariance, with split
//! R-hat and bulk/tail ESS. Every result stays conditional on the specified structure,
//! the fixed design and the fixed measurement or diffusion law. No coverage
//! calibration is claimed.

use crate::cyclic_dynamics::{
    OdeFit, OdeFitPolicy, OdeObservation, OuFit, OuParameter, fit_ode_likelihood, fit_ou_likelihood,
};
use crate::cyclic_gradient::CyclicGradientPolicy;
use crate::cyclic_likelihood::{CyclicBootstrapDesign, CyclicFitOptions};
use crate::ou_mechanisms::{
    OuEquationReplacement, OuMechanismFit, OuMechanismParameter, OuMechanismRegime, OuMechanisms,
    Summary, fit_ou_mechanisms,
};
use antecedent_core::execution::AdmissionLease;
use antecedent_core::{
    CausalRng, ChainId, DeclaredSamplingDesign, ExecutionContext, FoldId, NonZeroThreadCount,
    Parallelism, ReductionPolicy, ReplicateId, ResamplingRule, RngKey, SamplingDesignError,
    SemanticOperation, VarianceRule,
};
use antecedent_kernels::standard_normal;
use antecedent_model::continuous_cyclic::{
    OdeParameter, OdePolicy, OdeSystem, OrnsteinUhlenbeck, TimedClamp,
};
use antecedent_model::nonlinear_equilibrium::{Clamp, EquilibriumError, RegimeRow};
use antecedent_prob::metropolis::{
    MetropolisError, MetropolisOptions, sample_metropolis_dispersed,
};
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::collections::BTreeMap;

/// Largest squared Cholesky pivot ratio accepted as a regular information matrix.
const MAX_PIVOT_RATIO: f64 = 1e12;
/// Semantic stream tags, one per family; the low byte names the bootstrap method.
const ODE_STREAM: u64 = 0x4459_4e5f_4f44_4500;
const OU_STREAM: u64 = 0x4459_4e5f_4f55_5300;
const OU_MECHANISM_STREAM: u64 = 0x4459_4e5f_4f55_4d00;
const PARAMETRIC_WORD: u64 = 1;
const NONPARAMETRIC_WORD: u64 = 2;

/// Refusal of a dynamic-family inference request.
#[derive(Clone, Debug, PartialEq, thiserror::Error)]
pub enum DynamicInferenceError {
    /// The declared sampling design does not allow the attempted rule.
    #[error(transparent)]
    Design(#[from] SamplingDesignError),
    /// The model or data was refused before any replicate or draw ran.
    #[error(transparent)]
    Model(EquilibriumError),
    /// The supplied fit was not produced by the declared problem.
    #[error("fit does not match the declared dynamic problem: {0}")]
    FitMismatch(String),
    /// Malformed request: counts, shapes, prior, quantity or sampler policy.
    #[error("invalid dynamic inference request: {0}")]
    Invalid(String),
    /// Cancellation or admission refusal.
    #[error("dynamic inference resource refusal: {0}")]
    Resource(String),
}

impl From<EquilibriumError> for DynamicInferenceError {
    fn from(error: EquilibriumError) -> Self {
        match error {
            EquilibriumError::Resource(message) => Self::Resource(message),
            other => Self::Model(other),
        }
    }
}

fn invalid(message: &str) -> DynamicInferenceError {
    DynamicInferenceError::Invalid(message.into())
}

fn mismatch(message: &str) -> DynamicInferenceError {
    DynamicInferenceError::FitMismatch(message.into())
}

fn cancelled() -> DynamicInferenceError {
    DynamicInferenceError::Resource("cancelled".into())
}

fn model_invalid(message: &str) -> EquilibriumError {
    EquilibriumError::Invalid(message.into())
}

/// Declared ODE measurement-fitting procedure. Every replicate reruns it whole.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OdeInferenceProblem {
    /// Starting drift system; coordinates outside `parameters` stay fixed.
    pub template: OdeSystem,
    /// Estimated drift coordinates, in estimate order.
    pub parameters: Vec<OdeParameter>,
    /// Independent trajectories, the sampling units.
    pub data: Vec<OdeObservation>,
    /// Integration policy.
    pub integration: OdePolicy,
    /// Local optimizer policy.
    pub policy: OdeFitPolicy,
}

/// Declared stationary OU fitting procedure. Every replicate reruns it whole.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OuInferenceProblem {
    /// Starting process; coordinates outside `parameters`, including diffusion, stay fixed.
    pub template: OrnsteinUhlenbeck,
    /// Estimated drift/center coordinates, in estimate order.
    pub parameters: Vec<OuParameter>,
    /// Independent stationary replicates under known sustained clamps, the sampling units.
    pub rows: Vec<RegimeRow>,
    /// Coordinate-search policy.
    pub options: CyclicFitOptions,
}

/// Declared shared-mechanism OU fitting procedure. Every replicate reruns it whole.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OuMechanismInferenceProblem {
    /// Starting mechanisms; coordinates outside `parameters` stay fixed.
    pub template: OuMechanisms,
    /// Estimated mechanism coordinates, in estimate order.
    pub parameters: Vec<OuMechanismParameter>,
    /// Equation-replacement regimes; each observation is one sampling unit.
    pub regimes: Vec<OuMechanismRegime>,
    /// Gradient policy.
    pub policy: CyclicGradientPolicy,
}

/// A fitted dynamic family together with the procedure that produced it.
#[derive(Clone, Copy, Debug)]
pub enum DynamicFitRef<'a> {
    /// ODE drift fit under Gaussian measurement error.
    Ode {
        /// Declared procedure.
        problem: &'a OdeInferenceProblem,
        /// Its fit.
        fit: &'a OdeFit,
    },
    /// Stationary OU fit.
    Ou {
        /// Declared procedure.
        problem: &'a OuInferenceProblem,
        /// Its fit.
        fit: &'a OuFit,
    },
    /// Shared-mechanism OU fit.
    OuMechanisms {
        /// Declared procedure.
        problem: &'a OuMechanismInferenceProblem,
        /// Its fit.
        fit: &'a OuMechanismFit,
    },
}

/// Interventional or equilibrium quantity evaluated at every replicate or draw.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum DynamicQuantity {
    /// ODE state at `time` from `initial` at `start` under a sustained-clamp schedule.
    OdeState {
        /// Exact initial state.
        initial: Vec<f64>,
        /// Initial time.
        start: f64,
        /// Evaluation time, after `start`.
        time: f64,
        /// Timed mechanism replacements.
        schedule: Vec<TimedClamp>,
        /// State coordinate.
        variable: usize,
    },
    /// Stationary OU mean after sustained hard clamps.
    OuStationaryMean {
        /// Sustained clamps; empty is the unintervened process.
        clamps: Vec<Clamp>,
        /// State coordinate.
        variable: usize,
    },
    /// Stationary OU covariance entry after sustained hard clamps.
    OuStationaryCovariance {
        /// Sustained clamps; empty is the unintervened process.
        clamps: Vec<Clamp>,
        /// Row coordinate.
        variable: usize,
        /// Column coordinate.
        other: usize,
    },
    /// Stationary mean of shared OU mechanisms after equation replacements.
    OuMechanismMean {
        /// Equation replacements; empty is observation.
        replacements: Vec<OuEquationReplacement>,
        /// State coordinate.
        variable: usize,
    },
    /// Stationary covariance entry of shared OU mechanisms after equation replacements.
    OuMechanismCovariance {
        /// Equation replacements; empty is observation.
        replacements: Vec<OuEquationReplacement>,
        /// Row coordinate.
        variable: usize,
        /// Column coordinate.
        other: usize,
    },
}

/// How bootstrap replicate data are produced.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DynamicBootstrapMethod {
    /// Simulate every unit from the fitted model, keeping its design facts.
    Parametric,
    /// Resample whole independent units with replacement within strata.
    Nonparametric,
}

/// A bootstrap request and the sampling design it must respect.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DynamicBootstrapRequest {
    /// Parametric or nonparametric replicate data.
    pub method: DynamicBootstrapMethod,
    /// Replicate count, at most 10,000.
    pub replicates: usize,
    /// Declared sampling design of the units.
    pub design: DeclaredSamplingDesign,
    /// Attempted resampling rule; checked against `design` for a nonparametric draw.
    pub resampling: ResamplingRule,
    /// Strata of a nonparametric draw: fixed assignments, or targets only.
    pub stratification: CyclicBootstrapDesign,
}

impl DynamicBootstrapRequest {
    /// Independent subject trajectories, each drawn whole.
    #[must_use]
    pub const fn whole_units(
        method: DynamicBootstrapMethod,
        replicates: usize,
        stratification: CyclicBootstrapDesign,
    ) -> Self {
        Self {
            method,
            replicates,
            design: DeclaredSamplingDesign::IndependentSubjectTrajectories,
            resampling: ResamplingRule::RowSubjectIid,
            stratification,
        }
    }
}

/// Why a replicate produced no parameter vector.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ReplicateFailure {
    /// The rerun procedure stopped before its declared tolerance.
    NotConverged,
    /// Simulation or fitting failed numerically (instability, singularity, nonfinite).
    Numerical {
        /// Original diagnostic.
        detail: String,
    },
}

/// One bootstrap replicate in its original slot.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DynamicReplicate {
    /// Refitted parameters; absent for a failed replicate.
    pub parameters: Option<Vec<f64>>,
    /// Quantities at the refitted parameters; a failed evaluation stays `None`.
    pub quantities: Vec<Option<f64>>,
    /// Failure classification of a failed replicate.
    pub failure: Option<ReplicateFailure>,
}

impl DynamicReplicate {
    fn failed(failure: ReplicateFailure, quantities: usize) -> Self {
        Self { parameters: None, quantities: vec![None; quantities], failure: Some(failure) }
    }
}

/// Bootstrap sampling law of the whole fitting procedure.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DynamicSamplingLaw {
    /// Parametric or nonparametric.
    pub method: DynamicBootstrapMethod,
    /// Declared sampling design.
    pub design: String,
    /// Resampling rule actually applied, or the parametric unit simulation.
    pub resampling: String,
    /// Strata of a nonparametric draw; absent for a parametric one.
    pub stratification: Option<CyclicBootstrapDesign>,
    /// Original estimate, in parameter order.
    pub estimate: Vec<f64>,
    /// Quantities at the original estimate.
    pub quantity_estimates: Vec<Option<f64>>,
    /// Every replicate, failed ones included, in replicate order.
    pub replicates: Vec<DynamicReplicate>,
    /// Weight of each replicate slot; they sum to one, failed slots included.
    pub weights: Vec<f64>,
    /// Weight of failed replicates; never renormalized away.
    pub failed_mass: f64,
    /// Standard deviation of successful replicates per parameter (at least two needed).
    pub standard_errors: Option<Vec<f64>>,
    /// Standard deviation of successful quantity values (at least two needed).
    pub quantity_standard_errors: Vec<Option<f64>>,
    /// Weight of replicates without a value for each quantity, failed fits included.
    pub quantity_failed_mass: Vec<f64>,
}

/// Whether the likelihood is regular enough for an observed-information covariance.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum InformationRegularity {
    /// Converged estimate with a positive-definite, conditioned negative Hessian.
    Regular,
    /// The estimate did not reach its declared tolerance.
    EstimateNotConverged,
    /// The Hessian could not be evaluated near the estimate.
    NotEvaluable {
        /// Original diagnostic.
        detail: String,
    },
    /// The negative Hessian is not positive definite (a flat or unidentified direction).
    NotPositiveDefinite,
    /// Positive definite but too ill-conditioned to invert reliably.
    IllConditioned {
        /// Squared ratio of the largest to smallest Cholesky pivot.
        pivot_ratio: f64,
    },
}

/// Observed information and the delta-method covariance it licenses.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DynamicObservedInformation {
    /// Estimate, in parameter order.
    pub estimate: Vec<f64>,
    /// Row-major negative Hessian of the log likelihood, when evaluable.
    pub information: Option<Vec<f64>>,
    /// Regularity verdict; a covariance exists only when regular.
    pub regularity: InformationRegularity,
    /// Inverse information, row-major.
    pub covariance: Option<Vec<f64>>,
    /// Square roots of the covariance diagonal.
    pub standard_errors: Option<Vec<f64>>,
    /// Quantities at the estimate.
    pub quantity_estimates: Vec<Option<f64>>,
    /// Delta-method standard errors of the quantities.
    pub quantity_standard_errors: Vec<Option<f64>>,
    /// Derivative scheme used for the Hessian.
    pub derivative: String,
    /// Variance rule the declared design allowed.
    pub variance: String,
}

/// Explicit independent normal prior over the estimated coordinates.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DynamicPrior {
    /// Prior mean per coordinate.
    pub mean: Vec<f64>,
    /// Positive prior standard deviation per coordinate.
    pub standard_deviation: Vec<f64>,
}

impl DynamicPrior {
    fn validate(&self, p: usize) -> Result<(), DynamicInferenceError> {
        if self.mean.len() != p
            || self.standard_deviation.len() != p
            || self.mean.iter().any(|v| !v.is_finite())
            || self.standard_deviation.iter().any(|v| !v.is_finite() || *v <= 0.0)
        {
            return Err(invalid("prior needs one finite mean and positive sd per coordinate"));
        }
        Ok(())
    }

    fn log_density(&self, theta: &[f64]) -> f64 {
        -0.5 * theta
            .iter()
            .zip(&self.mean)
            .zip(&self.standard_deviation)
            .map(|((t, m), s)| ((t - m) / s).powi(2))
            .sum::<f64>()
    }
}

/// Metropolis policy in the whitened (preconditioned) coordinates.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DynamicPosteriorOptions {
    /// Independent chains, 2 to 16.
    pub chains: usize,
    /// Discarded iterations per chain.
    pub warmup: usize,
    /// Retained draws per chain, at least 8.
    pub draws: usize,
    /// Whitened proposal sd; `None` is `2.38/sqrt(p)`.
    pub proposal_scale: Option<f64>,
    /// Whitened sd of the dispersed chain starts around the estimate.
    pub start_spread: f64,
}

impl Default for DynamicPosteriorOptions {
    fn default() -> Self {
        Self { chains: 4, warmup: 500, draws: 1000, proposal_scale: None, start_spread: 1.0 }
    }
}

impl DynamicPosteriorOptions {
    fn sampler(self, p: usize) -> Result<MetropolisOptions, DynamicInferenceError> {
        let proposal_scale = self.proposal_scale.unwrap_or(2.38 / (p as f64).sqrt());
        if !(2..=16).contains(&self.chains)
            || !(8..=100_000).contains(&self.draws)
            || self.warmup > 100_000
            || !proposal_scale.is_finite()
            || proposal_scale <= 0.0
            || !self.start_spread.is_finite()
            || self.start_spread <= 0.0
        {
            return Err(invalid("posterior chains, draws, warmup, proposal or start spread"));
        }
        Ok(MetropolisOptions {
            chains: self.chains,
            warmup: self.warmup,
            draws: self.draws,
            proposal_scale,
        })
    }
}

/// Preconditioner of the whitened Metropolis walk.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PosteriorPreconditioner {
    /// Cholesky factor of `(information + prior precision)^-1` at the estimate.
    Laplace,
    /// Diagonal curvature plus prior precision, when the full matrix is not definite.
    DiagonalCurvature,
    /// Prior standard deviations, when the information is not evaluable.
    PriorScale,
}

/// Posterior over the dynamic parameters with draws of the derived quantities.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DynamicPosterior {
    /// Declared prior.
    pub prior: DynamicPrior,
    /// Declared sampling design.
    pub design: String,
    /// Proposal preconditioner actually used.
    pub preconditioner: PosteriorPreconditioner,
    /// Chain-major parameter draws.
    pub parameter_draws: Vec<Vec<f64>>,
    /// Quantities per draw; a failed evaluation (unstable regime) stays `None`.
    pub quantity_draws: Vec<Vec<Option<f64>>>,
    /// Fraction of draws without a value for each quantity.
    pub quantity_failed_mass: Vec<f64>,
    /// Chains.
    pub chains: usize,
    /// Draws per chain.
    pub draws_per_chain: usize,
    /// Accepted retained proposals per chain.
    pub accepted: Vec<usize>,
    /// Maximum rank/folded split R-hat over parameters, absent if undefined.
    pub max_rhat: Option<f64>,
    /// Minimum bulk ESS over parameters.
    pub min_bulk_ess: f64,
    /// Minimum tail ESS over parameters.
    pub min_tail_ess: f64,
    /// R-hat <= 1.01, bulk/tail ESS >= 100, and every chain moved.
    pub diagnostics_pass: bool,
}

type Fallible<T> = Result<T, EquilibriumError>;

/// One clamp assignment's stationary law: the clamps, the mean, the free coordinates and the
/// Cholesky factor of their covariance block.
type ClampLaw<'a> = (&'a [Clamp], Vec<f64>, Vec<usize>, Vec<f64>);

/// One fitted family: likelihood on the original data, unit resampling and simulation.
trait Family: Sync {
    type Data;
    fn stream(&self) -> u64;
    fn estimate(&self) -> &[f64];
    fn converged(&self) -> bool;
    fn reported_log_likelihood(&self) -> f64;
    fn log_likelihood(&self, theta: &[f64], ctx: &ExecutionContext) -> Fallible<f64>;
    fn has_score(&self) -> bool;
    fn score(&self, theta: &[f64], ctx: &ExecutionContext) -> Fallible<Vec<f64>>;
    fn units(&self) -> usize;
    fn strata(&self, design: CyclicBootstrapDesign) -> Vec<Vec<usize>>;
    fn gather(&self, picks: &[usize]) -> Self::Data;
    fn simulate(&self, rng: &mut CausalRng, ctx: &ExecutionContext) -> Fallible<Self::Data>;
    fn refit(&self, data: &Self::Data, ctx: &ExecutionContext) -> Fallible<(Vec<f64>, bool)>;
    fn check_quantity(&self, quantity: &DynamicQuantity) -> bool;
    fn quantity(
        &self,
        theta: &[f64],
        quantity: &DynamicQuantity,
        ctx: &ExecutionContext,
    ) -> Fallible<f64>;
    fn data_bytes(&self) -> usize;
}

fn family_mismatch() -> EquilibriumError {
    model_invalid("quantity belongs to another dynamic family")
}

fn clamps_valid(clamps: &[Clamp], n: usize) -> bool {
    clamps.iter().enumerate().all(|(k, c)| {
        c.variable < n
            && c.value.is_finite()
            && clamps[..k].iter().all(|other| other.variable != c.variable)
    })
}

// ---------------------------------------------------------------- ODE family

fn ode_values(model: &OdeSystem, parameters: &[OdeParameter]) -> Fallible<Vec<f64>> {
    if parameters.is_empty() {
        return Err(model_invalid("ODE parameter count"));
    }
    parameters
        .iter()
        .enumerate()
        .map(|(k, parameter)| {
            if parameters[..k].contains(parameter) {
                return Err(model_invalid("duplicate ODE parameter"));
            }
            let value = match *parameter {
                OdeParameter::ExpressionParameter { equation, parameter } => {
                    model.equations.get(equation).and_then(|e| e.parameters.get(parameter)).copied()
                }
                OdeParameter::Intercept { equation } => {
                    model.equations.get(equation).map(|e| e.intercept)
                }
                OdeParameter::Weight { equation, parent } => model
                    .equations
                    .get(equation)
                    .filter(|e| e.expression.is_none())
                    .and_then(|e| e.weights.get(parent))
                    .copied(),
                OdeParameter::Decay { equation } => model.decay.get(equation).copied(),
            };
            value.ok_or_else(|| model_invalid("ODE parameter coordinate"))
        })
        .collect()
}

fn schedule_key(schedule: &[TimedClamp], design: CyclicBootstrapDesign) -> Vec<u64> {
    match design {
        CyclicBootstrapDesign::FixedAssignments => {
            let mut key = Vec::new();
            for action in schedule {
                let mut clamps: Vec<_> =
                    action.clamps.iter().map(|c| (c.variable as u64, c.value.to_bits())).collect();
                clamps.sort_unstable();
                key.push(action.time.to_bits());
                key.push(clamps.len() as u64);
                key.extend(clamps.into_iter().flat_map(|(variable, value)| [variable, value]));
            }
            key
        }
        CyclicBootstrapDesign::RandomizedWithinTargets => {
            let mut targets: Vec<u64> = schedule
                .iter()
                .flat_map(|action| action.clamps.iter().map(|c| c.variable as u64))
                .collect();
            targets.sort_unstable();
            targets.dedup();
            targets
        }
    }
}

struct OdeFamily<'a> {
    problem: &'a OdeInferenceProblem,
    estimate: Vec<f64>,
    converged: bool,
    reported: f64,
}

impl<'a> OdeFamily<'a> {
    fn new(problem: &'a OdeInferenceProblem, fit: &OdeFit) -> Result<Self, DynamicInferenceError> {
        problem.template.validate()?;
        if problem.data.is_empty() {
            return Err(invalid("ODE problem has no trajectories"));
        }
        ode_values(&problem.template, &problem.parameters)?;
        let estimate = ode_values(&fit.model, &problem.parameters)?;
        if crate::cyclic_dynamics::bind(&problem.template, &problem.parameters, &estimate)
            != fit.model
        {
            return Err(mismatch("fitted ODE differs from the template outside its parameters"));
        }
        Ok(Self { problem, estimate, converged: fit.converged, reported: fit.log_likelihood })
    }

    fn model(&self, theta: &[f64]) -> OdeSystem {
        crate::cyclic_dynamics::bind(&self.problem.template, &self.problem.parameters, theta)
    }

    fn path(
        &self,
        model: &OdeSystem,
        initial: &[f64],
        times: &[f64],
        schedule: &[TimedClamp],
        ctx: &ExecutionContext,
    ) -> Fallible<Vec<Vec<f64>>> {
        // The likelihood integrates the sensitivity system; reuse that exact path.
        Ok(model
            .sensitivities(
                initial,
                times,
                schedule,
                &self.problem.parameters,
                self.problem.integration,
                ctx,
            )?
            .trajectory
            .states)
    }
}

impl Family for OdeFamily<'_> {
    type Data = Vec<OdeObservation>;

    fn stream(&self) -> u64 {
        ODE_STREAM
    }
    fn estimate(&self) -> &[f64] {
        &self.estimate
    }
    fn converged(&self) -> bool {
        self.converged
    }
    fn reported_log_likelihood(&self) -> f64 {
        self.reported
    }
    fn log_likelihood(&self, theta: &[f64], ctx: &ExecutionContext) -> Fallible<f64> {
        let p = self.problem;
        crate::cyclic_dynamics::likelihood_score(
            &self.model(theta),
            &p.parameters,
            &p.data,
            p.integration,
            ctx,
        )
        .map(|(likelihood, _)| likelihood)
    }
    fn has_score(&self) -> bool {
        true
    }
    fn score(&self, theta: &[f64], ctx: &ExecutionContext) -> Fallible<Vec<f64>> {
        let p = self.problem;
        crate::cyclic_dynamics::likelihood_score(
            &self.model(theta),
            &p.parameters,
            &p.data,
            p.integration,
            ctx,
        )
        .map(|(_, score)| score)
    }
    fn units(&self) -> usize {
        self.problem.data.len()
    }
    fn strata(&self, design: CyclicBootstrapDesign) -> Vec<Vec<usize>> {
        let mut groups: BTreeMap<Vec<u64>, Vec<usize>> = BTreeMap::new();
        for (index, observed) in self.problem.data.iter().enumerate() {
            groups.entry(schedule_key(&observed.schedule, design)).or_default().push(index);
        }
        groups.into_values().collect()
    }
    fn gather(&self, picks: &[usize]) -> Self::Data {
        picks.iter().map(|&index| self.problem.data[index].clone()).collect()
    }
    fn simulate(&self, rng: &mut CausalRng, ctx: &ExecutionContext) -> Fallible<Self::Data> {
        let model = self.model(&self.estimate);
        self.problem
            .data
            .iter()
            .map(|observed| {
                let states =
                    self.path(&model, &observed.initial, &observed.times, &observed.schedule, ctx)?;
                let mut values = Vec::with_capacity(states.len());
                // The first state is exact by design and is not a measurement.
                values.push(observed.initial.clone());
                for state in states.iter().skip(1) {
                    values.push(
                        state
                            .iter()
                            .zip(&observed.variances)
                            .map(|(x, variance)| x + variance.sqrt() * standard_normal(rng))
                            .collect(),
                    );
                }
                Ok(OdeObservation {
                    initial: observed.initial.clone(),
                    times: observed.times.clone(),
                    values,
                    variances: observed.variances.clone(),
                    schedule: observed.schedule.clone(),
                })
            })
            .collect()
    }
    fn refit(&self, data: &Self::Data, ctx: &ExecutionContext) -> Fallible<(Vec<f64>, bool)> {
        let p = self.problem;
        let fit =
            fit_ode_likelihood(&p.template, &p.parameters, data, p.integration, p.policy, ctx)?;
        Ok((ode_values(&fit.model, &p.parameters)?, fit.converged))
    }
    fn check_quantity(&self, quantity: &DynamicQuantity) -> bool {
        let n = self.problem.template.equations.len();
        match quantity {
            DynamicQuantity::OdeState { initial, start, time, schedule, variable } => {
                initial.len() == n
                    && initial.iter().all(|v| v.is_finite())
                    && start.is_finite()
                    && time.is_finite()
                    && time > start
                    && *variable < n
                    && schedule
                        .iter()
                        .all(|action| action.time.is_finite() && clamps_valid(&action.clamps, n))
            }
            _ => false,
        }
    }
    fn quantity(
        &self,
        theta: &[f64],
        quantity: &DynamicQuantity,
        ctx: &ExecutionContext,
    ) -> Fallible<f64> {
        let DynamicQuantity::OdeState { initial, start, time, schedule, variable } = quantity
        else {
            return Err(family_mismatch());
        };
        let states = self.path(&self.model(theta), initial, &[*start, *time], schedule, ctx)?;
        states
            .get(1)
            .and_then(|state| state.get(*variable))
            .copied()
            .ok_or_else(|| model_invalid("ODE state quantity"))
    }
    fn data_bytes(&self) -> usize {
        self.problem
            .data
            .iter()
            .map(|o| {
                let cells = o.initial.len()
                    + o.times.len()
                    + o.variances.len()
                    + o.values.iter().map(Vec::len).sum::<usize>()
                    + o.schedule.iter().map(|a| 1 + 2 * a.clamps.len()).sum::<usize>();
                cells * 8 + 160
            })
            .sum()
    }
}

// ----------------------------------------------------------------- OU family

fn ou_values(model: &OrnsteinUhlenbeck, parameters: &[OuParameter]) -> Fallible<Vec<f64>> {
    let n = model.center.len();
    if parameters.is_empty() || model.drift.len() != n * n {
        return Err(model_invalid("OU parameter count or shape"));
    }
    parameters
        .iter()
        .enumerate()
        .map(|(k, parameter)| {
            if parameters[..k].contains(parameter) {
                return Err(model_invalid("duplicate OU parameter"));
            }
            match *parameter {
                OuParameter::Drift { row, column } if row < n && column < n => {
                    Ok(model.drift[row * n + column])
                }
                OuParameter::Center { variable } if variable < n => Ok(model.center[variable]),
                _ => Err(model_invalid("OU parameter coordinate")),
            }
        })
        .collect()
}

fn bind_ou(
    template: &OrnsteinUhlenbeck,
    parameters: &[OuParameter],
    values: &[f64],
) -> OrnsteinUhlenbeck {
    let n = template.center.len();
    let mut model = template.clone();
    for (&parameter, &value) in parameters.iter().zip(values) {
        match parameter {
            OuParameter::Drift { row, column } => model.drift[row * n + column] = value,
            OuParameter::Center { variable } => model.center[variable] = value,
        }
    }
    model
}

struct OuFamily<'a> {
    problem: &'a OuInferenceProblem,
    estimate: Vec<f64>,
    converged: bool,
    reported: f64,
}

impl<'a> OuFamily<'a> {
    fn new(problem: &'a OuInferenceProblem, fit: &OuFit) -> Result<Self, DynamicInferenceError> {
        if problem.rows.is_empty() {
            return Err(invalid("OU problem has no stationary replicates"));
        }
        ou_values(&problem.template, &problem.parameters)?;
        let estimate = ou_values(&fit.model, &problem.parameters)?;
        if bind_ou(&problem.template, &problem.parameters, &estimate) != fit.model {
            return Err(mismatch("fitted OU differs from the template outside its parameters"));
        }
        Ok(Self { problem, estimate, converged: fit.converged, reported: fit.log_likelihood })
    }

    fn model(&self, theta: &[f64]) -> OrnsteinUhlenbeck {
        bind_ou(&self.problem.template, &self.problem.parameters, theta)
    }
}

impl Family for OuFamily<'_> {
    type Data = Vec<RegimeRow>;

    fn stream(&self) -> u64 {
        OU_STREAM
    }
    fn estimate(&self) -> &[f64] {
        &self.estimate
    }
    fn converged(&self) -> bool {
        self.converged
    }
    fn reported_log_likelihood(&self) -> f64 {
        self.reported
    }
    fn log_likelihood(&self, theta: &[f64], ctx: &ExecutionContext) -> Fallible<f64> {
        self.model(theta).stationary_log_likelihood(&self.problem.rows, ctx)
    }
    fn has_score(&self) -> bool {
        false
    }
    fn score(&self, _theta: &[f64], _ctx: &ExecutionContext) -> Fallible<Vec<f64>> {
        Err(model_invalid("stationary OU fits carry no analytic score"))
    }
    fn units(&self) -> usize {
        self.problem.rows.len()
    }
    fn strata(&self, design: CyclicBootstrapDesign) -> Vec<Vec<usize>> {
        crate::cyclic_likelihood::bootstrap_groups(&self.problem.rows, design)
            .into_values()
            .collect()
    }
    fn gather(&self, picks: &[usize]) -> Self::Data {
        picks.iter().map(|&index| self.problem.rows[index].clone()).collect()
    }
    fn simulate(&self, rng: &mut CausalRng, ctx: &ExecutionContext) -> Fallible<Self::Data> {
        let model = self.model(&self.estimate);
        let n = model.center.len();
        // Consecutive rows with one clamp assignment share a law and its factor.
        let mut previous: Option<ClampLaw<'_>> = None;
        let mut rows = Vec::with_capacity(self.problem.rows.len());
        for row in &self.problem.rows {
            if previous.as_ref().is_none_or(|(clamps, ..)| *clamps != row.clamps.as_slice()) {
                let law = model.stationary_intervention(&row.clamps, ctx)?;
                let full = &law.covariance;
                let block: Vec<f64> = law
                    .free
                    .iter()
                    .flat_map(|&i| law.free.iter().map(move |&j| full[i * n + j]))
                    .collect();
                let lower = cholesky(&block, law.free.len()).ok_or(EquilibriumError::Singular)?;
                previous = Some((row.clamps.as_slice(), law.mean, law.free, lower));
            }
            let Some((_, mean, free, lower)) = previous.as_ref() else {
                return Err(model_invalid("OU stationary law"));
            };
            let k = free.len();
            let z: Vec<f64> = (0..k).map(|_| standard_normal(rng)).collect();
            let mut values = mean.clone();
            for (a, &i) in free.iter().enumerate() {
                values[i] += (0..=a).map(|b| lower[a * k + b] * z[b]).sum::<f64>();
            }
            rows.push(RegimeRow { values, clamps: row.clamps.clone() });
        }
        Ok(rows)
    }
    fn refit(&self, data: &Self::Data, ctx: &ExecutionContext) -> Fallible<(Vec<f64>, bool)> {
        let p = self.problem;
        let fit = fit_ou_likelihood(&p.template, &p.parameters, data, p.options, ctx)?;
        Ok((ou_values(&fit.model, &p.parameters)?, fit.converged))
    }
    fn check_quantity(&self, quantity: &DynamicQuantity) -> bool {
        let n = self.problem.template.center.len();
        match quantity {
            DynamicQuantity::OuStationaryMean { clamps, variable } => {
                *variable < n && clamps_valid(clamps, n)
            }
            DynamicQuantity::OuStationaryCovariance { clamps, variable, other } => {
                *variable < n && *other < n && clamps_valid(clamps, n)
            }
            _ => false,
        }
    }
    fn quantity(
        &self,
        theta: &[f64],
        quantity: &DynamicQuantity,
        ctx: &ExecutionContext,
    ) -> Fallible<f64> {
        let model = self.model(theta);
        let n = model.center.len();
        let value = match quantity {
            DynamicQuantity::OuStationaryMean { clamps, variable } => {
                model.stationary_intervention(clamps, ctx)?.mean.get(*variable).copied()
            }
            DynamicQuantity::OuStationaryCovariance { clamps, variable, other } => model
                .stationary_intervention(clamps, ctx)?
                .covariance
                .get(variable * n + other)
                .copied(),
            _ => return Err(family_mismatch()),
        };
        value.ok_or_else(|| model_invalid("OU quantity coordinate"))
    }
    fn data_bytes(&self) -> usize {
        self.problem.rows.iter().map(|row| (row.values.len() + 2 * row.clamps.len()) * 8 + 64).sum()
    }
}

// -------------------------------------------------- shared-mechanism OU family

fn replacements_valid(template: &OuMechanisms, replacements: &[OuEquationReplacement]) -> bool {
    let n = template.equations.len();
    replacements.iter().enumerate().all(|(k, r)| {
        template.equations.contains(&r.equation)
            && replacements[..k].iter().all(|other| other.equation != r.equation)
            && r.drift.len() == n
            && r.diffusion_factor.len() == n
            && r.forcing.is_finite()
            && r.drift.iter().chain(&r.diffusion_factor).all(|v| v.is_finite())
    })
}

struct OuMechanismFamily<'a> {
    problem: &'a OuMechanismInferenceProblem,
    estimate: Vec<f64>,
    converged: bool,
    reported: f64,
    summaries: Vec<Summary>,
    /// Unit offsets of each regime; the last entry is the unit count.
    offsets: Vec<usize>,
    _summary_lease: AdmissionLease,
}

impl<'a> OuMechanismFamily<'a> {
    fn new(
        problem: &'a OuMechanismInferenceProblem,
        fit: &OuMechanismFit,
        ctx: &ExecutionContext,
    ) -> Result<Self, DynamicInferenceError> {
        let n = problem.template.validate()?;
        crate::ou_mechanisms::parameters(&problem.template, &problem.parameters)?;
        let estimate = crate::ou_mechanisms::parameters(&fit.model, &problem.parameters)?;
        if crate::ou_mechanisms::bind(&problem.template, &problem.parameters, &estimate)
            != fit.model
        {
            return Err(mismatch("fitted mechanisms differ from the template outside parameters"));
        }
        let regimes = problem.regimes.len();
        let bytes = regimes
            .checked_mul((n * n + n) * 8 + n * 16 + 96)
            .and_then(|b| b.checked_add((regimes + 1) * 8));
        let summary_lease = admit(ctx, bytes)?;
        let summaries = crate::ou_mechanisms::summaries(&problem.regimes, n, ctx)?;
        let mut offsets = Vec::with_capacity(regimes + 1);
        offsets.push(0);
        for regime in &problem.regimes {
            offsets.push(offsets[offsets.len() - 1] + regime.values.len());
        }
        Ok(Self {
            problem,
            estimate,
            converged: fit.converged,
            reported: fit.log_likelihood,
            summaries,
            offsets,
            _summary_lease: summary_lease,
        })
    }

    fn model(&self, theta: &[f64]) -> OuMechanisms {
        crate::ou_mechanisms::bind(&self.problem.template, &self.problem.parameters, theta)
    }

    fn scores(&self, theta: &[f64], ctx: &ExecutionContext) -> Fallible<(f64, Vec<f64>)> {
        crate::ou_mechanisms::scores(
            &self.model(theta),
            &self.summaries,
            &self.problem.parameters,
            ctx,
        )
    }
}

impl Family for OuMechanismFamily<'_> {
    type Data = Vec<OuMechanismRegime>;

    fn stream(&self) -> u64 {
        OU_MECHANISM_STREAM
    }
    fn estimate(&self) -> &[f64] {
        &self.estimate
    }
    fn converged(&self) -> bool {
        self.converged
    }
    fn reported_log_likelihood(&self) -> f64 {
        self.reported
    }
    fn log_likelihood(&self, theta: &[f64], ctx: &ExecutionContext) -> Fallible<f64> {
        self.scores(theta, ctx).map(|(likelihood, _)| likelihood)
    }
    fn has_score(&self) -> bool {
        true
    }
    fn score(&self, theta: &[f64], ctx: &ExecutionContext) -> Fallible<Vec<f64>> {
        self.scores(theta, ctx).map(|(_, score)| score)
    }
    fn units(&self) -> usize {
        self.offsets[self.offsets.len() - 1]
    }
    fn strata(&self, design: CyclicBootstrapDesign) -> Vec<Vec<usize>> {
        let units = |r: usize| (self.offsets[r]..self.offsets[r + 1]).collect::<Vec<_>>();
        match design {
            CyclicBootstrapDesign::FixedAssignments => {
                (0..self.problem.regimes.len()).map(units).collect()
            }
            CyclicBootstrapDesign::RandomizedWithinTargets => {
                let mut groups: BTreeMap<Vec<&str>, Vec<usize>> = BTreeMap::new();
                for (r, regime) in self.problem.regimes.iter().enumerate() {
                    let mut key: Vec<&str> =
                        regime.replacements.iter().map(|x| x.equation.as_str()).collect();
                    key.sort_unstable();
                    groups.entry(key).or_default().extend(units(r));
                }
                groups.into_values().collect()
            }
        }
    }
    fn gather(&self, picks: &[usize]) -> Self::Data {
        let regimes = &self.problem.regimes;
        let mut values: Vec<Vec<Vec<f64>>> = vec![Vec::new(); regimes.len()];
        for &unit in picks {
            // A drawn unit keeps its original regime, including under pooled strata.
            let r = self.offsets.partition_point(|&offset| offset <= unit) - 1;
            values[r].push(regimes[r].values[unit - self.offsets[r]].clone());
        }
        regimes
            .iter()
            .zip(values)
            .filter(|(_, values)| !values.is_empty())
            .map(|(regime, values)| OuMechanismRegime {
                id: regime.id.clone(),
                replacements: regime.replacements.clone(),
                values,
            })
            .collect()
    }
    fn simulate(&self, rng: &mut CausalRng, ctx: &ExecutionContext) -> Fallible<Self::Data> {
        let model = self.model(&self.estimate);
        let n = self.problem.template.equations.len();
        self.problem
            .regimes
            .iter()
            .map(|regime| {
                let law = model.stationary(&regime.replacements, ctx)?;
                let lower = cholesky(&law.covariance, n).ok_or(EquilibriumError::Singular)?;
                let values: Vec<Vec<f64>> = (0..regime.values.len())
                    .map(|_| {
                        let z: Vec<f64> = (0..n).map(|_| standard_normal(rng)).collect();
                        (0..n)
                            .map(|i| {
                                law.mean[i] + (0..=i).map(|j| lower[i * n + j] * z[j]).sum::<f64>()
                            })
                            .collect::<Vec<f64>>()
                    })
                    .collect();
                Ok(OuMechanismRegime {
                    id: regime.id.clone(),
                    replacements: regime.replacements.clone(),
                    values,
                })
            })
            .collect()
    }
    fn refit(&self, data: &Self::Data, ctx: &ExecutionContext) -> Fallible<(Vec<f64>, bool)> {
        let p = self.problem;
        let fit = fit_ou_mechanisms(&p.template, &p.parameters, data, p.policy, ctx)?;
        Ok((crate::ou_mechanisms::parameters(&fit.model, &p.parameters)?, fit.converged))
    }
    fn check_quantity(&self, quantity: &DynamicQuantity) -> bool {
        let template = &self.problem.template;
        let n = template.equations.len();
        match quantity {
            DynamicQuantity::OuMechanismMean { replacements, variable } => {
                *variable < n && replacements_valid(template, replacements)
            }
            DynamicQuantity::OuMechanismCovariance { replacements, variable, other } => {
                *variable < n && *other < n && replacements_valid(template, replacements)
            }
            _ => false,
        }
    }
    fn quantity(
        &self,
        theta: &[f64],
        quantity: &DynamicQuantity,
        ctx: &ExecutionContext,
    ) -> Fallible<f64> {
        let model = self.model(theta);
        let n = self.problem.template.equations.len();
        let value = match quantity {
            DynamicQuantity::OuMechanismMean { replacements, variable } => {
                model.stationary(replacements, ctx)?.mean.get(*variable).copied()
            }
            DynamicQuantity::OuMechanismCovariance { replacements, variable, other } => {
                model.stationary(replacements, ctx)?.covariance.get(variable * n + other).copied()
            }
            _ => return Err(family_mismatch()),
        };
        value.ok_or_else(|| model_invalid("OU mechanism quantity coordinate"))
    }
    fn data_bytes(&self) -> usize {
        let n = self.problem.template.equations.len();
        self.problem
            .regimes
            .iter()
            .map(|r| r.values.len() * (n * 8 + 24) + r.replacements.len() * (2 * n + 1) * 8 + 128)
            .sum()
    }
}

// ------------------------------------------------------------- shared helpers

fn admit(
    ctx: &ExecutionContext,
    bytes: Option<usize>,
) -> Result<AdmissionLease, DynamicInferenceError> {
    if ctx.cancellation.is_cancelled() {
        return Err(cancelled());
    }
    let bytes = bytes
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or_else(|| invalid("allocation size overflows the address space"))?;
    ctx.acquire_admission(0, bytes).map_err(|e| DynamicInferenceError::Resource(e.to_string()))
}

/// Context for an indexed map of whole fits. Each fit's integrator or Lyapunov solve
/// reserves its own thread slot, so the map takes at most half the pool's threads and
/// both reservations always fit together.
fn worker_context(ctx: &ExecutionContext, work: usize) -> ExecutionContext {
    let half = (ctx.admission.thread_budget() / 2).max(1);
    let workers = ctx
        .parallelism
        .max_threads
        .get()
        .min(half)
        .min(u32::try_from(work).unwrap_or(u32::MAX))
        .max(1);
    let mut bounded = ctx.clone();
    bounded.parallelism =
        Parallelism::bounded(NonZeroThreadCount::new(workers).unwrap_or(NonZeroThreadCount::MIN));
    bounded
}

fn check_quantities<F: Family>(
    family: &F,
    quantities: &[DynamicQuantity],
) -> Result<(), DynamicInferenceError> {
    if quantities.iter().all(|q| family.check_quantity(q)) {
        Ok(())
    } else {
        Err(invalid("quantity shape, coordinate or family differs from the fitted family"))
    }
}

/// The reported fit must reproduce on the declared data at the reported estimate.
fn check_reported<F: Family>(
    family: &F,
    ctx: &ExecutionContext,
) -> Result<(), DynamicInferenceError> {
    let likelihood = family.log_likelihood(family.estimate(), ctx)?;
    let reported = family.reported_log_likelihood();
    if (likelihood - reported).abs() > 1e-8 * (1.0 + reported.abs()) {
        return Err(mismatch("reported log likelihood does not reproduce on the declared data"));
    }
    Ok(())
}

/// Evaluate quantities; a numerical failure stays an empty slot, resources abort.
fn evaluate_quantities<F: Family>(
    family: &F,
    theta: &[f64],
    quantities: &[DynamicQuantity],
    ctx: &ExecutionContext,
) -> Result<Vec<Option<f64>>, DynamicInferenceError> {
    quantities
        .iter()
        .map(|quantity| match family.quantity(theta, quantity, ctx) {
            Ok(value) => Ok(value.is_finite().then_some(value)),
            Err(EquilibriumError::Resource(message)) => {
                Err(DynamicInferenceError::Resource(message))
            }
            Err(_) => Ok(None),
        })
        .collect()
}

/// Separate a resource refusal (abort) from a numerical failure (reported).
fn settle<T>(value: Fallible<T>) -> Result<Result<T, String>, DynamicInferenceError> {
    match value {
        Ok(value) => Ok(Ok(value)),
        Err(EquilibriumError::Resource(message)) => Err(DynamicInferenceError::Resource(message)),
        Err(error) => Ok(Err(error.to_string())),
    }
}

fn cholesky(matrix: &[f64], n: usize) -> Option<Vec<f64>> {
    crate::linalg::SpdFactor::new(matrix, n).map(crate::linalg::SpdFactor::into_lower)
}

/// Inverse of `L L^T` from its Cholesky factor, symmetrized.
fn spd_inverse(lower: &[f64], n: usize) -> Vec<f64> {
    crate::linalg::inverse_from_lower(lower, n)
}

fn spread(values: &[f64]) -> Option<f64> {
    if values.len() < 2 {
        return None;
    }
    let count = values.len() as f64;
    let mean = values.iter().sum::<f64>() / count;
    Some((values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (count - 1.0)).sqrt())
}

fn step(value: f64, relative: f64) -> f64 {
    relative * value.abs().max(1.0)
}

fn shifted(theta: &[f64], moves: &[(usize, f64)]) -> Vec<f64> {
    let mut point = theta.to_vec();
    for &(index, delta) in moves {
        point[index] += delta;
    }
    point
}

fn uniform_index(rng: &mut CausalRng, size: usize) -> usize {
    let bound = size as u64;
    // Rejection sampling avoids modulo bias.
    let ceiling = u64::MAX - u64::MAX % bound;
    loop {
        let draw = rng.next_u64();
        if draw < ceiling {
            return usize::try_from(draw % bound).unwrap_or(size - 1);
        }
    }
}

fn draw_units(strata: &[Vec<usize>], rng: &mut CausalRng) -> Vec<usize> {
    let mut picks = Vec::with_capacity(strata.iter().map(Vec::len).sum());
    for stratum in strata {
        for _ in stratum {
            picks.push(stratum[uniform_index(rng, stratum.len())]);
        }
    }
    picks
}

// --------------------------------------------------------- observed information

/// Negative Hessian at `theta`: central differences of the analytic score where the
/// family has one, otherwise central second differences of the log likelihood.
fn negative_hessian<F: Family>(
    family: &F,
    theta: &[f64],
    ctx: &ExecutionContext,
) -> Result<(Result<Vec<f64>, String>, &'static str), DynamicInferenceError> {
    if family.has_score() {
        Ok((score_hessian(family, theta, ctx)?, "central_difference_of_analytic_score"))
    } else {
        Ok((likelihood_hessian(family, theta, ctx)?, "central_second_difference_of_likelihood"))
    }
}

fn score_hessian<F: Family>(
    family: &F,
    theta: &[f64],
    ctx: &ExecutionContext,
) -> Result<Result<Vec<f64>, String>, DynamicInferenceError> {
    let p = theta.len();
    let mut derivative = vec![0.0; p * p];
    for j in 0..p {
        if ctx.cancellation.is_cancelled() {
            return Err(cancelled());
        }
        let h = step(theta[j], 1e-5);
        let up = match settle(family.score(&shifted(theta, &[(j, h)]), ctx))? {
            Ok(score) => score,
            Err(detail) => return Ok(Err(detail)),
        };
        let down = match settle(family.score(&shifted(theta, &[(j, -h)]), ctx))? {
            Ok(score) => score,
            Err(detail) => return Ok(Err(detail)),
        };
        for i in 0..p {
            derivative[i * p + j] = (up[i] - down[i]) / (2.0 * h);
        }
    }
    Ok(Ok((0..p * p)
        .map(|k| {
            let (i, j) = (k / p, k % p);
            -0.5 * (derivative[i * p + j] + derivative[j * p + i])
        })
        .collect()))
}

fn likelihoods<F: Family>(
    family: &F,
    points: &[Vec<f64>],
    ctx: &ExecutionContext,
) -> Result<Result<Vec<f64>, String>, DynamicInferenceError> {
    let mut values = Vec::with_capacity(points.len());
    for point in points {
        match settle(family.log_likelihood(point, ctx))? {
            Ok(value) => values.push(value),
            Err(detail) => return Ok(Err(detail)),
        }
    }
    Ok(Ok(values))
}

fn likelihood_hessian<F: Family>(
    family: &F,
    theta: &[f64],
    ctx: &ExecutionContext,
) -> Result<Result<Vec<f64>, String>, DynamicInferenceError> {
    let p = theta.len();
    let steps: Vec<f64> = theta.iter().map(|value| step(*value, 1e-4)).collect();
    let center = match likelihoods(family, &[theta.to_vec()], ctx)? {
        Ok(values) => values[0],
        Err(detail) => return Ok(Err(detail)),
    };
    let mut information = vec![0.0; p * p];
    for i in 0..p {
        for j in i..p {
            if ctx.cancellation.is_cancelled() {
                return Err(cancelled());
            }
            let (hi, hj) = (steps[i], steps[j]);
            let value = if i == j {
                let points = [shifted(theta, &[(i, hi)]), shifted(theta, &[(i, -hi)])];
                match likelihoods(family, &points, ctx)? {
                    Ok(f) => (f[0] - 2.0 * center + f[1]) / (hi * hi),
                    Err(detail) => return Ok(Err(detail)),
                }
            } else {
                let points = [
                    shifted(theta, &[(i, hi), (j, hj)]),
                    shifted(theta, &[(i, hi), (j, -hj)]),
                    shifted(theta, &[(i, -hi), (j, hj)]),
                    shifted(theta, &[(i, -hi), (j, -hj)]),
                ];
                match likelihoods(family, &points, ctx)? {
                    Ok(f) => (f[0] - f[1] - f[2] + f[3]) / (4.0 * hi * hj),
                    Err(detail) => return Ok(Err(detail)),
                }
            };
            information[i * p + j] = -value;
            information[j * p + i] = -value;
        }
    }
    Ok(Ok(information))
}

fn classify(converged: bool, information: &[f64], p: usize) -> InformationRegularity {
    if information.iter().any(|v| !v.is_finite()) {
        return InformationRegularity::NotEvaluable { detail: "nonfinite information".into() };
    }
    if !converged {
        return InformationRegularity::EstimateNotConverged;
    }
    let Some(lower) = cholesky(information, p) else {
        return InformationRegularity::NotPositiveDefinite;
    };
    let (low, high) = (0..p)
        .map(|i| lower[i * p + i])
        .fold((f64::INFINITY, 0.0_f64), |(low, high), v| (low.min(v), high.max(v)));
    let pivot_ratio = (high / low).powi(2);
    if pivot_ratio > MAX_PIVOT_RATIO {
        InformationRegularity::IllConditioned { pivot_ratio }
    } else {
        InformationRegularity::Regular
    }
}

fn delta_method<F: Family>(
    family: &F,
    theta: &[f64],
    covariance: &[f64],
    quantities: &[DynamicQuantity],
    estimates: &[Option<f64>],
    ctx: &ExecutionContext,
) -> Result<Vec<Option<f64>>, DynamicInferenceError> {
    let p = theta.len();
    let mut output = Vec::with_capacity(quantities.len());
    'quantity: for (quantity, estimate) in quantities.iter().zip(estimates) {
        if estimate.is_none() {
            output.push(None);
            continue;
        }
        let mut gradient = Vec::with_capacity(p);
        for j in 0..p {
            let h = step(theta[j], 1e-6);
            let up = settle(family.quantity(&shifted(theta, &[(j, h)]), quantity, ctx))?;
            let down = settle(family.quantity(&shifted(theta, &[(j, -h)]), quantity, ctx))?;
            match (up, down) {
                (Ok(a), Ok(b)) if a.is_finite() && b.is_finite() => {
                    gradient.push((a - b) / (2.0 * h));
                }
                _ => {
                    output.push(None);
                    continue 'quantity;
                }
            }
        }
        let variance: f64 = (0..p)
            .map(|i| (0..p).map(|j| gradient[i] * covariance[i * p + j] * gradient[j]).sum::<f64>())
            .sum();
        output.push(Some(variance.max(0.0).sqrt()));
    }
    Ok(output)
}

fn observed_information<F: Family>(
    family: &F,
    design: DeclaredSamplingDesign,
    quantities: &[DynamicQuantity],
    ctx: &ExecutionContext,
) -> Result<DynamicObservedInformation, DynamicInferenceError> {
    if ctx.cancellation.is_cancelled() {
        return Err(cancelled());
    }
    let variance = design.apply_variance(VarianceRule::RowSubjectIid)?;
    check_quantities(family, quantities)?;
    let p = family.estimate().len();
    let q = quantities.len();
    let bytes = p
        .checked_mul(p)
        .and_then(|cells| cells.checked_mul(48))
        .and_then(|bytes| bytes.checked_add((q * (p + 4) + 8 * p) * 8 + 1024));
    let _lease = admit(ctx, bytes)?;
    check_reported(family, ctx)?;
    let estimate = family.estimate().to_vec();
    let quantity_estimates = evaluate_quantities(family, &estimate, quantities, ctx)?;
    let (information, derivative) = negative_hessian(family, &estimate, ctx)?;
    let regularity = match &information {
        Ok(matrix) => classify(family.converged(), matrix, p),
        Err(detail) => InformationRegularity::NotEvaluable { detail: detail.clone() },
    };
    let covariance = match (&regularity, &information) {
        (InformationRegularity::Regular, Ok(matrix)) => {
            cholesky(matrix, p).map(|lower| spd_inverse(&lower, p))
        }
        _ => None,
    };
    let standard_errors =
        covariance.as_ref().map(|c| (0..p).map(|j| c[j * p + j].max(0.0).sqrt()).collect());
    let quantity_standard_errors = match &covariance {
        Some(c) => delta_method(family, &estimate, c, quantities, &quantity_estimates, ctx)?,
        None => vec![None; q],
    };
    Ok(DynamicObservedInformation {
        estimate,
        information: information.ok(),
        regularity,
        covariance,
        standard_errors,
        quantity_estimates,
        quantity_standard_errors,
        derivative: derivative.into(),
        variance: variance.rule().as_str().into(),
    })
}

// ------------------------------------------------------------------ bootstrap

fn replicate<F: Family>(
    family: &F,
    strata: Option<&[Vec<usize>]>,
    quantities: &[DynamicQuantity],
    operation: SemanticOperation,
    seed: u64,
    index: usize,
    ctx: &ExecutionContext,
) -> Result<DynamicReplicate, DynamicInferenceError> {
    if ctx.cancellation.is_cancelled() {
        return Err(cancelled());
    }
    let key = RngKey::new(
        operation,
        FoldId::new(0),
        ChainId::new(0),
        ReplicateId::new(index as u64),
        ReductionPolicy::Strict,
    );
    let mut rng = key.open(seed);
    let data = match strata {
        Some(strata) => Ok(family.gather(&draw_units(strata, &mut rng))),
        None => family.simulate(&mut rng, ctx),
    };
    match data.and_then(|data| family.refit(&data, ctx)) {
        Ok((theta, true)) => Ok(DynamicReplicate {
            quantities: evaluate_quantities(family, &theta, quantities, ctx)?,
            parameters: Some(theta),
            failure: None,
        }),
        Ok((_, false)) => {
            Ok(DynamicReplicate::failed(ReplicateFailure::NotConverged, quantities.len()))
        }
        Err(EquilibriumError::Resource(message)) => Err(DynamicInferenceError::Resource(message)),
        Err(error) => Ok(DynamicReplicate::failed(
            ReplicateFailure::Numerical { detail: error.to_string() },
            quantities.len(),
        )),
    }
}

fn validate_bootstrap_replicates(replicates: usize) -> Result<(), DynamicInferenceError> {
    if !(1..=10_000).contains(&replicates) {
        return Err(invalid("bootstrap replicate count"));
    }
    Ok(())
}

fn bootstrap<F: Family>(
    family: &F,
    request: DynamicBootstrapRequest,
    quantities: &[DynamicQuantity],
    ctx: &ExecutionContext,
) -> Result<DynamicSamplingLaw, DynamicInferenceError> {
    if ctx.cancellation.is_cancelled() {
        return Err(cancelled());
    }
    validate_bootstrap_replicates(request.replicates)?;
    check_quantities(family, quantities)?;
    let (word, resampling, stratification) = match request.method {
        DynamicBootstrapMethod::Nonparametric => {
            let bound = request.design.apply_resampling(request.resampling)?;
            if bound.rule() != ResamplingRule::RowSubjectIid {
                return Err(invalid(
                    "dynamic units carry no cluster labels; only whole independent units are drawn",
                ));
            }
            (NONPARAMETRIC_WORD, bound.rule().as_str(), Some(request.stratification))
        }
        DynamicBootstrapMethod::Parametric => {
            request.design.apply_variance(VarianceRule::RowSubjectIid)?;
            if !family.converged() {
                return Err(invalid("a parametric bootstrap simulates from a converged estimate"));
            }
            (PARAMETRIC_WORD, "simulated_independent_units", None)
        }
    };
    let p = family.estimate().len();
    let q = quantities.len();
    let map_ctx = worker_context(ctx, request.replicates);
    let workers = usize::try_from(map_ctx.parallelism.max_threads.get()).unwrap_or(1);
    // Replicate slots, one gathered or simulated data copy plus fit scratch per worker,
    // and the stratum index.
    let bytes = request
        .replicates
        .checked_mul(
            p.checked_add(q)
                .and_then(|v| v.checked_mul(16))
                .and_then(|v| v.checked_add(128))
                .ok_or_else(|| invalid("bootstrap size overflow"))?,
        )
        .and_then(|slots| {
            family
                .data_bytes()
                .checked_mul(2)
                .and_then(|data| data.checked_mul(workers))
                .and_then(|data| data.checked_add(slots))
        })
        .and_then(|bytes| bytes.checked_add(family.units().checked_mul(16)?));
    let _lease = admit(ctx, bytes)?;
    check_reported(family, ctx)?;
    let quantity_estimates = evaluate_quantities(family, family.estimate(), quantities, ctx)?;
    let strata = stratification.map(|design| family.strata(design));
    let operation = SemanticOperation::new(family.stream() | word);
    let seed = ctx.rng.master_seed();
    let replicates = map_ctx.map_indexed(request.replicates, |index, inner| {
        replicate(family, strata.as_deref(), quantities, operation, seed, index, inner)
    })?;
    Ok(sampling_law(
        request,
        resampling,
        stratification,
        family.estimate().to_vec(),
        quantity_estimates,
        replicates,
    ))
}

fn sampling_law(
    request: DynamicBootstrapRequest,
    resampling: &str,
    stratification: Option<CyclicBootstrapDesign>,
    estimate: Vec<f64>,
    quantity_estimates: Vec<Option<f64>>,
    replicates: Vec<DynamicReplicate>,
) -> DynamicSamplingLaw {
    let count = replicates.len();
    let weight = 1.0 / count as f64;
    let p = estimate.len();
    let q = quantity_estimates.len();
    let successes: Vec<&[f64]> =
        replicates.iter().filter_map(|r| r.parameters.as_deref()).collect();
    let failed = count - successes.len();
    let standard_errors = (successes.len() >= 2).then(|| {
        (0..p)
            .map(|j| {
                let column: Vec<f64> = successes.iter().map(|theta| theta[j]).collect();
                spread(&column).unwrap_or(0.0)
            })
            .collect()
    });
    let mut quantity_standard_errors = Vec::with_capacity(q);
    let mut quantity_failed_mass = Vec::with_capacity(q);
    for k in 0..q {
        let values: Vec<f64> = replicates.iter().filter_map(|r| r.quantities[k]).collect();
        quantity_failed_mass.push((count - values.len()) as f64 * weight);
        quantity_standard_errors.push(spread(&values));
    }
    DynamicSamplingLaw {
        method: request.method,
        design: request.design.as_str().into(),
        resampling: resampling.into(),
        stratification,
        estimate,
        quantity_estimates,
        weights: vec![weight; count],
        failed_mass: failed as f64 * weight,
        replicates,
        standard_errors,
        quantity_standard_errors,
        quantity_failed_mass,
    }
}

// ------------------------------------------------------------------ posterior

fn preconditioner<F: Family>(
    family: &F,
    center: &[f64],
    prior: &DynamicPrior,
    ctx: &ExecutionContext,
) -> Result<(Vec<f64>, PosteriorPreconditioner), DynamicInferenceError> {
    let p = center.len();
    let (information, _) = negative_hessian(family, center, ctx)?;
    let Ok(mut precision) = information else {
        let mut factor = vec![0.0; p * p];
        for j in 0..p {
            factor[j * p + j] = prior.standard_deviation[j];
        }
        return Ok((factor, PosteriorPreconditioner::PriorScale));
    };
    for j in 0..p {
        precision[j * p + j] += prior.standard_deviation[j].powi(-2);
    }
    if let Some(factor) =
        cholesky(&precision, p).and_then(|lower| cholesky(&spd_inverse(&lower, p), p))
    {
        return Ok((factor, PosteriorPreconditioner::Laplace));
    }
    let mut factor = vec![0.0; p * p];
    for j in 0..p {
        let curvature = precision[j * p + j].max(prior.standard_deviation[j].powi(-2));
        factor[j * p + j] = curvature.sqrt().recip();
    }
    Ok((factor, PosteriorPreconditioner::DiagonalCurvature))
}

#[allow(
    clippy::too_many_lines,
    reason = "one pass validates, admits, preconditions, samples and maps draws in order"
)]
fn posterior<F: Family>(
    family: &F,
    prior: &DynamicPrior,
    options: DynamicPosteriorOptions,
    design: DeclaredSamplingDesign,
    quantities: &[DynamicQuantity],
    ctx: &ExecutionContext,
) -> Result<DynamicPosterior, DynamicInferenceError> {
    if ctx.cancellation.is_cancelled() {
        return Err(cancelled());
    }
    design.apply_variance(VarianceRule::RowSubjectIid)?;
    check_quantities(family, quantities)?;
    let p = family.estimate().len();
    let q = quantities.len();
    prior.validate(p)?;
    let sampler = options.sampler(p)?;
    let total =
        options.chains.checked_mul(options.draws).ok_or_else(|| invalid("posterior draw count"))?;
    // Whitened and mapped draws, quantity slots, and the preconditioning matrices.
    let bytes = total
        .checked_mul(p * 24 + q * 16 + 64)
        .and_then(|bytes| bytes.checked_add(p * p * 64 + 4096));
    let _lease = admit(ctx, bytes)?;
    check_reported(family, ctx)?;
    let center = family.estimate().to_vec();
    let (factor, preconditioner) = preconditioner(family, &center, prior, ctx)?;
    let to_theta = |z: &[f64]| -> Vec<f64> {
        (0..p).map(|i| center[i] + (0..=i).map(|j| factor[i * p + j] * z[j]).sum::<f64>()).collect()
    };
    let resource: RefCell<Option<String>> = RefCell::new(None);
    let density = |z: &[f64]| {
        let theta = to_theta(z);
        match family.log_likelihood(&theta, ctx) {
            Ok(likelihood) => likelihood + prior.log_density(&theta),
            Err(EquilibriumError::Resource(message)) => {
                resource.borrow_mut().get_or_insert(message);
                f64::NAN
            }
            // Unstable or singular dynamics lie outside the likelihood's support.
            Err(_) => f64::NEG_INFINITY,
        }
    };
    let sampled = match sample_metropolis_dispersed(
        &vec![0.0; p],
        options.start_spread,
        sampler,
        ctx,
        density,
    ) {
        Ok(sampled) => sampled,
        Err(error) => {
            return Err(match (resource.into_inner(), error) {
                (Some(message), _) | (None, MetropolisError::Resource(message)) => {
                    DynamicInferenceError::Resource(message)
                }
                (None, MetropolisError::Invalid(reason)) => invalid(reason),
            });
        }
    };
    let flat: Vec<f64> = sampled.samples.chunks_exact(p).flat_map(&to_theta).collect();
    let summary = antecedent_prob::mcmc_summary(&flat, options.chains, options.draws, p);
    let moved = antecedent_prob::all_chains_moved(&flat, options.chains, options.draws, p);
    let parameter_draws: Vec<Vec<f64>> = flat.chunks_exact(p).map(<[f64]>::to_vec).collect();
    drop(flat);
    let quantity_draws = if q == 0 {
        vec![Vec::new(); total]
    } else {
        worker_context(ctx, total).map_indexed(total, |index, inner| {
            if inner.cancellation.is_cancelled() {
                return Err(cancelled());
            }
            evaluate_quantities(family, &parameter_draws[index], quantities, inner)
        })?
    };
    let quantity_failed_mass = (0..q)
        .map(|k| {
            quantity_draws.iter().filter(|draw| draw[k].is_none()).count() as f64 / total as f64
        })
        .collect();
    let max_rhat = summary.max_rhat.is_finite().then_some(summary.max_rhat);
    Ok(DynamicPosterior {
        prior: prior.clone(),
        design: design.as_str().into(),
        preconditioner,
        parameter_draws,
        quantity_draws,
        quantity_failed_mass,
        chains: options.chains,
        draws_per_chain: options.draws,
        accepted: sampled.accepted,
        max_rhat,
        min_bulk_ess: summary.min_bulk_ess,
        min_tail_ess: summary.min_tail_ess,
        diagnostics_pass: moved
            && max_rhat.is_some_and(|r| r <= 1.01)
            && summary.min_bulk_ess >= 100.0
            && summary.min_tail_ess >= 100.0,
    })
}

// ------------------------------------------------------------- public entries

/// Bootstrap sampling law of a fitted dynamic family. Every replicate reruns the
/// declared procedure from its template. A nonparametric replicate draws whole
/// independent units within strata and refuses any rule the declared design does not
/// allow, including row draws inside a trajectory. A parametric replicate simulates
/// every unit from the converged estimate. Failed replicates keep their slots as
/// failed mass. No coverage claim is made.
///
/// # Errors
/// Design refusal, a fit that does not reproduce on the declared problem, malformed
/// quantities or counts, and cancellation or admission refusal.
pub fn bootstrap_dynamic_fit(
    target: DynamicFitRef<'_>,
    request: DynamicBootstrapRequest,
    quantities: &[DynamicQuantity],
    ctx: &ExecutionContext,
) -> Result<DynamicSamplingLaw, DynamicInferenceError> {
    if ctx.cancellation.is_cancelled() {
        return Err(cancelled());
    }
    match target {
        DynamicFitRef::Ode { problem, fit } => {
            bootstrap(&OdeFamily::new(problem, fit)?, request, quantities, ctx)
        }
        DynamicFitRef::Ou { problem, fit } => {
            bootstrap(&OuFamily::new(problem, fit)?, request, quantities, ctx)
        }
        DynamicFitRef::OuMechanisms { problem, fit } => {
            bootstrap(&OuMechanismFamily::new(problem, fit, ctx)?, request, quantities, ctx)
        }
    }
}

/// Observed information at the estimate and, when the likelihood is regular there,
/// its inverse covariance and delta-method quantity standard errors. A non-regular
/// likelihood returns the reason in place of a covariance.
///
/// # Errors
/// Design refusal of the independent variance rule, a fit that does not reproduce,
/// malformed quantities, and cancellation or admission refusal.
pub fn dynamic_observed_information(
    target: DynamicFitRef<'_>,
    design: DeclaredSamplingDesign,
    quantities: &[DynamicQuantity],
    ctx: &ExecutionContext,
) -> Result<DynamicObservedInformation, DynamicInferenceError> {
    if ctx.cancellation.is_cancelled() {
        return Err(cancelled());
    }
    match target {
        DynamicFitRef::Ode { problem, fit } => {
            observed_information(&OdeFamily::new(problem, fit)?, design, quantities, ctx)
        }
        DynamicFitRef::Ou { problem, fit } => {
            observed_information(&OuFamily::new(problem, fit)?, design, quantities, ctx)
        }
        DynamicFitRef::OuMechanisms { problem, fit } => observed_information(
            &OuMechanismFamily::new(problem, fit, ctx)?,
            design,
            quantities,
            ctx,
        ),
    }
}

/// Posterior over the estimated dynamic coordinates under the declared independent
/// normal prior and the family's likelihood (Gaussian measurement for ODEs, the
/// stationary Gaussian law for OU families), with draws of the requested quantities.
/// Structure, design and the fixed noise or diffusion law are not sampled.
///
/// # Errors
/// Design refusal, malformed prior or sampler policy, a fit that does not reproduce,
/// malformed quantities, and cancellation or admission refusal.
pub fn sample_dynamic_posterior(
    target: DynamicFitRef<'_>,
    prior: &DynamicPrior,
    options: DynamicPosteriorOptions,
    design: DeclaredSamplingDesign,
    quantities: &[DynamicQuantity],
    ctx: &ExecutionContext,
) -> Result<DynamicPosterior, DynamicInferenceError> {
    if ctx.cancellation.is_cancelled() {
        return Err(cancelled());
    }
    match target {
        DynamicFitRef::Ode { problem, fit } => {
            posterior(&OdeFamily::new(problem, fit)?, prior, options, design, quantities, ctx)
        }
        DynamicFitRef::Ou { problem, fit } => {
            posterior(&OuFamily::new(problem, fit)?, prior, options, design, quantities, ctx)
        }
        DynamicFitRef::OuMechanisms { problem, fit } => posterior(
            &OuMechanismFamily::new(problem, fit, ctx)?,
            prior,
            options,
            design,
            quantities,
            ctx,
        ),
    }
}

#[cfg(test)]
#[test]
fn dynamic_replicate_limit_preflight_checks_documented_boundaries() {
    assert!(validate_bootstrap_replicates(1).is_ok());
    assert!(validate_bootstrap_replicates(10_000).is_ok());
    assert!(validate_bootstrap_replicates(0).is_err());
    assert!(validate_bootstrap_replicates(10_001).is_err());
}

#[cfg(test)]
mod tests;
