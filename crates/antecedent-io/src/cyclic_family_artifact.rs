// SPDX-License-Identifier: MIT OR Apache-2.0
//! Retained dynamic/constraint model execution, identity and independent replay.
//! Families retain distinct semantics; no conversion from ODE/OU/constraints to
//! a static SCM is implied by using a common artifact transport.

use crate::cyclic_system_artifact::{CyclicArtifactError, MAX_CYCLIC_ARTIFACT_BYTES};
use antecedent_core::{CanonicalWriter, ExecutionContext};
use antecedent_model::{
    continuous_cyclic::{
        OdePolicy, OdeSystem, OdeTrajectory, OrnsteinUhlenbeck, OuMoments, TimedClamp,
    },
    equation_system::{IsolatedSolutions, IsolationPolicy, PolynomialSystem},
};
use serde::{Deserialize, Serialize};

/// Explicit multi-chain sampling/replay policy.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CyclicSamplingPolicy {
    /// Independent chain count.
    pub chains: usize,
    /// Discarded iterations per chain.
    pub warmup: usize,
    /// Retained draws per chain.
    pub draws: usize,
    /// Fixed symmetric random-walk proposal scale.
    pub proposal_scale: f64,
    /// Reproducible semantic stream seed.
    pub seed: u64,
}
impl CyclicSamplingPolicy {
    fn options(&self) -> antecedent_prob::metropolis::MetropolisOptions {
        antecedent_prob::metropolis::MetropolisOptions {
            chains: self.chains,
            warmup: self.warmup,
            draws: self.draws,
            proposal_scale: self.proposal_scale,
        }
    }
    fn context(&self, ctx: &ExecutionContext) -> ExecutionContext {
        let mut context = ctx.clone();
        context.rng = antecedent_core::RngFactory::from_seed(self.seed);
        context
    }
}

/// A complete typed dynamic or algebraic computation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "family", rename_all = "snake_case", deny_unknown_fields)]
pub enum CyclicFamilyRequest {
    /// Continuous theorem-derived functional under a specified or fitted Gaussian observational density.
    GaussianFunctional {
        /// Original theorem-scoped identification or adjustment source.
        source: crate::cyclic_graph::CyclicGraphRequest,
        /// Separately retained source identity.
        source_identity: String,
        /// Exact source-derived functional IR.
        program: crate::FunctionalProgramWire,
        /// Explicit Gaussian observational law or independent-row fitting recipe.
        law: crate::cyclic_gaussian_functional::GaussianCyclicLaw,
        /// Original outcome/conditioning/intervention assignments.
        assignments: Vec<Vec<(u32, f64)>>,
    },
    /// Projected nonlinear likelihood fitting with separately declared validation regimes.
    AdamFit {
        /// Serializable mechanisms and specified joint Gaussian disturbance covariance.
        model: antecedent_model::nonlinear_equilibrium::EquilibriumModel,
        /// Explicit mutable mechanism coordinates.
        parameters: Vec<antecedent_estimate::cyclic_likelihood::CyclicParameter>,
        /// Independent known-regime training equilibria.
        training: Vec<antecedent_model::nonlinear_equilibrium::RegimeRow>,
        /// Separate validation regimes, not final test observations.
        validation: Vec<antecedent_model::nonlinear_equilibrium::RegimeRow>,
        /// Semantic minibatch streams, contraction and stopping policies.
        policy: antecedent_estimate::cyclic_adam::CyclicAdamPolicy,
    },
    /// Exact eligible-mechanism lowering and bounded equilibrium root isolation.
    EquilibriumRoots {
        /// Original structural/noise model, retaining stable mechanism identities.
        model: antecedent_model::nonlinear_equilibrium::EquilibriumModel,
        /// Explicit bounded solution domain.
        domain: Vec<antecedent_model::equation_system::Interval>,
        /// Specified joint disturbance realization; no probability law is inferred.
        noise: Vec<f64>,
        /// Joint hard-clamp intervention.
        clamps: Vec<antecedent_model::nonlinear_equilibrium::Clamp>,
        /// Isolation work and refinement policy.
        policy: IsolationPolicy,
    },
    /// Shared affine stochastic mechanisms with named equation replacements.
    OuMechanisms {
        /// Baseline drift, forcing and joint Brownian diffusion factor.
        model: antecedent_estimate::ou_mechanisms::OuMechanisms,
        /// Known-regime independent stationary observations; empty for a specified model.
        regimes: Vec<antecedent_estimate::ou_mechanisms::OuMechanismRegime>,
        /// Shared baseline coordinates; empty means no fitting.
        parameters: Vec<antecedent_estimate::ou_mechanisms::OuMechanismParameter>,
        /// Bounded analytic stationary-likelihood optimizer policy.
        policy: antecedent_estimate::cyclic_gradient::CyclicGradientPolicy,
        /// Prespecified held-out or observational equation-replacement worlds.
        worlds: Vec<Vec<antecedent_estimate::ou_mechanisms::OuEquationReplacement>>,
    },
    /// Consume a theorem-derived functional with an explicit finite observational law.
    Functional {
        /// Original graphical identification request and scientific assumptions.
        source: crate::cyclic_graph::CyclicGraphRequest,
        /// Independently retained identification artifact identity.
        source_identity: String,
        /// Exact functional produced by that source.
        program: crate::FunctionalProgramWire,
        /// Observational joint law or empirical table.
        law: crate::cyclic_functional::FiniteCyclicLaw,
        /// Dense variable assignments at which to evaluate the functional.
        assignments: Vec<Vec<(u32, f64)>>,
    },
    /// Analytic likelihood optimization of compositional feedback mechanisms.
    GradientFit {
        /// Specified mechanisms and joint Gaussian disturbance covariance.
        model: antecedent_model::nonlinear_equilibrium::EquilibriumModel,
        /// Checked local parameter coordinates.
        parameters: Vec<antecedent_estimate::cyclic_likelihood::CyclicParameter>,
        /// Independent equilibria in known intervention regimes.
        rows: Vec<antecedent_model::nonlinear_equilibrium::RegimeRow>,
        /// Bounded local optimizer policy.
        policy: antecedent_estimate::cyclic_gradient::CyclicGradientPolicy,
    },
    /// Finite polynomial SCM with a joint cross-world disturbance/selection law.
    PolynomialCausal {
        /// Parameter prior, independent training units, factual evidence and action worlds.
        program: antecedent_estimate::polynomial_scm::PolynomialCausalProgram,
    },
    /// Named hard clamps, mechanism changes/replacements and structural shocks.
    EquilibriumActions {
        /// Factual structural/noise model.
        model: antecedent_model::nonlinear_equilibrium::EquilibriumModel,
        /// Prespecified named structural-action worlds.
        worlds: Vec<antecedent_model::cyclic_actions::EquilibriumActionWorld>,
        /// Explicit aligned disturbances, shared across all worlds.
        noise: Vec<Vec<f64>>,
        /// Numerical initialization; not a causal initial state or selection law.
        initial: Vec<f64>,
        /// Optional fully observed factual state, abducted under the original model.
        factual: Option<Vec<f64>>,
        /// Bounded numerical/certificate policy.
        policy: antecedent_model::nonlinear_equilibrium::SolvePolicy,
    },
    /// Finite polynomial parameter candidates with explicit equilibrium-selection and measurement laws.
    PolynomialFit {
        /// Named candidate systems and root-region selection laws.
        candidates: Vec<antecedent_estimate::constraint_likelihood::PolynomialCandidate>,
        /// Independent measurements with known Gaussian variances.
        measurements: Vec<antecedent_estimate::constraint_likelihood::PolynomialMeasurement>,
        /// Root-isolation work/width policy, retained for every candidate.
        policy: IsolationPolicy,
    },
    /// Conditional-structure IID-within-regime frequentist bootstrap responses.
    FrequentistBootstrap {
        /// Independent sampling unit, retained in the artifact identity.
        #[serde(default)]
        design: antecedent_estimate::cyclic_likelihood::CyclicBootstrapDesign,
        /// Fixed structural/noise template.
        model: antecedent_model::nonlinear_equilibrium::EquilibriumModel,
        /// Fitted parameter coordinates.
        parameters: Vec<antecedent_estimate::cyclic_likelihood::CyclicParameter>,
        /// Independent equilibrium observations with known regimes.
        rows: Vec<antecedent_model::nonlinear_equilibrium::RegimeRow>,
        /// Local likelihood optimization policy.
        policy: antecedent_estimate::cyclic_likelihood::CyclicFitOptions,
        /// Bounded bootstrap attempts; failed attempts keep their slots.
        replicates: usize,
        /// Replay seed.
        seed: u64,
        /// Prespecified response worlds.
        worlds: Vec<Vec<antecedent_model::nonlinear_equilibrium::Clamp>>,
        /// Aligned explicit disturbances, not an automatic population law.
        noise: Vec<Vec<f64>>,
        /// Numerical initialization.
        initial: Vec<f64>,
        /// Bounded equilibrium solve policy.
        solve_policy: antecedent_model::nonlinear_equilibrium::SolvePolicy,
    },
    /// Conditional parameter posterior with aligned response propagation.
    ParameterPosterior {
        /// Specified mechanism/noise template.
        model: antecedent_model::nonlinear_equilibrium::EquilibriumModel,
        /// Parameter coordinates with a declared independent normal prior.
        parameters: Vec<antecedent_estimate::cyclic_likelihood::CyclicParameter>,
        /// Independent known-regime equilibrium observations.
        rows: Vec<antecedent_model::nonlinear_equilibrium::RegimeRow>,
        /// Fixed normal prior standard deviation, restricted to model support.
        prior_sd: f64,
        /// Multi-chain sampling and replay policy.
        sampling: CyclicSamplingPolicy,
        /// Prespecified response worlds, evaluated with shared parameters/noise.
        worlds: Vec<Vec<antecedent_model::nonlinear_equilibrium::Clamp>>,
        /// Explicit aligned structural disturbance realizations; not automatically a population law.
        noise: Vec<Vec<f64>>,
        /// Numerical initialization.
        initial: Vec<f64>,
        /// Equilibrium solve policy.
        solve_policy: antecedent_model::nonlinear_equilibrium::SolvePolicy,
    },
    /// Joint parameter/factual posterior and shared-noise counterfactual propagation.
    JointCounterfactual {
        /// Fixed structure and joint disturbance model.
        model: antecedent_model::nonlinear_equilibrium::EquilibriumModel,
        /// Coordinates with a declared normal parameter prior.
        parameters: Vec<antecedent_estimate::cyclic_likelihood::CyclicParameter>,
        /// Independent known-regime training data.
        rows: Vec<antecedent_model::nonlinear_equilibrium::RegimeRow>,
        /// Prior standard deviation, restricted to contractive likelihood support.
        prior_sd: f64,
        /// Factual observation evidence, missing values explicitly represented.
        observations: Vec<Option<f64>>,
        /// Known independent measurement variances, zero denotes exact.
        measurement_variances: Vec<f64>,
        /// Factual latent-state initialization, not an equilibrium selection law.
        initial: Vec<f64>,
        /// Joint posterior sampling/replay policy.
        sampling: CyclicSamplingPolicy,
        /// Prespecified intervention worlds, aligned across joint posterior draws.
        worlds: Vec<Vec<antecedent_model::nonlinear_equilibrium::Clamp>>,
        /// Numerical solve policy.
        solve_policy: antecedent_model::nonlinear_equilibrium::SolvePolicy,
    },
    /// Partial/noisy factual abduction and shared-noise intervention prediction.
    PartialCounterfactual {
        /// Specified structural/noise model (not a parameter-posterior mixture).
        model: antecedent_model::nonlinear_equilibrium::EquilibriumModel,
        /// Exact/noisy observed values, None for unobserved coordinates.
        observations: Vec<Option<f64>>,
        /// Independent Gaussian measurement variances; zero means exact observation.
        measurement_variances: Vec<f64>,
        /// Latent factual-state sampler initialization.
        initial: Vec<f64>,
        /// Sampling/replay policy.
        sampling: CyclicSamplingPolicy,
        /// Counterfactual hard-clamp worlds.
        worlds: Vec<Vec<antecedent_model::nonlinear_equilibrium::Clamp>>,
        /// Equilibrium solve policy.
        solve_policy: antecedent_model::nonlinear_equilibrium::SolvePolicy,
    },

    /// Identification or FCI under an explicit cyclic graphical interpretation.
    Graph {
        /// Complete graph/data/query/premise program.
        request: crate::cyclic_graph::CyclicGraphRequest,
    },
    /// Finite-horizon deterministic dynamics.
    Ode {
        /// Named continuous-time drift mechanisms.
        model: OdeSystem,
        /// State at the first observation time.
        initial: Vec<f64>,
        /// Strictly increasing finite observation times.
        times: Vec<f64>,
        /// Time-indexed sustained mechanism replacements.
        schedule: Vec<TimedClamp>,
        /// Adaptive integration policy.
        policy: OdePolicy,
    },
    /// Stationary stochastic dynamics, with stability verified before the law is returned.
    OuStationary {
        /// Linear stochastic drift, center and diffusion.
        model: OrnsteinUhlenbeck,
    },
    /// Nonstationary Gaussian moments from an explicit initial law.
    OuTransient {
        /// Linear stochastic dynamics.
        model: OrnsteinUhlenbeck,
        /// Initial mean.
        initial_mean: Vec<f64>,
        /// Initial row-major covariance.
        initial_covariance: Vec<f64>,
        /// Strictly increasing observation times.
        times: Vec<f64>,
        /// Adaptive moment-integration policy.
        policy: OdePolicy,
    },
    /// Explicit sparse nonlinear equilibrium structure-selection/refit procedure.
    StructureFit {
        /// Semantic resampling seed, retained independently of the consumer context.
        #[serde(default)]
        seed: u64,
        /// Independent sampling unit for whole-selection resampling.
        #[serde(default)]
        design: antecedent_estimate::cyclic_likelihood::CyclicBootstrapDesign,
        /// Mechanism activations/intercepts and specified joint noise covariance.
        model: antecedent_model::nonlinear_equilibrium::EquilibriumModel,
        /// Independent known-regime training observations.
        training: Vec<antecedent_model::nonlinear_equilibrium::RegimeRow>,
        /// Separately supplied whole held-out intervention regimes, evaluation only.
        heldout: Vec<antecedent_model::nonlinear_equilibrium::RegimeRow>,
        /// Positive L1 penalty and local optimization policy.
        policy: antecedent_estimate::cyclic_likelihood::CyclicFitOptions,
        /// Explicit coefficient threshold before unpenalized refit.
        threshold: f64,
        /// Optional complete-selection bootstrap attempts, preserving failed slots.
        #[serde(default)]
        bootstrap_replicates: usize,
    },
    /// Fit deterministic drift with analytic sensitivities and fixed observation laws.
    OdeFit {
        /// Specified drift structure.
        model: OdeSystem,
        /// Named drift parameter coordinates.
        parameters: Vec<antecedent_model::continuous_cyclic::OdeParameter>,
        /// Independent observed trajectories.
        observations: Vec<antecedent_estimate::cyclic_dynamics::OdeObservation>,
        /// Adaptive integration policy.
        integration: OdePolicy,
        /// Local optimization policy.
        policy: antecedent_estimate::cyclic_dynamics::OdeFitPolicy,
    },
    /// Fit stationary stochastic dynamics with known sustained intervention regimes.
    OuFit {
        /// Drift/center template and specified diffusion.
        model: OrnsteinUhlenbeck,
        /// Drift/center coordinates to estimate.
        parameters: Vec<antecedent_estimate::cyclic_dynamics::OuParameter>,
        /// Independent stationary observations and known clamps.
        rows: Vec<antecedent_model::nonlinear_equilibrium::RegimeRow>,
        /// Local pattern-search policy (no L1 penalty).
        policy: antecedent_estimate::cyclic_likelihood::CyclicFitOptions,
    },
    /// Bounded polynomial equation/constraint solution set.
    Polynomial {
        /// Named variables and residual equations, including any explicit replacements.
        model: PolynomialSystem,
        /// Root-isolation budget and minimum width.
        policy: IsolationPolicy,
    },
}

/// Distinct answer representations prevent equilibrium/dynamic/set coercions.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "family", content = "result", rename_all = "snake_case", deny_unknown_fields)]
pub enum CyclicFamilyAnswer {
    /// Continuous Gaussian plug-in evaluation, retaining density fitting and theorem premises.
    GaussianFunctional(crate::cyclic_gaussian_functional::GaussianCyclicFunctionalReport),
    /// Best-validation feasible model and independent stopping/score diagnostics.
    AdamFit(antecedent_estimate::cyclic_adam::CyclicAdamFit),
    /// Shared stochastic-mechanism fit and stationary laws in the requested worlds.
    OuMechanisms {
        /// Explicitly absent for specified-model evaluation.
        fit: Option<antecedent_estimate::ou_mechanisms::OuMechanismFit>,
        /// Actual interventional stationary laws, preserving world order.
        laws: Vec<antecedent_estimate::ou_mechanisms::OuMechanismOutcome>,
    },
    /// Identified probabilities conditional on a declared finite observational law.
    Functional(crate::cyclic_functional::CyclicFunctionalReport),
    /// Local fitted model with full free-system likelihood diagnostics.
    GradientFit(antecedent_estimate::cyclic_gradient::CyclicGradientFit),
    /// Joint parameter/exogenous posterior and coupled multiple-equilibrium outcomes.
    PolynomialCausal(antecedent_estimate::polynomial_scm::PolynomialCausalReport),

    /// Conditional frequentist bootstrap, retaining fitting and response failures.
    FrequentistBootstrap {
        /// Bootstrap parameter vectors; None marks a failed fit.
        parameters: Vec<Option<Vec<f64>>>,
        /// `(world,bootstrap_attempt,noise_row)` answers, None for failed fits.
        worlds: Vec<Vec<Option<Vec<antecedent_model::nonlinear_equilibrium::EquilibriumOutcome>>>>,
    },

    /// Named structural-action outputs, with explicit failed response mass.
    EquilibriumActions(Vec<CyclicActionWorldReport>),

    /// Bounded-candidate likelihood ranking and explicit unresolved alternatives.
    PolynomialFit(antecedent_estimate::constraint_likelihood::PolynomialFit),
    /// Parameter uncertainty and aligned `(world,parameter_draw,noise_row)` responses.
    ParameterPosterior {
        /// Conditional-structure/covariance draws and convergence diagnostics.
        posterior: antecedent_estimate::cyclic_likelihood::CyclicPosterior,
        /// Response evaluations retaining all alignment axes.
        worlds: Vec<Vec<Vec<antecedent_model::nonlinear_equilibrium::EquilibriumOutcome>>>,
    },
    /// Joint parameter/factual-state posterior, with shared disturbances across worlds.
    JointCounterfactual {
        /// Parameter and factual draws, updated with both training and factual evidence.
        posterior: antecedent_estimate::cyclic_likelihood::CyclicJointAbduction,
        /// `(world,joint_draw)` predictions preserving parameter/noise dependence.
        worlds: Vec<Vec<antecedent_model::nonlinear_equilibrium::EquilibriumOutcome>>,
    },
    /// Latent factual/noise posterior and `(world,abduction_draw)` counterfactuals.
    PartialCounterfactual {
        /// Explicit observation-model abduction and diagnostics.
        abduction: antecedent_estimate::cyclic_likelihood::CyclicAbductionPosterior,
        /// Predictions sharing each factual disturbance realization across worlds.
        worlds: Vec<Vec<antecedent_model::nonlinear_equilibrium::EquilibriumOutcome>>,
    },
    /// Graphical derivation or cyclic PAG evidence, never a specified-model fit.
    Graph(crate::cyclic_graph::CyclicGraphReport),
    /// Finite-horizon ODE path.
    Ode(OdeTrajectory),
    /// Row-major stationary OU covariance.
    OuStationary(Vec<f64>),
    /// Finite-horizon OU mean/covariance.
    OuTransient(OuMoments),
    /// Selected parametric topology and conditional refit, not nonparametric graph ID.
    StructureFit {
        /// Selected topology and conditional refit.
        fit: antecedent_estimate::cyclic_likelihood::CyclicStructureFit,
        /// Complete selection/refit bootstrap attempts; failures retain their slots.
        bootstrap: Vec<Option<antecedent_estimate::cyclic_likelihood::CyclicStructureFit>>,
    },
    /// Sensitivity-based deterministic drift fit and convergence facts.
    OdeFit(antecedent_estimate::cyclic_dynamics::OdeFit),
    /// Stationary OU parameter fit, conditional on declared diffusion/structure.
    OuFit(antecedent_estimate::cyclic_dynamics::OuFit),
    /// Certified roots and unresolved regions; no implicit selection law.
    Polynomial(IsolatedSolutions),
}

/// Draw-aligned outcome of one named structural-action world.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CyclicActionWorldReport {
    /// Prespecified world identifier.
    pub id: String,
    /// Actual post-action SCC schedule.
    pub blocks: Vec<Vec<usize>>,
    /// Outcomes retain failed draws in their original positions.
    pub rows: Vec<antecedent_model::nonlinear_equilibrium::EquilibriumOutcome>,
    /// Shared-disturbance prediction from the original factual state, when supplied.
    pub counterfactual: Option<antecedent_model::nonlinear_equilibrium::EquilibriumOutcome>,
}

/// Identity-bound answer of the specified model.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CyclicFamilyReport {
    /// Independently retained request digest.
    pub identity: String,
    /// Typed family answer.
    pub answer: CyclicFamilyAnswer,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Record {
    version: u32,
    request: CyclicFamilyRequest,
    report: CyclicFamilyReport,
}
fn invalid(e: impl std::fmt::Display) -> CyclicArtifactError {
    CyclicArtifactError::Invalid(e.to_string())
}

struct BoundedJsonSize {
    bytes: usize,
}
impl std::io::Write for BoundedJsonSize {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.bytes = self
            .bytes
            .checked_add(bytes.len())
            .filter(|&n| n <= MAX_CYCLIC_ARTIFACT_BYTES)
            .ok_or_else(|| std::io::Error::other("cyclic family byte limit"))?;
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}
fn bounded_json<T: Serialize>(
    value: &T,
    ctx: &ExecutionContext,
) -> Result<(Vec<u8>, antecedent_core::execution::AdmissionLease), CyclicArtifactError> {
    let mut size = BoundedJsonSize { bytes: 0 };
    serde_json::to_writer(&mut size, value).map_err(invalid)?;
    let retention =
        ctx.acquire_admission(0, (size.bytes as u64).saturating_mul(4)).map_err(invalid)?;
    let bytes = serde_json::to_vec(value).map_err(invalid)?;
    Ok((bytes, retention))
}
impl CyclicFamilyRequest {
    /// Stable semantic/operator family without executing or allocating a payload.
    #[must_use]
    pub const fn family_name(&self) -> &'static str {
        match self {
            Self::AdamFit { .. } => "adam_fit",
            Self::EquilibriumRoots { .. } => "equilibrium_roots",
            Self::OuMechanisms { .. } => "ou_mechanisms",
            Self::Functional { .. } => "functional",
            Self::GaussianFunctional { .. } => "gaussian_functional",
            Self::GradientFit { .. } => "gradient_fit",
            Self::PolynomialCausal { .. } => "polynomial_causal",
            Self::EquilibriumActions { .. } => "equilibrium_actions",
            Self::FrequentistBootstrap { .. } => "frequentist_bootstrap",
            Self::PolynomialFit { .. } => "polynomial_fit",
            Self::ParameterPosterior { .. } => "parameter_posterior",
            Self::JointCounterfactual { .. } => "joint_counterfactual",
            Self::PartialCounterfactual { .. } => "partial_counterfactual",
            Self::Graph { .. } => "graph",
            Self::Ode { .. } => "ode",
            Self::OuStationary { .. } => "ou_stationary",
            Self::OuTransient { .. } => "ou_transient",
            Self::StructureFit { .. } => "structure_fit",
            Self::OdeFit { .. } => "ode_fit",
            Self::OuFit { .. } => "ou_fit",
            Self::Polynomial { .. } => "polynomial",
        }
    }
    /// Execute on the common resource/cancellation context, preserving family semantics.
    #[allow(
        clippy::too_many_lines,
        reason = "typed scientific family dispatch retains distinct operator inputs and answer representations"
    )]
    pub fn execute(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<CyclicFamilyReport, CyclicArtifactError> {
        if ctx.cancellation.is_cancelled() {
            return Err(invalid("cyclic execution cancelled"));
        }
        // Validate recursive expression representations before serializing a typed
        // Rust request; the Python parser independently bounds nesting during decode.
        match self {
            Self::AdamFit { model, .. }
            | Self::EquilibriumRoots { model, .. }
            | Self::GradientFit { model, .. }
            | Self::EquilibriumActions { model, .. }
            | Self::FrequentistBootstrap { model, .. }
            | Self::ParameterPosterior { model, .. }
            | Self::JointCounterfactual { model, .. }
            | Self::PartialCounterfactual { model, .. }
            | Self::StructureFit { model, .. } => model.validate()?,
            Self::Ode { model, .. } | Self::OdeFit { model, .. } => model.validate()?,
            _ => {}
        }
        let (request, _request_retention) = bounded_json(self, ctx)?;
        let identity =
            CanonicalWriter::digest_bytes_hex("antecedent.cyclic_family.request.v1", &request);
        let answer = match self {
            Self::GaussianFunctional { source, source_identity, program, law, assignments } => {
                CyclicFamilyAnswer::GaussianFunctional(
                    crate::cyclic_gaussian_functional::evaluate_gaussian_cyclic_functional(
                        source,
                        source_identity,
                        program,
                        law,
                        assignments,
                        ctx,
                    )?,
                )
            }
            Self::AdamFit { model, parameters, training, validation, policy } => {
                CyclicFamilyAnswer::AdamFit(antecedent_estimate::cyclic_adam::fit_cyclic_adam(
                    model, parameters, training, validation, *policy, ctx,
                )?)
            }

            Self::EquilibriumRoots { model, domain, noise, clamps, policy } => {
                let system =
                    antecedent_model::equilibrium_polynomial::compile_polynomial_equilibrium(
                        model, domain, noise, clamps, ctx,
                    )?;
                CyclicFamilyAnswer::Polynomial(system.isolate(*policy, ctx)?)
            }

            Self::OuMechanisms { model, regimes, parameters, policy, worlds } => {
                if worlds.is_empty()
                    || worlds.len() > 128
                    || (parameters.is_empty() != regimes.is_empty())
                {
                    return Err(invalid("OU mechanism world/fit design"));
                }
                let fit = if parameters.is_empty() {
                    None
                } else {
                    Some(antecedent_estimate::ou_mechanisms::fit_ou_mechanisms(
                        model, parameters, regimes, *policy, ctx,
                    )?)
                };
                let fitted = fit.as_ref().map_or(model, |f| &f.model);
                let laws = worlds
                    .iter()
                    .map(|replacements| fitted.stationary_outcome(replacements, ctx))
                    .collect::<Result<Vec<_>, _>>()?;
                CyclicFamilyAnswer::OuMechanisms { fit, laws }
            }

            Self::Functional { source, source_identity, program, law, assignments } => {
                CyclicFamilyAnswer::Functional(
                    crate::cyclic_functional::evaluate_cyclic_functional(
                        source,
                        source_identity,
                        program,
                        law,
                        assignments,
                        ctx,
                    )?,
                )
            }
            Self::GradientFit { model, parameters, rows, policy } => {
                CyclicFamilyAnswer::GradientFit(
                    antecedent_estimate::cyclic_gradient::fit_cyclic_gradient(
                        model, parameters, rows, *policy, ctx,
                    )?,
                )
            }
            Self::PolynomialCausal { program } => {
                CyclicFamilyAnswer::PolynomialCausal(program.execute(ctx)?)
            }
            Self::FrequentistBootstrap {
                design,
                model,
                parameters,
                rows,
                policy,
                replicates,
                seed,
                worlds,
                noise,
                initial,
                solve_policy,
            } => {
                let count = replicates
                    .saturating_mul(worlds.len())
                    .saturating_mul(noise.len())
                    .saturating_mul(model.order());
                if *replicates == 0
                    || *replicates > 1000
                    || worlds.is_empty()
                    || noise.is_empty()
                    || count > 100_000
                    || rows.len() > 10_000
                {
                    return Err(invalid("frequentist bootstrap response/data limits"));
                }
                let _retention = ctx
                    .acquire_admission(0, (count * 128 + replicates * parameters.len() * 16) as u64)
                    .map_err(invalid)?;
                let mut context = ctx.clone();
                context.rng = antecedent_core::RngFactory::from_seed(*seed);
                let draws =
                    antecedent_estimate::cyclic_likelihood::bootstrap_cyclic_fit_with_design(
                        model,
                        parameters,
                        rows,
                        *policy,
                        *replicates,
                        *design,
                        &context,
                    )?;
                let mut output = Vec::with_capacity(worlds.len());
                for clamps in worlds {
                    let mut attempts = Vec::with_capacity(draws.len());
                    for draw in &draws {
                        let answer = draw
                            .as_ref()
                            .map(|values| {
                                let fitted =
                                    antecedent_estimate::cyclic_likelihood::with_parameters(
                                        model, parameters, values,
                                    )?;
                                noise
                                    .iter()
                                    .map(|u| {
                                        fitted.solve_outcome(u, clamps, initial, *solve_policy, ctx)
                                    })
                                    .collect::<Result<
                                        Vec<_>,
                                        antecedent_model::nonlinear_equilibrium::EquilibriumError,
                                    >>()
                            })
                            .transpose()?;
                        attempts.push(answer);
                    }
                    output.push(attempts);
                }
                CyclicFamilyAnswer::FrequentistBootstrap { parameters: draws, worlds: output }
            }
            Self::EquilibriumActions { model, worlds, noise, initial, factual, policy } => {
                model.validate()?;
                let count = worlds
                    .len()
                    .saturating_mul(noise.len() + usize::from(factual.is_some()))
                    .saturating_mul(model.order());
                if worlds.is_empty()
                    || worlds.len() > 256
                    || count > 100_000
                    || (noise.is_empty() && factual.is_none())
                {
                    return Err(invalid("action world/response limits"));
                }
                let mut ids = std::collections::BTreeSet::new();
                let mut plans = Vec::with_capacity(worlds.len());
                for world in worlds {
                    if world.id.trim().is_empty() || world.id.len() > 256 || !ids.insert(&world.id)
                    {
                        return Err(invalid("unique nonempty action world identities required"));
                    }
                    plans.push(model.compile_actions(&world.actions)?);
                }
                let _retention = ctx
                    .acquire_admission(0, (count * 128 + worlds.len() * 512) as u64)
                    .map_err(invalid)?;
                let factual_noise =
                    factual.as_ref().map(|state| model.abduce(state)).transpose()?;
                let mut output = Vec::with_capacity(worlds.len());
                for (world, plan) in worlds.iter().zip(plans) {
                    let rows = noise
                        .iter()
                        .map(|u| plan.solve(u, initial, *policy, ctx))
                        .collect::<Result<Vec<_>, _>>()?;
                    let counterfactual = factual_noise
                        .as_ref()
                        .map(|u| plan.solve(u, initial, *policy, ctx))
                        .transpose()?;
                    output.push(CyclicActionWorldReport {
                        id: world.id.clone(),
                        blocks: plan.blocks()?,
                        rows,
                        counterfactual,
                    });
                }
                CyclicFamilyAnswer::EquilibriumActions(output)
            }
            Self::PolynomialFit { candidates, measurements, policy } => {
                CyclicFamilyAnswer::PolynomialFit(
                    antecedent_estimate::constraint_likelihood::fit_polynomial_candidates(
                        candidates,
                        measurements,
                        *policy,
                        ctx,
                    )?,
                )
            }
            Self::ParameterPosterior {
                model,
                parameters,
                rows,
                prior_sd,
                sampling,
                worlds,
                noise,
                initial,
                solve_policy,
            } => {
                let count = sampling
                    .chains
                    .checked_mul(sampling.draws)
                    .and_then(|n| n.checked_mul(worlds.len()))
                    .and_then(|n| n.checked_mul(noise.len()))
                    .ok_or_else(|| invalid("posterior response size overflow"))?;
                if count > 100_000 || worlds.is_empty() || noise.is_empty() || rows.len() > 10_000 {
                    return Err(invalid("posterior response/data limits"));
                }
                let _retention = ctx
                    .acquire_admission(0, (count * (model.order() * 128 + 512)) as u64)
                    .map_err(invalid)?;
                let context = sampling.context(ctx);
                let posterior = antecedent_estimate::cyclic_likelihood::sample_cyclic_posterior(
                    model,
                    parameters,
                    rows,
                    *prior_sd,
                    sampling.options(),
                    &context,
                )?;
                let mut output = Vec::with_capacity(worlds.len());
                for clamps in worlds {
                    let mut by_parameter = Vec::new();
                    for draw in &posterior.samples {
                        let fitted = antecedent_estimate::cyclic_likelihood::with_parameters(
                            model, parameters, draw,
                        )?;
                        by_parameter.push(
                            noise
                                .iter()
                                .map(|noise| {
                                    fitted.solve_outcome(noise, clamps, initial, *solve_policy, ctx)
                                })
                                .collect::<Result<Vec<_>, _>>()?,
                        );
                    }
                    output.push(by_parameter);
                }
                CyclicFamilyAnswer::ParameterPosterior { posterior, worlds: output }
            }
            Self::JointCounterfactual {
                model,
                parameters,
                rows,
                prior_sd,
                observations,
                measurement_variances,
                initial,
                sampling,
                worlds,
                solve_policy,
            } => {
                let count = sampling
                    .chains
                    .checked_mul(sampling.draws)
                    .and_then(|draws| draws.checked_mul(worlds.len()))
                    .ok_or_else(|| invalid("joint counterfactual size overflow"))?;
                if count > 100_000 || worlds.is_empty() || rows.len() > 10_000 {
                    return Err(invalid("joint counterfactual retention/data limits"));
                }
                let _retention = ctx
                    .acquire_admission(0, (count * (model.order() * 128 + 512)) as u64)
                    .map_err(invalid)?;
                let posterior = antecedent_estimate::cyclic_likelihood::abduce_cyclic_joint(
                    model,
                    parameters,
                    rows,
                    *prior_sd,
                    observations,
                    measurement_variances,
                    initial,
                    sampling.options(),
                    &sampling.context(ctx),
                )?;
                let mut output = Vec::with_capacity(worlds.len());
                for clamps in worlds {
                    let mut draws = Vec::with_capacity(posterior.parameter_draws.len());
                    for ((values, factual), noise) in posterior
                        .parameter_draws
                        .iter()
                        .zip(&posterior.factual.states)
                        .zip(&posterior.factual.disturbances)
                    {
                        let draw_model = antecedent_estimate::cyclic_likelihood::with_parameters(
                            model, parameters, values,
                        )?;
                        draws.push(draw_model.solve_outcome(
                            noise,
                            clamps,
                            factual,
                            *solve_policy,
                            ctx,
                        )?);
                    }
                    output.push(draws);
                }
                CyclicFamilyAnswer::JointCounterfactual { posterior, worlds: output }
            }
            Self::PartialCounterfactual {
                model,
                observations,
                measurement_variances,
                initial,
                sampling,
                worlds,
                solve_policy,
            } => {
                let count = sampling
                    .chains
                    .checked_mul(sampling.draws)
                    .and_then(|n| n.checked_mul(worlds.len()))
                    .ok_or_else(|| invalid("counterfactual size overflow"))?;
                if count > 100_000 || worlds.is_empty() {
                    return Err(invalid("counterfactual retention limit"));
                }
                let _retention = ctx
                    .acquire_admission(0, (count * (model.order() * 128 + 512)) as u64)
                    .map_err(invalid)?;
                let context = sampling.context(ctx);
                let abduction = antecedent_estimate::cyclic_likelihood::abduce_cyclic_partial(
                    model,
                    observations,
                    measurement_variances,
                    initial,
                    sampling.options(),
                    &context,
                )?;
                let mut output = Vec::with_capacity(worlds.len());
                for clamps in worlds {
                    output.push(
                        abduction
                            .disturbances
                            .iter()
                            .zip(&abduction.states)
                            .map(|(noise, factual)| {
                                model.solve_outcome(noise, clamps, factual, *solve_policy, ctx)
                            })
                            .collect::<Result<Vec<_>, _>>()?,
                    );
                }
                CyclicFamilyAnswer::PartialCounterfactual { abduction, worlds: output }
            }
            Self::Graph { request } => CyclicFamilyAnswer::Graph(request.execute(ctx)?),
            Self::Ode { model, initial, times, schedule, policy } => {
                CyclicFamilyAnswer::Ode(model.trajectory(initial, times, schedule, *policy, ctx)?)
            }
            Self::OuStationary { model } => {
                CyclicFamilyAnswer::OuStationary(model.stationary_covariance(ctx)?)
            }
            Self::OuTransient { model, initial_mean, initial_covariance, times, policy } => {
                CyclicFamilyAnswer::OuTransient(model.moments(
                    initial_mean,
                    initial_covariance,
                    times,
                    *policy,
                    ctx,
                )?)
            }
            Self::StructureFit {
                seed,
                design,
                model,
                training,
                heldout,
                policy,
                threshold,
                bootstrap_replicates,
            } => {
                let fit = antecedent_estimate::cyclic_likelihood::learn_cyclic_structure(
                    model, training, heldout, *policy, *threshold, ctx,
                )?;
                let mut context = ctx.clone();
                context.rng = antecedent_core::RngFactory::from_seed(*seed);
                let bootstrap =
                    antecedent_estimate::cyclic_likelihood::bootstrap_cyclic_structure_with_design(
                        model,
                        training,
                        heldout,
                        *policy,
                        *threshold,
                        *bootstrap_replicates,
                        *design,
                        &context,
                    )?;
                CyclicFamilyAnswer::StructureFit { fit, bootstrap }
            }
            Self::OdeFit { model, parameters, observations, integration, policy } => {
                CyclicFamilyAnswer::OdeFit(
                    antecedent_estimate::cyclic_dynamics::fit_ode_likelihood(
                        model,
                        parameters,
                        observations,
                        *integration,
                        *policy,
                        ctx,
                    )?,
                )
            }
            Self::OuFit { model, parameters, rows, policy } => {
                CyclicFamilyAnswer::OuFit(antecedent_estimate::cyclic_dynamics::fit_ou_likelihood(
                    model, parameters, rows, *policy, ctx,
                )?)
            }
            Self::Polynomial { model, policy } => {
                CyclicFamilyAnswer::Polynomial(model.isolate(*policy, ctx)?)
            }
        };
        Ok(CyclicFamilyReport { identity, answer })
    }
    /// Execute and encode with the existing family-neutral artifact container.
    pub fn export(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<(CyclicFamilyReport, Vec<u8>), CyclicArtifactError> {
        #[derive(Serialize)]
        struct View<'a> {
            version: u16,
            request: &'a CyclicFamilyRequest,
            report: &'a CyclicFamilyReport,
        }
        let report = self.execute(ctx)?;
        let (bytes, _record_retention) =
            bounded_json(&View { version: 1, request: self, report: &report }, ctx)?;
        let artifact = crate::encode_container(
            &crate::ArtifactContainer {
                kind: "cyclic_family",
                note: "specified_dynamic_or_constraint_model",
                max_bytes: MAX_CYCLIC_ARTIFACT_BYTES,
                artifact_id: "cyclic_family",
                library_version: env!("CARGO_PKG_VERSION"),
            },
            &[crate::OutSection {
                id: "cyclic_family_v1",
                content_type: crate::CONTENT_BYTES,
                bytes: &bytes,
            }],
        )
        .map_err(|e| invalid(e.into_io("cyclic family layout")))?;
        Ok((report, artifact))
    }
}

/// Check a separately retained request identity and recompute the typed answer.
pub fn consume_cyclic_family(
    bytes: &[u8],
    expected_identity: &str,
    ctx: &ExecutionContext,
) -> Result<CyclicFamilyReport, CyclicArtifactError> {
    if bytes.len() > MAX_CYCLIC_ARTIFACT_BYTES || expected_identity.len() != 64 {
        return Err(invalid("family byte limit or expected identity"));
    }
    let _retention =
        ctx.acquire_admission(0, (bytes.len() as u64).saturating_mul(4)).map_err(invalid)?;
    let parts = crate::decode_container(
        "cyclic_family",
        MAX_CYCLIC_ARTIFACT_BYTES,
        &["cyclic_family_v1"],
        bytes,
    )
    .map_err(|e| invalid(e.into_io("cyclic family layout")))?;
    let record: Record =
        serde_json::from_slice(parts.first().ok_or_else(|| invalid("missing family section"))?)
            .map_err(invalid)?;
    if record.version != 1 {
        return Err(invalid("unsupported cyclic family version"));
    }
    let request_bytes = serde_json::to_vec(&record.request).map_err(invalid)?;
    if CanonicalWriter::digest_bytes_hex("antecedent.cyclic_family.request.v1", &request_bytes)
        != expected_identity
        || record.report.identity != expected_identity
    {
        return Err(CyclicArtifactError::Mismatch);
    }
    let report = record.request.execute(ctx)?;
    if report.identity != expected_identity || report != record.report {
        return Err(CyclicArtifactError::Mismatch);
    }
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ou_family_replay_is_bound_to_continuous_time_semantics() {
        let request = CyclicFamilyRequest::OuStationary {
            model: OrnsteinUhlenbeck { drift: vec![-2.0], center: vec![0.0], diffusion: vec![4.0] },
        };
        let ctx = ExecutionContext::for_tests(0);
        let (report, bytes) = request.export(&ctx).unwrap();
        assert_eq!(consume_cyclic_family(&bytes, &report.identity, &ctx).unwrap(), report);
        assert!(consume_cyclic_family(&bytes, &"0".repeat(64), &ctx).is_err());
    }
}

/// Independently replay a family and retain its verified executable request for
/// scientific coordinate/evidence adapters. The request is returned only after
/// the ordinary identity, semantic replay and exact answer checks succeed.
pub fn consume_cyclic_family_with_request(
    bytes: &[u8],
    expected_identity: &str,
    ctx: &ExecutionContext,
) -> Result<(CyclicFamilyRequest, CyclicFamilyReport), CyclicArtifactError> {
    let report = consume_cyclic_family(bytes, expected_identity, ctx)?;
    let parts = crate::decode_container(
        "cyclic_family",
        MAX_CYCLIC_ARTIFACT_BYTES,
        &["cyclic_family_v1"],
        bytes,
    )
    .map_err(|e| invalid(e.into_io("cyclic source layout")))?;
    let record: Record =
        serde_json::from_slice(parts.first().ok_or_else(|| invalid("missing family section"))?)
            .map_err(invalid)?;
    Ok((record.request, report))
}

#[cfg(test)]
mod seed_binding_tests {
    use super::*;
    use antecedent_model::nonlinear_equilibrium::{
        Activation, Clamp, Equation, EquilibriumModel, RegimeRow,
    };
    #[test]
    fn selection_bootstrap_replay_uses_request_seed_not_consumer_rng() {
        let model = EquilibriumModel {
            equations: vec![
                Equation {
                    id: "x".into(),
                    intercept: 0.,
                    weights: vec![0., 0.],
                    activation: Activation::Identity,
                    expression: None,
                    parameters: vec![],
                },
                Equation {
                    id: "y".into(),
                    intercept: 0.,
                    weights: vec![0., 0.],
                    activation: Activation::Identity,
                    expression: None,
                    parameters: vec![],
                },
            ],
            noise_covariance: vec![1., 0., 0., 1.],
        };
        let rows: Vec<_> = [-2., -1., 1., 2.]
            .into_iter()
            .flat_map(|x| {
                [-0.1, 0.1].into_iter().map(move |e| RegimeRow {
                    values: vec![x, 0.6 * x + e],
                    clamps: vec![Clamp { variable: 0, value: x }],
                })
            })
            .collect();
        let request=CyclicFamilyRequest::StructureFit{seed:71,design:antecedent_estimate::cyclic_likelihood::CyclicBootstrapDesign::RandomizedWithinTargets,model,training:rows,heldout:vec![],policy:antecedent_estimate::cyclic_likelihood::CyclicFitOptions{initial_step:0.2,tolerance:0.001,max_sweeps:200,l1_penalty:0.1},threshold:0.05,bootstrap_replicates:2};
        let (report, bytes) = request.export(&ExecutionContext::for_tests(1)).unwrap();
        let CyclicFamilyAnswer::StructureFit { bootstrap, .. } = &report.answer else { panic!() };
        assert!(bootstrap.iter().any(Option::is_some));
        assert_eq!(
            consume_cyclic_family(&bytes, &report.identity, &ExecutionContext::for_tests(999))
                .unwrap(),
            report
        );
    }
}
