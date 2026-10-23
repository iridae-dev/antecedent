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

#[path = "cyclic_family_program.rs"]
mod compiled;
pub use compiled::CompiledCyclicFamilyProgram;
use compiled::FrozenFamilyFitValue;
pub(crate) use compiled::FrozenGaussianDensity;

#[path = "cyclic_family_worlds.rs"]
mod worlds;

#[path = "cyclic_family_contract.rs"]
mod contract;
pub use contract::{
    CyclicProblem, CyclicRealization, CyclicUncertainty, cyclic_contract_payloads,
    cyclic_identification, cyclic_problem, cyclic_realization, design_unresolved,
};

/// Identity of the lowered program a prepared handle executed, stamped on its report.
///
/// Like an admission reservation it is local to the process: it is omitted from
/// every artifact and ignored by scientific equality, so an independently replayed
/// report equals the stamped one.
#[derive(Clone, Copy, Debug, Default)]
pub struct LoweredStamp(Option<antecedent_core::SemanticDigest>);

impl LoweredStamp {
    /// Stamp of `identity`.
    #[must_use]
    pub const fn of(identity: antecedent_core::SemanticDigest) -> Self {
        Self(Some(identity))
    }

    /// The stamped identity, when one was stamped.
    #[must_use]
    pub const fn identity(self) -> Option<antecedent_core::SemanticDigest> {
        self.0
    }
}

impl PartialEq for LoweredStamp {
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}
pub use worlds::{
    CertificateDemandWire, CoefficientCovarianceWire, CoefficientPriorWire,
    ConstraintWeightingWire, CyclicActionAnswer, CyclicComponentWire, CyclicInitialStateWire,
    CyclicInterventionWire, CyclicInterventionWorld, CyclicWorldAnswer, CyclicWorldOutcomeAnswer,
    CyclicWorldWire, DisturbanceDivisorWire, DisturbancePriorWire, DynamicFitAnswer,
    DynamicInferenceAnswer, DynamicMethodWire, DynamicProblemWire, ExperimentRegimeWire,
    ExperimentalFitOptionsWire, ExperimentalRecoveryReport, LinearCyclicWire, NextAttemptAnswer,
    RecoveredDrawOriginWire, RecoveredDrawStatusWire, RecoveredKeyedWorld, RecoveredLawCoefficient,
    RecoveredLawDraw, RecoveredLawProduct, RecoveryInferenceWire, RecoveryPriorWire,
    RegimeClampWire, RegimeSampleWire, ResamplingRuleWire, SamplingDesignWire, UnitColumnWire,
    WorldCertificateAnswer, WorldSolvedByAnswer,
};

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
    /// Experimentally recovered linear equilibrium, executed from the retained law.
    ///
    /// Preparation does not fit, sample, or treat the recovered draws as a new
    /// estimate. Specified models remain model-conditional; compilation grants
    /// no scientific license. Failed and inadmissible draws stay in the law.
    RecoveredLinear {
        /// Node ids, strictly increasing. Draw coordinates are positions in this list.
        nodes: Vec<u32>,
        /// Joint coefficient and disturbance draws, including failed and inadmissible draws.
        draws: Vec<RecoveredDrawWire>,
        /// Hard-clamp worlds. Indices are positions in `nodes`.
        worlds: Vec<RecoveredInterventionWire>,
        /// Outcome coordinate, a position in `nodes`.
        outcome: usize,
        /// Source of the recovered model. It does not certify the component.
        source_id: String,
        /// Evidence that may certify the component. Never a substitute for a draw.
        certifying_evidence: Vec<String>,
        /// How the draws were produced. Supplied draws declare no meaning and are
        /// omitted from the encoding, so existing requests keep their identity.
        #[serde(default, skip_serializing_if = "RecoveredDrawOriginWire::is_supplied")]
        uncertainty: RecoveredDrawOriginWire,
    },
    /// Linear equilibrium recovered from randomized clamp experiments.
    ///
    /// Preparation admits the worlds and does not fit. Execution fits the
    /// declared regimes, draws the declared uncertainty, and evaluates each
    /// hard-clamp world from the portable law. Unresolved directions stay
    /// unresolved in every draw; the law stays conditional on the recipe.
    ExperimentalRecovery {
        /// Declared regimes, in sample order.
        regimes: Vec<ExperimentRegimeWire>,
        /// Units the mask removes before recovery.
        #[serde(default)]
        dropped: Vec<u32>,
        /// One sample per regime.
        samples: Vec<RegimeSampleWire>,
        /// Fit options.
        #[serde(default)]
        options: ExperimentalFitOptionsWire,
        /// How the uncertainty of the law is produced.
        inference: RecoveryInferenceWire,
        /// Hard-clamp worlds named by variable id.
        worlds: Vec<CyclicWorldWire>,
        /// Outcome variable id; a retained unit.
        outcome: u32,
        /// Source of the recovered model. It does not certify the component.
        source_id: String,
        /// Evidence that may certify the component.
        certifying_evidence: Vec<String>,
    },
    /// Specified linear Gaussian equilibrium under typed interventions.
    LinearWorlds {
        /// Variable ids in model order.
        variables: Vec<u32>,
        /// Structural map and disturbance law.
        model: LinearCyclicWire,
        /// Component identity and certifying evidence.
        component: CyclicComponentWire,
        /// Worlds named by variable id.
        worlds: Vec<CyclicWorldWire>,
        /// Whether an uncertified world is answered.
        certificate: CertificateDemandWire,
    },
    /// Specified nonlinear equilibrium under typed interventions, over shared draws.
    EquilibriumWorlds {
        /// Variable ids; equation `k` is the mechanism of variable `k`.
        variables: Vec<u32>,
        /// Specified model.
        model: antecedent_model::nonlinear_equilibrium::EquilibriumModel,
        /// Worlds named by variable id.
        worlds: Vec<CyclicWorldWire>,
        /// Disturbance draws shared by every world, one row per draw.
        noise: Vec<Vec<f64>>,
        /// Numerical initial guess; not a causal initial state.
        initial: Vec<f64>,
        /// Numerical and certificate policy.
        policy: antecedent_model::nonlinear_equilibrium::SolvePolicy,
    },
    /// Specified continuous-time system under typed interventions and initial-state changes.
    OdeWorlds {
        /// Variable ids in state order.
        variables: Vec<u32>,
        /// Drift mechanisms.
        model: OdeSystem,
        /// Initial state at the first observation time.
        initial: Vec<f64>,
        /// Declared initial-state law name of each state.
        initial_laws: Vec<String>,
        /// Strictly increasing observation times.
        times: Vec<f64>,
        /// Worlds named by variable id.
        worlds: Vec<CyclicWorldWire>,
        /// Adaptive integration policy.
        policy: OdePolicy,
    },
    /// Sampling-law, observed-information, or posterior uncertainty of a fitted
    /// continuous-time family. The fit is shared across methods and quantities.
    DynamicInference {
        /// Declared fitting procedure.
        problem: DynamicProblemWire,
        /// Uncertainty method.
        method: DynamicMethodWire,
        /// Interventional or stationary quantities evaluated at every replicate or draw.
        quantities: Vec<antecedent_estimate::cyclic_dynamic_inference::DynamicQuantity>,
    },
    /// The single-system equilibrium request, carried as a family so it shares
    /// rebinding, evidence, and replay with every other family.
    System {
        /// Complete single-system request.
        request: crate::cyclic_system_artifact::CyclicExecutionRequest,
    },
}

/// One direct coefficient of a recovered linear law, `K[into, from]`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RecoveredCoefficientWire {
    /// Target equation, a raw node id.
    pub into: u32,
    /// Source variable, a raw node id.
    pub from: u32,
    /// Direct mechanism value. A structural zero is exact.
    pub value: f64,
}

/// Support coordinate of an unresolved recovered direction. It is not a filled entry of `K`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RecoveredSupportWire {
    /// Target equation, a raw node id.
    pub into: u32,
    /// Source variable, a raw node id.
    pub from: u32,
}

/// Why a recovered direction was left unresolved.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "reason", rename_all = "snake_case", deny_unknown_fields)]
pub enum RecoveredUnresolvedWire {
    /// At least one ordered pair into `target` fails the pair condition.
    IncompletePairCoverage {
        /// Equation coverage does not pin down.
        target: u32,
        /// Sources that are never clamped while `target` is passive.
        missing_clamps: Vec<u32>,
    },
    /// The pair condition holds and the declared constraint rank does not.
    DeficientConstraintRank {
        /// Equation whose constraint block is rank deficient.
        target: u32,
        /// Declared column rank.
        column_rank: usize,
        /// Number of incoming coefficients.
        width: usize,
    },
}

/// One unresolved direction retained with a draw. It does not complete `K`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RecoveredDirectionWire {
    /// Coordinates the direction moves.
    pub support: Vec<RecoveredSupportWire>,
    /// Why the direction is unresolved.
    pub reason: RecoveredUnresolvedWire,
}

/// One supplied draw of a recovered linear law.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RecoveredDrawWire {
    /// Direct effects named by this draw. Missing coordinates stay missing.
    pub coefficients: Vec<RecoveredCoefficientWire>,
    /// Intercept in node order.
    pub intercept: Vec<f64>,
    /// Disturbance mean of this draw, in node order.
    pub noise_mean: Vec<f64>,
    /// Row-major disturbance covariance of this draw.
    pub noise_covariance: Vec<f64>,
    /// Directions rank-aware recovery left free.
    #[serde(default)]
    pub unresolved: Vec<RecoveredDirectionWire>,
    /// Weight of this draw. The weights of the law sum to one.
    pub weight: f64,
    /// When set, the draw stays inadmissible even if the numbers would compile.
    #[serde(default)]
    pub marked_inadmissible: bool,
}

/// One hard-clamp intervention of a recovered linear law.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RecoveredInterventionWire {
    /// Clamped positions in the node list.
    pub clamp_nodes: Vec<usize>,
    /// Clamp values, aligned with `clamp_nodes`.
    pub clamp_values: Vec<f64>,
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
    /// Interventions evaluated from a retained recovered linear law.
    ///
    /// Preparation did not fit this law. Failed and inadmissible draws remain.
    /// Specified models stay model-conditional; compilation grants no scientific license.
    RecoveredLinear(RecoveredLinearReport),
    /// Law recovered from clamp experiments, its draw origin, and its keyed worlds.
    ExperimentalRecovery(ExperimentalRecoveryReport),
    /// Typed worlds of a specified model, keyed by scientific world id.
    Worlds(Vec<CyclicWorldAnswer>),
    /// Point fit of a continuous-time family and its uncertainty.
    DynamicInference {
        /// Point fit.
        fit: DynamicFitAnswer,
        /// Sampling law, observed information, or posterior.
        inference: Box<DynamicInferenceAnswer>,
    },
    /// Single-system equilibrium answer.
    System(crate::cyclic_system_artifact::CyclicExecutionReport),
}

/// Which algorithm certified one recovered draw.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "algorithm", rename_all = "snake_case", deny_unknown_fields)]
pub enum RecoveredSolvedBy {
    /// Topological DAG evaluation. The draw was not solved as a cyclic block.
    Acyclic,
    /// The DAG algorithm refused a directed cycle; the cyclic license compiled the draw.
    Cyclic {
        /// Existing next attempt named by the DAG refusal.
        next_attempt: String,
    },
}

/// One retained draw after intervention, including a failure slot.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case", deny_unknown_fields)]
pub enum RecoveredDrawAnswer {
    /// Certified equilibrium under this draw's own disturbance.
    Equilibrium {
        /// Equilibrium mean, in node order.
        mean: Vec<f64>,
        /// Row-major equilibrium covariance.
        covariance: Vec<f64>,
        /// Disturbance mean of this draw.
        disturbance_mean: Vec<f64>,
        /// Row-major disturbance covariance of this draw.
        disturbance_covariance: Vec<f64>,
        /// Algorithm that produced the equilibrium.
        solved_by: RecoveredSolvedBy,
    },
    /// The coefficient law kept the draw and did not compile it.
    Inadmissible {
        /// Stable reason name.
        reason: String,
    },
    /// An admissible draw was not certified. The weight remains.
    NotCertified {
        /// Display of the certification fault.
        fault: String,
    },
}

/// Draws of one intervention world, in law order.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RecoveredWorldAnswer {
    /// Law order, including failed and inadmissible draws.
    pub draws: Vec<RecoveredEvaluatedDraw>,
    /// Outcome coordinate.
    pub outcome: usize,
    /// Mass of every draw.
    pub total_mass: f64,
    /// Mass that did not enter the outcome mixture.
    pub withheld_mass: f64,
    /// Weighted mean of certified outcome means, when any exist.
    pub mixture_mean: Option<f64>,
    /// Total outcome variance across and within certified draws.
    pub mixture_variance: Option<f64>,
    /// Weighted variance of the certified draw means.
    pub between_draw_variance: Option<f64>,
}

/// Weight and outcome of one recovered draw.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RecoveredEvaluatedDraw {
    /// Weight copied from the recovered law.
    pub weight: f64,
    /// Equilibrium, inadmissible status, or certification failure.
    pub outcome: RecoveredDrawAnswer,
}

/// Intervention results for every retained world.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RecoveredLinearReport {
    /// Worlds in request order.
    pub worlds: Vec<RecoveredWorldAnswer>,
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
    #[serde(skip)]
    pub(crate) retention: crate::retained_allocation::RetainedAllocation,
    #[serde(skip)]
    lowered: LoweredStamp,
    /// Independently retained request digest.
    pub identity: String,
    /// Typed family answer.
    pub answer: CyclicFamilyAnswer,
}
impl CyclicFamilyReport {
    /// Admitted bytes held by this report; reservations are not process resident memory.
    #[must_use]
    pub fn retained_admitted_bytes(&self) -> u64 {
        self.retention.bytes()
    }

    /// Identity of the lowered program whose execution produced this report.
    ///
    /// `None` for a report no prepared handle stamped, such as an independently
    /// replayed artifact.
    #[must_use]
    pub const fn lowered_program_identity(&self) -> Option<antecedent_core::SemanticDigest> {
        self.lowered.identity()
    }

    /// Stamp the identity of the lowered program that produced this report.
    #[must_use]
    pub fn with_lowered_program_identity(
        mut self,
        identity: antecedent_core::SemanticDigest,
    ) -> Self {
        self.lowered = LoweredStamp::of(identity);
        self
    }
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
    let retention = ctx
        .acquire_admission(
            0,
            crate::retained_allocation::allocation_bytes(value)
                .map_err(invalid)?
                .checked_add(size.bytes as u64)
                .ok_or_else(|| invalid("retained request size overflow"))?,
        )
        .map_err(invalid)?;
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
            Self::RecoveredLinear { .. } => "recovered_linear",
            Self::ExperimentalRecovery { .. } => "experimental_recovery",
            Self::LinearWorlds { .. } => "linear_worlds",
            Self::EquilibriumWorlds { .. } => "equilibrium_worlds",
            Self::OdeWorlds { .. } => "ode_worlds",
            Self::DynamicInference { .. } => "dynamic_inference",
            Self::System { .. } => "system",
        }
    }

    /// Intervention worlds of `report`, keyed by scientific world id, or `None`
    /// for a family whose answer has no intervention worlds.
    #[must_use]
    pub fn intervention_worlds(
        &self,
        report: &CyclicFamilyReport,
    ) -> Option<Vec<CyclicInterventionWorld>> {
        worlds::intervention_worlds(self, &report.answer)
    }
    /// Execute on the common resource/cancellation context, preserving family semantics.
    pub fn execute(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<CyclicFamilyReport, CyclicArtifactError> {
        CompiledCyclicFamilyProgram::compile(self, ctx)?.execute(ctx)
    }

    #[allow(
        clippy::too_many_lines,
        reason = "typed scientific family dispatch retains distinct operator inputs and answer representations"
    )]
    fn execute_bound(
        &self,
        identity: &str,
        frozen: Option<&FrozenFamilyFitValue>,
        source_proof: Option<&crate::cyclic_functional::CheckedCyclicFunctionalSource>,
        ctx: &ExecutionContext,
    ) -> Result<CyclicFamilyReport, CyclicArtifactError> {
        if ctx.cancellation.is_cancelled() {
            return Err(invalid("cyclic execution cancelled"));
        }
        let mut output_leases = Vec::new();
        let base = compiled::output_bytes(self)?;
        output_leases.push(std::sync::Arc::new(ctx.acquire_admission(0, base).map_err(invalid)?));
        let answer = match self {
            Self::GaussianFunctional { law, assignments, .. } => {
                CyclicFamilyAnswer::GaussianFunctional(
                    crate::cyclic_gaussian_functional::evaluate_gaussian_cyclic_functional_checked(
                        source_proof
                            .ok_or_else(|| invalid("compiled functional source missing"))?,
                        law,
                        assignments,
                        match frozen {
                            Some(FrozenFamilyFitValue::Gaussian(density)) => Some(density),
                            _ => None,
                        },
                        ctx,
                    )?,
                )
            }
            Self::AdamFit { model, parameters, training, validation, policy } => {
                CyclicFamilyAnswer::AdamFit(
                    if let Some(FrozenFamilyFitValue::Adam(retained)) = frozen {
                        retained.clone()
                    } else {
                        antecedent_estimate::cyclic_adam::fit_cyclic_adam(
                            model, parameters, training, validation, *policy, ctx,
                        )?
                    },
                )
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
                } else if let Some(FrozenFamilyFitValue::Ou(retained)) = frozen {
                    retained.clone()
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

            Self::Functional { law, assignments, .. } => CyclicFamilyAnswer::Functional(
                crate::cyclic_functional::evaluate_cyclic_functional_checked(
                    source_proof.ok_or_else(|| invalid("compiled functional source missing"))?,
                    law,
                    assignments,
                    ctx,
                )?,
            ),
            Self::GradientFit { model, parameters, rows, policy } => {
                CyclicFamilyAnswer::GradientFit(
                    if let Some(FrozenFamilyFitValue::Gradient(retained)) = frozen {
                        retained.clone()
                    } else {
                        antecedent_estimate::cyclic_gradient::fit_cyclic_gradient(
                            model, parameters, rows, *policy, ctx,
                        )?
                    },
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
                let mut context = ctx.clone();
                context.rng = antecedent_core::RngFactory::from_seed(*seed);
                let draws = if let Some(FrozenFamilyFitValue::Bootstrap(retained)) = frozen {
                    retained.clone()
                } else {
                    antecedent_estimate::cyclic_likelihood::bootstrap_cyclic_fit_with_design(
                        model,
                        parameters,
                        rows,
                        *policy,
                        *replicates,
                        *design,
                        &context,
                    )?
                };
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
                    if let Some(FrozenFamilyFitValue::Polynomial(retained)) = frozen {
                        retained.clone()
                    } else {
                        antecedent_estimate::constraint_likelihood::fit_polynomial_candidates(
                            candidates,
                            measurements,
                            *policy,
                            ctx,
                        )?
                    },
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
                let context = sampling.context(ctx);
                let posterior = if let Some(FrozenFamilyFitValue::Parameter(retained)) = frozen {
                    retained.clone()
                } else {
                    antecedent_estimate::cyclic_likelihood::sample_cyclic_posterior(
                        model,
                        parameters,
                        rows,
                        *prior_sd,
                        sampling.options(),
                        &context,
                    )?
                };
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
                let posterior = if let Some(FrozenFamilyFitValue::Joint(retained)) = frozen {
                    retained.clone()
                } else {
                    antecedent_estimate::cyclic_likelihood::abduce_cyclic_joint(
                        model,
                        parameters,
                        rows,
                        *prior_sd,
                        observations,
                        measurement_variances,
                        initial,
                        sampling.options(),
                        &sampling.context(ctx),
                    )?
                };
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
                let context = sampling.context(ctx);
                let abduction = if let Some(FrozenFamilyFitValue::Partial(retained)) = frozen {
                    retained.clone()
                } else {
                    antecedent_estimate::cyclic_likelihood::abduce_cyclic_partial(
                        model,
                        observations,
                        measurement_variances,
                        initial,
                        sampling.options(),
                        &context,
                    )?
                };
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
                if let Some(FrozenFamilyFitValue::Structure { fit, bootstrap }) = frozen {
                    CyclicFamilyAnswer::StructureFit {
                        fit: fit.clone(),
                        bootstrap: bootstrap.clone(),
                    }
                } else {
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
            }
            Self::OdeFit { model, parameters, observations, integration, policy } => {
                CyclicFamilyAnswer::OdeFit(
                    if let Some(FrozenFamilyFitValue::Ode(retained)) = frozen {
                        retained.clone()
                    } else {
                        antecedent_estimate::cyclic_dynamics::fit_ode_likelihood(
                            model,
                            parameters,
                            observations,
                            *integration,
                            *policy,
                            ctx,
                        )?
                    },
                )
            }
            Self::OuFit { model, parameters, rows, policy } => CyclicFamilyAnswer::OuFit(
                if let Some(FrozenFamilyFitValue::OuFit(retained)) = frozen {
                    retained.clone()
                } else {
                    antecedent_estimate::cyclic_dynamics::fit_ou_likelihood(
                        model, parameters, rows, *policy, ctx,
                    )?
                },
            ),
            Self::Polynomial { model, policy } => {
                CyclicFamilyAnswer::Polynomial(model.isolate(*policy, ctx)?)
            }
            Self::RecoveredLinear {
                nodes,
                draws,
                worlds,
                outcome,
                source_id,
                certifying_evidence,
                uncertainty: _,
            } => compiled::execute_recovered_linear(
                nodes,
                draws,
                worlds,
                *outcome,
                source_id,
                certifying_evidence,
            )?,
            Self::ExperimentalRecovery {
                regimes,
                dropped,
                samples,
                options,
                inference,
                worlds,
                outcome,
                source_id,
                certifying_evidence,
            } => worlds::execute_experimental_recovery(
                regimes,
                dropped,
                samples,
                *options,
                inference,
                worlds,
                *outcome,
                source_id,
                certifying_evidence,
                match frozen {
                    Some(FrozenFamilyFitValue::Recovered(law)) => Some(law),
                    _ => None,
                },
                ctx,
            )?,
            Self::LinearWorlds { variables, model, component, worlds, certificate } => {
                worlds::execute_linear_worlds(
                    variables,
                    model,
                    component,
                    worlds,
                    *certificate,
                    ctx,
                )?
            }
            Self::EquilibriumWorlds { variables, model, worlds, noise, initial, policy } => {
                worlds::execute_equilibrium_worlds(
                    variables, model, worlds, noise, initial, *policy, ctx,
                )?
            }
            Self::OdeWorlds { variables, model, initial, initial_laws, times, worlds, policy } => {
                worlds::execute_ode_worlds(
                    variables,
                    model,
                    initial,
                    initial_laws,
                    times,
                    worlds,
                    *policy,
                    ctx,
                )?
            }
            Self::DynamicInference { problem, method, quantities } => {
                worlds::execute_dynamic_inference(
                    problem,
                    method,
                    quantities,
                    match frozen {
                        Some(FrozenFamilyFitValue::Dynamic(fit)) => Some(fit),
                        _ => None,
                    },
                    ctx,
                )?
            }
            Self::System { request } => {
                let fit = match (&request.fit, frozen) {
                    (None, _) => None,
                    (Some(_), Some(FrozenFamilyFitValue::System(retained))) => {
                        Some(retained.clone())
                    }
                    (Some(recipe), _) => {
                        Some(antecedent_estimate::cyclic_likelihood::fit_cyclic_likelihood(
                            &request.model,
                            &recipe.parameters,
                            &recipe.rows,
                            recipe.options,
                            ctx,
                        )?)
                    }
                };
                CyclicFamilyAnswer::System(request.execute_bound(fit.as_ref(), ctx)?)
            }
        };
        let required = crate::retained_allocation::allocation_bytes(&answer)
            .map_err(invalid)?
            .checked_add(identity.len() as u64 + std::mem::size_of::<CyclicFamilyReport>() as u64)
            .ok_or_else(|| invalid("report retention size overflow"))?;
        let held = output_leases
            .iter()
            .try_fold(0u64, |sum, lease| sum.checked_add(lease.remainder().bytes()))
            .ok_or_else(|| invalid("report retention size overflow"))?;
        if required > held {
            output_leases.push(std::sync::Arc::new(
                ctx.acquire_admission(0, required - held).map_err(invalid)?,
            ));
        }
        let retention = crate::retained_allocation::RetainedAllocation(output_leases);
        antecedent_core::execution_attempt::record_physical(
            antecedent_core::execution_attempt::PhysicalEvent::RetainedBytes,
            retention.bytes(),
        );
        antecedent_core::execution_attempt::record_physical(
            antecedent_core::execution_attempt::PhysicalEvent::RetainedFull,
            1,
        );
        Ok(CyclicFamilyReport {
            identity: identity.to_owned(),
            answer,
            retention,
            lowered: LoweredStamp::default(),
        })
    }
    /// Execute and encode with the existing family-neutral artifact container.
    pub fn export(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<(CyclicFamilyReport, Vec<u8>), CyclicArtifactError> {
        CompiledCyclicFamilyProgram::compile(self, ctx)?.export(ctx)
    }

    /// Bounded JSON record of `version` holding this request and `report`.
    fn record_bytes(
        &self,
        version: u16,
        report: &CyclicFamilyReport,
        ctx: &ExecutionContext,
    ) -> Result<Vec<u8>, CyclicArtifactError> {
        #[derive(Serialize)]
        struct View<'a> {
            version: u16,
            request: &'a CyclicFamilyRequest,
            report: &'a CyclicFamilyReport,
        }
        let (bytes, _record_retention) =
            bounded_json(&View { version, request: self, report }, ctx)?;
        Ok(bytes)
    }

    /// Encode `sections` in the family-neutral container, admitting encoding scratch first.
    fn encode_sections(
        sections: &[crate::OutSection<'_>],
        ctx: &ExecutionContext,
    ) -> Result<Vec<u8>, CyclicArtifactError> {
        let payload = sections
            .iter()
            .try_fold(0_u64, |total, section| total.checked_add(section.bytes.len() as u64));
        let scratch = payload
            .and_then(|n| n.checked_mul(3))
            .and_then(|n| n.checked_add(65536))
            .ok_or_else(|| invalid("artifact encoding size overflow"))?;
        let _encoding = ctx.acquire_admission(0, scratch).map_err(invalid)?;
        let artifact = crate::encode_container(
            &crate::ArtifactContainer {
                kind: "cyclic_family",
                note: "specified_dynamic_or_constraint_model",
                max_bytes: MAX_CYCLIC_ARTIFACT_BYTES,
                artifact_id: "cyclic_family",
                library_version: env!("CARGO_PKG_VERSION"),
            },
            sections,
        )
        .map_err(|e| invalid(e.into_io("cyclic family layout")))?;
        if ctx.cancellation.is_cancelled() {
            return Err(invalid("cyclic family artifact export cancelled"));
        }
        Ok(artifact)
    }

    /// Encode a version-two record: the request and report, and the contract
    /// section bound to that request.
    fn encode_report(
        &self,
        report: &CyclicFamilyReport,
        contract: &antecedent_io_core::contract_section::CyclicContractWire,
        ctx: &ExecutionContext,
    ) -> Result<Vec<u8>, CyclicArtifactError> {
        let bytes = self.record_bytes(2, report, ctx)?;
        let contract = crate::to_cbor(contract).map_err(invalid)?;
        Self::encode_sections(
            &[
                crate::OutSection {
                    id: FAMILY_V2_SECTION,
                    content_type: crate::CONTENT_BYTES,
                    bytes: &bytes,
                },
                crate::OutSection {
                    id: antecedent_io_core::contract_section::CYCLIC_CONTRACT_SECTION,
                    content_type: crate::CONTENT_CBOR,
                    bytes: &contract,
                },
            ],
            ctx,
        )
    }

    /// Encode the version-one record, which carries no contract section. Kept so
    /// the previous-version golden is written by the format it pins.
    #[cfg(test)]
    fn encode_report_v1(
        &self,
        report: &CyclicFamilyReport,
        ctx: &ExecutionContext,
    ) -> Result<Vec<u8>, CyclicArtifactError> {
        let bytes = self.record_bytes(1, report, ctx)?;
        Self::encode_sections(
            &[crate::OutSection {
                id: FAMILY_V1_SECTION,
                content_type: crate::CONTENT_BYTES,
                bytes: &bytes,
            }],
            ctx,
        )
    }
}

/// Section of a version-one record.
const FAMILY_V1_SECTION: &str = "cyclic_family_v1";
/// Record section of a version-two artifact; the contract section follows it.
const FAMILY_V2_SECTION: &str = "cyclic_family_v2";

/// A decoded record and, for version two, its contract section.
struct Decoded {
    record: Record,
    request_bytes: Vec<u8>,
    contract: Option<antecedent_io_core::contract_section::CyclicContractWire>,
}

/// Decode either record layout. A version-one record has no contract; a
/// version-two record must carry one.
fn decode_record(bytes: &[u8]) -> Result<Decoded, CyclicArtifactError> {
    let layout = |e: crate::CodecError| invalid(e.into_io("cyclic family layout"));
    let (record_bytes, contract) = match crate::decode_container(
        "cyclic_family",
        MAX_CYCLIC_ARTIFACT_BYTES,
        &[FAMILY_V1_SECTION],
        bytes,
    ) {
        Ok(mut parts) => (parts.pop().ok_or_else(|| invalid("missing family section"))?, None),
        Err(crate::CodecError::Layout) => {
            let mut parts = crate::decode_container(
                "cyclic_family",
                MAX_CYCLIC_ARTIFACT_BYTES,
                &[FAMILY_V2_SECTION, antecedent_io_core::contract_section::CYCLIC_CONTRACT_SECTION],
                bytes,
            )
            .map_err(layout)?;
            let contract = parts.pop().ok_or_else(|| invalid("missing contract section"))?;
            let record = parts.pop().ok_or_else(|| invalid("missing family section"))?;
            (record, Some(crate::from_cbor(&contract).map_err(invalid)?))
        }
        Err(other) => return Err(layout(other)),
    };
    let record: Record = serde_json::from_slice(&record_bytes).map_err(invalid)?;
    let expected = if contract.is_some() { 2 } else { 1 };
    if record.version != expected {
        return Err(invalid("unsupported cyclic family version"));
    }
    let request_bytes = serde_json::to_vec(&record.request).map_err(invalid)?;
    Ok(Decoded { record, request_bytes, contract })
}

/// Check a version-two contract against its request, independently of the producer.
///
/// The contract is verified from bytes ([`antecedent_io_core::contract_section::verify_cyclic_contract_against_body`]),
/// then every payload is re-derived from the decoded request and its compiled
/// solve schedules, so a consistently resealed contract over other payloads is
/// refused. Compilation does not fit or solve.
fn verify_contract(
    request: &CyclicFamilyRequest,
    request_bytes: &[u8],
    contract: &antecedent_io_core::contract_section::CyclicContractWire,
    ctx: &ExecutionContext,
) -> Result<(), CyclicArtifactError> {
    let unresolved = antecedent_io_core::contract_section::verify_cyclic_contract_against_body(
        contract,
        request_bytes,
    );
    if !unresolved.is_empty() {
        return Err(invalid(format!(
            "cyclic family contract does not verify: {}",
            unresolved.iter().map(|reason| &**reason).collect::<Vec<_>>().join(",")
        )));
    }
    let program = CompiledCyclicFamilyProgram::compile(request, ctx)?;
    if program.contract()? != *contract {
        return Err(CyclicArtifactError::Mismatch);
    }
    Ok(())
}

/// Check a separately retained request identity and recompute the typed answer.
///
/// Reads version-one records, which carry no contract, and version-two records,
/// whose contract section is verified against the request before the answer is
/// replayed.
pub fn consume_cyclic_family(
    bytes: &[u8],
    expected_identity: &str,
    ctx: &ExecutionContext,
) -> Result<CyclicFamilyReport, CyclicArtifactError> {
    consume_decoded(bytes, expected_identity, ctx).map(|(_, report)| report)
}

fn consume_decoded(
    bytes: &[u8],
    expected_identity: &str,
    ctx: &ExecutionContext,
) -> Result<(Decoded, CyclicFamilyReport), CyclicArtifactError> {
    if bytes.len() > MAX_CYCLIC_ARTIFACT_BYTES || expected_identity.len() != 64 {
        return Err(invalid("family byte limit or expected identity"));
    }
    let _retention = crate::cyclic_system_artifact::cyclic_decode_admission(bytes, ctx)?;
    let decoded = decode_record(bytes)?;
    if CanonicalWriter::digest_bytes_hex(
        "antecedent.cyclic_family.request.v1",
        &decoded.request_bytes,
    ) != expected_identity
        || decoded.record.report.identity != expected_identity
    {
        return Err(CyclicArtifactError::Mismatch);
    }
    if let Some(contract) = &decoded.contract {
        if contract.request_identity != expected_identity {
            return Err(CyclicArtifactError::Mismatch);
        }
        verify_contract(&decoded.record.request, &decoded.request_bytes, contract, ctx)?;
    }
    let report = decoded.record.request.execute(ctx)?;
    if report.identity != expected_identity || report != decoded.record.report {
        return Err(CyclicArtifactError::Mismatch);
    }
    Ok((decoded, report))
}

/// Independently consume a version-two family artifact and return its verified
/// contract with the replayed answer.
///
/// A version-one record has no contract and is refused here; it still replays
/// through [`consume_cyclic_family`].
pub fn consume_cyclic_family_contract(
    bytes: &[u8],
    expected_identity: &str,
    ctx: &ExecutionContext,
) -> Result<
    (antecedent_io_core::contract_section::CyclicContractWire, CyclicFamilyReport),
    CyclicArtifactError,
> {
    let (decoded, report) = consume_decoded(bytes, expected_identity, ctx)?;
    let contract = decoded.contract.ok_or_else(|| {
        invalid(
            "a version-one cyclic family record carries no contract; next attempt: re-export it \
             from a prepared family to write a contract section",
        )
    })?;
    Ok((contract, report))
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
    let (decoded, report) = consume_decoded(bytes, expected_identity, ctx)?;
    Ok((decoded.record.request, report))
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

/// The version-one record keeps replaying; the version-two record carries a
/// contract an independent consumer verifies against the request it travels with.
///
/// The previous-version golden lives at `tests/fixtures/cyclic_family_v1.bin` and
/// is written by the version-one encoder it pins. Generate it with
/// `cargo test -p antecedent-io --lib previous_version -- --ignored write_`.
#[cfg(test)]
mod previous_version {
    use super::*;
    use std::path::PathBuf;

    const GOLDEN: &str = "cyclic_family_v1.bin";
    const GENERATOR: &str =
        "cargo test -p antecedent-io --lib previous_version -- --ignored write_";

    fn fixture_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
    }

    fn ctx() -> ExecutionContext {
        ExecutionContext::for_tests(1)
    }

    /// `A = 0.4 B + U_A`, `B = 0.5 A + U_B` under the model as specified, a clamp and a shock.
    fn request() -> CyclicFamilyRequest {
        CyclicFamilyRequest::LinearWorlds {
            variables: vec![10, 11],
            model: LinearCyclicWire {
                coefficients: vec![0.0, 0.4, 0.5, 0.0],
                intercept: vec![0.0; 2],
                noise_mean: vec![0.0; 2],
                noise_covariance: vec![1.0, 0.0, 0.0, 1.0],
            },
            component: CyclicComponentWire {
                component_id: "feedback".into(),
                source_id: "specification".into(),
                certifying_evidence: vec!["independent-bench-test".into()],
            },
            worlds: vec![
                CyclicWorldWire { world: 7, interventions: vec![], initial_states: vec![] },
                CyclicWorldWire {
                    world: 8,
                    interventions: vec![CyclicInterventionWire::Set { variable: 10, value: 2.0 }],
                    initial_states: vec![],
                },
                CyclicWorldWire {
                    world: 9,
                    interventions: vec![CyclicInterventionWire::Shift { variable: 10, delta: 1.0 }],
                    initial_states: vec![],
                },
            ],
            certificate: CertificateDemandWire::Required,
        }
    }

    #[test]
    #[ignore = "writes tests/fixtures; run with `-- --ignored write_`"]
    fn write_cyclic_family_v1_golden() {
        let program = CompiledCyclicFamilyProgram::compile(&request(), &ctx()).unwrap();
        let report = program.execute(&ctx()).unwrap();
        let bytes = request().encode_report_v1(&report, &ctx()).unwrap();
        let dir = fixture_dir();
        std::fs::create_dir_all(&dir).expect("create the fixture directory");
        std::fs::write(dir.join(GOLDEN), bytes).expect("write the golden");
    }

    #[test]
    fn the_previous_version_golden_still_replays() {
        let bytes = antecedent_test_support::read_fixture_bytes(
            &fixture_dir(),
            GOLDEN,
            &format!("generate it with `{GENERATOR}`"),
        );
        let identity =
            CompiledCyclicFamilyProgram::compile(&request(), &ctx()).unwrap().identity().to_owned();
        let replayed = consume_cyclic_family(&bytes, &identity, &ctx()).unwrap();
        assert_eq!(replayed, request().execute(&ctx()).unwrap());
        let (decoded, _) = consume_cyclic_family_with_request(&bytes, &identity, &ctx()).unwrap();
        assert_eq!(decoded, request());
        assert!(
            consume_cyclic_family_contract(&bytes, &identity, &ctx()).is_err(),
            "a version-one record carries no contract to verify"
        );
        assert!(consume_cyclic_family(&bytes, &"0".repeat(64), &ctx()).is_err());
    }

    #[test]
    fn a_freshly_encoded_version_one_record_replays_beside_version_two() {
        let program = CompiledCyclicFamilyProgram::compile(&request(), &ctx()).unwrap();
        let report = program.execute(&ctx()).unwrap();
        let v1 = request().encode_report_v1(&report, &ctx()).unwrap();
        assert_eq!(consume_cyclic_family(&v1, program.identity(), &ctx()).unwrap(), report);
        let (exported, v2) = program.export(&ctx()).unwrap();
        assert_ne!(v1, v2);
        let (contract, replayed) =
            consume_cyclic_family_contract(&v2, program.identity(), &ctx()).unwrap();
        assert_eq!(replayed, exported);
        assert_eq!(contract, program.contract().unwrap());
        assert!(
            antecedent_io_core::contract_section::verify_cyclic_contract_against_body(
                &contract,
                &serde_json::to_vec(&request()).unwrap(),
            )
            .is_empty()
        );
    }

    #[test]
    fn a_tampered_or_resealed_contract_is_refused() {
        let program = CompiledCyclicFamilyProgram::compile(&request(), &ctx()).unwrap();
        let report = program.execute(&ctx()).unwrap();
        let contract = program.contract().unwrap();

        let mut tampered = contract.clone();
        tampered.observation.variables.push(12);
        let bytes = request().encode_report(&report, &tampered, &ctx()).unwrap();
        assert!(consume_cyclic_family(&bytes, program.identity(), &ctx()).is_err());

        // A self-consistent contract over another request's payloads, bound to this
        // request's identity, passes the byte-level check and fails re-derivation.
        let mut other = request();
        if let CyclicFamilyRequest::LinearWorlds { certificate, .. } = &mut other {
            *certificate = CertificateDemandWire::ReportUncertified;
        }
        let other_program = CompiledCyclicFamilyProgram::compile(&other, &ctx()).unwrap();
        let payloads = cyclic_contract_payloads(
            &other,
            other_program.execution_plans(),
            other_program.fit_dependency_identity(),
        )
        .unwrap();
        let resealed = antecedent_io_core::contract_section::seal_cyclic_contract(
            payloads,
            program.identity(),
        )
        .unwrap();
        assert!(
            antecedent_io_core::contract_section::verify_cyclic_contract_against_body(
                &resealed,
                &serde_json::to_vec(&request()).unwrap(),
            )
            .is_empty()
        );
        assert_ne!(resealed, contract);
        let bytes = request().encode_report(&report, &resealed, &ctx()).unwrap();
        assert!(matches!(
            consume_cyclic_family(&bytes, program.identity(), &ctx()),
            Err(CyclicArtifactError::Mismatch)
        ));
    }
}
