//! Typed requests for retained cyclic family execution.

use antecedent_model::continuous_cyclic::{OdePolicy, OdeSystem, OrnsteinUhlenbeck, TimedClamp};
use antecedent_model::equation_system::{IsolationPolicy, PolynomialSystem};
use serde::{Deserialize, Serialize};

use super::{
    CertificateDemandWire, CyclicComponentWire, CyclicSamplingPolicy, CyclicWorldWire,
    DynamicMethodWire, DynamicProblemWire, ExperimentRegimeWire, ExperimentalFitOptionsWire,
    LinearCyclicWire, RecoveredDrawOriginWire, RecoveredDrawWire, RecoveredInterventionWire,
    RecoveryInferenceWire, RegimeSampleWire, RetainedRecoveryInputs, RetainedRecoveryProvenance,
};

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
    /// Fit a continuous-time OU model from continuous trajectories by the exact Gaussian
    /// transition likelihood (Kalman filter under a declared linear Gaussian observation
    /// model), then predict held-out intervention trajectories and sustained worlds.
    OuTrajectoryFit {
        /// Baseline drift, forcing and diffusion factor; fitted coordinates start here.
        model: antecedent_estimate::ou_mechanisms::OuMechanisms,
        /// Explicit linear Gaussian observation model, so partial observation is declared.
        observation: antecedent_estimate::ou_trajectory::OuObservationModel,
        /// Shared baseline coordinates to estimate.
        parameters: Vec<antecedent_estimate::ou_mechanisms::OuMechanismParameter>,
        /// Independent trajectories with known piecewise-constant intervention schedules.
        trajectories: Vec<antecedent_estimate::ou_trajectory::OuTrajectory>,
        /// Bounded optimizer policy.
        policy: antecedent_estimate::cyclic_gradient::CyclicGradientPolicy,
        /// Held-out intervention trajectories to predict.
        #[serde(default)]
        predictions: Vec<antecedent_estimate::ou_trajectory::OuPredictionDesign>,
        /// Sustained-intervention worlds whose stationary law is predicted.
        #[serde(default)]
        stationary_worlds: Vec<antecedent_estimate::ou_trajectory::OuStationaryDesign>,
    },
    /// Simulate a bounded nonlinear stochastic differential equation with diagonal noise
    /// under named interventions and paired contrasts, and, when trajectories are supplied,
    /// fit declared parameters by the Euler quasi likelihood (or the extended-Kalman
    /// likelihood under a declared observation model) before simulating at the fit.
    SdeAnalysis {
        /// Drift and diagonal diffusion expressions, schedules, bounds and parameters.
        model: antecedent_model::nonlinear_sde::SdeSystem,
        /// Initial state of every simulated path.
        initial: Vec<f64>,
        /// Declared step, horizon, scheme, ensemble size, seed and numerical-error check.
        simulation: antecedent_model::nonlinear_sde::SdeSimulationPolicy,
        /// Named intervention worlds besides the implicit baseline.
        #[serde(default)]
        interventions: Vec<antecedent_model::nonlinear_sde::SdeWorld>,
        /// Paired contrasts of two worlds under shared random numbers.
        #[serde(default)]
        contrasts: Vec<antecedent_model::nonlinear_sde::SdeContrast>,
        /// Parameter coordinates (indices) to estimate; ignored without trajectories.
        #[serde(default)]
        parameters: Vec<usize>,
        /// Linear Gaussian observation model; absent means every state is observed directly.
        #[serde(default)]
        observation: Option<antecedent_estimate::nonlinear_sde_fit::SdeObservationModel>,
        /// Gaussian prior of the hidden state at each trajectory's first observation.
        #[serde(default)]
        filter_prior: Option<antecedent_estimate::nonlinear_sde_fit::SdeFilterPrior>,
        /// Independent trajectories with known intervention regimes.
        #[serde(default)]
        trajectories: Vec<antecedent_estimate::nonlinear_sde_fit::SdeTrajectory>,
        /// Optimizer and observation-interval admission policy of the fit.
        #[serde(default)]
        fit_policy: Option<antecedent_estimate::nonlinear_sde_fit::SdeFitPolicy>,
        /// Asymptotic-normal parameter draws propagated through the simulation (0 or 2..=512).
        #[serde(default)]
        parameter_draws: usize,
        /// Seed of those draws.
        #[serde(default)]
        parameter_seed: u64,
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
    /// no scientific support. Failed and inadmissible draws stay in the law.
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
    /// Process-local borrowed continuation of an executed or independently verified law.
    /// It encodes as the existing experimental-recovery request. Decoding always
    /// selects that original variant and independently refits/replays its source.
    #[serde(rename = "experimental_recovery", skip_deserializing)]
    RetainedExperimentalRecovery {
        /// Borrowed original regimes, observations and inference declaration.
        #[serde(flatten)]
        inputs: RetainedRecoveryInputs,
        /// New hard-clamp worlds, preserving scientific ids.
        worlds: Vec<CyclicWorldWire>,
        /// Outcome id in the original retained-unit namespace.
        outcome: u32,
        /// Borrowed original source/evidence declarations, serialized last.
        #[serde(flatten)]
        provenance: RetainedRecoveryProvenance,
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
