// SPDX-License-Identifier: MIT OR Apache-2.0
//! Typed cyclic worlds, laws recovered from clamp experiments, and inference on
//! fitted continuous-time families.
//!
//! World requests name variables by id and carry core interventions. They are
//! admitted by [`antecedent_model::cyclic_world`]: a refused operation names its
//! next attempt and is never rewritten into another action. Results are keyed by
//! scientific world id and keep the certificate, the solve route and the failed and
//! withheld mass. A recovered law keeps how its draws were produced, so a sampling
//! law is never read as a posterior. The index-based clamp wires of the older
//! families stay decodable and unchanged.

use std::collections::BTreeSet;
use std::sync::Arc;

use antecedent_core::intervention::CyclicFamily;
use antecedent_core::{
    ExecutionContext, Intervention, MechanismOverride, RngFactory, ScientificWorldId, Value,
    VariableId,
};
use antecedent_estimate::cyclic_dynamic_inference::{
    DynamicBootstrapMethod, DynamicBootstrapRequest, DynamicFitRef, DynamicInferenceError,
    DynamicObservedInformation, DynamicPosterior, DynamicPosteriorOptions, DynamicPrior,
    DynamicQuantity, DynamicSamplingLaw, OdeInferenceProblem, OuInferenceProblem,
    OuMechanismInferenceProblem, bootstrap_dynamic_fit, dynamic_observed_information,
    sample_dynamic_posterior,
};
use antecedent_estimate::cyclic_dynamics::{OdeFit, OuFit};
use antecedent_estimate::cyclic_likelihood::CyclicBootstrapDesign;
use antecedent_estimate::experimental_fit::{
    BootstrapSpec, CoefficientCovarianceEstimator, CoefficientPrior, ConstraintWeighting,
    DEFAULT_FIT_BYTES, DisturbanceDivisor, DisturbancePrior, DrawStatus, ExperimentRows,
    ExperimentalFitError, ExperimentalFitOptions, PosteriorSpec, RecoveredDrawOrigin,
    RecoveredEstimate, RecoveryPrior, SystemStability, asymptotic_normal_recovery,
    bootstrap_experimental_recovery, fit_experimental_recovery, posterior_recovery,
};
use antecedent_estimate::ou_mechanisms::OuMechanismFit;
use antecedent_estimate::recovery::{
    RecoveryPlanParts, RetainedRecoveryPlan, consume_retained_recovery_plan,
};
use antecedent_identify::experiment_moments::{RegimeSample, UnitColumn};
use antecedent_identify::experiment_regime::{
    DependencePremise, ExperimentRegime, PopulationId, RegimeContract, RegimeIntervention,
    SourceIdentity, UnitMask,
};
use antecedent_identify::experimental_recovery::{
    DirectCoefficient, JointRecoveryDraw, RecoveredCoefficientUncertainty, RecoveryDrawInput,
    UnresolvedDirection, UnresolvedReason,
};
use antecedent_model::compile::{CompiledSystemPlan, compile_equilibrium, compile_ode};
use antecedent_model::continuous_cyclic::{OdePolicy, OdeSystem, OdeTrajectory};
use antecedent_model::cyclic_world::{
    CertificateBasis, CertificateDemand, CertificateGap, ContinuousTimeSpecification, CyclicAction,
    CyclicVariables, CyclicWorldOutcome, CyclicWorldRequest, CyclicWorldResult,
    LinearCyclicSpecification, NonlinearCyclicSpecification, WorldCertificate, WorldSolvedBy,
    evaluate_continuous_time_world, evaluate_linear_worlds, evaluate_nonlinear_world,
    resolve_cyclic_world,
};
use antecedent_model::nonlinear_equilibrium::{
    Clamp, EquilibriumModel, EquilibriumOutcome, SolvePolicy,
};
use antecedent_model::scc_compile::{CompileBudget, GaussianDisturbance};
use antecedent_model::{HardClamp, LinearStructuralMap};
use serde::{Deserialize, Serialize};

use super::{
    CyclicFamilyAnswer as A, RecoveredDirectionWire, RecoveredSupportWire, RecoveredUnresolvedWire,
    RecoveredWorldAnswer, invalid,
};
use crate::cyclic_system_artifact::CyclicArtifactError;

/// Largest number of draws, replicates or disturbance rows one request *generates* or
/// retains from a bare count. Counts that index request data (worlds, regimes, quantities,
/// retained draws) are bounded by that data and carry no separate cap.
const MAX_DRAWS: usize = 10_000;
/// Largest number of observed values over every regime column.
const MAX_EXPERIMENT_VALUES: usize = 20_000_000;

fn refused(refusal: antecedent_model::cyclic_world::CyclicInterventionRefusal) -> CyclicArtifactError {
    CyclicArtifactError::WorldRefusal(refusal)
}

// ---------------------------------------------------------------------------
// Typed worlds
// ---------------------------------------------------------------------------

/// One intervention on a named variable.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum CyclicInterventionWire {
    /// Hard clamp `do(variable := value)`; the clamped equation is deleted.
    Set {
        /// Variable id.
        variable: u32,
        /// Assigned level.
        value: f64,
    },
    /// Additive shock on the structural disturbance of `variable`.
    Shift {
        /// Variable id.
        variable: u32,
        /// Deterministic perturbation.
        delta: f64,
    },
    /// Mechanism replacement or change of a named soft family (`constant`,
    /// `linear_mechanism` or `additive_shift`). The disturbance is kept.
    Soft {
        /// Variable id.
        variable: u32,
        /// Soft family.
        family: String,
        /// Packed family parameters.
        parameters: Vec<f64>,
    },
}

impl CyclicInterventionWire {
    fn intervention(&self) -> Intervention {
        match self {
            Self::Set { variable, value } => Intervention::Set {
                variable: VariableId::from_raw(*variable),
                value: Value::f64(*value),
            },
            Self::Shift { variable, delta } => Intervention::Shift {
                variable: VariableId::from_raw(*variable),
                delta: Value::f64(*delta),
            },
            Self::Soft { variable, family, parameters } => Intervention::Soft {
                variable: VariableId::from_raw(*variable),
                mechanism: MechanismOverride::named(family.as_str(), parameters.clone()),
            },
        }
    }
}

/// Replacement of one declared initial-state law.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CyclicInitialStateWire {
    /// Declared initial-state law name.
    pub law: String,
    /// Replacement initial level.
    pub level: f64,
}

/// The simultaneous interventions that define one scientific world.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CyclicWorldWire {
    /// Scientific world id; unique within a request.
    pub world: u32,
    /// Simultaneous untimed interventions; empty is the model as specified.
    #[serde(default)]
    pub interventions: Vec<CyclicInterventionWire>,
    /// Initial-state law replacements (continuous-time families only).
    #[serde(default)]
    pub initial_states: Vec<CyclicInitialStateWire>,
}

impl CyclicWorldWire {
    fn request(&self) -> Result<CyclicWorldRequest, CyclicArtifactError> {
        let mut request = CyclicWorldRequest::new(
            ScientificWorldId::from_raw(self.world),
            self.interventions.iter().map(CyclicInterventionWire::intervention),
        );
        for state in &self.initial_states {
            request =
                request.with_initial_state(state.law.as_str(), state.level).map_err(refused)?;
        }
        Ok(request)
    }
}

fn world_requests(
    worlds: &[CyclicWorldWire],
) -> Result<Vec<CyclicWorldRequest>, CyclicArtifactError> {
    if worlds.is_empty() {
        return Err(invalid("typed cyclic worlds: at least one world is required"));
    }
    let mut seen = BTreeSet::new();
    worlds
        .iter()
        .map(|world| {
            if !seen.insert(world.world) {
                return Err(invalid(
                    "typed cyclic worlds: a world id is requested twice; next attempt: give \
                     every world its own id",
                ));
            }
            world.request()
        })
        .collect()
}

fn variables_of(ids: &[u32]) -> Result<CyclicVariables, CyclicArtifactError> {
    CyclicVariables::try_new(ids.iter().copied().map(VariableId::from_raw).collect())
        .map_err(refused)
}

fn budget(ctx: &ExecutionContext) -> CompileBudget {
    CompileBudget { max_work: u64::MAX, max_bytes: ctx.memory_limit_bytes().unwrap_or(u64::MAX) }
}

/// Whether a linear world without a certificate is answered.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CertificateDemandWire {
    /// Withhold a world whose `ρ(|K|) < 1` certificate was not shown.
    Required,
    /// Answer a uniquely solvable world and mark it uncertified.
    ReportUncertified,
}

impl CertificateDemandWire {
    const fn engine(self) -> CertificateDemand {
        match self {
            Self::Required => CertificateDemand::Required,
            Self::ReportUncertified => CertificateDemand::ReportUncertified,
        }
    }
}

/// Specified linear Gaussian equilibrium `x = K x + c + ε`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LinearCyclicWire {
    /// Row-major `K[into * n + from]`, zero diagonal.
    pub coefficients: Vec<f64>,
    /// Intercept `c`, in variable order.
    pub intercept: Vec<f64>,
    /// Disturbance mean, in variable order.
    pub noise_mean: Vec<f64>,
    /// Row-major disturbance covariance.
    pub noise_covariance: Vec<f64>,
}

/// Identity and evidence of the component a specified linear model enters as.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CyclicComponentWire {
    /// Component id.
    pub component_id: String,
    /// Source of the model. It does not certify the component.
    pub source_id: String,
    /// Evidence other than the source that may certify the component.
    pub certifying_evidence: Vec<String>,
}

/// Action of one evaluated world.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CyclicActionAnswer {
    /// Stable action kind (`hard_clamp`, `replace_mechanism`, `change_mechanism`,
    /// `additive_shock`, `initial_state_change`).
    pub kind: String,
    /// Target variable id.
    pub variable: u32,
    /// Model position.
    pub node: usize,
}

/// What the world evaluated to.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case", deny_unknown_fields)]
pub enum CyclicWorldOutcomeAnswer {
    /// Gaussian equilibrium law in variable order.
    GaussianLaw {
        /// Equilibrium mean.
        mean: Vec<f64>,
        /// Row-major equilibrium covariance.
        covariance: Vec<f64>,
        /// Finite structural residual of the compile, when one was reported.
        residual_inf: Option<f64>,
    },
    /// One outcome per disturbance draw; failed draws stay in place.
    EquilibriumDraws {
        /// Draw-aligned outcomes.
        outcomes: Vec<EquilibriumOutcome>,
    },
    /// Trajectory at the declared observation times.
    Trajectory {
        /// Integrated trajectory.
        trajectory: OdeTrajectory,
    },
    /// No solution was returned.
    NoSolution {
        /// Why.
        cause: String,
        /// Next attempt name.
        next_attempt: String,
    },
    /// Not computed because a required certificate was not shown.
    Withheld,
}

/// Next attempt of a refusal or of an uncertified world.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NextAttemptAnswer {
    /// Stable name.
    pub name: String,
    /// What to declare instead.
    pub sentence: String,
}

/// Uniqueness certificate of one world. A certificate is not physical stability.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorldCertificateAnswer {
    /// Whether the world is certified for the declared model.
    pub certified: bool,
    /// What the certificate rests on, when certified.
    pub basis: Option<String>,
    /// Gap, when not certified.
    pub gap: Option<String>,
    /// Upper bound on `ρ(|K|)` of the intervened map, when the linear family was assessed.
    pub rho_abs_upper: Option<f64>,
    /// Draws covered by a nonlinear certificate.
    pub certified_draws: Option<usize>,
    /// Draws solved without a uniqueness certificate.
    pub uncertified_draws: Option<usize>,
    /// Draws that failed.
    pub failed_draws: Option<usize>,
    /// Draws withheld for lack of a certificate.
    pub withheld_draws: Option<usize>,
    /// What would earn the certificate, or what to try instead.
    pub next_attempt: Option<NextAttemptAnswer>,
}

/// How the world was solved.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorldSolvedByAnswer {
    /// Stable route name.
    pub method: String,
    /// Topological order or free positions, when the route has one.
    pub order: Vec<usize>,
    /// Solve blocks, sources first, when the route has them.
    pub blocks: Vec<Vec<usize>>,
    /// The DAG algorithm's refusal that sent the world to block factoring.
    pub dag_refusal: Option<String>,
    /// Accepted integration steps.
    pub accepted_steps: Option<usize>,
    /// Rejected integration steps.
    pub rejected_steps: Option<usize>,
}

/// One evaluated typed world, keyed by its scientific world id.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CyclicWorldAnswer {
    /// Scientific world id.
    pub world: u32,
    /// Family the world was admitted on.
    pub family: String,
    /// Variable ids in outcome order.
    pub variables: Vec<u32>,
    /// Resolved actions, sorted by model position.
    pub actions: Vec<CyclicActionAnswer>,
    /// Outcome.
    pub outcome: CyclicWorldOutcomeAnswer,
    /// Uniqueness certificate.
    pub certificate: WorldCertificateAnswer,
    /// Solve route.
    pub solved_by: WorldSolvedByAnswer,
    /// Share of mass without a solution. Never renormalized away.
    pub failed_mass: f64,
    /// Share of mass withheld for lack of a certificate.
    pub withheld_mass: f64,
    /// Physical stability. Never claimed by these evaluators.
    pub physical_stability: String,
}

fn attempt(next: antecedent_core::refusal::NextAttempt) -> NextAttemptAnswer {
    NextAttemptAnswer { name: next.name().to_owned(), sentence: next.sentence().to_owned() }
}

fn certificate_answer(certificate: &WorldCertificate) -> WorldCertificateAnswer {
    let mut answer = WorldCertificateAnswer {
        certified: certificate.is_certified(),
        basis: None,
        gap: None,
        rho_abs_upper: None,
        certified_draws: None,
        uncertified_draws: None,
        failed_draws: None,
        withheld_draws: None,
        next_attempt: certificate.next_attempt().map(attempt),
    };
    match certificate {
        WorldCertificate::Certified { basis } => match basis {
            CertificateBasis::LinearFamilyContraction { rho_abs_upper } => {
                answer.basis = Some("linear_family_contraction".into());
                answer.rho_abs_upper = Some(*rho_abs_upper);
            }
            CertificateBasis::NonlinearBlockContraction { draws } => {
                answer.basis = Some("nonlinear_block_contraction".into());
                answer.certified_draws = Some(*draws);
            }
            _ => answer.basis = Some("declared_certificate".into()),
        },
        WorldCertificate::NotCertified { gap, .. } => match gap {
            CertificateGap::LinearFamilyBoundInsufficient { rho_abs_upper } => {
                answer.gap = Some("linear_family_bound_insufficient".into());
                answer.rho_abs_upper = Some(*rho_abs_upper).filter(|rho| rho.is_finite());
            }
            CertificateGap::NonlinearUniquenessUnproved { uncertified_draws } => {
                answer.gap = Some("nonlinear_uniqueness_unproved".into());
                answer.uncertified_draws = Some(*uncertified_draws);
            }
            CertificateGap::NonlinearMassLost { failed_draws, withheld_draws } => {
                answer.gap = Some("nonlinear_mass_lost".into());
                answer.failed_draws = Some(*failed_draws);
                answer.withheld_draws = Some(*withheld_draws);
            }
            CertificateGap::NumericalIntegrationOnly => {
                answer.gap = Some("numerical_integration_only".into());
            }
            CertificateGap::NoSolution => answer.gap = Some("no_solution".into()),
            _ => answer.gap = Some("not_certified".into()),
        },
    }
    answer
}

fn solved_by_answer(solved_by: &WorldSolvedBy) -> WorldSolvedByAnswer {
    let mut answer = WorldSolvedByAnswer {
        method: solved_by.as_str().to_owned(),
        order: Vec::new(),
        blocks: Vec::new(),
        dag_refusal: None,
        accepted_steps: None,
        rejected_steps: None,
    };
    match solved_by {
        WorldSolvedBy::AcyclicOrder { order } => answer.order.clone_from(order),
        WorldSolvedBy::StronglyConnectedBlocks { dag_refusal, blocks } => {
            answer.dag_refusal = Some(dag_refusal.clone());
            answer.blocks.clone_from(blocks);
        }
        WorldSolvedBy::DirectFreeBlockSolve { free } => answer.order.clone_from(free),
        WorldSolvedBy::NonlinearBlocks { blocks, .. } => answer.blocks.clone_from(blocks),
        WorldSolvedBy::AdaptiveIntegration { accepted_steps, rejected_steps, .. } => {
            answer.accepted_steps = Some(*accepted_steps);
            answer.rejected_steps = Some(*rejected_steps);
        }
        _ => {}
    }
    answer
}

fn world_answer(result: &CyclicWorldResult) -> Result<CyclicWorldAnswer, CyclicArtifactError> {
    let outcome = match result.outcome() {
        CyclicWorldOutcome::GaussianLaw { mean, covariance, residual_inf } => {
            CyclicWorldOutcomeAnswer::GaussianLaw {
                mean: mean.clone(),
                covariance: covariance.clone(),
                residual_inf: residual_inf.filter(|value| value.is_finite()),
            }
        }
        CyclicWorldOutcome::EquilibriumDraws { outcomes } => {
            CyclicWorldOutcomeAnswer::EquilibriumDraws { outcomes: outcomes.clone() }
        }
        CyclicWorldOutcome::Trajectory { trajectory } => {
            CyclicWorldOutcomeAnswer::Trajectory { trajectory: trajectory.clone() }
        }
        CyclicWorldOutcome::NoSolution { failure } => CyclicWorldOutcomeAnswer::NoSolution {
            cause: format!("{:?}", failure.cause),
            next_attempt: failure.next_attempt.name().to_owned(),
        },
        CyclicWorldOutcome::Withheld => CyclicWorldOutcomeAnswer::Withheld,
        _ => return Err(invalid("cyclic world outcome has no portable answer")),
    };
    Ok(CyclicWorldAnswer {
        world: result.world().raw(),
        family: result.family().as_str().to_owned(),
        variables: result.variables().iter().map(|id| id.raw()).collect(),
        actions: result
            .actions()
            .iter()
            .map(|action| CyclicActionAnswer {
                kind: action.kind().to_owned(),
                variable: action.variable().raw(),
                node: action.node(),
            })
            .collect(),
        outcome,
        certificate: certificate_answer(result.certificate()),
        solved_by: solved_by_answer(result.solved_by()),
        failed_mass: result.mass().failed(),
        withheld_mass: result.mass().withheld(),
        physical_stability: format!("{:?}", result.physical_stability()).to_lowercase(),
    })
}

/// Hard-clamp positions and levels of every world, after admission on `family`.
fn admitted_clamps(
    family: CyclicFamily,
    variables: &CyclicVariables,
    initial_laws: &[Arc<str>],
    requests: &[CyclicWorldRequest],
) -> Result<Vec<Vec<(usize, f64)>>, CyclicArtifactError> {
    requests
        .iter()
        .map(|request| {
            let resolved =
                resolve_cyclic_world(family, variables, initial_laws, request).map_err(refused)?;
            Ok(resolved
                .actions()
                .iter()
                .filter_map(|action| match action {
                    CyclicAction::HardClamp { node, level, .. } => Some((*node, *level)),
                    _ => None,
                })
                .collect())
        })
        .collect()
}

pub(super) fn linear_world_plans(
    variables: &[u32],
    model: &LinearCyclicWire,
    worlds: &[CyclicWorldWire],
) -> Result<Vec<CompiledSystemPlan>, CyclicArtifactError> {
    let bound = variables_of(variables)?;
    let requests = world_requests(worlds)?;
    let map = LinearStructuralMap::try_new(model.coefficients.clone(), model.intercept.clone())
        .map_err(invalid)?;
    admitted_clamps(CyclicFamily::LinearEquilibrium, &bound, &[], &requests)?
        .into_iter()
        .map(|clamps| {
            let nodes: Vec<usize> = clamps.iter().map(|(node, _)| *node).collect();
            let clamp = HardClamp::try_new(bound.len(), &nodes).map_err(invalid)?;
            let route = antecedent_model::scc_compile::route_linear_support(&map, &clamp)
                .map_err(invalid)?;
            Ok(super::compiled::plan_from_route(&route))
        })
        .collect()
}

pub(super) fn equilibrium_world_plans(
    variables: &[u32],
    model: &EquilibriumModel,
    worlds: &[CyclicWorldWire],
) -> Result<Vec<CompiledSystemPlan>, CyclicArtifactError> {
    model.validate()?;
    let bound = variables_of(variables)?;
    let requests = world_requests(worlds)?;
    admitted_clamps(CyclicFamily::NonlinearEquilibrium, &bound, &[], &requests)?
        .into_iter()
        .map(|clamps| {
            let clamps: Vec<Clamp> =
                clamps.into_iter().map(|(variable, value)| Clamp { variable, value }).collect();
            compile_equilibrium(model, &clamps).map_err(CyclicArtifactError::from)
        })
        .collect()
}

pub(super) fn ode_world_plans(
    variables: &[u32],
    model: &OdeSystem,
    initial_laws: &[String],
    worlds: &[CyclicWorldWire],
) -> Result<Vec<CompiledSystemPlan>, CyclicArtifactError> {
    let bound = variables_of(variables)?;
    let requests = world_requests(worlds)?;
    let laws: Vec<Arc<str>> = initial_laws.iter().map(|law| Arc::from(law.as_str())).collect();
    admitted_clamps(CyclicFamily::ContinuousTime, &bound, &laws, &requests)?;
    Ok(vec![compile_ode(model)?])
}

pub(super) fn execute_linear_worlds(
    variables: &[u32],
    model: &LinearCyclicWire,
    component: &CyclicComponentWire,
    worlds: &[CyclicWorldWire],
    certificate: CertificateDemandWire,
    ctx: &ExecutionContext,
) -> Result<A, CyclicArtifactError> {
    let bound = variables_of(variables)?;
    let map = LinearStructuralMap::try_new(model.coefficients.clone(), model.intercept.clone())
        .map_err(invalid)?;
    let disturbance =
        GaussianDisturbance::try_new(model.noise_mean.clone(), model.noise_covariance.clone())
            .map_err(invalid)?;
    let evidence: Vec<&str> = component.certifying_evidence.iter().map(String::as_str).collect();
    let spec = LinearCyclicSpecification::try_new(
        bound,
        map,
        disturbance,
        &component.component_id,
        &component.source_id,
        &evidence,
    )
    .map_err(refused)?;
    let requests = world_requests(worlds)?;
    if ctx.cancellation.is_cancelled() {
        return Err(invalid("cyclic world evaluation cancelled"));
    }
    let results = evaluate_linear_worlds(&spec, &requests, certificate.engine(), budget(ctx))
        .map_err(refused)?;
    let answers = requests
        .iter()
        .map(|request| {
            let result = results
                .get(request.world())
                .ok_or_else(|| invalid("an evaluated world is missing"))?;
            world_answer(result)
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(A::Worlds(answers))
}

pub(super) fn execute_equilibrium_worlds(
    variables: &[u32],
    model: &EquilibriumModel,
    worlds: &[CyclicWorldWire],
    noise: &[Vec<f64>],
    initial: &[f64],
    policy: SolvePolicy,
    ctx: &ExecutionContext,
) -> Result<A, CyclicArtifactError> {
    let spec = NonlinearCyclicSpecification::try_new(variables_of(variables)?, model.clone())
        .map_err(refused)?;
    let n = model.order();
    if noise.is_empty()
        || noise.len() > MAX_DRAWS
        || noise.iter().any(|row| row.len() != n || row.iter().any(|value| !value.is_finite()))
        || initial.len() != n
        || noise.len().saturating_mul(worlds.len()).saturating_mul(n) > 1_000_000
    {
        return Err(invalid(
            "equilibrium worlds need finite disturbance rows of the model's order and an \
             initial guess of that order",
        ));
    }
    let flat = noise.concat();
    let requests = world_requests(worlds)?;
    let mut answers = Vec::with_capacity(requests.len());
    for request in &requests {
        if ctx.cancellation.is_cancelled() {
            return Err(invalid("cyclic world evaluation cancelled"));
        }
        let result = evaluate_nonlinear_world(&spec, request, &flat, initial, policy, ctx)
            .map_err(refused)?;
        answers.push(world_answer(&result)?);
    }
    Ok(A::Worlds(answers))
}

#[allow(clippy::too_many_arguments, reason = "each argument is one field of the typed request")]
pub(super) fn execute_ode_worlds(
    variables: &[u32],
    model: &OdeSystem,
    initial: &[f64],
    initial_laws: &[String],
    times: &[f64],
    worlds: &[CyclicWorldWire],
    policy: OdePolicy,
    ctx: &ExecutionContext,
) -> Result<A, CyclicArtifactError> {
    let laws: Vec<Arc<str>> = initial_laws.iter().map(|law| Arc::from(law.as_str())).collect();
    let spec = ContinuousTimeSpecification::try_new(
        variables_of(variables)?,
        model.clone(),
        initial.to_vec(),
        laws,
        times.to_vec(),
    )
    .map_err(refused)?;
    let requests = world_requests(worlds)?;
    let mut answers = Vec::with_capacity(requests.len());
    for request in &requests {
        if ctx.cancellation.is_cancelled() {
            return Err(invalid("cyclic world evaluation cancelled"));
        }
        let result =
            evaluate_continuous_time_world(&spec, request, policy, ctx).map_err(refused)?;
        answers.push(world_answer(&result)?);
    }
    Ok(A::Worlds(answers))
}

// ---------------------------------------------------------------------------
// Draw origin and recovered laws
// ---------------------------------------------------------------------------

/// Declared sampling design of the experimental units.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SamplingDesignWire {
    /// Independent subjects; a subject's trajectory is one unit.
    IndependentSubjectTrajectories,
    /// Rows nested in clusters; the cluster is the unit.
    ClusteredPanel,
    /// One dependent series.
    DependentTimeSeries,
}

impl SamplingDesignWire {
    const fn engine(self) -> antecedent_core::DeclaredSamplingDesign {
        use antecedent_core::DeclaredSamplingDesign as D;
        match self {
            Self::IndependentSubjectTrajectories => D::IndependentSubjectTrajectories,
            Self::ClusteredPanel => D::ClusteredPanel,
            Self::DependentTimeSeries => D::DependentTimeSeries,
        }
    }
    const fn of(design: antecedent_core::DeclaredSamplingDesign) -> Self {
        use antecedent_core::DeclaredSamplingDesign as D;
        match design {
            D::IndependentSubjectTrajectories => Self::IndependentSubjectTrajectories,
            D::ClusteredPanel => Self::ClusteredPanel,
            D::DependentTimeSeries => Self::DependentTimeSeries,
        }
    }
    /// Wire spelling.
    const fn label(self) -> &'static str {
        match self {
            Self::IndependentSubjectTrajectories => "independent_subject_trajectories",
            Self::ClusteredPanel => "clustered_panel",
            Self::DependentTimeSeries => "dependent_time_series",
        }
    }
}

/// Resampling unit.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResamplingRuleWire {
    /// Whole subjects.
    RowSubjectIid,
    /// Whole clusters.
    WholeCluster,
    /// Contiguous blocks.
    ContiguousBlock,
    /// Individual rows. No declared design allows it; kept so the refusal is typed.
    RowIid,
}

impl ResamplingRuleWire {
    const fn engine(self) -> antecedent_core::ResamplingRule {
        use antecedent_core::ResamplingRule as R;
        match self {
            Self::RowSubjectIid => R::RowSubjectIid,
            Self::WholeCluster => R::WholeCluster,
            Self::ContiguousBlock => R::ContiguousBlock,
            Self::RowIid => R::RowIid,
        }
    }
    const fn of(rule: antecedent_core::ResamplingRule) -> Self {
        use antecedent_core::ResamplingRule as R;
        match rule {
            R::RowSubjectIid => Self::RowSubjectIid,
            R::WholeCluster => Self::WholeCluster,
            R::ContiguousBlock => Self::ContiguousBlock,
            R::RowIid => Self::RowIid,
        }
    }
}

/// Coefficient prior of a recovered-law posterior.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum CoefficientPriorWire {
    /// Improper flat prior on the identified directions.
    Flat,
    /// Independent normal prior, projected onto the identified subspace only.
    Normal {
        /// Prior mean of every coefficient.
        mean: f64,
        /// Prior variance of every coefficient.
        variance: f64,
    },
}

/// Disturbance prior of a recovered-law posterior.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum DisturbancePriorWire {
    /// Known disturbance variances, in node order.
    Known {
        /// Variance of each node's disturbance.
        variances: Vec<f64>,
    },
    /// Scaled inverse chi-square on each diagonal disturbance variance.
    InverseWishart {
        /// Prior degrees of freedom.
        degrees_of_freedom: f64,
        /// Prior scale per node.
        scale: Vec<f64>,
    },
}

/// Prior of a recovered-law posterior. The intercept prior is flat.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RecoveryPriorWire {
    /// Coefficient prior.
    pub coefficients: CoefficientPriorWire,
    /// Disturbance prior.
    pub disturbance: DisturbancePriorWire,
}

impl RecoveryPriorWire {
    fn engine(&self) -> RecoveryPrior {
        RecoveryPrior {
            coefficients: match self.coefficients {
                CoefficientPriorWire::Flat => CoefficientPrior::Flat,
                CoefficientPriorWire::Normal { mean, variance } => {
                    CoefficientPrior::Normal { mean, variance }
                }
            },
            disturbance: match &self.disturbance {
                DisturbancePriorWire::Known { variances } => {
                    DisturbancePrior::Known { variances: variances.clone() }
                }
                DisturbancePriorWire::InverseWishart { degrees_of_freedom, scale } => {
                    DisturbancePrior::InverseWishart {
                        degrees_of_freedom: *degrees_of_freedom,
                        scale: scale.clone(),
                    }
                }
            },
        }
    }
    fn of(prior: &RecoveryPrior) -> Self {
        Self {
            coefficients: match prior.coefficients {
                CoefficientPrior::Flat => CoefficientPriorWire::Flat,
                CoefficientPrior::Normal { mean, variance } => {
                    CoefficientPriorWire::Normal { mean, variance }
                }
            },
            disturbance: match &prior.disturbance {
                DisturbancePrior::Known { variances } => {
                    DisturbancePriorWire::Known { variances: variances.clone() }
                }
                DisturbancePrior::InverseWishart { degrees_of_freedom, scale } => {
                    DisturbancePriorWire::InverseWishart {
                        degrees_of_freedom: *degrees_of_freedom,
                        scale: scale.clone(),
                    }
                }
            },
        }
    }
}

/// How the draws of a recovered law were produced.
///
/// Supplied draws declare no meaning: an evidence projection keeps them on the
/// interventional port only. A sampling law is never read as a posterior.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(tag = "origin", rename_all = "snake_case", deny_unknown_fields)]
pub enum RecoveredDrawOriginWire {
    /// The caller supplied the draws.
    #[default]
    Supplied,
    /// Joint asymptotic normal law of the experimental fit. A sampling law.
    AsymptoticNormal {
        /// Number of draws.
        draws: usize,
        /// Seed.
        seed: u64,
    },
    /// Whole-recovery bootstrap of the experimental units. A sampling law.
    Bootstrap {
        /// Replicates, failed ones included.
        replicates: usize,
        /// Seed.
        seed: u64,
        /// Declared sampling design.
        design: SamplingDesignWire,
        /// Resampling rule the design allowed.
        rule: ResamplingRuleWire,
    },
    /// Gaussian posterior over the identified coefficient subspace.
    Posterior {
        /// Prior.
        prior: RecoveryPriorWire,
        /// Retained draws.
        draws: usize,
        /// Discarded sweeps of each chain.
        burn_in: usize,
        /// Seed.
        seed: u64,
    },
}

impl RecoveredDrawOriginWire {
    /// Whether the draws were supplied without a declared meaning.
    #[must_use]
    pub const fn is_supplied(&self) -> bool {
        matches!(self, Self::Supplied)
    }

    /// Evidence meaning of the draws: `sampling_law`, `parameter_posterior`, or
    /// `None` for supplied draws.
    #[must_use]
    pub const fn law_meaning(&self) -> Option<&'static str> {
        match self {
            Self::Supplied => None,
            Self::AsymptoticNormal { .. } | Self::Bootstrap { .. } => Some("sampling_law"),
            Self::Posterior { .. } => Some("parameter_posterior"),
        }
    }

    fn of(origin: &RecoveredDrawOrigin) -> Self {
        match origin {
            RecoveredDrawOrigin::Supplied => Self::Supplied,
            RecoveredDrawOrigin::AsymptoticNormal { draws, seed } => {
                Self::AsymptoticNormal { draws: *draws, seed: *seed }
            }
            RecoveredDrawOrigin::Bootstrap { replicates, seed, design, rule } => Self::Bootstrap {
                replicates: *replicates,
                seed: *seed,
                design: SamplingDesignWire::of(*design),
                rule: ResamplingRuleWire::of(*rule),
            },
            RecoveredDrawOrigin::Posterior { prior, draws, burn_in, seed } => Self::Posterior {
                prior: RecoveryPriorWire::of(prior),
                draws: *draws,
                burn_in: *burn_in,
                seed: *seed,
            },
        }
    }
}

/// One direct coefficient of a generated draw. An absent value is not finite.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RecoveredLawCoefficient {
    /// Target equation id.
    pub into: u32,
    /// Source variable id.
    pub from: u32,
    /// Value; `None` when the generator left it non-finite.
    pub value: Option<f64>,
}

/// One generated draw of a recovered law. `None` entries mark unresolved rows,
/// whose values the generator did not complete.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RecoveredLawDraw {
    /// Direct effects named by this draw.
    pub coefficients: Vec<RecoveredLawCoefficient>,
    /// Intercept in node order.
    pub intercept: Vec<Option<f64>>,
    /// Disturbance mean in node order.
    pub noise_mean: Vec<Option<f64>>,
    /// Row-major disturbance covariance.
    pub noise_covariance: Vec<Option<f64>>,
    /// Unresolved directions carried by the draw.
    pub unresolved: Vec<RecoveredDirectionWire>,
    /// Weight; the weights of the law sum to one.
    pub weight: f64,
    /// Inadmissible regardless of its numbers (a failed replicate, for one).
    pub marked_inadmissible: bool,
}

/// Status of one generated draw.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case", deny_unknown_fields)]
pub enum RecoveredDrawStatusWire {
    /// Every row resolved.
    Complete {
        /// Invertibility of `I − U K` on every declared experiment, when computed.
        /// Not physical stability.
        weakly_stable: Option<bool>,
        /// Whether `ρ(|K|) < 1` was established, when computed.
        family_admitted: Option<bool>,
    },
    /// The draw carries unresolved directions; none was completed.
    Unresolved {
        /// Number of unresolved directions.
        directions: usize,
    },
    /// The replicate's recovery failed; its weight stays as failed mass.
    Failed {
        /// Why.
        error: String,
    },
    /// Marked inadmissible by the caller.
    Marked,
}

/// The recovered law an experimental fit produced, without its worlds.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RecoveredLawProduct {
    /// Retained unit ids, strictly increasing; positions index the draws.
    pub nodes: Vec<u32>,
    /// How the draws were produced.
    pub origin: RecoveredDrawOriginWire,
    /// Draws in law order, failed and inadmissible ones included.
    pub draws: Vec<RecoveredLawDraw>,
    /// One status per draw.
    pub draw_status: Vec<RecoveredDrawStatusWire>,
    /// Mass of replicates whose recovery failed.
    pub failed_mass: f64,
}

/// One hard-clamp world of a recovered law, keyed by its scientific world id.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RecoveredKeyedWorld {
    /// Scientific world id.
    pub world: u32,
    /// Draw-aligned evaluation; failed and inadmissible draws stay at their weights.
    pub answer: RecoveredWorldAnswer,
}

/// A law recovered from clamp experiments and its intervention worlds.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExperimentalRecoveryReport {
    /// The recovered law.
    #[serde(with = "shared_recovered_law")]
    pub law: std::sync::Arc<RecoveredLawProduct>,
    /// Evidence meaning of the draws.
    pub law_meaning: Option<String>,
    /// Worlds in request order.
    pub worlds: Vec<RecoveredKeyedWorld>,
}

mod shared_recovered_law {
    use super::RecoveredLawProduct;
    use serde::{Deserialize, Serialize};
    use std::sync::Arc;
    pub fn serialize<S: serde::Serializer>(
        law: &Arc<RecoveredLawProduct>,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        law.as_ref().serialize(serializer)
    }
    pub fn deserialize<'de, D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Arc<RecoveredLawProduct>, D::Error> {
        RecoveredLawProduct::deserialize(deserializer).map(Arc::new)
    }
}

fn finite(value: f64) -> Option<f64> {
    value.is_finite().then_some(value)
}

fn direction_wire(direction: &UnresolvedDirection) -> RecoveredDirectionWire {
    RecoveredDirectionWire {
        support: direction
            .support
            .iter()
            .map(|item| RecoveredSupportWire { into: item.into.raw(), from: item.from.raw() })
            .collect(),
        reason: match &direction.reason {
            UnresolvedReason::IncompletePairCoverage { target, missing_clamps } => {
                RecoveredUnresolvedWire::IncompletePairCoverage {
                    target: target.raw(),
                    missing_clamps: missing_clamps.iter().map(|id| id.raw()).collect(),
                }
            }
            UnresolvedReason::DeficientConstraintRank { target, column_rank, width } => {
                RecoveredUnresolvedWire::DeficientConstraintRank {
                    target: target.raw(),
                    column_rank: *column_rank,
                    width: *width,
                }
            }
        },
    }
}

fn law_draw(draw: &JointRecoveryDraw) -> RecoveredLawDraw {
    RecoveredLawDraw {
        coefficients: draw
            .coefficients()
            .iter()
            .map(|(coefficient, value)| RecoveredLawCoefficient {
                into: coefficient.into.raw(),
                from: coefficient.from.raw(),
                value: finite(*value),
            })
            .collect(),
        intercept: draw.intercept().iter().copied().map(finite).collect(),
        noise_mean: draw.noise_mean().iter().copied().map(finite).collect(),
        noise_covariance: draw.noise_covariance().iter().copied().map(finite).collect(),
        unresolved: draw.unresolved().iter().map(direction_wire).collect(),
        weight: draw.weight(),
        marked_inadmissible: draw.marked_inadmissible(),
    }
}

fn draw_input(draw: &RecoveredLawDraw) -> RecoveryDrawInput {
    let restore = |values: &[Option<f64>]| -> Vec<f64> {
        values.iter().map(|value| value.unwrap_or(f64::NAN)).collect()
    };
    RecoveryDrawInput {
        coefficients: draw
            .coefficients
            .iter()
            .map(|item| {
                (
                    DirectCoefficient {
                        into: VariableId::from_raw(item.into),
                        from: VariableId::from_raw(item.from),
                    },
                    item.value.unwrap_or(f64::NAN),
                )
            })
            .collect(),
        intercept: restore(&draw.intercept),
        noise_mean: restore(&draw.noise_mean),
        noise_covariance: restore(&draw.noise_covariance),
        unresolved: draw.unresolved.iter().map(super::compiled::direction_input).collect(),
        weight: draw.weight,
        marked_inadmissible: draw.marked_inadmissible,
    }
}

fn status_wire(status: &DrawStatus) -> RecoveredDrawStatusWire {
    match status {
        DrawStatus::Complete { stability } => RecoveredDrawStatusWire::Complete {
            weakly_stable: stability.as_ref().map(SystemStability::weakly_stable),
            family_admitted: stability.as_ref().map(|item| item.family_admitted),
        },
        DrawStatus::Unresolved { directions } => {
            RecoveredDrawStatusWire::Unresolved { directions: *directions }
        }
        DrawStatus::Failed { error } => {
            RecoveredDrawStatusWire::Failed { error: error.to_string() }
        }
        DrawStatus::Marked => RecoveredDrawStatusWire::Marked,
    }
}

impl RecoveredLawProduct {
    fn of(estimate: &RecoveredEstimate) -> Self {
        let law = estimate.law();
        Self {
            nodes: law.nodes().iter().map(|node| node.raw()).collect(),
            origin: RecoveredDrawOriginWire::of(estimate.origin()),
            draws: law.draws().iter().map(law_draw).collect(),
            draw_status: estimate.draw_status().iter().map(status_wire).collect(),
            failed_mass: estimate.failed_mass(),
        }
    }

    fn law(&self) -> Result<RecoveredCoefficientUncertainty, CyclicArtifactError> {
        if self.draws.is_empty() || self.nodes.len() > 64 {
            return Err(invalid("a recovered law needs a draw and at most 64 nodes"));
        }
        let ids: Vec<VariableId> = self.nodes.iter().copied().map(VariableId::from_raw).collect();
        RecoveredCoefficientUncertainty::retain(&ids, self.draws.iter().map(draw_input).collect())
            .map_err(invalid)
    }
}

// ---------------------------------------------------------------------------
// Experimental recovery request
// ---------------------------------------------------------------------------

/// One clamp of an experiment regime.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegimeClampWire {
    /// Clamped unit id.
    pub unit: u32,
    /// Declared level, or the mean of the randomized clamp draws.
    pub value: f64,
}

/// One declared experiment regime.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExperimentRegimeWire {
    /// Randomized surgical clamps of this regime; empty is the passive regime.
    pub clamps: Vec<RegimeClampWire>,
    /// Sampling units of the regime.
    pub units: Vec<u32>,
    /// Population the units are drawn from.
    #[serde(default)]
    pub population: u32,
    /// The regime claims to share its disturbance with another regime.
    #[serde(default)]
    pub dependent: bool,
    /// Source identity of the regime's sample.
    pub source: u64,
}

/// One observed column of a regime sample.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UnitColumnWire {
    /// Unit id.
    pub unit: u32,
    /// Observed values, one per row, randomized clamp draws included.
    pub values: Vec<f64>,
}

/// Sample of one regime.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegimeSampleWire {
    /// Source the columns were drawn from; must equal the regime's source.
    pub source: u64,
    /// Columns of the retained units.
    pub columns: Vec<UnitColumnWire>,
    /// Sampling-unit label of every row, for whole-subject or whole-cluster resampling.
    #[serde(default)]
    pub sampling_units: Option<Vec<u64>>,
}

/// Row weighting of the constraint least-squares solve.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConstraintWeightingWire {
    /// Inverse model-based covariance of the rows.
    #[default]
    Efficient,
    /// Ordinary least squares.
    Unweighted,
}

/// Joint coefficient covariance estimator.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CoefficientCovarianceWire {
    /// Heteroskedasticity-robust sandwich.
    #[default]
    Sandwich,
    /// Model-based covariance under the declared Gaussian law.
    ModelBased,
    /// Not computed; asymptotic draws are then refused.
    NotComputed,
}

/// Divisor of the pooled residual variance.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DisturbanceDivisorWire {
    /// Row count (maximum likelihood).
    #[default]
    RowCount,
    /// Row count minus one.
    RowCountMinusOne,
}

/// Options of the experimental fit. The byte budget comes from the execution context.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExperimentalFitOptionsWire {
    /// Row weighting.
    #[serde(default)]
    pub weighting: ConstraintWeightingWire,
    /// Coefficient covariance estimator.
    #[serde(default)]
    pub covariance: CoefficientCovarianceWire,
    /// Disturbance divisor.
    #[serde(default)]
    pub divisor: DisturbanceDivisorWire,
    /// Declared sampling design of the experimental rows, when the study declares one.
    ///
    /// An asymptotic-normal or posterior origin treats rows as independent within each
    /// regime, so it is refused under a clustered or time-series design; a bootstrap
    /// must resample under this same design. Absent, the encoding and the request
    /// identity are unchanged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sampling_design: Option<SamplingDesignWire>,
}

/// Refuse an uncertainty origin that the declared row design does not allow.
///
/// The refusal carries the registered `estimator_inference_mismatch` reason and names
/// the next attempt: a bootstrap under the declared design.
fn check_recovery_design(
    options: ExperimentalFitOptionsWire,
    inference: &RecoveryInferenceWire,
) -> Result<(), CyclicArtifactError> {
    let Some(declared) = options.sampling_design else {
        return Ok(());
    };
    let code = antecedent_core::reason_code!("estimator_inference_mismatch");
    let origin = match inference {
        RecoveryInferenceWire::Bootstrap { design, .. } if *design == declared => return Ok(()),
        RecoveryInferenceWire::Bootstrap { design, .. } => {
            return Err(CyclicArtifactError::ScientificRefusal {
                code, detail: "cyclic_recovery.bootstrap_sampling_design_mismatch",
                message: format!("the bootstrap resamples under the `{}` design but the rows declare `{}`",design.label(),declared.label()),
                next_attempt: format!("use a bootstrap under the declared `{}` design",declared.label()),
            });
        }
        RecoveryInferenceWire::AsymptoticNormal { .. } => "asymptotic_normal",
        RecoveryInferenceWire::Posterior { .. } => "posterior",
    };
    match declared {
        SamplingDesignWire::IndependentSubjectTrajectories => Ok(()),
        SamplingDesignWire::ClusteredPanel | SamplingDesignWire::DependentTimeSeries => {
            Err(CyclicArtifactError::ScientificRefusal {
                code, detail: "cyclic_recovery.row_independence_unlicensed",
                message: format!("the `{origin}` origin treats rows as independent within each regime, which the declared `{}` design does not allow",declared.label()),
                next_attempt: format!("use a bootstrap under the declared `{}` design, which resamples the unit the design names",declared.label()),
            })
        }
    }
}

impl ExperimentalFitOptionsWire {
    fn engine(self, ctx: &ExecutionContext) -> ExperimentalFitOptions {
        ExperimentalFitOptions {
            weighting: match self.weighting {
                ConstraintWeightingWire::Efficient => ConstraintWeighting::Efficient,
                ConstraintWeightingWire::Unweighted => ConstraintWeighting::Unweighted,
            },
            covariance: match self.covariance {
                CoefficientCovarianceWire::Sandwich => CoefficientCovarianceEstimator::Sandwich,
                CoefficientCovarianceWire::ModelBased => CoefficientCovarianceEstimator::ModelBased,
                CoefficientCovarianceWire::NotComputed => {
                    CoefficientCovarianceEstimator::NotComputed
                }
            },
            divisor: match self.divisor {
                DisturbanceDivisorWire::RowCount => DisturbanceDivisor::RowCount,
                DisturbanceDivisorWire::RowCountMinusOne => DisturbanceDivisor::RowCountMinusOne,
            },
            max_bytes: ctx
                .memory_limit_bytes()
                .map_or(DEFAULT_FIT_BYTES, |limit| limit.min(DEFAULT_FIT_BYTES)),
        }
    }
}

/// How the uncertainty of the recovered law is produced.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "method", rename_all = "snake_case", deny_unknown_fields)]
pub enum RecoveryInferenceWire {
    /// Joint asymptotic normal law of the fit. A sampling law.
    AsymptoticNormal {
        /// Number of draws.
        draws: usize,
        /// Seed.
        seed: u64,
    },
    /// Whole-recovery bootstrap of the experimental units. A sampling law.
    Bootstrap {
        /// Replicates, failed ones included.
        replicates: usize,
        /// Seed.
        seed: u64,
        /// Declared sampling design of the units.
        design: SamplingDesignWire,
        /// Resampling rule; the design must allow it.
        rule: ResamplingRuleWire,
        /// Block length of contiguous-block resampling.
        #[serde(default)]
        block_length: Option<usize>,
    },
    /// Gaussian posterior over the identified coefficient subspace.
    Posterior {
        /// Prior.
        prior: RecoveryPriorWire,
        /// Retained draws.
        draws: usize,
        /// Discarded sweeps of each Gibbs chain.
        #[serde(default)]
        burn_in: usize,
        /// Seed.
        seed: u64,
    },
}

/// Retained unit ids of a declared contract: every declared unit minus the mask.
pub(super) fn retained_units(regimes: &[ExperimentRegimeWire], dropped: &[u32]) -> Vec<u32> {
    let dropped: BTreeSet<u32> = dropped.iter().copied().collect();
    regimes
        .iter()
        .flat_map(|regime| regime.units.iter().copied())
        .filter(|unit| !dropped.contains(unit))
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn regime_contract(regimes: &[ExperimentRegimeWire], dropped: &[u32]) -> RegimeContract {
    RegimeContract {
        regimes: regimes
            .iter()
            .map(|regime| ExperimentRegime {
                targets: regime
                    .clamps
                    .iter()
                    .map(|clamp| RegimeIntervention {
                        unit: VariableId::from_raw(clamp.unit),
                        value: Value::f64(clamp.value),
                    })
                    .collect(),
                units: regime.units.iter().copied().map(VariableId::from_raw).collect(),
                population: PopulationId::from_raw(regime.population),
                premise: if regime.dependent {
                    DependencePremise::Dependence
                } else {
                    DependencePremise::Independence
                },
                source: SourceIdentity::from_raw(regime.source),
            })
            .collect(),
        mask: UnitMask { dropped: dropped.iter().copied().map(VariableId::from_raw).collect() },
    }
}

fn check_experiment(
    regimes: &[ExperimentRegimeWire],
    samples: &[RegimeSampleWire],
    source_id: &str,
    certifying_evidence: &[String],
) -> Result<(), CyclicArtifactError> {
    if regimes.is_empty() || samples.len() != regimes.len() {
        return Err(invalid("experimental recovery needs one sample per declared regime"));
    }
    let values = samples
        .iter()
        .flat_map(|sample| sample.columns.iter())
        .try_fold(0usize, |sum, column| sum.checked_add(column.values.len()))
        .ok_or_else(|| invalid("experiment value count overflow"))?;
    if values > MAX_EXPERIMENT_VALUES {
        return Err(invalid("experimental recovery value limit"));
    }
    if source_id.is_empty()
        || certifying_evidence.is_empty()
        || certifying_evidence.iter().any(|item| item.is_empty() || item == source_id)
    {
        return Err(invalid(
            "experimental recovery needs a source id and certifying evidence other than the \
             source; next attempt: cite a held-out regime or other evidence that may certify \
             the component",
        ));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments, reason = "each argument is one field of the typed request")]
pub(super) fn experimental_recovery_plans(
    regimes: &[ExperimentRegimeWire],
    dropped: &[u32],
    samples: &[RegimeSampleWire],
    options: ExperimentalFitOptionsWire,
    inference: &RecoveryInferenceWire,
    worlds: &[CyclicWorldWire],
    outcome: u32,
    source_id: &str,
    certifying_evidence: &[String],
) -> Result<Vec<CompiledSystemPlan>, CyclicArtifactError> {
    // Compilation admits the worlds on the retained units and the uncertainty
    // origin under the declared row design. It does not fit.
    check_experiment(regimes, samples, source_id, certifying_evidence)?;
    check_recovery_design(options, inference)?;
    let units = retained_units(regimes, dropped);
    let variables = variables_of(&units)?;
    if variables.position(VariableId::from_raw(outcome)).is_none() {
        return Err(invalid(
            "experimental recovery outcome is not a retained unit; next attempt: name an \
             unmasked unit",
        ));
    }
    let requests = world_requests(worlds)?;
    for request in &requests {
        recovered_clamp(&variables, request)?;
    }
    Ok(Vec::new())
}

fn recovered_clamp(
    variables: &CyclicVariables,
    request: &CyclicWorldRequest,
) -> Result<(Vec<usize>, Vec<f64>), CyclicArtifactError> {
    let resolved = resolve_cyclic_world(CyclicFamily::LinearEquilibrium, variables, &[], request)
        .map_err(refused)?;
    let mut nodes = Vec::new();
    let mut values = Vec::new();
    for action in resolved.actions() {
        match action {
            CyclicAction::HardClamp { node, level, .. } => {
                nodes.push(*node);
                values.push(*level);
            }
            other => {
                return Err(CyclicArtifactError::ScientificRefusal {
                    code: antecedent_core::reason_code!("route_not_supported"),
                    detail: "cyclic_recovery.hard_clamp_required",
                    message: format!("world {}: `{}` on variable {}: a recovered law supports hard clamps only",
                        request.world().raw(),other.kind(),other.variable().raw()),
                    next_attempt: "evaluate this action on a separately declared eligible specified model; keep the recovered law and its original uncertainty origin explicit".into(),
                });
            }
        }
    }
    Ok((nodes, values))
}

fn fit_error(error: &ExperimentalFitError) -> CyclicArtifactError {
    CyclicArtifactError::RecoveryFit(error.clone())
}

#[cfg(test)]
std::thread_local! { static RECOVERY_FITS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) }; }
#[cfg(test)]
pub(super) fn recovery_fit_count() -> usize {
    RECOVERY_FITS.with(std::cell::Cell::get)
}

fn estimate_law(
    regimes: &[ExperimentRegimeWire],
    dropped: &[u32],
    samples: &[RegimeSampleWire],
    options: ExperimentalFitOptionsWire,
    inference: &RecoveryInferenceWire,
    ctx: &ExecutionContext,
) -> Result<RecoveredLawProduct, CyclicArtifactError> {
    #[cfg(test)]
    RECOVERY_FITS.with(|count| count.set(count.get() + 1));
    check_recovery_design(options, inference)?;
    let contract = regime_contract(regimes, dropped);
    let columns: Vec<Vec<UnitColumn<'_>>> = samples
        .iter()
        .map(|sample| {
            sample
                .columns
                .iter()
                .map(|column| UnitColumn {
                    unit: VariableId::from_raw(column.unit),
                    values: column.values.as_slice(),
                })
                .collect()
        })
        .collect();
    let regime_samples: Vec<RegimeSample<'_>> = columns
        .iter()
        .zip(samples)
        .map(|(columns, sample)| RegimeSample {
            columns: columns.as_slice(),
            source: SourceIdentity::from_raw(sample.source),
        })
        .collect();
    let rows = ExperimentRows { contract: &contract, samples: &regime_samples };
    let options = options.engine(ctx);
    let estimate = match inference {
        RecoveryInferenceWire::AsymptoticNormal { draws, seed } => {
            if *draws > MAX_DRAWS {
                return Err(invalid("asymptotic draw limit"));
            }
            let fit = fit_experimental_recovery(&rows, &options).map_err(|e| fit_error(&e))?;
            asymptotic_normal_recovery(&fit, *draws, *seed)
        }
        RecoveryInferenceWire::Bootstrap { replicates, seed, design, rule, block_length } => {
            if *replicates > MAX_DRAWS {
                return Err(invalid("bootstrap replicate limit"));
            }
            let labelled = samples.iter().filter(|sample| sample.sampling_units.is_some()).count();
            if labelled != 0 && labelled != samples.len() {
                return Err(invalid(
                    "sampling-unit labels must be given for every regime or for none",
                ));
            }
            let labels: Vec<&[u64]> =
                samples.iter().filter_map(|sample| sample.sampling_units.as_deref()).collect();
            let spec = BootstrapSpec {
                design: design.engine(),
                rule: rule.engine(),
                replicates: *replicates,
                seed: *seed,
                sampling_units: (labelled == samples.len()).then_some(labels.as_slice()),
                block_length: *block_length,
                batch: (*replicates).clamp(1, 64),
            };
            bootstrap_experimental_recovery(&rows, &options, &spec, ctx)
        }
        RecoveryInferenceWire::Posterior { prior, draws, burn_in, seed } => {
            if *draws > MAX_DRAWS {
                return Err(invalid("posterior draw limit"));
            }
            let fit = fit_experimental_recovery(&rows, &options).map_err(|e| fit_error(&e))?;
            posterior_recovery(
                &fit,
                &PosteriorSpec {
                    prior: prior.engine(),
                    draws: *draws,
                    burn_in: *burn_in,
                    seed: *seed,
                },
            )
        }
    }
    .map_err(|e| fit_error(&e))?;
    Ok(RecoveredLawProduct::of(&estimate))
}

#[allow(clippy::too_many_arguments, reason = "each argument is one field of the typed request")]
pub(super) fn execute_experimental_recovery(
    regimes: &[ExperimentRegimeWire],
    dropped: &[u32],
    samples: &[RegimeSampleWire],
    options: ExperimentalFitOptionsWire,
    inference: &RecoveryInferenceWire,
    worlds: &[CyclicWorldWire],
    outcome: u32,
    source_id: &str,
    certifying_evidence: &[String],
    frozen: Option<&std::sync::Arc<RecoveredLawProduct>>,
    ctx: &ExecutionContext,
) -> Result<A, CyclicArtifactError> {
    check_experiment(regimes, samples, source_id, certifying_evidence)?;
    let law = match frozen {
        Some(retained) => retained.clone(),
        None => {
            std::sync::Arc::new(estimate_law(regimes, dropped, samples, options, inference, ctx)?)
        }
    };
    if ctx.cancellation.is_cancelled() {
        return Err(invalid("experimental recovery cancelled"));
    }
    let worlds = evaluate_recovered_worlds(&law, worlds, outcome, source_id, certifying_evidence)?;
    Ok(A::ExperimentalRecovery(ExperimentalRecoveryReport {
        law_meaning: law.origin.law_meaning().map(str::to_owned),
        law,
        worlds,
    }))
}

/// Evaluate new worlds from the exact original law; no recovery or sampling.
pub(super) fn execute_retained_recovery(
    law: &std::sync::Arc<RecoveredLawProduct>,
    worlds: &[CyclicWorldWire],
    outcome: u32,
    source_id: &str,
    evidence: &[String],
    ctx: &ExecutionContext,
) -> Result<A, CyclicArtifactError> {
    if ctx.cancellation.is_cancelled() {
        return Err(CyclicArtifactError::Cancelled);
    }
    let answers = evaluate_recovered_worlds(law, worlds, outcome, source_id, evidence)?;
    Ok(A::ExperimentalRecovery(ExperimentalRecoveryReport {
        law: std::sync::Arc::clone(law),
        law_meaning: law.origin.law_meaning().map(str::to_owned),
        worlds: answers,
    }))
}

fn evaluate_recovered_worlds(
    product: &RecoveredLawProduct,
    worlds: &[CyclicWorldWire],
    outcome: u32,
    source_id: &str,
    certifying_evidence: &[String],
) -> Result<Vec<RecoveredKeyedWorld>, CyclicArtifactError> {
    // Every world is evaluated from the portable law, so a replay reads the same numbers.
    let law = product.law()?;
    let variables = CyclicVariables::try_new(law.nodes().to_vec()).map_err(refused)?;
    let outcome = variables.position(VariableId::from_raw(outcome)).ok_or_else(|| {
        invalid("experimental recovery outcome is not a retained unit; next attempt: name an unmasked unit")
    })?;
    let requests = world_requests(worlds)?;
    requests
        .iter()
        .map(|request| {
            let (clamp_nodes, clamp_values) = recovered_clamp(&variables, request)?;
            let plan = RetainedRecoveryPlan::try_new(RecoveryPlanParts {
                uncertainty: law.clone(),
                clamp_nodes,
                clamp_values,
                outcome,
                source_id: source_id.to_owned(),
                certifying_evidence: certifying_evidence.to_vec(),
            })
            .map_err(invalid)?;
            let evaluation = consume_retained_recovery_plan(&plan);
            Ok(RecoveredKeyedWorld {
                world: request.world().raw(),
                answer: super::compiled::world_answer(&evaluation),
            })
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Dynamic-fit inference
// ---------------------------------------------------------------------------

/// The declared fitting procedure of a continuous-time family.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "family", content = "problem", rename_all = "snake_case")]
pub enum DynamicProblemWire {
    /// ODE drift fit under Gaussian measurement error.
    Ode(OdeInferenceProblem),
    /// Stationary OU fit.
    Ou(OuInferenceProblem),
    /// Shared-mechanism OU fit.
    OuMechanisms(OuMechanismInferenceProblem),
}

impl DynamicProblemWire {
    /// State order of the declared system.
    #[must_use]
    pub fn order(&self) -> usize {
        match self {
            Self::Ode(problem) => problem.template.equations.len(),
            Self::Ou(problem) => problem.template.center.len(),
            Self::OuMechanisms(problem) => problem.template.equations.len(),
        }
    }
}

/// How the uncertainty of the fitted dynamic family is produced.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "method", rename_all = "snake_case", deny_unknown_fields)]
pub enum DynamicMethodWire {
    /// Whole-procedure bootstrap. A sampling law.
    Bootstrap {
        /// Parametric or nonparametric replicate data.
        kind: DynamicBootstrapMethod,
        /// Replicates, failed ones included.
        replicates: usize,
        /// Declared sampling design of the units.
        design: SamplingDesignWire,
        /// Attempted resampling rule; the design must allow it.
        resampling: ResamplingRuleWire,
        /// Strata of a nonparametric draw.
        #[serde(default)]
        stratification: CyclicBootstrapDesign,
        /// Seed of the replicate streams.
        seed: u64,
    },
    /// Observed information and its delta-method covariance, when the likelihood is regular.
    Information {
        /// Declared sampling design of the units.
        design: SamplingDesignWire,
    },
    /// Posterior under an explicit independent normal prior.
    Posterior {
        /// Prior over the estimated coordinates.
        prior: DynamicPrior,
        /// Sampler policy.
        #[serde(default)]
        options: DynamicPosteriorOptions,
        /// Declared sampling design of the units.
        design: SamplingDesignWire,
        /// Seed of the chains.
        seed: u64,
    },
}

/// Point fit of a continuous-time family.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "family", content = "fit", rename_all = "snake_case")]
pub enum DynamicFitAnswer {
    /// ODE fit.
    Ode(OdeFit),
    /// Stationary OU fit.
    Ou(OuFit),
    /// Shared-mechanism OU fit.
    OuMechanisms(OuMechanismFit),
}

impl DynamicFitAnswer {
    /// Log likelihood at the estimate.
    #[must_use]
    pub const fn log_likelihood(&self) -> f64 {
        match self {
            Self::Ode(fit) => fit.log_likelihood,
            Self::Ou(fit) => fit.log_likelihood,
            Self::OuMechanisms(fit) => fit.log_likelihood,
        }
    }
}

/// Uncertainty of a fitted dynamic family.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "method", content = "result", rename_all = "snake_case")]
pub enum DynamicInferenceAnswer {
    /// Bootstrap sampling law.
    SamplingLaw(DynamicSamplingLaw),
    /// Observed information, or the reason no covariance exists.
    Information(DynamicObservedInformation),
    /// Parameter posterior with quantity draws.
    Posterior(DynamicPosterior),
}

fn dynamic_error(error: DynamicInferenceError) -> CyclicArtifactError {
    match error {
        DynamicInferenceError::Model(model) => CyclicArtifactError::Execution(model),
        other => invalid(format!("dynamic inference refused: {other}")),
    }
}

fn fit_dynamic(
    problem: &DynamicProblemWire,
    ctx: &ExecutionContext,
) -> Result<DynamicFitAnswer, CyclicArtifactError> {
    Ok(match problem {
        DynamicProblemWire::Ode(p) => {
            DynamicFitAnswer::Ode(antecedent_estimate::cyclic_dynamics::fit_ode_likelihood(
                &p.template,
                &p.parameters,
                &p.data,
                p.integration,
                p.policy,
                ctx,
            )?)
        }
        DynamicProblemWire::Ou(p) => {
            DynamicFitAnswer::Ou(antecedent_estimate::cyclic_dynamics::fit_ou_likelihood(
                &p.template,
                &p.parameters,
                &p.rows,
                p.options,
                ctx,
            )?)
        }
        DynamicProblemWire::OuMechanisms(p) => {
            DynamicFitAnswer::OuMechanisms(antecedent_estimate::ou_mechanisms::fit_ou_mechanisms(
                &p.template,
                &p.parameters,
                &p.regimes,
                p.policy,
                ctx,
            )?)
        }
    })
}

fn seeded(ctx: &ExecutionContext, seed: u64) -> ExecutionContext {
    let mut context = ctx.clone();
    context.rng = RngFactory::from_seed(seed);
    context
}

pub(super) fn execute_dynamic_inference(
    problem: &DynamicProblemWire,
    method: &DynamicMethodWire,
    quantities: &[DynamicQuantity],
    frozen: Option<&DynamicFitAnswer>,
    ctx: &ExecutionContext,
) -> Result<A, CyclicArtifactError> {
    let fit = match frozen {
        Some(retained) => retained.clone(),
        None => fit_dynamic(problem, ctx)?,
    };
    let target = match (problem, &fit) {
        (DynamicProblemWire::Ode(problem), DynamicFitAnswer::Ode(fit)) => {
            DynamicFitRef::Ode { problem, fit }
        }
        (DynamicProblemWire::Ou(problem), DynamicFitAnswer::Ou(fit)) => {
            DynamicFitRef::Ou { problem, fit }
        }
        (DynamicProblemWire::OuMechanisms(problem), DynamicFitAnswer::OuMechanisms(fit)) => {
            DynamicFitRef::OuMechanisms { problem, fit }
        }
        _ => return Err(invalid("retained dynamic fit belongs to a different family")),
    };
    let inference = match method {
        DynamicMethodWire::Bootstrap {
            kind,
            replicates,
            design,
            resampling,
            stratification,
            seed,
        } => DynamicInferenceAnswer::SamplingLaw(
            bootstrap_dynamic_fit(
                target,
                DynamicBootstrapRequest {
                    method: *kind,
                    replicates: *replicates,
                    design: design.engine(),
                    resampling: resampling.engine(),
                    stratification: *stratification,
                },
                quantities,
                &seeded(ctx, *seed),
            )
            .map_err(dynamic_error)?,
        ),
        DynamicMethodWire::Information { design } => DynamicInferenceAnswer::Information(
            dynamic_observed_information(target, design.engine(), quantities, ctx)
                .map_err(dynamic_error)?,
        ),
        DynamicMethodWire::Posterior { prior, options, design, seed } => {
            DynamicInferenceAnswer::Posterior(
                sample_dynamic_posterior(
                    target,
                    prior,
                    *options,
                    design.engine(),
                    quantities,
                    &seeded(ctx, *seed),
                )
                .map_err(dynamic_error)?,
            )
        }
    };
    Ok(A::DynamicInference { fit, inference: Box::new(inference) })
}

// ---------------------------------------------------------------------------
// Keyed intervention worlds of any family that has them
// ---------------------------------------------------------------------------

/// One intervention world of a family answer, keyed by its scientific world id.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "source", rename_all = "snake_case", deny_unknown_fields)]
pub enum CyclicInterventionWorld {
    /// A world of a specified model, with its certificate and solve route.
    Specified {
        /// Evaluated world.
        answer: Box<CyclicWorldAnswer>,
    },
    /// A hard-clamp world of a recovered law, draw-aligned.
    Recovered {
        /// Scientific world id. Older index-based requests use the world's position.
        world: u32,
        /// How the draws were produced.
        origin: RecoveredDrawOriginWire,
        /// Draw-aligned evaluation.
        answer: RecoveredWorldAnswer,
    },
}

/// Intervention worlds of `answer`, or `None` for a family without them.
pub(super) fn intervention_worlds(
    request: &super::CyclicFamilyRequest,
    answer: &A,
) -> Option<Vec<CyclicInterventionWorld>> {
    match (request, answer) {
        (_, A::Worlds(worlds)) => Some(
            worlds
                .iter()
                .map(|answer| CyclicInterventionWorld::Specified {
                    answer: Box::new(answer.clone()),
                })
                .collect(),
        ),
        (_, A::ExperimentalRecovery(report)) => Some(
            report
                .worlds
                .iter()
                .map(|world| CyclicInterventionWorld::Recovered {
                    world: world.world,
                    origin: report.law.origin.clone(),
                    answer: world.answer.clone(),
                })
                .collect(),
        ),
        (
            super::CyclicFamilyRequest::RecoveredLinear { uncertainty, .. },
            A::RecoveredLinear(report),
        ) => report
            .worlds
            .iter()
            .enumerate()
            .map(|(index, answer)| {
                Some(CyclicInterventionWorld::Recovered {
                    world: u32::try_from(index).ok()?,
                    origin: uncertainty.clone(),
                    answer: answer.clone(),
                })
            })
            .collect(),
        _ => None,
    }
}

#[cfg(test)]
mod preparation_status_tests {
    use super::*;
    use super::super::{CompiledCyclicFamilyProgram, CyclicFamilyRequest as R};
    use antecedent_core::ExecutionContext;
    use antecedent_model::cyclic_world::CyclicRefusalKind;

    fn request() -> R {
        serde_json::from_str(include_str!("../../../parity/cyclic/recovered-asymptotic-normal/request.json")).unwrap()
    }
    fn with_worlds(mut request: R, selected: Vec<CyclicWorldWire>) -> R {
        let R::ExperimentalRecovery { worlds, .. } = &mut request else { panic!("recovery fixture") };
        *worlds = selected;
        request
    }
    fn world(action: CyclicInterventionWire) -> CyclicWorldWire {
        CyclicWorldWire { world: 7, interventions: vec![action], initial_states: vec![] }
    }
    #[test]
    fn native_preparation_distinguishes_capability_and_invalid_worlds_without_fitting() {
        let ctx = ExecutionContext::for_tests(19);
        let fits = recovery_fit_count();
        let original = request();
        let scientific = with_worlds(original.clone(), vec![world(CyclicInterventionWire::Shift { variable: 0, delta: 1.0 })]);
        assert!(matches!(CompiledCyclicFamilyProgram::compile(&scientific,&ctx), Err(CyclicArtifactError::ScientificRefusal { code: "route_not_supported", detail: "cyclic_recovery.hard_clamp_required", .. })));
        let corrected = with_worlds(scientific, vec![world(CyclicInterventionWire::Set { variable: 0, value: 3.0 })]);
        assert!(CompiledCyclicFamilyProgram::compile(&corrected,&ctx).is_ok());
        let duplicate = world(CyclicInterventionWire::Set { variable: 0, value: 3.0 });
        assert!(matches!(CompiledCyclicFamilyProgram::compile(&with_worlds(original.clone(),vec![duplicate.clone(),duplicate]),&ctx), Err(CyclicArtifactError::Invalid(message)) if message.contains("next attempt")));
        let malformed = with_worlds(original,vec![world(CyclicInterventionWire::Soft { variable: 0, family: "linear_mechanism".into(), parameters: vec![1.0] })]);
        assert!(matches!(CompiledCyclicFamilyProgram::compile(&malformed,&ctx), Err(CyclicArtifactError::WorldRefusal(error)) if matches!(error.kind,CyclicRefusalKind::SoftParameterShape { .. })));
        assert_eq!(recovery_fit_count(),fits);
    }
    #[test]
    fn design_status_is_typed_without_a_fit_or_method_substitution() {
        let ctx = ExecutionContext::for_tests(19);
        let fits = recovery_fit_count();
        let mut request = request();
        let R::ExperimentalRecovery { options, .. } = &mut request else { panic!("recovery fixture") };
        options.sampling_design = Some(SamplingDesignWire::DependentTimeSeries);
        assert!(matches!(CompiledCyclicFamilyProgram::compile(&request,&ctx), Err(CyclicArtifactError::ScientificRefusal { code: "estimator_inference_mismatch", .. })));
        assert_eq!(recovery_fit_count(),fits);
    }
}
