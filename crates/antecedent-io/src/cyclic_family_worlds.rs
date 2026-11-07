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

#[path = "cyclic_family_worlds/answers.rs"]
mod answers;
#[cfg(test)]
use answers::block_diagnostic_answer;
use answers::world_answer;
pub use answers::{
    CyclicActionAnswer, CyclicWorldAnswer, CyclicWorldOutcomeAnswer, NextAttemptAnswer,
    WorldCertificateAnswer, WorldSolvedByAnswer,
};

#[path = "cyclic_family_worlds/execution.rs"]
mod execution;
pub(super) use execution::{
    equilibrium_world_plans, execute_equilibrium_worlds, execute_linear_worlds, execute_ode_worlds,
    linear_world_plans, ode_world_plans,
};

#[path = "cyclic_family_worlds/dynamic.rs"]
mod dynamic;
pub(super) use dynamic::execute_dynamic_inference;
pub use dynamic::{
    DynamicFitAnswer, DynamicInferenceAnswer, DynamicMethodWire, DynamicProblemWire,
};

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
use antecedent_model::compile::{CompiledSystemPlan, compile_equilibria, compile_ode};
use antecedent_model::continuous_cyclic::{OdePolicy, OdeSystem, OdeTrajectory};
use antecedent_model::cyclic_world::{
    CertificateBasis, CertificateDemand, CertificateGap, ContinuousTimeSpecification, CyclicAction,
    CyclicVariables, CyclicWorldOutcome, CyclicWorldRequest, CyclicWorldResult,
    LinearCyclicSpecification, NonlinearCyclicSpecification, WorldCertificate, WorldSolvedBy,
    evaluate_continuous_time_world, evaluate_linear_worlds, evaluate_linear_worlds_routed,
    evaluate_nonlinear_world, resolve_cyclic_world,
};
use antecedent_model::nonlinear_equilibrium::{
    Clamp, EquilibriumModel, EquilibriumOutcome, SolvePolicy,
};
use antecedent_model::scc_compile::{
    CompileBudget, GaussianDisturbance, LinearRouteSet, gaussian_equilibrium_peak_bytes,
};
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

fn refused(
    refusal: antecedent_model::cyclic_world::CyclicInterventionRefusal,
) -> CyclicArtifactError {
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

/// Most worlds one request may declare. Each world is admitted, compiled and, on a model that
/// compiles per world, re-validated, so the count is bounded before any of that work.
pub(super) const MAX_WORLDS: usize = 10_000;
/// Most nodes a recovered law carries.
pub(super) const MAX_RECOVERED_NODES: usize = 64;

/// Refuse a world count above [`MAX_WORLDS`] before sizing any work by it.
pub(super) fn check_world_count(count: usize) -> Result<(), CyclicArtifactError> {
    if count > MAX_WORLDS {
        return Err(invalid("typed cyclic worlds: more worlds than the request limit"));
    }
    Ok(())
}

fn world_requests(
    worlds: &[CyclicWorldWire],
) -> Result<Vec<CyclicWorldRequest>, CyclicArtifactError> {
    if worlds.is_empty() {
        return Err(invalid("typed cyclic worlds: at least one world is required"));
    }
    check_world_count(worlds.len())?;
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

/// Admission work all worlds of one request may charge together (a dense 1024-node block is
/// about 10^10). Each world is charged against what the earlier worlds left.
const MAX_REQUEST_WORLD_WORK: u64 = 10_000_000_000_000;

fn budget(ctx: &ExecutionContext) -> CompileBudget {
    CompileBudget {
        max_work: MAX_REQUEST_WORLD_WORK,
        max_bytes: ctx.memory_limit_bytes().unwrap_or(u64::MAX),
    }
}

/// Admit the memory of evaluating `count` worlds before any is evaluated: `peak` scratch of
/// the world in flight, and for every world the result and the answer built from it, which
/// both live until the answers are returned.
fn admit_worlds(
    ctx: &ExecutionContext,
    peak: u64,
    per_world: u64,
    count: usize,
) -> Result<antecedent_core::execution::AdmissionLease, CyclicArtifactError> {
    let bytes = per_world
        .checked_mul(2)
        .and_then(|both| both.checked_mul(count as u64))
        .and_then(|worlds| worlds.checked_add(peak))
        .ok_or_else(|| invalid("typed cyclic worlds: world memory estimate overflows"))?;
    ctx.acquire_admission(0, bytes).map_err(CyclicArtifactError::Admission)
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

// ---------------------------------------------------------------------------
// Draw origin and recovered laws
// ---------------------------------------------------------------------------

#[path = "cyclic_family_worlds/recovery.rs"]
mod recovery;
#[cfg(test)]
pub(super) use recovery::recovery_fit_count;
pub use recovery::{
    CoefficientCovarianceWire, CoefficientPriorWire, ConstraintWeightingWire,
    DisturbanceDivisorWire, DisturbancePriorWire, ExperimentRegimeWire, ExperimentalFitOptionsWire,
    ExperimentalRecoveryReport, RecoveredDrawOriginWire, RecoveredDrawStatusWire,
    RecoveredKeyedWorld, RecoveredLawCoefficient, RecoveredLawDraw, RecoveredLawProduct,
    RecoveryInferenceWire, RecoveryPriorWire, RegimeClampWire, RegimeSampleWire,
    ResamplingRuleWire, SamplingDesignWire, UnitColumnWire,
};
pub(super) use recovery::{
    execute_experimental_recovery, execute_retained_recovery, experimental_recovery_plans,
    retained_units,
};

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
    use super::super::{CompiledCyclicFamilyProgram, CyclicFamilyRequest as R};
    use super::*;
    use antecedent_core::ExecutionContext;
    use antecedent_model::cyclic_world::CyclicRefusalKind;

    fn request() -> R {
        serde_json::from_str(include_str!(
            "../../../parity/cyclic/recovered-asymptotic-normal/request.json"
        ))
        .unwrap()
    }
    fn with_worlds(mut request: R, selected: Vec<CyclicWorldWire>) -> R {
        let R::ExperimentalRecovery { worlds, .. } = &mut request else {
            panic!("recovery fixture")
        };
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
        let scientific = with_worlds(
            original.clone(),
            vec![world(CyclicInterventionWire::Shift { variable: 0, delta: 1.0 })],
        );
        assert!(matches!(
            CompiledCyclicFamilyProgram::compile(&scientific, &ctx),
            Err(CyclicArtifactError::ScientificRefusal {
                code: "route_not_supported",
                detail: "cyclic_recovery.hard_clamp_required",
                ..
            })
        ));
        let corrected = with_worlds(
            scientific,
            vec![world(CyclicInterventionWire::Set { variable: 0, value: 3.0 })],
        );
        assert!(CompiledCyclicFamilyProgram::compile(&corrected, &ctx).is_ok());
        let duplicate = world(CyclicInterventionWire::Set { variable: 0, value: 3.0 });
        assert!(
            matches!(CompiledCyclicFamilyProgram::compile(&with_worlds(original.clone(),vec![duplicate.clone(),duplicate]),&ctx), Err(CyclicArtifactError::Invalid(message)) if message.contains("next attempt"))
        );
        let malformed = with_worlds(
            original,
            vec![world(CyclicInterventionWire::Soft {
                variable: 0,
                family: "linear_mechanism".into(),
                parameters: vec![1.0],
            })],
        );
        assert!(
            matches!(CompiledCyclicFamilyProgram::compile(&malformed,&ctx), Err(CyclicArtifactError::WorldRefusal(error)) if matches!(error.kind,CyclicRefusalKind::SoftParameterShape { .. }))
        );
        assert_eq!(recovery_fit_count(), fits);
    }
    #[test]
    fn design_status_is_typed_without_a_fit_or_method_substitution() {
        let ctx = ExecutionContext::for_tests(19);
        let fits = recovery_fit_count();
        let mut request = request();
        let R::ExperimentalRecovery { options, .. } = &mut request else {
            panic!("recovery fixture")
        };
        options.sampling_design = Some(SamplingDesignWire::DependentTimeSeries);
        assert!(matches!(
            CompiledCyclicFamilyProgram::compile(&request, &ctx),
            Err(CyclicArtifactError::ScientificRefusal {
                code: "estimator_inference_mismatch",
                ..
            })
        ));
        assert_eq!(recovery_fit_count(), fits);
    }
}

#[cfg(test)]
mod block_diagnostic_wire_tests {
    use super::*;

    #[test]
    fn nonfinite_diagnostics_are_refused_before_json_serialization() {
        let mut block = antecedent_model::scc_compile::BlockDiagnostic {
            members: vec![0],
            residual_inf: 0.0,
            kappa_inf: f64::MAX,
            exact: false,
        };
        let valid = block_diagnostic_answer(&block).unwrap();
        let json = serde_json::to_value(valid).unwrap();
        assert!(json["kappa_inf"].is_number());
        block.kappa_inf = f64::INFINITY;
        assert!(block_diagnostic_answer(&block).is_err());
        block.kappa_inf = 1.0;
        block.residual_inf = f64::NAN;
        assert!(block_diagnostic_answer(&block).is_err());
    }
}
