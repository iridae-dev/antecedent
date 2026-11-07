use super::nonlinear::replacement_equation;
use super::{
    Arc, CertificateGap, Clamp, CyclicAction, CyclicFamily, CyclicInterventionRefusal,
    CyclicRefusalKind, CyclicVariables, CyclicWorldOutcome, CyclicWorldRequest, CyclicWorldResult,
    ExecutionContext, INTEGRATION_FAILURE_ATTEMPT, OdePolicy, OdeSystem, TimedClamp,
    VALIDATED_INTEGRATION_ATTEMPT, WorldCertificate, WorldFailure, WorldFailureCause, WorldMass,
    WorldSolvedBy, execution_refusal, failed_parts, refusal, resolve_cyclic_world,
    specification_refusal,
};
use crate::nonlinear_equilibrium::EquilibriumError;

/// Specified continuous-time system with its initial state, the name of each
/// state's initial-state law, and the observation times.
#[derive(Clone, Debug, PartialEq)]
pub struct ContinuousTimeSpecification {
    variables: CyclicVariables,
    system: OdeSystem,
    initial_state: Vec<f64>,
    initial_laws: Vec<Arc<str>>,
    times: Vec<f64>,
}

impl ContinuousTimeSpecification {
    /// Bind `system` to `variables`.
    ///
    /// # Errors
    ///
    /// A system that does not validate; an initial state, law list, or order
    /// that disagree; an empty or repeated law name; or observation times that
    /// are empty, non-finite, or not strictly increasing.
    pub fn try_new(
        variables: CyclicVariables,
        system: OdeSystem,
        initial_state: Vec<f64>,
        initial_laws: Vec<Arc<str>>,
        times: Vec<f64>,
    ) -> Result<Self, CyclicInterventionRefusal> {
        let family = CyclicFamily::ContinuousTime;
        system.validate().map_err(|error| specification_refusal(family, None, error))?;
        let n = variables.len();
        if system.equations.len() != n || initial_state.len() != n || initial_laws.len() != n {
            return Err(specification_refusal(
                family,
                None,
                "the system, the initial state, the laws, and the variables have different orders",
            ));
        }
        if initial_state.iter().any(|value| !value.is_finite()) {
            return Err(specification_refusal(family, None, "the initial state is not finite"));
        }
        let mut names = std::collections::BTreeSet::new();
        if initial_laws.iter().any(|law| law.is_empty() || !names.insert(Arc::clone(law))) {
            return Err(specification_refusal(
                family,
                None,
                "initial-state law names must be non-empty and distinct",
            ));
        }
        if times.is_empty()
            || times.iter().any(|time| !time.is_finite())
            || times.windows(2).any(|pair| pair[1] <= pair[0])
        {
            return Err(specification_refusal(
                family,
                None,
                "observation times must be finite and strictly increasing",
            ));
        }
        Ok(Self { variables, system, initial_state, initial_laws, times })
    }

    /// Bound variables.
    #[must_use]
    pub const fn variables(&self) -> &CyclicVariables {
        &self.variables
    }

    /// Declared initial-state law names, in variable order.
    #[must_use]
    pub fn initial_laws(&self) -> &[Arc<str>] {
        &self.initial_laws
    }

    /// Observation times.
    #[must_use]
    pub fn times(&self) -> &[f64] {
        &self.times
    }
}

/// Evaluate one world of a continuous-time system.
///
/// An untimed set clamps the state, and its derivative, from the first
/// observation time over the whole horizon. A mechanism replacement replaces
/// the drift equation; the declared decay stays. An initial-state change
/// replaces that state's initial level. The trajectory is not certified:
/// adaptive integration controls local error only. An integration that does not
/// reach the horizon is a result with failed mass one.
///
/// # Errors
///
/// A refusal of the request, a malformed policy or intervened system, or a
/// resource or cancellation refusal.
pub fn evaluate_continuous_time_world(
    spec: &ContinuousTimeSpecification,
    request: &CyclicWorldRequest,
    policy: OdePolicy,
    ctx: &ExecutionContext,
) -> Result<CyclicWorldResult, CyclicInterventionRefusal> {
    let family = CyclicFamily::ContinuousTime;
    let resolved = resolve_cyclic_world(family, &spec.variables, &spec.initial_laws, request)?;
    let world = resolved.world;
    let n = spec.variables.len();
    let mut system = spec.system.clone();
    let mut initial = spec.initial_state.clone();
    let mut clamps = Vec::new();
    for action in &resolved.actions {
        match action {
            CyclicAction::HardClamp { node, level, .. } => {
                clamps.push(Clamp { variable: *node, value: *level });
            }
            CyclicAction::ReplaceMechanism { node, replacement, .. } => {
                let id = system.equations[*node].id.clone();
                system.equations[*node] = replacement_equation(&id, n, replacement);
            }
            CyclicAction::ChangeMechanism { node, intercept_shift, .. } => {
                system.equations[*node].intercept += intercept_shift;
            }
            CyclicAction::InitialState { node, level, .. } => initial[*node] = *level,
            CyclicAction::AdditiveShock { .. } => {
                return Err(refusal(
                    Some(world),
                    Some(family),
                    "shift",
                    CyclicRefusalKind::Capability {
                        reason: "the continuous-time family does not take a disturbance shock",
                    },
                    family.next_attempt("shift"),
                ));
            }
        }
    }
    let schedule = if clamps.is_empty() {
        Vec::new()
    } else {
        vec![TimedClamp { time: spec.times[0], clamps }]
    };
    let retained = ctx
        .acquire_admission(0, CyclicWorldResult::retained_trajectory_estimate(n, spec.times.len()))
        .map_err(|error| execution_refusal(family, world, error))?;
    match system.trajectory(&initial, &spec.times, &schedule, policy, ctx) {
        Ok(trajectory) => {
            let solved_by = WorldSolvedBy::AdaptiveIntegration {
                accepted_steps: trajectory.accepted_steps,
                rejected_steps: trajectory.rejected_steps,
                policy,
            };
            Ok(CyclicWorldResult::from_parts(
                resolved,
                &spec.variables,
                request,
                (
                    CyclicWorldOutcome::Trajectory { trajectory },
                    WorldCertificate::NotCertified {
                        gap: CertificateGap::NumericalIntegrationOnly,
                        next_attempt: VALIDATED_INTEGRATION_ATTEMPT,
                    },
                    solved_by,
                    WorldMass::SOLVED,
                ),
            )
            .retained_by(retained))
        }
        Err(error) => {
            let integration_failure = match &error {
                EquilibriumError::Nonconvergence { .. } => true,
                EquilibriumError::Resource(message) => {
                    !ctx.cancellation.is_cancelled() && message.contains("step budget")
                }
                EquilibriumError::Invalid(message) => message.contains("nonfinite"),
                _ => false,
            };
            if integration_failure {
                return Ok(CyclicWorldResult::from_parts(
                    resolved,
                    &spec.variables,
                    request,
                    failed_parts(WorldFailure {
                        cause: WorldFailureCause::Integration { detail: error.to_string() },
                        next_attempt: INTEGRATION_FAILURE_ATTEMPT,
                    }),
                ));
            }
            match error {
                EquilibriumError::Resource(message) => {
                    Err(execution_refusal(family, world, message))
                }
                other => Err(specification_refusal(family, Some(world), other)),
            }
        }
    }
}
