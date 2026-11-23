use super::{
    Activation, BLOCK_CONTRACTION_ATTEMPT, CertificateBasis, CertificateGap, CyclicAction,
    CyclicFamily, CyclicInterventionRefusal, CyclicRefusalKind, CyclicVariables,
    CyclicWorldOutcome, CyclicWorldRequest, CyclicWorldResult, Equation, EquilibriumAction,
    EquilibriumError, EquilibriumModel, EquilibriumOutcome, ExecutionContext, MechanismReplacement,
    NONLINEAR_WORLD_MAX_WORK, OUTCOME_OVERHEAD_BYTES, RESULT_OVERHEAD_BYTES,
    RETAINED_FAILED_DRAWS_ATTEMPT, ResolvedCyclicWorld, SolvePolicy, WorldCertificate, WorldMass,
    WorldSolvedBy, execution_refusal, refusal, resolve_cyclic_world, specification_refusal,
};

/// Specified nonlinear additive-noise equilibrium with bound variables.
#[derive(Clone, Debug, PartialEq)]
pub struct NonlinearCyclicSpecification {
    variables: CyclicVariables,
    model: EquilibriumModel,
}

impl NonlinearCyclicSpecification {
    /// Bind `model` to `variables`. Equation `k` is the mechanism of variable `k`.
    ///
    /// # Errors
    ///
    /// A model that does not validate, or an order that differs from the variables.
    pub fn try_new(
        variables: CyclicVariables,
        model: EquilibriumModel,
    ) -> Result<Self, CyclicInterventionRefusal> {
        let family = CyclicFamily::NonlinearEquilibrium;
        model.validate().map_err(|error| specification_refusal(family, None, error))?;
        if model.order() != variables.len() {
            return Err(specification_refusal(
                family,
                None,
                "the model and the variables have different orders",
            ));
        }
        Ok(Self { variables, model })
    }

    /// Bound variables.
    #[must_use]
    pub const fn variables(&self) -> &CyclicVariables {
        &self.variables
    }

    /// Specified model.
    #[must_use]
    pub const fn model(&self) -> &EquilibriumModel {
        &self.model
    }
}

pub(super) fn replacement_equation(
    id: &str,
    order: usize,
    replacement: &MechanismReplacement,
) -> Equation {
    let (intercept, weights) = match replacement {
        MechanismReplacement::Constant { value } => (*value, vec![0.0; order]),
        MechanismReplacement::Linear { intercept, weights } => (*intercept, weights.clone()),
    };
    Equation {
        id: id.to_owned(),
        intercept,
        weights,
        activation: Activation::Identity,
        expression: None,
        parameters: Vec::new(),
    }
}

/// The named equilibrium actions of a resolved nonlinear world.
///
/// # Errors
///
/// An initial-state change, which an equilibrium does not have.
pub fn equilibrium_actions(
    model: &EquilibriumModel,
    resolved: &ResolvedCyclicWorld,
) -> Result<Vec<EquilibriumAction>, CyclicInterventionRefusal> {
    let family = resolved.family;
    resolved
        .actions
        .iter()
        .map(|action| {
            let Some(equation) = model.equations.get(action.node()) else {
                return Err(specification_refusal(
                    family,
                    Some(resolved.world),
                    "an action names a position outside the model",
                ));
            };
            let mechanism = equation.id.clone();
            match action {
                CyclicAction::HardClamp { level, .. } => {
                    Ok(EquilibriumAction::Clamp { mechanism, value: *level })
                }
                CyclicAction::ReplaceMechanism { replacement, .. } => {
                    let replacement =
                        replacement_equation(&equation.id, model.order(), replacement);
                    Ok(EquilibriumAction::ReplaceMechanism { mechanism, replacement })
                }
                CyclicAction::ChangeMechanism { intercept_shift, .. } => {
                    Ok(EquilibriumAction::ChangeMechanism {
                        mechanism,
                        intercept: Some(equation.intercept + intercept_shift),
                        weights: None,
                        activation: None,
                    })
                }
                CyclicAction::AdditiveShock { delta, .. } => {
                    Ok(EquilibriumAction::AdditiveShock { mechanism, delta: *delta })
                }
                CyclicAction::InitialState { .. } => Err(refusal(
                    Some(resolved.world),
                    Some(family),
                    "initial_state_change",
                    CyclicRefusalKind::Capability {
                        reason: "an equilibrium has no initial-state law",
                    },
                    family.next_attempt("initial_state_change"),
                )),
            }
        })
        .collect()
}

/// Evaluate one world of a nonlinear equilibrium over disturbance draws.
///
/// `noise_draws` is row-major, one draw per row of the model's order, and is
/// reused unchanged in every world so worlds share their disturbances.
/// `initial` is the numerical initial guess; it is not a causal initial state.
/// A draw refused for lack of a contraction certificate (under
/// `policy.require_contraction`) is withheld mass. A draw that did not converge
/// or met a singular Jacobian is failed mass.
///
/// # Errors
///
/// A refusal of the request, malformed draws or policy, or a resource or
/// cancellation refusal.
pub fn evaluate_nonlinear_world(
    spec: &NonlinearCyclicSpecification,
    request: &CyclicWorldRequest,
    noise_draws: &[f64],
    initial: &[f64],
    policy: SolvePolicy,
    ctx: &ExecutionContext,
) -> Result<CyclicWorldResult, CyclicInterventionRefusal> {
    let family = CyclicFamily::NonlinearEquilibrium;
    let resolved = resolve_cyclic_world(family, &spec.variables, &[], request)?;
    let world = resolved.world;
    let n = spec.model.order();
    if noise_draws.is_empty() || !noise_draws.len().is_multiple_of(n) {
        return Err(specification_refusal(
            family,
            Some(world),
            "disturbance draws must be non-empty rows of the model's order",
        ));
    }
    let actions = equilibrium_actions(&spec.model, &resolved)?;
    let classify = |error: EquilibriumError| match error {
        EquilibriumError::Resource(message) => execution_refusal(family, world, message),
        other => specification_refusal(family, Some(world), other),
    };
    let plan = spec.model.compile_actions(&actions).map_err(classify)?;
    let blocks = plan.blocks().map_err(classify)?;
    let draw_count = noise_draws.len() / n;
    let (work, retained) = nonlinear_world_cost(n, draw_count, blocks.len(), &policy);
    if work.is_none_or(|work| work > NONLINEAR_WORLD_MAX_WORK) {
        return Err(execution_refusal(
            family,
            world,
            format_args!(
                "{}{}: cyclic_world.work_limit: {draw_count} draws over {n} variables exceed \
                 the world work limit of {NONLINEAR_WORLD_MAX_WORK}. Next attempt: fewer draws, \
                 a smaller iteration bound, or split the draws across requests",
                antecedent_core::reason_code::PREFIX,
                antecedent_core::reason_code!("execution_resource_limit"),
            ),
        ));
    }
    let lease = retained
        .ok_or_else(|| execution_refusal(family, world, "retained outcome size overflows u64"))
        .and_then(|bytes| {
            ctx.acquire_admission(0, bytes).map_err(|error| execution_refusal(family, world, error))
        })?;
    let mut outcomes = Vec::with_capacity(draw_count);
    let (mut failed, mut withheld, mut uncertified) = (0_usize, 0_usize, 0_usize);
    for draw in noise_draws.chunks_exact(n) {
        if ctx.cancellation.is_cancelled() {
            return Err(execution_refusal(family, world, "execution cancelled"));
        }
        let outcome = plan.solve(draw, initial, policy, ctx).map_err(classify)?;
        match &outcome {
            EquilibriumOutcome::Solved(solution) => {
                if !(solution.globally_unique && solution.existence_certified) {
                    uncertified += 1;
                }
            }
            EquilibriumOutcome::Failed(failure) if failure.failure == "uncertified" => {
                withheld += 1;
            }
            EquilibriumOutcome::Failed(_) => failed += 1,
        }
        outcomes.push(outcome);
    }
    let draws = outcomes.len();
    let certificate = if withheld > 0 || failed > 0 {
        WorldCertificate::NotCertified {
            gap: CertificateGap::NonlinearMassLost {
                failed_draws: failed,
                withheld_draws: withheld,
            },
            next_attempt: if withheld > 0 {
                BLOCK_CONTRACTION_ATTEMPT
            } else {
                RETAINED_FAILED_DRAWS_ATTEMPT
            },
        }
    } else if uncertified > 0 {
        WorldCertificate::NotCertified {
            gap: CertificateGap::NonlinearUniquenessUnproved { uncertified_draws: uncertified },
            next_attempt: BLOCK_CONTRACTION_ATTEMPT,
        }
    } else {
        WorldCertificate::Certified { basis: CertificateBasis::NonlinearBlockContraction { draws } }
    };
    Ok(CyclicWorldResult::from_parts(
        resolved,
        &spec.variables,
        request,
        (
            CyclicWorldOutcome::EquilibriumDraws { outcomes },
            certificate,
            WorldSolvedBy::NonlinearBlocks { blocks, policy },
            WorldMass::from_counts(draws, failed, withheld),
        ),
    )
    .retained_by(lease))
}

/// Upper bound on the solver work of a nonlinear world and the bytes its outcomes retain.
/// Each is `None` when it overflows. Every draw retains its state, a copy of the block
/// partition and the evidence record.
pub(super) fn nonlinear_world_cost(
    order: usize,
    draws: usize,
    blocks: usize,
    policy: &SolvePolicy,
) -> (Option<u64>, Option<u64>) {
    let order = order as u64;
    let draws = draws as u64;
    let iterations = (policy.max_total_iterations as u64)
        .min((blocks as u64).saturating_mul(policy.max_iterations as u64));
    let work = draws
        .checked_mul(iterations)
        .and_then(|w| w.checked_mul(order.max(1)))
        .and_then(|w| w.checked_mul(order.max(1)));
    let per_outcome = order
        .checked_mul(16)
        .and_then(|b| b.checked_add((blocks as u64).checked_mul(24)?))
        .and_then(|b| b.checked_add(OUTCOME_OVERHEAD_BYTES as u64));
    let retained = per_outcome
        .and_then(|b| b.checked_mul(draws))
        .and_then(|b| b.checked_add(RESULT_OVERHEAD_BYTES as u64));
    (work, retained)
}
