// SPDX-License-Identifier: MIT OR Apache-2.0

//! Typed interventions on cyclic models and the worlds they define.
//!
//! A caller names variables by [`VariableId`] and passes core
//! [`Intervention`]s. [`resolve_cyclic_world`] admits each one on its
//! [`CyclicFamily`] and binds it to a model position. The result is a list of
//! [`CyclicAction`]s with distinct scientific identities:
//!
//! - an untimed [`Intervention::Set`] is a hard clamp, which deletes the
//!   clamped equation;
//! - an untimed [`Intervention::Shift`] is an additive shock on the structural
//!   disturbance of an equilibrium;
//! - an [`Intervention::Soft`] of family [`SOFT_CONSTANT`] or
//!   [`SOFT_LINEAR_MECHANISM`] replaces the deterministic mechanism and keeps
//!   the disturbance;
//! - an [`Intervention::Soft`] of family [`SOFT_ADDITIVE_SHIFT`] changes the
//!   mechanism's intercept (on the continuous-time family, a drift input);
//! - an [`InterventionOperation::InitialState`] replaces one initial-state law
//!   of a continuous-time system.
//!
//! None of them is rewritten into another. A numerically equal answer from two
//! actions does not make them the same world. Every refusal is a
//! [`CyclicInterventionRefusal`] that names the next attempt.
//!
//! The evaluators return a [`CyclicWorldResult`] keyed by
//! [`ScientificWorldId`]. It keeps the request that defined the world, the
//! resolved actions, the outcome, the [`WorldCertificate`], how the world was
//! solved ([`WorldSolvedBy`]), and the failed and withheld mass
//! ([`WorldMass`]). A world whose free block has no unique solution is a
//! result with failed mass one, not a dropped world. A certificate is a
//! uniqueness claim for the declared model; physical stability stays
//! unclaimed, and a specified-model evaluation is not identification.
//!
//! - Linear equilibrium: the intervened map is rechecked with
//!   [`revalidate_solvability`]. When `ρ(|K|) < 1` holds for the intervened
//!   map, the world compiles through an admitted [`CyclicComponent`] and is
//!   certified. Otherwise the free block is decided by a direct solve and the
//!   world is reported uncertified, or withheld under
//!   [`CertificateDemand::Required`].
//! - Nonlinear equilibrium: each disturbance draw is solved through the named
//!   [`EquilibriumAction`]s. A draw refused for lack of a contraction
//!   certificate is withheld mass; a draw that did not converge is failed mass.
//!   Neither is renormalized away.
//! - Continuous time: the trajectory is integrated from the intervened initial
//!   state with clamps held from the initial time. Adaptive integration
//!   controls local error only, so the trajectory is not certified.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

use antecedent_core::intervention::{
    AdmitError, CYCLIC_SUPPORTED_OPERATION_ATTEMPT, CapabilityRefusal, CyclicFamily,
    InterventionOperation,
};
use antecedent_core::refusal::NextAttempt;
use antecedent_core::{
    ExecutionContext, Intervention, InterventionError, MechanismOverride, ScientificWorldId,
    TemporalPolicy, Value, VariableId, WorldAction,
};
use antecedent_kernels::gauss_jordan_solve;

use crate::continuous_cyclic::{OdePolicy, OdeSystem, OdeTrajectory, TimedClamp};
use crate::cyclic_actions::EquilibriumAction;
use crate::cyclic_component::CyclicComponent;
use crate::intervention_request::{
    InterventionRequest, InterventionRequestError, hard_clamp_of, revalidate_solvability,
};
use crate::linear_solution::{
    HardClamp, LinearStructuralMap, NONZERO_DIAGONAL_NEXT_ATTEMPT, PhysicalStability,
    SolvabilityRefusal, assess, free_indices,
};
use crate::nonlinear_equilibrium::{
    Activation, Clamp, Equation, EquilibriumError, EquilibriumModel, EquilibriumOutcome,
    SolvePolicy,
};
use crate::scc_compile::{
    CompileBudget, GaussianDisturbance, LinearSupportRoute, SccCompileError,
    compile_routed_gaussian_equilibrium,
};

/// Soft family that replaces the mechanism by a constant. Parameters: `[value]`.
pub const SOFT_CONSTANT: &str = "constant";

/// Soft family that adds to the mechanism's intercept. Parameters: `[delta]`.
pub const SOFT_ADDITIVE_SHIFT: &str = "additive_shift";

/// Soft family that replaces the mechanism by an affine one.
/// Parameters: `[intercept, w_0, …, w_{n-1}]` in the specification's variable order.
pub const SOFT_LINEAR_MECHANISM: &str = "linear_mechanism";

/// Bind the variable in the specification.
pub const BIND_VARIABLE_ATTEMPT: NextAttempt = NextAttempt::new(
    "bind_variable",
    "bind the variable in the specification's variable order, or remove the action",
);

/// Declare at most one clamp per variable.
pub const ONE_ACTION_PER_VARIABLE_ATTEMPT: NextAttempt = NextAttempt::new(
    "one_action_per_variable",
    "declare at most one clamp per variable, or one mechanism change plus one additive shock",
);

/// Assign a finite float level.
pub const FINITE_FLOAT_LEVEL_ATTEMPT: NextAttempt = NextAttempt::new(
    "finite_float_level",
    "assign a finite float level; other value types are not coerced",
);

/// Use a soft family this layer resolves.
pub const SUPPORTED_SOFT_FAMILY_ATTEMPT: NextAttempt = NextAttempt::new(
    "supported_soft_family",
    "use soft family `constant` [value], `additive_shift` [delta], or `linear_mechanism` \
     [intercept, one weight per variable]",
);

/// Supply the packed parameters the soft family declares.
pub const SOFT_PARAMETER_SHAPE_ATTEMPT: NextAttempt = NextAttempt::new(
    "soft_parameter_shape",
    "pack `constant` and `additive_shift` as one parameter, and `linear_mechanism` as the \
     intercept followed by one weight per variable",
);

/// A self-coefficient belongs to another family or to a normalization.
pub const NONZERO_DIAGONAL_ATTEMPT: NextAttempt =
    NextAttempt::new("normalize_structural_diagonal", NONZERO_DIAGONAL_NEXT_ATTEMPT);

/// Name a declared initial-state law.
pub const DECLARED_INITIAL_LAW_ATTEMPT: NextAttempt = NextAttempt::new(
    "declared_initial_law",
    "name one of the specification's declared initial-state laws",
);

/// Repair a malformed intervention.
pub const WELL_FORMED_INTERVENTION_ATTEMPT: NextAttempt = NextAttempt::new(
    "well_formed_intervention",
    "repair the intervention so it validates: finite levels, a valid temporal policy, and an \
     untimed sequence",
);

/// Repair the specification.
pub const WELL_FORMED_SPECIFICATION_ATTEMPT: NextAttempt = NextAttempt::new(
    "well_formed_specification",
    "repair the specification's shape, coefficients, noise law, or numerical policy and \
     evaluate again",
);

/// Give each world its own id.
pub const DISTINCT_WORLD_ATTEMPT: NextAttempt =
    NextAttempt::new("distinct_world_ids", "give each requested world its own ScientificWorldId");

/// Name a component and certifying evidence.
pub const CERTIFYING_EVIDENCE_ATTEMPT: NextAttempt = NextAttempt::new(
    "certifying_evidence",
    "name a component id, a source id, and certifying evidence other than the source",
);

/// Resources or cancellation stopped the evaluation.
pub const EXECUTION_ATTEMPT: NextAttempt = NextAttempt::new(
    "execution_resources",
    "raise the admitted resources or clear the cancellation, then evaluate the world again",
);

/// A singular clamped loop needs a selection or a dynamic law.
pub const SELECTION_OR_DYNAMIC_LAW_ATTEMPT: NextAttempt = NextAttempt::new(
    "selection_or_dynamic_law",
    "declare an equilibrium selection rule or a dynamic law for the clamped loop, or intervene \
     with a mechanism change that keeps the free block regular",
);

/// A numerically singular block needs a rank-revealing factor.
pub const RANK_REVEALING_FACTOR_ATTEMPT: NextAttempt = NextAttempt::new(
    "rank_revealing_factor",
    "factor the free block with a rank-revealing decomposition before treating it as singular, \
     or rescale its coefficients",
);

/// A non-finite free block needs rescaled coefficients.
pub const RESCALE_COEFFICIENTS_ATTEMPT: NextAttempt = NextAttempt::new(
    "rescale_structural_coefficients",
    "rescale the structural coefficients so the free block and its solution are finite",
);

/// The family bound was not shown for a linear world.
pub const QUERY_SPECIFIC_CERTIFICATE_ATTEMPT: NextAttempt = NextAttempt::new(
    "query_specific_free_block",
    "the bound ρ(|K|) < 1 was not shown for the intervened map; certify this world's free set \
     by an exact factorization, or report the world uncertified",
);

/// A nonlinear root is not shown unique.
pub const BLOCK_CONTRACTION_ATTEMPT: NextAttempt = NextAttempt::new(
    "block_contraction_certificate",
    "bound each free block's derivative norm below one, or verify an affine bijection, before \
     treating a numerical root as the unique equilibrium",
);

/// Some nonlinear draws failed.
pub const RETAINED_FAILED_DRAWS_ATTEMPT: NextAttempt = NextAttempt::new(
    "retained_failed_draws",
    "report the failed draws as failed mass, or revise the solve policy and the numerical \
     initial guess and evaluate again",
);

/// A trajectory needs a validated integrator before it is certified.
pub const VALIDATED_INTEGRATION_ATTEMPT: NextAttempt = NextAttempt::new(
    "validated_integration",
    "bound the global integration error with a validated integrator before certifying the \
     trajectory",
);

/// Integration did not reach the horizon.
pub const INTEGRATION_FAILURE_ATTEMPT: NextAttempt = NextAttempt::new(
    "revised_integration_policy",
    "raise the step budget or revise the tolerances, or shorten the horizon, and integrate again",
);

// ---------------------------------------------------------------------------
// Refusals
// ---------------------------------------------------------------------------

/// What was refused.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum CyclicRefusalKind {
    /// The family does not perform the operation.
    Capability {
        /// Why the operation was not rewritten into one the family performs.
        reason: &'static str,
    },
    /// The intervention or its policy is malformed.
    Malformed {
        /// Validation message.
        reason: String,
    },
    /// The variable is not bound in the specification.
    UnboundVariable {
        /// Variable that was named.
        variable: VariableId,
    },
    /// Two actions on one variable conflict.
    ConflictingActions {
        /// Variable named twice.
        variable: VariableId,
    },
    /// A level is not a float.
    NonFloatLevel,
    /// A level or a soft parameter is NaN or infinite.
    NonFiniteLevel,
    /// The soft family is not one this layer resolves.
    UnsupportedSoftFamily {
        /// Family that was named.
        family: Arc<str>,
    },
    /// The soft parameters do not have the family's shape.
    SoftParameterShape {
        /// Soft family.
        family: &'static str,
        /// Parameter count the family takes.
        expected: usize,
        /// Parameter count supplied.
        found: usize,
    },
    /// A replacement gives the variable a structural self-coefficient.
    NonzeroDiagonal {
        /// Variable whose replacement reads itself.
        variable: VariableId,
    },
    /// The initial-state law is not declared.
    UnknownInitialLaw {
        /// Law that was named.
        law_name: Arc<str>,
    },
    /// The same variable id is bound twice, or no variable is bound.
    VariableBinding {
        /// What is inconsistent.
        message: &'static str,
    },
    /// Two requests share one world id.
    DuplicateWorld,
    /// The specification or the numerical policy is malformed.
    Specification {
        /// What is inconsistent.
        message: String,
    },
    /// The component gate refused the specification's evidence.
    Evidence {
        /// Component refusal.
        message: String,
    },
    /// Resources or cancellation stopped the evaluation.
    Execution {
        /// Resource refusal.
        message: String,
    },
}

impl fmt::Display for CyclicRefusalKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Capability { reason } => formatter.write_str(reason),
            Self::Malformed { reason } => write!(formatter, "malformed intervention: {reason}"),
            Self::UnboundVariable { variable } => {
                write!(formatter, "variable {} is not bound", variable.raw())
            }
            Self::ConflictingActions { variable } => {
                write!(formatter, "conflicting actions on variable {}", variable.raw())
            }
            Self::NonFloatLevel => formatter.write_str("the level is not a float"),
            Self::NonFiniteLevel => formatter.write_str("the level is not finite"),
            Self::UnsupportedSoftFamily { family } => {
                write!(formatter, "soft family `{family}` is not resolved on a cyclic model")
            }
            Self::SoftParameterShape { family, expected, found } => write!(
                formatter,
                "soft family `{family}` takes {expected} parameters and {found} were supplied"
            ),
            Self::NonzeroDiagonal { variable } => write!(
                formatter,
                "the replacement gives variable {} a structural self-coefficient",
                variable.raw()
            ),
            Self::UnknownInitialLaw { law_name } => {
                write!(formatter, "initial-state law `{law_name}` is not declared")
            }
            Self::VariableBinding { message } => formatter.write_str(message),
            Self::DuplicateWorld => formatter.write_str("the world id is requested twice"),
            Self::Specification { message } => write!(formatter, "specification: {message}"),
            Self::Evidence { message } => write!(formatter, "component evidence: {message}"),
            Self::Execution { message } => write!(formatter, "execution: {message}"),
        }
    }
}

/// A typed refusal. It always names the next attempt.
#[derive(Clone, Debug, PartialEq)]
pub struct CyclicInterventionRefusal {
    /// World being resolved, when the refusal belongs to one.
    pub world: Option<ScientificWorldId>,
    /// Family the refusal was decided on, when one was named.
    pub family: Option<CyclicFamily>,
    /// Operation or step that was refused.
    pub operation: &'static str,
    /// What was refused.
    pub kind: CyclicRefusalKind,
    /// What to declare instead.
    pub next_attempt: NextAttempt,
}

impl fmt::Display for CyclicInterventionRefusal {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("cyclic ")?;
        match self.family {
            Some(family) => write!(formatter, "{family}")?,
            None => formatter.write_str("model")?,
        }
        if let Some(world) = self.world {
            write!(formatter, " world {world}")?;
        }
        write!(
            formatter,
            ": {} refused: {}; next attempt: {}",
            self.operation,
            self.kind,
            self.next_attempt.sentence()
        )
    }
}

impl std::error::Error for CyclicInterventionRefusal {}

fn refusal(
    world: Option<ScientificWorldId>,
    family: Option<CyclicFamily>,
    operation: &'static str,
    kind: CyclicRefusalKind,
    next_attempt: NextAttempt,
) -> CyclicInterventionRefusal {
    CyclicInterventionRefusal { world, family, operation, kind, next_attempt }
}

fn specification_refusal(
    family: CyclicFamily,
    world: Option<ScientificWorldId>,
    message: impl fmt::Display,
) -> CyclicInterventionRefusal {
    refusal(
        world,
        Some(family),
        "specification",
        CyclicRefusalKind::Specification { message: message.to_string() },
        WELL_FORMED_SPECIFICATION_ATTEMPT,
    )
}

fn execution_refusal(
    family: CyclicFamily,
    world: ScientificWorldId,
    message: impl fmt::Display,
) -> CyclicInterventionRefusal {
    refusal(
        Some(world),
        Some(family),
        "evaluate",
        CyclicRefusalKind::Execution { message: message.to_string() },
        EXECUTION_ATTEMPT,
    )
}

// ---------------------------------------------------------------------------
// Variables and requests
// ---------------------------------------------------------------------------

/// Variable ids in model order: position `k` is model node `k`.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct CyclicVariables {
    ids: Vec<VariableId>,
}

impl CyclicVariables {
    /// Bind `ids` to model positions `0..ids.len()`.
    ///
    /// # Errors
    ///
    /// An empty list, or an id bound twice.
    pub fn try_new(ids: Vec<VariableId>) -> Result<Self, CyclicInterventionRefusal> {
        let binding = |message| {
            refusal(
                None,
                None,
                "bind_variables",
                CyclicRefusalKind::VariableBinding { message },
                BIND_VARIABLE_ATTEMPT,
            )
        };
        if ids.is_empty() {
            return Err(binding("a cyclic model binds at least one variable"));
        }
        let mut seen = std::collections::BTreeSet::new();
        if ids.iter().any(|id| !seen.insert(*id)) {
            return Err(binding("a variable id is bound to two model positions"));
        }
        Ok(Self { ids })
    }

    /// Ids in model order.
    #[must_use]
    pub fn ids(&self) -> &[VariableId] {
        &self.ids
    }

    /// Number of bound variables.
    #[must_use]
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    /// Always `false`: a binding holds at least one variable.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// Model position of `variable`.
    #[must_use]
    pub fn position(&self, variable: VariableId) -> Option<usize> {
        self.ids.iter().position(|id| *id == variable)
    }
}

/// The interventions that define one world.
///
/// Interventions are simultaneous. `temporal` applies to each of them and is
/// refused by the equilibrium families; it is kept so a refusal names the
/// timing that was asked for. Initial-state changes are
/// [`InterventionOperation::InitialState`] values.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CyclicWorldRequest {
    world: ScientificWorldId,
    interventions: Vec<Intervention>,
    temporal: Option<TemporalPolicy>,
    initial_states: Vec<InterventionOperation>,
}

impl CyclicWorldRequest {
    /// World `world` under the simultaneous untimed `interventions`.
    #[must_use]
    pub fn new(
        world: ScientificWorldId,
        interventions: impl IntoIterator<Item = Intervention>,
    ) -> Self {
        Self {
            world,
            interventions: interventions.into_iter().collect(),
            temporal: None,
            initial_states: Vec::new(),
        }
    }

    /// World without interventions: the model as specified.
    #[must_use]
    pub fn observational(world: ScientificWorldId) -> Self {
        Self::new(world, [])
    }

    /// World from a target's [`WorldAction`]. A schedule keeps its policy.
    #[must_use]
    pub fn from_action(world: ScientificWorldId, action: &WorldAction) -> Self {
        if let Some((intervention, policy)) = action.schedule_parts() {
            return Self::new(world, [intervention.clone()]).timed(policy.clone());
        }
        Self::new(world, action.at_decision_interventions().unwrap_or_default().to_vec())
    }

    /// Time every intervention of this world by `policy`.
    #[must_use]
    pub fn timed(mut self, policy: TemporalPolicy) -> Self {
        self.temporal = Some(policy);
        self
    }

    /// Replace the initial-state law `law_name` by the level `level`.
    ///
    /// # Errors
    ///
    /// An empty law name or a non-finite level.
    pub fn with_initial_state(
        mut self,
        law_name: impl Into<Arc<str>>,
        level: f64,
    ) -> Result<Self, CyclicInterventionRefusal> {
        let operation = InterventionOperation::initial_state(law_name, Value::f64(level))
            .map_err(|error| malformed(self.world, None, "initial_state_change", &error))?;
        self.initial_states.push(operation);
        Ok(self)
    }

    /// World id.
    #[must_use]
    pub const fn world(&self) -> ScientificWorldId {
        self.world
    }

    /// Simultaneous interventions, as declared.
    #[must_use]
    pub fn interventions(&self) -> &[Intervention] {
        &self.interventions
    }

    /// Timing applied to every intervention, when one was declared.
    #[must_use]
    pub const fn temporal(&self) -> Option<&TemporalPolicy> {
        self.temporal.as_ref()
    }

    /// Initial-state changes, as declared.
    #[must_use]
    pub fn initial_states(&self) -> &[InterventionOperation] {
        &self.initial_states
    }
}

// ---------------------------------------------------------------------------
// Resolved actions
// ---------------------------------------------------------------------------

/// Replacement deterministic mechanism. The disturbance is kept.
#[derive(Clone, Debug, PartialEq)]
pub enum MechanismReplacement {
    /// `x_i := value + u_i`.
    Constant {
        /// Replacement constant.
        value: f64,
    },
    /// `x_i := intercept + Σ_j weights[j] x_j + u_i`.
    Linear {
        /// Replacement intercept.
        intercept: f64,
        /// Weights in the specification's variable order.
        weights: Vec<f64>,
    },
}

/// One action bound to a model position.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum CyclicAction {
    /// Delete the equation and hold the variable at `level`.
    HardClamp {
        /// Clamped variable.
        variable: VariableId,
        /// Model position.
        node: usize,
        /// Assigned level.
        level: f64,
    },
    /// Replace the deterministic mechanism and keep its disturbance.
    ReplaceMechanism {
        /// Target variable.
        variable: VariableId,
        /// Model position.
        node: usize,
        /// Replacement.
        replacement: MechanismReplacement,
    },
    /// Add `intercept_shift` to the mechanism's intercept.
    ChangeMechanism {
        /// Target variable.
        variable: VariableId,
        /// Model position.
        node: usize,
        /// Added to the intercept.
        intercept_shift: f64,
    },
    /// Add `delta` to the structural disturbance.
    AdditiveShock {
        /// Target variable.
        variable: VariableId,
        /// Model position.
        node: usize,
        /// Deterministic perturbation of the disturbance.
        delta: f64,
    },
    /// Replace the initial-state law of a continuous-time state.
    InitialState {
        /// State variable.
        variable: VariableId,
        /// Model position.
        node: usize,
        /// Declared law that was replaced.
        law_name: Arc<str>,
        /// Replacement initial level.
        level: f64,
    },
}

const CLAMP_BIT: u8 = 1;
const MECHANISM_BIT: u8 = 2;
const SHOCK_BIT: u8 = 4;
const INITIAL_BIT: u8 = 8;

impl CyclicAction {
    /// Stable name of the action kind.
    #[must_use]
    pub const fn kind(&self) -> &'static str {
        match self {
            Self::HardClamp { .. } => "hard_clamp",
            Self::ReplaceMechanism { .. } => "replace_mechanism",
            Self::ChangeMechanism { .. } => "change_mechanism",
            Self::AdditiveShock { .. } => "additive_shock",
            Self::InitialState { .. } => "initial_state_change",
        }
    }

    /// Target variable.
    #[must_use]
    pub const fn variable(&self) -> VariableId {
        match self {
            Self::HardClamp { variable, .. }
            | Self::ReplaceMechanism { variable, .. }
            | Self::ChangeMechanism { variable, .. }
            | Self::AdditiveShock { variable, .. }
            | Self::InitialState { variable, .. } => *variable,
        }
    }

    /// Model position.
    #[must_use]
    pub const fn node(&self) -> usize {
        match self {
            Self::HardClamp { node, .. }
            | Self::ReplaceMechanism { node, .. }
            | Self::ChangeMechanism { node, .. }
            | Self::AdditiveShock { node, .. }
            | Self::InitialState { node, .. } => *node,
        }
    }

    const fn bit(&self) -> u8 {
        match self {
            Self::HardClamp { .. } => CLAMP_BIT,
            Self::ReplaceMechanism { .. } | Self::ChangeMechanism { .. } => MECHANISM_BIT,
            Self::AdditiveShock { .. } => SHOCK_BIT,
            Self::InitialState { .. } => INITIAL_BIT,
        }
    }
}

/// Actions of one world, admitted and bound, sorted by model position.
#[derive(Clone, Debug, PartialEq)]
pub struct ResolvedCyclicWorld {
    world: ScientificWorldId,
    family: CyclicFamily,
    actions: Vec<CyclicAction>,
}

impl ResolvedCyclicWorld {
    /// World id.
    #[must_use]
    pub const fn world(&self) -> ScientificWorldId {
        self.world
    }

    /// Family the actions were admitted on.
    #[must_use]
    pub const fn family(&self) -> CyclicFamily {
        self.family
    }

    /// Bound actions.
    #[must_use]
    pub fn actions(&self) -> &[CyclicAction] {
        &self.actions
    }
}

/// Admit every action of `request` on `family` and bind it to `variables`.
///
/// `initial_laws` names the initial-state law of each model position; it is
/// read only by the continuous-time family.
///
/// # Errors
///
/// A [`CyclicInterventionRefusal`] naming the next attempt: an operation the
/// family does not perform, a malformed intervention, an unbound variable, a
/// non-float or non-finite level, an unsupported soft family or parameter
/// shape, a self-coefficient on the linear family, an undeclared initial-state
/// law, or conflicting actions on one variable.
pub fn resolve_cyclic_world(
    family: CyclicFamily,
    variables: &CyclicVariables,
    initial_laws: &[Arc<str>],
    request: &CyclicWorldRequest,
) -> Result<ResolvedCyclicWorld, CyclicInterventionRefusal> {
    let world = request.world;
    let mut actions =
        Vec::with_capacity(request.interventions.len() + request.initial_states.len());
    for intervention in &request.interventions {
        let operation = family.admit(intervention, request.temporal.as_ref()).map_err(|error| {
            admit_refusal(world, family, intervention_name(intervention), error)
        })?;
        actions.push(action_of(world, family, variables, &operation)?);
    }
    for operation in &request.initial_states {
        family
            .admit_operation(operation)
            .map_err(|refused| capability_refusal(world, family, refused))?;
        actions.push(initial_state_action(world, family, variables, initial_laws, operation)?);
    }
    let mut seen: BTreeMap<usize, u8> = BTreeMap::new();
    for action in &actions {
        let bit = action.bit();
        let held = seen.entry(action.node()).or_insert(0);
        if *held & bit != 0 || *held & CLAMP_BIT != 0 || (bit == CLAMP_BIT && *held != 0) {
            return Err(refusal(
                Some(world),
                Some(family),
                action.kind(),
                CyclicRefusalKind::ConflictingActions { variable: action.variable() },
                ONE_ACTION_PER_VARIABLE_ATTEMPT,
            ));
        }
        *held |= bit;
    }
    actions.sort_by_key(|action| (action.node(), action.bit()));
    Ok(ResolvedCyclicWorld { world, family, actions })
}

fn intervention_name(intervention: &Intervention) -> &'static str {
    match intervention {
        Intervention::Set { .. } => "set",
        Intervention::Shift { .. } => "shift",
        Intervention::Stochastic { .. } => "stochastic",
        Intervention::Soft { .. } => "soft",
        Intervention::Transform { .. } => "transform",
        Intervention::Assign { .. } => "assign",
        Intervention::Sequence(_) => "sequence",
        _ => "intervention",
    }
}

fn capability_refusal(
    world: ScientificWorldId,
    family: CyclicFamily,
    refused: CapabilityRefusal,
) -> CyclicInterventionRefusal {
    refusal(
        Some(world),
        Some(family),
        refused.operation(),
        CyclicRefusalKind::Capability { reason: refused.reason() },
        family.next_attempt(refused.operation()),
    )
}

fn malformed(
    world: ScientificWorldId,
    family: Option<CyclicFamily>,
    operation: &'static str,
    error: &InterventionError,
) -> CyclicInterventionRefusal {
    let next_attempt = if matches!(error, InterventionError::NonFiniteLevel) {
        FINITE_FLOAT_LEVEL_ATTEMPT
    } else {
        WELL_FORMED_INTERVENTION_ATTEMPT
    };
    refusal(
        Some(world),
        family,
        operation,
        CyclicRefusalKind::Malformed { reason: error.to_string() },
        next_attempt,
    )
}

fn admit_refusal(
    world: ScientificWorldId,
    family: CyclicFamily,
    operation: &'static str,
    error: AdmitError,
) -> CyclicInterventionRefusal {
    match error {
        AdmitError::Refused(refused) => capability_refusal(world, family, refused),
        AdmitError::Invalid(invalid) => malformed(world, Some(family), operation, &invalid),
        other => refusal(
            Some(world),
            Some(family),
            operation,
            CyclicRefusalKind::Malformed { reason: other.to_string() },
            WELL_FORMED_INTERVENTION_ATTEMPT,
        ),
    }
}

fn bind(
    world: ScientificWorldId,
    family: CyclicFamily,
    operation: &'static str,
    variables: &CyclicVariables,
    variable: VariableId,
) -> Result<usize, CyclicInterventionRefusal> {
    variables.position(variable).ok_or_else(|| {
        refusal(
            Some(world),
            Some(family),
            operation,
            CyclicRefusalKind::UnboundVariable { variable },
            BIND_VARIABLE_ATTEMPT,
        )
    })
}

fn float_level(
    world: ScientificWorldId,
    family: CyclicFamily,
    operation: &'static str,
    value: &Value,
) -> Result<f64, CyclicInterventionRefusal> {
    let kind = match value {
        Value::Float64(level) if level.is_finite() => {
            // `-0.0` and `+0.0` are one level.
            return Ok(if level.to_bits() == (-0.0_f64).to_bits() { 0.0 } else { *level });
        }
        Value::Float64(_) => CyclicRefusalKind::NonFiniteLevel,
        _ => CyclicRefusalKind::NonFloatLevel,
    };
    Err(refusal(Some(world), Some(family), operation, kind, FINITE_FLOAT_LEVEL_ATTEMPT))
}

fn action_of(
    world: ScientificWorldId,
    family: CyclicFamily,
    variables: &CyclicVariables,
    operation: &InterventionOperation,
) -> Result<CyclicAction, CyclicInterventionRefusal> {
    match operation {
        InterventionOperation::Set { variable, value } => {
            let node = bind(world, family, "set", variables, *variable)?;
            let level = float_level(world, family, "set", value)?;
            Ok(CyclicAction::HardClamp { variable: *variable, node, level })
        }
        InterventionOperation::Shift { variable, delta } => {
            let node = bind(world, family, "shift", variables, *variable)?;
            let delta = float_level(world, family, "shift", delta)?;
            Ok(CyclicAction::AdditiveShock { variable: *variable, node, delta })
        }
        InterventionOperation::Soft { variable, mechanism } => {
            let node = bind(world, family, "soft", variables, *variable)?;
            soft_action(world, family, variables.len(), *variable, node, mechanism)
        }
        other => Err(refusal(
            Some(world),
            Some(family),
            other.name(),
            CyclicRefusalKind::Capability {
                reason: "the operation was admitted but has no cyclic action kind",
            },
            CYCLIC_SUPPORTED_OPERATION_ATTEMPT,
        )),
    }
}

fn soft_action(
    world: ScientificWorldId,
    family: CyclicFamily,
    order: usize,
    variable: VariableId,
    node: usize,
    mechanism: &MechanismOverride,
) -> Result<CyclicAction, CyclicInterventionRefusal> {
    let parameters: &[f64] = &mechanism.parameters;
    let refuse =
        |kind, next_attempt| refusal(Some(world), Some(family), "soft", kind, next_attempt);
    if parameters.iter().any(|parameter| !parameter.is_finite()) {
        return Err(refuse(CyclicRefusalKind::NonFiniteLevel, FINITE_FLOAT_LEVEL_ATTEMPT));
    }
    let shape = |family_name: &'static str, expected: usize| {
        if parameters.len() == expected {
            Ok(())
        } else {
            Err(refuse(
                CyclicRefusalKind::SoftParameterShape {
                    family: family_name,
                    expected,
                    found: parameters.len(),
                },
                SOFT_PARAMETER_SHAPE_ATTEMPT,
            ))
        }
    };
    match &*mechanism.family_id {
        SOFT_CONSTANT => {
            shape(SOFT_CONSTANT, 1)?;
            Ok(CyclicAction::ReplaceMechanism {
                variable,
                node,
                replacement: MechanismReplacement::Constant { value: parameters[0] },
            })
        }
        SOFT_ADDITIVE_SHIFT => {
            shape(SOFT_ADDITIVE_SHIFT, 1)?;
            Ok(CyclicAction::ChangeMechanism { variable, node, intercept_shift: parameters[0] })
        }
        SOFT_LINEAR_MECHANISM => {
            shape(SOFT_LINEAR_MECHANISM, order + 1)?;
            let weights = parameters[1..].to_vec();
            if family == CyclicFamily::LinearEquilibrium && weights[node].abs() > 0.0 {
                return Err(refuse(
                    CyclicRefusalKind::NonzeroDiagonal { variable },
                    NONZERO_DIAGONAL_ATTEMPT,
                ));
            }
            Ok(CyclicAction::ReplaceMechanism {
                variable,
                node,
                replacement: MechanismReplacement::Linear { intercept: parameters[0], weights },
            })
        }
        _ => Err(refuse(
            CyclicRefusalKind::UnsupportedSoftFamily { family: Arc::clone(&mechanism.family_id) },
            SUPPORTED_SOFT_FAMILY_ATTEMPT,
        )),
    }
}

fn initial_state_action(
    world: ScientificWorldId,
    family: CyclicFamily,
    variables: &CyclicVariables,
    initial_laws: &[Arc<str>],
    operation: &InterventionOperation,
) -> Result<CyclicAction, CyclicInterventionRefusal> {
    let InterventionOperation::InitialState { law_name, level } = operation else {
        return Err(refusal(
            Some(world),
            Some(family),
            operation.name(),
            CyclicRefusalKind::Capability {
                reason: "an initial-state slot holds another operation",
            },
            CYCLIC_SUPPORTED_OPERATION_ATTEMPT,
        ));
    };
    let Some(node) = initial_laws.iter().position(|law| law == law_name) else {
        return Err(refusal(
            Some(world),
            Some(family),
            "initial_state_change",
            CyclicRefusalKind::UnknownInitialLaw { law_name: Arc::clone(law_name) },
            DECLARED_INITIAL_LAW_ATTEMPT,
        ));
    };
    let Some(&variable) = variables.ids().get(node) else {
        return Err(specification_refusal(
            family,
            Some(world),
            "an initial-state law has no bound variable",
        ));
    };
    let level = float_level(world, family, "initial_state_change", level)?;
    Ok(CyclicAction::InitialState { variable, node, law_name: Arc::clone(law_name), level })
}

// ---------------------------------------------------------------------------
// World results
// ---------------------------------------------------------------------------

/// Why a world has no solution to return.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum WorldFailureCause {
    /// The intervened free block is singular, numerically singular, or non-finite.
    Solvability(SolvabilityRefusal),
    /// Integration did not reach the horizon.
    Integration {
        /// Integrator diagnostic.
        detail: String,
    },
}

/// A world with no solution, and what to try instead.
#[derive(Clone, Debug, PartialEq)]
pub struct WorldFailure {
    /// Why no solution was returned.
    pub cause: WorldFailureCause,
    /// What to declare instead.
    pub next_attempt: NextAttempt,
}

impl WorldFailure {
    fn solvability(refused: SolvabilityRefusal) -> Self {
        let next_attempt = match &refused {
            SolvabilityRefusal::Singular { .. } => SELECTION_OR_DYNAMIC_LAW_ATTEMPT,
            SolvabilityRefusal::NumericallySingular { .. } => RANK_REVEALING_FACTOR_ATTEMPT,
            SolvabilityRefusal::NonFinite => RESCALE_COEFFICIENTS_ATTEMPT,
        };
        Self { cause: WorldFailureCause::Solvability(refused), next_attempt }
    }
}

/// What the world evaluated to.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum CyclicWorldOutcome {
    /// Gaussian equilibrium law, in the specification's variable order.
    /// A clamped variable has its level as mean and a zero row and column.
    GaussianLaw {
        /// Equilibrium mean.
        mean: Vec<f64>,
        /// Row-major equilibrium covariance.
        covariance: Vec<f64>,
        /// Structural residual of the compile, when the compile reported one.
        residual_inf: Option<f64>,
    },
    /// One outcome per disturbance draw, in draw order. Failed draws stay in place.
    EquilibriumDraws {
        /// Draw-aligned outcomes.
        outcomes: Vec<EquilibriumOutcome>,
    },
    /// Trajectory at the specification's observation times.
    Trajectory {
        /// Integrated trajectory.
        trajectory: OdeTrajectory,
    },
    /// No solution was returned.
    NoSolution {
        /// Why, and what to try instead.
        failure: WorldFailure,
    },
    /// A solution was not computed because its certificate was required and not shown.
    Withheld,
}

/// What the certificate rests on.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum CertificateBasis {
    /// `ρ(|K|) < 1` for the intervened linear map. Every hard-clamp free block of
    /// that map is uniquely solvable.
    LinearFamilyContraction {
        /// Upper bound on `ρ(|K|)`.
        rho_abs_upper: f64,
    },
    /// Every free block of every draw contracts or is a verified affine bijection.
    NonlinearBlockContraction {
        /// Draws covered.
        draws: usize,
    },
}

/// Why a world is not certified.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum CertificateGap {
    /// The bound `ρ(|K|) < 1` was not shown for the intervened map.
    LinearFamilyBoundInsufficient {
        /// Upper bound that did not land below one.
        rho_abs_upper: f64,
    },
    /// Some draws solved numerically without a uniqueness certificate.
    NonlinearUniquenessUnproved {
        /// Draws without the certificate.
        uncertified_draws: usize,
    },
    /// Some draws failed or were withheld.
    NonlinearMassLost {
        /// Draws that did not converge or were singular.
        failed_draws: usize,
        /// Draws refused for lack of a contraction certificate.
        withheld_draws: usize,
    },
    /// The trajectory rests on locally error-controlled integration.
    NumericalIntegrationOnly,
    /// The world has no solution.
    NoSolution,
}

/// Uniqueness certificate of one world.
#[derive(Clone, Debug, PartialEq)]
pub enum WorldCertificate {
    /// Certified for the declared model.
    Certified {
        /// What the certificate rests on.
        basis: CertificateBasis,
    },
    /// Not certified.
    NotCertified {
        /// Why.
        gap: CertificateGap,
        /// What would earn the certificate, or what to try instead.
        next_attempt: NextAttempt,
    },
}

impl WorldCertificate {
    /// Whether the world is certified.
    #[must_use]
    pub const fn is_certified(&self) -> bool {
        matches!(self, Self::Certified { .. })
    }

    /// Gap, when not certified.
    #[must_use]
    pub const fn gap(&self) -> Option<&CertificateGap> {
        match self {
            Self::Certified { .. } => None,
            Self::NotCertified { gap, .. } => Some(gap),
        }
    }

    /// Next attempt, when not certified.
    #[must_use]
    pub const fn next_attempt(&self) -> Option<NextAttempt> {
        match self {
            Self::Certified { .. } => None,
            Self::NotCertified { next_attempt, .. } => Some(*next_attempt),
        }
    }

    fn gap_name(&self) -> Option<&'static str> {
        self.gap().map(|gap| match gap {
            CertificateGap::LinearFamilyBoundInsufficient { .. } => {
                "linear_family_bound_insufficient"
            }
            CertificateGap::NonlinearUniquenessUnproved { .. } => "nonlinear_uniqueness_unproved",
            CertificateGap::NonlinearMassLost { .. } => "nonlinear_mass_lost",
            CertificateGap::NumericalIntegrationOnly => "numerical_integration_only",
            CertificateGap::NoSolution => "no_solution",
        })
    }
}

/// How the world was solved.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum WorldSolvedBy {
    /// The free support is acyclic and was evaluated in this topological order.
    AcyclicOrder {
        /// Free positions, sources first.
        order: Vec<usize>,
    },
    /// The DAG algorithm refused a cycle; strongly connected blocks were factored.
    StronglyConnectedBlocks {
        /// The DAG algorithm's refusal.
        dag_refusal: String,
        /// Free blocks, sources first.
        blocks: Vec<Vec<usize>>,
    },
    /// The free block was solved directly without a family certificate.
    DirectFreeBlockSolve {
        /// Free positions.
        free: Vec<usize>,
    },
    /// Nonlinear blocks solved under an explicit policy.
    NonlinearBlocks {
        /// Post-action blocks, sources first.
        blocks: Vec<Vec<usize>>,
        /// Numerical policy.
        policy: SolvePolicy,
    },
    /// Adaptive Dormand–Prince integration.
    AdaptiveIntegration {
        /// Accepted steps.
        accepted_steps: usize,
        /// Rejected steps.
        rejected_steps: usize,
        /// Numerical policy.
        policy: OdePolicy,
    },
    /// Nothing was solved.
    NotSolved,
}

impl WorldSolvedBy {
    /// Stable name.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::AcyclicOrder { .. } => "acyclic_order",
            Self::StronglyConnectedBlocks { .. } => "strongly_connected_blocks",
            Self::DirectFreeBlockSolve { .. } => "direct_free_block_solve",
            Self::NonlinearBlocks { .. } => "nonlinear_blocks",
            Self::AdaptiveIntegration { .. } => "adaptive_integration",
            Self::NotSolved => "not_solved",
        }
    }
}

/// Share of the world's disturbance law, or of its draws, by status.
///
/// `failed` had no solution. `withheld` was not answered because a required
/// certificate was not shown. Neither is renormalized into the solved share.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WorldMass {
    failed: f64,
    withheld: f64,
}

impl WorldMass {
    /// Every unit of mass was solved.
    pub const SOLVED: Self = Self { failed: 0.0, withheld: 0.0 };
    /// No mass was solved.
    pub const FAILED: Self = Self { failed: 1.0, withheld: 0.0 };
    /// Every unit of mass was withheld.
    pub const WITHHELD: Self = Self { failed: 0.0, withheld: 1.0 };

    fn from_counts(total: usize, failed: usize, withheld: usize) -> Self {
        let total = total.max(1) as f64;
        Self { failed: failed as f64 / total, withheld: withheld as f64 / total }
    }

    /// Failed share.
    #[must_use]
    pub const fn failed(self) -> f64 {
        self.failed
    }

    /// Withheld share.
    #[must_use]
    pub const fn withheld(self) -> f64 {
        self.withheld
    }

    /// Solved share.
    #[must_use]
    pub fn solved(self) -> f64 {
        1.0 - self.failed - self.withheld
    }
}

/// One evaluated cyclic world.
#[derive(Clone, Debug, PartialEq)]
pub struct CyclicWorldResult {
    world: ScientificWorldId,
    family: CyclicFamily,
    variables: Vec<VariableId>,
    request: CyclicWorldRequest,
    actions: Vec<CyclicAction>,
    outcome: CyclicWorldOutcome,
    certificate: WorldCertificate,
    solved_by: WorldSolvedBy,
    mass: WorldMass,
    physical_stability: PhysicalStability,
}

impl CyclicWorldResult {
    fn from_parts(
        resolved: ResolvedCyclicWorld,
        variables: &CyclicVariables,
        request: &CyclicWorldRequest,
        (outcome, certificate, solved_by, mass): WorldParts,
    ) -> Self {
        Self {
            world: resolved.world,
            family: resolved.family,
            variables: variables.ids().to_vec(),
            request: request.clone(),
            actions: resolved.actions,
            outcome,
            certificate,
            solved_by,
            mass,
            physical_stability: PhysicalStability::Unclaimed,
        }
    }

    /// World id.
    #[must_use]
    pub const fn world(&self) -> ScientificWorldId {
        self.world
    }

    /// Family the world was evaluated on.
    #[must_use]
    pub const fn family(&self) -> CyclicFamily {
        self.family
    }

    /// Variables in outcome order.
    #[must_use]
    pub fn variables(&self) -> &[VariableId] {
        &self.variables
    }

    /// The request that defined the world.
    #[must_use]
    pub const fn request(&self) -> &CyclicWorldRequest {
        &self.request
    }

    /// Resolved actions.
    #[must_use]
    pub fn actions(&self) -> &[CyclicAction] {
        &self.actions
    }

    /// Outcome.
    #[must_use]
    pub const fn outcome(&self) -> &CyclicWorldOutcome {
        &self.outcome
    }

    /// Uniqueness certificate.
    #[must_use]
    pub const fn certificate(&self) -> &WorldCertificate {
        &self.certificate
    }

    /// How the world was solved.
    #[must_use]
    pub const fn solved_by(&self) -> &WorldSolvedBy {
        &self.solved_by
    }

    /// Failed and withheld mass.
    #[must_use]
    pub const fn mass(&self) -> WorldMass {
        self.mass
    }

    /// Physical stability. Never claimed here.
    #[must_use]
    pub const fn physical_stability(&self) -> PhysicalStability {
        self.physical_stability
    }

    /// Mean of `variable`, when the outcome has a complete mean.
    #[must_use]
    pub fn mean_of(&self, variable: VariableId) -> Option<f64> {
        let position = self.variables.iter().position(|id| *id == variable)?;
        self.complete_mean().map(|mean| mean[position])
    }

    /// Mean in variable order: the Gaussian mean, the average over draws when
    /// every draw solved, or the final state of a trajectory. `None` when some
    /// mass failed or was withheld: the solved share is not renormalized.
    fn complete_mean(&self) -> Option<Vec<f64>> {
        match &self.outcome {
            CyclicWorldOutcome::GaussianLaw { mean, .. } => Some(mean.clone()),
            CyclicWorldOutcome::EquilibriumDraws { outcomes } => {
                let mut sum = vec![0.0; self.variables.len()];
                for outcome in outcomes {
                    let EquilibriumOutcome::Solved(solution) = outcome else {
                        return None;
                    };
                    for (total, value) in sum.iter_mut().zip(&solution.values) {
                        *total += value;
                    }
                }
                if outcomes.is_empty() {
                    return None;
                }
                let count = outcomes.len() as f64;
                Some(sum.into_iter().map(|total| total / count).collect())
            }
            CyclicWorldOutcome::Trajectory { trajectory } => trajectory.states.last().cloned(),
            CyclicWorldOutcome::NoSolution { .. } | CyclicWorldOutcome::Withheld => None,
        }
    }

    /// Variable-keyed view for result types outside this crate.
    #[must_use]
    pub fn summary(&self) -> CyclicWorldSummary {
        let means =
            self.complete_mean().map(|mean| self.variables.iter().copied().zip(mean).collect());
        let variances = match &self.outcome {
            CyclicWorldOutcome::GaussianLaw { covariance, .. } => {
                let n = self.variables.len();
                Some(
                    self.variables
                        .iter()
                        .enumerate()
                        .map(|(index, id)| (*id, covariance[index * n + index]))
                        .collect(),
                )
            }
            _ => None,
        };
        CyclicWorldSummary {
            world: self.world,
            family: self.family,
            means,
            variances,
            certified: self.certificate.is_certified(),
            certificate_gap: self.certificate.gap_name(),
            next_attempt: self.certificate.next_attempt(),
            solved_by: self.solved_by.as_str(),
            failed_mass: self.mass.failed(),
            withheld_mass: self.mass.withheld(),
        }
    }
}

/// Variable-keyed summary of one world.
#[derive(Clone, Debug, PartialEq)]
pub struct CyclicWorldSummary {
    /// World id.
    pub world: ScientificWorldId,
    /// Family.
    pub family: CyclicFamily,
    /// Complete means, when the outcome has them.
    pub means: Option<Vec<(VariableId, f64)>>,
    /// Gaussian variances, when the outcome is a Gaussian law.
    pub variances: Option<Vec<(VariableId, f64)>>,
    /// Whether the world is certified.
    pub certified: bool,
    /// Stable name of the certificate gap.
    pub certificate_gap: Option<&'static str>,
    /// Next attempt when not certified.
    pub next_attempt: Option<NextAttempt>,
    /// Stable name of how the world was solved.
    pub solved_by: &'static str,
    /// Failed share.
    pub failed_mass: f64,
    /// Withheld share.
    pub withheld_mass: f64,
}

/// World results keyed by [`ScientificWorldId`].
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CyclicWorldResults {
    worlds: BTreeMap<ScientificWorldId, CyclicWorldResult>,
}

impl CyclicWorldResults {
    /// Key `results` by world.
    ///
    /// # Errors
    ///
    /// Two results for one world.
    pub fn try_from_results(
        results: impl IntoIterator<Item = CyclicWorldResult>,
    ) -> Result<Self, CyclicInterventionRefusal> {
        let mut worlds = BTreeMap::new();
        for result in results {
            let (world, family) = (result.world, result.family);
            if worlds.insert(world, result).is_some() {
                return Err(duplicate_world(world, family));
            }
        }
        Ok(Self { worlds })
    }

    /// Result of `world`.
    #[must_use]
    pub fn get(&self, world: ScientificWorldId) -> Option<&CyclicWorldResult> {
        self.worlds.get(&world)
    }

    /// Results in world order.
    #[allow(
        clippy::iter_without_into_iter,
        reason = "the public surface keeps one explicit world-order iterator"
    )]
    pub fn iter(
        &self,
    ) -> std::collections::btree_map::Iter<'_, ScientificWorldId, CyclicWorldResult> {
        self.worlds.iter()
    }

    /// Number of worlds.
    #[must_use]
    pub fn len(&self) -> usize {
        self.worlds.len()
    }

    /// Whether no world is held.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.worlds.is_empty()
    }

    /// Summaries in world order.
    #[must_use]
    pub fn summaries(&self) -> Vec<CyclicWorldSummary> {
        self.worlds.values().map(CyclicWorldResult::summary).collect()
    }
}

fn duplicate_world(world: ScientificWorldId, family: CyclicFamily) -> CyclicInterventionRefusal {
    refusal(
        Some(world),
        Some(family),
        "evaluate",
        CyclicRefusalKind::DuplicateWorld,
        DISTINCT_WORLD_ATTEMPT,
    )
}

// ---------------------------------------------------------------------------
// Linear equilibrium
// ---------------------------------------------------------------------------

/// Whether an uncertified linear world is answered.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum CertificateDemand {
    /// Withhold a world whose certificate was not shown.
    Required,
    /// Answer a uniquely solvable world and mark it uncertified.
    ReportUncertified,
}

/// Specified linear Gaussian equilibrium with bound variables and the
/// evidence an admitted component cites.
#[derive(Clone, Debug, PartialEq)]
pub struct LinearCyclicSpecification {
    variables: CyclicVariables,
    model: LinearStructuralMap,
    disturbance: GaussianDisturbance,
    component_id: String,
    source_id: String,
    certifying_evidence: Vec<String>,
}

impl LinearCyclicSpecification {
    /// Bind `model` and `disturbance` to `variables`.
    ///
    /// # Errors
    ///
    /// Dimensions that disagree, or a component id, source id, or evidence list
    /// the component gate would refuse.
    pub fn try_new(
        variables: CyclicVariables,
        model: LinearStructuralMap,
        disturbance: GaussianDisturbance,
        component_id: &str,
        source_id: &str,
        certifying_evidence: &[&str],
    ) -> Result<Self, CyclicInterventionRefusal> {
        let family = CyclicFamily::LinearEquilibrium;
        if model.dimension() != variables.len() || disturbance.dimension() != variables.len() {
            return Err(specification_refusal(
                family,
                None,
                "the map, the disturbance, and the variables have different orders",
            ));
        }
        let evidence_refused = component_id.is_empty()
            || source_id.is_empty()
            || certifying_evidence.is_empty()
            || certifying_evidence.iter().any(|cited| cited.is_empty() || *cited == source_id);
        if evidence_refused {
            return Err(refusal(
                None,
                Some(family),
                "specification",
                CyclicRefusalKind::Evidence {
                    message: "the component needs an id, a source, and certifying evidence other \
                              than the source"
                        .to_owned(),
                },
                CERTIFYING_EVIDENCE_ATTEMPT,
            ));
        }
        Ok(Self {
            variables,
            model,
            disturbance,
            component_id: component_id.to_owned(),
            source_id: source_id.to_owned(),
            certifying_evidence: certifying_evidence
                .iter()
                .map(|cited| (*cited).to_owned())
                .collect(),
        })
    }

    /// Bound variables.
    #[must_use]
    pub const fn variables(&self) -> &CyclicVariables {
        &self.variables
    }

    /// Structural map.
    #[must_use]
    pub const fn model(&self) -> &LinearStructuralMap {
        &self.model
    }

    /// Disturbance law.
    #[must_use]
    pub const fn disturbance(&self) -> &GaussianDisturbance {
        &self.disturbance
    }
}

/// Intervened map, disturbance and clamp of one linear world.
struct LinearWorldSystem {
    model: LinearStructuralMap,
    disturbance: GaussianDisturbance,
    clamp: InterventionRequest,
}

fn linear_world_system(
    spec: &LinearCyclicSpecification,
    resolved: &ResolvedCyclicWorld,
) -> Result<LinearWorldSystem, CyclicInterventionRefusal> {
    let family = CyclicFamily::LinearEquilibrium;
    let world = Some(resolved.world);
    let n = spec.model.dimension();
    let mut coefficients = spec.model.coefficients().to_vec();
    let mut intercept = spec.model.intercept().to_vec();
    let mut mean = spec.disturbance.mean().to_vec();
    let (mut nodes, mut levels) = (Vec::new(), Vec::new());
    for action in &resolved.actions {
        match action {
            CyclicAction::HardClamp { node, level, .. } => {
                nodes.push(*node);
                levels.push(*level);
            }
            CyclicAction::ReplaceMechanism { node, replacement, .. } => {
                let row = &mut coefficients[node * n..(node + 1) * n];
                match replacement {
                    MechanismReplacement::Constant { value } => {
                        row.fill(0.0);
                        intercept[*node] = *value;
                    }
                    MechanismReplacement::Linear { intercept: value, weights } => {
                        row.copy_from_slice(weights);
                        intercept[*node] = *value;
                    }
                }
            }
            CyclicAction::ChangeMechanism { node, intercept_shift, .. } => {
                intercept[*node] += intercept_shift;
            }
            CyclicAction::AdditiveShock { node, delta, .. } => mean[*node] += delta,
            CyclicAction::InitialState { .. } => {
                return Err(refusal(
                    world,
                    Some(family),
                    "initial_state_change",
                    CyclicRefusalKind::Capability {
                        reason: "an equilibrium has no initial-state law",
                    },
                    family.next_attempt("initial_state_change"),
                ));
            }
        }
    }
    let model = LinearStructuralMap::try_new(coefficients, intercept)
        .map_err(|error| specification_refusal(family, world, error))?;
    let disturbance = GaussianDisturbance::try_new(mean, spec.disturbance.covariance().to_vec())
        .map_err(|error| specification_refusal(family, world, error))?;
    let clamp = InterventionRequest::hard_clamp(n, &nodes, &levels)
        .map_err(|error| specification_refusal(family, world, error))?;
    Ok(LinearWorldSystem { model, disturbance, clamp })
}

/// Evaluate one world of a specified linear Gaussian equilibrium.
///
/// The intervened free block is rechecked first; a singular, numerically
/// singular, or non-finite block is a result with failed mass one. When
/// `ρ(|K|) < 1` holds for the intervened map, the world is compiled through an
/// admitted component and certified. Otherwise the world is withheld under
/// [`CertificateDemand::Required`], or solved directly and reported
/// uncertified under [`CertificateDemand::ReportUncertified`].
///
/// # Errors
///
/// A refusal of the request ([`resolve_cyclic_world`]), a malformed intervened
/// map or disturbance, a component-gate refusal, or a budget refusal.
pub fn evaluate_linear_world(
    spec: &LinearCyclicSpecification,
    request: &CyclicWorldRequest,
    demand: CertificateDemand,
    budget: CompileBudget,
) -> Result<CyclicWorldResult, CyclicInterventionRefusal> {
    let family = CyclicFamily::LinearEquilibrium;
    let resolved = resolve_cyclic_world(family, &spec.variables, &[], request)?;
    let world = resolved.world;
    let system = linear_world_system(spec, &resolved)?;
    let (clamp, values) = hard_clamp_of(&system.clamp)
        .map_err(|error| specification_refusal(family, Some(world), error))?;
    let assessment = assess(&system.model, clamp)
        .map_err(|error| specification_refusal(family, Some(world), error))?;
    let parts = match revalidate_solvability(&system.model, &system.clamp) {
        Err(InterventionRequestError::NotSolvable { refusal: refused }) => {
            failed_parts(WorldFailure::solvability(refused))
        }
        Err(error) => return Err(specification_refusal(family, Some(world), error)),
        Ok(_) if assessment.family.is_admitted() => {
            let evidence: Vec<&str> = spec.certifying_evidence.iter().map(String::as_str).collect();
            let component = CyclicComponent::try_new(
                &spec.component_id,
                &spec.source_id,
                &evidence,
                &system.model,
                &assessment.family,
            )
            .map_err(|error| {
                refusal(
                    Some(world),
                    Some(family),
                    "specification",
                    CyclicRefusalKind::Evidence { message: error.to_string() },
                    CERTIFYING_EVIDENCE_ATTEMPT,
                )
            })?;
            certified_linear_parts(world, &component, &system, (clamp, &values), budget)?
        }
        Ok(_) => uncertified_linear_parts(
            &system,
            (clamp.indices(), &values),
            assessment.family.rho_abs_upper(),
            demand,
        ),
    };
    Ok(CyclicWorldResult::from_parts(resolved, &spec.variables, request, parts))
}

/// Outcome, certificate, solver, and mass of one evaluated world.
type WorldParts = (CyclicWorldOutcome, WorldCertificate, WorldSolvedBy, WorldMass);

fn failed_parts(failure: WorldFailure) -> WorldParts {
    let next_attempt = failure.next_attempt;
    (
        CyclicWorldOutcome::NoSolution { failure },
        WorldCertificate::NotCertified { gap: CertificateGap::NoSolution, next_attempt },
        WorldSolvedBy::NotSolved,
        WorldMass::FAILED,
    )
}

/// The world under an admitted `ρ(|K|) < 1` certificate, compiled through `component`.
fn certified_linear_parts(
    world: ScientificWorldId,
    component: &CyclicComponent,
    system: &LinearWorldSystem,
    (clamp, values): (&HardClamp, &[f64]),
    budget: CompileBudget,
) -> Result<WorldParts, CyclicInterventionRefusal> {
    let family = CyclicFamily::LinearEquilibrium;
    let routed = match compile_routed_gaussian_equilibrium(
        component,
        &system.model,
        &system.disturbance,
        clamp,
        values,
        budget,
    ) {
        Ok(routed) => routed,
        Err(SccCompileError::NumericallySingularBlock { members }) => {
            return Ok(failed_parts(WorldFailure::solvability(
                SolvabilityRefusal::NumericallySingular { members },
            )));
        }
        Err(SccCompileError::NonFiniteEquilibrium) => {
            return Ok(failed_parts(WorldFailure::solvability(SolvabilityRefusal::NonFinite)));
        }
        Err(error @ SccCompileError::Budget { .. }) => {
            return Err(execution_refusal(family, world, error));
        }
        Err(error) => return Err(specification_refusal(family, Some(world), error)),
    };
    let solved_by = match routed.route() {
        LinearSupportRoute::Acyclic { order } => {
            WorldSolvedBy::AcyclicOrder { order: order.clone() }
        }
        LinearSupportRoute::Cyclic { next_attempt, blocks } => {
            WorldSolvedBy::StronglyConnectedBlocks {
                dag_refusal: next_attempt.clone(),
                blocks: blocks.clone(),
            }
        }
    };
    let equilibrium = routed.equilibrium();
    Ok((
        CyclicWorldOutcome::GaussianLaw {
            mean: equilibrium.mean().to_vec(),
            covariance: equilibrium.covariance().to_vec(),
            residual_inf: Some(equilibrium.residual_inf()),
        },
        WorldCertificate::Certified {
            basis: CertificateBasis::LinearFamilyContraction {
                rho_abs_upper: component.family_rho_abs_upper(),
            },
        },
        solved_by,
        WorldMass::SOLVED,
    ))
}

/// The world when `ρ(|K|) < 1` was not shown: withheld, or solved directly and
/// marked uncertified.
fn uncertified_linear_parts(
    system: &LinearWorldSystem,
    (clamp_nodes, values): (&[usize], &[f64]),
    rho_abs_upper: f64,
    demand: CertificateDemand,
) -> WorldParts {
    let certificate = WorldCertificate::NotCertified {
        gap: CertificateGap::LinearFamilyBoundInsufficient { rho_abs_upper },
        next_attempt: QUERY_SPECIFIC_CERTIFICATE_ATTEMPT,
    };
    if demand == CertificateDemand::Required {
        return (
            CyclicWorldOutcome::Withheld,
            certificate,
            WorldSolvedBy::NotSolved,
            WorldMass::WITHHELD,
        );
    }
    match direct_free_block_moments(&system.model, clamp_nodes, values, &system.disturbance) {
        Ok((mean, covariance, free)) => (
            CyclicWorldOutcome::GaussianLaw { mean, covariance, residual_inf: None },
            certificate,
            WorldSolvedBy::DirectFreeBlockSolve { free },
            WorldMass::SOLVED,
        ),
        Err(refused) => failed_parts(WorldFailure::solvability(refused)),
    }
}

/// Evaluate every request against one linear specification.
///
/// # Errors
///
/// Two requests with one world id, or the first refusal of
/// [`evaluate_linear_world`].
pub fn evaluate_linear_worlds(
    spec: &LinearCyclicSpecification,
    requests: &[CyclicWorldRequest],
    demand: CertificateDemand,
    budget: CompileBudget,
) -> Result<CyclicWorldResults, CyclicInterventionRefusal> {
    let mut seen = std::collections::BTreeSet::new();
    for request in requests {
        if !seen.insert(request.world) {
            return Err(duplicate_world(request.world, CyclicFamily::LinearEquilibrium));
        }
    }
    let results = requests
        .iter()
        .map(|request| evaluate_linear_world(spec, request, demand, budget))
        .collect::<Result<Vec<_>, _>>()?;
    CyclicWorldResults::try_from_results(results)
}

/// The free block's mean, its covariance and the free node indices.
type FreeBlockMoments = (Vec<f64>, Vec<f64>, Vec<usize>);

/// Gaussian moments of the free block without a family certificate.
///
/// `(I - K_FF) μ_F = c_F + K_FH a + μ_ε,F` and
/// `Σ_X,FF = (I - K_FF)^{-1} Σ_ε,FF (I - K_FF)^{-T}`, both by solves. The
/// inverse is not formed. `clamp_nodes` is ascending and `levels` follows it.
fn direct_free_block_moments(
    model: &LinearStructuralMap,
    clamp_nodes: &[usize],
    levels: &[f64],
    disturbance: &GaussianDisturbance,
) -> Result<FreeBlockMoments, SolvabilityRefusal> {
    let n = model.dimension();
    let (k, c) = (model.coefficients(), model.intercept());
    let (mu, sigma) = (disturbance.mean(), disturbance.covariance());
    let free = free_indices(n, clamp_nodes);
    let m = free.len();
    let mut mean = vec![0.0; n];
    for (&node, &level) in clamp_nodes.iter().zip(levels) {
        mean[node] = level;
    }
    let mut covariance = vec![0.0; n * n];
    if m == 0 {
        return Ok((mean, covariance, free));
    }
    let mut factor = vec![0.0; m * m];
    let mut rhs = vec![0.0; m * (m + 1)];
    for (p, &row) in free.iter().enumerate() {
        let mut value = c[row] + mu[row];
        for (&node, &level) in clamp_nodes.iter().zip(levels) {
            value += k[row * n + node] * level;
        }
        rhs[p] = value;
        for (q, &col) in free.iter().enumerate() {
            let identity = if p == q { 1.0 } else { 0.0 };
            factor[p * m + q] = identity - k[row * n + col];
            // Right-hand side q + 1 is column q of Σ_FF.
            rhs[(q + 1) * m + p] = sigma[row * n + col];
        }
    }
    let mut first = factor.clone();
    if !gauss_jordan_solve(&mut first, &mut rhs, m, m + 1) {
        return Err(SolvabilityRefusal::NumericallySingular { members: free });
    }
    // Y = A^{-1} Σ_FF has Y[p][q] at rhs[(q + 1) * m + p]. Column q of Yᵀ is row q of Y.
    let mut transposed = vec![0.0; m * m];
    for q in 0..m {
        for p in 0..m {
            transposed[q * m + p] = rhs[(p + 1) * m + q];
        }
    }
    if !gauss_jordan_solve(&mut factor, &mut transposed, m, m) {
        return Err(SolvabilityRefusal::NumericallySingular { members: free });
    }
    // Z = A^{-1} Yᵀ = A^{-1} Σ_FF A^{-T}, with Z[p][q] at transposed[q * m + p].
    for (p, &row) in free.iter().enumerate() {
        mean[row] = rhs[p];
        for (q, &col) in free.iter().enumerate() {
            covariance[row * n + col] = 0.5 * (transposed[q * m + p] + transposed[p * m + q]);
        }
    }
    if mean.iter().chain(&covariance).any(|value| !value.is_finite()) {
        return Err(SolvabilityRefusal::NonFinite);
    }
    Ok((mean, covariance, free))
}

// ---------------------------------------------------------------------------
// Nonlinear equilibrium
// ---------------------------------------------------------------------------

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

fn replacement_equation(id: &str, order: usize, replacement: &MechanismReplacement) -> Equation {
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
    if noise_draws.is_empty() || noise_draws.len() % n != 0 {
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
    let mut outcomes = Vec::with_capacity(noise_draws.len() / n);
    let (mut failed, mut withheld, mut uncertified) = (0_usize, 0_usize, 0_usize);
    for draw in noise_draws.chunks_exact(n) {
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
    ))
}

// ---------------------------------------------------------------------------
// Continuous time
// ---------------------------------------------------------------------------

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
            ))
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
