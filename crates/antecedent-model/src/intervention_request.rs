// SPDX-License-Identifier: MIT OR Apache-2.0

//! Intervention requests on a linear free block.
//!
//! A hard clamp is the only request this block executes. It deletes the clamped
//! equations and leaves
//! [`FREE_BLOCK_EQUATION`](crate::linear_solution::FREE_BLOCK_EQUATION). The
//! disturbance on the free variables stays marginal.
//!
//! A mechanism shift, a pulse, a sustained action, and an initial-state change
//! each bind their own target and identity. None of them is rewritten as a clamp.
//! A pulse is not a sustained action, including when the sustained window is a
//! single step. An initial-state change is not a hard clamp.
//!
//! After a hard clamp, [`revalidate_solvability`] decides the free block again.
//! A singular or non-finite block is a refusal and is not marked unidentified.
//! Numerically singular and ill-conditioned stay different statuses. An
//! ill-conditioned block is not reported unique. A unique free block is not a
//! stability certificate: physical stability stays unclaimed, including when
//! the free update has spectral radius above one. A refused request is not
//! compiled and is not replaced by another clamp.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

use thiserror::Error;

use crate::cyclic_component::CyclicComponent;
use crate::linear_solution::{
    HardClamp, IdentificationStatus, LinearSolutionError, LinearStructuralMap, PhysicalStability,
    QueryBlock, SolvabilityRefusal, SolverConvergence, revalidate_free_block,
};
use crate::scc_compile::{
    CompileBudget, CompiledEquilibrium, GaussianDisturbance, SccCompileError,
    compile_gaussian_equilibrium,
};

const MECHANISM_SHIFT_NEXT: &str = concat!(
    "replace the structural coefficient on its edge; ",
    "a mechanism shift is not a hard clamp",
);
const PULSE_NEXT: &str = concat!(
    "apply the action at its single offset; ",
    "a pulse is not a sustained action and is not a hard clamp",
);
const SUSTAINED_NEXT: &str = concat!(
    "hold the action on its window; ",
    "a sustained action is not a pulse and is not a hard clamp",
);
const INITIAL_STATE_NEXT: &str =
    concat!("replace the initial-state law; ", "an initial-state change is not a hard clamp",);

/// Which intervention was requested.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum InterventionKind {
    /// Hold free-block variables at assigned levels.
    HardClamp,
    /// Replace one structural coefficient.
    MechanismShift,
    /// Apply an action at one time offset.
    Pulse,
    /// Hold an action on an inclusive step window.
    SustainedAction,
    /// Replace an initial-state law.
    InitialStateChange,
}

impl InterventionKind {
    /// Stable name of this kind.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::HardClamp => "hard_clamp",
            Self::MechanismShift => "mechanism_shift",
            Self::Pulse => "pulse",
            Self::SustainedAction => "sustained_action",
            Self::InitialStateChange => "initial_state_change",
        }
    }

    /// Why this kind is not executed as a hard clamp.
    ///
    /// [`Self::HardClamp`] has no refusal text.
    #[must_use]
    pub const fn clamp_refusal(self) -> Option<&'static str> {
        match self {
            Self::HardClamp => None,
            Self::MechanismShift => Some(MECHANISM_SHIFT_NEXT),
            Self::Pulse => Some(PULSE_NEXT),
            Self::SustainedAction => Some(SUSTAINED_NEXT),
            Self::InitialStateChange => Some(INITIAL_STATE_NEXT),
        }
    }
}

impl std::fmt::Display for InterventionKind {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Variables whose structural equations a hard clamp deletes.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct ClampTarget {
    /// Model order the indices were checked against.
    pub dimension: usize,
    /// Clamped indices, ascending.
    pub indices: Vec<usize>,
}

/// Assigned levels of a hard clamp, in [`ClampTarget`] order.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct HardClampIdentity {
    /// Model order.
    pub dimension: usize,
    /// Clamped indices, ascending.
    pub indices: Vec<usize>,
    /// Bit patterns of the assigned levels. `-0.0` is stored as `+0.0`.
    pub level_bits: Vec<u64>,
}

impl HardClampIdentity {
    /// Assigned levels in index order.
    #[must_use]
    pub fn levels(&self) -> Vec<f64> {
        self.level_bits.iter().copied().map(f64::from_bits).collect()
    }
}

/// Structural edge a mechanism shift replaces. This is not a clamped variable.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct MechanismTarget {
    /// Model order.
    pub dimension: usize,
    /// Variable the coefficient enters.
    pub into_node: usize,
    /// Variable the coefficient comes from.
    pub from_node: usize,
}

/// Replacement coefficient of a [`MechanismTarget`].
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct MechanismShiftIdentity {
    /// Model order.
    pub dimension: usize,
    /// Variable the coefficient enters.
    pub into_node: usize,
    /// Variable the coefficient comes from.
    pub from_node: usize,
    /// Bit pattern of the replacement coefficient.
    pub replacement_bits: u64,
}

/// One variable at one time offset.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct PulseTarget {
    /// Model order.
    pub dimension: usize,
    /// Variable the pulse acts on.
    pub node: usize,
    /// Time offset of the pulse.
    pub at: i32,
}

/// Level of a [`PulseTarget`]. The offset stays on the target.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct PulseIdentity {
    /// Model order.
    pub dimension: usize,
    /// Variable the pulse acts on.
    pub node: usize,
    /// Time offset of the pulse.
    pub at: i32,
    /// Bit pattern of the pulse level.
    pub level_bits: u64,
}

/// One variable on an inclusive step window.
///
/// A window with `from == until` is still a sustained action. It is not a pulse.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct SustainedTarget {
    /// Model order.
    pub dimension: usize,
    /// Variable the action holds.
    pub node: usize,
    /// First intervened step.
    pub from: i32,
    /// Last intervened step, inclusive.
    pub until: i32,
}

/// Level held on a [`SustainedTarget`].
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct SustainedIdentity {
    /// Model order.
    pub dimension: usize,
    /// Variable the action holds.
    pub node: usize,
    /// First intervened step.
    pub from: i32,
    /// Last intervened step, inclusive.
    pub until: i32,
    /// Bit pattern of the held level.
    pub level_bits: u64,
}

/// Initial-state law a change replaces. This is not an equilibrium variable.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct InitialStateTarget {
    /// Name of the law being replaced.
    pub law_name: String,
}

/// Replacement level of an [`InitialStateTarget`].
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct InitialStateIdentity {
    /// Name of the law being replaced.
    pub law_name: String,
    /// Bit pattern of the replacement level.
    pub level_bits: u64,
}

/// One request bound to the target and identity of its kind.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum InterventionRequest {
    /// Hard clamp of equilibrium variables.
    HardClamp(HardClampRequest),
    /// Replacement of one structural coefficient.
    MechanismShift(MechanismShiftRequest),
    /// One-offset action.
    Pulse(PulseRequest),
    /// Action held on a window.
    SustainedAction(SustainedActionRequest),
    /// Replacement of an initial-state law.
    InitialStateChange(InitialStateChangeRequest),
}

/// Hard clamp bound to its variables and its assigned levels.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HardClampRequest {
    /// Variables the clamp holds.
    pub target: ClampTarget,
    /// Levels that distinguish this clamp from another on the same variables.
    pub identity: HardClampIdentity,
    clamp: HardClamp,
}

/// Mechanism shift bound to its edge and its replacement coefficient.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MechanismShiftRequest {
    /// Edge the shift replaces.
    pub target: MechanismTarget,
    /// Replacement that distinguishes this shift from another on the same edge.
    pub identity: MechanismShiftIdentity,
}

/// Pulse bound to its offset and its level.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PulseRequest {
    /// Variable and offset.
    pub target: PulseTarget,
    /// Level that distinguishes this pulse from another at the same offset.
    pub identity: PulseIdentity,
}

/// Sustained action bound to its window and its level.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SustainedActionRequest {
    /// Variable and window.
    pub target: SustainedTarget,
    /// Level that distinguishes this action from another on the same window.
    pub identity: SustainedIdentity,
}

/// Initial-state change bound to its law and its replacement level.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct InitialStateChangeRequest {
    /// Law being replaced.
    pub target: InitialStateTarget,
    /// Level that distinguishes this change from another of the same law.
    pub identity: InitialStateIdentity,
}

/// Target and identity of one request, one variant per kind.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum InterventionBinding {
    /// Clamp variables and assigned levels.
    HardClamp {
        /// Variables held.
        target: ClampTarget,
        /// Assigned levels.
        identity: HardClampIdentity,
    },
    /// Mechanism edge and replacement coefficient.
    MechanismShift {
        /// Edge replaced.
        target: MechanismTarget,
        /// Replacement coefficient.
        identity: MechanismShiftIdentity,
    },
    /// Pulse offset and level.
    Pulse {
        /// Variable and offset.
        target: PulseTarget,
        /// Pulse level.
        identity: PulseIdentity,
    },
    /// Sustained window and level.
    SustainedAction {
        /// Variable and window.
        target: SustainedTarget,
        /// Held level.
        identity: SustainedIdentity,
    },
    /// Initial-state law and replacement level.
    InitialStateChange {
        /// Law replaced.
        target: InitialStateTarget,
        /// Replacement level.
        identity: InitialStateIdentity,
    },
}

impl InterventionRequest {
    /// Hard clamp. `values[k]` is the level for `nodes[k]` before sorting.
    ///
    /// Stored indices and levels follow the ascending index order of [`HardClamp`].
    ///
    /// # Errors
    ///
    /// A non-positive dimension, a non-finite level, a value count that does not
    /// match `nodes`, or a [`LinearSolutionError`] from the clamp.
    #[must_use = "a refused clamp is not a request"]
    pub fn hard_clamp(
        dimension: usize,
        nodes: &[usize],
        values: &[f64],
    ) -> Result<Self, InterventionRequestError> {
        Ok(Self::HardClamp(HardClampRequest::try_new(dimension, nodes, values)?))
    }

    /// Replace the coefficient of `from_node` into `into_node`.
    ///
    /// # Errors
    ///
    /// A non-positive dimension, an out-of-range node, a diagonal target, or a
    /// non-finite replacement.
    #[must_use = "a refused mechanism shift is not a request"]
    pub fn mechanism_shift(
        dimension: usize,
        into_node: usize,
        from_node: usize,
        replacement: f64,
    ) -> Result<Self, InterventionRequestError> {
        Ok(Self::MechanismShift(MechanismShiftRequest::try_new(
            dimension,
            into_node,
            from_node,
            replacement,
        )?))
    }

    /// Pulse `node` at offset `at`.
    ///
    /// # Errors
    ///
    /// A non-positive dimension, an out-of-range node, or a non-finite level.
    #[must_use = "a refused pulse is not a request"]
    pub fn pulse(
        dimension: usize,
        node: usize,
        at: i32,
        level: f64,
    ) -> Result<Self, InterventionRequestError> {
        Ok(Self::Pulse(PulseRequest::try_new(dimension, node, at, level)?))
    }

    /// Hold `node` from `from` through `until`, inclusive.
    ///
    /// `from == until` stays a sustained action.
    ///
    /// # Errors
    ///
    /// A non-positive dimension, an out-of-range node, a window that ends before
    /// it starts, or a non-finite level.
    #[must_use = "a refused sustained action is not a request"]
    pub fn sustained(
        dimension: usize,
        node: usize,
        from: i32,
        until: i32,
        level: f64,
    ) -> Result<Self, InterventionRequestError> {
        Ok(Self::SustainedAction(SustainedActionRequest::try_new(
            dimension, node, from, until, level,
        )?))
    }

    /// Replace the initial-state law `law_name`.
    ///
    /// # Errors
    ///
    /// An empty law name or a non-finite level.
    #[must_use = "a refused initial-state change is not a request"]
    pub fn initial_state(law_name: &str, level: f64) -> Result<Self, InterventionRequestError> {
        Ok(Self::InitialStateChange(InitialStateChangeRequest::try_new(law_name, level)?))
    }

    /// Kind of this request.
    #[must_use]
    pub const fn kind(&self) -> InterventionKind {
        match self {
            Self::HardClamp(_) => InterventionKind::HardClamp,
            Self::MechanismShift(_) => InterventionKind::MechanismShift,
            Self::Pulse(_) => InterventionKind::Pulse,
            Self::SustainedAction(_) => InterventionKind::SustainedAction,
            Self::InitialStateChange(_) => InterventionKind::InitialStateChange,
        }
    }

    /// Target and identity of this request.
    #[must_use]
    pub fn binding(&self) -> InterventionBinding {
        match self {
            Self::HardClamp(request) => InterventionBinding::HardClamp {
                target: request.target.clone(),
                identity: request.identity.clone(),
            },
            Self::MechanismShift(request) => InterventionBinding::MechanismShift {
                target: request.target.clone(),
                identity: request.identity.clone(),
            },
            Self::Pulse(request) => {
                InterventionBinding::Pulse { target: request.target, identity: request.identity }
            }
            Self::SustainedAction(request) => InterventionBinding::SustainedAction {
                target: request.target,
                identity: request.identity,
            },
            Self::InitialStateChange(request) => InterventionBinding::InitialStateChange {
                target: request.target.clone(),
                identity: request.identity.clone(),
            },
        }
    }
}

impl HardClampRequest {
    fn try_new(
        dimension: usize,
        nodes: &[usize],
        values: &[f64],
    ) -> Result<Self, InterventionRequestError> {
        require_dimension(dimension)?;
        if nodes.len() != values.len() {
            return Err(InterventionRequestError::ValueCount {
                nodes: nodes.len(),
                values: values.len(),
            });
        }
        let mut paired = Vec::with_capacity(nodes.len());
        for (&node, &value) in nodes.iter().zip(values) {
            paired.push((node, finite_level(value)?));
        }
        paired.sort_by_key(|(node, _)| *node);
        let indices: Vec<usize> = paired.iter().map(|(node, _)| *node).collect();
        let levels: Vec<f64> = paired.iter().map(|(_, level)| *level).collect();
        let clamp =
            HardClamp::try_new(dimension, &indices).map_err(InterventionRequestError::Solution)?;
        let level_bits = levels.iter().copied().map(f64::to_bits).collect();
        Ok(Self {
            target: ClampTarget { dimension, indices: indices.clone() },
            identity: HardClampIdentity { dimension, indices, level_bits },
            clamp,
        })
    }

    fn values(&self) -> Vec<f64> {
        self.identity.levels()
    }
}

impl MechanismShiftRequest {
    fn try_new(
        dimension: usize,
        into_node: usize,
        from_node: usize,
        replacement: f64,
    ) -> Result<Self, InterventionRequestError> {
        require_dimension(dimension)?;
        require_node(dimension, into_node)?;
        require_node(dimension, from_node)?;
        if into_node == from_node {
            return Err(InterventionRequestError::DiagonalMechanism { node: into_node });
        }
        let replacement = finite_level(replacement)?;
        Ok(Self {
            target: MechanismTarget { dimension, into_node, from_node },
            identity: MechanismShiftIdentity {
                dimension,
                into_node,
                from_node,
                replacement_bits: replacement.to_bits(),
            },
        })
    }
}

impl PulseRequest {
    fn try_new(
        dimension: usize,
        node: usize,
        at: i32,
        level: f64,
    ) -> Result<Self, InterventionRequestError> {
        require_dimension(dimension)?;
        require_node(dimension, node)?;
        let level = finite_level(level)?;
        Ok(Self {
            target: PulseTarget { dimension, node, at },
            identity: PulseIdentity { dimension, node, at, level_bits: level.to_bits() },
        })
    }
}

impl SustainedActionRequest {
    fn try_new(
        dimension: usize,
        node: usize,
        from: i32,
        until: i32,
        level: f64,
    ) -> Result<Self, InterventionRequestError> {
        require_dimension(dimension)?;
        require_node(dimension, node)?;
        if until < from {
            return Err(InterventionRequestError::InvertedWindow { from, until });
        }
        let level = finite_level(level)?;
        Ok(Self {
            target: SustainedTarget { dimension, node, from, until },
            identity: SustainedIdentity {
                dimension,
                node,
                from,
                until,
                level_bits: level.to_bits(),
            },
        })
    }
}

impl InitialStateChangeRequest {
    fn try_new(law_name: &str, level: f64) -> Result<Self, InterventionRequestError> {
        if law_name.is_empty() {
            return Err(InterventionRequestError::EmptyInitialLaw);
        }
        let level = finite_level(level)?;
        let identity =
            InitialStateIdentity { law_name: law_name.to_owned(), level_bits: level.to_bits() };
        Ok(Self { target: InitialStateTarget { law_name: law_name.to_owned() }, identity })
    }
}

/// Free block after a hard clamp whose solvability was rechecked.
#[derive(Clone, Debug, PartialEq)]
pub struct RevalidatedFreeBlock {
    block: QueryBlock,
    identification: IdentificationStatus,
    physical_stability: PhysicalStability,
    solver: SolverConvergence,
}

impl RevalidatedFreeBlock {
    /// Intervened free block. Ill-conditioning is left as assessed.
    #[must_use]
    pub const fn block(&self) -> &QueryBlock {
        &self.block
    }

    /// Specified-model algebra is not an identification result.
    #[must_use]
    pub const fn identification(&self) -> IdentificationStatus {
        self.identification
    }

    /// Physical stability was not claimed.
    #[must_use]
    pub const fn physical_stability(&self) -> PhysicalStability {
        self.physical_stability
    }

    /// No iterative solver ran.
    #[must_use]
    pub const fn solver(&self) -> SolverConvergence {
        self.solver
    }
}

/// Why a request was not executed on the free block.
#[derive(Clone, Debug, Eq, PartialEq, Error)]
#[non_exhaustive]
pub enum InterventionRequestError {
    /// The kind is not a hard clamp and was not rewritten as one.
    #[error("{kind} is not a free-block hard clamp; next attempt: {next_attempt}")]
    UnsupportedSemantic {
        /// Kind that was refused.
        kind: InterventionKind,
        /// What would carry that kind, instead of a clamp.
        next_attempt: &'static str,
    },
    /// The clamp leaves a singular or non-finite free block.
    ///
    /// This refusal does not mark the model unidentified.
    #[error("free block after the hard clamp is not solvable: {refusal}")]
    NotSolvable {
        /// Singular, numerically singular, or non-finite.
        refusal: SolvabilityRefusal,
    },
    /// The request does not describe one intervention of its kind.
    #[error("intervention request shape: {message}")]
    Shape {
        /// What is inconsistent.
        message: &'static str,
    },
    /// A level or a replacement coefficient is NaN or infinite.
    #[error("intervention level is not finite")]
    NonFiniteLevel,
    /// The clamp's node list and level list have different lengths.
    #[error("hard clamp has {nodes} nodes and {values} values")]
    ValueCount {
        /// Number of nodes.
        nodes: usize,
        /// Number of levels.
        values: usize,
    },
    /// A node index is outside the model.
    #[error("node {node} is outside dimension {dimension}")]
    NodeOutOfRange {
        /// Index that was requested.
        node: usize,
        /// Model dimension.
        dimension: usize,
    },
    /// A sustained window ends before it starts.
    #[error("sustained window from {from} until {until} ends before it starts")]
    InvertedWindow {
        /// First step.
        from: i32,
        /// Last step.
        until: i32,
    },
    /// The initial-state law was not named.
    #[error("initial-state law name is empty")]
    EmptyInitialLaw,
    /// The shift would write a structural diagonal.
    #[error(
        "mechanism shift into node {node} from itself is a diagonal coefficient; next attempt: \
         keep the structural diagonal at zero"
    )]
    DiagonalMechanism {
        /// Variable whose diagonal was targeted.
        node: usize,
    },
    /// The clamp itself was refused.
    #[error(transparent)]
    Solution(LinearSolutionError),
    /// The hard clamp was solvable and the equilibrium compile refused it.
    #[error(transparent)]
    Compile(SccCompileError),
}

/// The hard clamp inside `request`.
///
/// Mechanism shifts, pulses, sustained actions, and initial-state changes are
/// refused. They are not rewritten as clamps.
///
/// # Errors
///
/// [`InterventionRequestError::UnsupportedSemantic`] for every kind other than
/// a hard clamp.
#[must_use = "a refused semantic is not a clamp"]
pub fn hard_clamp_of(
    request: &InterventionRequest,
) -> Result<(&HardClamp, Vec<f64>), InterventionRequestError> {
    match request {
        InterventionRequest::HardClamp(clamp) => Ok((&clamp.clamp, clamp.values())),
        other => Err(unsupported(other.kind())),
    }
}

/// Recheck the free block `request` leaves.
///
/// Only a hard clamp is checked. The returned block keeps
/// [`Uniqueness`](crate::linear_solution::Uniqueness) as assessed, including
/// [`Uniqueness::IllConditioned`](crate::linear_solution::Uniqueness::IllConditioned)
/// and [`Uniqueness::Undetermined`](crate::linear_solution::Uniqueness::Undetermined).
/// Identification stays [`IdentificationStatus::NotMarkedUnidentified`].
/// Physical stability stays unclaimed and no iterative solver runs.
///
/// # Errors
///
/// [`InterventionRequestError::UnsupportedSemantic`],
/// [`InterventionRequestError::NotSolvable`], or a clamp dimension mismatch.
#[must_use = "a refused free block is not a solution"]
pub fn revalidate_solvability(
    model: &LinearStructuralMap,
    request: &InterventionRequest,
) -> Result<RevalidatedFreeBlock, InterventionRequestError> {
    let (clamp, _) = hard_clamp_of(request)?;
    match revalidate_free_block(model, clamp) {
        Ok(block) => Ok(RevalidatedFreeBlock {
            block,
            identification: IdentificationStatus::NotMarkedUnidentified,
            physical_stability: PhysicalStability::Unclaimed,
            solver: SolverConvergence::NotRun,
        }),
        Err(LinearSolutionError::FreeBlockNotSolvable { refusal }) => {
            Err(InterventionRequestError::NotSolvable { refusal })
        }
        Err(other) => Err(InterventionRequestError::Solution(other)),
    }
}

/// Compile the Gaussian equilibrium under `request`.
///
/// A hard clamp is rechecked, then compiled. The free disturbance stays marginal.
/// Every other kind is refused before a clamp is built.
///
/// # Errors
///
/// The errors of [`revalidate_solvability`], or [`InterventionRequestError::Compile`].
#[must_use = "a refused request is not an equilibrium"]
pub fn compile_under_request(
    component: &CyclicComponent,
    model: &LinearStructuralMap,
    disturbance: &GaussianDisturbance,
    request: &InterventionRequest,
    budget: CompileBudget,
) -> Result<CompiledEquilibrium, InterventionRequestError> {
    let (clamp, values) = hard_clamp_of(request)?;
    revalidate_solvability(model, request)?;
    compile_gaussian_equilibrium(component, model, disturbance, clamp, &values, budget)
        .map_err(InterventionRequestError::Compile)
}

fn unsupported(kind: InterventionKind) -> InterventionRequestError {
    match kind.clamp_refusal() {
        Some(next_attempt) => InterventionRequestError::UnsupportedSemantic { kind, next_attempt },
        None => InterventionRequestError::Shape {
            message: "a hard clamp has no unsupported-semantic refusal",
        },
    }
}

fn require_dimension(dimension: usize) -> Result<(), InterventionRequestError> {
    if dimension == 0 {
        return Err(InterventionRequestError::Shape {
            message: "an intervention needs a positive model dimension",
        });
    }
    Ok(())
}

fn require_node(dimension: usize, node: usize) -> Result<(), InterventionRequestError> {
    if node >= dimension {
        return Err(InterventionRequestError::NodeOutOfRange { node, dimension });
    }
    Ok(())
}

fn finite_level(value: f64) -> Result<f64, InterventionRequestError> {
    if !value.is_finite() {
        return Err(InterventionRequestError::NonFiniteLevel);
    }
    if value.to_bits() == (-0.0_f64).to_bits() { Ok(0.0) } else { Ok(value) }
}

#[cfg(test)]
mod tests {
    use super::{
        InterventionBinding, InterventionKind, InterventionRequest, InterventionRequestError,
        compile_under_request, hard_clamp_of, revalidate_solvability,
    };
    use crate::cyclic_component::{CyclicComponent, CyclicComponentError};
    use crate::linear_solution::{
        Conditioning, ConsistencyWitness, FamilyCertificate, HardClamp, IdentificationStatus,
        LinearStructuralMap, NextAttempt, PhysicalStability, SolvabilityRefusal, SolverConvergence,
        Uniqueness, assess,
    };
    use crate::scc_compile::{CompileBudget, GaussianDisturbance, SccCompileError};

    fn close(left: f64, right: f64) -> bool {
        (left - right).abs() <= 1.0e-8
    }

    fn map(coefficients: Vec<f64>, intercept: Vec<f64>) -> LinearStructuralMap {
        LinearStructuralMap::try_new(coefficients, intercept).expect("fixture map")
    }

    fn component_for(model: &LinearStructuralMap) -> CyclicComponent {
        let family = assess(model, &HardClamp::none(model.dimension()).expect("no clamp"))
            .expect("assessment")
            .family;
        assert!(matches!(family, FamilyCertificate::Admitted { .. }));
        CyclicComponent::try_new("feedback", "source-measurements", &["solve-note"], model, &family)
            .expect("component")
    }

    fn identity_noise(dimension: usize) -> GaussianDisturbance {
        let mut covariance = vec![0.0; dimension * dimension];
        for node in 0..dimension {
            covariance[node * dimension + node] = 1.0;
        }
        GaussianDisturbance::try_new(vec![0.0; dimension], covariance).expect("disturbance")
    }

    /// Eigenvalues of a 2×2 update `K_FF` as `(real, imaginary)` pairs, the roots
    /// of `λ² - tr(K_FF) λ + det(K_FF)`. A negative discriminant gives the
    /// conjugate pair, with the positive imaginary part first.
    fn pair_eigenvalues(k_ff: &[f64]) -> [(f64, f64); 2] {
        assert_eq!(k_ff.len(), 4, "a 2×2 free update is four entries");
        let trace = k_ff[0] + k_ff[3];
        let determinant = k_ff[0] * k_ff[3] - k_ff[1] * k_ff[2];
        let half_trace = trace / 2.0;
        let discriminant = half_trace * half_trace - determinant;
        if discriminant >= 0.0 {
            let root = discriminant.sqrt();
            [(half_trace + root, 0.0), (half_trace - root, 0.0)]
        } else {
            let root = (-discriminant).sqrt();
            [(half_trace, root), (half_trace, -root)]
        }
    }

    fn modulus(eigenvalue: (f64, f64)) -> f64 {
        eigenvalue.0.hypot(eigenvalue.1)
    }

    fn matmul3(left: &[f64], right: &[f64]) -> [f64; 9] {
        let mut product = [0.0; 9];
        for row in 0..3 {
            for col in 0..3 {
                let mut sum = 0.0;
                for index in 0..3 {
                    sum += left[row * 3 + index] * right[index * 3 + col];
                }
                product[row * 3 + col] = sum;
            }
        }
        product
    }

    #[test]
    fn hard_clamp_request_keeps_the_marginal_free_disturbance() {
        // do(A = 2) on A = 0.4 B + ε_A, B = 0.5 A + ε_B, Cov(ε_A, ε_B) = 0.3.
        // Var(B) stays 1, the marginal variance, rather than 0.91.
        let model = map(vec![0.0, 0.4, 0.5, 0.0], vec![0.0, 0.0]);
        let request = InterventionRequest::hard_clamp(2, &[0], &[2.0]).expect("clamp A");
        let (left, right) = match (request.binding(), {
            let other = InterventionRequest::hard_clamp(2, &[0], &[3.0]).expect("other level");
            other.binding()
        }) {
            (
                InterventionBinding::HardClamp { target: left, identity: left_id },
                InterventionBinding::HardClamp { target: right, identity: right_id },
            ) => {
                assert_eq!(left, right);
                assert_ne!(left_id, right_id);
                (left_id.levels(), right_id.levels())
            }
            other => panic!("hard clamp bindings {other:?}"),
        };
        assert!(close(left[0], 2.0));
        assert!(close(right[0], 3.0));

        let swapped = InterventionRequest::hard_clamp(2, &[1, 0], &[5.0, 4.0]).expect("sorted");
        let (clamp, values) = hard_clamp_of(&swapped).expect("clamp");
        assert_eq!(clamp.indices(), &[0, 1]);
        assert!(close(values[0], 4.0));
        assert!(close(values[1], 5.0));
        let negative_zero = InterventionRequest::hard_clamp(2, &[0], &[-0.0]).expect("signed zero");
        let positive_zero = InterventionRequest::hard_clamp(2, &[0], &[0.0]).expect("zero");
        assert_eq!(negative_zero, positive_zero);

        let rechecked = revalidate_solvability(&model, &request).expect("solvable");
        assert_eq!(rechecked.block().factor.free, vec![1]);
        assert!(matches!(rechecked.block().uniqueness, Uniqueness::Unique));
        assert_eq!(rechecked.identification(), IdentificationStatus::NotMarkedUnidentified);
        assert_eq!(rechecked.physical_stability(), PhysicalStability::Unclaimed);
        assert_eq!(rechecked.solver(), SolverConvergence::NotRun);

        let noise = GaussianDisturbance::try_new(vec![0.0, 0.0], vec![1.0, 0.3, 0.3, 1.0])
            .expect("disturbance");
        let law = compile_under_request(
            &component_for(&model),
            &model,
            &noise,
            &request,
            CompileBudget::unlimited(),
        )
        .expect("equilibrium");
        assert!(close(law.mean()[0], 2.0));
        assert!(close(law.mean()[1], 1.0));
        assert!(close(law.covariance()[3], 1.0));
        assert!((law.covariance()[3] - 0.91).abs() > 0.05);
    }

    #[test]
    fn mechanism_shift_is_not_a_hard_clamp() {
        let shift = InterventionRequest::mechanism_shift(2, 1, 0, 0.2).expect("shift");
        match shift.binding() {
            InterventionBinding::MechanismShift { target, identity } => {
                assert_eq!(target.into_node, 1);
                assert_eq!(target.from_node, 0);
                assert_eq!(identity.replacement_bits, 0.2_f64.to_bits());
            }
            other => panic!("mechanism binding {other:?}"),
        }
        let clamp = InterventionRequest::hard_clamp(2, &[0], &[0.2]).expect("clamp");
        assert_ne!(shift.kind(), clamp.kind());
        assert_ne!(shift, clamp);
        assert_eq!(
            hard_clamp_of(&shift).expect_err("not a clamp"),
            InterventionRequestError::UnsupportedSemantic {
                kind: InterventionKind::MechanismShift,
                next_attempt: InterventionKind::MechanismShift.clamp_refusal().expect("text"),
            }
        );
        let diagonal = InterventionRequest::mechanism_shift(2, 1, 1, 0.2);
        assert!(matches!(diagonal, Err(InterventionRequestError::DiagonalMechanism { node: 1 })));

        // The overlapping loops are singular under do(C). A shift is refused as a
        // semantic, not as that singular clamp.
        let loops = map(vec![0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![0.0, 0.0, 0.0]);
        let edge = InterventionRequest::mechanism_shift(3, 0, 1, 0.0).expect("edge");
        let err = revalidate_solvability(&loops, &edge).expect_err("unsupported");
        assert!(matches!(
            err,
            InterventionRequestError::UnsupportedSemantic {
                kind: InterventionKind::MechanismShift,
                ..
            }
        ));
    }

    #[test]
    fn pulse_is_not_a_sustained_action() {
        let pulse = InterventionRequest::pulse(2, 0, 0, 1.0).expect("pulse");
        let sustained = InterventionRequest::sustained(2, 0, 0, 0, 1.0).expect("one step");
        assert_eq!(pulse.kind(), InterventionKind::Pulse);
        assert_eq!(sustained.kind(), InterventionKind::SustainedAction);
        assert_ne!(pulse, sustained);
        match (pulse.binding(), sustained.binding()) {
            (
                InterventionBinding::Pulse { target, identity },
                InterventionBinding::SustainedAction { target: window, identity: held },
            ) => {
                assert_eq!(target.at, 0);
                assert_eq!(identity.level_bits, 1.0_f64.to_bits());
                assert_eq!(window.from, 0);
                assert_eq!(window.until, 0);
                assert_eq!(held.level_bits, 1.0_f64.to_bits());
            }
            other => panic!("temporal bindings {other:?}"),
        }
        assert_eq!(
            hard_clamp_of(&pulse).expect_err("pulse"),
            InterventionRequestError::UnsupportedSemantic {
                kind: InterventionKind::Pulse,
                next_attempt: InterventionKind::Pulse.clamp_refusal().expect("text"),
            }
        );
        assert_eq!(
            hard_clamp_of(&sustained).expect_err("sustained"),
            InterventionRequestError::UnsupportedSemantic {
                kind: InterventionKind::SustainedAction,
                next_attempt: InterventionKind::SustainedAction.clamp_refusal().expect("text"),
            }
        );
        let inverted = InterventionRequest::sustained(2, 0, 2, 1, 1.0);
        assert_eq!(inverted, Err(InterventionRequestError::InvertedWindow { from: 2, until: 1 }));
    }

    #[test]
    fn initial_state_change_is_not_a_hard_clamp() {
        let change = InterventionRequest::initial_state("x0", 2.0).expect("law");
        let clamp = InterventionRequest::hard_clamp(2, &[0], &[2.0]).expect("clamp");
        assert_ne!(change, clamp);
        match change.binding() {
            InterventionBinding::InitialStateChange { target, identity } => {
                assert_eq!(target.law_name, "x0");
                assert_eq!(identity.level_bits, 2.0_f64.to_bits());
                assert_eq!(identity.law_name, "x0");
            }
            other => panic!("initial-state binding {other:?}"),
        }
        assert_eq!(
            hard_clamp_of(&change).expect_err("not a clamp"),
            InterventionRequestError::UnsupportedSemantic {
                kind: InterventionKind::InitialStateChange,
                next_attempt: InterventionKind::InitialStateChange.clamp_refusal().expect("text"),
            }
        );
        assert_eq!(
            InterventionRequest::initial_state("", 2.0).expect_err("unnamed"),
            InterventionRequestError::EmptyInitialLaw
        );
        assert_eq!(
            InterventionRequest::pulse(2, 0, 0, f64::NAN).expect_err("nan"),
            InterventionRequestError::NonFiniteLevel
        );
    }

    #[test]
    fn singular_free_block_after_a_clamp_is_refused() {
        let model = map(vec![0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![0.0, 0.0, 0.0]);
        let request = InterventionRequest::hard_clamp(3, &[2], &[0.0]).expect("clamp C");
        let err = revalidate_solvability(&model, &request).expect_err("singular");
        assert_eq!(
            err,
            InterventionRequestError::NotSolvable {
                refusal: SolvabilityRefusal::Singular { members: vec![0, 1] },
            }
        );
    }

    #[test]
    fn numerically_singular_stays_distinct_from_ill_conditioned() {
        let near = map(vec![0.0, 1.0, 1.0 + 1.0e-13, 0.0], vec![0.0, 0.0]);
        let request = InterventionRequest::hard_clamp(2, &[], &[]).expect("no clamp");
        let err = revalidate_solvability(&near, &request).expect_err("pivot");
        assert_eq!(
            err,
            InterventionRequestError::NotSolvable {
                refusal: SolvabilityRefusal::NumericallySingular { members: vec![0, 1] },
            }
        );

        let near_noise = identity_noise(2);
        let near_err = compile_under_request(
            &component_for(&map(vec![0.0, 0.4, 0.5, 0.0], vec![0.0, 0.0])),
            &near,
            &near_noise,
            &request,
            CompileBudget::unlimited(),
        )
        .expect_err("near-singular clamp stays failed");
        assert_eq!(
            near_err,
            InterventionRequestError::NotSolvable {
                refusal: SolvabilityRefusal::NumericallySingular { members: vec![0, 1] },
            }
        );

        let ill = map(vec![0.0, 1.0, 1.0 - 1.0e-9, 0.0], vec![0.0, 0.0]);
        let rechecked = revalidate_solvability(&ill, &request).expect("ill conditioned");
        match &rechecked.block().uniqueness {
            Uniqueness::IllConditioned { kappa } => {
                assert!((3.9e9..4.1e9).contains(kappa), "kappa {kappa}");
            }
            other => panic!("uniqueness {other:?}"),
        }
        assert_eq!(rechecked.identification(), IdentificationStatus::NotMarkedUnidentified);
        assert!(!matches!(
            rechecked.block().uniqueness,
            Uniqueness::Singular(_) | Uniqueness::NumericallySingular { .. } | Uniqueness::Unique
        ));
        let law = compile_under_request(
            &component_for(&ill),
            &ill,
            &identity_noise(2),
            &request,
            CompileBudget::unlimited(),
        )
        .expect("ill conditioning is not a refusal");
        assert!(close(law.mean()[0], 0.0));
        assert!(close(law.mean()[1], 0.0));
        assert_eq!(law.physical_stability(), PhysicalStability::Unclaimed);
    }

    #[test]
    fn non_finite_free_block_is_refused() {
        let model = map(vec![0.0, 1.0e200, 1.0e200, 0.0], vec![0.0, 0.0]);
        let request = InterventionRequest::hard_clamp(2, &[], &[]).expect("no clamp");
        let err = revalidate_solvability(&model, &request).expect_err("overflow");
        assert_eq!(
            err,
            InterventionRequestError::NotSolvable { refusal: SolvabilityRefusal::NonFinite }
        );
    }

    #[test]
    fn open_free_block_is_not_reported_unique() {
        let mut coefficients = vec![0.0; 16];
        for node in 0..4 {
            coefficients[node * 4 + (node + 1) % 4] = 1.0;
        }
        let model = map(coefficients, vec![0.0; 4]);
        let request = InterventionRequest::hard_clamp(4, &[], &[]).expect("no clamp");
        let rechecked = revalidate_solvability(&model, &request).expect("open");
        assert!(matches!(rechecked.block().uniqueness, Uniqueness::Undetermined { .. }));
        assert_eq!(rechecked.identification(), IdentificationStatus::NotMarkedUnidentified);
        assert_eq!(rechecked.solver(), SolverConvergence::NotRun);
    }

    #[test]
    fn overlapping_loops_are_uniquely_solvable_until_c_is_clamped() {
        // A = B + C, B = A, C = A. The full system is uniquely solvable.
        // do(C = 1) leaves (I - K_FF) singular and inconsistent, so the solution fails.
        let model = map(vec![0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![0.0, 0.0, 0.0]);
        let open = InterventionRequest::hard_clamp(3, &[], &[]).expect("no clamp");
        let full = revalidate_solvability(&model, &open).expect("full system");
        assert!(matches!(full.block().uniqueness, Uniqueness::Unique));
        assert_eq!(full.block().factor.free, vec![0, 1, 2]);
        assert_eq!(full.physical_stability(), PhysicalStability::Unclaimed);

        let request = InterventionRequest::hard_clamp(3, &[2], &[1.0]).expect("clamp C");
        let (clamp, values) = hard_clamp_of(&request).expect("clamp");
        let report = assess(&model, clamp).expect("assessment");
        assert!(matches!(report.original.uniqueness, Uniqueness::Unique));
        assert!(matches!(report.original.conditioning, Conditioning::NotEstimated { .. }));
        match &report.intervened.uniqueness {
            Uniqueness::Singular(block) => match &block.witness {
                ConsistencyWitness::LeftNull(weights) => {
                    let coupling = &report.intervened.factor.k_fh;
                    let intercept = &report.intervened.factor.intercept_f;
                    assert_eq!(weights.len(), coupling.len());
                    let mut residual = 0.0;
                    for (index, weight) in weights.iter().enumerate() {
                        residual += weight * (coupling[index] * values[0] + intercept[index]);
                    }
                    assert!(close(residual, 1.0));
                }
                ConsistencyWitness::NotFormed => panic!("2×2 left null should be formed"),
            },
            other => panic!("intervened uniqueness {other:?}"),
        }
        assert!(matches!(report.intervened.conditioning, Conditioning::Undefined));
        match &report.family {
            FamilyCertificate::Insufficient { rho_abs_upper, next_attempt, identification } => {
                assert!(*rho_abs_upper >= 1.0);
                assert_eq!(
                    next_attempt,
                    &NextAttempt::QuerySpecificFreeBlock { free: vec![0, 1] }
                );
                assert_eq!(*identification, IdentificationStatus::NotMarkedUnidentified);
            }
            FamilyCertificate::Admitted { .. } => {
                panic!("overlapping loops are not a contraction")
            }
        }
        assert_eq!(report.physical_stability, PhysicalStability::Unclaimed);
        assert_eq!(
            revalidate_solvability(&model, &request).expect_err("singular"),
            InterventionRequestError::NotSolvable {
                refusal: SolvabilityRefusal::Singular { members: vec![0, 1] },
            }
        );

        // The recheck refuses before compilation. A stand-in component is not
        // allowed to turn this failed clamp into an equilibrium.
        let stand_in = component_for(&map(vec![0.0, 0.4, 0.5, 0.0], vec![0.0, 0.0]));
        let err = compile_under_request(
            &stand_in,
            &model,
            &identity_noise(3),
            &request,
            CompileBudget::unlimited(),
        )
        .expect_err("singular clamp stays failed");
        assert_eq!(
            err,
            InterventionRequestError::NotSolvable {
                refusal: SolvabilityRefusal::Singular { members: vec![0, 1] },
            }
        );
        let shift = InterventionRequest::mechanism_shift(3, 0, 1, 0.0).expect("shift");
        let shift_err = compile_under_request(
            &stand_in,
            &model,
            &identity_noise(3),
            &shift,
            CompileBudget::unlimited(),
        )
        .expect_err("shift stays failed");
        assert!(matches!(
            shift_err,
            InterventionRequestError::UnsupportedSemantic {
                kind: InterventionKind::MechanismShift,
                ..
            }
        ));
    }

    #[test]
    fn unique_unstable_equilibrium_is_not_reported_as_stable() {
        // K = [[0, 2], [2, 0]]. det(I - K) = -3, so the equilibrium is unique.
        // The update eigenvalues are ±2, outside the unit circle.
        let model = map(vec![0.0, 2.0, 2.0, 0.0], vec![0.0, 0.0]);
        let request = InterventionRequest::hard_clamp(2, &[], &[]).expect("full update");
        let rechecked = revalidate_solvability(&model, &request).expect("unique");
        let [larger, smaller] = pair_eigenvalues(&rechecked.block().factor.k_ff);
        assert!(close(larger.0, 2.0) && close(larger.1, 0.0));
        assert!(close(smaller.0, -2.0) && close(smaller.1, 0.0));
        assert!(modulus(larger) > 1.0);
        assert!(modulus(smaller) > 1.0);
        assert!(matches!(rechecked.block().uniqueness, Uniqueness::Unique));
        assert_eq!(rechecked.physical_stability(), PhysicalStability::Unclaimed);
        assert_eq!(rechecked.identification(), IdentificationStatus::NotMarkedUnidentified);
        assert_eq!(rechecked.solver(), SolverConvergence::NotRun);

        let report = assess(&model, &HardClamp::none(2).expect("no clamp")).expect("assessment");
        assert!(matches!(report.original.uniqueness, Uniqueness::Unique));
        assert!(!report.family.is_admitted());
        assert!(report.family.rho_abs_upper() > 1.0);
        assert_eq!(report.physical_stability, PhysicalStability::Unclaimed);
        assert_eq!(report.solver, SolverConvergence::NotRun);
    }

    #[test]
    fn rotation_update_has_a_complex_pair_outside_the_unit_circle() {
        // K = [[0, -2], [2, 0]]. det(I - K) = 5, so the equilibrium is unique.
        // The update eigenvalues are ±2i; the real-root formula would give NaN.
        let model = map(vec![0.0, -2.0, 2.0, 0.0], vec![0.0, 0.0]);
        let request = InterventionRequest::hard_clamp(2, &[], &[]).expect("full update");
        let rechecked = revalidate_solvability(&model, &request).expect("unique");
        assert!(matches!(rechecked.block().uniqueness, Uniqueness::Unique));
        let [upper, lower] = pair_eigenvalues(&rechecked.block().factor.k_ff);
        assert!(upper.0.is_finite() && lower.0.is_finite());
        assert!(close(upper.0, 0.0) && close(upper.1, 2.0));
        assert!(close(lower.0, 0.0) && close(lower.1, -2.0));
        assert!(close(modulus(upper), 2.0));
        assert!(close(modulus(lower), 2.0));
        let report = assess(&model, &HardClamp::none(2).expect("no clamp")).expect("assessment");
        assert!(!report.family.is_admitted());
        assert_eq!(report.physical_stability, PhysicalStability::Unclaimed);
    }

    #[test]
    fn unique_unstable_equilibrium_reaches_no_consumer() {
        // K = [[0, 2], [2, 0]]: unique, with update eigenvalues ±2.
        let unstable = map(vec![0.0, 2.0, 2.0, 0.0], vec![0.0, 0.0]);
        let open = HardClamp::none(2).expect("no clamp");
        let family = assess(&unstable, &open).expect("assessment").family;
        // No component can be admitted for it.
        let refused = CyclicComponent::try_new(
            "feedback",
            "source-measurements",
            &["solve-note"],
            &unstable,
            &family,
        )
        .expect_err("an unstable family is not admitted");
        assert!(matches!(refused, CyclicComponentError::FamilyNotAdmitted { .. }));
        // A component admitted for another model does not compile it either.
        let stable = map(vec![0.0, 0.4, 0.5, 0.0], vec![0.0, 0.0]);
        let borrowed = component_for(&stable);
        for request in [
            InterventionRequest::hard_clamp(2, &[], &[]).expect("full update"),
            InterventionRequest::hard_clamp(2, &[0], &[1.0]).expect("do(A = 1)"),
        ] {
            // The free block is solvable, so the refusal is the component's.
            revalidate_solvability(&unstable, &request).expect("unique free block");
            let received = compile_under_request(
                &borrowed,
                &unstable,
                &identity_noise(2),
                &request,
                CompileBudget::unlimited(),
            );
            assert!(matches!(
                received,
                Err(InterventionRequestError::Compile(
                    SccCompileError::ComponentModelMismatch { .. }
                ))
            ));
        }
    }

    #[test]
    fn nilpotent_full_update_cubes_to_zero_and_third_clamp_leaves_pm_two() {
        // K = [[0, 2, 2], [2, 0, 0], [-2, 0, 0]]. K³ = 0 on the full update.
        // Clamping variable three leaves K_FF = [[0, 2], [2, 0]], eigenvalues ±2.
        let coefficients = vec![0.0, 2.0, 2.0, 2.0, 0.0, 0.0, -2.0, 0.0, 0.0];
        let square = matmul3(&coefficients, &coefficients);
        let cube = matmul3(&square, &coefficients);
        assert!(cube.iter().all(|entry| close(*entry, 0.0)));

        let model = map(coefficients, vec![0.0, 0.0, 0.0]);
        let full = InterventionRequest::hard_clamp(3, &[], &[]).expect("full update");
        let full_block = revalidate_solvability(&model, &full).expect("nilpotent update");
        assert_eq!(full_block.block().factor.k_ff.len(), model.coefficients().len());
        assert!(
            full_block
                .block()
                .factor
                .k_ff
                .iter()
                .zip(model.coefficients())
                .all(|(left, right)| close(*left, *right))
        );
        assert!(matches!(full_block.block().uniqueness, Uniqueness::Unique));
        assert_eq!(full_block.physical_stability(), PhysicalStability::Unclaimed);

        let clamped = InterventionRequest::hard_clamp(3, &[2], &[0.0]).expect("clamp third");
        let free = revalidate_solvability(&model, &clamped).expect("free block");
        assert_eq!(free.block().factor.free, vec![0, 1]);
        assert_eq!(free.block().factor.clamped, vec![2]);
        let k_ff = &free.block().factor.k_ff;
        assert!(close(k_ff[0], 0.0));
        assert!(close(k_ff[1], 2.0));
        assert!(close(k_ff[2], 2.0));
        assert!(close(k_ff[3], 0.0));
        let [larger, smaller] = pair_eigenvalues(k_ff);
        assert!(close(larger.0, 2.0) && close(larger.1, 0.0));
        assert!(close(smaller.0, -2.0) && close(smaller.1, 0.0));
        assert!(matches!(free.block().uniqueness, Uniqueness::Unique));
        assert_eq!(free.physical_stability(), PhysicalStability::Unclaimed);
        assert_eq!(free.identification(), IdentificationStatus::NotMarkedUnidentified);
        let report =
            assess(&model, &HardClamp::try_new(3, &[2]).expect("clamp")).expect("report");
        assert!(!report.family.is_admitted());
        assert_eq!(report.physical_stability, PhysicalStability::Unclaimed);
    }
}
