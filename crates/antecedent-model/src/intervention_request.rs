// SPDX-License-Identifier: MIT OR Apache-2.0

//! Interventions on a linear free block.
//!
//! Callers pass [`antecedent_core::Intervention`]. An untimed
//! [`Intervention::Set`](antecedent_core::Intervention::Set) is a hard clamp:
//! it deletes the clamped equations and leaves
//! [`FREE_BLOCK_EQUATION`](crate::linear_solution::FREE_BLOCK_EQUATION). The
//! disturbance on the free variables stays marginal.
//!
//! A shift, a soft mechanism replacement, a pulse, a sustained action, and an
//! initial-state change are refused. None of them is rewritten as a clamp. A
//! pulse is not a sustained action, including when the sustained window is a
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

use antecedent_core::intervention::{AdmitError, ExecutionSurface, InterventionOperation};
use antecedent_core::{Intervention, InterventionError, TemporalPolicy, Value};
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

/// One hard clamp. Other interventions are refused before a clamp is built.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum InterventionRequest {
    /// Hard clamp of equilibrium variables.
    HardClamp(HardClampRequest),
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

/// Clamp variables and the levels assigned to them.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct InterventionBinding {
    /// Variables held.
    pub target: ClampTarget,
    /// Assigned levels.
    pub identity: HardClampIdentity,
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

    /// Target and identity of this clamp.
    #[must_use]
    pub fn binding(&self) -> InterventionBinding {
        let Self::HardClamp(request) = self;
        InterventionBinding {
            target: request.target.clone(),
            identity: request.identity.clone(),
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
    /// The operation is not a hard clamp and was not rewritten as one.
    #[error("{operation} is not a free-block hard clamp: {reason}")]
    UnsupportedSemantic {
        /// Operation that was refused.
        operation: &'static str,
        /// Why it was not executed as a hard clamp.
        reason: &'static str,
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
    /// The initial-state law was not named.
    #[error("initial-state law name is empty")]
    EmptyInitialLaw,
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
    let InterventionRequest::HardClamp(clamp) = request;
    Ok((&clamp.clamp, clamp.values()))
}

/// Recheck the free block an [`Intervention`] leaves.
///
/// An untimed [`Intervention::Set`](antecedent_core::Intervention::Set) is the
/// hard clamp of that variable. Every other operation is refused by name.
///
/// # Errors
///
/// [`InterventionRequestError::UnsupportedSemantic`] when the operation is not
/// an untimed set, [`InterventionRequestError::NotSolvable`], or a clamp error.
#[must_use = "a refused free block is not a solution"]
pub fn revalidate_intervention(
    model: &LinearStructuralMap,
    intervention: &Intervention,
    temporal: Option<&TemporalPolicy>,
) -> Result<RevalidatedFreeBlock, InterventionRequestError> {
    let request = clamp_for_set(model.dimension(), intervention, temporal)?;
    revalidate_solvability(model, &request)
}

/// Compile the Gaussian equilibrium under `intervention`.
///
/// An untimed set is rechecked, then compiled. The free disturbance stays
/// marginal. A shift, a soft replacement, a pulse, and a sustained action are
/// refused before a clamp is built.
///
/// # Errors
///
/// The errors of [`revalidate_intervention`], or [`InterventionRequestError::Compile`].
#[must_use = "a refused intervention is not an equilibrium"]
pub fn compile_intervention(
    component: &CyclicComponent,
    model: &LinearStructuralMap,
    disturbance: &GaussianDisturbance,
    intervention: &Intervention,
    temporal: Option<&TemporalPolicy>,
    budget: CompileBudget,
) -> Result<CompiledEquilibrium, InterventionRequestError> {
    let request = clamp_for_set(model.dimension(), intervention, temporal)?;
    compile_under_request(component, model, disturbance, &request, budget)
}

/// Refuse an initial-state change. It is not compiled as a hard clamp.
///
/// # Errors
///
/// [`InterventionRequestError::UnsupportedSemantic`] naming `initial_state_change`,
/// or a malformed law name or level.
#[must_use = "a refused initial-state change is not a clamp"]
pub fn revalidate_initial_state(
    model: &LinearStructuralMap,
    law_name: &str,
    level: f64,
) -> Result<RevalidatedFreeBlock, InterventionRequestError> {
    let _ = model;
    let operation = InterventionOperation::initial_state(law_name, Value::f64(level))
        .map_err(|error| map_intervention_error(&error))?;
    match ExecutionSurface::Cyclic.admit_operation(&operation) {
        Err(refusal) => Err(refused(refusal)),
        Ok(()) => Err(InterventionRequestError::UnsupportedSemantic {
            operation: "initial_state_change",
            reason: "an initial-state change replaces the initial-state law and is not a \
                     hard clamp",
        }),
    }
}

fn clamp_for_set(
    dimension: usize,
    intervention: &Intervention,
    temporal: Option<&TemporalPolicy>,
) -> Result<InterventionRequest, InterventionRequestError> {
    let operation =
        ExecutionSurface::Cyclic.admit(intervention, temporal).map_err(map_admit_error)?;
    let InterventionOperation::Set { variable, value } = operation else {
        return Err(InterventionRequestError::UnsupportedSemantic {
            operation: operation.name(),
            reason: "only an untimed set is a free-block hard clamp",
        });
    };
    let Value::Float64(level) = value else {
        return Err(InterventionRequestError::UnsupportedSemantic {
            operation: "set",
            reason: "a hard clamp needs one finite float level and does not coerce other values",
        });
    };
    if !level.is_finite() {
        return Err(InterventionRequestError::NonFiniteLevel);
    }
    let node = usize::try_from(variable.raw()).map_err(|_| InterventionRequestError::Shape {
        message: "intervention variable index does not fit usize",
    })?;
    InterventionRequest::hard_clamp(dimension, &[node], &[level])
}

fn map_admit_error(error: AdmitError) -> InterventionRequestError {
    match error {
        AdmitError::Refused(refusal) => refused(refusal),
        AdmitError::Invalid(invalid) => map_intervention_error(&invalid),
        _ => InterventionRequestError::UnsupportedSemantic {
            operation: "intervention",
            reason: "the intervention was not admitted",
        },
    }
}

fn map_intervention_error(error: &InterventionError) -> InterventionRequestError {
    match error {
        InterventionError::NonFiniteLevel => InterventionRequestError::NonFiniteLevel,
        InterventionError::EmptyInitialLaw => InterventionRequestError::EmptyInitialLaw,
        _ => InterventionRequestError::Shape {
            message: "the intervention is not a cyclic hard clamp",
        },
    }
}

fn refused(refusal: antecedent_core::intervention::CapabilityRefusal) -> InterventionRequestError {
    InterventionRequestError::UnsupportedSemantic {
        operation: refusal.operation(),
        reason: refusal.reason(),
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

fn require_dimension(dimension: usize) -> Result<(), InterventionRequestError> {
    if dimension == 0 {
        return Err(InterventionRequestError::Shape {
            message: "an intervention needs a positive model dimension",
        });
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
        InterventionRequest, InterventionRequestError, compile_intervention, compile_under_request,
        hard_clamp_of, revalidate_initial_state, revalidate_intervention, revalidate_solvability,
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
        let left = request.binding();
        let right =
            InterventionRequest::hard_clamp(2, &[0], &[3.0]).expect("other level").binding();
        assert_eq!(left.target, right.target);
        assert_ne!(left.identity, right.identity);
        let (left, right) = (left.identity.levels(), right.identity.levels());
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

    fn node(raw: u32) -> antecedent_core::VariableId {
        antecedent_core::VariableId::from_raw(raw)
    }

    fn refused_operation(error: InterventionRequestError) -> &'static str {
        match error {
            InterventionRequestError::UnsupportedSemantic { operation, .. } => operation,
            other => panic!("expected a named refusal, got {other:?}"),
        }
    }

    #[test]
    fn the_same_set_is_the_cyclic_clamp_and_other_operations_are_refused() {
        use antecedent_core::{Intervention, MechanismOverride, TemporalPolicy, Value};
        use antecedent_graph::Dag;
        use crate::compile::CompiledCausalModel;
        use crate::overlay::InterventionOverlay;

        let model = map(vec![0.0, 0.4, 0.5, 0.0], vec![0.0, 0.0]);
        let set = Intervention::set(node(0), Value::f64(2.0));
        let static_model = CompiledCausalModel::compile(Dag::with_variables(2)).expect("dag");
        let overlay = InterventionOverlay::from_declared(&static_model, &set, None).expect("set");
        assert_eq!(overlay.hard_set[0], Some(2.0));
        let rechecked = revalidate_intervention(&model, &set, None).expect("clamp");
        assert_eq!(rechecked.block().factor.clamped, vec![0]);
        let law = compile_intervention(
            &component_for(&model),
            &model,
            &identity_noise(2),
            &set,
            None,
            CompileBudget::unlimited(),
        )
        .expect("equilibrium");
        assert!(close(law.mean()[0], overlay.hard_set[0].expect("assigned")));
        assert!(close(law.mean()[1], 1.0));

        let shift = Intervention::shift(node(0), Value::f64(0.2));
        let soft = Intervention::soft(node(0), MechanismOverride::constant(0.2));
        let loops = map(vec![0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![0.0, 0.0, 0.0]);
        assert_eq!(
            refused_operation(revalidate_intervention(&loops, &shift, None).expect_err("shift")),
            "shift"
        );
        assert_eq!(
            refused_operation(revalidate_intervention(&model, &soft, None).expect_err("soft")),
            "soft"
        );
        assert_eq!(
            refused_operation(
                revalidate_intervention(&model, &set, Some(&TemporalPolicy::pulse(0)))
                    .expect_err("pulse")
            ),
            "pulse"
        );
        assert_eq!(
            refused_operation(
                revalidate_intervention(&model, &set, Some(&TemporalPolicy::sustained(0, 0)))
                    .expect_err("sustained")
            ),
            "sustained"
        );
        assert_ne!(
            refused_operation(
                revalidate_intervention(&model, &set, Some(&TemporalPolicy::pulse(0)))
                    .expect_err("pulse")
            ),
            refused_operation(
                revalidate_intervention(&model, &set, Some(&TemporalPolicy::sustained(0, 0)))
                    .expect_err("sustained")
            )
        );
        assert_eq!(
            refused_operation(revalidate_initial_state(&model, "x0", 2.0).expect_err("law")),
            "initial_state_change"
        );
        assert_eq!(
            revalidate_initial_state(&model, "", 2.0).expect_err("unnamed"),
            InterventionRequestError::EmptyInitialLaw
        );
        assert_eq!(
            revalidate_intervention(
                &model,
                &Intervention::set(node(0), Value::f64(f64::NAN)),
                None
            )
            .expect_err("nan"),
            InterventionRequestError::NonFiniteLevel
        );
        let stand_in = component_for(&model);
        assert_eq!(
            refused_operation(
                compile_intervention(
                    &stand_in,
                    &loops,
                    &identity_noise(3),
                    &shift,
                    None,
                    CompileBudget::unlimited(),
                )
                .expect_err("shift stays failed")
            ),
            "shift"
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
        let shift = antecedent_core::Intervention::shift(node(0), antecedent_core::Value::f64(0.0));
        assert_eq!(
            refused_operation(
                compile_intervention(
                    &stand_in,
                    &model,
                    &identity_noise(3),
                    &shift,
                    None,
                    CompileBudget::unlimited(),
                )
                .expect_err("shift stays failed")
            ),
            "shift"
        );
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
