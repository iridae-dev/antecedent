// SPDX-License-Identifier: MIT OR Apache-2.0

//! Gaussian equilibrium of a [`LinearStructuralMap`], compiled by strongly connected components.
//!
//! The map is `X = KX + c + ε` with a zero diagonal. A nonzero diagonal is
//! refused by [`LinearStructuralMap::try_new`] and names
//! [`NONZERO_DIAGONAL_NEXT_ATTEMPT`]. This module does not normalize it.
//!
//! Free variables under a hard clamp obey
//! [`FREE_BLOCK_EQUATION`](crate::linear_solution::FREE_BLOCK_EQUATION). Each strongly
//! connected component of the free support is one square block `I - K_SS`,
//! scheduled sources before sinks. Each block is passed to
//! [`antecedent_kernels::gauss_jordan_solve`]. That call overwrites the block;
//! the inverse is not formed and is not stored. The block diagonal of `I - K`
//! is 1, so the kernel's diagonal scale is 1 and a pivot at or below
//! [`antecedent_kernels::EQUILIBRATED_PIVOT_FLOOR`] is numerically singular. The
//! same test decides [`assess`](crate::linear_solution::assess), so the two
//! agree on which blocks are refused. A numerically singular or non-finite block
//! is a refusal. No root is selected, and the refusal is not a completed
//! mechanism. Numerically singular is weaker than exactly singular: a pivot below
//! the floor does not show that the determinant is zero. After the budget admits
//! the compile, [`revalidate_free_block`](crate::linear_solution::revalidate_free_block)
//! rechecks the clamp's free block before any block is overwritten. Singular and
//! numerically singular free blocks are refused there. An ill-conditioned block
//! is not one of those refusals.
//!
//! `ε` is Gaussian with mean `μ` and covariance `Σ`. Entries of `Σ` between
//! components are kept. Along the condensation,
//! `Cov(X_F) = (I - K_FF)^{-1} Σ_FF (I - K_FF)^{-T}` is applied by solves, with
//! parents from earlier blocks on the right-hand side. A hard clamp substitutes
//! constants. It does not condition `ε_F` on the clamped disturbances, so the
//! free block keeps the marginal `Σ_FF`.
//!
//! Bongers, Forré, Peters, and Mooij, *Foundations of Structural Causal Models
//! with Cycles and Latent Variables* (arXiv:1611.06221), Section 2.2, define
//! solvability of a cyclic system. Hyttinen, Eberhardt, and Hoyer, *Learning
//! Linear Cyclic Causal Models with Latent Variables*, JMLR 13 (2012), Section
//! 2, define this linear equilibrium; weak stability there is invertibility.
//! Compilation is a specified-model solve. It does not run that estimator and
//! does not mark the model identified.
//!
//! The compiled value carries the structural residual and a condition
//! diagnostic. [`gaussian_equilibrium_admission`] is the work and byte charge.
//! A budget that does not cover the charge is refused before the factorization.
//!
//! An acyclic free support is not solved as a cyclic block.
//! [`route_linear_support`] asks the DAG algorithm first. A directed cycle is
//! refused there, with that algorithm's next attempt, and only then does the
//! strongly connected compiler run. Specified models remain model-conditional;
//! compilation grants no scientific support.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use antecedent_graph::{CyclicDigraph, DenseNodeId};
use antecedent_kernels::gauss_jordan_solve;
use thiserror::Error;

use crate::cyclic_component::{
    CyclicComponent, DAG_EQUILIBRIUM_ATTEMPT, ModelBinding, SCC_COMPILE_ATTEMPT,
};
use crate::linear_algebra::{
    MatrixError, add_times_transpose, eye_minus_determinant2, matmul, max_abs_row_sum, transpose,
    unit_diagonal_infinity_kappa2, zeros,
};
use crate::linear_solution::{
    FREE_BLOCK_EQUATION, HardClamp, IdentificationStatus, LinearSolutionError, LinearStructuralMap,
    PhysicalStability, SolvabilityRefusal, SolverConvergence, free_indices, pivots_regular,
    revalidate_free_block,
};

mod admission;
mod numerics;
mod propagation;
mod routing;
use admission::{plan_bytes, plan_work, work_overflow};
use numerics::{
    block_residual, condition_of, finite_matrix, positions, solve_row_major,
    structural_coefficient, structural_residual, symmetrize,
};
pub(crate) use numerics::{eye_minus_block, require_psd};
use propagation::propagate;
pub use routing::{
    LinearRouteSet, LinearSupportRoute, RoutedEquilibrium, compile_routed_gaussian_equilibrium,
    route_linear_support,
};
pub(crate) use routing::{compile_routed_with, dag_support_refusal};

/// Work and bytes a compilation may spend on the factorizations and diagnostics.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CompileBudget {
    /// Admission work. This is the charge from [`gaussian_equilibrium_admission`], not a
    /// hardware counter.
    pub max_work: u64,
    /// Upper bound on the floats the compilation allocates, in bytes.
    pub max_bytes: u64,
}

impl CompileBudget {
    /// A budget that admits every charge that fits in `u64`.
    #[must_use]
    pub const fn unlimited() -> Self {
        Self { max_work: u64::MAX, max_bytes: u64::MAX }
    }
}

/// Work and bytes charged before a factorization runs.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AdmissionCost {
    work: u64,
    bytes: u64,
}

impl AdmissionCost {
    /// Admission work for support routing, the factorization, the solves, the covariance
    /// check, and the diagnostics, charged cumulatively against one budget.
    #[must_use]
    pub const fn work(self) -> u64 {
        self.work
    }

    /// Byte upper bound on the peak dense storage: the block workspaces and covariance
    /// states of the solve plus the covariance the compiled result keeps.
    #[must_use]
    pub const fn bytes(self) -> u64 {
        self.bytes
    }
}

/// Why a linear Gaussian equilibrium was not compiled.
#[derive(Clone, Debug, Eq, PartialEq, Error)]
#[non_exhaustive]
pub enum SccCompileError {
    /// The dimensions or the block schedule do not describe one compile.
    #[error("SCC compile shape: {message}")]
    Shape {
        /// What is inconsistent.
        message: &'static str,
    },
    /// The disturbance and the map have different dimensions.
    #[error("disturbance dimension {disturbance} does not match model dimension {model}")]
    DisturbanceDimension {
        /// Dimension of the disturbance.
        disturbance: usize,
        /// Dimension of the map.
        model: usize,
    },
    /// The clamp was built for a different dimension than the map.
    #[error("clamp dimension {clamp} does not match model dimension {model}")]
    ClampDimension {
        /// Dimension stored on the clamp.
        clamp: usize,
        /// Dimension of the map.
        model: usize,
    },
    /// `clamp_values` does not align with the clamp indices.
    #[error("clamp has {nodes} nodes and {values} values")]
    ClampValues {
        /// Number of clamped nodes.
        nodes: usize,
        /// Number of supplied values.
        values: usize,
    },
    /// A disturbance mean is NaN or infinite.
    #[error("non-finite disturbance mean at node {node}")]
    NonFiniteDisturbanceMean {
        /// Variable whose disturbance mean is not finite.
        node: usize,
    },
    /// A disturbance covariance entry is NaN or infinite.
    #[error("non-finite disturbance covariance at row {row}, column {column}")]
    NonFiniteDisturbanceCovariance {
        /// Row of the covariance entry.
        row: usize,
        /// Column of the covariance entry.
        column: usize,
    },
    /// The covariance is not symmetric.
    #[error("disturbance covariance is not symmetric at row {row}, column {column}")]
    AsymmetricCovariance {
        /// Row of the disagreeing entry.
        row: usize,
        /// Column of the disagreeing entry.
        column: usize,
    },
    /// A clamp value is NaN or infinite.
    #[error("non-finite clamp value at node {node}")]
    NonFiniteClamp {
        /// Clamped variable.
        node: usize,
    },
    /// A computed equilibrium entry is NaN or infinite.
    #[error("non-finite equilibrium value; no mechanism was returned")]
    NonFiniteEquilibrium,
    /// `Σ` is not positive semidefinite.
    #[error("disturbance covariance is not positive semidefinite at index {index}")]
    NotPositiveSemidefinite {
        /// Pivot index where a negative direction appeared.
        index: usize,
    },
    /// `I - K_SS` has a pivot at or below the equilibrated floor. The block is
    /// numerically singular, which does not claim its determinant is exactly zero.
    /// No root was selected.
    #[error(
        "SCC block {members:?} is numerically singular (pivot at or below the equilibrated \
         floor); no equilibrium was selected"
    )]
    NumericallySingularBlock {
        /// Free variables in the block, ascending.
        members: Vec<usize>,
    },
    /// The component was admitted for a model of a different order.
    #[error("component was admitted for a model of order {component}, not {model}")]
    ComponentDimension {
        /// Order stored on the component's binding.
        component: usize,
        /// Order of the supplied model.
        model: usize,
    },
    /// The component was admitted for a different model than the one supplied.
    #[error("component `{component}` was admitted for a different model than the one supplied")]
    ComponentModelMismatch {
        /// Id of the component.
        component: String,
    },
    /// The admission charge exceeds the budget. No factorization was run.
    #[error("SCC compile {resource} {requested} exceeds the budget {limit}")]
    Budget {
        /// `"work"` or `"bytes"`.
        resource: &'static str,
        /// Charge that was refused.
        requested: u64,
        /// Budget entry that refused it.
        limit: u64,
    },
}

/// Center and covariance of `ε`.
///
/// `covariance` is row-major, length `mean.len() * mean.len()`. Off-block entries are part of
/// the law. Positive semidefiniteness is checked when the equilibrium is compiled, under the
/// compile budget, and the verdict stays with the covariance: every disturbance derived from
/// it with another center, and every clone, is checked once between them.
#[derive(Clone, Debug, PartialEq)]
pub struct GaussianDisturbance {
    mean: Vec<f64>,
    covariance: Arc<[f64]>,
    psd: PsdVerdict,
}

/// Whether the covariance shared by a family of disturbances has passed [`require_psd`].
///
/// Equal for every pair: it is a cache of a property of the covariance, not part of the law.
#[derive(Clone, Debug, Default)]
struct PsdVerdict(Arc<AtomicBool>);

impl PartialEq for PsdVerdict {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}

impl GaussianDisturbance {
    /// Disturbance with finite `mean` and finite symmetric `covariance`.
    ///
    /// # Errors
    ///
    /// An empty mean, a covariance that is not `n × n`, a non-finite entry, or an asymmetric
    /// entry.
    #[must_use = "a disturbance error is a refusal of the Gaussian law"]
    pub fn try_new(mean: Vec<f64>, covariance: Vec<f64>) -> Result<Self, SccCompileError> {
        let n = mean.len();
        if n == 0 {
            return Err(SccCompileError::Shape {
                message: "a disturbance needs at least one variable",
            });
        }
        let Some(len) = n.checked_mul(n) else {
            return Err(SccCompileError::Shape {
                message: "disturbance covariance length overflows usize",
            });
        };
        if covariance.len() != len {
            return Err(SccCompileError::Shape {
                message: "disturbance covariance must be row-major with length n*n",
            });
        }
        for (node, value) in mean.iter().copied().enumerate() {
            if !value.is_finite() {
                return Err(SccCompileError::NonFiniteDisturbanceMean { node });
            }
        }
        for row in 0..n {
            for col in 0..n {
                let value = covariance[row * n + col];
                if !value.is_finite() {
                    return Err(SccCompileError::NonFiniteDisturbanceCovariance {
                        row,
                        column: col,
                    });
                }
                #[allow(
                    clippy::float_cmp,
                    reason = "symmetry of the supplied covariance is an exact contract"
                )]
                if value != covariance[col * n + row] {
                    return Err(SccCompileError::AsymmetricCovariance { row, column: col });
                }
            }
        }
        Ok(Self { mean, covariance: covariance.into(), psd: PsdVerdict::default() })
    }

    /// This covariance under another finite `mean`, sharing the covariance and its
    /// positive-semidefiniteness verdict.
    pub(crate) fn with_mean(&self, mean: Vec<f64>) -> Result<Self, SccCompileError> {
        if mean.len() != self.mean.len() {
            return Err(SccCompileError::Shape {
                message: "a disturbance mean must keep the covariance's order",
            });
        }
        if let Some(node) = mean.iter().position(|value| !value.is_finite()) {
            return Err(SccCompileError::NonFiniteDisturbanceMean { node });
        }
        Ok(Self { mean, covariance: Arc::clone(&self.covariance), psd: self.psd.clone() })
    }

    /// Whether the covariance has already passed the positive-semidefinite check.
    pub(crate) fn psd_verified(&self) -> bool {
        self.psd.0.load(Ordering::Acquire)
    }

    /// Check that the covariance is positive semidefinite, once for every disturbance sharing it.
    pub(crate) fn require_psd_once(&self) -> Result<(), SccCompileError> {
        if self.psd_verified() {
            return Ok(());
        }
        require_psd(&self.covariance, self.mean.len())?;
        self.psd.0.store(true, Ordering::Release);
        Ok(())
    }

    /// Dimension of `ε`.
    #[must_use]
    pub fn dimension(&self) -> usize {
        self.mean.len()
    }

    /// Mean of `ε`, in variable order.
    #[must_use]
    pub fn mean(&self) -> &[f64] {
        &self.mean
    }

    /// Row-major covariance of `ε`.
    #[must_use]
    pub fn covariance(&self) -> &[f64] {
        &self.covariance
    }
}

/// One solved free block.
#[derive(Clone, Debug, PartialEq)]
pub struct BlockDiagnostic {
    /// Free variables in this block, ascending.
    pub members: Vec<usize>,
    /// Infinity norm of `(I - K_SS) μ_S - rhs_S`.
    pub residual_inf: f64,
    /// Condition number of `I - K_SS`.
    ///
    /// For order 1 and 2 this is the infinity-norm closed form and [`Self::exact`]
    /// is true. For a larger block it is the Hager–Higham estimate of that
    /// infinity-norm condition number, and [`Self::exact`] is false. The estimate
    /// is a diagnostic. It is not the one-vector lower bound `||A||∞ ||A^{-1} 1||∞`.
    pub kappa_inf: f64,
    /// Whether `kappa_inf` is the closed form rather than the Hager–Higham estimate.
    pub exact: bool,
}

/// Compiled Gaussian equilibrium of one admitted component.
///
/// The law is the pushforward of the disturbance through the solved free blocks.
/// [`Self::identification`] is [`IdentificationStatus::NotMarkedUnidentified`].
/// [`Self::physical_stability`] stays unclaimed. [`Self::solver`] stays
/// [`SolverConvergence::NotRun`]: Gauss-Jordan is a direct solve, not an iterative solver.
#[derive(Clone, Debug, PartialEq)]
#[must_use = "dropping it drops the compiled equilibrium"]
pub struct CompiledEquilibrium {
    component_id: String,
    free_block_equation: &'static str,
    compile_attempt: &'static str,
    identification: IdentificationStatus,
    physical_stability: PhysicalStability,
    solver: SolverConvergence,
    mean: Vec<f64>,
    covariance: Vec<f64>,
    residual_inf: f64,
    blocks: Vec<BlockDiagnostic>,
    admission: AdmissionCost,
}

impl CompiledEquilibrium {
    /// Id of the admitted component. This is not the component's source id.
    #[must_use]
    pub fn component_id(&self) -> &str {
        &self.component_id
    }

    /// [`FREE_BLOCK_EQUATION`](crate::linear_solution::FREE_BLOCK_EQUATION), copied from the
    /// component.
    #[must_use]
    pub const fn free_block_equation(&self) -> &'static str {
        self.free_block_equation
    }

    /// [`SCC_COMPILE_ATTEMPT`]. Compilation does not identify the model.
    #[must_use]
    pub const fn compile_attempt(&self) -> &'static str {
        self.compile_attempt
    }

    /// Specified-model compilation is not an identification result.
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

    /// Equilibrium mean, in variable order. Clamped variables hold their assigned values.
    #[must_use]
    pub fn mean(&self) -> &[f64] {
        &self.mean
    }

    /// Row-major equilibrium covariance. A clamped variable has a zero row and column.
    #[must_use]
    pub fn covariance(&self) -> &[f64] {
        &self.covariance
    }

    /// Infinity norm of the free structural residual `(I - K) μ - (c + μ_ε)`.
    ///
    /// Clamped rows are assignments and are not part of this norm.
    #[must_use]
    pub const fn residual_inf(&self) -> f64 {
        self.residual_inf
    }

    /// Solved blocks in condensation order.
    #[must_use]
    pub fn blocks(&self) -> &[BlockDiagnostic] {
        &self.blocks
    }

    /// Charge that admitted this compilation.
    #[must_use]
    pub const fn admission(&self) -> AdmissionCost {
        self.admission
    }
}

/// Admission charge for compiling `model` under `clamp`.
///
/// The charge depends on the free-block schedule, not on the numeric values. It is
/// checked before any factorization.
///
/// # Errors
///
/// [`SccCompileError::ClampDimension`], or [`SccCompileError::Shape`] when a block index or the
/// charge does not fit.
#[must_use = "an admission error refuses the budget before a solve"]
pub fn gaussian_equilibrium_admission(
    model: &LinearStructuralMap,
    clamp: &HardClamp,
) -> Result<AdmissionCost, SccCompileError> {
    Ok(schedule(model, clamp)?.cost)
}

/// Byte upper bound on the peak dense storage of compiling one world of a model of `order`
/// variables, whatever the clamp. This is the `bytes` of every [`AdmissionCost`] for that order.
///
/// # Errors
///
/// [`SccCompileError::Shape`] when the order or the bound does not fit in `u64`.
pub fn gaussian_equilibrium_peak_bytes(order: usize) -> Result<u64, SccCompileError> {
    plan_bytes(
        u64::try_from(order).map_err(|_| SccCompileError::Shape {
            message: "model dimension does not fit in u64",
        })?,
    )
}

/// Admission work of routing the free support of a model of `order` variables once: the
/// part of every [`AdmissionCost::work`] spent before any factorization.
///
/// # Errors
///
/// [`SccCompileError::Shape`] when the order or the work does not fit in `u64`.
pub fn gaussian_equilibrium_route_work(order: usize) -> Result<u64, SccCompileError> {
    route_work(model_order_u64(order)?)
}

fn model_order_u64(order: usize) -> Result<u64, SccCompileError> {
    u64::try_from(order)
        .map_err(|_| SccCompileError::Shape { message: "model dimension does not fit in u64" })
}

/// Compile the Gaussian equilibrium of `component`'s linear map.
///
/// `clamp_values[k]` is the constant for `clamp.indices()[k]`. An empty clamp compiles the
/// observational equilibrium. The component's source id is not read and is not evidence.
///
/// # Errors
///
/// A component bound to a different model, a dimension mismatch, a non-finite or asymmetric or
/// indefinite disturbance, a singular, numerically singular, or non-finite free block, or an
/// admission charge above `budget`. The solvability recheck runs only after the budget admits.
#[must_use = "a compile error is a refusal, not an equilibrium"]
pub fn compile_gaussian_equilibrium(
    component: &CyclicComponent,
    model: &LinearStructuralMap,
    disturbance: &GaussianDisturbance,
    clamp: &HardClamp,
    clamp_values: &[f64],
    budget: CompileBudget,
) -> Result<CompiledEquilibrium, SccCompileError> {
    let n = model.dimension();
    let binding = component.model_binding();
    if binding.dimension() != n {
        return Err(SccCompileError::ComponentDimension {
            component: binding.dimension(),
            model: n,
        });
    }
    if binding != ModelBinding::of(model) {
        return Err(SccCompileError::ComponentModelMismatch {
            component: component.id().to_owned(),
        });
    }
    compile_checked(component.id(), model, disturbance, clamp, clamp_values, budget)
}

/// The compilation after the component has been checked against the model.
///
/// Kept separate so the refusals of the solve itself (a numerically singular block, for one)
/// stay reachable for models that the family certificate would not admit.
fn compile_checked(
    component_id: &str,
    model: &LinearStructuralMap,
    disturbance: &GaussianDisturbance,
    clamp: &HardClamp,
    clamp_values: &[f64],
    budget: CompileBudget,
) -> Result<CompiledEquilibrium, SccCompileError> {
    reject_compile_inputs(model, disturbance, clamp, clamp_values)?;
    let planned = schedule(model, clamp)?;
    let routed_work = route_work(model_dimension_u64(model)?)?;
    finish_compile(
        component_id,
        model,
        disturbance,
        (clamp, clamp_values),
        budget,
        (&planned.blocks, routed_work),
        SCC_COMPILE_ATTEMPT,
    )
}

fn model_dimension_u64(model: &LinearStructuralMap) -> Result<u64, SccCompileError> {
    u64::try_from(model.dimension())
        .map_err(|_| SccCompileError::Shape { message: "model dimension does not fit in u64" })
}

/// Work the support routing spends before any factorization: one pass over the dense
/// coefficient matrix to build the support graph and one to condense it.
fn route_work(n: u64) -> Result<u64, SccCompileError> {
    n.checked_mul(n).and_then(|square| square.checked_mul(2)).ok_or_else(work_overflow)
}

fn reject_compile_inputs(
    model: &LinearStructuralMap,
    disturbance: &GaussianDisturbance,
    clamp: &HardClamp,
    clamp_values: &[f64],
) -> Result<(), SccCompileError> {
    let n = model.dimension();
    if disturbance.dimension() != n {
        return Err(SccCompileError::DisturbanceDimension {
            disturbance: disturbance.dimension(),
            model: n,
        });
    }
    if clamp_values.len() != clamp.indices().len() {
        return Err(SccCompileError::ClampValues {
            nodes: clamp.indices().len(),
            values: clamp_values.len(),
        });
    }
    for (&node, &value) in clamp.indices().iter().zip(clamp_values) {
        if !value.is_finite() {
            return Err(SccCompileError::NonFiniteClamp { node });
        }
    }
    Ok(())
}

fn finish_compile(
    component_id: &str,
    model: &LinearStructuralMap,
    disturbance: &GaussianDisturbance,
    clamp_binding: (&HardClamp, &[f64]),
    budget: CompileBudget,
    schedule: (&[Vec<usize>], u64),
    attempt: &'static str,
) -> Result<CompiledEquilibrium, SccCompileError> {
    // Routing was charged before the schedule existed; the factorization work adds to it, so
    // the budget bounds the whole compile and the recorded cost is that total.
    let (blocks, routed_work) = schedule;
    let (clamp, clamp_values) = clamp_binding;
    let n = model.dimension();
    let sizes: Vec<u64> = blocks
        .iter()
        .map(|block| u64::try_from(block.len()))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| SccCompileError::Shape { message: "block length does not fit in u64" })?;
    let n_u64 = u64::try_from(n)
        .map_err(|_| SccCompileError::Shape { message: "model dimension does not fit in u64" })?;
    let clamps = u64::try_from(clamp.indices().len())
        .map_err(|_| SccCompileError::Shape { message: "clamp length does not fit in u64" })?;
    let work =
        plan_work(n_u64, clamps, &sizes)?.checked_add(routed_work).ok_or_else(work_overflow)?;
    let cost = AdmissionCost { work, bytes: plan_bytes(n_u64)? };
    if cost.work > budget.max_work {
        return Err(SccCompileError::Budget {
            resource: "work",
            requested: cost.work,
            limit: budget.max_work,
        });
    }
    if cost.bytes > budget.max_bytes {
        return Err(SccCompileError::Budget {
            resource: "bytes",
            requested: cost.bytes,
            limit: budget.max_bytes,
        });
    }
    disturbance.require_psd_once()?;
    rechecked_solvability(model, clamp)?;
    let mut assigned = vec![None; n];
    for (&node, &value) in clamp.indices().iter().zip(clamp_values) {
        assigned[node] = Some(value);
    }
    let solved = propagate(model, disturbance, &assigned, blocks)?;
    Ok(CompiledEquilibrium {
        component_id: component_id.to_owned(),
        free_block_equation: FREE_BLOCK_EQUATION,
        compile_attempt: attempt,
        identification: IdentificationStatus::NotMarkedUnidentified,
        physical_stability: PhysicalStability::Unclaimed,
        solver: SolverConvergence::NotRun,
        mean: solved.mean,
        covariance: solved.covariance,
        residual_inf: solved.residual_inf,
        blocks: solved.blocks,
        admission: cost,
    })
}

/// Refuse a clamp whose free block is singular or non-finite before factorization.
fn rechecked_solvability(
    model: &LinearStructuralMap,
    clamp: &HardClamp,
) -> Result<(), SccCompileError> {
    match revalidate_free_block(model, clamp) {
        Ok(_) => Ok(()),
        Err(LinearSolutionError::FreeBlockNotSolvable { refusal }) => match refusal {
            SolvabilityRefusal::NonFinite => Err(SccCompileError::NonFiniteEquilibrium),
            SolvabilityRefusal::Singular { members }
            | SolvabilityRefusal::NumericallySingular { members } => {
                Err(SccCompileError::NumericallySingularBlock { members })
            }
        },
        Err(LinearSolutionError::ClampDimension { clamp, model }) => {
            Err(SccCompileError::ClampDimension { clamp, model })
        }
        Err(
            LinearSolutionError::Shape { .. }
            | LinearSolutionError::NonzeroDiagonal { .. }
            | LinearSolutionError::NonFiniteCoefficient { .. }
            | LinearSolutionError::NonFiniteIntercept { .. }
            | LinearSolutionError::ClampOutOfRange { .. }
            | LinearSolutionError::DuplicateClamp { .. },
        ) => Err(SccCompileError::Shape {
            message: "free-block solvability recheck was not a clamp of this model",
        }),
    }
}

struct PlannedSchedule {
    blocks: Vec<Vec<usize>>,
    cost: AdmissionCost,
}

fn schedule(
    model: &LinearStructuralMap,
    clamp: &HardClamp,
) -> Result<PlannedSchedule, SccCompileError> {
    let n = model.dimension();
    if clamp.dimension() != n {
        return Err(SccCompileError::ClampDimension { clamp: clamp.dimension(), model: n });
    }
    let free = free_indices(n, clamp.indices());
    let blocks = support_blocks(model, &free)?;
    let sizes: Vec<u64> = blocks
        .iter()
        .map(|block| u64::try_from(block.len()))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| SccCompileError::Shape { message: "block length does not fit in u64" })?;
    let n_u64 = u64::try_from(n)
        .map_err(|_| SccCompileError::Shape { message: "model dimension does not fit in u64" })?;
    let clamps = u64::try_from(clamp.indices().len())
        .map_err(|_| SccCompileError::Shape { message: "clamp length does not fit in u64" })?;
    let work = plan_work(n_u64, clamps, &sizes)?
        .checked_add(route_work(n_u64)?)
        .ok_or_else(work_overflow)?;
    let bytes = plan_bytes(n_u64)?;
    Ok(PlannedSchedule { blocks, cost: AdmissionCost { work, bytes } })
}

/// Strongly connected blocks of the free support, sources before sinks, as global indices.
pub(crate) fn support_blocks(
    model: &LinearStructuralMap,
    free: &[usize],
) -> Result<Vec<Vec<usize>>, SccCompileError> {
    let m = free.len();
    let width = u32::try_from(m)
        .map_err(|_| SccCompileError::Shape { message: "free block does not fit in a dense id" })?;
    let mut graph = CyclicDigraph::with_variables(width);
    for (into_local, &into_node) in free.iter().enumerate() {
        for (from_local, &from_node) in free.iter().enumerate() {
            if into_local == from_local {
                continue;
            }
            let coefficient = structural_coefficient(model, into_node, from_node)?;
            #[allow(clippy::float_cmp, reason = "a structural zero is not a directed edge")]
            if coefficient != 0.0 {
                let from = DenseNodeId::try_from_usize(from_local).map_err(|_| {
                    SccCompileError::Shape { message: "free-node index does not fit a dense id" }
                })?;
                let into = DenseNodeId::try_from_usize(into_local).map_err(|_| {
                    SccCompileError::Shape { message: "free-node index does not fit a dense id" }
                })?;
                // `K[into, from]` is the mechanism of `from` into `into`.
                graph.insert_directed(from, into).map_err(|_| SccCompileError::Shape {
                    message: "structural support edge was rejected",
                })?;
            }
        }
    }
    let mut blocks = Vec::new();
    for component in graph.condensation().components() {
        let mut members = Vec::with_capacity(component.len());
        for id in component {
            let local = id.as_usize();
            let Some(&global) = free.get(local) else {
                return Err(SccCompileError::Shape {
                    message: "condensation member is outside the free set",
                });
            };
            members.push(global);
        }
        blocks.push(members);
    }
    Ok(blocks)
}

impl From<MatrixError> for SccCompileError {
    fn from(error: MatrixError) -> Self {
        match error {
            MatrixError::Overflow => {
                Self::Shape { message: "block workspace length overflows usize" }
            }
            MatrixError::NonFinite => Self::NonFiniteEquilibrium,
        }
    }
}

#[cfg(test)]
mod tests;
