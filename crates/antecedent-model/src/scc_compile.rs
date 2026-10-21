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
//! compilation grants no scientific license.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

use antecedent_graph::{CyclicDigraph, Dag, DenseNodeId};
use antecedent_kernels::gauss_jordan_solve;
use thiserror::Error;

use crate::cyclic_component::{
    CyclicComponent, DAG_EQUILIBRIUM_ATTEMPT, ModelBinding, SCC_COMPILE_ATTEMPT,
};
use crate::linear_solution::{
    FREE_BLOCK_EQUATION, HardClamp, IdentificationStatus, LinearSolutionError, LinearStructuralMap,
    PhysicalStability, SolvabilityRefusal, SolverConvergence, free_indices, pivots_regular,
    revalidate_free_block,
};

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
    /// Admission work for the factorization, the solves, the covariance check, and the
    /// diagnostics.
    #[must_use]
    pub const fn work(self) -> u64 {
        self.work
    }

    /// Byte upper bound charged for the mean, the two covariance states, and the block
    /// workspaces.
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
/// compile budget.
#[derive(Clone, Debug, PartialEq)]
pub struct GaussianDisturbance {
    mean: Vec<f64>,
    covariance: Vec<f64>,
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
        Ok(Self { mean, covariance })
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

/// How the free support of a linear map was scheduled.
///
/// Specified models remain model-conditional. Choosing this schedule grants no
/// scientific license.
#[derive(Clone, Debug, PartialEq)]
pub enum LinearSupportRoute {
    /// Topological order of the free nodes, sources before sinks.
    ///
    /// Each index is one mechanism. None of them is a cyclic block.
    Acyclic {
        /// Free node indices in topological order.
        order: Vec<usize>,
    },
    /// The DAG algorithm refused a directed cycle.
    ///
    /// `next_attempt` is that refusal. `blocks` is the condensation schedule
    /// the cyclic compiler may then run. It is not applied by the DAG algorithm.
    Cyclic {
        /// Existing next attempt from the DAG refusal.
        next_attempt: String,
        /// Strongly connected free blocks, sources before sinks.
        blocks: Vec<Vec<usize>>,
    },
}

/// Equilibrium compiled by [`compile_routed_gaussian_equilibrium`].
#[derive(Clone, Debug, PartialEq)]
pub struct RoutedEquilibrium {
    route: LinearSupportRoute,
    equilibrium: CompiledEquilibrium,
}

impl RoutedEquilibrium {
    /// Schedule the DAG algorithm selected, including a cycle refusal when one was required.
    #[must_use]
    pub fn route(&self) -> &LinearSupportRoute {
        &self.route
    }

    /// Compiled equilibrium. An acyclic route is not a cyclic-block solve.
    pub fn equilibrium(&self) -> &CompiledEquilibrium {
        &self.equilibrium
    }
}

/// Ask the DAG algorithm which schedule the free support uses.
///
/// An acyclic support returns its topological order and does not build a
/// condensation. A directed cycle is refused by [`Dag::insert_directed`]; the
/// error text is the existing next attempt, and the condensation is returned
/// only as the later cyclic schedule.
///
/// # Errors
///
/// A clamp dimension mismatch, or a support edge the graph rejected for a
/// reason other than a directed cycle.
pub fn route_linear_support(
    model: &LinearStructuralMap,
    clamp: &HardClamp,
) -> Result<LinearSupportRoute, SccCompileError> {
    let n = model.dimension();
    if clamp.dimension() != n {
        return Err(SccCompileError::ClampDimension { clamp: clamp.dimension(), model: n });
    }
    let free = free_indices(n, clamp.indices());
    let width = u32::try_from(free.len())
        .map_err(|_| SccCompileError::Shape { message: "free block does not fit in a dense id" })?;
    let mut graph = Dag::with_variables(width);
    for (into_local, &into_node) in free.iter().enumerate() {
        for (from_local, &from_node) in free.iter().enumerate() {
            if into_local == from_local {
                continue;
            }
            let coefficient = structural_coefficient(model, into_node, from_node)?;
            #[allow(clippy::float_cmp, reason = "a structural zero is not a directed edge")]
            if coefficient == 0.0 {
                continue;
            }
            let from = DenseNodeId::try_from_usize(from_local).map_err(|_| {
                SccCompileError::Shape { message: "free-node index does not fit a dense id" }
            })?;
            let into = DenseNodeId::try_from_usize(into_local).map_err(|_| {
                SccCompileError::Shape { message: "free-node index does not fit a dense id" }
            })?;
            if let Err(error) = graph.insert_directed(from, into) {
                return match error {
                    antecedent_graph::error::GraphError::Cycle { .. } => {
                        let blocks = support_blocks(model, &free)?;
                        Ok(LinearSupportRoute::Cyclic { next_attempt: error.to_string(), blocks })
                    }
                    _ => Err(SccCompileError::Shape {
                        message: "structural support edge was rejected",
                    }),
                };
            }
        }
    }
    let Some(order) = graph.topological_order() else {
        let blocks = support_blocks(model, &free)?;
        return Ok(LinearSupportRoute::Cyclic { next_attempt: dag_license_refusal(), blocks });
    };
    let mut global = Vec::with_capacity(order.len());
    for id in order {
        let local = id.as_usize();
        let Some(&node) = free.get(local) else {
            return Err(SccCompileError::Shape {
                message: "topological member is outside the free set",
            });
        };
        global.push(node);
    }
    Ok(LinearSupportRoute::Acyclic { order: global })
}

fn dag_license_refusal() -> String {
    antecedent_graph::error::CyclicDagRefusal {
        operation: antecedent_graph::error::DagOperation::Simulation,
        next_attempt: antecedent_graph::error::CyclicNextAttempt::Unlicensed {
            consumer: "CompiledCausalModel",
            attempt: "a directed cycle belongs in a CyclicDigraph; condensation describes \
                      the feedback and does not sample or abduct. This compiler has no \
                      cyclic license",
        },
    }
    .to_string()
}

/// Compile a Gaussian equilibrium, using the DAG algorithm when the free support is acyclic.
///
/// A directed cycle is refused by that algorithm. The cyclic license then runs
/// the existing strongly connected compiler. Specified models remain
/// model-conditional; compilation grants no scientific license.
///
/// # Errors
///
/// The errors of [`compile_gaussian_equilibrium`]. A directed cycle is compiled
/// by the cyclic license after the DAG refusal is recorded on the route.
pub fn compile_routed_gaussian_equilibrium(
    component: &CyclicComponent,
    model: &LinearStructuralMap,
    disturbance: &GaussianDisturbance,
    clamp: &HardClamp,
    clamp_values: &[f64],
    budget: CompileBudget,
) -> Result<RoutedEquilibrium, SccCompileError> {
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
    reject_compile_inputs(model, disturbance, clamp, clamp_values)?;
    let route = route_linear_support(model, clamp)?;
    let equilibrium = match &route {
        LinearSupportRoute::Acyclic { order } => {
            let blocks: Vec<Vec<usize>> = order.iter().copied().map(|node| vec![node]).collect();
            finish_compile(
                component.id(),
                model,
                disturbance,
                (clamp, clamp_values),
                budget,
                &blocks,
                DAG_EQUILIBRIUM_ATTEMPT,
            )?
        }
        LinearSupportRoute::Cyclic { .. } => {
            let planned = schedule(model, clamp)?;
            finish_compile(
                component.id(),
                model,
                disturbance,
                (clamp, clamp_values),
                budget,
                &planned.blocks,
                SCC_COMPILE_ATTEMPT,
            )?
        }
    };
    Ok(RoutedEquilibrium { route, equilibrium })
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
    finish_compile(
        component_id,
        model,
        disturbance,
        (clamp, clamp_values),
        budget,
        &planned.blocks,
        SCC_COMPILE_ATTEMPT,
    )
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
    blocks: &[Vec<usize>],
    attempt: &'static str,
) -> Result<CompiledEquilibrium, SccCompileError> {
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
    let cost = AdmissionCost { work: plan_work(n_u64, clamps, &sizes)?, bytes: plan_bytes(n_u64)? };
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
    require_psd(disturbance.covariance(), n)?;
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

struct Propagated {
    mean: Vec<f64>,
    covariance: Vec<f64>,
    residual_inf: f64,
    blocks: Vec<BlockDiagnostic>,
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
    let work = plan_work(n_u64, clamps, &sizes)?;
    let bytes = plan_bytes(n_u64)?;
    Ok(PlannedSchedule { blocks, cost: AdmissionCost { work, bytes } })
}

fn work_overflow() -> SccCompileError {
    SccCompileError::Shape { message: "admission work overflows u64" }
}

fn plan_work(n: u64, clamps: u64, blocks: &[u64]) -> Result<u64, SccCompileError> {
    let square = n.checked_mul(n).ok_or_else(work_overflow)?;
    let mut work = square.checked_mul(n).ok_or_else(work_overflow)?;
    work = work.checked_add(square).ok_or_else(work_overflow)?;
    let mut earlier = 0u64;
    for &m in blocks {
        let step = block_work(m, earlier, n, clamps)?;
        let condition = condition_work(m)?;
        work = work
            .checked_add(step)
            .and_then(|sum| sum.checked_add(condition))
            .ok_or_else(work_overflow)?;
        earlier = earlier.checked_add(m).ok_or_else(work_overflow)?;
    }
    Ok(work)
}

fn block_work(m: u64, earlier: u64, n: u64, clamps: u64) -> Result<u64, SccCompileError> {
    let m2 = m.checked_mul(m).ok_or_else(work_overflow)?;
    let factor = m2.checked_mul(m).ok_or_else(work_overflow)?;
    let rhs_cols =
        m.checked_add(earlier).and_then(|sum| sum.checked_add(n)).ok_or_else(work_overflow)?;
    let solves = m2.checked_mul(rhs_cols).ok_or_else(work_overflow)?;
    let span =
        m.checked_add(earlier).and_then(|sum| sum.checked_add(n)).ok_or_else(work_overflow)?;
    let cross =
        m.checked_mul(earlier).and_then(|prod| prod.checked_mul(span)).ok_or_else(work_overflow)?;
    let mean = clamps
        .checked_add(earlier)
        .and_then(|sum| sum.checked_add(1))
        .and_then(|sum| m.checked_mul(sum))
        .ok_or_else(work_overflow)?;
    factor
        .checked_add(solves)
        .and_then(|sum| sum.checked_add(cross))
        .and_then(|sum| sum.checked_add(mean))
        .ok_or_else(work_overflow)
}

fn condition_work(m: u64) -> Result<u64, SccCompileError> {
    m.checked_mul(m).and_then(|square| square.checked_add(16)).ok_or_else(work_overflow)
}

fn byte_overflow() -> SccCompileError {
    SccCompileError::Shape { message: "admission bytes overflow u64" }
}

fn plan_bytes(n: u64) -> Result<u64, SccCompileError> {
    let square = n.checked_mul(n).ok_or_else(byte_overflow)?;
    // `12 n² + 2 n` floats: covariance states and block workspaces, times 8 bytes.
    let mean_and_scratch = n.checked_mul(2).ok_or_else(byte_overflow)?;
    let floats = square
        .checked_mul(12)
        .and_then(|big| big.checked_add(mean_and_scratch))
        .ok_or_else(byte_overflow)?;
    floats.checked_mul(8).ok_or_else(byte_overflow)
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

fn propagate(
    model: &LinearStructuralMap,
    disturbance: &GaussianDisturbance,
    assigned: &[Option<f64>],
    blocks: &[Vec<usize>],
) -> Result<Propagated, SccCompileError> {
    let n = model.dimension();
    let mut mean = vec![0.0; n];
    for (node, value) in assigned.iter().enumerate() {
        if let Some(clamped) = value {
            mean[node] = *clamped;
        }
    }
    let mut covariance = zeros(n, n)?;
    let mut eps_x = zeros(n, n)?;
    let mut diagnostics = Vec::with_capacity(blocks.len());
    let mut earlier: Vec<usize> = Vec::new();
    for members in blocks {
        let diagnostic = solve_block(
            model,
            disturbance,
            assigned,
            members,
            &earlier,
            &mut Moments { mean: &mut mean, covariance: &mut covariance, eps_x: &mut eps_x },
        )?;
        earlier.extend(members.iter().copied());
        diagnostics.push(diagnostic);
    }
    symmetrize(&mut covariance, n)?;
    let mean_bad = mean.iter().any(|value| !value.is_finite());
    let covariance_bad = covariance.iter().any(|value| !value.is_finite());
    if mean_bad || covariance_bad {
        return Err(SccCompileError::NonFiniteEquilibrium);
    }
    let free = blocks.iter().flatten().copied().collect::<Vec<_>>();
    let residual_inf = structural_residual(model, disturbance, &mean, &free)?;
    Ok(Propagated { mean, covariance, residual_inf, blocks: diagnostics })
}

struct Moments<'a> {
    mean: &'a mut [f64],
    covariance: &'a mut [f64],
    eps_x: &'a mut [f64],
}

fn solve_block(
    model: &LinearStructuralMap,
    disturbance: &GaussianDisturbance,
    assigned: &[Option<f64>],
    members: &[usize],
    earlier: &[usize],
    moments: &mut Moments<'_>,
) -> Result<BlockDiagnostic, SccCompileError> {
    let m = members.len();
    let p = earlier.len();
    let n = model.dimension();
    let matrix = eye_minus_block(model, members)?;
    if !pivots_regular(&matrix, m) {
        return Err(SccCompileError::NumericallySingularBlock { members: members.to_vec() });
    }
    let mean_rhs =
        mean_right_hand_side(model, disturbance, assigned, members, earlier, moments.mean)?;
    // Parent coefficients and the parent covariance are shared by the three inputs below.
    let k_fe = parent_coefficients(model, members, earlier)?;
    let c_ee = earlier_covariance(earlier, moments.covariance, n)?;
    let parents = ParentInputs { k_fe: &k_fe, c_ee: &c_ee };
    let cov_u = disturbance_block(disturbance, members, earlier, &parents, moments.eps_x, n)?;
    let cov_u_earlier = cross_input(members, earlier, &parents, moments.eps_x, n)?;
    let cov_u_eps = disturbance_cross(disturbance, members, earlier, &k_fe, moments.eps_x, n)?;
    let cols = batch_columns(m, p, n)?;
    let mut rhs = zeros(m, cols)?;
    let cov_at = 1;
    let cross_at = cov_at + m;
    let eps_at = cross_at + p;
    for row in 0..m {
        let base = row * cols;
        rhs[base] = mean_rhs[row];
        for col in 0..m {
            rhs[base + cov_at + col] = cov_u[row * m + col];
        }
        for col in 0..p {
            rhs[base + cross_at + col] = cov_u_earlier[row * p + col];
        }
        for col in 0..n {
            rhs[base + eps_at + col] = cov_u_eps[row * n + col];
        }
    }
    let solved = solve_row_major(&matrix, &rhs, m, cols, members)?;
    for (local, &node) in members.iter().enumerate() {
        moments.mean[node] = solved[local * cols];
    }
    let mut solved_cov_u = zeros(m, m)?;
    for row in 0..m {
        for col in 0..m {
            solved_cov_u[row * m + col] = solved[row * cols + cov_at + col];
        }
    }
    // `W = A^{-1} Cov(u)`. `Cov(X_S) = W A^{-T} = (A^{-1} W^T)^T`.
    let cov_factor = transpose(&solved_cov_u, m, m)?;
    let solved_cov = solve_row_major(&matrix, &cov_factor, m, m, members)?;
    let cov_block = transpose(&solved_cov, m, m)?;
    for (row, &into_node) in members.iter().enumerate() {
        for (col, &from_node) in members.iter().enumerate() {
            moments.covariance[into_node * n + from_node] = cov_block[row * m + col];
        }
        for (col, &prior) in earlier.iter().enumerate() {
            let cross = solved[row * cols + cross_at + col];
            moments.covariance[into_node * n + prior] = cross;
            moments.covariance[prior * n + into_node] = cross;
        }
        for noise in 0..n {
            moments.eps_x[noise * n + into_node] = solved[row * cols + eps_at + noise];
        }
    }
    let residual_inf = block_residual(&matrix, &solved, cols, &mean_rhs, m)?;
    let (kappa_inf, exact) = condition_of(&matrix, m, members)?;
    Ok(BlockDiagnostic { members: members.to_vec(), residual_inf, kappa_inf, exact })
}

fn batch_columns(m: usize, earlier: usize, n: usize) -> Result<usize, SccCompileError> {
    m.checked_add(earlier)
        .and_then(|sum| sum.checked_add(n))
        .and_then(|sum| sum.checked_add(1))
        .ok_or(SccCompileError::Shape { message: "block right-hand side width overflows usize" })
}

fn mean_right_hand_side(
    model: &LinearStructuralMap,
    disturbance: &GaussianDisturbance,
    assigned: &[Option<f64>],
    members: &[usize],
    earlier: &[usize],
    mean: &[f64],
) -> Result<Vec<f64>, SccCompileError> {
    let n = model.dimension();
    let member_at = positions(n, members)?;
    let earlier_at = positions(n, earlier)?;
    let mut rhs = vec![0.0; members.len()];
    for (local, &node) in members.iter().enumerate() {
        let mut value = model.intercept()[node] + disturbance.mean()[node];
        for other in 0..n {
            if other == node {
                continue;
            }
            let coefficient = structural_coefficient(model, node, other)?;
            #[allow(
                clippy::float_cmp,
                reason = "a structural zero does not enter the right-hand side"
            )]
            if coefficient == 0.0 {
                continue;
            }
            if let Some(clamped) = assigned[other] {
                value += coefficient * clamped;
            } else if earlier_at[other].is_some() {
                value += coefficient * mean[other];
            } else if member_at[other].is_none() {
                return Err(SccCompileError::Shape {
                    message: "structural edge enters a free variable before its block is solved",
                });
            }
        }
        if !value.is_finite() {
            return Err(SccCompileError::NonFiniteEquilibrium);
        }
        rhs[local] = value;
    }
    Ok(rhs)
}

/// Coefficients of the earlier blocks into this block, and the earlier blocks' covariance.
struct ParentInputs<'a> {
    /// `K[members, earlier]`, row-major `m × p`.
    k_fe: &'a [f64],
    /// `Cov(X_earlier)`, row-major `p × p`.
    c_ee: &'a [f64],
}

fn disturbance_block(
    disturbance: &GaussianDisturbance,
    members: &[usize],
    earlier: &[usize],
    parents: &ParentInputs<'_>,
    eps_x: &[f64],
    n: usize,
) -> Result<Vec<f64>, SccCompileError> {
    let m = members.len();
    let p = earlier.len();
    let sigma = disturbance.covariance();
    let mut cov_u = zeros(m, m)?;
    for (row, &into_node) in members.iter().enumerate() {
        for (col, &from_node) in members.iter().enumerate() {
            cov_u[row * m + col] = sigma[into_node * n + from_node];
        }
    }
    if p == 0 {
        return Ok(cov_u);
    }
    let mut c_ex = zeros(m, p)?;
    for (row, &into_node) in members.iter().enumerate() {
        for (col, &prior) in earlier.iter().enumerate() {
            c_ex[row * p + col] = eps_x[into_node * n + prior];
        }
    }
    // `Cov(u) = Σ_SS + K C(ε, X)^T + C(ε, X) K^T + K C(X) K^T`.
    add_times_transpose(parents.k_fe, &c_ex, &mut cov_u, m, p)?;
    add_times_transpose(&c_ex, parents.k_fe, &mut cov_u, m, p)?;
    let through_parents = matmul(parents.k_fe, m, p, parents.c_ee, p)?;
    add_times_transpose(&through_parents, parents.k_fe, &mut cov_u, m, p)?;
    finite_matrix(&cov_u)?;
    Ok(cov_u)
}

fn cross_input(
    members: &[usize],
    earlier: &[usize],
    parents: &ParentInputs<'_>,
    eps_x: &[f64],
    n: usize,
) -> Result<Vec<f64>, SccCompileError> {
    let m = members.len();
    let p = earlier.len();
    if p == 0 {
        return Ok(Vec::new());
    }
    let mut cov_u = zeros(m, p)?;
    for (row, &into_node) in members.iter().enumerate() {
        for (col, &prior) in earlier.iter().enumerate() {
            cov_u[row * p + col] = eps_x[into_node * n + prior];
        }
    }
    let through_parents = matmul(parents.k_fe, m, p, parents.c_ee, p)?;
    for (index, value) in through_parents.iter().copied().enumerate() {
        cov_u[index] += value;
    }
    finite_matrix(&cov_u)?;
    Ok(cov_u)
}

fn disturbance_cross(
    disturbance: &GaussianDisturbance,
    members: &[usize],
    earlier: &[usize],
    k_fe: &[f64],
    eps_x: &[f64],
    n: usize,
) -> Result<Vec<f64>, SccCompileError> {
    let m = members.len();
    let p = earlier.len();
    let sigma = disturbance.covariance();
    let mut cov_u = zeros(m, n)?;
    for (row, &into_node) in members.iter().enumerate() {
        for noise in 0..n {
            let mut value = sigma[into_node * n + noise];
            for (col, &prior) in earlier.iter().enumerate() {
                value += k_fe[row * p + col] * eps_x[noise * n + prior];
            }
            cov_u[row * n + noise] = value;
        }
    }
    finite_matrix(&cov_u)?;
    Ok(cov_u)
}

fn parent_coefficients(
    model: &LinearStructuralMap,
    members: &[usize],
    earlier: &[usize],
) -> Result<Vec<f64>, SccCompileError> {
    let m = members.len();
    let p = earlier.len();
    let mut coefficients = zeros(m, p)?;
    for (row, &into_node) in members.iter().enumerate() {
        for (col, &prior) in earlier.iter().enumerate() {
            coefficients[row * p + col] = structural_coefficient(model, into_node, prior)?;
        }
    }
    Ok(coefficients)
}

fn earlier_covariance(
    earlier: &[usize],
    covariance: &[f64],
    n: usize,
) -> Result<Vec<f64>, SccCompileError> {
    let p = earlier.len();
    let mut c_ee = zeros(p, p)?;
    for (row, &left) in earlier.iter().enumerate() {
        for (col, &right) in earlier.iter().enumerate() {
            c_ee[row * p + col] = covariance[left * n + right];
        }
    }
    Ok(c_ee)
}

/// `I - K_SS` for a block of the free support. `LinearStructuralMap::try_new` guarantees the
/// zero diagonal, so the diagonal here is exactly one.
pub(crate) fn eye_minus_block(
    model: &LinearStructuralMap,
    members: &[usize],
) -> Result<Vec<f64>, SccCompileError> {
    let m = members.len();
    let mut matrix = zeros(m, m)?;
    for (row, &into_node) in members.iter().enumerate() {
        for (col, &from_node) in members.iter().enumerate() {
            matrix[row * m + col] = if row == col {
                1.0
            } else {
                -structural_coefficient(model, into_node, from_node)?
            };
        }
    }
    finite_matrix(&matrix)?;
    Ok(matrix)
}

fn condition_of(
    matrix: &[f64],
    m: usize,
    members: &[usize],
) -> Result<(f64, bool), SccCompileError> {
    if m == 0 {
        return Ok((1.0, true));
    }
    if let Some(kappa) = exact_infinity_kappa(matrix, m) {
        return Ok((kappa, true));
    }
    let transposed = transpose(matrix, m, m)?;
    let inverse_inf = hager_one_norm_inverse(&transposed, members)?;
    let kappa = infinity_norm(matrix, m) * inverse_inf;
    if !kappa.is_finite() {
        return Err(SccCompileError::NonFiniteEquilibrium);
    }
    Ok((kappa, false))
}

/// Hager's 1-norm estimate of `||A^{-1}||_1` (Higham, 1988).
///
/// Applied to `A^T`, the estimate is `||A^{-1}||_∞`.
fn hager_one_norm_inverse(matrix: &[f64], members: &[usize]) -> Result<f64, SccCompileError> {
    let m = members.len();
    let Ok(count) = u32::try_from(m) else {
        return Err(SccCompileError::NonFiniteEquilibrium);
    };
    if count == 0 {
        return Err(SccCompileError::NonFiniteEquilibrium);
    }
    let mut x = vec![1.0 / f64::from(count); m];
    let transposed = transpose(matrix, m, m)?;
    let mut estimate = 0.0_f64;
    for _step in 0..4 {
        let y = solve_vector(matrix, &x, members)?;
        let norm_y = y.iter().fold(0.0_f64, |acc, value| acc + value.abs());
        if norm_y <= estimate {
            break;
        }
        estimate = norm_y;
        let mut xi = Vec::with_capacity(m);
        for value in &y {
            xi.push(if *value >= 0.0 { 1.0 } else { -1.0 });
        }
        let z = solve_vector(&transposed, &xi, members)?;
        let mut dot = 0.0_f64;
        let mut peak = 0.0_f64;
        let mut peak_at = 0_usize;
        for (index, value) in z.iter().enumerate() {
            dot += value * x[index];
            if value.abs() > peak {
                peak = value.abs();
                peak_at = index;
            }
        }
        if peak <= dot {
            break;
        }
        x = vec![0.0; m];
        x[peak_at] = 1.0;
    }
    if !estimate.is_finite() {
        return Err(SccCompileError::NonFiniteEquilibrium);
    }
    Ok(estimate)
}

fn solve_vector(
    matrix: &[f64],
    rhs: &[f64],
    members: &[usize],
) -> Result<Vec<f64>, SccCompileError> {
    let m = members.len();
    solve_row_major(matrix, rhs, m, 1, members)
}

fn exact_infinity_kappa(matrix: &[f64], m: usize) -> Option<f64> {
    match m {
        1 => Some(1.0),
        2 => {
            let row = (1.0 + matrix[1].abs()).max(1.0 + matrix[2].abs());
            let det = (1.0 - matrix[1] * matrix[2]).abs();
            #[allow(
                clippy::float_cmp,
                reason = "a zero determinant has already been refused; this guards the division"
            )]
            if det == 0.0 || !row.is_finite() {
                None
            } else {
                let kappa = (row * row) / det;
                kappa.is_finite().then_some(kappa)
            }
        }
        _ => None,
    }
}

fn infinity_norm(matrix: &[f64], m: usize) -> f64 {
    matrix.chunks_exact(m).fold(0.0, |best, row| {
        let sum = row.iter().fold(0.0, |acc, entry| acc + entry.abs());
        best.max(sum)
    })
}

fn block_residual(
    matrix: &[f64],
    solved: &[f64],
    cols: usize,
    rhs: &[f64],
    m: usize,
) -> Result<f64, SccCompileError> {
    let mut residual = 0.0_f64;
    for row in 0..m {
        let mut image = 0.0;
        for col in 0..m {
            image += matrix[row * m + col] * solved[col * cols];
        }
        if !image.is_finite() {
            return Err(SccCompileError::NonFiniteEquilibrium);
        }
        residual = residual.max((image - rhs[row]).abs());
    }
    Ok(residual)
}

fn structural_residual(
    model: &LinearStructuralMap,
    disturbance: &GaussianDisturbance,
    mean: &[f64],
    free: &[usize],
) -> Result<f64, SccCompileError> {
    let n = model.dimension();
    let mut residual = 0.0_f64;
    for &node in free {
        let mut image = mean[node] - model.intercept()[node] - disturbance.mean()[node];
        for (other, &value) in mean.iter().enumerate().take(n) {
            if other == node {
                continue;
            }
            image -= structural_coefficient(model, node, other)? * value;
        }
        if !image.is_finite() {
            return Err(SccCompileError::NonFiniteEquilibrium);
        }
        residual = residual.max(image.abs());
    }
    Ok(residual)
}

fn solve_row_major(
    matrix: &[f64],
    rhs_row_major: &[f64],
    m: usize,
    cols: usize,
    members: &[usize],
) -> Result<Vec<f64>, SccCompileError> {
    if m == 0 {
        return Ok(Vec::new());
    }
    let mut factor = matrix.to_vec();
    let mut packed = zeros(cols, m)?;
    for col in 0..cols {
        for row in 0..m {
            packed[col * m + row] = rhs_row_major[row * cols + col];
        }
    }
    if !gauss_jordan_solve(&mut factor, &mut packed, m, cols) {
        return Err(SccCompileError::NumericallySingularBlock { members: members.to_vec() });
    }
    let mut solved = zeros(m, cols)?;
    for col in 0..cols {
        for row in 0..m {
            let value = packed[col * m + row];
            if !value.is_finite() {
                return Err(SccCompileError::NonFiniteEquilibrium);
            }
            solved[row * cols + col] = value;
        }
    }
    Ok(solved)
}

/// Refuse a covariance that is not positive semidefinite.
///
/// The test runs on the correlation matrix `R = D^{-1/2} Σ D^{-1/2}` with
/// `D = diag(Σ)`, so the tolerance is relative to each variable's own scale rather than
/// to the largest entry of `Σ`. A variance that is negative beyond a relative tolerance of
/// its own row is refused, so `diag(1e6, -1e-9)` is not hidden by the `1e6` entry. A
/// variable of zero variance must have a zero row.
#[allow(clippy::float_cmp, reason = "a zero-variance row must be exactly zero off the diagonal")]
fn require_psd(covariance: &[f64], n: usize) -> Result<(), SccCompileError> {
    let tolerance = antecedent_kernels::EQUILIBRATED_PIVOT_FLOOR;
    let mut scale = vec![0.0_f64; n];
    for row in 0..n {
        let variance = covariance[row * n + row];
        let row_scale = (0..n).fold(0.0_f64, |best, col| best.max(covariance[row * n + col].abs()));
        if variance < -tolerance * row_scale {
            return Err(SccCompileError::NotPositiveSemidefinite { index: row });
        }
        scale[row] = if variance > 0.0 { variance.sqrt() } else { 0.0 };
    }
    let mut corr = zeros(n, n)?;
    for row in 0..n {
        for col in 0..n {
            let entry = covariance[row * n + col];
            if scale[row] > 0.0 && scale[col] > 0.0 {
                corr[row * n + col] = entry / scale[row] / scale[col];
            } else if entry != 0.0 && row != col {
                // A zero-variance variable cannot covary with anything.
                return Err(SccCompileError::NotPositiveSemidefinite { index: row.max(col) });
            }
        }
    }
    let mut factor = zeros(n, n)?;
    for row in 0..n {
        for col in 0..row {
            let mut sum = corr[row * n + col];
            for k in 0..col {
                sum -= factor[row * n + k] * factor[col * n + k];
            }
            if !sum.is_finite() {
                return Err(SccCompileError::NonFiniteEquilibrium);
            }
            let diag = factor[col * n + col];
            if diag > 0.0 {
                factor[row * n + col] = sum / diag;
            } else if sum.abs() > tolerance {
                return Err(SccCompileError::NotPositiveSemidefinite { index: row });
            } else {
                factor[row * n + col] = 0.0;
            }
        }
        let mut diag = corr[row * n + row];
        for k in 0..row {
            diag -= factor[row * n + k] * factor[row * n + k];
        }
        if !diag.is_finite() {
            return Err(SccCompileError::NonFiniteEquilibrium);
        }
        if diag < -tolerance {
            return Err(SccCompileError::NotPositiveSemidefinite { index: row });
        }
        factor[row * n + row] = if diag > 0.0 { diag.sqrt() } else { 0.0 };
    }
    Ok(())
}

fn add_times_transpose(
    left: &[f64],
    right: &[f64],
    out: &mut [f64],
    rows: usize,
    mid: usize,
) -> Result<(), SccCompileError> {
    for row in 0..rows {
        for col in 0..rows {
            let mut sum = 0.0;
            for k in 0..mid {
                sum += left[row * mid + k] * right[col * mid + k];
            }
            if !sum.is_finite() {
                return Err(SccCompileError::NonFiniteEquilibrium);
            }
            out[row * rows + col] += sum;
        }
    }
    Ok(())
}

fn matmul(
    left: &[f64],
    rows: usize,
    mid: usize,
    right: &[f64],
    cols: usize,
) -> Result<Vec<f64>, SccCompileError> {
    let mut out = zeros(rows, cols)?;
    for row in 0..rows {
        for col in 0..cols {
            let mut sum = 0.0;
            for k in 0..mid {
                sum += left[row * mid + k] * right[k * cols + col];
            }
            if !sum.is_finite() {
                return Err(SccCompileError::NonFiniteEquilibrium);
            }
            out[row * cols + col] = sum;
        }
    }
    Ok(out)
}

fn transpose(matrix: &[f64], rows: usize, cols: usize) -> Result<Vec<f64>, SccCompileError> {
    let mut out = zeros(cols, rows)?;
    for row in 0..rows {
        for col in 0..cols {
            out[col * rows + row] = matrix[row * cols + col];
        }
    }
    Ok(out)
}

fn symmetrize(covariance: &mut [f64], n: usize) -> Result<(), SccCompileError> {
    for row in 0..n {
        for col in (row + 1)..n {
            let value = 0.5 * (covariance[row * n + col] + covariance[col * n + row]);
            if !value.is_finite() {
                return Err(SccCompileError::NonFiniteEquilibrium);
            }
            covariance[row * n + col] = value;
            covariance[col * n + row] = value;
        }
    }
    Ok(())
}

fn finite_matrix(matrix: &[f64]) -> Result<(), SccCompileError> {
    if matrix.iter().any(|entry| !entry.is_finite()) {
        return Err(SccCompileError::NonFiniteEquilibrium);
    }
    Ok(())
}

fn structural_coefficient(
    model: &LinearStructuralMap,
    into_node: usize,
    from_node: usize,
) -> Result<f64, SccCompileError> {
    model.coefficient(into_node, from_node).ok_or(SccCompileError::Shape {
        message: "structural coefficient index is outside the model",
    })
}

fn positions(n: usize, nodes: &[usize]) -> Result<Vec<Option<usize>>, SccCompileError> {
    let mut pos = vec![None; n];
    for (index, &node) in nodes.iter().enumerate() {
        if node >= n {
            return Err(SccCompileError::Shape { message: "block member is outside the model" });
        }
        pos[node] = Some(index);
    }
    Ok(pos)
}

fn zeros(rows: usize, cols: usize) -> Result<Vec<f64>, SccCompileError> {
    let len = rows
        .checked_mul(cols)
        .ok_or(SccCompileError::Shape { message: "block workspace length overflows usize" })?;
    Ok(vec![0.0; len])
}

#[cfg(test)]
mod tests {
    use super::{
        CompileBudget, GaussianDisturbance, LinearSupportRoute, SccCompileError,
        compile_gaussian_equilibrium, compile_routed_gaussian_equilibrium,
        gaussian_equilibrium_admission,
    };
    use crate::cyclic_component::{CyclicComponent, DAG_EQUILIBRIUM_ATTEMPT, SCC_COMPILE_ATTEMPT};
    use crate::linear_solution::{
        FREE_BLOCK_EQUATION, HardClamp, IdentificationStatus, LinearSolutionError,
        LinearStructuralMap, NONZERO_DIAGONAL_NEXT_ATTEMPT, PhysicalStability, SolverConvergence,
        Uniqueness, assess,
    };

    fn close(left: f64, right: f64) -> bool {
        (left - right).abs() <= 1e-8
    }

    fn map(coefficients: Vec<f64>, intercept: Vec<f64>) -> LinearStructuralMap {
        LinearStructuralMap::try_new(coefficients, intercept).expect("fixture map")
    }

    /// Component admitted for `model` under the family certificate it yields. Only models
    /// with `ρ(|K|) < 1` can be wrapped.
    fn component_for(model: &LinearStructuralMap) -> CyclicComponent {
        let family = assess(model, &HardClamp::none(model.dimension()).expect("no clamp"))
            .expect("assessment")
            .family;
        CyclicComponent::try_new("feedback", "source-measurements", &["solve-note"], model, &family)
            .expect("component")
    }

    fn compile(
        model: &LinearStructuralMap,
        mean: Vec<f64>,
        covariance: Vec<f64>,
        clamp: &HardClamp,
        values: &[f64],
    ) -> super::CompiledEquilibrium {
        let disturbance = GaussianDisturbance::try_new(mean, covariance).expect("disturbance");
        compile_gaussian_equilibrium(
            &component_for(model),
            model,
            &disturbance,
            clamp,
            values,
            CompileBudget::unlimited(),
        )
        .expect("equilibrium")
    }

    #[test]
    fn two_node_cycle_matches_the_hand_solution() {
        // A = 0.4 B + ε_A, B = 0.5 A + ε_B. Indices: A = 0, B = 1.
        // I - K = [[1, -0.4], [-0.5, 1]], det = 0.8.
        // (I - K)^{-1} = [[1.25, 0.5], [0.625, 1.25]].
        // With μ_ε = (1, 0) and c = 0, μ = (1.25, 0.625).
        // With Σ = I, Cov = (I - K)^{-1}(I - K)^{-T}
        //         = [[1.8125, 1.40625], [1.40625, 1.953125]].
        // ||I - K||∞ = 1.5, ||(I - K)^{-1}||∞ = 1.875, κ∞ = 2.8125.
        let model = map(vec![0.0, 0.4, 0.5, 0.0], vec![0.0, 0.0]);
        let clamp = HardClamp::none(2).expect("no clamp");
        let cost = gaussian_equilibrium_admission(&model, &clamp).expect("admission");
        // n = 2, one block of order 2, no clamps:
        // PSD n³ = 8, residual n² = 4,
        // block = m³ + m²(m + n) + m = 8 + 16 + 2 = 26, condition = m² + 16 = 20.
        assert_eq!(cost.work(), 58);
        assert_eq!(cost.bytes(), 416);

        let law = compile(&model, vec![1.0, 0.0], vec![1.0, 0.0, 0.0, 1.0], &clamp, &[]);
        assert_eq!(law.component_id(), "feedback");
        assert_eq!(law.free_block_equation(), FREE_BLOCK_EQUATION);
        assert_eq!(law.compile_attempt(), SCC_COMPILE_ATTEMPT);
        assert_eq!(component_for(&model).scc_compile_attempt(), SCC_COMPILE_ATTEMPT);
        assert_eq!(law.identification(), IdentificationStatus::NotMarkedUnidentified);
        assert_eq!(law.physical_stability(), PhysicalStability::Unclaimed);
        assert_eq!(law.solver(), SolverConvergence::NotRun);
        assert_eq!(law.admission(), cost);
        let wrapped = component_for(&model);
        assert!(!wrapped.evidence_may_certify(wrapped.source_id()));

        assert!(close(law.mean()[0], 1.25));
        assert!(close(law.mean()[1], 0.625));
        assert!(close(law.covariance()[0], 1.8125));
        assert!(close(law.covariance()[1], 1.40625));
        assert!(close(law.covariance()[3], 1.953_125));
        assert!(close(law.covariance()[1] / law.covariance()[0], 45.0 / 58.0));
        assert!(law.residual_inf() < 1e-8);
        assert_eq!(law.blocks().len(), 1);
        assert_eq!(law.blocks()[0].members, vec![0, 1]);
        assert!(law.blocks()[0].exact);
        assert!(close(law.blocks()[0].kappa_inf, 2.8125));
        assert!(law.blocks()[0].residual_inf < 1e-8);
    }

    #[test]
    fn cross_block_disturbance_covariance_is_kept_without_a_structural_edge() {
        // Two singleton components and Σ_01 = 0.3. Dropping the cross entry would
        // report Cov(X_0, X_1) = 0.
        let model = map(vec![0.0, 0.0, 0.0, 0.0], vec![0.0, 0.0]);
        let clamp = HardClamp::none(2).expect("no clamp");
        let law = compile(&model, vec![0.0, 0.0], vec![1.0, 0.3, 0.3, 1.0], &clamp, &[]);
        assert_eq!(law.blocks().len(), 2);
        assert_eq!(law.blocks()[0].members, vec![0]);
        assert_eq!(law.blocks()[1].members, vec![1]);
        assert!(close(law.covariance()[0], 1.0));
        assert!(close(law.covariance()[1], 0.3));
        assert!(close(law.covariance()[3], 1.0));
        assert!(law.residual_inf() < 1e-8);
    }

    #[test]
    fn chain_keeps_structural_and_disturbance_cross_covariance() {
        // X0 = 0.5 + ε0, X1 = 0.25 + 0.5 X0 + ε1.
        // μ_ε = (1, -2), Σ = [[1, 0.3], [0.3, 1]].
        // μ = (1.5, -1), Cov(X0, X1) = 0.8, Var(X1) = 1.55.
        // Independence of the disturbances would give Cov = 0.5 and Var(X1) = 1.25.
        let model = map(vec![0.0, 0.0, 0.5, 0.0], vec![0.5, 0.25]);
        let clamp = HardClamp::none(2).expect("no clamp");
        let law = compile(&model, vec![1.0, -2.0], vec![1.0, 0.3, 0.3, 1.0], &clamp, &[]);
        assert_eq!(law.blocks()[0].members, vec![0]);
        assert_eq!(law.blocks()[1].members, vec![1]);
        assert!(close(law.mean()[0], 1.5));
        assert!(close(law.mean()[1], -1.0));
        assert!(close(law.covariance()[0], 1.0));
        assert!(close(law.covariance()[1], 0.8));
        assert!(close(law.covariance()[3], 1.55));
        assert!(law.residual_inf() < 1e-8);
    }

    #[test]
    fn upstream_parent_of_a_two_node_cycle_matches_the_hand_solution() {
        // X0 = ε0, X1 = 0.2 X0 + 0.4 X2 + ε1, X2 = 0.5 X1 + ε2, Σ = I, c = 0.
        // The cycle inverse is [[1.25, 0.5], [0.625, 1.25]], so
        // X1 = 0.25 ε0 + 1.25 ε1 + 0.5 ε2,
        // X2 = 0.125 ε0 + 0.625 ε1 + 1.25 ε2.
        // Cov(X0, X1) = 0.25, Cov(X0, X2) = 0.125,
        // Var(X1) = 1.875, Cov(X1, X2) = 1.4375, Var(X2) = 1.96875.
        let model = map(vec![0.0, 0.0, 0.0, 0.2, 0.0, 0.4, 0.0, 0.5, 0.0], vec![0.0, 0.0, 0.0]);
        let clamp = HardClamp::none(3).expect("no clamp");
        let law = compile(
            &model,
            vec![0.0; 3],
            vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            &clamp,
            &[],
        );
        assert_eq!(law.blocks().len(), 2);
        assert_eq!(law.blocks()[0].members, vec![0]);
        assert_eq!(law.blocks()[1].members, vec![1, 2]);
        let cov = law.covariance();
        assert!(close(cov[0], 1.0));
        assert!(close(cov[1], 0.25));
        assert!(close(cov[2], 0.125));
        assert!(close(cov[4], 1.875));
        assert!(close(cov[5], 1.4375));
        assert!(close(cov[8], 1.96875));
        assert!(law.residual_inf() < 1e-8);
        assert!(law.blocks()[1].exact);
        assert!(close(law.blocks()[1].kappa_inf, 2.8125));
    }

    #[test]
    fn hard_clamp_keeps_the_marginal_free_disturbance() {
        // do(A = 2) on A = 0.4 B + ε_A, B = 0.5 A + ε_B, with Cov(ε_A, ε_B) = 0.3.
        // B = 1 + ε_B. Var(B) stays 1, not the conditional variance 0.91.
        let model = map(vec![0.0, 0.4, 0.5, 0.0], vec![0.0, 0.0]);
        let clamp = HardClamp::try_new(2, &[0]).expect("clamp A");
        let law = compile(&model, vec![0.0, 0.0], vec![1.0, 0.3, 0.3, 1.0], &clamp, &[2.0]);
        assert!(close(law.mean()[0], 2.0));
        assert!(close(law.mean()[1], 1.0));
        assert!(close(law.covariance()[0], 0.0));
        assert!(close(law.covariance()[1], 0.0));
        assert!(close(law.covariance()[3], 1.0));
        assert_eq!(law.blocks().len(), 1);
        assert_eq!(law.blocks()[0].members, vec![1]);
        assert!(law.residual_inf() < 1e-8);

        let both = HardClamp::try_new(2, &[1, 0]).expect("clamp both");
        let assigned = compile(&model, vec![0.0, 0.0], identity(2), &both, &[4.0, 5.0]);
        assert!(assigned.blocks().is_empty());
        assert!(close(assigned.mean()[0], 4.0));
        assert!(close(assigned.mean()[1], 5.0));
        assert!(assigned.covariance().iter().all(|entry| close(*entry, 0.0)));
        assert!(close(assigned.residual_inf(), 0.0));
    }

    /// Compile without the component gate, for models the family certificate would not admit.
    fn compile_unbound(
        model: &LinearStructuralMap,
        covariance: Vec<f64>,
        clamp: &HardClamp,
        values: &[f64],
    ) -> Result<super::CompiledEquilibrium, SccCompileError> {
        let n = model.dimension();
        let noise = GaussianDisturbance::try_new(vec![0.0; n], covariance).expect("noise");
        super::compile_checked("unbound", model, &noise, clamp, values, CompileBudget::unlimited())
    }

    #[test]
    fn singular_blocks_are_refused() {
        let cycle = map(vec![0.0, 1.0, 1.0, 0.0], vec![0.0, 0.0]);
        let err = compile_unbound(&cycle, identity(2), &HardClamp::none(2).expect("none"), &[])
            .expect_err("singular cycle");
        assert!(
            matches!(err, SccCompileError::NumericallySingularBlock { members } if members == [0, 1])
        );

        // A = B + C + ε_A, B = A + ε_B, C = A + ε_C. do(C = 0) leaves a singular pair.
        let overlapping =
            map(vec![0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![0.0, 0.0, 0.0]);
        let clamp_c = HardClamp::try_new(3, &[2]).expect("clamp C");
        let err = compile_unbound(&overlapping, identity(3), &clamp_c, &[0.0])
            .expect_err("singular free block");
        assert!(
            matches!(err, SccCompileError::NumericallySingularBlock { members } if members == [0, 1])
        );

        let mut coefficients = vec![0.0; 16];
        for node in 0..4 {
            coefficients[node * 4 + (node + 1) % 4] = 1.0;
        }
        let four = map(coefficients, vec![0.0; 4]);
        let err = compile_unbound(&four, identity(4), &HardClamp::none(4).expect("none"), &[])
            .expect_err("singular 4-cycle");
        assert!(matches!(
            err,
            SccCompileError::NumericallySingularBlock { members } if members == [0, 1, 2, 3]
        ));
    }

    #[test]
    fn assess_and_compile_agree_on_a_pivot_below_the_floor() {
        // I - K = [[1, -1], [-(1 + 1e-13), 1]]: det = -1e-13 is not zero, but the second
        // pivot after row exchange is about 1e-13, below the 1e-12 floor.
        let near = map(vec![0.0, 1.0, 1.0 + 1.0e-13, 0.0], vec![0.0, 0.0]);
        let clamp = HardClamp::none(2).expect("none");
        let report = assess(&near, &clamp).expect("assessment");
        assert_eq!(
            report.original.uniqueness,
            Uniqueness::NumericallySingular { members: vec![0, 1] }
        );
        let err = compile_unbound(&near, identity(2), &clamp, &[]).expect_err("compile");
        assert_eq!(err, SccCompileError::NumericallySingularBlock { members: vec![0, 1] });
        // Neither route may call the block clean: the family bound also refuses it.
        assert!(!report.family.is_admitted());

        // A pivot above the floor is accepted by both.
        let ok = map(vec![0.0, 1.0, 1.0 - 1.0e-9, 0.0], vec![0.0, 0.0]);
        let report = assess(&ok, &clamp).expect("assessment");
        assert!(matches!(report.original.uniqueness, Uniqueness::IllConditioned { .. }));
        let law = compile_unbound(&ok, identity(2), &clamp, &[]).expect("compile");
        assert!(law.blocks()[0].kappa_inf > 1.0e9);
    }

    #[test]
    fn a_component_is_bound_to_its_model() {
        let model = map(vec![0.0, 0.4, 0.5, 0.0], vec![0.0, 0.0]);
        let wrapped = component_for(&model);
        let noise = GaussianDisturbance::try_new(vec![0.0; 2], identity(2)).expect("noise");
        let clamp = HardClamp::none(2).expect("none");

        let reweighted = map(vec![0.0, 0.4, 0.6, 0.0], vec![0.0, 0.0]);
        let err = compile_gaussian_equilibrium(
            &wrapped,
            &reweighted,
            &noise,
            &clamp,
            &[],
            CompileBudget::unlimited(),
        )
        .expect_err("different coefficients");
        assert_eq!(
            err,
            SccCompileError::ComponentModelMismatch { component: "feedback".to_owned() }
        );

        let larger = map(vec![0.0; 9], vec![0.0; 3]);
        let noise3 = GaussianDisturbance::try_new(vec![0.0; 3], identity(3)).expect("noise");
        let err = compile_gaussian_equilibrium(
            &wrapped,
            &larger,
            &noise3,
            &HardClamp::none(3).expect("none"),
            &[],
            CompileBudget::unlimited(),
        )
        .expect_err("different order");
        assert_eq!(err, SccCompileError::ComponentDimension { component: 2, model: 3 });
    }

    #[test]
    fn a_negative_variance_hidden_by_a_large_one_is_refused() {
        // Σ = diag(1e6, -1e-9). A tolerance scaled by the largest entry (1e6) would pass
        // it and emit Var = -1e-9.
        let model = map(vec![0.0, 0.0, 0.0, 0.0], vec![0.0, 0.0]);
        let noise = GaussianDisturbance::try_new(vec![0.0; 2], vec![1.0e6, 0.0, 0.0, -1.0e-9])
            .expect("finite symmetric");
        let err = compile_gaussian_equilibrium(
            &component_for(&model),
            &model,
            &noise,
            &HardClamp::none(2).expect("none"),
            &[],
            CompileBudget::unlimited(),
        )
        .expect_err("negative variance");
        assert_eq!(err, SccCompileError::NotPositiveSemidefinite { index: 1 });

        // A badly scaled but positive covariance is still accepted.
        let scaled = compile(
            &model,
            vec![0.0; 2],
            vec![1.0e6, 0.0, 0.0, 1.0e-9],
            &HardClamp::none(2).expect("none"),
            &[],
        );
        assert!((scaled.covariance()[3] - 1.0e-9).abs() < 1.0e-18);
    }

    #[test]
    fn weight_half_four_cycle_has_a_small_residual_and_a_condition_bound() {
        let mut coefficients = vec![0.0; 16];
        for node in 0..4 {
            coefficients[node * 4 + (node + 1) % 4] = 0.5;
        }
        let model = map(coefficients, vec![0.0; 4]);
        let mut mean = vec![0.0; 4];
        mean[0] = 1.0;
        let law = compile(&model, mean, identity(4), &HardClamp::none(4).expect("no clamp"), &[]);
        assert_eq!(law.blocks().len(), 1);
        assert_eq!(law.blocks()[0].members, vec![0, 1, 2, 3]);
        assert!(!law.blocks()[0].exact);
        assert!(law.blocks()[0].kappa_inf.is_finite());
        assert!(law.blocks()[0].kappa_inf > 0.0);
        assert!(law.residual_inf() < 1e-8);
        assert_eq!(law.identification(), IdentificationStatus::NotMarkedUnidentified);
    }

    #[test]
    fn a_third_order_block_uses_the_hager_higham_estimate() {
        let mut coefficients = vec![0.0; 9];
        coefficients[2] = 0.1;
        coefficients[3] = 0.1;
        coefficients[7] = 0.1;
        let model = map(coefficients, vec![0.0; 3]);
        let clamp = HardClamp::none(3).expect("none");
        let law = compile(&model, vec![0.0; 3], identity(3), &clamp, &[]);
        assert_eq!(law.blocks().len(), 1);
        assert_eq!(law.blocks()[0].members, vec![0, 1, 2]);
        assert!(!law.blocks()[0].exact);
        let matrix = [
            1.0, 0.0, -0.1, //
            -0.1, 1.0, 0.0, //
            0.0, -0.1, 1.0,
        ];
        assert!(close(law.blocks()[0].kappa_inf, infinity_condition(&matrix)));
    }

    fn infinity_condition(matrix: &[f64]) -> f64 {
        let mut factor = matrix.to_vec();
        let mut inverse = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        for pivot in 0..3 {
            let mut scale = factor[pivot * 3 + pivot];
            for column in 0..3 {
                factor[pivot * 3 + column] /= scale;
                inverse[pivot * 3 + column] /= scale;
            }
            for row in 0..3 {
                if row == pivot {
                    continue;
                }
                scale = factor[row * 3 + pivot];
                for column in 0..3 {
                    factor[row * 3 + column] -= scale * factor[pivot * 3 + column];
                    inverse[row * 3 + column] -= scale * inverse[pivot * 3 + column];
                }
            }
        }
        let norm = |rows: &[f64]| {
            rows.chunks(3).fold(0.0_f64, |best, row| {
                best.max(row.iter().fold(0.0_f64, |acc, entry| acc + entry.abs()))
            })
        };
        norm(matrix) * norm(&inverse)
    }

    #[test]
    fn nonzero_diagonal_names_the_next_attempt() {
        let diagonal = LinearStructuralMap::try_new(vec![0.2, 0.0, 0.0, 0.0], vec![0.0, 0.0]);
        let err = diagonal.expect_err("nonzero diagonal");
        assert_eq!(
            err,
            LinearSolutionError::NonzeroDiagonal {
                node: 0,
                next_attempt: NONZERO_DIAGONAL_NEXT_ATTEMPT,
            }
        );
        assert!(err.to_string().contains("next attempt"));
        assert!(err.to_string().contains(NONZERO_DIAGONAL_NEXT_ATTEMPT));
    }

    #[test]
    fn non_finite_indefinite_and_budget_shortfalls_are_refused() {
        let non_finite = GaussianDisturbance::try_new(vec![f64::NAN, 0.0], identity(2));
        assert!(matches!(non_finite, Err(SccCompileError::NonFiniteDisturbanceMean { node: 0 })));
        let asymmetric = GaussianDisturbance::try_new(vec![0.0, 0.0], vec![1.0, 0.2, 0.4, 1.0]);
        assert!(matches!(
            asymmetric,
            Err(SccCompileError::AsymmetricCovariance { row: 0, column: 1 })
        ));

        let model = map(vec![0.0, 0.4, 0.5, 0.0], vec![0.0, 0.0]);
        let clamp = HardClamp::none(2).expect("no clamp");
        let noise = GaussianDisturbance::try_new(vec![0.0, 0.0], identity(2)).expect("noise");
        let indefinite = GaussianDisturbance::try_new(vec![0.0, 0.0], vec![1.0, 2.0, 2.0, 1.0])
            .expect("indefinite shape is finite and symmetric");
        let err = compile_gaussian_equilibrium(
            &component_for(&model),
            &model,
            &indefinite,
            &clamp,
            &[],
            CompileBudget::unlimited(),
        )
        .expect_err("indefinite covariance");
        assert!(matches!(err, SccCompileError::NotPositiveSemidefinite { index: 1 }));

        let cost = gaussian_equilibrium_admission(&model, &clamp).expect("admission");
        let short_work = compile_gaussian_equilibrium(
            &component_for(&model),
            &model,
            &noise,
            &clamp,
            &[],
            CompileBudget { max_work: cost.work() - 1, max_bytes: u64::MAX },
        )
        .expect_err("work budget");
        assert_eq!(
            short_work,
            SccCompileError::Budget {
                resource: "work",
                requested: cost.work(),
                limit: cost.work() - 1,
            }
        );
        let short_bytes = compile_gaussian_equilibrium(
            &component_for(&model),
            &model,
            &noise,
            &clamp,
            &[],
            CompileBudget { max_work: u64::MAX, max_bytes: cost.bytes() - 1 },
        )
        .expect_err("byte budget");
        assert!(matches!(
            short_bytes,
            SccCompileError::Budget { resource: "bytes", requested, limit }
                if requested == cost.bytes() && limit == cost.bytes() - 1
        ));
    }

    #[test]
    fn acyclic_support_uses_the_dag_path_and_a_cycle_is_refused() {
        // x1 = 0.5 x0 + ε1, x2 = 0.25 x0 + 0.4 x1 + ε2, x0 = ε0. No directed cycle.
        let chain = map(vec![0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.25, 0.4, 0.0], vec![0.0, 0.0, 0.0]);
        let open = HardClamp::none(3).expect("open chain");
        let noise_mean = vec![1.0, 0.0, 0.0];
        let noise_cov = identity(3);
        let routed = compile_routed_gaussian_equilibrium(
            &component_for(&chain),
            &chain,
            &GaussianDisturbance::try_new(noise_mean.clone(), noise_cov.clone()).expect("noise"),
            &open,
            &[],
            CompileBudget::unlimited(),
        )
        .expect("acyclic route");
        match routed.route() {
            LinearSupportRoute::Acyclic { order } => assert_eq!(order, &vec![0, 1, 2]),
            LinearSupportRoute::Cyclic { next_attempt, .. } => {
                panic!("acyclic chain was refused: {next_attempt}")
            }
        }
        let law = routed.equilibrium();
        assert_eq!(law.compile_attempt(), DAG_EQUILIBRIUM_ATTEMPT);
        assert!(law.blocks().iter().all(|block| block.members.len() == 1));
        let scc = compile(&chain, noise_mean, noise_cov, &open, &[]);
        assert_eq!(scc.compile_attempt(), SCC_COMPILE_ATTEMPT);
        assert_eq!(law.mean(), scc.mean());
        assert_eq!(law.covariance(), scc.covariance());
        assert!(close(law.mean()[0], 1.0));
        assert!(close(law.mean()[1], 0.5));
        assert!(close(law.mean()[2], 0.45));

        let cycle = map(vec![0.0, 0.4, 0.5, 0.0], vec![0.0, 0.0]);
        let clamp = HardClamp::none(2).expect("open cycle");
        let routed = compile_routed_gaussian_equilibrium(
            &component_for(&cycle),
            &cycle,
            &GaussianDisturbance::try_new(vec![1.0, 0.0], identity(2)).expect("noise"),
            &clamp,
            &[],
            CompileBudget::unlimited(),
        )
        .expect("cycle is compiled only after the DAG refusal");
        let LinearSupportRoute::Cyclic { next_attempt, blocks } = routed.route() else {
            panic!("a directed cycle must be refused by the DAG algorithm");
        };
        assert!(next_attempt.contains("condensation"), "{next_attempt}");
        assert!(next_attempt.contains("sigma_separates"), "{next_attempt}");
        assert!(next_attempt.contains("next attempt"), "{next_attempt}");
        assert!(blocks.iter().any(|block| block.len() > 1), "{blocks:?}");
        assert_eq!(routed.equilibrium().compile_attempt(), SCC_COMPILE_ATTEMPT);
        assert!(routed.equilibrium().blocks().iter().any(|block| block.members.len() > 1));
    }

    fn identity(n: usize) -> Vec<f64> {
        let mut covariance = vec![0.0; n * n];
        for node in 0..n {
            covariance[node * n + node] = 1.0;
        }
        covariance
    }
}
