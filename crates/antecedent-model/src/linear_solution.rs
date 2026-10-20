//! Solution contracts for a specified linear equilibrium SCM.
//!
//! The structural model is `X = KX + c + ε`. `K[i, j]` is variable `j`'s direct
//! mechanism into variable `i`. This initial family requires a zero diagonal.
//! A nonzero diagonal is refused and names [`NONZERO_DIAGONAL_NEXT_ATTEMPT`]:
//! a later normalization would have to transform the intercept and the noise
//! law with it, and this module does not do that.
//!
//! A hard clamp `H = a` deletes the clamped equations and leaves the free block
//!
//! ```text
//! (I - K_FF) X_F = K_FH a + c_F + ε_F
//! ```
//!
//! [`FREE_BLOCK_EQUATION`] records that operator. The reports below describe
//! the block to factor. They do not form `(I - K_FF)⁻¹`. `ε_F` stays the
//! marginal disturbance on the free variables: clamping is not conditioning
//! on the factual value of a clamped variable.
//! [`revalidate_free_block`] repeats the decision on the clamp's free block.
//! A singular or non-finite block is a refusal. An ill-conditioned block stays
//! ill-conditioned and is not reported unique.
//!
//! [`hard_clamp_mean_batch`] evaluates several assignments of one hard clamp.
//! One factorization of the free block serves every row. That factor is
//! overwritten and is not stored. The batch returns equilibrium means only.
//! A disturbance law's mean and covariance belong to the Gaussian compile.
//!
//! Each assessment keeps four statuses apart:
//!
//! - [`QueryBlock`] — solvability and conditioning of one concrete free block,
//!   for the original system and again for the clamped query.
//! - [`FamilyCertificate`] — the sufficient family bound `ρ(|K|) < 1` on the
//!   entrywise absolute matrix. It applies to every hard-clamp free block at
//!   once. A bound that does not land strictly below one leaves this
//!   certificate unestablished and names the narrower query-specific check.
//!   That outcome is [`IdentificationStatus::NotMarkedUnidentified`].
//! - [`PhysicalStability`] — dynamical stability. Invertibility of `(I - K)`,
//!   a signed radius `ρ(K) < 1`, and `ρ(|K|) < 1` are different claims, and
//!   none of them is written into this status. `ρ(K) < 1` does not imply
//!   `ρ(|K|) < 1`.
//! - [`SolverConvergence`] — whether an iterative solver ran. No certificate
//!   above is a convergence result.
//!
//! Hyttinen, Eberhardt, and Hoyer's weak stability is invertibility of the
//! equilibrium map. It is the solvability status, when that status is
//! [`Uniqueness::Unique`], and it is not [`PhysicalStability`].
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]
#![deny(missing_docs)]

use antecedent_kernels::gauss_jordan_solve;
use thiserror::Error;

use crate::scc_compile::{eye_minus_block, support_blocks};

/// Equation of the free block under a hard clamp `H = a`.
///
/// `K_FF` and `K_FH` are the free-free and free-clamped blocks of `K`.
/// Factor or solve this operator; do not treat the string as an inverse.
pub const FREE_BLOCK_EQUATION: &str = "(I - K_FF) X_F = K_FH a + c_F + ε_F";

/// Next attempt when a structural diagonal is nonzero.
///
/// Rescaling that equation by itself would leave the intercept and the noise
/// law in the old coordinate. This family does not do that rescaling.
pub const NONZERO_DIAGONAL_NEXT_ATTEMPT: &str = concat!(
    "normalize the structural equation only by transforming the intercept and the noise law ",
    "with the diagonal",
);

/// A computed upper bound must sit at least this far below one before
/// `ρ(|K|) < 1` is treated as established. The gap absorbs a rounding-low
/// Collatz–Wielandt ratio so a bound that merely grazes one is not admitted.
const ADMISSION_GAP: f64 = 1.0e-12;

/// Power iterations used for the Collatz–Wielandt upper bound on `ρ(|K|)`.
const POWER_ITERATIONS: usize = 32;

/// Largest strongly connected block that [`assess`] pivot-tests directly.
/// Larger blocks are decided by a norm or by the family certificate, or stay open.
const PIVOT_TEST_ORDER_LIMIT: usize = 3;

/// A block whose exact closed-form infinity-norm condition number exceeds this value
/// is reported as [`Uniqueness::IllConditioned`] rather than as a clean
/// [`Uniqueness::Unique`]. With the equilibrated pivot floor
/// ([`antecedent_kernels::EQUILIBRATED_PIVOT_FLOOR`]) a block can pass the singularity test and still
/// carry a condition number near `1 / floor`.
pub const ILL_CONDITIONED_KAPPA: f64 = 1.0e8;

/// The singularity test shared by [`assess`] and
/// [`compile_gaussian_equilibrium`](crate::scc_compile::compile_gaussian_equilibrium).
///
/// `eye_minus` is `I - K_SS` row-major of order `m`, whose diagonal is one. It is
/// regular when [`gauss_jordan_solve`] with partial pivoting finds every pivot
/// above [`antecedent_kernels::EQUILIBRATED_PIVOT_FLOOR`]. A `false` means numerically singular: a
/// pivot sits at or below the floor, which includes but is not limited to an
/// exactly singular matrix.
pub(crate) fn pivots_regular(eye_minus: &[f64], m: usize) -> bool {
    if m == 0 {
        return true;
    }
    let mut factor = eye_minus.to_vec();
    gauss_jordan_solve(&mut factor, &mut [], m, 0)
}

/// Failure to construct a map or a clamp, before any certificate is issued.
#[derive(Clone, Debug, Eq, PartialEq, Error)]
#[non_exhaustive]
pub enum LinearSolutionError {
    /// The coefficient matrix and the intercept do not describe one finite square model.
    #[error("linear SCM shape: {message}")]
    Shape {
        /// What is inconsistent.
        message: &'static str,
    },
    /// `K[node, node]` is nonzero. This family does not rescale that equation.
    #[error("linear SCM requires a zero diagonal at node {node}; next attempt: {next_attempt}")]
    NonzeroDiagonal {
        /// Variable whose structural self-coefficient is nonzero.
        node: usize,
        /// [`NONZERO_DIAGONAL_NEXT_ATTEMPT`].
        next_attempt: &'static str,
    },
    /// A structural coefficient is NaN or infinite.
    #[error("non-finite structural coefficient into {into_node} from {from_node}")]
    NonFiniteCoefficient {
        /// Row index, the variable the mechanism enters.
        into_node: usize,
        /// Column index, the variable the mechanism comes from.
        from_node: usize,
    },
    /// An intercept is NaN or infinite.
    #[error("non-finite intercept at node {node}")]
    NonFiniteIntercept {
        /// Variable whose intercept is not finite.
        node: usize,
    },
    /// A clamp index is outside `0..dimension`.
    #[error("clamp index {node} is outside dimension {dimension}")]
    ClampOutOfRange {
        /// Index that was requested.
        node: usize,
        /// Model dimension the clamp was built against.
        dimension: usize,
    },
    /// The same variable was clamped twice.
    #[error("clamp index {node} is repeated")]
    DuplicateClamp {
        /// Index that occurred more than once.
        node: usize,
    },
    /// The clamp was built for a different dimension than the map.
    #[error("clamp dimension {clamp} does not match model dimension {model}")]
    ClampDimension {
        /// Dimension stored on the clamp.
        clamp: usize,
        /// Dimension of the map.
        model: usize,
    },
    /// The free block after a hard clamp is singular or non-finite.
    ///
    /// This is a refusal of the intervened block. It does not mark the model
    /// unidentified: the status on a completed assessment stays
    /// [`IdentificationStatus::NotMarkedUnidentified`].
    #[error("free block after the clamp is not solvable: {refusal}")]
    FreeBlockNotSolvable {
        /// Which solvability failure was rechecked.
        refusal: SolvabilityRefusal,
    },
}

/// Why the free block that remains after a hard clamp has no solution to return.
///
/// [`Uniqueness::IllConditioned`] is not one of these failures. A large condition
/// number is a status of a block that still has a unique solution.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SolvabilityRefusal {
    /// The closed form of `(I - K_FF)` is exactly singular.
    Singular {
        /// Free indices, ascending.
        members: Vec<usize>,
    },
    /// A strongly connected block failed the shared pivot test.
    NumericallySingular {
        /// Free variables of that block, ascending.
        members: Vec<usize>,
    },
    /// A closed form of the free block was not finite.
    NonFinite,
}

impl std::fmt::Display for SolvabilityRefusal {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Singular { members } => write!(formatter, "singular free block {members:?}"),
            Self::NumericallySingular { members } => {
                write!(formatter, "numerically singular free block {members:?}")
            }
            Self::NonFinite => formatter.write_str("non-finite free block"),
        }
    }
}

/// Row-major structural map `X = KX + c + ε` with a zero diagonal.
#[derive(Clone, Debug, PartialEq)]
pub struct LinearStructuralMap {
    n: usize,
    coefficients: Vec<f64>,
    intercept: Vec<f64>,
}

impl LinearStructuralMap {
    /// Build a map from row-major `K` and intercept `c`.
    ///
    /// `coefficients[i * n + j]` is `K[i, j]`, the direct mechanism of `j` into `i`.
    /// `n` is `intercept.len()`. The diagonal must be exactly zero, and every
    /// entry must be finite.
    ///
    /// # Errors
    ///
    /// [`LinearSolutionError::Shape`], [`LinearSolutionError::NonFiniteCoefficient`],
    /// [`LinearSolutionError::NonFiniteIntercept`], or [`LinearSolutionError::NonzeroDiagonal`].
    #[must_use = "construction errors are the refusal of this initial family"]
    pub fn try_new(
        coefficients: Vec<f64>,
        intercept: Vec<f64>,
    ) -> Result<Self, LinearSolutionError> {
        let n = intercept.len();
        if n == 0 {
            return Err(LinearSolutionError::Shape {
                message: "a linear SCM needs at least one variable",
            });
        }
        let Some(len) = n.checked_mul(n) else {
            return Err(LinearSolutionError::Shape {
                message: "coefficient length overflows usize",
            });
        };
        if coefficients.len() != len {
            return Err(LinearSolutionError::Shape {
                message: "coefficients must be row-major with length n*n",
            });
        }
        for (i, row) in coefficients.chunks_exact(n).enumerate() {
            if !intercept[i].is_finite() {
                return Err(LinearSolutionError::NonFiniteIntercept { node: i });
            }
            for (j, value) in row.iter().copied().enumerate() {
                if !value.is_finite() {
                    return Err(LinearSolutionError::NonFiniteCoefficient {
                        into_node: i,
                        from_node: j,
                    });
                }
                // The family invariant is an exact structural zero, not a tolerance.
                #[allow(
                    clippy::float_cmp,
                    reason = "zero diagonal is an exact structural constraint of this family"
                )]
                if i == j && value != 0.0 {
                    return Err(LinearSolutionError::NonzeroDiagonal {
                        node: i,
                        next_attempt: NONZERO_DIAGONAL_NEXT_ATTEMPT,
                    });
                }
            }
        }
        Ok(Self { n, coefficients, intercept })
    }

    /// Number of variables.
    #[must_use]
    pub fn dimension(&self) -> usize {
        self.n
    }

    /// Row-major `K`.
    #[must_use]
    pub fn coefficients(&self) -> &[f64] {
        &self.coefficients
    }

    /// Intercept `c`.
    #[must_use]
    pub fn intercept(&self) -> &[f64] {
        &self.intercept
    }

    /// `K[into_node, from_node]`, the mechanism of `from_node` into `into_node`.
    #[must_use]
    pub fn coefficient(&self, into_node: usize, from_node: usize) -> Option<f64> {
        if into_node >= self.n || from_node >= self.n {
            return None;
        }
        Some(self.coefficients[into_node * self.n + from_node])
    }
}

/// Variables held at assigned values. The values themselves do not enter the
/// solvability certificate; they enter the right-hand side `K_FH a`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HardClamp {
    dimension: usize,
    indices: Vec<usize>,
}

impl HardClamp {
    /// Clamp `nodes` inside a model of `dimension`.
    ///
    /// Indices are stored in increasing order. Duplicates are refused.
    ///
    /// # Errors
    ///
    /// [`LinearSolutionError::Shape`] when `dimension` is zero,
    /// [`LinearSolutionError::ClampOutOfRange`], or [`LinearSolutionError::DuplicateClamp`].
    pub fn try_new(dimension: usize, nodes: &[usize]) -> Result<Self, LinearSolutionError> {
        if dimension == 0 {
            return Err(LinearSolutionError::Shape {
                message: "a clamp needs a positive model dimension",
            });
        }
        let mut indices = nodes.to_vec();
        indices.sort_unstable();
        for (pos, &node) in indices.iter().enumerate() {
            if node >= dimension {
                return Err(LinearSolutionError::ClampOutOfRange { node, dimension });
            }
            if pos > 0 && indices[pos - 1] == node {
                return Err(LinearSolutionError::DuplicateClamp { node });
            }
        }
        Ok(Self { dimension, indices })
    }

    /// Clamp no variables. The intervened block is then the original system.
    ///
    /// # Errors
    ///
    /// [`LinearSolutionError::Shape`] when `dimension` is zero.
    pub fn none(dimension: usize) -> Result<Self, LinearSolutionError> {
        Self::try_new(dimension, &[])
    }

    /// Dimension the clamp was built for.
    #[must_use]
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Clamped variable indices, strictly increasing.
    #[must_use]
    pub fn indices(&self) -> &[usize] {
        &self.indices
    }
}

/// Equilibrium means for several assignments of one [`HardClamp`].
///
/// Row `s` is the solution of [`FREE_BLOCK_EQUATION`] at `assignments[s]`.
/// Clamped coordinates hold those assignments. Free coordinates are the solved
/// block. The factor is not retained.
#[derive(Clone, Debug, PartialEq)]
#[must_use = "dropping it drops the intervened means"]
pub struct HardClampMeanBatch {
    rows: usize,
    dimension: usize,
    means: Vec<f64>,
}

impl HardClampMeanBatch {
    /// Number of assignment rows.
    #[must_use = "the row count selects an intervened mean"]
    pub const fn rows(&self) -> usize {
        self.rows
    }

    /// Model dimension, the width of each mean.
    #[must_use = "the dimension is the width of each intervened mean"]
    pub const fn dimension(&self) -> usize {
        self.dimension
    }

    /// Row-major means, one equilibrium per assignment row.
    #[must_use = "the means are the intervened batch"]
    pub fn means(&self) -> &[f64] {
        &self.means
    }

    /// Equilibrium mean for assignment row `row`, or `None` when `row` is past the batch.
    #[must_use = "a missing row is outside the batch"]
    pub fn row(&self, row: usize) -> Option<&[f64]> {
        let start = row.checked_mul(self.dimension)?;
        let end = start.checked_add(self.dimension)?;
        self.means.get(start..end)
    }
}

/// Solvability, conditioning, family certificate, physical stability, and
/// solver convergence for one map and one hard clamp.
///
/// `original` ignores the clamp. `intervened` is the free block that remains.
/// Those two blocks can disagree; a determinant of the original system is not
/// a determinant of the free block.
#[derive(Clone, Debug, PartialEq)]
#[must_use]
pub struct SolutionAssessment {
    /// Full system, with every variable free.
    pub original: QueryBlock,
    /// Free block after the hard clamp.
    pub intervened: QueryBlock,
    /// Sufficient certificate `ρ(|K|) < 1` for every hard-clamp sub-block.
    pub family: FamilyCertificate,
    /// Physical dynamical stability. Always unclaimed by this module.
    pub physical_stability: PhysicalStability,
    /// Iterative solver status. Always [`SolverConvergence::NotRun`] here.
    pub solver: SolverConvergence,
}

/// One linear block: the operator to factor, whether it determines `X_F`,
/// and a condition number when this module has one without inverting.
#[derive(Clone, Debug, PartialEq)]
pub struct QueryBlock {
    /// `(I - K_FF)` described by its structural blocks.
    pub factor: FactorDescription,
    /// Existence and uniqueness for this block alone.
    pub uniqueness: Uniqueness,
    /// Conditioning of `(I - K_FF)`, kept separate from uniqueness.
    pub conditioning: Conditioning,
}

/// Structural pieces of [`FREE_BLOCK_EQUATION`].
///
/// `k_ff` is `K_FF`, not `(I - K_FF)` and not an inverse. A caller that
/// factors the block builds `I - K_FF` from `k_ff`.
#[derive(Clone, Debug, PartialEq)]
pub struct FactorDescription {
    /// Free indices `F`, ascending.
    pub free: Vec<usize>,
    /// Clamped indices `H`, ascending.
    pub clamped: Vec<usize>,
    /// [`FREE_BLOCK_EQUATION`].
    pub equation: &'static str,
    /// `K_FF` in free-index order, row-major. Entry `(p, q)` is `K[F_p, F_q]`.
    pub k_ff: Vec<f64>,
    /// `K_FH` row-major, length `free.len() * clamped.len()`.
    /// Entry `(p, r)` is `K[F_p, H_r]`.
    pub k_fh: Vec<f64>,
    /// Intercepts `c_F` in free-index order.
    pub intercept_f: Vec<f64>,
}

/// Whether `(I - K_FF) X_F = rhs` determines `X_F`.
///
/// A unique block is a query-specific claim. It does not establish
/// `ρ(|K|) < 1`, and it does not establish physical stability.
#[derive(Clone, Debug, PartialEq)]
pub enum Uniqueness {
    /// The free block has a unique solution for every finite right-hand side.
    Unique,
    /// `(I - K_FF)` is singular. Off the consistency constraint there is no
    /// solution; on it there are infinitely many. No root is selected.
    Singular(SingularBlock),
    /// A strongly connected block of the free system has a pivot at or below
    /// [`antecedent_kernels::EQUILIBRATED_PIVOT_FLOOR`]. This is the same test that makes
    /// [`compile_gaussian_equilibrium`](crate::scc_compile::compile_gaussian_equilibrium)
    /// refuse the block. It does not claim the block is exactly singular:
    /// [`Self::Singular`] is reserved for a closed-form determinant that evaluates
    /// to zero.
    NumericallySingular {
        /// Free variables of the failing strongly connected block, ascending.
        members: Vec<usize>,
    },
    /// The pivot test passed, so the block has a unique solution, but the exact
    /// infinity-norm condition number exceeds [`ILL_CONDITIONED_KAPPA`]. A solve
    /// loses roughly `log10(kappa)` digits.
    IllConditioned {
        /// Exact infinity-norm condition number of the block.
        kappa: f64,
    },
    /// The sufficient checks did not decide this block. The query stays open.
    /// This is not a verdict that the model is unidentified.
    Undetermined {
        /// Factor or solve `(I - K_FF)` on this free set.
        next_attempt: NextAttempt,
    },
}

/// Witness that a singular free block is a consistency constraint rather than
/// a selected root.
#[derive(Clone, Debug, PartialEq)]
pub struct SingularBlock {
    /// How the constraint was obtained.
    pub witness: ConsistencyWitness,
}

/// Left-null data for `w · (K_FH a + c_F + ε_F) = 0`.
#[derive(Clone, Debug, PartialEq)]
pub enum ConsistencyWitness {
    /// Nontrivial `w` with `wᵀ (I - K_FF) = 0`, in free-index order.
    ///
    /// A solution exists only when `w` is orthogonal to `K_FH a + c_F + ε_F`.
    LeftNull(Vec<f64>),
    /// The determinant is zero, so the solution is not unique, and a left
    /// null vector was not formed. The next exact step is a factorization of
    /// this same block.
    NotFormed,
}

/// Conditioning of `(I - K_FF)`, separate from whether a unique solution exists.
#[derive(Clone, Debug, PartialEq)]
pub enum Conditioning {
    /// Infinity-norm condition number.
    ///
    /// `exact` is true for the closed 1×1 and 2×2 forms. It is false when
    /// `kappa` is the Neumann bound `(1 + ||K_FF||∞) / (1 - ||K_FF||∞)`
    /// obtained from `||K_FF||∞ < 1`.
    InfinityNorm {
        /// Condition number, or a finite upper bound on it when `exact` is false.
        kappa: f64,
        /// `true` when `kappa` is the closed-form infinity-norm condition number.
        exact: bool,
    },
    /// Uniqueness may already be known. A condition number still needs a factor.
    NotEstimated {
        /// Factor `(I - K_FF)` and estimate its condition from that factor.
        next_attempt: NextAttempt,
    },
    /// The block is exactly or numerically singular, so it has no usable condition number.
    Undefined,
}

/// Sufficient family certificate `ρ(|K|) < 1`.
///
/// The bound is an upper bound: the infinity norm of `|K|`, and for order
/// above two a Collatz–Wielandt ratio `max_i (|K| v)_i / v_i` on positive
/// vectors from a short power iteration. For two variables it is the closed
/// form `sqrt(|K[0,1] K[1,0]|)`. Admission requires the bound to sit at least
/// `1e-12` below one, so a ratio that only rounds under one is not admitted.
///
/// Failure means the sufficient bound was not shown. It does not prove
/// `ρ(|K|) ≥ 1`, it does not prove a query-specific block is singular, and
/// it does not mark the model unidentified. `ρ(K) < 1` for the signed matrix
/// is not this certificate.
#[derive(Clone, Debug, PartialEq)]
pub enum FamilyCertificate {
    /// `ρ(|K|) ≤ rho_abs_upper < 1`, up to the admission gap.
    Admitted {
        /// Finite upper bound on `ρ(|K|)`.
        rho_abs_upper: f64,
    },
    /// The sufficient bound did not land strictly below one.
    Insufficient {
        /// Finite upper bound, or `+∞` when every bound overflowed.
        rho_abs_upper: f64,
        /// The narrower check to run for the clamp that was assessed.
        next_attempt: NextAttempt,
        /// Specified-model algebra is not an identification result.
        identification: IdentificationStatus,
    },
}

impl FamilyCertificate {
    /// Whether `ρ(|K|) < 1` was established for the whole intervention family.
    #[must_use]
    pub fn is_admitted(&self) -> bool {
        matches!(self, Self::Admitted { .. })
    }

    /// Upper bound carried by either variant.
    #[must_use]
    pub fn rho_abs_upper(&self) -> f64 {
        match self {
            Self::Admitted { rho_abs_upper } | Self::Insufficient { rho_abs_upper, .. } => {
                *rho_abs_upper
            }
        }
    }

    /// Identification status of this certificate.
    ///
    /// Both variants return [`IdentificationStatus::NotMarkedUnidentified`].
    /// A failed bound does not change the status.
    #[must_use]
    pub fn identification(&self) -> IdentificationStatus {
        match self {
            Self::Admitted { .. } => IdentificationStatus::NotMarkedUnidentified,
            Self::Insufficient { identification, .. } => *identification,
        }
    }
}

/// What to try after a sufficient certificate stops short.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum NextAttempt {
    /// Decide `(I - K_FF)` on this free set by a factorization or an exact
    /// small-block check. The claim, if it succeeds, covers this free set
    /// and not every other hard clamp.
    QuerySpecificFreeBlock {
        /// Free indices, ascending.
        free: Vec<usize>,
    },
}

/// Identification claim attached to a family certificate.
///
/// This module evaluates a specified map. It does not identify `K` from
/// observations, and a failed contraction bound is not an identification failure.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IdentificationStatus {
    /// The model was not marked unidentified.
    NotMarkedUnidentified,
}

/// Physical stability of the equilibrium as a dynamical system.
///
/// This module does not award or refuse stability. Unique solvability,
/// `ρ(K) < 1` for a signed synchronous update, and `ρ(|K|) < 1` remain
/// separate claims; a signed radius below one does not imply the absolute
/// certificate, and neither radius is copied into this status.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PhysicalStability {
    /// No physical-stability certificate was claimed.
    Unclaimed,
}

/// Convergence of an iterative linear solver.
///
/// Certificates are not solver iterations. Admitting `ρ(|K|) < 1` does not
/// mean a solver converged, and failing that bound does not mean a solver diverged.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SolverConvergence {
    /// No iterative solver was run.
    NotRun,
}

/// Assess `model` under `clamp`.
///
/// The original block and the clamped free block are decided independently.
/// The family certificate is the absolute-radius bound on the whole of `K`.
/// Physical stability stays [`PhysicalStability::Unclaimed`] and solver
/// convergence stays [`SolverConvergence::NotRun`].
///
/// # Errors
///
/// [`LinearSolutionError::ClampDimension`] when the clamp was built for a
/// different dimension.
#[must_use = "the assessment is the solution contract"]
pub fn assess(
    model: &LinearStructuralMap,
    clamp: &HardClamp,
) -> Result<SolutionAssessment, LinearSolutionError> {
    if clamp.dimension != model.n {
        return Err(LinearSolutionError::ClampDimension { clamp: clamp.dimension, model: model.n });
    }
    let intervened_free = free_indices(model.n, &clamp.indices);
    let family = certify_family(model, &intervened_free);
    // `ρ(|K|) < 1` bounds every principal sub-block, so it applies to both blocks.
    let admitted = family.is_admitted();
    let original = query_block(model, &[], admitted);
    let intervened = query_block(model, &clamp.indices, admitted);
    Ok(SolutionAssessment {
        original,
        intervened,
        family,
        physical_stability: PhysicalStability::Unclaimed,
        solver: SolverConvergence::NotRun,
    })
}

/// Recheck solvability of the free block `clamp` leaves.
///
/// The original system is not reused. The result is the intervened
/// [`QueryBlock`] from [`assess`]: [`Uniqueness::Unique`] and
/// [`Uniqueness::IllConditioned`] are returned as assessed, and an open block
/// larger than the closed form stays [`Uniqueness::Undetermined`]. A singular
/// block, a numerically singular block, and a non-finite closed form are
/// [`LinearSolutionError::FreeBlockNotSolvable`]. That refusal does not mark
/// the model unidentified. No inverse is stored, and no iterative solver runs.
///
/// # Errors
///
/// [`LinearSolutionError::ClampDimension`] or [`LinearSolutionError::FreeBlockNotSolvable`].
#[must_use = "a refused free block is not a solution"]
pub fn revalidate_free_block(
    model: &LinearStructuralMap,
    clamp: &HardClamp,
) -> Result<QueryBlock, LinearSolutionError> {
    let assessment = assess(model, clamp)?;
    if let Some(refusal) = solvability_refusal(&assessment.intervened) {
        return Err(LinearSolutionError::FreeBlockNotSolvable { refusal });
    }
    Ok(assessment.intervened)
}

/// Equilibrium means for every row of `assignment_rows` under one hard clamp.
///
/// `assignment_rows` is row-major. Each row has one entry per
/// [`HardClamp::indices`] entry, in that order. `disturbance_mean` is the mean
/// of `ε` in variable order and is added on the free right-hand side. The
/// clamp's free block is rechecked with [`revalidate_free_block`] before the
/// factor is built. One call to [`gauss_jordan_solve`] then serves every row.
/// The factor is overwritten and is not stored. A covariance is not part of
/// this batch: the level does not change it.
///
/// # Errors
///
/// [`LinearSolutionError::Shape`] when the disturbance mean or the assignment
/// rows do not describe a non-empty batch for this clamp,
/// [`LinearSolutionError::ClampDimension`], or
/// [`LinearSolutionError::FreeBlockNotSolvable`].
#[must_use = "a refused free block is not a batch of means"]
pub fn hard_clamp_mean_batch(
    model: &LinearStructuralMap,
    clamp: &HardClamp,
    assignment_rows: &[f64],
    disturbance_mean: &[f64],
) -> Result<HardClampMeanBatch, LinearSolutionError> {
    let n = model.n;
    if disturbance_mean.len() != n || disturbance_mean.iter().any(|value| !value.is_finite()) {
        return Err(LinearSolutionError::Shape {
            message: "disturbance mean must be finite and match the model dimension",
        });
    }
    let rows = assignment_row_count(clamp.indices().len(), assignment_rows)?;
    revalidate_free_block(model, clamp)?;
    let free = free_indices(n, clamp.indices());
    let means = if free.is_empty() {
        scatter_clamp_means(n, rows, clamp.indices(), &free, assignment_rows, &[])
    } else {
        solve_free_means(model, clamp, &free, rows, assignment_rows, disturbance_mean)?
    };
    Ok(HardClampMeanBatch { rows, dimension: n, means })
}

fn assignment_row_count(
    width: usize,
    assignment_rows: &[f64],
) -> Result<usize, LinearSolutionError> {
    let shape = LinearSolutionError::Shape {
        message: "assignments must be finite row-major rows of this clamp",
    };
    if width == 0 {
        return Err(shape);
    }
    let rows = assignment_rows.chunks_exact(width).len();
    let covered = rows.checked_mul(width);
    if covered != Some(assignment_rows.len()) || rows == 0 {
        return Err(shape);
    }
    if assignment_rows.iter().any(|value| !value.is_finite()) {
        return Err(shape);
    }
    Ok(rows)
}

fn solve_free_means(
    model: &LinearStructuralMap,
    clamp: &HardClamp,
    free: &[usize],
    rows: usize,
    assignment_rows: &[f64],
    disturbance_mean: &[f64],
) -> Result<Vec<f64>, LinearSolutionError> {
    let n = model.n;
    let m = free.len();
    let mut factor = eye_minus_k(&principal(&model.coefficients, n, free), m);
    let mut rhs = free_batch_rhs(model, clamp, free, rows, assignment_rows, disturbance_mean);
    if !gauss_jordan_solve(&mut factor, &mut rhs, m, rows) {
        return Err(LinearSolutionError::FreeBlockNotSolvable {
            refusal: SolvabilityRefusal::NumericallySingular { members: free.to_vec() },
        });
    }
    if rhs.iter().any(|value| !value.is_finite()) {
        return Err(LinearSolutionError::FreeBlockNotSolvable {
            refusal: SolvabilityRefusal::NonFinite,
        });
    }
    Ok(scatter_clamp_means(n, rows, clamp.indices(), free, assignment_rows, &rhs))
}

fn eye_minus_k(k_ff: &[f64], order: usize) -> Vec<f64> {
    let mut matrix = vec![0.0; order * order];
    for row in 0..order {
        for col in 0..order {
            let coefficient = k_ff[row * order + col];
            matrix[row * order + col] = if row == col { 1.0 - coefficient } else { -coefficient };
        }
    }
    matrix
}

fn free_batch_rhs(
    model: &LinearStructuralMap,
    clamp: &HardClamp,
    free: &[usize],
    rows: usize,
    assignment_rows: &[f64],
    disturbance_mean: &[f64],
) -> Vec<f64> {
    let n = model.n;
    let m = free.len();
    let width = clamp.indices().len();
    let k_fh = gather_k_fh(&model.coefficients, n, free, clamp.indices());
    let mut rhs = vec![0.0; rows * m];
    for (scenario, assignment) in assignment_rows.chunks_exact(width).enumerate() {
        for (local, &node) in free.iter().enumerate() {
            let mut value = model.intercept[node] + disturbance_mean[node];
            let start = local * width;
            for (coefficient, level) in k_fh[start..start + width].iter().zip(assignment) {
                value += coefficient * level;
            }
            rhs[scenario * m + local] = value;
        }
    }
    rhs
}

fn scatter_clamp_means(
    n: usize,
    rows: usize,
    clamped: &[usize],
    free: &[usize],
    assignment_rows: &[f64],
    solved: &[f64],
) -> Vec<f64> {
    let width = clamped.len();
    let m = free.len();
    let mut means = vec![0.0; rows * n];
    for (scenario, assignment) in assignment_rows.chunks_exact(width).enumerate() {
        let start = scenario * n;
        let row = &mut means[start..start + n];
        for (&node, &level) in clamped.iter().zip(assignment) {
            row[node] = level;
        }
        for (local, &node) in free.iter().enumerate() {
            row[node] = solved[scenario * m + local];
        }
    }
    means
}

fn solvability_refusal(block: &QueryBlock) -> Option<SolvabilityRefusal> {
    if small_block_non_finite(&block.factor.k_ff, block.factor.free.len()) {
        return Some(SolvabilityRefusal::NonFinite);
    }
    match &block.uniqueness {
        Uniqueness::Singular(_) => {
            Some(SolvabilityRefusal::Singular { members: block.factor.free.clone() })
        }
        Uniqueness::NumericallySingular { members } => {
            Some(SolvabilityRefusal::NumericallySingular { members: members.clone() })
        }
        Uniqueness::IllConditioned { kappa } if !kappa.is_finite() => {
            Some(SolvabilityRefusal::NonFinite)
        }
        Uniqueness::Unique
        | Uniqueness::IllConditioned { .. }
        | Uniqueness::Undetermined { .. } => None,
    }
}

/// `true` when the 2×2 or 3×3 closed form of `I - K_FF` is not finite.
fn small_block_non_finite(k_ff: &[f64], order: usize) -> bool {
    match order {
        2 => {
            let product = k_ff[1] * k_ff[2];
            !product.is_finite() || !(1.0 - product).is_finite()
        }
        3 => {
            let mut eye_minus = [0.0; 9];
            for row in 0..3 {
                for col in 0..3 {
                    let coefficient = k_ff[row * 3 + col];
                    eye_minus[row * 3 + col] =
                        if row == col { 1.0 - coefficient } else { -coefficient };
                }
            }
            !determinant3(&eye_minus).is_finite()
        }
        _ => false,
    }
}

/// The family certificate of `model` with every variable free.
pub(crate) fn family_certificate(model: &LinearStructuralMap) -> FamilyCertificate {
    certify_family(model, &free_indices(model.n, &[]))
}

fn certify_family(model: &LinearStructuralMap, query_free: &[usize]) -> FamilyCertificate {
    let upper = abs_radius_upper(&model.coefficients, model.n);
    let upper = if upper.is_finite() { upper } else { f64::INFINITY };
    if (0.0..1.0 - ADMISSION_GAP).contains(&upper) {
        FamilyCertificate::Admitted { rho_abs_upper: upper }
    } else {
        FamilyCertificate::Insufficient {
            rho_abs_upper: upper,
            next_attempt: NextAttempt::QuerySpecificFreeBlock { free: query_free.to_vec() },
            identification: IdentificationStatus::NotMarkedUnidentified,
        }
    }
}

fn query_block(
    model: &LinearStructuralMap,
    clamped: &[usize],
    family_admitted: bool,
) -> QueryBlock {
    let free = free_indices(model.n, clamped);
    let k_ff = principal(&model.coefficients, model.n, &free);
    let k_fh = gather_k_fh(&model.coefficients, model.n, &free, clamped);
    let intercept_f = free.iter().map(|&node| model.intercept[node]).collect();
    let (uniqueness, conditioning) = decide(model, &free, &k_ff, family_admitted);
    QueryBlock {
        factor: FactorDescription {
            free,
            clamped: clamped.to_vec(),
            equation: FREE_BLOCK_EQUATION,
            k_ff,
            k_fh,
            intercept_f,
        },
        uniqueness,
        conditioning,
    }
}

/// Free indices of a model of order `n` under the ascending clamp `clamped`.
pub(crate) fn free_indices(n: usize, clamped: &[usize]) -> Vec<usize> {
    let mut free = Vec::with_capacity(n.saturating_sub(clamped.len()));
    let mut cursor = 0;
    for node in 0..n {
        if cursor < clamped.len() && clamped[cursor] == node {
            cursor += 1;
        } else {
            free.push(node);
        }
    }
    free
}

fn principal(coefficients: &[f64], n: usize, free: &[usize]) -> Vec<f64> {
    let m = free.len();
    let mut block = vec![0.0; m * m];
    for (row, &into_node) in free.iter().enumerate() {
        for (col, &from_node) in free.iter().enumerate() {
            block[row * m + col] = coefficients[into_node * n + from_node];
        }
    }
    block
}

fn gather_k_fh(coefficients: &[f64], n: usize, free: &[usize], clamped: &[usize]) -> Vec<f64> {
    let mut block = Vec::with_capacity(free.len().saturating_mul(clamped.len()));
    for &into_node in free {
        for &from_node in clamped {
            block.push(coefficients[into_node * n + from_node]);
        }
    }
    block
}

/// Decide one free block with the criterion `compile_gaussian_equilibrium` uses.
///
/// A closed-form determinant that evaluates to exactly zero is
/// [`Uniqueness::Singular`]. Otherwise each strongly connected block up to
/// [`PIVOT_TEST_ORDER_LIMIT`] gets the pivot test of [`pivots_regular`], and a
/// failure is [`Uniqueness::NumericallySingular`]. A larger block is decided by
/// the Neumann norm bound or the admitted family certificate, else it stays open.
fn decide(
    model: &LinearStructuralMap,
    free: &[usize],
    k_ff: &[f64],
    family_admitted: bool,
) -> (Uniqueness, Conditioning) {
    let m = free.len();
    if m <= 1 {
        return (Uniqueness::Unique, Conditioning::InfinityNorm { kappa: 1.0, exact: true });
    }
    if let Some(verdict) = closed_form_singular(k_ff, free) {
        return verdict;
    }
    let Ok(blocks) = support_blocks(model, free) else {
        return undetermined(free);
    };
    let mut oversized = false;
    for block in &blocks {
        if block.len() < 2 {
            continue;
        }
        if block.len() > PIVOT_TEST_ORDER_LIMIT {
            oversized = true;
            continue;
        }
        let Ok(matrix) = eye_minus_block(model, block) else {
            return undetermined(free);
        };
        if !pivots_regular(&matrix, block.len()) {
            return (
                Uniqueness::NumericallySingular { members: block.clone() },
                Conditioning::Undefined,
            );
        }
    }
    let (uniqueness, conditioning) = if m == 2 {
        two_by_two_conditioning(k_ff, free)
    } else {
        let norm = max_abs_row_sum(k_ff, m);
        if norm.is_finite() && norm < 1.0 {
            neumann_block(norm, free)
        } else if oversized && !family_admitted {
            return undetermined(free);
        } else {
            (Uniqueness::Unique, not_estimated(free))
        }
    };
    let ill_conditioned = match (&uniqueness, &conditioning) {
        (Uniqueness::Unique, Conditioning::InfinityNorm { kappa, exact: true })
            if *kappa > ILL_CONDITIONED_KAPPA =>
        {
            Some(*kappa)
        }
        _ => None,
    };
    if let Some(kappa) = ill_conditioned {
        return (Uniqueness::IllConditioned { kappa }, conditioning);
    }
    (uniqueness, conditioning)
}

/// Exactly singular 2x2 or 3x3 free block by its closed-form determinant, or an
/// undecided verdict when that determinant is not finite.
fn closed_form_singular(k_ff: &[f64], free: &[usize]) -> Option<(Uniqueness, Conditioning)> {
    match free.len() {
        2 => {
            let product = k_ff[1] * k_ff[2];
            let det = 1.0 - product;
            if !product.is_finite() || !det.is_finite() {
                return Some(undetermined(free));
            }
            #[allow(
                clippy::float_cmp,
                reason = "the 2x2 closed form is exactly singular when 1 - K_01 K_10 evaluates to zero"
            )]
            if det == 0.0 {
                return Some((
                    Uniqueness::Singular(SingularBlock {
                        witness: ConsistencyWitness::LeftNull(vec![k_ff[2], 1.0]),
                    }),
                    Conditioning::Undefined,
                ));
            }
            None
        }
        3 => {
            let mut eye_minus = [0.0; 9];
            for row in 0..3 {
                for col in 0..3 {
                    let coefficient = k_ff[row * 3 + col];
                    eye_minus[row * 3 + col] =
                        if row == col { 1.0 - coefficient } else { -coefficient };
                }
            }
            let det = determinant3(&eye_minus);
            if !det.is_finite() {
                return Some(undetermined(free));
            }
            #[allow(
                clippy::float_cmp,
                reason = "the 3x3 closed form is exactly singular when its determinant evaluates to zero"
            )]
            if det == 0.0 {
                return Some((
                    Uniqueness::Singular(SingularBlock { witness: ConsistencyWitness::NotFormed }),
                    Conditioning::Undefined,
                ));
            }
            None
        }
        _ => None,
    }
}

fn two_by_two_conditioning(k_ff: &[f64], free: &[usize]) -> (Uniqueness, Conditioning) {
    let det = 1.0 - k_ff[1] * k_ff[2];
    let row_sum = (1.0 + k_ff[1].abs()).max(1.0 + k_ff[2].abs());
    let kappa = (row_sum * row_sum) / det.abs();
    if !kappa.is_finite() {
        return (Uniqueness::Unique, not_estimated(free));
    }
    (Uniqueness::Unique, Conditioning::InfinityNorm { kappa, exact: true })
}

fn neumann_block(norm: f64, free: &[usize]) -> (Uniqueness, Conditioning) {
    let kappa = (1.0 + norm) / (1.0 - norm);
    if !kappa.is_finite() {
        return (Uniqueness::Unique, not_estimated(free));
    }
    (Uniqueness::Unique, Conditioning::InfinityNorm { kappa, exact: false })
}

fn determinant3(m: &[f64; 9]) -> f64 {
    let (a, b, c) = (m[0], m[1], m[2]);
    let (d, e, f) = (m[3], m[4], m[5]);
    let (g, h, i) = (m[6], m[7], m[8]);
    a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
}

fn undetermined(free: &[usize]) -> (Uniqueness, Conditioning) {
    let next_attempt = NextAttempt::QuerySpecificFreeBlock { free: free.to_vec() };
    (
        Uniqueness::Undetermined { next_attempt: next_attempt.clone() },
        Conditioning::NotEstimated { next_attempt },
    )
}

fn not_estimated(free: &[usize]) -> Conditioning {
    Conditioning::NotEstimated {
        next_attempt: NextAttempt::QuerySpecificFreeBlock { free: free.to_vec() },
    }
}

fn abs_radius_upper(coefficients: &[f64], n: usize) -> f64 {
    if n == 2 {
        return (coefficients[1].abs() * coefficients[2].abs()).sqrt();
    }
    let by_norm = max_abs_row_sum(coefficients, n);
    if n < 2 {
        return by_norm;
    }
    smaller_upper(by_norm, collatz_wielandt_upper(coefficients, n))
}

fn max_abs_row_sum(coefficients: &[f64], n: usize) -> f64 {
    coefficients
        .chunks_exact(n)
        .map(|row| row.iter().map(|entry| entry.abs()).sum::<f64>())
        .fold(0.0_f64, f64::max)
}

fn smaller_upper(left: f64, right: f64) -> f64 {
    match (left.is_finite(), right.is_finite()) {
        (true, true) => left.min(right),
        (true, false) => left,
        (false, true) => right,
        (false, false) => f64::INFINITY,
    }
}

/// Collatz–Wielandt upper bound `ρ(|K|) ≤ max_i (|K| v)_i / v_i` for `v > 0`.
///
/// Successive iterates and their sum are both probed, so a period-2 cycle
/// still yields a valid upper bound. A bound that stays at or above one does
/// not prove `ρ(|K|) ≥ 1`.
fn collatz_wielandt_upper(coefficients: &[f64], n: usize) -> f64 {
    let mut vector = vec![1.0; n];
    let mut image = vec![0.0; n];
    let mut best = f64::INFINITY;
    for _ in 0..POWER_ITERATIONS {
        abs_matvec(coefficients, n, &vector, &mut image);
        if image.iter().any(|value| !value.is_finite()) {
            break;
        }
        best = best.min(positive_ratio_max(&image, &vector));
        let scale = image.iter().copied().fold(0.0_f64, f64::max);
        if scale <= 0.0 {
            return 0.0;
        }
        if !scale.is_finite() {
            break;
        }
        let mut next = vec![0.0; n];
        for i in 0..n {
            let scaled = image[i] / scale;
            next[i] = if scaled > 0.0 { scaled } else { f64::MIN_POSITIVE };
        }
        let mut summed = vec![0.0; n];
        for i in 0..n {
            summed[i] = vector[i] + next[i];
        }
        abs_matvec(coefficients, n, &summed, &mut image);
        if image.iter().all(|value| value.is_finite()) {
            best = best.min(positive_ratio_max(&image, &summed));
        }
        vector = next;
    }
    best
}

fn positive_ratio_max(image: &[f64], vector: &[f64]) -> f64 {
    let mut max_ratio = 0.0_f64;
    for (&image_i, &coordinate) in image.iter().zip(vector) {
        if coordinate <= 0.0 || !image_i.is_finite() {
            return f64::INFINITY;
        }
        let ratio = image_i / coordinate;
        if !ratio.is_finite() {
            return f64::INFINITY;
        }
        max_ratio = max_ratio.max(ratio);
    }
    max_ratio
}

fn abs_matvec(coefficients: &[f64], n: usize, vector: &[f64], out: &mut [f64]) {
    for (row, image) in out.iter_mut().enumerate() {
        let mut sum = 0.0;
        let start = row * n;
        for (column, &coordinate) in vector.iter().enumerate() {
            sum += coefficients[start + column].abs() * coordinate;
        }
        *image = sum;
    }
}

#[cfg(test)]
mod tests {
    use super::{
        Conditioning, ConsistencyWitness, FamilyCertificate, HardClamp, IdentificationStatus,
        LinearSolutionError, LinearStructuralMap, NONZERO_DIAGONAL_NEXT_ATTEMPT, NextAttempt,
        PhysicalStability, SolvabilityRefusal, SolverConvergence, Uniqueness, assess,
        hard_clamp_mean_batch, revalidate_free_block,
    };
    use crate::cyclic_component::CyclicComponent;
    use crate::intervention_request::{InterventionRequest, compile_under_request};
    use crate::scc_compile::{CompileBudget, GaussianDisturbance, compile_gaussian_equilibrium};

    fn close(left: f64, right: f64) -> bool {
        (left - right).abs() <= 1.0e-12
    }

    fn map(coefficients: Vec<f64>, intercept: Vec<f64>) -> LinearStructuralMap {
        LinearStructuralMap::try_new(coefficients, intercept).expect("fixture map")
    }

    fn independent_unit(mean: Vec<f64>) -> GaussianDisturbance {
        let n = mean.len();
        let mut covariance = vec![0.0; n * n];
        for node in 0..n {
            covariance[node * n + node] = 1.0;
        }
        GaussianDisturbance::try_new(mean, covariance).expect("unit disturbance")
    }

    fn admitted_component(model: &LinearStructuralMap) -> CyclicComponent {
        let family = assess(model, &HardClamp::none(model.dimension()).expect("no clamp"))
            .expect("assessment")
            .family;
        CyclicComponent::try_new(
            "feedback",
            "source-measurements",
            &["solve-note"],
            model,
            &family,
        )
        .expect("component")
    }

    #[test]
    fn two_cycle_association_clamp_derivative_and_mean_shift_are_different() {
        // A = 0.4 B + U_A, B = 0.5 A + U_B. Indices: A = 0, B = 1.
        // Independent unit disturbances. The three responses come from the
        // compiled equilibrium and the clamp batch, not from a local inverse.
        let model = map(vec![0.0, 0.4, 0.5, 0.0], vec![0.0, 0.0]);
        let clamp_a = HardClamp::try_new(2, &[0]).expect("clamp A");
        let report = assess(&model, &clamp_a).expect("assessment");
        let component = admitted_component(&model);
        let open = HardClamp::none(2).expect("no clamp");

        let observed = compile_gaussian_equilibrium(
            &component,
            &model,
            &independent_unit(vec![0.0, 0.0]),
            &open,
            &[],
            CompileBudget::unlimited(),
        )
        .expect("observational equilibrium");
        assert!(close(observed.mean()[0], 0.0));
        assert!(close(observed.mean()[1], 0.0));
        let covariance = observed.covariance();
        assert!(close(covariance[0], 1.8125));
        assert!(close(covariance[1], 1.40625));
        assert!(close(covariance[2], 1.40625));
        assert!(close(covariance[3], 1.953_125));
        // Observational regression of B on A is Cov(A, B) / Var(A).
        let association = covariance[1] / covariance[0];
        assert!(close(association, 45.0 / 58.0));
        assert!((association - 0.776).abs() < 1.0e-3);

        let shocked = compile_gaussian_equilibrium(
            &component,
            &model,
            &independent_unit(vec![1.0, 0.0]),
            &open,
            &[],
            CompileBudget::unlimited(),
        )
        .expect("unit disturbance shock");
        assert!(close(shocked.mean()[0], 1.25));
        let mean_shift_b = shocked.mean()[1];
        assert!(close(mean_shift_b, 0.625));

        let batch = hard_clamp_mean_batch(&model, &clamp_a, &[0.0, 1.0, 2.0], &[0.0, 0.0])
            .expect("clamp batch");
        assert_eq!(batch.rows(), 3);
        assert_eq!(batch.dimension(), 2);
        let at_zero = batch.row(0).expect("do(A = 0)");
        let at_one = batch.row(1).expect("do(A = 1)");
        let at_two = batch.row(2).expect("do(A = 2)");
        assert!(batch.row(3).is_none());
        assert!(close(at_zero[0], 0.0));
        assert!(close(at_zero[1], 0.0));
        assert!(close(at_one[0], 1.0));
        assert!(close(at_one[1], 0.5));
        assert!(close(at_two[0], 2.0));
        assert!(close(at_two[1], 1.0));
        let clamp_response_b = at_one[1] - at_zero[1];
        assert!(close(clamp_response_b, 0.5));
        assert!(close(at_two[1] - at_one[1], 0.5));

        let request = InterventionRequest::hard_clamp(2, &[0], &[1.0]).expect("do(A = 1)");
        let intervened = compile_under_request(
            &component,
            &model,
            &independent_unit(vec![0.0, 0.0]),
            &request,
            CompileBudget::unlimited(),
        )
        .expect("intervened equilibrium");
        assert!(close(intervened.mean()[0], 1.0));
        assert!(close(intervened.mean()[1], at_one[1]));
        assert!(close(intervened.covariance()[0], 0.0));
        assert!(close(intervened.covariance()[1], 0.0));
        assert!(close(intervened.covariance()[3], 1.0));

        assert!((association - clamp_response_b).abs() > 0.2);
        assert!((mean_shift_b - clamp_response_b).abs() > 0.1);
        assert!((association - mean_shift_b).abs() > 0.1);

        assert!(report.family.is_admitted());
        assert!(close(report.family.rho_abs_upper(), (0.2_f64).sqrt()));
        assert_eq!(report.family.identification(), IdentificationStatus::NotMarkedUnidentified);
        assert!(matches!(report.original.uniqueness, Uniqueness::Unique));
        match report.original.conditioning {
            Conditioning::InfinityNorm { kappa, exact: true } => assert!(close(kappa, 2.8125)),
            other => panic!("original conditioning {other:?}"),
        }
        assert_eq!(report.intervened.factor.free, vec![1]);
        assert_eq!(report.intervened.factor.clamped, vec![0]);
        assert!(matches!(report.intervened.uniqueness, Uniqueness::Unique));
        match report.intervened.conditioning {
            Conditioning::InfinityNorm { kappa, exact: true } => assert!(close(kappa, 1.0)),
            other => panic!("clamp conditioning {other:?}"),
        }
        assert_eq!(report.physical_stability, PhysicalStability::Unclaimed);
        assert_eq!(report.solver, SolverConvergence::NotRun);
    }

    #[test]
    fn overlapping_loops_are_uniquely_solvable_until_c_is_clamped() {
        // A = B + C + U_A, B = A + U_B, C = A + U_C.
        // Indices: A = 0, B = 1, C = 2.
        let model = map(vec![0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![0.0, 0.0, 0.0]);
        let clamp_c = HardClamp::try_new(3, &[2]).expect("clamp C");
        let report = assess(&model, &clamp_c).expect("assessment");

        assert!(matches!(report.original.uniqueness, Uniqueness::Unique));
        assert!(matches!(report.original.conditioning, Conditioning::NotEstimated { .. }));
        match report.intervened.uniqueness {
            Uniqueness::Singular(block) => match block.witness {
                ConsistencyWitness::LeftNull(w) => {
                    assert_eq!(w.len(), 2);
                    assert!(close(w[0], 1.0));
                    assert!(close(w[1], 1.0));
                    let k_fh = &report.intervened.factor.k_fh;
                    assert_eq!(k_fh.len(), 2);
                    assert!(close(k_fh[0], 1.0));
                    assert!(close(k_fh[1], 0.0));
                    let clamp_coeff = w[0] * k_fh[0] + w[1] * k_fh[1];
                    assert!(close(clamp_coeff, 1.0));
                    assert!(report.intervened.factor.intercept_f.iter().all(|c| close(*c, 0.0)));
                }
                ConsistencyWitness::NotFormed => panic!("2x2 left null should be formed"),
            },
            other => panic!("intervened uniqueness {other:?}"),
        }
        assert!(matches!(report.intervened.conditioning, Conditioning::Undefined));

        match &report.family {
            FamilyCertificate::Insufficient { rho_abs_upper, next_attempt, identification } => {
                assert!(*rho_abs_upper >= 1.0);
                assert_eq!(next_attempt, &NextAttempt::QuerySpecificFreeBlock { free: vec![0, 1] });
                assert_eq!(*identification, IdentificationStatus::NotMarkedUnidentified);
            }
            FamilyCertificate::Admitted { .. } => {
                panic!("ρ(|K|) < 1 must stay unestablished for the overlapping loops")
            }
        }
        assert_eq!(report.family.identification(), IdentificationStatus::NotMarkedUnidentified);
        assert_eq!(report.physical_stability, PhysicalStability::Unclaimed);
        assert_eq!(report.solver, SolverConvergence::NotRun);
    }

    #[test]
    fn nilpotent_signed_update_does_not_admit_the_abs_family() {
        // K = [[0, 2, 2], [2, 0, 0], [-2, 0, 0]]. K³ = 0, so every eigenvalue
        // of the signed matrix is 0. |K| does not inherit that radius.
        let coefficients = vec![0.0, 2.0, 2.0, 2.0, 0.0, 0.0, -2.0, 0.0, 0.0];
        let square = matmul3(&coefficients, &coefficients);
        let cube = matmul3(&square, &coefficients);
        assert!(cube.iter().all(|entry| close(*entry, 0.0)));

        let model = map(coefficients, vec![0.0, 0.0, 0.0]);
        let full = assess(&model, &HardClamp::none(3).expect("no clamp")).expect("full");
        assert!(!full.family.is_admitted());
        assert!(full.family.rho_abs_upper() >= 1.0);
        assert!(matches!(full.original.uniqueness, Uniqueness::Unique));
        assert_eq!(full.family.identification(), IdentificationStatus::NotMarkedUnidentified);
        assert_eq!(full.physical_stability, PhysicalStability::Unclaimed);
        assert_eq!(full.solver, SolverConvergence::NotRun);

        let clamped =
            assess(&model, &HardClamp::try_new(3, &[2]).expect("clamp third")).expect("clamped");
        assert!(matches!(clamped.intervened.uniqueness, Uniqueness::Unique));
        let k_ff = &clamped.intervened.factor.k_ff;
        let signed_radius = (k_ff[1].abs() * k_ff[2].abs()).sqrt();
        assert!(close(signed_radius, 2.0));
        assert_eq!(clamped.physical_stability, PhysicalStability::Unclaimed);
        assert_eq!(clamped.solver, SolverConvergence::NotRun);
        assert!(!clamped.family.is_admitted());
    }

    #[test]
    fn row_sum_above_one_can_still_admit_the_abs_certificate() {
        // Row sums reach 10, while the spectrum of |K| is {0, ±0.9}.
        let model = map(vec![0.0, 0.9, 0.0, 0.9, 0.0, 0.0, 10.0, 0.0, 0.0], vec![0.0, 0.0, 0.0]);
        let report = assess(&model, &HardClamp::none(3).expect("no clamp")).expect("assessment");
        assert!(report.family.is_admitted());
        assert!(report.family.rho_abs_upper() < 1.0);
        assert!(matches!(report.original.uniqueness, Uniqueness::Unique));
        assert_eq!(report.solver, SolverConvergence::NotRun);
        assert_eq!(report.physical_stability, PhysicalStability::Unclaimed);
    }

    #[test]
    fn unresolved_block_names_a_query_check_and_stays_identified() {
        // A 4-cycle of weight 1 has infinity norm 1, so ρ(|K|) < 1 is not
        // established, and the free block is past the 3×3 closed form.
        let mut coefficients = vec![0.0; 16];
        for node in 0..4 {
            coefficients[node * 4 + (node + 1) % 4] = 1.0;
        }
        let model = map(coefficients, vec![0.0; 4]);
        let report = assess(&model, &HardClamp::none(4).expect("no clamp")).expect("assessment");
        let free = vec![0, 1, 2, 3];
        match report.original.uniqueness {
            Uniqueness::Undetermined { next_attempt } => {
                assert_eq!(
                    next_attempt,
                    NextAttempt::QuerySpecificFreeBlock { free: free.clone() }
                );
            }
            other => panic!("original uniqueness {other:?}"),
        }
        match report.family {
            FamilyCertificate::Insufficient { rho_abs_upper, next_attempt, identification } => {
                assert!(rho_abs_upper >= 1.0);
                assert_eq!(next_attempt, NextAttempt::QuerySpecificFreeBlock { free });
                assert_eq!(identification, IdentificationStatus::NotMarkedUnidentified);
            }
            FamilyCertificate::Admitted { .. } => panic!("a unit 4-cycle is not a contraction"),
        }
        assert_eq!(report.physical_stability, PhysicalStability::Unclaimed);
        assert_eq!(report.solver, SolverConvergence::NotRun);
    }

    #[test]
    fn nonzero_diagonal_non_finite_entries_and_bad_clamps_are_refused() {
        let diagonal = LinearStructuralMap::try_new(vec![0.2, 0.0, 0.0, 0.0], vec![0.0, 0.0]);
        assert_eq!(
            diagonal,
            Err(LinearSolutionError::NonzeroDiagonal {
                node: 0,
                next_attempt: NONZERO_DIAGONAL_NEXT_ATTEMPT,
            })
        );
        assert!(diagonal.unwrap_err().to_string().contains(NONZERO_DIAGONAL_NEXT_ATTEMPT));
        let negative = LinearStructuralMap::try_new(vec![0.0, 1.0, 0.0, -0.2], vec![0.0, 0.0]);
        assert_eq!(
            negative,
            Err(LinearSolutionError::NonzeroDiagonal {
                node: 1,
                next_attempt: NONZERO_DIAGONAL_NEXT_ATTEMPT,
            })
        );
        let infinite =
            LinearStructuralMap::try_new(vec![0.0, f64::INFINITY, 0.0, 0.0], vec![0.0, 0.0]);
        assert!(matches!(
            infinite,
            Err(LinearSolutionError::NonFiniteCoefficient { into_node: 0, from_node: 1 })
        ));
        let shape = LinearStructuralMap::try_new(vec![0.0, 1.0, 0.0], vec![0.0, 0.0]);
        assert!(matches!(shape, Err(LinearSolutionError::Shape { .. })));

        assert_eq!(
            HardClamp::try_new(3, &[1, 0, 1]),
            Err(LinearSolutionError::DuplicateClamp { node: 1 })
        );
        assert_eq!(
            HardClamp::try_new(3, &[3]),
            Err(LinearSolutionError::ClampOutOfRange { node: 3, dimension: 3 })
        );
        let clamp = HardClamp::try_new(3, &[2, 0]).expect("sorted clamp");
        assert_eq!(clamp.indices(), &[0, 2]);
        let model = map(vec![0.0, 0.1, 0.2, 0.0], vec![0.0, 0.0]);
        let mismatch = assess(&model, &HardClamp::none(3).expect("other dimension"));
        assert_eq!(mismatch, Err(LinearSolutionError::ClampDimension { clamp: 3, model: 2 }));
    }

    #[test]
    fn pivot_below_the_floor_is_numerically_singular_not_a_clean_unique() {
        // I - K = [[1, -1], [-(1 + 1e-13), 1]]. After pivoting on the larger
        // first-column entry the second pivot is 1 - 1 / (1 + 1e-13), about 1e-13,
        // which is below the 1e-12 floor. The determinant is -1e-13, not zero, so
        // the block is numerically singular and not exactly singular.
        let model = map(vec![0.0, 1.0, 1.0 + 1.0e-13, 0.0], vec![0.0, 0.0]);
        let report = assess(&model, &HardClamp::none(2).expect("no clamp")).expect("assessment");
        assert_eq!(
            report.original.uniqueness,
            Uniqueness::NumericallySingular { members: vec![0, 1] }
        );
        assert_eq!(report.original.conditioning, Conditioning::Undefined);
        assert!(!matches!(report.original.uniqueness, Uniqueness::Singular(_)));
    }

    #[test]
    fn unique_but_ill_conditioned_block_is_not_reported_as_clean() {
        // det(I - K) = 1 - 1 * (1 - 1e-9) = 1e-9 clears the 1e-12 floor, and
        // kappa = (1 + 1)^2 / 1e-9 = 4e9 exceeds the ill-conditioned bound.
        let model = map(vec![0.0, 1.0, 1.0 - 1.0e-9, 0.0], vec![0.0, 0.0]);
        let report = assess(&model, &HardClamp::none(2).expect("no clamp")).expect("assessment");
        match report.original.uniqueness {
            Uniqueness::IllConditioned { kappa } => {
                assert!((3.9e9..4.1e9).contains(&kappa), "kappa {kappa}");
            }
            other => panic!("uniqueness {other:?}"),
        }
    }

    #[test]
    fn admitted_family_decides_a_large_block_whose_row_sums_exceed_one() {
        // One strongly connected block of order four with a first row summing to
        // 1.2. rho(|K|) is about 0.4606 (power iteration with numpy), so every
        // principal sub-block is uniquely solvable and the block is not left open.
        let mut coefficients = vec![0.1; 16];
        for node in 0..4 {
            coefficients[node * 4 + node] = 0.0;
        }
        coefficients[1] = 0.4;
        coefficients[2] = 0.4;
        coefficients[3] = 0.4;
        let model = map(coefficients, vec![0.0; 4]);
        let report = assess(&model, &HardClamp::none(4).expect("no clamp")).expect("assessment");
        assert!(report.family.is_admitted());
        assert!(matches!(report.original.uniqueness, Uniqueness::Unique));
        assert!(matches!(report.original.conditioning, Conditioning::NotEstimated { .. }));
    }

    #[test]
    fn revalidate_free_block_refuses_a_singular_clamp() {
        // A = B + C, B = A, C = A. do(C) leaves the singular pair (A, B).
        let model = map(vec![0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![0.0, 0.0, 0.0]);
        let clamp = HardClamp::try_new(3, &[2]).expect("clamp C");
        let err = revalidate_free_block(&model, &clamp).expect_err("singular free block");
        assert_eq!(
            err,
            LinearSolutionError::FreeBlockNotSolvable {
                refusal: SolvabilityRefusal::Singular { members: vec![0, 1] },
            }
        );
    }

    #[test]
    fn revalidate_free_block_keeps_ill_conditioning_distinct() {
        // det(I - K) = 1e-9 clears the pivot floor. kappa = 4e9 stays ill-conditioned.
        let ill = map(vec![0.0, 1.0, 1.0 - 1.0e-9, 0.0], vec![0.0, 0.0]);
        let block =
            revalidate_free_block(&ill, &HardClamp::none(2).expect("none")).expect("solvable");
        match block.uniqueness {
            Uniqueness::IllConditioned { kappa } => {
                assert!((3.9e9..4.1e9).contains(&kappa), "kappa {kappa}");
            }
            other => panic!("uniqueness {other:?}"),
        }

        // The second pivot is about 1e-13, below the floor, and the determinant is not zero.
        let near = map(vec![0.0, 1.0, 1.0 + 1.0e-13, 0.0], vec![0.0, 0.0]);
        let err = revalidate_free_block(&near, &HardClamp::none(2).expect("none"))
            .expect_err("numerically singular");
        assert_eq!(
            err,
            LinearSolutionError::FreeBlockNotSolvable {
                refusal: SolvabilityRefusal::NumericallySingular { members: vec![0, 1] },
            }
        );
    }

    fn matmul3(left: &[f64], right: &[f64]) -> [f64; 9] {
        let mut product = [0.0; 9];
        for row in 0..3 {
            for col in 0..3 {
                product[row * 3 + col] = left[row * 3] * right[col]
                    + left[row * 3 + 1] * right[3 + col]
                    + left[row * 3 + 2] * right[6 + col];
            }
        }
        product
    }
}
