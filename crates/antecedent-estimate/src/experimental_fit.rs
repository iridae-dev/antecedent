// SPDX-License-Identifier: MIT OR Apache-2.0

//! Statistical fit of the clamp-experiment recovery recipe from experiment rows.
//!
//! [`antecedent_identify::experimental_recovery`] judges a declared design and
//! reads direct effects from an assembled constraint block. This module is the
//! estimator that turns regime-labelled rows into a recovered linear law and its
//! uncertainty. The model is `x = K x + c + ε` on the retained units, with
//! `K[into, from]` the direct effect of `from` into `into`, a zero diagonal, an
//! intercept `c`, and independent Gaussian disturbances (diagonal `Σε`).
//!
//! # Point fit
//!
//! [`fit_experimental_recovery`] runs, in order:
//!
//! 1. [`validate_regime_contract`] on the declared regimes. A refused contract is
//!    [`ExperimentalFitError::Regime`]; no mechanism is fit on it.
//! 2. [`assemble_experiment_constraints`]: centered moments per regime and one
//!    experimental-effect block per covered target. A target that some ordered
//!    pair leaves uncovered has no block. Its whole incoming row is one
//!    [`UnresolvedReason::IncompletePairCoverage`] direction.
//! 3. The column rank of each block is read from its coefficient matrix with
//!    [`constraint_column_report`]. A rank-deficient block keeps its null
//!    directions as [`AssembledCoefficientRecovery::NullDirections`]; no value is
//!    filled in for them. Only the identified projections onto the row space are
//!    estimated ([`IdentifiedProjection`]).
//! 4. A full-column-rank block is solved by least squares over its stacked rows.
//!    With noise an overdetermined block is never exactly consistent, so the
//!    exact reader's [`ConstraintColumnReport::Inconsistent`] is not a refusal
//!    here: it is the ordinary case of a statistical fit. The rows of regime `r`
//!    are `Σ_JJ⁻¹ Cov(x_J, x_i − b'x_{-i})`, the randomized-clamp moment condition
//!    premultiplied by the clamp covariance. [`ConstraintWeighting::Efficient`]
//!    weights them by `n_r Σ_JJ`, the inverse of their model-based covariance;
//!    [`ConstraintWeighting::Unweighted`] is plain least squares.
//! 5. The disturbance of each resolved row is estimated from its residuals
//!    `x_i − K_i x`, pooled over every regime in which `i` is passive, including
//!    a regime that clamps nothing. The intercept is the pooled residual mean and
//!    the noise mean is zero. The variance uses the declared
//!    [`DisturbanceDivisor`]. Off-diagonal disturbance covariances are fixed at
//!    zero by the recipe's noise law; the largest residual correlation is reported
//!    as a diagnostic and is not used to fill them.
//! 6. The joint covariance of every resolved coefficient is the
//!    heteroskedasticity-robust sandwich ([`CoefficientCovarianceEstimator::Sandwich`])
//!    or the model-based form under the declared Gaussian law
//!    ([`CoefficientCovarianceEstimator::ModelBased`]). Coefficients of different
//!    targets that share a regime are correlated, and the sandwich keeps that
//!    cross covariance.
//! 7. Diagnostics: the condition number of every constraint block, weak stability
//!    (invertibility of `I − U K`) of every declared experiment at the point
//!    estimate, the family certificate `ρ(|K|) < 1`, and the largest residual
//!    correlation. A weak or unstable system is flagged, not refused, and weak
//!    stability is not physical dynamical stability.
//!
//! # Uncertainty
//!
//! Each generator returns a [`RecoveredEstimate`]: a
//! [`RecoveredCoefficientUncertainty`] whose draws keep their own disturbance,
//! the [`RecoveredDrawOrigin`] that produced it, and one [`DrawStatus`] per draw.
//! Unresolved directions are carried into every draw, which keeps such a draw
//! inadmissible. A failed draw stays in the law at its weight.
//!
//! - [`asymptotic_normal_recovery`] draws the resolved coefficients from their
//!   joint asymptotic normal law. A sampling law.
//! - [`bootstrap_experimental_recovery`] resamples experimental units within each
//!   regime under the declared [`DeclaredSamplingDesign`] and re-runs the entire
//!   recovery per replicate. A row-level draw is refused under every design. A
//!   replicate whose recovery fails stays as failed mass. A sampling law.
//! - [`posterior_recovery`] draws from a Gaussian posterior over the identified
//!   coefficient subspace. A null direction is never completed by the prior.
//!   A parameter posterior.
//!
//! Every random stream is keyed by its semantic operation, chain, and replicate
//! through [`RngKey`]. Worker count and batch size never enter a key, so the same
//! seed gives the same law on any schedule.

use std::collections::{BTreeMap, BTreeSet};
use std::convert::Infallible;
use std::fmt;

use antecedent_core::{
    CausalRng, ChainId, DeclaredSamplingDesign, ExecutionContext, FoldId, ReductionPolicy,
    ReplicateId, ResamplingRule, RngKey, SamplingDesignError, SemanticOperation, VariableId,
};
use antecedent_identify::experiment_moments::{
    AssembledExperimentConstraints, ConstraintColumnReport, MissingPair, MomentAssemblyRefusal,
    RegimeSample, TargetConstraintBlock, UnitColumn, assemble_experiment_constraints,
    constraint_column_report,
};
use antecedent_identify::experiment_regime::{
    AcceptedRegimeContract, RegimeContract, RegimeContractDecision, RegimeRefusal,
    validate_regime_contract,
};
use antecedent_identify::experimental_recovery::{
    AssembledCoefficientRecovery, DirectCoefficient, RecoveredCoefficientUncertainty,
    RecoveryDrawInput, UncertaintyFault, UnresolvedDirection, UnresolvedReason,
    joint_draw_from_blocks,
};
use antecedent_kernels::cholesky::cholesky_psd_into;
use antecedent_kernels::{
    back_solve_into, chol_inverse, chol_solve_into, cholesky_spd_into, invert_square,
    sample_inv_gamma, standard_normal, unbiased_index,
};
use antecedent_model::{HardClamp, ILL_CONDITIONED_KAPPA, LinearStructuralMap, assess};
use thiserror::Error;

use crate::recovery::CoefficientCovariance;

/// Semantic operation of one asymptotic-normal coefficient draw.
///
/// The replicate index of the stream is the draw index.
pub const ASYMPTOTIC_DRAW_OPERATION: SemanticOperation =
    SemanticOperation::new(0x4558_5246_4954_0001);

/// Semantic operation of one bootstrap replicate of the whole recovery.
///
/// The replicate index of the stream is the replicate index.
pub const BOOTSTRAP_REPLICATE_OPERATION: SemanticOperation =
    SemanticOperation::new(0x4558_5246_4954_0002);

/// Semantic operation of one posterior sweep.
///
/// The chain index is the constraint block, and the replicate index is the sweep.
pub const POSTERIOR_SWEEP_OPERATION: SemanticOperation =
    SemanticOperation::new(0x4558_5246_4954_0003);

/// Default byte budget of one fit.
pub const DEFAULT_FIT_BYTES: u64 = 1 << 30;

/// Relative tolerance of the semidefinite factor of a coefficient covariance.
const COVARIANCE_FACTOR_TOLERANCE: f64 = 1.0e-12;

/// Smallest norm a candidate basis vector keeps after orthogonalization.
const BASIS_NORM_FLOOR: f64 = 1.0e-8;

/// Bytes of one `f64`.
const F64_BYTES: u64 = 8;

/// How the stacked rows of a constraint block are weighted.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub enum ConstraintWeighting {
    /// Weight the rows of regime `r` by `n_r Σ_JJ`, the inverse of their
    /// model-based covariance under the declared homoskedastic Gaussian law.
    #[default]
    Efficient,
    /// Ordinary least squares on the stacked rows.
    Unweighted,
}

/// How the joint coefficient covariance is estimated.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub enum CoefficientCovarianceEstimator {
    /// Heteroskedasticity-robust sandwich (divisor `n_r`, no small-sample
    /// correction), including the cross covariance of targets that share a regime.
    #[default]
    Sandwich,
    /// Model-based covariance under independent homoskedastic disturbances. Cross
    /// covariances between targets are zero under that law.
    ModelBased,
    /// No covariance. Asymptotic draws are then refused.
    NotComputed,
}

/// Divisor of the pooled residual variance of a resolved row.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub enum DisturbanceDivisor {
    /// Divide the residual sum of squares by the pooled row count `N`. This is the
    /// maximum-likelihood divisor and the divisor of the assembled moments.
    #[default]
    RowCount,
    /// Divide by `N − 1`.
    RowCountMinusOne,
}

/// Options of one fit.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct ExperimentalFitOptions {
    /// Row weighting of the least-squares solve.
    pub weighting: ConstraintWeighting,
    /// Coefficient covariance estimator.
    pub covariance: CoefficientCovarianceEstimator,
    /// Divisor of the disturbance variance.
    pub divisor: DisturbanceDivisor,
    /// Byte budget of the assembly and of the covariance work. Checked before
    /// the memory is allocated.
    pub max_bytes: u64,
}

impl Default for ExperimentalFitOptions {
    fn default() -> Self {
        Self {
            weighting: ConstraintWeighting::default(),
            covariance: CoefficientCovarianceEstimator::default(),
            divisor: DisturbanceDivisor::default(),
            max_bytes: DEFAULT_FIT_BYTES,
        }
    }
}

/// Regime-labelled experiment rows.
///
/// `samples[r]` is the sample of `contract.regimes[r]`, with its declared source
/// identity. Rows of a regime are its experimental units, in one order shared by
/// every column of that regime.
#[derive(Clone, Copy, Debug)]
pub struct ExperimentRows<'a> {
    /// Declared regimes and mask. Validated before anything is fit.
    pub contract: &'a RegimeContract,
    /// One sample per declared regime, in regime order.
    pub samples: &'a [RegimeSample<'a>],
}

/// Why a prior was refused.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PriorFault {
    /// A normal coefficient prior needs a finite mean and a finite positive variance.
    CoefficientPrior,
    /// The disturbance prior names a different number of nodes.
    DisturbanceLength {
        /// Retained units.
        expected: usize,
        /// Entries supplied.
        found: usize,
    },
    /// A known disturbance variance is not finite and positive.
    KnownVariance {
        /// Node index.
        node: usize,
    },
    /// The inverse-Wishart degrees of freedom are not finite and positive.
    DegreesOfFreedom,
    /// An inverse-Wishart scale is not finite and positive.
    Scale {
        /// Node index.
        node: usize,
    },
}

impl fmt::Display for PriorFault {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CoefficientPrior => {
                f.write_str("a normal coefficient prior needs a finite mean and positive variance")
            }
            Self::DisturbanceLength { expected, found } => {
                write!(f, "the disturbance prior names {found} nodes, the law has {expected}")
            }
            Self::KnownVariance { node } => {
                write!(f, "known disturbance variance of node index {node} is not positive")
            }
            Self::DegreesOfFreedom => {
                f.write_str("inverse-Wishart degrees of freedom must be finite and positive")
            }
            Self::Scale { node } => {
                write!(f, "inverse-Wishart scale of node index {node} is not positive")
            }
        }
    }
}

/// Why a fit or an uncertainty law was not produced.
#[derive(Clone, Debug, PartialEq, Error)]
#[non_exhaustive]
pub enum ExperimentalFitError {
    /// The regime contract was refused. Repair it; do not fit on it.
    #[error("regime contract refused: {0}")]
    Regime(RegimeRefusal),
    /// Moment and constraint assembly refused the rows.
    #[error("moment assembly refused: {0}")]
    Assembly(MomentAssemblyRefusal),
    /// Fewer than two retained units: there is no direct coefficient.
    #[error("a recovery needs at least two retained units, found {units}")]
    TooFewUnits {
        /// Retained units.
        units: usize,
    },
    /// A constraint block is not a finite coefficient matrix.
    #[error("constraint block into {target} is not a finite coefficient matrix")]
    BlockNotAConstraint {
        /// Equation of the block.
        target: VariableId,
    },
    /// A full-column-rank block could not be factored.
    #[error("constraint block into {target} could not be solved")]
    BlockSolve {
        /// Equation of the block.
        target: VariableId,
    },
    /// The work estimate exceeds the byte budget. Nothing was allocated for it.
    #[error("the fit estimate of {requested} bytes exceeds the budget of {limit}")]
    BudgetExceeded {
        /// Estimated bytes.
        requested: u64,
        /// Budget.
        limit: u64,
    },
    /// A byte estimate overflowed.
    #[error("a byte estimate overflowed")]
    ResourceOverflow,
    /// The declared sampling design refuses the resampling rule.
    #[error(transparent)]
    SamplingDesign(SamplingDesignError),
    /// A row-level draw. No declared design allows it.
    #[error("resampling rule {0} draws individual rows; no declared design allows it")]
    RowLevelDraw(ResamplingRule),
    /// Whole-cluster resampling needs sampling-unit labels.
    #[error("whole-cluster resampling needs sampling-unit labels for every regime")]
    MissingSamplingUnits,
    /// Sampling-unit labels were supplied for a different number of regimes.
    #[error("{labels} sampling-unit label lists for {regimes} regimes")]
    SamplingUnitCount {
        /// Label lists supplied.
        labels: usize,
        /// Declared regimes.
        regimes: usize,
    },
    /// A regime's labels do not have one entry per row.
    #[error("regime {regime} has {labels} sampling-unit labels for {rows} rows")]
    SamplingUnitShape {
        /// Regime index.
        regime: usize,
        /// Labels supplied.
        labels: usize,
        /// Rows of the regime.
        rows: usize,
    },
    /// Contiguous-block resampling needs a block length in `1..=rows` of every regime.
    #[error("contiguous-block resampling needs a block length between 1 and {rows}")]
    BlockLength {
        /// Rows of the shortest regime.
        rows: usize,
    },
    /// No draws or replicates were requested.
    #[error("at least one draw or replicate must be requested")]
    DrawCount,
    /// A scheduling batch of zero replicates.
    #[error("a scheduling batch must hold at least one replicate")]
    BatchSize,
    /// The fit carries no coefficient covariance.
    #[error("the coefficient covariance was not computed for this fit")]
    CovarianceNotComputed,
    /// The coefficient covariance has a negative direction.
    #[error("the coefficient covariance is not positive semidefinite")]
    CovarianceNotPositiveSemidefinite,
    /// A posterior precision was not positive definite.
    #[error("the posterior precision of the block into {target} is not positive definite")]
    PosteriorPrecision {
        /// Equation of the block.
        target: VariableId,
    },
    /// The prior was refused.
    #[error("prior refused: {0}")]
    Prior(PriorFault),
    /// The coefficient law refused the draws.
    #[error(transparent)]
    Uncertainty(UncertaintyFault),
}

/// Estimated projections of a rank-deficient block onto its identified subspace.
///
/// `basis` is an orthonormal basis of the row space of the block's coefficient
/// matrix, each vector in source order. `estimate[k]` estimates `basis[k] · b`.
/// The orthogonal complement is the block's null space. It is not estimated, and
/// no coordinate of `b` is completed from these projections.
#[derive(Clone, Debug, PartialEq)]
pub struct IdentifiedProjection {
    /// Orthonormal row-space basis.
    pub basis: Vec<Vec<f64>>,
    /// Estimates of the projections, aligned with `basis`.
    pub estimate: Vec<f64>,
}

/// Fit of one constraint block.
#[derive(Clone, Debug, PartialEq)]
pub struct TargetBlockFit {
    /// Equation of the block.
    pub target: VariableId,
    /// Sources of the block, in unit order.
    pub sources: Vec<VariableId>,
    /// Stacked rows.
    pub rows: usize,
    /// Column rank of the coefficient matrix.
    pub column_rank: usize,
    /// Frobenius condition number `‖R‖_F ‖R⁻¹‖_F` of the unweighted coefficient
    /// matrix, an upper bound on its 2-norm condition. Infinite when the matrix is
    /// rank deficient.
    pub condition: f64,
    /// Identified projections, for a rank-deficient block.
    pub identified: Option<IdentifiedProjection>,
}

/// Joint covariance of every resolved coefficient.
#[derive(Clone, Debug, PartialEq)]
pub struct JointCoefficientCovariance {
    /// Coordinates, block by block in unit order, sources in unit order.
    pub coefficients: Vec<DirectCoefficient>,
    /// Row-major covariance, `coefficients.len()` squared.
    pub covariance: Vec<f64>,
    /// Estimator that produced it.
    pub estimator: CoefficientCovarianceEstimator,
}

impl JointCoefficientCovariance {
    /// Covariance of two coordinates, when both are resolved.
    #[must_use]
    pub fn entry(&self, left: DirectCoefficient, right: DirectCoefficient) -> Option<f64> {
        let p = self.coefficients.len();
        let row = self.coefficients.iter().position(|item| *item == left)?;
        let col = self.coefficients.iter().position(|item| *item == right)?;
        self.covariance.get(row * p + col).copied()
    }
}

/// Whether one declared experiment is weakly stable at a coefficient matrix.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ExperimentSolvability {
    /// `I − U K` is invertible with infinity-norm condition `kappa`.
    Invertible {
        /// `‖M‖∞ ‖M⁻¹‖∞` of the free block.
        kappa: f64,
    },
    /// Invertible, but the condition exceeds [`ILL_CONDITIONED_KAPPA`].
    IllConditioned {
        /// `‖M‖∞ ‖M⁻¹‖∞` of the free block.
        kappa: f64,
    },
    /// Singular under the equilibrated pivot policy: the experiment is not weakly stable.
    Singular,
}

/// Stability report of a complete coefficient matrix.
///
/// Weak stability is invertibility of `I − U K` on each declared experiment. The
/// family certificate is the sufficient bound `ρ(|K|) < 1`. Neither is physical
/// dynamical stability, and neither selects a coefficient.
#[derive(Clone, Debug, PartialEq)]
pub struct SystemStability {
    /// Upper bound on `ρ(|K|)`.
    pub rho_abs_upper: f64,
    /// Whether `ρ(|K|) < 1` was established.
    pub family_admitted: bool,
    /// Solvability of each declared regime, in regime order.
    pub experiments: Vec<ExperimentSolvability>,
}

impl SystemStability {
    /// Whether every declared experiment is invertible.
    #[must_use]
    pub fn weakly_stable(&self) -> bool {
        self.experiments.iter().all(|item| !matches!(item, ExperimentSolvability::Singular))
    }
}

/// Largest absolute residual correlation between two resolved rows in one regime.
///
/// The recipe's noise law fixes it at zero. A large value is evidence against
/// that law; it is not used to fill a disturbance covariance.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ResidualCorrelation {
    /// Regime index.
    pub regime: usize,
    /// First node.
    pub left: VariableId,
    /// Second node.
    pub right: VariableId,
    /// Sample correlation of the two residuals in that regime.
    pub correlation: f64,
}

/// A condition the fit records without refusing.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FitFlag {
    /// A target has no constraint block: some ordered pair is not covered.
    IncompleteCoverage {
        /// Equation without a block.
        target: VariableId,
    },
    /// A constraint block is rank deficient. Its null directions stay unresolved.
    RankDeficientConstraint {
        /// Equation of the block.
        target: VariableId,
        /// Column rank.
        column_rank: usize,
        /// Sources.
        width: usize,
    },
    /// A constraint block is ill conditioned: a weak experimental design.
    IllConditionedConstraint {
        /// Equation of the block.
        target: VariableId,
        /// Frobenius condition number.
        condition: f64,
    },
    /// A declared experiment is not weakly stable at the point estimate.
    ExperimentNotWeaklyStable {
        /// Regime index.
        regime: usize,
    },
    /// A declared experiment is weakly stable but ill conditioned at the point estimate.
    IllConditionedExperiment {
        /// Regime index.
        regime: usize,
        /// Infinity-norm condition number.
        kappa: f64,
    },
    /// `ρ(|K|) < 1` was not established at the point estimate.
    FamilyCertificateNotAdmitted {
        /// Upper bound on `ρ(|K|)`.
        rho_abs_upper: f64,
    },
}

/// Diagnostics of a fit.
#[derive(Clone, Debug, PartialEq)]
pub struct FitDiagnostics {
    /// Stability at the point estimate, when every row is resolved.
    pub stability: Option<SystemStability>,
    /// Largest residual correlation between resolved rows.
    pub largest_residual_correlation: Option<ResidualCorrelation>,
    /// Recorded conditions.
    pub flags: Vec<FitFlag>,
}

/// Per-regime layout and moments, in unit order.
#[derive(Clone, Debug, PartialEq)]
struct RegimeLayout {
    clamped: Vec<usize>,
    passive: Vec<usize>,
    rows: usize,
    mean: Vec<f64>,
    covariance: Vec<f64>,
    clamped_covariance: Vec<f64>,
    clamped_inverse: Option<Vec<f64>>,
}

/// Rows of one regime inside a block, contiguous and in clamp order.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct RowGroup {
    regime: usize,
    start: usize,
    len: usize,
}

#[derive(Clone, Debug, PartialEq)]
enum BlockRank {
    Full,
    Deficient { column_rank: usize, directions: Vec<Vec<f64>> },
}

/// Unweighted constraint system of one block.
#[derive(Clone, Debug, PartialEq)]
struct BlockSystem {
    target: usize,
    sources: Vec<usize>,
    a: Vec<f64>,
    y: Vec<f64>,
    groups: Vec<RowGroup>,
    rank: BlockRank,
}

/// Least-squares solution of a block, on the identified subspace when deficient.
#[derive(Clone, Debug, PartialEq)]
struct BlockSolution {
    estimate: Vec<f64>,
    r: Vec<f64>,
    r_inverse: Vec<f64>,
    dim: usize,
    basis: Option<Vec<Vec<f64>>>,
}

/// Pooled moments of the rows in which one node is passive.
#[derive(Clone, Debug, PartialEq)]
struct ResidualMoments {
    rows: usize,
    mean: Vec<f64>,
    covariance: Vec<f64>,
}

impl ResidualMoments {
    fn mean_of(&self, v: &[f64]) -> f64 {
        dot(v, &self.mean)
    }

    /// Variance about the pooled mean, divisor `N`.
    fn variance_of(&self, v: &[f64]) -> f64 {
        quadratic(v, &self.covariance, v).max(0.0)
    }
}

/// Point fit of the clamp-experiment recovery recipe.
///
/// The fit is a point and its sampling covariance. It is not a coefficient law:
/// draw a law with [`asymptotic_normal_recovery`],
/// [`bootstrap_experimental_recovery`], or [`posterior_recovery`].
#[derive(Clone, Debug, PartialEq)]
pub struct ExperimentalRecoveryFit {
    nodes: Vec<VariableId>,
    blocks: Vec<AssembledCoefficientRecovery>,
    block_fits: Vec<TargetBlockFit>,
    coverage_unresolved: Vec<UnresolvedDirection>,
    unresolved: Vec<UnresolvedDirection>,
    intercept: Vec<Option<f64>>,
    disturbance_variance: Vec<Option<f64>>,
    residual_rows: Vec<usize>,
    covariance: Option<JointCoefficientCovariance>,
    offsets: Vec<Option<usize>>,
    options: ExperimentalFitOptions,
    diagnostics: FitDiagnostics,
    systems: Vec<BlockSystem>,
    residuals: Vec<Option<ResidualMoments>>,
    regimes: Vec<RegimeLayout>,
}

impl ExperimentalRecoveryFit {
    /// Retained units, strictly increasing. Intercepts and disturbances use this order.
    #[must_use]
    pub fn nodes(&self) -> &[VariableId] {
        &self.nodes
    }

    /// One recovery per covered target, in unit order.
    ///
    /// [`AssembledCoefficientRecovery::Coefficients`] carries the least-squares
    /// estimate. [`AssembledCoefficientRecovery::NullDirections`] carries the
    /// matrix null space and no value.
    #[must_use]
    pub fn blocks(&self) -> &[AssembledCoefficientRecovery] {
        &self.blocks
    }

    /// Per-block fit details, aligned with [`Self::blocks`].
    #[must_use]
    pub fn block_fits(&self) -> &[TargetBlockFit] {
        &self.block_fits
    }

    /// Every unresolved direction: uncovered rows first, then the null directions
    /// of rank-deficient blocks. Empty only when the recovery goal is closed.
    #[must_use]
    pub fn unresolved(&self) -> &[UnresolvedDirection] {
        &self.unresolved
    }

    /// Whether every direct coefficient was recovered.
    #[must_use]
    pub fn goal_closed(&self) -> bool {
        self.unresolved.is_empty()
    }

    /// Estimated direct effect, when its row is resolved.
    #[must_use]
    pub fn coefficient(&self, coefficient: DirectCoefficient) -> Option<f64> {
        self.blocks.iter().find_map(|block| match block {
            AssembledCoefficientRecovery::Coefficients { target, sources, values }
                if *target == coefficient.into =>
            {
                sources.iter().position(|source| *source == coefficient.from).map(|at| values[at])
            }
            _ => None,
        })
    }

    /// Intercept per node, `None` for an unresolved row. The noise mean is zero.
    #[must_use]
    pub fn intercept(&self) -> &[Option<f64>] {
        &self.intercept
    }

    /// Disturbance variance per node under the declared divisor, `None` for an
    /// unresolved row. Off-diagonal disturbance covariances are zero by the
    /// recipe's noise law.
    #[must_use]
    pub fn disturbance_variance(&self) -> &[Option<f64>] {
        &self.disturbance_variance
    }

    /// Pooled residual rows per node, zero for an unresolved row.
    #[must_use]
    pub fn residual_rows(&self) -> &[usize] {
        &self.residual_rows
    }

    /// Joint coefficient covariance, unless [`CoefficientCovarianceEstimator::NotComputed`].
    #[must_use]
    pub fn coefficient_covariance(&self) -> Option<&JointCoefficientCovariance> {
        self.covariance.as_ref()
    }

    /// Standard error of a resolved coefficient.
    #[must_use]
    pub fn standard_error(&self, coefficient: DirectCoefficient) -> Option<f64> {
        let variance = self.covariance.as_ref()?.entry(coefficient, coefficient)?;
        (variance >= 0.0).then(|| variance.sqrt())
    }

    /// Per-block covariances for [`crate::recovery::asymptotic_normal_draws`].
    ///
    /// These are the diagonal blocks of [`Self::coefficient_covariance`]; the
    /// cross covariance between blocks is dropped there and kept by
    /// [`asymptotic_normal_recovery`].
    #[must_use]
    pub fn block_covariances(&self) -> Vec<CoefficientCovariance> {
        let Some(joint) = &self.covariance else {
            return Vec::new();
        };
        let p = joint.coefficients.len();
        let mut out = Vec::new();
        for (block, (offset, system)) in self.offsets.iter().zip(&self.systems).enumerate() {
            let Some(offset) = *offset else {
                continue;
            };
            let k = system.sources.len();
            let mut covariance = vec![0.0; k * k];
            for row in 0..k {
                for col in 0..k {
                    covariance[row * k + col] = joint.covariance[(offset + row) * p + offset + col];
                }
            }
            out.push(CoefficientCovariance { block, covariance });
        }
        out
    }

    /// Diagnostics of the fit.
    #[must_use]
    pub const fn diagnostics(&self) -> &FitDiagnostics {
        &self.diagnostics
    }

    /// Options the fit ran with.
    #[must_use]
    pub const fn options(&self) -> &ExperimentalFitOptions {
        &self.options
    }

    /// The point as one draw of weight one, with its disturbance.
    ///
    /// An unresolved row has a NaN intercept and a NaN disturbance variance: no
    /// value is filled in. Such a draw carries its unresolved directions and is
    /// inadmissible once retained.
    ///
    /// # Errors
    ///
    /// The faults of [`joint_draw_from_blocks`].
    pub fn point_draw(&self) -> Result<RecoveryDrawInput, UncertaintyFault> {
        self.draw_with_weight(1.0)
    }

    fn draw_with_weight(&self, weight: f64) -> Result<RecoveryDrawInput, UncertaintyFault> {
        let intercept = self.intercept.iter().map(|value| value.unwrap_or(f64::NAN)).collect();
        draw_input(
            &self.nodes,
            &self.blocks,
            &self.coverage_unresolved,
            intercept,
            diagonal(&self.disturbance_variance),
            weight,
        )
    }

    fn clamped_sets(&self) -> Vec<Vec<usize>> {
        self.regimes.iter().map(|layout| layout.clamped.clone()).collect()
    }

    fn status_of(&self, draw: &RecoveryDrawInput) -> DrawStatus {
        draw_status(&self.nodes, &self.clamped_sets(), draw)
    }

    fn divisor_scale(&self, rows: usize) -> f64 {
        divisor_scale(self.options.divisor, rows)
    }
}

/// Fit the clamp-experiment recipe to regime-labelled rows.
///
/// See the module documentation for the steps. A rank-deficient or uncovered row
/// is returned with its unresolved directions, never completed.
///
/// # Errors
///
/// [`ExperimentalFitError::Regime`] for a refused contract,
/// [`ExperimentalFitError::Assembly`] when assembly refuses the rows,
/// [`ExperimentalFitError::TooFewUnits`], block faults, and budget faults.
pub fn fit_experimental_recovery(
    rows: &ExperimentRows<'_>,
    options: &ExperimentalFitOptions,
) -> Result<ExperimentalRecoveryFit, ExperimentalFitError> {
    let accepted = accept(rows.contract)?;
    fit_core(&accepted, rows.samples, options)
}

fn accept(contract: &RegimeContract) -> Result<AcceptedRegimeContract, ExperimentalFitError> {
    match validate_regime_contract(contract) {
        RegimeContractDecision::Accepted(accepted) => Ok(accepted),
        RegimeContractDecision::Refused(refusal) => Err(ExperimentalFitError::Regime(refusal)),
    }
}

struct FittedBlocks {
    systems: Vec<BlockSystem>,
    solutions: Vec<BlockSolution>,
    blocks: Vec<AssembledCoefficientRecovery>,
    block_fits: Vec<TargetBlockFit>,
}

fn fit_core(
    accepted: &AcceptedRegimeContract,
    samples: &[RegimeSample<'_>],
    options: &ExperimentalFitOptions,
) -> Result<ExperimentalRecoveryFit, ExperimentalFitError> {
    let assembled = assemble_experiment_constraints(accepted, samples, options.max_bytes)
        .map_err(ExperimentalFitError::Assembly)?;
    let units = assembled.units.clone();
    let n = units.len();
    if n < 2 {
        return Err(ExperimentalFitError::TooFewUnits { units: n });
    }
    let unit_index: BTreeMap<VariableId, usize> =
        units.iter().enumerate().map(|(index, unit)| (*unit, index)).collect();
    let regimes = regime_layouts(accepted, &assembled, &unit_index);
    let fitted = fit_blocks(&assembled, &units, &unit_index, &regimes, options.weighting)?;
    let coverage_unresolved = coverage_directions(&units, &assembled.uncovered);
    let rows = resolved_rows(n, &fitted.systems, &fitted.blocks);
    let residuals: Vec<Option<ResidualMoments>> = (0..n)
        .map(|target| rows[target].as_ref().and_then(|_| pooled_moments(target, &regimes, n)))
        .collect();
    let vectors: Vec<Option<Vec<f64>>> = rows
        .iter()
        .enumerate()
        .map(|(i, row)| row.as_ref().map(|k| residual_vector(i, k)))
        .collect();
    let mut intercept = vec![None; n];
    let mut disturbance_variance = vec![None; n];
    let mut residual_rows = vec![0; n];
    for target in 0..n {
        if let (Some(moments), Some(v)) = (&residuals[target], &vectors[target]) {
            intercept[target] = Some(moments.mean_of(v));
            disturbance_variance[target] =
                Some(moments.variance_of(v) * divisor_scale(options.divisor, moments.rows));
            residual_rows[target] = moments.rows;
        }
    }
    let (offsets, coefficients) = parameter_layout(&units, &fitted.systems);
    let covariance = match options.covariance {
        CoefficientCovarianceEstimator::NotComputed => None,
        estimator => {
            let columns = regime_columns(&units, samples).ok_or(ExperimentalFitError::Assembly(
                MomentAssemblyRefusal::SampleUnits { regime: 0 },
            ))?;
            let inputs = CovarianceInputs {
                systems: &fitted.systems,
                solutions: &fitted.solutions,
                offsets: &offsets,
                vectors: &vectors,
                variances: &disturbance_variance,
                regimes: &regimes,
                columns: &columns,
                weighting: options.weighting,
                max_bytes: options.max_bytes,
            };
            let matrix = joint_covariance(&inputs, coefficients.len(), estimator)?;
            Some(JointCoefficientCovariance { coefficients, covariance: matrix, estimator })
        }
    };
    let clamped: Vec<Vec<usize>> = regimes.iter().map(|layout| layout.clamped.clone()).collect();
    let diagnostics =
        fit_diagnostics(&units, &fitted, &coverage_unresolved, &rows, &vectors, &regimes, &clamped);
    let mut fit = ExperimentalRecoveryFit {
        nodes: units,
        blocks: fitted.blocks,
        block_fits: fitted.block_fits,
        coverage_unresolved,
        unresolved: Vec::new(),
        intercept,
        disturbance_variance,
        residual_rows,
        covariance,
        offsets,
        options: *options,
        diagnostics,
        systems: fitted.systems,
        residuals,
        regimes,
    };
    let point = fit.point_draw().map_err(ExperimentalFitError::Uncertainty)?;
    fit.unresolved = point.unresolved;
    Ok(fit)
}

fn fit_blocks(
    assembled: &AssembledExperimentConstraints,
    units: &[VariableId],
    unit_index: &BTreeMap<VariableId, usize>,
    regimes: &[RegimeLayout],
    weighting: ConstraintWeighting,
) -> Result<FittedBlocks, ExperimentalFitError> {
    let mut fitted = FittedBlocks {
        systems: Vec::with_capacity(assembled.blocks.len()),
        solutions: Vec::with_capacity(assembled.blocks.len()),
        blocks: Vec::with_capacity(assembled.blocks.len()),
        block_fits: Vec::with_capacity(assembled.blocks.len()),
    };
    for block in &assembled.blocks {
        let target = block.target;
        let rank = match constraint_column_report(block) {
            ConstraintColumnReport::NotAConstraint => {
                return Err(ExperimentalFitError::BlockNotAConstraint { target });
            }
            ConstraintColumnReport::Unique { .. } => BlockRank::Full,
            ConstraintColumnReport::Inconsistent { column_rank }
                if column_rank == block.sources.len() =>
            {
                BlockRank::Full
            }
            ConstraintColumnReport::Inconsistent { .. } => {
                return Err(ExperimentalFitError::BlockSolve { target });
            }
            ConstraintColumnReport::Deficient { column_rank, directions } => {
                BlockRank::Deficient { column_rank, directions }
            }
        };
        let system = build_system(block, units, unit_index, regimes, rank)
            .ok_or(ExperimentalFitError::BlockNotAConstraint { target })?;
        let solution = solve_block(&system, regimes, weighting)
            .ok_or(ExperimentalFitError::BlockSolve { target })?;
        let (recovered, detail) = block_outcome(block, &system, &solution);
        fitted.systems.push(system);
        fitted.solutions.push(solution);
        fitted.blocks.push(recovered);
        fitted.block_fits.push(detail);
    }
    Ok(fitted)
}

fn block_outcome(
    block: &TargetConstraintBlock,
    system: &BlockSystem,
    solution: &BlockSolution,
) -> (AssembledCoefficientRecovery, TargetBlockFit) {
    let k = system.sources.len();
    match &system.rank {
        BlockRank::Full => (
            AssembledCoefficientRecovery::Coefficients {
                target: block.target,
                sources: block.sources.clone(),
                values: solution.estimate.clone(),
            },
            TargetBlockFit {
                target: block.target,
                sources: block.sources.clone(),
                rows: system.y.len(),
                column_rank: k,
                condition: unweighted_condition(system),
                identified: None,
            },
        ),
        BlockRank::Deficient { column_rank, directions } => (
            AssembledCoefficientRecovery::NullDirections {
                target: block.target,
                sources: block.sources.clone(),
                column_rank: *column_rank,
                directions: directions.clone(),
                full_rank_was_asserted: false,
            },
            TargetBlockFit {
                target: block.target,
                sources: block.sources.clone(),
                rows: system.y.len(),
                column_rank: *column_rank,
                condition: f64::INFINITY,
                identified: Some(IdentifiedProjection {
                    basis: solution.basis.clone().unwrap_or_default(),
                    estimate: solution.estimate.clone(),
                }),
            },
        ),
    }
}

fn regime_layouts(
    accepted: &AcceptedRegimeContract,
    assembled: &AssembledExperimentConstraints,
    unit_index: &BTreeMap<VariableId, usize>,
) -> Vec<RegimeLayout> {
    let n = assembled.units.len();
    accepted
        .regimes()
        .iter()
        .zip(&assembled.moments)
        .map(|(regime, moments)| {
            let clamped: Vec<usize> = regime
                .targets
                .iter()
                .filter_map(|target| unit_index.get(&target.unit).copied())
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect();
            let passive: Vec<usize> = (0..n).filter(|index| !clamped.contains(index)).collect();
            let j = clamped.len();
            let mut clamped_covariance = vec![0.0; j * j];
            for (p, &left) in clamped.iter().enumerate() {
                for (q, &right) in clamped.iter().enumerate() {
                    clamped_covariance[p * j + q] = moments.covariance[left * n + right];
                }
            }
            let clamped_inverse =
                if j == 0 { Some(Vec::new()) } else { invert_square(&clamped_covariance, j) };
            RegimeLayout {
                clamped,
                passive,
                rows: moments.rows,
                mean: moments.mean.clone(),
                covariance: moments.covariance.clone(),
                clamped_covariance,
                clamped_inverse,
            }
        })
        .collect()
}

fn build_system(
    block: &TargetConstraintBlock,
    units: &[VariableId],
    unit_index: &BTreeMap<VariableId, usize>,
    regimes: &[RegimeLayout],
    rank: BlockRank,
) -> Option<BlockSystem> {
    let k = block.sources.len();
    let target = *unit_index.get(&block.target)?;
    let sources: Vec<usize> = block
        .sources
        .iter()
        .map(|source| unit_index.get(source).copied())
        .collect::<Option<_>>()?;
    let mut a = Vec::with_capacity(block.coefficients.len());
    let mut y = Vec::with_capacity(block.rhs.len());
    let mut groups = Vec::new();
    let regime_ids: BTreeSet<usize> = block.rows.iter().map(|row| row.regime).collect();
    for regime in regime_ids {
        let layout = regimes.get(regime)?;
        let start = y.len();
        for &clamp in &layout.clamped {
            let probe = units[clamp];
            let row =
                block.rows.iter().position(|row| row.regime == regime && row.probe == probe)?;
            a.extend_from_slice(block.coefficients.get(row * k..(row + 1) * k)?);
            y.push(*block.rhs.get(row)?);
        }
        groups.push(RowGroup { regime, start, len: layout.clamped.len() });
    }
    if y.len() != block.rows.len() {
        return None;
    }
    Some(BlockSystem { target, sources, a, y, groups, rank })
}

/// Rows premultiplied by `L'`, where `L L' = W` is the regime weight.
fn weighted_rows(
    system: &BlockSystem,
    regimes: &[RegimeLayout],
    weighting: ConstraintWeighting,
) -> Option<(Vec<f64>, Vec<f64>)> {
    if weighting == ConstraintWeighting::Unweighted {
        return Some((system.a.clone(), system.y.clone()));
    }
    let k = system.sources.len();
    let mut a = vec![0.0; system.a.len()];
    let mut y = vec![0.0; system.y.len()];
    for group in &system.groups {
        let layout = regimes.get(group.regime)?;
        let j = group.len;
        let rows = layout.rows as f64;
        let weight: Vec<f64> = layout.clamped_covariance.iter().map(|value| value * rows).collect();
        let mut factor = vec![0.0; j * j];
        cholesky_spd_into(&weight, j, &mut factor).ok()?;
        for p in 0..j {
            for c in 0..k {
                a[(group.start + p) * k + c] =
                    (p..j).map(|q| factor[q * j + p] * system.a[(group.start + q) * k + c]).sum();
            }
            y[group.start + p] =
                (p..j).map(|q| factor[q * j + p] * system.y[group.start + q]).sum();
        }
    }
    Some((a, y))
}

fn solve_block(
    system: &BlockSystem,
    regimes: &[RegimeLayout],
    weighting: ConstraintWeighting,
) -> Option<BlockSolution> {
    let (a, y) = weighted_rows(system, regimes, weighting)?;
    let m = y.len();
    let k = system.sources.len();
    let (design, dim, basis) = match &system.rank {
        BlockRank::Full => (a, k, None),
        BlockRank::Deficient { directions, .. } => {
            let basis = row_space_basis(k, directions);
            let d = basis.len();
            let mut design = vec![0.0; m * d];
            for row in 0..m {
                for (col, vector) in basis.iter().enumerate() {
                    design[row * d + col] = dot(&a[row * k..(row + 1) * k], vector);
                }
            }
            (design, d, Some(basis))
        }
    };
    if dim == 0 {
        return Some(BlockSolution {
            estimate: Vec::new(),
            r: Vec::new(),
            r_inverse: Vec::new(),
            dim: 0,
            basis,
        });
    }
    let (estimate, r) = qr_least_squares(&design, &y, m, dim)?;
    let r_inverse = upper_inverse(&r, dim)?;
    Some(BlockSolution { estimate, r, r_inverse, dim, basis })
}

fn unweighted_condition(system: &BlockSystem) -> f64 {
    let k = system.sources.len();
    let m = system.y.len();
    let Some((_, r)) = qr_least_squares(&system.a, &system.y, m, k) else {
        return f64::INFINITY;
    };
    let Some(inverse) = upper_inverse(&r, k) else {
        return f64::INFINITY;
    };
    let norm = |values: &[f64]| values.iter().map(|v| v * v).sum::<f64>().sqrt();
    norm(&r) * norm(&inverse)
}

fn coverage_directions(
    units: &[VariableId],
    uncovered: &[MissingPair],
) -> Vec<UnresolvedDirection> {
    let mut by_target: BTreeMap<VariableId, BTreeSet<VariableId>> = BTreeMap::new();
    for pair in uncovered {
        by_target.entry(pair.into).or_default().insert(pair.from);
    }
    by_target
        .into_iter()
        .map(|(target, missing)| UnresolvedDirection {
            support: units
                .iter()
                .filter(|unit| **unit != target)
                .map(|from| DirectCoefficient { into: target, from: *from })
                .collect(),
            reason: UnresolvedReason::IncompletePairCoverage {
                target,
                missing_clamps: missing.into_iter().collect(),
            },
        })
        .collect()
}

/// Row `K_i` (length `n`, zero diagonal) of every resolved target.
fn resolved_rows(
    n: usize,
    systems: &[BlockSystem],
    blocks: &[AssembledCoefficientRecovery],
) -> Vec<Option<Vec<f64>>> {
    let mut rows = vec![None; n];
    for (system, block) in systems.iter().zip(blocks) {
        if let AssembledCoefficientRecovery::Coefficients { values, .. } = block {
            let mut row = vec![0.0; n];
            for (&source, &value) in system.sources.iter().zip(values) {
                row[source] = value;
            }
            rows[system.target] = Some(row);
        }
    }
    rows
}

/// `v` with `v_i = 1` and `v_j = −K_ij`, so `v · x` is the residual of row `i`.
fn residual_vector(target: usize, row: &[f64]) -> Vec<f64> {
    let mut v: Vec<f64> = row.iter().map(|value| -value).collect();
    v[target] = 1.0;
    v
}

fn pooled_moments(target: usize, regimes: &[RegimeLayout], n: usize) -> Option<ResidualMoments> {
    let members: Vec<&RegimeLayout> =
        regimes.iter().filter(|layout| layout.passive.contains(&target)).collect();
    let rows: usize = members.iter().map(|layout| layout.rows).sum();
    if rows == 0 {
        return None;
    }
    let total = rows as f64;
    let mut mean = vec![0.0; n];
    for layout in &members {
        let share = layout.rows as f64 / total;
        for (slot, value) in mean.iter_mut().zip(&layout.mean) {
            *slot += share * value;
        }
    }
    let mut covariance = vec![0.0; n * n];
    for layout in &members {
        let share = layout.rows as f64 / total;
        for row in 0..n {
            let shift_row = layout.mean[row] - mean[row];
            for col in 0..n {
                let shift_col = layout.mean[col] - mean[col];
                covariance[row * n + col] +=
                    share * (layout.covariance[row * n + col] + shift_row * shift_col);
            }
        }
    }
    Some(ResidualMoments { rows, mean, covariance })
}

fn divisor_scale(divisor: DisturbanceDivisor, rows: usize) -> f64 {
    match divisor {
        DisturbanceDivisor::RowCount => 1.0,
        DisturbanceDivisor::RowCountMinusOne if rows > 1 => rows as f64 / (rows - 1) as f64,
        DisturbanceDivisor::RowCountMinusOne => f64::NAN,
    }
}

fn parameter_layout(
    units: &[VariableId],
    systems: &[BlockSystem],
) -> (Vec<Option<usize>>, Vec<DirectCoefficient>) {
    let mut offsets = Vec::with_capacity(systems.len());
    let mut coefficients = Vec::new();
    for system in systems {
        if system.rank == BlockRank::Full {
            offsets.push(Some(coefficients.len()));
            for &source in &system.sources {
                coefficients
                    .push(DirectCoefficient { into: units[system.target], from: units[source] });
            }
        } else {
            offsets.push(None);
        }
    }
    (offsets, coefficients)
}

fn regime_columns<'a>(
    units: &[VariableId],
    samples: &[RegimeSample<'a>],
) -> Option<Vec<Vec<&'a [f64]>>> {
    samples
        .iter()
        .map(|sample| {
            units
                .iter()
                .map(|unit| {
                    sample.columns.iter().find(|column| column.unit == *unit).map(|c| c.values)
                })
                .collect::<Option<Vec<_>>>()
        })
        .collect()
}

struct CovarianceInputs<'s, 'a> {
    systems: &'s [BlockSystem],
    solutions: &'s [BlockSolution],
    offsets: &'s [Option<usize>],
    vectors: &'s [Option<Vec<f64>>],
    variances: &'s [Option<f64>],
    regimes: &'s [RegimeLayout],
    columns: &'s [Vec<&'a [f64]>],
    weighting: ConstraintWeighting,
    max_bytes: u64,
}

/// Joint covariance `Σ_r G_{s,r} S_{st,r} G_{t,r}'` of the resolved coefficients.
///
/// `G_{s,r} = B_s⁻¹ A_{s,r}' W_r Σ_JJ⁻¹ / n_r` is the influence of regime `r` on
/// block `s`. The sandwich uses `S_{st,r} = Σ_rows z z' ê_s ê_t`; the model-based
/// form uses `S_{ss,r} = σ̂_s² n_r Σ_JJ` and zero across blocks.
fn joint_covariance(
    inputs: &CovarianceInputs<'_, '_>,
    p: usize,
    estimator: CoefficientCovarianceEstimator,
) -> Result<Vec<f64>, ExperimentalFitError> {
    if p == 0 {
        return Ok(Vec::new());
    }
    check_covariance_budget(inputs, p)?;
    let mut covariance = vec![0.0; p * p];
    for (regime, layout) in inputs.regimes.iter().enumerate() {
        let j = layout.clamped.len();
        if j == 0 {
            continue;
        }
        let members: Vec<(usize, RowGroup)> = inputs
            .systems
            .iter()
            .enumerate()
            .filter(|(index, _)| inputs.offsets[*index].is_some())
            .filter_map(|(index, system)| {
                system.groups.iter().find(|group| group.regime == regime).map(|g| (index, *g))
            })
            .collect();
        if members.is_empty() {
            continue;
        }
        let gains = members
            .iter()
            .map(|(index, group)| {
                regime_gain(
                    &inputs.systems[*index],
                    *group,
                    &inputs.solutions[*index],
                    layout,
                    inputs.weighting,
                )
            })
            .collect::<Option<Vec<_>>>()
            .ok_or(ExperimentalFitError::CovarianceNotPositiveSemidefinite)?;
        let scatter = match estimator {
            CoefficientCovarianceEstimator::Sandwich => {
                sandwich_scatter(inputs, regime, layout, &members)
            }
            _ => model_scatter(inputs, layout, &members),
        };
        let t = members.len();
        for left in 0..t {
            for right in left..t {
                let Some(s) = &scatter[left * t + right] else {
                    continue;
                };
                add_gain_product(inputs, &members, &gains, (left, right), s, j, &mut covariance, p);
            }
        }
    }
    for row in 0..p {
        for col in (row + 1)..p {
            let mean = 0.5 * (covariance[row * p + col] + covariance[col * p + row]);
            covariance[row * p + col] = mean;
            covariance[col * p + row] = mean;
        }
    }
    Ok(covariance)
}

fn check_covariance_budget(
    inputs: &CovarianceInputs<'_, '_>,
    p: usize,
) -> Result<(), ExperimentalFitError> {
    let word =
        |value: usize| u64::try_from(value).map_err(|_| ExperimentalFitError::ResourceOverflow);
    let p64 = word(p)?;
    let mut cells = p64.checked_mul(p64).ok_or(ExperimentalFitError::ResourceOverflow)?;
    let mut largest = 0_u64;
    for (regime, layout) in inputs.regimes.iter().enumerate() {
        let members = inputs
            .systems
            .iter()
            .filter(|system| system.groups.iter().any(|group| group.regime == regime))
            .count();
        let j = word(layout.clamped.len())?;
        let t = word(members)?;
        let scatter = t
            .checked_mul(t)
            .and_then(|pairs| pairs.checked_mul(j))
            .and_then(|pairs| pairs.checked_mul(j))
            .ok_or(ExperimentalFitError::ResourceOverflow)?;
        largest = largest.max(scatter);
    }
    cells = cells.checked_add(largest).ok_or(ExperimentalFitError::ResourceOverflow)?;
    let requested = cells.checked_mul(F64_BYTES).ok_or(ExperimentalFitError::ResourceOverflow)?;
    if requested > inputs.max_bytes {
        return Err(ExperimentalFitError::BudgetExceeded { requested, limit: inputs.max_bytes });
    }
    Ok(())
}

/// `G = B⁻¹ M`, `k × j`, with `M = A_r'` (efficient) or `A_r' Σ_JJ⁻¹ / n_r` (unweighted).
fn regime_gain(
    system: &BlockSystem,
    group: RowGroup,
    solution: &BlockSolution,
    layout: &RegimeLayout,
    weighting: ConstraintWeighting,
) -> Option<Vec<f64>> {
    let k = system.sources.len();
    let j = group.len;
    let mut m = vec![0.0; k * j];
    match weighting {
        ConstraintWeighting::Efficient => {
            for c in 0..k {
                for p in 0..j {
                    m[c * j + p] = system.a[(group.start + p) * k + c];
                }
            }
        }
        ConstraintWeighting::Unweighted => {
            let inverse = layout.clamped_inverse.as_ref()?;
            let rows = layout.rows as f64;
            for c in 0..k {
                for p in 0..j {
                    m[c * j + p] = (0..j)
                        .map(|q| system.a[(group.start + q) * k + c] * inverse[q * j + p])
                        .sum::<f64>()
                        / rows;
                }
            }
        }
    }
    let b_inverse = gram_of_upper_inverse(&solution.r_inverse, k);
    let mut gain = vec![0.0; k * j];
    for row in 0..k {
        for p in 0..j {
            gain[row * j + p] = (0..k).map(|c| b_inverse[row * k + c] * m[c * j + p]).sum();
        }
    }
    Some(gain)
}

fn sandwich_scatter(
    inputs: &CovarianceInputs<'_, '_>,
    regime: usize,
    layout: &RegimeLayout,
    members: &[(usize, RowGroup)],
) -> Vec<Option<Vec<f64>>> {
    let t = members.len();
    let j = layout.clamped.len();
    let columns = &inputs.columns[regime];
    let n = columns.len();
    let vectors: Vec<&[f64]> = members
        .iter()
        .map(|(index, _)| inputs.vectors[inputs.systems[*index].target].as_deref().unwrap_or(&[]))
        .collect();
    let mut scatter: Vec<Option<Vec<f64>>> = vec![None; t * t];
    for left in 0..t {
        for right in left..t {
            scatter[left * t + right] = Some(vec![0.0; j * j]);
        }
    }
    let mut dx = vec![0.0; n];
    let mut z = vec![0.0; j];
    let mut residual = vec![0.0; t];
    for row in 0..layout.rows {
        for (c, column) in columns.iter().enumerate() {
            dx[c] = column[row] - layout.mean[c];
        }
        for (p, &clamp) in layout.clamped.iter().enumerate() {
            z[p] = dx[clamp];
        }
        for (slot, v) in residual.iter_mut().zip(&vectors) {
            *slot = dot(v, &dx);
        }
        for left in 0..t {
            for right in left..t {
                let weight = residual[left] * residual[right];
                if let Some(cell) = scatter[left * t + right].as_mut() {
                    for p in 0..j {
                        for q in 0..j {
                            cell[p * j + q] += weight * z[p] * z[q];
                        }
                    }
                }
            }
        }
    }
    scatter
}

fn model_scatter(
    inputs: &CovarianceInputs<'_, '_>,
    layout: &RegimeLayout,
    members: &[(usize, RowGroup)],
) -> Vec<Option<Vec<f64>>> {
    let t = members.len();
    let rows = layout.rows as f64;
    let mut scatter: Vec<Option<Vec<f64>>> = vec![None; t * t];
    for (slot, (index, _)) in members.iter().enumerate() {
        let variance = inputs.variances[inputs.systems[*index].target].unwrap_or(f64::NAN);
        scatter[slot * t + slot] =
            Some(layout.clamped_covariance.iter().map(|value| variance * rows * value).collect());
    }
    scatter
}

#[allow(clippy::too_many_arguments, reason = "one accumulation step over named pieces")]
fn add_gain_product(
    inputs: &CovarianceInputs<'_, '_>,
    members: &[(usize, RowGroup)],
    gains: &[Vec<f64>],
    (left, right): (usize, usize),
    scatter: &[f64],
    j: usize,
    covariance: &mut [f64],
    p: usize,
) {
    let (left_system, right_system) = (members[left].0, members[right].0);
    let (Some(left_offset), Some(right_offset)) =
        (inputs.offsets[left_system], inputs.offsets[right_system])
    else {
        return;
    };
    let kl = inputs.systems[left_system].sources.len();
    let kr = inputs.systems[right_system].sources.len();
    let gl = &gains[left];
    let gr = &gains[right];
    // middle = S G_right'  (j × kr)
    let mut middle = vec![0.0; j * kr];
    for a in 0..j {
        for c in 0..kr {
            middle[a * kr + c] = (0..j).map(|b| scatter[a * j + b] * gr[c * j + b]).sum();
        }
    }
    for row in 0..kl {
        for col in 0..kr {
            let value: f64 = (0..j).map(|a| gl[row * j + a] * middle[a * kr + col]).sum();
            covariance[(left_offset + row) * p + right_offset + col] += value;
            if left != right {
                covariance[(right_offset + col) * p + left_offset + row] += value;
            }
        }
    }
}

fn fit_diagnostics(
    units: &[VariableId],
    fitted: &FittedBlocks,
    coverage: &[UnresolvedDirection],
    rows: &[Option<Vec<f64>>],
    vectors: &[Option<Vec<f64>>],
    regimes: &[RegimeLayout],
    clamped: &[Vec<usize>],
) -> FitDiagnostics {
    let mut flags = Vec::new();
    for direction in coverage {
        if let UnresolvedReason::IncompletePairCoverage { target, .. } = direction.reason {
            flags.push(FitFlag::IncompleteCoverage { target });
        }
    }
    for detail in &fitted.block_fits {
        if detail.column_rank < detail.sources.len() {
            flags.push(FitFlag::RankDeficientConstraint {
                target: detail.target,
                column_rank: detail.column_rank,
                width: detail.sources.len(),
            });
        } else if detail.condition.is_nan() || detail.condition > ILL_CONDITIONED_KAPPA {
            flags.push(FitFlag::IllConditionedConstraint {
                target: detail.target,
                condition: detail.condition,
            });
        }
    }
    let n = units.len();
    let stability = complete_matrix(n, rows).and_then(|k| system_stability(&k, n, clamped));
    if let Some(stability) = &stability {
        for (regime, solvability) in stability.experiments.iter().enumerate() {
            match solvability {
                ExperimentSolvability::Singular => {
                    flags.push(FitFlag::ExperimentNotWeaklyStable { regime });
                }
                ExperimentSolvability::IllConditioned { kappa } => {
                    flags.push(FitFlag::IllConditionedExperiment { regime, kappa: *kappa });
                }
                ExperimentSolvability::Invertible { .. } => {}
            }
        }
        if !stability.family_admitted {
            flags.push(FitFlag::FamilyCertificateNotAdmitted {
                rho_abs_upper: stability.rho_abs_upper,
            });
        }
    }
    FitDiagnostics {
        stability,
        largest_residual_correlation: residual_correlation(units, regimes, vectors),
        flags,
    }
}

fn complete_matrix(n: usize, rows: &[Option<Vec<f64>>]) -> Option<Vec<f64>> {
    let mut k = Vec::with_capacity(n * n);
    for row in rows {
        k.extend_from_slice(row.as_ref()?);
    }
    Some(k)
}

fn system_stability(k: &[f64], n: usize, clamped: &[Vec<usize>]) -> Option<SystemStability> {
    if k.iter().any(|value| !value.is_finite()) {
        return None;
    }
    let model = LinearStructuralMap::try_new(k.to_vec(), vec![0.0; n]).ok()?;
    let open = HardClamp::none(n).ok()?;
    let assessment = assess(&model, &open).ok()?;
    let experiments = clamped.iter().map(|set| experiment_solvability(k, n, set)).collect();
    Some(SystemStability {
        rho_abs_upper: assessment.family.rho_abs_upper(),
        family_admitted: assessment.family.is_admitted(),
        experiments,
    })
}

fn experiment_solvability(k: &[f64], n: usize, clamped: &[usize]) -> ExperimentSolvability {
    let free: Vec<usize> = (0..n).filter(|index| !clamped.contains(index)).collect();
    let f = free.len();
    if f == 0 {
        return ExperimentSolvability::Invertible { kappa: 1.0 };
    }
    let mut block = vec![0.0; f * f];
    for (p, &row) in free.iter().enumerate() {
        for (q, &col) in free.iter().enumerate() {
            let identity = if p == q { 1.0 } else { 0.0 };
            block[p * f + q] = identity - k[row * n + col];
        }
    }
    let Some(inverse) = invert_square(&block, f) else {
        return ExperimentSolvability::Singular;
    };
    let kappa = infinity_norm(&block, f) * infinity_norm(&inverse, f);
    if !kappa.is_finite() {
        return ExperimentSolvability::Singular;
    }
    if kappa > ILL_CONDITIONED_KAPPA {
        ExperimentSolvability::IllConditioned { kappa }
    } else {
        ExperimentSolvability::Invertible { kappa }
    }
}

fn residual_correlation(
    units: &[VariableId],
    regimes: &[RegimeLayout],
    vectors: &[Option<Vec<f64>>],
) -> Option<ResidualCorrelation> {
    let n = units.len();
    let mut best: Option<ResidualCorrelation> = None;
    for (regime, layout) in regimes.iter().enumerate() {
        for (position, &left) in layout.passive.iter().enumerate() {
            let Some(vl) = &vectors[left] else {
                continue;
            };
            for &right in &layout.passive[position + 1..] {
                let Some(vr) = &vectors[right] else {
                    continue;
                };
                let cov = quadratic_n(vl, &layout.covariance, vr, n);
                let var_left = quadratic_n(vl, &layout.covariance, vl, n);
                let var_right = quadratic_n(vr, &layout.covariance, vr, n);
                if !(var_left > 0.0 && var_right > 0.0) {
                    continue;
                }
                let correlation = cov / (var_left * var_right).sqrt();
                if !correlation.is_finite() {
                    continue;
                }
                if best.is_none_or(|current| correlation.abs() > current.correlation.abs()) {
                    best = Some(ResidualCorrelation {
                        regime,
                        left: units[left],
                        right: units[right],
                        correlation,
                    });
                }
            }
        }
    }
    best
}

/// Build one draw: block coefficients, the uncovered rows, and the draw's disturbance.
fn draw_input(
    nodes: &[VariableId],
    blocks: &[AssembledCoefficientRecovery],
    coverage: &[UnresolvedDirection],
    intercept: Vec<f64>,
    noise_covariance: Vec<f64>,
    weight: f64,
) -> Result<RecoveryDrawInput, UncertaintyFault> {
    let noise_mean = vec![0.0; nodes.len()];
    let draw =
        joint_draw_from_blocks(nodes, blocks, intercept, noise_mean, noise_covariance, weight)?;
    let mut unresolved = coverage.to_vec();
    unresolved.extend_from_slice(draw.unresolved());
    Ok(RecoveryDrawInput {
        coefficients: draw.coefficients().to_vec(),
        intercept: draw.intercept().to_vec(),
        noise_mean: draw.noise_mean().to_vec(),
        noise_covariance: draw.noise_covariance().to_vec(),
        unresolved,
        weight,
        marked_inadmissible: false,
    })
}

/// Diagonal disturbance covariance; NaN marks an unresolved row.
fn diagonal(variances: &[Option<f64>]) -> Vec<f64> {
    let n = variances.len();
    let mut out = vec![0.0; n * n];
    for (index, variance) in variances.iter().enumerate() {
        out[index * n + index] = variance.unwrap_or(f64::NAN);
    }
    out
}

fn failed_draw(weight: f64) -> RecoveryDrawInput {
    RecoveryDrawInput {
        coefficients: Vec::new(),
        intercept: Vec::new(),
        noise_mean: Vec::new(),
        noise_covariance: Vec::new(),
        unresolved: Vec::new(),
        weight,
        marked_inadmissible: true,
    }
}

/// Status of one generated or supplied draw.
#[derive(Clone, Debug, PartialEq)]
pub enum DrawStatus {
    /// Every row is resolved. `stability` is the weak-stability and family report
    /// of the draw's matrix over the declared experiments, when it was computed.
    Complete {
        /// Stability of this draw's matrix.
        stability: Option<SystemStability>,
    },
    /// The draw carries unresolved directions. No coefficient was filled for them.
    Unresolved {
        /// Number of unresolved directions.
        directions: usize,
    },
    /// The recovery of this replicate failed. Its weight stays as failed mass.
    Failed {
        /// Why the replicate's recovery failed.
        error: ExperimentalFitError,
    },
    /// A supplied draw the caller marked inadmissible.
    Marked,
}

fn draw_status(
    nodes: &[VariableId],
    clamped: &[Vec<usize>],
    draw: &RecoveryDrawInput,
) -> DrawStatus {
    if draw.marked_inadmissible {
        return DrawStatus::Marked;
    }
    if !draw.unresolved.is_empty() {
        return DrawStatus::Unresolved { directions: draw.unresolved.len() };
    }
    let n = nodes.len();
    let index: BTreeMap<VariableId, usize> =
        nodes.iter().enumerate().map(|(at, node)| (*node, at)).collect();
    let mut k = vec![0.0; n * n];
    let mut named = 0_usize;
    for (coefficient, value) in &draw.coefficients {
        match (index.get(&coefficient.into), index.get(&coefficient.from)) {
            (Some(&into), Some(&from)) if into != from => {
                k[into * n + from] = *value;
                named += 1;
            }
            _ => return DrawStatus::Complete { stability: None },
        }
    }
    let stability = (named == n * (n - 1)).then(|| system_stability(&k, n, clamped)).flatten();
    DrawStatus::Complete { stability }
}

/// What the draws of a [`RecoveredEstimate`] mean to a downstream evidence port.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum DrawLawMeaning {
    /// Repeated-sampling law of the recovery (asymptotic normal or bootstrap).
    /// Not a belief over the true mechanism.
    SamplingLaw,
    /// Posterior over the identified coefficients.
    ParameterPosterior,
}

/// Coefficient prior of [`posterior_recovery`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CoefficientPrior {
    /// Improper flat prior on the identified directions.
    Flat,
    /// Independent `N(mean, variance)` on every coefficient, projected onto the
    /// identified subspace of each block. The projection onto a null direction is
    /// discarded: the prior never supplies an unresolved coordinate.
    Normal {
        /// Prior mean of every coefficient.
        mean: f64,
        /// Prior variance of every coefficient.
        variance: f64,
    },
}

/// Disturbance prior of [`posterior_recovery`].
#[derive(Clone, Debug, PartialEq)]
pub enum DisturbancePrior {
    /// Known disturbance variances, in node order.
    Known {
        /// Variance of each node's disturbance.
        variances: Vec<f64>,
    },
    /// Inverse-Wishart prior on the disturbance covariance.
    ///
    /// The recipe's noise law fixes the off-diagonal entries at zero, so each
    /// diagonal entry carries the one-dimensional inverse-Wishart
    /// `IW₁(degrees_of_freedom, scale[i])`, the scaled inverse chi-square.
    InverseWishart {
        /// Prior degrees of freedom.
        degrees_of_freedom: f64,
        /// Prior scale of each node, in node order.
        scale: Vec<f64>,
    },
}

/// Prior of [`posterior_recovery`]. The intercept prior is flat.
#[derive(Clone, Debug, PartialEq)]
pub struct RecoveryPrior {
    /// Coefficient prior.
    pub coefficients: CoefficientPrior,
    /// Disturbance prior.
    pub disturbance: DisturbancePrior,
}

/// How the draws of a [`RecoveredEstimate`] were produced.
#[derive(Clone, Debug, PartialEq)]
pub enum RecoveredDrawOrigin {
    /// The caller supplied the draws. Their meaning is not declared here.
    Supplied,
    /// [`asymptotic_normal_recovery`].
    AsymptoticNormal {
        /// Number of draws.
        draws: usize,
        /// Seed.
        seed: u64,
    },
    /// [`bootstrap_experimental_recovery`].
    Bootstrap {
        /// Number of replicates, failed ones included.
        replicates: usize,
        /// Seed.
        seed: u64,
        /// Declared sampling design.
        design: DeclaredSamplingDesign,
        /// Resampling rule that design allowed.
        rule: ResamplingRule,
    },
    /// [`posterior_recovery`].
    Posterior {
        /// Prior.
        prior: RecoveryPrior,
        /// Number of retained draws.
        draws: usize,
        /// Discarded sweeps of each chain.
        burn_in: usize,
        /// Seed.
        seed: u64,
    },
}

impl RecoveredDrawOrigin {
    /// Meaning of the draws, or `None` for supplied draws.
    #[must_use]
    pub const fn law_meaning(&self) -> Option<DrawLawMeaning> {
        match self {
            Self::Supplied => None,
            Self::AsymptoticNormal { .. } | Self::Bootstrap { .. } => {
                Some(DrawLawMeaning::SamplingLaw)
            }
            Self::Posterior { .. } => Some(DrawLawMeaning::ParameterPosterior),
        }
    }
}

/// Gaussian posterior of the identified projections of a rank-deficient block.
///
/// Available under a known disturbance variance. `mean[k]` and `covariance` are
/// the posterior of `basis[k] · b`. The null directions of the block stay
/// unresolved in every draw.
#[derive(Clone, Debug, PartialEq)]
pub struct IdentifiedPosteriorBlock {
    /// Equation of the block.
    pub target: VariableId,
    /// Sources of the block.
    pub sources: Vec<VariableId>,
    /// Orthonormal row-space basis.
    pub basis: Vec<Vec<f64>>,
    /// Posterior mean of the projections.
    pub mean: Vec<f64>,
    /// Row-major posterior covariance of the projections.
    pub covariance: Vec<f64>,
}

/// A recovered coefficient law with the record of how its draws were produced.
#[derive(Clone, Debug, PartialEq)]
#[must_use = "dropping the estimate drops failed and unresolved mass"]
pub struct RecoveredEstimate {
    law: RecoveredCoefficientUncertainty,
    origin: RecoveredDrawOrigin,
    statuses: Vec<DrawStatus>,
    identified_posteriors: Vec<IdentifiedPosteriorBlock>,
}

impl RecoveredEstimate {
    /// Record a caller-supplied law as [`RecoveredDrawOrigin::Supplied`].
    pub fn supplied(law: RecoveredCoefficientUncertainty) -> Self {
        let statuses = law
            .draws()
            .iter()
            .map(|draw| {
                if draw.marked_inadmissible() {
                    DrawStatus::Marked
                } else if draw.unresolved().is_empty() {
                    DrawStatus::Complete { stability: None }
                } else {
                    DrawStatus::Unresolved { directions: draw.unresolved().len() }
                }
            })
            .collect();
        Self {
            law,
            origin: RecoveredDrawOrigin::Supplied,
            statuses,
            identified_posteriors: Vec::new(),
        }
    }

    /// The coefficient law, failed and inadmissible draws included.
    pub const fn law(&self) -> &RecoveredCoefficientUncertainty {
        &self.law
    }

    /// The law, consuming the record.
    pub fn into_law(self) -> RecoveredCoefficientUncertainty {
        self.law
    }

    /// How the draws were produced.
    #[must_use]
    pub const fn origin(&self) -> &RecoveredDrawOrigin {
        &self.origin
    }

    /// Meaning of the draws for a downstream evidence port.
    #[must_use]
    pub const fn law_meaning(&self) -> Option<DrawLawMeaning> {
        self.origin.law_meaning()
    }

    /// One status per draw, in law order.
    #[must_use]
    pub fn draw_status(&self) -> &[DrawStatus] {
        &self.statuses
    }

    /// Mass of replicates whose recovery failed.
    #[must_use]
    pub fn failed_mass(&self) -> f64 {
        self.law
            .draws()
            .iter()
            .zip(&self.statuses)
            .filter(|(_, status)| matches!(status, DrawStatus::Failed { .. }))
            .map(|(draw, _)| draw.weight())
            .sum()
    }

    /// Identified-subspace posteriors of rank-deficient blocks.
    #[must_use]
    pub fn identified_posteriors(&self) -> &[IdentifiedPosteriorBlock] {
        &self.identified_posteriors
    }
}

fn keyed_stream(
    operation: SemanticOperation,
    chain: usize,
    replicate: usize,
    seed: u64,
) -> CausalRng {
    RngKey::new(
        operation,
        FoldId::new(0),
        ChainId::new(index_word(chain)),
        ReplicateId::new(index_word(replicate)),
        ReductionPolicy::Strict,
    )
    .open(seed)
}

fn index_word(index: usize) -> u64 {
    u64::try_from(index).unwrap_or(u64::MAX)
}

/// Draw the recovered law from the joint asymptotic normal law of the fit.
///
/// Draw `d` opens the stream [`ASYMPTOTIC_DRAW_OPERATION`] with replicate `d`.
/// It adds `L z` to the resolved coefficients, where `L L'` is the joint
/// covariance, so the cross covariance between blocks is kept. Each resolved
/// row's intercept and disturbance variance are then recomputed from the drawn
/// coefficients and perturbed by their own asymptotic sampling error: the
/// residual mean by `N(0, σ²/N)` and the variance on the log scale by
/// `N(−1/N, 2/N)`. Under randomized clamps and a symmetric disturbance both are
/// asymptotically independent of the coefficient estimate. Unresolved
/// directions are carried into every draw.
///
/// # Errors
///
/// [`ExperimentalFitError::DrawCount`], [`ExperimentalFitError::CovarianceNotComputed`],
/// [`ExperimentalFitError::CovarianceNotPositiveSemidefinite`], or a law fault.
pub fn asymptotic_normal_recovery(
    fit: &ExperimentalRecoveryFit,
    draws: usize,
    seed: u64,
) -> Result<RecoveredEstimate, ExperimentalFitError> {
    if draws == 0 {
        return Err(ExperimentalFitError::DrawCount);
    }
    let covariance = fit.covariance.as_ref().ok_or(ExperimentalFitError::CovarianceNotComputed)?;
    let p = covariance.coefficients.len();
    let mut factor = vec![0.0; p * p];
    if p > 0 {
        cholesky_psd_into(&covariance.covariance, p, COVARIANCE_FACTOR_TOLERANCE, &mut factor)
            .map_err(|_| ExperimentalFitError::CovarianceNotPositiveSemidefinite)?;
    }
    let weight = 1.0 / draws as f64;
    let mut inputs = Vec::with_capacity(draws);
    let mut statuses = Vec::with_capacity(draws);
    for draw in 0..draws {
        let mut rng = keyed_stream(ASYMPTOTIC_DRAW_OPERATION, 0, draw, seed);
        let z: Vec<f64> = (0..p).map(|_| standard_normal(&mut rng)).collect();
        let delta: Vec<f64> =
            (0..p).map(|row| (0..=row).map(|col| factor[row * p + col] * z[col]).sum()).collect();
        let blocks = perturbed_blocks(fit, &delta);
        let rows = resolved_rows(fit.nodes.len(), &fit.systems, &blocks);
        let (intercept, variance) = asymptotic_disturbance(fit, &rows, &mut rng);
        let input = draw_input(
            &fit.nodes,
            &blocks,
            &fit.coverage_unresolved,
            intercept,
            diagonal(&variance),
            weight,
        )
        .map_err(ExperimentalFitError::Uncertainty)?;
        statuses.push(fit.status_of(&input));
        inputs.push(input);
    }
    let law = RecoveredCoefficientUncertainty::retain(&fit.nodes, inputs)
        .map_err(ExperimentalFitError::Uncertainty)?;
    Ok(RecoveredEstimate {
        law,
        origin: RecoveredDrawOrigin::AsymptoticNormal { draws, seed },
        statuses,
        identified_posteriors: Vec::new(),
    })
}

fn perturbed_blocks(
    fit: &ExperimentalRecoveryFit,
    delta: &[f64],
) -> Vec<AssembledCoefficientRecovery> {
    fit.blocks
        .iter()
        .zip(&fit.offsets)
        .map(|(block, offset)| match (block, offset) {
            (AssembledCoefficientRecovery::Coefficients { target, sources, values }, Some(at)) => {
                AssembledCoefficientRecovery::Coefficients {
                    target: *target,
                    sources: sources.clone(),
                    values: values.iter().zip(&delta[*at..]).map(|(v, d)| v + d).collect(),
                }
            }
            (other, _) => other.clone(),
        })
        .collect()
}

fn asymptotic_disturbance(
    fit: &ExperimentalRecoveryFit,
    rows: &[Option<Vec<f64>>],
    rng: &mut CausalRng,
) -> (Vec<f64>, Vec<Option<f64>>) {
    let n = fit.nodes.len();
    let mut intercept = vec![f64::NAN; n];
    let mut variance = vec![None; n];
    for target in 0..n {
        let (Some(row), Some(moments)) = (&rows[target], &fit.residuals[target]) else {
            continue;
        };
        let v = residual_vector(target, row);
        let count = moments.rows as f64;
        let base = moments.variance_of(&v);
        let mean_noise = standard_normal(rng);
        let log_noise = standard_normal(rng);
        intercept[target] = moments.mean_of(&v) + (base / count).sqrt() * mean_noise;
        let scaled = base * fit.divisor_scale(moments.rows);
        variance[target] = Some(scaled * ((2.0 / count).sqrt() * log_noise - 1.0 / count).exp());
    }
    (intercept, variance)
}

/// Sampling design and schedule of [`bootstrap_experimental_recovery`].
#[derive(Clone, Copy, Debug)]
pub struct BootstrapSpec<'a> {
    /// Declared sampling design of the experimental units.
    pub design: DeclaredSamplingDesign,
    /// Resampling rule. It must be the rule `design` allows.
    pub rule: ResamplingRule,
    /// Replicates, failed ones included.
    pub replicates: usize,
    /// Seed.
    pub seed: u64,
    /// Sampling-unit label of every row, one list per regime. Rows sharing a label
    /// are one subject's trajectory or one cluster and are drawn together.
    /// Required for [`ResamplingRule::WholeCluster`]; optional for
    /// [`ResamplingRule::RowSubjectIid`], where its absence makes each row a subject.
    pub sampling_units: Option<&'a [&'a [u64]]>,
    /// Block length of [`ResamplingRule::ContiguousBlock`].
    pub block_length: Option<usize>,
    /// Replicates per scheduled batch. Scheduling only: it never changes a draw.
    pub batch: usize,
}

/// Validated resampling plan of every regime.
struct ResamplingPlan {
    rule: ResamplingRule,
    rows: Vec<usize>,
    groups: Option<Vec<Vec<Vec<usize>>>>,
    block_length: usize,
}

impl ResamplingPlan {
    fn new(
        rule: ResamplingRule,
        samples: &[RegimeSample<'_>],
        spec: &BootstrapSpec<'_>,
    ) -> Result<Self, ExperimentalFitError> {
        let rows: Vec<usize> = samples
            .iter()
            .map(|sample| sample.columns.first().map_or(0, |column| column.values.len()))
            .collect();
        let groups = match spec.sampling_units {
            None => None,
            Some(labels) => Some(group_rows(labels, &rows)?),
        };
        let shortest = rows.iter().copied().min().unwrap_or(0);
        let mut block_length = 1;
        match rule {
            ResamplingRule::RowSubjectIid => {}
            ResamplingRule::WholeCluster => {
                if groups.is_none() {
                    return Err(ExperimentalFitError::MissingSamplingUnits);
                }
            }
            ResamplingRule::ContiguousBlock => {
                block_length = spec
                    .block_length
                    .filter(|length| (1..=shortest).contains(length))
                    .ok_or(ExperimentalFitError::BlockLength { rows: shortest })?;
            }
            ResamplingRule::RowIid => return Err(ExperimentalFitError::RowLevelDraw(rule)),
        }
        Ok(Self { rule, rows, groups, block_length })
    }

    fn draw(&self, regime: usize, rng: &mut CausalRng) -> Vec<usize> {
        let rows = self.rows[regime];
        let mut out = Vec::with_capacity(rows);
        match (self.rule, &self.groups) {
            (ResamplingRule::ContiguousBlock, _) => {
                let starts = rows - self.block_length + 1;
                while out.len() < rows {
                    let start = unbiased_index(rng, starts);
                    out.extend(start..start + self.block_length);
                }
                out.truncate(rows);
            }
            (_, Some(groups)) => {
                let units = &groups[regime];
                for _ in 0..units.len() {
                    out.extend_from_slice(&units[unbiased_index(rng, units.len())]);
                }
            }
            (_, None) => {
                for _ in 0..rows {
                    out.push(unbiased_index(rng, rows));
                }
            }
        }
        out
    }
}

fn group_rows(
    labels: &[&[u64]],
    rows: &[usize],
) -> Result<Vec<Vec<Vec<usize>>>, ExperimentalFitError> {
    if labels.len() != rows.len() {
        return Err(ExperimentalFitError::SamplingUnitCount {
            labels: labels.len(),
            regimes: rows.len(),
        });
    }
    labels
        .iter()
        .zip(rows)
        .enumerate()
        .map(|(regime, (regime_labels, &count))| {
            if regime_labels.len() != count {
                return Err(ExperimentalFitError::SamplingUnitShape {
                    regime,
                    labels: regime_labels.len(),
                    rows: count,
                });
            }
            let mut by_label: BTreeMap<u64, Vec<usize>> = BTreeMap::new();
            for (row, label) in regime_labels.iter().enumerate() {
                by_label.entry(*label).or_default().push(row);
            }
            Ok(by_label.into_values().collect())
        })
        .collect()
}

fn run_replicate(
    accepted: &AcceptedRegimeContract,
    samples: &[RegimeSample<'_>],
    plan: &ResamplingPlan,
    options: &ExperimentalFitOptions,
    replicate: usize,
    seed: u64,
) -> Result<ExperimentalRecoveryFit, ExperimentalFitError> {
    let mut rng = keyed_stream(BOOTSTRAP_REPLICATE_OPERATION, 0, replicate, seed);
    let data: Vec<Vec<Vec<f64>>> = samples
        .iter()
        .enumerate()
        .map(|(regime, sample)| {
            let rows = plan.draw(regime, &mut rng);
            sample
                .columns
                .iter()
                .map(|column| rows.iter().map(|&row| column.values[row]).collect())
                .collect()
        })
        .collect();
    let columns: Vec<Vec<UnitColumn<'_>>> = data
        .iter()
        .zip(samples)
        .map(|(regime, sample)| {
            regime
                .iter()
                .zip(sample.columns)
                .map(|(values, column)| UnitColumn { unit: column.unit, values: values.as_slice() })
                .collect()
        })
        .collect();
    let resampled: Vec<RegimeSample<'_>> = columns
        .iter()
        .zip(samples)
        .map(|(regime, sample)| RegimeSample { columns: regime.as_slice(), source: sample.source })
        .collect();
    fit_core(accepted, &resampled, options)
}

/// Whole-procedure nonparametric bootstrap of the recovery.
///
/// `spec.design` must allow `spec.rule`
/// ([`DeclaredSamplingDesign::apply_resampling`]): a row-level draw is refused
/// under every design, a subject trajectory or a cluster is drawn whole, and a
/// dependent series is drawn in contiguous blocks. Units are resampled within
/// each regime, with replacement, to the regime's own count, so regimes stay
/// independent and keep their clamp sets. Each replicate re-runs the moment
/// assembly, the rank-aware block solves, and the disturbance estimate. A
/// replicate whose recovery fails is retained as a marked draw with
/// [`DrawStatus::Failed`]; its weight is not renormalized away.
///
/// Replicate `b` opens [`BOOTSTRAP_REPLICATE_OPERATION`] with replicate `b`.
/// Replicates are scheduled in batches of `spec.batch` on `ctx`; neither the
/// batch size nor the worker count changes a draw.
///
/// # Errors
///
/// A refused design or rule, a malformed label or block length, a refused
/// contract, a fit of the original rows that fails, or a law fault.
pub fn bootstrap_experimental_recovery(
    rows: &ExperimentRows<'_>,
    options: &ExperimentalFitOptions,
    spec: &BootstrapSpec<'_>,
    ctx: &ExecutionContext,
) -> Result<RecoveredEstimate, ExperimentalFitError> {
    if spec.replicates == 0 {
        return Err(ExperimentalFitError::DrawCount);
    }
    if spec.batch == 0 {
        return Err(ExperimentalFitError::BatchSize);
    }
    let bound =
        spec.design.apply_resampling(spec.rule).map_err(ExperimentalFitError::SamplingDesign)?;
    let accepted = accept(rows.contract)?;
    let replicate_options = ExperimentalFitOptions {
        covariance: CoefficientCovarianceEstimator::NotComputed,
        ..*options
    };
    let reference = fit_core(&accepted, rows.samples, &replicate_options)?;
    let plan = ResamplingPlan::new(bound.rule(), rows.samples, spec)?;
    let batches = spec.replicates.div_ceil(spec.batch);
    let outcomes = ctx.map_indexed(batches, |batch, _inner| {
        let start = batch * spec.batch;
        let end = (start + spec.batch).min(spec.replicates);
        Ok::<_, Infallible>(
            (start..end)
                .map(|replicate| {
                    run_replicate(
                        &accepted,
                        rows.samples,
                        &plan,
                        &replicate_options,
                        replicate,
                        spec.seed,
                    )
                })
                .collect::<Vec<_>>(),
        )
    });
    let outcomes: Vec<Result<ExperimentalRecoveryFit, ExperimentalFitError>> = match outcomes {
        Ok(batches) => batches.into_iter().flatten().collect(),
        Err(never) => match never {},
    };
    let weight = 1.0 / spec.replicates as f64;
    let mut inputs = Vec::with_capacity(outcomes.len());
    let mut statuses = Vec::with_capacity(outcomes.len());
    for outcome in outcomes {
        let drawn = outcome.and_then(|fit| {
            fit.draw_with_weight(weight).map_err(ExperimentalFitError::Uncertainty)
        });
        match drawn {
            Ok(input) => {
                statuses.push(reference.status_of(&input));
                inputs.push(input);
            }
            Err(error) => {
                statuses.push(DrawStatus::Failed { error });
                inputs.push(failed_draw(weight));
            }
        }
    }
    let law = RecoveredCoefficientUncertainty::retain(&reference.nodes, inputs)
        .map_err(ExperimentalFitError::Uncertainty)?;
    Ok(RecoveredEstimate {
        law,
        origin: RecoveredDrawOrigin::Bootstrap {
            replicates: spec.replicates,
            seed: spec.seed,
            design: spec.design,
            rule: spec.rule,
        },
        statuses,
        identified_posteriors: Vec::new(),
    })
}

/// Prior and schedule of [`posterior_recovery`].
#[derive(Clone, Debug, PartialEq)]
pub struct PosteriorSpec {
    /// Prior.
    pub prior: RecoveryPrior,
    /// Retained draws.
    pub draws: usize,
    /// Discarded sweeps of each Gibbs chain. Unused under a known disturbance.
    pub burn_in: usize,
    /// Seed.
    pub seed: u64,
}

/// One posterior draw of a resolved row.
struct RowDraw {
    coefficients: Vec<f64>,
    intercept: f64,
    variance: f64,
}

/// Draw the recovered law from a Gaussian posterior over the identified subspace.
///
/// The likelihood is the asymptotic Gaussian law of the efficient constraint
/// estimate, `b̂ | b, σ² ~ N(b, σ² B⁻¹)` with `B = Σ_r A_r' n_r Σ_JJ A_r`, a
/// limited-information likelihood of the experimental-effect moments rather than
/// the full cyclic likelihood. The pooled residual scatter informs `σ²`
/// (`N σ̂² | σ² ~ σ² χ²_{N−1}`) and the pooled residual mean informs the
/// intercept under a flat prior (`c | b, σ² ~ N(x̄_i − b'x̄_{-i}, σ²/N)`).
///
/// - [`DisturbancePrior::Known`]: draws are independent; stream
///   [`POSTERIOR_SWEEP_OPERATION`], chain = block, replicate = draw.
/// - [`DisturbancePrior::InverseWishart`]: a Gibbs chain per block alternates
///   `b | σ²` (Gaussian) and `σ² | b` (inverse gamma); `burn_in` sweeps are
///   discarded; replicate = sweep.
///
/// Rows are independent under the diagonal noise law. A rank-deficient block
/// has a posterior over its identified projections only
/// ([`IdentifiedPosteriorBlock`], under a known disturbance variance); its null
/// directions stay unresolved in every draw, whatever the prior. An uncovered
/// row has no posterior.
///
/// # Errors
///
/// [`ExperimentalFitError::DrawCount`], [`ExperimentalFitError::Prior`],
/// [`ExperimentalFitError::PosteriorPrecision`], a block fault, or a law fault.
pub fn posterior_recovery(
    fit: &ExperimentalRecoveryFit,
    spec: &PosteriorSpec,
) -> Result<RecoveredEstimate, ExperimentalFitError> {
    if spec.draws == 0 {
        return Err(ExperimentalFitError::DrawCount);
    }
    let n = fit.nodes.len();
    validate_prior(&spec.prior, n).map_err(ExperimentalFitError::Prior)?;
    let mut row_draws: Vec<Option<Vec<RowDraw>>> = Vec::with_capacity(fit.systems.len());
    let mut identified = Vec::new();
    for (chain, system) in fit.systems.iter().enumerate() {
        let target = fit.nodes[system.target];
        let solution = solve_block(system, &fit.regimes, ConstraintWeighting::Efficient)
            .ok_or(ExperimentalFitError::BlockSolve { target })?;
        match system.rank {
            BlockRank::Full => {
                row_draws.push(Some(posterior_row(fit, system, &solution, spec, chain)?));
            }
            BlockRank::Deficient { .. } => {
                row_draws.push(None);
                if let Some(block) = identified_posterior(fit, system, &solution, &spec.prior)? {
                    identified.push(block);
                }
            }
        }
    }
    let weight = 1.0 / spec.draws as f64;
    let mut inputs = Vec::with_capacity(spec.draws);
    let mut statuses = Vec::with_capacity(spec.draws);
    for draw in 0..spec.draws {
        let mut intercept = vec![f64::NAN; n];
        let mut variance = vec![None; n];
        let blocks: Vec<AssembledCoefficientRecovery> = fit
            .blocks
            .iter()
            .zip(fit.systems.iter().zip(&row_draws))
            .map(|(block, (system, draws))| match (block, draws) {
                (
                    AssembledCoefficientRecovery::Coefficients { target, sources, .. },
                    Some(draws),
                ) => {
                    let row = &draws[draw];
                    intercept[system.target] = row.intercept;
                    variance[system.target] = Some(row.variance);
                    AssembledCoefficientRecovery::Coefficients {
                        target: *target,
                        sources: sources.clone(),
                        values: row.coefficients.clone(),
                    }
                }
                (other, _) => other.clone(),
            })
            .collect();
        let input = draw_input(
            &fit.nodes,
            &blocks,
            &fit.coverage_unresolved,
            intercept,
            diagonal(&variance),
            weight,
        )
        .map_err(ExperimentalFitError::Uncertainty)?;
        statuses.push(fit.status_of(&input));
        inputs.push(input);
    }
    let law = RecoveredCoefficientUncertainty::retain(&fit.nodes, inputs)
        .map_err(ExperimentalFitError::Uncertainty)?;
    Ok(RecoveredEstimate {
        law,
        origin: RecoveredDrawOrigin::Posterior {
            prior: spec.prior.clone(),
            draws: spec.draws,
            burn_in: spec.burn_in,
            seed: spec.seed,
        },
        statuses,
        identified_posteriors: identified,
    })
}

fn validate_prior(prior: &RecoveryPrior, n: usize) -> Result<(), PriorFault> {
    if let CoefficientPrior::Normal { mean, variance } = prior.coefficients {
        if !(mean.is_finite() && variance.is_finite() && variance > 0.0) {
            return Err(PriorFault::CoefficientPrior);
        }
    }
    match &prior.disturbance {
        DisturbancePrior::Known { variances } => {
            if variances.len() != n {
                return Err(PriorFault::DisturbanceLength { expected: n, found: variances.len() });
            }
            if let Some(node) = variances.iter().position(|v| !(v.is_finite() && *v > 0.0)) {
                return Err(PriorFault::KnownVariance { node });
            }
        }
        DisturbancePrior::InverseWishart { degrees_of_freedom, scale } => {
            if !(degrees_of_freedom.is_finite() && *degrees_of_freedom > 0.0) {
                return Err(PriorFault::DegreesOfFreedom);
            }
            if scale.len() != n {
                return Err(PriorFault::DisturbanceLength { expected: n, found: scale.len() });
            }
            if let Some(node) = scale.iter().position(|v| !(v.is_finite() && *v > 0.0)) {
                return Err(PriorFault::Scale { node });
            }
        }
    }
    Ok(())
}

/// Prior precision (scalar times identity) and prior shift `Λ₀ m₀` on a subspace.
fn prior_terms(prior: CoefficientPrior, basis: Option<&[Vec<f64>]>, dim: usize) -> (f64, Vec<f64>) {
    match prior {
        CoefficientPrior::Flat => (0.0, vec![0.0; dim]),
        CoefficientPrior::Normal { mean, variance } => {
            let precision = 1.0 / variance;
            let shift = match basis {
                None => vec![mean * precision; dim],
                Some(basis) => basis
                    .iter()
                    .map(|vector| vector.iter().sum::<f64>() * mean * precision)
                    .collect(),
            };
            (precision, shift)
        }
    }
}

/// Precision `B/σ² + λ₀ I` and right-hand side `B b̂/σ² + shift`, factored.
fn conditional_gaussian(
    b: &[f64],
    estimate: &[f64],
    precision: f64,
    shift: &[f64],
    variance: f64,
    dim: usize,
) -> Option<(Vec<f64>, Vec<f64>)> {
    let mut matrix = vec![0.0; dim * dim];
    let mut rhs = shift.to_vec();
    for row in 0..dim {
        for col in 0..dim {
            matrix[row * dim + col] = b[row * dim + col] / variance;
        }
        matrix[row * dim + row] += precision;
        rhs[row] += (0..dim).map(|col| b[row * dim + col] * estimate[col]).sum::<f64>() / variance;
    }
    let mut factor = vec![0.0; dim * dim];
    cholesky_spd_into(&matrix, dim, &mut factor).ok()?;
    let mut scratch = vec![0.0; dim];
    let mut mean = vec![0.0; dim];
    chol_solve_into(&factor, dim, &rhs, &mut scratch, &mut mean);
    Some((factor, mean))
}

fn gaussian_draw(factor: &[f64], mean: &[f64], dim: usize, rng: &mut CausalRng) -> Vec<f64> {
    let z: Vec<f64> = (0..dim).map(|_| standard_normal(rng)).collect();
    let mut shift = vec![0.0; dim];
    back_solve_into(factor, dim, &z, &mut shift);
    mean.iter().zip(&shift).map(|(m, s)| m + s).collect()
}

fn posterior_row(
    fit: &ExperimentalRecoveryFit,
    system: &BlockSystem,
    solution: &BlockSolution,
    spec: &PosteriorSpec,
    chain: usize,
) -> Result<Vec<RowDraw>, ExperimentalFitError> {
    let target = fit.nodes[system.target];
    let refused = ExperimentalFitError::PosteriorPrecision { target };
    let moments =
        fit.residuals[system.target].as_ref().ok_or(ExperimentalFitError::BlockSolve { target })?;
    let k = solution.dim;
    let gram = gram_of_upper(&solution.r, k);
    let (precision, shift) = prior_terms(spec.prior.coefficients, None, k);
    let count = moments.rows as f64;
    let row_of = |values: &[f64]| {
        let mut row = vec![0.0; fit.nodes.len()];
        for (&source, &value) in system.sources.iter().zip(values) {
            row[source] = value;
        }
        residual_vector(system.target, &row)
    };
    let draw_intercept = |values: &[f64], variance: f64, rng: &mut CausalRng| {
        moments.mean_of(&row_of(values)) + (variance / count).sqrt() * standard_normal(rng)
    };
    let mut draws = Vec::with_capacity(spec.draws);
    match &spec.prior.disturbance {
        DisturbancePrior::Known { variances } => {
            let variance = variances[system.target];
            let (factor, mean) =
                conditional_gaussian(&gram, &solution.estimate, precision, &shift, variance, k)
                    .ok_or(refused)?;
            for draw in 0..spec.draws {
                let mut rng = keyed_stream(POSTERIOR_SWEEP_OPERATION, chain, draw, spec.seed);
                let coefficients = gaussian_draw(&factor, &mean, k, &mut rng);
                let intercept = draw_intercept(&coefficients, variance, &mut rng);
                draws.push(RowDraw { coefficients, intercept, variance });
            }
        }
        DisturbancePrior::InverseWishart { degrees_of_freedom, scale } => {
            let scatter = count * moments.variance_of(&row_of(&solution.estimate));
            let prior_scale = scale[system.target];
            let shape = 0.5 * (degrees_of_freedom + (count - 1.0) + k as f64);
            let mut variance = (prior_scale + scatter) / (degrees_of_freedom + count - 1.0);
            for sweep in 0..spec.burn_in + spec.draws {
                let mut rng = keyed_stream(POSTERIOR_SWEEP_OPERATION, chain, sweep, spec.seed);
                let (factor, mean) =
                    conditional_gaussian(&gram, &solution.estimate, precision, &shift, variance, k)
                        .ok_or_else(|| refused.clone())?;
                let coefficients = gaussian_draw(&factor, &mean, k, &mut rng);
                let offset: Vec<f64> =
                    coefficients.iter().zip(&solution.estimate).map(|(b, e)| b - e).collect();
                let distance = quadratic_n(&offset, &gram, &offset, k);
                let rate = 0.5 * (prior_scale + scatter + distance.max(0.0));
                variance = sample_inv_gamma(shape, rate, &mut rng);
                let intercept = draw_intercept(&coefficients, variance, &mut rng);
                if sweep >= spec.burn_in {
                    draws.push(RowDraw { coefficients, intercept, variance });
                }
            }
        }
    }
    Ok(draws)
}

fn identified_posterior(
    fit: &ExperimentalRecoveryFit,
    system: &BlockSystem,
    solution: &BlockSolution,
    prior: &RecoveryPrior,
) -> Result<Option<IdentifiedPosteriorBlock>, ExperimentalFitError> {
    let DisturbancePrior::Known { variances } = &prior.disturbance else {
        return Ok(None);
    };
    let target = fit.nodes[system.target];
    let sources: Vec<VariableId> = system.sources.iter().map(|&s| fit.nodes[s]).collect();
    let basis = solution.basis.clone().unwrap_or_default();
    let d = solution.dim;
    if d == 0 {
        return Ok(Some(IdentifiedPosteriorBlock {
            target,
            sources,
            basis,
            mean: Vec::new(),
            covariance: Vec::new(),
        }));
    }
    let gram = gram_of_upper(&solution.r, d);
    let (precision, shift) = prior_terms(prior.coefficients, Some(basis.as_slice()), d);
    let (factor, mean) = conditional_gaussian(
        &gram,
        &solution.estimate,
        precision,
        &shift,
        variances[system.target],
        d,
    )
    .ok_or(ExperimentalFitError::PosteriorPrecision { target })?;
    Ok(Some(IdentifiedPosteriorBlock {
        target,
        sources,
        basis,
        mean,
        covariance: chol_inverse(&factor, d),
    }))
}

fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter().zip(right).map(|(a, b)| a * b).sum()
}

/// `left' M right` for a square `M` whose side is `left.len()`.
fn quadratic(left: &[f64], matrix: &[f64], right: &[f64]) -> f64 {
    quadratic_n(left, matrix, right, left.len())
}

fn quadratic_n(left: &[f64], matrix: &[f64], right: &[f64], n: usize) -> f64 {
    let mut total = 0.0;
    for row in 0..n {
        let mut inner = 0.0;
        for col in 0..n {
            inner += matrix[row * n + col] * right[col];
        }
        total += left[row] * inner;
    }
    total
}

fn infinity_norm(matrix: &[f64], n: usize) -> f64 {
    (0..n)
        .map(|row| matrix[row * n..(row + 1) * n].iter().map(|v| v.abs()).sum::<f64>())
        .fold(0.0, f64::max)
}

/// `R'R` of an upper-triangular row-major `R`.
fn gram_of_upper(r: &[f64], k: usize) -> Vec<f64> {
    let mut gram = vec![0.0; k * k];
    for row in 0..k {
        for col in 0..k {
            gram[row * k + col] = (0..k).map(|inner| r[inner * k + row] * r[inner * k + col]).sum();
        }
    }
    gram
}

/// `R⁻¹ R⁻ᵀ = (R'R)⁻¹` from `R⁻¹`.
fn gram_of_upper_inverse(r_inverse: &[f64], k: usize) -> Vec<f64> {
    let mut out = vec![0.0; k * k];
    for row in 0..k {
        for col in 0..k {
            out[row * k + col] =
                (0..k).map(|inner| r_inverse[row * k + inner] * r_inverse[col * k + inner]).sum();
        }
    }
    out
}

/// Householder least squares of row-major `m × k` `a` (`m ≥ k`) against `y`.
///
/// Returns the solution and the upper factor `R`, or `None` when a column
/// vanishes or a value is not finite.
fn qr_least_squares(a: &[f64], y: &[f64], m: usize, k: usize) -> Option<(Vec<f64>, Vec<f64>)> {
    if k == 0 || m < k || a.len() != m * k || y.len() != m {
        return None;
    }
    let mut work = a.to_vec();
    let mut rhs = y.to_vec();
    let mut v = vec![0.0; m];
    for j in 0..k {
        let norm = (j..m).map(|i| work[i * k + j] * work[i * k + j]).sum::<f64>().sqrt();
        if !(norm.is_finite() && norm > 0.0) {
            return None;
        }
        let alpha = if work[j * k + j] > 0.0 { -norm } else { norm };
        for i in j..m {
            v[i] = work[i * k + j];
        }
        v[j] -= alpha;
        let vv: f64 = (j..m).map(|i| v[i] * v[i]).sum();
        if !(vv.is_finite() && vv > 0.0) {
            return None;
        }
        for col in j..k {
            let s: f64 = (j..m).map(|i| v[i] * work[i * k + col]).sum();
            let f = 2.0 * s / vv;
            for i in j..m {
                work[i * k + col] -= f * v[i];
            }
        }
        let s: f64 = (j..m).map(|i| v[i] * rhs[i]).sum();
        let f = 2.0 * s / vv;
        for i in j..m {
            rhs[i] -= f * v[i];
        }
    }
    let mut r = vec![0.0; k * k];
    for row in 0..k {
        for col in row..k {
            r[row * k + col] = work[row * k + col];
        }
    }
    let mut x = vec![0.0; k];
    for row in (0..k).rev() {
        let mut acc = rhs[row];
        for col in (row + 1)..k {
            acc -= r[row * k + col] * x[col];
        }
        let pivot = r[row * k + row];
        if !(pivot.is_finite() && pivot.abs() > 0.0) {
            return None;
        }
        x[row] = acc / pivot;
    }
    x.iter().all(|value| value.is_finite()).then_some((x, r))
}

fn upper_inverse(r: &[f64], k: usize) -> Option<Vec<f64>> {
    let mut inverse = vec![0.0; k * k];
    for col in 0..k {
        for row in (0..k).rev() {
            let mut acc = if row == col { 1.0 } else { 0.0 };
            for inner in (row + 1)..k {
                acc -= r[row * k + inner] * inverse[inner * k + col];
            }
            inverse[row * k + col] = acc / r[row * k + row];
        }
    }
    inverse.iter().all(|value| value.is_finite()).then_some(inverse)
}

/// Orthonormal basis of the orthogonal complement of `null` in `R^k`.
fn row_space_basis(k: usize, null: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let mut nullspace: Vec<Vec<f64>> = Vec::with_capacity(null.len());
    for direction in null {
        if direction.len() != k {
            continue;
        }
        let mut candidate = direction.clone();
        if !normalize(&mut candidate) {
            continue;
        }
        orthogonalize(&mut candidate, &nullspace);
        orthogonalize(&mut candidate, &nullspace);
        if normalize(&mut candidate) {
            nullspace.push(candidate);
        }
    }
    let wanted = k.saturating_sub(nullspace.len());
    let mut basis: Vec<Vec<f64>> = Vec::with_capacity(wanted);
    for axis in 0..k {
        if basis.len() == wanted {
            break;
        }
        let mut candidate = vec![0.0; k];
        candidate[axis] = 1.0;
        for _ in 0..2 {
            orthogonalize(&mut candidate, &nullspace);
            orthogonalize(&mut candidate, &basis);
        }
        if normalize(&mut candidate) {
            basis.push(candidate);
        }
    }
    basis
}

fn orthogonalize(vector: &mut [f64], against: &[Vec<f64>]) {
    for other in against {
        let projection = dot(vector, other);
        for (slot, value) in vector.iter_mut().zip(other) {
            *slot -= projection * value;
        }
    }
}

fn normalize(vector: &mut [f64]) -> bool {
    let norm = dot(vector, vector).sqrt();
    if !(norm.is_finite() && norm > BASIS_NORM_FLOOR) {
        return false;
    }
    for value in vector.iter_mut() {
        *value /= norm;
    }
    true
}

#[cfg(test)]
#[path = "experimental_fit_tests.rs"]
mod tests;
