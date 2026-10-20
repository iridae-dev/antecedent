//! Minimum experimental recovery recipe for fully observed linear Gaussian systems.
//!
//! This is the Hyttinen–Eberhardt–Hoyer multi-experiment specification for a
//! declared set of randomized clamp experiments. The design judge accepts or
//! refuses the design. It does not read a sample or assemble a mechanism matrix.
//! Numeric direct effects come only from an assembled constraint block, and only
//! for directions that block's column rank fixes.
//!
//! The system is `x = Kx + ε` on a fully observed node set, with `K[into, from]`
//! the direct mechanism of `from` into `into` and a zero diagonal (canonical
//! model). A clamp experiment partitions the nodes into intervened `J` and
//! passive `U` and replaces each clamped equation with an exogenous assignment.
//!
//! The recipe, in order:
//!
//! - **Exogeneity.** The clamp is a randomized surgical intervention, uncorrelated
//!   with the disturbances. Conditioning on the factual disturbance is not a clamp.
//! - **Disturbances and regimes.** Regimes are independent, and the noise law is
//!   the fully observed independent Gaussian law (diagonal disturbance covariance).
//!   Correlated disturbances are a latent-variable expansion.
//! - **Weak stability.** `I − U K` is invertible on every experiment in the set.
//!   That invertibility is not physical dynamical stability (eigenvalues of `U K`
//!   inside the unit disk, or a contraction bound such as `ρ(|K|) < 1`).
//! - **Coverage.** The pair condition: for ordered pair `(from, into)`, some
//!   experiment clamps `from` while `into` stays passive. If any source into a
//!   target fails it, the whole incoming row is one unresolved direction.
//! - **Rank.** When every pair into a target holds, generic experimental effects
//!   give that row full column rank. A measure-zero cancellation of those effects
//!   can still drop the rank, so a closed report is generic identification, not a
//!   verified rank. A declared column rank below the row width names the unresolved
//!   directions and leaves the recovery goal open. The Moore-Penrose completion of
//!   that block is not applied. That declaration is not the column rank of an
//!   assembled constraint matrix. The matrix rank, its null directions, and any
//!   direct effects are read from the matrix. [`DeclaredEquationRank`] and
//!   [`WeakStability::InvertibilityDeclared`] do not by themselves return
//!   coefficients, and a caller-asserted full rank does not fill a free direction
//!   of a rank-deficient matrix. A rank declared on a target the pair condition
//!   does not cover is a warning on a partial report. It does not refuse the
//!   design and it does not solve that row.
//! - **Finite-sample conditioning.** A declared sample rank or finite condition
//!   number of the constraint matrix is not the population rank. Sampling error
//!   can make an underdetermined system look full rank (Hyttinen, Eberhardt, and
//!   Hoyer, JMLR 2012, Section 4.2). That declaration does not recover a
//!   coordinate, and it does not open a row the population rank already closes.
//! - **Uniqueness restrictions.** A minimum-norm completion, a background zero,
//!   or faithfulness selects one matrix in an open solution set (Section 4.1,
//!   step 4, and Section 4.2). The selection is not applied. It is not physical
//!   dynamical stability, and physical dynamical stability does not select the
//!   unresolved directions. Weak stability remains invertibility of `I − U K`
//!   (Definition 6), which is not a stability certificate either.
//!
//! Observational equilibrium regression and prior regularization are not this
//! recipe. A latent-variable expansion is an optional later attempt, recorded as
//! [`NotThisRecipe`], not as an identification verdict.
//!
//! [`RecoveredCoefficientUncertainty`] is the sampling law of a recovery, not a
//! point. Each draw keeps its coefficients and its disturbance together.
//! A failed or inadmissible draw stays in the law at its own weight.
//! Rank-aware blocks contribute [`UnresolvedDirection`]s. They do not complete `K`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use antecedent_core::VariableId;

use crate::experiment_moments::{
    ConstraintColumnReport, TargetConstraintBlock, constraint_column_report,
};

/// Name of this clamp-experiment recipe.
///
/// A refused observational-equilibrium ordinary-least-squares attempt reports
/// this string as its next attempt.
pub const CLAMP_EXPERIMENT_RECIPE: &str =
    "hyttinen_eberhardt_hoyer.fully_observed_clamp_experiment";

/// Name of the optional later latent-variable expansion.
pub const LATENT_VARIABLE_EXPANSION_ATTEMPT: &str =
    "hyttinen_eberhardt_hoyer.latent_variable_expansion";

/// Name of prior or penalty regularization, which is not this recipe.
pub const PRIOR_REGULARIZATION_ATTEMPT: &str = "hyttinen_eberhardt_hoyer.prior_regularization";

/// Direct mechanism coordinate `K[into, from]`.
///
/// The coordinate names an entry. It does not carry a numeric value.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct DirectCoefficient {
    /// Equation the mechanism writes.
    pub into: VariableId,
    /// Source of the direct mechanism.
    pub from: VariableId,
}

/// One declared clamp experiment: a partition of the observed nodes.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeclaredClampExperiment {
    /// Intervened nodes `J`.
    pub clamped: Vec<VariableId>,
    /// Passively observed nodes `U`.
    pub passive: Vec<VariableId>,
    /// How the clamp is claimed to be assigned.
    pub exogeneity: ClampExogeneity,
}

/// Whether a clamp is the randomized intervention this recipe requires.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum ClampExogeneity {
    /// Surgical randomization, independent of every disturbance.
    ///
    /// The clamp replaces the structural equation. It is not conditioning on the
    /// factual disturbance.
    RandomizedIntervention,
    /// The assigned values depend on the factual disturbance.
    ConditionedOnFactualDisturbance,
}

/// Disturbance law the design declares.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum DeclaredNoiseLaw {
    /// Fully observed independent Gaussian disturbances: diagonal `Σε`.
    ///
    /// Second moments exhaust a Gaussian law. Off-diagonal disturbance
    /// covariances are fixed at zero by this declaration, not estimated.
    IndependentGaussian,
    /// No noise law was declared.
    Undeclared,
    /// Correlated disturbances (latent confounding). Not this recipe.
    LatentConfounding,
}

/// Whether each experiment is its own disturbance regime.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum RegimeAssumption {
    /// Independent regimes: a fresh disturbance draw in every experiment.
    Independent,
    /// One disturbance shared across experiments, or a regime defined by
    /// conditioning a single draw.
    Dependent,
}

/// Invertibility of `I − U K` on every experiment in the design.
///
/// This is weak stability. It is not [`PhysicalDynamicalStability`].
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum WeakStability {
    /// The design declares invertibility of `I − U K` for every experiment.
    InvertibilityDeclared,
    /// Invertibility was not declared.
    NotDeclared,
}

/// Physical dynamical stability of the equilibration map.
///
/// Asymptotic stability requires every eigenvalue of every `U K` to have modulus
/// strictly less than one. A contraction bound such as `ρ(|K|) < 1` is the same
/// kind of certificate. Declaring it does not declare [`WeakStability`], and it
/// does not select unresolved coefficient directions.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum PhysicalDynamicalStability {
    /// A physical dynamical-stability certificate was declared.
    Declared,
    /// No physical dynamical-stability certificate was declared.
    NotDeclared,
}

/// Declared numerical condition of the estimated constraint matrix.
///
/// This is a property of a finite sample. It is not the population column rank
/// of a [`DeclaredEquationRank`], not a [`UniquenessRestriction`], and not
/// [`PhysicalDynamicalStability`].
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum FiniteSampleConditioning {
    /// No sample rank or condition number was declared.
    NotDeclared,
    /// The sample constraint matrix was declared full column rank.
    ///
    /// Sampling error can give that rank when the population system is
    /// underdetermined. The declaration does not recover a coefficient.
    SampleFullColumnRank,
    /// A finite condition number of the sample constraints was declared.
    ///
    /// A finite number is not a rank deficiency, and it is not a uniqueness
    /// restriction. No numeric condition number is stored: `f64` is not `Eq`.
    FiniteConditionNumber,
}

/// An extra assumption offered to select one mechanism matrix.
///
/// The assumption names a way to pick a point in an open solution set. This
/// recipe records it and does not apply it. Applying it would fill unresolved
/// directions. It does not certify [`WeakStability`] or
/// [`PhysicalDynamicalStability`].
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum UniquenessRestriction {
    /// No extra restriction was offered.
    NotOffered,
    /// Minimum-norm completion of the constraint system.
    ///
    /// This is the Moore-Penrose solution of an underdetermined block. It
    /// returns a value on every coordinate, including the nullspace.
    MinimumNormCompletion,
    /// A known zero on a coordinate the experiments leave unresolved.
    BackgroundZero,
    /// Faithfulness, offered as a further constraint on the open directions.
    Faithfulness,
}

/// Declared column rank of the constraint block for one equation.
///
/// The block has one column per other node. This recipe does not estimate the
/// block. When the declaration is absent and the pair condition holds for every
/// source into `target`, the row is treated as full generic rank.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeclaredEquationRank {
    /// Equation the block constrains.
    pub target: VariableId,
    /// Column rank of the block.
    pub column_rank: usize,
    /// Declared nullspace basis. Each inner vector is one unresolved direction,
    /// listed by the coordinates it moves.
    pub nullspace: Vec<Vec<DirectCoefficient>>,
}

/// Assumptions and experiments of one clamp-experiment design.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ClampExperimentSet {
    /// Declared experiments. A null experiment (nothing clamped) is allowed and
    /// does not by itself satisfy the pair condition.
    pub experiments: Vec<DeclaredClampExperiment>,
    /// Declared disturbance law.
    pub noise: DeclaredNoiseLaw,
    /// Declared regime dependence.
    pub regimes: RegimeAssumption,
    /// Weak stability (invertibility) declaration.
    pub weak_stability: WeakStability,
    /// Physical dynamical stability declaration, recorded separately.
    pub physical_dynamical_stability: PhysicalDynamicalStability,
    /// Finite-sample condition of the constraint matrix, recorded separately.
    pub finite_sample_conditioning: FiniteSampleConditioning,
    /// Uniqueness restriction offered for the open directions, not applied.
    pub uniqueness_restriction: UniquenessRestriction,
    /// Non-generic rank declarations. Empty means generic rank.
    pub equation_ranks: Vec<DeclaredEquationRank>,
}

/// What the caller is asking this recipe to judge.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RecoveryAttempt {
    /// Randomized clamp experiments under the declarations in the set.
    ClampExperiments(ClampExperimentSet),
    /// Per-node ordinary least squares of the equilibrium equations, including a
    /// study that only offers the observational regression.
    ObservationalEquilibriumOls,
    /// A prior or penalty used to pick one mechanism matrix.
    PriorRegularization,
    /// Expand the fully observed model with latent confounding and treat that
    /// expansion as identification.
    LatentVariableExpansion,
}

/// Declared design: the observed nodes and the attempt offered for them.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExperimentDesign {
    /// Fully observed nodes. At least two, without duplicates, for a clamp attempt.
    pub nodes: Vec<VariableId>,
    /// Attempt offered for `nodes`.
    pub attempt: RecoveryAttempt,
}

/// Why a coordinate or a whole incoming row stays unresolved.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum UnresolvedReason {
    /// At least one ordered pair into `target` fails the pair condition.
    ///
    /// The support is the whole incoming row: a missing pair may move every
    /// coefficient into that target, so this recipe does not separate the row.
    IncompletePairCoverage {
        /// Equation that coverage does not pin down.
        target: VariableId,
        /// Sources that are never clamped while `target` is passive.
        missing_clamps: Vec<VariableId>,
    },
    /// The pair condition holds, and the declared constraint rank does not.
    DeficientConstraintRank {
        /// Equation whose constraint block is rank deficient.
        target: VariableId,
        /// Declared column rank.
        column_rank: usize,
        /// Number of incoming coefficients (`nodes − 1`).
        width: usize,
    },
}

/// One basis direction of the unresolved coefficient subspace.
///
/// `support` lists coordinates the direction moves. It is not a filled-in row of `K`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct UnresolvedDirection {
    /// Coordinates with a non-zero weight in this direction.
    pub support: Vec<DirectCoefficient>,
    /// Why the direction is unresolved.
    pub reason: UnresolvedReason,
}

/// Coefficient coordinates this recipe can and cannot recover.
///
/// `recoverable` and the supports of `unresolved` name coordinates. Neither
/// stores a mechanism value. [`Self::finite_sample_conditioning`] and
/// [`Self::uniqueness_restriction`] are the declarations that were offered.
/// They do not move a coordinate between the two lists.
///
/// [`Self::generic`] is set when the closed report rests on generic experimental
/// effects. [`WeakStability::InvertibilityDeclared`] and a full
/// [`DeclaredEquationRank`] are not a verified column rank. Measure-zero
/// cancellations can break that rank.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CoefficientRecovery {
    /// Recipe that produced the report.
    pub recipe: &'static str,
    /// Coordinates pinned down by coverage and generic rank.
    pub recoverable: Vec<DirectCoefficient>,
    /// Unresolved directions. Empty only when the coefficient goal is closed.
    pub unresolved: Vec<UnresolvedDirection>,
    /// Finite-sample condition offered with the design. Not applied.
    pub finite_sample_conditioning: FiniteSampleConditioning,
    /// Uniqueness restriction offered with the design. Not applied.
    pub uniqueness_restriction: UniquenessRestriction,
    /// The closed report is generic identification, not a verified rank.
    pub generic: bool,
    /// Declarations that were recorded and not applied.
    pub warnings: Vec<RecoveryWarning>,
}

/// A declaration recorded on a coefficient report. It does not close the goal.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum RecoveryWarning {
    /// A rank was declared for a target the pair condition does not cover.
    ///
    /// The incoming row stays unresolved. The declaration is not a refusal of
    /// the rest of the design.
    RankDeclarationOnUncoveredTarget {
        /// Equation the declaration named.
        target: VariableId,
    },
}

impl RecoveryWarning {
    /// Design fault this warning records.
    #[must_use]
    pub const fn design_fault(self) -> DesignFault {
        match self {
            Self::RankDeclarationOnUncoveredTarget { .. } => {
                DesignFault::RankDeclarationOnUncoveredTarget
            }
        }
    }
}

impl CoefficientRecovery {
    /// Whether every direct coefficient is recoverable.
    ///
    /// Finite-sample conditioning and a uniqueness restriction do not close the
    /// goal. A closed goal is coverage plus generic rank. Measure-zero
    /// cancellations can still break that rank; [`Self::generic`] records it.
    #[must_use]
    pub fn goal_closed(&self) -> bool {
        self.unresolved.is_empty()
    }
}

/// Why a clamp attempt was refused before any coefficient report.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum RefusalReason {
    /// The design only offers observational equilibrium regression.
    ObservationalEquilibriumOls,
    /// A clamp is conditioned on the factual disturbance.
    ClampNotExogenous {
        /// Index of the first experiment that fails exogeneity.
        experiment: usize,
    },
    /// No noise law was declared.
    NoiseLawUndeclared,
    /// Regimes are not independent.
    DependentRegimes,
    /// Weak stability (invertibility) was not declared.
    WeakStabilityNotDeclared,
    /// Physical dynamical stability was offered in place of weak stability.
    PhysicalDynamicalStabilityIsNotWeakStability,
    /// A uniqueness restriction was offered in place of weak stability.
    ///
    /// The restriction selects a mechanism. It does not certify invertibility
    /// of `I − U K`, and it does not certify physical dynamical stability.
    UniquenessRestrictionIsNotAStabilityCertificate,
    /// The clamp declaration is not a well-formed experiment set.
    DesignFault(DesignFault),
}

/// Structural faults in a clamp-experiment declaration.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum DesignFault {
    /// No observed nodes.
    EmptySystem,
    /// A node label was repeated.
    DuplicateNode,
    /// Fewer than two nodes, so there is no direct coefficient.
    NoDirectCoefficient,
    /// A clamp attempt listed no experiment.
    NoExperiment,
    /// An experiment does not partition the declared nodes.
    ExperimentDoesNotPartition {
        /// Index of the experiment.
        experiment: usize,
    },
    /// A rank declaration names a node outside the design.
    RankTargetUnknown,
    /// Two rank declarations name the same equation.
    DuplicateRankDeclaration,
    /// Declared column rank exceeds the incoming-row width.
    RankAboveWidth,
    /// The nullspace basis does not have one vector per missing rank.
    NullspaceWidthMismatch,
    /// A nullspace coordinate is not an incoming coefficient of the target.
    NullspaceCoefficientOffTarget,
    /// A nullspace coordinate lies on the diagonal.
    DiagonalCoefficient,
    /// The same coordinate appears twice in one direction.
    DuplicateCoefficientInDirection,
    /// Two nullspace vectors have the same support.
    DuplicateDirection,
    /// A nullspace vector names no coordinate.
    EmptyDirection,
    /// A rank was declared for an equation the pair condition does not cover.
    ///
    /// The decision is [`ExperimentalRecoveryDecision::Partial`] with
    /// [`RecoveryWarning::RankDeclarationOnUncoveredTarget`]. The row stays
    /// unresolved. This fault is not a refusal of the rest of the design.
    RankDeclarationOnUncoveredTarget,
}

/// The attempt a refusal points to.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum NextAttempt {
    /// Satisfy this module's clamp-experiment recipe.
    ClampExperimentRecipe,
}

impl NextAttempt {
    /// Stable name of the next attempt.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::ClampExperimentRecipe => CLAMP_EXPERIMENT_RECIPE,
        }
    }
}

/// A refusal of an attempt that is not yet this recipe, or that breaks a
/// precondition of it. The next attempt is the clamp-experiment recipe.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct RecipeRefusal {
    /// Why the attempt was refused.
    pub reason: RefusalReason,
    /// What to attempt next.
    pub next_attempt: NextAttempt,
}

/// An attempt this module records without judging identification.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum NotThisRecipe {
    /// Optional later expansion to correlated disturbances.
    LatentVariableExpansion,
    /// Prior or penalty selection of a mechanism matrix.
    PriorRegularization,
}

impl NotThisRecipe {
    /// Stable name of the attempt that was not judged.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::LatentVariableExpansion => LATENT_VARIABLE_EXPANSION_ATTEMPT,
            Self::PriorRegularization => PRIOR_REGULARIZATION_ATTEMPT,
        }
    }
}

/// Outcome of judging a declared design against the minimum recipe.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ExperimentalRecoveryDecision {
    /// Generic identification of every off-diagonal coordinate.
    ///
    /// Pair coverage holds and no declared deficiency remains. No mechanism
    /// matrix is attached. Measure-zero cancellations of the experimental
    /// effects can break the column rank, so this is not a verified rank.
    /// [`WeakStability::InvertibilityDeclared`] and [`DeclaredEquationRank`]
    /// do not supply that rank.
    Recovered(CoefficientRecovery),
    /// Coverage or rank leaves unresolved directions. The recovery goal stays open.
    Partial(CoefficientRecovery),
    /// The attempt fails a precondition of this recipe.
    Refused(RecipeRefusal),
    /// The attempt is not this recipe and is not an identification verdict.
    NotThisRecipe(NotThisRecipe),
}

impl ExperimentalRecoveryDecision {
    /// Whether the coefficient recovery goal is closed.
    ///
    /// Only a [`Self::Recovered`] report closes it. A rank-deficient or
    /// under-covered design, a refusal, and [`NotThisRecipe`] all leave it open.
    #[must_use]
    pub fn goal_closed(&self) -> bool {
        match self {
            Self::Recovered(recovery) => recovery.goal_closed(),
            Self::Partial(_) | Self::Refused(_) | Self::NotThisRecipe(_) => false,
        }
    }

    /// Coefficient report, when the decision is [`Self::Recovered`] or [`Self::Partial`].
    #[must_use]
    pub const fn coefficient_recovery(&self) -> Option<&CoefficientRecovery> {
        match self {
            Self::Recovered(recovery) | Self::Partial(recovery) => Some(recovery),
            Self::Refused(_) | Self::NotThisRecipe(_) => None,
        }
    }
}

/// Judge `design` against the minimum clamp-experiment recipe.
///
/// The function inspects declarations only. It does not estimate experimental
/// effects, invert a constraint matrix, or complete `K`. A finite-sample rank
/// or condition number is not read as the population rank. A uniqueness
/// restriction is not applied, and it is not a physical-stability certificate.
#[must_use]
pub fn decide_experimental_recovery(design: &ExperimentDesign) -> ExperimentalRecoveryDecision {
    match &design.attempt {
        RecoveryAttempt::ObservationalEquilibriumOls => {
            refuse(RefusalReason::ObservationalEquilibriumOls)
        }
        RecoveryAttempt::PriorRegularization => {
            ExperimentalRecoveryDecision::NotThisRecipe(NotThisRecipe::PriorRegularization)
        }
        RecoveryAttempt::LatentVariableExpansion => {
            ExperimentalRecoveryDecision::NotThisRecipe(NotThisRecipe::LatentVariableExpansion)
        }
        RecoveryAttempt::ClampExperiments(set) => decide_clamp_set(&design.nodes, set),
    }
}

/// What a rank declaration and an invertibility declaration return on their own.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum DeclarationRecovery {
    /// No direct-effect vector was produced.
    NoCoefficients,
}

/// [`DeclaredEquationRank`] and [`WeakStability::InvertibilityDeclared`] are not
/// an assembled constraint matrix.
///
/// Both may be present, including a declaration of full column rank together
/// with invertibility of `I − U K`. The result is still no coefficients.
/// Physical dynamical stability is not an argument of this function and is not
/// a substitute for either declaration.
#[must_use]
pub const fn recover_declared_coefficients(
    equation_rank: Option<&DeclaredEquationRank>,
    weak_stability: WeakStability,
) -> DeclarationRecovery {
    match (equation_rank, weak_stability) {
        (Some(_) | None, WeakStability::InvertibilityDeclared | WeakStability::NotDeclared) => {
            DeclarationRecovery::NoCoefficients
        }
    }
}

/// Direct effects read from one assembled block, or the directions the block leaves free.
#[derive(Clone, Debug, PartialEq)]
pub enum AssembledCoefficientRecovery {
    /// Unique solution of a full-column-rank block whose rows agree.
    Coefficients {
        /// Passive equation the block constrains.
        target: VariableId,
        /// Sources, in the same order as `values`.
        sources: Vec<VariableId>,
        /// Direct effects of `sources` into `target`.
        values: Vec<f64>,
    },
    /// Null directions of a rank-deficient coefficient matrix. No value was filled in.
    NullDirections {
        /// Passive equation the block constrains.
        target: VariableId,
        /// Sources, in the same order as each direction.
        sources: Vec<VariableId>,
        /// Rank of the coefficient matrix.
        column_rank: usize,
        /// Basis of the matrix nullspace, one vector per free source.
        directions: Vec<Vec<f64>>,
        /// The caller asserted full column rank for this target. The matrix did not have it.
        full_rank_was_asserted: bool,
    },
    /// The columns span the sources, but the rows disagree.
    Inconsistent {
        /// Rank of the coefficient matrix.
        column_rank: usize,
    },
    /// `block` is not a finite coefficient matrix of its stated shape.
    NotAConstraint,
}

impl AssembledCoefficientRecovery {
    /// Direct effects, when the matrix determined them.
    #[must_use]
    pub fn coefficients(&self) -> Option<&[f64]> {
        match self {
            Self::Coefficients { values, .. } => Some(values),
            Self::NullDirections { .. } | Self::Inconsistent { .. } | Self::NotAConstraint => None,
        }
    }
}

/// Read direct effects from `block`, ignoring claims that are not the matrix.
///
/// `asserted_rank` and `weak_stability` do not supply coefficients.
/// [`WeakStability::InvertibilityDeclared`] stays invertibility of `I − U K`;
/// it is not physical dynamical stability and it does not select a null
/// direction. When `asserted_rank` says this target has full column rank and
/// the matrix does not, the result is the matrix nullspace.
#[must_use]
pub fn recover_assembled_coefficients(
    block: &TargetConstraintBlock,
    asserted_rank: Option<&DeclaredEquationRank>,
    weak_stability: WeakStability,
) -> AssembledCoefficientRecovery {
    let report = constraint_column_report(block);
    let asserted_full = asserted_rank
        .is_some_and(|rank| rank.target == block.target && rank.column_rank >= block.sources.len());
    match weak_stability {
        WeakStability::InvertibilityDeclared | WeakStability::NotDeclared => {
            recovery_from_matrix(block, report, asserted_full)
        }
    }
}

fn recovery_from_matrix(
    block: &TargetConstraintBlock,
    report: ConstraintColumnReport,
    full_rank_was_asserted: bool,
) -> AssembledCoefficientRecovery {
    match report {
        ConstraintColumnReport::Unique { coefficients } => {
            AssembledCoefficientRecovery::Coefficients {
                target: block.target,
                sources: block.sources.clone(),
                values: coefficients,
            }
        }
        ConstraintColumnReport::Deficient { column_rank, directions } => {
            AssembledCoefficientRecovery::NullDirections {
                target: block.target,
                sources: block.sources.clone(),
                column_rank,
                directions,
                full_rank_was_asserted,
            }
        }
        ConstraintColumnReport::Inconsistent { column_rank } => {
            AssembledCoefficientRecovery::Inconsistent { column_rank }
        }
        ConstraintColumnReport::NotAConstraint => AssembledCoefficientRecovery::NotAConstraint,
    }
}

fn refuse(reason: RefusalReason) -> ExperimentalRecoveryDecision {
    ExperimentalRecoveryDecision::Refused(RecipeRefusal {
        reason,
        next_attempt: NextAttempt::ClampExperimentRecipe,
    })
}

fn decide_clamp_set(
    nodes: &[VariableId],
    set: &ClampExperimentSet,
) -> ExperimentalRecoveryDecision {
    let node_set = match validated_nodes(nodes) {
        Ok(node_set) => node_set,
        Err(fault) => return refuse(RefusalReason::DesignFault(fault)),
    };
    if set.experiments.is_empty() {
        return refuse(RefusalReason::DesignFault(DesignFault::NoExperiment));
    }
    if let Some(fault) = validate_experiments(&node_set, &set.experiments) {
        return refuse(RefusalReason::DesignFault(fault));
    }
    if let Some(fault) = validate_ranks(&node_set, &set.equation_ranks) {
        return refuse(RefusalReason::DesignFault(fault));
    }
    if let Some(experiment) = first_non_exogenous(&set.experiments) {
        return refuse(RefusalReason::ClampNotExogenous { experiment });
    }
    match set.noise {
        DeclaredNoiseLaw::Undeclared => return refuse(RefusalReason::NoiseLawUndeclared),
        DeclaredNoiseLaw::LatentConfounding => {
            return ExperimentalRecoveryDecision::NotThisRecipe(
                NotThisRecipe::LatentVariableExpansion,
            );
        }
        DeclaredNoiseLaw::IndependentGaussian => {}
    }
    match set.regimes {
        RegimeAssumption::Dependent => return refuse(RefusalReason::DependentRegimes),
        RegimeAssumption::Independent => {}
    }
    if set.weak_stability == WeakStability::NotDeclared {
        return refuse(stability_stand_in(set));
    }
    recover(&node_set, set)
}

fn stability_stand_in(set: &ClampExperimentSet) -> RefusalReason {
    match (set.physical_dynamical_stability, set.uniqueness_restriction) {
        (PhysicalDynamicalStability::Declared, _) => {
            RefusalReason::PhysicalDynamicalStabilityIsNotWeakStability
        }
        (PhysicalDynamicalStability::NotDeclared, UniquenessRestriction::NotOffered) => {
            RefusalReason::WeakStabilityNotDeclared
        }
        (
            PhysicalDynamicalStability::NotDeclared,
            UniquenessRestriction::MinimumNormCompletion
            | UniquenessRestriction::BackgroundZero
            | UniquenessRestriction::Faithfulness,
        ) => RefusalReason::UniquenessRestrictionIsNotAStabilityCertificate,
    }
}

fn validated_nodes(nodes: &[VariableId]) -> Result<BTreeSet<VariableId>, DesignFault> {
    if nodes.is_empty() {
        return Err(DesignFault::EmptySystem);
    }
    let node_set = nodes.iter().copied().collect::<BTreeSet<_>>();
    if node_set.len() != nodes.len() {
        return Err(DesignFault::DuplicateNode);
    }
    if node_set.len() < 2 {
        return Err(DesignFault::NoDirectCoefficient);
    }
    Ok(node_set)
}

fn validate_experiments(
    nodes: &BTreeSet<VariableId>,
    experiments: &[DeclaredClampExperiment],
) -> Option<DesignFault> {
    for (index, experiment) in experiments.iter().enumerate() {
        if !experiment_partitions(nodes, experiment) {
            return Some(DesignFault::ExperimentDoesNotPartition { experiment: index });
        }
    }
    None
}

fn experiment_partitions(
    nodes: &BTreeSet<VariableId>,
    experiment: &DeclaredClampExperiment,
) -> bool {
    let mut seen = BTreeSet::new();
    for node in experiment.clamped.iter().chain(experiment.passive.iter()) {
        if !nodes.contains(node) || !seen.insert(*node) {
            return false;
        }
    }
    seen.len() == nodes.len()
}

fn validate_ranks(
    nodes: &BTreeSet<VariableId>,
    ranks: &[DeclaredEquationRank],
) -> Option<DesignFault> {
    let width = nodes.len() - 1;
    let mut seen_targets = BTreeSet::new();
    for rank in ranks {
        if !nodes.contains(&rank.target) {
            return Some(DesignFault::RankTargetUnknown);
        }
        if !seen_targets.insert(rank.target) {
            return Some(DesignFault::DuplicateRankDeclaration);
        }
        if rank.column_rank > width {
            return Some(DesignFault::RankAboveWidth);
        }
        let nullity = width - rank.column_rank;
        if rank.nullspace.len() != nullity {
            return Some(DesignFault::NullspaceWidthMismatch);
        }
        if let Some(fault) = validate_nullspace(nodes, rank) {
            return Some(fault);
        }
    }
    None
}

fn validate_nullspace(
    nodes: &BTreeSet<VariableId>,
    rank: &DeclaredEquationRank,
) -> Option<DesignFault> {
    let mut seen_directions = BTreeSet::new();
    for direction in &rank.nullspace {
        if direction.is_empty() {
            return Some(DesignFault::EmptyDirection);
        }
        let mut support = BTreeSet::new();
        for coefficient in direction {
            if coefficient.into != rank.target || !nodes.contains(&coefficient.from) {
                return Some(DesignFault::NullspaceCoefficientOffTarget);
            }
            if coefficient.from == coefficient.into {
                return Some(DesignFault::DiagonalCoefficient);
            }
            if !support.insert(*coefficient) {
                return Some(DesignFault::DuplicateCoefficientInDirection);
            }
        }
        if !seen_directions.insert(support) {
            return Some(DesignFault::DuplicateDirection);
        }
    }
    None
}

fn first_non_exogenous(experiments: &[DeclaredClampExperiment]) -> Option<usize> {
    experiments.iter().position(|experiment| {
        matches!(experiment.exogeneity, ClampExogeneity::ConditionedOnFactualDisturbance)
    })
}

fn recover(nodes: &BTreeSet<VariableId>, set: &ClampExperimentSet) -> ExperimentalRecoveryDecision {
    let width = nodes.len() - 1;
    let mut recoverable = Vec::new();
    let mut unresolved = Vec::new();
    let mut warnings = Vec::new();
    for target in nodes {
        let missing = missing_clamps(nodes, &set.experiments, *target);
        if !missing.is_empty() {
            if set.equation_ranks.iter().any(|rank| rank.target == *target) {
                warnings.push(RecoveryWarning::RankDeclarationOnUncoveredTarget {
                    target: *target,
                });
            }
            unresolved.push(UnresolvedDirection {
                support: incoming(nodes, *target),
                reason: UnresolvedReason::IncompletePairCoverage {
                    target: *target,
                    missing_clamps: missing,
                },
            });
            continue;
        }
        if let Some(rank) = equation_rank(set, *target) {
            if rank.column_rank < width {
                push_rank_deficiency(nodes, rank, width, &mut recoverable, &mut unresolved);
                continue;
            }
        }
        recoverable.extend(incoming(nodes, *target));
    }
    // Coverage decides the coordinates. A full declared rank is the generic rank,
    // not a matrix factorization. The sample condition and the uniqueness
    // restriction are copied and not applied.
    let generic = unresolved.is_empty();
    let recovery = CoefficientRecovery {
        recipe: CLAMP_EXPERIMENT_RECIPE,
        recoverable,
        unresolved,
        finite_sample_conditioning: set.finite_sample_conditioning,
        uniqueness_restriction: set.uniqueness_restriction,
        generic,
        warnings,
    };
    if recovery.goal_closed() {
        return ExperimentalRecoveryDecision::Recovered(recovery);
    }
    ExperimentalRecoveryDecision::Partial(recovery)
}

/// Whether `regime` is an identified clamp in `design`.
///
/// A regime that clamps nothing is not identified. A regime that is not one of
/// `design` was never clamped by this decision, including a joint intervention
/// that the design did not run, and is not identified. Generic identification
/// of other regimes does not transfer to it. A partial or refused decision
/// identifies no regime.
#[must_use]
pub fn regime_is_identified(
    decision: &ExperimentalRecoveryDecision,
    design: &[DeclaredClampExperiment],
    regime: &DeclaredClampExperiment,
) -> bool {
    if regime.clamped.is_empty() {
        return false;
    }
    let Some(recovery) = decision.coefficient_recovery() else {
        return false;
    };
    if !recovery.generic {
        return false;
    }
    let ran = design.iter().any(|experiment| same_clamp(experiment, regime));
    if !ran {
        return false;
    }
    regime.clamped.iter().all(|from| {
        regime.passive.iter().all(|into| {
            recovery.recoverable.iter().any(|coefficient| {
                coefficient.into == *into && coefficient.from == *from
            })
        })
    })
}

fn same_clamp(left: &DeclaredClampExperiment, right: &DeclaredClampExperiment) -> bool {
    if left.exogeneity != right.exogeneity {
        return false;
    }
    let clamped = |experiment: &DeclaredClampExperiment| {
        experiment.clamped.iter().copied().collect::<BTreeSet<_>>()
    };
    let passive = |experiment: &DeclaredClampExperiment| {
        experiment.passive.iter().copied().collect::<BTreeSet<_>>()
    };
    clamped(left) == clamped(right) && passive(left) == passive(right)
}

/// Coefficients of an admissible draw.
///
/// An inadmissible draw, including one that still carries unresolved directions
/// from a rank-deficient block, is not a recovered mechanism.
#[must_use]
pub fn as_recovered_mechanism(draw: &JointRecoveryDraw) -> Option<&[(DirectCoefficient, f64)]> {
    if draw.is_inadmissible() { None } else { Some(draw.coefficients()) }
}

fn push_rank_deficiency(
    nodes: &BTreeSet<VariableId>,
    rank: &DeclaredEquationRank,
    width: usize,
    recoverable: &mut Vec<DirectCoefficient>,
    unresolved: &mut Vec<UnresolvedDirection>,
) {
    let mut blocked = BTreeSet::new();
    for direction in &rank.nullspace {
        let mut support = direction.clone();
        support.sort();
        for coefficient in &support {
            blocked.insert(*coefficient);
        }
        unresolved.push(UnresolvedDirection {
            support,
            reason: UnresolvedReason::DeficientConstraintRank {
                target: rank.target,
                column_rank: rank.column_rank,
                width,
            },
        });
    }
    for coefficient in incoming(nodes, rank.target) {
        if !blocked.contains(&coefficient) {
            recoverable.push(coefficient);
        }
    }
}

fn equation_rank(set: &ClampExperimentSet, target: VariableId) -> Option<&DeclaredEquationRank> {
    set.equation_ranks.iter().find(|rank| rank.target == target)
}

fn missing_clamps(
    nodes: &BTreeSet<VariableId>,
    experiments: &[DeclaredClampExperiment],
    target: VariableId,
) -> Vec<VariableId> {
    nodes
        .iter()
        .copied()
        .filter(|source| *source != target && !pair_condition(experiments, *source, target))
        .collect()
}

fn pair_condition(
    experiments: &[DeclaredClampExperiment],
    source: VariableId,
    target: VariableId,
) -> bool {
    experiments.iter().any(|experiment| {
        experiment.clamped.contains(&source) && experiment.passive.contains(&target)
    })
}

fn incoming(nodes: &BTreeSet<VariableId>, target: VariableId) -> Vec<DirectCoefficient> {
    nodes
        .iter()
        .copied()
        .filter(|source| *source != target)
        .map(|from| DirectCoefficient { into: target, from })
        .collect()
}

/// Why a joint coefficient-and-noise draw cannot be compiled as a mechanism.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum InadmissibleCoefficientDraw {
    /// The caller marked the draw inadmissible. The mark is not cleared.
    Marked,
    /// The weight is not a finite positive mass.
    WeightNotMass,
    /// The draw names unresolved directions. They were not filled in.
    UnresolvedDirections,
    /// A coefficient lies on the structural diagonal.
    StructuralDiagonal,
    /// A coefficient or the intercept is not finite.
    NonFiniteParameter,
    /// The disturbance on this draw is not a finite symmetric covariance of the node set.
    NoiseNotJoint,
    /// The draw does not name each off-diagonal coordinate once.
    IncompleteMechanism,
}

/// Admissible or inadmissible status of one draw. Inadmissible status is kept.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum DrawAdmissibility {
    /// The draw is a complete finite mechanism with a joint disturbance.
    Admissible,
    /// The draw stays in the law and is not compiled as a mechanism.
    Inadmissible(InadmissibleCoefficientDraw),
}

/// One supplied draw: coefficients, intercept, and the disturbance of the same draw.
///
/// `marked_inadmissible` is kept even when the numbers would otherwise compile.
#[derive(Clone, Debug, PartialEq)]
pub struct RecoveryDrawInput {
    /// Direct effects named by this draw. Missing coordinates stay missing.
    pub coefficients: Vec<(DirectCoefficient, f64)>,
    /// Intercept in node order.
    pub intercept: Vec<f64>,
    /// Disturbance mean of this draw, in node order.
    pub noise_mean: Vec<f64>,
    /// Row-major disturbance covariance of this draw.
    pub noise_covariance: Vec<f64>,
    /// Directions rank-aware recovery left free. Not a filled row of `K`.
    pub unresolved: Vec<UnresolvedDirection>,
    /// Unnormalized weight of this draw.
    pub weight: f64,
    /// When set, the draw stays [`InadmissibleCoefficientDraw::Marked`].
    pub marked_inadmissible: bool,
}

/// One retained draw. Fields are the supplied draw; status is not a point estimate.
#[derive(Clone, Debug, PartialEq)]
pub struct JointRecoveryDraw {
    coefficients: Vec<(DirectCoefficient, f64)>,
    intercept: Vec<f64>,
    noise_mean: Vec<f64>,
    noise_covariance: Vec<f64>,
    unresolved: Vec<UnresolvedDirection>,
    weight: f64,
    marked_inadmissible: bool,
    admissibility: DrawAdmissibility,
}

impl JointRecoveryDraw {
    /// Coefficients as supplied, including any incomplete or diagonal entries.
    #[must_use]
    pub fn coefficients(&self) -> &[(DirectCoefficient, f64)] {
        &self.coefficients
    }

    /// Intercept as supplied.
    #[must_use]
    pub fn intercept(&self) -> &[f64] {
        &self.intercept
    }

    /// Disturbance mean paired with these coefficients.
    #[must_use]
    pub fn noise_mean(&self) -> &[f64] {
        &self.noise_mean
    }

    /// Disturbance covariance paired with these coefficients.
    #[must_use]
    pub fn noise_covariance(&self) -> &[f64] {
        &self.noise_covariance
    }

    /// Unresolved directions. Empty when the draw names a complete mechanism.
    #[must_use]
    pub fn unresolved(&self) -> &[UnresolvedDirection] {
        &self.unresolved
    }

    /// Weight as supplied.
    #[must_use]
    pub const fn weight(&self) -> f64 {
        self.weight
    }

    /// Whether the caller marked the draw inadmissible.
    #[must_use]
    pub const fn marked_inadmissible(&self) -> bool {
        self.marked_inadmissible
    }

    /// Status after retention. An inadmissible input stays inadmissible.
    #[must_use]
    pub const fn admissibility(&self) -> DrawAdmissibility {
        self.admissibility
    }

    /// Whether compilation must refuse this draw.
    #[must_use]
    pub const fn is_inadmissible(&self) -> bool {
        matches!(self.admissibility, DrawAdmissibility::Inadmissible(_))
    }
}

/// Plug-in coefficients and one disturbance.
///
/// This is not [`RecoveredCoefficientUncertainty`]. A plug-in has no draw weight
/// and no inadmissible mass.
#[derive(Clone, Debug, PartialEq)]
pub struct RecoveredCoefficientPoint {
    coefficients: Vec<(DirectCoefficient, f64)>,
    noise_covariance: Vec<f64>,
}

impl RecoveredCoefficientPoint {
    /// A single mechanism and a single disturbance, with no draw weight.
    #[must_use]
    pub fn plug_in(
        coefficients: Vec<(DirectCoefficient, f64)>,
        noise_covariance: Vec<f64>,
    ) -> Self {
        Self { coefficients, noise_covariance }
    }

    /// Plug-in coefficients.
    #[must_use]
    pub fn coefficients(&self) -> &[(DirectCoefficient, f64)] {
        &self.coefficients
    }

    /// Plug-in disturbance covariance.
    #[must_use]
    pub fn noise_covariance(&self) -> &[f64] {
        &self.noise_covariance
    }
}

/// Why a coefficient law was not retained.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum UncertaintyFault {
    /// No nodes were named.
    EmptyNodes,
    /// A node label was repeated.
    DuplicateNode,
    /// Nodes were not strictly increasing.
    UnsortedNodes,
    /// The law contained no draws.
    NoDraws,
    /// A coefficient block names a different number of values than sources.
    BlockShape {
        /// Index of the block in the supplied slice.
        block: usize,
        /// Passive equation the block constrains.
        target: VariableId,
        /// Number of sources.
        sources: usize,
        /// Number of values.
        values: usize,
    },
    /// A block whose rows disagree cannot be paired with a disturbance.
    BlockInconsistent {
        /// Index of the block in the supplied slice.
        block: usize,
        /// Rank of the coefficient matrix.
        column_rank: usize,
    },
    /// A block that is not a finite coefficient matrix cannot be paired with a disturbance.
    BlockNotAConstraint {
        /// Index of the block in the supplied slice.
        block: usize,
    },
}

impl fmt::Display for UncertaintyFault {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyNodes => {
                formatter.write_str("a recovered coefficient law needs the observed nodes")
            }
            Self::DuplicateNode => formatter.write_str("a node was repeated"),
            Self::UnsortedNodes => formatter.write_str("nodes must be strictly increasing"),
            Self::NoDraws => formatter.write_str("a coefficient law needs its draws"),
            Self::BlockShape { block, target, sources, values } => write!(
                formatter,
                "coefficient block {block} into node {} names {sources} sources and {values} values",
                target.raw()
            ),
            Self::BlockInconsistent { block, column_rank } => write!(
                formatter,
                "coefficient block {block} has rows that disagree (column rank {column_rank})"
            ),
            Self::BlockNotAConstraint { block } => {
                write!(formatter, "coefficient block {block} is not a finite coefficient matrix")
            }
        }
    }
}

impl std::error::Error for UncertaintyFault {}

/// Refusal to read a coefficient law as a plug-in point.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PointCollapseRefusal {
    /// Number of retained draws.
    pub draws: usize,
    /// Mass of draws that stayed inadmissible.
    pub inadmissible_mass: f64,
}

impl fmt::Display for PointCollapseRefusal {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "a coefficient law of {} draws is not a point estimate (inadmissible mass {})",
            self.draws, self.inadmissible_mass
        )
    }
}

impl std::error::Error for PointCollapseRefusal {}

/// Joint law of recovered coefficients and disturbances.
///
/// Draws are not dropped and are not replaced by [`RecoveredCoefficientPoint`].
/// The disturbance on a draw is that draw's disturbance.
#[derive(Clone, Debug, PartialEq)]
#[must_use = "dropping the law drops inadmissible mass with the admissible draws"]
pub struct RecoveredCoefficientUncertainty {
    nodes: Vec<VariableId>,
    draws: Vec<JointRecoveryDraw>,
}

impl RecoveredCoefficientUncertainty {
    /// Retain every draw. Inadmissible draws stay inadmissible.
    ///
    /// # Errors
    ///
    /// [`UncertaintyFault`] when `nodes` is empty, unsorted, or duplicated, or
    /// when `draws` is empty.
    pub fn retain(
        nodes: &[VariableId],
        draws: Vec<RecoveryDrawInput>,
    ) -> Result<Self, UncertaintyFault> {
        checked_nodes(nodes)?;
        if draws.is_empty() {
            return Err(UncertaintyFault::NoDraws);
        }
        let draws = draws.into_iter().map(|draw| retain_draw(nodes, draw)).collect();
        Ok(Self { nodes: nodes.to_vec(), draws })
    }

    /// Nodes in the order that indexes intercepts and disturbances.
    #[must_use]
    pub fn nodes(&self) -> &[VariableId] {
        &self.nodes
    }

    /// Retained draws, in input order.
    #[must_use]
    pub fn draws(&self) -> &[JointRecoveryDraw] {
        &self.draws
    }

    /// Sum of finite non-negative weights.
    #[must_use]
    pub fn total_mass(&self) -> f64 {
        self.draws.iter().map(|draw| mass_weight(draw.weight)).sum()
    }

    /// Mass of draws whose status is inadmissible.
    #[must_use]
    pub fn inadmissible_mass(&self) -> f64 {
        self.draws
            .iter()
            .filter(|draw| draw.is_inadmissible())
            .map(|draw| mass_weight(draw.weight))
            .sum()
    }

    /// This law is not a plug-in.
    ///
    /// Collapsing the draws, or dropping the inadmissible ones and renormalizing,
    /// is refused. The refusal reports the retained mass.
    #[must_use = "a refusal is what keeps this law from being a point estimate"]
    pub fn try_collapse(&self) -> Result<RecoveredCoefficientPoint, PointCollapseRefusal> {
        Err(PointCollapseRefusal {
            draws: self.draws.len(),
            inadmissible_mass: self.inadmissible_mass(),
        })
    }
}

/// Pair `blocks` with one disturbance.
///
/// A [`AssembledCoefficientRecovery::NullDirections`] block adds unresolved
/// directions and no filled coordinates. A coefficient block contributes only
/// the entries it solved. Missing entries stay missing.
///
/// # Errors
///
/// [`UncertaintyFault`] when `nodes` is empty, unsorted, or duplicated, or when
/// a block cannot contribute: [`UncertaintyFault::BlockShape`] for a value list
/// that does not match its sources, [`UncertaintyFault::BlockInconsistent`] and
/// [`UncertaintyFault::BlockNotAConstraint`] for blocks that solved nothing. Each
/// names the offending block.
#[must_use = "an unretained draw is not part of the coefficient law"]
pub fn joint_draw_from_blocks(
    nodes: &[VariableId],
    blocks: &[AssembledCoefficientRecovery],
    intercept: Vec<f64>,
    noise_mean: Vec<f64>,
    noise_covariance: Vec<f64>,
    weight: f64,
) -> Result<JointRecoveryDraw, UncertaintyFault> {
    checked_nodes(nodes)?;
    let mut coefficients = Vec::new();
    let mut unresolved = Vec::new();
    for (index, block) in blocks.iter().enumerate() {
        match block {
            AssembledCoefficientRecovery::Coefficients { target, sources, values } => {
                if values.len() != sources.len() {
                    return Err(UncertaintyFault::BlockShape {
                        block: index,
                        target: *target,
                        sources: sources.len(),
                        values: values.len(),
                    });
                }
                for (&from, &value) in sources.iter().zip(values) {
                    coefficients.push((DirectCoefficient { into: *target, from }, value));
                }
            }
            AssembledCoefficientRecovery::NullDirections {
                target,
                sources,
                column_rank,
                directions,
                full_rank_was_asserted: _,
            } => {
                push_null_directions(*target, sources, *column_rank, directions, &mut unresolved);
            }
            AssembledCoefficientRecovery::Inconsistent { column_rank } => {
                return Err(UncertaintyFault::BlockInconsistent {
                    block: index,
                    column_rank: *column_rank,
                });
            }
            AssembledCoefficientRecovery::NotAConstraint => {
                return Err(UncertaintyFault::BlockNotAConstraint { block: index });
            }
        }
    }
    Ok(retain_draw(
        nodes,
        RecoveryDrawInput {
            coefficients,
            intercept,
            noise_mean,
            noise_covariance,
            unresolved,
            weight,
            marked_inadmissible: false,
        },
    ))
}

fn push_null_directions(
    target: VariableId,
    sources: &[VariableId],
    column_rank: usize,
    directions: &[Vec<f64>],
    unresolved: &mut Vec<UnresolvedDirection>,
) {
    let reason =
        UnresolvedReason::DeficientConstraintRank { target, column_rank, width: sources.len() };
    let whole_row = |source: &VariableId| DirectCoefficient { into: target, from: *source };
    if directions.is_empty() {
        unresolved.push(UnresolvedDirection {
            support: sources.iter().map(whole_row).collect(),
            reason: reason.clone(),
        });
        return;
    }
    for direction in directions {
        let support = if direction.len() == sources.len() {
            let mut named: Vec<DirectCoefficient> = sources
                .iter()
                .zip(direction)
                .filter(|(_, weight)| !is_structural_zero(**weight))
                .map(|(source, _)| whole_row(source))
                .collect();
            if named.is_empty() {
                named = sources.iter().map(whole_row).collect();
            }
            named
        } else {
            sources.iter().map(whole_row).collect()
        };
        unresolved.push(UnresolvedDirection { support, reason: reason.clone() });
    }
}

fn is_structural_zero(weight: f64) -> bool {
    #[allow(clippy::float_cmp, reason = "a nullspace weight of exactly zero is not support")]
    let zero = weight == 0.0;
    zero
}

fn retain_draw(nodes: &[VariableId], draw: RecoveryDrawInput) -> JointRecoveryDraw {
    let admissibility = classify_draw(nodes, &draw);
    JointRecoveryDraw {
        coefficients: draw.coefficients,
        intercept: draw.intercept,
        noise_mean: draw.noise_mean,
        noise_covariance: draw.noise_covariance,
        unresolved: draw.unresolved,
        weight: draw.weight,
        marked_inadmissible: draw.marked_inadmissible,
        admissibility,
    }
}

fn classify_draw(nodes: &[VariableId], draw: &RecoveryDrawInput) -> DrawAdmissibility {
    if draw.marked_inadmissible {
        return DrawAdmissibility::Inadmissible(InadmissibleCoefficientDraw::Marked);
    }
    if !(draw.weight.is_finite() && draw.weight > 0.0) {
        return DrawAdmissibility::Inadmissible(InadmissibleCoefficientDraw::WeightNotMass);
    }
    if !draw.unresolved.is_empty() {
        return DrawAdmissibility::Inadmissible(InadmissibleCoefficientDraw::UnresolvedDirections);
    }
    if draw.coefficients.iter().any(|(coefficient, _)| coefficient.into == coefficient.from) {
        return DrawAdmissibility::Inadmissible(InadmissibleCoefficientDraw::StructuralDiagonal);
    }
    if !parameters_finite(draw) {
        return DrawAdmissibility::Inadmissible(InadmissibleCoefficientDraw::NonFiniteParameter);
    }
    if !noise_is_joint(nodes, draw) {
        return DrawAdmissibility::Inadmissible(InadmissibleCoefficientDraw::NoiseNotJoint);
    }
    if !mechanism_complete(nodes, draw) {
        return DrawAdmissibility::Inadmissible(InadmissibleCoefficientDraw::IncompleteMechanism);
    }
    DrawAdmissibility::Admissible
}

fn parameters_finite(draw: &RecoveryDrawInput) -> bool {
    draw.coefficients.iter().all(|(_, value)| value.is_finite())
        && draw.intercept.iter().all(|value| value.is_finite())
}

fn noise_is_joint(nodes: &[VariableId], draw: &RecoveryDrawInput) -> bool {
    let n = nodes.len();
    if draw.noise_mean.len() != n || draw.noise_covariance.len() != n.saturating_mul(n) {
        return false;
    }
    if draw.noise_mean.iter().any(|value| !value.is_finite()) {
        return false;
    }
    for row in 0..n {
        for column in 0..n {
            let value = draw.noise_covariance[row * n + column];
            if !value.is_finite() {
                return false;
            }
            #[allow(
                clippy::float_cmp,
                reason = "symmetry of a supplied covariance is an exact contract"
            )]
            let symmetric = value == draw.noise_covariance[column * n + row];
            if !symmetric {
                return false;
            }
        }
    }
    true
}

fn mechanism_complete(nodes: &[VariableId], draw: &RecoveryDrawInput) -> bool {
    if draw.intercept.len() != nodes.len() {
        return false;
    }
    let mut named = BTreeMap::<DirectCoefficient, f64>::new();
    for (coefficient, value) in &draw.coefficients {
        if !nodes.contains(&coefficient.into) || !nodes.contains(&coefficient.from) {
            return false;
        }
        if named.insert(*coefficient, *value).is_some() {
            return false;
        }
    }
    nodes.iter().enumerate().all(|(into_index, into)| {
        nodes.iter().enumerate().all(|(from_index, from)| {
            into_index == from_index
                || named.contains_key(&DirectCoefficient { into: *into, from: *from })
        })
    })
}

fn checked_nodes(nodes: &[VariableId]) -> Result<(), UncertaintyFault> {
    if nodes.is_empty() {
        return Err(UncertaintyFault::EmptyNodes);
    }
    for pair in nodes.windows(2) {
        if pair[0] == pair[1] {
            return Err(UncertaintyFault::DuplicateNode);
        }
        if pair[0] > pair[1] {
            return Err(UncertaintyFault::UnsortedNodes);
        }
    }
    Ok(())
}

fn mass_weight(weight: f64) -> f64 {
    if weight.is_finite() && weight > 0.0 { weight } else { 0.0 }
}

#[cfg(test)]
mod tests {
    use super::{
        AssembledCoefficientRecovery, CLAMP_EXPERIMENT_RECIPE, ClampExogeneity, ClampExperimentSet,
        DeclarationRecovery, DeclaredClampExperiment, DeclaredEquationRank, DeclaredNoiseLaw,
        DesignFault, DirectCoefficient, DrawAdmissibility, ExperimentDesign,
        ExperimentalRecoveryDecision, FiniteSampleConditioning, InadmissibleCoefficientDraw,
        JointRecoveryDraw, LATENT_VARIABLE_EXPANSION_ATTEMPT, NextAttempt, NotThisRecipe,
        PRIOR_REGULARIZATION_ATTEMPT, PhysicalDynamicalStability, RecoveredCoefficientPoint,
        RecoveredCoefficientUncertainty, RecoveryAttempt, RecoveryDrawInput, RecoveryWarning,
        RefusalReason, RegimeAssumption, UncertaintyFault, UniquenessRestriction, UnresolvedReason,
        WeakStability, as_recovered_mechanism, decide_experimental_recovery, joint_draw_from_blocks,
        recover_assembled_coefficients, recover_declared_coefficients, regime_is_identified,
    };
    use crate::experiment_moments::{
        ConstraintRow, RegimeSample, TargetConstraintBlock, UnitColumn,
        assemble_experiment_constraints,
    };
    use crate::experiment_regime::{
        DependencePremise, ExperimentRegime, PopulationId, RegimeContract, RegimeContractDecision,
        RegimeIntervention, SourceIdentity, UnitMask, validate_regime_contract,
    };
    use antecedent_core::{Value, VariableId};

    fn node(raw: u32) -> VariableId {
        VariableId::from_raw(raw)
    }

    fn coefficient(into: u32, from: u32) -> DirectCoefficient {
        DirectCoefficient { into: node(into), from: node(from) }
    }

    fn all_but_one(nodes: &[VariableId]) -> Vec<DeclaredClampExperiment> {
        nodes
            .iter()
            .copied()
            .map(|passive| DeclaredClampExperiment {
                clamped: nodes.iter().copied().filter(|other| *other != passive).collect(),
                passive: vec![passive],
                exogeneity: ClampExogeneity::RandomizedIntervention,
            })
            .collect()
    }

    fn clamp_design(
        nodes: Vec<VariableId>,
        experiments: Vec<DeclaredClampExperiment>,
        equation_ranks: Vec<DeclaredEquationRank>,
    ) -> ExperimentDesign {
        ExperimentDesign {
            nodes,
            attempt: RecoveryAttempt::ClampExperiments(ClampExperimentSet {
                experiments,
                noise: DeclaredNoiseLaw::IndependentGaussian,
                regimes: RegimeAssumption::Independent,
                weak_stability: WeakStability::InvertibilityDeclared,
                physical_dynamical_stability: PhysicalDynamicalStability::NotDeclared,
                finite_sample_conditioning: FiniteSampleConditioning::NotDeclared,
                uniqueness_restriction: UniquenessRestriction::NotOffered,
                equation_ranks,
            }),
        }
    }

    fn three() -> Vec<VariableId> {
        vec![node(0), node(1), node(2)]
    }

    #[test]
    fn accepts_all_but_one_randomized_clamps_under_the_minimum_recipe() {
        let nodes = three();
        let design = clamp_design(nodes.clone(), all_but_one(&nodes), Vec::new());
        let decision = decide_experimental_recovery(&design);
        let ExperimentalRecoveryDecision::Recovered(recovery) = decision else {
            panic!("expected a closed coefficient recovery");
        };
        assert_eq!(recovery.recipe, CLAMP_EXPERIMENT_RECIPE);
        assert!(recovery.goal_closed());
        assert!(recovery.generic);
        assert!(recovery.warnings.is_empty());
        assert!(recovery.unresolved.is_empty());
        assert_eq!(
            recovery.recoverable,
            vec![
                coefficient(0, 1),
                coefficient(0, 2),
                coefficient(1, 0),
                coefficient(1, 2),
                coefficient(2, 0),
                coefficient(2, 1),
            ]
        );
    }

    #[test]
    fn refuses_observational_equilibrium_ols_and_names_the_clamp_experiment_recipe() {
        let design = ExperimentDesign {
            nodes: three(),
            attempt: RecoveryAttempt::ObservationalEquilibriumOls,
        };
        let decision = decide_experimental_recovery(&design);
        assert!(!decision.goal_closed());
        assert!(decision.coefficient_recovery().is_none());
        let ExperimentalRecoveryDecision::Refused(refusal) = decision else {
            panic!("expected a refusal");
        };
        assert_eq!(refusal.reason, RefusalReason::ObservationalEquilibriumOls);
        assert_eq!(refusal.next_attempt, NextAttempt::ClampExperimentRecipe);
        assert_eq!(refusal.next_attempt.name(), CLAMP_EXPERIMENT_RECIPE);
    }

    #[test]
    fn insufficient_coverage_returns_partial_recovery_without_a_mechanism_matrix() {
        let nodes = three();
        let experiments = vec![DeclaredClampExperiment {
            clamped: vec![node(1), node(2)],
            passive: vec![node(0)],
            exogeneity: ClampExogeneity::RandomizedIntervention,
        }];
        let decision = decide_experimental_recovery(&clamp_design(nodes, experiments, Vec::new()));
        assert!(!decision.goal_closed());
        let ExperimentalRecoveryDecision::Partial(recovery) = decision else {
            panic!("expected a partial recovery");
        };
        assert_eq!(recovery.recipe, CLAMP_EXPERIMENT_RECIPE);
        assert_eq!(recovery.recoverable, vec![coefficient(0, 1), coefficient(0, 2)]);
        assert_eq!(recovery.unresolved.len(), 2);
        assert_eq!(recovery.unresolved[0].support, vec![coefficient(1, 0), coefficient(1, 2)]);
        assert_eq!(
            recovery.unresolved[0].reason,
            UnresolvedReason::IncompletePairCoverage {
                target: node(1),
                missing_clamps: vec![node(0), node(2)],
            }
        );
        assert_eq!(recovery.unresolved[1].support, vec![coefficient(2, 0), coefficient(2, 1)]);
        assert_eq!(
            recovery.unresolved[1].reason,
            UnresolvedReason::IncompletePairCoverage {
                target: node(2),
                missing_clamps: vec![node(0), node(1)],
            }
        );
    }

    #[test]
    fn rank_deficiency_names_unresolved_directions_and_leaves_the_goal_open() {
        let nodes = three();
        let ranks = vec![DeclaredEquationRank {
            target: node(0),
            column_rank: 1,
            nullspace: vec![vec![coefficient(0, 1)]],
        }];
        let decision =
            decide_experimental_recovery(&clamp_design(nodes.clone(), all_but_one(&nodes), ranks));
        assert!(!decision.goal_closed());
        let ExperimentalRecoveryDecision::Partial(recovery) = decision else {
            panic!("expected the goal to stay open");
        };
        assert_eq!(
            recovery.recoverable,
            vec![
                coefficient(0, 2),
                coefficient(1, 0),
                coefficient(1, 2),
                coefficient(2, 0),
                coefficient(2, 1),
            ]
        );
        assert_eq!(recovery.unresolved.len(), 1);
        assert_eq!(recovery.unresolved[0].support, vec![coefficient(0, 1)]);
        assert_eq!(
            recovery.unresolved[0].reason,
            UnresolvedReason::DeficientConstraintRank { target: node(0), column_rank: 1, width: 2 }
        );
    }

    #[test]
    fn records_latent_variable_expansion_as_not_this_recipe() {
        let as_attempt =
            ExperimentDesign { nodes: three(), attempt: RecoveryAttempt::LatentVariableExpansion };
        let decision = decide_experimental_recovery(&as_attempt);
        assert!(!decision.goal_closed());
        assert!(decision.coefficient_recovery().is_none());
        let ExperimentalRecoveryDecision::NotThisRecipe(attempt) = decision else {
            panic!("expected not_this_recipe");
        };
        assert_eq!(attempt, NotThisRecipe::LatentVariableExpansion);
        assert_eq!(attempt.name(), LATENT_VARIABLE_EXPANSION_ATTEMPT);

        let nodes = three();
        let mut design = clamp_design(nodes.clone(), all_but_one(&nodes), Vec::new());
        clamp_set_mut(&mut design).noise = DeclaredNoiseLaw::LatentConfounding;
        let decision = decide_experimental_recovery(&design);
        assert!(matches!(
            decision,
            ExperimentalRecoveryDecision::NotThisRecipe(NotThisRecipe::LatentVariableExpansion)
        ));
    }

    #[test]
    fn records_prior_regularization_as_not_this_recipe() {
        let design =
            ExperimentDesign { nodes: three(), attempt: RecoveryAttempt::PriorRegularization };
        let decision = decide_experimental_recovery(&design);
        assert!(!decision.goal_closed());
        assert!(decision.coefficient_recovery().is_none());
        let ExperimentalRecoveryDecision::NotThisRecipe(attempt) = decision else {
            panic!("expected not_this_recipe");
        };
        assert_eq!(attempt, NotThisRecipe::PriorRegularization);
        assert_eq!(attempt.name(), PRIOR_REGULARIZATION_ATTEMPT);
    }

    #[test]
    fn refuses_a_clamp_conditioned_on_the_factual_disturbance() {
        let nodes = three();
        let mut experiments = all_but_one(&nodes);
        experiments[0].exogeneity = ClampExogeneity::ConditionedOnFactualDisturbance;
        let decision = decide_experimental_recovery(&clamp_design(nodes, experiments, Vec::new()));
        assert!(decision.coefficient_recovery().is_none());
        let ExperimentalRecoveryDecision::Refused(refusal) = decision else {
            panic!("expected an exogeneity refusal");
        };
        assert_eq!(refusal.reason, RefusalReason::ClampNotExogenous { experiment: 0 });
        assert_eq!(refusal.next_attempt.name(), CLAMP_EXPERIMENT_RECIPE);
    }

    #[test]
    fn keeps_weak_stability_distinct_from_physical_dynamical_stability() {
        let nodes = three();
        let mut design = clamp_design(nodes.clone(), all_but_one(&nodes), Vec::new());
        {
            let set = clamp_set_mut(&mut design);
            set.weak_stability = WeakStability::NotDeclared;
            set.physical_dynamical_stability = PhysicalDynamicalStability::Declared;
        }
        let decision = decide_experimental_recovery(&design);
        let ExperimentalRecoveryDecision::Refused(refusal) = decision else {
            panic!("physical dynamical stability must not stand in for invertibility");
        };
        assert_eq!(refusal.reason, RefusalReason::PhysicalDynamicalStabilityIsNotWeakStability);

        let mut design = clamp_design(nodes.clone(), all_but_one(&nodes), Vec::new());
        clamp_set_mut(&mut design).weak_stability = WeakStability::NotDeclared;
        let decision = decide_experimental_recovery(&design);
        let ExperimentalRecoveryDecision::Refused(refusal) = decision else {
            panic!("missing invertibility must be refused");
        };
        assert_eq!(refusal.reason, RefusalReason::WeakStabilityNotDeclared);

        let mut design = clamp_design(nodes.clone(), all_but_one(&nodes), Vec::new());
        clamp_set_mut(&mut design).physical_dynamical_stability =
            PhysicalDynamicalStability::Declared;
        let decision = decide_experimental_recovery(&design);
        assert!(matches!(decision, ExperimentalRecoveryDecision::Recovered(_)));
        assert!(decision.goal_closed());
    }

    #[test]
    fn refuses_dependent_regimes() {
        let nodes = three();
        let mut design = clamp_design(nodes.clone(), all_but_one(&nodes), Vec::new());
        clamp_set_mut(&mut design).regimes = RegimeAssumption::Dependent;
        let decision = decide_experimental_recovery(&design);
        let ExperimentalRecoveryDecision::Refused(refusal) = decision else {
            panic!("expected a regime refusal");
        };
        assert_eq!(refusal.reason, RefusalReason::DependentRegimes);
        assert_eq!(refusal.next_attempt, NextAttempt::ClampExperimentRecipe);
    }

    #[test]
    fn refuses_an_undeclared_noise_law() {
        let nodes = three();
        let mut design = clamp_design(nodes.clone(), all_but_one(&nodes), Vec::new());
        clamp_set_mut(&mut design).noise = DeclaredNoiseLaw::Undeclared;
        let decision = decide_experimental_recovery(&design);
        let ExperimentalRecoveryDecision::Refused(refusal) = decision else {
            panic!("expected a noise-law refusal");
        };
        assert_eq!(refusal.reason, RefusalReason::NoiseLawUndeclared);
    }

    #[test]
    fn null_experiment_does_not_satisfy_pair_coverage() {
        let nodes = three();
        let experiments = vec![DeclaredClampExperiment {
            clamped: Vec::new(),
            passive: nodes.clone(),
            exogeneity: ClampExogeneity::RandomizedIntervention,
        }];
        let decision = decide_experimental_recovery(&clamp_design(nodes, experiments, Vec::new()));
        let ExperimentalRecoveryDecision::Partial(recovery) = decision else {
            panic!("a null experiment is under-covered, not an OLS refusal");
        };
        assert!(!recovery.goal_closed());
        assert!(recovery.recoverable.is_empty());
        assert_eq!(recovery.unresolved.len(), 3);
    }

    #[test]
    fn refuses_a_clamp_set_that_does_not_partition_the_nodes() {
        let nodes = three();
        let experiments = vec![DeclaredClampExperiment {
            clamped: vec![node(0), node(1)],
            passive: vec![node(1)],
            exogeneity: ClampExogeneity::RandomizedIntervention,
        }];
        let decision = decide_experimental_recovery(&clamp_design(nodes, experiments, Vec::new()));
        let ExperimentalRecoveryDecision::Refused(refusal) = decision else {
            panic!("expected a design fault");
        };
        assert_eq!(
            refusal.reason,
            RefusalReason::DesignFault(DesignFault::ExperimentDoesNotPartition { experiment: 0 })
        );
    }

    fn deficient_into_zero() -> DeclaredEquationRank {
        DeclaredEquationRank {
            target: node(0),
            column_rank: 0,
            nullspace: vec![vec![coefficient(0, 1)], vec![coefficient(0, 2)]],
        }
    }

    fn one_passive_experiment() -> Vec<DeclaredClampExperiment> {
        vec![DeclaredClampExperiment {
            clamped: vec![node(1), node(2)],
            passive: vec![node(0)],
            exogeneity: ClampExogeneity::RandomizedIntervention,
        }]
    }

    #[test]
    fn sample_full_column_rank_does_not_recover_a_weak_design() {
        let nodes = three();
        let mut design = clamp_design(nodes, one_passive_experiment(), Vec::new());
        clamp_set_mut(&mut design).finite_sample_conditioning =
            FiniteSampleConditioning::SampleFullColumnRank;
        let decision = decide_experimental_recovery(&design);
        let ExperimentalRecoveryDecision::Partial(recovery) = decision else {
            panic!("a sample full rank must not close an under-covered row");
        };
        assert!(!recovery.goal_closed());
        assert_eq!(
            recovery.finite_sample_conditioning,
            FiniteSampleConditioning::SampleFullColumnRank
        );
        assert_eq!(recovery.uniqueness_restriction, UniquenessRestriction::NotOffered);
        assert_eq!(recovery.recoverable, vec![coefficient(0, 1), coefficient(0, 2)]);
        assert_eq!(recovery.unresolved.len(), 2);
        assert!(recovery.unresolved.iter().all(|direction| {
            matches!(direction.reason, UnresolvedReason::IncompletePairCoverage { .. })
        }));
        assert!(!recovery.recoverable.contains(&coefficient(1, 0)));
        assert!(!recovery.recoverable.contains(&coefficient(2, 0)));
    }

    #[test]
    fn sample_full_column_rank_does_not_license_a_rank_on_an_uncovered_row() {
        let nodes = three();
        let ranks =
            vec![DeclaredEquationRank { target: node(1), column_rank: 2, nullspace: Vec::new() }];
        let mut design = clamp_design(nodes, one_passive_experiment(), ranks);
        clamp_set_mut(&mut design).finite_sample_conditioning =
            FiniteSampleConditioning::SampleFullColumnRank;
        let decision = decide_experimental_recovery(&design);
        let ExperimentalRecoveryDecision::Partial(recovery) = decision else {
            panic!("an uncovered rank declaration is a partial report, not a refusal");
        };
        assert!(!recovery.goal_closed());
        assert!(!recovery.generic);
        assert_eq!(
            recovery.warnings,
            vec![RecoveryWarning::RankDeclarationOnUncoveredTarget { target: node(1) }]
        );
        assert_eq!(
            recovery.warnings[0].design_fault(),
            DesignFault::RankDeclarationOnUncoveredTarget
        );
        assert!(recovery.recoverable.iter().all(|coefficient| coefficient.into == node(0)));
        assert!(recovery.unresolved.iter().any(|direction| {
            matches!(
                direction.reason,
                UnresolvedReason::IncompletePairCoverage { target, .. } if target == node(1)
            )
        }));
    }

    #[test]
    fn finite_condition_number_does_not_fill_a_rank_deficient_row() {
        let nodes = three();
        let mut design =
            clamp_design(nodes.clone(), all_but_one(&nodes), vec![deficient_into_zero()]);
        clamp_set_mut(&mut design).finite_sample_conditioning =
            FiniteSampleConditioning::FiniteConditionNumber;
        let decision = decide_experimental_recovery(&design);
        let ExperimentalRecoveryDecision::Partial(recovery) = decision else {
            panic!("a finite condition number must not fill a nullspace");
        };
        assert!(!recovery.goal_closed());
        assert_eq!(
            recovery.finite_sample_conditioning,
            FiniteSampleConditioning::FiniteConditionNumber
        );
        assert_eq!(
            recovery.recoverable,
            vec![coefficient(1, 0), coefficient(1, 2), coefficient(2, 0), coefficient(2, 1)]
        );
        assert_eq!(recovery.unresolved.len(), 2);
        assert_eq!(recovery.unresolved[0].support, vec![coefficient(0, 1)]);
        assert_eq!(recovery.unresolved[1].support, vec![coefficient(0, 2)]);
        assert!(recovery.unresolved.iter().all(|direction| {
            direction.reason
                == UnresolvedReason::DeficientConstraintRank {
                    target: node(0),
                    column_rank: 0,
                    width: 2,
                }
        }));
    }

    #[test]
    fn finite_condition_number_does_not_open_a_covered_full_rank_design() {
        let nodes = three();
        let mut design = clamp_design(nodes.clone(), all_but_one(&nodes), Vec::new());
        clamp_set_mut(&mut design).finite_sample_conditioning =
            FiniteSampleConditioning::FiniteConditionNumber;
        let decision = decide_experimental_recovery(&design);
        let ExperimentalRecoveryDecision::Recovered(recovery) = decision else {
            panic!("a finite condition number is not a rank deficiency");
        };
        assert!(recovery.goal_closed());
        assert!(recovery.unresolved.is_empty());
        assert_eq!(recovery.recoverable.len(), 6);
        assert_eq!(
            recovery.finite_sample_conditioning,
            FiniteSampleConditioning::FiniteConditionNumber
        );
    }

    #[test]
    fn physical_stability_does_not_resolve_rank_deficiency() {
        let nodes = three();
        let mut design =
            clamp_design(nodes.clone(), all_but_one(&nodes), vec![deficient_into_zero()]);
        clamp_set_mut(&mut design).physical_dynamical_stability =
            PhysicalDynamicalStability::Declared;
        let decision = decide_experimental_recovery(&design);
        let ExperimentalRecoveryDecision::Partial(recovery) = decision else {
            panic!("physical stability must not select the nullspace");
        };
        assert!(!recovery.goal_closed());
        assert_eq!(recovery.uniqueness_restriction, UniquenessRestriction::NotOffered);
        assert_eq!(recovery.unresolved.len(), 2);
        assert!(!recovery.recoverable.contains(&coefficient(0, 1)));
        assert!(!recovery.recoverable.contains(&coefficient(0, 2)));
    }

    #[test]
    fn uniqueness_restriction_is_not_a_stability_certificate() {
        for restriction in [
            UniquenessRestriction::MinimumNormCompletion,
            UniquenessRestriction::BackgroundZero,
            UniquenessRestriction::Faithfulness,
        ] {
            let nodes = three();
            let mut design = clamp_design(nodes.clone(), all_but_one(&nodes), Vec::new());
            {
                let set = clamp_set_mut(&mut design);
                set.weak_stability = WeakStability::NotDeclared;
                set.uniqueness_restriction = restriction;
            }
            let decision = decide_experimental_recovery(&design);
            assert!(decision.coefficient_recovery().is_none());
            let ExperimentalRecoveryDecision::Refused(refusal) = decision else {
                panic!("a uniqueness restriction must not stand in for invertibility");
            };
            assert_eq!(
                refusal.reason,
                RefusalReason::UniquenessRestrictionIsNotAStabilityCertificate
            );
            assert_eq!(refusal.next_attempt, NextAttempt::ClampExperimentRecipe);
        }

        let nodes = three();
        let mut design = clamp_design(nodes.clone(), all_but_one(&nodes), Vec::new());
        {
            let set = clamp_set_mut(&mut design);
            set.weak_stability = WeakStability::NotDeclared;
            set.physical_dynamical_stability = PhysicalDynamicalStability::Declared;
            set.uniqueness_restriction = UniquenessRestriction::MinimumNormCompletion;
        }
        let decision = decide_experimental_recovery(&design);
        let ExperimentalRecoveryDecision::Refused(refusal) = decision else {
            panic!("physical stability stays a distinct refusal");
        };
        assert_eq!(refusal.reason, RefusalReason::PhysicalDynamicalStabilityIsNotWeakStability);

        let nodes = three();
        let mut design = clamp_design(nodes.clone(), all_but_one(&nodes), Vec::new());
        clamp_set_mut(&mut design).uniqueness_restriction =
            UniquenessRestriction::MinimumNormCompletion;
        let decision = decide_experimental_recovery(&design);
        let ExperimentalRecoveryDecision::Recovered(recovery) = decision else {
            panic!("a restriction on a closed design is not a stability certificate");
        };
        assert!(recovery.goal_closed());
        assert_eq!(recovery.uniqueness_restriction, UniquenessRestriction::MinimumNormCompletion);
        assert_eq!(
            clamp_set_mut(&mut design).physical_dynamical_stability,
            PhysicalDynamicalStability::NotDeclared
        );
    }

    #[test]
    fn minimum_norm_and_background_zero_do_not_fill_rank_deficiency() {
        for restriction in [
            UniquenessRestriction::MinimumNormCompletion,
            UniquenessRestriction::BackgroundZero,
            UniquenessRestriction::Faithfulness,
        ] {
            let nodes = three();
            let mut design =
                clamp_design(nodes.clone(), all_but_one(&nodes), vec![deficient_into_zero()]);
            clamp_set_mut(&mut design).uniqueness_restriction = restriction;
            let decision = decide_experimental_recovery(&design);
            let ExperimentalRecoveryDecision::Partial(recovery) = decision else {
                panic!("a uniqueness restriction must not fill the nullspace");
            };
            assert!(!recovery.goal_closed());
            assert_eq!(recovery.uniqueness_restriction, restriction);
            assert_eq!(
                recovery.recoverable,
                vec![coefficient(1, 0), coefficient(1, 2), coefficient(2, 0), coefficient(2, 1)]
            );
            assert_eq!(recovery.unresolved[0].support, vec![coefficient(0, 1)]);
            assert_eq!(recovery.unresolved[1].support, vec![coefficient(0, 2)]);
        }
    }

    fn clamp_set_mut(design: &mut ExperimentDesign) -> &mut ClampExperimentSet {
        match &mut design.attempt {
            RecoveryAttempt::ClampExperiments(set) => set,
            RecoveryAttempt::ObservationalEquilibriumOls
            | RecoveryAttempt::PriorRegularization
            | RecoveryAttempt::LatentVariableExpansion => {
                panic!("expected a clamp-experiment attempt")
            }
        }
    }

    fn parallel_constraint_into_z() -> TargetConstraintBlock {
        let independent = DependencePremise::Independence;
        let contract = {
            let decision = validate_regime_contract(&RegimeContract {
                regimes: vec![
                    clamp_regime(&[0, 1, 2], &[0], independent, 1),
                    clamp_regime(&[0, 1, 2], &[1], independent, 2),
                ],
                mask: UnitMask { dropped: Vec::new() },
            });
            let RegimeContractDecision::Accepted(contract) = decision else {
                panic!("the fixture regime contract was refused");
            };
            contract
        };
        let (x, y, z) = ([1.0, -1.0], [1.0, -1.0], [1.0, -1.0]);
        let columns = [
            UnitColumn { unit: node(0), values: &x },
            UnitColumn { unit: node(1), values: &y },
            UnitColumn { unit: node(2), values: &z },
        ];
        let samples = [
            RegimeSample { columns: &columns, source: SourceIdentity::from_raw(1) },
            RegimeSample { columns: &columns, source: SourceIdentity::from_raw(2) },
        ];
        let assembled =
            assemble_experiment_constraints(&contract, &samples, u64::MAX).expect("z is covered");
        assembled.blocks.into_iter().find(|block| block.target == node(2)).expect("the block for z")
    }

    fn clamp_regime(
        units: &[u32],
        targets: &[u32],
        premise: DependencePremise,
        source: u64,
    ) -> ExperimentRegime {
        ExperimentRegime {
            targets: targets
                .iter()
                .map(|&raw| RegimeIntervention { unit: node(raw), value: Value::Int64(0) })
                .collect(),
            units: units.iter().copied().map(node).collect(),
            population: PopulationId::from_raw(1),
            premise,
            source: SourceIdentity::from_raw(source),
        }
    }

    #[test]
    fn declared_rank_and_invertibility_do_not_return_coefficients() {
        let full = DeclaredEquationRank { target: node(2), column_rank: 2, nullspace: Vec::new() };
        assert_eq!(
            recover_declared_coefficients(Some(&full), WeakStability::InvertibilityDeclared),
            DeclarationRecovery::NoCoefficients
        );
        assert_eq!(
            recover_declared_coefficients(None, WeakStability::InvertibilityDeclared),
            DeclarationRecovery::NoCoefficients
        );
        let block = TargetConstraintBlock {
            target: node(2),
            sources: vec![node(0), node(1)],
            rows: vec![
                ConstraintRow { regime: 0, probe: node(0) },
                ConstraintRow { regime: 1, probe: node(1) },
            ],
            coefficients: vec![1.0, 0.0, 0.0, 1.0],
            rhs: vec![0.25, 0.5],
        };
        let from_matrix = recover_assembled_coefficients(
            &block,
            Some(&full),
            WeakStability::InvertibilityDeclared,
        );
        assert_eq!(from_matrix.coefficients(), Some(&[0.25, 0.5][..]));
    }

    #[test]
    fn caller_asserted_full_rank_on_a_deficient_matrix_does_not_recover_coefficients() {
        let block = parallel_constraint_into_z();
        assert_eq!(block.coefficients, vec![1.0, 1.0, 1.0, 1.0]);
        // Full column rank, with a support that is not the matrix null direction.
        let asserted = DeclaredEquationRank {
            target: block.target,
            column_rank: block.sources.len(),
            nullspace: vec![vec![coefficient(2, 0)]],
        };
        let recovery = recover_assembled_coefficients(
            &block,
            Some(&asserted),
            WeakStability::InvertibilityDeclared,
        );
        assert!(recovery.coefficients().is_none());
        let AssembledCoefficientRecovery::NullDirections {
            column_rank,
            directions,
            full_rank_was_asserted,
            ..
        } = recovery
        else {
            panic!("a full-rank claim must not replace the matrix nullspace");
        };
        assert!(full_rank_was_asserted);
        assert_eq!(column_rank, 1);
        assert_eq!(directions, vec![vec![-1.0, 1.0]]);
        let without_invertibility =
            recover_assembled_coefficients(&block, Some(&asserted), WeakStability::NotDeclared);
        assert_eq!(without_invertibility.coefficients(), None);
        assert!(matches!(
            without_invertibility,
            AssembledCoefficientRecovery::NullDirections { full_rank_was_asserted: true, .. }
        ));
    }

    fn two_nodes() -> Vec<VariableId> {
        vec![node(0), node(1)]
    }

    fn paired_noise(variance: f64) -> (Vec<f64>, Vec<f64>) {
        (vec![0.0, 0.0], vec![1.0, 0.0, 0.0, variance])
    }

    fn complete_draw(variance: f64, weight: f64, marked_inadmissible: bool) -> RecoveryDrawInput {
        let (noise_mean, noise_covariance) = paired_noise(variance);
        RecoveryDrawInput {
            coefficients: vec![(coefficient(0, 1), 0.4), (coefficient(1, 0), 0.5)],
            intercept: vec![0.0, 0.0],
            noise_mean,
            noise_covariance,
            unresolved: Vec::new(),
            weight,
            marked_inadmissible,
        }
    }

    #[test]
    fn inadmissible_draw_stays_inadmissible_and_uncertainty_is_not_a_point() {
        let nodes = two_nodes();
        let point = RecoveredCoefficientPoint::plug_in(
            vec![(coefficient(0, 1), 0.4), (coefficient(1, 0), 0.5)],
            paired_noise(1.0).1,
        );
        let uncertainty = RecoveredCoefficientUncertainty::retain(
            &nodes,
            vec![complete_draw(1.0, 0.75, false), complete_draw(4.0, 0.25, true)],
        )
        .expect("two draws");
        let refused = uncertainty.try_collapse().expect_err("a law is not a plug-in");
        assert_eq!(refused.draws, 2);
        assert!((refused.inadmissible_mass - 0.25).abs() < 1e-12);
        assert!((uncertainty.inadmissible_mass() - 0.25).abs() < 1e-12);
        assert!((uncertainty.total_mass() - 1.0).abs() < 1e-12);
        assert_eq!(uncertainty.draws().len(), 2);
        assert_eq!(
            uncertainty.draws()[1].admissibility(),
            DrawAdmissibility::Inadmissible(InadmissibleCoefficientDraw::Marked)
        );
        assert_eq!(uncertainty.draws()[1].noise_covariance(), paired_noise(4.0).1.as_slice());
        assert_ne!(uncertainty.draws()[1].noise_covariance(), point.noise_covariance());
        assert!(uncertainty.draws().iter().any(JointRecoveryDraw::is_inadmissible));
    }

    #[test]
    fn rank_deficient_block_keeps_unresolved_directions_without_a_mechanism_matrix() {
        let nodes = two_nodes();
        let (noise_mean, noise_covariance) = paired_noise(1.0);
        let block = AssembledCoefficientRecovery::NullDirections {
            target: node(1),
            sources: vec![node(0)],
            column_rank: 0,
            directions: vec![vec![1.0]],
            full_rank_was_asserted: false,
        };
        let draw = joint_draw_from_blocks(
            &nodes,
            &[block],
            vec![0.0, 0.0],
            noise_mean,
            noise_covariance.clone(),
            1.0,
        )
        .expect("null directions are a draw");
        assert_eq!(
            draw.admissibility(),
            DrawAdmissibility::Inadmissible(InadmissibleCoefficientDraw::UnresolvedDirections)
        );
        assert!(draw.coefficients().is_empty());
        assert_eq!(draw.unresolved().len(), 1);
        assert_eq!(draw.unresolved()[0].support, vec![coefficient(1, 0)]);
        assert_eq!(draw.noise_covariance(), noise_covariance.as_slice());
        assert!(draw.coefficients().len() < nodes.len() * nodes.len());
        assert!(as_recovered_mechanism(&draw).is_none());
    }

    fn draw_from(
        block: AssembledCoefficientRecovery,
    ) -> Result<JointRecoveryDraw, UncertaintyFault> {
        let (noise_mean, noise_covariance) = paired_noise(1.0);
        joint_draw_from_blocks(
            &two_nodes(),
            &[block],
            vec![0.0, 0.0],
            noise_mean,
            noise_covariance,
            1.0,
        )
    }

    #[test]
    fn block_with_mismatched_values_is_refused_and_named() {
        let refused = draw_from(AssembledCoefficientRecovery::Coefficients {
            target: node(1),
            sources: vec![node(0)],
            values: vec![0.5, 0.6],
        })
        .expect_err("a value list that does not match its sources");
        assert_eq!(
            refused,
            UncertaintyFault::BlockShape { block: 0, target: node(1), sources: 1, values: 2 }
        );
    }

    #[test]
    fn inconsistent_and_non_constraint_blocks_are_refused_and_named() {
        let solved = AssembledCoefficientRecovery::Coefficients {
            target: node(0),
            sources: vec![node(1)],
            values: vec![0.4],
        };
        let (noise_mean, noise_covariance) = paired_noise(1.0);
        let refused = joint_draw_from_blocks(
            &two_nodes(),
            &[solved.clone(), AssembledCoefficientRecovery::Inconsistent { column_rank: 1 }],
            vec![0.0, 0.0],
            noise_mean.clone(),
            noise_covariance.clone(),
            1.0,
        )
        .expect_err("rows that disagree");
        assert_eq!(refused, UncertaintyFault::BlockInconsistent { block: 1, column_rank: 1 });
        let refused = joint_draw_from_blocks(
            &two_nodes(),
            &[solved, AssembledCoefficientRecovery::NotAConstraint],
            vec![0.0, 0.0],
            noise_mean,
            noise_covariance,
            1.0,
        )
        .expect_err("not a coefficient matrix");
        assert_eq!(refused, UncertaintyFault::BlockNotAConstraint { block: 1 });
        assert!(refused.to_string().contains("block 1"));
    }

    /// `x1 = 0.5 x0 + e1`, `x2 = 0.25 x0 + 0.4 x1 + e2`, `x0 = e0`.
    fn simulate_chain(seed: u64, rows: usize, clamped: &[u32]) -> [Vec<f64>; 3] {
        let mut rng = Lcg(seed);
        let mut columns = [vec![0.0; rows], vec![0.0; rows], vec![0.0; rows]];
        for row in 0..rows {
            let x0 = rng.normal();
            let x1 = if clamped.contains(&1) { rng.normal() } else { 0.5 * x0 + rng.normal() };
            let x2 = if clamped.contains(&2) {
                rng.normal()
            } else {
                0.25 * x0 + 0.4 * x1 + rng.normal()
            };
            columns[0][row] = x0;
            columns[1][row] = x1;
            columns[2][row] = x2;
        }
        columns
    }

    fn fixed_joint_holdout(rows: usize) -> [Vec<f64>; 3] {
        let mut rng = Lcg(90);
        let mut columns = [vec![2.0; rows], vec![1.25; rows], vec![0.0; rows]];
        for row in 0..rows {
            columns[2][row] = 0.25 * 2.0 + 0.4 * 1.25 + rng.normal();
        }
        columns
    }

    struct Lcg(u64);

    impl Lcg {
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
            self.0
        }

        fn uniform(&mut self) -> f64 {
            let bits = (self.next_u64() >> 11) as f64;
            bits / ((1_u64 << 53) as f64)
        }

        fn normal(&mut self) -> f64 {
            let u1 = self.uniform().max(1.0e-12);
            let u2 = self.uniform();
            (-2.0 * u1.ln()).sqrt() * (std::f64::consts::PI * 2.0 * u2).cos()
        }
    }

    fn single_clamp(clamped: u32) -> DeclaredClampExperiment {
        let nodes = [0_u32, 1, 2];
        DeclaredClampExperiment {
            clamped: vec![node(clamped)],
            passive: nodes.into_iter().filter(|other| *other != clamped).map(node).collect(),
            exogeneity: ClampExogeneity::RandomizedIntervention,
        }
    }

    fn effect(coefficients: &[(DirectCoefficient, f64)], into: u32, from: u32) -> f64 {
        coefficients
            .iter()
            .find(|(coefficient, _)| {
                coefficient.into == node(into) && coefficient.from == node(from)
            })
            .map(|(_, value)| *value)
            .expect("recovered coefficient")
    }

    #[test]
    fn held_out_joint_regime_uses_coefficient_uncertainty() {
        let rows = 2_000_u16;
        let row_count = usize::from(rows);
        let training = [
            simulate_chain(1, row_count, &[0]),
            simulate_chain(2, row_count, &[1]),
            simulate_chain(3, row_count, &[2]),
        ];
        let experiments = vec![single_clamp(0), single_clamp(1), single_clamp(2)];
        let decision = decide_experimental_recovery(&clamp_design(
            three(),
            experiments.clone(),
            Vec::new(),
        ));
        let ExperimentalRecoveryDecision::Recovered(report) = &decision else {
            panic!("single-node clamps are generic identification");
        };
        assert!(report.generic);
        assert!(regime_is_identified(&decision, &experiments, &experiments[0]));
        let joint = DeclaredClampExperiment {
            clamped: vec![node(0), node(1)],
            passive: vec![node(2)],
            exogeneity: ClampExogeneity::RandomizedIntervention,
        };
        assert!(!regime_is_identified(&decision, &experiments, &joint));
        let null = DeclaredClampExperiment {
            clamped: Vec::new(),
            passive: three(),
            exogeneity: ClampExogeneity::RandomizedIntervention,
        };
        assert!(!regime_is_identified(&decision, &experiments, &null));

        let regimes = vec![
            clamp_regime(&[0, 1, 2], &[0], DependencePremise::Independence, 11),
            clamp_regime(&[0, 1, 2], &[1], DependencePremise::Independence, 12),
            clamp_regime(&[0, 1, 2], &[2], DependencePremise::Independence, 13),
        ];
        let contract = {
            let accepted = validate_regime_contract(&RegimeContract {
                regimes,
                mask: UnitMask { dropped: Vec::new() },
            });
            let RegimeContractDecision::Accepted(contract) = accepted else {
                panic!("training regimes");
            };
            contract
        };
        let c0 = [
            UnitColumn { unit: node(0), values: &training[0][0] },
            UnitColumn { unit: node(1), values: &training[0][1] },
            UnitColumn { unit: node(2), values: &training[0][2] },
        ];
        let c1 = [
            UnitColumn { unit: node(0), values: &training[1][0] },
            UnitColumn { unit: node(1), values: &training[1][1] },
            UnitColumn { unit: node(2), values: &training[1][2] },
        ];
        let c2 = [
            UnitColumn { unit: node(0), values: &training[2][0] },
            UnitColumn { unit: node(1), values: &training[2][1] },
            UnitColumn { unit: node(2), values: &training[2][2] },
        ];
        let samples = [
            RegimeSample { columns: &c0, source: SourceIdentity::from_raw(11) },
            RegimeSample { columns: &c1, source: SourceIdentity::from_raw(12) },
            RegimeSample { columns: &c2, source: SourceIdentity::from_raw(13) },
        ];
        let assembled = assemble_experiment_constraints(&contract, &samples, u64::MAX)
            .expect("single clamps cover every target");
        let mut coefficients = Vec::new();
        for block in &assembled.blocks {
            let solved = recover_assembled_coefficients(block, None, WeakStability::NotDeclared);
            let values = solved.coefficients().expect("a covered block is a mechanism row");
            for (source, value) in block.sources.iter().zip(values) {
                coefficients.push((
                    DirectCoefficient { into: block.target, from: *source },
                    *value,
                ));
            }
        }
        let close_to = |into: u32, from: u32, truth: f64| {
            let got = effect(&coefficients, into, from);
            assert!((got - truth).abs() < 0.12, "K[{into},{from}] = {got}, truth {truth}");
        };
        close_to(1, 0, 0.5);
        close_to(2, 0, 0.25);
        close_to(2, 1, 0.4);
        close_to(0, 1, 0.0);
        close_to(0, 2, 0.0);
        close_to(1, 2, 0.0);

        let perturb = |scale: f64, marked: bool, weight: f64| {
            let shifted = coefficients
                .iter()
                .map(|(coefficient, value)| {
                    let delta = if coefficient.into == node(2) && coefficient.from == node(0) {
                        scale
                    } else {
                        0.0
                    };
                    (*coefficient, value + delta)
                })
                .collect();
            RecoveryDrawInput {
                coefficients: shifted,
                intercept: vec![0.0, 0.0, 0.0],
                noise_mean: vec![0.0, 0.0, 0.0],
                noise_covariance: vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                unresolved: Vec::new(),
                weight,
                marked_inadmissible: marked,
            }
        };
        let nodes = three();
        let uncertainty = RecoveredCoefficientUncertainty::retain(
            &nodes,
            vec![
                perturb(0.0, false, 0.3),
                perturb(0.05, false, 0.3),
                perturb(-0.05, false, 0.3),
                perturb(20.0, true, 0.1),
            ],
        )
        .expect("coefficient law");
        let held = fixed_joint_holdout(row_count);
        let held_mean = held[2].iter().sum::<f64>() / f64::from(rows);
        let predict = |mechanism: &[(DirectCoefficient, f64)]| {
            effect(mechanism, 2, 0) * 2.0 + effect(mechanism, 2, 1) * 1.25
        };
        let admissible: Vec<_> =
            uncertainty.draws().iter().filter_map(as_recovered_mechanism).collect();
        assert_eq!(admissible.len(), 3);
        assert!(uncertainty.draws().iter().any(|draw| as_recovered_mechanism(draw).is_none()));
        let center = admissible.iter().map(|mechanism| predict(mechanism)).sum::<f64>() / 3.0;
        let spread = admissible
            .iter()
            .map(|mechanism| (predict(mechanism) - center).abs())
            .fold(0.0, f64::max);
        assert!(spread > 0.05, "held-out check must use the coefficient draws");
        assert!(
            (held_mean - center).abs() < 0.25,
            "held-out joint regime mean {held_mean} vs draw center {center}"
        );
    }

    #[test]
    fn an_unclamped_regime_and_a_deficient_design_are_not_recovered() {
        let nodes = three();
        let only_zero = vec![single_clamp(0)];
        let decision =
            decide_experimental_recovery(&clamp_design(nodes, only_zero.clone(), Vec::new()));
        assert!(matches!(decision, ExperimentalRecoveryDecision::Partial(_)));
        assert!(!decision.coefficient_recovery().is_some_and(|recovery| recovery.generic));
        let never = single_clamp(1);
        assert!(!regime_is_identified(&decision, &only_zero, &never));
        assert!(!regime_is_identified(&decision, &only_zero, &only_zero[0]));
    }
}
