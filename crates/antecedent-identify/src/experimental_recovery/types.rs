//! Declarations, decisions and outcome types of the clamp-experiment recipe.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

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

/// The attempt a refusal points to: satisfy this module's clamp-experiment
/// recipe.
pub const CLAMP_EXPERIMENT_RECIPE_ATTEMPT: NextAttempt =
    NextAttempt::new(CLAMP_EXPERIMENT_RECIPE, "satisfy the clamp-experiment recipe");

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
