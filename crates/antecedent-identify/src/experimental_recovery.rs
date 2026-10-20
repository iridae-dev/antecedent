//! Minimum experimental recovery recipe for fully observed linear Gaussian systems.
//!
//! This is the Hyttinen–Eberhardt–Hoyer multi-experiment specification for a
//! declared set of randomized clamp experiments, not an estimator. It accepts or
//! refuses the design. It does not read a sample, fit a coefficient, or assemble
//! a mechanism matrix.
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
//!   give that row full column rank. A declared column rank below the row width
//!   names the unresolved directions and leaves the recovery goal open. The
//!   Moore-Penrose completion of that block is not applied.
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
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeSet;

use antecedent_core::VariableId;

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
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CoefficientRecovery {
    /// Recipe that produced the report.
    pub recipe: &'static str,
    /// Coordinates pinned down by coverage and rank.
    pub recoverable: Vec<DirectCoefficient>,
    /// Unresolved directions. Empty only when the coefficient goal is closed.
    pub unresolved: Vec<UnresolvedDirection>,
    /// Finite-sample condition offered with the design. Not applied.
    pub finite_sample_conditioning: FiniteSampleConditioning,
    /// Uniqueness restriction offered with the design. Not applied.
    pub uniqueness_restriction: UniquenessRestriction,
}

impl CoefficientRecovery {
    /// Whether every direct coefficient is recoverable.
    ///
    /// Finite-sample conditioning and a uniqueness restriction do not close the
    /// goal. A closed goal is coverage plus population rank.
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
    /// Coverage and rank stay separate diagnostics. A covered equation may
    /// declare a deficient rank; an uncovered equation is already an unresolved
    /// row and does not take a rank declaration.
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
    /// Every off-diagonal coordinate is recoverable. No mechanism matrix is attached.
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
    for target in nodes {
        let missing = missing_clamps(nodes, &set.experiments, *target);
        if !missing.is_empty() && set.equation_ranks.iter().any(|rank| rank.target == *target) {
            return refuse(RefusalReason::DesignFault(
                DesignFault::RankDeclarationOnUncoveredTarget,
            ));
        }
        if !missing.is_empty() {
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
    // Coverage and the declared population rank decide the coordinates.
    // The sample condition and the uniqueness restriction are copied and not applied.
    let recovery = CoefficientRecovery {
        recipe: CLAMP_EXPERIMENT_RECIPE,
        recoverable,
        unresolved,
        finite_sample_conditioning: set.finite_sample_conditioning,
        uniqueness_restriction: set.uniqueness_restriction,
    };
    if recovery.goal_closed() {
        return ExperimentalRecoveryDecision::Recovered(recovery);
    }
    ExperimentalRecoveryDecision::Partial(recovery)
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

#[cfg(test)]
mod tests {
    use super::{
        CLAMP_EXPERIMENT_RECIPE, ClampExogeneity, ClampExperimentSet, DeclaredClampExperiment,
        DeclaredEquationRank, DeclaredNoiseLaw, DesignFault, DirectCoefficient, ExperimentDesign,
        ExperimentalRecoveryDecision, FiniteSampleConditioning, LATENT_VARIABLE_EXPANSION_ATTEMPT,
        NextAttempt, NotThisRecipe, PRIOR_REGULARIZATION_ATTEMPT, PhysicalDynamicalStability,
        RecoveryAttempt, RefusalReason, RegimeAssumption, UniquenessRestriction, UnresolvedReason,
        WeakStability, decide_experimental_recovery,
    };
    use antecedent_core::VariableId;

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
        let ExperimentalRecoveryDecision::Refused(refusal) = decision else {
            panic!("sample rank must not turn an uncovered row into a rank solve");
        };
        assert_eq!(
            refusal.reason,
            RefusalReason::DesignFault(DesignFault::RankDeclarationOnUncoveredTarget)
        );
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
}
