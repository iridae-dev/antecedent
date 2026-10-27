//! Design judgement and coefficient recovery from declared and assembled blocks.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

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

pub(super) fn recovery_from_matrix(
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

pub(super) fn refuse(reason: RefusalReason) -> ExperimentalRecoveryDecision {
    ExperimentalRecoveryDecision::Refused(RecipeRefusal {
        reason,
        next_attempt: CLAMP_EXPERIMENT_RECIPE_ATTEMPT,
    })
}

pub(super) fn decide_clamp_set(
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

pub(super) fn stability_stand_in(set: &ClampExperimentSet) -> RefusalReason {
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

pub(super) fn validated_nodes(nodes: &[VariableId]) -> Result<BTreeSet<VariableId>, DesignFault> {
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

pub(super) fn validate_experiments(
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

pub(super) fn experiment_partitions(
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

pub(super) fn validate_ranks(
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

pub(super) fn validate_nullspace(
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

pub(super) fn first_non_exogenous(experiments: &[DeclaredClampExperiment]) -> Option<usize> {
    experiments.iter().position(|experiment| {
        matches!(experiment.exogeneity, ClampExogeneity::ConditionedOnFactualDisturbance)
    })
}

pub(super) fn recover(
    nodes: &BTreeSet<VariableId>,
    set: &ClampExperimentSet,
) -> ExperimentalRecoveryDecision {
    let width = nodes.len() - 1;
    let mut recoverable = Vec::new();
    let mut unresolved = Vec::new();
    let mut warnings = Vec::new();
    for target in nodes {
        let missing = missing_clamps(nodes, &set.experiments, *target);
        if !missing.is_empty() {
            if set.equation_ranks.iter().any(|rank| rank.target == *target) {
                warnings
                    .push(RecoveryWarning::RankDeclarationOnUncoveredTarget { target: *target });
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
            recovery
                .recoverable
                .iter()
                .any(|coefficient| coefficient.into == *into && coefficient.from == *from)
        })
    })
}

pub(super) fn same_clamp(left: &DeclaredClampExperiment, right: &DeclaredClampExperiment) -> bool {
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

pub(super) fn push_rank_deficiency(
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

pub(super) fn equation_rank(
    set: &ClampExperimentSet,
    target: VariableId,
) -> Option<&DeclaredEquationRank> {
    set.equation_ranks.iter().find(|rank| rank.target == target)
}

pub(super) fn missing_clamps(
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

pub(super) fn pair_condition(
    experiments: &[DeclaredClampExperiment],
    source: VariableId,
    target: VariableId,
) -> bool {
    experiments.iter().any(|experiment| {
        experiment.clamped.contains(&source) && experiment.passive.contains(&target)
    })
}

pub(super) fn incoming(nodes: &BTreeSet<VariableId>, target: VariableId) -> Vec<DirectCoefficient> {
    nodes
        .iter()
        .copied()
        .filter(|source| *source != target)
        .map(|from| DirectCoefficient { into: target, from })
        .collect()
}
