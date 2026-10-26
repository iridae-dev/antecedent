//! Joint recovery draws and the sampling law of a recovery.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

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

pub(super) fn push_null_directions(
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

pub(super) fn is_structural_zero(weight: f64) -> bool {
    #[allow(clippy::float_cmp, reason = "a nullspace weight of exactly zero is not support")]
    let zero = weight == 0.0;
    zero
}

pub(super) fn retain_draw(nodes: &[VariableId], draw: RecoveryDrawInput) -> JointRecoveryDraw {
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

pub(super) fn classify_draw(nodes: &[VariableId], draw: &RecoveryDrawInput) -> DrawAdmissibility {
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

pub(super) fn parameters_finite(draw: &RecoveryDrawInput) -> bool {
    draw.coefficients.iter().all(|(_, value)| value.is_finite())
        && draw.intercept.iter().all(|value| value.is_finite())
}

pub(super) fn noise_is_joint(nodes: &[VariableId], draw: &RecoveryDrawInput) -> bool {
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

pub(super) fn mechanism_complete(nodes: &[VariableId], draw: &RecoveryDrawInput) -> bool {
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

pub(super) fn checked_nodes(nodes: &[VariableId]) -> Result<(), UncertaintyFault> {
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

pub(super) fn mass_weight(weight: f64) -> f64 {
    if weight.is_finite() && weight > 0.0 { weight } else { 0.0 }
}
