//! Contrasts of intervention worlds and the dose × time surface.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use crate::canonical_encode::{CanonicalEncode, count};
use crate::identity::CanonicalWriter;
use crate::ids::VariableId;
use crate::intervention::{Intervention, TemporalPolicy};
use crate::value::Value;
use std::collections::BTreeMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use super::model::{ActionKind, TimeKind};
use super::{
    HistoryWindow, InitialLaw, InterventionWorld, ScientificTargetError, ScientificWorldId,
    WorldCollection,
};

/// One finite nonzero term of an explicitly declared linear world functional.
#[derive(Clone, Copy, Debug)]
pub struct ContrastTerm {
    /// World the coefficient multiplies.
    pub world: ScientificWorldId,
    /// Finite nonzero coefficient; ordinary differences and interactions use `+1`/`-1`.
    pub coefficient: f64,
}

impl PartialEq for ContrastTerm {
    fn eq(&self, other: &Self) -> bool {
        self.world == other.world && self.coefficient.to_bits() == other.coefficient.to_bits()
    }
}
impl Eq for ContrastTerm {}
impl Hash for ContrastTerm {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.world.hash(state);
        self.coefficient.to_bits().hash(state);
    }
}

/// Shape of an [`InterventionContrast`]. Not a mediation and not a dose × time grid.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum ContrastForm {
    /// Explicit finite linear combination, including higher-order factorial contrasts.
    /// Per-world time frames remain declared; no exponential world expansion is implied.
    Linear,
    /// `E[Y(plus)] − E[Y(minus)]`, with `minus` the reference world.
    Difference {
        /// World with coefficient `+1`.
        plus: ScientificWorldId,
        /// World with coefficient `-1`. This is the reference world.
        minus: ScientificWorldId,
    },
    /// `E[Y(a,b)] − E[Y(a,0)] − E[Y(0,b)] + E[Y(0,0)]`.
    ///
    /// `0` is the contrast's reference world, not the absence of treatment.
    Interaction {
        /// World `Y(a,b)`, coefficient `+1`.
        joint: ScientificWorldId,
        /// World `Y(a,0)`, coefficient `-1`.
        a_at_reference: ScientificWorldId,
        /// World `Y(0,b)`, coefficient `-1`.
        b_at_reference: ScientificWorldId,
    },
}

/// A contrast of intervention worlds, including the two-intervention reference interaction.
///
/// The collection is exactly the worlds in the formula. Marginal intervals are not
/// subtracted here; the coefficients are the target.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct InterventionContrast {
    worlds: WorldCollection,
    form: ContrastForm,
    reference: ScientificWorldId,
    linear_terms: Arc<[ContrastTerm]>,
}

impl InterventionContrast {
    /// Declare an arbitrary finite linear combination of intervention-world means.
    ///
    /// Every supplied world appears exactly once with a finite nonzero coefficient.
    /// The reference is explicit. World quantities/populations/units/conditioning agree;
    /// each world keeps its own declared time frame and history. This supports explicit
    /// higher-order factorial contrasts and time-window functionals without generating
    /// undeclared worlds, identifying them, or subtracting marginal intervals.
    ///
    /// # Errors
    /// Missing/repeated worlds, an invalid coefficient, an empty functional, or a missing
    /// reference. Statistical and scientific support remain the compiler's responsibility.
    pub fn linear(
        worlds: WorldCollection,
        terms: impl IntoIterator<Item = (ScientificWorldId, f64)>,
        reference: ScientificWorldId,
    ) -> Result<Self, ScientificTargetError> {
        let mut terms: Vec<_> = terms
            .into_iter()
            .map(|(world, coefficient)| ContrastTerm { world, coefficient })
            .collect();
        if terms.is_empty()
            || terms.iter().any(|term| !term.coefficient.is_finite() || term.coefficient == 0.0)
        {
            return Err(ScientificTargetError::InvalidLinearContrast {
                reason: "coefficients must be finite and nonzero",
            });
        }
        terms.sort_by_key(|term| term.world);
        require_ids(&worlds, &terms.iter().map(|term| term.world).collect::<Vec<_>>())?;
        require_one_reference(&worlds, reference)?;
        Ok(Self { worlds, form: ContrastForm::Linear, reference, linear_terms: terms.into() })
    }

    /// Difference `E[Y(plus)] − E[Y(minus)]`.
    ///
    /// `reference` must be `minus`, and that world must be flagged. The two worlds
    /// share decision time, horizon, history, and initial law.
    ///
    /// # Errors
    ///
    /// See [`ScientificTargetError`] for an unknown, repeated, extra, or unflagged
    /// world, a reference that is not the baseline, worlds whose time frames differ, or
    /// [`ScientificTargetError::IncoherentContrast`] when the two worlds do not assign
    /// the same coordinates.
    pub fn difference(
        worlds: WorldCollection,
        plus: ScientificWorldId,
        minus: ScientificWorldId,
        reference: ScientificWorldId,
    ) -> Result<Self, ScientificTargetError> {
        require_ids(&worlds, &[plus, minus])?;
        if reference != minus {
            return Err(ScientificTargetError::ReferenceIsNotBaseline { id: reference });
        }
        require_one_reference(&worlds, reference)?;
        require_same_time(&worlds)?;
        require_coherent_difference(&worlds, plus, minus)?;
        Ok(Self {
            worlds,
            form: ContrastForm::Difference { plus, minus },
            reference,
            linear_terms: Arc::from([]),
        })
    }

    /// Interaction `E[Y(a,b)] − E[Y(a,0)] − E[Y(0,b)] + E[Y(0,0)]`.
    ///
    /// `reference` is the world `Y(0,0)`. The four worlds share one time frame and
    /// assign the same coordinates (a variable's level, or the schedule). Two disjoint
    /// factors `A` and `B` each move away from the reference: `a_at_reference` changes
    /// only `A`, `b_at_reference` only `B`, and `joint` takes both. The reference level
    /// is whatever the reference world assigns, not necessarily zero.
    /// There is no constructor that builds this from four separately estimated intervals.
    ///
    /// # Errors
    ///
    /// See [`ScientificTargetError`] for an unknown, repeated, extra, or wrongly
    /// flagged world, worlds whose time frames differ, or
    /// [`ScientificTargetError::IncoherentContrast`] when the four worlds are not the
    /// 2x2 cells `Y(a,b)`, `Y(a,0)`, `Y(0,b)`, `Y(0,0)` of two disjoint factors.
    pub fn interaction(
        worlds: WorldCollection,
        joint: ScientificWorldId,
        a_at_reference: ScientificWorldId,
        b_at_reference: ScientificWorldId,
        reference: ScientificWorldId,
    ) -> Result<Self, ScientificTargetError> {
        require_ids(&worlds, &[joint, a_at_reference, b_at_reference, reference])?;
        require_one_reference(&worlds, reference)?;
        require_same_time(&worlds)?;
        require_coherent_interaction(&worlds, joint, a_at_reference, b_at_reference, reference)?;
        Ok(Self {
            worlds,
            form: ContrastForm::Interaction { joint, a_at_reference, b_at_reference },
            reference,
            linear_terms: Arc::from([]),
        })
    }

    /// Worlds in the formula.
    #[must_use]
    pub const fn worlds(&self) -> &WorldCollection {
        &self.worlds
    }

    /// Difference or interaction.
    #[must_use]
    pub const fn form(&self) -> ContrastForm {
        self.form
    }

    /// Explicit reference world.
    #[must_use]
    pub const fn reference(&self) -> ScientificWorldId {
        self.reference
    }

    /// Coefficients in formula order.
    ///
    /// A difference is `(plus, +1)`, `(minus, -1)`. An interaction is
    /// `(joint, +1)`, `(a_at_reference, -1)`, `(b_at_reference, -1)`, `(reference, +1)`.
    #[must_use]
    pub fn terms(&self) -> Vec<ContrastTerm> {
        match self.form {
            ContrastForm::Linear => self.linear_terms.to_vec(),
            ContrastForm::Difference { plus, minus } => {
                vec![term(plus, 1), term(minus, -1)]
            }
            ContrastForm::Interaction { joint, a_at_reference, b_at_reference } => {
                vec![
                    term(joint, 1),
                    term(a_at_reference, -1),
                    term(b_at_reference, -1),
                    term(self.reference, 1),
                ]
            }
        }
    }
}

fn term(world: ScientificWorldId, coefficient: i8) -> ContrastTerm {
    ContrastTerm { world, coefficient: f64::from(coefficient) }
}

fn require_ids(
    worlds: &WorldCollection,
    ids: &[ScientificWorldId],
) -> Result<(), ScientificTargetError> {
    let mut sorted = ids.to_vec();
    sorted.sort_unstable();
    if sorted.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err(ScientificTargetError::WorldsNotDistinct);
    }
    for id in ids {
        if worlds.get(*id).is_none() {
            return Err(ScientificTargetError::UnknownWorld { id: *id });
        }
    }
    let actual: Vec<ScientificWorldId> =
        worlds.worlds().iter().map(InterventionWorld::id).collect();
    if actual != sorted {
        return Err(ScientificTargetError::ContrastWorldSet);
    }
    Ok(())
}

fn require_one_reference(
    worlds: &WorldCollection,
    reference: ScientificWorldId,
) -> Result<(), ScientificTargetError> {
    let Some(chosen) = worlds.get(reference) else {
        return Err(ScientificTargetError::UnknownWorld { id: reference });
    };
    if !chosen.is_reference() {
        return Err(ScientificTargetError::ReferenceNotMarked { id: reference });
    }
    if let Some(other) =
        worlds.worlds().iter().find(|world| world.id() != reference && world.is_reference())
    {
        return Err(ScientificTargetError::UnexpectedReference { id: other.id() });
    }
    Ok(())
}

fn require_same_time(worlds: &WorldCollection) -> Result<(), ScientificTargetError> {
    let Some((first, rest)) = worlds.worlds().split_first() else {
        return Err(ScientificTargetError::EmptyWorlds);
    };
    if rest.iter().any(|world| world.time() != first.time()) {
        Err(ScientificTargetError::IncompatibleTime)
    } else {
        Ok(())
    }
}

/// What a world fixes: a variable's intervention, or the schedule it runs under.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
enum Coordinate {
    Variable(VariableId),
    Schedule,
    /// One structural action kind on one variable.
    Structural(VariableId, &'static str),
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum Assigned<'a> {
    Intervention(Intervention),
    Schedule(&'a TemporalPolicy),
    Structural(&'a super::StructuralAction),
}

fn coordinates(world: &InterventionWorld) -> BTreeMap<Coordinate, Assigned<'_>> {
    let mut map = BTreeMap::new();
    match &world.action().kind {
        ActionKind::Interventions(interventions) => {
            for intervention in interventions.iter() {
                for variable in intervention.target_variables() {
                    map.insert(
                        Coordinate::Variable(variable),
                        Assigned::Intervention(
                            projected_intervention(intervention, variable)
                                .expect("intervention declares this target"),
                        ),
                    );
                }
            }
        }
        ActionKind::Schedule { intervention, policy } => {
            for variable in intervention.target_variables() {
                map.insert(
                    Coordinate::Variable(variable),
                    Assigned::Intervention(
                        projected_intervention(intervention, variable)
                            .expect("intervention declares this target"),
                    ),
                );
            }
            map.insert(Coordinate::Schedule, Assigned::Schedule(policy));
        }
        ActionKind::Structural(actions) => {
            for action in actions.iter() {
                map.insert(
                    Coordinate::Structural(action.variable(), action.kind_name()),
                    Assigned::Structural(action),
                );
            }
        }
    }
    map
}

/// A sequence may set several channels at different times. Compare only the steps
/// that set this coordinate, otherwise changing A incorrectly appears to change B too.
fn projected_intervention(
    intervention: &Intervention,
    variable: VariableId,
) -> Option<Intervention> {
    match intervention {
        Intervention::Sequence(sequence) => {
            let steps = sequence
                .steps
                .iter()
                .filter_map(|step| {
                    projected_intervention(&step.intervention, variable).map(|intervention| {
                        crate::intervention::SequencedIntervention {
                            intervention,
                            temporal: step.temporal.clone(),
                        }
                    })
                })
                .collect::<Vec<_>>();
            (!steps.is_empty()).then(|| {
                Intervention::Sequence(crate::intervention::InterventionSequence::new(steps))
            })
        }
        _ => intervention.target_variables().contains(&variable).then(|| intervention.clone()),
    }
}

fn incoherent(reason: &'static str) -> ScientificTargetError {
    ScientificTargetError::IncoherentContrast { reason }
}

fn require_coherent_difference(
    worlds: &WorldCollection,
    plus: ScientificWorldId,
    minus: ScientificWorldId,
) -> Result<(), ScientificTargetError> {
    let (Some(plus), Some(minus)) = (worlds.get(plus), worlds.get(minus)) else {
        return Err(ScientificTargetError::EmptyWorlds);
    };
    let plus = coordinates(plus);
    let minus = coordinates(minus);
    if !plus.keys().eq(minus.keys()) {
        return Err(incoherent("the two worlds assign different variables or schedules"));
    }
    if plus == minus {
        return Err(incoherent("the two worlds are the same intervention"));
    }
    Ok(())
}

fn require_coherent_interaction(
    worlds: &WorldCollection,
    joint: ScientificWorldId,
    a_at_reference: ScientificWorldId,
    b_at_reference: ScientificWorldId,
    reference: ScientificWorldId,
) -> Result<(), ScientificTargetError> {
    let (Some(joint), Some(a_world), Some(b_world), Some(reference)) = (
        worlds.get(joint),
        worlds.get(a_at_reference),
        worlds.get(b_at_reference),
        worlds.get(reference),
    ) else {
        return Err(ScientificTargetError::EmptyWorlds);
    };
    let joint = coordinates(joint);
    let a_world = coordinates(a_world);
    let b_world = coordinates(b_world);
    let reference = coordinates(reference);
    for other in [&joint, &a_world, &b_world] {
        if !other.keys().eq(reference.keys()) {
            return Err(incoherent("the four worlds assign different variables or schedules"));
        }
    }
    let mut moves_a = false;
    let mut moves_b = false;
    for (coordinate, at_reference) in &reference {
        let a = &a_world[coordinate];
        let b = &b_world[coordinate];
        let both = &joint[coordinate];
        let a_moves = a != at_reference;
        let b_moves = b != at_reference;
        if a_moves && b_moves {
            return Err(incoherent("the two factors move the same variable or schedule"));
        }
        let expected = if a_moves {
            a
        } else if b_moves {
            b
        } else {
            at_reference
        };
        if both != expected {
            return Err(incoherent("the joint world is not the reference moved by both factors"));
        }
        moves_a |= a_moves;
        moves_b |= b_moves;
    }
    if !moves_a || !moves_b {
        return Err(incoherent("each factor must move away from the reference"));
    }
    Ok(())
}

/// Dose × time surface: one world for every dose on every horizon.
///
/// Cell order is dose-major, then horizon. The surface is not a difference of
/// corner intervals; worlds on the grid may have different horizons, which a
/// contrast refuses.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct DoseTimeSurface {
    worlds: WorldCollection,
    dose: VariableId,
    level_bits: Arc<[u64]>,
    horizons: Arc<[u32]>,
    cells: Arc<[ScientificWorldId]>,
}

impl DoseTimeSurface {
    /// Grid. `cells` has one id per dose and horizon, in dose-major order.
    ///
    /// Each cell is a temporal schedule that hard-sets `dose` to that level.
    /// Every cell shares the policy, history, and initial law. Dose levels are
    /// finite and strictly increasing, with `-0.0` stored as `+0.0`. Horizons
    /// are strictly increasing and at least 1.
    ///
    /// # Errors
    ///
    /// [`ScientificTargetError::NotADoseTimeGrid`] when the declaration is not
    /// that rectangle, or [`ScientificTargetError::UnknownWorld`] when a cell id
    /// is missing.
    pub fn new(
        worlds: WorldCollection,
        dose: VariableId,
        levels: impl IntoIterator<Item = f64>,
        horizons: impl IntoIterator<Item = u32>,
        cells: impl IntoIterator<Item = ScientificWorldId>,
    ) -> Result<Self, ScientificTargetError> {
        let level_bits = canonical_levels(levels)?;
        let horizons = canonical_horizons(horizons)?;
        let cells: Arc<[ScientificWorldId]> = cells.into_iter().collect();
        check_grid(&worlds, dose, &level_bits, &horizons, &cells)?;
        Ok(Self { worlds, dose, level_bits, horizons, cells })
    }

    /// Worlds of the grid.
    #[must_use]
    pub const fn worlds(&self) -> &WorldCollection {
        &self.worlds
    }

    /// Dose variable.
    #[must_use]
    pub const fn dose(&self) -> VariableId {
        self.dose
    }

    /// Dose levels as IEEE-754 bits. `-0.0` is stored as `+0.0`.
    #[must_use]
    pub fn level_bits(&self) -> &[u64] {
        &self.level_bits
    }

    /// Horizons in axis order.
    #[must_use]
    pub fn horizons(&self) -> &[u32] {
        &self.horizons
    }

    /// Cell world ids, dose-major then horizon.
    #[must_use]
    pub fn cells(&self) -> &[ScientificWorldId] {
        &self.cells
    }
}

fn canonical_levels(
    levels: impl IntoIterator<Item = f64>,
) -> Result<Arc<[u64]>, ScientificTargetError> {
    let mut bits = Vec::new();
    let mut previous: Option<f64> = None;
    for level in levels {
        if !level.is_finite() {
            return Err(grid("dose levels must be finite and strictly increasing"));
        }
        let level = if level.to_bits() == (-0.0_f64).to_bits() { 0.0 } else { level };
        if previous.is_some_and(|prev| level.total_cmp(&prev) != std::cmp::Ordering::Greater) {
            return Err(grid("dose levels must be finite and strictly increasing"));
        }
        previous = Some(level);
        bits.push(level.to_bits());
    }
    if bits.is_empty() {
        return Err(grid("a dose-time grid names at least one dose and one horizon"));
    }
    Ok(Arc::from(bits))
}

fn canonical_horizons(
    horizons: impl IntoIterator<Item = u32>,
) -> Result<Arc<[u32]>, ScientificTargetError> {
    let horizons: Vec<u32> = horizons.into_iter().collect();
    if horizons.is_empty() {
        return Err(grid("a dose-time grid names at least one dose and one horizon"));
    }
    let mut previous = 0_u32;
    for horizon in &horizons {
        if *horizon == 0 || *horizon <= previous {
            return Err(grid("horizons must be strictly increasing and at least 1"));
        }
        previous = *horizon;
    }
    Ok(Arc::from(horizons))
}

fn check_grid(
    worlds: &WorldCollection,
    dose: VariableId,
    levels: &[u64],
    horizons: &[u32],
    cells: &[ScientificWorldId],
) -> Result<(), ScientificTargetError> {
    let Some(expected) = levels.len().checked_mul(horizons.len()) else {
        return Err(grid("dose-time grid is too large"));
    };
    if cells.len() != expected || worlds.len() != cells.len() {
        return Err(grid("cells must be one world per dose and horizon, and no others"));
    }
    let mut seen = cells.to_vec();
    seen.sort_unstable();
    if seen.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err(grid("a world is used for more than one cell"));
    }
    let mut anchor: Option<(TemporalPolicy, HistoryWindow, InitialLaw)> = None;
    for (index, id) in cells.iter().enumerate() {
        let Some(world) = worlds.get(*id) else {
            return Err(ScientificTargetError::UnknownWorld { id: *id });
        };
        if world.binding().outcome().outcome() == dose {
            return Err(grid("dose and outcome are the same variable"));
        }
        let dose_index = index / horizons.len();
        let horizon_index = index % horizons.len();
        let (policy, history, initial_law, horizon_steps, assigned) = cell_parts(world, dose)?;
        if horizon_steps != horizons[horizon_index] || assigned != levels[dose_index] {
            return Err(grid("a cell does not match its dose and horizon"));
        }
        if let Some((policy0, history0, law0)) = &anchor {
            if policy != policy0 || history != *history0 || initial_law != law0 {
                return Err(grid("cells must share a policy, history, and initial law"));
            }
        } else {
            anchor = Some((policy.clone(), history, initial_law.clone()));
        }
    }
    Ok(())
}

fn cell_parts(
    world: &InterventionWorld,
    dose: VariableId,
) -> Result<(&TemporalPolicy, HistoryWindow, &InitialLaw, u32, u64), ScientificTargetError> {
    let TimeKind::Temporal { horizon_steps, history, initial_law } = &world.time().kind else {
        return Err(grid("each cell is a temporal world"));
    };
    let ActionKind::Schedule { intervention, policy } = &world.action().kind else {
        return Err(grid("each cell is a scheduled dose assignment"));
    };
    let Intervention::Set { variable, value: Value::Float64(level) } = intervention else {
        return Err(grid("each cell hard-sets the dose"));
    };
    if *variable != dose {
        return Err(grid("each cell hard-sets the dose"));
    }
    let bits =
        if level.to_bits() == (-0.0_f64).to_bits() { 0.0_f64.to_bits() } else { level.to_bits() };
    Ok((policy, *history, initial_law, *horizon_steps, bits))
}

fn grid(reason: &'static str) -> ScientificTargetError {
    ScientificTargetError::NotADoseTimeGrid { reason }
}

/// A contrast is its form and its signed worlds in formula order. Caller-chosen world
/// labels and the reference flag are not written: the reference is the baseline term
/// of the formula, so relabelling the worlds is the same target.
impl CanonicalEncode for InterventionContrast {
    fn encode(&self, writer: &mut CanonicalWriter) {
        let form = match self.form {
            ContrastForm::Linear => {
                writer.word(3);
                let mut terms: Vec<_> = self
                    .terms()
                    .iter()
                    .map(|term| {
                        let mut item = CanonicalWriter::new("antecedent.linear_world_term.v1");
                        item.real(term.coefficient);
                        if let Some(world) = self.worlds.get(term.world) {
                            world.encode(&mut item);
                        }
                        item.finish()
                    })
                    .collect();
                terms.sort_unstable();
                writer.word(count(terms.len()));
                for term in terms {
                    writer.bytes(&term);
                }
                // Unlike the fixed difference/interaction forms, arbitrary coefficients do not
                // determine a unique reference. Its scientific content is part of this question.
                if let Some(reference) = self.worlds.get(self.reference) {
                    reference.encode(writer);
                }
                return;
            }
            ContrastForm::Difference { .. } => 1,
            ContrastForm::Interaction { .. } => 2,
        };
        writer.word(form);
        let terms = self.terms();
        writer.word(count(terms.len()));
        for term in terms {
            // Keep previous fixed-form identities byte-for-byte compatible.
            writer.word(if term.coefficient.is_sign_negative() { u64::MAX } else { 1 });
            if let Some(world) = self.worlds.get(term.world) {
                world.encode(writer);
            }
        }
    }
}

impl CanonicalEncode for DoseTimeSurface {
    fn encode(&self, writer: &mut CanonicalWriter) {
        self.dose.encode(writer);
        writer.word(count(self.level_bits.len()));
        for bits in self.level_bits.as_ref() {
            writer.word(*bits);
        }
        writer.word(count(self.horizons.len()));
        for horizon in self.horizons.as_ref() {
            writer.word(u64::from(*horizon));
        }
        writer.word(count(self.cells.len()));
        for id in self.cells.as_ref() {
            match self.worlds.get(*id) {
                Some(world) => {
                    writer.word(1);
                    world.encode(writer);
                }
                None => {
                    writer.word(0);
                }
            }
        }
    }
}
