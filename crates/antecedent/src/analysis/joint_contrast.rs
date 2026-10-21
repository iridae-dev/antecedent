//! One joint intervention contrast.
//!
//! The functional is `E[Y(a,b)] − E[Y(a,0)] − E[Y(0,b)] + E[Y(0,0)]`. `0` is the
//! flagged reference world. The four worlds share population, outcome, units,
//! conditioning, and reference regime, and they are accepted only by
//! [`InterventionContrast::interaction`]. Frequentist uncertainty is the quadratic
//! form of that contrast in the covariance of the four world means. Bayesian
//! uncertainty is one draw of the contrast per aligned posterior row. A mediation
//! target keeps its own premises. A saturation or Gaussian-panel response is not
//! estimated here.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![cfg_attr(
    not(test),
    allow(dead_code, reason = "the parent wires the module; until then only the tests call it")
)]

use antecedent_core::{
    AssumptionSet, Conditioning, ContrastForm, GraphFamily, HistoryWindow, InferenceSelection,
    InitialLaw, Intervention, InterventionContrast, InterventionWorld, ModelAssumptions,
    ObservationUnit, OutcomeQuantity, QuantityUnits, ScientificTarget, ScientificTargetError,
    ScientificWorldId, TargetForm, TargetPopulation, TemporalPolicy, TimeFrame, Value, VariableId,
    WorldAction, WorldBinding, WorldCollection,
};

/// Dose variable. In a temporal world this is the level factor.
const DOSE: u32 = 0;
/// Outcome whose mean the contrast takes.
const OUTCOME: u32 = 1;
/// Second static factor. Temporal factor B is a schedule, not this variable.
const STATIC_FACTOR_B: u32 = 2;

/// Flagged reference world `Y(0,0)`.
const REFERENCE_WORLD: ScientificWorldId = ScientificWorldId::from_raw(0);
/// World `Y(a,0)`.
const FACTOR_A_WORLD: ScientificWorldId = ScientificWorldId::from_raw(1);
/// World `Y(0,b)`.
const FACTOR_B_WORLD: ScientificWorldId = ScientificWorldId::from_raw(2);
/// World `Y(a,b)`.
const JOINT_WORLD: ScientificWorldId = ScientificWorldId::from_raw(3);

/// Term order of [`InterventionContrast::terms`] for an interaction.
const TERM_WORLDS: [ScientificWorldId; 4] =
    [JOINT_WORLD, FACTOR_A_WORLD, FACTOR_B_WORLD, REFERENCE_WORLD];

/// Declared reference level. The flag, not this number, marks the reference world.
const REFERENCE_LEVEL: f64 = 0.0_f64;
/// Level of a factor that has moved off the reference.
const ACTIVE_LEVEL: f64 = 1.0;

/// Steps after the policy origin.
const HORIZON_STEPS: u32 = 2;
/// Explicit history cap. This is not the planner default.
const HISTORY_LAG: u32 = 1;
/// Reference schedule: a pulse at the policy origin.
const REFERENCE_OFFSET: i32 = 0;
/// Last step of the active sustained schedule, inclusive and inside the horizon.
const ACTIVE_UNTIL: i32 = 1;

/// Nominal level of the normal interval. Coverage of that interval is unmeasured.
const INTERVAL_LEVEL: f64 = 0.95;
/// Nearest `f64` to the standard normal 0.975 quantile.
const NORMAL_975: f64 = 1.959_963_984_540_054;
/// Absolute slack for a quadratic form that is negative only by rounding.
const VARIANCE_SLACK: f64 = 1.0e-8;

/// Refusal to assemble or estimate the joint contrast.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub(crate) enum JointContrastError {
    /// The scientific-target constructors refused the worlds.
    #[error(transparent)]
    Target(#[from] ScientificTargetError),
    /// `E[Y(a)]` is not the four-world functional.
    #[error("a single-world response is not Delta_AB")]
    SingleWorld,
    /// `E[Y(a)] − E[Y(0)]` is not the four-world functional.
    #[error("a two-world difference is not Delta_AB")]
    TwoWorldDifference,
    /// Mediation is a different target and carries its own premises.
    #[error("mediation keeps its own cross-world premises")]
    Mediation,
    /// The object is not the interaction contrast.
    #[error("the target is not the joint intervention contrast")]
    NotTheJointContrast,
    /// The covariance of a mean needs at least two aligned units.
    #[error("cross-world covariance needs at least two aligned units")]
    TooFewUnits,
    /// Columns are missing, repeated, or of unequal length.
    #[error("world outcomes are not the four aligned contrast worlds")]
    UnalignedOutcomes,
    /// An outcome or a posterior coordinate is not finite.
    #[error("a world outcome is not finite")]
    NonFiniteOutcome,
    /// `c' Σ c` is not a finite variance.
    #[error("the cross-world quadratic form is not a variance")]
    NotACovariance,
    /// No joint posterior rows were supplied.
    #[error("aligned posterior draws are empty")]
    NoDraws,
    /// A row does not carry each contrast world once.
    #[error("a posterior row does not carry each contrast world once")]
    UnalignedDraw,
}

/// Outcomes of one world, aligned by unit with the other three worlds.
#[derive(Clone, Debug, PartialEq)]
#[must_use = "pass the four world columns into the contrast"]
pub(crate) struct WorldOutcomes {
    /// World these outcomes belong to.
    pub(crate) world: ScientificWorldId,
    /// One outcome per unit, in unit order.
    pub(crate) values: Vec<f64>,
}

/// One joint posterior draw of the four world means.
#[derive(Clone, Debug, PartialEq)]
#[must_use = "contract the row into one draw of Delta_AB"]
pub(crate) struct PosteriorDraw {
    /// Four world means from the same posterior draw.
    pub(crate) worlds: Vec<(ScientificWorldId, f64)>,
}

/// Frequentist contrast and its cross-world interval.
///
/// `covariance` is the covariance of the four world means in
/// [`InterventionContrast::terms`] order. `variance` is the quadratic form of the
/// contrast in that matrix. The interval is not the result of subtracting four
/// marginal intervals.
#[derive(Clone, Debug, PartialEq)]
#[must_use = "the interval is the quadratic form of the four worlds"]
pub(crate) struct FrequentistContrast {
    /// `c' μ`.
    pub(crate) point: f64,
    /// `c' Σ c`.
    pub(crate) variance: f64,
    /// Square root of [`Self::variance`].
    pub(crate) standard_error: f64,
    /// Nominal normal level. Coverage is unmeasured.
    pub(crate) level: f64,
    /// Lower normal endpoint.
    pub(crate) lower: f64,
    /// Upper normal endpoint.
    pub(crate) upper: f64,
    /// Covariance of the four world means, in term order.
    pub(crate) covariance: [[f64; 4]; 4],
}

/// Posterior of `Delta_AB` obtained by contracting each aligned row.
#[derive(Clone, Debug, PartialEq)]
#[must_use = "each draw is one joint posterior realisation of Delta_AB"]
pub(crate) struct AlignedPosterior {
    draws: Vec<f64>,
}

impl AlignedPosterior {
    /// Aligned draws, in the order the rows were supplied.
    #[must_use = "the draws are the posterior of Delta_AB"]
    pub(crate) fn draws(&self) -> &[f64] {
        &self.draws
    }

    /// Mean of the aligned draws.
    #[must_use = "the posterior mean is the mean of the aligned draws"]
    pub(crate) fn mean(&self) -> f64 {
        let total: f64 = self.draws.iter().copied().sum();
        total / self.draws.len() as f64
    }
}

struct SignedWorlds {
    ids: [ScientificWorldId; 4],
    coefficients: [f64; 4],
}

/// Static interaction at the declared reference level.
///
/// # Errors
///
/// [`JointContrastError::Target`] when the four worlds are not the interaction.
#[must_use = "the target is the joint contrast, not a single world"]
pub(crate) fn static_target(
    inference: InferenceSelection,
) -> Result<ScientificTarget, JointContrastError> {
    static_cell(
        inference,
        (REFERENCE_LEVEL, REFERENCE_LEVEL),
        (ACTIVE_LEVEL, REFERENCE_LEVEL),
        (REFERENCE_LEVEL, ACTIVE_LEVEL),
        (ACTIVE_LEVEL, ACTIVE_LEVEL),
    )
}

/// Temporal interaction of a dose and a schedule.
///
/// The reference regime is a pulse at offset 0, dose [`REFERENCE_LEVEL`], horizon
/// [`HORIZON_STEPS`], history lag [`HISTORY_LAG`], and the contemporaneous observed
/// initial law. Factor A moves only the dose. `schedule_b` is factor B and the
/// joint world's schedule. A pulse or window outside the horizon is refused, as is
/// a schedule that does not move off the reference pulse.
///
/// # Errors
///
/// [`JointContrastError::Target`] when the schedule leaves the horizon or the four
/// worlds are not the interaction.
#[must_use = "the target is the scheduled contrast, not a single program"]
pub(crate) fn temporal_target(
    inference: InferenceSelection,
    schedule_b: TemporalPolicy,
) -> Result<ScientificTarget, JointContrastError> {
    let binding = shared_binding("series", Conditioning::unconditional())?;
    let time = declared_time()?;
    let reference_policy = TemporalPolicy::pulse(REFERENCE_OFFSET);
    let worlds = WorldCollection::new([
        world(
            REFERENCE_WORLD,
            true,
            schedule_action(REFERENCE_LEVEL, reference_policy.clone())?,
            binding.clone(),
            time.clone(),
        )?,
        world(
            FACTOR_A_WORLD,
            false,
            schedule_action(ACTIVE_LEVEL, reference_policy)?,
            binding.clone(),
            time.clone(),
        )?,
        world(
            FACTOR_B_WORLD,
            false,
            schedule_action(REFERENCE_LEVEL, schedule_b.clone())?,
            binding.clone(),
            time.clone(),
        )?,
        world(JOINT_WORLD, false, schedule_action(ACTIVE_LEVEL, schedule_b)?, binding, time)?,
    ])?;
    package(worlds, GraphFamily::TemporalDag, inference)
}

/// One contrast, with the covariance of the four world means.
///
/// # Errors
///
/// [`JointContrastError::SingleWorld`], [`JointContrastError::TwoWorldDifference`],
/// [`JointContrastError::Mediation`], or [`JointContrastError::NotTheJointContrast`]
/// when `target` is not the interaction. [`JointContrastError::UnalignedOutcomes`],
/// [`JointContrastError::TooFewUnits`], [`JointContrastError::NonFiniteOutcome`], or
/// [`JointContrastError::NotACovariance`] when the columns do not support the
/// quadratic form.
#[must_use = "read the cross-world interval before discarding the contrast"]
pub(crate) fn frequentist_contrast(
    target: &ScientificTarget,
    worlds: &[WorldOutcomes],
) -> Result<FrequentistContrast, JointContrastError> {
    let signed = signed_worlds(target)?;
    let rows = align_units(&signed.ids, worlds)?;
    let (means, covariance) = moments(&rows)?;
    let point = dot(signed.coefficients, means);
    let variance = variance_or_refuse(quadratic(signed.coefficients, covariance))?;
    let standard_error = variance.sqrt();
    let half_width = NORMAL_975 * standard_error;
    Ok(FrequentistContrast {
        point,
        variance,
        standard_error,
        level: INTERVAL_LEVEL,
        lower: point - half_width,
        upper: point + half_width,
        covariance,
    })
}

/// Contract each aligned posterior row into one draw of `Delta_AB`.
///
/// Rows are not four independent posterior samples. Draw `s` uses only row `s`.
///
/// # Errors
///
/// The target errors of [`frequentist_contrast`], plus [`JointContrastError::NoDraws`]
/// and [`JointContrastError::UnalignedDraw`].
#[must_use = "the draws are the posterior of the joint contrast"]
pub(crate) fn aligned_posterior(
    target: &ScientificTarget,
    draws: &[PosteriorDraw],
) -> Result<AlignedPosterior, JointContrastError> {
    let signed = signed_worlds(target)?;
    if draws.is_empty() {
        return Err(JointContrastError::NoDraws);
    }
    let mut contracted = Vec::with_capacity(draws.len());
    for draw in draws {
        contracted.push(contrast_of_row(&signed, &draw.worlds)?);
    }
    Ok(AlignedPosterior { draws: contracted })
}

fn static_cell(
    inference: InferenceSelection,
    reference: (f64, f64),
    factor_a: (f64, f64),
    factor_b: (f64, f64),
    joint: (f64, f64),
) -> Result<ScientificTarget, JointContrastError> {
    let binding = shared_binding("subject", Conditioning::unconditional())?;
    let time = TimeFrame::static_decision();
    let worlds = WorldCollection::new([
        world(
            REFERENCE_WORLD,
            true,
            static_action(reference.0, reference.1)?,
            binding.clone(),
            time.clone(),
        )?,
        world(
            FACTOR_A_WORLD,
            false,
            static_action(factor_a.0, factor_a.1)?,
            binding.clone(),
            time.clone(),
        )?,
        world(
            FACTOR_B_WORLD,
            false,
            static_action(factor_b.0, factor_b.1)?,
            binding.clone(),
            time.clone(),
        )?,
        world(JOINT_WORLD, false, static_action(joint.0, joint.1)?, binding, time)?,
    ])?;
    package(worlds, GraphFamily::Dag, inference)
}

fn package(
    worlds: WorldCollection,
    graph: GraphFamily,
    inference: InferenceSelection,
) -> Result<ScientificTarget, JointContrastError> {
    let contrast = InterventionContrast::interaction(
        worlds,
        JOINT_WORLD,
        FACTOR_A_WORLD,
        FACTOR_B_WORLD,
        REFERENCE_WORLD,
    )?;
    Ok(ScientificTarget::new(
        TargetForm::Contrast(contrast),
        ModelAssumptions::new(graph, AssumptionSet::new()),
        inference,
    ))
}

fn shared_binding(
    unit: &str,
    conditioning: Conditioning,
) -> Result<WorldBinding, ScientificTargetError> {
    WorldBinding::new(
        TargetPopulation::AllObserved,
        ObservationUnit::new(unit)?,
        OutcomeQuantity::mean(VariableId::from_raw(OUTCOME), QuantityUnits::dimensionless())?,
        conditioning,
    )
}

fn declared_time() -> Result<TimeFrame, ScientificTargetError> {
    TimeFrame::temporal(
        HORIZON_STEPS,
        HistoryWindow::max_lag(HISTORY_LAG),
        InitialLaw::contemporaneous_observed(),
    )
}

fn static_action(factor_a: f64, factor_b: f64) -> Result<WorldAction, ScientificTargetError> {
    WorldAction::at_decision([
        Intervention::set(VariableId::from_raw(DOSE), Value::f64(factor_a)),
        Intervention::set(VariableId::from_raw(STATIC_FACTOR_B), Value::f64(factor_b)),
    ])
}

fn schedule_action(
    dose: f64,
    policy: TemporalPolicy,
) -> Result<WorldAction, ScientificTargetError> {
    let dose = Intervention::set(VariableId::from_raw(DOSE), Value::f64(dose));
    WorldAction::schedule(dose, policy)
}

fn world(
    id: ScientificWorldId,
    reference: bool,
    action: WorldAction,
    binding: WorldBinding,
    time: TimeFrame,
) -> Result<InterventionWorld, ScientificTargetError> {
    InterventionWorld::new(id, action, reference, binding, time)
}

#[allow(
    clippy::wildcard_enum_match_arm,
    reason = "a world bundle or dose-time grid is not the joint contrast"
)]
fn signed_worlds(target: &ScientificTarget) -> Result<SignedWorlds, JointContrastError> {
    match target.form() {
        TargetForm::World(_) => Err(JointContrastError::SingleWorld),
        TargetForm::Mediation(_) => Err(JointContrastError::Mediation),
        TargetForm::Contrast(contrast) => match contrast.form() {
            ContrastForm::Difference { .. } => Err(JointContrastError::TwoWorldDifference),
            ContrastForm::Interaction { .. } => coefficients_of(contrast),
            _ => Err(JointContrastError::NotTheJointContrast),
        },
        _ => Err(JointContrastError::NotTheJointContrast),
    }
}

fn coefficients_of(contrast: &InterventionContrast) -> Result<SignedWorlds, JointContrastError> {
    let terms = contrast.terms();
    if terms.len() != TERM_WORLDS.len() {
        return Err(JointContrastError::NotTheJointContrast);
    }
    let mut ids = TERM_WORLDS;
    let mut coefficients = [0.0_f64; 4];
    for (index, term) in terms.iter().enumerate() {
        ids[index] = term.world;
        if term.coefficient.to_bits() != 1.0_f64.to_bits()
            && term.coefficient.to_bits() != (-1.0_f64).to_bits()
        {
            return Err(JointContrastError::NotTheJointContrast);
        }
        coefficients[index] = term.coefficient;
    }
    Ok(SignedWorlds { ids, coefficients })
}

fn align_units(
    ids: &[ScientificWorldId; 4],
    worlds: &[WorldOutcomes],
) -> Result<Vec<[f64; 4]>, JointContrastError> {
    if worlds.len() == 1 {
        return Err(JointContrastError::SingleWorld);
    }
    if worlds.len() != ids.len() {
        return Err(JointContrastError::UnalignedOutcomes);
    }
    let mut columns: [Option<&[f64]>; 4] = [None; 4];
    for world in worlds {
        let Some(index) = ids.iter().position(|id| *id == world.world) else {
            return Err(JointContrastError::UnalignedOutcomes);
        };
        if columns[index].is_some() {
            return Err(JointContrastError::UnalignedOutcomes);
        }
        columns[index] = Some(world.values.as_slice());
    }
    let mut slices: [&[f64]; 4] = [&[], &[], &[], &[]];
    for (index, column) in columns.iter().enumerate() {
        let Some(slice) = column else {
            return Err(JointContrastError::UnalignedOutcomes);
        };
        slices[index] = slice;
    }
    let n = slices[0].len();
    if slices.iter().any(|slice| slice.len() != n) {
        return Err(JointContrastError::UnalignedOutcomes);
    }
    if n < 2 {
        return Err(JointContrastError::TooFewUnits);
    }
    let mut rows = vec![[0.0_f64; 4]; n];
    for (column, slice) in slices.iter().enumerate() {
        for (row, value) in rows.iter_mut().zip(*slice) {
            if !value.is_finite() {
                return Err(JointContrastError::NonFiniteOutcome);
            }
            row[column] = *value;
        }
    }
    Ok(rows)
}

fn moments(rows: &[[f64; 4]]) -> Result<([f64; 4], [[f64; 4]; 4]), JointContrastError> {
    let n = rows.len();
    if n < 2 {
        return Err(JointContrastError::TooFewUnits);
    }
    let scale = n as f64;
    let mut means = [0.0_f64; 4];
    for row in rows {
        for (mean, value) in means.iter_mut().zip(row) {
            *mean += *value;
        }
    }
    for mean in &mut means {
        *mean /= scale;
    }
    let mut scatter = [[0.0_f64; 4]; 4];
    for row in rows {
        let deviation: [f64; 4] = std::array::from_fn(|index| row[index] - means[index]);
        for (left_index, left) in deviation.iter().enumerate() {
            for (right_index, right) in deviation.iter().enumerate().skip(left_index) {
                let product = left * right;
                scatter[left_index][right_index] += product;
                if left_index != right_index {
                    scatter[right_index][left_index] += product;
                }
            }
        }
    }
    let divisor = scale * (n - 1) as f64;
    let mut covariance = [[0.0_f64; 4]; 4];
    for (left_index, row) in scatter.iter().enumerate() {
        for (right_index, product) in row.iter().enumerate() {
            covariance[left_index][right_index] = product / divisor;
        }
    }
    Ok((means, covariance))
}

fn dot(coefficients: [f64; 4], means: [f64; 4]) -> f64 {
    coefficients.iter().zip(means).map(|(coef, mean)| coef * mean).sum()
}

fn quadratic(coefficients: [f64; 4], covariance: [[f64; 4]; 4]) -> f64 {
    let mut total = 0.0_f64;
    for (coef_row, row) in coefficients.iter().zip(covariance) {
        let inner: f64 = coefficients.iter().zip(row).map(|(coef, entry)| entry * coef).sum();
        total += coef_row * inner;
    }
    total
}

#[allow(clippy::float_cmp, reason = "a negative quadratic form is not a variance")]
fn variance_or_refuse(raw: f64) -> Result<f64, JointContrastError> {
    if !raw.is_finite() {
        return Err(JointContrastError::NotACovariance);
    }
    if raw < 0.0_f64 {
        if raw >= -VARIANCE_SLACK {
            return Ok(0.0_f64);
        }
        return Err(JointContrastError::NotACovariance);
    }
    Ok(raw)
}

fn contrast_of_row(
    signed: &SignedWorlds,
    row: &[(ScientificWorldId, f64)],
) -> Result<f64, JointContrastError> {
    if row.len() == 1 {
        return Err(JointContrastError::SingleWorld);
    }
    if row.len() != signed.ids.len() {
        return Err(JointContrastError::UnalignedDraw);
    }
    let mut seen = [false; 4];
    let mut total = 0.0_f64;
    for (world, outcome) in row {
        if !outcome.is_finite() {
            return Err(JointContrastError::NonFiniteOutcome);
        }
        let Some(index) = signed.ids.iter().position(|id| id == world) else {
            return Err(JointContrastError::UnalignedDraw);
        };
        if seen[index] {
            return Err(JointContrastError::UnalignedDraw);
        }
        seen[index] = true;
        total += signed.coefficients[index] * outcome;
    }
    Ok(total)
}

#[cfg(test)]
mod tests {
    use super::*;
    use antecedent_core::{
        CrossWorldPremise, MediationArms, MediationComponent, MediationTarget, OutcomeFunctional,
    };

    fn bits(value: f64) -> u64 {
        value.to_bits()
    }

    #[allow(clippy::float_cmp, reason = "the tolerance is the comparison")]
    fn close(left: f64, right: f64) {
        let scale = left.abs().max(right.abs()).max(1.0);
        assert!((left - right).abs() <= scale * 1.0e-12, "{left} vs {right}");
    }

    #[allow(
        clippy::float_cmp,
        reason = "the subtracted endpoint width must exceed the quadratic-form width"
    )]
    fn assert_wider(endpoint: f64, quadratic: f64) {
        assert!(endpoint > quadratic, "{endpoint} vs {quadratic}");
    }

    fn columns(rows: &[[f64; 4]]) -> Vec<WorldOutcomes> {
        TERM_WORLDS
            .into_iter()
            .enumerate()
            .map(|(index, world)| WorldOutcomes {
                world,
                values: rows.iter().map(|row| row[index]).collect(),
            })
            .collect()
    }

    fn shared_shock(means: [f64; 4]) -> Vec<[f64; 4]> {
        [-1.0, 0.0_f64, 1.0]
            .into_iter()
            .map(|shock| std::array::from_fn(|index| means[index] + shock))
            .collect()
    }

    fn endpoint_half_width(covariance: [[f64; 4]; 4]) -> f64 {
        let sum_se: f64 = (0..4).map(|index| covariance[index][index].sqrt()).sum();
        NORMAL_975 * sum_se
    }

    fn column_span(rows: &[[f64; 4]], index: usize) -> (f64, f64) {
        let mut min = rows[0][index];
        let mut max = rows[0][index];
        for row in rows {
            min = min.min(row[index]);
            max = max.max(row[index]);
        }
        (min, max)
    }

    fn subtracted_bounds(rows: &[[f64; 4]]) -> (f64, f64) {
        let mut lower = 0.0_f64;
        let mut upper = 0.0_f64;
        for index in [0, 3] {
            let (min, max) = column_span(rows, index);
            lower += min;
            upper += max;
        }
        for index in [1, 2] {
            let (min, max) = column_span(rows, index);
            lower -= max;
            upper -= min;
        }
        (lower, upper)
    }

    fn active_schedule() -> TemporalPolicy {
        TemporalPolicy::sustained(REFERENCE_OFFSET, ACTIVE_UNTIL)
    }

    #[test]
    fn static_worlds_share_population_outcome_units_conditioning_and_reference() {
        let target = static_target(InferenceSelection::Frequentist).unwrap();
        let TargetForm::Contrast(contrast) = target.form() else {
            panic!("interaction contrast");
        };
        assert!(matches!(contrast.form(), ContrastForm::Interaction { .. }));
        assert_eq!(contrast.reference(), REFERENCE_WORLD);
        assert_eq!(
            contrast.terms().iter().map(|term| term.coefficient.to_bits()).collect::<Vec<_>>(),
            [1.0_f64, -1.0, -1.0, 1.0].map(f64::to_bits).to_vec()
        );
        let worlds = contrast.worlds().worlds();
        let references: Vec<_> = worlds.iter().filter(|world| world.is_reference()).collect();
        assert_eq!(references.len(), 1);
        assert_eq!(references[0].id(), REFERENCE_WORLD);
        let binding = worlds[0].binding();
        assert_eq!(binding.population(), &TargetPopulation::AllObserved);
        assert_eq!(binding.unit().name(), "subject");
        assert_eq!(binding.outcome().outcome(), VariableId::from_raw(OUTCOME));
        assert_eq!(binding.outcome().functional(), &OutcomeFunctional::Mean);
        assert_eq!(binding.outcome().units().name(), "1");
        assert!(binding.conditioning().variables().is_empty());
        assert!(worlds.iter().all(|world| world.binding() == binding && world.time().is_static()));
        assert_eq!(target.assumptions().graph(), GraphFamily::Dag);
    }

    #[test]
    fn the_four_worlds_must_realise_the_declared_reference_contrast() {
        let incoherent = static_cell(
            InferenceSelection::Frequentist,
            (REFERENCE_LEVEL, REFERENCE_LEVEL),
            (ACTIVE_LEVEL, REFERENCE_LEVEL),
            (REFERENCE_LEVEL, ACTIVE_LEVEL),
            (2.0, ACTIVE_LEVEL),
        )
        .unwrap_err();
        assert!(matches!(
            incoherent,
            JointContrastError::Target(ScientificTargetError::IncoherentContrast { .. })
        ));

        let binding = shared_binding("subject", Conditioning::unconditional()).unwrap();
        let time = TimeFrame::static_decision();
        let unmarked = WorldCollection::new([
            world(
                REFERENCE_WORLD,
                false,
                static_action(REFERENCE_LEVEL, REFERENCE_LEVEL).unwrap(),
                binding.clone(),
                time.clone(),
            )
            .unwrap(),
            world(
                FACTOR_A_WORLD,
                false,
                static_action(ACTIVE_LEVEL, REFERENCE_LEVEL).unwrap(),
                binding.clone(),
                time.clone(),
            )
            .unwrap(),
            world(
                FACTOR_B_WORLD,
                false,
                static_action(REFERENCE_LEVEL, ACTIVE_LEVEL).unwrap(),
                binding.clone(),
                time.clone(),
            )
            .unwrap(),
            world(
                JOINT_WORLD,
                false,
                static_action(ACTIVE_LEVEL, ACTIVE_LEVEL).unwrap(),
                binding,
                time,
            )
            .unwrap(),
        ])
        .unwrap();
        let unmarked = InterventionContrast::interaction(
            unmarked,
            JOINT_WORLD,
            FACTOR_A_WORLD,
            FACTOR_B_WORLD,
            REFERENCE_WORLD,
        )
        .unwrap_err();
        assert_eq!(unmarked, ScientificTargetError::ReferenceNotMarked { id: REFERENCE_WORLD });

        let shared = shared_binding("subject", Conditioning::unconditional()).unwrap();
        let conditioned =
            shared_binding("subject", Conditioning::on([VariableId::from_raw(3)])).unwrap();
        let time = TimeFrame::static_decision();
        let mismatch = WorldCollection::new([
            world(
                REFERENCE_WORLD,
                true,
                static_action(REFERENCE_LEVEL, REFERENCE_LEVEL).unwrap(),
                conditioned,
                time.clone(),
            )
            .unwrap(),
            world(
                FACTOR_A_WORLD,
                false,
                static_action(ACTIVE_LEVEL, REFERENCE_LEVEL).unwrap(),
                shared,
                time,
            )
            .unwrap(),
        ])
        .unwrap_err();
        assert_eq!(mismatch, ScientificTargetError::IncompatibleBinding { field: "conditioning" });

        let unmoved =
            temporal_target(InferenceSelection::Frequentist, TemporalPolicy::pulse(0)).unwrap_err();
        assert!(matches!(
            unmoved,
            JointContrastError::Target(ScientificTargetError::IncoherentContrast { .. })
        ));
    }

    #[test]
    fn a_single_world_is_not_the_contrast() {
        let binding = shared_binding("subject", Conditioning::unconditional()).unwrap();
        let time = TimeFrame::static_decision();
        let one = world(
            REFERENCE_WORLD,
            true,
            static_action(REFERENCE_LEVEL, REFERENCE_LEVEL).unwrap(),
            binding.clone(),
            time.clone(),
        )
        .unwrap();
        let single = ScientificTarget::new(
            TargetForm::World(one),
            ModelAssumptions::new(GraphFamily::Dag, AssumptionSet::new()),
            InferenceSelection::Frequentist,
        );
        let rows = shared_shock([6.0, 3.0, 1.0, 2.0]);
        let err = frequentist_contrast(&single, &columns(&rows)).unwrap_err();
        assert!(matches!(err, JointContrastError::SingleWorld));

        let interaction = static_target(InferenceSelection::Frequentist).unwrap();
        let one_column = [columns(&rows).remove(0)];
        let err = frequentist_contrast(&interaction, &one_column).unwrap_err();
        assert!(matches!(err, JointContrastError::SingleWorld));
        let err =
            aligned_posterior(&interaction, &[PosteriorDraw { worlds: vec![(JOINT_WORLD, 1.0)] }])
                .unwrap_err();
        assert!(matches!(err, JointContrastError::SingleWorld));

        let other = world(
            FACTOR_A_WORLD,
            false,
            static_action(ACTIVE_LEVEL, REFERENCE_LEVEL).unwrap(),
            binding,
            time,
        )
        .unwrap();
        let reference = world(
            REFERENCE_WORLD,
            true,
            static_action(REFERENCE_LEVEL, REFERENCE_LEVEL).unwrap(),
            shared_binding("subject", Conditioning::unconditional()).unwrap(),
            TimeFrame::static_decision(),
        )
        .unwrap();
        let worlds = WorldCollection::new([reference, other]).unwrap();
        let difference = InterventionContrast::difference(
            worlds,
            FACTOR_A_WORLD,
            REFERENCE_WORLD,
            REFERENCE_WORLD,
        )
        .unwrap();
        let difference = ScientificTarget::new(
            TargetForm::Contrast(difference),
            ModelAssumptions::new(GraphFamily::Dag, AssumptionSet::new()),
            InferenceSelection::Frequentist,
        );
        let err = frequentist_contrast(&difference, &columns(&rows)).unwrap_err();
        assert!(matches!(err, JointContrastError::TwoWorldDifference));
    }

    #[test]
    fn mediation_keeps_its_own_premises() {
        let binding = shared_binding("subject", Conditioning::unconditional()).unwrap();
        let arms = MediationArms::new(
            VariableId::from_raw(DOSE),
            [VariableId::from_raw(STATIC_FACTOR_B)],
            Intervention::set(VariableId::from_raw(DOSE), Value::f64(REFERENCE_LEVEL)),
            Intervention::set(VariableId::from_raw(DOSE), Value::f64(ACTIVE_LEVEL)),
            MediationComponent::NaturalDirect,
        )
        .unwrap();
        let mediation = MediationTarget::new(
            binding,
            TimeFrame::static_decision(),
            None,
            arms,
            [
                CrossWorldPremise::Composition,
                CrossWorldPremise::CrossWorldIndependence,
                CrossWorldPremise::PathSpecificNested,
            ],
        )
        .unwrap();
        let target = ScientificTarget::new(
            TargetForm::Mediation(mediation),
            ModelAssumptions::new(GraphFamily::Dag, AssumptionSet::new()),
            InferenceSelection::Bayesian,
        );
        let rows = shared_shock([6.0, 3.0, 1.0, 2.0]);
        assert!(matches!(
            frequentist_contrast(&target, &columns(&rows)).unwrap_err(),
            JointContrastError::Mediation
        ));
        let draws = rows
            .iter()
            .map(|row| PosteriorDraw { worlds: TERM_WORLDS.into_iter().zip(*row).collect() })
            .collect::<Vec<_>>();
        assert!(matches!(
            aligned_posterior(&target, &draws).unwrap_err(),
            JointContrastError::Mediation
        ));
    }

    #[test]
    fn shared_shock_cancels_in_one_cross_world_quadratic_form() {
        let target = static_target(InferenceSelection::Frequentist).unwrap();
        let means = [6.0, 3.0, 1.0, 2.0];
        let rows = shared_shock(means);
        let mut permuted = columns(&rows);
        permuted.rotate_left(1);
        let fit = frequentist_contrast(&target, &permuted).unwrap();
        assert_eq!(bits(fit.point), bits(4.0));
        assert_eq!(bits(fit.variance), bits(0.0_f64));
        assert_eq!(bits(fit.standard_error), bits(0.0_f64));
        assert_eq!(bits(fit.lower), bits(fit.point));
        assert_eq!(bits(fit.upper), bits(fit.point));
        assert_eq!(bits(fit.level), bits(INTERVAL_LEVEL));
        assert_ne!(bits(fit.covariance[0][1]), bits(0.0_f64));
        assert_wider(endpoint_half_width(fit.covariance), fit.standard_error);

        let one_unit = columns(&rows[..1]);
        assert!(matches!(
            frequentist_contrast(&target, &one_unit).unwrap_err(),
            JointContrastError::TooFewUnits
        ));
        let mut non_finite = columns(&rows);
        non_finite[0].values[0] = f64::NAN;
        assert!(matches!(
            frequentist_contrast(&target, &non_finite).unwrap_err(),
            JointContrastError::NonFiniteOutcome
        ));
    }

    #[test]
    fn independent_world_noise_is_still_one_quadratic_form() {
        let target = static_target(InferenceSelection::Frequentist).unwrap();
        let means = [6.0, 3.0, 1.0, 2.0];
        let mut rows = Vec::new();
        for world in 0..4 {
            for sign in [1.0, -1.0] {
                let mut row = means;
                row[world] += sign;
                rows.push(row);
            }
        }
        let fit = frequentist_contrast(&target, &columns(&rows)).unwrap();
        assert_eq!(bits(fit.point), bits(4.0));
        close(fit.variance, 1.0 / 7.0);
        for left in 0..4 {
            for right in 0..4 {
                if left != right {
                    assert_eq!(bits(fit.covariance[left][right]), bits(0.0_f64));
                }
            }
        }
        let marginal: f64 = (0..4).map(|index| fit.covariance[index][index]).sum();
        close(fit.variance, marginal);
        assert_wider(endpoint_half_width(fit.covariance), fit.standard_error);
        close(fit.upper - fit.lower, 2.0 * NORMAL_975 * fit.standard_error);
    }

    #[test]
    fn aligned_posterior_draws_are_not_four_marginal_ranges() {
        let target = static_target(InferenceSelection::Bayesian).unwrap();
        let rows = shared_shock([6.0, 3.0, 1.0, 2.0]);
        let draws = rows
            .iter()
            .map(|row| PosteriorDraw { worlds: TERM_WORLDS.into_iter().zip(*row).collect() })
            .collect::<Vec<_>>();
        let posterior = aligned_posterior(&target, &draws).unwrap();
        assert_eq!(posterior.draws().len(), rows.len());
        for (draw, row) in posterior.draws().iter().zip(&rows) {
            let by_row = row[0] - row[1] - row[2] + row[3];
            assert_eq!(bits(*draw), bits(by_row));
            assert_eq!(bits(*draw), bits(4.0));
        }
        assert_eq!(bits(posterior.mean()), bits(4.0));
        let crossed = rows[0][0] - rows[1][1] - rows[1][2] + rows[1][3];
        assert_ne!(bits(crossed), bits(4.0));
        let (lower, upper) = subtracted_bounds(&rows);
        assert_wider(upper - lower, 0.0_f64);
        assert_ne!(bits(lower), bits(4.0));
        assert_ne!(bits(upper), bits(4.0));

        assert!(matches!(
            aligned_posterior(&target, &[]).unwrap_err(),
            JointContrastError::NoDraws
        ));
        let mut repeated = draws[0].clone();
        repeated.worlds[1].0 = JOINT_WORLD;
        assert!(matches!(
            aligned_posterior(&target, &[repeated]).unwrap_err(),
            JointContrastError::UnalignedDraw
        ));
    }

    #[test]
    fn temporal_schedules_declare_timing_horizon_and_initial_law() {
        let target = temporal_target(InferenceSelection::Frequentist, active_schedule()).unwrap();
        let TargetForm::Contrast(contrast) = target.form() else {
            panic!("interaction contrast");
        };
        assert_eq!(target.assumptions().graph(), GraphFamily::TemporalDag);
        let binding = contrast.worlds().worlds()[0].binding();
        assert_eq!(binding.unit().name(), "series");
        assert_eq!(binding.population(), &TargetPopulation::AllObserved);
        assert_eq!(binding.outcome().functional(), &OutcomeFunctional::Mean);
        assert!(binding.conditioning().variables().is_empty());
        for world in contrast.worlds().worlds() {
            let time = world.time();
            assert_eq!(time.horizon_steps(), Some(HORIZON_STEPS));
            assert_eq!(time.history().unwrap().max_lag_steps(), Some(HISTORY_LAG));
            assert!(time.initial_law().unwrap().name().is_none());
            let (_, policy) = world.action().schedule_parts().unwrap();
            let offsets = policy.active_offsets().unwrap();
            let bound = i64::from(HORIZON_STEPS);
            assert!(offsets.iter().all(|step| (-bound..=bound).contains(&i64::from(*step))));
        }
        let reference = contrast.worlds().get(REFERENCE_WORLD).unwrap();
        let factor_a = contrast.worlds().get(FACTOR_A_WORLD).unwrap();
        let factor_b = contrast.worlds().get(FACTOR_B_WORLD).unwrap();
        let joint = contrast.worlds().get(JOINT_WORLD).unwrap();
        assert!(reference.is_reference());
        assert!(!factor_a.is_reference());
        assert!(matches!(
            reference.action().schedule_parts().unwrap().1,
            TemporalPolicy::Pulse { at: REFERENCE_OFFSET }
        ));
        assert!(matches!(
            factor_a.action().schedule_parts().unwrap().1,
            TemporalPolicy::Pulse { at: REFERENCE_OFFSET }
        ));
        assert!(matches!(
            factor_b.action().schedule_parts().unwrap().1,
            TemporalPolicy::Sustained { from: REFERENCE_OFFSET, until: ACTIVE_UNTIL }
        ));
        assert!(matches!(
            joint.action().schedule_parts().unwrap().1,
            TemporalPolicy::Sustained { from: REFERENCE_OFFSET, until: ACTIVE_UNTIL }
        ));
        assert_eq!(bits(hard_dose(reference.action())), bits(REFERENCE_LEVEL));
        assert_eq!(bits(hard_dose(factor_b.action())), bits(REFERENCE_LEVEL));
        assert_eq!(bits(hard_dose(factor_a.action())), bits(ACTIVE_LEVEL));
        assert_eq!(bits(hard_dose(joint.action())), bits(ACTIVE_LEVEL));
    }

    #[test]
    fn a_pulse_outside_the_horizon_is_refused() {
        for policy in [
            TemporalPolicy::pulse(3),
            TemporalPolicy::pulse(-3),
            TemporalPolicy::sustained(REFERENCE_OFFSET, 3),
        ] {
            let err = temporal_target(InferenceSelection::Frequentist, policy).unwrap_err();
            assert!(matches!(
                err,
                JointContrastError::Target(ScientificTargetError::PolicyOutsideHorizon {
                    horizon_steps: HORIZON_STEPS,
                    ..
                })
            ));
        }
    }

    fn hard_dose(action: &WorldAction) -> f64 {
        let (intervention, _) = action.schedule_parts().expect("schedule");
        match intervention {
            Intervention::Set { value, .. } => value.as_f64().expect("dose"),
            _ => panic!("temporal factor is a hard dose"),
        }
    }

    fn program_mean(dose: f64, steps: f64) -> f64 {
        // Main effect of dose, main effect per active step, and their product.
        0.5 * dose + 0.25 * steps + 1.5 * dose * steps
    }

    fn program_means(target: &ScientificTarget) -> [f64; 4] {
        let TargetForm::Contrast(contrast) = target.form() else {
            panic!("interaction contrast");
        };
        let mut means = [0.0_f64; 4];
        for (index, term) in contrast.terms().iter().enumerate() {
            let world = contrast.worlds().get(term.world).unwrap();
            let (_, policy) = world.action().schedule_parts().unwrap();
            let steps = policy.active_offsets().unwrap().len() as f64;
            means[index] = program_mean(hard_dose(world.action()), steps);
        }
        means
    }

    #[test]
    fn temporal_program_contrast_matches_the_analytic_interaction() {
        let target = temporal_target(InferenceSelection::Frequentist, active_schedule()).unwrap();
        let means = program_means(&target);
        let expected = means[0] - means[1] - means[2] + means[3];
        assert_eq!(bits(expected), bits(1.5));
        let fit = frequentist_contrast(&target, &columns(&shared_shock(means))).unwrap();
        assert_eq!(bits(fit.point), bits(expected));
        assert_eq!(bits(fit.variance), bits(0.0_f64));
        assert_wider(endpoint_half_width(fit.covariance), fit.standard_error);
        assert!(matches!(target.form(), TargetForm::Contrast(_)));
        let TargetForm::Contrast(contrast) = target.form() else {
            panic!("interaction contrast");
        };
        assert_eq!(contrast.worlds().worlds()[0].time().horizon_steps(), Some(HORIZON_STEPS));
    }

    #[test]
    fn temporal_aligned_draws_follow_each_joint_row() {
        let target = temporal_target(InferenceSelection::Bayesian, active_schedule()).unwrap();
        let means = program_means(&target);
        let rows = shared_shock(means);
        let draws = rows
            .iter()
            .map(|row| PosteriorDraw { worlds: TERM_WORLDS.into_iter().zip(*row).collect() })
            .collect::<Vec<_>>();
        let posterior = aligned_posterior(&target, &draws).unwrap();
        for (draw, row) in posterior.draws().iter().zip(&rows) {
            assert_eq!(bits(*draw), bits(row[0] - row[1] - row[2] + row[3]));
            assert_eq!(bits(*draw), bits(1.5));
        }
        let crossed = rows[0][0] - rows[1][1] - rows[1][2] + rows[1][3];
        assert_ne!(bits(crossed), bits(posterior.mean()));
        let (lower, upper) = subtracted_bounds(&rows);
        assert_wider(upper - lower, 0.0_f64);
    }

    #[test]
    fn inference_label_does_not_change_the_scientific_identity() {
        let frequentist = static_target(InferenceSelection::Frequentist).unwrap();
        let bayesian = static_target(InferenceSelection::Bayesian).unwrap();
        assert_eq!(frequentist.identity(), bayesian.identity());
        assert_eq!(frequentist, bayesian);
        assert_ne!(frequentist.inference(), bayesian.inference());
        let temporal = temporal_target(InferenceSelection::Frequentist, active_schedule()).unwrap();
        let temporal_bayes =
            temporal_target(InferenceSelection::Bayesian, active_schedule()).unwrap();
        assert_eq!(temporal.identity(), temporal_bayes.identity());
        assert_ne!(frequentist.identity(), temporal.identity());
    }
}
