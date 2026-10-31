//! Licensed temporal intervention plans.

use std::sync::Arc;

use crate::error::EstimationError;
use crate::temporal_sequential::{SequentialMechanismOverlay, SequentialNodeOverlay};
use antecedent_core::{
    Intervention, InterventionSequence, MechanismOverride, ResponseFunctional, ResponseQuery,
    TemporalResponseSpec, Value, VariableId,
};

use super::surface::SequenceLeaf;

/// Licensed temporal intervention shape. The containing plan owns its lease.
#[derive(Clone, Debug, PartialEq)]
pub enum TemporalInterventionKind {
    /// One direct intervention, whose schedule is borrowed from the response spec.
    Single {
        /// Intervened variable.
        treatment: VariableId,
        /// Hard set or constant soft intervention level.
        level: Option<f64>,
        /// Additive shift when `level` is `None`.
        shift: f64,
    },
    /// A sequential intervention.
    Sequential {
        /// Set, constant soft intervention, or shift overlays.
        overlays: Vec<SequentialNodeOverlay>,
    },
    /// Deterministic mean mechanisms.
    Mechanisms {
        /// Mean-mechanism overlays, optionally mixed with set or shift actions.
        overlays: Vec<SequentialMechanismOverlay>,
    },
}

/// A plan whose clones share both the overlay storage and its admission lease.
#[derive(Clone, Debug)]
pub struct TemporalInterventionPlan(Arc<TemporalPlanInner>);
#[derive(Debug)]
struct TemporalPlanInner {
    kind: TemporalInterventionKind,
    _lease: antecedent_core::execution::AdmissionLease,
}

/// An admitted identifier schedule. Clones share its storage and lease.
#[derive(Clone, Debug)]
pub struct AdmittedIdentificationSchedule(Arc<IdentificationScheduleInner>);
#[derive(Debug)]
struct IdentificationScheduleInner {
    values: Vec<(VariableId, i32, Option<f64>)>,
    _lease: antecedent_core::execution::AdmissionLease,
}
impl std::ops::Deref for AdmittedIdentificationSchedule {
    type Target = [(VariableId, i32, Option<f64>)];
    fn deref(&self) -> &Self::Target {
        &self.0.values
    }
}

/// A materialized mechanism schedule, sharing data and admission across clones.
#[derive(Clone, Debug)]
pub struct AdmittedMechanismOverlays(Arc<MechanismOverlaysInner>);
#[derive(Debug)]
struct MechanismOverlaysInner {
    values: Vec<SequentialMechanismOverlay>,
    _lease: antecedent_core::execution::AdmissionLease,
}
impl std::ops::Deref for AdmittedMechanismOverlays {
    type Target = [SequentialMechanismOverlay];
    fn deref(&self) -> &Self::Target {
        &self.0.values
    }
}

impl TemporalInterventionPlan {
    /// Borrow the admitted plan shape; a caller-created copy is caller-owned.
    #[must_use]
    pub fn kind(&self) -> &TemporalInterventionKind {
        &self.0.kind
    }

    /// Iterate overlays without cloning their backing allocation.
    pub fn mechanism_overlays(
        &self,
    ) -> Option<impl Iterator<Item = SequentialMechanismOverlay> + '_> {
        let (sequential, mechanisms): (&[SequentialNodeOverlay], &[SequentialMechanismOverlay]) =
            match self.kind() {
                TemporalInterventionKind::Single { .. } => return None,
                TemporalInterventionKind::Sequential { overlays } => (overlays, &[]),
                TemporalInterventionKind::Mechanisms { overlays } => (&[], overlays),
            };
        Some(sequential.iter().copied().map(Into::into).chain(mechanisms.iter().copied()))
    }

    /// Materialize mechanism overlays only when a slice is required by a consumer.
    pub fn admitted_mechanism_overlays(
        &self,
        budget: &mut antecedent_core::TemporalBudget,
        ctx: &antecedent_core::ExecutionContext,
    ) -> Result<Option<AdmittedMechanismOverlays>, EstimationError> {
        use antecedent_core::TemporalMaterializationError as T;
        if ctx.cancellation.is_cancelled() {
            return Err(T::Cancelled.into());
        }
        let count = match self.kind() {
            TemporalInterventionKind::Single { .. } => return Ok(None),
            TemporalInterventionKind::Sequential { overlays } => overlays.len() as u64,
            TemporalInterventionKind::Mechanisms { overlays } => overlays.len() as u64,
        };
        budget.charge_offsets(count)?;
        budget.charge_work(count)?;
        let bytes = count
            .checked_mul(256)
            .and_then(|n| n.checked_add(512))
            .ok_or(T::Limit { dimension: "bytes" })?;
        if bytes > budget.limits().max_bytes {
            return Err(T::Limit { dimension: "bytes" }.into());
        }
        let lease = ctx.acquire_admission(0, bytes).map_err(|e| match e {
            antecedent_core::execution::AdmissionError::Cancelled => T::Cancelled,
            other => T::Admission(other),
        })?;
        let mut values = Vec::new();
        values
            .try_reserve_exact(usize::try_from(count).map_err(|_| T::Allocation)?)
            .map_err(|_| T::Allocation)?;
        if let Some(iter) = self.mechanism_overlays() {
            for overlay in iter {
                if ctx.cancellation.is_cancelled() {
                    return Err(T::Cancelled.into());
                }
                values.push(overlay);
            }
        }
        Ok(Some(AdmittedMechanismOverlays(Arc::new(MechanismOverlaysInner {
            values,
            _lease: lease,
        }))))
    }

    /// Build an admitted identifier schedule under the same operation budget.
    pub fn identification_schedule(
        &self,
        spec: &TemporalResponseSpec,
        budget: &mut antecedent_core::TemporalBudget,
        ctx: &antecedent_core::ExecutionContext,
    ) -> Result<AdmittedIdentificationSchedule, EstimationError> {
        self.schedule_impl(spec, budget, ctx, false).map(|(schedule, _)| schedule)
    }

    /// An admitted schedule with fixed levels removed, plus whether any were removed.
    pub fn identification_schedule_without_levels(
        &self,
        spec: &TemporalResponseSpec,
        budget: &mut antecedent_core::TemporalBudget,
        ctx: &antecedent_core::ExecutionContext,
    ) -> Result<(AdmittedIdentificationSchedule, bool), EstimationError> {
        self.schedule_impl(spec, budget, ctx, true)
    }

    fn schedule_impl(
        &self,
        spec: &TemporalResponseSpec,
        budget: &mut antecedent_core::TemporalBudget,
        ctx: &antecedent_core::ExecutionContext,
        strip_levels: bool,
    ) -> Result<(AdmittedIdentificationSchedule, bool), EstimationError> {
        use antecedent_core::TemporalMaterializationError as T;
        if ctx.cancellation.is_cancelled() {
            return Err(T::Cancelled.into());
        }
        let count = match self.kind() {
            TemporalInterventionKind::Single { .. } => spec.policy.offset_count()?,
            TemporalInterventionKind::Sequential { overlays } => overlays.len() as u64,
            TemporalInterventionKind::Mechanisms { overlays } => overlays.len() as u64,
        };
        budget.charge_offsets(count)?;
        budget.charge_work(count.checked_mul(2).ok_or(T::Limit { dimension: "work" })?)?;
        let bytes = count
            .checked_mul(128)
            .and_then(|n| n.checked_add(512))
            .ok_or(T::Limit { dimension: "bytes" })?;
        if bytes > budget.limits().max_bytes {
            return Err(T::Limit { dimension: "bytes" }.into());
        }
        let lease = ctx.acquire_admission(0, bytes).map_err(|e| match e {
            antecedent_core::execution::AdmissionError::Cancelled => T::Cancelled,
            other => T::Admission(other),
        })?;
        let mut values = Vec::new();
        values
            .try_reserve_exact(usize::try_from(count).map_err(|_| T::Allocation)?)
            .map_err(|_| T::Allocation)?;
        match self.kind() {
            TemporalInterventionKind::Single { treatment, level, .. } => {
                for offset in spec.policy.offsets()? {
                    if ctx.cancellation.is_cancelled() {
                        return Err(T::Cancelled.into());
                    }
                    values.push((*treatment, offset, *level));
                }
            }
            TemporalInterventionKind::Sequential { overlays } => {
                for overlay in overlays {
                    if ctx.cancellation.is_cancelled() {
                        return Err(T::Cancelled.into());
                    }
                    values.push((overlay.variable, overlay.offset, overlay.level));
                }
            }
            TemporalInterventionKind::Mechanisms { overlays } => {
                for overlay in overlays {
                    if ctx.cancellation.is_cancelled() {
                        return Err(T::Cancelled.into());
                    }
                    values.push((overlay.node.variable, overlay.node.offset, overlay.node.level));
                }
            }
        }
        let mut stripped = false;
        if strip_levels {
            for value in &mut values {
                if ctx.cancellation.is_cancelled() {
                    return Err(T::Cancelled.into());
                }
                stripped |= value.2.is_some();
                value.2 = None;
            }
        }
        Ok((
            AdmittedIdentificationSchedule(Arc::new(IdentificationScheduleInner {
                values,
                _lease: lease,
            })),
            stripped,
        ))
    }
}

/// Inspect a direct single intervention without materializing a schedule.
/// Sequence and mean-mechanism plans return `None`; malformed direct inputs
/// return the same refusal as planning.
pub fn single_temporal_intervention(
    query: &ResponseQuery,
) -> Result<Option<(VariableId, Option<f64>, f64)>, EstimationError> {
    if query.temporal.is_none() {
        return Ok(None);
    }
    let ResponseFunctional::InterventionResponse { interventions, .. } = &query.functional else {
        return Ok(None);
    };
    if interventions.len() != 1 {
        return Err(EstimationError::unsupported(
            "temporal InterventionResponse needs one primary intervention",
        ));
    }
    match &interventions[0] {
        Intervention::Sequence(_) => Ok(None),
        Intervention::Soft { mechanism, .. }
            if matches!(mechanism.family_id.as_ref(), "multiplicative" | "truncated_shift") =>
        {
            Ok(None)
        }
        other => resolve_one(other, 0).map(Some),
    }
}

/// Whether the declared temporal response needs a sequence or mean-mechanism plan.
/// This shape inspection does not materialize or validate that plan.
#[must_use]
pub fn requires_temporal_overlays(query: &ResponseQuery) -> bool {
    if query.temporal.is_none() {
        return false;
    }
    let ResponseFunctional::InterventionResponse { interventions, .. } = &query.functional else {
        return false;
    };
    if interventions.len() != 1 {
        return false;
    }
    match &interventions[0] {
        Intervention::Sequence(_) => true,
        Intervention::Soft { mechanism, .. } => {
            matches!(mechanism.family_id.as_ref(), "multiplicative" | "truncated_shift")
        }
        _ => false,
    }
}

/// Bound borrowed temporal-response validation before its recursive or allocating work.
/// Intervention trees and expressions are visited without allocation before
/// generic validation, under the same cumulative work and admission limits.
/// # Errors
/// Invalid query, unsupported route, cancellation, or resource refusal.
pub fn validate_temporal_response_within(
    query: &ResponseQuery,
    budget: &mut antecedent_core::TemporalBudget,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<(), EstimationError> {
    use antecedent_core::TemporalMaterializationError as T;
    if ctx.cancellation.is_cancelled() {
        return Err(T::Cancelled.into());
    }
    let temporal = query.temporal.as_ref().ok_or_else(|| {
        EstimationError::unsupported("temporal response needs a temporal specification")
    })?;
    let mut work =
        (temporal.horizons.len() as u64).checked_mul(4).ok_or(T::Limit { dimension: "work" })?;
    match &query.outcome_functional {
        antecedent_core::OutcomeFunctional::ExceedanceGrid(grid) => {
            work = work
                .checked_add(
                    (grid.len() as u64).checked_mul(4).ok_or(T::Limit { dimension: "work" })?,
                )
                .ok_or(T::Limit { dimension: "work" })?;
        }
        _ => {}
    }
    match &query.functional {
        ResponseFunctional::MeanCurve { treatment, .. } => {
            if let antecedent_core::GridSpec::Values(values) = &treatment.grid {
                work = work
                    .checked_add(
                        (values.len() as u64)
                            .checked_mul(4)
                            .ok_or(T::Limit { dimension: "work" })?,
                    )
                    .ok_or(T::Limit { dimension: "work" })?;
            }
        }
        ResponseFunctional::AverageDerivative {
            weighting: antecedent_core::DerivativeWeighting::Custom(weights),
            ..
        } => {
            work = work
                .checked_add(
                    (weights.len() as u64).checked_mul(4).ok_or(T::Limit { dimension: "work" })?,
                )
                .ok_or(T::Limit { dimension: "work" })?;
        }
        ResponseFunctional::DirectionalDerivative { outcomes, treatments, at, direction } => {
            let pairs = (outcomes.len() as u64)
                .checked_mul(treatments.len() as u64)
                .ok_or(T::Limit { dimension: "work" })?;
            work = work
                .checked_add(pairs)
                .and_then(|n| n.checked_add(at.len() as u64))
                .and_then(|n| n.checked_add(direction.len() as u64))
                .ok_or(T::Limit { dimension: "work" })?;
        }
        ResponseFunctional::Jacobian { outcomes, treatments, at, .. } => {
            let pairs = (outcomes.len() as u64)
                .checked_mul(treatments.len() as u64)
                .ok_or(T::Limit { dimension: "work" })?;
            work = work
                .checked_add(pairs)
                .and_then(|n| n.checked_add(at.len() as u64))
                .ok_or(T::Limit { dimension: "work" })?;
        }
        ResponseFunctional::InterventionResponse { interventions, .. } => {
            work = work
                .checked_add(interventions.len() as u64)
                .ok_or(T::Limit { dimension: "work" })?;
        }
        _ => {}
    }
    budget.charge_work(work)?;
    let validation_lease = match &query.functional {
        ResponseFunctional::InterventionResponse { interventions, .. } => {
            Some(antecedent_core::admit_temporal_intervention_validation(
                interventions.iter(),
                budget,
                ctx,
            )?)
        }
        _ => None,
    };
    query.validate()?;
    if ctx.cancellation.is_cancelled() {
        return Err(T::Cancelled.into());
    }
    drop(validation_lease);
    Ok(())
}

/// Plan overlays for a temporal [`ResponseQuery`], if it is an `InterventionResponse`.
///
/// # Errors
///
/// Unlicensed Sequence / Soft forms.
pub fn plan_from_response_query(
    query: &ResponseQuery,
    budget: &mut antecedent_core::TemporalBudget,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<Option<TemporalInterventionPlan>, EstimationError> {
    let Some(temporal) = query.temporal.as_ref() else {
        return Ok(None);
    };
    match &query.functional {
        ResponseFunctional::InterventionResponse { interventions, .. } => {
            Ok(Some(plan_temporal_intervention(interventions, temporal, budget, ctx)?))
        }
        _ => Ok(None),
    }
}

/// Classify a temporal `InterventionResponse` as a single-node overlay or a
/// sequential schedule. Nested Sequence stays refused. Multi-step never
/// collapses to the last step.
///
/// # Errors
///
/// Empty, nested, stochastic, unlicensed Soft, or ambiguous schedules.
pub(crate) fn plan_temporal_intervention(
    interventions: &[Intervention],
    spec: &TemporalResponseSpec,
    budget: &mut antecedent_core::TemporalBudget,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<TemporalInterventionPlan, EstimationError> {
    use antecedent_core::TemporalMaterializationError as T;
    if ctx.cancellation.is_cancelled() {
        return Err(T::Cancelled.into());
    }
    if interventions.len() != 1 {
        return Err(EstimationError::unsupported(
            "temporal InterventionResponse needs one primary intervention",
        ));
    }
    let mut offsets = 0u64;
    let mut steps = 1u64;
    let primary = &interventions[0];
    if let Intervention::Sequence(sequence) = primary {
        steps = u64::try_from(sequence.steps.len()).map_err(|_| T::Limit { dimension: "work" })?;
        budget.charge_work(steps)?;
        for step in sequence.steps.iter() {
            if ctx.cancellation.is_cancelled() {
                return Err(T::Cancelled.into());
            }
            // The planner does not license nested sequences or arbitrary symbolic expressions.
            if matches!(&step.intervention, Intervention::Sequence(_)) {
                return Err(EstimationError::unsupported_coded(
                    antecedent_core::reason_code!("route_not_supported"),
                    "Intervention::Sequence nested inside a Sequence is not licensed for \
                     temporal InterventionResponse",
                ));
            }
            if !matches!(
                &step.intervention,
                Intervention::Set { .. } | Intervention::Shift { .. } | Intervention::Soft { .. }
            ) {
                return Err(EstimationError::unsupported(
                    "temporal sequence step is not licensed for this route",
                ));
            }
            offsets = offsets
                .checked_add(step.temporal.offset_count()?)
                .ok_or(T::Limit { dimension: "offsets" })?;
        }
    } else if let Intervention::Soft { mechanism, .. } = primary {
        if matches!(mechanism.family_id.as_ref(), "multiplicative" | "truncated_shift") {
            offsets = spec.policy.offset_count()?;
        }
    }
    budget.charge_offsets(offsets)?;
    // Multiple copies, sorting, and duplicate checks are bounded by these
    // conservative logical units. The allocator itself is not a peak-memory proof.
    let work = offsets
        .checked_mul(8)
        .and_then(|n| steps.checked_mul(steps).and_then(|q| n.checked_add(q)))
        .and_then(|n| steps.checked_mul(16).and_then(|q| n.checked_add(q)))
        .ok_or(T::Limit { dimension: "work" })?;
    budget.charge_work(work)?;
    let bytes = offsets
        .checked_mul(512)
        .and_then(|n| steps.checked_mul(1024).and_then(|q| n.checked_add(q)))
        .and_then(|n| n.checked_add(4096))
        .ok_or(T::Limit { dimension: "bytes" })?;
    if bytes > budget.limits().max_bytes {
        return Err(T::Limit { dimension: "bytes" }.into());
    }
    let lease = ctx.acquire_admission(0, bytes).map_err(|e| match e {
        antecedent_core::execution::AdmissionError::Cancelled => T::Cancelled,
        other => T::Admission(other),
    })?;
    let kind = plan_temporal_intervention_raw(interventions, spec, ctx)?;
    if ctx.cancellation.is_cancelled() {
        return Err(T::Cancelled.into());
    }
    Ok(TemporalInterventionPlan(Arc::new(TemporalPlanInner { kind, _lease: lease })))
}

fn bounded_offsets(
    policy: &antecedent_core::TemporalPolicy,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<Arc<[i32]>, EstimationError> {
    use antecedent_core::TemporalMaterializationError as T;
    let count = usize::try_from(policy.offset_count()?).map_err(|_| T::Allocation)?;
    let mut values = Vec::new();
    values.try_reserve_exact(count).map_err(|_| T::Allocation)?;
    for offset in policy.offsets()? {
        if ctx.cancellation.is_cancelled() {
            return Err(T::Cancelled.into());
        }
        values.push(offset);
    }
    Ok(Arc::from(values))
}

fn plan_temporal_intervention_raw(
    interventions: &[Intervention],
    spec: &TemporalResponseSpec,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<TemporalInterventionKind, EstimationError> {
    if interventions.is_empty() {
        return Err(EstimationError::unsupported(
            "intervention response requires at least one intervention",
        ));
    }
    if interventions.len() > 1 {
        return Err(EstimationError::unsupported_coded(
            antecedent_core::reason_code!("route_not_supported"),
            "temporal InterventionResponse supports one primary intervention \
             (use Sequence for multi-step or joint policies)",
        ));
    }
    if let Some(plan) = plan_mean_mechanisms(&interventions[0], spec, ctx)? {
        return Ok(plan);
    }
    match &interventions[0] {
        Intervention::Sequence(seq) => plan_sequence(seq, spec, 0, ctx),
        other => {
            let (treatment, level, shift) = resolve_one(other, 0)?;
            Ok(TemporalInterventionKind::Single { treatment, level, shift })
        }
    }
}

fn plan_mean_mechanisms(
    intervention: &Intervention,
    spec: &TemporalResponseSpec,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<Option<TemporalInterventionKind>, EstimationError> {
    let leaves = match intervention {
        Intervention::Sequence(sequence) => {
            sequence.steps.iter().map(|step| &step.intervention).collect::<Vec<_>>()
        }
        other => vec![other],
    };
    if !leaves.iter().any(|leaf| matches!(leaf,
        Intervention::Soft { mechanism, .. } if matches!(mechanism.family_id.as_ref(), "multiplicative" | "truncated_shift"))) {
        return Ok(None);
    }
    let mut modifiers = Vec::new();
    let mut replacements = Vec::new();
    for leaf in leaves {
        if ctx.cancellation.is_cancelled() {
            return Err(antecedent_core::TemporalMaterializationError::Cancelled.into());
        }
        let mut multiplier = 1.0;
        let mut bounds = None;
        let mut replacement = leaf.clone();
        if let Intervention::Soft { variable, mechanism } = leaf {
            match mechanism.family_id.as_ref() {
                "multiplicative" => {
                    if mechanism.parameters.len() != 1 || !mechanism.parameters[0].is_finite() {
                        return Err(EstimationError::unsupported(
                            "multiplicative requires one finite mean multiplier",
                        ));
                    }
                    multiplier = mechanism.parameters[0];
                    replacement =
                        Intervention::soft(*variable, MechanismOverride::additive_shift(0.0));
                }
                "truncated_shift" => {
                    let p = &mechanism.parameters;
                    if p.len() != 3 || p.iter().any(|v| !v.is_finite()) || p[1] > p[2] {
                        return Err(EstimationError::unsupported(
                            "truncated_shift requires finite [shift, lower, upper] with lower <= upper",
                        ));
                    }
                    bounds = Some((p[1], p[2]));
                    replacement =
                        Intervention::soft(*variable, MechanismOverride::additive_shift(p[0]));
                }
                _ => {}
            }
        }
        modifiers.push((multiplier, bounds));
        replacements.push(replacement);
    }
    let surrogate = if let Intervention::Sequence(sequence) = intervention {
        let mut sequence = sequence.clone();
        let mut steps = sequence.steps.to_vec();
        for (step, replacement) in steps.iter_mut().zip(replacements) {
            step.intervention = replacement;
        }
        sequence.steps = Arc::from(steps);
        Intervention::Sequence(sequence)
    } else {
        replacements.remove(0)
    };
    let base = plan_temporal_intervention_raw(&[surrogate], spec, ctx)?;
    let nodes = match base {
        TemporalInterventionKind::Single { treatment, level, shift } => {
            let offsets = bounded_offsets(&spec.policy, ctx)?;
            let mut nodes = Vec::with_capacity(offsets.len());
            for &offset in offsets.iter() {
                if ctx.cancellation.is_cancelled() {
                    return Err(antecedent_core::TemporalMaterializationError::Cancelled.into());
                }
                nodes.push(SequentialNodeOverlay { variable: treatment, offset, level, shift });
            }
            nodes
        }
        TemporalInterventionKind::Sequential { overlays } => overlays,
        TemporalInterventionKind::Mechanisms { .. } => {
            return Err(EstimationError::unsupported(
                "a set or shift surrogate cannot plan as a mean mechanism",
            ));
        }
    };
    let mut overlays = Vec::with_capacity(nodes.len());
    for (index, node) in nodes.into_iter().enumerate() {
        if ctx.cancellation.is_cancelled() {
            return Err(antecedent_core::TemporalMaterializationError::Cancelled.into());
        }
        let (multiplier, bounds) = modifiers[if modifiers.len() == 1 { 0 } else { index }];
        overlays.push(SequentialMechanismOverlay { node, multiplier, bounds });
    }
    Ok(Some(TemporalInterventionKind::Mechanisms { overlays }))
}

/// `depth` counts levels of `Intervention::Sequence` nesting already entered.
/// `resolve_sequence` is only reachable from a Sequence itself, so a `depth > 0`
/// arrival there means a Sequence nested inside a Sequence — refused explicitly
/// rather than silently recursing into a leaf (ADR 0021 fail-closed contract).
pub(super) fn resolve_one(
    iv: &Intervention,
    depth: usize,
) -> Result<(VariableId, Option<f64>, f64), EstimationError> {
    let finite_numeric = |value: &Value, kind: &'static str| {
        value.as_f64().filter(|v| v.is_finite()).ok_or_else(|| {
            EstimationError::unsupported(match kind {
                "set" => "intervention Set requires a finite numeric value",
                _ => "intervention Shift requires a finite numeric delta",
            })
        })
    };
    let one_finite_parameter = |mechanism: &MechanismOverride, family: &'static str| {
        if mechanism.parameters.len() != 1 || !mechanism.parameters[0].is_finite() {
            return Err(EstimationError::unsupported(match family {
                "constant" => "Soft(constant) requires exactly one finite parameter",
                _ => "Soft(additive_shift) requires exactly one finite parameter",
            }));
        }
        Ok(mechanism.parameters[0])
    };
    match iv {
        Intervention::Set { variable, value } => {
            let level = finite_numeric(value, "set")?;
            Ok((*variable, Some(level), 0.0))
        }
        Intervention::Shift { variable, delta } => {
            let d = finite_numeric(delta, "shift")?;
            Ok((*variable, None, d))
        }
        Intervention::Soft { variable, mechanism } => match mechanism.family_id.as_ref() {
            "constant" => {
                let level = one_finite_parameter(mechanism, "constant")?;
                Ok((*variable, Some(level), 0.0))
            }
            "additive_shift" => {
                let d = one_finite_parameter(mechanism, "additive_shift")?;
                Ok((*variable, None, d))
            }
            _ => Err(EstimationError::unsupported(
                "Soft mechanism family is not licensed for temporal InterventionResponse; use constant or additive_shift",
            )),
        },
        Intervention::Sequence(seq) => {
            if depth > 0 {
                return Err(EstimationError::unsupported_coded(
                    antecedent_core::reason_code!("route_not_supported"),
                    "Intervention::Sequence nested inside a Sequence is not licensed for \
                     temporal InterventionResponse",
                ));
            }
            if seq.steps.len() != 1 {
                return Err(EstimationError::unsupported(
                    "multi-step Sequence must be planned as sequential overlays; \
                     refuse rather than collapse to the last step",
                ));
            }
            resolve_one(&seq.steps[0].intervention, depth + 1)
        }
        Intervention::Stochastic { .. } => Err(EstimationError::unsupported_coded(
            antecedent_core::reason_code!("route_not_supported"),
            "stochastic interventions are not licensed on the temporal InterventionResponse path",
        )),
        _ => Err(EstimationError::unsupported(
            "unsupported intervention variant for temporal response",
        )),
    }
}

fn plan_sequence(
    seq: &InterventionSequence,
    spec: &TemporalResponseSpec,
    depth: usize,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<TemporalInterventionKind, EstimationError> {
    if seq.is_empty() {
        return Err(EstimationError::unsupported("empty Intervention::Sequence"));
    }
    if depth > 0 {
        return Err(EstimationError::unsupported_coded(
            antecedent_core::reason_code!("route_not_supported"),
            "Intervention::Sequence nested inside a Sequence is not licensed for \
             temporal InterventionResponse",
        ));
    }
    let origin = spec.treatment_offset()?;
    let mut leaves = Vec::with_capacity(seq.steps.len());
    for step in seq.steps.iter() {
        if ctx.cancellation.is_cancelled() {
            return Err(antecedent_core::TemporalMaterializationError::Cancelled.into());
        }
        if matches!(step.intervention, Intervention::Sequence(_)) {
            return Err(EstimationError::unsupported_coded(
                antecedent_core::reason_code!("route_not_supported"),
                "Intervention::Sequence nested inside a Sequence is not licensed for \
                 temporal InterventionResponse",
            ));
        }
        let (variable, level, shift) = resolve_one(&step.intervention, depth + 1)?;
        let offsets = bounded_offsets(&step.temporal, ctx)?;
        leaves.push((variable, level, shift, offsets));
    }
    if leaves.len() == 1 {
        let (variable, level, shift, offsets) = &leaves[0];
        // Pulse(0) / a singleton window at 0 is the implicit Sequence shorthand
        // and attaches to the spec origin. Any other native step policy — Pulse(-2),
        // Sustained(-3, -1), or a window that already includes time 0 among other
        // times — is executed as written.
        let resolved = match offsets.as_ref() {
            [0] => Arc::<[i32]>::from([origin]),
            _ => Arc::clone(offsets),
        };
        let mut overlays = Vec::with_capacity(resolved.len());
        for &offset in resolved.iter() {
            if ctx.cancellation.is_cancelled() {
                return Err(antecedent_core::TemporalMaterializationError::Cancelled.into());
            }
            overlays.push(SequentialNodeOverlay {
                variable: *variable,
                offset,
                level: *level,
                shift: *shift,
            });
        }
        return Ok(TemporalInterventionKind::Sequential { overlays });
    }
    let offsets = sequence_overlay_offsets(origin, &leaves, ctx)?;
    let mut overlays = Vec::with_capacity(leaves.len());
    let mut seen = Vec::new();
    for ((variable, level, shift, _), offset) in leaves.iter().zip(offsets) {
        if ctx.cancellation.is_cancelled() {
            return Err(antecedent_core::TemporalMaterializationError::Cancelled.into());
        }
        let key = (*variable, offset);
        if seen.contains(&key) {
            return Err(EstimationError::unsupported(
                "Sequence assigns the same (variable, time) twice; refuse rather than collapse",
            ));
        }
        seen.push(key);
        overlays.push(SequentialNodeOverlay {
            variable: *variable,
            offset,
            level: *level,
            shift: *shift,
        });
    }
    Ok(TemporalInterventionKind::Sequential { overlays })
}

/// Multi-step same variable → consecutive times ending at the spec origin.
/// Distinct variables, each once → joint at the origin.
/// Explicit distinct Pulse offsets are honored.
fn sequence_overlay_offsets(
    origin: i32,
    leaves: &[SequenceLeaf],
    ctx: &antecedent_core::ExecutionContext,
) -> Result<Vec<i32>, EstimationError> {
    let n = i32::try_from(leaves.len())
        .map_err(|_| EstimationError::unsupported("Sequence is too long"))?;
    let mut explicit = Vec::with_capacity(leaves.len());
    for (_, _, _, offsets) in leaves {
        if ctx.cancellation.is_cancelled() {
            return Err(antecedent_core::TemporalMaterializationError::Cancelled.into());
        }
        let [at] = offsets.as_ref() else {
            return Err(EstimationError::unsupported_coded(
                antecedent_core::reason_code!("route_not_supported"),
                "each Sequence step must be a Pulse or single-time window",
            ));
        };
        explicit.push(*at);
    }
    let same_var = leaves.iter().all(|(variable, _, _, _)| *variable == leaves[0].0);
    let all_default = explicit.iter().all(|&at| at == 0);
    let all_origin = explicit.iter().all(|&at| at == origin);
    let distinct_explicit = {
        let mut seen = explicit.clone();
        seen.sort_unstable();
        seen.dedup();
        seen.len() == explicit.len() && !(all_default || all_origin)
    };
    if distinct_explicit {
        return Ok(explicit);
    }
    if same_var && (all_default || all_origin) {
        // Consecutive times ending at the policy origin. Two Pulse{0} (or
        // Pulse{origin}) steps are a two-step policy, not last-step collapse.
        let mut offsets = Vec::with_capacity(leaves.len());
        for i in 0..n {
            if ctx.cancellation.is_cancelled() {
                return Err(antecedent_core::TemporalMaterializationError::Cancelled.into());
            }
            offsets.push(
                origin
                    .checked_sub(n - 1 - i)
                    .ok_or_else(|| EstimationError::unsupported("Sequence times exceed i32"))?,
            );
        }
        return Ok(offsets);
    }
    let unique_vars = {
        let mut vars: Vec<VariableId> =
            leaves.iter().map(|(variable, _, _, _)| *variable).collect();
        vars.sort_by_key(|variable| variable.raw());
        vars.dedup();
        vars.len() == leaves.len()
    };
    if unique_vars && (all_default || all_origin) {
        return Ok(vec![origin; leaves.len()]);
    }
    Err(EstimationError::unsupported(
        "Sequence schedule is ambiguous; use distinct Pulse offsets or a same-variable \
         consecutive policy",
    ))
}

#[cfg(test)]
mod admitted_plan_tests {
    use super::*;
    use antecedent_core::execution::AdmissionPool;
    use antecedent_core::{SequencedIntervention, TemporalMaterializationError, TemporalPolicy};

    #[test]
    fn bounded_sequence_refuses_large_ranges_and_preserves_lease_across_clones() {
        let ctx = antecedent_core::ExecutionContext::for_tests(3)
            .with_admission(AdmissionPool::fixed(1, 65_536));
        let spec = TemporalResponseSpec::new(vec![1], TemporalPolicy::pulse(0), None).unwrap();
        let step = |policy| {
            Intervention::Sequence(InterventionSequence::new([SequencedIntervention {
                intervention: Intervention::set(VariableId::from_raw(0), Value::f64(1.0)),
                temporal: policy,
            }]))
        };
        let mut budget =
            antecedent_core::TemporalBudget::new(ctx.temporal_materialization_limits());
        let huge = step(TemporalPolicy::sustained(i32::MIN, i32::MAX));
        assert!(matches!(
            plan_temporal_intervention(&[huge], &spec, &mut budget, &ctx),
            Err(EstimationError::TemporalMaterialization(
                TemporalMaterializationError::Limit { .. }
            ))
        ));
        let before = ctx.admission.available().bytes();
        let mut budget =
            antecedent_core::TemporalBudget::new(ctx.temporal_materialization_limits());
        let plan = plan_temporal_intervention(
            &[step(TemporalPolicy::sustained(-1, 1))],
            &spec,
            &mut budget,
            &ctx,
        )
        .unwrap();
        let during = ctx.admission.available().bytes();
        assert!(during < before);
        let clone = plan.clone();
        drop(plan);
        assert_eq!(ctx.admission.available().bytes(), during);
        drop(clone);
        assert_eq!(ctx.admission.available().bytes(), before);
        ctx.cancellation.cancel();
        assert!(matches!(
            plan_temporal_intervention(&[step(TemporalPolicy::pulse(0))], &spec, &mut budget, &ctx),
            Err(EstimationError::TemporalMaterialization(TemporalMaterializationError::Cancelled))
        ));
    }
}
