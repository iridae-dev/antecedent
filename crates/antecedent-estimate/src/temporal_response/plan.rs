//! Licensed temporal intervention plans.

use std::sync::Arc;

use crate::error::EstimationError;
use crate::temporal_sequential::{SequentialMechanismOverlay, SequentialNodeOverlay};
use antecedent_core::{
    Intervention, InterventionSequence, MechanismOverride, ResponseFunctional, ResponseQuery,
    TemporalResponseSpec, Value, VariableId,
};

use super::surface::SequenceLeaf;

/// Licensed temporal `InterventionResponse` overlay.
#[derive(Clone, Debug, PartialEq)]
pub enum TemporalInterventionPlan {
    /// One treatment schedule from [`TemporalResponseSpec::policy`] (plain Set / Shift / Soft).
    Single {
        /// Intervened variable.
        treatment: VariableId,
        /// Hard set / Soft constant.
        level: Option<f64>,
        /// Additive shift when `level` is `None`.
        shift: f64,
    },
    /// Explicit Sequence: one overlay per intervened unfolded node.
    Sequential {
        /// Licensed Set / Soft constant / Soft shift overlays.
        overlays: Vec<SequentialNodeOverlay>,
    },
    /// Deterministic mean mechanisms, evaluated through the unfolded engine.
    Mechanisms {
        /// Multiplicative or bounded-mean overlays, optionally mixed with Set/Shift.
        overlays: Vec<SequentialMechanismOverlay>,
    },
}

impl TemporalInterventionPlan {
    /// Sequential mean-mechanism overlays; single Set/Shift keeps its direct path.
    #[must_use]
    pub fn mechanism_overlays(&self) -> Option<Vec<SequentialMechanismOverlay>> {
        match self {
            Self::Single { .. } => None,
            Self::Sequential { overlays } => {
                Some(overlays.iter().copied().map(Into::into).collect())
            }
            Self::Mechanisms { overlays } => Some(overlays.clone()),
        }
    }
    /// Treatment nodes the identifier must cover, with the `Set` level actually
    /// requested at each node (`None` for a Soft/shift step, which has no
    /// fixed value).
    #[must_use]
    pub fn identification_schedule(
        &self,
        spec: &TemporalResponseSpec,
    ) -> Vec<(VariableId, i32, Option<f64>)> {
        match self {
            Self::Single { treatment, level, .. } => spec
                .policy
                .active_offsets()
                .map(|offsets| offsets.iter().map(|&offset| (*treatment, offset, *level)).collect())
                .unwrap_or_default(),
            Self::Sequential { overlays } => overlays
                .iter()
                .map(|overlay| (overlay.variable, overlay.offset, overlay.level))
                .collect(),
            Self::Mechanisms { overlays } => overlays
                .iter()
                .map(|overlay| (overlay.node.variable, overlay.node.offset, overlay.node.level))
                .collect(),
        }
    }
}

/// Plan overlays for a temporal [`ResponseQuery`], if it is an `InterventionResponse`.
///
/// # Errors
///
/// Unlicensed Sequence / Soft forms.
pub fn plan_from_response_query(
    query: &ResponseQuery,
) -> Result<Option<TemporalInterventionPlan>, EstimationError> {
    let Some(temporal) = query.temporal.as_ref() else {
        return Ok(None);
    };
    match &query.functional {
        ResponseFunctional::InterventionResponse { interventions, .. } => {
            Ok(Some(plan_temporal_intervention(interventions, temporal)?))
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
) -> Result<TemporalInterventionPlan, EstimationError> {
    if interventions.is_empty() {
        return Err(EstimationError::unsupported(
            "intervention response requires at least one intervention",
        ));
    }
    if interventions.len() > 1 {
        return Err(EstimationError::unsupported(
            "temporal InterventionResponse supports one primary intervention \
             (use Sequence for multi-step or joint policies)",
        ));
    }
    if let Some(plan) = plan_mean_mechanisms(&interventions[0], spec)? {
        return Ok(plan);
    }
    match &interventions[0] {
        Intervention::Sequence(seq) => plan_sequence(seq, spec, 0),
        other => {
            let (treatment, level, shift) = resolve_one(other, 0)?;
            Ok(TemporalInterventionPlan::Single { treatment, level, shift })
        }
    }
}

fn plan_mean_mechanisms(
    intervention: &Intervention,
    spec: &TemporalResponseSpec,
) -> Result<Option<TemporalInterventionPlan>, EstimationError> {
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
    let base = plan_temporal_intervention(&[surrogate], spec)?;
    let nodes = match base {
        TemporalInterventionPlan::Single { treatment, level, shift } => spec
            .policy
            .active_offsets()?
            .iter()
            .map(|&offset| SequentialNodeOverlay { variable: treatment, offset, level, shift })
            .collect::<Vec<_>>(),
        TemporalInterventionPlan::Sequential { overlays } => overlays,
        TemporalInterventionPlan::Mechanisms { .. } => {
            return Err(EstimationError::unsupported(
                "a set or shift surrogate cannot plan as a mean mechanism",
            ));
        }
    };
    let overlays = nodes
        .into_iter()
        .enumerate()
        .map(|(index, node)| {
            let (multiplier, bounds) = modifiers[if modifiers.len() == 1 { 0 } else { index }];
            SequentialMechanismOverlay { node, multiplier, bounds }
        })
        .collect();
    Ok(Some(TemporalInterventionPlan::Mechanisms { overlays }))
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
            other => Err(EstimationError::data_msg(format!(
                "Soft mechanism family `{other}` is not licensed for temporal InterventionResponse; \
                 use constant or additive_shift"
            ))),
        },
        Intervention::Sequence(seq) => {
            if depth > 0 {
                return Err(EstimationError::unsupported(
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
        Intervention::Stochastic { .. } => Err(EstimationError::unsupported(
            "stochastic interventions are not licensed on the temporal InterventionResponse path",
        )),
        other => Err(EstimationError::data_msg(format!(
            "unsupported intervention variant for temporal response: {other:?}"
        ))),
    }
}

fn plan_sequence(
    seq: &InterventionSequence,
    spec: &TemporalResponseSpec,
    depth: usize,
) -> Result<TemporalInterventionPlan, EstimationError> {
    if seq.is_empty() {
        return Err(EstimationError::unsupported("empty Intervention::Sequence"));
    }
    if depth > 0 {
        return Err(EstimationError::unsupported(
            "Intervention::Sequence nested inside a Sequence is not licensed for \
             temporal InterventionResponse",
        ));
    }
    let origin = spec.treatment_offset()?;
    let mut leaves = Vec::with_capacity(seq.steps.len());
    for step in seq.steps.iter() {
        if matches!(step.intervention, Intervention::Sequence(_)) {
            return Err(EstimationError::unsupported(
                "Intervention::Sequence nested inside a Sequence is not licensed for \
                 temporal InterventionResponse",
            ));
        }
        let (variable, level, shift) = resolve_one(&step.intervention, depth + 1)?;
        let offsets = step.temporal.active_offsets()?;
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
        let overlays = resolved
            .iter()
            .map(|&offset| SequentialNodeOverlay {
                variable: *variable,
                offset,
                level: *level,
                shift: *shift,
            })
            .collect();
        return Ok(TemporalInterventionPlan::Sequential { overlays });
    }
    let offsets = sequence_overlay_offsets(origin, &leaves)?;
    let mut overlays = Vec::with_capacity(leaves.len());
    let mut seen = Vec::new();
    for ((variable, level, shift, _), offset) in leaves.iter().zip(offsets) {
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
    Ok(TemporalInterventionPlan::Sequential { overlays })
}

/// Multi-step same variable → consecutive times ending at the spec origin.
/// Distinct variables, each once → joint at the origin.
/// Explicit distinct Pulse offsets are honored.
fn sequence_overlay_offsets(
    origin: i32,
    leaves: &[SequenceLeaf],
) -> Result<Vec<i32>, EstimationError> {
    let n = i32::try_from(leaves.len())
        .map_err(|_| EstimationError::unsupported("Sequence is too long"))?;
    let mut explicit = Vec::with_capacity(leaves.len());
    for (_, _, _, offsets) in leaves {
        let [at] = offsets.as_ref() else {
            return Err(EstimationError::unsupported(
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
        return Ok((0..n).map(|i| origin - (n - 1 - i)).collect());
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
