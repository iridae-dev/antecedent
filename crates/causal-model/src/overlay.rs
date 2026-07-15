//! Intervention overlays on an immutable compiled plan (DESIGN.md §15.4).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use causal_core::{
    Intervention, InterventionSequence, MechanismOverride, StochasticPolicy, TemporalPolicy,
    VariableId,
};
use causal_graph::DenseNodeId;

use crate::compile::CompiledCausalModel;
use crate::error::ModelError;

/// Compact overlay describing how sampling differs from the observational plan.
///
/// The underlying [`CompiledCausalModel`] is never cloned; overlays are applied
/// during ancestral sampling.
#[derive(Clone, Debug, Default)]
pub struct InterventionOverlay {
    /// Per-node hard sets (dense index → value), `None` = not hard-set.
    pub hard_set: Vec<Option<f64>>,
    /// Per-node additive shifts.
    pub shifts: Vec<f64>,
    /// Per-node stochastic policies.
    pub stochastic: Vec<Option<StochasticPolicy>>,
    /// Per-node soft mechanism overrides.
    pub soft: Vec<Option<MechanismOverride>>,
    /// Optional temporal activation mask per node (`true` = active at current step).
    pub active: Vec<bool>,
}

impl InterventionOverlay {
    /// Empty overlay (observational) for `n_nodes`.
    #[must_use]
    pub fn observational(n_nodes: usize) -> Self {
        Self {
            hard_set: vec![None; n_nodes],
            shifts: vec![0.0; n_nodes],
            stochastic: vec![None; n_nodes],
            soft: vec![None; n_nodes],
            active: vec![true; n_nodes],
        }
    }

    /// Whether any node is intervened.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.hard_set.iter().all(Option::is_none)
            && self.shifts.iter().all(|s| *s == 0.0)
            && self.stochastic.iter().all(Option::is_none)
            && self.soft.iter().all(Option::is_none)
            && self.active.iter().all(|a| *a)
    }

    /// Compile interventions against a model (simultaneous / single-step).
    ///
    /// # Errors
    ///
    /// Unknown variables or invalid interventions.
    pub fn from_interventions(
        model: &CompiledCausalModel,
        interventions: &[Intervention],
    ) -> Result<Self, ModelError> {
        let mut overlay = Self::observational(model.n_nodes());
        for iv in interventions {
            apply_intervention(model, &mut overlay, iv, true)?;
        }
        Ok(overlay)
    }

    /// Overlay for a temporal sequence at discrete step `t`.
    ///
    /// # Errors
    ///
    /// Invalid sequence or unknown variables.
    pub fn from_sequence_at(
        model: &CompiledCausalModel,
        seq: &InterventionSequence,
        t: i32,
    ) -> Result<Self, ModelError> {
        let mut overlay = Self::observational(model.n_nodes());
        for step in seq.steps.iter() {
            if temporal_active(step.temporal, t) {
                apply_intervention(model, &mut overlay, &step.intervention, true)?;
            }
        }
        Ok(overlay)
    }
}

fn temporal_active(policy: TemporalPolicy, t: i32) -> bool {
    match policy {
        TemporalPolicy::Pulse { at } => t == at,
        TemporalPolicy::Sustained { from, until } => t >= from && t <= until,
        _ => false,
    }
}

fn apply_intervention(
    model: &CompiledCausalModel,
    overlay: &mut InterventionOverlay,
    iv: &Intervention,
    allow_nested_sequence: bool,
) -> Result<(), ModelError> {
    match iv {
        Intervention::Set { variable, value } => {
            let dense = require_dense(model, *variable)?;
            let v = value.as_f64().ok_or_else(|| ModelError::Unsupported {
                message: "hard set requires numeric value".into(),
            })?;
            overlay.hard_set[dense.as_usize()] = Some(v);
            Ok(())
        }
        Intervention::Shift { variable, delta } => {
            let dense = require_dense(model, *variable)?;
            let d = delta.as_f64().ok_or_else(|| ModelError::Unsupported {
                message: "shift requires numeric delta".into(),
            })?;
            overlay.shifts[dense.as_usize()] += d;
            Ok(())
        }
        Intervention::Stochastic { variable, policy } => {
            policy.validate().map_err(|e| ModelError::Unsupported { message: e.to_string() })?;
            let dense = require_dense(model, *variable)?;
            overlay.stochastic[dense.as_usize()] = Some(policy.clone());
            Ok(())
        }
        Intervention::Soft { variable, mechanism } => {
            let dense = require_dense(model, *variable)?;
            // Unify with `Intervention::Shift`: additive soft overrides are shifts, so
            // ancestral and structural sampling share the same noise semantics.
            if mechanism.family_id.as_ref() == "additive_shift" {
                let d = mechanism.parameters.first().copied().unwrap_or(0.0);
                overlay.shifts[dense.as_usize()] += d;
                return Ok(());
            }
            overlay.soft[dense.as_usize()] = Some(mechanism.clone());
            Ok(())
        }
        Intervention::Sequence(seq) => {
            if !allow_nested_sequence {
                return Err(ModelError::Unsupported {
                    message: "nested intervention sequences are not supported here".into(),
                });
            }
            // Simultaneous interpretation at t=0 for static models.
            for step in seq.steps.iter() {
                if temporal_active(step.temporal, 0) {
                    apply_intervention(model, overlay, &step.intervention, false)?;
                }
            }
            Ok(())
        }
        _ => Err(ModelError::Unsupported { message: "unknown intervention variant".into() }),
    }
}

fn require_dense(model: &CompiledCausalModel, var: VariableId) -> Result<DenseNodeId, ModelError> {
    model.dense_of(var).ok_or_else(|| ModelError::Shape {
        message: format!("variable {var} not in compiled model"),
    })
}

/// Shared immutable model plus overlay (no model clone).
#[derive(Clone, Debug)]
pub struct ModelView<'a> {
    /// Borrowed compiled plan.
    pub model: &'a CompiledCausalModel,
    /// Intervention overlay.
    pub overlay: Arc<InterventionOverlay>,
}

impl<'a> ModelView<'a> {
    /// Observational view.
    #[must_use]
    pub fn observational(model: &'a CompiledCausalModel) -> Self {
        Self { model, overlay: Arc::new(InterventionOverlay::observational(model.n_nodes())) }
    }

    /// Interventional view.
    #[must_use]
    pub fn with_overlay(model: &'a CompiledCausalModel, overlay: InterventionOverlay) -> Self {
        Self { model, overlay: Arc::new(overlay) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use causal_core::{Intervention, Value, VariableId};
    use causal_graph::Dag;

    #[test]
    fn hard_set_overlay() {
        let g = Dag::with_variables(2);
        let model = CompiledCausalModel::compile(g).unwrap();
        let t = VariableId::from_raw(0);
        let overlay = InterventionOverlay::from_interventions(
            &model,
            &[Intervention::set(t, Value::f64(1.0))],
        )
        .unwrap();
        assert_eq!(overlay.hard_set[0], Some(1.0));
        assert!(overlay.hard_set[1].is_none());
    }
}
