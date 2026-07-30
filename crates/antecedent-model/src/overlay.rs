//! Intervention overlays on an immutable compiled plan.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{
    Intervention, InterventionSequence, MechanismOverride, StochasticPolicy, TemporalPolicy,
    VariableId,
};
use antecedent_graph::DenseNodeId;

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

    /// Reject a node carrying both a hard set and an additive shift.
    ///
    /// A hard set replaces a node's assignment outright, so an additive shift on the
    /// same node has nothing well-defined to add to: the mechanism it would shift is
    /// exactly the one the set discarded. Rather than pick a winner silently, both
    /// overlay construction and the overlay-accepting samplers refuse the pair and
    /// make the caller say which they meant.
    ///
    /// Called for you by [`Self::from_interventions`] and [`Self::from_sequence_at`];
    /// call it yourself if you populate the public fields directly.
    ///
    /// # Errors
    ///
    /// [`ModelError::Unsupported`] naming the first offending dense node index.
    pub fn validate(&self) -> Result<(), ModelError> {
        for (idx, set) in self.hard_set.iter().enumerate() {
            if set.is_some() && self.shifts.get(idx).is_some_and(|s| *s != 0.0) {
                return Err(ModelError::Unsupported {
                    message: format!(
                        "dense node {idx} carries both a hard set and an additive shift; \
                         a variable cannot be simultaneously pinned (do(X := v)) and shifted \
                         (do(X := X + delta))"
                    ),
                });
            }
        }
        Ok(())
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
    /// Unknown variables, invalid interventions, or a node both set and shifted
    /// (see [`Self::validate`]).
    pub fn from_interventions(
        model: &CompiledCausalModel,
        interventions: &[Intervention],
    ) -> Result<Self, ModelError> {
        let mut overlay = Self::observational(model.n_nodes());
        for iv in interventions {
            apply_intervention(model, &mut overlay, iv, true)?;
        }
        // Checked once here rather than per-variant inside `apply_intervention`: the
        // conflict is order-independent, and any future shift-producing variant is
        // covered without having to remember to re-add the check.
        overlay.validate()?;
        Ok(overlay)
    }

    /// Overlay for a temporal sequence at discrete step `t`.
    ///
    /// # Errors
    ///
    /// Invalid sequence, unknown variables, or a node both set and shifted at `t`
    /// (see [`Self::validate`]).
    pub fn from_sequence_at(
        model: &CompiledCausalModel,
        seq: &InterventionSequence,
        t: i32,
    ) -> Result<Self, ModelError> {
        let mut overlay = Self::observational(model.n_nodes());
        for step in seq.steps.iter() {
            if temporal_active(&step.temporal, t)? {
                apply_intervention(model, &mut overlay, &step.intervention, true)?;
            }
        }
        // Only steps active at `t` compose, so a set and a shift that never overlap in
        // time remain legal — the conflict is evaluated per step, not across the sequence.
        overlay.validate()?;
        Ok(overlay)
    }
}

fn temporal_active(policy: &TemporalPolicy, t: i32) -> Result<bool, ModelError> {
    policy.validate().map_err(|e| ModelError::Unsupported { message: e.to_string() })?;
    Ok(policy.is_active_at(t))
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
                if temporal_active(&step.temporal, 0)? {
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
    use antecedent_core::{
        DynamicRuleId, Intervention, InterventionSequence, SequencedIntervention, TemporalPolicy,
        Value, VariableId,
    };
    use antecedent_graph::Dag;

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

    #[test]
    fn shift_overlay_accumulates_and_is_independent_of_hard_set() {
        let g = Dag::with_variables(2);
        let model = CompiledCausalModel::compile(g).unwrap();
        let x = VariableId::from_raw(0);
        let y = VariableId::from_raw(1);
        let overlay = InterventionOverlay::from_interventions(
            &model,
            &[Intervention::shift(x, Value::f64(1.5)), Intervention::shift(x, Value::f64(0.5))],
        )
        .unwrap();
        // Multiple shifts on the same variable accumulate additively.
        assert!((overlay.shifts[0] - 2.0).abs() < 1e-12);
        assert!(overlay.hard_set[0].is_none());
        // An unrelated variable is untouched.
        assert!(overlay.shifts[1].abs() < 1e-12);
        assert!(overlay.hard_set[y.as_usize()].is_none());
    }

    /// Order-independence matters: `apply_intervention` writes `hard_set` and `shifts`
    /// into separate arrays, so neither one "sees" the other as it lands.
    #[test]
    fn set_and_shift_on_same_variable_rejected_in_either_order() {
        let g = Dag::with_variables(2);
        let model = CompiledCausalModel::compile(g).unwrap();
        let x = VariableId::from_raw(0);

        for ivs in [
            vec![Intervention::set(x, Value::f64(1.0)), Intervention::shift(x, Value::f64(0.5))],
            vec![Intervention::shift(x, Value::f64(0.5)), Intervention::set(x, Value::f64(1.0))],
        ] {
            let err = InterventionOverlay::from_interventions(&model, &ivs).unwrap_err();
            assert!(
                matches!(&err, ModelError::Unsupported { message }
                    if message.contains("hard set") && message.contains("additive shift")),
                "unexpected error: {err}"
            );
        }
    }

    /// `Intervention::Soft` with the `additive_shift` family folds into `shifts`, so it
    /// collides with a hard set exactly like `Intervention::Shift` does.
    #[test]
    fn additive_shift_soft_override_collides_with_hard_set() {
        let g = Dag::with_variables(1);
        let model = CompiledCausalModel::compile(g).unwrap();
        let x = VariableId::from_raw(0);
        let err = InterventionOverlay::from_interventions(
            &model,
            &[
                Intervention::set(x, Value::f64(2.0)),
                Intervention::soft(x, MechanismOverride::additive_shift(0.25)),
            ],
        )
        .unwrap_err();
        assert!(matches!(err, ModelError::Unsupported { .. }), "unexpected error: {err}");
    }

    /// A zero shift is not a shift: it leaves the assignment untouched, so pairing it
    /// with a set discards nothing and stays legal.
    #[test]
    fn zero_shift_alongside_hard_set_is_allowed() {
        let g = Dag::with_variables(1);
        let model = CompiledCausalModel::compile(g).unwrap();
        let x = VariableId::from_raw(0);
        let overlay = InterventionOverlay::from_interventions(
            &model,
            &[Intervention::shift(x, Value::f64(0.0)), Intervention::set(x, Value::f64(3.0))],
        )
        .unwrap();
        assert_eq!(overlay.hard_set[0], Some(3.0));
    }

    /// Shifts on a *different* variable never conflict — the check is per node, not global.
    #[test]
    fn set_and_shift_on_different_variables_compose() {
        let g = Dag::with_variables(2);
        let model = CompiledCausalModel::compile(g).unwrap();
        let x = VariableId::from_raw(0);
        let y = VariableId::from_raw(1);
        let overlay = InterventionOverlay::from_interventions(
            &model,
            &[Intervention::set(x, Value::f64(1.0)), Intervention::shift(y, Value::f64(0.5))],
        )
        .unwrap();
        assert_eq!(overlay.hard_set[0], Some(1.0));
        assert!((overlay.shifts[1] - 0.5).abs() < 1e-12);
    }

    /// Sequences compose only the steps active at `t`, so a set and a shift scheduled on
    /// disjoint steps are legal at every `t` even though they name the same variable.
    #[test]
    fn sequence_set_and_shift_conflict_only_when_steps_overlap() {
        let g = Dag::with_variables(1);
        let model = CompiledCausalModel::compile(g).unwrap();
        let x = VariableId::from_raw(0);

        let disjoint = InterventionSequence::new(vec![
            SequencedIntervention::new(
                Intervention::set(x, Value::f64(1.0)),
                TemporalPolicy::dynamic(DynamicRuleId::from_raw(0), [0]),
            ),
            SequencedIntervention::new(
                Intervention::shift(x, Value::f64(0.5)),
                TemporalPolicy::dynamic(DynamicRuleId::from_raw(1), [1]),
            ),
        ]);
        assert_eq!(
            InterventionOverlay::from_sequence_at(&model, &disjoint, 0).unwrap().hard_set[0],
            Some(1.0)
        );
        let at_one = InterventionOverlay::from_sequence_at(&model, &disjoint, 1).unwrap();
        assert!(at_one.hard_set[0].is_none());
        assert!((at_one.shifts[0] - 0.5).abs() < 1e-12);

        let overlapping = InterventionSequence::new(vec![
            SequencedIntervention::new(
                Intervention::set(x, Value::f64(1.0)),
                TemporalPolicy::dynamic(DynamicRuleId::from_raw(0), [0]),
            ),
            SequencedIntervention::new(
                Intervention::shift(x, Value::f64(0.5)),
                TemporalPolicy::dynamic(DynamicRuleId::from_raw(1), [0]),
            ),
        ]);
        assert!(InterventionOverlay::from_sequence_at(&model, &overlapping, 0).is_err());
    }

    /// The fields are public, so an overlay can be built without going through
    /// `from_interventions`. `validate` is what such a caller has to reach for.
    #[test]
    fn hand_built_overlay_validates() {
        let mut overlay = InterventionOverlay::observational(2);
        assert!(overlay.validate().is_ok());
        overlay.hard_set[0] = Some(1.0);
        assert!(overlay.validate().is_ok());
        overlay.shifts[0] = 0.5;
        let err = overlay.validate().unwrap_err();
        assert!(
            matches!(&err, ModelError::Unsupported { message } if message.contains("dense node 0")),
            "error should name the offending node: {err}"
        );
    }

    #[test]
    fn dynamic_policy_sequence_activates_on_schedule() {
        let g = Dag::with_variables(1);
        let model = CompiledCausalModel::compile(g).unwrap();
        let t = VariableId::from_raw(0);
        let seq = InterventionSequence::new(vec![SequencedIntervention::new(
            Intervention::set(t, Value::f64(1.0)),
            TemporalPolicy::dynamic(DynamicRuleId::from_raw(0), [0, 2]),
        )]);
        let overlay = InterventionOverlay::from_sequence_at(&model, &seq, 0).unwrap();
        assert_eq!(overlay.hard_set[0], Some(1.0));
        let idle = InterventionOverlay::from_sequence_at(&model, &seq, 1).unwrap();
        assert!(idle.hard_set[0].is_none());
        let again = InterventionOverlay::from_sequence_at(&model, &seq, 2).unwrap();
        assert_eq!(again.hard_set[0], Some(1.0));
    }
}
