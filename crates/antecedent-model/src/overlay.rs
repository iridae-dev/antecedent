//! Intervention overlays on an immutable compiled plan.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::intervention::{AdmitError, ExecutionSurface};
use antecedent_core::{
    Intervention, InterventionSequence, MechanismOverride, PolicyReads, StochasticPolicy,
    TemporalPolicy, Value, ValueExpr, VariableId,
};
use antecedent_graph::DenseNodeId;

use crate::compile::{CompiledCausalModel, ParentGatherPlan};
use crate::error::ModelError;
use crate::program::{NodeProgram, PolicyRefusal, ProgramKind, execution_order};

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
    /// Per-node value programs (transform of the generated value, or assignment). A
    /// vector shorter than the node count means "no program" for the missing nodes.
    pub programs: Vec<Option<NodeProgram>>,
    /// Evaluation order (indices into the model's parent gathers) once program reads are
    /// added to the graph's own edges; `None` keeps the model's topological order.
    pub order: Option<Arc<[u32]>>,
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
            programs: vec![None; n_nodes],
            order: None,
        }
    }

    /// Derive [`Self::order`] from the programs' reads and the model's parent edges.
    ///
    /// Called for you by [`Self::from_interventions`] and [`Self::from_sequence_at`]; call
    /// it yourself after populating [`Self::programs`] directly.
    ///
    /// # Errors
    ///
    /// A circular-dependency refusal naming the cycle: a program reads a variable that
    /// depends on its own node, directly or through other programs.
    pub fn plan_order(&mut self, model: &CompiledCausalModel) -> Result<(), ModelError> {
        self.order = execution_order(model, &self.programs).map_err(PolicyRefusal::into_error)?;
        Ok(())
    }

    /// The program on dense node `idx`, if any.
    #[must_use]
    pub fn program_at(&self, idx: usize) -> Option<&NodeProgram> {
        self.programs.get(idx).and_then(Option::as_ref)
    }

    /// The model's parent gathers in the order this overlay evaluates them: the model's
    /// own topological order, or the dependency order that places every node after the
    /// variables its program reads.
    pub fn gathers<'a>(
        &'a self,
        model: &'a CompiledCausalModel,
    ) -> impl Iterator<Item = &'a ParentGatherPlan> + 'a {
        (0..model.parent_gathers.len()).map(move |i| {
            let at = self.order.as_ref().and_then(|o| o.get(i)).map_or(i, |g| *g as usize);
            &model.parent_gathers[at]
        })
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
        // No evaluation path reads the activation mask, so a node declared inactive would be
        // evaluated as if it were active. Refuse the declaration instead of ignoring it.
        if let Some(idx) = self.active.iter().position(|a| !*a) {
            return Err(ModelError::Unsupported {
                message: format!(
                    "dense node {idx} is declared inactive, but temporal activation masks are \
                     not supported by evaluation and would be silently ignored"
                ),
            });
        }
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
        // Replacement directives (hard set, stochastic policy, soft mechanism) each
        // define the node's whole assignment, so two of them on one node have no
        // well-defined combination; evaluation order would silently pick a winner.
        for idx in 0..self.hard_set.len() {
            let kinds = [
                self.hard_set[idx].is_some(),
                self.stochastic.get(idx).is_some_and(Option::is_some),
                self.soft.get(idx).is_some_and(Option::is_some),
            ];
            if kinds.iter().filter(|k| **k).count() > 1 {
                return Err(conflicting_directives(idx));
            }
        }
        // A program composes with at most a stochastic policy or soft mechanism (a
        // transform then acts on the value they generate). Anything else either replaces
        // the generated value (hard set), leaves the order of two edits undefined (a shift
        // before or after the program), or defines the assignment a second time.
        for (idx, program) in self.programs.iter().enumerate() {
            let Some(program) = program else { continue };
            let assigns = program.kind() == ProgramKind::Assign;
            let other = if self.hard_set.get(idx).is_some_and(Option::is_some) {
                Some("hard set")
            } else if self.shifts.get(idx).is_some_and(|s| *s != 0.0) {
                Some("additive shift")
            } else if assigns && self.stochastic.get(idx).is_some_and(Option::is_some) {
                Some("stochastic policy")
            } else if assigns && self.soft.get(idx).is_some_and(Option::is_some) {
                Some("soft mechanism")
            } else {
                None
            };
            if let Some(other) = other {
                let kind = if assigns { "assignment" } else { "transform" };
                return Err(program_conflict(format!(
                    "dense node {idx} carries an {kind} program together with a {other}"
                )));
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
            && self.programs.iter().all(Option::is_none)
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
            admit_static(iv, None)?;
            apply_intervention(model, &mut overlay, iv, true, 0)?;
        }
        // Checked once here rather than per-variant inside `apply_intervention`: the
        // conflict is order-independent, and any future shift-producing variant is
        // covered without having to remember to re-add the check.
        overlay.validate()?;
        overlay.plan_order(model)?;
        Ok(overlay)
    }

    /// Execute one intervention on a static model.
    ///
    /// `temporal` of `None` is the untimed intervention. A pulse, a sustained
    /// window, and an initial-state change are refused. A shift stays a shift
    /// and a soft replacement stays a mechanism replacement.
    ///
    /// # Errors
    ///
    /// [`ModelError::Unsupported`] naming the refused operation, or the errors
    /// of [`Self::from_interventions`].
    pub fn from_declared(
        model: &CompiledCausalModel,
        intervention: &Intervention,
        temporal: Option<&TemporalPolicy>,
    ) -> Result<Self, ModelError> {
        admit_static(intervention, temporal)?;
        Self::from_interventions(model, std::slice::from_ref(intervention))
    }

    /// Refuse an initial-state change. A static model has no initial-state law
    /// to replace, and the change is not applied as a hard set.
    ///
    /// # Errors
    ///
    /// [`ModelError::Unsupported`] naming `initial_state_change`.
    pub fn from_initial_state(
        model: &CompiledCausalModel,
        law_name: &str,
        level: f64,
    ) -> Result<Self, ModelError> {
        let _ = model;
        let operation = antecedent_core::intervention::InterventionOperation::initial_state(
            law_name,
            Value::f64(level),
        )
        .map_err(|error| ModelError::Unsupported { message: error.to_string() })?;
        ExecutionSurface::Static
            .admit_operation(&operation)
            .map_err(|refusal| ModelError::Unsupported { message: refusal.to_string() })?;
        Err(ModelError::Unsupported {
            message: "an initial-state change replaces the initial-state law and is not a \
                      hard clamp"
                .into(),
        })
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
        Self::from_sequence_with_history(model, seq, t, 0)
    }

    /// [`Self::from_sequence_at`] for a temporal route that keeps `window` steps of history,
    /// so a value program active at `t` may read a variable up to `window` steps back.
    ///
    /// # Errors
    ///
    /// As [`Self::from_sequence_at`]; a history read deeper than `window` is refused.
    pub fn from_sequence_with_history(
        model: &CompiledCausalModel,
        seq: &InterventionSequence,
        t: i32,
        window: u32,
    ) -> Result<Self, ModelError> {
        let mut overlay = Self::observational(model.n_nodes());
        for step in seq.steps.iter() {
            if temporal_active(&step.temporal, t)? {
                apply_intervention(model, &mut overlay, &step.intervention, true, window)?;
            }
        }
        // Only steps active at `t` compose, so a set and a shift that never overlap in
        // time remain legal — the conflict is evaluated per step, not across the sequence.
        overlay.validate()?;
        overlay.plan_order(model)?;
        Ok(overlay)
    }
}

fn admit_static(
    intervention: &Intervention,
    temporal: Option<&TemporalPolicy>,
) -> Result<(), ModelError> {
    ExecutionSurface::Static.admit(intervention, temporal).map_err(|error| match error {
        AdmitError::Refused(refusal) => ModelError::Unsupported { message: refusal.to_string() },
        AdmitError::Invalid(invalid) => ModelError::Unsupported { message: invalid.to_string() },
        other => ModelError::Unsupported { message: other.to_string() },
    })?;
    Ok(())
}

fn program_conflict(subject: String) -> ModelError {
    crate::program::refusal_directive_conflict(subject).into_error()
}

fn conflicting_directives(idx: usize) -> ModelError {
    ModelError::Unsupported {
        message: format!(
            "dense node {idx} is targeted by two conflicting interventions (a hard set, a \
             stochastic policy or a soft mechanism replaces the node's whole assignment); \
             state one, or make the repeated directive identical"
        ),
    }
}

fn temporal_active(policy: &TemporalPolicy, t: i32) -> Result<bool, ModelError> {
    policy.validate().map_err(|e| ModelError::Unsupported { message: e.to_string() })?;
    Ok(policy.is_active_at(t))
}

/// Compose a finite additive shift, refusing a composition that overflows.
fn add_shift(overlay: &mut InterventionOverlay, idx: usize, delta: f64) -> Result<(), ModelError> {
    let total = overlay.shifts[idx] + delta;
    if !total.is_finite() {
        return Err(ModelError::Unsupported {
            message: "composed shifts overflow to a non-finite delta".into(),
        });
    }
    overlay.shifts[idx] = total;
    Ok(())
}

#[allow(
    clippy::float_cmp,
    reason = "a repeated identical set is idempotent: exact equality is intended"
)]
fn apply_intervention(
    model: &CompiledCausalModel,
    overlay: &mut InterventionOverlay,
    iv: &Intervention,
    allow_nested_sequence: bool,
    window: u32,
) -> Result<(), ModelError> {
    match iv {
        Intervention::Set { variable, value } => {
            let dense = require_dense(model, *variable)?;
            let v = value.as_f64().ok_or_else(|| ModelError::Unsupported {
                message: "hard set requires numeric value".into(),
            })?;
            let idx = dense.as_usize();
            // A repeated identical set is idempotent; a different value would silently
            // overwrite the first.
            if overlay.hard_set[idx].is_some_and(|prev| prev != v)
                || overlay.stochastic[idx].is_some()
                || overlay.soft[idx].is_some()
            {
                return Err(conflicting_directives(idx));
            }
            overlay.hard_set[idx] = Some(v);
            Ok(())
        }
        Intervention::Shift { variable, delta } => {
            let dense = require_dense(model, *variable)?;
            let d = delta.as_f64().filter(|d| d.is_finite()).ok_or_else(|| {
                ModelError::Unsupported { message: "shift requires a finite numeric delta".into() }
            })?;
            add_shift(overlay, dense.as_usize(), d)
        }
        Intervention::Stochastic { variable, policy } => {
            policy.validate().map_err(|e| ModelError::Unsupported { message: e.to_string() })?;
            let dense = require_dense(model, *variable)?;
            let idx = dense.as_usize();
            if overlay.hard_set[idx].is_some()
                || overlay.soft[idx].is_some()
                || overlay.stochastic[idx].as_ref().is_some_and(|prev| prev != policy)
            {
                return Err(conflicting_directives(idx));
            }
            overlay.stochastic[idx] = Some(policy.clone());
            Ok(())
        }
        Intervention::Soft { variable, mechanism } => {
            let dense = require_dense(model, *variable)?;
            let idx = dense.as_usize();
            // Unify with `Intervention::Shift`: additive soft overrides are shifts, so
            // ancestral and structural sampling share the same noise semantics.
            if mechanism.family_id.as_ref() == "additive_shift" {
                let d = match &mechanism.parameters[..] {
                    [d] if d.is_finite() => *d,
                    _ => {
                        return Err(ModelError::Unsupported {
                            message: "additive_shift override needs exactly one finite delta"
                                .into(),
                        });
                    }
                };
                return add_shift(overlay, idx, d);
            }
            if overlay.hard_set[idx].is_some()
                || overlay.stochastic[idx].is_some()
                || overlay.soft[idx].as_ref().is_some_and(|prev| prev != mechanism)
            {
                return Err(conflicting_directives(idx));
            }
            overlay.soft[idx] = Some(mechanism.clone());
            Ok(())
        }
        Intervention::Transform { variable, expr, reads } => {
            apply_program(model, overlay, *variable, ProgramKind::Transform, expr, reads, window)
        }
        Intervention::Assign { variable, expr, reads } => {
            apply_program(model, overlay, *variable, ProgramKind::Assign, expr, reads, window)
        }
        Intervention::Sequence(seq) => {
            if !allow_nested_sequence {
                return Err(ModelError::Unsupported {
                    message: "nested intervention sequences are not supported here".into(),
                });
            }
            // A static model has a single time point, t = 0. Steps active there compose as
            // one simultaneous intervention; a step scheduled for another time has no
            // meaning in a static evaluation and dropping it would report a counterfactual
            // that omits part of the request.
            for step in seq.steps.iter() {
                if !temporal_active(&step.temporal, 0)? {
                    return Err(ModelError::Unsupported {
                        message: "an intervention sequence has steps not active at t = 0; a \
                                  static evaluation has only t = 0, so those steps cannot be \
                                  applied (evaluate the sequence per time step instead)"
                            .into(),
                    });
                }
                apply_intervention(model, overlay, &step.intervention, false, window)?;
            }
            Ok(())
        }
        _ => Err(ModelError::Unsupported { message: "unknown intervention variant".into() }),
    }
}

fn apply_program(
    model: &CompiledCausalModel,
    overlay: &mut InterventionOverlay,
    variable: VariableId,
    kind: ProgramKind,
    expr: &ValueExpr,
    reads: &PolicyReads,
    window: u32,
) -> Result<(), ModelError> {
    let dense = require_dense(model, variable)?;
    let program = NodeProgram::compile_windowed(model, variable, kind, expr, reads, window)
        .map_err(PolicyRefusal::into_error)?;
    let idx = dense.as_usize();
    // A repeated identical program is idempotent; a different one on the same node has no
    // well-defined composition.
    if overlay.programs[idx].as_ref().is_some_and(|prev| prev != &program) {
        return Err(program_conflict(format!(
            "variable {variable} is targeted by two different value programs"
        )));
    }
    overlay.programs[idx] = Some(program);
    Ok(())
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

    #[test]
    fn non_finite_shift_is_refused() {
        let model = CompiledCausalModel::compile(Dag::with_variables(1)).unwrap();
        let x = VariableId::from_raw(0);
        for delta in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let err = InterventionOverlay::from_interventions(
                &model,
                &[Intervention::shift(x, Value::f64(delta))],
            )
            .unwrap_err();
            assert!(matches!(err, ModelError::Unsupported { .. }), "{delta}: {err}");
        }
        let err = InterventionOverlay::from_interventions(
            &model,
            &[
                Intervention::shift(x, Value::f64(f64::MAX)),
                Intervention::shift(x, Value::f64(f64::MAX)),
            ],
        )
        .unwrap_err();
        assert!(matches!(err, ModelError::Unsupported { .. }), "{err}");
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

    /// Two different hard sets, or a set beside a soft mechanism or stochastic policy,
    /// used to resolve by evaluation order with no error; an identical repeat is fine.
    #[test]
    fn conflicting_replacement_directives_on_one_node_are_rejected() {
        let g = Dag::with_variables(1);
        let model = CompiledCausalModel::compile(g).unwrap();
        let x = VariableId::from_raw(0);
        let conflicts = [
            vec![Intervention::set(x, Value::f64(1.0)), Intervention::set(x, Value::f64(2.0))],
            vec![
                Intervention::set(x, Value::f64(1.0)),
                Intervention::soft(x, MechanismOverride::constant(3.0)),
            ],
            vec![
                Intervention::soft(x, MechanismOverride::constant(3.0)),
                Intervention::set(x, Value::f64(1.0)),
            ],
            vec![
                Intervention::soft(x, MechanismOverride::constant(3.0)),
                Intervention::soft(x, MechanismOverride::constant(4.0)),
            ],
        ];
        for ivs in conflicts {
            let err = InterventionOverlay::from_interventions(&model, &ivs).unwrap_err();
            assert!(
                matches!(&err, ModelError::Unsupported { message }
                    if message.contains("conflicting interventions")),
                "unexpected error: {err}"
            );
        }
        let repeated = InterventionOverlay::from_interventions(
            &model,
            &[Intervention::set(x, Value::f64(1.0)), Intervention::set(x, Value::f64(1.0))],
        )
        .unwrap();
        assert_eq!(repeated.hard_set[0], Some(1.0));
    }

    /// A hand-built overlay carrying two replacement directives on one node fails
    /// `validate`.
    #[test]
    fn hand_built_overlay_with_two_replacements_fails_validate() {
        let mut overlay = InterventionOverlay::observational(1);
        overlay.hard_set[0] = Some(1.0);
        overlay.soft[0] = Some(MechanismOverride::constant(2.0));
        assert!(overlay.validate().is_err());
    }

    /// A static evaluation has only t = 0; a sequence step scheduled later must not be
    /// dropped silently.
    #[test]
    fn static_sequence_with_a_step_after_t0_is_refused() {
        let g = Dag::with_variables(2);
        let model = CompiledCausalModel::compile(g).unwrap();
        let (x, y) = (VariableId::from_raw(0), VariableId::from_raw(1));
        let seq = Intervention::sequence(InterventionSequence::new(vec![
            SequencedIntervention::new(
                Intervention::set(x, Value::f64(1.0)),
                TemporalPolicy::dynamic(DynamicRuleId::from_raw(0), [0]),
            ),
            SequencedIntervention::new(
                Intervention::set(y, Value::f64(2.0)),
                TemporalPolicy::dynamic(DynamicRuleId::from_raw(1), [1]),
            ),
        ]));
        let err = InterventionOverlay::from_interventions(&model, &[seq]).unwrap_err();
        assert!(
            matches!(&err, ModelError::Unsupported { message } if message.contains("t = 0")),
            "{err}"
        );
    }

    /// The additive-shift family needs exactly one finite delta; an empty or NaN list
    /// used to become a zero shift.
    #[test]
    fn additive_shift_override_requires_a_finite_delta() {
        let g = Dag::with_variables(1);
        let model = CompiledCausalModel::compile(g).unwrap();
        let x = VariableId::from_raw(0);
        for parameters in [vec![], vec![f64::NAN], vec![1.0, 2.0]] {
            let iv = Intervention::soft(x, MechanismOverride::named("additive_shift", parameters));
            assert!(InterventionOverlay::from_interventions(&model, &[iv]).is_err());
        }
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

    #[test]
    fn a_static_set_matches_the_cyclic_operation_and_other_actions_are_not_clamps() {
        let g = Dag::with_variables(1);
        let model = CompiledCausalModel::compile(g).unwrap();
        let variable = VariableId::from_raw(0);
        let set = Intervention::set(variable, Value::f64(2.0));
        let static_op =
            antecedent_core::intervention::ExecutionSurface::Static.admit(&set, None).unwrap();
        let cyclic_op =
            antecedent_core::intervention::ExecutionSurface::Cyclic.admit(&set, None).unwrap();
        assert_eq!(static_op, cyclic_op);
        let overlay = InterventionOverlay::from_declared(&model, &set, None).unwrap();
        assert_eq!(overlay.hard_set[0], Some(2.0));
        assert_eq!(overlay.shifts[0].to_bits(), 0.0_f64.to_bits());
        assert!(overlay.soft[0].is_none());

        let shift = Intervention::shift(variable, Value::f64(2.0));
        let shifted = InterventionOverlay::from_declared(&model, &shift, None).unwrap();
        assert!(shifted.hard_set[0].is_none());
        assert!((shifted.shifts[0] - 2.0).abs() < 1e-12);
        let soft = Intervention::soft(variable, MechanismOverride::constant(2.0));
        let replaced = InterventionOverlay::from_declared(&model, &soft, None).unwrap();
        assert!(replaced.hard_set[0].is_none());
        assert!(replaced.soft[0].is_some());

        let pulse =
            InterventionOverlay::from_declared(&model, &set, Some(&TemporalPolicy::pulse(0)));
        assert!(pulse.unwrap_err().to_string().contains("pulse"));
        let sustained = InterventionOverlay::from_declared(
            &model,
            &set,
            Some(&TemporalPolicy::sustained(0, 0)),
        );
        assert!(sustained.unwrap_err().to_string().contains("sustained"));
        let initial = InterventionOverlay::from_initial_state(&model, "x0", 2.0);
        assert!(initial.unwrap_err().to_string().contains("initial_state_change"));
    }
}
