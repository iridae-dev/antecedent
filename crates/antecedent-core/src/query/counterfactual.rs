//! Query submodule.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::fault::InterventionFault;
use std::sync::Arc;

use crate::canonical_encode::{CanonicalEncode, encode_list, encode_variables, flag};
use crate::identity::CanonicalWriter;
use crate::ids::VariableId;
use crate::intervention::{Intervention, InterventionSequence};
use crate::value::Value;

use super::error::QueryError;

#[derive(Clone, Debug, PartialEq)]
/// Counterfactual query over factual observations and interventions.
pub struct CounterfactualQuery {
    /// Outcome variable(s) to predict under the counterfactual world.
    pub outcomes: Arc<[VariableId]>,
    /// Interventions defining the counterfactual (active) world after abduction.
    pub interventions: Arc<[Intervention]>,
    /// Control world for a two-world ITE (typically a hard set of treatment to 0).
    ///
    /// When [`Self::joint_control`] is non-empty this is the first intervention of a joint
    /// baseline; the whole baseline is [`Self::control_world`].
    pub control: Intervention,
    /// The further simultaneous interventions of a joint baseline, applied together with
    /// [`Self::control`] in the control world. Empty for the single-intervention baseline.
    pub joint_control: Arc<[Intervention]>,
    /// When true, allow nested counterfactual interventions under invertible SCMs.
    pub allow_nested: bool,
}

/// Tag written in place of the control intervention when the baseline is joint.
///
/// [`Intervention`] encodes a tag in `1..=7`, so a single-intervention baseline can never
/// begin with this word: the bytes of a query with one control are exactly those it always
/// had, and a joint baseline is a different canonical form.
const JOINT_CONTROL_TAG: u64 = 8;

/// Every field is written, in the order given: `outcomes` index the result,
/// and `interventions` keep their declared order (a later intervention on the
/// same variable is not assumed to commute with an earlier one). A joint baseline
/// is written as [`JOINT_CONTROL_TAG`] followed by the whole control world in order.
impl CanonicalEncode for CounterfactualQuery {
    fn encode(&self, writer: &mut CanonicalWriter) {
        encode_variables(writer, &self.outcomes);
        encode_list(writer, &self.interventions);
        if self.joint_control.is_empty() {
            self.control.encode(writer);
        } else {
            writer.word(JOINT_CONTROL_TAG);
            encode_list(writer, &self.control_world());
        }
        writer.word(flag(self.allow_nested));
    }
}

/// Every variable an intervention sets, shifts or replaces, in order of appearance
/// (a sequence contributes the variables of its steps).
fn intervened_variables(intervention: &Intervention, out: &mut Vec<VariableId>) {
    match intervention {
        Intervention::Sequence(sequence) => {
            for step in sequence.steps.iter() {
                intervened_variables(&step.intervention, out);
            }
        }
        other => out.extend(other.primary_variable()),
    }
}

impl CounterfactualQuery {
    /// Hard set of the unique first intervention target to `0.0`.
    ///
    /// When the active world is empty or its first intervention has multiple
    /// targets, returns an unresolved empty sequence. Validation refuses it
    /// until the caller supplies an explicit control. This also applies when
    /// format-0.4 query decoding supplies an omitted control.
    #[must_use]
    pub fn default_control(interventions: &[Intervention]) -> Intervention {
        match interventions.first().and_then(Intervention::primary_variable) {
            Some(variable) => Intervention::set(variable, Value::f64(0.0)),
            None => Intervention::sequence(InterventionSequence { steps: Arc::from([]) }),
        }
    }

    /// Construct a single-outcome counterfactual query.
    ///
    /// Control defaults to a hard set of the unique first intervention target to `0.0`;
    /// otherwise it remains unresolved until explicitly supplied.
    #[must_use]
    pub fn new(outcome: VariableId, interventions: impl Into<Arc<[Intervention]>>) -> Self {
        let interventions = interventions.into();
        let control = Self::default_control(&interventions);
        Self {
            outcomes: Arc::from([outcome]),
            interventions,
            control,
            joint_control: Arc::from([]),
            allow_nested: false,
        }
    }

    /// The whole control world: [`Self::control`] followed by [`Self::joint_control`].
    #[must_use]
    pub fn control_world(&self) -> Vec<Intervention> {
        let mut world = Vec::with_capacity(1 + self.joint_control.len());
        world.push(self.control.clone());
        world.extend(self.joint_control.iter().cloned());
        world
    }

    /// Every variable the active world intervenes on, in order of appearance.
    #[must_use]
    pub fn active_variables(&self) -> Vec<VariableId> {
        let mut out = Vec::new();
        for intervention in self.interventions.iter() {
            intervened_variables(intervention, &mut out);
        }
        out
    }

    /// Every variable the control world intervenes on, in order of appearance.
    #[must_use]
    pub fn control_variables(&self) -> Vec<VariableId> {
        let mut out = Vec::new();
        for intervention in self.control_world() {
            intervened_variables(&intervention, &mut out);
        }
        out
    }

    /// Replace the control world by a joint baseline of simultaneous interventions: the
    /// first becomes [`Self::control`], the rest [`Self::joint_control`].
    ///
    /// An empty baseline leaves the control world unchanged.
    #[must_use]
    pub fn with_joint_control(mut self, baseline: impl IntoIterator<Item = Intervention>) -> Self {
        let mut baseline = baseline.into_iter();
        if let Some(first) = baseline.next() {
            self.control = first;
            self.joint_control = baseline.collect();
        }
        self
    }

    /// Replace the control world (ITE baseline).
    #[must_use]
    pub fn with_control(mut self, control: Intervention) -> Self {
        self.control = control;
        self.joint_control = Arc::from([]);
        self
    }

    /// Hard-set the control level on the current control (or first intervention) target.
    ///
    /// This states a single-intervention baseline: a joint baseline is replaced.
    /// If neither the current control nor first intervention has a unique target,
    /// the new control remains unresolved until a target is supplied explicitly.
    #[must_use]
    pub fn with_control_level(self, level: f64) -> Self {
        let variable = self
            .control
            .primary_variable()
            .or_else(|| self.interventions.first().and_then(Intervention::primary_variable));
        match variable {
            Some(variable) => self.with_control(Intervention::set(variable, Value::f64(level))),
            None => self.with_control(Intervention::sequence(InterventionSequence {
                steps: Arc::from([]),
            })),
        }
    }

    /// Enable nested interventions where the model supports them.
    #[must_use]
    pub const fn with_nested(mut self, allow_nested: bool) -> Self {
        self.allow_nested = allow_nested;
        self
    }

    /// Validate interventions.
    ///
    /// # Errors
    ///
    /// Empty outcomes, an unresolved default, invalid interventions, or incompatible
    /// control targets. Explicit multi-target controls must match the active variables
    /// when the active world is nonempty.
    pub fn validate(&self) -> Result<(), QueryError> {
        if self.outcomes.is_empty() {
            return Err(QueryError::EmptyCounterfactualOutcomes);
        }
        for iv in self.interventions.iter() {
            iv.validate()
                .map_err(|e| QueryError::InvalidIntervention(InterventionFault::Rejected(e)))?;
        }
        let first = self.interventions.first();
        // An unresolved default is represented by an empty control sequence;
        // active intervention errors have already been reported above.
        if first.and_then(Intervention::primary_variable).is_none()
            && matches!(&self.control, Intervention::Sequence(seq) if seq.is_empty())
        {
            return Err(QueryError::AmbiguousInterventionTarget);
        }
        self.control
            .validate()
            .map_err(|e| QueryError::InvalidIntervention(InterventionFault::Rejected(e)))?;
        for iv in self.joint_control.iter() {
            iv.validate()
                .map_err(|e| QueryError::InvalidIntervention(InterventionFault::Rejected(e)))?;
        }
        // Preserve a natural active world with an explicit single-form control;
        // route-specific contrasts can still require matching intervention sets.
        if first.is_none() && self.joint_control.is_empty() {
            return Ok(());
        }
        if !self.joint_control.is_empty() || self.control.primary_variable().is_none() {
            // Joint and multi-target sequence baselines compare the set of variables,
            // not the number or order of nested steps on each variable.
            let mut active = self.active_variables();
            let mut control = self.control_variables();
            active.sort_unstable();
            active.dedup();
            control.sort_unstable();
            control.dedup();
            if active.is_empty() {
                return Err(QueryError::AmbiguousInterventionTarget);
            }
            if active != control {
                let at = active
                    .iter()
                    .zip(&control)
                    .position(|(a, c)| a != c)
                    .unwrap_or_else(|| active.len().min(control.len()));
                let expected = active
                    .get(at)
                    .or(active.last())
                    .copied()
                    .ok_or(QueryError::AmbiguousInterventionTarget)?;
                let got = control.get(at).or(control.last()).copied().unwrap_or(expected);
                return Err(QueryError::InterventionVariableMismatch { expected, got });
            }
            return Ok(());
        }
        let control_var =
            self.control.primary_variable().ok_or(QueryError::AmbiguousInterventionTarget)?;
        if let Some(active_var) = first.and_then(Intervention::primary_variable) {
            if control_var != active_var {
                return Err(QueryError::InterventionVariableMismatch {
                    expected: active_var,
                    got: control_var,
                });
            }
        } else {
            // A single explicit control can name any variable in the active
            // world, as for the established partial single-control form.
            let active = self.active_variables();
            if !active.contains(&control_var) {
                return Err(QueryError::InterventionVariableMismatch {
                    expected: active
                        .first()
                        .copied()
                        .ok_or(QueryError::AmbiguousInterventionTarget)?,
                    got: control_var,
                });
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod target_binding_tests {
    use super::*;
    use crate::intervention::{InterventionSequence, SequencedIntervention, TemporalPolicy};

    fn set(variable: u32, value: f64) -> Intervention {
        Intervention::set(VariableId::from_raw(variable), Value::f64(value))
    }
    fn sequence(steps: &[u32]) -> Intervention {
        Intervention::sequence(InterventionSequence::new(
            steps
                .iter()
                .map(|variable| {
                    SequencedIntervention::new(set(*variable, 1.0), TemporalPolicy::pulse(0))
                })
                .collect::<Vec<_>>(),
        ))
    }

    #[test]
    fn missing_or_ambiguous_default_target_is_not_licensed() {
        let outcome = VariableId::from_raw(2);
        let empty = CounterfactualQuery::new(outcome, Vec::<Intervention>::new());
        assert!(matches!(empty.validate(), Err(QueryError::AmbiguousInterventionTarget)));
        let natural =
            CounterfactualQuery::new(outcome, Vec::<Intervention>::new()).with_control(set(1, 0.0));
        assert!(natural.validate().is_ok());
        let explicit_nested_control = natural.clone().with_control(sequence(&[0, 1]));
        assert!(explicit_nested_control.validate().is_ok());
        let invalid_control = natural.clone().with_control(set(1, f64::NAN));
        assert!(matches!(invalid_control.validate(), Err(QueryError::InvalidIntervention(_))));
        // An explicitly empty sequence and the unresolved default share the
        // same representation and both refuse as ambiguous.
        assert!(matches!(
            natural.with_control(sequence(&[])).validate(),
            Err(QueryError::AmbiguousInterventionTarget)
        ));
        let invalid_active = CounterfactualQuery::new(outcome, [sequence(&[])]);
        assert!(matches!(invalid_active.validate(), Err(QueryError::InvalidIntervention(_))));
        // The unresolved marker cannot silently choose variable 0, including
        // after asking for a level without specifying its target.
        let ambiguous = CounterfactualQuery::new(outcome, [sequence(&[0, 1])]);
        assert!(matches!(ambiguous.validate(), Err(QueryError::AmbiguousInterventionTarget)));
        assert!(matches!(
            ambiguous.with_control_level(2.0).validate(),
            Err(QueryError::AmbiguousInterventionTarget)
        ));
        let valid = CounterfactualQuery::new(outcome, [set(0, 1.0)]);
        assert!(valid.validate().is_ok());
    }

    #[test]
    fn explicit_multi_target_controls_compare_variable_sets() {
        let outcome = VariableId::from_raw(2);
        let active = sequence(&[0, 0, 1]);
        let control = sequence(&[1, 0]);
        let nested = CounterfactualQuery::new(outcome, [active])
            .with_control(control.clone())
            .with_nested(true);
        assert!(nested.validate().is_ok());
        let partial = CounterfactualQuery::new(outcome, [sequence(&[0, 1])])
            .with_control(set(1, 0.0))
            .with_nested(true);
        assert!(partial.validate().is_ok());
        let unrelated = partial.with_control(set(3, 0.0));
        assert!(matches!(
            unrelated.validate(),
            Err(QueryError::InterventionVariableMismatch { .. })
        ));
        // The first active item can be single-target while the explicit control
        // sequence names the complete multi-intervention active world.
        let inverse = CounterfactualQuery::new(outcome, [set(0, 1.0), set(1, 1.0)])
            .with_control(control)
            .with_nested(true);
        assert!(inverse.validate().is_ok());
        let bad = inverse.with_control(sequence(&[0, 3]));
        assert!(matches!(bad.validate(), Err(QueryError::InterventionVariableMismatch { .. })));
    }

    #[test]
    fn joint_controls_and_intervening_on_outcome_remain_valid() {
        let outcome = VariableId::from_raw(2);
        let mut query = CounterfactualQuery::new(outcome, [set(0, 1.0), set(2, 1.0)])
            .with_joint_control([set(2, 0.0), set(0, 0.0)]);
        query.outcomes = Arc::from([outcome, VariableId::from_raw(0)]);
        assert!(query.validate().is_ok());
        query.joint_control = Arc::from([set(3, 0.0)]);
        assert!(matches!(query.validate(), Err(QueryError::InterventionVariableMismatch { .. })));
    }
}
