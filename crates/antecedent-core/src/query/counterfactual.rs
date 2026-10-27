//! Query submodule.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::fault::InterventionFault;
use std::sync::Arc;

use crate::canonical_encode::{CanonicalEncode, encode_list, encode_variables, flag};
use crate::identity::CanonicalWriter;
use crate::ids::VariableId;
use crate::intervention::Intervention;
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
    /// Hard set of the first intervention's target to `0.0`.
    ///
    /// [`Self::new`] uses this; format-0.4 query decode uses it when the wire omits `control`.
    #[must_use]
    pub fn default_control(interventions: &[Intervention]) -> Intervention {
        let variable = interventions
            .first()
            .and_then(Intervention::primary_variable)
            .unwrap_or(VariableId::from_raw(0));
        Intervention::set(variable, Value::f64(0.0))
    }

    /// Construct a single-outcome counterfactual query.
    ///
    /// Control defaults to a hard set of the first intervention's target to `0.0`.
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
    #[must_use]
    pub fn with_control_level(self, level: f64) -> Self {
        let variable = self
            .control
            .primary_variable()
            .or_else(|| self.interventions.first().and_then(Intervention::primary_variable))
            .unwrap_or(VariableId::from_raw(0));
        self.with_control(Intervention::set(variable, Value::f64(level)))
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
    /// Empty outcomes, invalid interventions, or a control that targets a different
    /// variable than the first active intervention.
    pub fn validate(&self) -> Result<(), QueryError> {
        if self.outcomes.is_empty() {
            return Err(QueryError::EmptyCounterfactualOutcomes);
        }
        for iv in self.interventions.iter() {
            iv.validate()
                .map_err(|e| QueryError::InvalidIntervention(InterventionFault::Rejected(e)))?;
        }
        self.control
            .validate()
            .map_err(|e| QueryError::InvalidIntervention(InterventionFault::Rejected(e)))?;
        for iv in self.joint_control.iter() {
            iv.validate()
                .map_err(|e| QueryError::InvalidIntervention(InterventionFault::Rejected(e)))?;
        }
        if !self.joint_control.is_empty() {
            // A joint baseline contrasts like with like: both worlds intervene on the same
            // set of variables, in any order.
            let mut active = self.active_variables();
            let mut control = self.control_variables();
            active.sort_unstable();
            control.sort_unstable();
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
        if let Some(active_var) =
            self.interventions.first().and_then(Intervention::primary_variable)
        {
            let control_var =
                self.control.primary_variable().ok_or(QueryError::AmbiguousInterventionTarget)?;
            if control_var != active_var {
                return Err(QueryError::InterventionVariableMismatch {
                    expected: active_var,
                    got: control_var,
                });
            }
        }
        Ok(())
    }
}
