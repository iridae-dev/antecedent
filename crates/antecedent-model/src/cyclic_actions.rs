// SPDX-License-Identifier: MIT OR Apache-2.0
//! Named equilibrium actions. A hard clamp, a changed/replaced mechanism and a
//! structural shock retain different scientific identities even when a special
//! parameter choice makes their numerical answers coincide. Numerical solver
//! initialization is separate from every action here; causal initial states
//! belong to the continuous-time family.
use crate::nonlinear_equilibrium::{
    Activation, Clamp, Equation, EquilibriumError, EquilibriumModel, EquilibriumOutcome,
    SolvePolicy,
};
use antecedent_core::ExecutionContext;
use serde::{Deserialize, Serialize};

/// Serializable action targeting a stable structural mechanism identity.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum EquilibriumAction {
    /// Remove the target mechanism and force its endogenous variable.
    Clamp {
        /// Stable mechanism/output identity.
        mechanism: String,
        /// Forced endogenous value.
        value: f64,
    },
    /// Replace the deterministic part of a mechanism, preserving its additive disturbance.
    ReplaceMechanism {
        /// Stable target/output identity; replacement must preserve it.
        mechanism: String,
        /// Complete serializable replacement mechanism.
        replacement: Equation,
    },
    /// Change specified parts of the deterministic mechanism, preserving its disturbance.
    ChangeMechanism {
        /// Stable target/output identity.
        mechanism: String,
        /// Optional new intercept.
        intercept: Option<f64>,
        /// Optional complete incoming coefficient row.
        weights: Option<Vec<f64>>,
        /// Optional new scalar activation.
        activation: Option<Activation>,
    },
    /// Add to the existing structural disturbance after factual abduction.
    AdditiveShock {
        /// Stable target/output identity.
        mechanism: String,
        /// Deterministic additive perturbation, not a fresh independent noise draw.
        delta: f64,
    },
}
/// A named intervention world; actions are retained in artifact target identity.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EquilibriumActionWorld {
    /// Prespecified stable world identifier.
    pub id: String,
    /// Distinct structural actions; ambiguous duplicate targets are refused.
    pub actions: Vec<EquilibriumAction>,
}
/// Compiled action block, distinct from a factual/observational model.
#[derive(Clone, Debug)]
pub struct EquilibriumActionPlan {
    model: EquilibriumModel,
    clamps: Vec<Clamp>,
    shocks: Vec<f64>,
}
fn invalid(message: &str) -> EquilibriumError {
    EquilibriumError::Invalid(message.into())
}
impl EquilibriumModel {
    /// Compile named mechanism actions, rejecting conflicting changes/clamps.
    /// One deterministic change and one shock may target the same mechanism.
    pub fn compile_actions(
        &self,
        actions: &[EquilibriumAction],
    ) -> Result<EquilibriumActionPlan, EquilibriumError> {
        self.validate()?;
        if actions.len() > self.order() * 2 {
            return Err(invalid("action count limit"));
        }
        let mut model = self.clone();
        let mut clamps = Vec::new();
        let mut shocks = vec![0.0; self.order()];
        let mut touched = vec![0_u8; self.order()];
        for action in actions {
            let mechanism = match action {
                EquilibriumAction::Clamp { mechanism, .. }
                | EquilibriumAction::ReplaceMechanism { mechanism, .. }
                | EquilibriumAction::ChangeMechanism { mechanism, .. }
                | EquilibriumAction::AdditiveShock { mechanism, .. } => mechanism,
            };
            let i = self
                .equations
                .iter()
                .position(|e| e.id == *mechanism)
                .ok_or_else(|| invalid("unknown action mechanism identity"))?;
            let kind = match action {
                EquilibriumAction::Clamp { .. } => 1,
                EquilibriumAction::AdditiveShock { .. } => 4,
                _ => 2,
            };
            if touched[i] & kind != 0 || (kind == 1 && touched[i] != 0) || touched[i] & 1 != 0 {
                return Err(invalid("conflicting or duplicate mechanism actions"));
            }
            touched[i] |= kind;
            match action {
                EquilibriumAction::Clamp { value, .. } => {
                    if !value.is_finite() {
                        return Err(invalid("nonfinite clamp"));
                    }
                    clamps.push(Clamp { variable: i, value: *value });
                }
                EquilibriumAction::ReplaceMechanism { replacement, .. } => {
                    if replacement.id != *mechanism {
                        return Err(invalid("replacement must preserve target mechanism identity"));
                    }
                    model.equations[i] = replacement.clone();
                }
                EquilibriumAction::ChangeMechanism { intercept, weights, activation, .. } => {
                    if intercept.is_none() && weights.is_none() && activation.is_none() {
                        return Err(invalid("empty mechanism change"));
                    }
                    if let Some(value) = intercept {
                        model.equations[i].intercept = *value;
                    }
                    if let Some(row) = weights {
                        model.equations[i].weights.clone_from(row);
                    }
                    if let Some(value) = activation {
                        model.equations[i].activation = *value;
                    }
                }
                EquilibriumAction::AdditiveShock { delta, .. } => {
                    if !delta.is_finite() {
                        return Err(invalid("nonfinite shock"));
                    }
                    shocks[i] = *delta;
                }
            }
        }
        model.validate()?;
        Ok(EquilibriumActionPlan { model, clamps, shocks })
    }
}
impl EquilibriumActionPlan {
    /// Inspect the actual post-action SCC schedule.
    pub fn blocks(&self) -> Result<Vec<Vec<usize>>, EquilibriumError> {
        self.model.blocks(&self.clamps)
    }
    /// Reuse the supplied/factually abducted disturbances, then apply shocks.
    /// Numerical failures remain aligned outcomes; resource/cancellation errors abort.
    pub fn solve(
        &self,
        noise: &[f64],
        initial: &[f64],
        policy: SolvePolicy,
        ctx: &ExecutionContext,
    ) -> Result<EquilibriumOutcome, EquilibriumError> {
        if noise.len() != self.shocks.len() {
            return Err(invalid("action disturbance order"));
        }
        let shifted: Vec<_> = noise.iter().zip(&self.shocks).map(|(u, shock)| u + shock).collect();
        self.model.solve_outcome(&shifted, &self.clamps, initial, policy, ctx)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn named_clamps_changes_replacements_and_shocks_have_distinct_shared_noise_truths() {
        let model = EquilibriumModel {
            equations: vec![
                Equation {
                    id: "a".into(),
                    intercept: 0.0,
                    weights: vec![0.0, 0.4],
                    activation: Activation::Identity,
                    expression: None,
                    parameters: vec![],
                },
                Equation {
                    id: "b".into(),
                    intercept: 0.0,
                    weights: vec![0.5, 0.0],
                    activation: Activation::Identity,
                    expression: None,
                    parameters: vec![],
                },
            ],
            noise_covariance: vec![1.0, 0.0, 0.0, 1.0],
        };
        let disturbance = model.abduce(&[1.25, 0.625]).unwrap();
        let cases = [
            (EquilibriumAction::Clamp { mechanism: "a".into(), value: 2.0 }, [2.0, 1.0]),
            (EquilibriumAction::AdditiveShock { mechanism: "a".into(), delta: 1.0 }, [2.5, 1.25]),
            (
                EquilibriumAction::ChangeMechanism {
                    mechanism: "b".into(),
                    intercept: None,
                    weights: Some(vec![0.0, 0.0]),
                    activation: None,
                },
                [1.0, 0.0],
            ),
            (
                EquilibriumAction::ReplaceMechanism {
                    mechanism: "a".into(),
                    replacement: Equation {
                        id: "a".into(),
                        intercept: 3.0,
                        weights: vec![0.0, 0.0],
                        activation: Activation::Identity,
                        expression: None,
                        parameters: vec![],
                    },
                },
                [4.0, 2.0],
            ),
        ];
        for (action, expected) in cases {
            let plan = model.compile_actions(&[action]).unwrap();
            let EquilibriumOutcome::Solved(answer) = plan
                .solve(
                    &disturbance,
                    &[0.0, 0.0],
                    SolvePolicy::default(),
                    &ExecutionContext::for_tests(1),
                )
                .unwrap()
            else {
                panic!("analytic action should solve")
            };
            for (value, truth) in answer.values.iter().zip(expected) {
                assert!((value - truth).abs() < 1e-9);
            }
        }
        assert!(
            model
                .compile_actions(&[
                    EquilibriumAction::Clamp { mechanism: "a".into(), value: 2.0 },
                    EquilibriumAction::AdditiveShock { mechanism: "a".into(), delta: 1.0 }
                ])
                .is_err()
        );
    }
}
