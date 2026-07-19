//! Decision analysis primitives.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::fmt;
use std::sync::Arc;

/// Handle for a registered decision problem.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct DecisionProblemId(u32);

impl DecisionProblemId {
    /// Create from a raw dense index.
    #[must_use]
    pub const fn from_raw(raw: u32) -> Self {
        Self(raw)
    }

    /// Underlying dense index.
    #[must_use]
    pub const fn raw(self) -> u32 {
        self.0
    }
}

impl fmt::Display for DecisionProblemId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "D{}", self.0)
    }
}

/// Batch utility evaluation over actions × outcome draws.
pub trait Utility<A, O>: Send + Sync {
    /// Evaluate utilities for each `(action, outcome)` pair in row-major order
    /// `actions.len() * outcomes.len()`, writing into `out`.
    fn evaluate_batch(&self, actions: &[A], outcomes: &[O], out: &mut [f64]);
}

/// Batch chance / hard constraint over actions × outcome draws.
pub trait DecisionConstraint<A, O>: Send + Sync {
    /// Constraint name for diagnostics.
    fn name(&self) -> &str;

    /// For each action, write the fraction of outcomes that satisfy the constraint
    /// into `out` (length `actions.len()`).
    fn satisfaction_batch(&self, actions: &[A], outcomes: &[O], out: &mut [f64]);
}

/// Decision problem: choose an action under utility and constraints.
///
/// The library returns expected utility and regret; it does not dispatch actions.
pub struct DecisionProblem<A, O> {
    /// Feasible actions.
    pub actions: Vec<A>,
    /// Utility function.
    pub utility: Arc<dyn Utility<A, O>>,
    /// Constraints (chance or hard via thresholding).
    pub constraints: Vec<Arc<dyn DecisionConstraint<A, O>>>,
    /// Minimum chance-constraint satisfaction required (default 1.0 = hard).
    pub chance_threshold: f64,
}

impl<A, O> DecisionProblem<A, O> {
    /// Construct with hard constraints by default.
    #[must_use]
    pub fn new(
        actions: Vec<A>,
        utility: Arc<dyn Utility<A, O>>,
        constraints: Vec<Arc<dyn DecisionConstraint<A, O>>>,
    ) -> Self {
        Self { actions, utility, constraints, chance_threshold: 1.0 }
    }
}

/// Decision evaluation summary for one candidate / posterior.
#[derive(Clone, Debug, PartialEq)]
pub struct DecisionEvaluation {
    /// Expected utility of the optimal feasible action.
    pub expected_utility: f64,
    /// Posterior regret vs oracle (max EU − chosen EU when infeasible → NaN path uses 0).
    pub posterior_regret: f64,
    /// Per-constraint satisfaction probabilities for the chosen action.
    pub chance_constraint_probs: Arc<[f64]>,
    /// Index of chosen action, if any feasible.
    pub chosen_action: Option<usize>,
}

/// Evaluate expected utility, regret, and chance constraints under outcome draws.
pub fn evaluate_decision<A, O>(
    problem: &DecisionProblem<A, O>,
    outcomes: &[O],
) -> DecisionEvaluation
where
    A: Clone,
{
    let n_a = problem.actions.len();
    let n_o = outcomes.len().max(1);
    let mut util = vec![0.0; n_a * n_o];
    if !outcomes.is_empty() && n_a > 0 {
        problem.utility.evaluate_batch(&problem.actions, outcomes, &mut util);
    }
    let mut eu = vec![0.0; n_a];
    if !outcomes.is_empty() {
        for (a, slot) in eu.iter_mut().enumerate() {
            let mut s = 0.0;
            for o in 0..n_o {
                s += util[a * n_o + o];
            }
            *slot = s / n_o as f64;
        }
    }
    let mut sat = vec![1.0; n_a];
    let mut per_constraint: Vec<Vec<f64>> = Vec::with_capacity(problem.constraints.len());
    for c in &problem.constraints {
        let mut row = vec![0.0; n_a];
        if !outcomes.is_empty() && n_a > 0 {
            c.satisfaction_batch(&problem.actions, outcomes, &mut row);
        } else {
            row.fill(1.0);
        }
        for a in 0..n_a {
            sat[a] = f64::min(sat[a], row[a]);
        }
        per_constraint.push(row);
    }
    let mut best_feas = None;
    let mut best_eu = f64::NEG_INFINITY;
    let mut best_any = None;
    let mut best_any_eu = f64::NEG_INFINITY;
    for (a, &u) in eu.iter().enumerate() {
        if u > best_any_eu {
            best_any_eu = u;
            best_any = Some(a);
        }
        if sat[a] + 1e-12 >= problem.chance_threshold && u > best_eu {
            best_eu = u;
            best_feas = Some(a);
        }
    }
    let chosen = best_feas.or(best_any);
    let chosen_eu = chosen.map_or(0.0, |a| eu[a]);
    let oracle = best_any_eu.max(chosen_eu);
    let regret = (oracle - chosen_eu).max(0.0);
    let chance = if let Some(a) = chosen {
        per_constraint.iter().map(|row| row[a]).collect::<Arc<[_]>>()
    } else {
        Arc::from([])
    };
    DecisionEvaluation {
        expected_utility: chosen_eu,
        posterior_regret: regret,
        chance_constraint_probs: chance,
        chosen_action: chosen,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct LinearUtil;
    impl Utility<f64, f64> for LinearUtil {
        fn evaluate_batch(&self, actions: &[f64], outcomes: &[f64], out: &mut [f64]) {
            let n_o = outcomes.len();
            for (ai, a) in actions.iter().enumerate() {
                for (oi, o) in outcomes.iter().enumerate() {
                    out[ai * n_o + oi] = a * o;
                }
            }
        }
    }

    struct NonNegOutcome;
    impl DecisionConstraint<f64, f64> for NonNegOutcome {
        fn name(&self) -> &str {
            "nonneg"
        }
        fn satisfaction_batch(&self, actions: &[f64], outcomes: &[f64], out: &mut [f64]) {
            let frac = outcomes.iter().filter(|o| **o >= 0.0).count() as f64
                / outcomes.len().max(1) as f64;
            for (i, _) in actions.iter().enumerate() {
                out[i] = frac;
            }
        }
    }

    #[test]
    fn picks_higher_expected_utility() {
        let problem = DecisionProblem::new(
            vec![1.0, 2.0],
            Arc::new(LinearUtil),
            vec![Arc::new(NonNegOutcome)],
        );
        let ev = evaluate_decision(&problem, &[1.0, 2.0]);
        assert_eq!(ev.chosen_action, Some(1));
        assert!((ev.expected_utility - 3.0).abs() < 1e-12);
    }
}
