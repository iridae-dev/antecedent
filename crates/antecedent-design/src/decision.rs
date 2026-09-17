//! Decision analysis primitives.
//!
//! A decision problem is a set of terminal actions, a utility `U(a, θ)` over the
//! uncertain decision state `θ`, and chance constraints that decide which actions
//! are admissible. [`evaluate_decision`] chooses the Bayes action under the current
//! belief (equally weighted draws of `θ`) and reports its expected regret against
//! perfect information. Value of *sample* information for a candidate design is in
//! [`crate::preposterior`].
//!
//! References: H. Raiffa and R. Schlaifer, *Applied Statistical Decision Theory*
//! (Harvard, 1961), ch. 4 (terminal opportunity loss and the expected value of
//! perfect information).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::fmt;
use std::sync::Arc;

use crate::error::DesignError;

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
    ///
    /// # Errors
    ///
    /// Callback or shape failures. Do not write NaN as a silent fallback.
    fn evaluate_batch(
        &self,
        actions: &[A],
        outcomes: &[O],
        out: &mut [f64],
    ) -> Result<(), DesignError>;

    /// Declared coefficients `(α_a, β_a)` when `U(a, θ) = α_a + β_a·θ` for a scalar
    /// state `θ`, one pair per action. `None` (the default) means the utility makes
    /// no such declaration; the conjugate-normal preposterior analysis then refuses.
    fn affine_in_state(&self, actions: &[A]) -> Option<Vec<(f64, f64)>> {
        let _ = actions;
        None
    }
}

/// Utility affine in a scalar state: `U(a, θ) = intercepts[a] + slopes[a]·θ`, with
/// actions given as indices into the coefficient tables.
#[derive(Clone, Debug, PartialEq)]
pub struct AffineUtility {
    /// Per-action intercepts `α_a`.
    pub intercepts: Arc<[f64]>,
    /// Per-action state slopes `β_a`.
    pub slopes: Arc<[f64]>,
}

impl AffineUtility {
    /// Construct from per-action intercepts and slopes.
    ///
    /// # Errors
    ///
    /// Length mismatch, no actions, or non-finite coefficients.
    pub fn new(intercepts: Vec<f64>, slopes: Vec<f64>) -> Result<Self, DesignError> {
        if intercepts.is_empty() || intercepts.len() != slopes.len() {
            return Err(DesignError::Shape(format!(
                "affine utility needs one intercept and one slope per action (got {} and {})",
                intercepts.len(),
                slopes.len()
            )));
        }
        if intercepts.iter().chain(&slopes).any(|v| !v.is_finite()) {
            return Err(DesignError::Numerical(
                "affine utility coefficients must be finite".into(),
            ));
        }
        Ok(Self { intercepts: intercepts.into(), slopes: slopes.into() })
    }

    fn coefficients(&self, action: usize) -> Result<(f64, f64), DesignError> {
        match (self.intercepts.get(action), self.slopes.get(action)) {
            (Some(&alpha), Some(&beta)) => Ok((alpha, beta)),
            _ => Err(DesignError::Shape(format!(
                "affine utility has no coefficients for action index {action}"
            ))),
        }
    }
}

impl Utility<usize, f64> for AffineUtility {
    fn evaluate_batch(
        &self,
        actions: &[usize],
        outcomes: &[f64],
        out: &mut [f64],
    ) -> Result<(), DesignError> {
        let n_o = outcomes.len();
        for (row, &action) in actions.iter().enumerate() {
            let (alpha, beta) = self.coefficients(action)?;
            for (col, theta) in outcomes.iter().enumerate() {
                out[row * n_o + col] = alpha + beta * theta;
            }
        }
        Ok(())
    }

    fn affine_in_state(&self, actions: &[usize]) -> Option<Vec<(f64, f64)>> {
        actions.iter().map(|&action| self.coefficients(action).ok()).collect()
    }
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
    /// Candidate terminal actions.
    pub actions: Vec<A>,
    /// Utility function.
    pub utility: Arc<dyn Utility<A, O>>,
    /// Constraints (chance or hard via thresholding). They decide which actions are
    /// admissible; they never enter the utility as a cost.
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

/// Decision evaluation summary under the current belief.
#[derive(Clone, Debug, PartialEq)]
pub struct DecisionEvaluation {
    /// Expected utility of the Bayes action (the admissible action with the highest
    /// expected utility).
    pub expected_utility: f64,
    /// Expected regret (terminal opportunity loss) of the Bayes action against
    /// perfect information over the admissible actions:
    /// `E_θ[max_{a∈F} U(a,θ)] − max_{a∈F} E_θ[U(a,θ)]`. This is the expected value of
    /// perfect information (EVPI). It is non-negative, unchanged when a constant is
    /// added to the utility, and scales with a positive rescaling of the utility.
    pub posterior_regret: f64,
    /// Per-constraint satisfaction probabilities for the chosen action.
    pub chance_constraint_probs: Arc<[f64]>,
    /// Index of the chosen (Bayes) action.
    pub chosen_action: usize,
}

/// Utilities and admissibility of every action under equally weighted state draws.
pub(crate) struct DecisionTable {
    /// Number of state draws.
    pub(crate) n_outcomes: usize,
    /// Row-major `actions × outcomes` utility matrix.
    pub(crate) utilities: Vec<f64>,
    /// Per-constraint satisfaction probability of every action.
    pub(crate) satisfaction: Vec<Vec<f64>>,
    /// Admissible action indices (ascending).
    pub(crate) admissible: Vec<usize>,
}

impl DecisionTable {
    /// Evaluate `problem` on `outcomes` and determine the admissible action set.
    ///
    /// # Errors
    ///
    /// No actions, no draws, an invalid chance threshold, non-finite utilities or
    /// satisfaction values, callback failures, or no admissible action
    /// ([`DesignError::NoAdmissibleAction`]).
    pub(crate) fn build<A, O>(
        problem: &DecisionProblem<A, O>,
        outcomes: &[O],
    ) -> Result<Self, DesignError> {
        let n_a = problem.actions.len();
        let n_o = outcomes.len();
        if n_a == 0 {
            return Err(DesignError::Config("decision problem has no actions".into()));
        }
        if n_o == 0 {
            return Err(DesignError::Shape(
                "decision evaluation needs at least one draw of the decision state".into(),
            ));
        }
        let threshold = problem.chance_threshold;
        if !(0.0..=1.0).contains(&threshold) {
            return Err(DesignError::Config(format!(
                "chance_threshold must lie in [0, 1] (got {threshold})"
            )));
        }
        let mut utilities = vec![0.0; n_a * n_o];
        problem.utility.evaluate_batch(&problem.actions, outcomes, &mut utilities)?;
        if utilities.iter().any(|u| !u.is_finite()) {
            return Err(DesignError::Numerical("utility returned a non-finite value".into()));
        }
        let mut satisfaction = Vec::with_capacity(problem.constraints.len());
        let mut worst = vec![1.0_f64; n_a];
        for constraint in &problem.constraints {
            let mut row = vec![0.0; n_a];
            constraint.satisfaction_batch(&problem.actions, outcomes, &mut row);
            if row.iter().any(|s| !s.is_finite()) {
                return Err(DesignError::Numerical(format!(
                    "constraint {} returned a non-finite satisfaction",
                    constraint.name()
                )));
            }
            for (slot, &s) in worst.iter_mut().zip(&row) {
                *slot = slot.min(s);
            }
            satisfaction.push(row);
        }
        let admissible: Vec<usize> = (0..n_a).filter(|&a| worst[a] + 1e-12 >= threshold).collect();
        if admissible.is_empty() {
            return Err(DesignError::NoAdmissibleAction(format!(
                "every one of {n_a} actions violates chance_threshold {threshold}"
            )));
        }
        Ok(Self { n_outcomes: n_o, utilities, satisfaction, admissible })
    }

    /// Utility of action `a` under draw `k`.
    pub(crate) fn utility(&self, a: usize, k: usize) -> f64 {
        self.utilities[a * self.n_outcomes + k]
    }

    /// Admissible action with the highest mean utility (lowest index on ties).
    pub(crate) fn bayes_action(&self) -> (usize, f64) {
        let mut best = (self.admissible[0], f64::NEG_INFINITY);
        for &a in &self.admissible {
            let eu = (0..self.n_outcomes).map(|k| self.utility(a, k)).sum::<f64>()
                / self.n_outcomes as f64;
            if eu > best.1 {
                best = (a, eu);
            }
        }
        best
    }
}

/// Evaluate the Bayes action and its expected regret under equally weighted draws of
/// the decision state.
///
/// Constraints define the admissible actions (minimum satisfaction plus `1e-12` at
/// least `chance_threshold`); inadmissible actions are excluded from both the choice
/// and the perfect-information benchmark.
///
/// # Errors
///
/// Utility callback failures, empty actions or draws, and
/// [`DesignError::NoAdmissibleAction`] when no action meets the threshold (there is
/// no decision to value, so no number is fabricated).
pub fn evaluate_decision<A, O>(
    problem: &DecisionProblem<A, O>,
    outcomes: &[O],
) -> Result<DecisionEvaluation, DesignError> {
    let table = DecisionTable::build(problem, outcomes)?;
    let (chosen, expected_utility) = table.bayes_action();
    // E[max_a U(a,θ) − U(a*,θ)]: a sum of non-negative differences, so the
    // regret is exactly invariant to the utility's origin.
    let regret = (0..table.n_outcomes)
        .map(|k| {
            let best = table
                .admissible
                .iter()
                .map(|&a| table.utility(a, k))
                .fold(f64::NEG_INFINITY, f64::max);
            best - table.utility(chosen, k)
        })
        .sum::<f64>()
        / table.n_outcomes as f64;
    let chance = table.satisfaction.iter().map(|row| row[chosen]).collect::<Arc<[_]>>();
    Ok(DecisionEvaluation {
        expected_utility,
        posterior_regret: regret,
        chance_constraint_probs: chance,
        chosen_action: chosen,
    })
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    struct LinearUtil(f64);
    impl Utility<f64, f64> for LinearUtil {
        fn evaluate_batch(
            &self,
            actions: &[f64],
            outcomes: &[f64],
            out: &mut [f64],
        ) -> Result<(), DesignError> {
            let n_o = outcomes.len();
            for (ai, a) in actions.iter().enumerate() {
                for (oi, o) in outcomes.iter().enumerate() {
                    out[ai * n_o + oi] = a * o + self.0;
                }
            }
            Ok(())
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
            Arc::new(LinearUtil(0.0)),
            vec![Arc::new(NonNegOutcome)],
        );
        let ev = evaluate_decision(&problem, &[1.0, 2.0]).unwrap();
        assert_eq!(ev.chosen_action, 1);
        assert!((ev.expected_utility - 3.0).abs() < 1e-12);
    }

    /// Outcomes [-1, 3] under U = a·θ: the Bayes action is a=2 (EU 2), perfect
    /// information earns E[max(-1·1, -1·2), max(3, 6)] = (−1 + 6)/2 = 2.5, so the
    /// expected regret is 0.5. A "chosen feasible vs best overall" score is 0 here.
    #[test]
    fn unconstrained_regret_is_expected_value_of_perfect_information() {
        let problem = DecisionProblem::new(vec![1.0, 2.0], Arc::new(LinearUtil(0.0)), vec![]);
        let ev = evaluate_decision(&problem, &[-1.0, 3.0]).unwrap();
        assert_eq!(ev.chosen_action, 1);
        assert!((ev.expected_utility - 2.0).abs() < 1e-12);
        assert!((ev.posterior_regret - 0.5).abs() < 1e-12, "{}", ev.posterior_regret);
    }

    #[test]
    fn regret_is_shift_invariant_and_scale_equivariant() {
        let base = evaluate_decision(
            &DecisionProblem::new(vec![1.0, 2.0, -0.5], Arc::new(LinearUtil(0.0)), vec![]),
            &[-1.0, 3.0, 0.25, -2.0],
        )
        .unwrap();
        for shift in [-1.0e3, 100.0, 7.5] {
            let shifted = evaluate_decision(
                &DecisionProblem::new(vec![1.0, 2.0, -0.5], Arc::new(LinearUtil(shift)), vec![]),
                &[-1.0, 3.0, 0.25, -2.0],
            )
            .unwrap();
            assert_eq!(shifted.chosen_action, base.chosen_action);
            assert!((shifted.posterior_regret - base.posterior_regret).abs() < 1e-9);
        }
        // Scaling actions by 3 scales U = a·θ by 3.
        let scaled = evaluate_decision(
            &DecisionProblem::new(vec![3.0, 6.0, -1.5], Arc::new(LinearUtil(0.0)), vec![]),
            &[-1.0, 3.0, 0.25, -2.0],
        )
        .unwrap();
        assert!((scaled.posterior_regret - 3.0 * base.posterior_regret).abs() < 1e-12);
    }

    /// Only action 1 is admissible, so there is nothing to learn: regret 0 even
    /// though the excluded action has higher utility (constraints are not costs).
    struct OnlySecond;
    impl DecisionConstraint<f64, f64> for OnlySecond {
        fn name(&self) -> &str {
            "only_second"
        }
        fn satisfaction_batch(&self, actions: &[f64], _outcomes: &[f64], out: &mut [f64]) {
            for (a, slot) in (0..actions.len()).zip(out.iter_mut()) {
                *slot = if a == 1 { 1.0 } else { 0.0 };
            }
        }
    }

    #[test]
    fn inadmissible_actions_are_excluded_not_scored_as_cost() {
        let problem = DecisionProblem::new(
            vec![10.0, 1.0, 2.0],
            Arc::new(LinearUtil(0.0)),
            vec![Arc::new(OnlySecond)],
        );
        let ev = evaluate_decision(&problem, &[-1.0, 3.0]).unwrap();
        assert_eq!(ev.chosen_action, 1);
        assert!(ev.posterior_regret.abs() < 1e-15);
        assert_eq!(ev.chance_constraint_probs.as_ref(), &[1.0]);
    }

    struct Never;
    impl DecisionConstraint<f64, f64> for Never {
        fn name(&self) -> &str {
            "never"
        }
        fn satisfaction_batch(&self, actions: &[f64], _outcomes: &[f64], out: &mut [f64]) {
            out[..actions.len()].fill(0.0);
        }
    }

    /// No admissible action is a typed refusal at every utility origin, never a
    /// regret measured from a fabricated zero utility.
    #[test]
    fn all_inadmissible_is_a_typed_refusal_at_any_utility_origin() {
        for shift in [0.0, 100.0] {
            let problem = DecisionProblem::new(
                vec![1.0, 2.0],
                Arc::new(LinearUtil(shift)),
                vec![Arc::new(Never)],
            );
            let err = evaluate_decision(&problem, &[-1.0, 3.0]).unwrap_err();
            assert!(matches!(err, DesignError::NoAdmissibleAction(_)), "{err:?}");
        }
    }

    #[test]
    fn empty_draws_and_actions_are_refused() {
        let problem = DecisionProblem::new(vec![1.0], Arc::new(LinearUtil(0.0)), vec![]);
        assert!(matches!(evaluate_decision(&problem, &[]), Err(DesignError::Shape(_))));
        let empty = DecisionProblem::<f64, f64>::new(vec![], Arc::new(LinearUtil(0.0)), vec![]);
        assert!(matches!(evaluate_decision(&empty, &[1.0]), Err(DesignError::Config(_))));
    }

    #[test]
    fn affine_utility_declares_its_coefficients() {
        let u = AffineUtility::new(vec![1.0, -2.0], vec![0.5, 3.0]).unwrap();
        assert_eq!(u.affine_in_state(&[1, 0]), Some(vec![(-2.0, 3.0), (1.0, 0.5)]));
        assert_eq!(u.affine_in_state(&[2]), None);
        let mut out = [0.0; 4];
        u.evaluate_batch(&[0, 1], &[0.0, 2.0], &mut out).unwrap();
        assert_eq!(out, [1.0, 2.0, -2.0, 4.0]);
        assert!(AffineUtility::new(vec![1.0], vec![]).is_err());
    }
}
