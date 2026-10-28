//! Bounded families of history-dependent intervention policies.
//!
//! A family declares the observed features a rule may read (the current value of a variable
//! or a lagged value), the rule's shape and a closed parameter box. One parameter vector
//! fixes one policy. The same policy is available two ways that agree bit for bit: directly
//! ([`AdaptivePolicyFamily::action`]) for independent simulators, and lowered to a
//! [`ValueExpr`] ([`AdaptivePolicyFamily::to_value_expr`]) that a temporal rollout executes
//! as an assigned intervention. Every action is clipped into the family's action interval, so
//! the action bound is a property of the family rather than a penalty.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{CanonicalWriter, ValueExpr, VariableId};

/// Largest decision-tree depth a family may declare; deeper rules are not a small tree.
pub const MAX_TREE_DEPTH: u32 = 3;
/// Largest number of cut points of a piecewise-constant rule.
pub const MAX_PIECEWISE_CUTS: usize = 8;
/// Largest number of features a rule may read.
pub const MAX_POLICY_FEATURES: usize = 16;
/// Largest number of steps of a fixed schedule family.
pub const MAX_SCHEDULE_STEPS: usize = 256;

/// One observed quantity a rule may read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PolicyFeature {
    /// The current-step value of a variable the model computes before the action.
    Current(VariableId),
    /// The value of a variable `lag` (at least one) steps earlier.
    Lag {
        /// Variable read.
        variable: VariableId,
        /// Steps back.
        lag: u32,
    },
}

impl PolicyFeature {
    /// The value expression reading this feature in a rollout.
    #[must_use]
    pub fn expr(self) -> ValueExpr {
        match self {
            Self::Current(variable) => ValueExpr::var(variable),
            Self::Lag { variable, lag } => ValueExpr::lag(variable, lag),
        }
    }
    /// Steps of history the feature needs.
    #[must_use]
    pub const fn lag(self) -> u32 {
        match self {
            Self::Current(_) => 0,
            Self::Lag { lag, .. } => lag,
        }
    }
    /// The variable read.
    #[must_use]
    pub const fn variable(self) -> VariableId {
        match self {
            Self::Current(variable) | Self::Lag { variable, .. } => variable,
        }
    }
}

/// Closed interval of one rule parameter.
#[derive(Clone, Debug, PartialEq)]
pub struct ParameterBound {
    /// Stable parameter name.
    pub name: String,
    /// Inclusive lower bound.
    pub lower: f64,
    /// Inclusive upper bound.
    pub upper: f64,
}

/// The shape of a rule. Parameter layouts are documented per variant.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PolicyRule {
    /// Open-loop schedule: `theta[t]` is the action at step `t`; later steps repeat the last.
    /// This is the best-fixed-schedule baseline a history-dependent rule is compared with.
    Schedule {
        /// Steps with their own parameter.
        steps: usize,
    },
    /// `theta = [threshold, below, at_or_above]` on one feature.
    Threshold {
        /// Feature index read.
        feature: usize,
    },
    /// Piecewise-constant on one feature: `theta = [cut_0..cut_{k-1}, level_0..level_k]`. The
    /// cuts are ordered ascending before use, so any point of the box is a valid rule.
    Piecewise {
        /// Feature index read.
        feature: usize,
        /// Number of cut points.
        cuts: usize,
    },
    /// `u = clip(a + sum_i b_i f_i)`: `theta = [a, b_0..b_{m-1}]` over `features`.
    LinearFeedback {
        /// Feature indices read.
        features: Vec<usize>,
    },
    /// Complete binary tree in heap order. `splits[i]` is the feature tested at internal node
    /// `i`; `theta = [threshold_0.., leaf_0..]`; a value at or above the threshold goes right.
    Tree {
        /// Feature tested at each of the `2^depth - 1` internal nodes.
        splits: Vec<usize>,
    },
}

/// Why a family or a parameter vector is invalid.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("adaptive policy family: {0}")]
pub struct AdaptivePolicyError(String);

fn bad(message: impl Into<String>) -> AdaptivePolicyError {
    AdaptivePolicyError(message.into())
}

/// A bounded family of policies with declared features, rule shape and parameter box.
#[derive(Clone, Debug, PartialEq)]
pub struct AdaptivePolicyFamily {
    id: String,
    features: Vec<PolicyFeature>,
    rule: PolicyRule,
    action_lower: f64,
    action_upper: f64,
    bounds: Vec<ParameterBound>,
}

fn bound(name: String, lower: f64, upper: f64) -> ParameterBound {
    ParameterBound { name, lower, upper }
}

impl AdaptivePolicyFamily {
    /// Build and validate a family. `feature_ranges` gives, per feature, the interval on which
    /// thresholds and cuts may lie; `coefficient_range` bounds linear-feedback slopes.
    ///
    /// # Errors
    /// Non-finite or inverted bounds, a feature index out of range, an over-large rule, or a
    /// tree whose split list does not match a complete tree.
    pub fn new(
        id: impl Into<String>,
        features: Vec<PolicyFeature>,
        feature_ranges: &[(f64, f64)],
        rule: PolicyRule,
        action_bounds: (f64, f64),
        coefficient_range: (f64, f64),
    ) -> Result<Self, AdaptivePolicyError> {
        let id = id.into();
        let (lo, hi) = action_bounds;
        let finite_interval = |a: f64, b: f64| a.is_finite() && b.is_finite() && a <= b;
        if id.is_empty() {
            return Err(bad("empty family id"));
        }
        if !finite_interval(lo, hi) {
            return Err(bad("action bounds must be finite with lower <= upper"));
        }
        if features.len() > MAX_POLICY_FEATURES || feature_ranges.len() != features.len() {
            return Err(bad("one finite range per declared feature, at most the feature limit"));
        }
        if feature_ranges.iter().any(|(a, b)| !finite_interval(*a, *b)) {
            return Err(bad("feature ranges must be finite with lower <= upper"));
        }
        let range = |feature: usize| -> Result<(f64, f64), AdaptivePolicyError> {
            feature_ranges.get(feature).copied().ok_or_else(|| bad("feature index out of range"))
        };
        let mut bounds = Vec::new();
        match &rule {
            PolicyRule::Schedule { steps } => {
                if *steps == 0 || *steps > MAX_SCHEDULE_STEPS {
                    return Err(bad("a schedule needs 1..=256 steps"));
                }
                for t in 0..*steps {
                    bounds.push(bound(format!("action_{t}"), lo, hi));
                }
            }
            PolicyRule::Threshold { feature } => {
                let (a, b) = range(*feature)?;
                bounds.push(bound("threshold".into(), a, b));
                bounds.push(bound("below".into(), lo, hi));
                bounds.push(bound("at_or_above".into(), lo, hi));
            }
            PolicyRule::Piecewise { feature, cuts } => {
                if *cuts == 0 || *cuts > MAX_PIECEWISE_CUTS {
                    return Err(bad("a piecewise rule needs 1..=8 cut points"));
                }
                let (a, b) = range(*feature)?;
                for j in 0..*cuts {
                    bounds.push(bound(format!("cut_{j}"), a, b));
                }
                for j in 0..=*cuts {
                    bounds.push(bound(format!("level_{j}"), lo, hi));
                }
            }
            PolicyRule::LinearFeedback { features: used } => {
                if used.is_empty() {
                    return Err(bad("linear feedback reads at least one feature"));
                }
                if !finite_interval(coefficient_range.0, coefficient_range.1) {
                    return Err(bad("coefficient range must be finite with lower <= upper"));
                }
                for feature in used {
                    range(*feature)?;
                }
                bounds.push(bound("intercept".into(), lo, hi));
                for (i, _) in used.iter().enumerate() {
                    bounds.push(bound(
                        format!("slope_{i}"),
                        coefficient_range.0,
                        coefficient_range.1,
                    ));
                }
            }
            PolicyRule::Tree { splits } => {
                let nodes = splits.len();
                let depth = (nodes + 1).trailing_zeros();
                if nodes == 0 || (nodes + 1).count_ones() != 1 || depth > MAX_TREE_DEPTH {
                    return Err(bad("a tree needs 1, 3 or 7 internal nodes (depth 1 to 3)"));
                }
                for (i, feature) in splits.iter().enumerate() {
                    let (a, b) = range(*feature)?;
                    bounds.push(bound(format!("threshold_{i}"), a, b));
                }
                for leaf in 0..=nodes {
                    bounds.push(bound(format!("leaf_{leaf}"), lo, hi));
                }
            }
        }
        Ok(Self { id, features, rule, action_lower: lo, action_upper: hi, bounds })
    }

    /// Stable family id.
    #[must_use]
    pub fn id(&self) -> &str {
        &self.id
    }
    /// Declared features.
    #[must_use]
    pub fn features(&self) -> &[PolicyFeature] {
        &self.features
    }
    /// The rule shape.
    #[must_use]
    pub const fn rule(&self) -> &PolicyRule {
        &self.rule
    }
    /// Inclusive action interval every action is clipped into.
    #[must_use]
    pub const fn action_bounds(&self) -> (f64, f64) {
        (self.action_lower, self.action_upper)
    }
    /// The parameter box.
    #[must_use]
    pub fn bounds(&self) -> &[ParameterBound] {
        &self.bounds
    }
    /// Number of parameters.
    #[must_use]
    pub fn dimension(&self) -> usize {
        self.bounds.len()
    }
    /// Whether the rule reads no observation (an open-loop schedule).
    #[must_use]
    pub const fn is_open_loop(&self) -> bool {
        matches!(self.rule, PolicyRule::Schedule { .. })
    }
    /// Steps of history the rule reads.
    #[must_use]
    pub fn max_lag(&self) -> u32 {
        self.used_features().into_iter().map(|f| self.features[f].lag()).max().unwrap_or(0)
    }
    /// Distinct variables the rule reads.
    #[must_use]
    pub fn read_variables(&self) -> Vec<VariableId> {
        let mut variables: Vec<VariableId> =
            self.used_features().into_iter().map(|f| self.features[f].variable()).collect();
        variables.sort_by_key(|v| v.raw());
        variables.dedup();
        variables
    }
    fn used_features(&self) -> Vec<usize> {
        let mut used = match &self.rule {
            PolicyRule::Schedule { .. } => Vec::new(),
            PolicyRule::Threshold { feature } | PolicyRule::Piecewise { feature, .. } => {
                vec![*feature]
            }
            PolicyRule::LinearFeedback { features } => features.clone(),
            PolicyRule::Tree { splits } => splits.clone(),
        };
        used.sort_unstable();
        used.dedup();
        used
    }

    /// Check a parameter vector against the box.
    ///
    /// # Errors
    /// A wrong length, a non-finite value or a value outside its bound.
    pub fn check_parameters(&self, theta: &[f64]) -> Result<(), AdaptivePolicyError> {
        if theta.len() != self.bounds.len() {
            return Err(bad(format!(
                "expected {} parameters, got {}",
                self.bounds.len(),
                theta.len()
            )));
        }
        for (value, b) in theta.iter().zip(&self.bounds) {
            if !value.is_finite() || *value < b.lower || *value > b.upper {
                return Err(bad(format!("parameter {} is outside its declared bound", b.name)));
            }
        }
        Ok(())
    }

    /// The clipped action at `step` given the feature values `features` (one per declared
    /// feature, in declaration order).
    ///
    /// # Errors
    /// An invalid parameter vector or a feature slice of the wrong length.
    pub fn action(
        &self,
        theta: &[f64],
        step: usize,
        features: &[f64],
    ) -> Result<f64, AdaptivePolicyError> {
        self.check_parameters(theta)?;
        if features.len() != self.features.len() {
            return Err(bad("feature slice does not match the declared features"));
        }
        let raw = match &self.rule {
            PolicyRule::Schedule { steps } => theta[step.min(steps - 1)],
            PolicyRule::Threshold { feature } => {
                if features[*feature] >= theta[0] {
                    theta[2]
                } else {
                    theta[1]
                }
            }
            PolicyRule::Piecewise { feature, cuts } => {
                let mut sorted: Vec<f64> = theta[..*cuts].to_vec();
                sorted.sort_by(f64::total_cmp);
                let level = sorted.iter().take_while(|c| features[*feature] >= **c).count();
                theta[*cuts + level]
            }
            PolicyRule::LinearFeedback { features: used } => {
                let mut value = theta[0];
                for (i, feature) in used.iter().enumerate() {
                    value += theta[1 + i] * features[*feature];
                }
                value
            }
            PolicyRule::Tree { splits } => {
                let nodes = splits.len();
                let mut node = 0_usize;
                while node < nodes {
                    node = 2 * node + 1 + usize::from(features[splits[node]] >= theta[node]);
                }
                theta[node]
            }
        };
        Ok(antecedent_core::clip_value(raw, self.action_lower, self.action_upper))
    }

    /// The same policy as a value expression for an assigned intervention at `step`. A rollout
    /// executing it produces exactly the values [`Self::action`] returns.
    ///
    /// # Errors
    /// An invalid parameter vector.
    pub fn to_value_expr(
        &self,
        theta: &[f64],
        step: usize,
    ) -> Result<ValueExpr, AdaptivePolicyError> {
        self.check_parameters(theta)?;
        let c = ValueExpr::constant;
        let feature = |i: usize| self.features[i].expr();
        let (lo, hi) = (self.action_lower, self.action_upper);
        let raw = match &self.rule {
            PolicyRule::Schedule { steps } => c(theta[step.min(steps - 1)]),
            PolicyRule::Threshold { feature: f } => {
                feature(*f).ge(c(theta[0])).if_then_else(c(theta[2]), c(theta[1]))
            }
            PolicyRule::Piecewise { feature: f, cuts } => {
                let mut sorted: Vec<f64> = theta[..*cuts].to_vec();
                sorted.sort_by(f64::total_cmp);
                // Fold from the top level down: the last level applies at or above the top cut.
                let mut expr = c(theta[*cuts + *cuts]);
                for j in (0..*cuts).rev() {
                    expr = feature(*f).ge(c(sorted[j])).if_then_else(expr, c(theta[*cuts + j]));
                }
                expr
            }
            PolicyRule::LinearFeedback { features: used } => {
                let mut expr = c(theta[0]);
                for (i, f) in used.iter().enumerate() {
                    expr = expr.add(c(theta[1 + i]).mul(feature(*f)));
                }
                expr
            }
            PolicyRule::Tree { splits } => self.tree_expr(splits, theta, 0),
        };
        Ok(raw.clip(c(lo), c(hi)))
    }

    fn tree_expr(&self, splits: &[usize], theta: &[f64], node: usize) -> ValueExpr {
        let nodes = splits.len();
        if node >= nodes {
            return ValueExpr::constant(theta[node]);
        }
        let right = self.tree_expr(splits, theta, 2 * node + 2);
        let left = self.tree_expr(splits, theta, 2 * node + 1);
        self.features[splits[node]]
            .expr()
            .ge(ValueExpr::constant(theta[node]))
            .if_then_else(right, left)
    }

    /// Canonical identity of the family (features, rule, bounds, action interval).
    #[must_use]
    pub fn identity(&self) -> [u8; 32] {
        let mut w = CanonicalWriter::new("antecedent.adaptive_policy_family.v1");
        w.text(&self.id).word(self.features.len() as u64);
        for f in &self.features {
            match f {
                PolicyFeature::Current(v) => w.word(1).word(u64::from(v.raw())).word(0),
                PolicyFeature::Lag { variable, lag } => {
                    w.word(2).word(u64::from(variable.raw())).word(u64::from(*lag))
                }
            };
        }
        match &self.rule {
            PolicyRule::Schedule { steps } => w.word(1).word(*steps as u64),
            PolicyRule::Threshold { feature } => w.word(2).word(*feature as u64),
            PolicyRule::Piecewise { feature, cuts } => {
                w.word(3).word(*feature as u64).word(*cuts as u64)
            }
            PolicyRule::LinearFeedback { features } => {
                w.word(4).word(features.len() as u64);
                for f in features {
                    w.word(*f as u64);
                }
                &mut w
            }
            PolicyRule::Tree { splits } => {
                w.word(5).word(splits.len() as u64);
                for f in splits {
                    w.word(*f as u64);
                }
                &mut w
            }
        };
        w.real(self.action_lower).real(self.action_upper);
        for b in &self.bounds {
            w.text(&b.name).real(b.lower).real(b.upper);
        }
        w.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(raw: u32) -> VariableId {
        VariableId::from_raw(raw)
    }

    fn family(rule: PolicyRule) -> AdaptivePolicyFamily {
        AdaptivePolicyFamily::new(
            "f",
            vec![
                PolicyFeature::Lag { variable: v(1), lag: 1 },
                PolicyFeature::Lag { variable: v(1), lag: 2 },
            ],
            &[(-5.0, 5.0), (-5.0, 5.0)],
            rule,
            (0.0, 3.0),
            (-2.0, 2.0),
        )
        .unwrap()
    }

    #[test]
    fn threshold_piecewise_linear_and_tree_rules_pick_the_documented_action() {
        let t = family(PolicyRule::Threshold { feature: 0 });
        assert_eq!(t.action(&[1.0, 0.5, 2.0], 0, &[1.0, 0.0]).unwrap(), 2.0);
        assert_eq!(t.action(&[1.0, 0.5, 2.0], 0, &[0.99, 0.0]).unwrap(), 0.5);
        let p = family(PolicyRule::Piecewise { feature: 0, cuts: 2 });
        // Cuts given out of order are sorted: [-1, 1]; levels 0, 1, 2.
        let theta = [1.0, -1.0, 0.0, 1.0, 2.0];
        assert_eq!(p.action(&theta, 0, &[-3.0, 0.0]).unwrap(), 0.0);
        assert_eq!(p.action(&theta, 0, &[0.0, 0.0]).unwrap(), 1.0);
        assert_eq!(p.action(&theta, 0, &[4.0, 0.0]).unwrap(), 2.0);
        let l = family(PolicyRule::LinearFeedback { features: vec![0, 1] });
        assert_eq!(l.action(&[1.0, 0.5, -1.0], 0, &[2.0, 1.0]).unwrap(), 1.0);
        assert_eq!(l.action(&[1.0, 2.0, 0.0], 0, &[2.0, 0.0]).unwrap(), 3.0, "clipped above");
        let tree = family(PolicyRule::Tree { splits: vec![0, 1, 1] });
        // Root on feature 0 at 0; left node on feature 1 at -1; right node on feature 1 at 1.
        let theta = [0.0, -1.0, 1.0, 0.0, 0.5, 1.5, 2.5];
        assert_eq!(tree.action(&theta, 0, &[-1.0, -2.0]).unwrap(), 0.0);
        assert_eq!(tree.action(&theta, 0, &[-1.0, 0.0]).unwrap(), 0.5);
        assert_eq!(tree.action(&theta, 0, &[1.0, 0.0]).unwrap(), 1.5);
        assert_eq!(tree.action(&theta, 0, &[1.0, 2.0]).unwrap(), 2.5);
    }

    #[test]
    fn invalid_families_and_parameters_are_refused() {
        let ok = |rule| {
            AdaptivePolicyFamily::new(
                "f",
                vec![PolicyFeature::Current(v(0))],
                &[(0.0, 1.0)],
                rule,
                (0.0, 1.0),
                (-1.0, 1.0),
            )
        };
        assert!(ok(PolicyRule::Threshold { feature: 3 }).is_err());
        assert!(ok(PolicyRule::Tree { splits: vec![0, 0] }).is_err());
        assert!(ok(PolicyRule::Tree { splits: vec![0; 15] }).is_err());
        assert!(ok(PolicyRule::Piecewise { feature: 0, cuts: 0 }).is_err());
        let fam = ok(PolicyRule::Threshold { feature: 0 }).unwrap();
        assert!(fam.check_parameters(&[0.5, 0.0]).is_err());
        assert!(fam.check_parameters(&[2.0, 0.0, 0.0]).is_err());
        assert!(fam.check_parameters(&[f64::NAN, 0.0, 0.0]).is_err());
    }

    #[test]
    fn the_lowered_expression_agrees_with_the_direct_action() {
        let cases = [
            (PolicyRule::Threshold { feature: 1 }, vec![0.5, 0.25, 2.0]),
            (PolicyRule::Piecewise { feature: 0, cuts: 2 }, vec![1.0, -1.0, 0.0, 1.0, 2.0]),
            (PolicyRule::LinearFeedback { features: vec![0, 1] }, vec![1.0, 0.5, -1.0]),
            (PolicyRule::Tree { splits: vec![0, 1, 1] }, vec![0.0, -1.0, 1.0, 0.0, 0.5, 1.5, 2.5]),
        ];
        for (rule, theta) in cases {
            let fam = AdaptivePolicyFamily::new(
                "now",
                vec![PolicyFeature::Current(v(0)), PolicyFeature::Current(v(1))],
                &[(-5.0, 5.0), (-5.0, 5.0)],
                rule,
                (0.0, 3.0),
                (-2.0, 2.0),
            )
            .unwrap();
            let expr = fam.to_value_expr(&theta, 0).unwrap();
            for x0 in [-3.0, -1.0, 0.0, 0.7, 4.0] {
                for x1 in [-2.0, -1.0, 0.0, 1.0, 3.0] {
                    let lookup = |var: VariableId| Some(if var == v(0) { x0 } else { x1 });
                    let via_expr = expr.evaluate(None, &lookup).unwrap();
                    assert_eq!(via_expr, fam.action(&theta, 0, &[x0, x1]).unwrap());
                }
            }
        }
    }
}
