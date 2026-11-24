//! Serde wires of the decision contract parts every decision artifact stores: utility
//! expressions, criteria, hard constraints, constraint exclusions, and the names of action kinds
//! and structural policies.
//!
//! The ordinary contract and result artifacts and the artifacts of decisions across a
//! preference set write these parts with the same wires, so the same part has the same bytes
//! in every artifact, and a host that declares a contract part as JSON reads it with the same
//! wire. Each wire converts to and from the decision type it carries.

// SPDX-License-Identifier: MIT OR Apache-2.0

use serde::{Deserialize, Serialize};

use crate::decision_contract::{
    ActionKind, DecisionCriterion, HardConstraint, StructuralPolicy, UtilityExpr,
};
use crate::decision_eval::ConstraintExclusion;

/// A utility expression.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExprWire {
    /// [`UtilityExpr::Const`].
    Const(f64),
    /// [`UtilityExpr::Input`].
    Input(usize),
    /// [`UtilityExpr::Neg`].
    Neg(Box<ExprWire>),
    /// [`UtilityExpr::Add`].
    Add(Box<ExprWire>, Box<ExprWire>),
    /// [`UtilityExpr::Sub`].
    Sub(Box<ExprWire>, Box<ExprWire>),
    /// [`UtilityExpr::Mul`].
    Mul(Box<ExprWire>, Box<ExprWire>),
    /// [`UtilityExpr::Min`].
    Min(Box<ExprWire>, Box<ExprWire>),
    /// [`UtilityExpr::Max`].
    Max(Box<ExprWire>, Box<ExprWire>),
}

impl From<&UtilityExpr> for ExprWire {
    fn from(expr: &UtilityExpr) -> Self {
        let b = |e: &UtilityExpr| Box::new(Self::from(e));
        match expr {
            UtilityExpr::Const(v) => Self::Const(*v),
            UtilityExpr::Input(k) => Self::Input(*k),
            UtilityExpr::Neg(a) => Self::Neg(b(a)),
            UtilityExpr::Add(x, y) => Self::Add(b(x), b(y)),
            UtilityExpr::Sub(x, y) => Self::Sub(b(x), b(y)),
            UtilityExpr::Mul(x, y) => Self::Mul(b(x), b(y)),
            UtilityExpr::Min(x, y) => Self::Min(b(x), b(y)),
            UtilityExpr::Max(x, y) => Self::Max(b(x), b(y)),
        }
    }
}

impl From<ExprWire> for UtilityExpr {
    fn from(wire: ExprWire) -> Self {
        let b = |e: Box<ExprWire>| Box::new(Self::from(*e));
        match wire {
            ExprWire::Const(v) => Self::Const(v),
            ExprWire::Input(k) => Self::Input(k),
            ExprWire::Neg(a) => Self::Neg(b(a)),
            ExprWire::Add(x, y) => Self::Add(b(x), b(y)),
            ExprWire::Sub(x, y) => Self::Sub(b(x), b(y)),
            ExprWire::Mul(x, y) => Self::Mul(b(x), b(y)),
            ExprWire::Min(x, y) => Self::Min(b(x), b(y)),
            ExprWire::Max(x, y) => Self::Max(b(x), b(y)),
        }
    }
}

/// A decision criterion.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CriterionWire {
    /// [`DecisionCriterion::PosteriorExpectedUtility`].
    PosteriorExpectedUtility,
    /// [`DecisionCriterion::PosteriorExpectedLoss`].
    PosteriorExpectedLoss,
    /// [`DecisionCriterion::ThresholdProbability`].
    ThresholdProbability {
        /// The utility threshold.
        threshold: f64,
    },
    /// [`DecisionCriterion::Quantile`].
    Quantile {
        /// The level.
        p: f64,
    },
    /// [`DecisionCriterion::MinimaxOverIdentifiedSet`].
    MinimaxOverIdentifiedSet,
    /// [`DecisionCriterion::MaximinOverStructures`].
    MaximinOverStructures,
    /// [`DecisionCriterion::Regret`].
    Regret,
    /// [`DecisionCriterion::ExpectedRegret`].
    ExpectedRegret,
}

impl From<DecisionCriterion> for CriterionWire {
    fn from(c: DecisionCriterion) -> Self {
        match c {
            DecisionCriterion::PosteriorExpectedUtility => Self::PosteriorExpectedUtility,
            DecisionCriterion::PosteriorExpectedLoss => Self::PosteriorExpectedLoss,
            DecisionCriterion::ThresholdProbability { threshold } => {
                Self::ThresholdProbability { threshold }
            }
            DecisionCriterion::Quantile { p } => Self::Quantile { p },
            DecisionCriterion::MinimaxOverIdentifiedSet => Self::MinimaxOverIdentifiedSet,
            DecisionCriterion::MaximinOverStructures => Self::MaximinOverStructures,
            DecisionCriterion::Regret => Self::Regret,
            DecisionCriterion::ExpectedRegret => Self::ExpectedRegret,
        }
    }
}

impl From<CriterionWire> for DecisionCriterion {
    fn from(c: CriterionWire) -> Self {
        match c {
            CriterionWire::PosteriorExpectedUtility => Self::PosteriorExpectedUtility,
            CriterionWire::PosteriorExpectedLoss => Self::PosteriorExpectedLoss,
            CriterionWire::ThresholdProbability { threshold } => {
                Self::ThresholdProbability { threshold }
            }
            CriterionWire::Quantile { p } => Self::Quantile { p },
            CriterionWire::MinimaxOverIdentifiedSet => Self::MinimaxOverIdentifiedSet,
            CriterionWire::MaximinOverStructures => Self::MaximinOverStructures,
            CriterionWire::Regret => Self::Regret,
            CriterionWire::ExpectedRegret => Self::ExpectedRegret,
        }
    }
}

/// A hard constraint.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConstraintWire {
    id: String,
    expr: ExprWire,
    bound: f64,
    min_probability: f64,
    units: String,
    applies_to: Vec<String>,
}

impl From<&HardConstraint> for ConstraintWire {
    fn from(c: &HardConstraint) -> Self {
        Self {
            id: c.id.clone(),
            expr: ExprWire::from(&c.expr),
            bound: c.bound,
            min_probability: c.min_probability,
            units: c.units.clone(),
            applies_to: c.applies_to.clone(),
        }
    }
}

impl From<ConstraintWire> for HardConstraint {
    fn from(c: ConstraintWire) -> Self {
        Self {
            id: c.id,
            expr: c.expr.into(),
            bound: c.bound,
            min_probability: c.min_probability,
            units: c.units,
            applies_to: c.applies_to,
        }
    }
}

/// A hard-constraint exclusion of one action.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExclusionWire {
    constraint_id: String,
    probability: f64,
    required: f64,
}

impl From<&ConstraintExclusion> for ExclusionWire {
    fn from(e: &ConstraintExclusion) -> Self {
        Self {
            constraint_id: e.constraint_id.clone(),
            probability: e.probability,
            required: e.required,
        }
    }
}

impl From<ExclusionWire> for ConstraintExclusion {
    fn from(e: ExclusionWire) -> Self {
        Self { constraint_id: e.constraint_id, probability: e.probability, required: e.required }
    }
}

/// The stored name of an action kind.
#[must_use]
pub const fn kind_name(kind: ActionKind) -> &'static str {
    match kind {
        ActionKind::Intervention => "intervention",
        ActionKind::Policy => "policy",
        ActionKind::Regime => "regime",
        ActionKind::Study => "study",
        ActionKind::External => "external",
    }
}

/// The action kind a stored name names; `None` for any other name.
#[must_use]
pub fn kind_named(name: &str) -> Option<ActionKind> {
    Some(match name {
        "intervention" => ActionKind::Intervention,
        "policy" => ActionKind::Policy,
        "regime" => ActionKind::Regime,
        "study" => ActionKind::Study,
        "external" => ActionKind::External,
        _ => return None,
    })
}

/// The stored name of a structural policy.
#[must_use]
pub const fn structural_name(policy: StructuralPolicy) -> &'static str {
    match policy {
        StructuralPolicy::RequireInvariantBestAction => "require_invariant_best_action",
        StructuralPolicy::Maximin => "maximin",
        StructuralPolicy::BayesOverStructures => "bayes_over_structures",
        StructuralPolicy::ReportOnly => "report_only",
    }
}

/// The structural policy a stored name names; `None` for any other name.
#[must_use]
pub fn structural_named(name: &str) -> Option<StructuralPolicy> {
    Some(match name {
        "require_invariant_best_action" => StructuralPolicy::RequireInvariantBestAction,
        "maximin" => StructuralPolicy::Maximin,
        "bayes_over_structures" => StructuralPolicy::BayesOverStructures,
        "report_only" => StructuralPolicy::ReportOnly,
        _ => return None,
    })
}
