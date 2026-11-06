//! Closed utility expressions over sensitivity surfaces.
// SPDX-License-Identifier: MIT OR Apache-2.0

use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, HashMap};

/// A closed utility expression over named surface quantities.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UtilityTerm {
    /// A constant.
    Const(f64),
    /// The surface quantity with this `variable_id`.
    Quantity(String),
    /// Negation.
    Neg(Box<UtilityTerm>),
    /// Sum.
    Add(Box<UtilityTerm>, Box<UtilityTerm>),
    /// Difference.
    Sub(Box<UtilityTerm>, Box<UtilityTerm>),
    /// Product.
    Mul(Box<UtilityTerm>, Box<UtilityTerm>),
    /// Pointwise minimum.
    Min(Box<UtilityTerm>, Box<UtilityTerm>),
    /// Pointwise maximum.
    Max(Box<UtilityTerm>, Box<UtilityTerm>),
}

impl UtilityTerm {
    /// The surface quantity `variable_id`.
    #[must_use]
    pub fn quantity(variable_id: &str) -> Self {
        Self::Quantity(variable_id.to_owned())
    }

    /// Build `left + right`.
    #[must_use]
    pub fn sum(left: Self, right: Self) -> Self {
        Self::Add(Box::new(left), Box::new(right))
    }

    /// Build `left - right`.
    #[must_use]
    pub fn difference(left: Self, right: Self) -> Self {
        Self::Sub(Box::new(left), Box::new(right))
    }

    /// Build `left * right`.
    #[must_use]
    pub fn product(left: Self, right: Self) -> Self {
        Self::Mul(Box::new(left), Box::new(right))
    }

    pub(super) fn depth(&self) -> usize {
        match self {
            Self::Const(_) | Self::Quantity(_) => 1,
            Self::Neg(a) => 1 + a.depth(),
            Self::Add(a, b)
            | Self::Sub(a, b)
            | Self::Mul(a, b)
            | Self::Min(a, b)
            | Self::Max(a, b) => 1 + a.depth().max(b.depth()),
        }
    }

    /// Nodes of the expression. The recursion is as deep as the expression, which the caller
    /// has already bounded by the depth limit.
    pub(super) fn node_count(&self) -> usize {
        match self {
            Self::Const(_) | Self::Quantity(_) => 1,
            Self::Neg(a) => a.node_count().saturating_add(1),
            Self::Add(a, b)
            | Self::Sub(a, b)
            | Self::Mul(a, b)
            | Self::Min(a, b)
            | Self::Max(a, b) => a.node_count().saturating_add(b.node_count()).saturating_add(1),
        }
    }

    pub(super) fn all_finite(&self) -> bool {
        match self {
            Self::Const(value) => value.is_finite(),
            Self::Quantity(_) => true,
            Self::Neg(a) => a.all_finite(),
            Self::Add(a, b)
            | Self::Sub(a, b)
            | Self::Mul(a, b)
            | Self::Min(a, b)
            | Self::Max(a, b) => a.all_finite() && b.all_finite(),
        }
    }

    fn collect(&self, into: &mut Vec<String>) {
        match self {
            Self::Const(_) => {}
            Self::Quantity(id) => into.push(id.clone()),
            Self::Neg(a) => a.collect(into),
            Self::Add(a, b)
            | Self::Sub(a, b)
            | Self::Mul(a, b)
            | Self::Min(a, b)
            | Self::Max(a, b) => {
                a.collect(into);
                b.collect(into);
            }
        }
    }

    /// Variable ids the expression reads, ascending and distinct.
    #[must_use]
    pub fn references(&self) -> Vec<String> {
        let mut found = Vec::new();
        self.collect(&mut found);
        found.sort();
        found.dedup();
        found
    }

    /// Whether the expression is multilinear in its quantities (no minimum or maximum,
    /// no quantity multiplied by itself). The difference of two multilinear utilities
    /// attains its extremes at the vertices of a box, so the vertex scenarios then
    /// certify the whole assumption range at a grid point.
    #[must_use]
    pub fn is_multilinear(&self) -> bool {
        self.multilinear_quantities().is_some()
    }

    /// Quantity ids when `self` is multilinear, in one walk. `None` when a minimum,
    /// a maximum, or a product of overlapping quantities is present.
    fn multilinear_quantities(&self) -> Option<BTreeSet<&str>> {
        match self {
            Self::Const(_) => Some(BTreeSet::new()),
            Self::Quantity(id) => Some(BTreeSet::from([id.as_str()])),
            Self::Neg(a) => a.multilinear_quantities(),
            Self::Add(a, b) | Self::Sub(a, b) => {
                let mut left = a.multilinear_quantities()?;
                left.append(&mut b.multilinear_quantities()?);
                Some(left)
            }
            Self::Mul(a, b) => {
                let mut left = a.multilinear_quantities()?;
                let mut right = b.multilinear_quantities()?;
                if left.intersection(&right).next().is_some() {
                    return None;
                }
                left.append(&mut right);
                Some(left)
            }
            Self::Min(..) | Self::Max(..) => None,
        }
    }

    // Quantities outside the active box are constants at this point. A min/max of
    // constants is also constant; a min/max depending on the box is not multilinear.
    pub(super) fn multilinear_over<'a>(
        &'a self,
        active: &BTreeSet<&str>,
    ) -> Option<BTreeSet<&'a str>> {
        match self {
            Self::Const(_) => Some(BTreeSet::new()),
            Self::Quantity(id) => Some(if active.contains(id.as_str()) {
                BTreeSet::from([id.as_str()])
            } else {
                BTreeSet::new()
            }),
            Self::Neg(a) => a.multilinear_over(active),
            Self::Add(a, b) | Self::Sub(a, b) | Self::Mul(a, b) => {
                let mut left = a.multilinear_over(active)?;
                let mut right = b.multilinear_over(active)?;
                if matches!(self, Self::Mul(..)) && !left.is_disjoint(&right) {
                    return None;
                }
                left.append(&mut right);
                Some(left)
            }
            Self::Min(a, b) | Self::Max(a, b) => {
                let left = a.multilinear_over(active)?;
                let right = b.multilinear_over(active)?;
                if left.is_empty() && right.is_empty() { Some(left) } else { None }
            }
        }
    }

    /// Evaluate with each quantity's value in `values`; `None` for an unknown quantity.
    #[must_use]
    pub fn evaluate(&self, values: &HashMap<&str, f64>) -> Option<f64> {
        Some(match self {
            Self::Const(value) => *value,
            Self::Quantity(id) => *values.get(id.as_str())?,
            Self::Neg(a) => -a.evaluate(values)?,
            Self::Add(a, b) => a.evaluate(values)? + b.evaluate(values)?,
            Self::Sub(a, b) => a.evaluate(values)? - b.evaluate(values)?,
            Self::Mul(a, b) => a.evaluate(values)? * b.evaluate(values)?,
            Self::Min(a, b) => a.evaluate(values)?.min(b.evaluate(values)?),
            Self::Max(a, b) => a.evaluate(values)?.max(b.evaluate(values)?),
        })
    }

    /// Units of the expression: `None` for a pure constant, which adopts any units.
    pub(super) fn units(&self, by_id: &HashMap<&str, &str>) -> Result<Option<String>, String> {
        match self {
            Self::Const(_) => Ok(None),
            Self::Quantity(id) => by_id
                .get(id.as_str())
                .map(|units| Some((*units).to_owned()))
                .ok_or_else(|| format!("unknown quantity `{id}`")),
            Self::Neg(a) => a.units(by_id),
            Self::Add(a, b) | Self::Sub(a, b) | Self::Min(a, b) | Self::Max(a, b) => {
                merge_units(a.units(by_id)?, b.units(by_id)?)
            }
            Self::Mul(a, b) => Ok(match (a.units(by_id)?, b.units(by_id)?) {
                (Some(x), Some(y)) => Some(format!("{x}*{y}")),
                (Some(x), None) | (None, Some(x)) => Some(x),
                (None, None) => None,
            }),
        }
    }
}

pub(super) fn merge_units(
    left: Option<String>,
    right: Option<String>,
) -> Result<Option<String>, String> {
    match (left, right) {
        (Some(x), Some(y)) if x != y => Err(format!("`{x}` and `{y}`")),
        (Some(x), _) | (None, Some(x)) => Ok(Some(x)),
        (None, None) => Ok(None),
    }
}
