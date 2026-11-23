//! Value models over outcome attributes and preference parameters.
//!
//! A [`ValueModel`] states how outcome attributes (benefit, harm, cost) combine into one
//! utility, as a closed [`PreferenceExpr`] over the attributes and a vector of named
//! preference parameters. Each parameter carries a declared admissible range and the
//! assumption that justifies it; [`LinearRelation`]s restrict the parameters jointly (an
//! ordering of slopes, a recorded choice). The admissible set is a [`PreferenceSet`]: a box
//! cut by linear relations. A result read over that set is an assumption range, not a
//! probability statement.
//!
//! When the value is affine in every parameter, [`ValueModel::linear_form`] returns the exact
//! symbolic decomposition `value = fixed + sum_j theta_j * terms[j]` with parameter-free parts;
//! otherwise [`ValueModel::shape_parameters`] names the parameters that block it. [`shape`] and
//! [`Curve`] build the common parameter-free shapes and piecewise-linear curves.

// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::{BTreeSet, HashSet};
use std::convert::Infallible;

use antecedent_contract::ExternalRefusal;
use antecedent_core::{CanonicalWriter, is_blank};

use crate::decision_contract::UtilityExpr;
use crate::preference_refusal::{DECLARE, at, refusal};

/// Largest expression depth a value model accepts, the same bound as a decision utility.
pub const MAX_PREFERENCE_DEPTH: usize = 64;

/// Closed expression over attributes and preference parameters.
#[derive(Clone, Debug, PartialEq)]
pub enum PreferenceExpr {
    /// A constant.
    Const(f64),
    /// The `k`th attribute of the value model.
    Attribute(usize),
    /// The `j`th preference parameter.
    Parameter(usize),
    /// Negation.
    Neg(Box<PreferenceExpr>),
    /// Sum.
    Add(Box<PreferenceExpr>, Box<PreferenceExpr>),
    /// Difference.
    Sub(Box<PreferenceExpr>, Box<PreferenceExpr>),
    /// Product.
    Mul(Box<PreferenceExpr>, Box<PreferenceExpr>),
    /// Pointwise minimum.
    Min(Box<PreferenceExpr>, Box<PreferenceExpr>),
    /// Pointwise maximum.
    Max(Box<PreferenceExpr>, Box<PreferenceExpr>),
    /// Natural exponential.
    Exp(Box<PreferenceExpr>),
    /// Natural logarithm.
    Ln(Box<PreferenceExpr>),
}

impl PreferenceExpr {
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

    /// Build `max(left, right)`.
    #[must_use]
    pub fn maximum(left: Self, right: Self) -> Self {
        Self::Max(Box::new(left), Box::new(right))
    }

    /// Build `min(left, right)`.
    #[must_use]
    pub fn minimum(left: Self, right: Self) -> Self {
        Self::Min(Box::new(left), Box::new(right))
    }

    /// Build `exp(inner)`.
    #[must_use]
    pub fn exp(inner: Self) -> Self {
        Self::Exp(Box::new(inner))
    }

    /// Build `ln(inner)`.
    #[must_use]
    pub fn ln(inner: Self) -> Self {
        Self::Ln(Box::new(inner))
    }

    /// Build `-inner`.
    #[must_use]
    pub fn negate(inner: Self) -> Self {
        Self::Neg(Box::new(inner))
    }

    /// Visit every node, parents before children, left before right.
    fn visit<'a>(&'a self, f: &mut impl FnMut(&'a Self)) {
        f(self);
        match self {
            Self::Const(_) | Self::Attribute(_) | Self::Parameter(_) => {}
            Self::Neg(a) | Self::Exp(a) | Self::Ln(a) => a.visit(f),
            Self::Add(a, b)
            | Self::Sub(a, b)
            | Self::Mul(a, b)
            | Self::Min(a, b)
            | Self::Max(a, b) => {
                a.visit(f);
                b.visit(f);
            }
        }
    }

    /// Attribute indices the expression reads, ascending and distinct.
    #[must_use]
    pub fn attributes_used(&self) -> Vec<usize> {
        let mut used = BTreeSet::new();
        self.visit(&mut |node| {
            if let Self::Attribute(k) = node {
                used.insert(*k);
            }
        });
        used.into_iter().collect()
    }

    /// Parameter indices the expression reads, ascending and distinct.
    #[must_use]
    pub fn parameters_used(&self) -> Vec<usize> {
        let mut used = BTreeSet::new();
        self.visit(&mut |node| {
            if let Self::Parameter(j) = node {
                used.insert(*j);
            }
        });
        used.into_iter().collect()
    }

    fn has_parameter(&self) -> bool {
        let mut found = false;
        self.visit(&mut |node| found |= matches!(node, Self::Parameter(_)));
        found
    }

    fn has_attribute(&self) -> bool {
        let mut found = false;
        self.visit(&mut |node| found |= matches!(node, Self::Attribute(_)));
        found
    }

    fn depth(&self) -> usize {
        match self {
            Self::Const(_) | Self::Attribute(_) | Self::Parameter(_) => 1,
            Self::Neg(a) | Self::Exp(a) | Self::Ln(a) => 1 + a.depth(),
            Self::Add(a, b)
            | Self::Sub(a, b)
            | Self::Mul(a, b)
            | Self::Min(a, b)
            | Self::Max(a, b) => 1 + a.depth().max(b.depth()),
        }
    }

    fn constants_finite(&self) -> bool {
        let mut finite = true;
        self.visit(&mut |node| {
            if let Self::Const(value) = node {
                finite &= value.is_finite();
            }
        });
        finite
    }

    /// Evaluate at one point of the attributes and parameters.
    ///
    /// Every intermediate value must be finite, so the logarithm of a non-positive number and
    /// an overflowing exponential refuse rather than propagate.
    ///
    /// # Errors
    /// An attribute or parameter index outside the given slices, or a non-finite value at
    /// any node.
    pub fn evaluate(
        &self,
        attributes: &[f64],
        parameters: &[f64],
    ) -> Result<f64, PreferenceModelError> {
        use PreferenceModelError as E;
        let value = match self {
            Self::Const(value) => *value,
            Self::Attribute(k) => *attributes.get(*k).ok_or(E::UnknownAttribute { index: *k })?,
            Self::Parameter(j) => *parameters.get(*j).ok_or(E::UnknownParameter { index: *j })?,
            Self::Neg(a) => -a.evaluate(attributes, parameters)?,
            Self::Add(a, b) => {
                a.evaluate(attributes, parameters)? + b.evaluate(attributes, parameters)?
            }
            Self::Sub(a, b) => {
                a.evaluate(attributes, parameters)? - b.evaluate(attributes, parameters)?
            }
            Self::Mul(a, b) => {
                a.evaluate(attributes, parameters)? * b.evaluate(attributes, parameters)?
            }
            Self::Min(a, b) => {
                a.evaluate(attributes, parameters)?.min(b.evaluate(attributes, parameters)?)
            }
            Self::Max(a, b) => {
                a.evaluate(attributes, parameters)?.max(b.evaluate(attributes, parameters)?)
            }
            Self::Exp(a) => a.evaluate(attributes, parameters)?.exp(),
            Self::Ln(a) => a.evaluate(attributes, parameters)?.ln(),
        };
        if value.is_finite() { Ok(value) } else { Err(E::NonFiniteValue) }
    }

    /// Rebuild the expression with each leaf optionally replaced.
    fn map_leaves<Fail>(
        &self,
        f: &mut impl FnMut(&Self) -> Result<Option<Self>, Fail>,
    ) -> Result<Self, Fail> {
        Ok(match self {
            Self::Const(_) | Self::Attribute(_) | Self::Parameter(_) => {
                f(self)?.unwrap_or_else(|| self.clone())
            }
            Self::Neg(a) => Self::negate(a.map_leaves(f)?),
            Self::Exp(a) => Self::exp(a.map_leaves(f)?),
            Self::Ln(a) => Self::ln(a.map_leaves(f)?),
            Self::Add(a, b) => Self::sum(a.map_leaves(f)?, b.map_leaves(f)?),
            Self::Sub(a, b) => Self::difference(a.map_leaves(f)?, b.map_leaves(f)?),
            Self::Mul(a, b) => Self::product(a.map_leaves(f)?, b.map_leaves(f)?),
            Self::Min(a, b) => Self::minimum(a.map_leaves(f)?, b.map_leaves(f)?),
            Self::Max(a, b) => Self::maximum(a.map_leaves(f)?, b.map_leaves(f)?),
        })
    }

    /// Replace every `Parameter(j)` with `Const(parameters[j])`. The structure is otherwise
    /// unchanged; nothing is folded.
    ///
    /// # Errors
    /// A non-finite parameter value, or a parameter index outside `parameters`.
    pub fn bind_parameters(&self, parameters: &[f64]) -> Result<Self, PreferenceModelError> {
        if let Some(index) = parameters.iter().position(|value| !value.is_finite()) {
            return Err(PreferenceModelError::NonFiniteParameterValue { index });
        }
        self.map_leaves(&mut |leaf| match leaf {
            Self::Parameter(j) => parameters
                .get(*j)
                .map(|value| Some(Self::Const(*value)))
                .ok_or(PreferenceModelError::UnknownParameter { index: *j }),
            _ => Ok(None),
        })
    }

    /// Replace every `Attribute(attribute)` with `Const(value)`.
    #[must_use]
    pub fn bind_attribute(&self, attribute: usize, value: f64) -> Self {
        let Ok(bound) = self.map_leaves::<Infallible>(&mut |leaf| {
            Ok(matches!(leaf, Self::Attribute(k) if *k == attribute).then_some(Self::Const(value)))
        });
        bound
    }

    /// Whether the expression is affine in the attributes, with every parameter read as a
    /// constant. A subexpression free of attributes is a constant whatever its operator.
    #[must_use]
    pub fn is_affine_in_attributes(&self) -> bool {
        match self {
            Self::Const(_) | Self::Attribute(_) | Self::Parameter(_) => true,
            Self::Neg(a) => a.is_affine_in_attributes(),
            Self::Add(a, b) | Self::Sub(a, b) => {
                a.is_affine_in_attributes() && b.is_affine_in_attributes()
            }
            Self::Mul(a, b) => {
                (a.is_affine_in_attributes() && !b.has_attribute())
                    || (!a.has_attribute() && b.is_affine_in_attributes())
            }
            Self::Min(a, b) | Self::Max(a, b) => !a.has_attribute() && !b.has_attribute(),
            Self::Exp(a) | Self::Ln(a) => !a.has_attribute(),
        }
    }

    /// Lower to the decision layer's [`UtilityExpr`]. `None` when a `Parameter`, `Exp` or
    /// `Ln` remains, or an attribute has no input.
    ///
    /// `attribute_to_input[k]` is the input index for attribute `k`; `None` means the
    /// attribute must already be bound.
    #[must_use]
    pub fn to_utility_expr(&self, attribute_to_input: &[Option<usize>]) -> Option<UtilityExpr> {
        let lower = |e: &Self| e.to_utility_expr(attribute_to_input);
        Some(match self {
            Self::Const(value) => UtilityExpr::Const(*value),
            Self::Attribute(k) => {
                UtilityExpr::Input(attribute_to_input.get(*k).copied().flatten()?)
            }
            Self::Parameter(_) | Self::Exp(_) | Self::Ln(_) => return None,
            Self::Neg(a) => UtilityExpr::Neg(Box::new(lower(a)?)),
            Self::Add(a, b) => UtilityExpr::sum(lower(a)?, lower(b)?),
            Self::Sub(a, b) => UtilityExpr::difference(lower(a)?, lower(b)?),
            Self::Mul(a, b) => UtilityExpr::product(lower(a)?, lower(b)?),
            Self::Min(a, b) => UtilityExpr::minimum(lower(a)?, lower(b)?),
            Self::Max(a, b) => UtilityExpr::maximum(lower(a)?, lower(b)?),
        })
    }

    fn encode(&self, out: &mut CanonicalWriter) {
        match self {
            Self::Const(value) => {
                out.word(1).real(*value);
            }
            Self::Attribute(k) => {
                out.word(2).word(*k as u64);
            }
            Self::Parameter(j) => {
                out.word(3).word(*j as u64);
            }
            Self::Neg(a) => unary(out, 4, a),
            Self::Add(a, b) => binary(out, 5, a, b),
            Self::Sub(a, b) => binary(out, 6, a, b),
            Self::Mul(a, b) => binary(out, 7, a, b),
            Self::Min(a, b) => binary(out, 8, a, b),
            Self::Max(a, b) => binary(out, 9, a, b),
            Self::Exp(a) => unary(out, 10, a),
            Self::Ln(a) => unary(out, 11, a),
        }
    }
}

fn unary(out: &mut CanonicalWriter, tag: u64, a: &PreferenceExpr) {
    out.word(tag);
    a.encode(out);
}

fn binary(out: &mut CanonicalWriter, tag: u64, a: &PreferenceExpr, b: &PreferenceExpr) {
    out.word(tag);
    a.encode(out);
    b.encode(out);
}

/// One outcome attribute of a value model.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Attribute {
    /// Stable attribute identity.
    pub id: String,
    /// Units of the attribute.
    pub units: String,
}

/// One preference parameter with its declared admissible range.
#[derive(Clone, Debug, PartialEq)]
pub struct PreferenceParameter {
    /// Stable parameter identity.
    pub id: String,
    /// Utility units per unit of what the parameter multiplies, such as `usd/event`; free
    /// text, non-blank.
    pub units: String,
    /// Lower end of the admissible range; finite.
    pub lower: f64,
    /// Upper end of the admissible range; finite and at least `lower`. `lower == upper`
    /// fixes the parameter.
    pub upper: f64,
    /// Names the declared judgment the range stands on; non-blank.
    pub assumption_id: String,
}

/// Direction of a linear relation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RelationSense {
    /// `sum >= bound`.
    AtLeast,
    /// `sum <= bound`.
    AtMost,
    /// `sum == bound`.
    Equal,
}

impl RelationSense {
    const fn tag(self) -> u64 {
        match self {
            Self::AtLeast => 1,
            Self::AtMost => 2,
            Self::Equal => 3,
        }
    }
}

/// Where a linear relation came from. Kept so a result can name which declaration or which
/// recorded choice binds.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RelationOrigin {
    /// Declared directly by the analyst.
    Declared,
    /// Generated from a declared shape rule (monotone or diminishing slopes).
    ShapeRule,
    /// Implied by a recorded choice: the chosen option was valued at least as highly as
    /// the rejected one.
    Choice {
        /// Identity of the choice record.
        record_id: String,
        /// Identity of the rejected option.
        rejected_option: String,
    },
}

/// `sum_j coefficient_j * theta_j  (>= | <= | ==)  bound`.
#[derive(Clone, Debug, PartialEq)]
pub struct LinearRelation {
    /// Stable relation identity.
    pub id: String,
    /// `(parameter index, coefficient)` pairs; indices distinct.
    pub terms: Vec<(usize, f64)>,
    /// Direction.
    pub sense: RelationSense,
    /// Right-hand side.
    pub bound: f64,
    /// Where the relation came from; part of the identity.
    pub origin: RelationOrigin,
}

impl LinearRelation {
    fn lhs(&self, parameters: &[f64]) -> Option<f64> {
        self.terms.iter().try_fold(0.0, |sum, (index, coefficient)| {
            parameters.get(*index).map(|value| sum + coefficient * value)
        })
    }
}

/// The admissible preference parameters: a box of declared ranges cut by linear relations.
#[derive(Clone, Debug, PartialEq)]
pub struct PreferenceSet {
    /// Parameters, in index order.
    pub parameters: Vec<PreferenceParameter>,
    /// Linear relations between parameters, in any order.
    pub relations: Vec<LinearRelation>,
}

/// Rows of `A theta <= b`.
#[derive(Clone, Debug, PartialEq)]
pub struct PreferenceRows {
    /// Dense rows of `A`, each of the set's dimension.
    pub coefficients: Vec<Vec<f64>>,
    /// Right-hand sides `b`.
    pub rhs: Vec<f64>,
    /// `row_relation[i]` is the index of the relation row `i` came from.
    pub row_relation: Vec<usize>,
}

impl PreferenceSet {
    /// Check the parameters and relations.
    ///
    /// # Errors
    /// A blank or duplicate parameter id, blank units or assumption, a non-finite or
    /// reversed range; a relation with a blank or duplicate id, a blank choice origin, no
    /// terms, an unknown or repeated parameter index, or a non-finite coefficient or bound.
    pub fn validate(&self) -> Result<(), PreferenceModelError> {
        use PreferenceModelError as E;
        let mut ids = HashSet::new();
        for (position, parameter) in self.parameters.iter().enumerate() {
            if is_blank(&parameter.id) {
                return Err(E::BlankParameterId { position });
            }
            if !ids.insert(parameter.id.as_str()) {
                return Err(E::DuplicateParameter { id: parameter.id.clone() });
            }
            if is_blank(&parameter.units) {
                return Err(E::BlankParameterUnits { parameter: parameter.id.clone() });
            }
            if !parameter.lower.is_finite() || !parameter.upper.is_finite() {
                return Err(E::NonFiniteRange { parameter: parameter.id.clone() });
            }
            if parameter.lower > parameter.upper {
                return Err(E::ReversedRange { parameter: parameter.id.clone() });
            }
            if is_blank(&parameter.assumption_id) {
                return Err(E::BlankAssumption { parameter: parameter.id.clone() });
            }
        }
        let mut relation_ids = HashSet::new();
        for (position, relation) in self.relations.iter().enumerate() {
            if is_blank(&relation.id) {
                return Err(E::BlankRelationId { position });
            }
            if !relation_ids.insert(relation.id.as_str()) {
                return Err(E::DuplicateRelation { id: relation.id.clone() });
            }
            if let RelationOrigin::Choice { record_id, rejected_option } = &relation.origin
                && (is_blank(record_id) || is_blank(rejected_option))
            {
                return Err(E::BlankRelationOrigin { relation: relation.id.clone() });
            }
            if relation.terms.is_empty() {
                return Err(E::EmptyRelation { relation: relation.id.clone() });
            }
            let mut seen = HashSet::new();
            for (index, coefficient) in &relation.terms {
                if *index >= self.parameters.len() {
                    return Err(E::RelationUnknownParameter {
                        relation: relation.id.clone(),
                        index: *index,
                    });
                }
                if !seen.insert(*index) {
                    return Err(E::RelationRepeatedParameter {
                        relation: relation.id.clone(),
                        index: *index,
                    });
                }
                if !coefficient.is_finite() {
                    return Err(E::RelationNonFiniteCoefficient { relation: relation.id.clone() });
                }
            }
            if !relation.bound.is_finite() {
                return Err(E::RelationNonFiniteBound { relation: relation.id.clone() });
            }
        }
        Ok(())
    }

    /// Number of parameters.
    #[must_use]
    pub fn dimension(&self) -> usize {
        self.parameters.len()
    }

    /// Whether `parameters` lies in the set, every range and relation allowed to be off by
    /// at most `tolerance`. A length mismatch, a non-finite value, or a negative or
    /// non-finite tolerance is outside.
    #[must_use]
    pub fn contains(&self, parameters: &[f64], tolerance: f64) -> bool {
        if parameters.len() != self.dimension() || !tolerance.is_finite() || tolerance < 0.0 {
            return false;
        }
        let in_box = self.parameters.iter().zip(parameters).all(|(declared, value)| {
            value.is_finite()
                && *value >= declared.lower - tolerance
                && *value <= declared.upper + tolerance
        });
        in_box
            && self.relations.iter().all(|relation| {
                relation.lhs(parameters).is_some_and(|lhs| match relation.sense {
                    RelationSense::AtLeast => lhs >= relation.bound - tolerance,
                    RelationSense::AtMost => lhs <= relation.bound + tolerance,
                    RelationSense::Equal => (lhs - relation.bound).abs() <= tolerance,
                })
            })
    }

    /// Rows of `A theta <= b` for every relation, in relation order, without the parameter
    /// bounds. An `AtLeast` relation is negated; an `Equal` relation gives the `<=` row then
    /// the negated row. A term whose index is outside the dimension is skipped (an invalid
    /// set).
    #[must_use]
    pub fn inequality_rows(&self) -> PreferenceRows {
        let dimension = self.dimension();
        let mut rows =
            PreferenceRows { coefficients: Vec::new(), rhs: Vec::new(), row_relation: Vec::new() };
        for (position, relation) in self.relations.iter().enumerate() {
            let mut dense = vec![0.0; dimension];
            for (index, coefficient) in &relation.terms {
                if let Some(slot) = dense.get_mut(*index) {
                    *slot += coefficient;
                }
            }
            let negated: Vec<f64> = dense.iter().map(|value| -value).collect();
            let mut push = |row: Vec<f64>, rhs: f64| {
                rows.coefficients.push(row);
                rows.rhs.push(rhs);
                rows.row_relation.push(position);
            };
            match relation.sense {
                RelationSense::AtMost => push(dense, relation.bound),
                RelationSense::AtLeast => push(negated, -relation.bound),
                RelationSense::Equal => {
                    push(dense, relation.bound);
                    push(negated, -relation.bound);
                }
            }
        }
        rows
    }

    /// `(lower, upper)` of each parameter, in index order.
    #[must_use]
    pub fn bounds(&self) -> Vec<(f64, f64)> {
        self.parameters.iter().map(|p| (p.lower, p.upper)).collect()
    }

    /// Midpoint of each declared range. It may violate the relations.
    #[must_use]
    pub fn midpoint(&self) -> Vec<f64> {
        self.parameters.iter().map(|p| 0.5 * p.lower + 0.5 * p.upper).collect()
    }
}

/// A value model: the utility of an outcome as a closed expression over its attributes and
/// the preference parameters.
#[derive(Clone, Debug, PartialEq)]
pub struct ValueModel {
    /// Outcome attributes, in index order; the order is semantic.
    pub attributes: Vec<Attribute>,
    /// The admissible preference parameters.
    pub preferences: PreferenceSet,
    /// The value as a function of attributes and parameters.
    pub value: PreferenceExpr,
    /// Units of the value.
    pub utility_units: String,
}

/// `value = fixed + sum_j theta_j * terms[j]`, every part free of parameters.
#[derive(Clone, Debug, PartialEq)]
pub struct LinearForm {
    /// The part that does not move with any parameter.
    pub fixed: PreferenceExpr,
    /// One coefficient expression per parameter (`terms.len()` is the dimension); exactly
    /// `Const(0.0)` where the parameter does not enter.
    pub terms: Vec<PreferenceExpr>,
}

impl ValueModel {
    /// Check the attributes, parameters, relations and expression.
    ///
    /// # Errors
    /// No attributes; a blank or duplicate attribute id or blank attribute units; any
    /// [`PreferenceSet::validate`] refusal; blank utility units; a non-finite constant or an
    /// expression deeper than [`MAX_PREFERENCE_DEPTH`]; an expression reading an unknown
    /// attribute or parameter; a parameter or attribute the expression never reads.
    pub fn validate(&self) -> Result<(), PreferenceModelError> {
        use PreferenceModelError as E;
        if self.attributes.is_empty() {
            return Err(E::NoAttributes);
        }
        let mut ids = HashSet::new();
        for (position, attribute) in self.attributes.iter().enumerate() {
            if is_blank(&attribute.id) {
                return Err(E::BlankAttributeId { position });
            }
            if !ids.insert(attribute.id.as_str()) {
                return Err(E::DuplicateAttribute { id: attribute.id.clone() });
            }
            if is_blank(&attribute.units) {
                return Err(E::BlankAttributeUnits { attribute: attribute.id.clone() });
            }
        }
        self.preferences.validate()?;
        if is_blank(&self.utility_units) {
            return Err(E::BlankUtilityUnits);
        }
        let depth = self.value.depth();
        if depth > MAX_PREFERENCE_DEPTH {
            return Err(E::ExpressionTooDeep { depth });
        }
        if !self.value.constants_finite() {
            return Err(E::NonFiniteConstant);
        }
        let attributes = self.value.attributes_used();
        if let Some(index) = attributes.iter().find(|k| **k >= self.attributes.len()) {
            return Err(E::UnknownAttribute { index: *index });
        }
        let parameters = self.value.parameters_used();
        if let Some(index) = parameters.iter().find(|j| **j >= self.preferences.dimension()) {
            return Err(E::UnknownParameter { index: *index });
        }
        if let Some(unused) = (0..self.preferences.dimension()).find(|j| !parameters.contains(j)) {
            return Err(E::UnusedParameter {
                parameter: self.preferences.parameters[unused].id.clone(),
            });
        }
        if let Some(unused) = (0..self.attributes.len()).find(|k| !attributes.contains(k)) {
            return Err(E::UnusedAttribute { attribute: self.attributes[unused].id.clone() });
        }
        Ok(())
    }

    /// Canonical BLAKE3 identity as lowercase hex.
    ///
    /// Attribute and parameter order is semantic (expressions index them); relation order
    /// is not, and the order of terms within a relation is not. Any semantic edit (an
    /// attribute id or units, a parameter id, units, range or assumption, a relation or its
    /// origin, the expression, the utility units) changes it.
    ///
    /// # Errors
    /// An invalid model has no identity.
    pub fn identity(&self) -> Result<String, PreferenceModelError> {
        self.validate()?;
        let mut relations: Vec<[u8; 32]> =
            self.preferences.relations.iter().map(encode_relation).collect();
        relations.sort_unstable();
        let mut out = CanonicalWriter::new("antecedent.value_model.v1");
        out.word(self.attributes.len() as u64);
        for attribute in &self.attributes {
            out.text(&attribute.id).text(&attribute.units);
        }
        out.word(self.preferences.parameters.len() as u64);
        for parameter in &self.preferences.parameters {
            out.text(&parameter.id)
                .text(&parameter.units)
                .real(parameter.lower)
                .real(parameter.upper)
                .text(&parameter.assumption_id);
        }
        out.word(relations.len() as u64);
        for relation in &relations {
            out.bytes(relation);
        }
        self.value.encode(&mut out);
        out.text(&self.utility_units);
        Ok(out.finish_hex())
    }

    /// The exact symbolic decomposition `value = fixed + sum_j theta_j * terms[j]`.
    ///
    /// `Some` exactly when the value is affine in every parameter: a parameter never
    /// multiplies another parameter-dependent factor and never sits inside `Min`, `Max`,
    /// `Exp` or `Ln`. Parameter-free subexpressions are kept whole (a kink or an exponential
    /// of attributes stays in its part). Trivial simplification drops `+ 0`, `* 1` and
    /// `* 0`, so an absent term is exactly `Const(0.0)`. Intended for a validated model; a
    /// parameter index outside the dimension gives `None`.
    #[must_use]
    pub fn linear_form(&self) -> Option<LinearForm> {
        let affine = decompose(&self.value, self.preferences.dimension())?;
        Some(LinearForm {
            fixed: affine.fixed,
            terms: affine.terms.into_iter().map(|t| if is_zero(&t) { zero() } else { t }).collect(),
        })
    }

    /// Parameters the value is not affine in (shape parameters): those inside `Min`, `Max`,
    /// `Exp` or `Ln`, or in a product whose two factors both depend on parameters.
    /// Ascending and distinct; empty exactly when [`Self::linear_form`] is `Some` for a
    /// validated model.
    #[must_use]
    pub fn shape_parameters(&self) -> Vec<usize> {
        let mut blocking = BTreeSet::new();
        blockers(&self.value, &mut blocking);
        blocking.into_iter().collect()
    }

    /// The value at one point. Checks the lengths only; call [`Self::validate`] first.
    ///
    /// # Errors
    /// A length mismatch with the attributes or parameters, or a non-finite value.
    pub fn value_at(
        &self,
        attributes: &[f64],
        parameters: &[f64],
    ) -> Result<f64, PreferenceModelError> {
        use PreferenceModelError as E;
        if attributes.len() != self.attributes.len() {
            return Err(E::AttributeCount {
                expected: self.attributes.len(),
                found: attributes.len(),
            });
        }
        if parameters.len() != self.preferences.dimension() {
            return Err(E::ParameterCount {
                expected: self.preferences.dimension(),
                found: parameters.len(),
            });
        }
        self.value.evaluate(attributes, parameters)
    }
}

fn encode_relation(relation: &LinearRelation) -> [u8; 32] {
    let mut out = CanonicalWriter::new("antecedent.value_model.relation.v1");
    out.text(&relation.id);
    let mut terms = relation.terms.clone();
    terms.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.total_cmp(&b.1)));
    out.word(terms.len() as u64);
    for (index, coefficient) in terms {
        out.word(index as u64).real(coefficient);
    }
    out.word(relation.sense.tag()).real(relation.bound);
    match &relation.origin {
        RelationOrigin::Declared => {
            out.word(1);
        }
        RelationOrigin::ShapeRule => {
            out.word(2);
        }
        RelationOrigin::Choice { record_id, rejected_option } => {
            out.word(3).text(record_id).text(rejected_option);
        }
    }
    out.finish()
}

/// Running decomposition of a subexpression: its parameter-free part and the coefficient of
/// each parameter.
struct Affine {
    fixed: PreferenceExpr,
    terms: Vec<PreferenceExpr>,
}

const fn zero() -> PreferenceExpr {
    PreferenceExpr::Const(0.0)
}

/// Exactly zero, either sign.
fn is_zero(expr: &PreferenceExpr) -> bool {
    matches!(expr, PreferenceExpr::Const(c) if c.abs().total_cmp(&0.0).is_eq())
}

fn is_one(expr: &PreferenceExpr) -> bool {
    matches!(expr, PreferenceExpr::Const(c) if c.total_cmp(&1.0).is_eq())
}

fn add(left: PreferenceExpr, right: PreferenceExpr) -> PreferenceExpr {
    use PreferenceExpr as P;
    match (left, right) {
        (l, r) if is_zero(&l) => r,
        (l, r) if is_zero(&r) => l,
        (P::Const(a), P::Const(b)) => P::Const(a + b),
        (l, r) => P::sum(l, r),
    }
}

fn sub(left: PreferenceExpr, right: PreferenceExpr) -> PreferenceExpr {
    use PreferenceExpr as P;
    match (left, right) {
        (l, r) if is_zero(&r) => l,
        (l, r) if is_zero(&l) => neg(r),
        (P::Const(a), P::Const(b)) => P::Const(a - b),
        (l, r) => P::difference(l, r),
    }
}

fn neg(inner: PreferenceExpr) -> PreferenceExpr {
    use PreferenceExpr as P;
    match inner {
        e if is_zero(&e) => zero(),
        P::Const(c) => P::Const(-c),
        P::Neg(e) => *e,
        e => P::negate(e),
    }
}

/// Product with `* 0` dropped. The model's evaluation refuses non-finite values, so a zero
/// factor makes the product zero wherever the model is defined.
fn mul(left: PreferenceExpr, right: PreferenceExpr) -> PreferenceExpr {
    use PreferenceExpr as P;
    match (left, right) {
        (l, r) if is_zero(&l) || is_zero(&r) => zero(),
        (l, r) if is_one(&l) => r,
        (l, r) if is_one(&r) => l,
        (P::Const(a), P::Const(b)) => P::Const(a * b),
        (l, r) => P::product(l, r),
    }
}

fn combine(
    left: Affine,
    right: Affine,
    op: fn(PreferenceExpr, PreferenceExpr) -> PreferenceExpr,
) -> Affine {
    Affine {
        fixed: op(left.fixed, right.fixed),
        terms: left.terms.into_iter().zip(right.terms).map(|(l, r)| op(l, r)).collect(),
    }
}

fn scale(factor: &PreferenceExpr, inner: Affine, factor_first: bool) -> Affine {
    let apply = |e: PreferenceExpr| {
        if factor_first { mul(factor.clone(), e) } else { mul(e, factor.clone()) }
    };
    Affine { fixed: apply(inner.fixed), terms: inner.terms.into_iter().map(apply).collect() }
}

fn decompose(expr: &PreferenceExpr, dimension: usize) -> Option<Affine> {
    use PreferenceExpr as P;
    if !expr.has_parameter() {
        return Some(Affine { fixed: expr.clone(), terms: vec![zero(); dimension] });
    }
    match expr {
        P::Parameter(j) => {
            let mut terms = vec![zero(); dimension];
            *terms.get_mut(*j)? = P::Const(1.0);
            Some(Affine { fixed: zero(), terms })
        }
        P::Neg(a) => {
            let a = decompose(a, dimension)?;
            Some(Affine { fixed: neg(a.fixed), terms: a.terms.into_iter().map(neg).collect() })
        }
        P::Add(a, b) => Some(combine(decompose(a, dimension)?, decompose(b, dimension)?, add)),
        P::Sub(a, b) => Some(combine(decompose(a, dimension)?, decompose(b, dimension)?, sub)),
        P::Mul(a, b) => {
            if !a.has_parameter() {
                Some(scale(a, decompose(b, dimension)?, true))
            } else if !b.has_parameter() {
                Some(scale(b, decompose(a, dimension)?, false))
            } else {
                None
            }
        }
        P::Const(_) | P::Attribute(_) | P::Min(..) | P::Max(..) | P::Exp(_) | P::Ln(_) => None,
    }
}

fn blockers(expr: &PreferenceExpr, into: &mut BTreeSet<usize>) {
    use PreferenceExpr as P;
    match expr {
        P::Const(_) | P::Attribute(_) | P::Parameter(_) => {}
        P::Neg(a) => blockers(a, into),
        P::Add(a, b) | P::Sub(a, b) => {
            blockers(a, into);
            blockers(b, into);
        }
        P::Mul(a, b) => {
            if a.has_parameter() && b.has_parameter() {
                into.extend(expr.parameters_used());
            } else {
                blockers(a, into);
                blockers(b, into);
            }
        }
        P::Min(..) | P::Max(..) | P::Exp(_) | P::Ln(_) => into.extend(expr.parameters_used()),
    }
}

/// Building blocks: parameter-free shapes of one or two attributes.
pub mod shape {
    use super::PreferenceExpr as P;

    /// `x`.
    #[must_use]
    pub fn rate(attribute: usize) -> P {
        P::Attribute(attribute)
    }

    /// `clamp(x, from, to) - from`: the part of `x` that falls in `[from, to]`, measured
    /// from `from`. Expects `from <= to`.
    #[must_use]
    pub fn segment(attribute: usize, from: f64, to: f64) -> P {
        P::difference(
            P::minimum(P::maximum(P::Attribute(attribute), P::Const(from)), P::Const(to)),
            P::Const(from),
        )
    }

    /// `max(x - above, 0)`.
    #[must_use]
    pub fn excess(attribute: usize, above: f64) -> P {
        P::maximum(P::difference(P::Attribute(attribute), P::Const(above)), P::Const(0.0))
    }

    /// `max(below - x, 0)`.
    #[must_use]
    pub fn shortfall(attribute: usize, below: f64) -> P {
        P::maximum(P::difference(P::Const(below), P::Attribute(attribute)), P::Const(0.0))
    }

    /// `x_a * x_b`.
    #[must_use]
    pub fn interaction(a: usize, b: usize) -> P {
        P::product(P::Attribute(a), P::Attribute(b))
    }

    /// `min(x_a, x_b)`.
    #[must_use]
    pub fn both(a: usize, b: usize) -> P {
        P::minimum(P::Attribute(a), P::Attribute(b))
    }
}

/// A piecewise-linear curve of one attribute between declared breakpoints; one slope
/// parameter per segment. `slopes[i]` is the parameter index for segment
/// `[breakpoints[i], breakpoints[i+1]]`. The curve is zero at `breakpoints[0]` and flat
/// outside the first and last breakpoints.
#[derive(Clone, Debug, PartialEq)]
pub struct Curve {
    /// Attribute index the curve reads.
    pub attribute: usize,
    /// Strictly increasing, finite breakpoints; at least two.
    pub breakpoints: Vec<f64>,
    /// One parameter index per segment.
    pub slopes: Vec<usize>,
}

impl Curve {
    /// Check the breakpoints and slope count.
    ///
    /// # Errors
    /// Fewer than two breakpoints, a slope count other than one per segment, a non-finite
    /// breakpoint, or breakpoints that do not strictly increase.
    pub fn validate(&self) -> Result<(), PreferenceModelError> {
        use PreferenceModelError as E;
        if self.breakpoints.len() < 2 {
            return Err(E::CurveTooShort { breakpoints: self.breakpoints.len() });
        }
        if self.slopes.len() + 1 != self.breakpoints.len() {
            return Err(E::CurveSlopeCount {
                breakpoints: self.breakpoints.len(),
                slopes: self.slopes.len(),
            });
        }
        if let Some(position) = self.breakpoints.iter().position(|b| !b.is_finite()) {
            return Err(E::CurveBreakpointNonFinite { position });
        }
        if let Some(position) = self.breakpoints.windows(2).position(|w| w[0] >= w[1]) {
            return Err(E::CurveBreakpointsNotIncreasing { position: position + 1 });
        }
        Ok(())
    }

    /// `sum_i Parameter(slopes[i]) * segment(attribute, breakpoints[i], breakpoints[i+1])`,
    /// summed left to right. `Const(0.0)` for a curve with no segment.
    #[must_use]
    pub fn expr(&self) -> PreferenceExpr {
        self.slopes
            .iter()
            .zip(self.breakpoints.windows(2))
            .map(|(slope, edge)| {
                PreferenceExpr::product(
                    PreferenceExpr::Parameter(*slope),
                    shape::segment(self.attribute, edge[0], edge[1]),
                )
            })
            .reduce(PreferenceExpr::sum)
            .unwrap_or_else(zero)
    }

    fn single(&self, id_prefix: &str, rule: &str, sense: RelationSense) -> Vec<LinearRelation> {
        self.slopes
            .iter()
            .enumerate()
            .map(|(i, slope)| LinearRelation {
                id: format!("{id_prefix}.{rule}.{i}"),
                terms: vec![(*slope, 1.0)],
                sense,
                bound: 0.0,
                origin: RelationOrigin::ShapeRule,
            })
            .collect()
    }

    fn adjacent(&self, id_prefix: &str, rule: &str, sign: f64) -> Vec<LinearRelation> {
        self.slopes
            .windows(2)
            .enumerate()
            .filter(|(_, pair)| pair[0] != pair[1])
            .map(|(i, pair)| LinearRelation {
                id: format!("{id_prefix}.{rule}.{i}"),
                terms: vec![(pair[0], sign), (pair[1], -sign)],
                sense: RelationSense::AtLeast,
                bound: 0.0,
                origin: RelationOrigin::ShapeRule,
            })
            .collect()
    }

    /// `slope_i >= 0` for each segment, ids `"{id_prefix}.increasing.{i}"`, origin
    /// [`RelationOrigin::ShapeRule`].
    #[must_use]
    pub fn increasing(&self, id_prefix: &str) -> Vec<LinearRelation> {
        self.single(id_prefix, "increasing", RelationSense::AtLeast)
    }

    /// `slope_i <= 0` for each segment, ids `"{id_prefix}.decreasing.{i}"`.
    #[must_use]
    pub fn decreasing(&self, id_prefix: &str) -> Vec<LinearRelation> {
        self.single(id_prefix, "decreasing", RelationSense::AtMost)
    }

    /// `slope_i - slope_{i+1} >= 0` for each adjacent pair, ids
    /// `"{id_prefix}.diminishing.{i}"`. A pair sharing one parameter is trivially satisfied
    /// and gives no relation.
    #[must_use]
    pub fn diminishing(&self, id_prefix: &str) -> Vec<LinearRelation> {
        self.adjacent(id_prefix, "diminishing", 1.0)
    }

    /// `slope_{i+1} - slope_i >= 0` for each adjacent pair, ids
    /// `"{id_prefix}.accelerating.{i}"`. A pair sharing one parameter gives no relation.
    #[must_use]
    pub fn accelerating(&self, id_prefix: &str) -> Vec<LinearRelation> {
        self.adjacent(id_prefix, "accelerating", -1.0)
    }
}

/// Why a value model, preference set, curve or evaluation is refused.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum PreferenceModelError {
    /// The model declares no attribute.
    #[error("the value model declares no attribute")]
    NoAttributes,
    /// An attribute id is blank.
    #[error("attribute at position {position} has a blank id")]
    BlankAttributeId {
        /// Position of the attribute.
        position: usize,
    },
    /// Two attributes share an id.
    #[error("duplicate attribute {id}")]
    DuplicateAttribute {
        /// The repeated id.
        id: String,
    },
    /// An attribute's units are blank.
    #[error("attribute {attribute} has blank units")]
    BlankAttributeUnits {
        /// Attribute id.
        attribute: String,
    },
    /// A parameter id is blank.
    #[error("parameter at position {position} has a blank id")]
    BlankParameterId {
        /// Position of the parameter.
        position: usize,
    },
    /// Two parameters share an id.
    #[error("duplicate parameter {id}")]
    DuplicateParameter {
        /// The repeated id.
        id: String,
    },
    /// A parameter's units are blank.
    #[error("parameter {parameter} has blank units")]
    BlankParameterUnits {
        /// Parameter id.
        parameter: String,
    },
    /// A parameter range has a non-finite end.
    #[error("parameter {parameter} has a non-finite range")]
    NonFiniteRange {
        /// Parameter id.
        parameter: String,
    },
    /// A parameter range has `lower > upper`.
    #[error("parameter {parameter} has an empty range")]
    ReversedRange {
        /// Parameter id.
        parameter: String,
    },
    /// A parameter names no assumption.
    #[error("parameter {parameter} names no assumption")]
    BlankAssumption {
        /// Parameter id.
        parameter: String,
    },
    /// A relation id is blank.
    #[error("relation at position {position} has a blank id")]
    BlankRelationId {
        /// Position of the relation.
        position: usize,
    },
    /// Two relations share an id.
    #[error("duplicate relation {id}")]
    DuplicateRelation {
        /// The repeated id.
        id: String,
    },
    /// A choice origin names a blank record or rejected option.
    #[error("relation {relation} has a blank choice origin")]
    BlankRelationOrigin {
        /// Relation id.
        relation: String,
    },
    /// A relation has no terms.
    #[error("relation {relation} has no terms")]
    EmptyRelation {
        /// Relation id.
        relation: String,
    },
    /// A relation names a parameter index outside the set.
    #[error("relation {relation} reads unknown parameter {index}")]
    RelationUnknownParameter {
        /// Relation id.
        relation: String,
        /// The unknown index.
        index: usize,
    },
    /// A relation names one parameter twice.
    #[error("relation {relation} repeats parameter {index}")]
    RelationRepeatedParameter {
        /// Relation id.
        relation: String,
        /// The repeated index.
        index: usize,
    },
    /// A relation coefficient is not finite.
    #[error("relation {relation} has a non-finite coefficient")]
    RelationNonFiniteCoefficient {
        /// Relation id.
        relation: String,
    },
    /// A relation bound is not finite.
    #[error("relation {relation} has a non-finite bound")]
    RelationNonFiniteBound {
        /// Relation id.
        relation: String,
    },
    /// The expression holds a non-finite constant.
    #[error("the value expression holds a non-finite constant")]
    NonFiniteConstant,
    /// The expression is deeper than [`MAX_PREFERENCE_DEPTH`].
    #[error("the value expression has depth {depth}")]
    ExpressionTooDeep {
        /// Its depth.
        depth: usize,
    },
    /// An expression reads an attribute index that does not exist.
    #[error("unknown attribute {index}")]
    UnknownAttribute {
        /// The unknown index.
        index: usize,
    },
    /// An expression reads a parameter index that does not exist.
    #[error("unknown parameter {index}")]
    UnknownParameter {
        /// The unknown index.
        index: usize,
    },
    /// A declared parameter is never read by the value.
    #[error("parameter {parameter} is never read by the value")]
    UnusedParameter {
        /// Parameter id.
        parameter: String,
    },
    /// A declared attribute is never read by the value.
    #[error("attribute {attribute} is never read by the value")]
    UnusedAttribute {
        /// Attribute id.
        attribute: String,
    },
    /// The utility units are blank.
    #[error("the value model has blank utility units")]
    BlankUtilityUnits,
    /// An evaluation reached a non-finite value.
    #[error("the value is not finite at this point")]
    NonFiniteValue,
    /// A point has the wrong number of attribute values.
    #[error("expected {expected} attribute values, found {found}")]
    AttributeCount {
        /// Attributes declared.
        expected: usize,
        /// Values given.
        found: usize,
    },
    /// A point has the wrong number of parameter values.
    #[error("expected {expected} parameter values, found {found}")]
    ParameterCount {
        /// Parameters declared.
        expected: usize,
        /// Values given.
        found: usize,
    },
    /// A parameter value to bind is not finite.
    #[error("parameter value {index} is not finite")]
    NonFiniteParameterValue {
        /// Position of the value.
        index: usize,
    },
    /// A curve has fewer than two breakpoints.
    #[error("a curve needs two breakpoints, found {breakpoints}")]
    CurveTooShort {
        /// Breakpoints given.
        breakpoints: usize,
    },
    /// A curve does not have one slope per segment.
    #[error("{breakpoints} breakpoints with {slopes} slopes; a curve has one slope per segment")]
    CurveSlopeCount {
        /// Breakpoints given.
        breakpoints: usize,
        /// Slopes given.
        slopes: usize,
    },
    /// A curve breakpoint is not finite.
    #[error("curve breakpoint {position} is not finite")]
    CurveBreakpointNonFinite {
        /// Position of the breakpoint.
        position: usize,
    },
    /// Curve breakpoints do not strictly increase.
    #[error("curve breakpoint {position} does not exceed the one before it")]
    CurveBreakpointsNotIncreasing {
        /// Position of the first breakpoint not above its predecessor.
        position: usize,
    },
}

impl PreferenceModelError {
    /// Stable detail id, prefixed `preference_model.`.
    #[must_use]
    pub const fn detail(&self) -> &'static str {
        use antecedent_core::reason_details;
        match self {
            Self::NoAttributes => reason_details::preference_model::NO_ATTRIBUTES,
            Self::BlankAttributeId { .. } => reason_details::preference_model::ATTRIBUTE_ID_BLANK,
            Self::DuplicateAttribute { .. } => {
                reason_details::preference_model::ATTRIBUTE_DUPLICATE
            }
            Self::BlankAttributeUnits { .. } => {
                reason_details::preference_model::ATTRIBUTE_UNITS_BLANK
            }
            Self::BlankParameterId { .. } => reason_details::preference_model::PARAMETER_ID_BLANK,
            Self::DuplicateParameter { .. } => {
                reason_details::preference_model::PARAMETER_DUPLICATE
            }
            Self::BlankParameterUnits { .. } => {
                reason_details::preference_model::PARAMETER_UNITS_BLANK
            }
            Self::NonFiniteRange { .. } => {
                reason_details::preference_model::PARAMETER_RANGE_NON_FINITE
            }
            Self::ReversedRange { .. } => reason_details::preference_model::PARAMETER_RANGE_EMPTY,
            Self::BlankAssumption { .. } => {
                reason_details::preference_model::PARAMETER_ASSUMPTION_BLANK
            }
            Self::BlankRelationId { .. } => reason_details::preference_model::RELATION_ID_BLANK,
            Self::DuplicateRelation { .. } => reason_details::preference_model::RELATION_DUPLICATE,
            Self::BlankRelationOrigin { .. } => {
                reason_details::preference_model::RELATION_ORIGIN_BLANK
            }
            Self::EmptyRelation { .. } => reason_details::preference_model::RELATION_WITHOUT_TERMS,
            Self::RelationUnknownParameter { .. } => {
                reason_details::preference_model::RELATION_UNKNOWN_PARAMETER
            }
            Self::RelationRepeatedParameter { .. } => {
                reason_details::preference_model::RELATION_REPEATED_PARAMETER
            }
            Self::RelationNonFiniteCoefficient { .. } => {
                reason_details::preference_model::RELATION_COEFFICIENT_NON_FINITE
            }
            Self::RelationNonFiniteBound { .. } => {
                reason_details::preference_model::RELATION_BOUND_NON_FINITE
            }
            Self::NonFiniteConstant => {
                reason_details::preference_model::EXPRESSION_CONSTANT_NON_FINITE
            }
            Self::ExpressionTooDeep { .. } => reason_details::preference_model::EXPRESSION_TOO_DEEP,
            Self::UnknownAttribute { .. } => reason_details::preference_model::UNKNOWN_ATTRIBUTE,
            Self::UnknownParameter { .. } => reason_details::preference_model::UNKNOWN_PARAMETER,
            Self::UnusedParameter { .. } => reason_details::preference_model::PARAMETER_UNUSED,
            Self::UnusedAttribute { .. } => reason_details::preference_model::ATTRIBUTE_UNUSED,
            Self::BlankUtilityUnits => reason_details::preference_model::UTILITY_UNITS_BLANK,
            Self::NonFiniteValue => reason_details::preference_model::VALUE_NON_FINITE,
            Self::AttributeCount { .. } => {
                reason_details::preference_model::ATTRIBUTE_COUNT_MISMATCH
            }
            Self::ParameterCount { .. } => {
                reason_details::preference_model::PARAMETER_COUNT_MISMATCH
            }
            Self::NonFiniteParameterValue { .. } => {
                reason_details::preference_model::PARAMETER_VALUE_NON_FINITE
            }
            Self::CurveTooShort { .. } => reason_details::preference_model::CURVE_TOO_SHORT,
            Self::CurveSlopeCount { .. } => reason_details::preference_model::CURVE_SLOPE_COUNT,
            Self::CurveBreakpointNonFinite { .. } => {
                reason_details::preference_model::CURVE_BREAKPOINT_NON_FINITE
            }
            Self::CurveBreakpointsNotIncreasing { .. } => {
                reason_details::preference_model::CURVE_BREAKPOINTS_NOT_INCREASING
            }
        }
    }
}

impl PreferenceModelError {
    /// Structured refusal for a value-model, preference-set or curve failure.
    ///
    /// Every cause is a malformed declaration or point, refused as `invalid_argument` at the
    /// `declare` stage.
    #[must_use]
    pub fn to_refusal(&self) -> ExternalRefusal {
        let (offending, remedy) = self.offending_and_remedy();
        refusal(antecedent_core::reason_code!("invalid_argument"), DECLARE, self.detail())
            .offending_opt(offending)
            .remedy(remedy)
            .build()
    }

    #[allow(clippy::too_many_lines, reason = "one arm per declaration failure of the value model")]
    fn offending_and_remedy(&self) -> (Option<String>, &'static str) {
        match self {
            Self::NoAttributes => (None, "declare at least one attribute the value reads"),
            Self::BlankAttributeId { position } => {
                (Some(position.to_string()), "give every attribute a non-blank id")
            }
            Self::DuplicateAttribute { id } => {
                (Some(id.clone()), "give every attribute a distinct id")
            }
            Self::BlankAttributeUnits { attribute } => {
                (Some(attribute.clone()), "state the units of every attribute")
            }
            Self::BlankParameterId { position } => {
                (Some(position.to_string()), "give every parameter a non-blank id")
            }
            Self::DuplicateParameter { id } => {
                (Some(id.clone()), "give every parameter a distinct id")
            }
            Self::BlankParameterUnits { parameter } => {
                (Some(parameter.clone()), "state the units of every parameter")
            }
            Self::NonFiniteRange { parameter } => {
                (Some(parameter.clone()), "declare finite lower and upper values for the range")
            }
            Self::ReversedRange { parameter } => {
                (Some(parameter.clone()), "declare a range whose lower value is at most its upper")
            }
            Self::BlankAssumption { parameter } => {
                (Some(parameter.clone()), "name the assumption the parameter's range stands on")
            }
            Self::BlankRelationId { position } => {
                (Some(position.to_string()), "give every relation a non-blank id")
            }
            Self::DuplicateRelation { id } => {
                (Some(id.clone()), "give every relation a distinct id")
            }
            Self::BlankRelationOrigin { relation } => (
                Some(relation.clone()),
                "name the record and the rejected option the relation comes from",
            ),
            Self::EmptyRelation { relation } => {
                (Some(relation.clone()), "give the relation at least one parameter coefficient")
            }
            Self::RelationUnknownParameter { relation, index } => {
                (Some(at(relation, index)), "read only parameters the preference set declares")
            }
            Self::RelationRepeatedParameter { relation, index } => (
                Some(at(relation, index)),
                "combine the coefficients of the parameter into one term",
            ),
            Self::RelationNonFiniteCoefficient { relation } => {
                (Some(relation.clone()), "supply finite relation coefficients")
            }
            Self::RelationNonFiniteBound { relation } => {
                (Some(relation.clone()), "supply a finite relation bound")
            }
            Self::NonFiniteConstant => (None, "supply finite constants in the value expression"),
            Self::ExpressionTooDeep { depth } => {
                (Some(depth.to_string()), "flatten the value expression")
            }
            Self::UnknownAttribute { index } => {
                (Some(index.to_string()), "read only attributes the value model declares")
            }
            Self::UnknownParameter { index } => {
                (Some(index.to_string()), "read only parameters the value model declares")
            }
            Self::UnusedParameter { parameter } => {
                (Some(parameter.clone()), "remove the parameter or read it in the value")
            }
            Self::UnusedAttribute { attribute } => {
                (Some(attribute.clone()), "remove the attribute or read it in the value")
            }
            Self::BlankUtilityUnits => (None, "state the units the value is measured in"),
            Self::NonFiniteValue => (
                None,
                "narrow the ranges or change the value so it is finite at every evaluated point",
            ),
            Self::AttributeCount { found, .. } => {
                (Some(found.to_string()), "supply one value per declared attribute")
            }
            Self::ParameterCount { found, .. } => {
                (Some(found.to_string()), "supply one value per declared parameter")
            }
            Self::NonFiniteParameterValue { index } => {
                (Some(index.to_string()), "supply finite parameter values")
            }
            Self::CurveTooShort { breakpoints } => {
                (Some(breakpoints.to_string()), "supply at least two curve breakpoints")
            }
            Self::CurveSlopeCount { slopes, .. } => {
                (Some(slopes.to_string()), "supply one slope fewer than the number of breakpoints")
            }
            Self::CurveBreakpointNonFinite { position } => {
                (Some(position.to_string()), "supply finite curve breakpoints")
            }
            Self::CurveBreakpointsNotIncreasing { position } => {
                (Some(position.to_string()), "order the curve breakpoints and remove repeats")
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::PreferenceExpr as P;
    use super::*;

    #[test]
    fn simplification_leaves_exact_zero_and_drops_units() {
        assert_eq!(add(zero(), P::Attribute(0)), P::Attribute(0));
        assert_eq!(sub(P::Attribute(0), zero()), P::Attribute(0));
        assert_eq!(sub(zero(), P::Attribute(0)), P::negate(P::Attribute(0)));
        assert_eq!(mul(P::Const(1.0), P::Attribute(1)), P::Attribute(1));
        assert_eq!(mul(P::Const(-0.0), P::Attribute(1)), zero());
        assert_eq!(neg(P::Const(0.0)), zero());
        assert_eq!(neg(P::negate(P::Attribute(2))), P::Attribute(2));
        assert_eq!(add(P::Const(2.0), P::Const(0.5)), P::Const(2.5));
        assert!(is_zero(&P::Const(-0.0)));
        assert!(!is_zero(&P::Const(1e-300)));
    }

    #[test]
    fn blockers_name_parameters_inside_kinks_and_parameter_products() {
        let value = P::sum(
            P::product(P::Parameter(0), P::Parameter(1)),
            P::sum(
                P::maximum(P::Parameter(2), P::Attribute(0)),
                P::product(P::Parameter(3), P::Attribute(0)),
            ),
        );
        let mut found = BTreeSet::new();
        blockers(&value, &mut found);
        assert_eq!(found.into_iter().collect::<Vec<_>>(), vec![0, 1, 2]);
        assert!(decompose(&value, 4).is_none());
        let affine = P::product(P::Parameter(3), P::Attribute(0));
        let parts = decompose(&affine, 4).unwrap();
        assert_eq!(parts.fixed, zero());
        assert_eq!(parts.terms[3], P::Attribute(0));
        assert_eq!(parts.terms[0], zero());
    }

    #[test]
    fn a_parameter_index_beyond_the_dimension_has_no_decomposition() {
        assert!(decompose(&P::Parameter(2), 2).is_none());
    }
}
