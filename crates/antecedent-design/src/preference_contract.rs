//! Decision contracts whose actions are valued through a preference value model.
//!
//! A [`PreferenceContract`] declares actions by stable id, each with one input per value-model
//! attribute: a scientific quantity whose law a source supplies, or a value known without
//! uncertainty. The utility of an action is the model's value with that action's inputs in
//! place of the attributes, so it moves with the preference parameters. At one parameter
//! value, [`PreferenceContract::bind`] gives the ordinary [`DecisionContract`] the existing
//! evaluators read. [`PreferenceContract::term_contract`] exposes each part of the model's
//! linear form as its own pseudo-action, so one evaluation gives the expectation of every
//! part; [`PreferenceContract::probe_contract`] exposes each quantity input on its own, so
//! per-draw attribute values can be read. A [`PreferencePolicy`] says how a result over the
//! preference set may be consumed.

// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::HashSet;

use antecedent_contract::ScientificQuantity;
use antecedent_core::{CanonicalWriter, is_blank};

use crate::decision_contract::{
    ActionKind, DecisionAction, DecisionContract, DecisionContractError, DecisionCriterion,
    HardConstraint, StructuralPolicy, UtilityExpr,
};
use crate::preference_model::{PreferenceExpr, PreferenceModelError, ValueModel};

/// What an action supplies for one model attribute.
#[derive(Clone, Debug, PartialEq)]
pub enum ActionInput {
    /// A scientific quantity whose law a source supplies.
    Quantity(ScientificQuantity),
    /// A value known without uncertainty.
    Fixed {
        /// The value; finite.
        value: f64,
        /// Its units; equal to the attribute's units.
        units: String,
    },
}

/// One action valued through the model.
#[derive(Clone, Debug, PartialEq)]
pub struct PreferenceAction {
    /// Stable semantic identity.
    pub id: String,
    /// What kind of action this is.
    pub kind: ActionKind,
    /// One input per model attribute, in attribute order. At least one is a quantity.
    pub inputs: Vec<ActionInput>,
}

/// How a result over the preference set may be consumed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PreferencePolicy {
    /// Choose only an action best at every admissible parameter value.
    RequireInvariantBestAction,
    /// Choose the action with the best worst-case criterion value over the set.
    Maximin,
    /// Choose the action with the smallest worst-case regret over the set.
    MinimaxRegret,
    /// Weight parameter values by a measure over the set; refuses without one.
    Weighted,
    /// Report the answer across the set without choosing.
    ReportOnly,
}

impl PreferencePolicy {
    const fn tag(self) -> u64 {
        match self {
            Self::RequireInvariantBestAction => 1,
            Self::Maximin => 2,
            Self::MinimaxRegret => 3,
            Self::Weighted => 4,
            Self::ReportOnly => 5,
        }
    }
}

/// A decision problem whose utility comes from a value model with uncertain preferences.
#[derive(Clone, Debug, PartialEq)]
pub struct PreferenceContract {
    /// The value model.
    pub model: ValueModel,
    /// Candidate actions, in any order.
    pub actions: Vec<PreferenceAction>,
    /// Scoring criterion at each parameter value.
    pub criterion: DecisionCriterion,
    /// Hard constraints. `expr` indexes model attributes (`Input(k)` is attribute `k`), not
    /// parameters, so a constraint never depends on preferences.
    pub constraints: Vec<HardConstraint>,
    /// Population the decision concerns; every quantity input must be in it.
    pub target_population: String,
    /// Horizon the decision concerns; every quantity input must be at it.
    pub horizon: u32,
    /// Structural-uncertainty policy.
    pub structural_policy: StructuralPolicy,
    /// Preference-uncertainty policy.
    pub preference_policy: PreferencePolicy,
}

/// The ordinary contract that reads each quantity input on its own, with where
/// each action's attributes sit in it.
#[derive(Clone, Debug, PartialEq)]
pub struct ProbeContract {
    /// Contract whose pseudo-actions each have one quantity input and utility `Input(0)`.
    pub contract: DecisionContract,
    /// `probe_index[a][k]` is the pseudo-action reading attribute `k` of action `a`;
    /// `None` when that input is fixed.
    pub probe_index: Vec<Vec<Option<usize>>>,
}

impl PreferenceContract {
    /// Check the model, actions, inputs, scope and constraints.
    ///
    /// # Errors
    /// Any model refusal; a blank target population; an invalid criterion threshold or
    /// level; fewer than two actions; a blank or duplicate action id; an input count other
    /// than the attribute count; an action with no quantity input; a non-finite fixed value;
    /// fixed or quantity units that differ from the attribute's; an invalid quantity or one
    /// outside the target population or horizon; a constraint with a blank or duplicate id,
    /// blank units, a non-finite bound or constant, a probability outside `(0, 1]`, an
    /// unknown attribute or an unknown action.
    pub fn validate(&self) -> Result<(), PreferenceContractError> {
        use PreferenceContractError as E;
        self.model.validate().map_err(E::Model)?;
        if is_blank(&self.target_population) {
            return Err(E::BlankTargetPopulation);
        }
        let criterion_valid = match self.criterion {
            DecisionCriterion::ThresholdProbability { threshold } => threshold.is_finite(),
            DecisionCriterion::Quantile { p } => p > 0.0 && p < 1.0,
            _ => true,
        };
        if !criterion_valid {
            return Err(E::InvalidCriterion);
        }
        if self.actions.len() < 2 {
            return Err(E::TooFewActions { count: self.actions.len() });
        }
        let mut ids = HashSet::new();
        for (position, action) in self.actions.iter().enumerate() {
            if is_blank(&action.id) {
                return Err(E::BlankActionId { position });
            }
            if !ids.insert(action.id.as_str()) {
                return Err(E::DuplicateAction { id: action.id.clone() });
            }
            self.validate_inputs(action)?;
        }
        let mut constraint_ids = HashSet::new();
        for (position, constraint) in self.constraints.iter().enumerate() {
            if is_blank(&constraint.id) {
                return Err(E::BlankConstraintId { position });
            }
            if !constraint_ids.insert(constraint.id.as_str()) {
                return Err(E::DuplicateConstraint { id: constraint.id.clone() });
            }
            let named = || constraint.id.clone();
            if is_blank(&constraint.units) {
                return Err(E::BlankConstraintUnits { constraint: named() });
            }
            if !constraint.bound.is_finite() || !utility_constants_finite(&constraint.expr) {
                return Err(E::ConstraintNonFinite { constraint: named() });
            }
            if !(constraint.min_probability > 0.0 && constraint.min_probability <= 1.0) {
                return Err(E::ConstraintProbability { constraint: named() });
            }
            if let Some(index) = constraint
                .expr
                .inputs_used()
                .into_iter()
                .find(|k| *k >= self.model.attributes.len())
            {
                return Err(E::ConstraintUnknownAttribute { constraint: named(), index });
            }
            if let Some(unknown) = constraint.applies_to.iter().find(|a| !ids.contains(a.as_str()))
            {
                return Err(E::ConstraintUnknownAction {
                    constraint: named(),
                    action: unknown.clone(),
                });
            }
        }
        Ok(())
    }

    fn validate_inputs(&self, action: &PreferenceAction) -> Result<(), PreferenceContractError> {
        use PreferenceContractError as E;
        let expected = self.model.attributes.len();
        if action.inputs.len() != expected {
            return Err(E::InputCount {
                action: action.id.clone(),
                expected,
                found: action.inputs.len(),
            });
        }
        let mut any_quantity = false;
        for (input, declared) in action.inputs.iter().zip(&self.model.attributes) {
            let at = || (action.id.clone(), declared.id.clone());
            match input {
                ActionInput::Fixed { value, units } => {
                    if !value.is_finite() {
                        let (action, attribute) = at();
                        return Err(E::NonFiniteFixedInput { action, attribute });
                    }
                    if *units != declared.units {
                        let (action, attribute) = at();
                        return Err(E::FixedUnitsMismatch { action, attribute });
                    }
                }
                ActionInput::Quantity(quantity) => {
                    any_quantity = true;
                    if quantity.validate().is_err() {
                        let (action, attribute) = at();
                        return Err(E::InvalidQuantity { action, attribute });
                    }
                    if quantity.units != declared.units {
                        let (action, attribute) = at();
                        return Err(E::QuantityUnitsMismatch { action, attribute });
                    }
                    if quantity.population_id != self.target_population
                        || quantity.horizon != self.horizon
                    {
                        let (action, attribute) = at();
                        return Err(E::OutsideScope { action, attribute });
                    }
                }
            }
        }
        if !any_quantity {
            return Err(E::NoQuantityInput { action: action.id.clone() });
        }
        Ok(())
    }

    /// Canonical BLAKE3 identity as lowercase hex, over the model identity, the actions and
    /// their inputs, the constraints, the criterion, scope and both policies. Independent of
    /// the order actions and constraints are declared in.
    ///
    /// # Errors
    /// An invalid contract has no identity.
    pub fn identity(&self) -> Result<String, PreferenceContractError> {
        self.validate()?;
        let model = self.model.identity().map_err(PreferenceContractError::Model)?;
        let mut actions: Vec<[u8; 32]> = self.actions.iter().map(encode_action).collect();
        actions.sort_unstable();
        let mut constraints: Vec<[u8; 32]> =
            self.constraints.iter().map(encode_constraint).collect();
        constraints.sort_unstable();
        let mut out = CanonicalWriter::new("antecedent.preference_contract.v1");
        out.text(&model).text(&self.target_population).word(u64::from(self.horizon));
        encode_criterion(self.criterion, &mut out);
        out.word(structural_tag(self.structural_policy)).word(self.preference_policy.tag());
        for list in [&actions, &constraints] {
            out.word(list.len() as u64);
            for item in list {
                out.bytes(item);
            }
        }
        Ok(out.finish_hex())
    }

    /// The ordinary contract at one parameter value.
    ///
    /// Each action's utility is the model value with the parameters and the action's fixed
    /// inputs substituted, every subexpression that is then free of attributes folded to its
    /// constant, and the remaining attributes renumbered to the action's quantity inputs:
    /// the quantities, in attribute order, become `Input(0), Input(1), ...`. The bound
    /// utility evaluated on quantity values equals [`ValueModel::value_at`] with those values
    /// and the fixed values in place.
    ///
    /// Constraint expressions are rewritten the same way per action (an attribute whose input
    /// is fixed becomes its constant). When the rewrite is the same for every action a
    /// constraint applies to, the constraint keeps its id and `applies_to`; otherwise it
    /// becomes one constraint per action, id `"{constraint}@{action}"`, applying to that
    /// action alone, in action declaration order.
    ///
    /// # Errors
    /// An invalid contract; a parameter count other than the dimension or a non-finite
    /// parameter; a non-finite folded constant; a bound value that still holds `Exp` or `Ln`
    /// of a quantity attribute (`not_lowerable`); a refusal of the ordinary contract.
    pub fn bind(&self, parameters: &[f64]) -> Result<DecisionContract, PreferenceContractError> {
        use PreferenceContractError as E;
        self.validate()?;
        let dimension = self.model.preferences.dimension();
        if parameters.len() != dimension {
            return Err(E::ParameterCount { expected: dimension, found: parameters.len() });
        }
        if let Some(index) = parameters.iter().position(|value| !value.is_finite()) {
            return Err(E::NonFiniteParameter { index });
        }
        let value = self.model.value.bind_parameters(parameters).map_err(E::Model)?;
        let mut actions = Vec::with_capacity(self.actions.len());
        for action in &self.actions {
            let (map, inputs) = input_map(action);
            let utility = lower(&value, action, &map)?
                .ok_or_else(|| E::NotLowerable { action: action.id.clone() })?;
            actions.push(DecisionAction {
                id: action.id.clone(),
                kind: action.kind,
                inputs,
                utility,
            });
        }
        let constraints = self.bound_constraints()?;
        self.ordinary(actions, constraints, self.criterion)
    }

    fn bound_constraints(&self) -> Result<Vec<HardConstraint>, PreferenceContractError> {
        let mut out = Vec::new();
        for constraint in &self.constraints {
            let mut rewritten = Vec::new();
            for action in self.actions.iter().filter(|a| {
                constraint.applies_to.is_empty() || constraint.applies_to.contains(&a.id)
            }) {
                let (map, _) = input_map(action);
                let expr = rewrite_constraint(&constraint.expr, action, &map).ok_or_else(|| {
                    PreferenceContractError::ConstraintUnknownAttribute {
                        constraint: constraint.id.clone(),
                        index: constraint.expr.inputs_used().last().copied().unwrap_or(0),
                    }
                })?;
                rewritten.push((action.id.clone(), expr));
            }
            let shared = rewritten
                .windows(2)
                .all(|pair| pair[0].1 == pair[1].1)
                .then(|| rewritten.first().map(|(_, expr)| expr.clone()))
                .flatten();
            if let Some(expr) = shared {
                out.push(HardConstraint { expr, ..constraint.clone() });
            } else {
                for (action, expr) in rewritten {
                    out.push(HardConstraint {
                        id: format!("{}@{action}", constraint.id),
                        expr,
                        applies_to: vec![action],
                        ..constraint.clone()
                    });
                }
            }
        }
        Ok(out)
    }

    fn ordinary(
        &self,
        actions: Vec<DecisionAction>,
        constraints: Vec<HardConstraint>,
        criterion: DecisionCriterion,
    ) -> Result<DecisionContract, PreferenceContractError> {
        let contract = DecisionContract {
            actions,
            utility_units: self.model.utility_units.clone(),
            criterion,
            constraints,
            target_population: self.target_population.clone(),
            horizon: self.horizon,
            structural_policy: self.structural_policy,
        };
        contract.validate().map_err(PreferenceContractError::Contract)?;
        Ok(contract)
    }

    /// Ordinary contract whose pseudo-actions are the parts of the model's linear form, so
    /// the existing evaluators give the expectation of each part.
    ///
    /// With `K = 1 + dimension`, pseudo-action `a * K + j` (actions in declaration order) has
    /// id `"{action}#term{j}"`, the action's kind and quantity inputs, and utility
    /// `fixed` for `j = 0` and `terms[j - 1]` otherwise, lowered as in [`Self::bind`]. A part
    /// that reads no quantity (a constant `c`, including an absent term) is written
    /// `c + 0 * Input(0)` so the ordinary contract accepts it; its expectation is `c`. The
    /// criterion is [`DecisionCriterion::PosteriorExpectedUtility`] and there are no
    /// constraints (they are read through [`Self::bind`]). Every part shares the model's
    /// utility units in the ordinary contract, although part `j > 0` is in utility units per
    /// unit of parameter `j - 1`.
    ///
    /// `None` when the model has no linear form or a part holds `Exp` or `Ln` of a quantity
    /// attribute.
    ///
    /// # Errors
    /// An invalid contract, a non-finite folded constant, or a refusal of the ordinary
    /// contract (such as more pseudo-actions than it admits).
    pub fn term_contract(&self) -> Result<Option<DecisionContract>, PreferenceContractError> {
        self.validate()?;
        let Some(form) = self.model.linear_form() else {
            return Ok(None);
        };
        let parts: Vec<&PreferenceExpr> = std::iter::once(&form.fixed).chain(&form.terms).collect();
        let mut actions = Vec::with_capacity(self.actions.len() * parts.len());
        for action in &self.actions {
            let (map, inputs) = input_map(action);
            for (j, part) in parts.iter().enumerate() {
                let Some(mut utility) = lower(part, action, &map)? else {
                    return Ok(None);
                };
                if utility.inputs_used().is_empty() {
                    utility = UtilityExpr::sum(
                        utility,
                        UtilityExpr::product(UtilityExpr::Const(0.0), UtilityExpr::Input(0)),
                    );
                }
                actions.push(DecisionAction {
                    id: format!("{}#term{j}", action.id),
                    kind: action.kind,
                    inputs: inputs.clone(),
                    utility,
                });
            }
        }
        self.ordinary(actions, Vec::new(), DecisionCriterion::PosteriorExpectedUtility).map(Some)
    }

    /// Ordinary contract whose pseudo-actions each read one quantity input, with utility
    /// `Input(0)`, used to read per-draw attribute values.
    ///
    /// For each action (declaration order) and each attribute `k` (index order) whose input
    /// is a quantity, one pseudo-action with id `"{action}#probe{k}"`, the action's kind and
    /// that quantity alone. `probe_index[a][k]` is that pseudo-action's index, `None` for a
    /// fixed input. A valid contract has at least two actions with a quantity each, so there
    /// are always at least two pseudo-actions. The criterion is
    /// [`DecisionCriterion::PosteriorExpectedUtility`] and there are no constraints.
    ///
    /// # Errors
    /// An invalid contract or a refusal of the ordinary contract.
    pub fn probe_contract(&self) -> Result<ProbeContract, PreferenceContractError> {
        self.validate()?;
        let mut actions = Vec::new();
        let mut probe_index = Vec::with_capacity(self.actions.len());
        for action in &self.actions {
            let mut row = Vec::with_capacity(action.inputs.len());
            for (k, input) in action.inputs.iter().enumerate() {
                match input {
                    ActionInput::Quantity(quantity) => {
                        row.push(Some(actions.len()));
                        actions.push(DecisionAction {
                            id: format!("{}#probe{k}", action.id),
                            kind: action.kind,
                            inputs: vec![quantity.clone()],
                            utility: UtilityExpr::Input(0),
                        });
                    }
                    ActionInput::Fixed { .. } => row.push(None),
                }
            }
            probe_index.push(row);
        }
        let contract =
            self.ordinary(actions, Vec::new(), DecisionCriterion::PosteriorExpectedUtility)?;
        Ok(ProbeContract { contract, probe_index })
    }
}

/// `attribute_to_input` for an action (quantities numbered densely in attribute order) and
/// the quantities in that order.
fn input_map(action: &PreferenceAction) -> (Vec<Option<usize>>, Vec<ScientificQuantity>) {
    let mut map = Vec::with_capacity(action.inputs.len());
    let mut quantities = Vec::new();
    for input in &action.inputs {
        match input {
            ActionInput::Quantity(quantity) => {
                map.push(Some(quantities.len()));
                quantities.push(quantity.clone());
            }
            ActionInput::Fixed { .. } => map.push(None),
        }
    }
    (map, quantities)
}

/// Substitute the action's fixed inputs, fold closed subexpressions, and lower.
fn lower(
    expr: &PreferenceExpr,
    action: &PreferenceAction,
    map: &[Option<usize>],
) -> Result<Option<UtilityExpr>, PreferenceContractError> {
    let mut bound = expr.clone();
    for (k, input) in action.inputs.iter().enumerate() {
        if let ActionInput::Fixed { value, .. } = input {
            bound = bound.bind_attribute(k, *value);
        }
    }
    let folded = fold(&bound).map_err(PreferenceContractError::Model)?;
    Ok(folded.to_utility_expr(map))
}

/// Replace every subexpression free of attributes and parameters by its value. The fold
/// evaluates with the same arithmetic as [`PreferenceExpr::evaluate`], so the folded
/// expression gives bit-identical values.
fn fold(expr: &PreferenceExpr) -> Result<PreferenceExpr, PreferenceModelError> {
    use PreferenceExpr as P;
    if !matches!(expr, P::Const(_))
        && expr.attributes_used().is_empty()
        && expr.parameters_used().is_empty()
    {
        return Ok(P::Const(expr.evaluate(&[], &[])?));
    }
    Ok(match expr {
        P::Const(_) | P::Attribute(_) | P::Parameter(_) => expr.clone(),
        P::Neg(a) => P::negate(fold(a)?),
        P::Exp(a) => P::exp(fold(a)?),
        P::Ln(a) => P::ln(fold(a)?),
        P::Add(a, b) => P::sum(fold(a)?, fold(b)?),
        P::Sub(a, b) => P::difference(fold(a)?, fold(b)?),
        P::Mul(a, b) => P::product(fold(a)?, fold(b)?),
        P::Min(a, b) => P::minimum(fold(a)?, fold(b)?),
        P::Max(a, b) => P::maximum(fold(a)?, fold(b)?),
    })
}

/// Rewrite a constraint over attributes into one over the action's quantity inputs.
fn rewrite_constraint(
    expr: &UtilityExpr,
    action: &PreferenceAction,
    map: &[Option<usize>],
) -> Option<UtilityExpr> {
    use UtilityExpr as U;
    let go = |e: &UtilityExpr| rewrite_constraint(e, action, map).map(Box::new);
    Some(match expr {
        U::Const(value) => U::Const(*value),
        U::Input(k) => match (map.get(*k)?, action.inputs.get(*k)?) {
            (Some(input), _) => U::Input(*input),
            (None, ActionInput::Fixed { value, .. }) => U::Const(*value),
            (None, ActionInput::Quantity(_)) => return None,
        },
        U::Neg(a) => U::Neg(go(a)?),
        U::Add(a, b) => U::Add(go(a)?, go(b)?),
        U::Sub(a, b) => U::Sub(go(a)?, go(b)?),
        U::Mul(a, b) => U::Mul(go(a)?, go(b)?),
        U::Min(a, b) => U::Min(go(a)?, go(b)?),
        U::Max(a, b) => U::Max(go(a)?, go(b)?),
    })
}

fn utility_constants_finite(expr: &UtilityExpr) -> bool {
    use UtilityExpr as U;
    match expr {
        U::Const(value) => value.is_finite(),
        U::Input(_) => true,
        U::Neg(a) => utility_constants_finite(a),
        U::Add(a, b) | U::Sub(a, b) | U::Mul(a, b) | U::Min(a, b) | U::Max(a, b) => {
            utility_constants_finite(a) && utility_constants_finite(b)
        }
    }
}

fn encode_utility(expr: &UtilityExpr, out: &mut CanonicalWriter) {
    use UtilityExpr as U;
    fn pair(tag: u64, a: &UtilityExpr, b: &UtilityExpr, out: &mut CanonicalWriter) {
        out.word(tag);
        encode_utility(a, out);
        encode_utility(b, out);
    }
    match expr {
        U::Const(value) => {
            out.word(1).real(*value);
        }
        U::Input(k) => {
            out.word(2).word(*k as u64);
        }
        U::Neg(a) => {
            out.word(3);
            encode_utility(a, out);
        }
        U::Add(a, b) => pair(4, a, b, out),
        U::Sub(a, b) => pair(5, a, b, out),
        U::Mul(a, b) => pair(6, a, b, out),
        U::Min(a, b) => pair(7, a, b, out),
        U::Max(a, b) => pair(8, a, b, out),
    }
}

const fn kind_tag(kind: ActionKind) -> u64 {
    match kind {
        ActionKind::Intervention => 1,
        ActionKind::Policy => 2,
        ActionKind::Regime => 3,
        ActionKind::Study => 4,
        ActionKind::External => 5,
    }
}

const fn structural_tag(policy: StructuralPolicy) -> u64 {
    match policy {
        StructuralPolicy::RequireInvariantBestAction => 1,
        StructuralPolicy::Maximin => 2,
        StructuralPolicy::BayesOverStructures => 3,
        StructuralPolicy::ReportOnly => 4,
    }
}

fn encode_criterion(criterion: DecisionCriterion, out: &mut CanonicalWriter) {
    use DecisionCriterion as C;
    match criterion {
        C::PosteriorExpectedUtility => {
            out.word(1);
        }
        C::PosteriorExpectedLoss => {
            out.word(2);
        }
        C::ThresholdProbability { threshold } => {
            out.word(3).real(threshold);
        }
        C::Quantile { p } => {
            out.word(4).real(p);
        }
        C::MinimaxOverIdentifiedSet => {
            out.word(5);
        }
        C::MaximinOverStructures => {
            out.word(6);
        }
        C::Regret => {
            out.word(7);
        }
        C::ExpectedRegret => {
            out.word(8);
        }
    }
}

fn encode_action(action: &PreferenceAction) -> [u8; 32] {
    let mut out = CanonicalWriter::new("antecedent.preference_contract.action.v1");
    out.text(&action.id).word(kind_tag(action.kind)).word(action.inputs.len() as u64);
    for input in &action.inputs {
        match input {
            ActionInput::Quantity(quantity) => {
                out.word(1);
                quantity.write_canonical(&mut out);
            }
            ActionInput::Fixed { value, units } => {
                out.word(2).real(*value).text(units);
            }
        }
    }
    out.finish()
}

fn encode_constraint(constraint: &HardConstraint) -> [u8; 32] {
    let mut out = CanonicalWriter::new("antecedent.preference_contract.constraint.v1");
    out.text(&constraint.id);
    encode_utility(&constraint.expr, &mut out);
    out.real(constraint.bound).real(constraint.min_probability).text(&constraint.units);
    let mut applies: Vec<&String> = constraint.applies_to.iter().collect();
    applies.sort();
    out.texts(&applies);
    out.finish()
}

/// Why a preference contract, or a contract derived from it, is refused.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum PreferenceContractError {
    /// The value model is refused.
    #[error("value model: {0}")]
    Model(PreferenceModelError),
    /// The derived ordinary contract is refused.
    #[error("decision contract: {0}")]
    Contract(DecisionContractError),
    /// The target population is blank.
    #[error("the target population is blank")]
    BlankTargetPopulation,
    /// The criterion threshold or level is invalid.
    #[error("the criterion threshold or level is invalid")]
    InvalidCriterion,
    /// Fewer than two actions.
    #[error("a decision compares at least two actions, found {count}")]
    TooFewActions {
        /// Actions declared.
        count: usize,
    },
    /// An action id is blank.
    #[error("action at position {position} has a blank id")]
    BlankActionId {
        /// Position of the action.
        position: usize,
    },
    /// Two actions share an id.
    #[error("duplicate action {id}")]
    DuplicateAction {
        /// The repeated id.
        id: String,
    },
    /// An action does not give one input per attribute.
    #[error("action {action} gives {found} inputs for {expected} attributes")]
    InputCount {
        /// Action id.
        action: String,
        /// Attributes declared.
        expected: usize,
        /// Inputs given.
        found: usize,
    },
    /// An action reads no quantity, so no source informs it.
    #[error("action {action} has no quantity input")]
    NoQuantityInput {
        /// Action id.
        action: String,
    },
    /// A fixed input is not finite.
    #[error("action {action} has a non-finite fixed {attribute}")]
    NonFiniteFixedInput {
        /// Action id.
        action: String,
        /// Attribute id.
        attribute: String,
    },
    /// A fixed input's units differ from the attribute's.
    #[error("action {action} fixes {attribute} in other units")]
    FixedUnitsMismatch {
        /// Action id.
        action: String,
        /// Attribute id.
        attribute: String,
    },
    /// A quantity coordinate is invalid.
    #[error("action {action} has an invalid quantity for {attribute}")]
    InvalidQuantity {
        /// Action id.
        action: String,
        /// Attribute id.
        attribute: String,
    },
    /// A quantity's units differ from the attribute's.
    #[error("action {action} reads {attribute} in other units")]
    QuantityUnitsMismatch {
        /// Action id.
        action: String,
        /// Attribute id.
        attribute: String,
    },
    /// A quantity is outside the target population or horizon.
    #[error("action {action} reads {attribute} outside the decision scope")]
    OutsideScope {
        /// Action id.
        action: String,
        /// Attribute id.
        attribute: String,
    },
    /// A constraint id is blank.
    #[error("constraint at position {position} has a blank id")]
    BlankConstraintId {
        /// Position of the constraint.
        position: usize,
    },
    /// Two constraints share an id.
    #[error("duplicate constraint {id}")]
    DuplicateConstraint {
        /// The repeated id.
        id: String,
    },
    /// A constraint's units are blank.
    #[error("constraint {constraint} has blank units")]
    BlankConstraintUnits {
        /// Constraint id.
        constraint: String,
    },
    /// A constraint bound or constant is not finite.
    #[error("constraint {constraint} has a non-finite bound or constant")]
    ConstraintNonFinite {
        /// Constraint id.
        constraint: String,
    },
    /// A constraint probability is outside `(0, 1]`.
    #[error("constraint {constraint} has a probability outside (0, 1]")]
    ConstraintProbability {
        /// Constraint id.
        constraint: String,
    },
    /// A constraint reads an attribute that does not exist.
    #[error("constraint {constraint} reads unknown attribute {index}")]
    ConstraintUnknownAttribute {
        /// Constraint id.
        constraint: String,
        /// The unknown attribute index.
        index: usize,
    },
    /// A constraint names an action that is not declared.
    #[error("constraint {constraint} names unknown action {action}")]
    ConstraintUnknownAction {
        /// Constraint id.
        constraint: String,
        /// The unknown action id.
        action: String,
    },
    /// A parameter point has the wrong length.
    #[error("expected {expected} parameter values, found {found}")]
    ParameterCount {
        /// The set's dimension.
        expected: usize,
        /// Values given.
        found: usize,
    },
    /// A parameter value is not finite.
    #[error("parameter value {index} is not finite")]
    NonFiniteParameter {
        /// Position of the value.
        index: usize,
    },
    /// The bound value still holds `Exp` or `Ln` of a quantity attribute, which the
    /// ordinary contract cannot express.
    #[error("the value of action {action} cannot be lowered to a decision utility")]
    NotLowerable {
        /// Action id.
        action: String,
    },
}

impl PreferenceContractError {
    /// Stable detail id, prefixed `preference_contract.`; a model refusal keeps its own
    /// `preference_model.` detail.
    #[must_use]
    pub const fn detail(&self) -> &'static str {
        use antecedent_core::reason_details;
        match self {
            Self::Model(inner) => inner.detail(),
            Self::Contract(_) => reason_details::preference_contract::DECISION_CONTRACT_REFUSED,
            Self::BlankTargetPopulation => {
                reason_details::preference_contract::TARGET_POPULATION_BLANK
            }
            Self::InvalidCriterion => reason_details::preference_contract::CRITERION_INVALID,
            Self::TooFewActions { .. } => reason_details::preference_contract::TOO_FEW_ACTIONS,
            Self::BlankActionId { .. } => reason_details::preference_contract::ACTION_ID_BLANK,
            Self::DuplicateAction { .. } => reason_details::preference_contract::ACTION_DUPLICATE,
            Self::InputCount { .. } => reason_details::preference_contract::INPUT_COUNT_MISMATCH,
            Self::NoQuantityInput { .. } => {
                reason_details::preference_contract::ACTION_WITHOUT_QUANTITY
            }
            Self::NonFiniteFixedInput { .. } => {
                reason_details::preference_contract::FIXED_INPUT_NON_FINITE
            }
            Self::FixedUnitsMismatch { .. } => {
                reason_details::preference_contract::FIXED_UNITS_MISMATCH
            }
            Self::InvalidQuantity { .. } => reason_details::preference_contract::QUANTITY_INVALID,
            Self::QuantityUnitsMismatch { .. } => {
                reason_details::preference_contract::QUANTITY_UNITS_MISMATCH
            }
            Self::OutsideScope { .. } => {
                reason_details::preference_contract::QUANTITY_OUTSIDE_SCOPE
            }
            Self::BlankConstraintId { .. } => {
                reason_details::preference_contract::CONSTRAINT_ID_BLANK
            }
            Self::DuplicateConstraint { .. } => {
                reason_details::preference_contract::CONSTRAINT_DUPLICATE
            }
            Self::BlankConstraintUnits { .. } => {
                reason_details::preference_contract::CONSTRAINT_UNITS_BLANK
            }
            Self::ConstraintNonFinite { .. } => {
                reason_details::preference_contract::CONSTRAINT_NON_FINITE
            }
            Self::ConstraintProbability { .. } => {
                reason_details::preference_contract::CONSTRAINT_PROBABILITY
            }
            Self::ConstraintUnknownAttribute { .. } => {
                reason_details::preference_contract::CONSTRAINT_UNKNOWN_ATTRIBUTE
            }
            Self::ConstraintUnknownAction { .. } => {
                reason_details::preference_contract::CONSTRAINT_UNKNOWN_ACTION
            }
            Self::ParameterCount { .. } => {
                reason_details::preference_contract::PARAMETER_COUNT_MISMATCH
            }
            Self::NonFiniteParameter { .. } => {
                reason_details::preference_contract::PARAMETER_NON_FINITE
            }
            Self::NotLowerable { .. } => reason_details::preference_contract::NOT_LOWERABLE,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn folding_replaces_closed_subexpressions_and_keeps_attributes() {
        use PreferenceExpr as P;
        let expr = P::sum(P::exp(P::Const(0.0)), P::product(P::ln(P::Const(1.0)), P::Attribute(0)));
        assert_eq!(
            fold(&expr).unwrap(),
            P::sum(P::Const(1.0), P::product(P::Const(0.0), P::Attribute(0)))
        );
        assert_eq!(fold(&P::ln(P::Const(-1.0))), Err(PreferenceModelError::NonFiniteValue));
    }

    #[test]
    fn utility_encoding_separates_operators() {
        let digest = |e: &UtilityExpr| {
            let mut out = CanonicalWriter::new("t");
            encode_utility(e, &mut out);
            out.finish_hex()
        };
        let a = UtilityExpr::sum(UtilityExpr::Input(0), UtilityExpr::Input(1));
        let b = UtilityExpr::difference(UtilityExpr::Input(0), UtilityExpr::Input(1));
        assert_ne!(digest(&a), digest(&b));
    }
}
