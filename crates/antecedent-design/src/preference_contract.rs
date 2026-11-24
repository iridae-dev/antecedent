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

use antecedent_contract::{ExternalRefusal, ScientificQuantity};
use antecedent_core::{CanonicalWriter, is_blank};

use crate::decision_contract::{
    ActionKind, DecisionAction, DecisionContract, DecisionContractError, DecisionCriterion,
    HardConstraint, StructuralPolicy, UtilityExpr, constraint_digest,
};
use crate::decision_eval::ConstraintExclusion;
use crate::preference_model::{PreferenceExpr, PreferenceModelError, ValueModel};
use crate::preference_refusal::{CERTIFIED_ROUTE, DECLARE, at, refusal};
use crate::signal::at_stage;

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
            if !constraint.bound.is_finite() || !constraint.expr.all_finite() {
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
    /// their inputs, the constraints, the criterion, scope and both policies. The actions and
    /// constraints are written as sorted digests, then their ids in declaration order: results
    /// are reported in action order, so contracts that declare the same actions in another
    /// order have different identities.
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
        self.criterion.encode(&mut out);
        out.word(u64::from(self.structural_policy.tag())).word(self.preference_policy.tag());
        for list in [&actions, &constraints] {
            out.word(list.len() as u64);
            for item in list {
                out.bytes(item);
            }
        }
        let action_ids: Vec<&str> = self.actions.iter().map(|action| action.id.as_str()).collect();
        let constraint_ids: Vec<&str> =
            self.constraints.iter().map(|constraint| constraint.id.as_str()).collect();
        out.texts(&action_ids).texts(&constraint_ids);
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
    /// action alone, in action declaration order. The evaluations across the preference set
    /// report an exclusion by such a part under the declared constraint id.
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
        let constraints = self.bound_constraints()?.into_iter().map(|(c, _)| c).collect();
        self.ordinary(actions, constraints, self.criterion)
    }

    /// The ordinary [`DecisionContract`] that states this decision, when one exists.
    ///
    /// One exists exactly when the value model declares no preference parameter (every
    /// trade-off is a plain number inside the value) and the value has an ordinary form: it is
    /// the contract [`Self::bind`] gives at the empty parameter point, so the evaluators of
    /// ordinary contracts read it unchanged. A named parameter, even one with a single admissible
    /// value, keeps the decision a decision across a preference set.
    ///
    /// # Errors
    /// An invalid contract; a value model that declares preference parameters
    /// (`no_ordinary_form`); a value that still holds `Exp` or `Ln` of a quantity attribute
    /// (`not_lowerable`).
    pub fn ordinary_contract(&self) -> Result<DecisionContract, PreferenceContractError> {
        self.validate()?;
        let parameters = self.model.preferences.dimension();
        if parameters > 0 {
            return Err(PreferenceContractError::NoOrdinaryForm { parameters });
        }
        self.bind(&[])
    }

    /// Exclusions read from the contract [`Self::bind`] gives, each under the id its constraint
    /// was declared with: a constraint split per action is reported under its own id, not the
    /// `"{constraint}@{action}"` id of its part.
    pub(crate) fn declared_exclusions(
        &self,
        exclusions: Vec<Vec<ConstraintExclusion>>,
    ) -> Result<Vec<Vec<ConstraintExclusion>>, PreferenceContractError> {
        let bound = self.bound_constraints()?;
        Ok(exclusions
            .into_iter()
            .map(|list| {
                list.into_iter()
                    .map(|mut exclusion| {
                        if let Some((_, origin)) =
                            bound.iter().find(|(c, _)| c.id == exclusion.constraint_id)
                        {
                            exclusion.constraint_id.clone_from(&self.constraints[*origin].id);
                        }
                        exclusion
                    })
                    .collect()
            })
            .collect())
    }

    /// The constraints of [`Self::bind`], each with the index of the declared constraint it
    /// comes from.
    fn bound_constraints(&self) -> Result<Vec<(HardConstraint, usize)>, PreferenceContractError> {
        let mut out = Vec::new();
        for (origin, constraint) in self.constraints.iter().enumerate() {
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
                out.push((HardConstraint { expr, ..constraint.clone() }, origin));
            } else {
                for (action, expr) in rewritten {
                    out.push((
                        HardConstraint {
                            id: format!("{}@{action}", constraint.id),
                            expr,
                            applies_to: vec![action],
                            ..constraint.clone()
                        },
                        origin,
                    ));
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

fn encode_action(action: &PreferenceAction) -> [u8; 32] {
    let mut out = CanonicalWriter::new("antecedent.preference_contract.action.v1");
    out.text(&action.id).word(u64::from(action.kind.tag())).word(action.inputs.len() as u64);
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
    constraint_digest("antecedent.preference_contract.constraint.v1", constraint)
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
    /// The value model declares preference parameters, so no single ordinary contract states
    /// the decision.
    #[error(
        "the value model declares {parameters} preference parameter(s); an ordinary contract \
         declares none"
    )]
    NoOrdinaryForm {
        /// Number of declared preference parameters.
        parameters: usize,
    },
}

impl PreferenceContractError {
    /// Stable detail id, prefixed `preference_contract.`; a model refusal keeps its own
    /// `preference_model.` detail and an ordinary-contract refusal its `decision_contract.`
    /// detail, the detail the returned refusal carries.
    #[must_use]
    pub const fn detail(&self) -> &'static str {
        use antecedent_core::reason_details;
        match self {
            Self::Model(inner) => inner.detail(),
            Self::Contract(inner) => inner.detail(),
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
            Self::NoOrdinaryForm { .. } => reason_details::preference_contract::NO_ORDINARY_FORM,
        }
    }
}

impl PreferenceContractError {
    /// Structured refusal for a preference-contract declaration or binding failure.
    ///
    /// A value-model or derived decision-contract refusal keeps its own refusal.
    #[must_use]
    #[allow(clippy::too_many_lines, reason = "one arm per contract declaration failure")]
    pub fn to_refusal(&self) -> ExternalRefusal {
        let invalid = |remedy| {
            refusal(antecedent_core::reason_code!("invalid_argument"), DECLARE, self.detail())
                .remedy(remedy)
        };
        let quantity = |action: &str, attribute: &str, remedy| {
            refusal(
                antecedent_core::reason_code!("quantity_semantics_mismatch"),
                DECLARE,
                self.detail(),
            )
            .offending(at(action, attribute))
            .remedy(remedy)
            .build()
        };
        match self {
            Self::Model(error) => at_stage(error.to_refusal(), DECLARE),
            Self::Contract(error) => at_stage(error.to_refusal(), DECLARE),
            Self::BlankTargetPopulation => {
                invalid("name the population the decision is for").build()
            }
            Self::InvalidCriterion => {
                invalid("supply a finite criterion threshold and a level inside its range").build()
            }
            Self::TooFewActions { count } => invalid("declare at least two actions")
                .expected("2")
                .supplied(count.to_string())
                .build(),
            Self::BlankActionId { position } => {
                invalid("give every action a non-blank id").offending(position.to_string()).build()
            }
            Self::DuplicateAction { id } => {
                invalid("give every action a distinct id").offending(id.clone()).build()
            }
            Self::InputCount { action, expected, found } => {
                invalid("supply one input per attribute of the value model")
                    .offending(action.clone())
                    .expected(expected.to_string())
                    .supplied(found.to_string())
                    .build()
            }
            Self::NoQuantityInput { action } => {
                invalid("read at least one attribute from a scientific quantity")
                    .offending(action.clone())
                    .build()
            }
            Self::NonFiniteFixedInput { action, attribute } => {
                invalid("supply a finite fixed value").offending(at(action, attribute)).build()
            }
            Self::FixedUnitsMismatch { action, attribute } => {
                quantity(action, attribute, "state the fixed value in the attribute's units")
            }
            Self::InvalidQuantity { action, attribute } => quantity(
                action,
                attribute,
                "declare a complete scientific quantity for this attribute",
            ),
            Self::QuantityUnitsMismatch { action, attribute } => {
                quantity(action, attribute, "read a quantity in the attribute's units")
            }
            Self::OutsideScope { action, attribute } => quantity(
                action,
                attribute,
                "read quantities for the decision's target population and horizon",
            ),
            Self::BlankConstraintId { position } => invalid("give every constraint a non-blank id")
                .offending(position.to_string())
                .build(),
            Self::DuplicateConstraint { id } => {
                invalid("give every constraint a distinct id").offending(id.clone()).build()
            }
            Self::BlankConstraintUnits { constraint } => {
                invalid("state the units of the constraint bound")
                    .offending(constraint.clone())
                    .build()
            }
            Self::ConstraintNonFinite { constraint } => {
                invalid("supply a finite bound and finite constants")
                    .offending(constraint.clone())
                    .build()
            }
            Self::ConstraintProbability { constraint } => {
                invalid("supply a constraint probability in (0, 1]")
                    .offending(constraint.clone())
                    .expected("(0, 1]")
                    .build()
            }
            Self::ConstraintUnknownAttribute { constraint, index } => {
                invalid("read only attributes the value model declares")
                    .offending(at(constraint, index))
                    .build()
            }
            Self::ConstraintUnknownAction { constraint, action } => {
                invalid("name only actions the contract declares")
                    .offending(at(constraint, action))
                    .build()
            }
            Self::ParameterCount { expected, found } => {
                invalid("supply one value per preference parameter")
                    .expected(expected.to_string())
                    .supplied(found.to_string())
                    .build()
            }
            Self::NonFiniteParameter { index } => {
                invalid("supply finite parameter values").offending(index.to_string()).build()
            }
            Self::NotLowerable { action } => refusal(
                antecedent_core::reason_code!("route_not_supported"),
                DECLARE,
                self.detail(),
            )
            .offending(action.clone())
            .remedy(CERTIFIED_ROUTE)
            .build(),
            Self::NoOrdinaryForm { parameters } => refusal(
                antecedent_core::reason_code!("route_not_supported"),
                DECLARE,
                self.detail(),
            )
            .remedy(
                "evaluate the contract across its preference set, or state every trade-off as a \
                 plain number",
            )
            .expected("0")
            .supplied(parameters.to_string())
            .build(),
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
            e.encode(&mut out);
            out.finish_hex()
        };
        let a = UtilityExpr::sum(UtilityExpr::Input(0), UtilityExpr::Input(1));
        let b = UtilityExpr::difference(UtilityExpr::Input(0), UtilityExpr::Input(1));
        assert_ne!(digest(&a), digest(&b));
    }
}
