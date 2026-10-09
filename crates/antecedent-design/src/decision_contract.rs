//! Durable 2.3 decision contract and the source each functional needs.
//!
//! A [`DecisionContract`] declares actions by stable semantic ID, each with the
//! scientific quantities its utility reads and a closed, replayable utility
//! expression. Hard constraints exclude actions; they never become penalties.
//! A [`DecisionFunctional`] says which source representation it needs: a mean
//! can satisfy an expectation of a linear utility, a quantile needs a
//! distribution, and a nonlinear utility over several inputs needs aligned
//! joint draws. The contract's [`DecisionContract::identity`] is independent of
//! the order in which actions and constraints were declared and changes with
//! any semantic edit.

use std::collections::HashSet;

use antecedent_core::{ExternalRefusal, ScientificQuantity, SupportStatus};

/// What kind of thing an action is. The kind is part of the action identity.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ActionKind {
    /// A single intervention.
    Intervention,
    /// A treatment policy.
    Policy,
    /// A named regime.
    Regime,
    /// A proposed study.
    Study,
    /// An action defined by an external system.
    External,
}

impl ActionKind {
    const fn tag(self) -> u8 {
        match self {
            Self::Intervention => 1,
            Self::Policy => 2,
            Self::Regime => 3,
            Self::Study => 4,
            Self::External => 5,
        }
    }
}

/// A closed utility expression over an action's input quantities.
#[derive(Clone, Debug, PartialEq)]
pub enum UtilityExpr {
    /// A constant.
    Const(f64),
    /// The `k`th input quantity of the action.
    Input(usize),
    /// Sum.
    Add(Box<UtilityExpr>, Box<UtilityExpr>),
    /// Difference.
    Sub(Box<UtilityExpr>, Box<UtilityExpr>),
    /// Product.
    Mul(Box<UtilityExpr>, Box<UtilityExpr>),
    /// Negation.
    Neg(Box<UtilityExpr>),
    /// Pointwise minimum.
    Min(Box<UtilityExpr>, Box<UtilityExpr>),
    /// Pointwise maximum.
    Max(Box<UtilityExpr>, Box<UtilityExpr>),
}

impl UtilityExpr {
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

    /// Input indices the expression reads, ascending and distinct.
    #[must_use]
    pub fn inputs_used(&self) -> Vec<usize> {
        let mut used = Vec::new();
        self.collect_inputs(&mut used);
        used.sort_unstable();
        used.dedup();
        used
    }

    fn collect_inputs(&self, into: &mut Vec<usize>) {
        match self {
            Self::Const(_) => {}
            Self::Input(k) => into.push(*k),
            Self::Neg(a) => a.collect_inputs(into),
            Self::Add(a, b)
            | Self::Sub(a, b)
            | Self::Mul(a, b)
            | Self::Min(a, b)
            | Self::Max(a, b) => {
                a.collect_inputs(into);
                b.collect_inputs(into);
            }
        }
    }

    fn is_constant(&self) -> bool {
        self.inputs_used().is_empty()
    }

    /// Whether the expression is affine in its inputs, so its expectation is
    /// determined by the input means.
    #[must_use]
    pub fn is_affine(&self) -> bool {
        match self {
            Self::Const(_) | Self::Input(_) => true,
            Self::Neg(a) => a.is_affine(),
            Self::Add(a, b) | Self::Sub(a, b) => a.is_affine() && b.is_affine(),
            Self::Mul(a, b) => {
                (a.is_affine() && b.is_constant()) || (a.is_constant() && b.is_affine())
            }
            Self::Min(..) | Self::Max(..) => false,
        }
    }

    /// Largest depth, bounded at validation.
    fn depth(&self) -> usize {
        match self {
            Self::Const(_) | Self::Input(_) => 1,
            Self::Neg(a) => 1 + a.depth(),
            Self::Add(a, b)
            | Self::Sub(a, b)
            | Self::Mul(a, b)
            | Self::Min(a, b)
            | Self::Max(a, b) => 1 + a.depth().max(b.depth()),
        }
    }

    fn all_finite(&self) -> bool {
        match self {
            Self::Const(value) => value.is_finite(),
            Self::Input(_) => true,
            Self::Neg(a) => a.all_finite(),
            Self::Add(a, b)
            | Self::Sub(a, b)
            | Self::Mul(a, b)
            | Self::Min(a, b)
            | Self::Max(a, b) => a.all_finite() && b.all_finite(),
        }
    }

    /// Evaluate on one realization of the inputs.
    ///
    /// # Errors
    /// An input index outside `inputs` refuses.
    pub fn evaluate(&self, inputs: &[f64]) -> Result<f64, DecisionContractError> {
        Ok(match self {
            Self::Const(value) => *value,
            Self::Input(k) => *inputs.get(*k).ok_or(DecisionContractError::UnknownInput(*k))?,
            Self::Neg(a) => -a.evaluate(inputs)?,
            Self::Add(a, b) => a.evaluate(inputs)? + b.evaluate(inputs)?,
            Self::Sub(a, b) => a.evaluate(inputs)? - b.evaluate(inputs)?,
            Self::Mul(a, b) => a.evaluate(inputs)? * b.evaluate(inputs)?,
            Self::Min(a, b) => a.evaluate(inputs)?.min(b.evaluate(inputs)?),
            Self::Max(a, b) => a.evaluate(inputs)?.max(b.evaluate(inputs)?),
        })
    }

    fn encode(&self, out: &mut Vec<u8>) {
        match self {
            Self::Const(value) => {
                out.push(1);
                out.extend_from_slice(&value.to_bits().to_le_bytes());
            }
            Self::Input(k) => {
                out.push(2);
                out.extend_from_slice(&(*k as u64).to_le_bytes());
            }
            Self::Neg(a) => {
                out.push(3);
                a.encode(out);
            }
            Self::Add(a, b) => binary(out, 4, a, b),
            Self::Sub(a, b) => binary(out, 5, a, b),
            Self::Mul(a, b) => binary(out, 6, a, b),
            Self::Min(a, b) => binary(out, 7, a, b),
            Self::Max(a, b) => binary(out, 8, a, b),
        }
    }
}

fn binary(out: &mut Vec<u8>, tag: u8, a: &UtilityExpr, b: &UtilityExpr) {
    out.push(tag);
    a.encode(out);
    b.encode(out);
}

/// Representation a provider or artifact can supply for a scientific claim.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum SourceRepresentation {
    /// Means only.
    Mean,
    /// Means and the full covariance.
    MeanAndCovariance,
    /// A cumulative distribution function of each input.
    Cdf,
    /// A quantile function of each input.
    QuantileFunction,
    /// Draws of each input, not paired across inputs.
    MarginalDraws,
    /// Aligned joint draws: row `i` is one realization of every input.
    JointDraws,
}

/// The ways a source can satisfy a functional.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SourceRequirement {
    /// Any one of these representations suffices.
    pub any_of: Vec<SourceRepresentation>,
    /// Whether a sampled representation must also report a Monte Carlo error
    /// receipt, because the functional is then estimated rather than exact.
    pub sampled_needs_error_receipt: bool,
}

impl SourceRequirement {
    /// Check `available` against the requirement.
    ///
    /// # Errors
    /// Reports the representations that would have sufficed.
    pub fn check(
        &self,
        available: &[SourceRepresentation],
    ) -> Result<SourceMode, DecisionContractError> {
        for needed in &self.any_of {
            if available.contains(needed) {
                let sampled = matches!(
                    needed,
                    SourceRepresentation::MarginalDraws | SourceRepresentation::JointDraws
                );
                return Ok(SourceMode {
                    representation: *needed,
                    needs_error_receipt: sampled && self.sampled_needs_error_receipt,
                });
            }
        }
        Err(DecisionContractError::MissingSource { needed: self.any_of.clone() })
    }
}

/// The representation chosen to satisfy a requirement.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SourceMode {
    /// Representation that will be used.
    pub representation: SourceRepresentation,
    /// Whether the result must carry a Monte Carlo error receipt.
    pub needs_error_receipt: bool,
}

/// Which side of a threshold a probability or tail refers to.
#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum Tail {
    /// At or below.
    Lower,
    /// At or above.
    Upper,
}

/// A typed functional of an action's utility expression.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum DecisionFunctional {
    /// `E[expr]`.
    Expectation,
    /// `Var[expr]`.
    Variance,
    /// `P(expr <= threshold)` or `P(expr >= threshold)`.
    Probability {
        /// Threshold.
        threshold: f64,
        /// Direction.
        tail: Tail,
    },
    /// The `p`th quantile of `expr`.
    Quantile {
        /// Level in `(0, 1)`.
        p: f64,
    },
    /// `E[expr]` of the action's utility specifically.
    ExpectedUtility,
    /// `E[expr | expr in the p-tail]`.
    TailExpectation {
        /// Tail mass in `(0, 1)`.
        p: f64,
        /// Which tail.
        tail: Tail,
    },
}

impl DecisionFunctional {
    /// Source representations that can answer this functional for `expr`.
    ///
    /// A generic functional type does not license every native route: this only
    /// states what the source must supply.
    #[must_use]
    pub fn requirement(&self, expr: &UtilityExpr) -> SourceRequirement {
        use SourceRepresentation as S;
        let used = expr.inputs_used().len();
        let draws: Vec<S> =
            if used > 1 { vec![S::JointDraws] } else { vec![S::JointDraws, S::MarginalDraws] };
        let identity = matches!(expr, UtilityExpr::Input(_));
        match self {
            Self::Expectation | Self::ExpectedUtility => {
                if expr.is_affine() {
                    let mut any = vec![S::Mean, S::MeanAndCovariance];
                    any.extend(draws);
                    SourceRequirement { any_of: any, sampled_needs_error_receipt: true }
                } else {
                    SourceRequirement { any_of: draws, sampled_needs_error_receipt: true }
                }
            }
            Self::Variance => {
                if expr.is_affine() {
                    let mut any = vec![S::MeanAndCovariance];
                    any.extend(draws);
                    SourceRequirement { any_of: any, sampled_needs_error_receipt: true }
                } else {
                    SourceRequirement { any_of: draws, sampled_needs_error_receipt: true }
                }
            }
            Self::Probability { .. } if identity => {
                let mut any = vec![S::Cdf];
                any.extend(draws);
                SourceRequirement { any_of: any, sampled_needs_error_receipt: true }
            }
            Self::Quantile { .. } if identity => {
                let mut any = vec![S::QuantileFunction, S::Cdf];
                any.extend(draws);
                SourceRequirement { any_of: any, sampled_needs_error_receipt: true }
            }
            Self::Probability { .. } | Self::Quantile { .. } | Self::TailExpectation { .. } => {
                SourceRequirement { any_of: draws, sampled_needs_error_receipt: true }
            }
        }
    }

    fn valid(&self) -> bool {
        match self {
            Self::Probability { threshold, .. } => threshold.is_finite(),
            Self::Quantile { p } | Self::TailExpectation { p, .. } => *p > 0.0 && *p < 1.0,
            _ => true,
        }
    }
}

/// The initial, separately scoped decision criteria.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DecisionCriterion {
    /// Maximize posterior expected utility.
    PosteriorExpectedUtility,
    /// Minimize posterior expected loss.
    PosteriorExpectedLoss,
    /// Maximize the probability that utility reaches a threshold.
    ThresholdProbability {
        /// Utility threshold.
        threshold: f64,
    },
    /// Maximize a utility quantile.
    Quantile {
        /// Level in `(0, 1)`.
        p: f64,
    },
    /// Minimize the worst case over an identified set.
    MinimaxOverIdentifiedSet,
    /// Maximize the worst case over structures.
    MaximinOverStructures,
    /// Minimize the maximum regret against the best action per state.
    Regret,
    /// Minimize expected regret.
    ExpectedRegret,
}

impl DecisionCriterion {
    /// The functional the criterion scores each action with, when it is one.
    #[must_use]
    pub fn functional(&self) -> Option<DecisionFunctional> {
        match self {
            Self::PosteriorExpectedUtility => Some(DecisionFunctional::ExpectedUtility),
            Self::PosteriorExpectedLoss | Self::ExpectedRegret => {
                Some(DecisionFunctional::Expectation)
            }
            Self::ThresholdProbability { threshold } => {
                Some(DecisionFunctional::Probability { threshold: *threshold, tail: Tail::Upper })
            }
            Self::Quantile { p } => Some(DecisionFunctional::Quantile { p: *p }),
            Self::MinimaxOverIdentifiedSet | Self::MaximinOverStructures | Self::Regret => None,
        }
    }

    /// Whether the criterion compares actions state by state, so every action's
    /// inputs must come from one aligned joint source.
    #[must_use]
    pub const fn needs_state_alignment(&self) -> bool {
        matches!(self, Self::Regret | Self::ExpectedRegret)
    }

    fn valid(&self) -> bool {
        match self {
            Self::ThresholdProbability { threshold } => threshold.is_finite(),
            Self::Quantile { p } => *p > 0.0 && *p < 1.0,
            _ => true,
        }
    }

    fn encode(&self, out: &mut Vec<u8>) {
        match self {
            Self::PosteriorExpectedUtility => out.push(1),
            Self::PosteriorExpectedLoss => out.push(2),
            Self::ThresholdProbability { threshold } => {
                out.push(3);
                out.extend_from_slice(&threshold.to_bits().to_le_bytes());
            }
            Self::Quantile { p } => {
                out.push(4);
                out.extend_from_slice(&p.to_bits().to_le_bytes());
            }
            Self::MinimaxOverIdentifiedSet => out.push(5),
            Self::MaximinOverStructures => out.push(6),
            Self::Regret => out.push(7),
            Self::ExpectedRegret => out.push(8),
        }
    }
}

/// How a result built on structural uncertainty may be consumed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StructuralPolicy {
    /// Choose only an action best under every structure.
    RequireInvariantBestAction,
    /// Maximize the worst case over structures.
    Maximin,
    /// Weight structures by genuine structure probabilities; completion counts are
    /// not probabilities.
    BayesOverStructures,
    /// Report each structure's answer without choosing.
    ReportOnly,
}

impl StructuralPolicy {
    const fn tag(self) -> u8 {
        match self {
            Self::RequireInvariantBestAction => 1,
            Self::Maximin => 2,
            Self::BayesOverStructures => 3,
            Self::ReportOnly => 4,
        }
    }
}

/// One action in a decision contract.
#[derive(Clone, Debug, PartialEq)]
pub struct DecisionAction {
    /// Stable semantic identity; a label alone never defines the action.
    pub id: String,
    /// What kind of action this is.
    pub kind: ActionKind,
    /// Ordered quantities the utility reads, each under this action's regime.
    pub inputs: Vec<ScientificQuantity>,
    /// Utility over the inputs, in `utility_units`.
    pub utility: UtilityExpr,
}

/// A hard constraint: `P(expr <= bound) >= min_probability`. A violated
/// constraint excludes the action; it never enters the utility as a penalty.
#[derive(Clone, Debug, PartialEq)]
pub struct HardConstraint {
    /// Stable constraint identity.
    pub id: String,
    /// Expression over the action's inputs, in the constraint's own units.
    pub expr: UtilityExpr,
    /// Upper bound.
    pub bound: f64,
    /// Required probability in `(0, 1]`; `1.0` is a sure constraint.
    pub min_probability: f64,
    /// Units of `expr` and `bound`.
    pub units: String,
    /// Actions it applies to; empty applies to every action.
    pub applies_to: Vec<String>,
}

/// A durable decision problem.
#[derive(Clone, Debug, PartialEq)]
pub struct DecisionContract {
    /// Candidate actions, in any order.
    pub actions: Vec<DecisionAction>,
    /// Units of every action's utility; compared actions share them.
    pub utility_units: String,
    /// Scoring criterion.
    pub criterion: DecisionCriterion,
    /// Hard constraints.
    pub constraints: Vec<HardConstraint>,
    /// Population the decision concerns; every input must be in it.
    pub target_population: String,
    /// Horizon the decision concerns; every input must be at it.
    pub horizon: u32,
    /// Structural-uncertainty policy.
    pub structural_policy: StructuralPolicy,
}

/// Why a contract or source is refused.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DecisionContractError {
    /// A blank identity, units or population, or an out-of-bounds size.
    InvalidDeclaration(&'static str),
    /// Two actions or constraints share an identity.
    DuplicateId(String),
    /// A quantity coordinate is invalid.
    InvalidQuantity {
        /// Offending action.
        action: String,
        /// Input position.
        input: usize,
    },
    /// An input is outside the decision's target population or horizon.
    OutsideDecisionScope {
        /// Offending action.
        action: String,
        /// Input position.
        input: usize,
    },
    /// An expression reads an input that does not exist.
    UnknownInput(usize),
    /// A constraint names an action that is not declared.
    UnknownAction(String),
    /// A numeric parameter is not finite or out of range.
    InvalidParameter(&'static str),
    /// The declared sources cannot answer the functional.
    MissingSource {
        /// Representations that would have sufficed.
        needed: Vec<SourceRepresentation>,
    },
}

const MAX_ACTIONS: usize = 1_024;
const MAX_INPUTS: usize = 256;
const MAX_DEPTH: usize = 64;

impl DecisionContract {
    /// Check structure, units and scope.
    ///
    /// # Errors
    /// Refuses an under-specified contract before any evaluation.
    pub fn validate(&self) -> Result<(), DecisionContractError> {
        use DecisionContractError as E;
        if self.actions.len() < 2 {
            return Err(E::InvalidDeclaration("a decision compares at least two actions"));
        }
        if self.actions.len() > MAX_ACTIONS {
            return Err(E::InvalidDeclaration("too many actions"));
        }
        if blank(&self.utility_units) || blank(&self.target_population) {
            return Err(E::InvalidDeclaration("utility units and target population are required"));
        }
        if !self.criterion.valid() {
            return Err(E::InvalidParameter("criterion"));
        }
        if self.criterion.functional().is_some_and(|f| !f.valid()) {
            return Err(E::InvalidParameter("criterion functional"));
        }
        let mut ids = HashSet::new();
        for action in &self.actions {
            if blank(&action.id) {
                return Err(E::InvalidDeclaration("action id"));
            }
            if !ids.insert(action.id.as_str()) {
                return Err(E::DuplicateId(action.id.clone()));
            }
            self.validate_action(action)?;
        }
        let mut constraint_ids = HashSet::new();
        for constraint in &self.constraints {
            if blank(&constraint.id) || blank(&constraint.units) {
                return Err(E::InvalidDeclaration("constraint id and units"));
            }
            if !constraint_ids.insert(constraint.id.as_str()) {
                return Err(E::DuplicateId(constraint.id.clone()));
            }
            if !constraint.bound.is_finite()
                || !constraint.min_probability.is_finite()
                || constraint.min_probability <= 0.0
                || constraint.min_probability > 1.0
                || !constraint.expr.all_finite()
                || constraint.expr.depth() > MAX_DEPTH
            {
                return Err(E::InvalidParameter("constraint"));
            }
            if let Some(unknown) = constraint.applies_to.iter().find(|a| !ids.contains(a.as_str()))
            {
                return Err(E::UnknownAction(unknown.clone()));
            }
            let targets: Vec<&DecisionAction> = self
                .actions
                .iter()
                .filter(|a| {
                    constraint.applies_to.is_empty() || constraint.applies_to.contains(&a.id)
                })
                .collect();
            for action in targets {
                if constraint.expr.inputs_used().last().is_some_and(|k| *k >= action.inputs.len()) {
                    return Err(E::UnknownInput(action.inputs.len()));
                }
            }
        }
        Ok(())
    }

    fn validate_action(&self, action: &DecisionAction) -> Result<(), DecisionContractError> {
        use DecisionContractError as E;
        if action.inputs.is_empty() || action.inputs.len() > MAX_INPUTS {
            return Err(E::InvalidDeclaration("an action reads between one and 256 inputs"));
        }
        for (position, quantity) in action.inputs.iter().enumerate() {
            if quantity.validate().is_err() {
                return Err(E::InvalidQuantity { action: action.id.clone(), input: position });
            }
            if quantity.population_id != self.target_population || quantity.horizon != self.horizon
            {
                return Err(E::OutsideDecisionScope { action: action.id.clone(), input: position });
            }
        }
        if !action.utility.all_finite() || action.utility.depth() > MAX_DEPTH {
            return Err(E::InvalidParameter("utility expression"));
        }
        if let Some(k) =
            action.utility.inputs_used().into_iter().find(|k| *k >= action.inputs.len())
        {
            return Err(E::UnknownInput(k));
        }
        if action.utility.inputs_used().is_empty() {
            return Err(E::InvalidDeclaration("a utility reads at least one input"));
        }
        Ok(())
    }

    /// What every action's source must supply for the criterion's functional.
    ///
    /// The most demanding action decides: one nonlinear action makes the whole
    /// comparison need draws.
    ///
    /// # Errors
    /// An invalid contract refuses.
    pub fn source_requirement(&self) -> Result<Option<SourceRequirement>, DecisionContractError> {
        self.validate()?;
        let Some(functional) = self.criterion.functional() else {
            return Ok(None);
        };
        let mut combined: Option<SourceRequirement> = None;
        for action in &self.actions {
            let mut requirement = functional.requirement(&action.utility);
            if self.criterion.needs_state_alignment() {
                // State-by-state comparison needs one aligned source across actions.
                requirement.any_of.retain(|r| *r == SourceRepresentation::JointDraws);
            }
            combined = Some(match combined {
                None => requirement,
                Some(previous) => SourceRequirement {
                    any_of: previous
                        .any_of
                        .into_iter()
                        .filter(|r| requirement.any_of.contains(r))
                        .collect(),
                    sampled_needs_error_receipt: previous.sampled_needs_error_receipt
                        || requirement.sampled_needs_error_receipt,
                },
            });
        }
        Ok(combined)
    }

    /// Canonical BLAKE3 identity as lowercase hex.
    ///
    /// Reordering actions or constraints leaves it unchanged; any semantic
    /// change (an action's kind, quantities or utility, a constraint, the
    /// criterion, scope, units or structural policy) changes it.
    ///
    /// # Errors
    /// An invalid contract has no identity.
    pub fn identity(&self) -> Result<String, DecisionContractError> {
        self.validate()?;
        let mut actions: Vec<Vec<u8>> = self.actions.iter().map(encode_action).collect();
        actions.sort();
        let mut constraints: Vec<Vec<u8>> =
            self.constraints.iter().map(encode_constraint).collect();
        constraints.sort();
        let mut out = b"antecedent.decision_contract.v1".to_vec();
        put_str(&mut out, &self.utility_units);
        put_str(&mut out, &self.target_population);
        out.extend_from_slice(&self.horizon.to_le_bytes());
        self.criterion.encode(&mut out);
        out.push(self.structural_policy.tag());
        put_list(&mut out, &actions);
        put_list(&mut out, &constraints);
        Ok(blake3::hash(&out).to_hex().to_string())
    }
}

fn blank(value: &str) -> bool {
    value.trim().is_empty()
}

fn put_str(out: &mut Vec<u8>, value: &str) {
    out.extend_from_slice(&(value.len() as u64).to_le_bytes());
    out.extend_from_slice(value.as_bytes());
}

fn put_list(out: &mut Vec<u8>, items: &[Vec<u8>]) {
    out.extend_from_slice(&(items.len() as u64).to_le_bytes());
    for item in items {
        out.extend_from_slice(&(item.len() as u64).to_le_bytes());
        out.extend_from_slice(item);
    }
}

fn encode_quantity(out: &mut Vec<u8>, q: &ScientificQuantity) {
    for field in [
        &q.variable_id,
        &q.variable_name,
        &q.units,
        &q.population_id,
        &q.regime_id,
        &q.functional_id,
        &q.transform_id,
    ] {
        put_str(out, field);
    }
    out.push(match q.role {
        antecedent_core::QuantityRole::Treatment => 1,
        antecedent_core::QuantityRole::Outcome => 2,
        antecedent_core::QuantityRole::Covariate => 3,
        antecedent_core::QuantityRole::Mediator => 4,
        antecedent_core::QuantityRole::Selection => 5,
        antecedent_core::QuantityRole::Utility => 6,
    });
    out.extend_from_slice(&q.horizon.to_le_bytes());
    out.extend_from_slice(&(q.conditioning.len() as u64).to_le_bytes());
    for condition in &q.conditioning {
        put_str(out, &condition.variable_id);
        put_str(out, &condition.value_id);
    }
}

fn encode_action(action: &DecisionAction) -> Vec<u8> {
    let mut out = Vec::new();
    put_str(&mut out, &action.id);
    out.push(action.kind.tag());
    out.extend_from_slice(&(action.inputs.len() as u64).to_le_bytes());
    for quantity in &action.inputs {
        encode_quantity(&mut out, quantity);
    }
    action.utility.encode(&mut out);
    out
}

fn encode_constraint(constraint: &HardConstraint) -> Vec<u8> {
    let mut out = Vec::new();
    put_str(&mut out, &constraint.id);
    constraint.expr.encode(&mut out);
    out.extend_from_slice(&constraint.bound.to_bits().to_le_bytes());
    out.extend_from_slice(&constraint.min_probability.to_bits().to_le_bytes());
    put_str(&mut out, &constraint.units);
    let mut applies: Vec<&String> = constraint.applies_to.iter().collect();
    applies.sort();
    out.extend_from_slice(&(applies.len() as u64).to_le_bytes());
    for id in applies {
        put_str(&mut out, id);
    }
    out
}

/// The uncertainty representation a claim carries.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum UncertaintyKind {
    /// A single value or exact finite law with no structural spread.
    Point,
    /// A range over structures or an identified set; not a probability law.
    StructuralEnvelope,
    /// Genuine probabilities (a posterior over structures or parameters).
    Credible,
}

impl UncertaintyKind {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Point => "point",
            Self::StructuralEnvelope => "structural_envelope",
            Self::Credible => "credible",
        }
    }
}

/// The uncertainty representation a decision requires of the claims it reads.
///
/// A structural envelope is not a probability law and a point is not an
/// envelope, so a requirement is met only by its own kind.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub enum UncertaintyRequirement {
    /// Any representation is accepted.
    #[default]
    None,
    /// Only point claims are accepted.
    PointOnly,
    /// Only structural envelopes (identified sets, scenario ranges) are accepted.
    StructuralEnvelope,
    /// Only credible claims with genuine probabilities are accepted.
    Credible,
}

impl UncertaintyRequirement {
    /// Whether a claim of kind `supplied` satisfies the requirement.
    #[must_use]
    pub const fn satisfied_by(self, supplied: UncertaintyKind) -> bool {
        matches!(
            (self, supplied),
            (Self::None, _)
                | (Self::PointOnly, UncertaintyKind::Point)
                | (Self::StructuralEnvelope, UncertaintyKind::StructuralEnvelope)
                | (Self::Credible, UncertaintyKind::Credible)
        )
    }

    /// Stable `snake_case` name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::PointOnly => "point_only",
            Self::StructuralEnvelope => "structural_envelope",
            Self::Credible => "credible",
        }
    }

    const fn tag(self) -> u8 {
        match self {
            Self::None => 0,
            Self::PointOnly => 1,
            Self::StructuralEnvelope => 2,
            Self::Credible => 3,
        }
    }
}

/// The weakest empirical support one input quantity may have.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SupportRule {
    /// Action whose input the rule governs.
    pub action_id: String,
    /// Input position within the action.
    pub input: usize,
    /// Weakest status that still licenses the action; a weaker one excludes the
    /// action in the structure that reports it.
    pub weakest_allowed: SupportStatus,
}

/// An action removed by declaration (a legal, ethical or logistical rule), with
/// the reason.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeclaredExclusion {
    /// Excluded action.
    pub action_id: String,
    /// Why it is not admissible.
    pub reason: String,
}

/// Admissibility rules beyond hard constraints.
///
/// Every rule can only remove an action; none adds a penalty or a bonus to a
/// utility, so a hard constraint is never softened and a rule never trades
/// against value.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct AdmissibilityRules {
    /// Weakest allowed support for every input without its own rule; `None`
    /// leaves unruled inputs unchecked.
    pub default_weakest_support: Option<SupportStatus>,
    /// Per-input support rules.
    pub support_rules: Vec<SupportRule>,
    /// Actions removed by declaration.
    pub declared_exclusions: Vec<DeclaredExclusion>,
    /// Uncertainty representation the claims must carry.
    pub uncertainty: UncertaintyRequirement,
}

impl AdmissibilityRules {
    /// Weakest status allowed for `action_id`'s input, when any rule applies.
    #[must_use]
    pub fn weakest_allowed(&self, action_id: &str, input: usize) -> Option<SupportStatus> {
        self.support_rules
            .iter()
            .find(|rule| rule.action_id == action_id && rule.input == input)
            .map(|rule| rule.weakest_allowed)
            .or(self.default_weakest_support)
    }

    /// Whether `status` licenses the input.
    #[must_use]
    pub fn support_met(&self, action_id: &str, input: usize, status: SupportStatus) -> bool {
        self.weakest_allowed(action_id, input)
            .is_none_or(|weakest| status.severity() <= weakest.severity())
    }

    /// Declared reason for removing `action_id`, when it is removed.
    #[must_use]
    pub fn exclusion_reason(&self, action_id: &str) -> Option<&str> {
        self.declared_exclusions
            .iter()
            .find(|exclusion| exclusion.action_id == action_id)
            .map(|exclusion| exclusion.reason.as_str())
    }
}

/// Why admissibility rules or their contract are refused.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AdmissibilityError {
    /// The underlying contract is invalid.
    Contract(DecisionContractError),
    /// A rule names an action that is not declared.
    UnknownAction(String),
    /// A rule names an input the action does not have.
    UnknownInput {
        /// Action named by the rule.
        action: String,
        /// Input position.
        input: usize,
    },
    /// Two rules govern the same input or exclusion.
    DuplicateRule(String),
    /// A rule is malformed (for example a blank exclusion reason).
    InvalidRule(&'static str),
}

impl AdmissibilityError {
    /// Structured refusal with a registered code and a namespaced detail.
    #[must_use]
    pub fn to_refusal(&self) -> ExternalRefusal {
        let invalid = antecedent_core::reason_code!("invalid_argument");
        let build =
            |detail: &str, offending: Option<String>, expected: Option<&str>| ExternalRefusal {
                code: invalid,
                stage: "declare",
                detail: detail.to_owned(),
                offending,
                expected: expected.map(str::to_owned),
                supplied: None,
                capability: None,
                remedy: None,
            };
        match self {
            Self::Contract(error) => error.to_refusal(),
            Self::UnknownAction(id) => {
                build("decision_admissibility.unknown_action", Some(id.clone()), None)
            }
            Self::UnknownInput { action, input } => build(
                "decision_admissibility.unknown_input",
                Some(format!("{action}[{input}]")),
                None,
            ),
            Self::DuplicateRule(key) => {
                build("decision_admissibility.duplicate_rule", Some(key.clone()), None)
            }
            Self::InvalidRule(why) => {
                build("decision_admissibility.invalid_rule", None, Some(*why))
            }
        }
    }
}

/// A [`DecisionContract`] with admissibility rules beyond hard constraints.
///
/// The base contract keeps its own identity and struct shape; this wrapper
/// adds the rules and an identity that covers both. Reordering actions,
/// constraints, support rules or exclusions leaves the identity unchanged; any
/// semantic edit to the contract or the rules changes it.
#[derive(Clone, Debug, PartialEq)]
pub struct AdmissibleDecisionContract {
    /// The decision problem, with its hard constraints.
    pub contract: DecisionContract,
    /// Rules that may further remove actions or refuse claims.
    pub rules: AdmissibilityRules,
}

impl AdmissibleDecisionContract {
    /// Check the contract and that every rule names a declared action and input.
    ///
    /// # Errors
    /// Refuses an invalid contract or a rule that names nothing declared.
    pub fn validate(&self) -> Result<(), AdmissibilityError> {
        self.contract.validate().map_err(AdmissibilityError::Contract)?;
        let find = |id: &str| self.contract.actions.iter().find(|a| a.id == id);
        let mut keys = HashSet::new();
        for rule in &self.rules.support_rules {
            let action = find(&rule.action_id)
                .ok_or_else(|| AdmissibilityError::UnknownAction(rule.action_id.clone()))?;
            if rule.input >= action.inputs.len() {
                return Err(AdmissibilityError::UnknownInput {
                    action: rule.action_id.clone(),
                    input: rule.input,
                });
            }
            if !keys.insert((rule.action_id.as_str(), rule.input)) {
                return Err(AdmissibilityError::DuplicateRule(format!(
                    "{}[{}]",
                    rule.action_id, rule.input
                )));
            }
        }
        let mut excluded = HashSet::new();
        for exclusion in &self.rules.declared_exclusions {
            if find(&exclusion.action_id).is_none() {
                return Err(AdmissibilityError::UnknownAction(exclusion.action_id.clone()));
            }
            if blank(&exclusion.reason) {
                return Err(AdmissibilityError::InvalidRule("an exclusion states its reason"));
            }
            if !excluded.insert(exclusion.action_id.as_str()) {
                return Err(AdmissibilityError::DuplicateRule(exclusion.action_id.clone()));
            }
        }
        Ok(())
    }

    /// Canonical BLAKE3 identity as lowercase hex, covering the base contract
    /// identity and every rule.
    ///
    /// # Errors
    /// An invalid contract or rule set has no identity.
    pub fn identity(&self) -> Result<String, AdmissibilityError> {
        self.validate()?;
        let base = self.contract.identity().map_err(AdmissibilityError::Contract)?;
        let mut support: Vec<Vec<u8>> =
            self.rules.support_rules.iter().map(encode_support_rule).collect();
        support.sort();
        let mut exclusions: Vec<Vec<u8>> = self
            .rules
            .declared_exclusions
            .iter()
            .map(|exclusion| {
                let mut out = Vec::new();
                put_str(&mut out, &exclusion.action_id);
                put_str(&mut out, &exclusion.reason);
                out
            })
            .collect();
        exclusions.sort();
        let mut out = b"antecedent.decision_contract.admissibility.v1".to_vec();
        put_str(&mut out, &base);
        match self.rules.default_weakest_support {
            None => out.push(0),
            Some(status) => {
                out.push(1);
                out.push(status.severity());
            }
        }
        put_list(&mut out, &support);
        put_list(&mut out, &exclusions);
        out.push(self.rules.uncertainty.tag());
        Ok(blake3::hash(&out).to_hex().to_string())
    }
}

fn encode_support_rule(rule: &SupportRule) -> Vec<u8> {
    let mut out = Vec::new();
    put_str(&mut out, &rule.action_id);
    out.extend_from_slice(&(rule.input as u64).to_le_bytes());
    out.push(rule.weakest_allowed.severity());
    out
}

#[cfg(test)]
mod tests {
    use antecedent_core::QuantityRole;

    use super::*;

    fn input(variable: &str, regime: &str) -> ScientificQuantity {
        ScientificQuantity {
            variable_id: variable.into(),
            variable_name: variable.into(),
            role: QuantityRole::Outcome,
            units: "units".into(),
            population_id: "target".into(),
            regime_id: regime.into(),
            horizon: 0,
            functional_id: "outcome".into(),
            conditioning: vec![],
            transform_id: "identity".into(),
        }
    }

    fn action(id: &str, regime: &str, utility: UtilityExpr) -> DecisionAction {
        DecisionAction {
            id: id.into(),
            kind: ActionKind::Intervention,
            inputs: vec![input("benefit", regime), input("cost", regime)],
            utility,
        }
    }

    fn contract() -> DecisionContract {
        // Nonlinear: benefit * cost interaction requires aligned joint draws.
        let nonlinear = UtilityExpr::product(UtilityExpr::Input(0), UtilityExpr::Input(1));
        let linear = UtilityExpr::difference(UtilityExpr::Input(0), UtilityExpr::Input(1));
        DecisionContract {
            actions: vec![action("treat", "do(a=1)", nonlinear), action("wait", "do(a=0)", linear)],
            utility_units: "qaly".into(),
            criterion: DecisionCriterion::PosteriorExpectedUtility,
            constraints: vec![HardConstraint {
                id: "cost-cap".into(),
                expr: UtilityExpr::Input(1),
                bound: 10.0,
                min_probability: 1.0,
                units: "usd".into(),
                applies_to: vec![],
            }],
            target_population: "target".into(),
            horizon: 0,
            structural_policy: StructuralPolicy::ReportOnly,
        }
    }

    #[test]
    fn a_mean_satisfies_only_an_affine_expectation() {
        use SourceRepresentation as S;
        let linear = UtilityExpr::difference(UtilityExpr::Input(0), UtilityExpr::Input(1));
        let requirement = DecisionFunctional::Expectation.requirement(&linear);
        assert_eq!(requirement.check(&[S::Mean]).unwrap().representation, S::Mean);
        let nonlinear = UtilityExpr::product(UtilityExpr::Input(0), UtilityExpr::Input(1));
        let requirement = DecisionFunctional::ExpectedUtility.requirement(&nonlinear);
        assert_eq!(
            requirement.check(&[S::Mean, S::MeanAndCovariance, S::MarginalDraws]),
            Err(DecisionContractError::MissingSource { needed: vec![S::JointDraws] })
        );
        let mode = requirement.check(&[S::JointDraws]).unwrap();
        assert!(mode.needs_error_receipt);
        // One nonlinear input needs its law, not its pairing with another input.
        let clipped = UtilityExpr::maximum(UtilityExpr::Input(0), UtilityExpr::Const(0.0));
        let requirement = DecisionFunctional::Expectation.requirement(&clipped);
        assert!(requirement.check(&[S::MarginalDraws]).is_ok());
        assert!(requirement.check(&[S::Mean]).is_err());
    }

    #[test]
    fn quantiles_and_probabilities_need_a_distribution_not_a_mean() {
        use SourceRepresentation as S;
        let identity = UtilityExpr::Input(0);
        let quantile = DecisionFunctional::Quantile { p: 0.1 }.requirement(&identity);
        assert!(quantile.check(&[S::Mean, S::MeanAndCovariance]).is_err());
        assert_eq!(
            quantile.check(&[S::QuantileFunction]).unwrap().representation,
            S::QuantileFunction
        );
        assert!(!quantile.check(&[S::QuantileFunction]).unwrap().needs_error_receipt);
        let probability = DecisionFunctional::Probability { threshold: 1.0, tail: Tail::Upper }
            .requirement(&identity);
        assert_eq!(probability.check(&[S::Cdf]).unwrap().representation, S::Cdf);
        // A function of two inputs has no per-input CDF route.
        let sum = UtilityExpr::sum(UtilityExpr::Input(0), UtilityExpr::Input(1));
        let requirement =
            DecisionFunctional::Probability { threshold: 0.0, tail: Tail::Lower }.requirement(&sum);
        assert!(requirement.check(&[S::Cdf]).is_err());
        assert!(requirement.check(&[S::JointDraws]).is_ok());
    }

    #[test]
    fn contract_needs_joint_draws_when_any_action_is_nonlinear() {
        let c = contract();
        let requirement = c.source_requirement().unwrap().unwrap();
        assert_eq!(requirement.any_of, vec![SourceRepresentation::JointDraws]);
        let mut regret = c.clone();
        regret.criterion = DecisionCriterion::ExpectedRegret;
        assert_eq!(
            regret.source_requirement().unwrap().unwrap().any_of,
            vec![SourceRepresentation::JointDraws]
        );
        let mut minimax = c;
        minimax.criterion = DecisionCriterion::MaximinOverStructures;
        assert_eq!(minimax.source_requirement().unwrap(), None);
    }

    #[test]
    fn identity_ignores_declaration_order_but_not_semantics() {
        let base = contract();
        let id = base.identity().unwrap();
        let mut reordered = base.clone();
        reordered.actions.reverse();
        assert_eq!(reordered.identity().unwrap(), id);

        let edits: Vec<Box<dyn Fn(&mut DecisionContract)>> = vec![
            Box::new(|c| c.actions[0].id = "treat-2".into()),
            Box::new(|c| c.actions[0].kind = ActionKind::Policy),
            Box::new(|c| c.actions[0].inputs[0].regime_id = "do(a=2)".into()),
            Box::new(|c| c.actions[1].utility = UtilityExpr::Input(0)),
            Box::new(|c| c.constraints[0].bound = 11.0),
            Box::new(|c| c.constraints[0].min_probability = 0.9),
            Box::new(|c| c.criterion = DecisionCriterion::PosteriorExpectedLoss),
            Box::new(|c| c.structural_policy = StructuralPolicy::Maximin),
            Box::new(|c| c.utility_units = "usd".into()),
            Box::new(|c| c.constraints.clear()),
        ];
        for edit in edits {
            let mut changed = base.clone();
            edit(&mut changed);
            assert_ne!(changed.identity().unwrap(), id);
        }
    }

    #[test]
    fn under_specified_contracts_refuse() {
        use DecisionContractError as E;
        let mut one = contract();
        one.actions.pop();
        assert!(matches!(one.validate(), Err(E::InvalidDeclaration(_))));
        let mut dup = contract();
        dup.actions[1].id = "treat".into();
        assert_eq!(dup.validate(), Err(E::DuplicateId("treat".into())));
        let mut scope = contract();
        scope.actions[0].inputs[1].population_id = "other".into();
        assert_eq!(
            scope.validate(),
            Err(E::OutsideDecisionScope { action: "treat".into(), input: 1 })
        );
        let mut horizon = contract();
        horizon.horizon = 3;
        assert!(matches!(horizon.validate(), Err(E::OutsideDecisionScope { .. })));
        let mut index = contract();
        index.actions[0].utility = UtilityExpr::Input(5);
        assert_eq!(index.validate(), Err(E::UnknownInput(5)));
        let mut constraint = contract();
        constraint.constraints[0].applies_to = vec!["ghost".into()];
        assert_eq!(constraint.validate(), Err(E::UnknownAction("ghost".into())));
        let mut prob = contract();
        prob.constraints[0].min_probability = 0.0;
        assert_eq!(prob.validate(), Err(E::InvalidParameter("constraint")));
        let mut nan = contract();
        nan.actions[0].utility = UtilityExpr::Const(f64::NAN);
        assert!(nan.validate().is_err());
        let mut p = contract();
        p.criterion = DecisionCriterion::Quantile { p: 1.0 };
        assert_eq!(p.validate(), Err(E::InvalidParameter("criterion")));
        let mut units = contract();
        units.utility_units = " ".into();
        assert!(matches!(units.validate(), Err(E::InvalidDeclaration(_))));
        assert!(
            UtilityExpr::product(UtilityExpr::Input(0), UtilityExpr::Input(1))
                .evaluate(&[2.0])
                .is_err()
        );
    }

    #[test]
    fn utility_evaluates_exactly() {
        let expr = UtilityExpr::maximum(
            UtilityExpr::difference(
                UtilityExpr::product(UtilityExpr::Input(0), UtilityExpr::Input(1)),
                UtilityExpr::Const(1.0),
            ),
            UtilityExpr::Const(0.0),
        );
        assert!((expr.evaluate(&[3.0, 2.0]).unwrap() - 5.0).abs() < f64::EPSILON);
        assert!(expr.evaluate(&[0.1, 0.1]).unwrap().abs() < f64::EPSILON);
        assert!(!expr.is_affine());
        assert!(UtilityExpr::product(UtilityExpr::Const(2.0), UtilityExpr::Input(0)).is_affine());
        assert!(!UtilityExpr::product(UtilityExpr::Input(0), UtilityExpr::Input(1)).is_affine());
    }
}
