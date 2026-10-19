//! Durable decision contract and the source each functional needs.
//!
//! A [`DecisionContract`] declares actions by stable semantic ID, each with the
//! scientific quantities its utility reads and a closed, replayable utility
//! expression. Hard constraints exclude actions; they never become penalties.
//! A [`DecisionFunctional`] says which source representation it needs: a mean
//! can satisfy an expectation of a linear utility, covariance a linear variance,
//! a quantile or threshold a law, and a nonlinear utility over several inputs
//! aligned joint draws. [`DecisionContract::source_requirement`] compiles the
//! minimal request for the declared actions, including regret, statewise
//! optimality, robust sets, and hard constraints. A [`ProgramSearchContract`]
//! searches a space; scoring a prespecified program does not become that search.
//! The contract's [`DecisionContract::identity`] is independent of the order in
//! which actions and constraints were declared and changes with any semantic edit.

use std::collections::HashSet;

use antecedent_contract::{ExternalRefusal, ScientificQuantity};

use crate::signal::RefusalBuilder;
use antecedent_core::{CanonicalWriter, SupportStatus, is_blank};

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

    fn encode(&self, out: &mut CanonicalWriter) {
        match self {
            Self::Const(value) => {
                out.word(1).real(*value);
            }
            Self::Input(k) => {
                out.word(2).word(*k as u64);
            }
            Self::Neg(a) => {
                out.word(3);
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

fn binary(out: &mut CanonicalWriter, tag: u64, a: &UtilityExpr, b: &UtilityExpr) {
    out.word(tag);
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

impl SourceRepresentation {
    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::Mean => "mean",
            Self::MeanAndCovariance => "mean_and_covariance",
            Self::Cdf => "cdf",
            Self::QuantileFunction => "quantile_function",
            Self::MarginalDraws => "marginal_draws",
            Self::JointDraws => "joint_draws",
        }
    }
}

/// The ways a source can satisfy a functional.
///
/// An empty [`Self::any_of`] is the identified set or the set of structures a
/// robust criterion reads. It is not an unspecified probabilistic request:
/// [`Self::is_identified_set`] is true, and no mean, covariance, or law satisfies it.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SourceRequirement {
    /// Any one of these representations suffices. Empty for an identified set.
    pub any_of: Vec<SourceRepresentation>,
    /// Whether a sampled representation must also report a Monte Carlo error
    /// receipt, because the functional is then estimated rather than exact.
    pub sampled_needs_error_receipt: bool,
}

impl SourceRequirement {
    /// The identified set or structure set. No probabilistic representation suffices.
    #[must_use]
    pub const fn identified_set() -> Self {
        Self { any_of: Vec::new(), sampled_needs_error_receipt: false }
    }

    /// Whether this requirement is an identified set or a set of structures.
    #[must_use]
    pub fn is_identified_set(&self) -> bool {
        self.any_of.is_empty()
    }

    /// Check `available` against the requirement.
    ///
    /// An identified-set requirement matches none of `available`, so a mean is refused.
    ///
    /// # Errors
    /// Reports the representations that would have sufficed. The list is empty
    /// when the requirement is an identified set.
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

    fn encode(&self, out: &mut CanonicalWriter) {
        match self {
            Self::PosteriorExpectedUtility => {
                out.word(1);
            }
            Self::PosteriorExpectedLoss => {
                out.word(2);
            }
            Self::ThresholdProbability { threshold } => {
                out.word(3).real(*threshold);
            }
            Self::Quantile { p } => {
                out.word(4).real(*p);
            }
            Self::MinimaxOverIdentifiedSet => {
                out.word(5);
            }
            Self::MaximinOverStructures => {
                out.word(6);
            }
            Self::Regret => {
                out.word(7);
            }
            Self::ExpectedRegret => {
                out.word(8);
            }
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
        if is_blank(&self.utility_units) || is_blank(&self.target_population) {
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
            if is_blank(&action.id) {
                return Err(E::InvalidDeclaration("action id"));
            }
            if !ids.insert(action.id.as_str()) {
                return Err(E::DuplicateId(action.id.clone()));
            }
            self.validate_action(action)?;
        }
        let mut constraint_ids = HashSet::new();
        for constraint in &self.constraints {
            if is_blank(&constraint.id) || is_blank(&constraint.units) {
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

    /// The minimal source that can score the declared actions.
    ///
    /// Affine expected utility accepts a mean. A threshold or quantile asks for
    /// the law that determines it, and a hard constraint is a threshold, so it
    /// refuses a means-only source. Nonlinear coupling, regret, and statewise
    /// optimality (the best action in each state) ask for one joint law. Minimax
    /// over an identified set and maximin over structures ask for that set.
    /// Linear variance is [`DecisionFunctional::Variance`], not a declared-action
    /// criterion; a search states it as [`ProgramObjective::Variance`]. The most
    /// demanding action decides.
    ///
    /// Compiling the request does not change [`Self::identity`]. Scoring these
    /// declared actions does not build a [`ProgramSearchContract`].
    ///
    /// # Errors
    /// An invalid contract refuses.
    pub fn source_requirement(&self) -> Result<Option<SourceRequirement>, DecisionContractError> {
        self.validate()?;
        let mut requirement = if let Some(functional) = self.criterion.functional() {
            let mut combined: Option<SourceRequirement> = None;
            for action in &self.actions {
                let requirement = functional.requirement(&action.utility);
                combined = Some(match combined {
                    None => requirement,
                    Some(previous) => intersect(previous, &requirement),
                });
            }
            combined.ok_or(DecisionContractError::InvalidDeclaration(
                "a decision compares at least two actions",
            ))?
        } else if self.criterion.needs_state_alignment() {
            // Regret compares each action with the best action in that state.
            joint_draws()
        } else {
            SourceRequirement::identified_set()
        };
        requirement = tighten(requirement, &self.constraints);
        if self.criterion.needs_state_alignment() {
            // State-by-state comparison needs one aligned source across actions.
            requirement.any_of.retain(|r| *r == SourceRepresentation::JointDraws);
        }
        Ok(Some(requirement))
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
        let mut actions: Vec<[u8; 32]> = self.actions.iter().map(encode_action).collect();
        actions.sort_unstable();
        let mut constraints: Vec<[u8; 32]> =
            self.constraints.iter().map(encode_constraint).collect();
        constraints.sort_unstable();
        let mut out = CanonicalWriter::new("antecedent.decision_contract.v1");
        out.text(&self.utility_units).text(&self.target_population).word(u64::from(self.horizon));
        self.criterion.encode(&mut out);
        out.word(u64::from(self.structural_policy.tag()));
        put_list(&mut out, &actions);
        put_list(&mut out, &constraints);
        Ok(out.finish_hex())
    }
}

/// A count-prefixed set of item digests (already sorted by the caller so order is canonical).
fn put_list(out: &mut CanonicalWriter, items: &[[u8; 32]]) {
    out.word(items.len() as u64);
    for item in items {
        out.bytes(item);
    }
}

fn encode_action(action: &DecisionAction) -> [u8; 32] {
    let mut out = CanonicalWriter::new("antecedent.decision_contract.action.v1");
    out.text(&action.id).word(u64::from(action.kind.tag())).word(action.inputs.len() as u64);
    for quantity in &action.inputs {
        quantity.write_canonical(&mut out);
    }
    action.utility.encode(&mut out);
    out.finish()
}

fn encode_constraint(constraint: &HardConstraint) -> [u8; 32] {
    let mut out = CanonicalWriter::new("antecedent.decision_contract.constraint.v1");
    out.text(&constraint.id);
    constraint.expr.encode(&mut out);
    out.real(constraint.bound).real(constraint.min_probability).text(&constraint.units);
    let mut applies: Vec<&String> = constraint.applies_to.iter().collect();
    applies.sort();
    out.texts(&applies);
    out.finish()
}

fn joint_draws() -> SourceRequirement {
    SourceRequirement {
        any_of: vec![SourceRepresentation::JointDraws],
        sampled_needs_error_receipt: true,
    }
}

fn intersect(previous: SourceRequirement, requirement: &SourceRequirement) -> SourceRequirement {
    SourceRequirement {
        any_of: previous.any_of.into_iter().filter(|r| requirement.any_of.contains(r)).collect(),
        sampled_needs_error_receipt: previous.sampled_needs_error_receipt
            || requirement.sampled_needs_error_receipt,
    }
}

/// A law that answers a threshold also answers an affine expectation. Any other
/// criterion keeps only the representations that were already sufficient for it.
fn tighten(requirement: SourceRequirement, constraints: &[HardConstraint]) -> SourceRequirement {
    constraints.iter().fold(requirement, |requirement, constraint| {
        if requirement.is_identified_set() {
            return requirement;
        }
        let law =
            DecisionFunctional::Probability { threshold: constraint.bound, tail: Tail::Lower }
                .requirement(&constraint.expr);
        let any_of = if requirement.any_of.contains(&SourceRepresentation::Mean) {
            law.any_of
        } else {
            law.any_of.into_iter().filter(|r| requirement.any_of.contains(r)).collect()
        };
        SourceRequirement {
            any_of,
            sampled_needs_error_receipt: requirement.sampled_needs_error_receipt
                || law.sampled_needs_error_receipt,
        }
    })
}

/// Controllable variables a search ranges over.
///
/// A list of declared actions is not a space. Those actions are scored by a
/// [`DecisionContract`] and are not promoted into a search.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProgramSearchSpace {
    /// Stable identity of the space.
    pub id: String,
    /// Controllable variables, in the order an objective's inputs read them.
    pub variables: Vec<String>,
}

/// What a search optimizes. This is not a declared-action criterion.
#[derive(Clone, Debug, PartialEq)]
pub enum ProgramObjective {
    /// Expected utility of `expr`. Affine expressions accept a mean.
    ExpectedUtility(UtilityExpr),
    /// Variance of `expr`. Affine expressions accept a covariance.
    Variance(UtilityExpr),
    /// Probability that `expr` falls on `tail` of `threshold`.
    Threshold {
        /// Expression over the space's variables.
        expr: UtilityExpr,
        /// Threshold.
        threshold: f64,
        /// Direction.
        tail: Tail,
    },
    /// The `p`th quantile of `expr`.
    Quantile {
        /// Expression over the space's variables.
        expr: UtilityExpr,
        /// Level in `(0, 1)`.
        p: f64,
    },
    /// An optimum over the space. A joint law is required; a mean does not
    /// certify optimality of a searched program.
    Optimality(UtilityExpr),
    /// Regret of the chosen program against the best feasible program. A joint law.
    Regret(UtilityExpr),
    /// A robust optimum over an identified set or a set of structures.
    Robust,
}

impl ProgramObjective {
    fn expr(&self) -> Option<&UtilityExpr> {
        match self {
            Self::ExpectedUtility(expr)
            | Self::Variance(expr)
            | Self::Optimality(expr)
            | Self::Regret(expr)
            | Self::Threshold { expr, .. }
            | Self::Quantile { expr, .. } => Some(expr),
            Self::Robust => None,
        }
    }

    fn parameters_valid(&self) -> bool {
        match self {
            Self::Threshold { threshold, .. } => threshold.is_finite(),
            Self::Quantile { p, .. } => *p > 0.0 && *p < 1.0,
            _ => true,
        }
    }

    fn requirement(&self) -> SourceRequirement {
        match self {
            Self::ExpectedUtility(expr) => DecisionFunctional::ExpectedUtility.requirement(expr),
            Self::Variance(expr) => DecisionFunctional::Variance.requirement(expr),
            Self::Threshold { expr, threshold, tail } => {
                DecisionFunctional::Probability { threshold: *threshold, tail: *tail }
                    .requirement(expr)
            }
            Self::Quantile { expr, p } => DecisionFunctional::Quantile { p: *p }.requirement(expr),
            // Optimality and regret couple the searched program across states.
            Self::Optimality(_) | Self::Regret(_) => joint_draws(),
            Self::Robust => SourceRequirement::identified_set(),
        }
    }
}

/// A search over programs: a space, an objective, and hard constraints.
///
/// Distinct from [`DecisionContract`], which scores a prespecified list of
/// actions. Evaluating those actions does not construct this contract and does
/// not search.
#[derive(Clone, Debug, PartialEq)]
pub struct ProgramSearchContract {
    /// Variables the search ranges over.
    pub space: ProgramSearchSpace,
    /// Objective.
    pub objective: ProgramObjective,
    /// Hard constraints on every candidate. `applies_to` stays empty: a search
    /// constraint is not attached to a declared action. Constraints exclude
    /// candidates and never become penalties.
    pub constraints: Vec<HardConstraint>,
}

impl ProgramSearchContract {
    /// Check the space, the objective, and the constraints.
    ///
    /// # Errors
    /// Refuses a blank identity, a variable an expression does not have, or a
    /// parameter out of range.
    pub fn validate(&self) -> Result<(), DecisionContractError> {
        use DecisionContractError as E;
        if is_blank(&self.space.id) {
            return Err(E::InvalidDeclaration("search space id"));
        }
        if self.space.variables.is_empty() || self.space.variables.len() > MAX_INPUTS {
            return Err(E::InvalidDeclaration("a search names between one and 256 variables"));
        }
        let mut names = HashSet::new();
        for variable in &self.space.variables {
            if is_blank(variable) {
                return Err(E::InvalidDeclaration("search variable"));
            }
            if !names.insert(variable.as_str()) {
                return Err(E::DuplicateId(variable.clone()));
            }
        }
        if !self.objective.parameters_valid() {
            return Err(E::InvalidParameter("search objective"));
        }
        if let Some(expr) = self.objective.expr() {
            check_search_expr(expr, self.space.variables.len())?;
            if expr.inputs_used().is_empty() {
                return Err(E::InvalidDeclaration("an objective reads at least one variable"));
            }
        }
        let mut constraint_ids = HashSet::new();
        for constraint in &self.constraints {
            if is_blank(&constraint.id) || is_blank(&constraint.units) {
                return Err(E::InvalidDeclaration("constraint id and units"));
            }
            if !constraint_ids.insert(constraint.id.as_str()) {
                return Err(E::DuplicateId(constraint.id.clone()));
            }
            if !constraint.applies_to.is_empty() {
                return Err(E::InvalidDeclaration(
                    "a search constraint applies to every candidate",
                ));
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
            check_search_expr(&constraint.expr, self.space.variables.len())?;
        }
        Ok(())
    }

    /// The minimal source for the objective and the constraints.
    ///
    /// The same matrix as [`DecisionContract::source_requirement`]: means for an
    /// affine expected utility, covariance for a linear variance, the appropriate
    /// law for a threshold or quantile, a joint law for nonlinear coupling, regret,
    /// or optimality, and a set for a robust objective. A hard constraint refuses
    /// a means-only source.
    ///
    /// # Errors
    /// An invalid search contract refuses.
    pub fn source_requirement(&self) -> Result<SourceRequirement, DecisionContractError> {
        self.validate()?;
        Ok(tighten(self.objective.requirement(), &self.constraints))
    }
}

fn check_search_expr(expr: &UtilityExpr, variables: usize) -> Result<(), DecisionContractError> {
    use DecisionContractError as E;
    if !expr.all_finite() || expr.depth() > MAX_DEPTH {
        return Err(E::InvalidParameter("utility expression"));
    }
    if let Some(k) = expr.inputs_used().into_iter().find(|k| *k >= variables) {
        return Err(E::UnknownInput(k));
    }
    Ok(())
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
        let build = |detail: &str| RefusalBuilder::new(invalid, "declare", detail);
        match self {
            Self::Contract(error) => error.to_refusal(),
            Self::UnknownAction(id) => {
                build("decision_admissibility.unknown_action").offending(id.clone()).build()
            }
            Self::UnknownInput { action, input } => build("decision_admissibility.unknown_input")
                .offending(format!("{action}[{input}]"))
                .build(),
            Self::DuplicateRule(key) => {
                build("decision_admissibility.duplicate_rule").offending(key.clone()).build()
            }
            Self::InvalidRule(why) => {
                build("decision_admissibility.invalid_rule").expected(*why).build()
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
            if is_blank(&exclusion.reason) {
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
        let mut support: Vec<[u8; 32]> =
            self.rules.support_rules.iter().map(encode_support_rule).collect();
        support.sort_unstable();
        let mut exclusions: Vec<[u8; 32]> = self
            .rules
            .declared_exclusions
            .iter()
            .map(|exclusion| {
                let mut item =
                    CanonicalWriter::new("antecedent.decision_contract.admissibility.exclusion.v1");
                item.text(&exclusion.action_id).text(&exclusion.reason);
                item.finish()
            })
            .collect();
        exclusions.sort_unstable();
        let mut out = CanonicalWriter::new("antecedent.decision_contract.admissibility.v1");
        out.text(&base);
        out.opt_word(self.rules.default_weakest_support.map(|s| u64::from(s.severity())));
        put_list(&mut out, &support);
        put_list(&mut out, &exclusions);
        out.word(u64::from(self.rules.uncertainty.tag()));
        Ok(out.finish_hex())
    }
}

fn encode_support_rule(rule: &SupportRule) -> [u8; 32] {
    let mut out = CanonicalWriter::new("antecedent.decision_contract.admissibility.support.v1");
    out.text(&rule.action_id)
        .word(rule.input as u64)
        .word(u64::from(rule.weakest_allowed.severity()));
    out.finish()
}

#[cfg(test)]
mod tests {
    use antecedent_contract::QuantityRole;

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
        let mut statewise = c.clone();
        statewise.criterion = DecisionCriterion::Regret;
        let statewise = statewise.source_requirement().unwrap().unwrap();
        assert_eq!(statewise.any_of, vec![SourceRepresentation::JointDraws]);
        assert!(statewise.check(&[SourceRepresentation::Mean]).is_err());
        let mut minimax = c;
        minimax.criterion = DecisionCriterion::MaximinOverStructures;
        let set = minimax.source_requirement().unwrap().unwrap();
        assert!(set.is_identified_set());
        assert!(set.check(&[SourceRepresentation::Mean]).is_err());
    }

    fn one_input(id: &str, regime: &str, utility: UtilityExpr) -> DecisionAction {
        DecisionAction {
            id: id.into(),
            kind: ActionKind::Intervention,
            inputs: vec![input("benefit", regime)],
            utility,
        }
    }

    fn declared(criterion: DecisionCriterion, utility: UtilityExpr) -> DecisionContract {
        DecisionContract {
            actions: vec![
                one_input("treat", "do(a=1)", utility.clone()),
                one_input("wait", "do(a=0)", utility),
            ],
            utility_units: "qaly".into(),
            criterion,
            constraints: vec![],
            target_population: "target".into(),
            horizon: 0,
            structural_policy: StructuralPolicy::ReportOnly,
        }
    }

    fn cap() -> HardConstraint {
        HardConstraint {
            id: "cap".into(),
            expr: UtilityExpr::Input(0),
            bound: 1.0,
            min_probability: 0.9,
            units: "qaly".into(),
            applies_to: vec![],
        }
    }

    fn search(objective: ProgramObjective) -> ProgramSearchContract {
        ProgramSearchContract {
            space: ProgramSearchSpace { id: "dose".into(), variables: vec!["dose".into()] },
            objective,
            constraints: vec![],
        }
    }

    #[test]
    fn source_requirement_compiles_the_minimal_sufficient_request() {
        use DecisionCriterion as C;
        use SourceRepresentation as S;

        // Affine expected utility: a mean is sufficient, and compiling the request
        // leaves the contract identity unchanged.
        let affine = declared(C::PosteriorExpectedUtility, UtilityExpr::Input(0));
        let identity = affine.identity().unwrap();
        let means = affine.source_requirement().unwrap().unwrap();
        assert_eq!(
            means.any_of,
            vec![S::Mean, S::MeanAndCovariance, S::JointDraws, S::MarginalDraws]
        );
        assert_eq!(means.check(&[S::Mean]).unwrap().representation, S::Mean);
        assert_eq!(affine.identity().unwrap(), identity);
        // Two affine inputs still accept a mean; the draw alternative is joint.
        let gap = UtilityExpr::difference(UtilityExpr::Input(0), UtilityExpr::Input(1));
        let mut wide = contract();
        wide.actions[0].utility = gap.clone();
        wide.actions[1].utility = gap;
        wide.constraints.clear();
        wide.criterion = C::PosteriorExpectedUtility;
        assert_eq!(
            wide.source_requirement().unwrap().unwrap().any_of,
            vec![S::Mean, S::MeanAndCovariance, S::JointDraws]
        );
        // A threshold of that same contrast has no marginal law.
        wide.criterion = C::ThresholdProbability { threshold: 0.0 };
        let contrast = wide.source_requirement().unwrap().unwrap();
        assert_eq!(contrast.any_of, vec![S::JointDraws]);
        assert!(contrast.check(&[S::Mean]).is_err());
        assert!(contrast.check(&[S::Cdf]).is_err());

        // Linear variance accepts covariance and refuses a mean. A nonlinear
        // variance has no covariance route.
        let scaled = UtilityExpr::product(UtilityExpr::Const(2.0), UtilityExpr::Input(0));
        let variance = DecisionFunctional::Variance.requirement(&scaled);
        assert!(scaled.is_affine());
        assert_eq!(variance.any_of, vec![S::MeanAndCovariance, S::JointDraws, S::MarginalDraws]);
        assert!(variance.check(&[S::Mean]).is_err());
        assert_eq!(
            variance.check(&[S::MeanAndCovariance]).unwrap().representation,
            S::MeanAndCovariance
        );
        let curved = UtilityExpr::maximum(UtilityExpr::Input(0), UtilityExpr::Const(0.0));
        let curved_variance = DecisionFunctional::Variance.requirement(&curved);
        assert!(curved_variance.check(&[S::MeanAndCovariance]).is_err());

        // Thresholds and quantiles of an input ask for that law, not a mean.
        let threshold = declared(C::ThresholdProbability { threshold: 1.0 }, UtilityExpr::Input(0))
            .source_requirement()
            .unwrap()
            .unwrap();
        assert!(threshold.check(&[S::Mean]).is_err());
        assert_eq!(threshold.check(&[S::Cdf]).unwrap().representation, S::Cdf);
        let quantile = declared(C::Quantile { p: 0.1 }, UtilityExpr::Input(0))
            .source_requirement()
            .unwrap()
            .unwrap();
        assert!(quantile.check(&[S::Mean]).is_err());
        assert_eq!(
            quantile.check(&[S::QuantileFunction]).unwrap().representation,
            S::QuantileFunction
        );

        // A hard constraint is a threshold, so an otherwise affine utility refuses a mean.
        let mut constrained = affine.clone();
        constrained.constraints.push(cap());
        assert_ne!(constrained.identity().unwrap(), identity);
        let constrained = constrained.source_requirement().unwrap().unwrap();
        assert!(constrained.check(&[S::Mean]).is_err());
        assert_eq!(constrained.check(&[S::Cdf]).unwrap().representation, S::Cdf);

        // Nonlinear coupling, regret, and statewise optimality need one joint law.
        let coupled = contract().source_requirement().unwrap().unwrap();
        assert!(coupled.check(&[S::Mean]).is_err());
        assert_eq!(coupled.any_of, vec![S::JointDraws]);
        for criterion in [C::Regret, C::ExpectedRegret] {
            let requirement =
                declared(criterion, UtilityExpr::Input(0)).source_requirement().unwrap().unwrap();
            assert_eq!(requirement.any_of, vec![S::JointDraws]);
            assert!(requirement.check(&[S::Mean]).is_err());
        }

        // Robust decisions read a set. A mean and a joint law are both refused.
        for criterion in [C::MinimaxOverIdentifiedSet, C::MaximinOverStructures] {
            let requirement =
                declared(criterion, UtilityExpr::Input(0)).source_requirement().unwrap().unwrap();
            assert!(requirement.is_identified_set());
            assert_eq!(
                requirement.check(&[S::Mean]),
                Err(DecisionContractError::MissingSource { needed: vec![] })
            );
            assert!(requirement.check(&[S::JointDraws]).is_err());
        }
    }

    #[test]
    fn a_prespecified_program_is_not_promoted_into_a_search() {
        use SourceRepresentation as S;
        let prespecified =
            declared(DecisionCriterion::PosteriorExpectedUtility, UtilityExpr::Input(0));
        let identity = prespecified.identity().unwrap();
        let mode = prespecified.source_requirement().unwrap().unwrap().check(&[S::Mean]).unwrap();
        assert_eq!(mode.representation, S::Mean);
        assert_eq!(prespecified.identity().unwrap(), identity);

        let affine = UtilityExpr::Input(0);
        let expectation =
            search(ProgramObjective::ExpectedUtility(affine.clone())).source_requirement().unwrap();
        assert_eq!(expectation.check(&[S::Mean]).unwrap().representation, S::Mean);
        let variance =
            search(ProgramObjective::Variance(affine.clone())).source_requirement().unwrap();
        assert!(variance.check(&[S::Mean]).is_err());
        assert_eq!(
            variance.check(&[S::MeanAndCovariance]).unwrap().representation,
            S::MeanAndCovariance
        );
        let optimum =
            search(ProgramObjective::Optimality(affine.clone())).source_requirement().unwrap();
        assert_eq!(optimum.any_of, vec![S::JointDraws]);
        assert!(optimum.check(&[S::Mean]).is_err());
        let regret = search(ProgramObjective::Regret(affine)).source_requirement().unwrap();
        assert!(regret.check(&[S::Mean]).is_err());
        assert_eq!(regret.any_of, vec![S::JointDraws]);
        let robust = search(ProgramObjective::Robust).source_requirement().unwrap();
        assert!(robust.is_identified_set());
        assert!(robust.check(&[S::Mean]).is_err());

        let mut searched = search(ProgramObjective::ExpectedUtility(UtilityExpr::Input(0)));
        searched.constraints.push(cap());
        let searched = searched.source_requirement().unwrap();
        assert!(searched.check(&[S::Mean]).is_err());
        assert_eq!(searched.check(&[S::Cdf]).unwrap().representation, S::Cdf);
    }

    #[test]
    #[allow(
        clippy::type_complexity,
        reason = "one-off tuple returned from a single private helper"
    )]
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
