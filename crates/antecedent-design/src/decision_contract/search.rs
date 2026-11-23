//! Program-search contracts, validation and canonical identity.

use std::collections::HashSet;

use antecedent_core::{CanonicalWriter, is_blank};

use super::{
    DecisionContractError, DecisionFunctional, HardConstraint, MAX_DEPTH, MAX_INPUTS,
    SourceRequirement, Tail, UtilityExpr, encode_constraint, joint_draws, put_list, tighten,
};

/// Controllable variables a search ranges over.
///
/// A list of declared actions is not a space. Those actions are scored by a
/// [`super::DecisionContract`] and are not promoted into a search.
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
/// Distinct from [`super::DecisionContract`], which scores a prespecified list of
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
    /// Complete semantic search request, independent of constraint declaration order.
    ///
    /// # Errors
    /// An invalid search contract has no identity.
    pub fn identity(&self) -> Result<[u8; 32], DecisionContractError> {
        self.validate()?;
        let mut writer = CanonicalWriter::new("antecedent.program_search_contract.v1");
        writer.text(&self.space.id).texts(&self.space.variables);
        match &self.objective {
            ProgramObjective::ExpectedUtility(expr) => {
                writer.word(1);
                expr.encode(&mut writer);
            }
            ProgramObjective::Variance(expr) => {
                writer.word(2);
                expr.encode(&mut writer);
            }
            ProgramObjective::Threshold { expr, threshold, tail } => {
                writer.word(3);
                expr.encode(&mut writer);
                writer.real(*threshold).word(match tail {
                    Tail::Lower => 1,
                    Tail::Upper => 2,
                });
            }
            ProgramObjective::Quantile { expr, p } => {
                writer.word(4);
                expr.encode(&mut writer);
                writer.real(*p);
            }
            ProgramObjective::Optimality(expr) => {
                writer.word(5);
                expr.encode(&mut writer);
            }
            ProgramObjective::Regret(expr) => {
                writer.word(6);
                expr.encode(&mut writer);
            }
            ProgramObjective::Robust => {
                writer.word(7);
            }
        }
        let mut constraints: Vec<_> = self.constraints.iter().map(encode_constraint).collect();
        constraints.sort_unstable();
        put_list(&mut writer, &constraints);
        Ok(writer.finish())
    }

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
    /// The same matrix as [`super::DecisionContract::source_requirement`]: means for an
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
