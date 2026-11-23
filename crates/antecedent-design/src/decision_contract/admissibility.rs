//! Decision admissibility rules, uncertainty and support contracts.

use super::{DecisionContract, DecisionContractError, put_list};
use crate::signal::RefusalBuilder;
use antecedent_contract::ExternalRefusal;
use antecedent_core::{CanonicalWriter, SupportStatus, is_blank};
use std::collections::HashSet;

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
    /// Weakest status that still supports the action; a weaker one excludes the
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

    /// Whether `status` supports the input.
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
                build(antecedent_core::reason_details::decision_admissibility::UNKNOWN_ACTION)
                    .offending(id.clone())
                    .build()
            }
            Self::UnknownInput { action, input } => {
                build(antecedent_core::reason_details::decision_admissibility::UNKNOWN_INPUT)
                    .offending(format!("{action}[{input}]"))
                    .build()
            }
            Self::DuplicateRule(key) => {
                build(antecedent_core::reason_details::decision_admissibility::DUPLICATE_RULE)
                    .offending(key.clone())
                    .build()
            }
            Self::InvalidRule(why) => {
                build(antecedent_core::reason_details::decision_admissibility::INVALID_RULE)
                    .expected(*why)
                    .build()
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
