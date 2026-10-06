//! Structured refusals for decision contracts, evaluation, structures and design.
//!
//! Each typed error maps to one registered reason code and a namespaced detail
//! (`family.slot`). Every detail is a string literal so the promotion gate can
//! match it against the declared refusals; nothing is parsed out of a message.
//! The existing reason codes are unchanged.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::ExternalRefusal;

use crate::decision_contract::SourceRepresentation;
use crate::decision_eval::DecisionEvalError;
use crate::decision_structural::StructuralError;
use crate::{DecisionContractError, DesignError};

fn refusal(code: &'static str, stage: &'static str, detail: &str) -> ExternalRefusal {
    ExternalRefusal {
        code,
        stage,
        detail: detail.to_owned(),
        offending: None,
        expected: None,
        supplied: None,
        capability: None,
        remedy: None,
    }
}

const fn representation_name(representation: SourceRepresentation) -> &'static str {
    match representation {
        SourceRepresentation::Mean => "mean",
        SourceRepresentation::MeanAndCovariance => "mean_and_covariance",
        SourceRepresentation::Cdf => "cdf",
        SourceRepresentation::QuantileFunction => "quantile_function",
        SourceRepresentation::MarginalDraws => "marginal_draws",
        SourceRepresentation::JointDraws => "joint_draws",
    }
}

impl DecisionContractError {
    /// Structured refusal for a contract declaration failure.
    #[must_use]
    pub fn to_refusal(&self) -> ExternalRefusal {
        let invalid = antecedent_core::reason_code!("invalid_argument");
        let quantity = antecedent_core::reason_code!("quantity_semantics_mismatch");
        match self {
            Self::InvalidDeclaration(why) => ExternalRefusal {
                expected: Some((*why).to_owned()),
                ..refusal(invalid, "declare", "decision_contract.invalid_declaration")
            },
            Self::DuplicateId(id) => ExternalRefusal {
                offending: Some(id.clone()),
                ..refusal(invalid, "declare", "decision_contract.duplicate_id")
            },
            Self::InvalidQuantity { action, input } => ExternalRefusal {
                offending: Some(format!("{action}[{input}]")),
                remedy: Some("declare a complete scientific quantity for this input"),
                ..refusal(quantity, "declare", "decision_contract.invalid_quantity")
            },
            Self::OutsideDecisionScope { action, input } => ExternalRefusal {
                offending: Some(format!("{action}[{input}]")),
                remedy: Some("read inputs for the decision's target population and horizon"),
                ..refusal(quantity, "declare", "decision_contract.outside_decision_scope")
            },
            Self::UnknownInput(position) => ExternalRefusal {
                offending: Some(position.to_string()),
                ..refusal(invalid, "declare", "decision_contract.unknown_input")
            },
            Self::UnknownAction(id) => ExternalRefusal {
                offending: Some(id.clone()),
                ..refusal(invalid, "declare", "decision_contract.unknown_action")
            },
            Self::InvalidParameter(name) => ExternalRefusal {
                offending: Some((*name).to_owned()),
                ..refusal(invalid, "declare", "decision_contract.invalid_parameter")
            },
            Self::MissingSource { needed } => ExternalRefusal {
                expected: Some(
                    needed.iter().map(|r| representation_name(*r)).collect::<Vec<_>>().join(","),
                ),
                remedy: Some("supply one of the needed source representations"),
                ..refusal(
                    antecedent_core::reason_code!("decision_contract_unsatisfied"),
                    "declare",
                    "decision_contract.missing_source",
                )
            },
        }
    }
}

impl DecisionEvalError {
    /// Structured refusal for an evaluation failure.
    #[must_use]
    pub fn to_refusal(&self) -> ExternalRefusal {
        let at = |code, detail: &str| refusal(code, "evaluate", detail);
        let quantity = antecedent_core::reason_code!("quantity_semantics_mismatch");
        match self {
            Self::Contract(error) => {
                let mut value = error.to_refusal();
                value.stage = "evaluate";
                value
            }
            Self::JointLawRequired => ExternalRefusal {
                remedy: Some("supply aligned joint draws, not independent marginals"),
                ..at(
                    antecedent_core::reason_code!("joint_law_required"),
                    "decision_evaluation.joint_law_required",
                )
            },
            Self::QuantityNotFound { action, input } => ExternalRefusal {
                offending: Some(format!("{action}[{input}]")),
                remedy: Some("supply a source coordinate with this input's exact quantity"),
                ..at(quantity, "decision_evaluation.quantity_not_found")
            },
            Self::UnsupportedCoordinate { action, input } => ExternalRefusal {
                offending: Some(format!("{action}[{input}]")),
                remedy: Some("supply support for this coordinate or remove the action"),
                ..at(quantity, "decision_evaluation.unsupported_coordinate")
            },
            Self::MeaningMismatch { action, input } => ExternalRefusal {
                offending: Some(format!("{action}[{input}]")),
                remedy: Some("supply interventional predictive draws for outcome-law inputs"),
                ..at(
                    antecedent_core::reason_code!("distribution_meaning_mismatch"),
                    "decision_evaluation.distribution_meaning",
                )
            },
            Self::StructureInputsRequired(why) => ExternalRefusal {
                expected: Some((*why).to_owned()),
                remedy: Some("evaluate under structures with evaluate_structural"),
                ..at(
                    antecedent_core::reason_code!("route_not_supported"),
                    "decision_evaluation.structure_inputs_required",
                )
            },
            Self::NonFiniteUtility { action } => ExternalRefusal {
                offending: Some(action.clone()),
                ..at(
                    antecedent_core::reason_code!("decision_contract_unsatisfied"),
                    "decision_evaluation.non_finite_utility",
                )
            },
            Self::MeanSourceInsufficient { needed } => ExternalRefusal {
                expected: Some(
                    needed.iter().map(|r| representation_name(*r)).collect::<Vec<_>>().join(","),
                ),
                supplied: Some("mean".to_owned()),
                remedy: Some(
                    "supply aligned joint draws, or use an affine utility with the expectation criteria and no hard constraints",
                ),
                ..at(
                    antecedent_core::reason_code!("decision_contract_unsatisfied"),
                    "decision_evaluation.mean_source_insufficient",
                )
            },
        }
    }
}

impl StructuralError {
    /// Structured refusal for a structural evaluation failure.
    #[must_use]
    pub fn to_refusal(&self) -> ExternalRefusal {
        let at = |code, detail: &str| refusal(code, "structural", detail);
        let unsatisfied = antecedent_core::reason_code!("decision_contract_unsatisfied");
        match self {
            Self::Contract(error) => {
                let mut value = error.to_refusal();
                value.stage = "structural";
                value
            }
            Self::InvalidAtoms => ExternalRefusal {
                remedy: Some("supply at least one structure with a distinct, non-blank identity"),
                ..at(
                    antecedent_core::reason_code!("invalid_argument"),
                    "decision_structural.invalid_atoms",
                )
            },
            Self::InvalidProbabilities => ExternalRefusal {
                remedy: Some("give every structure a finite positive probability summing to one"),
                ..at(unsatisfied, "decision_structural.invalid_probabilities")
            },
            Self::ProbabilitiesRequired => ExternalRefusal {
                remedy: Some("supply genuine structure probabilities for Bayes over structures"),
                ..at(unsatisfied, "decision_structural.probabilities_required")
            },
        }
    }
}

impl DesignError {
    /// Structured refusal for a design evaluation failure.
    ///
    /// No variant carries a typed signal, update or cost-unit failure yet, so
    /// those registered codes are not claimed here.
    #[must_use]
    pub fn to_refusal(&self) -> ExternalRefusal {
        let invalid = antecedent_core::reason_code!("invalid_argument");
        let at = |code, detail: &str| refusal(code, "design", detail);
        match self {
            Self::EmptyCandidates => at(invalid, "design_refusal.empty_candidates"),
            Self::EmptyPosterior => at(invalid, "design_refusal.empty_posterior"),
            Self::Shape(message) => ExternalRefusal {
                supplied: Some(message.clone()),
                ..at(invalid, "design_refusal.shape")
            },
            Self::Config(message) => ExternalRefusal {
                supplied: Some(message.clone()),
                ..at(invalid, "design_refusal.config")
            },
            Self::Numerical(message) => ExternalRefusal {
                supplied: Some(message.clone()),
                ..at(invalid, "design_refusal.numerical")
            },
            Self::Prob(message) => ExternalRefusal {
                supplied: Some(message.clone()),
                ..at(invalid, "design_refusal.probability")
            },
            Self::NoAdmissibleAction(message) => ExternalRefusal {
                supplied: Some(message.clone()),
                ..at(
                    antecedent_core::reason_code!("decision_contract_unsatisfied"),
                    "design_refusal.no_admissible_action",
                )
            },
            Self::Callback { name, message } => ExternalRefusal {
                offending: Some(name.clone()),
                supplied: Some(message.clone()),
                ..at(
                    antecedent_core::reason_code!("decision_contract_unsatisfied"),
                    "design_refusal.callback",
                )
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check(
        value: &ExternalRefusal,
        code: &str,
        stage: &str,
        detail: &str,
        offending: Option<&str>,
    ) {
        assert_eq!(value.code, code);
        assert_eq!(value.stage, stage);
        assert_eq!(value.detail, detail);
        assert_eq!(value.offending.as_deref(), offending);
        assert!(value.capability.is_none());
        assert!(antecedent_core::reason_code::is_runtime_refusal(value.code), "{code}");
        assert_eq!(detail.matches('.').count(), 1);
    }

    #[test]
    fn contract_errors_have_exact_codes_and_details() {
        use DecisionContractError as E;
        let quantity = "quantity_semantics_mismatch";
        let a = || "treat".to_owned();
        let cases = [
            (E::InvalidDeclaration("why"), "invalid_argument", "invalid_declaration", None),
            (E::DuplicateId("treat".into()), "invalid_argument", "duplicate_id", Some("treat")),
            (
                E::InvalidQuantity { action: a(), input: 1 },
                quantity,
                "invalid_quantity",
                Some("treat[1]"),
            ),
            (
                E::OutsideDecisionScope { action: a(), input: 0 },
                quantity,
                "outside_decision_scope",
                Some("treat[0]"),
            ),
            (E::UnknownInput(3), "invalid_argument", "unknown_input", Some("3")),
            (E::UnknownAction("x".into()), "invalid_argument", "unknown_action", Some("x")),
            (
                E::InvalidParameter("criterion"),
                "invalid_argument",
                "invalid_parameter",
                Some("criterion"),
            ),
        ];
        for (error, code, slot, offending) in cases {
            let detail = format!("decision_contract.{slot}");
            check(&error.to_refusal(), code, "declare", &detail, offending);
        }
        let value = E::InvalidDeclaration("why").to_refusal();
        assert_eq!(value.expected.as_deref(), Some("why"));
        let value = E::OutsideDecisionScope { action: a(), input: 0 }.to_refusal();
        assert!(value.remedy.is_some());
    }

    #[test]
    fn missing_source_names_the_needed_representations() {
        let value = DecisionContractError::MissingSource {
            needed: vec![SourceRepresentation::JointDraws, SourceRepresentation::MeanAndCovariance],
        }
        .to_refusal();
        check(
            &value,
            "decision_contract_unsatisfied",
            "declare",
            "decision_contract.missing_source",
            None,
        );
        assert_eq!(value.expected.as_deref(), Some("joint_draws,mean_and_covariance"));
        assert!(value.remedy.is_some());
    }

    #[test]
    fn evaluation_errors_have_exact_codes_details_and_remedies() {
        use DecisionEvalError as E;
        let a = || "treat".to_owned();
        let value = E::JointLawRequired.to_refusal();
        check(
            &value,
            "joint_law_required",
            "evaluate",
            "decision_evaluation.joint_law_required",
            None,
        );
        assert_eq!(value.remedy, Some("supply aligned joint draws, not independent marginals"));

        let value = E::QuantityNotFound { action: a(), input: 2 }.to_refusal();
        check(
            &value,
            "quantity_semantics_mismatch",
            "evaluate",
            "decision_evaluation.quantity_not_found",
            Some("treat[2]"),
        );
        let value = E::UnsupportedCoordinate { action: a(), input: 0 }.to_refusal();
        check(
            &value,
            "quantity_semantics_mismatch",
            "evaluate",
            "decision_evaluation.unsupported_coordinate",
            Some("treat[0]"),
        );
        let value = E::MeaningMismatch { action: a(), input: 1 }.to_refusal();
        check(
            &value,
            "distribution_meaning_mismatch",
            "evaluate",
            "decision_evaluation.distribution_meaning",
            Some("treat[1]"),
        );
        assert_eq!(
            value.remedy,
            Some("supply interventional predictive draws for outcome-law inputs")
        );
        let value = E::StructureInputsRequired("regret").to_refusal();
        check(
            &value,
            "route_not_supported",
            "evaluate",
            "decision_evaluation.structure_inputs_required",
            None,
        );
        assert_eq!(value.expected.as_deref(), Some("regret"));
        assert_eq!(value.remedy, Some("evaluate under structures with evaluate_structural"));
        let value = E::NonFiniteUtility { action: a() }.to_refusal();
        check(
            &value,
            "decision_contract_unsatisfied",
            "evaluate",
            "decision_evaluation.non_finite_utility",
            Some("treat"),
        );
        let nested = E::Contract(DecisionContractError::UnknownInput(1)).to_refusal();
        check(
            &nested,
            "invalid_argument",
            "evaluate",
            "decision_contract.unknown_input",
            Some("1"),
        );
        for error in [E::JointLawRequired, E::StructureInputsRequired("x")] {
            assert_eq!(error.to_refusal().code, error.reason_code());
        }
    }

    #[test]
    fn structural_errors_have_exact_codes_and_details() {
        use StructuralError as E;
        let unsatisfied = "decision_contract_unsatisfied";
        let value = E::InvalidAtoms.to_refusal();
        check(&value, "invalid_argument", "structural", "decision_structural.invalid_atoms", None);
        assert!(value.remedy.is_some());
        let value = E::InvalidProbabilities.to_refusal();
        check(&value, unsatisfied, "structural", "decision_structural.invalid_probabilities", None);
        let value = E::ProbabilitiesRequired.to_refusal();
        check(
            &value,
            unsatisfied,
            "structural",
            "decision_structural.probabilities_required",
            None,
        );
        assert!(value.remedy.is_some());
        let nested = E::Contract(DecisionContractError::DuplicateId("t".into())).to_refusal();
        check(
            &nested,
            "invalid_argument",
            "structural",
            "decision_contract.duplicate_id",
            Some("t"),
        );
        for error in [E::InvalidAtoms, E::InvalidProbabilities, E::ProbabilitiesRequired] {
            assert_eq!(error.to_refusal().code, error.reason_code());
        }
    }

    #[test]
    fn design_errors_have_exact_codes_and_details() {
        let invalid = "invalid_argument";
        let unsatisfied = "decision_contract_unsatisfied";
        let cases = [
            (DesignError::EmptyCandidates, invalid, "design_refusal.empty_candidates", None),
            (DesignError::EmptyPosterior, invalid, "design_refusal.empty_posterior", None),
            (DesignError::Shape("s".into()), invalid, "design_refusal.shape", Some("s")),
            (DesignError::Config("c".into()), invalid, "design_refusal.config", Some("c")),
            (DesignError::Numerical("n".into()), invalid, "design_refusal.numerical", Some("n")),
            (DesignError::Prob("p".into()), invalid, "design_refusal.probability", Some("p")),
            (
                DesignError::NoAdmissibleAction("m".into()),
                unsatisfied,
                "design_refusal.no_admissible_action",
                Some("m"),
            ),
        ];
        for (error, code, detail, supplied) in cases {
            let value = error.to_refusal();
            check(&value, code, "design", detail, None);
            assert_eq!(value.supplied.as_deref(), supplied);
        }
        let value =
            DesignError::Callback { name: "utility".into(), message: "boom".into() }.to_refusal();
        check(&value, unsatisfied, "design", "design_refusal.callback", Some("utility"));
        assert_eq!(value.supplied.as_deref(), Some("boom"));
    }
}
