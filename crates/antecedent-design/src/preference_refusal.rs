//! Shared stage names, remedies and builders for the structured refusals of value models,
//! preference contracts, recorded choices and decisions across a preference set.
//!
//! Each typed error's `to_refusal()` lives beside its `detail()` in the error's own module and
//! maps the error to one registered reason code, a stage and its registered detail
//! (`family.slot`), with a remedy wherever the caller can act. A wrapped error that has its own
//! refusal keeps that refusal and names the stage that surfaced it; a wrapped linear-program or
//! conditional-logit refusal is reported under the wrapping error's detail with the inner detail
//! and message as the supplied text.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::signal::RefusalBuilder;

/// Stage of a value-model, preference-contract or choice-record declaration.
pub(crate) const DECLARE: &str = "declare";
/// Stage of an evaluation across a preference set.
pub(crate) const EVALUATE: &str = "evaluate";
/// Stage of the structure checks of an evaluation across structures.
pub(crate) const STRUCTURAL: &str = "structural";
/// Stage of learning the preference set compatible with recorded choices.
pub(crate) const LEARN: &str = "learn";
/// Stage of estimating preference parameters from recorded choices.
pub(crate) const ESTIMATE: &str = "estimate";
/// Stage of accepting, acknowledging and versioning preferences.
pub(crate) const ACCEPT: &str = "accept";
/// Stage of writing a preference artifact.
pub(crate) const EXPORT: &str = "export";
/// Stage of reading, checking and replaying a preference artifact.
pub(crate) const CONSUME: &str = "consume";

/// Remedy pointing at the certified route.
pub(crate) const CERTIFIED_ROUTE: &str =
    "evaluate with evaluate_preference_certified on aligned joint draws";
/// Remedy pointing at the compatible-set method.
pub(crate) const COMPATIBLE_SET: &str = "use the compatible-set method (learn_compatible_set)";

/// Refusal builder for a registered code, a stage and a registered detail.
pub(crate) fn refusal(code: &'static str, stage: &'static str, detail: &str) -> RefusalBuilder {
    RefusalBuilder::new(code, stage, detail)
}

/// `action[coordinate]`, the offending-input spelling of an indexed input.
pub(crate) fn at(action: &str, coordinate: impl std::fmt::Display) -> String {
    format!("{action}[{coordinate}]")
}

#[cfg(test)]
mod tests {
    use antecedent_contract::ExternalRefusal;
    use antecedent_core::reason_details;

    use crate::accepted_preferences::AcceptedPreferencesError;
    use crate::choice_record::ChoiceRecordError;
    use crate::preference_artifact::PreferenceArtifactError;
    use crate::preference_certified::PreferenceCertifiedError;
    use crate::preference_choice_model::ChoiceModelError;
    use crate::preference_contract::PreferenceContractError;
    use crate::preference_decision::{CertifiedRouteReason, PreferenceDecisionError};
    use crate::preference_evaluation::PreferenceEvaluationError;
    use crate::preference_model::PreferenceModelError;
    use crate::preference_set_learning::SetLearningError;

    use super::{CERTIFIED_ROUTE, COMPATIBLE_SET};

    /// The refusal is well formed and its code is one its detail is registered under.
    fn registered(value: &ExternalRefusal, stage: &str) {
        assert_eq!(value.stage, stage);
        assert!(value.validate().is_ok(), "{value:?}");
        let codes = reason_details::codes_of(&value.detail);
        assert!(codes.is_some_and(|codes| codes.contains(&value.code)), "{value:?}");
    }

    #[test]
    fn value_model_refusals_are_registered_declarations() {
        let value = PreferenceModelError::ReversedRange { parameter: "cost".into() }.to_refusal();
        registered(&value, "declare");
        assert_eq!(value.code, "invalid_argument");
        assert_eq!(value.detail, reason_details::preference_model::PARAMETER_RANGE_EMPTY);
        assert_eq!(value.offending.as_deref(), Some("cost"));
        assert!(value.remedy.is_some());
        registered(&PreferenceModelError::NonFiniteValue.to_refusal(), "declare");
    }

    #[test]
    fn contract_refusals_keep_wrapped_details_and_name_units() {
        let value = PreferenceContractError::QuantityUnitsMismatch {
            action: "treat".into(),
            attribute: "cost".into(),
        }
        .to_refusal();
        registered(&value, "declare");
        assert_eq!(value.code, "quantity_semantics_mismatch");
        assert_eq!(value.offending.as_deref(), Some("treat[cost]"));
        let nested =
            PreferenceContractError::Model(PreferenceModelError::BlankUtilityUnits).to_refusal();
        registered(&nested, "declare");
        assert_eq!(nested.detail, reason_details::preference_model::UTILITY_UNITS_BLANK);
        let value = PreferenceContractError::NotLowerable { action: "treat".into() }.to_refusal();
        registered(&value, "declare");
        assert_eq!(value.remedy, Some(CERTIFIED_ROUTE));
    }

    #[test]
    fn record_refusals_name_double_counting_and_needed_draws() {
        let value = ChoiceRecordError::ObservationRowDoubleCounted {
            row: "visit".into(),
            first_record: "r1".into(),
            second_record: "r2".into(),
        }
        .to_refusal();
        registered(&value, "declare");
        assert_eq!(value.code, "evidence_double_counted");
        assert_eq!(value.supplied.as_deref(), Some("r1,r2"));
        let value = ChoiceRecordError::BeliefLawRequired {
            record: "r1".into(),
            option: "a".into(),
            term: None,
        }
        .to_refusal();
        registered(&value, "declare");
        assert_eq!(value.expected.as_deref(), Some("draws"));
    }

    #[test]
    fn decision_refusals_point_at_the_certified_route_and_structures() {
        let value = PreferenceDecisionError::CertifiedRouteRequired {
            reason: CertifiedRouteReason::ShapeParameters { parameters: vec!["p".into()] },
        }
        .to_refusal();
        registered(&value, "evaluate");
        assert_eq!(value.code, "route_not_supported");
        assert_eq!(value.remedy, Some(CERTIFIED_ROUTE));
        registered(
            &PreferenceDecisionError::StructureProbabilitiesRequired.to_refusal(),
            "structural",
        );
        registered(&PreferenceDecisionError::MeasureRequired.to_refusal(), "evaluate");
        let nested =
            PreferenceDecisionError::Contract(PreferenceContractError::TooFewActions { count: 1 })
                .to_refusal();
        registered(&nested, "evaluate");
        assert_eq!(nested.detail, reason_details::preference_contract::TOO_FEW_ACTIONS);
    }

    #[test]
    fn certified_refusals_are_registered() {
        for error in [
            PreferenceCertifiedError::StructureInputsRequired,
            PreferenceCertifiedError::MinimaxRegretNeedsExpectation,
            PreferenceCertifiedError::MeasureIncomplete { parameters: vec!["p".into()] },
            PreferenceCertifiedError::EmptyPreferenceSet,
        ] {
            registered(&error.to_refusal(), "evaluate");
        }
        for error in [
            PreferenceCertifiedError::InvalidAtoms,
            PreferenceCertifiedError::ProbabilitiesRequired,
            PreferenceCertifiedError::MixtureCriterionUnsupported { criterion: "quantile" },
        ] {
            registered(&error.to_refusal(), "structural");
        }
    }

    #[test]
    fn learning_and_estimation_refusals_are_registered() {
        registered(&SetLearningError::ToleranceNegative { tolerance: -1.0 }.to_refusal(), "learn");
        registered(&SetLearningError::DeclaredSetEmpty.to_refusal(), "learn");
        let value = ChoiceModelError::Separated { decider: Some("d1".into()) }.to_refusal();
        registered(&value, "estimate");
        assert_eq!(value.code, "glm_fit_refused");
        assert_eq!(value.remedy, Some(COMPATIBLE_SET));
        assert_eq!(value.offending.as_deref(), Some("d1"));
        let value = ChoiceModelError::RankDeficient {
            decider: None,
            unit_of_account: true,
            parameters: vec!["p".into()],
        }
        .to_refusal();
        registered(&value, "estimate");
        assert_eq!(value.supplied.as_deref(), Some("unit_of_account,p"));
        let nested = ChoiceModelError::Records(ChoiceRecordError::RecordsMissing).to_refusal();
        registered(&nested, "estimate");
        assert_eq!(nested.detail, reason_details::choice_record::RECORDS_MISSING);
    }

    #[test]
    fn a_contradiction_states_the_tolerance_and_the_records() {
        use crate::preference_set_learning::{ChoiceConflict, ConflictKind};
        let value = AcceptedPreferencesError::Contradictory {
            decider: Some("d1".into()),
            conflict: ChoiceConflict {
                records: vec!["r1".into(), "r2".into()],
                kind: ConflictKind::Smallest,
                tolerance_needed: 0.5,
            },
        }
        .to_refusal();
        registered(&value, "accept");
        assert_eq!(value.detail, reason_details::accepted_preferences::CHOICES_CONTRADICTORY);
        assert_eq!(value.expected.as_deref(), Some("tolerance >= 0.5"));
        assert_eq!(value.supplied.as_deref(), Some("r1,r2"));
        assert_eq!(
            value.remedy,
            Some("raise the tolerance to the stated value or remove one of the named records")
        );
    }

    #[test]
    fn shape_learning_refusals_are_registered() {
        let value =
            PreferenceModelError::ValueRelationRowsInvalid { relation: "choice.r1.b".into() }
                .to_refusal();
        registered(&value, "declare");
        assert_eq!(value.offending.as_deref(), Some("choice.r1.b"));
        registered(
            &PreferenceModelError::ValueRelationToleranceInvalid { relation: "v".into() }
                .to_refusal(),
            "declare",
        );
        let value = SetLearningError::RegionBudgetInvalid { option: "max_boxes" }.to_refusal();
        registered(&value, "learn");
        assert_eq!(value.offending.as_deref(), Some("max_boxes"));
        let value = PreferenceDecisionError::CertifiedRouteRequired {
            reason: CertifiedRouteReason::ValueRelations { relations: vec!["choice.r1.b".into()] },
        }
        .to_refusal();
        registered(&value, "evaluate");
        assert_eq!(value.code, "route_not_supported");
        assert_eq!(value.remedy, Some(CERTIFIED_ROUTE));
        let value =
            ChoiceModelError::SeparatedAtShape { decider: None, shape: vec![("risk".into(), 0.5)] }
                .to_refusal();
        registered(&value, "estimate");
        assert_eq!(value.detail, reason_details::preference_choice_model::SEPARATED);
        assert_eq!(value.supplied.as_deref(), Some("risk=0.5"));
        for error in [
            ChoiceModelError::ShapeNotIdentified {
                decider: None,
                parameter: "risk".into(),
                profile_range: 0.1,
            },
            ChoiceModelError::ShapeEstimateAtBound {
                decider: Some("d1".into()),
                parameter: "risk".into(),
                estimate: 0.3,
                interval: (0.3, 0.6),
            },
            ChoiceModelError::ProfileMultimodal {
                decider: None,
                parameter: "risk".into(),
                modes: (0.2, 1.8),
            },
            ChoiceModelError::ShapeSearchBudget { decider: None, dimensions: 4, limit: 3 },
        ] {
            registered(&error.to_refusal(), "estimate");
        }
        let value = AcceptedPreferencesError::Undetermined { decider: None, boundary_boxes: 3 }
            .to_refusal();
        registered(&value, "accept");
        assert_eq!(value.detail, reason_details::accepted_preferences::SET_UNDETERMINED);
    }

    #[test]
    fn refusals_of_too_few_clusters_and_foreign_learned_preferences_are_registered() {
        let value = ChoiceModelError::TooFewDecidersToCluster {
            decider: None,
            deciders: 3,
            coefficients: 2,
            required: 5,
        }
        .to_refusal();
        registered(&value, "estimate");
        assert_eq!(value.supplied.as_deref(), Some("3"));
        registered(
            &AcceptedPreferencesError::DeclarationNotKept { part: "relations" }.to_refusal(),
            "accept",
        );
        registered(
            &AcceptedPreferencesError::RecordsMismatch { expected: "a".into(), found: "b".into() }
                .to_refusal(),
            "accept",
        );
        let nested = PreferenceCertifiedError::SetProgram(PreferenceDecisionError::Lp(
            antecedent_stats::linear_program::LpError::BoundsLength {
                variables: 1,
                lower: 0,
                upper: 0,
            },
        ))
        .to_refusal();
        registered(&nested, "evaluate");
        assert_eq!(nested.detail, reason_details::preference_decision::LINEAR_PROGRAM_REFUSED);
    }

    #[test]
    fn evaluation_and_artifact_refusals_are_registered() {
        let value =
            PreferenceEvaluationError::JointDrawsRequired { reason: "criterion" }.to_refusal();
        registered(&value, "evaluate");
        assert_eq!(value.code, "joint_law_required");
        registered(
            &PreferenceArtifactError::ReplayMismatch { field: "verdict" }.to_refusal(),
            "consume",
        );
        registered(
            &PreferenceArtifactError::Malformed { what: "kind".into() }.to_refusal(),
            "consume",
        );
        let nested =
            PreferenceArtifactError::Accepted(AcceptedPreferencesError::NoItems).to_refusal();
        registered(&nested, "consume");
        assert_eq!(nested.detail, reason_details::accepted_preferences::ACKNOWLEDGE_EMPTY);
    }
}
