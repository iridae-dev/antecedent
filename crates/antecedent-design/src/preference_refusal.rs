//! Structured refusals for value models, preference contracts, recorded choices and decisions
//! across a preference set.
//!
//! Each typed error maps to one registered reason code, a stage and its registered detail
//! (`family.slot`, the constant its `detail()` returns), with a remedy wherever the caller can
//! act. A wrapped error that has its own refusal keeps that refusal and names the stage that
//! surfaced it; a wrapped linear-program or conditional-logit refusal is reported under the
//! wrapping error's detail with the inner detail and message as the supplied text.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_contract::ExternalRefusal;

use crate::choice_record::ChoiceRecordError;
use crate::preference_certified::PreferenceCertifiedError;
use crate::preference_choice_model::ChoiceModelError;
use crate::preference_contract::PreferenceContractError;
use crate::preference_decision::{CertifiedRouteReason, PreferenceDecisionError};
use crate::preference_model::PreferenceModelError;
use crate::preference_set_learning::SetLearningError;
use crate::signal::{RefusalBuilder, at_stage};

/// Stage of a value-model, preference-contract or choice-record declaration.
const DECLARE: &str = "declare";
/// Stage of an evaluation across a preference set.
const EVALUATE: &str = "evaluate";
/// Stage of the structure checks of an evaluation across structures.
const STRUCTURAL: &str = "structural";
/// Stage of learning the preference set compatible with recorded choices.
const LEARN: &str = "learn";
/// Stage of estimating preference parameters from recorded choices.
const ESTIMATE: &str = "estimate";

const CERTIFIED_ROUTE: &str = "evaluate with evaluate_preference_certified on aligned joint draws";
const COMPATIBLE_SET: &str = "use the compatible-set method (learn_compatible_set)";

fn refusal(code: &'static str, stage: &'static str, detail: &str) -> RefusalBuilder {
    RefusalBuilder::new(code, stage, detail)
}

fn at(action: &str, coordinate: impl std::fmt::Display) -> String {
    format!("{action}[{coordinate}]")
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
        }
    }
}

impl ChoiceRecordError {
    /// Structured refusal for a choice-record declaration or feature-extraction failure.
    ///
    /// A value-model refusal keeps its own refusal.
    #[must_use]
    #[allow(clippy::too_many_lines, reason = "one arm per choice-record declaration failure")]
    pub fn to_refusal(&self) -> ExternalRefusal {
        let invalid = |remedy| {
            refusal(antecedent_core::reason_code!("invalid_argument"), DECLARE, self.detail())
                .remedy(remedy)
        };
        let unsatisfied = |remedy| {
            refusal(
                antecedent_core::reason_code!("decision_contract_unsatisfied"),
                DECLARE,
                self.detail(),
            )
            .remedy(remedy)
        };
        let describe = |attribute: Option<&crate::preference_model::Attribute>| {
            attribute.map_or_else(|| "none".to_owned(), |a| format!("{} [{}]", a.id, a.units))
        };
        match self {
            Self::Model(error) => at_stage(error.to_refusal(), DECLARE),
            Self::AttributesMissing => {
                invalid("declare the attributes the recorded options are described by").build()
            }
            Self::AttributeIdBlank { index } => {
                invalid("give every attribute a non-blank id").offending(index.to_string()).build()
            }
            Self::AttributeUnitsBlank { attribute } => {
                invalid("state the units of every attribute").offending(attribute.clone()).build()
            }
            Self::AttributeDuplicate { attribute } => {
                invalid("give every attribute a distinct id").offending(attribute.clone()).build()
            }
            Self::PopulationBlank => {
                invalid("name the population whose choices were recorded").build()
            }
            Self::PoolingAssumptionBlank => {
                invalid("name the assumption that the deciders share one set of preferences")
                    .build()
            }
            Self::RecordsMissing => invalid("supply at least one recorded choice").build(),
            Self::RecordIdBlank { index } => {
                invalid("give every record a non-blank id").offending(index.to_string()).build()
            }
            Self::RecordDuplicate { record } => {
                invalid("give every record a distinct id").offending(record.clone()).build()
            }
            Self::DeciderBlank { record } => {
                invalid("name who made the choice").offending(record.clone()).build()
            }
            Self::OptionsTooFew { record, options } => {
                invalid("record every option that was available, at least two")
                    .offending(record.clone())
                    .expected("2")
                    .supplied(options.to_string())
                    .build()
            }
            Self::OptionIdBlank { record, index } => {
                invalid("give every option a non-blank id").offending(at(record, index)).build()
            }
            Self::OptionDuplicate { record, option } => {
                invalid("give every option a distinct id").offending(at(record, option)).build()
            }
            Self::ChosenUnknown { record, chosen } => {
                invalid("name one of the recorded options as chosen")
                    .offending(record.clone())
                    .supplied(chosen.clone())
                    .build()
            }
            Self::ForecastSourceBlank { record } => {
                invalid("name where the forecast came from").offending(record.clone()).build()
            }
            Self::AnalysisIdentityBlank { record } => {
                invalid("name the identity of the result the decider read")
                    .offending(record.clone())
                    .build()
            }
            Self::AnalysisDateBlank { record } => {
                invalid("name the date the decider read the result")
                    .offending(record.clone())
                    .build()
            }
            Self::RealizedAssumptionBlank { record } => {
                invalid("name the assumption that the decider expected the realized outcome")
                    .offending(record.clone())
                    .build()
            }
            Self::MeansWidthMismatch { record, option, attributes, means } => {
                invalid("supply one mean per attribute")
                    .offending(at(record, option))
                    .expected(attributes.to_string())
                    .supplied(means.to_string())
                    .build()
            }
            Self::DrawsEmpty { record, option } => {
                invalid("supply at least one draw row").offending(at(record, option)).build()
            }
            Self::DrawsWidthMismatch { record, option, attributes, values } => {
                invalid("supply one value per attribute in every draw row")
                    .offending(at(record, option))
                    .expected(format!("a multiple of {attributes}"))
                    .supplied(values.to_string())
                    .build()
            }
            Self::BeliefNotFinite { record, option } => {
                invalid("supply finite means or draws").offending(at(record, option)).build()
            }
            Self::WeightsLengthMismatch { record, option, rows, weights } => {
                invalid("supply one weight per draw row")
                    .offending(at(record, option))
                    .expected(rows.to_string())
                    .supplied(weights.to_string())
                    .build()
            }
            Self::WeightNotFinite { record, option, row } => invalid("supply finite draw weights")
                .offending(format!("{}[{row}]", at(record, option)))
                .build(),
            Self::WeightNegative { record, option, row } => {
                invalid("supply non-negative draw weights")
                    .offending(format!("{}[{row}]", at(record, option)))
                    .build()
            }
            Self::WeightsSumInvalid { record, option } => {
                invalid("supply draw weights with a positive, finite total")
                    .offending(at(record, option))
                    .build()
            }
            Self::ObservationRowsMissing { record } => {
                invalid("name the observation rows the record rests on")
                    .offending(record.clone())
                    .build()
            }
            Self::ObservationRowBlank { record, index } => {
                invalid("give every observation row a non-blank id")
                    .offending(at(record, index))
                    .build()
            }
            Self::ObservationRowRepeated { record, row } => {
                invalid("name each observation row once").offending(at(record, row)).build()
            }
            Self::ObservationRowDoubleCounted { row, first_record, second_record } => refusal(
                antecedent_core::reason_code!("evidence_double_counted"),
                DECLARE,
                self.detail(),
            )
            .offending(row.clone())
            .supplied(format!("{first_record},{second_record}"))
            .remedy("keep the observation row in one record only")
            .build(),
            Self::AttributesDiffer { index, model, records } => refusal(
                antecedent_core::reason_code!("quantity_semantics_mismatch"),
                DECLARE,
                self.detail(),
            )
            .offending(index.to_string())
            .expected(describe(model.as_ref()))
            .supplied(describe(records.as_ref()))
            .remedy("describe the recorded options in the value model's attributes and units")
            .build(),
            Self::ShapeParametersNotFixed { parameters } => refusal(
                antecedent_core::reason_code!("route_not_supported"),
                DECLARE,
                self.detail(),
            )
            .offending(parameters.join(","))
            .remedy("fix these parameters to a single value, or evaluate on the certified route")
            .build(),
            Self::BeliefLawRequired { record, option, .. } => {
                unsatisfied("record the belief as draws")
                    .offending(at(record, option))
                    .expected("draws")
                    .supplied("means")
                    .build()
            }
            Self::TermNotFinite { record, option } => {
                unsatisfied("check the recorded belief and the domain of the value")
                    .offending(at(record, option))
                    .build()
            }
        }
    }
}

impl PreferenceDecisionError {
    /// Structured refusal for a failure of the exact evaluation across a preference set.
    ///
    /// A contract or source refusal keeps its own refusal; a linear-program refusal is
    /// reported under `preference_decision.linear_program_refused` with the linear program's
    /// detail and message as the supplied text.
    #[must_use]
    #[allow(clippy::too_many_lines, reason = "one arm per exact-evaluation failure")]
    pub fn to_refusal(&self) -> ExternalRefusal {
        let invalid = |remedy| {
            refusal(antecedent_core::reason_code!("invalid_argument"), EVALUATE, self.detail())
                .remedy(remedy)
        };
        let unsatisfied = |stage, remedy| {
            refusal(
                antecedent_core::reason_code!("decision_contract_unsatisfied"),
                stage,
                self.detail(),
            )
            .remedy(remedy)
        };
        let structural = |remedy| {
            refusal(antecedent_core::reason_code!("invalid_argument"), STRUCTURAL, self.detail())
                .remedy(remedy)
        };
        match self {
            Self::Contract(error) => at_stage(error.to_refusal(), EVALUATE),
            Self::Eval(error) => at_stage(error.to_refusal(), EVALUATE),
            Self::Lp(error) => refusal(
                antecedent_core::reason_code!("invalid_argument"),
                EVALUATE,
                antecedent_core::reason_details::preference_decision::LINEAR_PROGRAM_REFUSED,
            )
            .supplied(format!("{}: {error}", error.detail()))
            .remedy(
                "relax the linear-program options, or rescale the declared ranges and relations",
            )
            .build(),
            Self::CertifiedRouteRequired { reason } => {
                let builder = refusal(
                    antecedent_core::reason_code!("route_not_supported"),
                    EVALUATE,
                    self.detail(),
                )
                .remedy(CERTIFIED_ROUTE);
                match reason {
                    CertifiedRouteReason::Criterion(criterion) => {
                        builder.offending(format!("{criterion:?}")).build()
                    }
                    CertifiedRouteReason::ShapeParameters { parameters } => {
                        builder.offending(parameters.join(",")).build()
                    }
                    CertifiedRouteReason::PartNotLowerable
                    | CertifiedRouteReason::WorstCaseOfExpectedRegret => builder.build(),
                }
            }
            Self::InvalidOptions { option } => {
                invalid("supply the named option inside its documented range")
                    .offending(*option)
                    .build()
            }
            Self::EmptyPreferenceSet => {
                invalid("widen the declared ranges or remove a relation").build()
            }
            Self::MeasureRequired => unsatisfied(
                EVALUATE,
                "supply a measure over the preference set, or choose another preference policy",
            )
            .build(),
            Self::MeasureIncomplete { parameters } => {
                unsatisfied(EVALUATE, "supply a measure that resolves every parameter")
                    .offending(parameters.join(","))
                    .build()
            }
            Self::MeasureEmpty => invalid("supply at least one measure point").build(),
            Self::MeasurePointDimension { index, expected, found } => {
                invalid("supply one value per preference parameter")
                    .offending(index.to_string())
                    .expected(expected.to_string())
                    .supplied(found.to_string())
                    .build()
            }
            Self::MeasurePointNonFinite { index } => {
                invalid("supply finite measure points").offending(index.to_string()).build()
            }
            Self::MeasurePointOutsideSet { index } => {
                invalid("supply measure points inside the declared ranges and relations")
                    .offending(index.to_string())
                    .build()
            }
            Self::MeasureWeightCount { expected, found } => {
                invalid("supply one weight per measure point")
                    .expected(expected.to_string())
                    .supplied(found.to_string())
                    .build()
            }
            Self::MeasureWeightInvalid { index } => {
                invalid("supply finite, non-negative measure weights")
                    .offending(index.to_string())
                    .build()
            }
            Self::MeasureWeightTotalZero => {
                invalid("give at least one measure point a positive weight").build()
            }
            Self::MeasureLabelCount { expected, found } => {
                invalid("supply one label per measure point")
                    .expected(expected.to_string())
                    .supplied(found.to_string())
                    .build()
            }
            Self::MeasureLabelBlank { index } => {
                invalid("give every measure point a non-blank label")
                    .offending(index.to_string())
                    .build()
            }
            Self::MeasureLabelDuplicate { label } => {
                invalid("give every measure point a distinct label")
                    .offending(label.clone())
                    .build()
            }
            Self::MeasureOriginBlank => {
                invalid("name the assumption or estimate the measure comes from").build()
            }
            Self::MeasureUnresolvedUnknown { index } => {
                invalid("name only declared parameter positions as unresolved")
                    .offending(index.to_string())
                    .build()
            }
            Self::MeasureUnresolvedRepeated { index } => {
                invalid("name each unresolved parameter once").offending(index.to_string()).build()
            }
            Self::NoStructures => structural("supply at least one structure").build(),
            Self::StructureIdBlank { position } => {
                structural("give every structure a non-blank id")
                    .offending(position.to_string())
                    .build()
            }
            Self::StructureDuplicate { id } => {
                structural("give every structure a distinct id").offending(id.clone()).build()
            }
            Self::StructureProbabilitiesPartial => {
                unsatisfied(STRUCTURAL, "give every structure a probability, or none").build()
            }
            Self::StructureProbabilityInvalid { structure } => {
                unsatisfied(STRUCTURAL, "supply finite, non-negative structure probabilities")
                    .offending(structure.clone())
                    .build()
            }
            Self::StructureProbabilitiesExceedOne => {
                unsatisfied(STRUCTURAL, "supply structure probabilities whose total is at most one")
                    .build()
            }
            Self::StructureProbabilitiesRequired => unsatisfied(
                STRUCTURAL,
                "supply genuine structure probabilities, or choose another structural policy",
            )
            .build(),
            Self::NonFiniteCoefficient { action } => {
                unsatisfied(EVALUATE, "check the source draws and the domain of the value")
                    .offending(action.clone())
                    .build()
            }
            Self::UnboundedProgram | Self::UnexpectedInfeasible => {
                refusal(antecedent_core::reason_code!("invalid_argument"), EVALUATE, self.detail())
                    .build()
            }
        }
    }
}

impl PreferenceCertifiedError {
    /// Structured refusal for a failure of the certified evaluation across a preference set.
    ///
    /// A contract, value-model or source refusal keeps its own refusal.
    #[must_use]
    #[allow(clippy::too_many_lines, reason = "one arm per certified-evaluation failure")]
    pub fn to_refusal(&self) -> ExternalRefusal {
        let invalid = |stage, remedy| {
            refusal(antecedent_core::reason_code!("invalid_argument"), stage, self.detail())
                .remedy(remedy)
        };
        let unsatisfied = |stage, remedy| {
            refusal(
                antecedent_core::reason_code!("decision_contract_unsatisfied"),
                stage,
                self.detail(),
            )
            .remedy(remedy)
        };
        let unsupported = |stage, remedy| {
            refusal(antecedent_core::reason_code!("route_not_supported"), stage, self.detail())
                .remedy(remedy)
        };
        let corners = |lower: &[f64], upper: &[f64]| format!("{lower:?}..{upper:?}");
        match self {
            Self::Contract(error) => at_stage(error.to_refusal(), EVALUATE),
            Self::Model(error) => at_stage(error.to_refusal(), EVALUATE),
            Self::Eval(error) => at_stage(error.to_refusal(), EVALUATE),
            Self::StructureInputsRequired => unsupported(
                EVALUATE,
                "evaluate across structures with evaluate_preference_certified_structural",
            )
            .build(),
            Self::InvalidOptions { option } => {
                invalid(EVALUATE, "supply the named option inside its documented range")
                    .offending(*option)
                    .build()
            }
            Self::MinimaxRegretNeedsExpectation => unsupported(
                EVALUATE,
                "use an expectation criterion, or choose another preference policy",
            )
            .build(),
            Self::MeasureRequired => unsatisfied(
                EVALUATE,
                "supply a measure over the preference set, or choose another preference policy",
            )
            .build(),
            Self::MeasureEmpty => invalid(EVALUATE, "supply at least one measure point").build(),
            Self::MeasureUnknownParameter { index } => {
                invalid(EVALUATE, "name only declared parameter positions as unresolved")
                    .offending(index.to_string())
                    .build()
            }
            Self::MeasureIncomplete { parameters } => {
                unsatisfied(EVALUATE, "supply a measure that resolves every parameter")
                    .offending(parameters.join(","))
                    .build()
            }
            Self::MeasurePointDimension { index, expected, found } => {
                invalid(EVALUATE, "supply one value per preference parameter")
                    .offending(index.to_string())
                    .expected(expected.to_string())
                    .supplied(found.to_string())
                    .build()
            }
            Self::MeasureNonFinitePoint { index } => {
                invalid(EVALUATE, "supply finite measure points")
                    .offending(index.to_string())
                    .build()
            }
            Self::MeasurePointOutsideSet { index } => {
                invalid(EVALUATE, "supply measure points inside the declared ranges and relations")
                    .offending(index.to_string())
                    .build()
            }
            Self::MeasureWeightsInvalid => invalid(
                EVALUATE,
                "supply one finite, non-negative weight per measure point with a positive total",
            )
            .build(),
            Self::MeasureLabelsInvalid => {
                invalid(EVALUATE, "supply one label per measure point").build()
            }
            Self::EmptyPreferenceSet => {
                invalid(EVALUATE, "widen the declared ranges or remove a relation").build()
            }
            Self::LogarithmDomain { action, lower, upper } => invalid(
                EVALUATE,
                "keep the logarithm's argument positive: narrow the ranges or change the value",
            )
            .offending(action.clone())
            .supplied(corners(lower, upper))
            .build(),
            Self::NonFiniteEnclosure { subject, lower, upper } => invalid(
                EVALUATE,
                "narrow the parameter ranges, or change the value so it stays finite",
            )
            .offending(subject.clone())
            .supplied(corners(lower, upper))
            .build(),
            Self::InvalidAtoms => {
                invalid(STRUCTURAL, "supply at least one structure with a distinct, non-blank id")
                    .build()
            }
            Self::InvalidProbabilities => unsatisfied(
                STRUCTURAL,
                "give each structure a finite, non-negative probability summing to at most one",
            )
            .build(),
            Self::ProbabilitiesRequired => unsatisfied(
                STRUCTURAL,
                "supply genuine structure probabilities, or choose another structural policy",
            )
            .build(),
            Self::MixtureCriterionUnsupported { criterion } => unsupported(
                STRUCTURAL,
                "use an expectation criterion, or choose another structural policy",
            )
            .offending(*criterion)
            .build(),
        }
    }
}

impl SetLearningError {
    /// Structured refusal for a failure to learn the preference set compatible with recorded
    /// choices.
    ///
    /// A value-model or choice-record refusal keeps its own refusal; a linear-program refusal
    /// is reported under `preference_set_learning.linear_program_refused` with the linear
    /// program's detail and message as the supplied text. Contradictory records are an
    /// outcome, not a refusal.
    #[must_use]
    pub fn to_refusal(&self) -> ExternalRefusal {
        let invalid = |remedy| {
            refusal(antecedent_core::reason_code!("invalid_argument"), LEARN, self.detail())
                .remedy(remedy)
        };
        let tolerance = "supply a finite, non-negative tolerance in utility units";
        match self {
            Self::Model(error) => at_stage(error.to_refusal(), LEARN),
            Self::Records(error) => at_stage(error.to_refusal(), LEARN),
            Self::Lp(error) => invalid(
                "relax SetLearningOptions::lp, or rescale the declared ranges and relations",
            )
            .supplied(format!("{}: {error}", error.detail()))
            .build(),
            Self::ToleranceNegative { tolerance: value } => {
                invalid(tolerance).supplied(value.to_string()).build()
            }
            Self::ToleranceNonFinite => invalid(tolerance).build(),
            Self::DeclaredSetEmpty => {
                invalid("widen the declared ranges or remove a declared relation").build()
            }
            Self::RelationIdCollision { id } => {
                invalid("rename the declared relation or the record whose choice relation collides")
                    .offending(id.clone())
                    .build()
            }
            Self::ProgramInconsistent { stage } => {
                invalid("tighten the feasibility tolerance in SetLearningOptions::lp")
                    .supplied(*stage)
                    .build()
            }
        }
    }
}

impl ChoiceModelError {
    /// Structured refusal for a failure to estimate preference parameters from recorded
    /// choices.
    ///
    /// A value-model or choice-record refusal keeps its own refusal; a conditional-logit
    /// refusal is reported under `preference_choice_model.fit_refused` with the fit's detail
    /// and message as the supplied text.
    #[must_use]
    #[allow(clippy::too_many_lines, reason = "one arm per estimation failure")]
    pub fn to_refusal(&self) -> ExternalRefusal {
        let invalid = |remedy| {
            refusal(antecedent_core::reason_code!("invalid_argument"), ESTIMATE, self.detail())
                .remedy(remedy)
        };
        let design = |decider: Option<String>, remedy| {
            refusal(antecedent_core::reason_code!("design_rank_deficient"), ESTIMATE, self.detail())
                .offending_opt(decider)
                .remedy(remedy)
        };
        let fit = |decider: Option<String>, remedy| {
            refusal(antecedent_core::reason_code!("glm_fit_refused"), ESTIMATE, self.detail())
                .offending_opt(decider)
                .remedy(remedy)
        };
        match self {
            Self::LevelInvalid { level } => invalid("supply a level strictly between 0 and 1")
                .expected("(0, 1)")
                .supplied(level.to_string())
                .build(),
            Self::DrawsMissing => invalid("request at least one measure draw").build(),
            Self::FitOptionsInvalid => {
                invalid("supply a finite, positive gradient tolerance and separation bound").build()
            }
            Self::Model(error) => at_stage(error.to_refusal(), ESTIMATE),
            Self::Records(error) => at_stage(error.to_refusal(), ESTIMATE),
            Self::NoUnitOfAccount { decider } => design(
                decider.clone(),
                "record choices whose options differ in the fixed part of the value",
            )
            .build(),
            Self::NothingIdentified { decider } => design(
                decider.clone(),
                "record choices whose options differ in the terms of the value",
            )
            .build(),
            Self::RankDeficient { decider, unit_of_account, parameters } => {
                let mut columns = parameters.clone();
                if *unit_of_account {
                    columns.insert(0, "unit_of_account".to_owned());
                }
                design(
                    decider.clone(),
                    "record choices that vary the named parameters apart, or fix one of them",
                )
                .supplied(columns.join(","))
                .build()
            }
            Self::Separated { decider } => fit(decider.clone(), COMPATIBLE_SET).build(),
            Self::ConsistencyNotPositive { decider, consistency } => fit(
                decider.clone(),
                "check the sign of the fixed part of the value, or use the compatible-set method",
            )
            .supplied(consistency.to_string())
            .build(),
            Self::NotConverged { decider, iterations, gradient_norm } => fit(
                decider.clone(),
                "raise the fit's iteration limit, or use the compatible-set method",
            )
            .supplied(format!("{iterations} iterations, gradient norm {gradient_norm}"))
            .build(),
            Self::Fit { decider, error } => fit(decider.clone(), COMPATIBLE_SET)
                .supplied(format!("{}: {error}", error.detail()))
                .build(),
            Self::CovarianceNotPositiveDefinite { decider, kind } => {
                fit(decider.clone(), "record more choices, or use the compatible-set method")
                    .supplied(format!("{kind:?}"))
                    .build()
            }
            Self::NoDrawInsideSet { decider, drawn } => refusal(
                antecedent_core::reason_code!("uncertainty_representation_insufficient"),
                ESTIMATE,
                self.detail(),
            )
            .offending_opt(decider.clone())
            .supplied(drawn.to_string())
            .remedy("draw more, record more choices, or use the compatible-set method")
            .build(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use antecedent_core::reason_details;

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
}
