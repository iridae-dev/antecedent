//! Semantic contract for observation-conditioned scientific queries.
//!
//! A query that conditions on an observation and one that intervenes are
//! different claims, including when both are computed from the same causal
//! model, and a value read off an instrument is not the construct the
//! instrument measures. This module decides whether a *request* is of the kind
//! the numerical layer may execute at all, and says which request it refused
//! and what declaration would license it. It computes no probability.
//!
//! Refusals are [`ExternalRefusal`]s under the `observation_query` detail
//! namespace, each with a registered reason code and an actionable remedy:
//!
//! - an intervention (or any value that was assigned rather than recorded)
//!   placed in a conditioning slot, and an intervention supplied to a
//!   conditioning query;
//! - a conditional intervention (observations and an intervention in one
//!   request), which has no bounded route here;
//! - the recorded value of an indicator used as the construct it measures,
//!   unless a measurement model relating the two is declared and applies in
//!   the request's population, regime and arm;
//! - a level that no admissible evidence reaches, stating the most extreme
//!   value that is reachable;
//! - an interval or a calibrated-probability claim, both closed.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeSet;

use antecedent_core::measurement_model::{
    ArmBinding, ErrorStructure, MeasurementKind, MeasurementModel, MeasurementParameters,
    RepresentationVersion,
};

use crate::measurement::{check_representation_transfer, model_refusal};
use crate::{ExternalRefusal, MechanismOriginKind};

/// Stage reported by every refusal of this module.
pub const OBSERVATION_QUERY_STAGE: &str = "observe";

/// Which operation a query performs. The tag is part of every claim's
/// identity, so the same variables, levels and model give two distinct claims.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ObservationOperation {
    /// `P(outcome | observed values)`: the observational law, keeping every
    /// association between the observed variables and the outcome.
    Conditioning,
    /// `P(outcome | do(assignment))`: the truncated factorization.
    Intervention,
}

impl ObservationOperation {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Conditioning => "conditioning",
            Self::Intervention => "intervention",
        }
    }
}

/// How a value placed in a slot came to be.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum SlotProvenance {
    /// Recorded for a unit: the only value a conditioning slot may hold.
    Recorded,
    /// Assigned by an action or an intervention.
    Assigned,
    /// A what-if value that was never recorded.
    Hypothetical,
}

impl SlotProvenance {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Recorded => "recorded",
            Self::Assigned => "assigned",
            Self::Hypothetical => "hypothetical",
        }
    }
}

/// An indicator whose recorded value is offered about a model variable.
#[derive(Clone, Debug, PartialEq)]
pub struct MeasuredIndicator {
    /// Stable identity of the recorded column.
    pub indicator_id: String,
    /// What kind of quantity the recorded column is.
    pub kind: MeasurementKind,
    /// The measurement model relating the indicator to the variable. `None`
    /// states that the recorded value is being used as the variable itself,
    /// which is refused.
    pub model: Option<MeasurementModel>,
    /// For a binary indicator, which level of the variable is the positive
    /// class (the one `sensitivity` is the rate for).
    pub positive_level: Option<String>,
    /// The representation the caller expects the indicator in.
    pub representation: RepresentationVersion,
}

/// What a slot's value is a value of.
#[derive(Clone, Debug, PartialEq)]
#[allow(
    clippy::large_enum_variant,
    reason = "a slot is declared once per request and boxing the indicator would change the public variant shape"
)]
pub enum SlotValue {
    /// The model variable itself, recorded without error.
    Direct,
    /// The recorded value of an indicator of the model variable.
    Measured(MeasuredIndicator),
}

/// One value placed in a conditioning slot.
#[derive(Clone, Debug, PartialEq)]
pub struct ObservedSlot {
    /// The model variable the value is about.
    pub variable_id: String,
    /// The level. For a measured value this is the level of the *recorded*
    /// indicator, in the variable's level coding.
    pub level: String,
    /// How the value came to be.
    pub provenance: SlotProvenance,
    /// What the value is a value of.
    pub value: SlotValue,
}

/// A request to be checked before any probability is formed.
#[derive(Clone, Debug)]
pub struct ObservationSlotsRequest<'a> {
    /// The operation the request performs.
    pub operation: ObservationOperation,
    /// The outcome variable.
    pub outcome_id: &'a str,
    /// Values placed in conditioning slots.
    pub conditioning: &'a [ObservedSlot],
    /// `(variable, level)` assignments of `do()`.
    pub interventions: &'a [(String, String)],
}

fn refuse(
    code: &'static str,
    detail: &str,
    offending: impl Into<String>,
    remedy: &'static str,
) -> ExternalRefusal {
    ExternalRefusal::new(code, OBSERVATION_QUERY_STAGE, detail)
        .with_offending(offending)
        .with_remedy(remedy)
}

fn blank(text: &str) -> bool {
    text.trim().is_empty()
}

/// Check the slots of a request.
///
/// # Errors
/// The first failed check, as a coded structured refusal: a blank name; an
/// assigned or hypothetical value (or an intervention) in a conditioning
/// slot; observations and an intervention in one request; a variable assigned
/// twice; the outcome among the assigned variables; an indicator of a kind
/// that is not a proxy; a measured value without a measurement model (the
/// recorded value used as the construct); more than one measured signal.
#[allow(
    clippy::result_large_err,
    reason = "a refusal is the cold path of a once-per-call check; boxing the error would not pay"
)]
#[allow(
    clippy::too_many_lines,
    reason = "a flat sequence of independent slot checks, each with its own coded refusal"
)]
pub fn check_observation_slots(
    request: &ObservationSlotsRequest<'_>,
) -> Result<(), ExternalRefusal> {
    let invalid = antecedent_core::reason_code!("invalid_argument");
    if blank(request.outcome_id) {
        return Err(refuse(
            invalid,
            "observation_query.blank_field",
            "outcome_id",
            "name the outcome variable",
        ));
    }
    for slot in request.conditioning {
        if blank(&slot.variable_id) || blank(&slot.level) {
            return Err(refuse(
                invalid,
                "observation_query.blank_field",
                "conditioning",
                "name the variable and the level of every conditioning slot",
            ));
        }
    }
    for slot in request.conditioning {
        if slot.provenance != SlotProvenance::Recorded {
            return Err(ExternalRefusal::new(
                antecedent_core::reason_code!("quantity_semantics_mismatch"),
                OBSERVATION_QUERY_STAGE,
                "observation_query.intervention_in_conditioning_slot",
            )
            .with_offending(slot.variable_id.clone())
            .with_expected(SlotProvenance::Recorded.as_str())
            .with_supplied(slot.provenance.as_str())
            .with_remedy(
                "a conditioning slot holds only values that were recorded; state an assigned value as an intervention (a separate interventional claim), never as an observation",
            ));
        }
    }
    if request.operation == ObservationOperation::Conditioning {
        if let Some((variable, _)) = request.interventions.first() {
            return Err(ExternalRefusal::new(
                antecedent_core::reason_code!("quantity_semantics_mismatch"),
                OBSERVATION_QUERY_STAGE,
                "observation_query.intervention_in_conditioning_slot",
            )
            .with_offending(variable.clone())
            .with_expected(ObservationOperation::Conditioning.as_str())
            .with_supplied(ObservationOperation::Intervention.as_str())
            .with_remedy(
                "an intervention is its own claim: ask for the interventional distribution as an intervention query, and for the conditional distribution as a conditioning query",
            ));
        }
    } else if !request.conditioning.is_empty() {
        return Err(ExternalRefusal::new(
            antecedent_core::reason_code!("route_not_supported"),
            OBSERVATION_QUERY_STAGE,
            "observation_query.mixed_conditioning_and_intervention",
        )
        .with_offending(request.conditioning[0].variable_id.clone())
        .with_remedy(
            "a distribution that conditions on observations after an intervention has no bounded route; ask the conditional and the interventional claims separately",
        ));
    }
    let mut seen = BTreeSet::new();
    let assigned = request
        .conditioning
        .iter()
        .map(|s| s.variable_id.as_str())
        .chain(request.interventions.iter().map(|(v, _)| v.as_str()));
    for variable in assigned {
        if variable == request.outcome_id {
            return Err(refuse(
                invalid,
                "observation_query.outcome_assigned",
                variable,
                "ask about an outcome that is not itself conditioned on or intervened on",
            ));
        }
        if !seen.insert(variable) {
            return Err(refuse(
                invalid,
                "observation_query.repeated_slot",
                variable,
                "assign each variable at most once",
            ));
        }
    }
    let mut measured = 0_usize;
    for slot in request.conditioning {
        if let SlotValue::Measured(indicator) = &slot.value {
            measured += 1;
            if matches!(
                indicator.kind,
                MeasurementKind::Observed | MeasurementKind::LatentScientific
            ) {
                return Err(refuse(
                    invalid,
                    "observation_query.indicator_kind",
                    indicator.indicator_id.clone(),
                    "a measured value is offered for a noisy, assigned, learned or derived indicator; an error-free recorded value is a direct slot",
                ));
            }
            if indicator.model.is_none() {
                return Err(measured_as_construct(slot, indicator));
            }
        }
    }
    if measured > 1 {
        return Err(refuse(
            antecedent_core::reason_code!("route_not_supported"),
            "observation_query.multiple_measured_signals",
            "conditioning",
            "condition on one measured signal per query; several indicators need a joint measurement model",
        ));
    }
    Ok(())
}

fn measured_as_construct(slot: &ObservedSlot, indicator: &MeasuredIndicator) -> ExternalRefusal {
    ExternalRefusal::new(
        antecedent_core::reason_code!("proxy_offered_as_construct"),
        OBSERVATION_QUERY_STAGE,
        "observation_query.measured_as_construct",
    )
    .with_offending(slot.variable_id.clone())
    .with_supplied(format!("{}:{}", indicator.kind.as_str(), indicator.indicator_id))
    .with_expected(MeasurementKind::LatentScientific.as_str())
    .with_remedy(
        "declare a measurement model (a confusion matrix or reliability) relating the recorded indicator to the construct and condition on the indicator through it, or ask the question about the recorded indicator itself",
    )
}

/// Population, regime and arm a query is about.
#[derive(Clone, Copy, Debug)]
pub struct ObservationContext<'a> {
    /// Population the query is about.
    pub population_id: &'a str,
    /// Regime the query is about.
    pub regime_id: &'a str,
    /// Arm the query is about; `None` for all arms.
    pub arm_id: Option<&'a str>,
}

fn mismatch(detail: &str, expected: &str, supplied: &str) -> ExternalRefusal {
    ExternalRefusal::new(
        antecedent_core::reason_code!("measurement_binding_mismatch"),
        OBSERVATION_QUERY_STAGE,
        detail,
    )
    .with_expected(expected)
    .with_supplied(supplied)
    .with_remedy(
        "establish the measurement model for this population, regime and arm, or declare the invariance premise under a transport contract",
    )
}

/// `P(recorded indicator value | construct = level)` for every level of the
/// construct, aligned with `construct_levels`.
///
/// This is the likelihood the Bayes inversion needs. It exists only for a
/// measured slot whose measurement model is non-differential, is declared for
/// exactly this construct and indicator, applies in the request's population,
/// regime and arm and representation, and is a binary-misclassification or
/// categorical-confusion route.
///
/// # Errors
/// A measured value without a model (the recorded value used as the
/// construct); a malformed model; a different indicator, construct,
/// population, regime, arm or representation; error declared to differ by
/// arm, population or time; a route with no confusion matrix; levels that do
/// not match the model's.
#[allow(
    clippy::result_large_err,
    reason = "a refusal is the cold path of a once-per-call check; boxing the error would not pay"
)]
pub fn measured_likelihood(
    slot: &ObservedSlot,
    construct_levels: &[String],
    context: &ObservationContext<'_>,
) -> Result<Vec<f64>, ExternalRefusal> {
    let SlotValue::Measured(indicator) = &slot.value else {
        return Err(refuse(
            antecedent_core::reason_code!("invalid_argument"),
            "observation_query.indicator_kind",
            slot.variable_id.clone(),
            "only a measured slot has an indicator likelihood",
        ));
    };
    let Some(model) = &indicator.model else {
        return Err(measured_as_construct(slot, indicator));
    };
    model.validate().map_err(|error| model_refusal(&error))?;
    if indicator.indicator_id != model.indicator.id {
        return Err(mismatch(
            "observation_query.indicator_mismatch",
            &model.indicator.id,
            &indicator.indicator_id,
        ));
    }
    if slot.variable_id != model.construct.id {
        return Err(mismatch(
            "observation_query.construct_mismatch",
            &model.construct.id,
            &slot.variable_id,
        ));
    }
    if context.population_id != model.binding.population_id {
        return Err(mismatch(
            "observation_query.population_mismatch",
            &model.binding.population_id,
            context.population_id,
        ));
    }
    if context.regime_id != model.binding.regime_id {
        return Err(mismatch(
            "observation_query.regime_mismatch",
            &model.binding.regime_id,
            context.regime_id,
        ));
    }
    if let ArmBinding::Arm(bound) = &model.binding.arm {
        if context.arm_id != Some(bound.as_str()) {
            return Err(mismatch(
                "observation_query.arm_mismatch",
                bound,
                context.arm_id.unwrap_or("all_arms"),
            ));
        }
    }
    check_representation_transfer(
        &model.representation,
        &indicator.representation,
        None,
        false,
        model.indicator.kind,
    )?;
    if model.structure != ErrorStructure::NonDifferential {
        return Err(ExternalRefusal::new(
            antecedent_core::reason_code!("differential_error_not_pooled"),
            OBSERVATION_QUERY_STAGE,
            "observation_query.differential_error_unsupported",
        )
        .with_offending(indicator.indicator_id.clone())
        .with_remedy(
            "the Bayes inversion assumes error independent of every other variable given the construct; error that differs by arm, population or time has no bounded conditioning route",
        ));
    }
    if !construct_levels.iter().any(|l| l == &slot.level) {
        return Err(refuse(
            antecedent_core::reason_code!("invalid_argument"),
            "observation_query.unknown_level",
            slot.level.clone(),
            "name a recorded level in the construct's level coding",
        ));
    }
    match &model.parameters {
        MeasurementParameters::Binary { rates, .. } => {
            binary_likelihood(slot, indicator, construct_levels, rates)
        }
        MeasurementParameters::Categorical { levels, confusion, .. } => {
            categorical_likelihood(slot, construct_levels, levels, confusion)
        }
        _ => Err(ExternalRefusal::new(
            antecedent_core::reason_code!("route_not_supported"),
            OBSERVATION_QUERY_STAGE,
            "observation_query.measurement_route",
        )
        .with_supplied(model.route.as_str())
        .with_remedy(
            "conditioning on a measured signal needs a confusion matrix: declare a binary-misclassification or categorical-confusion measurement model",
        )),
    }
}

#[allow(
    clippy::result_large_err,
    reason = "a refusal is the cold path of a once-per-call check; boxing the error would not pay"
)]
fn binary_likelihood(
    slot: &ObservedSlot,
    indicator: &MeasuredIndicator,
    construct_levels: &[String],
    rates: &[antecedent_core::measurement_model::BinaryArmRates],
) -> Result<Vec<f64>, ExternalRefusal> {
    let invalid = antecedent_core::reason_code!("invalid_argument");
    let Some(positive) = indicator.positive_level.as_deref() else {
        return Err(refuse(
            invalid,
            "observation_query.positive_level",
            indicator.indicator_id.clone(),
            "declare which level of the construct is the positive class the sensitivity refers to",
        ));
    };
    if construct_levels.len() != 2 || !construct_levels.iter().any(|l| l == positive) {
        return Err(refuse(
            invalid,
            "observation_query.positive_level",
            positive.to_owned(),
            "a binary indicator needs a construct with exactly two levels, one of them the declared positive level",
        ));
    }
    let Some(rate) = rates.first() else {
        return Err(refuse(
            invalid,
            "observation_query.invalid_declaration",
            indicator.indicator_id.clone(),
            "the binary model carries no sensitivity and specificity",
        ));
    };
    let observed_positive = slot.level == positive;
    Ok(construct_levels
        .iter()
        .map(|level| {
            let truly_positive = level == positive;
            match (observed_positive, truly_positive) {
                (true, true) => rate.sensitivity,
                (false, true) => 1.0 - rate.sensitivity,
                (true, false) => 1.0 - rate.specificity,
                (false, false) => rate.specificity,
            }
        })
        .collect())
}

#[allow(
    clippy::result_large_err,
    reason = "a refusal is the cold path of a once-per-call check; boxing the error would not pay"
)]
fn categorical_likelihood(
    slot: &ObservedSlot,
    construct_levels: &[String],
    model_levels: &[String],
    confusion: &[f64],
) -> Result<Vec<f64>, ExternalRefusal> {
    let k = model_levels.len();
    let same_set =
        construct_levels.len() == k && construct_levels.iter().all(|l| model_levels.contains(l));
    if !same_set {
        return Err(mismatch(
            "observation_query.construct_mismatch",
            &model_levels.join(","),
            &construct_levels.join(","),
        ));
    }
    let Some(observed) = model_levels.iter().position(|l| l == &slot.level) else {
        return Err(refuse(
            antecedent_core::reason_code!("invalid_argument"),
            "observation_query.unknown_level",
            slot.level.clone(),
            "name a recorded level in the confusion model's level coding",
        ));
    };
    // Row-major [observed][true].
    Ok(construct_levels
        .iter()
        .map(|level| {
            let truth = model_levels.iter().position(|l| l == level).unwrap_or(0);
            confusion[observed * k + truth]
        })
        .collect())
}

// ------------------------------------------------------------ closed routes

/// The refusal of an interval (credible, confidence or prediction) for any
/// observation-conditioned value.
#[must_use]
pub fn interval_refusal() -> ExternalRefusal {
    ExternalRefusal::new(
        antecedent_core::reason_code!("cell_not_licensed"),
        OBSERVATION_QUERY_STAGE,
        "observation_query.interval_route_closed",
    )
    .with_remedy(
        "none available yet: the exact value under the declared model, or the labelled point value of a fitted model, is the result; an interval needs a measured whole-method coverage record",
    )
}

/// The refusal of a calibration claim for a posterior or decision probability.
#[must_use]
pub fn calibration_refusal() -> ExternalRefusal {
    ExternalRefusal::new(
        antecedent_core::reason_code!("cell_not_licensed"),
        OBSERVATION_QUERY_STAGE,
        "observation_query.calibrated_probability_closed",
    )
    .with_remedy(
        "none available yet: a probability computed from a model fitted to a sample is a point value with no claim about how often it is right; calibration needs a measured whole-method record at this query's coordinate",
    )
}

/// The refusal of a level that no admissible evidence reaches, stating the
/// most extreme reachable value (a supremum when raising, an infimum when
/// lowering) and why it is the limit.
#[must_use]
pub fn level_unreachable_refusal(
    level: f64,
    extremum: f64,
    direction: &str,
    bound: &str,
) -> ExternalRefusal {
    ExternalRefusal::new(
        antecedent_core::reason_code!("evidence_cannot_reach_level"),
        OBSERVATION_QUERY_STAGE,
        "observation_query.level_unreachable",
    )
    .with_offending(format!("{direction}:{bound}"))
    .with_expected(format!("{level}"))
    .with_supplied(format!("{extremum}"))
    .with_remedy(
        "state a level inside the reachable range, supply a signal with a larger likelihood ratio, or allow more observations",
    )
}

/// The refusal of conditioning on evidence the model gives probability zero.
#[must_use]
pub fn zero_probability_evidence_refusal() -> ExternalRefusal {
    ExternalRefusal::new(
        antecedent_core::reason_code!("invalid_argument"),
        OBSERVATION_QUERY_STAGE,
        "observation_query.zero_probability_evidence",
    )
    .with_remedy(
        "condition on values the declared model gives positive probability, or revise the model's support",
    )
}

/// How an observation-conditioned value is to be read.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum InferenceBasis {
    /// Exact given a model the analyst declared: a statement about the
    /// declared model, not about any sample.
    DeclaredModel,
    /// Computed from a model fitted to a sample (or supplied by an external
    /// provider): a point value with no coverage or calibration claim.
    SampleFittedPointOnly,
}

impl InferenceBasis {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::DeclaredModel => "exact_given_declared_model",
            Self::SampleFittedPointOnly => "point_only_sample_fitted",
        }
    }

    /// The basis implied by the origins of the mechanisms behind a value:
    /// declared only when every mechanism was written down by the analyst.
    #[must_use]
    pub fn from_origins(origins: &[MechanismOriginKind]) -> Self {
        if origins.iter().all(|o| *o == MechanismOriginKind::Declared) {
            Self::DeclaredModel
        } else {
            Self::SampleFittedPointOnly
        }
    }
}

#[cfg(test)]
mod tests {
    use antecedent_core::measurement_model::{
        BinaryArmRates, MeasurementBinding, MeasurementRoute, MeasurementVariable,
        ParameterEvidence,
    };

    use super::*;

    fn representation() -> RepresentationVersion {
        RepresentationVersion { coordinate_system_id: "assay".into(), version: 1 }
    }

    fn binary_model() -> MeasurementModel {
        MeasurementModel {
            model_id: "assay-validation".into(),
            route: MeasurementRoute::BinaryMisclassification,
            indicator: MeasurementVariable {
                id: "test_result".into(),
                kind: MeasurementKind::NoisyMisclassified,
                units: "binary".into(),
            },
            construct: MeasurementVariable {
                id: "disease".into(),
                kind: MeasurementKind::LatentScientific,
                units: "binary".into(),
            },
            binding: MeasurementBinding {
                population_id: "clinic".into(),
                regime_id: "observational".into(),
                arm: ArmBinding::AllArms,
            },
            structure: ErrorStructure::NonDifferential,
            parameters: MeasurementParameters::Binary {
                rates: vec![BinaryArmRates {
                    arm_id: None,
                    sensitivity: 0.9,
                    specificity: 0.8,
                    sensitivity_variance: 0.0,
                    specificity_variance: 0.0,
                }],
                min_youden: 0.1,
            },
            representation: representation(),
            evidence: ParameterEvidence::Declared { declarer: "lab".into() },
        }
    }

    fn measured_slot(model: Option<MeasurementModel>, level: &str) -> ObservedSlot {
        ObservedSlot {
            variable_id: "disease".into(),
            level: level.into(),
            provenance: SlotProvenance::Recorded,
            value: SlotValue::Measured(MeasuredIndicator {
                indicator_id: "test_result".into(),
                kind: MeasurementKind::NoisyMisclassified,
                model,
                positive_level: Some("1".into()),
                representation: representation(),
            }),
        }
    }

    fn direct(variable: &str, level: &str, provenance: SlotProvenance) -> ObservedSlot {
        ObservedSlot {
            variable_id: variable.into(),
            level: level.into(),
            provenance,
            value: SlotValue::Direct,
        }
    }

    fn levels() -> Vec<String> {
        vec!["0".into(), "1".into()]
    }

    const CONTEXT: ObservationContext<'static> =
        ObservationContext { population_id: "clinic", regime_id: "observational", arm_id: None };

    fn conditioning(slots: &[ObservedSlot]) -> ObservationSlotsRequest<'_> {
        ObservationSlotsRequest {
            operation: ObservationOperation::Conditioning,
            outcome_id: "outcome",
            conditioning: slots,
            interventions: &[],
        }
    }

    #[test]
    fn recorded_values_pass_and_an_assigned_value_is_an_intervention_in_a_conditioning_slot() {
        let ok = [direct("signal", "1", SlotProvenance::Recorded)];
        assert!(check_observation_slots(&conditioning(&ok)).is_ok());

        for provenance in [SlotProvenance::Assigned, SlotProvenance::Hypothetical] {
            let smuggled = [direct("signal", "1", provenance)];
            let refusal = check_observation_slots(&conditioning(&smuggled)).unwrap_err();
            assert_eq!(refusal.code, "quantity_semantics_mismatch");
            assert_eq!(refusal.detail, "observation_query.intervention_in_conditioning_slot");
            assert_eq!(refusal.offending.as_deref(), Some("signal"));
            assert_eq!(refusal.supplied.as_deref(), Some(provenance.as_str()));
            assert!(refusal.remedy.is_some());
            assert!(refusal.validate().is_ok());
        }
    }

    #[test]
    fn an_intervention_supplied_to_a_conditioning_query_is_refused_and_mixing_is_closed() {
        let interventions = [("signal".to_owned(), "1".to_owned())];
        let request = ObservationSlotsRequest {
            operation: ObservationOperation::Conditioning,
            outcome_id: "outcome",
            conditioning: &[],
            interventions: &interventions,
        };
        let refusal = check_observation_slots(&request).unwrap_err();
        assert_eq!(refusal.detail, "observation_query.intervention_in_conditioning_slot");
        assert_eq!(refusal.supplied.as_deref(), Some("intervention"));

        let observed = [direct("other", "0", SlotProvenance::Recorded)];
        let mixed = ObservationSlotsRequest {
            operation: ObservationOperation::Intervention,
            outcome_id: "outcome",
            conditioning: &observed,
            interventions: &interventions,
        };
        let refusal = check_observation_slots(&mixed).unwrap_err();
        assert_eq!(refusal.code, "route_not_supported");
        assert_eq!(refusal.detail, "observation_query.mixed_conditioning_and_intervention");

        let alone = ObservationSlotsRequest {
            operation: ObservationOperation::Intervention,
            outcome_id: "outcome",
            conditioning: &[],
            interventions: &interventions,
        };
        assert!(check_observation_slots(&alone).is_ok());
    }

    #[test]
    fn repeated_slots_and_an_assigned_outcome_are_refused() {
        let repeated = [
            direct("signal", "1", SlotProvenance::Recorded),
            direct("signal", "0", SlotProvenance::Recorded),
        ];
        assert_eq!(
            check_observation_slots(&conditioning(&repeated)).unwrap_err().detail,
            "observation_query.repeated_slot"
        );
        let outcome = [direct("outcome", "1", SlotProvenance::Recorded)];
        assert_eq!(
            check_observation_slots(&conditioning(&outcome)).unwrap_err().detail,
            "observation_query.outcome_assigned"
        );
        let blank_level = [direct("signal", " ", SlotProvenance::Recorded)];
        assert_eq!(
            check_observation_slots(&conditioning(&blank_level)).unwrap_err().detail,
            "observation_query.blank_field"
        );
    }

    #[test]
    fn a_measured_value_used_as_the_construct_is_refused_until_a_model_is_declared() {
        let bare = [measured_slot(None, "1")];
        let refusal = check_observation_slots(&conditioning(&bare)).unwrap_err();
        assert_eq!(refusal.code, "proxy_offered_as_construct");
        assert_eq!(refusal.detail, "observation_query.measured_as_construct");
        assert_eq!(refusal.offending.as_deref(), Some("disease"));
        assert!(refusal.remedy.is_some());
        assert!(refusal.validate().is_ok());
        // The likelihood function refuses the same way.
        let same = measured_likelihood(&bare[0], &levels(), &CONTEXT).unwrap_err();
        assert_eq!(same.detail, "observation_query.measured_as_construct");

        let modelled = [measured_slot(Some(binary_model()), "1")];
        assert!(check_observation_slots(&conditioning(&modelled)).is_ok());

        // An error-free kind is not a measured indicator.
        let mut wrong_kind = measured_slot(None, "1");
        if let SlotValue::Measured(m) = &mut wrong_kind.value {
            m.kind = MeasurementKind::Observed;
        }
        assert_eq!(
            check_observation_slots(&conditioning(&[wrong_kind])).unwrap_err().detail,
            "observation_query.indicator_kind"
        );

        // Two measured signals are not combined.
        let two = [
            measured_slot(Some(binary_model()), "1"),
            ObservedSlot {
                variable_id: "other".into(),
                ..measured_slot(Some(binary_model()), "0")
            },
        ];
        let refusal = check_observation_slots(&conditioning(&two)).unwrap_err();
        assert_eq!(refusal.code, "route_not_supported");
        assert_eq!(refusal.detail, "observation_query.multiple_measured_signals");
    }

    #[test]
    fn the_binary_likelihood_is_the_declared_sensitivity_and_specificity() {
        let positive = measured_slot(Some(binary_model()), "1");
        let l = measured_likelihood(&positive, &levels(), &CONTEXT).unwrap();
        // P(M=1 | C=0) = 1 - 0.8, P(M=1 | C=1) = 0.9.
        assert!((l[0] - 0.2).abs() < 1e-15 && (l[1] - 0.9).abs() < 1e-15, "{l:?}");
        let negative = measured_slot(Some(binary_model()), "0");
        let l = measured_likelihood(&negative, &levels(), &CONTEXT).unwrap();
        // P(M=0 | C=0) = 0.8, P(M=0 | C=1) = 1 - 0.9.
        assert!((l[0] - 0.8).abs() < 1e-15 && (l[1] - 0.1).abs() < 1e-15, "{l:?}");
        // Level order of the construct, not of the declaration, decides alignment.
        let swapped = vec!["1".to_owned(), "0".to_owned()];
        let l = measured_likelihood(&positive, &swapped, &CONTEXT).unwrap();
        assert!((l[0] - 0.9).abs() < 1e-15 && (l[1] - 0.2).abs() < 1e-15, "{l:?}");
    }

    #[test]
    fn the_categorical_likelihood_reads_the_observed_row_of_the_confusion_matrix() {
        let mut model = binary_model();
        model.route =
            antecedent_core::measurement_model::MeasurementRoute::ProbabilisticCategorical;
        model.indicator.kind = MeasurementKind::ProbabilisticFeatureAssignment;
        // Column-stochastic, row-major [observed][true]; levels a, b, c.
        model.parameters = MeasurementParameters::Categorical {
            levels: vec!["a".into(), "b".into(), "c".into()],
            confusion: vec![0.8, 0.1, 0.0, 0.1, 0.7, 0.2, 0.1, 0.2, 0.8],
            column_counts: None,
            max_condition: 100.0,
        };
        let mut slot = measured_slot(Some(model), "b");
        if let SlotValue::Measured(m) = &mut slot.value {
            m.kind = MeasurementKind::ProbabilisticFeatureAssignment;
            m.positive_level = None;
        }
        let construct: Vec<String> = vec!["c".into(), "a".into(), "b".into()];
        let l = measured_likelihood(&slot, &construct, &CONTEXT).unwrap();
        // Observed b: row [0.1, 0.7, 0.2] over true (a, b, c) -> aligned (c, a, b).
        assert!((l[0] - 0.2).abs() < 1e-15, "{l:?}");
        assert!((l[1] - 0.1).abs() < 1e-15, "{l:?}");
        assert!((l[2] - 0.7).abs() < 1e-15, "{l:?}");
        // Mismatched level sets are refused.
        let narrower: Vec<String> = vec!["a".into(), "b".into()];
        assert_eq!(
            measured_likelihood(&slot, &narrower, &CONTEXT).unwrap_err().detail,
            "observation_query.construct_mismatch"
        );
    }

    #[test]
    fn a_model_for_another_construct_population_or_error_structure_is_not_applied() {
        let slot = measured_slot(Some(binary_model()), "1");
        let other_population = ObservationContext { population_id: "hospital", ..CONTEXT };
        let refusal = measured_likelihood(&slot, &levels(), &other_population).unwrap_err();
        assert_eq!(refusal.code, "measurement_binding_mismatch");
        assert_eq!(refusal.detail, "observation_query.population_mismatch");
        assert_eq!(refusal.expected.as_deref(), Some("clinic"));
        assert_eq!(refusal.supplied.as_deref(), Some("hospital"));

        let other_regime = ObservationContext { regime_id: "randomized", ..CONTEXT };
        assert_eq!(
            measured_likelihood(&slot, &levels(), &other_regime).unwrap_err().detail,
            "observation_query.regime_mismatch"
        );

        let mut wrong_construct = slot.clone();
        wrong_construct.variable_id = "something_else".into();
        assert_eq!(
            measured_likelihood(&wrong_construct, &levels(), &CONTEXT).unwrap_err().detail,
            "observation_query.construct_mismatch"
        );

        let mut wrong_indicator = slot.clone();
        if let SlotValue::Measured(m) = &mut wrong_indicator.value {
            m.indicator_id = "another_column".into();
        }
        assert_eq!(
            measured_likelihood(&wrong_indicator, &levels(), &CONTEXT).unwrap_err().detail,
            "observation_query.indicator_mismatch"
        );

        let mut differential = binary_model();
        differential.structure =
            ErrorStructure::Differential(antecedent_core::measurement_model::DifferentialAxis::Arm);
        differential.parameters = MeasurementParameters::Binary {
            rates: vec![
                BinaryArmRates {
                    arm_id: Some("a".into()),
                    sensitivity: 0.9,
                    specificity: 0.8,
                    sensitivity_variance: 0.0,
                    specificity_variance: 0.0,
                },
                BinaryArmRates {
                    arm_id: Some("b".into()),
                    sensitivity: 0.7,
                    specificity: 0.8,
                    sensitivity_variance: 0.0,
                    specificity_variance: 0.0,
                },
            ],
            min_youden: 0.1,
        };
        let refusal =
            measured_likelihood(&measured_slot(Some(differential), "1"), &levels(), &CONTEXT)
                .unwrap_err();
        assert_eq!(refusal.code, "differential_error_not_pooled");
        assert_eq!(refusal.detail, "observation_query.differential_error_unsupported");

        let mut moved = slot.clone();
        if let SlotValue::Measured(m) = &mut moved.value {
            m.representation.version = 2;
        }
        assert_eq!(
            measured_likelihood(&moved, &levels(), &CONTEXT).unwrap_err().code,
            "representation_mapping_required"
        );
    }

    #[test]
    fn closed_routes_and_unreachable_levels_carry_registered_codes_and_remedies() {
        for refusal in [
            interval_refusal(),
            calibration_refusal(),
            level_unreachable_refusal(0.9, 0.78, "raise", "no_observation_crosses"),
            zero_probability_evidence_refusal(),
        ] {
            assert!(refusal.validate().is_ok(), "{refusal:?}");
            assert!(refusal.remedy.is_some());
        }
        let unreachable = level_unreachable_refusal(0.9, 0.78, "raise", "no_observation_crosses");
        assert_eq!(unreachable.code, "evidence_cannot_reach_level");
        assert_eq!(unreachable.expected.as_deref(), Some("0.9"));
        assert_eq!(unreachable.supplied.as_deref(), Some("0.78"));
        assert_eq!(interval_refusal().code, "cell_not_licensed");
        assert_eq!(calibration_refusal().detail, "observation_query.calibrated_probability_closed");
    }

    #[test]
    fn the_inference_basis_is_declared_only_when_every_mechanism_was_written_down() {
        assert_eq!(
            InferenceBasis::from_origins(&[MechanismOriginKind::Declared; 2]),
            InferenceBasis::DeclaredModel
        );
        assert_eq!(
            InferenceBasis::from_origins(&[
                MechanismOriginKind::Declared,
                MechanismOriginKind::Fitted
            ]),
            InferenceBasis::SampleFittedPointOnly
        );
        assert_eq!(InferenceBasis::from_origins(&[]), InferenceBasis::DeclaredModel);
    }
}
