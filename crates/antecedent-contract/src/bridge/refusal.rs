//! Typed refusals of the scientific bridge: a registered reason code, a
//! `scientific_bridge.*` detail and the declaration or evidence that would
//! support the request.
//!
//! Every detail literal is written here, beside the reason code it is emitted
//! under, as one named constructor per refusal. The rest of the module tree
//! calls the constructors and never spells a detail.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::kind::{AssumptionKind, QuantityClass, ValueKind};
use crate::ExternalRefusal;

/// A refusal of a bridge request.
pub type BridgeRefusal = ExternalRefusal;

const STAGE_DECLARE: &str = "declare";
const STAGE_BIND: &str = "bind";

// -- helpers that fix the code and stage of a family ------------------------------------------------

fn invalid(detail: &str, remedy: &'static str) -> BridgeRefusal {
    ExternalRefusal::new(
        antecedent_core::reason_code!("scientific_bridge_invalid"),
        STAGE_DECLARE,
        detail,
    )
    .with_remedy(remedy)
}

fn substitution(detail: &str, source: QuantityClass, remedy: &'static str) -> BridgeRefusal {
    ExternalRefusal::new(
        antecedent_core::reason_code!("bridge_substitution_refused"),
        STAGE_BIND,
        detail,
    )
    .with_supplied(source.as_str())
    .with_remedy(remedy)
}

fn mismatch(
    detail: &str,
    expected: impl Into<String>,
    supplied: impl Into<String>,
) -> BridgeRefusal {
    ExternalRefusal::new(antecedent_core::reason_code!("bridge_binding_mismatch"), STAGE_BIND, detail)
        .with_expected(expected)
        .with_supplied(supplied)
        .with_remedy(
            "bridge a claim about the same population and conditioning as its target, with the bridge's own evidence established for the claim's population and regime or carried by a declared transportability premise",
        )
}

fn not_supported(
    detail: &str,
    expected: impl Into<String>,
    supplied: impl Into<String>,
    remedy: &'static str,
) -> BridgeRefusal {
    ExternalRefusal::new(
        antecedent_core::reason_code!("bridge_mapping_not_supported"),
        STAGE_BIND,
        detail,
    )
    .with_expected(expected)
    .with_supplied(supplied)
    .with_remedy(remedy)
}

// -- malformed declarations (scientific_bridge_invalid) -----------------------------------------------

pub(super) fn blank_field(field: &str) -> BridgeRefusal {
    invalid(
        antecedent_core::reason_details::scientific_bridge::BLANK_FIELD,
        "name every identity and statement, in at most the declared number of bytes",
    )
    .with_offending(field)
}

pub(super) fn invalid_quantity(which: &str) -> BridgeRefusal {
    invalid(
        antecedent_core::reason_details::scientific_bridge::INVALID_QUANTITY,
        "name the variable, units, population, regime, functional and transform of each quantity, with canonical conditioning",
    )
    .with_offending(which)
}

pub(super) fn duplicate_assumption(id: &str) -> BridgeRefusal {
    invalid(
        antecedent_core::reason_details::scientific_bridge::DUPLICATE_ASSUMPTION,
        "give every declared assumption its own identity",
    )
    .with_offending(id)
}

pub(super) fn endpoint_classes(
    kind: &str,
    source: QuantityClass,
    target: QuantityClass,
) -> BridgeRefusal {
    invalid(
        antecedent_core::reason_details::scientific_bridge::ENDPOINT_CLASSES,
        "connect the class pair the bridge kind relates",
    )
    .with_offending(kind)
    .with_supplied(format!("{}->{}", source.as_str(), target.as_str()))
}

pub(super) fn mapping_not_allowed(mapping: &str, kind: &str) -> BridgeRefusal {
    invalid(
        antecedent_core::reason_details::scientific_bridge::MAPPING_NOT_ALLOWED,
        "carry the value by a mapping the bridge kind allows",
    )
    .with_offending(mapping)
    .with_supplied(kind)
}

pub(super) fn malformed_mapping(field: &str, remedy: &'static str) -> BridgeRefusal {
    invalid(antecedent_core::reason_details::scientific_bridge::MALFORMED_MAPPING, remedy)
        .with_offending(field)
}

pub(super) fn non_finite_value(field: &str, remedy: &'static str) -> BridgeRefusal {
    invalid(antecedent_core::reason_details::scientific_bridge::NON_FINITE_VALUE, remedy)
        .with_offending(field)
}

pub(super) fn provenance_failed() -> BridgeRefusal {
    invalid(
        antecedent_core::reason_details::scientific_bridge::PROVENANCE,
        "derive a claim whose identity differs from its source",
    )
}

pub(super) fn mapping_uncertainty_invalid(field: &str, remedy: &'static str) -> BridgeRefusal {
    invalid(antecedent_core::reason_details::scientific_bridge::MAPPING_UNCERTAINTY_INVALID, remedy)
        .with_offending(field)
}

pub(super) fn source_dependence_invalid(field: &str, remedy: &'static str) -> BridgeRefusal {
    invalid(antecedent_core::reason_details::scientific_bridge::SOURCE_DEPENDENCE_INVALID, remedy)
        .with_offending(field)
}

/// A source moment (mean) that is not a finite number.
pub(super) fn source_moment_not_finite(field: &str, remedy: &'static str) -> BridgeRefusal {
    invalid(antecedent_core::reason_details::scientific_bridge::SOURCE_MOMENT_NOT_FINITE, remedy)
        .with_offending(field)
}

/// A propagated value or map image that overflowed the finite range.
pub(super) fn propagation_overflow(field: &str, remedy: &'static str) -> BridgeRefusal {
    invalid(antecedent_core::reason_details::scientific_bridge::PROPAGATION_OVERFLOW, remedy)
        .with_offending(field)
}

/// A bridge whose dose or spend units disagree between its sides.
pub(super) fn estimand_units_mismatch(expected: String, supplied: String) -> BridgeRefusal {
    mismatch(
        antecedent_core::reason_details::scientific_bridge::ESTIMAND_UNITS_MISMATCH,
        expected,
        supplied,
    )
    .with_offending("estimand_bridge")
    .with_remedy(
        "state the experiment dose, the target dose and the spend difference in one unit identity",
    )
}

/// A spend difference that is not the target dose minus the experiment dose.
pub(super) fn estimand_spend_inconsistent(expected: String, supplied: String) -> BridgeRefusal {
    mismatch(antecedent_core::reason_details::scientific_bridge::ESTIMAND_SPEND_INCONSISTENT, expected, supplied)
        .with_offending("spend_delta")
        .with_remedy(
            "state a spend difference equal to the target dose minus the experiment dose, or correct the doses",
        )
}

pub(super) fn estimand_bridge_incomplete(field: &str, remedy: &'static str) -> BridgeRefusal {
    invalid(antecedent_core::reason_details::scientific_bridge::ESTIMAND_BRIDGE_INCOMPLETE, remedy)
        .with_offending(field)
}

// -- named substitutions (bridge_substitution_refused, proxy_offered_as_construct) ---------------------

/// The refusal for a bridge whose principal assumption is not declared.
///
/// The class pairs with a named substitution refuse under that name; any other
/// pair refuses as a missing assumption that names the assumption kind.
pub(super) fn missing_principal(
    source: QuantityClass,
    target: QuantityClass,
    principal: AssumptionKind,
) -> BridgeRefusal {
    use QuantityClass as C;
    match (source, target) {
        (C::MeasuredOutcome, C::LatentConstruct) => proxy_as_construct(),
        (C::StatedPreference, C::RealizedBehavior) => substitution(
            antecedent_core::reason_details::scientific_bridge::STATED_PREFERENCE_AS_REALIZED_BEHAVIOR,
            source,
            "declare the behavioral linkage by which the stated preference determines the realized choice (no hypothetical bias, or a declared calibration of it) and name its evidence",
        )
        .with_expected(target.as_str()),
        (C::PredictiveQuantity | C::AssociationalQuantity, C::CausalQuantity) => substitution(
            antecedent_core::reason_details::scientific_bridge::ASSOCIATION_AS_INTERVENTION,
            source,
            "declare and name the identification (adjustment set, exchangeability, positivity, consistency) under which the predictive quantity equals the causal quantity, or identify the causal quantity from the data",
        )
        .with_expected(target.as_str()),
        (C::ScientificEndpoint, C::DecisionUtility) => scientific_endpoint_as_utility(),
        _ => assumption_missing(principal),
    }
}

/// A measured quantity offered as the construct without a measurement model
/// (or with a mapping that leaves it unchanged).
pub(super) fn proxy_as_construct() -> BridgeRefusal {
    ExternalRefusal::new(
        antecedent_core::reason_code!("proxy_offered_as_construct"),
        STAGE_BIND,
        antecedent_core::reason_details::scientific_bridge::PROXY_AS_CONSTRUCT,
    )
    .with_supplied(QuantityClass::MeasuredOutcome.as_str())
    .with_expected(QuantityClass::LatentConstruct.as_str())
    .with_remedy(
        "state a measurement model that relates the measured quantity to the construct and bridge through its correction; a measured value is not the construct",
    )
}

/// A scientific endpoint offered as a utility without a utility mapping.
pub(super) fn scientific_endpoint_as_utility() -> BridgeRefusal {
    substitution(
        antecedent_core::reason_details::scientific_bridge::SCIENTIFIC_ENDPOINT_AS_UTILITY,
        QuantityClass::ScientificEndpoint,
        "declare the utility function that maps the endpoint to decision utility (its identity and form) and bridge through it; an effect on its natural scale is not a utility",
    )
    .with_expected(QuantityClass::DecisionUtility.as_str())
}

/// A predictive importance offered as a causal effect. Closed: no assumption
/// turns the importance of a feature in a predictive model into an effect.
pub(super) fn predictive_importance_as_causal_effect() -> BridgeRefusal {
    substitution(
        antecedent_core::reason_details::scientific_bridge::PREDICTIVE_IMPORTANCE_AS_CAUSAL_EFFECT,
        QuantityClass::PredictiveImportance,
        "ask the causal question directly: identify the effect of the feature under a declared graph and estimate it; the importance of a feature in a predictive model is a property of the predictor, not of the mechanism",
    )
    .with_expected(QuantityClass::CausalQuantity.as_str())
}

/// A model score offered as a probability. Not supported..
pub(super) fn model_score_as_probability() -> BridgeRefusal {
    substitution(
        antecedent_core::reason_details::scientific_bridge::MODEL_SCORE_AS_PROBABILITY,
        QuantityClass::ModelScore,
        "a calibrated predictive probability needs a held-out calibration of the score; no calibration bridge is supported for a score, so use the model's calibrated predictive law where it supplies one",
    )
    .with_expected(QuantityClass::PredictiveQuantity.as_str())
}

/// A value of one kind requested as another.
#[must_use]
pub fn wrong_value_kind(requested: ValueKind, held: ValueKind) -> BridgeRefusal {
    ExternalRefusal::new(
        antecedent_core::reason_code!("bridge_substitution_refused"),
        STAGE_BIND,
        antecedent_core::reason_details::scientific_bridge::VALUE_KIND_SUBSTITUTION,
    )
    .with_supplied(held.as_str())
    .with_expected(requested.as_str())
    .with_remedy(
        "a mean, a predictive law, a posterior, an identified set and a joint outcome law are not interchangeable; supply the requested object from its own route",
    )
}

// -- missing assumptions (bridge_assumption_missing) ---------------------------------------------------

/// Mapping uncertainty was required to propagate and was not declared.
pub(super) fn mapping_uncertainty_missing() -> BridgeRefusal {
    ExternalRefusal::new(
        antecedent_core::reason_code!("bridge_assumption_missing"),
        STAGE_BIND,
        antecedent_core::reason_details::scientific_bridge::MAPPING_UNCERTAINTY_MISSING,
    )
    .with_offending("mapping_uncertainty")
    .with_remedy(
        "declare the variance of the mapping's intercept and slope; an undeclared map is not an exact map",
    )
}

/// Source and mapping dependence was required to propagate and was not declared.
pub(super) fn source_dependence_missing() -> BridgeRefusal {
    ExternalRefusal::new(
        antecedent_core::reason_code!("sampling_dependence_unknown"),
        STAGE_BIND,
        antecedent_core::reason_details::scientific_bridge::SOURCE_DEPENDENCE_MISSING,
    )
    .with_offending("source_dependence")
    .with_remedy(
        "declare that the source estimate and the mapping are independent, or declare their covariance; disjoint identities do not establish it",
    )
}

/// A source variance that is not a finite non-negative number.
pub(super) fn source_variance_invalid(supplied: String) -> BridgeRefusal {
    ExternalRefusal::new(
        antecedent_core::reason_code!("scientific_bridge_invalid"),
        STAGE_DECLARE,
        antecedent_core::reason_details::scientific_bridge::SOURCE_VARIANCE_INVALID,
    )
    .with_offending("source_variance")
    .with_supplied(supplied)
    .with_remedy("state a finite non-negative variance of the source estimate")
}

/// An experimental prior offered with no estimand bridge.
pub(super) fn experimental_prior_without_estimand_bridge() -> BridgeRefusal {
    ExternalRefusal::new(
        antecedent_core::reason_code!("bridge_assumption_missing"),
        STAGE_BIND,
        antecedent_core::reason_details::scientific_bridge::EXPERIMENTAL_PRIOR_WITHOUT_ESTIMAND_BRIDGE,
    )
    .with_offending("estimand_bridge")
    .with_remedy(
        "translate the experiment's population, time window, dose, duration and exposure regime into the target, and state the recency and spend differences, before importing its prior",
    )
}

/// A required assumption that is not declared.
pub(super) fn assumption_missing(kind: AssumptionKind) -> BridgeRefusal {
    ExternalRefusal::new(
        antecedent_core::reason_code!("bridge_assumption_missing"),
        STAGE_BIND,
        antecedent_core::reason_details::scientific_bridge::ASSUMPTION_MISSING,
    )
    .with_offending(kind.as_str())
    .with_remedy(
        "declare the assumption the bridge rests on, with an identity, and name the evidence it rests on",
    )
}

/// A declared assumption that must name its evidence and does not.
pub(super) fn assumption_evidence_missing(id: &str, kind: AssumptionKind) -> BridgeRefusal {
    ExternalRefusal::new(
        antecedent_core::reason_code!("bridge_assumption_missing"),
        STAGE_BIND,
        antecedent_core::reason_details::scientific_bridge::ASSUMPTION_EVIDENCE_MISSING,
    )
    .with_offending(id)
    .with_expected(kind.as_str())
    .with_remedy(
        "name the identity of the validation study, measurement model, identification or utility function the assumption rests on",
    )
}

/// A bridge whose evidence was established elsewhere than the claim's
/// population or regime, with no transportability premise declared.
pub(super) fn transport_premise_missing(premise: String) -> BridgeRefusal {
    ExternalRefusal::new(
        antecedent_core::reason_code!("bridge_assumption_missing"),
        STAGE_BIND,
        antecedent_core::reason_details::scientific_bridge::TRANSPORT_PREMISE_MISSING,
    )
    .with_offending("transportability_premise")
    .with_expected(premise)
    .with_remedy(
        "declare the transportability premise (the relation the bridge rests on is the same in the claim's population and regime as in the one it was established for) with its identity and evidence, or establish the bridge in the claim's own population",
    )
}

// -- binding mismatches (bridge_binding_mismatch) ---------------------------------------------------------

pub(super) fn transport_premise_mismatch(expected: String) -> BridgeRefusal {
    mismatch(
        antecedent_core::reason_details::scientific_bridge::TRANSPORT_PREMISE_MISMATCH,
        expected,
        "another transport",
    )
    .with_offending("transportability_premise")
}

pub(super) fn estimand_bridge_mismatch(expected: String, supplied: String) -> BridgeRefusal {
    mismatch(antecedent_core::reason_details::scientific_bridge::ESTIMAND_BRIDGE_MISMATCH, expected, supplied)
        .with_offending("estimand_bridge")
        .with_remedy(
            "name the experiment and the target with the same population and exposure regime as the contract's evidence scope and target quantity; time window, dose, duration, recency and spend stay on the bridge",
        )
}

pub(super) fn transport_premise_unused() -> BridgeRefusal {
    mismatch(
        antecedent_core::reason_details::scientific_bridge::TRANSPORT_PREMISE_UNUSED,
        "no transportability premise: the claim is about the scope the bridge was established for",
        "a transportability premise",
    )
    .with_offending("transportability_premise")
}

pub(super) fn measurement_model_mismatch(
    which: &str,
    expected: impl Into<String>,
    supplied: impl Into<String>,
) -> BridgeRefusal {
    mismatch(
        antecedent_core::reason_details::scientific_bridge::MEASUREMENT_MODEL_MISMATCH,
        expected,
        supplied,
    )
    .with_offending(which)
}

pub(super) fn model_identity_mismatch(
    expected: impl Into<String>,
    supplied: impl Into<String>,
) -> BridgeRefusal {
    mismatch(
        antecedent_core::reason_details::scientific_bridge::MODEL_IDENTITY_MISMATCH,
        expected,
        supplied,
    )
}

pub(super) fn mapping_scope_mismatch(
    expected: impl Into<String>,
    supplied: impl Into<String>,
) -> BridgeRefusal {
    mismatch(
        antecedent_core::reason_details::scientific_bridge::MAPPING_SCOPE_MISMATCH,
        expected,
        supplied,
    )
}

pub(super) fn population_mismatch(
    expected: impl Into<String>,
    supplied: impl Into<String>,
) -> BridgeRefusal {
    mismatch(
        antecedent_core::reason_details::scientific_bridge::POPULATION_MISMATCH,
        expected,
        supplied,
    )
}

pub(super) fn regime_mismatch(
    expected: impl Into<String>,
    supplied: impl Into<String>,
) -> BridgeRefusal {
    mismatch(
        antecedent_core::reason_details::scientific_bridge::REGIME_MISMATCH,
        expected,
        supplied,
    )
}

pub(super) fn horizon_mismatch(
    expected: impl Into<String>,
    supplied: impl Into<String>,
) -> BridgeRefusal {
    mismatch(
        antecedent_core::reason_details::scientific_bridge::HORIZON_MISMATCH,
        expected,
        supplied,
    )
}

pub(super) fn conditioning_mismatch(
    expected: impl Into<String>,
    supplied: impl Into<String>,
) -> BridgeRefusal {
    mismatch(
        antecedent_core::reason_details::scientific_bridge::CONDITIONING_MISMATCH,
        expected,
        supplied,
    )
}

pub(super) fn quantity_mismatch(
    dimension: &str,
    expected: impl Into<String>,
    supplied: impl Into<String>,
) -> BridgeRefusal {
    mismatch(
        antecedent_core::reason_details::scientific_bridge::QUANTITY_MISMATCH,
        expected,
        supplied,
    )
    .with_offending(dimension)
}

pub(super) fn units_mismatch(
    expected: impl Into<String>,
    supplied: impl Into<String>,
) -> BridgeRefusal {
    mismatch(antecedent_core::reason_details::scientific_bridge::UNITS_MISMATCH, expected, supplied)
}

// -- values outside what the mapping supports (bridge_mapping_not_supported) ------------

/// A value kind the route does not carry: a law or a posterior on the point route, an
/// identified set or a joint law on either.
pub(super) fn value_kind_not_carried(kind: ValueKind) -> BridgeRefusal {
    ExternalRefusal::new(
        antecedent_core::reason_code!("bridge_mapping_not_supported"),
        STAGE_BIND,
        antecedent_core::reason_details::scientific_bridge::VALUE_KIND_NOT_CARRIED,
    )
    .with_supplied(kind.as_str())
    .with_expected(ValueKind::Mean.as_str())
    .with_remedy(
        "a point mean takes the point route and a posterior or predictive law takes the draws route; an identified set or joint law has no executed bridge",
    )
}

/// The source claim is not a finite point value.
pub(super) fn source_not_point(supplied: &str) -> BridgeRefusal {
    ExternalRefusal::new(
        antecedent_core::reason_code!("bridge_mapping_not_supported"),
        STAGE_BIND,
        antecedent_core::reason_details::scientific_bridge::SOURCE_NOT_POINT,
    )
    .with_supplied(supplied)
    .with_expected("point")
    .with_remedy(
        "bridge a claim that carries one finite point value; refusals, bounds and incomplete claims have none",
    )
}

/// An interval, posterior or joint law requested of a bridged claim.
#[must_use]
pub fn interval_not_supported() -> BridgeRefusal {
    ExternalRefusal::new(
        antecedent_core::reason_code!("route_not_supported"),
        STAGE_BIND,
        antecedent_core::reason_details::scientific_bridge::INTERVAL_NOT_SUPPORTED,
    )
    .with_remedy(
        "a bridged claim carries one mean and the contract layer states no uncertainty; name a level and ask for the interval of the bridged value, which needs the source claim's declared standard error and is published only inside a measured design. A posterior and a joint law have no executed bridge",
    )
}

pub(super) fn validation_trials_too_few(declared: u32, supplied: usize) -> BridgeRefusal {
    not_supported(
        antecedent_core::reason_details::scientific_bridge::VALIDATION_TRIALS_TOO_FEW,
        declared.to_string(),
        supplied.to_string(),
        "validate the surrogate in at least the declared number of trials, or declare the number of trials the evidence supports",
    )
}

pub(super) fn degenerate_validation(
    expected: &str,
    supplied: &str,
    remedy: &'static str,
) -> BridgeRefusal {
    not_supported(
        antecedent_core::reason_details::scientific_bridge::DEGENERATE_VALIDATION,
        expected,
        supplied,
        remedy,
    )
}

pub(super) fn validation_r_squared_too_low(declared: f64, supplied: f64) -> BridgeRefusal {
    not_supported(
        antecedent_core::reason_details::scientific_bridge::VALIDATION_R_SQUARED_TOO_LOW,
        declared.to_string(),
        supplied.to_string(),
        "the surrogate does not explain enough of the trial-level terminal effect; validate a better surrogate, or declare the coefficient of determination the evidence supports",
    )
}

pub(super) fn outside_validated_range(low: f64, high: f64, value: f64) -> BridgeRefusal {
    not_supported(
        antecedent_core::reason_details::scientific_bridge::OUTSIDE_VALIDATED_RANGE,
        format!("[{low}, {high}]"),
        value.to_string(),
        "the relationship is applied only within the surrogate effects the validation trials covered; extrapolation is not supported, so validate trials that cover this effect",
    )
}

pub(super) fn measurement_route_not_supported(expected: &str, supplied: &str) -> BridgeRefusal {
    not_supported(
        antecedent_core::reason_details::scientific_bridge::MEASUREMENT_ROUTE_NOT_SUPPORTED,
        expected,
        supplied,
        "only the binary-misclassification correction of a prevalence or risk is bridged; another measurement route has its own corrected-functional record",
    )
}

pub(super) fn outside_attainable_range(
    expected: String,
    supplied: f64,
    remedy: &'static str,
) -> BridgeRefusal {
    not_supported(
        antecedent_core::reason_details::scientific_bridge::OUTSIDE_ATTAINABLE_RANGE,
        expected,
        supplied.to_string(),
        remedy,
    )
}

pub(super) fn nonlinear_mean_not_expectation(map: &str) -> BridgeRefusal {
    not_supported(
        antecedent_core::reason_details::scientific_bridge::NONLINEAR_MEAN_NOT_EXPECTATION,
        "E[g(X)]",
        format!("g(E[X]) via {map}"),
        "the mean of a nonlinear map is not the map of the mean; supply a second moment for a square, or a normal law for an exponential, and read the expectation of the map",
    )
    .with_offending(map)
}

pub(super) fn nonlinear_law_insufficient(supplied: &str) -> BridgeRefusal {
    not_supported(
        antecedent_core::reason_details::scientific_bridge::NONLINEAR_LAW_INSUFFICIENT,
        "normal",
        supplied,
        "an exponential expectation is supported for a normal law; a mean and a variance do not determine it",
    )
    .with_offending("exponential")
}

pub(super) fn uncertain_mapping_not_affine(supplied: &str) -> BridgeRefusal {
    not_supported(
        antecedent_core::reason_details::scientific_bridge::UNCERTAIN_MAPPING_NOT_AFFINE,
        "identity or affine",
        supplied,
        "this propagator is the delta method of a declared affine map; a trial-level line or a measurement correction keeps its own measured interval",
    )
}

pub(super) fn dependence_not_a_covariance(supplied: String) -> BridgeRefusal {
    not_supported(
        antecedent_core::reason_details::scientific_bridge::DEPENDENCE_NOT_A_COVARIANCE,
        "covariance within the slope and source variances",
        supplied,
        "state a covariance whose square is at most the product of the slope variance and the source variance",
    )
}

pub(super) fn functional_not_supported(supplied: String) -> BridgeRefusal {
    not_supported(
        antecedent_core::reason_details::scientific_bridge::FUNCTIONAL_NOT_SUPPORTED,
        "prevalence or risk on the identity scale",
        supplied,
        "the binary-misclassification correction inverts a prevalence or risk on the probability scale; state the quantity as one",
    )
}

/// A differential measurement model has no single pair of rates for a claim
/// that names no arm.
pub(super) fn differential_error() -> BridgeRefusal {
    ExternalRefusal::new(
        antecedent_core::reason_code!("differential_error_not_pooled"),
        STAGE_BIND,
        antecedent_core::reason_details::scientific_bridge::DIFFERENTIAL_ERROR,
    )
    .with_remedy(
        "a bridge through differential error needs the arm or population of the claim; bridge through a non-differential model, or analyse the arms with their own rates",
    )
}

/// Youden's index at or below the model's declared minimum.
pub(super) fn youden_too_small(youden: f64, min_youden: f64) -> BridgeRefusal {
    ExternalRefusal::new(
        antecedent_core::reason_code!("measurement_not_identified"),
        STAGE_BIND,
        antecedent_core::reason_details::scientific_bridge::YOUDEN_TOO_SMALL,
    )
    .with_supplied(youden.to_string())
    .with_expected(min_youden.to_string())
    .with_remedy(
        "the indicator is not informative enough to invert; a more accurate measurement or a larger declared minimum-evidence study is needed",
    )
}

// -- bridges that carry draws ---------------------------------------------------------------------------

/// A source offered to a distribution bridge that is not a claim about draws:
/// an associational claim, a refusal, bounds or an incomplete contract.
pub(super) fn source_not_draws(supplied: &str) -> BridgeRefusal {
    ExternalRefusal::new(
        antecedent_core::reason_code!("bridge_mapping_not_supported"),
        STAGE_BIND,
        antecedent_core::reason_details::scientific_bridge::SOURCE_NOT_DRAWS,
    )
    .with_supplied(supplied)
    .with_expected("posterior or predictive draws")
    .with_remedy(
        "bridge the aligned draws of a posterior or predictive law; an associational claim, a refusal, bounds and an incomplete claim carry none",
    )
}

/// The meaning of the supplied draws is not the value kind the contract declares.
pub(super) fn draws_meaning_mismatch(expected: ValueKind, supplied: &str) -> BridgeRefusal {
    ExternalRefusal::new(
        antecedent_core::reason_code!("distribution_meaning_mismatch"),
        STAGE_BIND,
        antecedent_core::reason_details::scientific_bridge::DRAWS_MEANING_MISMATCH,
    )
    .with_expected(expected.as_str())
    .with_supplied(supplied)
    .with_remedy(
        "declare the value kind the draws are: a posterior for posterior draws of a parameter or causal functional, a predictive law for posterior predictive draws; a sampling or bootstrap law, or an empirical law, is neither",
    )
}

/// Draws that are independent marginals have no row pairing to carry.
pub(super) fn draws_not_aligned() -> BridgeRefusal {
    ExternalRefusal::new(
        antecedent_core::reason_code!("joint_law_required"),
        STAGE_BIND,
        antecedent_core::reason_details::scientific_bridge::DRAWS_NOT_ALIGNED,
    )
    .with_remedy(
        "supply the draws as columns of one aligned joint draw table; independently sampled marginals have no row pairing, so the bridged draws could not keep one",
    )
}

/// Draws that cannot be bridged: empty, non-finite, or without the spread a
/// declared covariance needs.
#[must_use]
pub fn draws_unusable(field: &str, remedy: &'static str) -> BridgeRefusal {
    invalid(antecedent_core::reason_details::scientific_bridge::DRAWS_UNUSABLE, remedy)
        .with_offending(field)
}

/// A draw outside the range the mapping is supported for. The first offending
/// draw is named.
#[must_use]
pub fn draws_outside_domain(count: usize, total: usize, first: f64) -> BridgeRefusal {
    ExternalRefusal::new(
        antecedent_core::reason_code!("bridge_mapping_not_supported"),
        STAGE_BIND,
        antecedent_core::reason_details::scientific_bridge::DRAWS_OUTSIDE_DOMAIN,
    )
    .with_offending(format!("{count} of {total} draws; first {first}"))
    .with_remedy(
        "every draw must lie inside the range the mapping is supported for (the validated surrogate range, the attainable recorded prevalence); the bridge neither truncates nor extrapolates a law",
    )
}

/// A mapping that has no propagation through a chain stage.
#[must_use]
pub fn chain_stage_not_affine(supplied: &str) -> BridgeRefusal {
    ExternalRefusal::new(
        antecedent_core::reason_code!("route_not_supported"),
        STAGE_BIND,
        antecedent_core::reason_details::scientific_bridge::CHAIN_STAGE_NOT_AFFINE,
    )
    .with_expected("identity, affine or trial-level mapping of a posterior")
    .with_supplied(supplied)
    .with_remedy(
        "express the bridge as a measurement-mapping stage only where it is an affine map of its inputs; bridge a predictive law or a measurement correction by deriving the bridged draws and supplying them to the chain as a source",
    )
}
