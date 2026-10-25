//! Typed refusals of the scientific bridge: a registered reason code, a
//! `scientific_bridge.*` detail and the declaration or evidence that would
//! license the request.
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

fn not_licensed(
    detail: &str,
    expected: impl Into<String>,
    supplied: impl Into<String>,
    remedy: &'static str,
) -> BridgeRefusal {
    ExternalRefusal::new(
        antecedent_core::reason_code!("bridge_mapping_not_licensed"),
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
        "scientific_bridge.blank_field",
        "name every identity and statement, in at most the declared number of bytes",
    )
    .with_offending(field)
}

pub(super) fn invalid_quantity(which: &str) -> BridgeRefusal {
    invalid(
        "scientific_bridge.invalid_quantity",
        "name the variable, units, population, regime, functional and transform of each quantity, with canonical conditioning",
    )
    .with_offending(which)
}

pub(super) fn too_many_assumptions() -> BridgeRefusal {
    invalid(
        "scientific_bridge.too_many_assumptions",
        "declare at most the bounded number of assumptions",
    )
    .with_offending("assumptions")
}

pub(super) fn duplicate_assumption(id: &str) -> BridgeRefusal {
    invalid(
        "scientific_bridge.duplicate_assumption",
        "give every declared assumption its own identity",
    )
    .with_offending(id)
}

pub(super) fn endpoint_classes(
    kind: &str,
    source: QuantityClass,
    target: QuantityClass,
) -> BridgeRefusal {
    invalid("scientific_bridge.endpoint_classes", "connect the class pair the bridge kind relates")
        .with_offending(kind)
        .with_supplied(format!("{}->{}", source.as_str(), target.as_str()))
}

pub(super) fn mapping_not_allowed(mapping: &str, kind: &str) -> BridgeRefusal {
    invalid(
        "scientific_bridge.mapping_not_allowed",
        "carry the value by a mapping the bridge kind allows",
    )
    .with_offending(mapping)
    .with_supplied(kind)
}

pub(super) fn malformed_mapping(field: &str, remedy: &'static str) -> BridgeRefusal {
    invalid("scientific_bridge.malformed_mapping", remedy).with_offending(field)
}

pub(super) fn non_finite_value(field: &str, remedy: &'static str) -> BridgeRefusal {
    invalid("scientific_bridge.non_finite_value", remedy).with_offending(field)
}

pub(super) fn provenance_failed() -> BridgeRefusal {
    invalid("scientific_bridge.provenance", "derive a claim whose identity differs from its source")
}

pub(super) fn mapping_uncertainty_invalid(field: &str, remedy: &'static str) -> BridgeRefusal {
    invalid("scientific_bridge.mapping_uncertainty_invalid", remedy).with_offending(field)
}

pub(super) fn source_dependence_invalid(field: &str, remedy: &'static str) -> BridgeRefusal {
    invalid("scientific_bridge.source_dependence_invalid", remedy).with_offending(field)
}

/// A source moment (mean) that is not a finite number.
pub(super) fn source_moment_not_finite(field: &str, remedy: &'static str) -> BridgeRefusal {
    invalid("scientific_bridge.source_moment_not_finite", remedy).with_offending(field)
}

/// A propagated value or map image that overflowed the finite range.
pub(super) fn propagation_overflow(field: &str, remedy: &'static str) -> BridgeRefusal {
    invalid("scientific_bridge.propagation_overflow", remedy).with_offending(field)
}

/// A bridge whose dose or spend units disagree between its sides.
pub(super) fn estimand_units_mismatch(expected: String, supplied: String) -> BridgeRefusal {
    mismatch("scientific_bridge.estimand_units_mismatch", expected, supplied)
        .with_offending("estimand_bridge")
        .with_remedy(
            "state the experiment dose, the target dose and the spend difference in one unit identity",
        )
}

/// A spend difference that is not the target dose minus the experiment dose.
pub(super) fn estimand_spend_inconsistent(expected: String, supplied: String) -> BridgeRefusal {
    mismatch("scientific_bridge.estimand_spend_inconsistent", expected, supplied)
        .with_offending("spend_delta")
        .with_remedy(
            "state a spend difference equal to the target dose minus the experiment dose, or correct the doses",
        )
}

pub(super) fn estimand_bridge_incomplete(field: &str, remedy: &'static str) -> BridgeRefusal {
    invalid("scientific_bridge.estimand_bridge_incomplete", remedy).with_offending(field)
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
            "scientific_bridge.stated_preference_as_realized_behavior",
            source,
            "declare the behavioral linkage by which the stated preference determines the realized choice (no hypothetical bias, or a declared calibration of it) and name its evidence",
        )
        .with_expected(target.as_str()),
        (C::PredictiveQuantity | C::AssociationalQuantity, C::CausalQuantity) => substitution(
            "scientific_bridge.association_as_intervention",
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
        "scientific_bridge.proxy_as_construct",
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
        "scientific_bridge.scientific_endpoint_as_utility",
        QuantityClass::ScientificEndpoint,
        "declare the utility function that maps the endpoint to decision utility (its identity and form) and bridge through it; an effect on its natural scale is not a utility",
    )
    .with_expected(QuantityClass::DecisionUtility.as_str())
}

/// A predictive importance offered as a causal effect. Closed: no assumption
/// turns the importance of a feature in a predictive model into an effect.
pub(super) fn predictive_importance_as_causal_effect() -> BridgeRefusal {
    substitution(
        "scientific_bridge.predictive_importance_as_causal_effect",
        QuantityClass::PredictiveImportance,
        "ask the causal question directly: identify the effect of the feature under a declared graph and estimate it; the importance of a feature in a predictive model is a property of the predictor, not of the mechanism",
    )
    .with_expected(QuantityClass::CausalQuantity.as_str())
}

/// A model score offered as a probability. Closed in this release.
pub(super) fn model_score_as_probability() -> BridgeRefusal {
    substitution(
        "scientific_bridge.model_score_as_probability",
        QuantityClass::ModelScore,
        "a calibrated predictive probability needs a held-out calibration of the score; no calibration bridge is licensed for a score, so use the model's calibrated predictive law where it supplies one",
    )
    .with_expected(QuantityClass::PredictiveQuantity.as_str())
}

/// A value of one kind requested as another.
#[must_use]
pub fn wrong_value_kind(requested: ValueKind, held: ValueKind) -> BridgeRefusal {
    ExternalRefusal::new(
        antecedent_core::reason_code!("bridge_substitution_refused"),
        STAGE_BIND,
        "scientific_bridge.value_kind_substitution",
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
        "scientific_bridge.mapping_uncertainty_missing",
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
        "scientific_bridge.source_dependence_missing",
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
        "scientific_bridge.source_variance_invalid",
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
        "scientific_bridge.experimental_prior_without_estimand_bridge",
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
        "scientific_bridge.assumption_missing",
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
        "scientific_bridge.assumption_evidence_missing",
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
        "scientific_bridge.transport_premise_missing",
    )
    .with_offending("transportability_premise")
    .with_expected(premise)
    .with_remedy(
        "declare the transportability premise (the relation the bridge rests on is the same in the claim's population and regime as in the one it was established for) with its identity and evidence, or establish the bridge in the claim's own population",
    )
}

// -- binding mismatches (bridge_binding_mismatch) ---------------------------------------------------------

pub(super) fn transport_premise_mismatch(expected: String) -> BridgeRefusal {
    mismatch("scientific_bridge.transport_premise_mismatch", expected, "another transport")
        .with_offending("transportability_premise")
}

pub(super) fn estimand_bridge_mismatch(expected: String, supplied: String) -> BridgeRefusal {
    mismatch("scientific_bridge.estimand_bridge_mismatch", expected, supplied)
        .with_offending("estimand_bridge")
        .with_remedy(
            "name the experiment and the target with the same population and exposure regime as the contract's evidence scope and target quantity; time window, dose, duration, recency and spend stay on the bridge",
        )
}

pub(super) fn transport_premise_unused() -> BridgeRefusal {
    mismatch(
        "scientific_bridge.transport_premise_unused",
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
    mismatch("scientific_bridge.measurement_model_mismatch", expected, supplied)
        .with_offending(which)
}

pub(super) fn model_identity_mismatch(
    expected: impl Into<String>,
    supplied: impl Into<String>,
) -> BridgeRefusal {
    mismatch("scientific_bridge.model_identity_mismatch", expected, supplied)
}

pub(super) fn mapping_scope_mismatch(
    expected: impl Into<String>,
    supplied: impl Into<String>,
) -> BridgeRefusal {
    mismatch("scientific_bridge.mapping_scope_mismatch", expected, supplied)
}

pub(super) fn population_mismatch(
    expected: impl Into<String>,
    supplied: impl Into<String>,
) -> BridgeRefusal {
    mismatch("scientific_bridge.population_mismatch", expected, supplied)
}

pub(super) fn regime_mismatch(
    expected: impl Into<String>,
    supplied: impl Into<String>,
) -> BridgeRefusal {
    mismatch("scientific_bridge.regime_mismatch", expected, supplied)
}

pub(super) fn horizon_mismatch(
    expected: impl Into<String>,
    supplied: impl Into<String>,
) -> BridgeRefusal {
    mismatch("scientific_bridge.horizon_mismatch", expected, supplied)
}

pub(super) fn conditioning_mismatch(
    expected: impl Into<String>,
    supplied: impl Into<String>,
) -> BridgeRefusal {
    mismatch("scientific_bridge.conditioning_mismatch", expected, supplied)
}

pub(super) fn quantity_mismatch(
    dimension: &str,
    expected: impl Into<String>,
    supplied: impl Into<String>,
) -> BridgeRefusal {
    mismatch("scientific_bridge.quantity_mismatch", expected, supplied).with_offending(dimension)
}

pub(super) fn units_mismatch(
    expected: impl Into<String>,
    supplied: impl Into<String>,
) -> BridgeRefusal {
    mismatch("scientific_bridge.units_mismatch", expected, supplied)
}

// -- values outside what the mapping licenses (cell_not_licensed, bridge_mapping_not_licensed) ------------

/// A value kind the route does not carry: a law or a posterior on the point route, an
/// identified set or a joint law on either.
pub(super) fn value_kind_not_carried(kind: ValueKind) -> BridgeRefusal {
    ExternalRefusal::new(
        antecedent_core::reason_code!("cell_not_licensed"),
        STAGE_BIND,
        "scientific_bridge.value_kind_not_carried",
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
        antecedent_core::reason_code!("cell_not_licensed"),
        STAGE_BIND,
        "scientific_bridge.source_not_point",
    )
    .with_supplied(supplied)
    .with_expected("point")
    .with_remedy(
        "bridge a claim that carries one finite point value; refusals, bounds and incomplete claims have none",
    )
}

/// An interval, posterior or joint law requested of a bridged claim.
#[must_use]
pub fn interval_not_licensed() -> BridgeRefusal {
    ExternalRefusal::new(
        antecedent_core::reason_code!("cell_not_licensed"),
        STAGE_BIND,
        "scientific_bridge.interval_not_licensed",
    )
    .with_remedy(
        "a bridged claim carries one mean and the contract layer states no uncertainty; name a level and ask for the interval of the bridged value, which needs the source claim's declared standard error and is published only inside a measured design. A posterior and a joint law have no executed bridge",
    )
}

pub(super) fn validation_trials_too_few(declared: u32, supplied: usize) -> BridgeRefusal {
    not_licensed(
        "scientific_bridge.validation_trials_too_few",
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
    not_licensed("scientific_bridge.degenerate_validation", expected, supplied, remedy)
}

pub(super) fn validation_r_squared_too_low(declared: f64, supplied: f64) -> BridgeRefusal {
    not_licensed(
        "scientific_bridge.validation_r_squared_too_low",
        declared.to_string(),
        supplied.to_string(),
        "the surrogate does not explain enough of the trial-level terminal effect; validate a better surrogate, or declare the coefficient of determination the evidence supports",
    )
}

pub(super) fn outside_validated_range(low: f64, high: f64, value: f64) -> BridgeRefusal {
    not_licensed(
        "scientific_bridge.outside_validated_range",
        format!("[{low}, {high}]"),
        value.to_string(),
        "the relationship is applied only within the surrogate effects the validation trials covered; extrapolation is not licensed, so validate trials that cover this effect",
    )
}

pub(super) fn measurement_route_not_supported(expected: &str, supplied: &str) -> BridgeRefusal {
    not_licensed(
        "scientific_bridge.measurement_route_not_supported",
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
    not_licensed(
        "scientific_bridge.outside_attainable_range",
        expected,
        supplied.to_string(),
        remedy,
    )
}

pub(super) fn nonlinear_mean_not_expectation(map: &str) -> BridgeRefusal {
    not_licensed(
        "scientific_bridge.nonlinear_mean_not_expectation",
        "E[g(X)]",
        format!("g(E[X]) via {map}"),
        "the mean of a nonlinear map is not the map of the mean; supply a second moment for a square, or a normal law for an exponential, and read the expectation of the map",
    )
    .with_offending(map)
}

pub(super) fn nonlinear_law_insufficient(supplied: &str) -> BridgeRefusal {
    not_licensed(
        "scientific_bridge.nonlinear_law_insufficient",
        "normal",
        supplied,
        "an exponential expectation is licensed for a normal law; a mean and a variance do not determine it",
    )
    .with_offending("exponential")
}

pub(super) fn uncertain_mapping_not_affine(supplied: &str) -> BridgeRefusal {
    not_licensed(
        "scientific_bridge.uncertain_mapping_not_affine",
        "identity or affine",
        supplied,
        "this propagator is the delta method of a declared affine map; a trial-level line or a measurement correction keeps its own measured interval",
    )
}

pub(super) fn dependence_not_a_covariance(supplied: String) -> BridgeRefusal {
    not_licensed(
        "scientific_bridge.dependence_not_a_covariance",
        "covariance within the slope and source variances",
        supplied,
        "state a covariance whose square is at most the product of the slope variance and the source variance",
    )
}

pub(super) fn functional_not_supported(supplied: String) -> BridgeRefusal {
    not_licensed(
        "scientific_bridge.functional_not_supported",
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
        "scientific_bridge.differential_error",
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
        "scientific_bridge.youden_too_small",
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
        antecedent_core::reason_code!("cell_not_licensed"),
        STAGE_BIND,
        "scientific_bridge.source_not_draws",
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
        "scientific_bridge.draws_meaning_mismatch",
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
        "scientific_bridge.draws_not_aligned",
    )
    .with_remedy(
        "supply the draws as columns of one aligned joint draw table; independently sampled marginals have no row pairing, so the bridged draws could not keep one",
    )
}

/// Draws that cannot be bridged: empty, non-finite, or without the spread a
/// declared covariance needs.
#[must_use]
pub fn draws_unusable(field: &str, remedy: &'static str) -> BridgeRefusal {
    invalid("scientific_bridge.draws_unusable", remedy).with_offending(field)
}

/// A draw outside the range the mapping is licensed for. The first offending
/// draw is named.
#[must_use]
pub fn draws_outside_domain(count: usize, total: usize, first: f64) -> BridgeRefusal {
    ExternalRefusal::new(
        antecedent_core::reason_code!("bridge_mapping_not_licensed"),
        STAGE_BIND,
        "scientific_bridge.draws_outside_domain",
    )
    .with_offending(format!("{count} of {total} draws; first {first}"))
    .with_remedy(
        "every draw must lie inside the range the mapping is licensed for (the validated surrogate range, the attainable recorded prevalence); the bridge neither truncates nor extrapolates a law",
    )
}

/// A mapping that has no propagation through a chain stage.
#[must_use]
pub fn chain_stage_not_affine(supplied: &str) -> BridgeRefusal {
    ExternalRefusal::new(
        antecedent_core::reason_code!("route_not_supported"),
        STAGE_BIND,
        "scientific_bridge.chain_stage_not_affine",
    )
    .with_expected("identity, affine or trial-level mapping of a posterior")
    .with_supplied(supplied)
    .with_remedy(
        "express the bridge as a measurement-mapping stage only where it is an affine map of its inputs; bridge a predictive law or a measurement correction by deriving the bridged draws and supplying them to the chain as a source",
    )
}
