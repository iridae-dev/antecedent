//! Typed refusals of the interface contract: a registered reason code, an
//! `interface_contract.*` detail and the declaration that would support the
//! request.
//!
//! Every detail literal is written here, beside the reason code it is emitted
//! under, as one named constructor per refusal. The rest of the module tree
//! calls the constructors and never spells a detail.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use crate::ExternalRefusal;

/// A refusal of an interface declaration, connection or composition.
pub type InterfaceRefusal = ExternalRefusal;

const STAGE_DECLARE: &str = "declare";
const STAGE_BIND: &str = "bind";

// -- helpers that fix the code and stage of a family ------------------------------------------------

fn invalid(detail: &str, remedy: &'static str) -> InterfaceRefusal {
    ExternalRefusal::new(
        antecedent_core::reason_code!("interface_contract_invalid"),
        STAGE_DECLARE,
        detail,
    )
    .with_remedy(remedy)
}

fn quantity_mismatch(
    detail: &str,
    expected: impl Into<String>,
    supplied: impl Into<String>,
    remedy: &'static str,
) -> InterfaceRefusal {
    ExternalRefusal::new(
        antecedent_core::reason_code!("interface_contract_quantity_mismatch"),
        STAGE_BIND,
        detail,
    )
    .with_expected(expected)
    .with_supplied(supplied)
    .with_remedy(remedy)
}

fn assumption_mismatch(
    detail: &str,
    expected: impl Into<String>,
    supplied: impl Into<String>,
    remedy: &'static str,
) -> InterfaceRefusal {
    ExternalRefusal::new(
        antecedent_core::reason_code!("interface_contract_assumption_mismatch"),
        STAGE_BIND,
        detail,
    )
    .with_expected(expected)
    .with_supplied(supplied)
    .with_remedy(remedy)
}

fn uncertainty_mismatch(
    detail: &str,
    expected: impl Into<String>,
    supplied: impl Into<String>,
    remedy: &'static str,
) -> InterfaceRefusal {
    ExternalRefusal::new(
        antecedent_core::reason_code!("interface_contract_uncertainty_mismatch"),
        STAGE_BIND,
        detail,
    )
    .with_expected(expected)
    .with_supplied(supplied)
    .with_remedy(remedy)
}

// -- malformed declarations (interface_contract_invalid) ---------------------------------------------

pub(super) fn blank_field(field: &str) -> InterfaceRefusal {
    invalid(
        antecedent_core::reason_details::interface_contract::BLANK_FIELD,
        "name every identity, in at most the declared number of bytes",
    )
    .with_offending(field)
}

pub(super) fn invalid_quantity(port: &str) -> InterfaceRefusal {
    invalid(
        antecedent_core::reason_details::interface_contract::INVALID_QUANTITY,
        "name the variable, units, population, regime, functional and transform of each quantity, with canonical conditioning",
    )
    .with_offending(port)
}

pub(super) fn duplicate_port(port: &str) -> InterfaceRefusal {
    invalid(
        antecedent_core::reason_details::interface_contract::DUPLICATE_PORT,
        "give every produced port, and every consumed port, its own name",
    )
    .with_offending(port)
}

pub(super) fn invalid_support(port: &str) -> InterfaceRefusal {
    invalid(
        antecedent_core::reason_details::interface_contract::INVALID_SUPPORT,
        "declare a support domain with finite ordered bounds or sorted, distinct, non-blank levels",
    )
    .with_offending(port)
}

pub(super) fn missing_operations(port: &str) -> InterfaceRefusal {
    invalid(
        antecedent_core::reason_details::interface_contract::MISSING_OPERATIONS,
        "declare at least one operation the port offers or requires",
    )
    .with_offending(port)
}

pub(super) fn uncertainty_meaning_required(port: &str) -> InterfaceRefusal {
    invalid(
        antecedent_core::reason_details::interface_contract::UNCERTAINTY_MEANING_REQUIRED,
        "state what the covariance, law, draws, interval or posterior means (posterior of a causal functional, estimator sampling, bootstrap, ...)",
    )
    .with_offending(port)
}

pub(super) fn uncertainty_meaning_unused(port: &str) -> InterfaceRefusal {
    invalid(
        antecedent_core::reason_details::interface_contract::UNCERTAINTY_MEANING_UNUSED,
        "state an uncertainty meaning only for a port that offers or requires a covariance, law, draws, interval or posterior",
    )
    .with_offending(port)
}

pub(super) fn posterior_without_posterior_meaning(port: &str) -> InterfaceRefusal {
    invalid(
        antecedent_core::reason_details::interface_contract::POSTERIOR_WITHOUT_POSTERIOR_MEANING,
        "a posterior operation carries a posterior meaning; declare the sampling or resampling distribution as draws or an interval",
    )
    .with_offending(port)
}

pub(super) fn point_only_with_uncertainty(port: &str) -> InterfaceRefusal {
    invalid(
        antecedent_core::reason_details::interface_contract::POINT_ONLY_WITH_UNCERTAINTY,
        "a point-only label cannot accompany a covariance, law, draws, interval or posterior; use none or not_measured, or drop the operation",
    )
    .with_offending(port)
}

pub(super) fn calibration_record_blank(port: &str) -> InterfaceRefusal {
    invalid(
        antecedent_core::reason_details::interface_contract::CALIBRATION_RECORD_BLANK,
        "a measured calibration label names the identity of its measured record",
    )
    .with_offending(port)
}

pub(super) fn calibration_without_interval(port: &str) -> InterfaceRefusal {
    invalid(
        antecedent_core::reason_details::interface_contract::CALIBRATION_WITHOUT_INTERVAL,
        "calibration is measured for an interval or a posterior; declare that operation or use another calibration label",
    )
    .with_offending(port)
}

pub(super) fn inherited_input_unknown(port: &str, input: &str) -> InterfaceRefusal {
    invalid(
        antecedent_core::reason_details::interface_contract::INHERITED_INPUT_UNKNOWN,
        "inherit calibration from a consumed port the contract declares",
    )
    .with_offending(port)
    .with_supplied(input)
}

pub(super) fn invalid_digest(field: &str) -> InterfaceRefusal {
    invalid(
        antecedent_core::reason_details::interface_contract::INVALID_DIGEST,
        "name a measurement model or evidence source by its 64-character lowercase hex identity",
    )
    .with_offending(field)
}

pub(super) fn unknown_provenance_stage(stage: &str) -> InterfaceRefusal {
    invalid(
        antecedent_core::reason_details::interface_contract::UNKNOWN_PROVENANCE_STAGE,
        "name provenance stages by the composition stage vocabulary",
    )
    .with_offending(stage)
}

pub(super) fn unknown_component(component: &str) -> InterfaceRefusal {
    invalid(
        antecedent_core::reason_details::interface_contract::UNKNOWN_COMPONENT,
        "connect components that are part of the declared composition",
    )
    .with_offending(component)
}

pub(super) fn unknown_port(component: &str, port: &str) -> InterfaceRefusal {
    invalid(
        antecedent_core::reason_details::interface_contract::UNKNOWN_PORT,
        "connect a port the component's contract declares, produced at the upstream end and consumed at the downstream end",
    )
    .with_offending(format!("{component}.{port}"))
}

pub(super) fn duplicate_component(component: &str) -> InterfaceRefusal {
    invalid(
        antecedent_core::reason_details::interface_contract::DUPLICATE_COMPONENT,
        "give every component in a composition its own identity",
    )
    .with_offending(component)
}

pub(super) fn port_multiply_fed(component: &str, port: &str) -> InterfaceRefusal {
    invalid(
        antecedent_core::reason_details::interface_contract::PORT_MULTIPLY_FED,
        "feed a consumed port from one produced port; combine evidence in a component that declares the combination",
    )
    .with_offending(format!("{component}.{port}"))
}

// -- quantity and support mismatch (interface_contract_quantity_mismatch) -------------------------------

pub(super) fn variable_mismatch(expected: &str, supplied: &str) -> InterfaceRefusal {
    quantity_mismatch(
        antecedent_core::reason_details::interface_contract::VARIABLE_MISMATCH,
        expected,
        supplied,
        "connect a producer of the variable (and role) the consumer names, or declare a typed bridge between the two quantities",
    )
}

pub(super) fn units_mismatch(expected: &str, supplied: &str) -> InterfaceRefusal {
    quantity_mismatch(
        antecedent_core::reason_details::interface_contract::UNITS_MISMATCH,
        expected,
        supplied,
        "produce the quantity in the consumer's units through a declared, checked unit conversion; units are never converted implicitly",
    )
}

pub(super) fn population_mismatch(expected: &str, supplied: &str) -> InterfaceRefusal {
    quantity_mismatch(
        antecedent_core::reason_details::interface_contract::POPULATION_MISMATCH,
        expected,
        supplied,
        "produce the quantity for the consumer's population, or carry it with a declared transport of evidence between the two populations",
    )
}

pub(super) fn regime_mismatch(expected: &str, supplied: &str) -> InterfaceRefusal {
    quantity_mismatch(
        antecedent_core::reason_details::interface_contract::REGIME_MISMATCH,
        expected,
        supplied,
        "produce the quantity under the consumer's intervention or regime; an observational quantity is not an interventional one",
    )
}

pub(super) fn horizon_mismatch(expected: u32, supplied: u32) -> InterfaceRefusal {
    quantity_mismatch(
        antecedent_core::reason_details::interface_contract::HORIZON_MISMATCH,
        expected.to_string(),
        supplied.to_string(),
        "produce the quantity at the consumer's causal horizon",
    )
}

pub(super) fn functional_mismatch(expected: &str, supplied: &str) -> InterfaceRefusal {
    quantity_mismatch(
        antecedent_core::reason_details::interface_contract::FUNCTIONAL_MISMATCH,
        expected,
        supplied,
        "produce the functional the consumer names (a mean is not a risk, a risk is not a quantile)",
    )
}

pub(super) fn conditioning_mismatch(expected: String, supplied: String) -> InterfaceRefusal {
    quantity_mismatch(
        antecedent_core::reason_details::interface_contract::CONDITIONING_MISMATCH,
        expected,
        supplied,
        "produce the quantity under exactly the consumer's conditioning events",
    )
}

pub(super) fn transform_mismatch(expected: &str, supplied: &str) -> InterfaceRefusal {
    quantity_mismatch(
        antecedent_core::reason_details::interface_contract::TRANSFORM_MISMATCH,
        expected,
        supplied,
        "produce the quantity on the consumer's scale, or declare the transform between the scales",
    )
}

pub(super) fn semantic_identity_mismatch(expected: &str, supplied: &str) -> InterfaceRefusal {
    quantity_mismatch(
        antecedent_core::reason_details::interface_contract::SEMANTIC_IDENTITY_MISMATCH,
        expected,
        supplied,
        "connect a producer of the same estimand or semantic identity; matching names, units or shapes do not establish it",
    )
}

pub(super) fn support_mismatch(expected: String, supplied: String) -> InterfaceRefusal {
    quantity_mismatch(
        antecedent_core::reason_details::interface_contract::SUPPORT_MISMATCH,
        expected,
        supplied,
        "produce values whose declared support lies inside the support the consumer accepts, or widen the consumer's declared domain with evidence that it handles the wider one",
    )
}

// -- assumptions (interface_contract_assumption_mismatch) -----------------------------------------------

pub(super) fn measurement_assumption_mismatch(
    expected: String,
    supplied: String,
) -> InterfaceRefusal {
    assumption_mismatch(
        antecedent_core::reason_details::interface_contract::MEASUREMENT_ASSUMPTION_MISMATCH,
        expected,
        supplied,
        "produce the quantity under the measurement models the consumer names (by model identity), or declare the consumer under the producer's models",
    )
}

pub(super) fn dependence_unknown(port: &str) -> InterfaceRefusal {
    assumption_mismatch(
        antecedent_core::reason_details::interface_contract::DEPENDENCE_UNKNOWN,
        "declared evidence sources",
        "undeclared",
        "declare the evidence sources the produced quantity rests on (their evidence-ledger identities), so overlap with other inputs is computable",
    )
    .with_offending(port)
}

pub(super) fn evidence_overlap_unknown(first: &str, second: &str) -> InterfaceRefusal {
    assumption_mismatch(
        antecedent_core::reason_details::interface_contract::DEPENDENCE_UNKNOWN,
        "known disjoint observations for the declared evidence sources",
        "overlap unknown",
        "supply an evidence ledger with known observation sets for both inputs, or declare the dependence the combination uses",
    )
    .with_offending(format!("{first}, {second}"))
}

pub(super) fn shared_evidence(first: &str, second: &str, shared: usize) -> InterfaceRefusal {
    assumption_mismatch(
        antecedent_core::reason_details::interface_contract::SHARED_EVIDENCE,
        "disjoint evidence among the inputs the consumer requires disjoint",
        format!("{shared} shared source(s)"),
        "feed the consumer from inputs with disjoint evidence, or remove the disjointness requirement and declare the dependence the combination uses",
    )
    .with_offending(format!("{first}, {second}"))
}

pub(super) fn provenance_missing(stage: &str) -> InterfaceRefusal {
    assumption_mismatch(
        antecedent_core::reason_details::interface_contract::PROVENANCE_MISSING,
        stage,
        "not supplied",
        "supply the provenance stage the consumer requires with the produced quantity, or remove the requirement",
    )
    .with_offending(stage)
}

// -- operations, uncertainty meaning, calibration ----------------------------------------------------

pub(super) fn operation_unavailable(
    operation: &str,
    offered: String,
    remedy: &'static str,
) -> InterfaceRefusal {
    ExternalRefusal::new(
        antecedent_core::reason_code!("interface_contract_operation_unavailable"),
        STAGE_BIND,
        antecedent_core::reason_details::interface_contract::OPERATION_UNAVAILABLE,
    )
    .with_offending(operation)
    .with_expected(operation)
    .with_supplied(offered)
    .with_remedy(remedy)
}

pub(super) fn uncertainty_meaning_missing(expected: &str) -> InterfaceRefusal {
    uncertainty_mismatch(
        antecedent_core::reason_details::interface_contract::UNCERTAINTY_MEANING_MISSING,
        expected,
        "no uncertainty meaning",
        "state what the producer's covariance, law, draws, interval or posterior means, or produce one with the meaning the consumer requires",
    )
}

pub(super) fn uncertainty_paradigm_mismatch(expected: &str, supplied: &str) -> InterfaceRefusal {
    uncertainty_mismatch(
        antecedent_core::reason_details::interface_contract::UNCERTAINTY_PARADIGM_MISMATCH,
        expected,
        supplied,
        "produce uncertainty of the paradigm the consumer requires; a posterior is not a sampling distribution and neither is a bootstrap",
    )
}

pub(super) fn uncertainty_meaning_mismatch(expected: &str, supplied: &str) -> InterfaceRefusal {
    uncertainty_mismatch(
        antecedent_core::reason_details::interface_contract::UNCERTAINTY_MEANING_MISMATCH,
        expected,
        supplied,
        "produce the object the consumer requires; a posterior over parameters is not a posterior over a causal functional or an outcome",
    )
}

// -- cycles ------------------------------------------------------------------------------------------

pub(super) fn cyclic_composition(path: String) -> InterfaceRefusal {
    ExternalRefusal::new(
        antecedent_core::reason_code!("interface_contract_cyclic_composition"),
        STAGE_BIND,
        antecedent_core::reason_details::interface_contract::CYCLIC_COMPOSITION,
    )
    .with_offending(path)
    .with_remedy(
        "remove the feedback between a component's output and another's input; cyclic scientific composition is not supported, so compose the components as a feed-forward graph or model the loop inside one component's own contract",
    )
}
