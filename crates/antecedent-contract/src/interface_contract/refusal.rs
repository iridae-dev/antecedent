//! Typed refusals of the interface contract: a registered reason code, an
//! `interface_contract.*` detail and the declaration that would license the
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
        "interface_contract.blank_field",
        "name every identity, in at most the declared number of bytes",
    )
    .with_offending(field)
}

pub(super) fn invalid_quantity(port: &str) -> InterfaceRefusal {
    invalid(
        "interface_contract.invalid_quantity",
        "name the variable, units, population, regime, functional and transform of each quantity, with canonical conditioning",
    )
    .with_offending(port)
}

pub(super) fn duplicate_port(port: &str) -> InterfaceRefusal {
    invalid(
        "interface_contract.duplicate_port",
        "give every produced port, and every consumed port, its own name",
    )
    .with_offending(port)
}

pub(super) fn invalid_support(port: &str) -> InterfaceRefusal {
    invalid(
        "interface_contract.invalid_support",
        "declare a support domain with finite ordered bounds or sorted, distinct, non-blank levels",
    )
    .with_offending(port)
}

pub(super) fn missing_operations(port: &str) -> InterfaceRefusal {
    invalid(
        "interface_contract.missing_operations",
        "declare at least one operation the port offers or requires",
    )
    .with_offending(port)
}

pub(super) fn uncertainty_meaning_required(port: &str) -> InterfaceRefusal {
    invalid(
        "interface_contract.uncertainty_meaning_required",
        "state what the covariance, law, draws, interval or posterior means (posterior of a causal functional, estimator sampling, bootstrap, ...)",
    )
    .with_offending(port)
}

pub(super) fn uncertainty_meaning_unused(port: &str) -> InterfaceRefusal {
    invalid(
        "interface_contract.uncertainty_meaning_unused",
        "state an uncertainty meaning only for a port that offers or requires a covariance, law, draws, interval or posterior",
    )
    .with_offending(port)
}

pub(super) fn posterior_without_posterior_meaning(port: &str) -> InterfaceRefusal {
    invalid(
        "interface_contract.posterior_without_posterior_meaning",
        "a posterior operation carries a posterior meaning; declare the sampling or resampling distribution as draws or an interval",
    )
    .with_offending(port)
}

pub(super) fn point_only_with_uncertainty(port: &str) -> InterfaceRefusal {
    invalid(
        "interface_contract.point_only_with_uncertainty",
        "a point-only label cannot accompany a covariance, law, draws, interval or posterior; use none or unmeasured, or drop the operation",
    )
    .with_offending(port)
}

pub(super) fn calibration_record_blank(port: &str) -> InterfaceRefusal {
    invalid(
        "interface_contract.calibration_record_blank",
        "a measured calibration label names the identity of its measured record",
    )
    .with_offending(port)
}

pub(super) fn calibration_without_interval(port: &str) -> InterfaceRefusal {
    invalid(
        "interface_contract.calibration_without_interval",
        "calibration is measured for an interval or a posterior; declare that operation or use another calibration label",
    )
    .with_offending(port)
}

pub(super) fn inherited_input_unknown(port: &str, input: &str) -> InterfaceRefusal {
    invalid(
        "interface_contract.inherited_input_unknown",
        "inherit calibration from a consumed port the contract declares",
    )
    .with_offending(port)
    .with_supplied(input)
}

pub(super) fn invalid_digest(field: &str) -> InterfaceRefusal {
    invalid(
        "interface_contract.invalid_digest",
        "name a measurement model or evidence source by its 64-character lowercase hex identity",
    )
    .with_offending(field)
}

pub(super) fn unknown_provenance_stage(stage: &str) -> InterfaceRefusal {
    invalid(
        "interface_contract.unknown_provenance_stage",
        "name provenance stages by the composition stage vocabulary",
    )
    .with_offending(stage)
}

pub(super) fn unknown_component(component: &str) -> InterfaceRefusal {
    invalid(
        "interface_contract.unknown_component",
        "connect components that are part of the declared composition",
    )
    .with_offending(component)
}

pub(super) fn unknown_port(component: &str, port: &str) -> InterfaceRefusal {
    invalid(
        "interface_contract.unknown_port",
        "connect a port the component's contract declares, produced at the upstream end and consumed at the downstream end",
    )
    .with_offending(format!("{component}.{port}"))
}

pub(super) fn duplicate_component(component: &str) -> InterfaceRefusal {
    invalid(
        "interface_contract.duplicate_component",
        "give every component in a composition its own identity",
    )
    .with_offending(component)
}

pub(super) fn port_multiply_fed(component: &str, port: &str) -> InterfaceRefusal {
    invalid(
        "interface_contract.port_multiply_fed",
        "feed a consumed port from one produced port; combine evidence in a component that declares the combination",
    )
    .with_offending(format!("{component}.{port}"))
}

// -- quantity and support mismatch (interface_contract_quantity_mismatch) -------------------------------

pub(super) fn variable_mismatch(expected: &str, supplied: &str) -> InterfaceRefusal {
    quantity_mismatch(
        "interface_contract.variable_mismatch",
        expected,
        supplied,
        "connect a producer of the variable (and role) the consumer names, or declare a typed bridge between the two quantities",
    )
}

pub(super) fn units_mismatch(expected: &str, supplied: &str) -> InterfaceRefusal {
    quantity_mismatch(
        "interface_contract.units_mismatch",
        expected,
        supplied,
        "produce the quantity in the consumer's units through a declared, checked unit conversion; units are never converted implicitly",
    )
}

pub(super) fn population_mismatch(expected: &str, supplied: &str) -> InterfaceRefusal {
    quantity_mismatch(
        "interface_contract.population_mismatch",
        expected,
        supplied,
        "produce the quantity for the consumer's population, or carry it with a declared transport of evidence between the two populations",
    )
}

pub(super) fn regime_mismatch(expected: &str, supplied: &str) -> InterfaceRefusal {
    quantity_mismatch(
        "interface_contract.regime_mismatch",
        expected,
        supplied,
        "produce the quantity under the consumer's intervention or regime; an observational quantity is not an interventional one",
    )
}

pub(super) fn horizon_mismatch(expected: u32, supplied: u32) -> InterfaceRefusal {
    quantity_mismatch(
        "interface_contract.horizon_mismatch",
        expected.to_string(),
        supplied.to_string(),
        "produce the quantity at the consumer's causal horizon",
    )
}

pub(super) fn functional_mismatch(expected: &str, supplied: &str) -> InterfaceRefusal {
    quantity_mismatch(
        "interface_contract.functional_mismatch",
        expected,
        supplied,
        "produce the functional the consumer names (a mean is not a risk, a risk is not a quantile)",
    )
}

pub(super) fn conditioning_mismatch(expected: String, supplied: String) -> InterfaceRefusal {
    quantity_mismatch(
        "interface_contract.conditioning_mismatch",
        expected,
        supplied,
        "produce the quantity under exactly the consumer's conditioning events",
    )
}

pub(super) fn transform_mismatch(expected: &str, supplied: &str) -> InterfaceRefusal {
    quantity_mismatch(
        "interface_contract.transform_mismatch",
        expected,
        supplied,
        "produce the quantity on the consumer's scale, or declare the transform between the scales",
    )
}

pub(super) fn semantic_identity_mismatch(expected: &str, supplied: &str) -> InterfaceRefusal {
    quantity_mismatch(
        "interface_contract.semantic_identity_mismatch",
        expected,
        supplied,
        "connect a producer of the same estimand or semantic identity; matching names, units or shapes do not establish it",
    )
}

pub(super) fn support_mismatch(expected: String, supplied: String) -> InterfaceRefusal {
    quantity_mismatch(
        "interface_contract.support_mismatch",
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
        "interface_contract.measurement_assumption_mismatch",
        expected,
        supplied,
        "produce the quantity under the measurement models the consumer names (by model identity), or declare the consumer under the producer's models",
    )
}

pub(super) fn dependence_unknown(port: &str) -> InterfaceRefusal {
    assumption_mismatch(
        "interface_contract.dependence_unknown",
        "declared evidence sources",
        "undeclared",
        "declare the evidence sources the produced quantity rests on (their evidence-ledger identities), so overlap with other inputs is computable",
    )
    .with_offending(port)
}

pub(super) fn shared_evidence(first: &str, second: &str, shared: usize) -> InterfaceRefusal {
    assumption_mismatch(
        "interface_contract.shared_evidence",
        "disjoint evidence among the inputs the consumer requires disjoint",
        format!("{shared} shared source(s)"),
        "feed the consumer from inputs with disjoint evidence, or remove the disjointness requirement and declare the dependence the combination uses",
    )
    .with_offending(format!("{first}, {second}"))
}

pub(super) fn provenance_missing(stage: &str) -> InterfaceRefusal {
    assumption_mismatch(
        "interface_contract.provenance_missing",
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
        "interface_contract.operation_unavailable",
    )
    .with_offending(operation)
    .with_expected(operation)
    .with_supplied(offered)
    .with_remedy(remedy)
}

pub(super) fn uncertainty_meaning_missing(expected: &str) -> InterfaceRefusal {
    uncertainty_mismatch(
        "interface_contract.uncertainty_meaning_missing",
        expected,
        "no uncertainty meaning",
        "state what the producer's covariance, law, draws, interval or posterior means, or produce one with the meaning the consumer requires",
    )
}

pub(super) fn uncertainty_paradigm_mismatch(expected: &str, supplied: &str) -> InterfaceRefusal {
    uncertainty_mismatch(
        "interface_contract.uncertainty_paradigm_mismatch",
        expected,
        supplied,
        "produce uncertainty of the paradigm the consumer requires; a posterior is not a sampling distribution and neither is a bootstrap",
    )
}

pub(super) fn uncertainty_meaning_mismatch(expected: &str, supplied: &str) -> InterfaceRefusal {
    uncertainty_mismatch(
        "interface_contract.uncertainty_meaning_mismatch",
        expected,
        supplied,
        "produce the object the consumer requires; a posterior over parameters is not a posterior over a causal functional or an outcome",
    )
}

fn calibration_not_held(
    detail: &str,
    expected: &str,
    supplied: String,
    remedy: &'static str,
) -> InterfaceRefusal {
    ExternalRefusal::new(
        antecedent_core::reason_code!("interface_contract_calibration_not_held"),
        STAGE_BIND,
        detail,
    )
    .with_expected(expected)
    .with_supplied(supplied)
    .with_remedy(remedy)
}

pub(super) fn calibration_not_measured(supplied: String) -> InterfaceRefusal {
    calibration_not_held(
        "interface_contract.calibration_not_held",
        "measured",
        supplied,
        "connect a producer that holds a measured calibration record for the composed quantity, or drop the consumer's calibration requirement; calibration of a component is not calibration of a composition",
    )
}

pub(super) fn calibration_inherited_unresolved(port: &str) -> InterfaceRefusal {
    calibration_not_held(
        "interface_contract.calibration_inherited_unresolved",
        "measured",
        "inherited".to_owned(),
        "check the whole composition so the inherited calibration resolves to the upstream record that holds it",
    )
    .with_offending(port)
}

pub(super) fn calibration_inherited_not_held(
    port: &str,
    input: &str,
    upstream: String,
) -> InterfaceRefusal {
    calibration_not_held(
        "interface_contract.calibration_inherited_not_held",
        "measured",
        upstream,
        "inherit calibration only from an input whose producer holds a measured record; otherwise declare the output unmeasured or point-only",
    )
    .with_offending(format!("{port} <- {input}"))
}

pub(super) fn calibration_inherited_unfed(port: &str, input: &str) -> InterfaceRefusal {
    calibration_not_held(
        "interface_contract.calibration_inherited_unfed",
        "measured",
        "input not connected".to_owned(),
        "connect the input the output inherits its calibration from, or declare the output unmeasured",
    )
    .with_offending(format!("{port} <- {input}"))
}

// -- cycles ------------------------------------------------------------------------------------------

pub(super) fn cyclic_composition(path: String) -> InterfaceRefusal {
    ExternalRefusal::new(
        antecedent_core::reason_code!("interface_contract_cyclic_composition"),
        STAGE_BIND,
        "interface_contract.cyclic_composition",
    )
    .with_offending(path)
    .with_remedy(
        "remove the feedback between a component's output and another's input; cyclic scientific composition is not supported, so compose the components as a feed-forward graph or model the loop inside one component's own contract",
    )
}
