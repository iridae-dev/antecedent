//! Python bridge for the interface-contract adapters of a declared mechanism, a scientific bridge
//! and a measurement model.
//!
//! Python declares the component (in the wire the component's own module already speaks) and the
//! port-level facts its declaration does not state: the support of each quantity, the estimand it
//! answers, what its uncertainty means, the measurement models it is established under, the
//! calibration it holds and the evidence it rests on. Nothing is inferred: an omitted support is
//! the real line, an omitted uncertainty meaning is none, an omitted measurement model is none.
//! The projection of the declaration to an interface contract, and every refusal of a malformed
//! or unbound declaration, are Rust's. The result is the contract in the interface-contract wire
//! layout, ready for the one connection and composition check.

use std::collections::{BTreeMap, BTreeSet};

use antecedent::analysis::evidence_ledger::SourceId;
use antecedent::analysis::interface_contract::{
    CalibrationStatus, ContractOperation, EvidenceBasis, InterfaceContract, PortFacts,
    SupportDomain, UncertaintyMeaning,
};
use antecedent::contract::bridge::BridgeInterfaceBasis;
use antecedent::contract::measurement::{
    MeasurementInterfaceBasis, measurement_interface_contract,
};
use antecedent::contract::{MechanismInterfaceBasis, MechanismOriginKind};
use antecedent::io::interface_contract_artifact::contract_to_wire;
use pyo3::prelude::*;
use serde_json::{Map, Value as Json};

use crate::bridge::external_refusal_json;
use crate::measurement_model_api::{
    flag, list, member, object, opt_text, parse, parse_model, present, text, texts,
};
use crate::scientific_bridge_api::parse_contract;

type Checked = (Option<String>, Option<String>);

fn quantity(value: &Json, what: &str) -> PyResult<antecedent::scientific::ScientificQuantity> {
    let wire: antecedent::io::quantity_wire::ScientificQuantityWire =
        serde_json::from_value(value.clone())
            .map_err(|error| crate::value_err(format!("invalid {what}: {error}")))?;
    antecedent::scientific::ScientificQuantity::try_from(wire).map_err(crate::value_err)
}

fn support(map: &Map<String, Json>) -> PyResult<SupportDomain> {
    let Some(value) = present(map, "support") else {
        return Ok(SupportDomain::Real);
    };
    let support = object(value, "support")?;
    match text(support, "kind")?.as_str() {
        "real" => Ok(SupportDomain::Real),
        "non_negative" => Ok(SupportDomain::NonNegative),
        "binary" => Ok(SupportDomain::Binary),
        "interval" => {
            let bound = |key: &str| {
                support
                    .get(key)
                    .and_then(Json::as_f64)
                    .ok_or_else(|| crate::value_err(format!("an interval support names `{key}`")))
            };
            Ok(SupportDomain::interval(bound("lower")?, bound("upper")?))
        }
        "categorical" => Ok(SupportDomain::Categorical { levels: texts(support, "levels")? }),
        other => Err(crate::value_err(format!("unknown support `{other}`"))),
    }
}

fn facts(value: &Json, what: &str) -> PyResult<PortFacts> {
    let map = object(value, what)?;
    let mut facts = PortFacts::new(support(map)?);
    if let Some(identity) = opt_text(map, "semantic_identity")? {
        facts = facts.with_semantic_identity(identity);
    }
    if let Some(name) = opt_text(map, "uncertainty")? {
        let meaning = UncertaintyMeaning::from_name(&name)
            .ok_or_else(|| crate::value_err(format!("unknown uncertainty meaning `{name}`")))?;
        facts = facts.with_uncertainty(meaning.meaning);
    }
    if present(map, "measurement_models").is_some() {
        facts = facts.with_measurement_models(texts(map, "measurement_models")?);
    }
    Ok(facts)
}

fn operations(map: &Map<String, Json>, key: &str) -> PyResult<BTreeSet<ContractOperation>> {
    texts(map, key)?
        .iter()
        .map(|name| {
            ContractOperation::from_name(name)
                .ok_or_else(|| crate::value_err(format!("unknown operation `{name}`")))
        })
        .collect()
}

fn calibration(map: &Map<String, Json>) -> PyResult<Option<CalibrationStatus>> {
    let Some(value) = present(map, "calibration") else {
        return Ok(None);
    };
    let wire = object(value, "calibration")?;
    Ok(Some(match text(wire, "status")?.as_str() {
        "none" => CalibrationStatus::None,
        "point_only" => CalibrationStatus::PointOnly,
        "not_measured" => CalibrationStatus::NotMeasured,
        "measured" => CalibrationStatus::Measured { record_id: text(wire, "record_id")? },
        "inherited" => CalibrationStatus::Inherited { input: text(wire, "input")? },
        other => return Err(crate::value_err(format!("unknown calibration `{other}`"))),
    }))
}

/// `None` is undeclared evidence; a list is the declared ledger sources it rests on.
fn evidence(map: &Map<String, Json>) -> PyResult<Option<EvidenceBasis>> {
    let Some(value) = present(map, "evidence") else {
        return Ok(None);
    };
    let wire = object(value, "evidence")?;
    if !flag(wire, "declared")? {
        return Ok(Some(EvidenceBasis::Undeclared));
    }
    let mut sources = BTreeSet::new();
    for hex in texts(wire, "sources")? {
        sources.insert(
            SourceId::from_hex(&hex)
                .ok_or_else(|| crate::value_err("an evidence source is not a 64-digit digest"))?,
        );
    }
    Ok(Some(EvidenceBasis::Declared(sources)))
}

fn contract_json(contract: &InterfaceContract) -> PyResult<String> {
    serde_json::to_value(contract_to_wire(contract))
        .map(|value| value.to_string())
        .map_err(|error| crate::serialization_err(error.to_string()))
}

/// The interface contract of a declared mechanism: its output as the produced port and one
/// consumed port per parent. `declaration_json` is the mechanism declaration; `basis_json` the
/// port facts, the origin (`fitted`, `declared` or `external_binding`), the operations offered
/// and what is required of the parents.
#[pyfunction]
fn mechanism_interface_contract(declaration_json: &str, basis_json: &str) -> PyResult<Checked> {
    crate::catch_ffi(|| {
        let mut declaration = crate::mechanism_composition_api::parse_declaration(&parse(
            declaration_json,
            "mechanism declaration",
        )?)?;
        let basis_value = parse(basis_json, "mechanism interface basis")?;
        let map = object(&basis_value, "mechanism interface basis")?;
        declaration.origin = match text(map, "origin")?.as_str() {
            "fitted" => MechanismOriginKind::Fitted,
            "declared" => MechanismOriginKind::Declared,
            "external_binding" => MechanismOriginKind::ExternalBinding,
            other => return Err(crate::value_err(format!("unknown mechanism origin `{other}`"))),
        };
        let mut basis = MechanismInterfaceBasis::new(
            facts(member(map, "output")?, "output facts")?,
            operations(map, "offered")?,
        );
        let mut parents = BTreeMap::new();
        for entry in list(map, "parents")? {
            let pair = entry
                .as_array()
                .filter(|pair| pair.len() == 2)
                .ok_or_else(|| crate::value_err("`parents` holds an entry that is not a pair"))?;
            let name = pair[0]
                .as_str()
                .ok_or_else(|| crate::value_err("`parents` holds a pair without a name"))?;
            parents.insert(name.to_owned(), facts(&pair[1], "parent facts")?);
        }
        basis.parents = parents;
        let parent_operation = text(map, "parent_operation")?;
        basis.parent_operation = ContractOperation::from_name(&parent_operation)
            .ok_or_else(|| crate::value_err(format!("unknown operation `{parent_operation}`")))?;
        basis.calibration = calibration(map)?;
        basis.evidence = evidence(map)?;
        basis.disjoint_parent_evidence = flag(map, "disjoint_parent_evidence")?;
        Ok((Some(contract_json(&declaration.interface_contract(&basis))?), None))
    })
}

/// The interface contract of a scientific bridge: the source quantity as the consumed port and
/// the target quantity as the produced port. `contract_json` is the bridge contract.
#[pyfunction]
fn bridge_interface_contract(
    contract_json_text: &str,
    component: &str,
    basis_json: &str,
) -> PyResult<Checked> {
    crate::catch_ffi(|| {
        let contract = parse_contract(&parse(contract_json_text, "bridge contract")?)?;
        let basis_value = parse(basis_json, "bridge interface basis")?;
        let map = object(&basis_value, "bridge interface basis")?;
        let basis = BridgeInterfaceBasis {
            source: facts(member(map, "source")?, "source facts")?,
            target: facts(member(map, "target")?, "target facts")?,
            calibration: calibration(map)?,
            evidence: evidence(map)?.unwrap_or(EvidenceBasis::Undeclared),
        };
        Ok((Some(contract_json(&contract.interface_contract(component, &basis))?), None))
    })
}

/// The interface contract of a measurement model: the recorded indicator it consumes and the
/// corrected construct it produces under the model's identity. A model that does not bind the two
/// quantities, or an operation the route does not support, is refused.
#[pyfunction]
fn measurement_interface_contract_of(
    component: &str,
    model_json: &str,
    indicator_json: &str,
    construct_json: &str,
    basis_json: &str,
) -> PyResult<Checked> {
    crate::catch_ffi(|| {
        let model = parse_model(&parse(model_json, "measurement model")?)?;
        let indicator = quantity(&parse(indicator_json, "indicator")?, "indicator")?;
        let construct = quantity(&parse(construct_json, "construct")?, "construct")?;
        let basis_value = parse(basis_json, "measurement interface basis")?;
        let map = object(&basis_value, "measurement interface basis")?;
        let basis = MeasurementInterfaceBasis {
            indicator: facts(member(map, "indicator")?, "indicator facts")?,
            construct: facts(member(map, "construct")?, "construct facts")?,
            operations: operations(map, "operations")?,
            calibration: calibration(map)?,
            evidence: evidence(map)?.unwrap_or(EvidenceBasis::Undeclared),
        };
        Ok(
            match measurement_interface_contract(component, &model, &indicator, &construct, &basis)
            {
                Ok(contract) => (Some(contract_json(&contract)?), None),
                Err(refusal) => (None, Some(external_refusal_json(&refusal))),
            },
        )
    })
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(mechanism_interface_contract, module)?)?;
    module.add_function(wrap_pyfunction!(bridge_interface_contract, module)?)?;
    module.add_function(wrap_pyfunction!(measurement_interface_contract_of, module)?)?;
    Ok(())
}
