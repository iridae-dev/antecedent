//! The connection decision: whether the quantity one component produces is
//! the quantity another consumes, decided from the two declarations alone.
//!
//! Matching shape, dtype or name is not scientific matching. A connection is
//! accepted only when the semantic coordinate (variable, role, units,
//! population, regime, horizon, functional, conditioning, transform), the
//! semantic identity, the support, the measurement assumptions, the declared
//! evidence, the requested operations, the meaning of the uncertainty, the
//! calibration the consumer relies on and the provenance the consumer requires
//! all agree. Every defect is reported, each as a coded refusal with the
//! declaration that would license the connection.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeSet;

use antecedent_core::identity::{CanonicalWriter, SemanticDigest};

use super::declaration::{
    CalibrationRequirement, CalibrationStatus, ConsumedPort, ContractOperation, EvidenceBasis,
    INTERFACE_CONTRACT_VERSION, InterfaceContract, ProducedPort, UncertaintyMeaning,
    operations_text,
};
use super::refusal::{
    InterfaceRefusal, calibration_inherited_unresolved, calibration_not_measured,
    conditioning_mismatch, dependence_unknown, functional_mismatch, horizon_mismatch,
    measurement_assumption_mismatch, operation_unavailable, population_mismatch,
    provenance_missing, regime_mismatch, semantic_identity_mismatch, support_mismatch,
    transform_mismatch, uncertainty_meaning_mismatch, uncertainty_meaning_missing,
    uncertainty_paradigm_mismatch, units_mismatch, unknown_port, variable_mismatch,
};
use crate::ScientificQuantity;

const TAG_CONNECTION: &str = "antecedent.interface_contract.connection";

/// An accepted connection of one produced port to one consumed port.
///
/// The connection carries what the producer holds, unchanged: the calibration
/// label is the producer's, never raised by the consumer, and the evidence is
/// the producer's declared basis.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Connection {
    /// Producing component.
    pub producer: String,
    /// Produced port.
    pub produced_port: String,
    /// Consuming component.
    pub consumer: String,
    /// Consumed port.
    pub consumed_port: String,
    /// The operations the consumer requires, every one of which the producer
    /// offers.
    pub operations: BTreeSet<ContractOperation>,
    /// The meaning of the uncertainty the producer offers, when it does.
    pub uncertainty: Option<UncertaintyMeaning>,
    /// The calibration the producer holds (a resolved value inside a checked
    /// composition).
    pub calibration: CalibrationStatus,
    /// The evidence the producer declares.
    pub evidence: EvidenceBasis,
    /// Identity binding both contracts, both ports and the negotiated
    /// operations.
    pub identity: SemanticDigest,
}

fn conditioning_text(quantity: &ScientificQuantity) -> String {
    quantity
        .conditioning
        .iter()
        .map(|c| format!("{}={}", c.variable_id, c.value_id))
        .collect::<Vec<_>>()
        .join(",")
}

fn set_text(set: &BTreeSet<String>) -> String {
    if set.is_empty() {
        "none".to_owned()
    } else {
        set.iter().map(String::as_str).collect::<Vec<_>>().join(",")
    }
}

fn operation_remedy(operation: ContractOperation) -> &'static str {
    match operation {
        ContractOperation::Point => "supply a producer that offers a point value of the quantity",
        ContractOperation::Mean => {
            "supply a producer that offers the mean; a law or draws are a different operation until the mean is declared"
        }
        ContractOperation::Law => {
            "a mean or a point does not determine a law; supply a producer that offers the law (distribution function, quantile or log density), or lower the consumer's requirement to the mean"
        }
        ContractOperation::Draws => {
            "a mean or a law does not supply draws; supply a producer that offers draws, or require the law or the mean the consumer actually uses"
        }
        ContractOperation::JointDraws => {
            "draws are aligned only under a declared joint law; supply a producer that offers joint draws, since draws of unrelated marginals are never paired by index"
        }
        ContractOperation::Covariance => {
            "a covariance needs a declared joint law with its dependence; supply a producer that offers the covariance or the joint draws it is computed from"
        }
        ContractOperation::Interval => {
            "supply a producer that offers an interval of the meaning the consumer states, with measured calibration where its coverage is relied on"
        }
        ContractOperation::Posterior => {
            "a posterior needs a producer in the posterior paradigm; an interval or draws of another paradigm are not a posterior"
        }
    }
}

fn compare_quantities(
    producer: &ScientificQuantity,
    consumer: &ScientificQuantity,
    refusals: &mut Vec<InterfaceRefusal>,
) {
    if producer.variable_id != consumer.variable_id || producer.role != consumer.role {
        refusals.push(variable_mismatch(
            &format!("{}/{:?}", consumer.variable_id, consumer.role),
            &format!("{}/{:?}", producer.variable_id, producer.role),
        ));
    }
    if producer.units != consumer.units {
        refusals.push(units_mismatch(&consumer.units, &producer.units));
    }
    if producer.population_id != consumer.population_id {
        refusals.push(population_mismatch(&consumer.population_id, &producer.population_id));
    }
    if producer.regime_id != consumer.regime_id {
        refusals.push(regime_mismatch(&consumer.regime_id, &producer.regime_id));
    }
    if producer.horizon != consumer.horizon {
        refusals.push(horizon_mismatch(consumer.horizon, producer.horizon));
    }
    if producer.functional_id != consumer.functional_id {
        refusals.push(functional_mismatch(&consumer.functional_id, &producer.functional_id));
    }
    if producer.conditioning != consumer.conditioning {
        refusals
            .push(conditioning_mismatch(conditioning_text(consumer), conditioning_text(producer)));
    }
    if producer.transform_id != consumer.transform_id {
        refusals.push(transform_mismatch(&consumer.transform_id, &producer.transform_id));
    }
}

fn compare_uncertainty(
    producer: &ProducedPort,
    consumer: &ConsumedPort,
    refusals: &mut Vec<InterfaceRefusal>,
) {
    let Some(required) = consumer.uncertainty else {
        return;
    };
    match producer.uncertainty {
        None => refusals.push(uncertainty_meaning_missing(required.name())),
        Some(offered) if offered.paradigm() != required.paradigm() => {
            refusals.push(uncertainty_paradigm_mismatch(
                required.paradigm().as_str(),
                offered.paradigm().as_str(),
            ));
        }
        Some(offered) if offered.meaning != required.meaning => {
            refusals.push(uncertainty_meaning_mismatch(required.name(), offered.name()));
        }
        Some(_) => {}
    }
}

/// The decision for two ports, with the producer's calibration already
/// resolved (`None` when it is declared inherited and cannot be resolved from
/// the pair alone).
pub(super) fn check_ports(
    producer: &InterfaceContract,
    out: &ProducedPort,
    consumer: &InterfaceContract,
    inp: &ConsumedPort,
    resolved: Option<&CalibrationStatus>,
) -> Result<Connection, Vec<InterfaceRefusal>> {
    let mut refusals = Vec::new();
    compare_quantities(&out.quantity, &inp.quantity, &mut refusals);
    if out.semantic_identity != inp.semantic_identity {
        refusals.push(semantic_identity_mismatch(&inp.semantic_identity, &out.semantic_identity));
    }
    if !inp.support.accepts(&out.support) {
        refusals.push(support_mismatch(inp.support.describe(), out.support.describe()));
    }
    if out.measurement_models != inp.measurement_models {
        refusals.push(measurement_assumption_mismatch(
            set_text(&inp.measurement_models),
            set_text(&out.measurement_models),
        ));
    }
    if inp.dependence.requires_declared()
        && match &out.evidence {
            EvidenceBasis::Undeclared => true,
            EvidenceBasis::Declared(sources) => sources.is_empty(),
        }
    {
        refusals.push(dependence_unknown(&out.name));
    }
    for stage in inp.provenance.difference(&out.provenance) {
        refusals.push(provenance_missing(stage));
    }
    for operation in inp.operations.difference(&out.operations) {
        refusals.push(operation_unavailable(
            operation.as_str(),
            operations_text(&out.operations),
            operation_remedy(*operation),
        ));
    }
    compare_uncertainty(out, inp, &mut refusals);
    if inp.calibration == CalibrationRequirement::Measured {
        match resolved {
            Some(CalibrationStatus::Measured { .. }) => {}
            Some(other) => refusals.push(calibration_not_measured(other.label().to_owned())),
            None => refusals.push(calibration_inherited_unresolved(&out.name)),
        }
    }
    if !refusals.is_empty() {
        return Err(refusals);
    }
    let calibration = resolved.cloned().unwrap_or_else(|| out.calibration.clone());
    let producer_identity = producer.canonical_identity();
    let consumer_identity = consumer.canonical_identity();
    let mut writer = CanonicalWriter::new(TAG_CONNECTION);
    writer
        .word(INTERFACE_CONTRACT_VERSION)
        .text(&producer_identity.to_hex())
        .text(&out.name)
        .text(&consumer_identity.to_hex())
        .text(&inp.name)
        .word(inp.operations.len() as u64);
    for operation in &inp.operations {
        writer.word(operation.tag());
    }
    writer.text(calibration.label());
    Ok(Connection {
        producer: producer.component.clone(),
        produced_port: out.name.clone(),
        consumer: consumer.component.clone(),
        consumed_port: inp.name.clone(),
        operations: inp.operations.clone(),
        uncertainty: out.uncertainty,
        calibration,
        evidence: out.evidence.clone(),
        identity: SemanticDigest::from_bytes(writer.finish()),
    })
}

/// Decide, from the declarations alone and before any numerical execution,
/// whether `produced_port` of `producer` may feed `consumed_port` of
/// `consumer`.
///
/// # Errors
/// Every defect found, in a fixed order, each a coded refusal with an
/// `interface_contract.*` detail and the declaration that would license the
/// connection: a malformed declaration or unknown port first (alone), then
/// quantity, units, population, regime, support and semantic-identity
/// mismatches, measurement assumptions, unknown dependence, missing
/// provenance, operations the producer does not offer, uncertainty meaning,
/// and calibration the producer does not hold.
pub fn check_connection(
    producer: &InterfaceContract,
    produced_port: &str,
    consumer: &InterfaceContract,
    consumed_port: &str,
) -> Result<Connection, Vec<InterfaceRefusal>> {
    producer.validate().map_err(|refusal| vec![refusal])?;
    consumer.validate().map_err(|refusal| vec![refusal])?;
    let Some(out) = producer.produced(produced_port) else {
        return Err(vec![unknown_port(&producer.component, produced_port)]);
    };
    let Some(inp) = consumer.consumed(consumed_port) else {
        return Err(vec![unknown_port(&consumer.component, consumed_port)]);
    };
    let resolved = match &out.calibration {
        CalibrationStatus::Inherited { .. } => None,
        other => Some(other),
    };
    check_ports(producer, out, consumer, inp, resolved)
}
