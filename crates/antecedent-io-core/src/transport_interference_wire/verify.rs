//! Verification of a decoded structural transport identification.
//!
//! A `Transportable` result is a proof object: a rule name, a formula and the premises the
//! identifier checked. The consumer cannot re-run the identifier (the diagram is not
//! carried), so it re-checks everything the proof states about itself: the rule is one the
//! identifier emits, the formula has the shape that rule produces, and the factors and
//! variable sets are consistent with each other. A formula that is not the output of any
//! implemented rule is refused instead of being trusted.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeSet;

use super::{
    InterferenceTransportCertificateWire, NotCertifiedCertificateWire, PopulationFactorWire,
    TransportFormulaWire, TransportIdentificationWire,
};
use crate::IoError;

const MAX_IDS: usize = 4096;
const MAX_FACTORS: usize = 4096;
const MAX_PREMISES: usize = 64;
const MAX_TEXT_BYTES: usize = 4096;

const RULE_DIRECT: &str = "transport.sid.direct";
const RULE_STANDARDIZE: &str = "transport.sid.standardize";
const RULE_S_ADMISSIBLE: &str = "transport.sid.s_admissible";
const RULE_SINGLETON: &str = "transport.sid.singleton_c_components";
const RULE_TARGET_G: &str = "transport.sid.target_g_formula";
const RULE_IRRELEVANT: &str = "transport.sid.treatment_irrelevant";

fn refuse(detail: &str) -> IoError {
    IoError::Convert(format!("transport identification: {detail}"))
}

fn text(value: &str, what: &str) -> Result<(), IoError> {
    if value.len() > MAX_TEXT_BYTES {
        return Err(IoError::TooLarge);
    }
    if value.trim().is_empty() {
        return Err(refuse(&format!("{what} must be non-blank")));
    }
    Ok(())
}

fn id_set(ids: &[u32], what: &str) -> Result<BTreeSet<u32>, IoError> {
    if ids.len() > MAX_IDS {
        return Err(IoError::TooLarge);
    }
    let set: BTreeSet<u32> = ids.iter().copied().collect();
    if set.len() != ids.len() {
        return Err(refuse(&format!("{what} repeats a variable id")));
    }
    Ok(set)
}

struct Factor {
    variables: BTreeSet<u32>,
    conditioned_on: BTreeSet<u32>,
    interventions: BTreeSet<u32>,
    ordered_conditioning: Vec<u32>,
}

fn factor(wire: &PopulationFactorWire) -> Result<Factor, IoError> {
    text(&wire.population, "a factor population")?;
    let variables = id_set(&wire.variables, "a factor's variables")?;
    let conditioned_on = id_set(&wire.conditioned_on, "a factor's conditioning set")?;
    let interventions = id_set(&wire.interventions, "a factor's interventions")?;
    if !variables.is_disjoint(&conditioned_on) {
        return Err(refuse("a factor conditions on a variable it also models"));
    }
    Ok(Factor {
        variables,
        conditioned_on,
        interventions,
        ordered_conditioning: wire.conditioned_on.clone(),
    })
}

fn standardize(
    over: &[u32],
    source: &PopulationFactorWire,
    target: &PopulationFactorWire,
) -> Result<BTreeSet<u32>, IoError> {
    let over = id_set(over, "the standardized variables")?;
    let source = factor(source)?;
    let target = factor(target)?;
    if source.variables.is_empty()
        || source.conditioned_on != over
        || !source.variables.is_disjoint(&over)
    {
        return Err(refuse(
            "the source response must model the outcomes given exactly the standardized variables",
        ));
    }
    if target.variables != over
        || !target.conditioned_on.is_empty()
        || !target.interventions.is_empty()
    {
        return Err(refuse(
            "the target law must be the unconditional observational law of the standardized variables",
        ));
    }
    Ok(over)
}

fn factorization(sum_out: &[u32], factors: &[PopulationFactorWire]) -> Result<(), IoError> {
    if factors.len() > MAX_FACTORS {
        return Err(IoError::TooLarge);
    }
    let sum_out = id_set(sum_out, "the summed-out variables")?;
    let factors = factors.iter().map(factor).collect::<Result<Vec<_>, _>>()?;
    let mut modelled = BTreeSet::new();
    for entry in &factors {
        let mut variables = entry.variables.iter().copied();
        let (Some(variable), None) = (variables.next(), variables.next()) else {
            return Err(refuse("a factorization factor must model exactly one variable"));
        };
        if !entry.interventions.is_empty() || !modelled.insert(variable) {
            return Err(refuse(
                "factorization factors must be observational and model distinct variables",
            ));
        }
    }
    if !sum_out.is_subset(&modelled) {
        return Err(refuse("a summed-out variable has no factor"));
    }
    let mut seen = BTreeSet::new();
    for entry in &factors {
        if entry
            .ordered_conditioning
            .iter()
            .any(|parent| modelled.contains(parent) && !seen.contains(parent))
        {
            return Err(refuse("factorization factors are not in topological order"));
        }
        seen.extend(entry.variables.iter().copied());
    }
    Ok(())
}

fn transportable(
    formula: &TransportFormulaWire,
    certificate: &InterferenceTransportCertificateWire,
) -> Result<(), IoError> {
    text(&certificate.rule, "the certificate rule")?;
    if certificate.premises.is_empty() {
        return Err(refuse("a positive certificate must state the premises it checked"));
    }
    if certificate.premises.len() > MAX_PREMISES {
        return Err(IoError::TooLarge);
    }
    for premise in &certificate.premises {
        text(premise, "a certificate premise")?;
    }
    let targets = id_set(&certificate.selection_targets, "the certificate's selection targets")?;
    let rule = certificate.rule.as_str();
    match (rule, formula) {
        (RULE_DIRECT, TransportFormulaWire::Direct(wire)) => {
            let direct = factor(wire)?;
            if !targets.is_empty() || direct.variables.is_empty() {
                return Err(refuse(
                    "a direct transport has outcomes and depends on no selection target",
                ));
            }
        }
        (RULE_IRRELEVANT, TransportFormulaWire::Direct(wire)) => {
            let direct = factor(wire)?;
            if direct.variables.is_empty()
                || !direct.conditioned_on.is_empty()
                || !direct.interventions.is_empty()
            {
                return Err(refuse(
                    "a treatment-irrelevant transport is the unconditional observational law of the outcomes",
                ));
            }
        }
        (
            RULE_STANDARDIZE | RULE_S_ADMISSIBLE,
            TransportFormulaWire::Standardize { over, source_response, target_law },
        ) => {
            let over = standardize(over, source_response, target_law)?;
            if rule == RULE_STANDARDIZE && targets != over {
                return Err(refuse(
                    "a pre-treatment standardization certifies exactly its standardized variables",
                ));
            }
        }
        (
            RULE_SINGLETON | RULE_TARGET_G,
            TransportFormulaWire::RecursiveFactorization { sum_out, factors },
        ) => factorization(sum_out, factors)?,
        (
            RULE_DIRECT | RULE_IRRELEVANT | RULE_STANDARDIZE | RULE_S_ADMISSIBLE | RULE_SINGLETON
            | RULE_TARGET_G,
            _,
        ) => return Err(refuse("the formula is not the shape its rule produces")),
        _ => return Err(refuse("the certificate rule is not an implemented transport rule")),
    }
    Ok(())
}

fn refusal(certificate: &NotCertifiedCertificateWire) -> Result<(), IoError> {
    text(&certificate.reason, "the reason")?;
    text(&certificate.message, "the message")?;
    id_set(&certificate.witness, "the witness")?;
    Ok(())
}

/// Check a decoded identification before it becomes a native proof.
///
/// # Errors
/// An oversized list, a blank text, a repeated variable id, a rule the identifier does not
/// emit, or a formula inconsistent with its rule or its own factors.
pub(super) fn verify(value: &TransportIdentificationWire) -> Result<(), IoError> {
    match value {
        TransportIdentificationWire::Transportable { formula, certificate } => {
            transportable(formula, certificate)
        }
        TransportIdentificationWire::NotCertified(certificate)
        | TransportIdentificationWire::MissingEvidence(certificate) => refusal(certificate),
    }
}
