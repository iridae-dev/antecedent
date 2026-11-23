//! Canonical identity for retained propagation-chain records.

use antecedent_contract::ScientificQuantity;
use antecedent_contract::dependence::SharedFactorKind;
use antecedent_contract::propagation_chain::{
    ChainReading, LawMeaning, Representation, StageKind, TableConstruction,
};
use antecedent_core::{CanonicalEncode, CanonicalWriter};
use std::collections::BTreeSet;

use super::{
    ChainChangeWire, ChainDeferredTableWire, ChainFactorWire, Error,
    PROPAGATION_CHAIN_ASSUMPTION_TAG, PROPAGATION_CHAIN_ASSUMPTION_VERSION,
    PROPAGATION_CHAIN_DEFERRED_TAG, PROPAGATION_CHAIN_JOINT_LAW_TAG,
    PROPAGATION_CHAIN_JOINT_LAW_VERSION, PROPAGATION_CHAIN_LAW_TAG, PROPAGATION_CHAIN_LAW_VERSION,
    PROPAGATION_CHAIN_TAG, PropagationChainRecordWire, operator_class, parse, stage_op, word_of,
};

fn write_factors(writer: &mut CanonicalWriter, factors: &[ChainFactorWire]) -> Result<(), Error> {
    let mut canonical: BTreeSet<(u64, String)> = BTreeSet::new();
    for factor in factors {
        let kind = parse(SharedFactorKind::from_name(&factor.kind), "factor kind", &factor.kind)?;
        canonical.insert((factor_kind_tag(kind), factor.id.clone()));
    }
    writer.word(u64::try_from(canonical.len()).unwrap_or(u64::MAX));
    for (tag, id) in &canonical {
        writer.word(*tag).text(id);
    }
    Ok(())
}

/// Canonical tag of a factor kind. Append-only.
const fn factor_kind_tag(kind: SharedFactorKind) -> u64 {
    match kind {
        SharedFactorKind::Row => 1,
        SharedFactorKind::Cluster => 2,
        SharedFactorKind::Model => 3,
        SharedFactorKind::Parameter => 4,
        SharedFactorKind::Prior => 5,
        SharedFactorKind::Disturbance => 6,
        SharedFactorKind::Period => 7,
        _ => 0,
    }
}

/// The deferred family table's declaration: its construction, rows, seed, slots, the
/// within-study blocks and the declared cross-study blocks.
fn write_deferred(writer: &mut CanonicalWriter, deferred: &ChainDeferredTableWire) {
    writer
        .text(&deferred.construction)
        .word(deferred.rows)
        .word(deferred.seed)
        .texts(&deferred.slots)
        .word(word_of(deferred.within_blocks.len()));
    for group in &deferred.within_blocks {
        writer.texts(group);
    }
    writer.word(word_of(deferred.declared_cross.len()));
    for block in &deferred.declared_cross {
        writer.word(block.first).word(block.second).word(word_of(block.terms.len()));
        for term in &block.terms {
            writer.word(*term);
        }
    }
}

/// Identity of everything the record rests on, excluding its results.
///
/// The stream is a [`CanonicalWriter`] under [`PROPAGATION_CHAIN_TAG`]. Every
/// enum enters as its append-only numeric tag, every real as its bit pattern,
/// and the factors of a source or stage as a sorted set, so the order a caller
/// listed them in does not reach the digest.
///
/// # Errors
/// A stored name that is not one of the format's names.
pub fn propagation_chain_identity(record: &PropagationChainRecordWire) -> Result<String, Error> {
    let tag = if record.deferred.is_some() {
        // A draft whose family table is bound at execution is another kind of object than any
        // sealed record: its identity lives in its own domain.
        PROPAGATION_CHAIN_DEFERRED_TAG
    } else if record.version == PROPAGATION_CHAIN_JOINT_LAW_VERSION {
        PROPAGATION_CHAIN_JOINT_LAW_TAG
    } else if record.version == PROPAGATION_CHAIN_ASSUMPTION_VERSION {
        PROPAGATION_CHAIN_ASSUMPTION_TAG
    } else if record.version == PROPAGATION_CHAIN_LAW_VERSION {
        PROPAGATION_CHAIN_LAW_TAG
    } else {
        PROPAGATION_CHAIN_TAG
    };
    let mut writer = CanonicalWriter::new(tag);
    writer.word(u64::from(record.version));
    let construction = parse(
        TableConstruction::from_name(&record.construction),
        "construction",
        &record.construction,
    )?;
    let reading = parse(ChainReading::from_name(&record.reading), "reading", &record.reading)?;
    writer.word(construction.tag()).word(reading.tag()).text(&record.output);
    if matches!(
        record.version,
        PROPAGATION_CHAIN_LAW_VERSION
            | PROPAGATION_CHAIN_ASSUMPTION_VERSION
            | PROPAGATION_CHAIN_JOINT_LAW_VERSION
    ) {
        if let Some(wire) = &record.output_quantity {
            let quantity: ScientificQuantity = wire
                .clone()
                .try_into()
                .map_err(|message| Error::Malformed(format!("output quantity: {message}")))?;
            writer.word(1).text(&quantity.canonical_identity());
        } else {
            writer.word(0);
        }
    }
    if record.version == PROPAGATION_CHAIN_JOINT_LAW_VERSION {
        writer.word(word_of(record.joint_outputs.len()));
        for output in &record.joint_outputs {
            let quantity: ScientificQuantity =
                output.quantity.clone().try_into().map_err(|message| {
                    Error::Malformed(format!("joint output quantity: {message}"))
                })?;
            writer.text(&output.stage).text(&quantity.canonical_identity());
        }
    }
    writer.word(word_of(record.sources.len()));
    for source in &record.sources {
        let representation = parse(
            Representation::from_name(&source.representation),
            "representation",
            &source.representation,
        )?;
        let meaning = match &source.meaning {
            Some(name) => Some(parse(LawMeaning::from_name(name), "law meaning", name)?.tag()),
            None => None,
        };
        writer
            .text(&source.id)
            .word(representation.tag())
            .opt_word(meaning)
            .opt_text(source.alignment.as_deref())
            .word(u64::from(source.complete_disclosure));
        write_factors(&mut writer, &source.factors)?;
        writer.word(word_of(source.values.len()));
        for value in &source.values {
            writer.word(*value);
        }
    }
    if let Some(deferred) = &record.deferred {
        write_deferred(&mut writer, deferred);
    }
    writer.word(word_of(record.stages.len()));
    for stage in &record.stages {
        let kind = parse(StageKind::from_name(&stage.kind), "stage kind", &stage.kind)?;
        writer
            .text(&stage.id)
            .word(kind.tag())
            .word(operator_class(&stage.op).tag())
            .texts(&stage.inputs);
        write_factors(&mut writer, &stage.own_factors)?;
        stage_op(&stage.op)?.encode(&mut writer);
        writer.opt_word(stage.decision_reference);
    }
    writer.word(word_of(record.changes.len()));
    for change in &record.changes {
        match change {
            ChainChangeWire::Withdraw { source } => {
                writer.word(1).text(source);
            }
            ChainChangeWire::Reweight { source, log_weights } => {
                writer.word(2).text(source).word(word_of(log_weights.len()));
                for weight in log_weights {
                    writer.word(*weight);
                }
            }
        }
    }
    writer.text(&record.label);
    Ok(writer.finish_hex())
}
