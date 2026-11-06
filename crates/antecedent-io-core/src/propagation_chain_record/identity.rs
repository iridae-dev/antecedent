//! Canonical identity for retained propagation-chain records.

use antecedent_contract::dependence::SharedFactorKind;
use antecedent_contract::propagation_chain::{
    ChainReading, LawMeaning, Representation, StageKind, TableConstruction,
};
use antecedent_core::{CanonicalEncode, CanonicalWriter};
use std::collections::BTreeSet;

use super::{
    ChainChangeWire, ChainFactorWire, Error, PROPAGATION_CHAIN_TAG, PropagationChainRecordWire,
    operator_class, parse, stage_op, word_of,
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
    let mut writer = CanonicalWriter::new(PROPAGATION_CHAIN_TAG);
    writer.word(u64::from(record.version));
    let construction = parse(
        TableConstruction::from_name(&record.construction),
        "construction",
        &record.construction,
    )?;
    let reading = parse(ChainReading::from_name(&record.reading), "reading", &record.reading)?;
    writer.word(construction.tag()).word(reading.tag()).text(&record.output);
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
