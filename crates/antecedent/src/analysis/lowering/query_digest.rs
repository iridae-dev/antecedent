//! Canonical digests of a study's query and of the inputs its identification reads.
//!
//! [`query_digest`] keys a typed functional target: two queries of one functional
//! kind (different interventions, levels, conditioning or evaluation points) get
//! different lowered identities. Every field is written through the query's own
//! canonical encoding, which writes set-valued inputs in a canonical order.
//!
//! [`retention_key`] keys the identification a lowered program retains: dispatch
//! installs a retained product only into a study whose key matches.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{CanonicalEncode, CanonicalWriter, CausalQuery, VariableId};
use antecedent_data::TableView;

use crate::Study;
use crate::analysis::builder::DataInput;

const QUERY_DIGEST_TAG: &str = "antecedent.lowered_query";
const RETENTION_KEY_TAG: &str = "antecedent.retained_identification_key";

/// Canonical digest of `query`.
pub(crate) fn query_digest(query: &CausalQuery) -> [u8; 32] {
    let mut writer = CanonicalWriter::new(QUERY_DIGEST_TAG);
    query.encode(&mut writer);
    writer.finish()
}

/// Digest of every study input identification reads: the query (with its effect
/// modifiers), the data modality and schema variables, the structure of any graph
/// class or the atoms and weights of a graph posterior, the tier background, the
/// completion cap and class prior, the selection diagram and transport trial, the
/// identifier, estimator and regression-discontinuity design, and the inference
/// mode.
///
/// `None` when an input has no canonical encoding (a graph whose identity wire
/// cannot be built). Such a study never reuses a retained product.
pub(crate) fn retention_key(study: &Study) -> Option<[u8; 32]> {
    let mut writer = CanonicalWriter::new(RETENTION_KEY_TAG);
    writer.bytes(&query_digest(&study.query));
    data(&mut writer, &study.data);
    // The supplied structure of any class. A graph-posterior study holds a
    // placeholder here that an atom study rebinds, so it is written when it has an
    // identity; the posterior's own atoms are written below.
    writer.text(study.graph.class().as_str());
    match crate::analysis::contract::accepted_graph_identity(&study.graph) {
        Ok(graph) => {
            writer.word(1).bytes(&antecedent_io::to_cbor(&graph).ok()?);
        }
        Err(_) if study.graph_posterior.is_some() => {
            writer.word(0);
        }
        Err(_) => return None,
    }
    match study.graph_posterior.as_ref() {
        Some(posterior) => {
            writer
                .word(1)
                .word(posterior.n_vars as u64)
                .word(posterior.n_graphs as u64)
                .text(posterior.atom_kind.as_str())
                .opt_word(posterior.max_lag.map(u64::from))
                .reals(&posterior.weights);
            writer.word(posterior.graph_keys.len() as u64);
            for key in posterior.graph_keys.iter() {
                writer.word(*key);
            }
        }
        None => {
            writer.word(0);
        }
    }
    match study.tiered.as_ref() {
        Some(background) => {
            writer.word(1);
            background.encode(&mut writer);
        }
        None => {
            writer.word(0);
        }
    }
    writer.opt_word(study.max_completions.map(|cap| cap as u64));
    match study.class_prior.as_ref() {
        Some(prior) => {
            writer.word(1).bytes(&antecedent_io::to_cbor(&prior.identity_wire()).ok()?);
        }
        None => {
            writer.word(0);
        }
    }
    match study.selection_diagram.as_ref() {
        Some(diagram) => {
            let graph = antecedent_io::admg_identity(diagram.causal_graph()).ok()?;
            writer.word(1).bytes(&antecedent_io::to_cbor(&graph).ok()?);
            variable_set(&mut writer, diagram.selection_targets());
        }
        None => {
            writer.word(0);
        }
    }
    match study.transport_trial.as_ref() {
        Some(trial) => {
            writer.word(1);
            trial.trial.encode(&mut writer);
            trial.selection_probability.encode(&mut writer);
            trial.treatment_probability.encode(&mut writer);
        }
        None => {
            writer.word(0);
        }
    }
    match study.rd.as_ref() {
        Some(rd) => {
            writer.word(1);
            rd.running_variable.encode(&mut writer);
            writer.real(rd.cutoff).real(rd.bandwidth);
        }
        None => {
            writer.word(0);
        }
    }
    writer
        .opt_text(study.identifier.as_ref().map(crate::IdentifierId::as_str))
        .opt_text(study.estimator.as_ref().map(crate::EstimatorId::as_str))
        .opt_text(study.estimator_spec.as_deref().map(|spec| spec.id().as_str()))
        .word(u64::from(study.inference.is_bayesian()));
    Some(writer.finish())
}

/// The data modality and the schema's variable ids. Identification reads the
/// variables a schema declares, never the values.
fn data(writer: &mut CanonicalWriter, data: &DataInput) {
    let (modality, schema) = match data {
        DataInput::Tabular(data) => ("tabular", data.schema()),
        DataInput::Temporal(data) => ("temporal", data.schema()),
        DataInput::Event(data) => ("event", data.schema()),
        DataInput::MultiEnv(data) => ("multi_environment", data.schema()),
        DataInput::Panel(data) => ("panel", data.schema()),
    };
    writer.text(modality).word(schema.variables().len() as u64);
    for variable in schema.variables() {
        variable.id.encode(writer);
    }
}

fn variable_set(writer: &mut CanonicalWriter, variables: &[VariableId]) {
    let mut sorted = variables.to_vec();
    sorted.sort_unstable();
    writer.word(sorted.len() as u64);
    for variable in &sorted {
        variable.encode(writer);
    }
}
