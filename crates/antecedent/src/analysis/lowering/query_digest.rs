//! Canonical digests of a study's query and of the inputs its identification reads.
//!
//! [`query_digest`] keys a typed functional target: two queries of one functional
//! kind (different interventions, levels, conditioning or evaluation points) get
//! different lowered identities. Interventions, populations, outcome functionals
//! and temporal policies are written through their own canonical encodings. Fields
//! with no canonical encoding of their own (grids, weightings, observation
//! declarations, row selectors, design arrays) are written as their `Debug`
//! rendering, which is deterministic for these plain value types.
//!
//! [`retention_key`] keys the identification a lowered program retains: dispatch
//! installs a retained product only into a study whose key matches.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{
    AverageEffectQuery, CanonicalEncode, CanonicalWriter, CausalQuery, Intervention,
    ResponseFunctional, ResponseQuery, VariableId,
};

use crate::Study;

const QUERY_DIGEST_TAG: &str = "antecedent.lowered_query";
const RETENTION_KEY_TAG: &str = "antecedent.retained_identification_key";

/// Canonical digest of `query`.
pub(crate) fn query_digest(query: &CausalQuery) -> [u8; 32] {
    let mut writer = CanonicalWriter::new(QUERY_DIGEST_TAG);
    writer.text(&format!("{:?}", query.kind()));
    match query {
        CausalQuery::AverageEffect(query) => average(&mut writer, query),
        CausalQuery::ConditionalEffect(query) => average(&mut writer, &query.inner),
        CausalQuery::TemporalEffect(query) => {
            writer.word(u64::from(query.treatment.raw())).word(u64::from(query.outcome.raw()));
            query.policy.encode(&mut writer);
            query.control.encode(&mut writer);
            query.active.encode(&mut writer);
            writer
                .word(u64::from(query.horizon_steps))
                .opt_word(query.max_history_lag.map(u64::from));
            query.target_population.encode(&mut writer);
            variables(&mut writer, &query.effect_modifiers);
        }
        CausalQuery::Distribution(query) => {
            variables(&mut writer, &query.outcomes);
            interventions(&mut writer, &query.interventions);
            variables(&mut writer, &query.conditioning);
            query.target_population.encode(&mut writer);
        }
        CausalQuery::PathSpecific(query) => {
            writer.word(u64::from(query.treatment.raw())).word(u64::from(query.outcome.raw()));
            variables(&mut writer, &query.path_nodes);
            query.control.encode(&mut writer);
            query.active.encode(&mut writer);
            query.target_population.encode(&mut writer);
            writer.word(query.max_paths as u64).word(query.max_len as u64);
        }
        CausalQuery::Counterfactual(query) => {
            variables(&mut writer, &query.outcomes);
            interventions(&mut writer, &query.interventions);
            query.control.encode(&mut writer);
            writer.word(u64::from(query.allow_nested));
        }
        CausalQuery::NestedCounterfactual(query) => {
            writer
                .word(u64::from(query.treatment.raw()))
                .word(u64::from(query.mediator.raw()))
                .word(u64::from(query.outcome.raw()))
                .real(query.control_value())
                .real(query.active_value());
        }
        CausalQuery::Mediation(query) => {
            writer.word(u64::from(query.treatment.raw())).word(u64::from(query.outcome.raw()));
            variables(&mut writer, &query.mediators);
            writer.text(&format!("{:?}", query.contrast));
            query.control.encode(&mut writer);
            query.active.encode(&mut writer);
            query.target_population.encode(&mut writer);
            writer.word(query.horizons.len() as u64);
            for horizon in query.horizons.iter() {
                writer.word(u64::from(*horizon));
            }
        }
        CausalQuery::Response(query) => response(&mut writer, query),
        // Attribution, change, matching and design queries hold plain values only.
        other => {
            writer.text(&format!("{other:?}"));
        }
    }
    writer.finish()
}

/// Digest of the study inputs identification reads: the query, the structure (one
/// graph, or the atoms and weights of a graph posterior), the identifier, the
/// completion cap and the inference mode.
pub(crate) fn retention_key(study: &Study) -> [u8; 32] {
    let mut writer = CanonicalWriter::new(RETENTION_KEY_TAG);
    writer.bytes(&query_digest(&study.query)).text(study.graph.class().as_str());
    if let Some(dag) = study.graph.as_dag() {
        writer.word(dag.nodes().len() as u64);
        for node in dag.nodes() {
            writer.word(u64::from(node.variable().raw()));
        }
        edges(&mut writer, &dag.directed_edge_pairs_sorted());
    } else if let Some(dag) = study.graph.as_temporal_dag() {
        writer.word(dag.nodes().len() as u64);
        for node in dag.nodes() {
            writer.word(u64::from(node.variable().raw()));
        }
        edges(&mut writer, &dag.directed_edge_pairs_sorted());
    }
    writer
        .opt_text(study.identifier.as_ref().map(|identifier| identifier.as_str()))
        .opt_word(study.max_completions.map(|cap| cap as u64))
        .word(u64::from(study.inference.is_bayesian()));
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
    writer.finish()
}

fn average(writer: &mut CanonicalWriter, query: &AverageEffectQuery) {
    writer.word(u64::from(query.treatment.raw())).word(u64::from(query.outcome.raw()));
    variables(writer, &query.effect_modifiers);
    query.control.encode(writer);
    query.active.encode(writer);
    query.target_population.encode(writer);
    query.outcome_functional.encode(writer);
}

fn response(writer: &mut CanonicalWriter, query: &ResponseQuery) {
    match &query.functional {
        ResponseFunctional::InterventionResponse { outcome, interventions: actions } => {
            writer.text("intervention_response").word(u64::from(outcome.raw()));
            interventions(writer, actions);
        }
        functional => {
            // Grids, weightings, evaluation points and directions are plain values.
            writer.text(&format!("{functional:?}"));
        }
    }
    query.target_population.encode(writer);
    query.outcome_functional.encode(writer);
    writer
        .text(&format!("{:?}", query.observation))
        .text(&format!("{:?}", query.observation_assumptions))
        .text(&format!("{:?}", query.temporal));
}

fn variables(writer: &mut CanonicalWriter, variables: &[VariableId]) {
    writer.word(variables.len() as u64);
    for variable in variables {
        variable.encode(writer);
    }
}

fn interventions(writer: &mut CanonicalWriter, interventions: &[Intervention]) {
    writer.word(interventions.len() as u64);
    for intervention in interventions {
        intervention.encode(writer);
    }
}

fn edges(writer: &mut CanonicalWriter, edges: &[(u32, u32)]) {
    writer.word(edges.len() as u64);
    for &(from, to) in edges {
        writer.word(u64::from(from)).word(u64::from(to));
    }
}
