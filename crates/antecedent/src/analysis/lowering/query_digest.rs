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
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{CanonicalEncode, CanonicalWriter, CausalQuery, VariableId};
use antecedent_data::TableView;

use crate::Study;
use crate::analysis::builder::DataInput;
#[cfg(test)]
use crate::analysis::execute::{StudyDeclarations, StudyInference};

const QUERY_DIGEST_TAG: &str = "antecedent.lowered_query";
const RETENTION_KEY_TAG: &str = "antecedent.retained_identification_key";
const POSTERIOR_MARKS_TAG: &str = "antecedent.posterior_circle_marks.v1";

/// Canonical digest of `query`.
pub(crate) fn query_digest(query: &CausalQuery) -> [u8; 32] {
    let mut writer = CanonicalWriter::new(QUERY_DIGEST_TAG);
    query.encode(&mut writer);
    writer.finish()
}

/// Whether the encoded query contains a runtime callback, including nested expressions.
pub(crate) fn has_runtime_callback(query: &CausalQuery) -> bool {
    let mut writer = CanonicalWriter::new(QUERY_DIGEST_TAG);
    query.encode(&mut writer);
    writer.has_runtime_callback()
}

/// Digest of every study input identification reads: the query (with its effect
/// modifiers), the data modality and schema variables, the structure of any graph
/// class or the atoms and weights of a graph posterior, the tier background, the
/// completion cap and class prior, the selection diagram and transport trial, the
/// lag-0 average effect a series build compiled, the identifier, estimator and
/// regression-discontinuity design, and the inference mode. A posterior whose atoms carry
/// circle marks also writes them; one without marks is keyed by its adjacency sets alone.
///
/// `None` when an input has no canonical encoding (a graph whose identity wire
/// cannot be built). Such a study never reuses a retained product.
pub(crate) fn retention_key(study: &Study) -> Option<[u8; 32]> {
    if has_runtime_callback(&study.query) {
        return None;
    }
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
    graph_posterior(&mut writer, study.graph_posterior.as_ref());
    match study.declared.tiered.as_ref() {
        Some(background) => {
            writer.word(1);
            background.encode(&mut writer);
        }
        None => {
            writer.word(0);
        }
    }
    writer.opt_word(study.declared.max_completions.map(|cap| cap as u64));
    match study.declared.class_prior.as_ref() {
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
    // The average effect a series build compiled into its lag-0 pulse contrast: identification
    // is retained for that compiled query, and the executed contract discloses this reading.
    match study.lag_zero_average_effect.as_ref() {
        Some(average) => {
            writer.word(1);
            average.encode(&mut writer);
        }
        None => {
            writer.word(0);
        }
    }
    writer
        .opt_text(study.declared.identifier.as_ref().map(crate::IdentifierId::as_str))
        .opt_text(study.declared.estimator.as_ref().map(crate::EstimatorId::as_str))
        .opt_text(study.declared.estimator_spec.as_deref().map(|spec| spec.id().as_str()))
        .word(u64::from(study.settings.inference.is_bayesian()));
    Some(writer.finish())
}

/// The atoms and weights of a graph posterior, `0` without one. Circle marks distinguish PAG
/// atoms of one adjacency set. They are written only when some atom carries one, after a tag no
/// word before them can be mistaken for, so a posterior without marks is keyed by its adjacency
/// sets alone.
fn graph_posterior(
    writer: &mut CanonicalWriter,
    posterior: Option<&antecedent_discovery::GraphPosterior>,
) {
    let Some(posterior) = posterior else {
        writer.word(0);
        return;
    };
    writer
        .word(1)
        .word(posterior.n_vars() as u64)
        .word(posterior.n_graphs() as u64)
        .text(posterior.atom_kind().as_str())
        .opt_word(posterior.max_lag().map(u64::from))
        .reals(posterior.weights());
    writer.word(posterior.graph_keys().len() as u64);
    for key in posterior.graph_keys().iter() {
        writer.word(*key);
    }
    if let Some(marks) = posterior.mark_atoms()
        && marks.iter().any(|mark| !mark.is_empty())
    {
        writer.text(POSTERIOR_MARKS_TAG).word(marks.len() as u64);
        for mark in marks.iter() {
            writer.word(mark.key());
        }
    }
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

/// The audit of what identification reads.
///
/// Every field of [`Study`] and its shared configuration is named here and classified. Adding a
/// field to either struct stops this function compiling until the new field is classified: either
/// [`retention_key`] writes it, or the comment says why identification never reads it. That is
/// what keeps a retained identification from being served to a study that differs in an input it
/// reads.
#[cfg(test)]
#[allow(
    dead_code,
    reason = "compiled for its exhaustive destructuring; calling it would assert nothing at run time"
)]
fn audit_identification_inputs(study: &Study) {
    let crate::analysis::execute::StudyConfig {
        // Not read by identification: caller-facing notes and sinks.
        refute_default_downgrade: _,
        data_diagnostics: _,
        latency_bootstrap_not_applied: _,
        stage_sink: _,
        // Estimation options: identification chooses an estimand, never reads these.
        estimator_spec_identity: _,
        response_options: _,
        observation_options: _,
        draw_controls: _,
        observation_delayed_entry: _,
        // Written by `retention_key`.
        rd: _,
        selection_diagram: _,
        transport_trial: _,
        lag_zero_average_effect: _,
        // Development-build disclosure; never part of a study's identity.
        would_refuse: _,
        // Read only when the evidence of separately estimated outcomes is judged for overlap.
        evidence_rows: _,
        // A built study never carries this: the build refuses it.
        continuous_cell: _,
        // A structure search and a mechanism fit are executed after identification and neither
        // changes the estimand it selects.
        discovery: _,
        mechanisms: _,
        posterior_provenance: _,
    } = &*study.config;
    let Study {
        // Written by `retention_key`.
        data: _,
        graph: _,
        graph_posterior: _,
        query: _,
        // Shared configuration, destructured above.
        config: _,
        // Derived from the query and graph at build.
        structure_source: _,
        support_status: _,
        declared,
        settings,
        latency_mode: _,
        interference: _,
        shared_batch_design: _,
        // Operational evidence capture does not change the statistical program.
        retain_response_witnesses: _,
        temporal_validation_budget: _,
        counterfactual_selection: _,
        // The product of identification, not an input to it.
        identification: _,
    } = study;
    let StudyDeclarations {
        // Written by `retention_key`.
        class_prior: _,
        max_completions: _,
        identifier: _,
        estimator: _,
        estimator_spec: _,
        tiered: _,
        // Refutation, validation and presentation choices run after identification.
        split: _,
        custom_validators: _,
        // Caller learner factories enter the estimator identity, not the query's.
        learner_factories: _,
        // Declared policy limits are checked at estimation; they enter the lowered program
        // through a model slot of the policy-value cell, never identification.
        binary_policy_constraints: _,
        // A declared randomized adjustment is a choice of estimator; it enters the lowered
        // program through the method and covariate of the randomized cell's outcome slot.
        randomized_adjustment: _,
        // Estimation-stage declarations: overlap, population, row binding, missing values and
        // sampling design.
        overlap_policy: _,
        population_registry: _,
        // Reported beside the result; its weights are in the registry the query is digested with.
        weighted_target: _,
        weighted_transport: _,
        transport_outcome_regressions: _,
        participants: _,
        missing: _,
        sampling_design: _,
        score_dependence: _,
    } = declared;
    let StudyInference {
        // Written by `retention_key`.
        inference: _,
        // Refutation and resampling choices run after identification.
        refute: _,
        bootstrap_replicates: _,
        retain_joint_horizons: _,
    } = settings;
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use antecedent_core::{
        AverageEffectQuery, CanonicalWriter, CausalQuery, ContinuousDomain, GridSpec,
        ResponseFunctional, ResponseQuery, VariableId,
    };
    use antecedent_data::TabularData;
    use antecedent_discovery::adjacency_mask::AdjacencyMask;
    use antecedent_discovery::{GraphPosterior, GraphPosteriorAtomKind, SamplingSummary};
    use antecedent_graph::{Admg, SelectionDiagram};

    use super::{graph_posterior, query_digest, retention_key, variable_set};
    use crate::identity_laws::{all_distinct, bit_exact, ordered_laws, set_laws};
    use crate::{BayesianConfig, InferenceMode, RefuteSuite, Study};

    const T: usize = 0;
    const Y: usize = 1;
    const Z: usize = 2;

    /// A one-atom PAG posterior over `t, y, z` whose adjacency is `z * t -> y` with both bits
    /// of `z, t` set, and whose circle marks are `marks`.
    fn study(marks: Option<AdjacencyMask>) -> Study {
        let mut adjacency = AdjacencyMask::EMPTY;
        adjacency.set_edge(3, Z, T, true);
        adjacency.set_edge(3, T, Z, true);
        adjacency.set_edge(3, T, Y, true);
        let summary = SamplingSummary {
            ess: 1.0,
            diagnostics: antecedent_prob::InferenceDiagnostics::analytic("retention_key_marks"),
            rejected_invalid: 0,
        };
        let mut posterior = GraphPosterior::from_atoms(
            3,
            vec![1.0],
            vec![adjacency],
            vec![0.0; 9],
            vec![0.0; 9],
            summary,
        )
        .unwrap()
        .with_atom_kind(GraphPosteriorAtomKind::Pag);
        if let Some(marks) = marks {
            posterior = posterior.with_mark_atoms([marks]).unwrap();
        }
        let t: Vec<f64> = (0..40).map(|i| f64::from(i % 7)).collect();
        let y: Vec<f64> = (0..40).map(|i| f64::from((3 * i) % 11)).collect();
        let z: Vec<f64> = (0..40).map(|i| f64::from(i % 2)).collect();
        let data = TabularData::from_f64_columns([
            ("t", t.as_slice()),
            ("y", y.as_slice()),
            ("z", z.as_slice()),
        ])
        .unwrap();
        Study::tabular(data)
            .graph_posterior(posterior)
            .query(CausalQuery::Response(ResponseQuery::new(ResponseFunctional::MeanCurve {
                outcome: VariableId::from_raw(1),
                treatment: ContinuousDomain::new(
                    VariableId::from_raw(0),
                    GridSpec::Values(Arc::from([0.5, 1.5])),
                ),
            })))
            .inference(InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(16)))
            .refute(RefuteSuite::None)
            .bootstrap_replicates(0)
            .build()
            .unwrap()
    }

    #[test]
    fn circle_marks_key_a_posterior_and_absent_marks_change_nothing() {
        let mut circles = AdjacencyMask::EMPTY;
        circles.set_edge(3, Z, T, true);
        circles.set_edge(3, T, Z, true);
        let mut one_circle = AdjacencyMask::EMPTY;
        one_circle.set_edge(3, Z, T, true);
        // `z <-> t -> y`, written with no marks or with empty marks, and `z o-o t -> y`,
        // `z o-> t -> y`: one adjacency set, three classes.
        let bare = retention_key(&study(None)).unwrap();
        assert_eq!(retention_key(&study(Some(AdjacencyMask::EMPTY))).unwrap(), bare);
        let both = retention_key(&study(Some(circles))).unwrap();
        let one = retention_key(&study(Some(one_circle))).unwrap();
        assert_ne!(both, bare);
        assert_ne!(one, bare);
        assert_ne!(one, both);
    }

    #[test]
    fn site_query_digest_frames_the_query() {
        let ate = |treatment: u32, outcome: u32| {
            query_digest(&CausalQuery::AverageEffect(AverageEffectQuery::binary_ate(
                VariableId::from_raw(treatment),
                VariableId::from_raw(outcome),
            )))
        };
        let response = |levels: Vec<f64>| {
            query_digest(&CausalQuery::Response(ResponseQuery::new(
                ResponseFunctional::MeanCurve {
                    outcome: VariableId::from_raw(1),
                    treatment: ContinuousDomain::new(
                        VariableId::from_raw(0),
                        GridSpec::Values(Arc::from(levels)),
                    ),
                },
            )))
        };
        assert_eq!(ate(0, 1), ate(0, 1));
        all_distinct(&[
            ate(0, 1),
            ate(1, 0),
            ate(0, 2),
            response(vec![0.5, 1.5]),
            response(vec![0.5]),
            response(vec![0.5, 2.5]),
        ]);
    }

    #[test]
    fn site_query_digest_selection_targets_are_a_set() {
        // A selection diagram refuses a repeated target, so sorting is the whole
        // canonicalization of the set the retention key writes.
        set_laws(&[0_u32, 2], |targets: &[u32]| {
            let diagram = SelectionDiagram::try_new(
                Admg::with_variables(3),
                targets.iter().copied().map(VariableId::from_raw).collect::<Vec<_>>(),
            )
            .ok()?;
            let mut writer = CanonicalWriter::new("site");
            variable_set(&mut writer, diagram.selection_targets());
            Some(writer.finish())
        });
    }

    /// A DAG posterior over `t, y, z` whose atoms are `(weight, t -> target)`.
    fn posterior(atoms: &[(f64, usize)]) -> Option<GraphPosterior> {
        let summary = SamplingSummary {
            ess: 1.0,
            diagnostics: antecedent_prob::InferenceDiagnostics::analytic("site"),
            rejected_invalid: 0,
        };
        let masks: Vec<AdjacencyMask> = atoms
            .iter()
            .map(|&(_, target)| {
                let mut mask = AdjacencyMask::EMPTY;
                mask.set_edge(3, T, target, true);
                mask
            })
            .collect();
        let weights: Vec<f64> = atoms.iter().map(|&(weight, _)| weight).collect();
        GraphPosterior::from_atoms(3, weights, masks, vec![0.0; 9], vec![0.0; 9], summary).ok()
    }

    fn posterior_digest(posterior: Option<&GraphPosterior>) -> [u8; 32] {
        let mut writer = CanonicalWriter::new("site");
        graph_posterior(&mut writer, posterior);
        writer.finish()
    }

    #[test]
    fn site_query_digest_posterior_atoms_are_ordered() {
        // A retained per-atom identification is index-aligned with the posterior's atoms; with
        // equal weights only the atoms' order tells the two posteriors apart.
        ordered_laws(&[(0.5, Y), (0.5, Z)], |atoms: &[(f64, usize)]| {
            posterior(atoms).map(|posterior| posterior_digest(Some(&posterior)))
        });
        let forward = posterior(&[(0.25, Y), (0.75, Z)]).unwrap();
        let swapped_weights = posterior(&[(0.75, Y), (0.25, Z)]).unwrap();
        all_distinct(&[
            posterior_digest(Some(&forward)),
            posterior_digest(Some(&swapped_weights)),
            posterior_digest(None),
        ]);
        bit_exact(|weight| {
            posterior(&[(weight, Y), (1.0, Z)]).map(|posterior| posterior_digest(Some(&posterior)))
        });
    }
}
