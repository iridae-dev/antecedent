//! The structures of a graph posterior, each compiled as a study of its own structure.
//!
//! Every distinct atom of the posterior becomes a study whose accepted graph is that atom,
//! compiled through the route the compiler uses for that graph kind in a single-graph study:
//!
//! * a directed acyclic atom as a DAG study;
//! * an acyclic directed mixed atom (bidirected marks for latent common causes), and a
//!   contemporaneous temporal ADMG atom, whose identification is the static ADMG functional,
//!   as an ADMG study;
//! * an equivalence-class atom (a CPDAG or a PAG) per completion, the way the class-aware
//!   identified-set route evaluates a class: each DAG member of a CPDAG as a DAG study, each
//!   MAG completion of a PAG as the most-confounded ADMG it allows (sound for identification
//!   in every DAG the MAG represents).
//!
//! The mass bookkeeping follows the structural-weight rules. Posterior weights of DAG and
//! ADMG atoms are probabilities. A class atom's weight is split evenly over its enumerated
//! completions; that split is a completion-enumeration weight, not a probability, so the
//! structures are read as graph-dependent claims and `BayesOverStructures` is refused at
//! compile. Mass on structures under which the quantity is not identified stays unidentified
//! and is never renormalized; enumeration that stops at its cap or budget is refused rather
//! than accounted partially.
//!
//! What the single-graph compiler cannot compile for a kind (lagged temporal atoms, which
//! need a temporal query on temporal data, or a structure its route does not support) is a
//! typed compile-time refusal naming the kind.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_contract::ExternalRefusal;
use antecedent_core::reason_details::preference_compile;
use antecedent_core::{CanonicalWriter, ExecutionContext, SearchLimits, SearchStop};
use antecedent_design::decision_adapters::AdapterError;
use antecedent_design::decision_contract::StructuralPolicy;
use antecedent_discovery::adjacency_mask::{
    admg_from_adjacency_mask, cpdag_from_adjacency_mask, dag_from_adjacency_mask,
    pag_from_adjacency_mask,
};
use antecedent_discovery::{GraphPosterior, GraphPosteriorAtomKind};
use antecedent_graph::{CompletionSampler, CpdagCompletionSampler, DEFAULT_COMPLETION_OPERATIONS};

use super::{
    CANCELLED_STAGE, CompiledStructure, PreferenceRouteError, STAGE_COMPILE,
    StructureIdentification, StructureOutcome, compile_structure,
};
use crate::accepted::AcceptedGraph;
use crate::analysis::lowering::CompileGoal;
use crate::result::StructuralWeightBasis;
use crate::support::StructureSource;
use crate::{CausalError, Study};

/// Most completions of one class atom compiled, the class-aware identification default.
pub(super) const CLASS_COMPLETIONS: usize = 32;

const COMPLETION_KEY_TAG: &str = "antecedent.preference_route.completion_key";

fn cancelled() -> PreferenceRouteError {
    CausalError::Cancelled { stage: CANCELLED_STAGE }.into()
}

fn kind_refusal(kind: &str, supplied: impl Into<String>) -> PreferenceRouteError {
    ExternalRefusal::new(
        antecedent_core::reason_code!("route_not_supported"),
        STAGE_COMPILE,
        preference_compile::ATOM_KIND_NOT_COMPILED,
    )
    .with_offending(format!("graph posterior atoms of kind {kind}"))
    .with_supplied(supplied)
    .with_remedy("decide over a posterior whose atom kind the single-graph compiler compiles")
    .into()
}

fn truncated(kind: &str, stop: Option<SearchStop>) -> PreferenceRouteError {
    let supplied = stop.map_or_else(
        || format!("more than {CLASS_COMPLETIONS} completions"),
        |stop| format!("the completion search stopped at its {stop:?} bound"),
    );
    ExternalRefusal::new(
        antecedent_core::reason_code!("route_not_supported"),
        STAGE_COMPILE,
        preference_compile::CLASS_COMPLETIONS_TRUNCATED,
    )
    .with_offending(format!("a graph posterior atom of kind {kind}"))
    .with_supplied(supplied)
    .with_remedy(
        "orient edges with background knowledge or an intervention, or declare the structure",
    )
    .into()
}

/// The study answering the query under `graph`.
fn study_under(study: &Study, graph: AcceptedGraph) -> Study {
    let mut atom = study.clone();
    atom.graph_posterior = None;
    atom.graph = Arc::new(graph);
    atom.structure_source = StructureSource::Explicit;
    atom.clear_identification();
    atom
}

/// Compile `study` under `graph`. For a kind other than a DAG, a compile refusal saying the
/// route does not support the structure, or that its estimator cannot evaluate it (the
/// mixed-graph functional's discrete plug-in on a continuous column), becomes the kind
/// refusal.
fn compile_under(
    study: &Study,
    graph: AcceptedGraph,
    kind: GraphPosteriorAtomKind,
    goal: &CompileGoal,
    ctx: &ExecutionContext,
) -> Result<StructureOutcome, PreferenceRouteError> {
    match compile_structure(&study_under(study, graph), goal, None, ctx) {
        Err(PreferenceRouteError::Compiler(error))
            if kind != GraphPosteriorAtomKind::Dag
                && (matches!(
                    error.peeled(),
                    CausalError::Unsupported { .. }
                        | CausalError::Support { .. }
                        | CausalError::Estimate(
                            antecedent_estimate::EstimationError::Unsupported { .. }
                        )
                ) || error.reason_code() == Some("route_not_supported")) =>
        {
            Err(kind_refusal(kind.as_str(), error.to_string()))
        }
        other => other,
    }
}

/// A structure that compiled, or one under which the quantity is not identified.
fn structure(
    key: Option<u64>,
    class: Option<u64>,
    weight: f64,
    source: usize,
    outcome: StructureOutcome,
) -> CompiledStructure {
    match outcome {
        Ok((plan, identification, runner)) => CompiledStructure {
            key,
            class,
            weight,
            source,
            identification,
            plan: Some(plan),
            runner: Some(runner),
        },
        Err(why) => CompiledStructure {
            key,
            class,
            weight,
            source,
            identification: StructureIdentification::NotIdentified(why),
            plan: None,
            runner: None,
        },
    }
}

/// The key of completion `index` of the class atom with key `class`.
fn completion_key(class: u64, index: usize) -> u64 {
    let mut writer = CanonicalWriter::new(COMPLETION_KEY_TAG);
    writer.word(class).word(index as u64);
    let bytes = writer.finish();
    let mut word = [0_u8; 8];
    word.copy_from_slice(&bytes[..8]);
    u64::from_le_bytes(word)
}

/// The accepted graphs of the completions of class atom `index`.
fn completions(
    posterior: &GraphPosterior,
    index: usize,
    ctx: &ExecutionContext,
) -> Result<Option<Vec<AcceptedGraph>>, PreferenceRouteError> {
    let kind = posterior.atom_kind();
    let Some(mask) = posterior.atoms().get(index).cloned() else {
        return Ok(None);
    };
    let limits = SearchLimits { operations: DEFAULT_COMPLETION_OPERATIONS, depth: usize::MAX };
    let stopped = |stop: Option<SearchStop>| match stop {
        Some(SearchStop::Cancelled) => cancelled(),
        other => truncated(kind.as_str(), other),
    };
    let graphs = if kind == GraphPosteriorAtomKind::Cpdag {
        let Ok(cpdag) = cpdag_from_adjacency_mask(mask, posterior.n_vars()) else {
            return Ok(None);
        };
        let Ok(mut sampler) =
            CpdagCompletionSampler::with_budget(cpdag, CLASS_COMPLETIONS, limits, ctx)
        else {
            return Ok(None);
        };
        let graphs: Vec<AcceptedGraph> =
            sampler.by_ref().map(|completion| AcceptedGraph::dag(completion.graph)).collect();
        if sampler.hit_cap() || sampler.search_stop().is_some() {
            return Err(stopped(sampler.search_stop()));
        }
        graphs
    } else {
        let mark = posterior.mark_atom(index);
        let Ok(pag) = pag_from_adjacency_mask(mask, mark, posterior.n_vars()) else {
            return Ok(None);
        };
        let Ok(sampler) = CompletionSampler::with_budget(&pag, CLASS_COMPLETIONS, limits, ctx)
        else {
            return Ok(None);
        };
        if sampler.hit_cap() {
            return Err(stopped(sampler.search_stop()));
        }
        let mut graphs = Vec::new();
        for completion in sampler {
            match antecedent_identify::mag_to_confounded_admg(&completion.graph) {
                Some(admg) => graphs.push(AcceptedGraph::admg(admg)),
                None => return Ok(None),
            }
        }
        graphs
    };
    if ctx.cancellation.is_cancelled() {
        return Err(cancelled());
    }
    Ok((!graphs.is_empty()).then_some(graphs))
}

/// The single structure of atom `index` of a DAG or ADMG posterior.
fn atom_graph(posterior: &GraphPosterior, index: usize) -> Option<AcceptedGraph> {
    let mask = posterior.atoms().get(index)?.clone();
    match posterior.atom_kind() {
        GraphPosteriorAtomKind::Dag => {
            dag_from_adjacency_mask(mask, posterior.n_vars()).ok().map(AcceptedGraph::dag)
        }
        GraphPosteriorAtomKind::Admg | GraphPosteriorAtomKind::TemporalAdmg => {
            admg_from_adjacency_mask(mask, posterior.n_vars()).ok().map(AcceptedGraph::admg)
        }
        _ => None,
    }
}

/// Compile every completion of class atom `index` with key `key` and weight `weight`.
fn compile_class(
    study: &Study,
    posterior: &GraphPosterior,
    (index, key, weight): (usize, u64, f64),
    source: usize,
    goal: &CompileGoal,
    ctx: &ExecutionContext,
) -> Result<Vec<CompiledStructure>, PreferenceRouteError> {
    let kind = posterior.atom_kind();
    let Some(graphs) = completions(posterior, index, ctx)? else {
        let why = format!("the {} atom has no completion", kind.as_str());
        return Ok(vec![structure(Some(key), Some(key), weight, source, Err(why))]);
    };
    let share = weight / graphs.len() as f64;
    let mut structures = Vec::with_capacity(graphs.len());
    for (completion, graph) in graphs.into_iter().enumerate() {
        if ctx.cancellation.is_cancelled() {
            return Err(cancelled());
        }
        let outcome = compile_under(study, graph, kind, goal, ctx)?;
        let completion = completion_key(key, completion);
        structures.push(structure(Some(completion), Some(key), share, source, outcome));
    }
    Ok(structures)
}

/// Compile every distinct structure of the study's graph posterior, with the basis its
/// weights are read under.
///
/// # Errors
/// Lagged temporal atoms, a structure the single-graph route does not support for the
/// atom's kind (`preference_compile.atom_kind_not_compiled`); a class whose completions
/// exceed the cap or the search budget (`preference_compile.class_completions_truncated`);
/// `BayesOverStructures` over class atoms, whose completion weights are not probabilities;
/// every other compile refusal; cancellation.
pub(super) fn compile_posterior(
    study: &Study,
    goal: &CompileGoal,
    source: usize,
    policy: StructuralPolicy,
    ctx: &ExecutionContext,
) -> Result<(Vec<CompiledStructure>, StructuralWeightBasis), PreferenceRouteError> {
    let Some(posterior) = study.graph_posterior.as_ref() else {
        return Ok((Vec::new(), StructuralWeightBasis::PosteriorProbability));
    };
    let kind = posterior.atom_kind();
    if posterior.lag_masks().is_some() || posterior.max_lag().is_some() {
        return Err(kind_refusal(
            &format!("lagged {}", kind.as_str()),
            "lagged atoms need a temporal query on temporal data; the decision binds a static \
             mean response curve of tabular data",
        ));
    }
    let class = matches!(kind, GraphPosteriorAtomKind::Cpdag | GraphPosteriorAtomKind::Pag);
    if class && policy == StructuralPolicy::BayesOverStructures {
        let mut refusal =
            AdapterError::CompletionCountNotProbability(kind.as_str().to_owned()).to_refusal();
        refusal.stage = STAGE_COMPILE;
        return Err(refusal.into());
    }
    let total: f64 = posterior.weights().iter().sum();
    let mut structures: Vec<CompiledStructure> = Vec::new();
    for index in 0..posterior.n_graphs() {
        if ctx.cancellation.is_cancelled() {
            return Err(cancelled());
        }
        let key = posterior.graph_keys().get(index).copied();
        let weight = posterior.weights().get(index).copied().unwrap_or(0.0) / total;
        if class {
            let class_key = key.unwrap_or(index as u64);
            let members = structures.iter().filter(|s| s.class == Some(class_key)).count();
            if members > 0 {
                for member in structures.iter_mut().filter(|s| s.class == Some(class_key)) {
                    member.weight += weight / members as f64;
                }
                continue;
            }
            structures.extend(compile_class(
                study,
                posterior,
                (index, class_key, weight),
                source,
                goal,
                ctx,
            )?);
            continue;
        }
        if let Some(existing) = structures.iter_mut().find(|s| s.key == key) {
            existing.weight += weight;
            continue;
        }
        let outcome = match atom_graph(posterior, index) {
            Some(graph) => compile_under(study, graph, kind, goal, ctx)?,
            None => Err(format!("the structure is not a valid {} atom", kind.as_str())),
        };
        structures.push(structure(key, None, weight, source, outcome));
    }
    let basis = if class {
        StructuralWeightBasis::CompletionEnumeration
    } else {
        StructuralWeightBasis::PosteriorProbability
    };
    Ok((structures, basis))
}
