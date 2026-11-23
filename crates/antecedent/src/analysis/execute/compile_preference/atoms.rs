//! The structures of a graph posterior, each compiled as a study of its own structure.
//!
//! Every distinct atom of the posterior becomes a study whose accepted graph is that atom,
//! compiled through the route the compiler uses for that graph kind in a single-graph study:
//!
//! * a directed acyclic atom as a DAG study;
//! * an acyclic directed mixed atom (bidirected marks for latent common causes), and a
//!   contemporaneous temporal ADMG atom, whose identification is the static ADMG functional,
//!   as an ADMG study;
//! * an equivalence-class atom (a CPDAG or a PAG) per member, the way the class-aware
//!   identified-set route evaluates a class: each DAG member of a CPDAG as a DAG study, each
//!   MAG member of a PAG as the most-confounded ADMG it allows (sound for identification in
//!   every DAG the MAG represents).
//!
//! The mass bookkeeping follows the structural-weight rules. Posterior weights of DAG and
//! ADMG atoms are probabilities. Without a declared class prior, a class atom's weight is
//! split evenly over its enumerated members; that split is a completion-enumeration weight,
//! not a probability, so the structures are read as graph-dependent claims and
//! `BayesOverStructures` is refused at compile. With a declared [`ClassPrior`] (uniform under
//! a named assumption, or keyed by member fingerprint) a member's weight is its atom's
//! posterior weight times its declared conditional mass: a probability under that declaration,
//! which every policy reads, `BayesOverStructures` included. Mass on structures under which the
//! quantity is not identified stays unidentified and is never renormalized.
//!
//! A class with at most the declared exact limit of members ([`ClassPrior::completion_limit`],
//! 32 by default) is enumerated exactly. A larger class is refused
//! (`preference_compile.class_completions_truncated`) unless a uniform measure declares a
//! sample: members are then drawn uniformly with replacement from the declared seed (a CPDAG by
//! exact clique-picking, a PAG by budget-bounded rejection over its circle endpoints), each
//! distinct member is compiled once and weighted by the share of the draws that produced it,
//! and the draw is recorded ([`ClassSampling`]). Graph-dependent claims need every member and
//! keyed masses need every key, so neither is sampled.
//!
//! What the single-graph compiler cannot compile for a kind (lagged temporal atoms, which
//! need a temporal query on temporal data, or a structure its route does not support) is a
//! typed compile-time refusal naming the kind.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::{BTreeMap, BTreeSet};
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
use antecedent_graph::{
    CompletionSampler, Cpdag, CpdagCompletionSampler, DEFAULT_COMPLETION_OPERATIONS, Dag,
    MecUniform, MecUniformError, Pag, PagUniformError, dag_fingerprint, mag_fingerprint,
    sample_pag_members,
};

use super::structures::{
    CompiledStructure, StructureIdentification, StructureOutcome, compile_structure,
};
use super::{CANCELLED_STAGE, PreferenceRouteError, STAGE_COMPILE};
use crate::accepted::AcceptedGraph;
use crate::analysis::lowering::CompileGoal;
use crate::analysis::preference_decision::{
    ClassMember, ClassSampling, ClassSamplingScheme, ClassSize,
};
use crate::class_prior::{ClassMassMismatch, ClassPrior, ClassSamplingRequest};
use crate::result::StructuralWeightBasis;
use crate::support::StructureSource;
use crate::{CausalError, Study};

const MEMBER_KEY_TAG: &str = "antecedent.preference_route.class_member_key";

/// Operations a class sample may charge: four times the enumeration budget, since every
/// accepted PAG member is audited against a full m-separation model.
const SAMPLING_OPERATIONS: usize = 4 * DEFAULT_COMPLETION_OPERATIONS;

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

/// Above the exact limit with no uniform declared measure that draws a sample.
fn truncated(kind: &str, supplied: impl Into<String>) -> PreferenceRouteError {
    ExternalRefusal::new(
        antecedent_core::reason_code!("route_not_supported"),
        STAGE_COMPILE,
        preference_compile::CLASS_COMPLETIONS_TRUNCATED,
    )
    .with_offending(format!("a graph posterior atom of kind {kind}"))
    .with_expected(
        "at most the exact limit of members, or a uniform declared class prior that draws a \
         sample",
    )
    .with_supplied(supplied)
    .with_remedy(
        "declare ClassPrior::uniform(assumption).sample(members, seed), raise its exact limit, \
         orient edges with background knowledge or an intervention, or declare the structure",
    )
    .into()
}

fn over_limit(limit: usize, stop: Option<SearchStop>) -> String {
    stop.map_or_else(
        || format!("more than {limit} members and no sample declared under a uniform measure"),
        |stop| format!("the member search stopped at its {stop:?} bound"),
    )
}

fn keys_mismatch(kind: &str, supplied: impl Into<String>) -> PreferenceRouteError {
    ExternalRefusal::new(
        antecedent_core::reason_code!("invalid_argument"),
        STAGE_COMPILE,
        preference_compile::CLASS_PRIOR_KEYS_MISMATCH,
    )
    .with_offending(format!("the class prior declared over {kind} atoms"))
    .with_expected("one mass for every member of every class atom, keyed by member fingerprint")
    .with_supplied(supplied)
    .with_remedy(
        "key the masses by the member fingerprints the compile reports, or declare a uniform \
         measure under a named assumption",
    )
    .into()
}

fn sampling_needs_uniform() -> PreferenceRouteError {
    ExternalRefusal::new(
        antecedent_core::reason_code!("invalid_argument"),
        STAGE_COMPILE,
        preference_compile::CLASS_PRIOR_SAMPLING_NEEDS_UNIFORM,
    )
    .with_offending("the class prior's declared sample")
    .with_expected("a uniform measure: keyed masses need every member's key")
    .with_supplied("a keyed or ordered class prior with a sample")
    .with_remedy("declare ClassPrior::uniform(assumption) to sample, or drop the sample")
    .into()
}

fn sampling_budget(kind: &str, supplied: impl Into<String>) -> PreferenceRouteError {
    ExternalRefusal::new(
        antecedent_core::reason_code!("route_not_supported"),
        STAGE_COMPILE,
        preference_compile::CLASS_SAMPLING_BUDGET,
    )
    .with_offending(format!("a graph posterior atom of kind {kind}"))
    .with_expected("the declared number of uniform member draws within the search budget")
    .with_supplied(supplied)
    .with_remedy(
        "draw fewer members, orient edges with background knowledge or an intervention, or \
         declare the structure",
    )
    .into()
}

/// The study answering the query under `graph`.
fn study_under(study: &Study, graph: AcceptedGraph) -> Study {
    let mut atom = study.clone();
    atom.graph_posterior = None;
    atom.graph = Arc::new(graph);
    atom.structure_source = StructureSource::Explicit;
    atom.declared.class_prior = None;
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
    (key, class): (Option<u64>, Option<u64>),
    weight: f64,
    source: usize,
    member: Option<ClassMember>,
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
            member,
        },
        Err(why) => CompiledStructure {
            key,
            class,
            weight,
            source,
            identification: StructureIdentification::NotIdentified(why),
            plan: None,
            runner: None,
            member,
        },
    }
}

/// The key of the member with content fingerprint `fingerprint` of the class atom with key
/// `class`: independent of the order the members were enumerated or drawn in.
fn member_key(class: u64, fingerprint: u64) -> u64 {
    let mut writer = CanonicalWriter::new(MEMBER_KEY_TAG);
    writer.word(class).word(fingerprint);
    let bytes = writer.finish();
    let mut word = [0_u8; 8];
    word.copy_from_slice(&bytes[..8]);
    u64::from_le_bytes(word)
}

/// One distinct member of a class atom: its graph, content fingerprint and how many draws
/// produced it (one when enumerated).
struct Member {
    graph: AcceptedGraph,
    fingerprint: u64,
    multiplicity: usize,
}

/// The members of one class atom, and the draw that produced them when it was sampled.
struct ClassMembers {
    members: Vec<Member>,
    sampling: Option<Arc<ClassSampling>>,
}

/// Distinct items in first-draw order with their multiplicities.
fn tally<T>(drawn: Vec<T>, fingerprint: impl Fn(&T) -> u64) -> Vec<(T, u64, usize)> {
    let mut slots: BTreeMap<u64, usize> = BTreeMap::new();
    let mut out: Vec<(T, u64, usize)> = Vec::new();
    for item in drawn {
        let print = fingerprint(&item);
        if let Some(&slot) = slots.get(&print) {
            out[slot].2 += 1;
        } else {
            slots.insert(print, out.len());
            out.push((item, print, 1));
        }
    }
    out
}

fn sampling_record(
    class_key: u64,
    scheme: ClassSamplingScheme,
    (class_size, attempts): (ClassSize, Option<u64>),
    (request, distinct): (ClassSamplingRequest, usize),
    prior: &ClassPrior,
) -> Arc<ClassSampling> {
    Arc::new(ClassSampling {
        class_key,
        scheme,
        class_size,
        drawn: request.members,
        distinct,
        seed: request.seed,
        attempts,
        assumption: prior.assumption().map(str::to_owned),
    })
}

/// Uniform draws of the members of a CPDAG class, counted exactly by clique-picking.
fn sample_cpdag(
    cpdag: &Cpdag,
    class_key: u64,
    (request, prior): (ClassSamplingRequest, &ClassPrior),
    ctx: &ExecutionContext,
) -> Result<ClassMembers, PreferenceRouteError> {
    let kind = GraphPosteriorAtomKind::Cpdag.as_str();
    let limits = SearchLimits { operations: SAMPLING_OPERATIONS, depth: usize::MAX };
    let refused = |error: MecUniformError| match error {
        MecUniformError::Stopped(SearchStop::Cancelled) => cancelled(),
        MecUniformError::Stopped(_)
        | MecUniformError::CountOverflow
        | MecUniformError::OrderNotDrawn => sampling_budget(kind, error.to_string()),
        MecUniformError::ConflictEdge
        | MecUniformError::NotChordal
        | MecUniformError::NotEssential
        | MecUniformError::MemberCheckFailed => truncated(
            kind,
            format!(
                "the atom is not the essential graph of a class, so it is not sampled: {error}"
            ),
        ),
    };
    let class = MecUniform::new(cpdag, limits, ctx).map_err(refused)?;
    let drawn: Vec<Dag> =
        class.sample_members(request.members, request.seed, class_key).map_err(refused)?;
    let distinct = tally(drawn, dag_fingerprint);
    let sampling = sampling_record(
        class_key,
        ClassSamplingScheme::CliquePicking,
        (ClassSize::Exact(class.class_size()), None),
        (request, distinct.len()),
        prior,
    );
    let members = distinct
        .into_iter()
        .map(|(dag, fingerprint, multiplicity)| Member {
            graph: AcceptedGraph::dag(dag),
            fingerprint,
            multiplicity,
        })
        .collect();
    Ok(ClassMembers { members, sampling: Some(sampling) })
}

/// Uniform draws of the members of a PAG class within the sampling budget.
fn sample_pag(
    pag: &Pag,
    reference: &Pag,
    (class_key, represented, size_exact): (u64, usize, bool),
    (request, prior): (ClassSamplingRequest, &ClassPrior),
    ctx: &ExecutionContext,
) -> Result<Option<ClassMembers>, PreferenceRouteError> {
    let kind = GraphPosteriorAtomKind::Pag.as_str();
    let limits = SearchLimits { operations: SAMPLING_OPERATIONS, depth: usize::MAX };
    let sample =
        sample_pag_members(pag, reference, request.members, (request.seed, class_key), limits, ctx)
            .map_err(|error| match error {
                PagUniformError::BudgetExhausted { stop: SearchStop::Cancelled, .. } => cancelled(),
                PagUniformError::BudgetExhausted { .. } | PagUniformError::AuditUnaffordable => {
                    sampling_budget(kind, error.to_string())
                }
                PagUniformError::ReferenceNotMember | PagUniformError::AmbiguousClass => truncated(
                    kind,
                    format!("the atom's members are not one certified class: {error}"),
                ),
            })?;
    let attempts = sample.attempts;
    let distinct = tally(sample.members, mag_fingerprint);
    let represented = u128::try_from(represented.max(distinct.len())).unwrap_or(u128::MAX);
    let class_size =
        if size_exact { ClassSize::Exact(represented) } else { ClassSize::AtLeast(represented) };
    let sampling = sampling_record(
        class_key,
        ClassSamplingScheme::CircleEndpointRejection,
        (class_size, Some(attempts)),
        (request, distinct.len()),
        prior,
    );
    let mut members = Vec::with_capacity(distinct.len());
    for (mag, fingerprint, multiplicity) in distinct {
        let Some(admg) = antecedent_identify::mag_to_confounded_admg(&mag) else {
            return Ok(None);
        };
        members.push(Member { graph: AcceptedGraph::admg(admg), fingerprint, multiplicity });
    }
    Ok(Some(ClassMembers { members, sampling: Some(sampling) }))
}

fn enumerated(members: Vec<(AcceptedGraph, u64)>) -> ClassMembers {
    let members = members
        .into_iter()
        .map(|(graph, fingerprint)| Member { graph, fingerprint, multiplicity: 1 })
        .collect();
    ClassMembers { members, sampling: None }
}

/// The members of class atom `index` with key `class_key`: every member when there are at
/// most the exact limit, else a uniform draw when the class prior declares one. `None` when
/// the atom has no member.
fn class_members(
    posterior: &GraphPosterior,
    (index, class_key): (usize, u64),
    prior: Option<&ClassPrior>,
    ctx: &ExecutionContext,
) -> Result<Option<ClassMembers>, PreferenceRouteError> {
    let kind = posterior.atom_kind();
    let Some(mask) = posterior.atoms().get(index).cloned() else {
        return Ok(None);
    };
    let limit =
        prior.map_or(crate::class_prior::DEFAULT_CLASS_EXACT_LIMIT, ClassPrior::completion_limit);
    let request = prior.and_then(|prior| prior.sampling().map(|request| (request, prior)));
    let limits = SearchLimits { operations: DEFAULT_COMPLETION_OPERATIONS, depth: usize::MAX };
    let members = if kind == GraphPosteriorAtomKind::Cpdag {
        let Ok(cpdag) = cpdag_from_adjacency_mask(mask, posterior.n_vars()) else {
            return Ok(None);
        };
        let Ok(mut sampler) =
            CpdagCompletionSampler::with_budget(cpdag.clone(), limit + 1, limits, ctx)
        else {
            return Ok(None);
        };
        let dags: Vec<Dag> = sampler.by_ref().map(|completion| completion.graph).collect();
        let stop = sampler.search_stop();
        if stop == Some(SearchStop::Cancelled) {
            return Err(cancelled());
        }
        if !sampler.hit_cap() && stop.is_none() && dags.len() <= limit {
            enumerated(
                dags.into_iter()
                    .map(|dag| {
                        let print = dag_fingerprint(&dag);
                        (AcceptedGraph::dag(dag), print)
                    })
                    .collect(),
            )
        } else {
            let Some(declared) = request else {
                return Err(truncated(kind.as_str(), over_limit(limit, stop)));
            };
            sample_cpdag(&cpdag, class_key, declared, ctx)?
        }
    } else {
        let mark = posterior.mark_atom(index);
        let Ok(pag) = pag_from_adjacency_mask(mask, mark, posterior.n_vars()) else {
            return Ok(None);
        };
        let Ok(sampler) = CompletionSampler::with_budget(&pag, limit + 1, limits, ctx) else {
            return Ok(None);
        };
        let report = sampler.validation_report();
        let stop = sampler.search_stop();
        let over = sampler.hit_cap();
        let mags: Vec<Pag> = sampler.map(|completion| completion.graph).collect();
        if stop == Some(SearchStop::Cancelled) {
            return Err(cancelled());
        }
        if !over && mags.len() <= limit {
            let mut members = Vec::with_capacity(mags.len());
            for mag in mags {
                let print = mag_fingerprint(&mag);
                match antecedent_identify::mag_to_confounded_admg(&mag) {
                    Some(admg) => members.push((AcceptedGraph::admg(admg), print)),
                    None => return Ok(None),
                }
            }
            enumerated(members)
        } else {
            let Some(declared) = request else {
                return Err(truncated(kind.as_str(), over_limit(limit, stop)));
            };
            let Some(reference) = mags.first() else {
                return Err(truncated(kind.as_str(), over_limit(limit, stop)));
            };
            let size_exact = stop.is_none()
                && !report.equivalence_audit_skipped
                && !report.ambiguous_global_class;
            let facts = (class_key, report.represented_completions, size_exact);
            let Some(sampled) = sample_pag(&pag, reference, facts, declared, ctx)? else {
                return Ok(None);
            };
            sampled
        }
    };
    if ctx.cancellation.is_cancelled() {
        return Err(cancelled());
    }
    Ok((!members.members.is_empty()).then_some(members))
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

/// The conditional mass of each member within its class: the share of the draws for a
/// sampled class, the declared masses for a class prior, an even split otherwise.
fn conditional_masses(
    class: &ClassMembers,
    prior: Option<&ClassPrior>,
    kind: &str,
) -> Result<Vec<f64>, PreferenceRouteError> {
    if let Some(sampling) = &class.sampling {
        let drawn = sampling.drawn.max(1) as f64;
        return Ok(class.members.iter().map(|m| m.multiplicity as f64 / drawn).collect());
    }
    let Some(prior) = prior else {
        let share = 1.0 / class.members.len().max(1) as f64;
        return Ok(vec![share; class.members.len()]);
    };
    let prints: Vec<u64> = class.members.iter().map(|m| m.fingerprint).collect();
    prior.conditional_masses(&prints).map_err(|mismatch| {
        keys_mismatch(
            kind,
            match mismatch {
                ClassMassMismatch::Ordered => {
                    "masses in enumeration order, which name no member of a graph-posterior \
                     class"
                        .to_owned()
                }
                ClassMassMismatch::Missing(print) => {
                    format!("no mass for the member with fingerprint {print}")
                }
                ClassMassMismatch::ZeroTotal => {
                    "masses that sum to zero over the members of one class".to_owned()
                }
            },
        )
    })
}

/// Compile every member of class atom `index` with key `key` and weight `weight`; returns
/// the structures and each one's conditional mass within the class.
fn compile_class(
    study: &Study,
    posterior: &GraphPosterior,
    (index, key, weight): (usize, u64, f64),
    (source, prior): (usize, Option<&ClassPrior>),
    goal: &CompileGoal,
    ctx: &ExecutionContext,
) -> Result<(Vec<CompiledStructure>, Vec<f64>), PreferenceRouteError> {
    let kind = posterior.atom_kind();
    let Some(class) = class_members(posterior, (index, key), prior, ctx)? else {
        let why = format!("the {} atom has no completion", kind.as_str());
        let lone = structure((Some(key), Some(key)), weight, source, None, Err(why));
        return Ok((vec![lone], vec![1.0]));
    };
    let shares = conditional_masses(&class, prior, kind.as_str())?;
    let mut structures = Vec::with_capacity(class.members.len());
    for (member, share) in class.members.into_iter().zip(shares.iter().copied()) {
        if ctx.cancellation.is_cancelled() {
            return Err(cancelled());
        }
        let outcome = compile_under(study, member.graph, kind, goal, ctx)?;
        let declared = ClassMember {
            fingerprint: member.fingerprint,
            multiplicity: member.multiplicity,
            sampling: class.sampling.clone(),
        };
        let keys = (Some(member_key(key, member.fingerprint)), Some(key));
        structures.push(structure(keys, weight * share, source, Some(declared), outcome));
    }
    Ok((structures, shares))
}

/// Compile every distinct structure of the study's graph posterior, with the basis its
/// weights are read under.
///
/// # Errors
/// Lagged temporal atoms, a structure the single-graph route does not support for the
/// atom's kind (`preference_compile.atom_kind_not_compiled`); a class with more members than
/// the exact limit (or whose member search stopped at its budget) without a uniform declared
/// measure that draws a sample (`preference_compile.class_completions_truncated`); a sample
/// the budget cannot draw (`preference_compile.class_sampling_budget`); a keyed class prior
/// with a missing or extra member key, or ordered masses
/// (`preference_compile.class_prior_keys_mismatch`); a sample declared on a measure that is not
/// uniform (`preference_compile.class_prior_sampling_needs_uniform`); `BayesOverStructures`
/// over class atoms without a declared class prior, whose completion weights are not
/// probabilities; every other compile refusal; cancellation.
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
    let prior = if class { study.declared.class_prior.as_ref() } else { None };
    if class && prior.is_none() && policy == StructuralPolicy::BayesOverStructures {
        let mut refusal =
            AdapterError::CompletionCountNotProbability(kind.as_str().to_owned()).to_refusal();
        refusal.stage = STAGE_COMPILE;
        return Err(refusal.into());
    }
    if prior.is_some_and(|prior| prior.sampling().is_some() && !prior.is_uniform()) {
        return Err(sampling_needs_uniform());
    }
    let total: f64 = posterior.weights().iter().sum();
    let mut structures: Vec<CompiledStructure> = Vec::new();
    // Where each class atom's members start, with their conditional masses, so a repeated
    // atom adds its weight in the same proportions.
    let mut classes: BTreeMap<u64, (usize, Vec<f64>)> = BTreeMap::new();
    for index in 0..posterior.n_graphs() {
        if ctx.cancellation.is_cancelled() {
            return Err(cancelled());
        }
        let key = posterior.graph_keys().get(index).copied();
        let weight = posterior.weights().get(index).copied().unwrap_or(0.0) / total;
        if class {
            let class_key = key.unwrap_or(index as u64);
            if let Some((start, shares)) = classes.get(&class_key) {
                for (offset, share) in shares.iter().enumerate() {
                    if let Some(member) = structures.get_mut(start + offset) {
                        member.weight += weight * share;
                    }
                }
                continue;
            }
            let start = structures.len();
            let (compiled, shares) = compile_class(
                study,
                posterior,
                (index, class_key, weight),
                (source, prior),
                goal,
                ctx,
            )?;
            structures.extend(compiled);
            classes.insert(class_key, (start, shares));
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
        structures.push(structure((key, None), weight, source, None, outcome));
    }
    if let Some(keys) = prior.and_then(ClassPrior::keys) {
        let matched: BTreeSet<u64> = structures
            .iter()
            .filter_map(|s| s.member.as_ref().map(|member| member.fingerprint))
            .collect();
        let extra: Vec<u64> = keys.into_iter().filter(|key| !matched.contains(key)).collect();
        if !extra.is_empty() {
            return Err(keys_mismatch(
                kind.as_str(),
                format!("masses for fingerprints {extra:?}, which are no member of any class atom"),
            ));
        }
    }
    let basis = match (class, prior.is_some()) {
        (true, true) => StructuralWeightBasis::PosteriorWithDeclaredClassPrior,
        (true, false) => StructuralWeightBasis::CompletionEnumeration,
        (false, _) => StructuralWeightBasis::PosteriorProbability,
    };
    Ok((structures, basis))
}
