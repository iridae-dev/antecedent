//! Generalized adjustment for CPDAG/PAG classes.
//!
//! Identification over a PAG streams bounded completions and aggregates an
//! [`IdentificationEnvelope`], preserving unidentified mass.
//!
//! # Search completeness
//!
//! Per MAG completion, adjustment sets are searched among subsets of
//! `An({T,Y}) ∪ (De(T) \ Forb(T,Y))` excluding `Forb(T,Y) ∪ {T,Y}`, where
//! `Forb(T,Y) = De(cn(T,Y))` and `cn(T,Y) = De(T) ∩ An(Y) \ {T}` (nodes on
//! proper causal paths from `T` to `Y`). Each subset is tested for m-separation of
//! `T` and `Y` in `G_{\underline{T}}` (outgoing edges from `T` removed). Enumeration is
//! by increasing set size and stops at the first valid set (minimal-first). Completions
//! that are not MAGs, or MAGs with no qualifying set in this candidate family, contribute
//! unidentified mass. A completion whose candidate family exceeds `max_candidates` is
//! folded into unidentified mass with status [`IdentificationStatus::NotIdentified`]
//! (the 1.0 public surface; there is no third identification outcome). Enumeration
//! was never attempted, so this is not a scientific open-back-door: the result
//! carries an Execution diagnostic [`CAPPED_COMPLETION_DIAGNOSTIC_CODE`] and is
//! counted in [`IdentificationEnvelope::truncated_completions`]. A completed
//! search that finds no set is also [`IdentificationStatus::NotIdentified`], but
//! with a Scientific diagnostic.
//!
//! This is **generalized adjustment**, not the full ID/IDC algorithm (see roadmap P5.3).
//! Sets outside the ancestor candidate family are not searched.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::many_single_char_names,
    clippy::too_many_arguments
)]

use std::sync::Arc;

use antecedent_core::{
    AssumptionSet, AverageEffectQuery, CausalQuery, Diagnostic, DiagnosticKind, DiagnosticSeverity,
    ResponseQuery, Value, VariableId,
};
use antecedent_expr::CausalExprArena;
use antecedent_graph::{
    Admg, BitSet, CompletionSampler, Cpdag, CpdagCompletionSampler, DSeparationWorkspace,
    DenseNodeId, Endpoint, Pag,
};

use crate::backdoor::BackdoorIdentifier;
use crate::identifier::IdentificationWorkspace;

use crate::envelope::{
    GraphFeature, GraphIdentificationCase, IdentificationEnvelope, ProbabilityMass,
};
use crate::error::IdentificationError;
use crate::result::{
    DerivationTrace, IdentificationPerformanceRecord, IdentificationResult, IdentificationStatus,
    IdentifiedEstimand,
};

/// Diagnostic code attached to a per-completion [`IdentificationResult`] when adjustment-set
/// enumeration was capped by `max_candidates` before it could search — as opposed to
/// searching exhaustively and finding no valid set. A cap keeps
/// [`IdentificationStatus::NotIdentified`] for the 1.0 freeze and is an
/// Execution diagnostic, not a scientific open-back-door. Both cases still fold
/// into [`IdentificationEnvelope::unidentified_weight`] (unidentified mass is
/// preserved either way); [`IdentificationEnvelope::truncated_completions`]
/// counts only this diagnostic.
pub const CAPPED_COMPLETION_DIAGNOSTIC_CODE: &str =
    "identify.generalized_adjustment.completion_capped";

/// Config for PAG generalized-adjustment envelopes.
#[derive(Clone, Debug)]
pub struct GeneralizedAdjustmentConfig {
    /// Max completions to stream (hard bound; never retain unbounded).
    pub max_completions: usize,
    /// Uniform weight per streamed completion.
    pub per_completion_weight: f64,
    /// Max candidate covariates to enumerate (bitmask width).
    pub max_candidates: usize,
}

impl Default for GeneralizedAdjustmentConfig {
    fn default() -> Self {
        Self { max_completions: 32, per_completion_weight: 1.0, max_candidates: 16 }
    }
}

/// Class-aware identifier for PAGs via completion envelopes.
#[derive(Clone, Debug, Default)]
pub struct GeneralizedAdjustmentIdentifier {
    /// Config.
    pub config: GeneralizedAdjustmentConfig,
}

impl GeneralizedAdjustmentIdentifier {
    /// Default identifier.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Identify an average effect over a PAG by streaming completions.
    ///
    /// # Errors
    ///
    /// Query type unsupported or graph errors.
    pub fn identify_pag_envelope(
        &self,
        pag: &Pag,
        query: &AverageEffectQuery,
    ) -> Result<IdentificationEnvelope<Pag>, IdentificationError> {
        let t = query.treatment;
        let y = query.outcome;
        let t_d = pag_var_to_dense(pag, t)?;
        let y_d = pag_var_to_dense(pag, y)?;
        let (active, control) = match (&query.active, &query.control) {
            (
                antecedent_core::Intervention::Set { value: active, .. },
                antecedent_core::Intervention::Set { value: control, .. },
            ) => (active.clone(), control.clone()),
            _ => {
                return Err(IdentificationError::UnsupportedQuery {
                    message: "generalized adjustment ATE requires Set interventions",
                });
            }
        };

        self.pag_envelope_with(pag, |completion| {
            let mut result = identify_on_mag_completion(
                completion,
                t,
                y,
                t_d,
                y_d,
                active.clone(),
                control.clone(),
                self.config.max_candidates,
            )?;
            result.query = CausalQuery::AverageEffect(query.clone());
            if let (Some(admg), Some(cut)) =
                (mag_to_admg(completion), proper_backdoor_mag(completion, t_d, y_d))
            {
                validate_conditional_adjustment(
                    &admg,
                    &cut,
                    completion.nodes(),
                    query,
                    &mut result,
                    &crate::backdoor::AdjustmentSearchConfig {
                        max_candidates: self.config.max_candidates,
                        ..Default::default()
                    },
                )?;
            }
            Ok(result)
        })
    }

    pub(crate) fn pag_envelope_with(
        &self,
        pag: &Pag,
        mut identify: impl FnMut(&Pag) -> Result<IdentificationResult, IdentificationError>,
    ) -> Result<IdentificationEnvelope<Pag>, IdentificationError> {
        let mut sampler = CompletionSampler::new(pag.clone(), self.config.max_completions)
            .map_err(IdentificationError::from)?;
        let mut cases = Vec::new();
        let w = ProbabilityMass(self.config.per_completion_weight);
        for completion in sampler.by_ref() {
            let result = identify(&completion.graph)?;
            cases.push(GraphIdentificationCase { graph: completion.graph, result, weight: w });
        }
        let mut envelope = IdentificationEnvelope::from_cases(cases);
        envelope.push_features(pag_circle_features(pag));
        let validation = sampler.validation_report();
        envelope.push_features([GraphFeature {
            kind: Arc::from("pag_completion_validation"),
            detail: Arc::from(format!(
                "exhaustively audited {} endpoint assignment(s): represented={}, \
                 rejected_non_ancestral={}, rejected_nonmaximal={}, \
                 rejected_local_incompatible={}, ambiguous_global_class={}, \
                 equivalence_audit_skipped={}",
                validation.assignments_examined,
                validation.represented_completions,
                validation.rejected_non_ancestral,
                validation.rejected_nonmaximal,
                validation.rejected_local_incompatible,
                validation.ambiguous_global_class,
                validation.equivalence_audit_skipped,
            )),
        }]);
        // The global class audit is exponential in node count and is skipped on larger
        // graphs. Local validity still holds for every case, but "identified in every
        // member of the equivalence class" is exactly the claim the skipped audit would
        // have supported, so it is downgraded here rather than asserted.
        if sampler.class_audit_incomplete() {
            envelope.push_features([GraphFeature {
                kind: Arc::from("completion_equivalence_audit_skipped"),
                detail: Arc::from(
                    "the global m-separation equivalence audit was too expensive for this graph; \
                     completions are locally valid but not certified to form one Markov class",
                ),
            }]);
            if envelope.status == IdentificationStatus::NonparametricallyIdentified {
                envelope.status = IdentificationStatus::PartiallyIdentified;
            }
        }
        // `NonparametricallyIdentified` on this path asserts identification for *every* member
        // of the equivalence class. When the sampler hit its retention cap, identification
        // ran only on a deterministic low-mask prefix, so the assertion is unearned — a
        // verified but unretained completion may well be unidentified. Downgrade rather than
        // overclaim.
        if sampler.hit_cap() {
            envelope.push_features([GraphFeature {
                kind: Arc::from("completion_enumeration_capped"),
                detail: Arc::from(format!(
                    "retained {} of {} verified MAG completion(s) under max_completions={}; \
                     identification is established only over the deterministic retained subset",
                    envelope.cases.len(),
                    validation.represented_completions,
                    self.config.max_completions
                )),
            }]);
            if envelope.status == IdentificationStatus::NonparametricallyIdentified {
                envelope.status = IdentificationStatus::PartiallyIdentified;
            }
        }
        Ok(envelope)
    }

    /// Identify an average effect over a CPDAG by streaming MEC DAG completions.
    ///
    /// Each completion is identified with ordinary backdoor search. Completions
    /// that do not identify contribute unidentified mass. The runtime class of
    /// the source graph stays `Cpdag`; a caller who completed the graph
    /// themselves holds a `Dag`.
    ///
    /// # Errors
    ///
    /// Query type unsupported, conflict marks, or graph errors.
    pub fn identify_cpdag_envelope(
        &self,
        cpdag: &Cpdag,
        query: &AverageEffectQuery,
    ) -> Result<IdentificationEnvelope<antecedent_graph::Dag>, IdentificationError> {
        match (&query.active, &query.control) {
            (
                antecedent_core::Intervention::Set { .. },
                antecedent_core::Intervention::Set { .. },
            ) => {}
            _ => {
                return Err(IdentificationError::UnsupportedQuery {
                    message: "CPDAG envelope ATE requires Set interventions",
                });
            }
        }

        let backdoor = BackdoorIdentifier::new().with_max_candidates(self.config.max_candidates);
        let mut workspace = IdentificationWorkspace::default();
        let cq = CausalQuery::AverageEffect(query.clone());
        self.cpdag_envelope_with(cpdag, |graph| {
            let prepared = backdoor.prepare(graph)?;
            backdoor.identify(&prepared, &cq, &mut workspace)
        })
    }

    pub(crate) fn cpdag_envelope_with(
        &self,
        cpdag: &Cpdag,
        mut identify: impl FnMut(
            &antecedent_graph::Dag,
        ) -> Result<IdentificationResult, IdentificationError>,
    ) -> Result<IdentificationEnvelope<antecedent_graph::Dag>, IdentificationError> {
        let mut sampler = CpdagCompletionSampler::new(cpdag.clone(), self.config.max_completions)?;
        let mut cases = Vec::new();
        let w = ProbabilityMass(self.config.per_completion_weight);
        for completion in sampler.by_ref() {
            let result = identify(&completion.graph)?;
            cases.push(GraphIdentificationCase { graph: completion.graph, result, weight: w });
        }
        let mut envelope = IdentificationEnvelope::from_cases(cases);
        envelope.push_features(cpdag_undirected_features(cpdag));
        if sampler.hit_cap() {
            envelope.push_features([GraphFeature {
                kind: Arc::from("completion_enumeration_capped"),
                detail: Arc::from(format!(
                    "retained {} MEC DAG completion(s) under max_completions={}; \
                     identification is established only over the deterministic retained subset",
                    envelope.cases.len(),
                    self.config.max_completions
                )),
            }]);
            if envelope.status == IdentificationStatus::NonparametricallyIdentified {
                envelope.status = IdentificationStatus::PartiallyIdentified;
            }
        }
        Ok(envelope)
    }
}

fn cpdag_undirected_features(cpdag: &Cpdag) -> Vec<GraphFeature> {
    let n = cpdag.undirected_edge_count();
    if n == 0 {
        return Vec::new();
    }
    vec![GraphFeature {
        kind: Arc::from("cpdag_undirected_marks"),
        detail: Arc::from(format!("{n} undirected edge(s) in source CPDAG")),
    }]
}

fn pag_circle_features(pag: &Pag) -> Vec<GraphFeature> {
    let review = antecedent_graph::PagReview::from_pag(pag.clone(), "generalized.adjustment");
    if review.pending_circles.is_empty() {
        return Vec::new();
    }
    vec![GraphFeature {
        kind: Arc::from("pag_circle_marks"),
        detail: Arc::from(format!(
            "{} edge(s) with circle endpoints in source PAG",
            review.pending_circles.len()
        )),
    }]
}

pub(crate) fn pag_var_to_dense(
    pag: &Pag,
    id: VariableId,
) -> Result<DenseNodeId, IdentificationError> {
    for (i, node) in pag.nodes().iter().enumerate() {
        if let antecedent_graph::NodeRef::Static(v) = node {
            if *v == id {
                return Ok(DenseNodeId::from_raw(u32::try_from(i).expect("fit")));
            }
        }
    }
    Err(IdentificationError::UnknownVariable { id })
}

pub(crate) fn validate_dag_conditional_adjustment(
    dag: &antecedent_graph::Dag,
    query: &AverageEffectQuery,
    result: &mut IdentificationResult,
    config: &crate::backdoor::AdjustmentSearchConfig,
) -> Result<(), IdentificationError> {
    if query.effect_modifiers.is_empty() {
        return Ok(());
    }
    let mut admg = Admg::with_variables(dag.node_count() as u32);
    for edge in dag.edges() {
        if let Some((from, to)) = edge.parent_child() {
            admg.insert_directed(from, to)?;
        }
    }
    let t = dag
        .nodes()
        .iter()
        .position(|node| *node == antecedent_graph::NodeRef::Static(query.treatment))
        .ok_or(IdentificationError::UnknownVariable { id: query.treatment })?;
    validate_conditional_adjustment(
        &admg,
        &mutilate_outgoing(&admg, DenseNodeId::from_raw(t as u32)),
        dag.nodes(),
        query,
        result,
        config,
    )
}

/// The regression conditions on Z union W. A marginal adjustment certificate
/// for Z alone does not license this conditional regression: W may be a
/// mediator, or may open a collider path. This is a sufficient backdoor check,
/// not a complete conditional-ID algorithm.
fn validate_conditional_adjustment(
    graph: &Admg,
    mutilated: &Admg,
    nodes: &[antecedent_graph::NodeRef],
    query: &AverageEffectQuery,
    result: &mut IdentificationResult,
    config: &crate::backdoor::AdjustmentSearchConfig,
) -> Result<(), IdentificationError> {
    if query.effect_modifiers.is_empty() {
        return Ok(());
    }
    let dense = |v: VariableId| {
        nodes
            .iter()
            .position(|node| *node == antecedent_graph::NodeRef::Static(v))
            .map(|i| DenseNodeId::from_raw(i as u32))
            .ok_or(IdentificationError::UnknownVariable { id: v })
    };
    let t = dense(query.treatment)?;
    let y = dense(query.outcome)?;
    let modifiers =
        query.effect_modifiers.iter().copied().map(dense).collect::<Result<Vec<_>, _>>()?;
    let descendants = directed_closure(graph, &[t], false);
    let pretreatment = modifiers.iter().all(|w| *w != y && !descendants.contains(*w));
    let mut ws = DSeparationWorkspace::default();
    let mut valid = Vec::new();
    if pretreatment {
        for estimand in &result.estimands {
            let mut conditioning = estimand
                .adjustment_set
                .iter()
                .copied()
                .map(dense)
                .collect::<Result<Vec<_>, _>>()?;
            conditioning.extend_from_slice(&modifiers);
            conditioning.sort_unstable();
            conditioning.dedup();
            if mutilated.is_m_separated(t, y, &conditioning, &mut ws)? {
                valid.push(estimand.clone());
            }
        }
    }
    if valid.is_empty() && pretreatment {
        let found =
            constrained_conditional_set(graph, mutilated, nodes, query, &modifiers, config)?;
        match found {
            ConditionalSearch::Found(adjustments, examined) => {
                for adjustment in adjustments {
                    let (active, control) = conditional_levels(query)?;
                    let functional = result.arena.backdoor_ate(
                        query.treatment,
                        query.outcome,
                        &adjustment,
                        active,
                        control,
                    );
                    valid.push(IdentifiedEstimand::backdoor(
                        "backdoor.adjustment",
                        adjustment.into(),
                        functional,
                    ));
                }
                result.status = IdentificationStatus::NonparametricallyIdentified;
                result.query = CausalQuery::AverageEffect(query.clone());
                result.performance.candidates_examined += examined;
                result.performance.sets_returned = valid.len() as u64;
                result.diagnostics.retain(|d| d.code.as_ref() != CAPPED_COMPLETION_DIAGNOSTIC_CODE);
                result.derivation.push("conditional.adjustment.search", "searched with the fixed pre-treatment modifier in every separation test; alternative joint conditioning set certified");
                if result.required_assumptions.entries.is_empty() {
                    result
                        .required_assumptions
                        .push(crate::assumptions::causal_markov("conditional.adjustment"));
                }
            }
            ConditionalSearch::Capped(n) => {
                *result = capped_completion_result(
                    CausalQuery::AverageEffect(query.clone()),
                    n,
                    config.max_candidates,
                );
                return Ok(());
            }
            ConditionalSearch::Absent => {}
        }
    }
    if valid.is_empty() {
        *result = not_identified(
            result.query.clone(),
            "conditional adjustment not certified: modifiers must be pre-treatment and \
             Z union modifiers must block backdoor paths; marginal ATE identification \
             alone is insufficient; constrained adjustment search found no set (general conditional ID not attempted)",
        );
        result.diagnostics.push(Diagnostic::new(
            "identify.conditional.adjustment_unverified",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Warning,
            "conditional adjustment is not certified on this completion; its mass is retained as unidentified",
        ));
    } else {
        result.estimands = valid;
    }
    Ok(())
}

fn conditional_levels(query: &AverageEffectQuery) -> Result<(Value, Value), IdentificationError> {
    match (&query.active, &query.control) {
        (
            antecedent_core::Intervention::Set { value: a, .. },
            antecedent_core::Intervention::Set { value: c, .. },
        ) => Ok((a.clone(), c.clone())),
        _ => Err(IdentificationError::unsupported(
            "conditional adjustment requires Set interventions",
        )),
    }
}

enum ConditionalSearch {
    Found(Vec<Vec<VariableId>>, u64),
    Capped(usize),
    Absent,
}

fn constrained_conditional_set(
    graph: &Admg,
    cut: &Admg,
    nodes: &[antecedent_graph::NodeRef],
    query: &AverageEffectQuery,
    modifiers: &[DenseNodeId],
    config: &crate::backdoor::AdjustmentSearchConfig,
) -> Result<ConditionalSearch, IdentificationError> {
    let dense = |v| {
        nodes
            .iter()
            .position(|node| *node == antecedent_graph::NodeRef::Static(v))
            .map(|i| DenseNodeId::from_raw(i as u32))
            .ok_or(IdentificationError::UnknownVariable { id: v })
    };
    let t = dense(query.treatment)?;
    let y = dense(query.outcome)?;
    let mut seeds = vec![t, y];
    seeds.extend_from_slice(modifiers);
    let ancestors = directed_closure(graph, &seeds, true);
    let descendants = directed_closure(graph, &[t], false);
    let candidates: Vec<_> = nodes
        .iter()
        .enumerate()
        .filter_map(|(i, node)| {
            let antecedent_graph::NodeRef::Static(variable) = node else {
                return None;
            };
            let id = DenseNodeId::from_raw(i as u32);
            let outside_history = config.max_history_lag.is_some_and(|cap| {
                config.history_lags.iter().any(|(v, lag)| v == variable && *lag > cap)
            });
            (id != y
                && !descendants.contains(id)
                && ancestors.contains(id)
                && !modifiers.contains(&id)
                && !config.forbidden.contains(variable)
                && !outside_history)
                .then_some(id)
        })
        .collect();
    if candidates.len() > config.max_candidates {
        return Ok(ConditionalSearch::Capped(candidates.len()));
    }
    if config.max_results == 0 {
        return Ok(ConditionalSearch::Absent);
    }
    let mut ws = DSeparationWorkspace::default();
    let mut found: Vec<Vec<DenseNodeId>> = Vec::new();
    let mut examined = 0;
    let sizes: Vec<_> = if config.maximal_only && !config.minimal_only {
        (0..=candidates.len()).rev().collect()
    } else {
        (0..=candidates.len()).collect()
    };
    for size in sizes {
        let mut error = None;
        crate::enum_masks::for_each_mask_of_size(&candidates, size, |z| {
            if config.minimal_only && found.iter().any(|old| old.iter().all(|v| z.contains(v))) {
                return false;
            }
            if config.maximal_only && found.iter().any(|old| z.iter().all(|v| old.contains(v))) {
                return false;
            }
            examined += 1;
            let mut conditioned = z.to_vec();
            conditioned.extend_from_slice(modifiers);
            match cut.is_m_separated(t, y, &conditioned, &mut ws) {
                Ok(true) => {
                    found.push(z.to_vec());
                    found.len() >= config.max_results
                }
                Ok(false) => false,
                Err(e) => {
                    error = Some(IdentificationError::from(e));
                    true
                }
            }
        });
        if let Some(e) = error {
            return Err(e);
        }
        if found.len() >= config.max_results {
            break;
        }
    }
    let mut sets: Vec<Vec<VariableId>> = found
        .iter()
        .map(|set| {
            set.iter()
                .map(|id| match nodes[id.as_usize()] {
                    antecedent_graph::NodeRef::Static(v) => v,
                    _ => unreachable!("static graph"),
                })
                .collect()
        })
        .collect();
    rank_conditional_sets(&mut sets, config);
    Ok(if sets.is_empty() {
        ConditionalSearch::Absent
    } else {
        ConditionalSearch::Found(sets, examined)
    })
}

fn rank_conditional_sets(
    sets: &mut [Vec<VariableId>],
    config: &crate::backdoor::AdjustmentSearchConfig,
) {
    if !config.measurement_costs.is_empty() {
        let cost = |set: &[VariableId]| {
            set.iter()
                .map(|v| {
                    config
                        .measurement_costs
                        .iter()
                        .find(|(id, _)| id == v)
                        .map_or(1.0, |(_, cost)| *cost)
                })
                .sum::<f64>()
        };
        sets.sort_by(|a, b| cost(a).total_cmp(&cost(b)));
    }
}

// Perkovic et al. (2018), Definition 6 and Theorem 7. A MAG arrow need not
// exclude hidden confounding. Delete only visible first edges of causal paths;
// keep side-effect edges so conditioning cannot conceal collider bias.
fn proper_backdoor_mag(mag: &Pag, t: DenseNodeId, y: DenseNodeId) -> Option<Admg> {
    let graph = mag_to_admg(mag)?;
    let ancestors_y = directed_closure(&graph, &[y], true);
    let causal_children: Vec<_> =
        graph.children(t).iter().copied().filter(|v| ancestors_y.contains(*v)).collect();
    if causal_children.iter().any(|&v| !crate::joint_response::visible(mag, t, v)) {
        return None;
    }
    let mut cut = Admg::with_variables(graph.node_count() as u32);
    for i in 0..graph.node_count() {
        let a = DenseNodeId::from_raw(i as u32);
        for &b in graph.children(a) {
            if a != t || !causal_children.contains(&b) {
                cut.insert_directed(a, b).ok()?;
            }
        }
        for &b in graph.bidirected_neighbors(a) {
            if b.raw() > a.raw() {
                cut.insert_bidirected(a, b).ok()?;
            }
        }
    }
    Some(cut)
}

enum MagAdjustment {
    Found { z_vars: Arc<[VariableId]>, examined: u64 },
    Failed(IdentificationResult),
}

fn mag_adjustment_search(
    mag: &Pag,
    _t: VariableId,
    _y: VariableId,
    t_d: DenseNodeId,
    y_d: DenseNodeId,
    query: CausalQuery,
    max_candidates: usize,
) -> Result<MagAdjustment, IdentificationError> {
    let Some(admg) = mag_to_admg(mag) else {
        return Ok(MagAdjustment::Failed(not_identified(
            query,
            "completion is not a MAG (undirected marks remain)",
        )));
    };

    let Some(mutilated) = proper_backdoor_mag(mag, t_d, y_d) else {
        return Ok(MagAdjustment::Failed(not_identified(
            query,
            "MAG is not adjustment amenable: a causal path starts with an invisible edge; no adjustment set identifies this effect",
        )));
    };
    let candidates = adjustment_candidates(&admg, t_d, y_d);
    if candidates.len() > max_candidates {
        return Ok(MagAdjustment::Failed(capped_completion_result(
            query,
            candidates.len(),
            max_candidates,
        )));
    }

    let mut ws = DSeparationWorkspace::default();
    let mut examined = 0u64;
    let mut found: Option<Vec<DenseNodeId>> = None;
    'sizes: for size in 0..=candidates.len() {
        let mut early = false;
        let mut enum_err: Option<IdentificationError> = None;
        crate::enum_masks::for_each_mask_of_size(&candidates, size, |z| {
            if enum_err.is_some() {
                return true;
            }
            examined += 1;
            match mutilated.is_m_separated(t_d, y_d, z, &mut ws) {
                Ok(true) => {
                    found = Some(z.to_vec());
                    early = true;
                    true
                }
                Ok(false) => false,
                Err(e) => {
                    enum_err = Some(IdentificationError::from(e));
                    true
                }
            }
        });
        if let Some(e) = enum_err {
            return Err(e);
        }
        if early {
            break 'sizes;
        }
    }

    let Some(z_dense) = found else {
        return Ok(MagAdjustment::Failed(IdentificationResult::not_identified(
            query,
            {
                let mut d = DerivationTrace::default();
                d.push(
                    "generalized.adjustment",
                    "no generalized adjustment set among ancestor candidates on completion",
                );
                d
            },
            AssumptionSet::default(),
            IdentificationPerformanceRecord { candidates_examined: examined, sets_returned: 0 },
        )));
    };

    let z_vars: Arc<[VariableId]> =
        z_dense.iter().map(|&d| mag_dense_to_var(mag, d)).collect::<Result<Vec<_>, _>>()?.into();
    Ok(MagAdjustment::Found { z_vars, examined })
}

fn mag_adjustment_identified(
    query: CausalQuery,
    z_vars: Arc<[VariableId]>,
    examined: u64,
    functional: antecedent_expr::ExprId,
    arena: CausalExprArena,
) -> IdentificationResult {
    let n_z = z_vars.len();
    let label = if n_z == 0 { "generalized.adjustment.empty" } else { "generalized.adjustment" };
    let estimand = IdentifiedEstimand::backdoor(label, z_vars, functional);
    let mut assumptions = AssumptionSet::default();
    assumptions.push(crate::assumptions::causal_markov("generalized.adjustment.mag"));
    IdentificationResult::identified(
        query,
        vec![estimand],
        arena,
        {
            let mut d = DerivationTrace::default();
            d.push(
                "generalized.adjustment",
                format!(
                    "Z (size {n_z}) m-separates T from Y in the proper back-door graph after MAG amenability and forbidden-set checks"
                ),
            );
            d
        },
        assumptions,
        IdentificationPerformanceRecord { candidates_examined: examined, sets_returned: 1 },
    )
}

pub(crate) fn identify_on_mag_completion(
    mag: &Pag,
    t: VariableId,
    y: VariableId,
    t_d: DenseNodeId,
    y_d: DenseNodeId,
    active: Value,
    control: Value,
    max_candidates: usize,
) -> Result<IdentificationResult, IdentificationError> {
    let query = CausalQuery::AverageEffect(AverageEffectQuery::new(
        t,
        y,
        Arc::from([]),
        antecedent_core::Intervention::set(t, control.clone()),
        antecedent_core::Intervention::set(t, active.clone()),
        antecedent_core::TargetPopulation::AllObserved,
    ));
    match mag_adjustment_search(mag, t, y, t_d, y_d, query.clone(), max_candidates)? {
        MagAdjustment::Failed(result) => Ok(result),
        MagAdjustment::Found { z_vars, examined } => {
            let mut arena = CausalExprArena::new();
            let functional = arena.backdoor_ate(t, y, &z_vars, active, control);
            Ok(mag_adjustment_identified(query, z_vars, examined, functional, arena))
        }
    }
}

/// Single-arm MAG adjustment mean `E[Y | do(T=level)]` at the requested Set.
pub(crate) fn identify_on_mag_completion_mean(
    mag: &Pag,
    t: VariableId,
    y: VariableId,
    t_d: DenseNodeId,
    y_d: DenseNodeId,
    level: Value,
    max_candidates: usize,
    response: &ResponseQuery,
) -> Result<IdentificationResult, IdentificationError> {
    let query = CausalQuery::Response(response.clone());
    match mag_adjustment_search(mag, t, y, t_d, y_d, query.clone(), max_candidates)? {
        MagAdjustment::Failed(result) => Ok(result),
        MagAdjustment::Found { z_vars, examined } => {
            let mut arena = CausalExprArena::new();
            let functional = arena.backdoor_mean(t, y, &z_vars, level);
            Ok(mag_adjustment_identified(query, z_vars, examined, functional, arena))
        }
    }
}

fn mag_dense_to_var(mag: &Pag, id: DenseNodeId) -> Result<VariableId, IdentificationError> {
    match mag.nodes().get(id.as_usize()) {
        Some(antecedent_graph::NodeRef::Static(v)) => Ok(*v),
        _ => Err(IdentificationError::UnknownVariable { id: VariableId::from_raw(id.raw()) }),
    }
}

/// Candidates = `(An({T,Y}) ∪ De(T)) \ (Forb(T,Y) ∪ {T,Y})`.
///
/// `Forb(T,Y) = De(cn)` with `cn = De(T) ∩ An(Y) \ {T}`. Side-effect descendants
/// of `T` that are not on a proper causal path to `Y` are allowed; mediators
/// and their descendants are not.
fn adjustment_candidates(admg: &Admg, t: DenseNodeId, y: DenseNodeId) -> Vec<DenseNodeId> {
    let an = directed_closure(admg, &[t, y], true);
    let de_t = directed_closure(admg, &[t], false);
    let an_y = directed_closure(admg, &[y], true);
    let mut cn = Vec::new();
    for i in 0..admg.node_count() {
        let id = DenseNodeId::from_raw(i as u32);
        if id != t && de_t.contains(id) && an_y.contains(id) {
            cn.push(id);
        }
    }
    let forb = directed_closure(admg, &cn, false);
    let mut out = Vec::new();
    for i in 0..admg.node_count() {
        let id = DenseNodeId::from_raw(i as u32);
        if id == t || id == y || forb.contains(id) {
            continue;
        }
        if an.contains(id) || de_t.contains(id) {
            out.push(id);
        }
    }
    out
}

pub(crate) fn directed_closure(admg: &Admg, seeds: &[DenseNodeId], ancestors: bool) -> BitSet {
    let mut out = BitSet::with_len(admg.node_count());
    let mut stack: Vec<DenseNodeId> = seeds.to_vec();
    for &s in seeds {
        out.insert(s);
    }
    while let Some(u) = stack.pop() {
        let nbrs = if ancestors { admg.parents(u) } else { admg.children(u) };
        for &v in nbrs {
            if !out.contains(v) {
                out.insert(v);
                stack.push(v);
            }
        }
    }
    out
}

pub(crate) fn not_identified(query: CausalQuery, detail: &str) -> IdentificationResult {
    let mut derivation = DerivationTrace::default();
    derivation.push("generalized.adjustment", detail);
    IdentificationResult::not_identified(
        query,
        derivation,
        AssumptionSet::default(),
        IdentificationPerformanceRecord::default(),
    )
}

/// Execution-capped result: candidate family exceeded `max_candidates` before
/// enumeration could start. Status stays [`IdentificationStatus::NotIdentified`]
/// (1.0 freeze). Honesty is [`CAPPED_COMPLETION_DIAGNOSTIC_CODE`] as
/// [`DiagnosticKind::Execution`], not a scientific open-back-door.
pub(crate) fn capped_completion_result(
    query: CausalQuery,
    n_candidates: usize,
    max_candidates: usize,
) -> IdentificationResult {
    let mut result =
        not_identified(query, "generalized adjustment candidate set exceeds enumeration limit");
    result.diagnostics.push(Diagnostic::new(
        CAPPED_COMPLETION_DIAGNOSTIC_CODE,
        DiagnosticKind::Execution,
        DiagnosticSeverity::Warning,
        format!(
            "generalized adjustment candidate set ({n_candidates}) exceeds max_candidates \
             ({max_candidates}); enumeration was not attempted on this completion, so \
             identifiability could not be determined (not the same as a proven non-\
             identifiable completion)"
        ),
    ));
    result
}

pub(crate) fn mag_to_admg(mag: &Pag) -> Option<Admg> {
    let n = mag.node_count() as u32;
    let mut admg = Admg::with_variables(n);
    for i in 0..mag.node_count() {
        let a = DenseNodeId::from_raw(i as u32);
        for (b, at_a, at_b) in mag.neighbors(a) {
            if b.raw() < a.raw() {
                continue;
            }
            if matches!(at_a, Endpoint::Circle) || matches!(at_b, Endpoint::Circle) {
                return None;
            }
            match (at_a, at_b) {
                (Endpoint::Tail, Endpoint::Arrow) => {
                    admg.insert_directed(a, b).ok()?;
                }
                (Endpoint::Arrow, Endpoint::Tail) => {
                    admg.insert_directed(b, a).ok()?;
                }
                (Endpoint::Arrow, Endpoint::Arrow) => {
                    admg.insert_bidirected(a, b).ok()?;
                }
                (Endpoint::Tail, Endpoint::Tail) => {
                    return None;
                }
                _ => return None,
            }
        }
    }
    Some(admg)
}

fn mutilate_outgoing(admg: &Admg, t: DenseNodeId) -> Admg {
    let n = admg.node_count() as u32;
    let mut out = Admg::with_variables(n);
    for i in 0..admg.node_count() {
        let u = DenseNodeId::from_raw(i as u32);
        for &v in admg.children(u) {
            if u == t {
                continue;
            }
            let _ = out.insert_directed(u, v);
        }
        for &v in admg.bidirected_neighbors(u) {
            if v.raw() > u.raw() {
                let _ = out.insert_bidirected(u, v);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    // Completion weights here are exact counts of unit-weight cases.
    #![allow(clippy::float_cmp)]
    use super::*;
    use crate::result::IdentificationStatus;
    use antecedent_graph::Pag;

    #[test]
    fn conditional_modifier_cannot_be_a_mediator() {
        let mut cpdag = Cpdag::with_variables(3);
        let t = DenseNodeId::from_raw(0);
        let w = DenseNodeId::from_raw(1);
        let y = DenseNodeId::from_raw(2);
        cpdag.insert_directed(t, w).unwrap();
        cpdag.insert_directed(w, y).unwrap();
        let id = GeneralizedAdjustmentIdentifier::new();
        let ate = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(2));
        assert_eq!(id.identify_cpdag_envelope(&cpdag, &ate).unwrap().identified_weight.0, 1.0);
        let conditional = ate.with_effect_modifiers([VariableId::from_raw(1)]);
        let env = id.identify_cpdag_envelope(&cpdag, &conditional).unwrap();
        assert_eq!(env.identified_weight.0, 0.0);
        assert_eq!(env.unidentified_weight.0, 1.0);
        assert!(
            env.cases[0]
                .result
                .diagnostics
                .iter()
                .any(|d| d.code.as_ref() == "identify.conditional.adjustment_unverified")
        );
    }

    #[test]
    fn pretreatment_modifier_can_open_a_collider_path() {
        // T <- A -> W <- B -> Y, T -> Y. Empty Z identifies ATE,
        // Conditioning on pre-treatment W opens the noncausal path; adding A closes it.
        let mut pag = Pag::with_variables(5);
        for (a, b) in [(1, 0), (1, 2), (3, 2), (3, 4), (0, 4)] {
            pag.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
        }
        let id = GeneralizedAdjustmentIdentifier::new();
        let ate = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(4));
        assert!(id.identify_pag_envelope(&pag, &ate).unwrap().identified_weight.0 > 0.0);
        let conditional = ate.with_effect_modifiers([VariableId::from_raw(2)]);
        let env = id.identify_pag_envelope(&pag, &conditional).unwrap();
        assert!(env.identified_weight.0 > 0.0);
        assert_eq!(env.unidentified_weight.0, 0.0);
        assert_eq!(
            env.cases[0].result.estimands[0].adjustment_set.as_ref(),
            &[VariableId::from_raw(1)]
        );
    }

    #[test]
    fn envelope_preserves_mass_on_mixed_pag() {
        let mut pag = Pag::with_variables(2);
        pag.insert_circle_arrow(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let id = GeneralizedAdjustmentIdentifier {
            config: GeneralizedAdjustmentConfig {
                max_completions: 8,
                per_completion_weight: 1.0,
                max_candidates: 16,
            },
        };
        let q = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        let env = id.identify_pag_envelope(&pag, &q).unwrap();
        let total = env.identified_weight.0 + env.unidentified_weight.0;
        assert!(total > 0.0);
        assert!((total - env.cases.len() as f64).abs() < 1e-9);
    }

    #[test]
    fn invisible_directed_edge_does_not_identify_even_conditionally() {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../../conformance/identify/generalized_adjustment/expected.json"
        ))
        .unwrap();
        assert_eq!(fixture["cases"][0]["status"], "not_identified_by_adjustment");
        let mut pag = Pag::with_variables(3);
        pag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let id = GeneralizedAdjustmentIdentifier::new();
        let q = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        for query in [q.clone(), q.with_effect_modifiers([VariableId::from_raw(2)])] {
            let env = id.identify_pag_envelope(&pag, &query).unwrap();
            assert_eq!(env.status, IdentificationStatus::NotIdentified);
            assert_eq!(env.identified_weight.0, 0.0);
            assert!(env.cases[0].result.estimands.is_empty());
        }
        // R -> T with R nonadjacent to Y makes T -> Y visible.
        pag.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(0)).unwrap();
        let env = id
            .identify_pag_envelope(
                &pag,
                &AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1)),
            )
            .unwrap();
        assert_eq!(env.status, IdentificationStatus::NonparametricallyIdentified);
        assert!(env.cases[0].result.estimands[0].adjustment_set.is_empty());
    }

    #[test]
    fn proper_backdoor_preserves_side_effect_collider_edges() {
        // R -> T -> Y; T -> W <- U -> Y. R witnesses visibility of T -> Y.
        let mut mag = Pag::with_variables(5);
        for (a, b) in [(4, 0), (0, 1), (0, 2), (3, 2), (3, 1)] {
            mag.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
        }
        let t = DenseNodeId::from_raw(0);
        let y = DenseNodeId::from_raw(1);
        let w = DenseNodeId::from_raw(2);
        let cut = proper_backdoor_mag(&mag, t, y).unwrap();
        assert!(cut.children(t).contains(&w));
        assert!(!cut.children(t).contains(&y));
        let mut workspace = DSeparationWorkspace::default();
        assert!(cut.is_m_separated(t, y, &[], &mut workspace).unwrap());
        assert!(!cut.is_m_separated(t, y, &[w], &mut workspace).unwrap());
    }

    #[test]
    fn confounder_identifies_with_nonempty_z() {
        // Z → T, Z → Y, T → Y  (backdoor {Z}).
        let mut pag = Pag::with_variables(4);
        let z = DenseNodeId::from_raw(0);
        let t = DenseNodeId::from_raw(1);
        let y = DenseNodeId::from_raw(2);
        pag.insert_directed(z, t).unwrap();
        pag.insert_directed(z, y).unwrap();
        pag.insert_directed(t, y).unwrap();
        pag.insert_directed(DenseNodeId::from_raw(3), t).unwrap();
        let id = GeneralizedAdjustmentIdentifier::new();
        let q = AverageEffectQuery::binary_ate(VariableId::from_raw(1), VariableId::from_raw(2));
        let env = id.identify_pag_envelope(&pag, &q).unwrap();
        assert_eq!(env.status, IdentificationStatus::NonparametricallyIdentified);
        assert_eq!(env.cases[0].result.status, IdentificationStatus::NonparametricallyIdentified);
        let z_set = &env.cases[0].result.estimands[0].adjustment_set;
        assert!(z_set.iter().any(|v| v.raw() == 0), "expected Z in adjustment, got {z_set:?}");
    }

    #[test]
    fn capped_enumeration_is_distinguishable_from_genuine_nonidentification() {
        // Case A: Z → T, Z → Y, T → Y, but max_candidates=0 forces the single-candidate set
        // {Z} over the cap before enumeration can even start — the search was truncated, not
        // exhausted.
        let mut capped_pag = Pag::with_variables(4);
        let z = DenseNodeId::from_raw(0);
        let t = DenseNodeId::from_raw(1);
        let y = DenseNodeId::from_raw(2);
        capped_pag.insert_directed(z, t).unwrap();
        capped_pag.insert_directed(z, y).unwrap();
        capped_pag.insert_directed(t, y).unwrap();
        capped_pag.insert_directed(DenseNodeId::from_raw(3), t).unwrap();
        let capped_id = GeneralizedAdjustmentIdentifier {
            config: GeneralizedAdjustmentConfig {
                max_completions: 8,
                per_completion_weight: 1.0,
                max_candidates: 0,
            },
        };
        let q = AverageEffectQuery::binary_ate(VariableId::from_raw(1), VariableId::from_raw(2));
        let capped_env = capped_id.identify_pag_envelope(&capped_pag, &q).unwrap();
        assert!(capped_env.unidentified_weight.0 > 0.0);
        assert_eq!(
            capped_env.truncated_completions,
            capped_env.cases.len(),
            "every case in this fixture should be capped"
        );
        assert_eq!(
            capped_env.cases[0].result.status,
            IdentificationStatus::NotIdentified,
            "1.0 freeze: cap keeps NotIdentified, honesty is the Execution diagnostic"
        );
        assert_eq!(capped_env.status, IdentificationStatus::NotIdentified);
        assert!(
            capped_env.cases[0].result.diagnostics.iter().any(|d| {
                d.code.as_ref() == CAPPED_COMPLETION_DIAGNOSTIC_CODE
                    && d.kind == DiagnosticKind::Execution
            }),
            "capped case should carry the capped-completion execution diagnostic"
        );
        assert!(
            !capped_env.cases[0]
                .result
                .diagnostics
                .iter()
                .any(|d| d.kind == DiagnosticKind::Scientific),
            "cap must not be stamped scientific: {:?}",
            capped_env.cases[0].result.diagnostics
        );

        // Case B: bare T ↔ Y (unmeasured confounding, no other variables). The search is
        // never truncated (candidates.len() == 0 never exceeds the default max_candidates),
        // but no adjustment set can ever separate T from Y, so this is a genuine
        // non-identification — proved, not merely "could not tell".
        let mut confounded_pag = Pag::with_variables(2);
        confounded_pag
            .insert_bidirected(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1))
            .unwrap();
        let genuine_id = GeneralizedAdjustmentIdentifier::new();
        let q2 = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        let genuine_env = genuine_id.identify_pag_envelope(&confounded_pag, &q2).unwrap();
        assert!(genuine_env.unidentified_weight.0 > 0.0);
        assert_eq!(
            genuine_env.truncated_completions, 0,
            "genuinely-blocked completions must not count as truncated"
        );

        // Both envelopes report positive unidentified mass (neither is silently dropped), but
        // only the capped one is flagged as truncated -- proving the two are distinguishable
        // rather than both being folded indistinguishably into `unidentified_weight`.
        assert!(capped_env.truncated_completions > genuine_env.truncated_completions);
    }

    #[test]
    fn gac_candidates_allow_side_effect_descendants_of_t() {
        // T → M → Y, T → W, U → T, U → Y. W ∈ De(T) but W ∉ Forb(T,Y)=De({M,Y}).
        let mut g = Admg::with_variables(5);
        let t = DenseNodeId::from_raw(0);
        let y = DenseNodeId::from_raw(1);
        let m = DenseNodeId::from_raw(2);
        let w = DenseNodeId::from_raw(3);
        let u = DenseNodeId::from_raw(4);
        g.insert_directed(t, m).unwrap();
        g.insert_directed(m, y).unwrap();
        g.insert_directed(t, w).unwrap();
        g.insert_directed(u, t).unwrap();
        g.insert_directed(u, y).unwrap();
        let c = adjustment_candidates(&g, t, y);
        assert!(c.contains(&u), "confounder must remain a candidate: {c:?}");
        assert!(c.contains(&w), "GAC allows side-effect descendant W: {c:?}");
        assert!(!c.contains(&m), "mediator is in Forb: {c:?}");
        assert!(!c.contains(&t) && !c.contains(&y));
    }

    #[test]
    fn cpdag_confounded_undirected_has_two_identified_completions() {
        // Z — T → Y, Z → Y. Completions: Z→T (backdoor {Z}) and T→Z (empty Z).
        let mut cpdag = Cpdag::with_variables(3);
        let z = DenseNodeId::from_raw(0);
        let t = DenseNodeId::from_raw(1);
        let y = DenseNodeId::from_raw(2);
        cpdag.insert_undirected(z, t).unwrap();
        cpdag.insert_directed(z, y).unwrap();
        cpdag.insert_directed(t, y).unwrap();
        let id = GeneralizedAdjustmentIdentifier::new();
        let q = AverageEffectQuery::binary_ate(VariableId::from_raw(1), VariableId::from_raw(2));
        let env = id.identify_cpdag_envelope(&cpdag, &q).unwrap();
        assert_eq!(env.cases.len(), 2);
        assert!(env.unidentified_weight.0 == 0.0);
        assert_eq!(env.identified_weight.0, 2.0);
        assert!(
            env.cases
                .iter()
                .all(|c| { c.result.status == IdentificationStatus::NonparametricallyIdentified })
        );
    }

    #[test]
    fn fully_oriented_cpdag_is_a_one_case_envelope() {
        let mut cpdag = Cpdag::with_variables(2);
        cpdag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let env = GeneralizedAdjustmentIdentifier::new()
            .identify_cpdag_envelope(
                &cpdag,
                &AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1)),
            )
            .unwrap();
        assert_eq!(env.cases.len(), 1);
        assert_eq!(env.status, IdentificationStatus::NonparametricallyIdentified);
        assert!(env.unidentified_weight.0 == 0.0);
    }
}

#[cfg(test)]
mod conditional_search_tests {
    use super::*;

    #[test]
    fn modifier_collider_requires_an_alternative_adjustment_set() {
        let mut dag = antecedent_graph::Dag::with_variables(5);
        for (a, b) in [(3, 0), (3, 2), (4, 2), (4, 1), (0, 1)] {
            dag.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
        }
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1))
                .with_effect_modifiers([VariableId::from_raw(2)]);
        let mut identifier = BackdoorIdentifier::new();
        let prepared = identifier.prepare(&dag).unwrap();
        let result = identifier
            .identify(
                &prepared,
                &CausalQuery::AverageEffect(query.clone()),
                &mut IdentificationWorkspace::default(),
            )
            .unwrap();
        assert_eq!(result.status, IdentificationStatus::NonparametricallyIdentified);
        assert_eq!(result.estimands[0].adjustment_set.as_ref(), &[VariableId::from_raw(3)]);
        identifier.config.forbidden = Arc::from([VariableId::from_raw(3)]);
        let result = identifier
            .identify(
                &prepared,
                &CausalQuery::AverageEffect(query.clone()),
                &mut IdentificationWorkspace::default(),
            )
            .unwrap();
        assert_eq!(result.estimands[0].adjustment_set.as_ref(), &[VariableId::from_raw(4)]);
        identifier.config.forbidden = Arc::from([VariableId::from_raw(3), VariableId::from_raw(4)]);
        let result = identifier
            .identify(
                &prepared,
                &CausalQuery::AverageEffect(query),
                &mut IdentificationWorkspace::default(),
            )
            .unwrap();
        assert_eq!(result.status, IdentificationStatus::NotIdentified);
    }
}
