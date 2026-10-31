//! Natural mediation on one series under a temporal PAG.
//!
//! A [`TemporalPag`] is a stationary template of a Markov equivalence class of maximal
//! ancestral graphs. A pulse at the treatment period and an outcome at the horizon period
//! need the template over a finite window of consecutive periods, and a natural direct or
//! indirect effect needs more than the MAG of that window: a path-specific effect is not a
//! function of the m-separation model (see [`crate::pag_path_specific`]), so every completion
//! is read through the ADMGs that share its MAG, and the effect is reported only when each of
//! those identifies it.
//!
//! # The window
//!
//! The window holds the template's longest lag of history before the pulse (plus any earlier
//! treated offset) and runs to the outcome period, the layout of a `TemporalIndexer` over the
//! schema: node `(variable, offset)` is dense id `(offset + history) * variables + variable`.
//! A cause that precedes the window is not erased: each completion is unfolded over a window
//! that reaches one further template lag into the past and then marginalized onto the
//! window by the inducing-path rule (Richardson and Spirtes 2002, Cor. 5.3): two window nodes
//! are adjacent in the marginal MAG iff no subset of the window nodes that are ancestors of
//! either separates them in the larger MAG, and an edge is directed from the ancestor. A
//! common cause that precedes the window therefore stays a bidirected edge between its
//! in-window effects, as a latent common cause does in the window of a temporal ADMG. History
//! deeper than that one further lag is read as the initial state, the stationary reading every
//! window contrast makes.
//!
//! The marginal MAG is restricted to the ancestors of the treatment, the mediators and the
//! outcome (an ancestral set of a MAG induces the MAG of its marginal model); the other
//! window nodes stay isolated so that every node keeps the dense id its table column has.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{
    CausalQuery, Diagnostic, DiagnosticKind, DiagnosticSeverity, Intervention, MediationQuery,
    TemporalEffectQuery, TemporalIndexer, TemporalNodeKey, TemporalPolicy, VariableId,
};
use antecedent_graph::{
    Admg, DSeparationWorkspace, DenseNodeId, Endpoint, NodeRef, Pag, TemporalPag,
    TemporalPagCompletionSampler,
};

use crate::envelope::{
    GraphFeature, GraphIdentificationCase, IdentificationEnvelope, ProbabilityMass,
    carries_identified_mass,
};
use crate::error::IdentificationError;
use crate::generalized::{GeneralizedAdjustmentIdentifier, not_identified};
use crate::identifier::IdentificationWorkspace;
use crate::path_specific::PathSpecificIdentifier;
use crate::result::{IdentificationResult, IdentificationStatus};
use crate::temporal_generalized::{TemporalClassEnvelope, TemporalCompletionGraph};

/// Diagnostic code of a window whose completion could not be read as a maximal ancestral graph.
pub const TEMPORAL_PAG_WINDOW_NOT_MAG_DIAGNOSTIC_CODE: &str =
    "identify.temporal_pag_path_specific.window_not_mag";

/// A natural-mediation contrast stated on the finite window of a temporal PAG.
#[derive(Clone, Debug)]
pub struct TemporalPagMediationWindow {
    /// Layout of the window: node `(variable, offset)` is dense id
    /// `(offset + history) * variables + variable`, also the column of the window table.
    pub indexer: TemporalIndexer,
    /// The contrast on window coordinates: treatment at the pulse period, outcome at the
    /// horizon period, the declared mediators at every period between them.
    query: std::sync::Arc<MediationQuery>,
    /// Offset of the pulse from the window origin.
    pub treatment_offset: i32,
    /// Offset of the outcome from the window origin.
    pub outcome_offset: i32,
    // Covers the internally produced window and its bounded component copies.
    _lease: std::sync::Arc<antecedent_core::execution::AdmissionLease>,
}

impl TemporalPagMediationWindow {
    /// Borrow the admitted window query. Caller-created copies are outside this owner's lease.
    #[must_use]
    pub fn query(&self) -> &MediationQuery {
        &self.query
    }

    /// Make a contrast variant while sharing this bounded window owner.
    #[must_use]
    pub fn with_contrast(&self, contrast: antecedent_core::MediationContrast) -> Self {
        let mut next = self.clone();
        std::sync::Arc::make_mut(&mut next.query).contrast = contrast;
        next
    }
}

/// State the contrast `query` on the window of `template` that the pulse `effect` (same
/// treatment, outcome and horizon) needs.
///
/// `variable_count` is the number of variables of the series schema, so a window coordinate is
/// the column of the window table the series is read into.
///
/// # Errors
///
/// A policy other than a pulse, an outcome before the pulse, a variable outside the schema, a
/// non-`Set` intervention, or no mediator between the pulse period and the outcome period.
pub fn temporal_pag_mediation_window(
    template: &TemporalPag,
    effect: &TemporalEffectQuery,
    query: &MediationQuery,
    variable_count: u32,
) -> Result<TemporalPagMediationWindow, IdentificationError> {
    let ctx = antecedent_core::ExecutionContext::production_default(0);
    let mut budget = antecedent_core::TemporalBudget::new(ctx.temporal_materialization_limits());
    temporal_pag_mediation_window_within(template, effect, query, variable_count, &mut budget, &ctx)
}

/// Construct a mediation window under the caller's shared temporal materialization budget.
pub fn temporal_pag_mediation_window_within(
    template: &TemporalPag,
    effect: &TemporalEffectQuery,
    query: &MediationQuery,
    variable_count: u32,
    budget: &mut antecedent_core::TemporalBudget,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<TemporalPagMediationWindow, IdentificationError> {
    if ctx.cancellation.is_cancelled() {
        return Err(IdentificationError::Cancelled);
    }
    let _validation = antecedent_core::admit_temporal_intervention_validation(
        [&effect.control, &effect.active, &query.control, &query.active],
        budget,
        ctx,
    )
    .map_err(crate::error::temporal_materialization_error)?;
    let input_work = (template.nodes().len() as u64)
        .checked_add(query.mediators.len() as u64)
        .and_then(|n| n.checked_add(query.horizons.len() as u64))
        .and_then(|n| n.checked_add(effect.effect_modifiers.len() as u64))
        .ok_or_else(|| IdentificationError::budget(crate::error::IdentificationBudget::Steps))?;
    budget.charge_work(input_work).map_err(crate::error::temporal_materialization_error)?;
    effect.validate().map_err(|_| IdentificationError::unsupported("invalid temporal effect"))?;
    query.validate().map_err(|_| IdentificationError::unsupported("invalid mediation query"))?;
    if !matches!(effect.policy, TemporalPolicy::Pulse { .. }) {
        return Err(IdentificationError::unsupported(
            "natural mediation on a temporal PAG is a contrast of pulses",
        ));
    }
    let treatment_offset = effect
        .try_treatment_offset()
        .map_err(|_| IdentificationError::unsupported("the pulse has no treatment offset"))?;
    let outcome_offset = effect.outcome_offset().map_err(IdentificationError::construction)?;
    if outcome_offset < treatment_offset {
        return Err(IdentificationError::unsupported(
            "the outcome period precedes the pulse period of the mediation",
        ));
    }
    let schema = |variable: VariableId| -> Result<(), IdentificationError> {
        if variable.raw() < variable_count {
            Ok(())
        } else {
            Err(IdentificationError::VariableOutsideSchema {
                variable: variable.raw(),
                count: variable_count,
            })
        }
    };
    let mut longest_lag = 0u32;
    for node in template.nodes() {
        if ctx.cancellation.is_cancelled() {
            return Err(IdentificationError::Cancelled);
        }
        let NodeRef::Lagged { variable, lag } = node else {
            return Err(IdentificationError::unsupported(
                "natural mediation on a temporal PAG requires a lagged template",
            ));
        };
        schema(*variable)?;
        longest_lag = longest_lag.max(lag.raw());
    }
    for variable in
        [query.treatment, query.outcome].into_iter().chain(query.mediators.iter().copied())
    {
        schema(variable)?;
    }
    let history = longest_lag
        .checked_add(treatment_offset.min(0).unsigned_abs())
        .ok_or_else(|| IdentificationError::out_of_range("temporal mediation history overflow"))?;
    let last = u32::try_from(outcome_offset.max(0)).map_err(|_| {
        IdentificationError::out_of_range("the outcome offset exceeds the window range")
    })?;
    let horizon = last
        .checked_add(1)
        .ok_or_else(|| IdentificationError::out_of_range("temporal mediation horizon overflow"))?;
    let span = u64::try_from(i64::from(outcome_offset) - i64::from(treatment_offset) + 1)
        .map_err(|_| IdentificationError::out_of_range("temporal mediation span overflow"))?;
    budget.charge_offsets(span).map_err(crate::error::temporal_materialization_error)?;
    let count = span
        .checked_mul(query.mediators.len() as u64)
        .ok_or_else(|| IdentificationError::budget(crate::error::IdentificationBudget::Steps))?;
    budget
        .charge_work(count.checked_mul(8).ok_or_else(|| {
            IdentificationError::budget(crate::error::IdentificationBudget::Steps)
        })?)
        .map_err(crate::error::temporal_materialization_error)?;
    let bytes = count
        .checked_mul(64)
        .and_then(|n| n.checked_add(4096))
        .ok_or_else(|| IdentificationError::budget(crate::error::IdentificationBudget::Memory))?;
    if bytes > budget.limits().max_bytes {
        return Err(IdentificationError::budget(crate::error::IdentificationBudget::Memory));
    }
    let lease =
        std::sync::Arc::new(ctx.acquire_admission(0, bytes).map_err(|error| match error {
            antecedent_core::execution::AdmissionError::Cancelled => IdentificationError::Cancelled,
            _ => IdentificationError::budget(crate::error::IdentificationBudget::Memory),
        })?);
    let indexer = TemporalIndexer::new(variable_count, history, horizon)
        .map_err(|error| IdentificationError::construction(error))?;
    let coordinate =
        |variable: VariableId, offset: i32| -> Result<VariableId, IdentificationError> {
            indexer
                .dense_id(TemporalNodeKey { variable, offset })
                .map(VariableId::from_raw)
                .map_err(|error| IdentificationError::construction(error))
        };
    let treatment = coordinate(query.treatment, treatment_offset)?;
    let outcome = coordinate(query.outcome, outcome_offset)?;
    let mut mediators = Vec::new();
    for &variable in query.mediators.iter() {
        for offset in treatment_offset..=outcome_offset {
            if ctx.cancellation.is_cancelled() {
                return Err(IdentificationError::Cancelled);
            }
            let node = coordinate(variable, offset)?;
            if node != treatment && node != outcome {
                mediators.push(node);
            }
        }
    }
    if mediators.is_empty() {
        return Err(IdentificationError::unsupported(
            "no mediator occurs in the window between the pulse and the outcome; a mediator must \
             lie between the treatment period and the outcome period",
        ));
    }
    let level = |intervention: &Intervention| match intervention {
        Intervention::Set { value, .. } => Ok(Intervention::set(treatment, value.clone())),
        _ => Err(IdentificationError::unsupported(
            "natural mediation on a temporal PAG requires Set interventions on both sides",
        )),
    };
    let mut window = MediationQuery::binary(treatment, outcome, mediators, query.contrast);
    window.control = level(&query.control)?;
    window.active = level(&query.active)?;
    window.target_population = query.target_population.clone();
    Ok(TemporalPagMediationWindow {
        indexer,
        query: std::sync::Arc::new(window),
        treatment_offset,
        outcome_offset,
        _lease: lease,
    })
}

/// Longest lag the template reaches back.
fn longest_lag(template: &TemporalPag) -> u32 {
    template
        .nodes()
        .iter()
        .filter_map(|node| match node {
            NodeRef::Lagged { lag, .. } => Some(lag.raw()),
            _ => None,
        })
        .max()
        .unwrap_or(0)
}

/// `big` (a directed/bidirected MAG) as an ADMG over the same dense ids.
fn mag_as_admg(big: &Pag) -> Result<Admg, IdentificationError> {
    let nodes = u32::try_from(big.node_count())
        .map_err(|_| IdentificationError::out_of_range("window exceeds the node id range"))?;
    let mut admg = Admg::with_variables(nodes);
    for index in 0..big.node_count() {
        let a = DenseNodeId::from_index(index);
        for (b, at_a, at_b) in big.neighbors(a) {
            if b.raw() < a.raw() {
                continue;
            }
            let inserted = match (at_a, at_b) {
                (Endpoint::Tail, Endpoint::Arrow) => admg.insert_directed(a, b),
                (Endpoint::Arrow, Endpoint::Tail) => admg.insert_directed(b, a),
                (Endpoint::Arrow, Endpoint::Arrow) => admg.insert_bidirected(a, b),
                _ => {
                    return Err(IdentificationError::unsupported(
                        "a temporal PAG completion must carry directed and bidirected marks only",
                    ));
                }
            };
            inserted.map_err(IdentificationError::from)?;
        }
    }
    Ok(admg)
}

/// The MAG of the window nodes that are the seeds or their ancestors, marginalized out of
/// `big` (a MAG whose first `shift` dense ids precede the window), over the window's own dense
/// ids. Window nodes that are neither a seed nor an ancestor of one stay isolated.
///
/// # Errors
///
/// A `big` that is not a directed/bidirected MAG, or a graph error.
fn window_mag(
    big: &Pag,
    window: &TemporalIndexer,
    shift: usize,
    seeds: &[usize],
    ctx: &antecedent_core::ExecutionContext,
) -> Result<Pag, IdentificationError> {
    if ctx.cancellation.is_cancelled() {
        return Err(IdentificationError::Cancelled);
    }
    let admg = mag_as_admg(big)?;
    let ancestors = admg.strict_ancestor_sets();
    let n = window.dense_len();
    let mut keep = vec![false; n];
    for &seed in seeds {
        if seed >= n {
            return Err(IdentificationError::out_of_range("a seed lies outside the window"));
        }
        keep[seed] = true;
        for (inner, retained) in keep.iter_mut().enumerate() {
            if ancestors[seed + shift].contains(DenseNodeId::from_index(inner + shift)) {
                *retained = true;
            }
        }
    }
    let nodes = u32::try_from(n)
        .map_err(|_| IdentificationError::out_of_range("window exceeds the node id range"))?;
    let mut marginal = Pag::with_variables(nodes);
    let mut workspace = DSeparationWorkspace::default();
    let kept: Vec<usize> = (0..n).filter(|&i| keep[i]).collect();
    for (position, &i) in kept.iter().enumerate() {
        let (a_big, a_inner) = (DenseNodeId::from_index(i + shift), DenseNodeId::from_index(i));
        for &j in &kept[position + 1..] {
            if ctx.cancellation.is_cancelled() {
                return Err(IdentificationError::Cancelled);
            }
            let (b_big, b_inner) = (DenseNodeId::from_index(j + shift), DenseNodeId::from_index(j));
            let direct = admg.children(a_big).contains(&b_big)
                || admg.children(b_big).contains(&a_big)
                || admg.bidirected_neighbors(a_big).contains(&b_big);
            let adjacent = direct || {
                let given: Vec<DenseNodeId> = kept
                    .iter()
                    .copied()
                    .filter(|&k| k != i && k != j)
                    .map(|k| DenseNodeId::from_index(k + shift))
                    .filter(|&k| {
                        ancestors[a_big.as_usize()].contains(k)
                            || ancestors[b_big.as_usize()].contains(k)
                    })
                    .collect();
                !admg
                    .is_m_separated(a_big, b_big, &given, &mut workspace)
                    .map_err(IdentificationError::from)?
            };
            if !adjacent {
                continue;
            }
            let inserted = if ancestors[b_big.as_usize()].contains(a_big) {
                marginal.insert_directed(a_inner, b_inner)
            } else if ancestors[a_big.as_usize()].contains(b_big) {
                marginal.insert_directed(b_inner, a_inner)
            } else {
                marginal.insert_bidirected(a_inner, b_inner)
            };
            inserted.map_err(IdentificationError::from)?;
        }
    }
    Ok(marginal)
}

/// Canonical text of a result's first functional, the form candidates are compared by.
fn functional_form(result: &IdentificationResult) -> Option<String> {
    let estimand = result.estimands.first()?;
    ((estimand.functional.raw() as usize) < result.arena.len())
        .then(|| result.arena.pretty(estimand.functional))
}

/// Publish the deterministic-subset limitation whenever completion enumeration
/// or the finite equivalence audit could not establish the complete class.
fn mark_incomplete_window(
    envelope: &mut IdentificationEnvelope<TemporalCompletionGraph>,
    sampler: &TemporalPagCompletionSampler,
    finite_audit: Option<bool>,
    max_completions: usize,
) {
    let incomplete =
        sampler.hit_cap() || sampler.class_audit_incomplete() || finite_audit.is_none();
    if incomplete {
        envelope.truncated_completions = envelope.truncated_completions.max(1);
        envelope.invariant = None;
        let stopped = sampler
            .search_stop()
            .map_or_else(String::new, |stop| format!(" (search stopped: {})", stop.code()));
        envelope.push_features([GraphFeature {
            kind: Arc::from("completion_enumeration_capped"),
            detail: Arc::from(format!(
                "retained {} temporal completion case(s) under max_completions={}{stopped}; \
                 identification is established only over the deterministic retained subset",
                envelope.cases.len(),
                max_completions,
            )),
        }]);
        if envelope.status == IdentificationStatus::NonparametricallyIdentified {
            envelope.status = IdentificationStatus::PartiallyIdentified;
        }
    }
}

impl GeneralizedAdjustmentIdentifier {
    /// Identify a natural direct, indirect or total effect over the window of a temporal PAG.
    ///
    /// Each stationary MAG completion of `pag` is unfolded over the window one template lag
    /// further back than `window`, marginalized onto `window` and read through the ADMGs that
    /// share the resulting MAG (see the module documentation). Completions whose candidates
    /// all agree on one functional identify the effect by it, candidates that differ yield one
    /// case per distinct functional, and a refused candidate leaves the completion
    /// unidentified. Every case carries the window's indexer.
    ///
    /// # Errors
    ///
    /// A template the completion search rejects, or a window the graph crate cannot hold.
    pub fn identify_temporal_pag_path_specific_envelope(
        &self,
        pag: &TemporalPag,
        window: &TemporalPagMediationWindow,
    ) -> Result<TemporalClassEnvelope, IdentificationError> {
        let inner = &window.indexer;
        let depth = longest_lag(pag);
        let history = inner.history().checked_add(depth).ok_or_else(|| {
            IdentificationError::out_of_range("temporal completion history overflow")
        })?;
        let big = TemporalIndexer::new(inner.variable_count(), history, inner.horizon())
            .map_err(|error| IdentificationError::construction(error))?;
        let shift = depth as usize * inner.variable_count() as usize;
        let mut seeds = vec![window.query.treatment.as_usize(), window.query.outcome.as_usize()];
        seeds.extend(window.query.mediators.iter().map(|m| m.as_usize()));
        let (limits, ctx) = self.config.completion_search();
        let mut temporal_budget = self.config.temporal_budget.clone().unwrap_or_else(|| {
            antecedent_core::TemporalBudget::new(ctx.temporal_materialization_limits())
        });
        let mut sampler = TemporalPagCompletionSampler::for_window_with_borrowed_temporal_budget(
            pag,
            self.config.max_completions,
            &big,
            limits,
            &mut temporal_budget,
            &ctx,
        )
        .map_err(crate::error::temporal_graph_error)?;
        let query = CausalQuery::Mediation((*window.query).clone());
        let identifier = PathSpecificIdentifier::new();
        let mut workspace = IdentificationWorkspace::default();
        let mut cases = Vec::new();
        let mut unfolded = Vec::new();
        let (mut completions, mut candidates, mut distinct) = (0usize, 0usize, 0usize);
        for completion in sampler.by_ref() {
            completions += 1;
            let finite = completion
                .graph
                .unfold_within(big.clone(), &mut temporal_budget, &ctx)
                .map_err(crate::error::temporal_graph_error)?;
            let results = if antecedent_graph::completion::is_mag_completion(&finite.pag) {
                let marginal = window_mag(&finite.pag, inner, shift, &seeds, &ctx)?;
                self.window_candidate_results(
                    &marginal,
                    &query,
                    &identifier,
                    &mut workspace,
                    &mut candidates,
                    &mut distinct,
                )?
            } else {
                let mut result = not_identified(
                    query.clone(),
                    "stationary temporal completion is not a maximal ancestral graph in the \
                     mediation window",
                );
                result.diagnostics.push(Diagnostic::new(
                    TEMPORAL_PAG_WINDOW_NOT_MAG_DIAGNOSTIC_CODE,
                    DiagnosticKind::Scientific,
                    DiagnosticSeverity::Info,
                    "the unfolded completion is not a maximal ancestral graph, so its \
                     path-specific candidates are not defined",
                ));
                vec![result]
            };
            unfolded.push(finite);
            #[allow(
                clippy::cast_precision_loss,
                reason = "a completion carries a handful of candidate functionals"
            )]
            let weight =
                ProbabilityMass(self.config.per_completion_weight / results.len().max(1) as f64);
            for mut result in results {
                result.retain_temporal_lease(std::sync::Arc::clone(&window._lease));
                cases.push(GraphIdentificationCase {
                    graph: TemporalCompletionGraph::Mag(completion.graph.clone()),
                    result,
                    weight,
                });
            }
        }
        crate::generalized::check_completion_stop(sampler.search_stop())?;
        let indexers = vec![inner.clone(); cases.len()];
        let mut envelope = IdentificationEnvelope::from_cases(cases);
        let report = sampler.validation_report();
        let refs: Vec<&antecedent_graph::Pag> = unfolded.iter().map(|window| &window.pag).collect();
        let finite_audit =
            antecedent_graph::completion::audit_finite_mag_equivalence_refs_within(&refs, &ctx)
                .map_err(crate::error::temporal_graph_error)?;
        envelope.push_features([GraphFeature {
            kind: Arc::from("temporal_pag_path_specific_window"),
            detail: Arc::from(format!(
                "{completions} stationary completion(s) unfolded over {} slice(s) and \
                 marginalized onto the {}-slice mediation window, read through {candidates} \
                 ADMG candidate(s) sharing their m-separation models; {distinct} distinct \
                 functional(s); stationary assignments={}, rejected_temporal_constraints={}, \
                 represented={}, finite_query_equivalence={finite_audit:?}",
                big.history() + big.horizon(),
                inner.history() + inner.horizon(),
                report.assignments_examined,
                report.rejected_constraints,
                report.represented_completions,
            )),
        }]);
        if finite_audit == Some(false) {
            return Err(IdentificationError::NotCertified {
                message: "temporal PAG endpoint refinements have different m-separation models \
                          in the unfolded mediation window",
            });
        }
        mark_incomplete_window(&mut envelope, &sampler, finite_audit, self.config.max_completions);
        Ok(TemporalClassEnvelope { envelope, indexers })
    }

    /// The distinct functionals of the ADMGs sharing `marginal`, or the one refusal when a
    /// candidate leaves the effect unidentified or the enumeration is bounded.
    fn window_candidate_results(
        &self,
        marginal: &Pag,
        query: &CausalQuery,
        identifier: &PathSpecificIdentifier,
        workspace: &mut IdentificationWorkspace,
        candidates: &mut usize,
        distinct: &mut usize,
    ) -> Result<Vec<IdentificationResult>, IdentificationError> {
        let found = self.admg_candidates_of_mag(marginal)?;
        *candidates += found.members.len();
        if found.bounded {
            let result = crate::pag_path_specific::search_bounded_result(query);
            return Ok(vec![result]);
        }
        let mut forms: Vec<Option<String>> = Vec::new();
        let mut results = Vec::new();
        for admg in &found.members {
            let prepared = identifier.prepare(admg)?;
            let mut result = identifier.identify(&prepared, query, workspace)?;
            if !carries_identified_mass(result.status) {
                result.diagnostics.push(Diagnostic::new(
                    "identify.path_specific_pag.unidentified_candidate",
                    DiagnosticKind::Scientific,
                    DiagnosticSeverity::Info,
                    format!(
                        "one of the {} ADMG(s) with this window MAG leaves the effect \
                         unidentified; no functional is correct for every model the MAG \
                         represents",
                        found.members.len()
                    ),
                ));
                return Ok(vec![result]);
            }
            let form = functional_form(&result);
            if forms.contains(&form) {
                continue;
            }
            forms.push(form);
            results.push(result);
        }
        *distinct += results.len();
        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use antecedent_core::{Lag, MediationContrast, Value};

    fn v(raw: u32) -> VariableId {
        VariableId::from_raw(raw)
    }

    fn lagged(graph: &mut TemporalPag, variable: u32, lag: u32) -> DenseNodeId {
        graph.add_lagged(v(variable), Lag::from_raw(lag)).unwrap()
    }

    /// x -> m -> y within a period and x(t-1) -> x(t): treatment 0, mediator 1, outcome 2.
    fn chain() -> TemporalPag {
        let mut graph = TemporalPag::empty();
        let x = lagged(&mut graph, 0, 0);
        let m = lagged(&mut graph, 1, 0);
        let y = lagged(&mut graph, 2, 0);
        let x_past = lagged(&mut graph, 0, 1);
        graph.insert_directed(x, m).unwrap();
        graph.insert_directed(m, y).unwrap();
        graph.insert_directed(x_past, x).unwrap();
        graph
    }

    fn mediation(contrast: MediationContrast) -> MediationQuery {
        MediationQuery::binary(v(0), v(2), [v(1)], contrast)
    }

    fn pulse() -> TemporalEffectQuery {
        TemporalEffectQuery::pulse(v(0), v(2), 1.0).with_horizon_steps(1)
    }

    #[test]
    fn mediation_window_charges_expansion_and_shares_its_owner() {
        let ctx = antecedent_core::ExecutionContext::for_tests(2);
        let mut limits = ctx.temporal_materialization_limits();
        limits.max_offsets = 0;
        let mut denied = antecedent_core::TemporalBudget::new(limits);
        assert!(matches!(
            temporal_pag_mediation_window_within(
                &chain(),
                &pulse(),
                &mediation(MediationContrast::Total),
                3,
                &mut denied,
                &ctx
            ),
            Err(IdentificationError::Budget { .. })
        ));
        let mut allowed =
            antecedent_core::TemporalBudget::new(ctx.temporal_materialization_limits());
        let window = temporal_pag_mediation_window_within(
            &chain(),
            &pulse(),
            &mediation(MediationContrast::Total),
            3,
            &mut allowed,
            &ctx,
        )
        .unwrap();
        let clone = window.clone();
        assert!(std::sync::Arc::ptr_eq(&window.query, &clone.query));
        assert!(std::sync::Arc::ptr_eq(&window._lease, &clone._lease));
        let mut cancelled = antecedent_core::ExecutionContext::for_tests(3);
        cancelled.cancellation = antecedent_core::CancellationToken::cancel_after_checks(0);
        let mut budget =
            antecedent_core::TemporalBudget::new(cancelled.temporal_materialization_limits());
        assert!(matches!(
            temporal_pag_mediation_window_within(
                &chain(),
                &pulse(),
                &mediation(MediationContrast::Total),
                3,
                &mut budget,
                &cancelled
            ),
            Err(IdentificationError::Cancelled)
        ));
    }

    #[test]
    fn marginal_window_pair_search_observes_context_cancellation() {
        let graph = Pag::with_variables(3);
        let inner = TemporalIndexer::new(3, 0, 1).unwrap();
        let mut ctx = antecedent_core::ExecutionContext::for_tests(1);
        ctx.cancellation = antecedent_core::CancellationToken::cancel_after_checks(2);
        assert!(matches!(
            window_mag(&graph, &inner, 0, &[0, 1, 2], &ctx),
            Err(IdentificationError::Cancelled)
        ));
    }

    #[test]
    fn temporal_history_overflow_refuses_before_unfolding() {
        let mut pag = chain();
        lagged(&mut pag, 0, u32::MAX);
        let earlier = pulse().with_policy(TemporalPolicy::pulse(-1));
        assert!(matches!(
            temporal_pag_mediation_window(&pag, &earlier, &mediation(MediationContrast::Total), 3),
            Err(IdentificationError::OutOfRange { .. })
        ));
        let window = TemporalPagMediationWindow {
            indexer: TemporalIndexer::new(3, 1, 1).unwrap(),
            query: std::sync::Arc::new(mediation(MediationContrast::Total)),
            treatment_offset: 0,
            outcome_offset: 0,
            _lease: std::sync::Arc::new(
                antecedent_core::ExecutionContext::production_default(0)
                    .acquire_admission(0, 0)
                    .unwrap(),
            ),
        };
        assert!(matches!(
            GeneralizedAdjustmentIdentifier::new()
                .identify_temporal_pag_path_specific_envelope(&pag, &window),
            Err(IdentificationError::OutOfRange { .. })
        ));
    }

    #[test]
    fn the_window_states_the_contrast_on_indexer_coordinates() {
        let window = temporal_pag_mediation_window(
            &chain(),
            &pulse(),
            &mediation(MediationContrast::Total),
            3,
        )
        .unwrap();
        // One template lag of history, the pulse period only.
        assert_eq!(window.indexer.history(), 1);
        assert_eq!(window.indexer.horizon(), 1);
        let key = |id: VariableId| window.indexer.key_of(id.raw()).unwrap();
        assert_eq!(key(window.query.treatment), TemporalNodeKey { variable: v(0), offset: 0 });
        assert_eq!(key(window.query.outcome), TemporalNodeKey { variable: v(2), offset: 0 });
        assert_eq!(
            window.query.mediators.iter().map(|m| key(*m)).collect::<Vec<_>>(),
            vec![TemporalNodeKey { variable: v(1), offset: 0 }]
        );
        match &window.query.active {
            Intervention::Set { value, .. } => assert_eq!(value, &Value::f64(1.0)),
            other => panic!("expected a Set level, got {other:?}"),
        }
    }

    #[test]
    fn a_horizon_without_a_mediator_period_is_refused() {
        let mut query = mediation(MediationContrast::Total);
        query.mediators = Arc::from([v(0)]);
        // The only mediator coordinate is the treatment's own, so none remains in the window.
        assert!(temporal_pag_mediation_window(&chain(), &pulse(), &query, 3).is_err());
    }

    #[test]
    fn the_marginal_window_keeps_a_pre_window_common_cause_as_a_bidirected_edge() {
        // u(t-1) -> a(t), u(t-1) -> b(t): marginalizing u(t-1) out leaves a <-> b, never the
        // absent edge the induced subgraph would show.
        let mut graph = TemporalPag::empty();
        let u = lagged(&mut graph, 0, 1);
        let a = lagged(&mut graph, 1, 0);
        let b = lagged(&mut graph, 2, 0);
        graph.insert_directed(u, a).unwrap();
        graph.insert_directed(u, b).unwrap();
        // Window: no history beyond the pulse period at offset 0; one further lag unfolded.
        let inner = TemporalIndexer::new(3, 0, 1).unwrap();
        let big = TemporalIndexer::new(3, 1, 1).unwrap();
        let unfolded = graph.unfold(big).unwrap();
        let a_id = inner.dense_id(TemporalNodeKey { variable: v(1), offset: 0 }).unwrap() as usize;
        let b_id = inner.dense_id(TemporalNodeKey { variable: v(2), offset: 0 }).unwrap() as usize;
        let marginal = window_mag(
            &unfolded.pag,
            &inner,
            3,
            &[a_id, b_id],
            &antecedent_core::ExecutionContext::for_tests(1),
        )
        .unwrap();
        let edge = marginal
            .edge_between(DenseNodeId::from_index(a_id), DenseNodeId::from_index(b_id))
            .expect("the pre-window cause leaves a confounding edge");
        assert_eq!((edge.at_a, edge.at_b), (Endpoint::Arrow, Endpoint::Arrow));
    }

    #[test]
    fn the_marginal_window_orients_a_pre_window_ancestor_through_the_window() {
        // x(t-2) -> x(t-1) -> x(t): with x(t-1) in the window the edge x(t-1) -> x(t) is
        // directed, and x(t-2) is no node of the window.
        let mut graph = TemporalPag::empty();
        let x0 = lagged(&mut graph, 0, 0);
        let x1 = lagged(&mut graph, 0, 1);
        graph.insert_directed(x1, x0).unwrap();
        let inner = TemporalIndexer::new(1, 1, 1).unwrap();
        let big = TemporalIndexer::new(1, 2, 1).unwrap();
        let unfolded = graph.unfold(big).unwrap();
        let marginal = window_mag(
            &unfolded.pag,
            &inner,
            1,
            &[1],
            &antecedent_core::ExecutionContext::for_tests(1),
        )
        .unwrap();
        let edge = marginal
            .edge_between(DenseNodeId::from_index(0), DenseNodeId::from_index(1))
            .expect("x(t-1) -> x(t) survives");
        assert_eq!((edge.at_a, edge.at_b), (Endpoint::Tail, Endpoint::Arrow));
    }
    #[test]
    fn mediation_window_honors_caller_cancellation_and_memory_without_changing_enumeration() {
        let graph = chain();
        let window = temporal_pag_mediation_window(
            &graph,
            &pulse(),
            &mediation(MediationContrast::Total),
            3,
        )
        .unwrap();
        let ordinary = GeneralizedAdjustmentIdentifier::new()
            .identify_temporal_pag_path_specific_envelope(&graph, &window)
            .unwrap();
        assert!(ordinary.envelope.cases.iter().all(|case| !case.result.temporal_leases.is_empty()));
        let cancelled = antecedent_core::ExecutionContext::for_tests(84);
        cancelled.cancellation.cancel();
        let configured = |ctx| GeneralizedAdjustmentIdentifier {
            config: crate::GeneralizedAdjustmentConfig { context: Some(ctx), ..Default::default() },
        };
        assert!(matches!(
            configured(cancelled).identify_temporal_pag_path_specific_envelope(&graph, &window),
            Err(IdentificationError::Cancelled)
        ));
        let limited = antecedent_core::ExecutionContext::for_tests(84).with_memory_limit(Some(1));
        assert!(matches!(
            configured(limited).identify_temporal_pag_path_specific_envelope(&graph, &window),
            Err(IdentificationError::Budget { budget: crate::IdentificationBudget::Memory })
        ));
        let actual = configured(antecedent_core::ExecutionContext::for_tests(85))
            .identify_temporal_pag_path_specific_envelope(&graph, &window)
            .unwrap();
        assert_eq!(ordinary.indexers, actual.indexers);
        let (ordinary, actual) = (ordinary.envelope, actual.envelope);
        assert_eq!(ordinary.status, actual.status);
        assert_eq!(ordinary.invariant, actual.invariant);
        assert_eq!(ordinary.identified_weight, actual.identified_weight);
        assert_eq!(ordinary.unidentified_weight, actual.unidentified_weight);
        assert_eq!(ordinary.truncated_completions, actual.truncated_completions);
        assert_eq!(ordinary.critical_graph_features, actual.critical_graph_features);
        assert_eq!(ordinary.cases.len(), actual.cases.len());
        for (expected, observed) in ordinary.cases.iter().zip(&actual.cases) {
            // Compare each slot's canonical marked graph, then its complete scientific
            // result and ordered arena tables. Hash-cons lookup-map iteration is immaterial.
            assert_eq!(expected.graph.digest(), observed.graph.digest());
            assert_eq!(expected.weight, observed.weight);
            let (expected, observed) = (&expected.result, &observed.result);
            assert_eq!(expected.status, observed.status);
            assert_eq!(expected.query, observed.query);
            assert_eq!(expected.estimands, observed.estimands);
            assert_eq!(expected.derivation, observed.derivation);
            assert_eq!(expected.required_assumptions, observed.required_assumptions);
            assert_eq!(expected.diagnostics, observed.diagnostics);
            assert_eq!(expected.performance, observed.performance);
            assert_eq!(expected.hedge, observed.hedge);
            assert_eq!(expected.estimand_claims, observed.estimand_claims);
            let (expected, observed) = (&expected.arena, &observed.arena);
            assert_eq!(expected.len(), observed.len());
            assert_eq!(expected.var_set_count(), observed.var_set_count());
            assert_eq!(expected.intervention_set_count(), observed.intervention_set_count());
            assert_eq!(expected.list_count(), observed.list_count());
            for index in 0..expected.len() {
                let id = antecedent_expr::ExprId::from_raw(u32::try_from(index).unwrap());
                assert_eq!(expected.node(id), observed.node(id));
                assert_eq!(expected.derivation(id), observed.derivation(id));
            }
            for index in 0..expected.var_set_count() {
                let id = antecedent_expr::VarSetId::from_raw(u32::try_from(index).unwrap());
                assert_eq!(expected.var_set(id), observed.var_set(id));
            }
            for index in 0..expected.intervention_set_count() {
                let id =
                    antecedent_expr::InterventionSetId::from_raw(u32::try_from(index).unwrap());
                assert_eq!(
                    expected.intervention_assignments(id),
                    observed.intervention_assignments(id)
                );
            }
            for index in 0..expected.list_count() {
                let id = antecedent_expr::ExprListId::from_raw(u32::try_from(index).unwrap());
                assert_eq!(expected.list(id), observed.list(id));
            }
        }
    }
}
