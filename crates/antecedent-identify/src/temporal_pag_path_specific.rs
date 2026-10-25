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
    pub query: MediationQuery,
    /// Offset of the pulse from the window origin.
    pub treatment_offset: i32,
    /// Offset of the outcome from the window origin.
    pub outcome_offset: i32,
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
    let outcome_offset = effect.outcome_offset();
    if outcome_offset < treatment_offset {
        return Err(IdentificationError::unsupported(
            "the outcome period precedes the pulse period of the mediation",
        ));
    }
    let schema = |variable: VariableId| -> Result<(), IdentificationError> {
        if variable.raw() < variable_count {
            Ok(())
        } else {
            Err(IdentificationError::msg(format!(
                "variable {} is outside the series schema of {variable_count} variables",
                variable.raw()
            )))
        }
    };
    let mut longest_lag = 0u32;
    for node in template.nodes() {
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
    let history = longest_lag + treatment_offset.min(0).unsigned_abs();
    let last = u32::try_from(outcome_offset.max(0))
        .map_err(|_| IdentificationError::msg("the outcome offset exceeds the window range"))?;
    let indexer = TemporalIndexer::new(variable_count, history, last + 1)
        .map_err(|error| IdentificationError::msg(error.to_string()))?;
    let coordinate =
        |variable: VariableId, offset: i32| -> Result<VariableId, IdentificationError> {
            indexer
                .dense_id(TemporalNodeKey { variable, offset })
                .map(VariableId::from_raw)
                .map_err(|error| IdentificationError::msg(error.to_string()))
        };
    let treatment = coordinate(query.treatment, treatment_offset)?;
    let outcome = coordinate(query.outcome, outcome_offset)?;
    let mut mediators = Vec::new();
    for &variable in query.mediators.iter() {
        for offset in treatment_offset..=outcome_offset {
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
    Ok(TemporalPagMediationWindow { indexer, query: window, treatment_offset, outcome_offset })
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
        .map_err(|_| IdentificationError::msg("window exceeds the node id range"))?;
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
) -> Result<Pag, IdentificationError> {
    let admg = mag_as_admg(big)?;
    let ancestors = admg.strict_ancestor_sets();
    let n = window.dense_len();
    let mut keep = vec![false; n];
    for &seed in seeds {
        if seed >= n {
            return Err(IdentificationError::msg("a seed lies outside the window"));
        }
        keep[seed] = true;
        for (inner, retained) in keep.iter_mut().enumerate() {
            if ancestors[seed + shift].contains(DenseNodeId::from_index(inner + shift)) {
                *retained = true;
            }
        }
    }
    let nodes = u32::try_from(n)
        .map_err(|_| IdentificationError::msg("window exceeds the node id range"))?;
    let mut marginal = Pag::with_variables(nodes);
    let mut workspace = DSeparationWorkspace::default();
    let kept: Vec<usize> = (0..n).filter(|&i| keep[i]).collect();
    for (position, &i) in kept.iter().enumerate() {
        let (a_big, a_inner) = (DenseNodeId::from_index(i + shift), DenseNodeId::from_index(i));
        for &j in &kept[position + 1..] {
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
        let big =
            TemporalIndexer::new(inner.variable_count(), inner.history() + depth, inner.horizon())
                .map_err(|error| IdentificationError::msg(error.to_string()))?;
        let shift = depth as usize * inner.variable_count() as usize;
        let mut seeds = vec![window.query.treatment.as_usize(), window.query.outcome.as_usize()];
        seeds.extend(window.query.mediators.iter().map(|m| m.as_usize()));
        let (limits, ctx) = self.config.completion_search();
        let mut sampler = TemporalPagCompletionSampler::for_window_with_budget(
            pag.clone(),
            self.config.max_completions,
            &big,
            limits,
            &ctx,
        )?;
        let query = CausalQuery::Mediation(window.query.clone());
        let identifier = PathSpecificIdentifier::new();
        let mut workspace = IdentificationWorkspace::default();
        let mut cases = Vec::new();
        let mut unfolded = Vec::new();
        let (mut completions, mut candidates, mut distinct) = (0usize, 0usize, 0usize);
        for completion in sampler.by_ref() {
            completions += 1;
            let finite = completion.graph.unfold(big.clone())?;
            let results = if antecedent_graph::completion::is_mag_completion(&finite.pag) {
                let marginal = window_mag(&finite.pag, inner, shift, &seeds)?;
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
            unfolded.push(finite.pag);
            #[allow(
                clippy::cast_precision_loss,
                reason = "a completion carries a handful of candidate functionals"
            )]
            let weight =
                ProbabilityMass(self.config.per_completion_weight / results.len().max(1) as f64);
            for result in results {
                cases.push(GraphIdentificationCase {
                    graph: TemporalCompletionGraph::Mag(completion.graph.clone()),
                    result,
                    weight,
                });
            }
        }
        let indexers = vec![inner.clone(); cases.len()];
        let mut envelope = IdentificationEnvelope::from_cases(cases);
        let report = sampler.validation_report();
        let finite_audit = antecedent_graph::completion::audit_finite_mag_equivalence(&unfolded);
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
            let mut result = not_identified(
                query.clone(),
                "the ADMG candidates of this MAG exceed the enumeration bound",
            );
            result.diagnostics.push(Diagnostic::new(
                crate::PAG_ADMG_SEARCH_BOUNDED_DIAGNOSTIC_CODE,
                DiagnosticKind::Execution,
                DiagnosticSeverity::Warning,
                "the ADMGs that share this MAG's m-separation model could not all be enumerated \
                 within the search bound, so the path-specific effect was not decided on every \
                 one of them; the completion is refused, not read as a point",
            ));
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
        let marginal = window_mag(&unfolded.pag, &inner, 3, &[a_id, b_id]).unwrap();
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
        let marginal = window_mag(&unfolded.pag, &inner, 1, &[1]).unwrap();
        let edge = marginal
            .edge_between(DenseNodeId::from_index(0), DenseNodeId::from_index(1))
            .expect("x(t-1) -> x(t) survives");
        assert_eq!((edge.at_a, edge.at_b), (Endpoint::Tail, Endpoint::Arrow));
    }
}
