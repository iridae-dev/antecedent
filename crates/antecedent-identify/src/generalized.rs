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
//! `T` and `Y` in `G_{\underline{T}}` (outgoing edges from `T` removed). Whether any set
//! exists is decided first by one m-separation test on the ancestral candidates
//! `An({T,Y})` within that family (Perkovic et al. 2018, Thm. 5): if it fails, no
//! generalized adjustment set exists on the completion and the search does not run.
//! Enumeration is by increasing set size and stops at the first valid set (minimal-first);
//! if `max_examinations` ends it early the ancestral set, already certified, is returned
//! with an Execution diagnostic. Completions that are not MAGs, or MAGs with no qualifying
//! set in this candidate family, contribute unidentified mass. A completion whose
//! ancestral candidates pass that test but whose candidate family exceeds
//! `max_candidates` is identified by the ancestral set, which the test already certified
//! (valid, not minimal): only the choice of a smaller set is skipped, and the result carries
//! an Execution diagnostic [`MAG_SEARCH_BOUNDED_DIAGNOSTIC_CODE`]. The joint multi-treatment
//! search decides existence the same way (one test per treatment on the maximal candidate set,
//! see the `joint_response` module) and returns that set, with the Execution diagnostic
//! [`crate::JOINT_SEARCH_BOUNDED_DIAGNOSTIC_CODE`], when the search for a smaller one is
//! stopped. A decided absence of any set is [`IdentificationStatus::NotIdentified`] with a
//! Scientific diagnostic.
//!
//! This is **generalized adjustment**, not the full ID/IDC algorithm (see roadmap P5.3).
//! Sets outside the ancestor candidate family are not searched.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{
    AssumptionSet, AverageEffectQuery, CausalQuery, Diagnostic, DiagnosticKind, DiagnosticSeverity,
    ExecutionContext, ResponseQuery, Value, VariableId,
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

mod conditional;
pub(crate) use conditional::validate_dag_conditional_adjustment;
#[cfg(test)]
use conditional::{ConditionalSearch, constrained_conditional_set};
use conditional::{fund_adjustment_enumeration, validate_conditional_adjustment};

/// Diagnostic code of a modifier-constrained adjustment search that spent
/// `max_examinations` before it finished: identifiability is undecided, not refuted.
pub(crate) const CONDITIONAL_SEARCH_BOUNDED_DIAGNOSTIC_CODE: &str =
    "identify.conditional.search_bounded";

/// Diagnostic code of a modifier-constrained adjustment search that stopped at `max_results`
/// sets. Every returned set is certified; only the list of alternatives is incomplete, so this
/// is not a bounded-search (undecided) outcome.
pub(crate) const CONDITIONAL_RESULTS_TRUNCATED_DIAGNOSTIC_CODE: &str =
    "identify.conditional.results_truncated";

/// Diagnostic code of a per-completion [`IdentificationResult`] whose adjustment-set search
/// was capped before it could decide — as opposed to searching exhaustively and finding no
/// valid set. Such a result keeps [`IdentificationStatus::NotIdentified`] and is an Execution
/// diagnostic, not a scientific open-back-door. The in-tree searches no longer stop undecided
/// (each decides existence in one test and falls back to the certified set), so none emits it;
/// it stays recognised by [`search_truncated`](crate::envelope::search_truncated) for stored
/// and externally built results, which [`IdentificationEnvelope::truncated_completions`]
/// counts.
pub const CAPPED_COMPLETION_DIAGNOSTIC_CODE: &str =
    "identify.generalized_adjustment.completion_capped";

/// Resource interruption cannot be read as a scientific or declared work-cap outcome.
pub(crate) fn check_completion_stop(
    stop: Option<antecedent_core::SearchStop>,
) -> Result<(), IdentificationError> {
    match stop {
        Some(antecedent_core::SearchStop::Cancelled) => Err(IdentificationError::Cancelled),
        Some(antecedent_core::SearchStop::Memory) => {
            Err(IdentificationError::Budget { budget: crate::error::IdentificationBudget::Memory })
        }
        _ => Ok(()),
    }
}

/// Config for PAG generalized-adjustment envelopes.
#[derive(Clone, Debug)]
pub struct GeneralizedAdjustmentConfig {
    /// Max completions to stream (hard bound; never retain unbounded).
    pub max_completions: usize,
    /// Uniform weight per streamed completion.
    pub per_completion_weight: f64,
    /// Caller-declared ceiling on the candidate covariates of an exact enumeration. The default
    /// (`usize::MAX`) declares none: the enumeration's storage is leased from the context's
    /// admission pool before it is allocated and its work is charged to
    /// [`Self::max_examinations`].
    pub max_candidates: usize,
    /// The caller's execution context: its admission pool funds every search's storage and
    /// completion enumeration, and its cancellation token stops them. `None` runs under a
    /// serial production context with an unlimited pool.
    pub context: Option<ExecutionContext>,
    /// Shared cumulative budget for nested temporal materializations, if this search is part of a larger operation.
    pub temporal_budget: Option<antecedent_core::TemporalBudget>,
    /// Separation tests one adjustment search may run before it stops looking for a smaller
    /// set (work budget, not an output cap).
    ///
    /// Whether a single-treatment or a joint multi-treatment set exists is decided by one
    /// test on the maximal candidate set, which is returned when the budget ends the search
    /// for a smaller one.
    pub max_examinations: u64,
    /// Operations the completion search may charge: one per endpoint assignment or partial
    /// orientation visited, and one per m-separation query of the equivalence audit. A stop is
    /// a resource outcome that withdraws every class-wide claim (the envelope is at most
    /// partially identified), never a statement that the unvisited completions identify.
    pub completion_operations: usize,
}

impl GeneralizedAdjustmentConfig {
    /// Limits and context of the completion search these settings configure.
    pub(crate) fn completion_search(&self) -> (antecedent_core::SearchLimits, ExecutionContext) {
        (
            antecedent_core::SearchLimits {
                operations: self.completion_operations,
                depth: usize::MAX,
            },
            self.context.clone().unwrap_or_else(|| ExecutionContext::production(0, 1)),
        )
    }
}

impl Default for GeneralizedAdjustmentConfig {
    fn default() -> Self {
        Self {
            max_completions: 32,
            per_completion_weight: 1.0,
            max_candidates: usize::MAX,
            context: None,
            temporal_budget: None,
            max_examinations: 1_000_000,
            completion_operations: antecedent_graph::DEFAULT_COMPLETION_OPERATIONS,
        }
    }
}

/// Diagnostic code of a MAG generalized-adjustment search that spent `max_examinations`
/// before it found a smaller set. The set it returns is valid; it is not minimal.
pub const MAG_SEARCH_BOUNDED_DIAGNOSTIC_CODE: &str =
    "identify.generalized_adjustment.search_bounded";

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
                MagTarget { t, y, t_d, y_d },
                active.clone(),
                control.clone(),
                self.config.budget(),
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
                        context: self.config.context.clone(),
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
        self.pag_envelope_with_candidates(pag, |completion| identify(completion).map(|r| vec![r]))
    }

    /// The completion envelope where one MAG completion can carry several cases: `identify`
    /// returns one result per distinct functional the completion leaves open, and the
    /// completion's mass is shared equally between them.
    pub(crate) fn pag_envelope_with_candidates(
        &self,
        pag: &Pag,
        mut identify: impl FnMut(&Pag) -> Result<Vec<IdentificationResult>, IdentificationError>,
    ) -> Result<IdentificationEnvelope<Pag>, IdentificationError> {
        let (limits, ctx) = self.config.completion_search();
        let mut sampler =
            CompletionSampler::with_budget(pag, self.config.max_completions, limits, &ctx)
                .map_err(IdentificationError::from)?;
        let mut cases = Vec::new();
        for completion in sampler.by_ref() {
            let results = identify(&completion.graph)?;
            #[allow(
                clippy::cast_precision_loss,
                reason = "a completion carries a handful of candidate functionals"
            )]
            let w =
                ProbabilityMass(self.config.per_completion_weight / results.len().max(1) as f64);
            for result in results {
                cases.push(GraphIdentificationCase {
                    graph: completion.graph.clone(),
                    result,
                    weight: w,
                });
            }
        }
        check_completion_stop(sampler.search_stop())?;
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
        // The global class audit is exponential in node count and is skipped when one
        // signature exceeds the search budget; a budget stop leaves the class only partly
        // enumerated. Local validity still holds for every case, but "identified in every
        // member of the equivalence class" is exactly the claim the incomplete audit would
        // have supported, so it is downgraded here rather than asserted.
        if sampler.class_audit_incomplete() {
            let reason = sampler.search_stop().map_or_else(
                || {
                    "the global m-separation equivalence audit was too expensive for this graph"
                        .to_owned()
                },
                |stop| {
                    format!(
                        "the completion search stopped ({}) before every endpoint assignment \
                         was classified",
                        stop.code()
                    )
                },
            );
            envelope.push_features([GraphFeature {
                kind: Arc::from("completion_equivalence_audit_skipped"),
                detail: Arc::from(format!(
                    "{reason}; completions are locally valid but not certified to form one \
                     Markov class"
                )),
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

        let mut backdoor =
            BackdoorIdentifier::new().with_max_candidates(self.config.max_candidates);
        backdoor.config.context.clone_from(&self.config.context);
        let mut workspace = IdentificationWorkspace::default();
        let cq = CausalQuery::AverageEffect(query.clone());
        self.cpdag_envelope_with(cpdag, |graph| {
            let prepared = backdoor.prepare(graph)?;
            backdoor.identify(&prepared, &cq, &mut workspace)
        })
    }

    /// Identify a path-restricted natural effect, or a natural direct effect, in every DAG
    /// completion of a CPDAG.
    ///
    /// A CPDAG assumes causal sufficiency, so each completion is a DAG and the
    /// Avin-Shpitser-Pearl identifier decides it on its own: a completion with a recanting
    /// witness carries unidentified mass, and completions that identify different
    /// functionals leave the effect partially identified over the class.
    ///
    /// # Errors
    ///
    /// A query the path-specific identifier does not read, or a completion that fails to
    /// prepare.
    pub fn identify_cpdag_path_specific_envelope(
        &self,
        cpdag: &Cpdag,
        query: &CausalQuery,
    ) -> Result<IdentificationEnvelope<antecedent_graph::Dag>, IdentificationError> {
        let identifier = crate::path_specific::PathSpecificIdentifier::new();
        let mut workspace = IdentificationWorkspace::default();
        self.cpdag_envelope_with(cpdag, |graph| {
            let prepared = identifier.prepare_dag(graph)?;
            identifier.identify(&prepared, query, &mut workspace)
        })
    }

    /// Identify an interventional distribution in every DAG completion of a CPDAG.
    ///
    /// A CPDAG assumes causal sufficiency, so each completion is a DAG read as the ADMG with
    /// no bidirected edge; the complete ID algorithm (IDC when the query conditions) decides
    /// it on its own. A completion that identifies a different functional leaves the
    /// distribution partially identified over the class.
    ///
    /// # Errors
    ///
    /// A query the algorithm does not read, or a completion that fails to prepare.
    pub fn identify_cpdag_distribution_envelope(
        &self,
        cpdag: &Cpdag,
        query: &CausalQuery,
    ) -> Result<IdentificationEnvelope<antecedent_graph::Dag>, IdentificationError> {
        let mut workspace = IdentificationWorkspace::default();
        self.cpdag_envelope_with(cpdag, |dag| {
            let admg =
                antecedent_graph::Admg::from_dag(dag).map_err(IdentificationError::construction)?;
            crate::pag_path_specific::identify_distribution_on(&admg, query, &mut workspace)
        })
    }

    pub(crate) fn cpdag_envelope_with(
        &self,
        cpdag: &Cpdag,
        mut identify: impl FnMut(
            &antecedent_graph::Dag,
        ) -> Result<IdentificationResult, IdentificationError>,
    ) -> Result<IdentificationEnvelope<antecedent_graph::Dag>, IdentificationError> {
        let (limits, ctx) = self.config.completion_search();
        let mut sampler = CpdagCompletionSampler::with_budget(
            cpdag.clone(),
            self.config.max_completions,
            limits,
            &ctx,
        )?;
        let mut cases = Vec::new();
        let w = ProbabilityMass(self.config.per_completion_weight);
        for completion in sampler.by_ref() {
            let result = identify(&completion.graph)?;
            cases.push(GraphIdentificationCase { graph: completion.graph, result, weight: w });
        }
        check_completion_stop(sampler.search_stop())?;
        let mut envelope = IdentificationEnvelope::from_cases(cases);
        envelope.push_features(cpdag_undirected_features(cpdag));
        if sampler.hit_cap() {
            let stopped = sampler
                .search_stop()
                .map_or_else(String::new, |stop| format!(" (search stopped: {})", stop.code()));
            envelope.push_features([GraphFeature {
                kind: Arc::from("completion_enumeration_capped"),
                detail: Arc::from(format!(
                    "retained {} MEC DAG completion(s) under max_completions={}{stopped}; \
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

/// Converts a node position to the `u32` payload of a dense id.
#[allow(
    clippy::cast_possible_truncation,
    reason = "positions index u32-addressed dense node ids, so they are below u32::MAX"
)]
fn dense_u32(index: usize) -> u32 {
    index as u32
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
    crate::prepared::dense_of_static(pag.nodes(), id)
}

fn proper_backdoor_mag(mag: &Pag, t: DenseNodeId, y: DenseNodeId) -> Option<Admg> {
    let graph = mag_to_admg(mag)?;
    let ancestors_y = directed_closure(&graph, &[y], true);
    let causal_children: Vec<_> =
        graph.children(t).iter().copied().filter(|v| ancestors_y.contains(*v)).collect();
    if causal_children.iter().any(|&v| !crate::joint_response::visible(mag, t, v)) {
        return None;
    }
    let mut cut = Admg::with_variables(dense_u32(graph.node_count()));
    for i in 0..graph.node_count() {
        let a = DenseNodeId::from_raw(dense_u32(i));
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
    Found { z_vars: Arc<[VariableId]>, examined: u64, budget_exhausted: bool },
    Failed(IdentificationResult),
}

/// Work limits of one MAG adjustment search.
#[derive(Clone, Copy, Debug)]
pub(crate) struct AdjustmentBudget<'a> {
    pub max_candidates: usize,
    pub max_examinations: u64,
    /// The caller's context, which funds the search's storage and observes cancellation.
    pub context: Option<&'a ExecutionContext>,
}

impl GeneralizedAdjustmentConfig {
    pub(crate) const fn budget(&self) -> AdjustmentBudget<'_> {
        AdjustmentBudget {
            max_candidates: self.max_candidates,
            max_examinations: self.max_examinations,
            context: self.context.as_ref(),
        }
    }
}

/// Treatment and outcome of a MAG adjustment search, as variables and as dense nodes.
#[derive(Clone, Copy, Debug)]
pub(crate) struct MagTarget {
    pub t: VariableId,
    pub y: VariableId,
    pub t_d: DenseNodeId,
    pub y_d: DenseNodeId,
}

fn mag_adjustment_search(
    mag: &Pag,
    target: MagTarget,
    query: CausalQuery,
    budget: AdjustmentBudget<'_>,
) -> Result<MagAdjustment, IdentificationError> {
    let MagTarget { t_d, y_d, .. } = target;
    let AdjustmentBudget { max_candidates, max_examinations, context } = budget;
    let context = crate::exact_search::context_or_default(context);
    let mut exact = crate::exact_search::ExactSearch::open(&context)?;
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

    // A generalized adjustment set exists iff An({T,Y}) restricted to the candidates
    // separates T from Y in the proper back-door graph (Perkovic et al. 2018, Thm. 5, after
    // van der Zander et al. 2014). The ancestors of {T,Y} agree in the cut graph and in the
    // MAG because the cut removes only edges leaving T, so one m-separation test decides
    // existence and the subset search below only picks a small set.
    let ancestral = directed_closure(&admg, &[t_d, y_d], true);
    let constructive: Vec<DenseNodeId> =
        candidates.iter().copied().filter(|&v| ancestral.contains(v)).collect();
    let mut ws = DSeparationWorkspace::default();
    if !mutilated.is_m_separated(t_d, y_d, &constructive, &mut ws)? {
        return Ok(MagAdjustment::Failed(IdentificationResult::not_identified(
            query,
            {
                let mut d = DerivationTrace::default();
                d.push(
                    "generalized.adjustment",
                    "no generalized adjustment set on completion: the ancestral candidate set does \
                     not m-separate treatment from outcome in the proper back-door graph",
                );
                d
            },
            AssumptionSet::default(),
            IdentificationPerformanceRecord { candidates_examined: 1, sets_returned: 0 },
        )));
    }
    // A single adjustment set is retained, so the enumeration's retained storage is one set.
    // Above the declared ceiling the subset search is skipped, not the decision: the ancestral
    // set just tested is a valid adjustment set, so the effect is identified by it (valid, not
    // minimal) and the unfinished search is reported as a bounded search.
    if !fund_adjustment_enumeration(&mut exact, candidates.len(), max_candidates, 1)? {
        let z_vars: Arc<[VariableId]> = constructive
            .iter()
            .map(|&d| mag_dense_to_var(mag, d))
            .collect::<Result<Vec<_>, _>>()?
            .into();
        return Ok(MagAdjustment::Found { z_vars, examined: 1, budget_exhausted: true });
    }

    let search = crate::enum_masks::first_set_by_size(&candidates, 1, max_examinations, |z| {
        exact.observe()?;
        if z == constructive.as_slice() {
            return Ok(true);
        }
        mutilated.is_m_separated(t_d, y_d, z, &mut ws).map_err(IdentificationError::from)
    })?;
    // The budget ended the search for a smaller set; the ancestral set is already certified.
    let z_dense = search.found.unwrap_or(constructive);
    let z_vars: Arc<[VariableId]> =
        z_dense.iter().map(|&d| mag_dense_to_var(mag, d)).collect::<Result<Vec<_>, _>>()?.into();
    Ok(MagAdjustment::Found {
        z_vars,
        examined: search.examined,
        budget_exhausted: search.budget_exhausted,
    })
}

fn mag_adjustment_identified(
    query: CausalQuery,
    z_vars: Arc<[VariableId]>,
    examined: u64,
    budget_exhausted: bool,
    functional: antecedent_expr::ExprId,
    arena: CausalExprArena,
) -> IdentificationResult {
    let n_z = z_vars.len();
    let label = if n_z == 0 { "generalized.adjustment.empty" } else { "generalized.adjustment" };
    let estimand = IdentifiedEstimand::backdoor(label, z_vars, functional);
    let mut assumptions = AssumptionSet::default();
    assumptions.push(crate::assumptions::causal_markov("generalized.adjustment.mag"));
    let mut result = IdentificationResult::identified(
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
    );
    if budget_exhausted {
        result.diagnostics.push(crate::result::search_bounded_diagnostic(
            MAG_SEARCH_BOUNDED_DIAGNOSTIC_CODE,
            format!(
                "generalized adjustment search stopped after {examined} separation tests (its \
                 examination budget, declared candidate ceiling or leasable storage) before it \
                 found a smaller set; the returned set m-separates treatment from outcome but \
                 is not minimal"
            ),
        ));
    }
    result
}

pub(crate) fn identify_on_mag_completion(
    mag: &Pag,
    target: MagTarget,
    active: Value,
    control: Value,
    budget: AdjustmentBudget<'_>,
) -> Result<IdentificationResult, IdentificationError> {
    let MagTarget { t, y, .. } = target;
    let query = CausalQuery::AverageEffect(AverageEffectQuery::new(
        t,
        y,
        Arc::from([]),
        antecedent_core::Intervention::set(t, control.clone()),
        antecedent_core::Intervention::set(t, active.clone()),
        antecedent_core::TargetPopulation::AllObserved,
    ));
    match mag_adjustment_search(mag, target, query.clone(), budget)? {
        MagAdjustment::Failed(result) => Ok(result),
        MagAdjustment::Found { z_vars, examined, budget_exhausted } => {
            let mut arena = CausalExprArena::new();
            let functional = arena.backdoor_ate(t, y, &z_vars, active, control);
            Ok(mag_adjustment_identified(
                query,
                z_vars,
                examined,
                budget_exhausted,
                functional,
                arena,
            ))
        }
    }
}

/// MAG adjustment for a single-treatment response: the functional the found adjustment set
/// supports is built by `functional` from the set (the mean at a hard level, or the surface a
/// declared policy is a functional of), never a stand-in contrast.
pub(crate) fn identify_on_mag_completion_response(
    mag: &Pag,
    target: MagTarget,
    budget: AdjustmentBudget<'_>,
    response: &ResponseQuery,
    functional: impl FnOnce(&mut CausalExprArena, &[VariableId]) -> antecedent_expr::ExprId,
) -> Result<IdentificationResult, IdentificationError> {
    let query = CausalQuery::Response(response.clone());
    match mag_adjustment_search(mag, target, query.clone(), budget)? {
        MagAdjustment::Failed(result) => Ok(result),
        MagAdjustment::Found { z_vars, examined, budget_exhausted } => {
            let mut arena = CausalExprArena::new();
            let functional = functional(&mut arena, &z_vars[..]);
            Ok(mag_adjustment_identified(
                query,
                z_vars,
                examined,
                budget_exhausted,
                functional,
                arena,
            ))
        }
    }
}

pub(crate) fn mag_dense_to_var(
    mag: &Pag,
    id: DenseNodeId,
) -> Result<VariableId, IdentificationError> {
    crate::prepared::node_variable_id(mag.nodes(), id.as_usize())
        .ok_or(IdentificationError::UnknownVariable { id: VariableId::from_raw(id.raw()) })
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
        let id = DenseNodeId::from_raw(dense_u32(i));
        if id != t && de_t.contains(id) && an_y.contains(id) {
            cn.push(id);
        }
    }
    let forb = directed_closure(admg, &cn, false);
    let mut out = Vec::new();
    for i in 0..admg.node_count() {
        let id = DenseNodeId::from_raw(dense_u32(i));
        if id == t || id == y || forb.contains(id) {
            continue;
        }
        if an.contains(id) || de_t.contains(id) {
            out.push(id);
        }
    }
    out
}

/// Directed ancestors (or descendants) of `seeds`, seeds included.
pub(crate) fn directed_closure(admg: &Admg, seeds: &[DenseNodeId], ancestors: bool) -> BitSet {
    let mut ws = antecedent_graph::GraphWorkspace::default();
    let mut out = BitSet::with_len(admg.node_count());
    if ancestors {
        admg.ancestors_of(seeds, &mut out, &mut ws);
    } else {
        admg.descendants_of(seeds, &mut out, &mut ws);
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

pub(crate) fn mag_to_admg(mag: &Pag) -> Option<Admg> {
    let n = dense_u32(mag.node_count());
    let mut admg = Admg::with_variables(n);
    for i in 0..mag.node_count() {
        let a = DenseNodeId::from_raw(dense_u32(i));
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

/// Directed edges of a MAG that are not visible (Zhang 2008, Definition 8),
/// as `(tail, head)` pairs in dense order.
pub fn invisible_directed_edges(mag: &Pag) -> Vec<(DenseNodeId, DenseNodeId)> {
    let mut out = Vec::new();
    for i in 0..mag.node_count() {
        let a = DenseNodeId::from_raw(dense_u32(i));
        for (b, at_a, at_b) in mag.neighbors(a) {
            if at_a == Endpoint::Tail
                && at_b == Endpoint::Arrow
                && !crate::joint_response::visible(mag, a, b)
            {
                out.push((a, b));
            }
        }
    }
    out
}

/// The most-confounded ADMG compatible with a directed/bidirected MAG.
///
/// A MAG edge `A -> B` excludes a latent common cause of `A` and `B` only when
/// it is visible (Zhang 2008, Lemma 9); an invisible edge is compatible with
/// `A <- L -> B`. This graph keeps every MAG edge and adds `A <-> B` beside
/// each invisible `A -> B`. The latent projection of every DAG the MAG
/// represents is an edge-subgraph of it: a projected `A -> B` makes `A` an
/// ancestor of `B`, so the MAG edge is `A -> B`; a projected `A <-> B` is an
/// inducing path into both ends, so the MAG edge is `A <-> B` or an invisible
/// directed edge. A causal model compatible with a graph is compatible with
/// every acyclic supergraph, so a functional that Shpitser–Pearl ID derives
/// here holds in every represented DAG. The converse fails: the reduction is
/// sound, not complete, for MAG identification.
pub fn mag_to_confounded_admg(mag: &Pag) -> Option<Admg> {
    let mut admg = mag_to_admg(mag)?;
    for (a, b) in invisible_directed_edges(mag) {
        admg.insert_bidirected(a, b).ok()?;
    }
    Some(admg)
}

fn mutilate_outgoing(admg: &Admg, t: DenseNodeId) -> Admg {
    admg.copy_filtered(
        antecedent_graph::CopyLabels::StaticIndex { extra: 0 },
        |from, _to| from != t,
        // Outgoing surgery leaves latent arrows in place.
        |_a, _b| true,
    )
    .expect("outgoing-mutilated ADMG dropped an edge")
}

#[cfg(test)]
mod conditional_search_tests;
#[cfg(test)]
mod enumeration_admission_tests;
#[cfg(test)]
mod tests;
