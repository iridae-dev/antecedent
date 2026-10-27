//! Backdoor adjustment identification for DAGs.
//!
//! Existence of an admissible adjustment set is decided by one d-separation test on the
//! ancestral set `An({T,Y})` minus the forbidden variables (van der Zander, Liskiewicz &
//! Textor 2014); the size-ordered subset search then only chooses which sets to return, and
//! its `max_examinations` budget never turns a decided query into an undecided one.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![cfg_attr(
    test,
    allow(
        clippy::cast_possible_truncation,
        reason = "test fixtures compare exact constants and index with small literals"
    )
)]

use std::sync::Arc;

use antecedent_core::{
    AssumptionSet, AverageEffectQuery, CausalQuery, ExecutionContext, VariableId,
};
use antecedent_expr::CausalExprArena;
use antecedent_graph::{BitSet, DSeparationWorkspace, Dag, DenseNodeId, GraphWorkspace};

use crate::error::IdentificationError;
use crate::identification_run::IdentificationRun;
use crate::identifier::IdentificationWorkspace;
use crate::node_map::NodeMap;
use crate::result::{
    DerivationTrace, IdentificationPerformanceRecord, IdentificationResult, IdentifiedEstimand,
};

/// Diagnostic code of a back-door search that spent `max_examinations` before it finished
/// the subset enumeration. A valid set may still be returned; only the choice among sets is
/// unfinished.
pub(crate) const BACKDOOR_SEARCH_BOUNDED_DIAGNOSTIC_CODE: &str = "identify.backdoor.search_bounded";

/// Configuration for adjustment-set enumeration.
#[derive(Clone, Debug)]
pub struct AdjustmentSearchConfig {
    /// Maximum number of adjustment sets to return.
    pub max_results: usize,
    /// Maximum candidate sets to d-separate (work budget, not an output cap).
    ///
    /// Exhaustion ends the search for smaller sets, not the decision: existence is settled by
    /// one separation test on the ancestral set, which is returned (valid, not minimal) with
    /// an execution diagnostic when nothing smaller was found first.
    pub max_examinations: u64,
    /// Caller-declared ceiling on the candidate variables of an exact subset enumeration.
    ///
    /// The default (`usize::MAX`) declares none: the enumeration is funded by the context's
    /// admission pool (its cursor and retained sets are leased before they are allocated) and
    /// its work is charged to [`Self::max_examinations`]. Above a declared ceiling, or when the
    /// pool cannot fund the enumeration, search restricts to ancestors of `{T,Y}` first; if
    /// still unfunded, tries the Henckel O-set before the certified ancestral set.
    pub max_candidates: usize,
    /// The caller's execution context: its admission pool funds the search's storage and its
    /// cancellation token is observed at every test. `None` runs under a serial production
    /// context with an unlimited pool.
    pub context: Option<ExecutionContext>,
    /// Variables forbidden from appearing in Z.
    pub forbidden: Arc<[VariableId]>,
    /// If true, only return inclusion-minimal sets.
    pub minimal_only: bool,
    /// If true, only return inclusion-maximal valid sets (supersets preferred).
    ///
    /// Incompatible with [`Self::minimal_only`]; when both are set, minimal wins.
    pub maximal_only: bool,
    /// Optional per-variable measurement costs (lower is better). Missing → cost 1.0.
    pub measurement_costs: Arc<[(VariableId, f64)]>,
    /// Optional max history lag (steps before the query reference time).
    /// When set, candidates listed in [`Self::history_lags`] with lag `>` this
    /// value are excluded from adjustment-set enumeration. Variables absent from
    /// `history_lags` are unrestricted (static DAGs / unknown lag).
    pub max_history_lag: Option<u32>,
    /// Per-variable history lag in steps (0 = contemporaneous or future relative
    /// to the query). Used with [`Self::max_history_lag`]. Unfolded temporal
    /// graphs populate this from [`antecedent_core::TemporalIndexer`].
    pub history_lags: Arc<[(VariableId, u32)]>,
}

impl Default for AdjustmentSearchConfig {
    fn default() -> Self {
        Self {
            max_results: 64,
            max_examinations: 1_000_000,
            max_candidates: usize::MAX,
            context: None,
            forbidden: Arc::from([]),
            minimal_only: true,
            maximal_only: false,
            measurement_costs: Arc::from([]),
            max_history_lag: None,
            history_lags: Arc::from([]),
        }
    }
}

impl AdjustmentSearchConfig {
    /// Validate the invariants the search relies on.
    ///
    /// Measurement costs must be finite and non-negative: they are summed and ordered, and a
    /// NaN or infinite cost would make the cost ordering inconsistent.
    ///
    /// # Errors
    ///
    /// A cost that is not finite and non-negative.
    pub fn validate(&self) -> Result<(), IdentificationError> {
        self.validate_costs()
    }

    fn validate_costs(&self) -> Result<(), IdentificationError> {
        if self.measurement_costs.iter().all(|&(_, cost)| cost.is_finite() && cost >= 0.0) {
            Ok(())
        } else {
            Err(IdentificationError::invalid_input(
                "adjustment.measurement_cost_not_finite_nonnegative",
            ))
        }
    }

    /// Set the per-variable measurement costs after validating them.
    ///
    /// # Errors
    ///
    /// A cost that is not finite and non-negative.
    pub fn with_measurement_costs(
        mut self,
        costs: impl Into<Arc<[(VariableId, f64)]>>,
    ) -> Result<Self, IdentificationError> {
        self.measurement_costs = costs.into();
        self.validate_costs()?;
        Ok(self)
    }

    /// Set the per-variable history lags and the largest admitted lag.
    #[must_use]
    pub fn with_history_lags(
        mut self,
        max_history_lag: Option<u32>,
        lags: impl Into<Arc<[(VariableId, u32)]>>,
    ) -> Self {
        self.max_history_lag = max_history_lag;
        self.history_lags = lags.into();
        self
    }

    /// Whether `id` exceeds [`Self::max_history_lag`] given [`Self::history_lags`].
    #[must_use]
    pub fn exceeds_history_lag(&self, id: VariableId) -> bool {
        let Some(max_lag) = self.max_history_lag else {
            return false;
        };
        self.history_lags.iter().find(|(v, _)| *v == id).is_some_and(|(_, lag)| *lag > max_lag)
    }
}

/// The adjustment instance both back-door searches start from.
///
/// History lag and measurement cost are dense in the DAG's node order, so the
/// subset search does not scan the configuration lists per candidate or per
/// comparison. The first listed lag or cost for a variable wins, matching
/// [`AdjustmentSearchConfig::exceeds_history_lag`].
pub(crate) struct AdjustmentProblem {
    /// Treatment node.
    pub(crate) t: DenseNodeId,
    /// Outcome node.
    pub(crate) y: DenseNodeId,
    /// Configured exclusions, `T`, `Y`, and descendants of `T`.
    pub(crate) forbidden: BitSet,
    /// Nodes that are neither forbidden nor past the history-lag limit.
    pub(crate) candidates: Vec<DenseNodeId>,
    /// `G` with edges out of `T` removed.
    pub(crate) mutilated: Dag,
    /// Per-node measurement cost. A variable with no listed cost is `1.0`.
    costs: Vec<f64>,
    /// First listed history lag of each node. `None` means the variable is unrestricted.
    lags: Vec<Option<u32>>,
    /// Copied from the search config so lag checks do not re-read the arc.
    max_history_lag: Option<u32>,
}

impl AdjustmentProblem {
    /// Build the shared preamble.
    ///
    /// # Errors
    ///
    /// Unknown treatment, outcome, or forbidden variable.
    pub(crate) fn new(
        dag: &Dag,
        map: &NodeMap,
        treatment: VariableId,
        outcome: VariableId,
        config: &AdjustmentSearchConfig,
        gws: &mut antecedent_graph::GraphWorkspace,
    ) -> Result<Self, IdentificationError> {
        let t = map.dense(treatment)?;
        let y = map.dense(outcome)?;
        let n = dag.node_count();
        let mut forbidden = BitSet::with_len(n);
        for &v in config.forbidden.iter() {
            forbidden.insert(map.dense(v)?);
        }
        forbidden.insert(t);
        forbidden.insert(y);
        let mut desc = BitSet::with_len(n);
        dag.descendants_of(&[t], &mut desc, gws);
        for id in DenseNodeId::all(n) {
            if desc.contains(id) {
                forbidden.insert(id);
            }
        }
        let lags = dense_first_wins(
            map,
            config.history_lags.iter().map(|&(variable, lag)| (variable, Some(lag))),
            None,
            n,
        )?;
        let max_history_lag = config.max_history_lag;
        // History lag is not applied. Efficient backdoor never consulted it;
        // the backdoor search filters afterwards.
        let candidates = DenseNodeId::all(n).filter(|id| !forbidden.contains(*id)).collect();
        let mutilated = remove_outgoing(dag, t)?;
        Ok(Self {
            t,
            y,
            forbidden,
            candidates,
            mutilated,
            costs: dense_costs(map, config, n)?,
            lags,
            max_history_lag,
        })
    }

    /// Whether `id`'s first listed lag is past the configured maximum.
    pub(crate) fn exceeds_lag(&self, id: DenseNodeId) -> bool {
        lag_exceeds(&self.lags, self.max_history_lag, id)
    }

    /// Sum of per-node measurement costs. An unlisted node costs `1.0`.
    ///
    /// Every listed cost is finite and non-negative and their sum over all nodes is finite
    /// ([`dense_costs`]), so a subset's total is finite and totally ordered.
    pub(crate) fn total_cost(&self, z: &[DenseNodeId]) -> f64 {
        z.iter().map(|d| self.costs[d.as_usize()]).sum()
    }

    /// Stable-sort `sets` by total measurement cost, lowest first.
    ///
    /// Totals are computed once per set, not per comparison.
    pub(crate) fn order_by_cost(&self, sets: &mut Vec<Vec<DenseNodeId>>) {
        let totals: Vec<f64> = sets.iter().map(|z| self.total_cost(z)).collect();
        let mut order: Vec<usize> = (0..sets.len()).collect();
        order.sort_by(|&a, &b| totals[a].total_cmp(&totals[b]));
        let mut taken: Vec<Option<Vec<DenseNodeId>>> =
            std::mem::take(sets).into_iter().map(Some).collect();
        *sets = order.into_iter().filter_map(|i| taken[i].take()).collect();
    }
}

/// A per-node table from a configuration list in which the first listed entry for a variable
/// wins and unlisted nodes take `default`.
///
/// # Errors
///
/// An entry names a variable the graph does not contain: a mistyped id would otherwise change
/// the search without any diagnostic.
pub(crate) fn dense_first_wins<T: Copy>(
    map: &NodeMap,
    entries: impl IntoIterator<Item = (VariableId, T)>,
    default: T,
    len: usize,
) -> Result<Vec<T>, IdentificationError> {
    let mut table = vec![default; len];
    let mut seen = vec![false; len];
    for (variable, value) in entries {
        let index = map.dense(variable)?.as_usize();
        if !seen[index] {
            seen[index] = true;
            table[index] = value;
        }
    }
    Ok(table)
}

/// Per-node measurement costs, `1.0` for unlisted nodes.
///
/// # Errors
///
/// A cost that is not finite and non-negative, a total over all nodes that overflows, or a
/// cost for a variable the graph does not contain.
fn dense_costs(
    map: &NodeMap,
    config: &AdjustmentSearchConfig,
    len: usize,
) -> Result<Vec<f64>, IdentificationError> {
    config.validate_costs()?;
    // `+ 0.0` turns a negative zero into zero so `total_cmp` orders equal costs as equal.
    let costs = dense_first_wins(
        map,
        config.measurement_costs.iter().map(|&(variable, cost)| (variable, cost + 0.0)),
        1.0,
        len,
    )?;
    if !costs.iter().sum::<f64>().is_finite() {
        return Err(IdentificationError::invalid_input(
            "adjustment.measurement_cost_total_not_finite",
        ));
    }
    Ok(costs)
}

fn lag_exceeds(lags: &[Option<u32>], max_history_lag: Option<u32>, id: DenseNodeId) -> bool {
    let Some(max_lag) = max_history_lag else {
        return false;
    };
    lags.get(id.as_usize()).and_then(|lag| *lag).is_some_and(|lag| lag > max_lag)
}

/// Prepared DAG for identification (cached ancestry helpers via workspaces).
#[derive(Clone, Debug)]
pub struct PreparedIdentificationGraph {
    dag: Dag,
    /// Variable <-> node index of `dag`, built once at preparation.
    nodes: Arc<NodeMap>,
    /// Caller-declared assumptions captured at [`BackdoorIdentifier::prepare_with_assumptions`].
    declared_assumptions: AssumptionSet,
}

impl PreparedIdentificationGraph {
    /// Wrap a DAG with no extra declared assumptions.
    #[must_use]
    pub fn new(dag: Dag) -> Self {
        Self::with_assumptions(dag, AssumptionSet::new())
    }

    /// Wrap a DAG together with caller-declared assumptions.
    #[must_use]
    pub fn with_assumptions(dag: Dag, declared_assumptions: AssumptionSet) -> Self {
        let nodes = Arc::new(NodeMap::from_dag(&dag));
        Self { dag, nodes, declared_assumptions }
    }

    /// The variable <-> node index of the prepared DAG.
    pub(crate) fn node_map(&self) -> &NodeMap {
        &self.nodes
    }

    /// The prepared DAG the back-door, front-door, IV, and efficient searches share.
    pub(crate) fn wrap(dag: &Dag, assumptions: AssumptionSet) -> Self {
        Self::with_assumptions(dag.clone(), assumptions)
    }

    /// Borrow the DAG.
    #[must_use]
    pub fn dag(&self) -> &Dag {
        &self.dag
    }

    /// Assumptions declared at prepare time (merged into identification results).
    #[must_use]
    pub fn declared_assumptions(&self) -> &AssumptionSet {
        &self.declared_assumptions
    }
}

/// Identifier for static DAGs .
#[derive(Clone, Debug, Default)]
pub struct BackdoorIdentifier {
    /// Search configuration.
    pub config: AdjustmentSearchConfig,
}

impl BackdoorIdentifier {
    /// Create with default config.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Declare a ceiling on exact candidate-subset enumeration (default: none, the context's
    /// pool funds it).
    #[must_use]
    pub fn with_max_candidates(mut self, max_candidates: usize) -> Self {
        self.config.max_candidates = max_candidates;
        self
    }

    /// Run the search under the caller's context: its pool funds the search's storage and its
    /// cancellation token stops it.
    #[must_use]
    pub fn with_context(mut self, context: ExecutionContext) -> Self {
        self.config.context = Some(context);
        self
    }

    /// Prepare a graph with no extra declared assumptions.
    ///
    /// # Errors
    ///
    /// Currently infallible; reserved for validation.
    pub fn prepare(&self, graph: &Dag) -> Result<PreparedIdentificationGraph, IdentificationError> {
        self.prepare_with_assumptions(graph, AssumptionSet::new())
    }

    /// Prepare a graph, retaining caller-declared assumptions for the result.
    ///
    /// # Errors
    ///
    /// Currently infallible; reserved for validation.
    pub fn prepare_with_assumptions(
        &self,
        graph: &Dag,
        assumptions: AssumptionSet,
    ) -> Result<PreparedIdentificationGraph, IdentificationError> {
        Ok(PreparedIdentificationGraph::wrap(graph, assumptions))
    }

    /// Identify an average-effect query via backdoor adjustment.
    ///
    /// At most `config.max_results` adjustment sets are returned; when more
    /// qualifying sets exist the result is truncated (noted in the derivation
    /// trace) rather than treated as an error.
    ///
    /// # Errors
    ///
    /// Unsupported query, unknown variables, or a candidate pool too large
    /// for exact enumeration.
    pub fn identify(
        &self,
        prepared: &PreparedIdentificationGraph,
        query: &CausalQuery,
        workspace: &mut IdentificationWorkspace,
    ) -> Result<IdentificationResult, IdentificationError> {
        antecedent_core::execution_attempt::run_operation(
            antecedent_core::execution_attempt::Operation::Identification,
            || {
                crate::execution_counts::note_check();
                let CausalQuery::AverageEffect(ate) = query else {
                    return Err(IdentificationError::UnsupportedQuery {
                        message: match query {
                            CausalQuery::Distribution(_) => {
                                "backdoor does not identify Distribution queries; use IdIdentifier / IdcIdentifier / AutoIdentifier"
                            }
                            CausalQuery::PathSpecific(_) => {
                                "backdoor does not identify PathSpecific queries; use PathSpecificIdentifier / AutoIdentifier"
                            }
                            _ => "backdoor only supports AverageEffect",
                        },
                    });
                };
                ate.validate().map_err(|_| IdentificationError::UnsupportedQuery {
                    message: "invalid average-effect query",
                })?;
                let mut result = match self.identify_ate(prepared, ate, query.clone(), workspace) {
                    Ok(result) => result,
                    Err(IdentificationError::NotCertified { .. })
                        if !ate.effect_modifiers.is_empty() =>
                    {
                        crate::generalized::not_identified(
                            query.clone(),
                            "marginal search capped; attempting modifier-constrained search",
                        )
                    }
                    Err(error) => return Err(error),
                };
                crate::generalized::validate_dag_conditional_adjustment(
                    prepared.dag(),
                    ate,
                    &mut result,
                    &self.config,
                )?;
                Ok(result)
            },
        )
    }

    fn identify_ate(
        &self,
        prepared: &PreparedIdentificationGraph,
        ate: &AverageEffectQuery,
        query: CausalQuery,
        workspace: &mut IdentificationWorkspace,
    ) -> Result<IdentificationResult, IdentificationError> {
        let run = IdentificationRun::open(prepared, ate, "backdoor")?;
        let dag = run.dag();
        let context = crate::exact_search::context_or_default(self.config.context.as_ref());
        let mut search = crate::exact_search::ExactSearch::open(&context)?;
        // The mutilated graph and its search scratch are leased before they are built; a pool
        // that cannot hold them is a resource refusal, never a scientific negative.
        if !search.hold(crate::exact_search::graph_bytes(dag.node_count()))? {
            return Err(IdentificationError::budget(crate::IdentificationBudget::Memory));
        }
        let problem = AdjustmentProblem::new(
            dag,
            prepared.node_map(),
            ate.treatment,
            ate.outcome,
            &self.config,
            &mut workspace.graph,
        )?;

        let mut screening = self.screen(&mut search, &problem, dag, workspace)?;
        let mut outcome = SearchOutcome::default();
        if screening.exists && screening.valid.is_empty() && !screening.ancestral_fallback {
            outcome = self.enumerate(&mut search, &problem, &mut screening, workspace)?;
        }
        let Screening { constructive, mut valid, ancestral_fallback, notes, examined, .. } =
            screening;
        if ancestral_fallback {
            valid.push(constructive);
        }
        if !self.config.measurement_costs.is_empty() {
            problem.order_by_cost(&mut valid);
        }

        let assumptions = run.base_assumptions();
        let mut derivation = DerivationTrace::default();
        derivation.push(
            "backdoor.criterion",
            "Z blocks all backdoor paths and contains no descendants of T",
        );
        for note in &notes {
            derivation.push("backdoor.search", note.clone());
        }
        if outcome.truncated {
            derivation.push(
                "backdoor.enumeration",
                format!(
                    "result limit reached; returning first {} adjustment sets (more exist)",
                    valid.len()
                ),
            );
        }
        if outcome.budget_exhausted {
            derivation.push(
                "backdoor.enumeration",
                format!(
                    "examination budget exhausted after {examined} d-separations \
                     (max_examinations={}); search is incomplete",
                    self.config.max_examinations
                ),
            );
        }

        if valid.is_empty() {
            // Reached only when the ancestral set failed its separation test, which proves no
            // admissible set exists: a scientific negative, never a search bound.
            return Ok(IdentificationResult::not_identified(
                query,
                derivation,
                assumptions,
                IdentificationPerformanceRecord { candidates_examined: examined, sets_returned: 0 },
            ));
        }

        let active = crate::intervention_support::require_set_value(&ate.active, "backdoor")?;
        let control = crate::intervention_support::require_set_value(&ate.control, "backdoor")?;
        let mut arena = CausalExprArena::new();
        let mut estimands = Vec::with_capacity(valid.len());
        for z in &valid {
            let vars: Vec<VariableId> = run.variables(z)?;
            let functional = arena.backdoor_ate(
                ate.treatment,
                ate.outcome,
                &vars,
                active.clone(),
                control.clone(),
            );
            estimands.push(IdentifiedEstimand::backdoor(
                "backdoor.adjustment",
                Arc::from(vars),
                functional,
            ));
            derivation.push("backdoor.adjustment_set", format!("|Z|={}", z.len()));
        }

        let mut result = IdentificationResult::identified(
            query,
            estimands,
            arena,
            derivation,
            assumptions,
            IdentificationPerformanceRecord {
                candidates_examined: examined,
                sets_returned: u64::try_from(valid.len()).unwrap_or(u64::MAX),
            },
        );
        if outcome.budget_exhausted {
            result.diagnostics.push(crate::result::search_bounded_diagnostic(
                BACKDOOR_SEARCH_BOUNDED_DIAGNOSTIC_CODE,
                "backdoor search exhausted max_examinations; returned sets are valid but \
                 the enumeration is incomplete",
            ));
        }
        Ok(result)
    }

    /// Decide existence and, where the exact enumeration is not funded, settle on a certified set.
    ///
    /// A valid set exists iff the ancestral one does: `An({T,Y}) \ forbidden` separates T from Y
    /// in `G underbar T` whenever any admissible set does (van der Zander, Liskiewicz & Textor
    /// 2014), so one separation test decides existence and the subset search only chooses which
    /// valid sets to return.
    fn screen(
        &self,
        search: &mut crate::exact_search::ExactSearch<'_>,
        problem: &AdjustmentProblem,
        dag: &Dag,
        workspace: &mut IdentificationWorkspace,
    ) -> Result<Screening, IdentificationError> {
        let (t, y) = (problem.t, problem.y);
        let mut candidates: Vec<DenseNodeId> =
            problem.candidates.iter().copied().filter(|id| !problem.exceeds_lag(*id)).collect();
        let constructive = screen_ancestor_candidates(
            dag,
            t,
            y,
            &problem.forbidden,
            problem,
            true,
            &mut workspace.graph,
        );
        let mut examined = 1u64;
        let exists =
            is_backdoor_adjustment(&problem.mutilated, t, y, &constructive, &mut workspace.dsep)?;

        let cap = self.config.max_candidates;
        let mut valid: Vec<Vec<DenseNodeId>> = Vec::new();
        let mut notes: Vec<String> = Vec::new();
        let mut ancestral_fallback = false;
        if !exists {
            notes.push(format!(
                "backdoor.existence: the ancestral set An({{T,Y}}) minus forbidden \
                 ({} variables) does not separate T from Y, so no admissible set exists",
                constructive.len()
            ));
        } else if self.config.minimal_only
            && is_mandatory_fork_set(dag, problem, &candidates, &constructive)
        {
            // Each fork T <- Z -> Y has exactly one internal vertex. Every
            // adjustment set must contain Z, regardless of the other variables
            // it conditions on. If these mandatory vertices already separate
            // T and Y, they are the unique inclusion-minimal set. This also
            // covers the empty set without enumerating its supersets.
            notes.push(
                "backdoor.mandatory_forks: the ancestral adjustment set An({T,Y}) minus forbidden consists entirely \
                 of mandatory common parents of T and Y; it is the unique minimal set"
                    .to_owned(),
            );
            ancestral_fallback = true;
        } else if !search.fund_enumeration(candidates.len(), cap, self.config.max_results)? {
            // Screen to An({T,Y}) \ forbidden when the enumeration is over the declared
            // ceiling or the pool cannot fund it.
            notes.push(format!(
                "backdoor.screen: restricted to An({{T,Y}}); {} → {} candidates",
                candidates.len(),
                constructive.len()
            ));
            candidates.clone_from(&constructive);
            if !search.fund_enumeration(candidates.len(), cap, self.config.max_results)? {
                // Still unfunded: the Henckel O-set is the preferred valid set; the ancestral
                // set already tested valid is the certified fallback.
                let o_set =
                    crate::efficient::optimal_adjustment_set(dag, t, y, &mut workspace.graph);
                examined += 1;
                if o_set.iter().all(|v| !problem.forbidden.contains(*v) && !problem.exceeds_lag(*v))
                    && is_backdoor_adjustment(
                        &problem.mutilated,
                        t,
                        y,
                        &o_set,
                        &mut workspace.dsep,
                    )?
                {
                    notes.push(format!(
                        "exact enumeration skipped ({} candidates over the declared ceiling {cap} or the \
                         storage the context can lease); returned O-set",
                        candidates.len()
                    ));
                    valid.push(o_set);
                } else {
                    notes.push(format!(
                        "exact enumeration skipped ({} candidates over the declared ceiling {cap} or the \
                         storage the context can lease); returned the ancestral set An({{T,Y}}) minus forbidden, valid but not minimal",
                        candidates.len()
                    ));
                    ancestral_fallback = true;
                }
            }
        }
        Ok(Screening {
            constructive,
            candidates,
            valid,
            ancestral_fallback,
            exists,
            notes,
            examined,
        })
    }

    /// Enumerate subsets by increasing size for minimal-first (or all / maximal).
    ///
    /// When more than `max_results` qualifying sets exist, the first `max_results` (in
    /// size-then-mask order, optionally cost-sorted afterward) are returned. Found sets are
    /// kept as bit masks so a superset/subset test against each is word-wise.
    fn enumerate(
        &self,
        search: &mut crate::exact_search::ExactSearch<'_>,
        problem: &AdjustmentProblem,
        screening: &mut Screening,
        workspace: &mut IdentificationWorkspace,
    ) -> Result<SearchOutcome, IdentificationError> {
        let (t, y) = (problem.t, problem.y);
        let n = problem.mutilated.node_count();
        let candidates = screening.candidates.as_slice();
        let constructive = screening.constructive.as_slice();
        let examined = &mut screening.examined;
        let valid = &mut screening.valid;
        let mut masks: Vec<BitSet> = Vec::new();
        let mut scratch = BitSet::with_len(n);
        let mut outcome = SearchOutcome::default();
        let m = candidates.len();
        let descending = self.config.maximal_only && !self.config.minimal_only;
        'sizes: for step in 0..=m {
            let size = if descending { m - step } else { step };
            let mut early_stop = false;
            let mut enum_err: Option<IdentificationError> = None;
            crate::enum_masks::for_each_mask_of_size(candidates, size, |z| {
                if enum_err.is_some() {
                    return true;
                }
                if let Err(e) = search.observe() {
                    enum_err = Some(e);
                    return true;
                }
                // A superset of a set already in hand is not d-separated again, but it still
                // spends the examination budget. Skipping it for free lets one minimal set
                // walk an exponential family (up to the candidate power set) that the cap
                // never sees.
                let skipped = !masks.is_empty() && {
                    scratch.clear();
                    for &d in z {
                        scratch.insert(d);
                    }
                    (self.config.minimal_only
                        && masks.iter().any(|prev| prev.is_subset_of(&scratch)))
                        || (descending && masks.iter().any(|prev| scratch.is_subset_of(prev)))
                };
                if skipped {
                    return !charge_examination(
                        &mut *examined,
                        self.config.max_examinations,
                        &mut outcome.budget_exhausted,
                        &mut early_stop,
                    );
                }
                // The ancestral set was tested above; do not spend a second test on it.
                if z != constructive {
                    if !charge_examination(
                        &mut *examined,
                        self.config.max_examinations,
                        &mut outcome.budget_exhausted,
                        &mut early_stop,
                    ) {
                        return true;
                    }
                    match is_backdoor_adjustment(&problem.mutilated, t, y, z, &mut workspace.dsep) {
                        Ok(false) => return false,
                        Err(e) => {
                            enum_err = Some(e);
                            return true;
                        }
                        Ok(true) => {}
                    }
                }
                valid.push(z.to_vec());
                let mut mask = BitSet::with_len(n);
                for &d in z {
                    mask.insert(d);
                }
                masks.push(mask);
                if self.config.minimal_only && z.is_empty() {
                    early_stop = true;
                    return true;
                }
                if valid.len() >= self.config.max_results {
                    outcome.truncated = true;
                    early_stop = true;
                    return true;
                }
                false
            });
            if let Some(e) = enum_err {
                return Err(e);
            }
            if early_stop {
                break 'sizes;
            }
        }
        if valid.is_empty() && outcome.budget_exhausted {
            // The budget stopped the search for *minimal* sets, but a valid set is already
            // in hand: identifiability is decided, only the choice of set is unfinished.
            screening.notes.push(
                "backdoor.enumeration: examination budget exhausted before a smaller set \
                 was found; returned the ancestral set An({T,Y}) minus forbidden, valid but \
                 not minimal"
                    .to_owned(),
            );
            screening.ancestral_fallback = true;
        }
        Ok(outcome)
    }
}

/// What the screening phase settled before any subset enumeration.
struct Screening {
    /// `An({T,Y}) \ forbidden`, minus variables past the history-lag limit: the certified set.
    constructive: Vec<DenseNodeId>,
    /// Candidates the enumeration draws subsets from.
    candidates: Vec<DenseNodeId>,
    /// Sets already chosen (the O-set fallback, then everything enumeration finds).
    valid: Vec<Vec<DenseNodeId>>,
    /// The answer is `constructive` itself.
    ancestral_fallback: bool,
    /// Whether any admissible set exists.
    exists: bool,
    /// Derivation notes in the order they were reached.
    notes: Vec<String>,
    /// Separation tests run so far.
    examined: u64,
}

/// How an enumeration ended.
#[derive(Default)]
struct SearchOutcome {
    /// `max_results` was reached.
    truncated: bool,
    /// `max_examinations` was spent.
    budget_exhausted: bool,
}

/// Whether every member of the ancestral set is a candidate that is a common parent of T and Y.
fn is_mandatory_fork_set(
    dag: &Dag,
    problem: &AdjustmentProblem,
    candidates: &[DenseNodeId],
    constructive: &[DenseNodeId],
) -> bool {
    let n = dag.node_count();
    let mask = |nodes: &[DenseNodeId]| {
        let mut set = BitSet::with_len(n);
        for &node in nodes {
            set.insert(node);
        }
        set
    };
    let candidate_mask = mask(candidates);
    let parents_of_t = mask(dag.parents(problem.t));
    let parents_of_y = mask(dag.parents(problem.y));
    constructive.iter().all(|&z| {
        candidate_mask.contains(z) && parents_of_t.contains(z) && parents_of_y.contains(z)
    })
}

/// Charge one candidate against `max_examinations`.
///
/// Returns false, and marks the search bounded, when the cap is already spent. The candidate
/// that hits the cap is refused rather than tested, so a search that lands exactly on the cap
/// has examined every candidate it accepted.
fn charge_examination(
    examined: &mut u64,
    max_examinations: u64,
    budget_exhausted: &mut bool,
    early_stop: &mut bool,
) -> bool {
    if *examined >= max_examinations {
        *budget_exhausted = true;
        *early_stop = true;
        return false;
    }
    *examined += 1;
    true
}

/// `An({T,Y}) \ forbidden`, minus variables past the history-lag limit.
///
/// A sound screen for the subset search, and the constructive adjustment set: when any
/// admissible set exists this one separates T from Y in `G underbar T`.
pub(crate) fn screen_ancestor_candidates(
    dag: &Dag,
    t: DenseNodeId,
    y: DenseNodeId,
    forbidden: &BitSet,
    problem: &AdjustmentProblem,
    respect_history_lag: bool,
    gws: &mut antecedent_graph::GraphWorkspace,
) -> Vec<DenseNodeId> {
    let mut an = BitSet::with_len(dag.node_count());
    dag.ancestors_of(&[t, y], &mut an, gws);
    DenseNodeId::all(dag.node_count())
        .filter(|id| {
            an.contains(*id)
                && !forbidden.contains(*id)
                && !(respect_history_lag && problem.exceeds_lag(*id))
        })
        .collect()
}

pub(crate) fn is_backdoor_adjustment(
    mutilated: &Dag,
    t: DenseNodeId,
    y: DenseNodeId,
    z: &[DenseNodeId],
    ws: &mut DSeparationWorkspace,
) -> Result<bool, IdentificationError> {
    mutilated.is_d_separated(t, y, z, ws).map_err(IdentificationError::from)
}

pub(crate) fn remove_outgoing(dag: &Dag, t: DenseNodeId) -> Result<Dag, IdentificationError> {
    remove_outgoing_set(dag, &[t])
}

/// `G` with outgoing edges from every node in `nodes` removed (node ids preserved).
pub(crate) fn remove_outgoing_set(
    dag: &Dag,
    nodes: &[DenseNodeId],
) -> Result<Dag, IdentificationError> {
    dag.copy_filtered(|from, _to| !nodes.contains(&from)).map_err(IdentificationError::from)
}

/// `G` with every node in `nodes` fully removed (both incoming and outgoing
/// edges dropped; node ids and count preserved).
pub(crate) fn remove_nodes(dag: &Dag, nodes: &[DenseNodeId]) -> Result<Dag, IdentificationError> {
    dag.copy_filtered(|from, to| !nodes.contains(&from) && !nodes.contains(&to))
        .map_err(IdentificationError::from)
}

pub(crate) fn var_to_dense(id: VariableId, dag: &Dag) -> Result<DenseNodeId, IdentificationError> {
    crate::prepared::dense_of_static(dag.nodes(), id)
}

pub(crate) fn dense_to_var(id: DenseNodeId, dag: &Dag) -> Result<VariableId, IdentificationError> {
    crate::prepared::node_variable_id(dag.nodes(), id.as_usize())
        .ok_or_else(|| IdentificationError::ExpectedStaticNode)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::oracle_dot::dag_from_oracle_dot;
    use crate::result::IdentificationStatus;
    use antecedent_core::AverageEffectQuery;

    #[test]
    fn mandatory_forks_certify_a_wide_unique_minimal_set() {
        let mut graph = Dag::with_variables(32);
        let t = DenseNodeId::from_raw(30);
        let y = DenseNodeId::from_raw(31);
        for index in 0..30 {
            let z = DenseNodeId::from_raw(index);
            graph.insert_directed(z, t).unwrap();
            graph.insert_directed(z, y).unwrap();
        }
        graph.insert_directed(t, y).unwrap();
        let mut identifier = BackdoorIdentifier::new();
        identifier.config.max_examinations = 1;
        let query = CausalQuery::average_effect(AverageEffectQuery::binary_ate(
            VariableId::from_raw(30),
            VariableId::from_raw(31),
        ));
        let result = identifier
            .identify(
                &identifier.prepare(&graph).unwrap(),
                &query,
                &mut IdentificationWorkspace::default(),
            )
            .unwrap();
        assert_eq!(result.estimands.len(), 1);
        assert_eq!(result.estimands[0].adjustment_set.len(), 30);
        assert_eq!(result.performance.candidates_examined, 1);
        assert!(!crate::envelope::search_truncated(&result));
    }

    #[test]
    fn minimal_sets_match_exhaustive_all_sets_on_every_five_node_ordered_dag() {
        let edges: Vec<_> = (0..5).flat_map(|a| ((a + 1)..5).map(move |b| (a, b))).collect();
        let query = CausalQuery::average_effect(AverageEffectQuery::binary_ate(
            VariableId::from_raw(3),
            VariableId::from_raw(4),
        ));
        for mask in 0u32..(1 << edges.len()) {
            let mut graph = Dag::with_variables(5);
            for (index, &(a, b)) in edges.iter().enumerate() {
                if mask & (1 << index) != 0 {
                    graph
                        .insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b))
                        .unwrap();
                }
            }
            let identifier = BackdoorIdentifier::new();
            let prepared = identifier.prepare(&graph).unwrap();
            let actual = identifier
                .identify(&prepared, &query, &mut IdentificationWorkspace::default())
                .unwrap();
            let mut exhaustive = BackdoorIdentifier::new();
            exhaustive.config.minimal_only = false;
            let all = exhaustive
                .identify(&prepared, &query, &mut IdentificationWorkspace::default())
                .unwrap();
            let mut expected: Vec<Vec<_>> = all
                .estimands
                .iter()
                .filter(|candidate| {
                    !all.estimands.iter().any(|other| {
                        other.adjustment_set.len() < candidate.adjustment_set.len()
                            && other
                                .adjustment_set
                                .iter()
                                .all(|v| candidate.adjustment_set.contains(v))
                    })
                })
                .map(|e| e.adjustment_set.to_vec())
                .collect();
            let mut got: Vec<_> =
                actual.estimands.iter().map(|e| e.adjustment_set.to_vec()).collect();
            expected.sort();
            got.sort();
            assert_eq!(got, expected, "graph mask {mask}");
        }
    }

    #[test]
    fn invalid_costs_and_unknown_configured_variables_are_refused_not_ignored() {
        let mut graph = Dag::with_variables(3);
        let (t, y, z) =
            (DenseNodeId::from_raw(0), DenseNodeId::from_raw(1), DenseNodeId::from_raw(2));
        graph.insert_directed(z, t).unwrap();
        graph.insert_directed(z, y).unwrap();
        graph.insert_directed(t, y).unwrap();
        let query = CausalQuery::average_effect(AverageEffectQuery::binary_ate(
            VariableId::from_raw(0),
            VariableId::from_raw(1),
        ));
        let run = |config: AdjustmentSearchConfig| {
            let identifier = BackdoorIdentifier { config };
            identifier.identify(
                &identifier.prepare(&graph).unwrap(),
                &query,
                &mut IdentificationWorkspace::default(),
            )
        };
        let costs = |costs: Vec<(VariableId, f64)>| AdjustmentSearchConfig {
            measurement_costs: Arc::from(costs),
            ..Default::default()
        };
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -1.0] {
            assert!(
                matches!(
                    run(costs(vec![(VariableId::from_raw(2), bad)])),
                    Err(IdentificationError::InvalidInput { .. })
                ),
                "{bad}"
            );
        }
        // Finite costs whose total overflows are refused too.
        assert!(matches!(
            run(costs(vec![
                (VariableId::from_raw(2), f64::MAX),
                (VariableId::from_raw(0), f64::MAX)
            ])),
            Err(IdentificationError::InvalidInput { .. })
        ));
        assert!(matches!(
            run(costs(vec![(VariableId::from_raw(9), 1.0)])),
            Err(IdentificationError::UnknownVariable { .. })
        ));
        assert!(run(costs(vec![(VariableId::from_raw(2), 0.5)])).is_ok());
        let lagged = AdjustmentSearchConfig::default()
            .with_history_lags(Some(1), vec![(VariableId::from_raw(9), 2u32)]);
        assert!(matches!(run(lagged), Err(IdentificationError::UnknownVariable { .. })));
        assert!(
            AdjustmentSearchConfig::default()
                .with_measurement_costs(vec![(VariableId::from_raw(2), f64::NAN)])
                .is_err()
        );
    }

    #[test]
    fn cost_order_is_stable_and_total() {
        let mut graph = Dag::with_variables(3);
        graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let config = AdjustmentSearchConfig {
            measurement_costs: Arc::from([(VariableId::from_raw(2), 5.0)]),
            ..Default::default()
        };
        let problem = AdjustmentProblem::new(
            &graph,
            &NodeMap::from_dag(&graph),
            VariableId::from_raw(0),
            VariableId::from_raw(1),
            &config,
            &mut GraphWorkspace::default(),
        )
        .unwrap();
        let a = vec![DenseNodeId::from_raw(2)];
        let b = vec![DenseNodeId::from_raw(0)];
        let c = vec![DenseNodeId::from_raw(1)];
        let mut sets = vec![a.clone(), b.clone(), c.clone()];
        problem.order_by_cost(&mut sets);
        // `b` and `c` cost 1.0 each and keep their relative order; `a` costs 5.0.
        assert_eq!(sets, vec![b, c, a]);
    }

    #[test]
    fn var_to_dense_respects_node_labels_not_raw_ids() {
        // Nodes labeled with non-dense VariableIds (raw 10, 20, 30 at dense 0,1,2).
        let mut g = Dag::empty();
        let t = g.add_node(antecedent_graph::NodeRef::Static(VariableId::from_raw(10))).unwrap();
        let y = g.add_node(antecedent_graph::NodeRef::Static(VariableId::from_raw(20))).unwrap();
        let z = g.add_node(antecedent_graph::NodeRef::Static(VariableId::from_raw(30))).unwrap();
        g.insert_directed(z, t).unwrap();
        g.insert_directed(z, y).unwrap();
        g.insert_directed(t, y).unwrap();

        assert_eq!(var_to_dense(VariableId::from_raw(10), &g).unwrap(), t);
        assert_eq!(var_to_dense(VariableId::from_raw(20), &g).unwrap(), y);
        assert_eq!(var_to_dense(VariableId::from_raw(30), &g).unwrap(), z);
        assert!(var_to_dense(VariableId::from_raw(0), &g).is_err());

        let id = BackdoorIdentifier::new();
        let prep = id.prepare(&g).unwrap();
        let q = CausalQuery::average_effect(AverageEffectQuery::binary_ate(
            VariableId::from_raw(10),
            VariableId::from_raw(20),
        ));
        let mut ws = IdentificationWorkspace::default();
        let res = id.identify(&prep, &q, &mut ws).unwrap();
        assert_eq!(res.status, IdentificationStatus::NonparametricallyIdentified);
        assert_eq!(res.estimands[0].adjustment_set.as_ref(), &[VariableId::from_raw(30)]);
    }

    #[test]
    fn confounding_requires_z() {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../../conformance/identify/general_id_backdoor_chain/expected.json"
        ))
        .unwrap();
        assert_eq!(fixture["case"], "identifiable_backdoor");
        assert_eq!(fixture["treatment"], "t");
        assert_eq!(fixture["outcome"], "y");
        assert_eq!(fixture["expected_status_family"], "identified");
        assert!(
            fixture["reference"]["outputs"]["estimand"]
                .as_str()
                .unwrap()
                .contains("Estimand name: backdoor\nEstimand expression:"),
            "external oracle must identify Z -> T, Z -> Y, T -> Y by backdoor"
        );

        let (g, nodes) = dag_from_oracle_dot(fixture["graph_dot"].as_str().unwrap());
        let treatment = nodes.id(fixture["treatment"].as_str().unwrap());
        let outcome = nodes.id(fixture["outcome"].as_str().unwrap());

        let id = BackdoorIdentifier::new();
        let prep = id.prepare(&g).unwrap();
        let q = CausalQuery::average_effect(AverageEffectQuery::binary_ate(treatment, outcome));
        let mut ws = IdentificationWorkspace::default();
        let res = id.identify(&prep, &q, &mut ws).unwrap();
        assert_eq!(res.status, IdentificationStatus::NonparametricallyIdentified);
        assert_eq!(res.estimands.len(), 1);
        assert_eq!(res.estimands[0].adjustment_set.as_ref(), &[nodes.id("z")]);
    }

    #[test]
    fn result_limit_truncates_instead_of_erroring() {
        // T <- A -> B -> C -> Y, T -> Y: minimal singletons {A}, {B}, {C}.
        // With max_results = 2 the first two are returned, not an error.
        let mut g = Dag::with_variables(5);
        let t = DenseNodeId::from_raw(0);
        let y = DenseNodeId::from_raw(1);
        let a = DenseNodeId::from_raw(2);
        let b = DenseNodeId::from_raw(3);
        let c = DenseNodeId::from_raw(4);
        g.insert_directed(a, t).unwrap();
        g.insert_directed(a, b).unwrap();
        g.insert_directed(b, c).unwrap();
        g.insert_directed(c, y).unwrap();
        g.insert_directed(t, y).unwrap();

        let mut id = BackdoorIdentifier::new();
        id.config.max_results = 2;
        let prep = id.prepare(&g).unwrap();
        let q = CausalQuery::average_effect(AverageEffectQuery::binary_ate(
            VariableId::from_raw(0),
            VariableId::from_raw(1),
        ));
        let mut ws = IdentificationWorkspace::default();
        let res = id.identify(&prep, &q, &mut ws).unwrap();
        assert_eq!(res.status, IdentificationStatus::NonparametricallyIdentified);
        assert_eq!(res.estimands.len(), 2);
        assert!(res.derivation.steps.iter().any(|s| s.rule.as_ref() == "backdoor.enumeration"));
    }

    #[test]
    fn inclusion_minimal_keeps_distinct_cardinalities() {
        // Two disjoint backdoor paths: T <- A -> Y and T <- B <- C -> Y.
        // {A,B}, {A,C} are min-cardinality; with minimal_only we still want only
        // inclusion-minimal sets. Here singletons fail, so {A,B} and {A,C} both
        // qualify at size 2 — and continuing sizes must not add their supersets.
        let mut g = Dag::with_variables(5);
        let t = DenseNodeId::from_raw(0);
        let y = DenseNodeId::from_raw(1);
        let a = DenseNodeId::from_raw(2);
        let b = DenseNodeId::from_raw(3);
        let c = DenseNodeId::from_raw(4);
        g.insert_directed(a, t).unwrap();
        g.insert_directed(a, y).unwrap();
        g.insert_directed(b, t).unwrap();
        g.insert_directed(c, b).unwrap();
        g.insert_directed(c, y).unwrap();
        g.insert_directed(t, y).unwrap();

        let id = BackdoorIdentifier::new();
        let prep = id.prepare(&g).unwrap();
        let q = CausalQuery::average_effect(AverageEffectQuery::binary_ate(
            VariableId::from_raw(0),
            VariableId::from_raw(1),
        ));
        let mut ws = IdentificationWorkspace::default();
        let res = id.identify(&prep, &q, &mut ws).unwrap();
        assert_eq!(res.status, IdentificationStatus::NonparametricallyIdentified);
        let sets: Vec<Vec<VariableId>> =
            res.estimands.iter().map(|e| e.adjustment_set.iter().copied().collect()).collect();
        // Inclusion-minimal: {A,B} and {A,C} (size 2). Supersets like {A,B,C} excluded.
        assert!(sets.iter().all(|s| s.len() == 2), "sets={sets:?}");
        assert_eq!(sets.len(), 2, "sets={sets:?}");
    }

    #[test]
    fn empty_adjustment_when_no_backdoor() {
        // T -> Y only
        let mut g = Dag::with_variables(2);
        g.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let id = BackdoorIdentifier::new();
        let prep = id.prepare(&g).unwrap();
        let q = CausalQuery::average_effect(AverageEffectQuery::binary_ate(
            VariableId::from_raw(0),
            VariableId::from_raw(1),
        ));
        let mut ws = IdentificationWorkspace::default();
        let res = id.identify(&prep, &q, &mut ws).unwrap();
        assert_eq!(res.status, IdentificationStatus::NonparametricallyIdentified);
        assert!(res.estimands[0].adjustment_set.is_empty());
        assert_eq!(res.performance.candidates_examined, 1);
    }

    #[test]
    fn empty_minimal_set_does_not_enumerate_isolates() {
        // T → Y plus 16 isolated covariates: only ∅ is inclusion-minimal.
        let mut g = Dag::with_variables(18);
        g.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let id = BackdoorIdentifier::new();
        let prep = id.prepare(&g).unwrap();
        let q = CausalQuery::average_effect(AverageEffectQuery::binary_ate(
            VariableId::from_raw(0),
            VariableId::from_raw(1),
        ));
        let mut ws = IdentificationWorkspace::default();
        let res = id.identify(&prep, &q, &mut ws).unwrap();
        assert_eq!(res.status, IdentificationStatus::NonparametricallyIdentified);
        assert!(res.estimands[0].adjustment_set.is_empty());
        assert_eq!(res.performance.candidates_examined, 1);
        assert_eq!(res.estimands.len(), 1);
    }

    #[test]
    fn missing_admissible_set_is_proved_by_one_test_not_a_budgeted_search() {
        // Thirty measured confounders C_i (T <- C_i -> Y) and a confounder U (T <- U -> Y)
        // that the caller forbids. The back-door path through U cannot be blocked by any
        // admissible set, so identification fails because of the graph; the old subset
        // search hit its budget here and reported that as undecided.
        let n_conf = 30u32;
        let mut g = Dag::with_variables(3 + n_conf);
        let t = DenseNodeId::from_raw(0);
        let y = DenseNodeId::from_raw(1);
        let u = DenseNodeId::from_raw(2);
        g.insert_directed(t, y).unwrap();
        g.insert_directed(u, t).unwrap();
        g.insert_directed(u, y).unwrap();
        for i in 0..n_conf {
            let c = DenseNodeId::from_raw(3 + i);
            g.insert_directed(c, t).unwrap();
            g.insert_directed(c, y).unwrap();
        }
        let mut id = BackdoorIdentifier::new();
        id.config.forbidden = Arc::from([VariableId::from_raw(2)]);
        let prep = id.prepare(&g).unwrap();
        let q = CausalQuery::average_effect(AverageEffectQuery::binary_ate(
            VariableId::from_raw(0),
            VariableId::from_raw(1),
        ));
        let res = id.identify(&prep, &q, &mut IdentificationWorkspace::default()).unwrap();
        assert_eq!(res.status, IdentificationStatus::NotIdentified);
        assert!(
            !crate::envelope::search_truncated(&res),
            "a proof is not a bounded search: {:?}",
            res.diagnostics
        );
        assert_eq!(res.performance.candidates_examined, 1);
    }

    #[test]
    fn over_cap_falls_back_to_the_ancestral_set_when_the_o_set_is_forbidden() {
        // C2..C5 confound T and Y; C6 only causes Y and is forbidden. With max_candidates = 2
        // the exact search is skipped and the O-set pa(Y) minus T holds the forbidden C6, so
        // the certified fallback is the ancestral set {C2..C5}, exactly the confounders.
        let mut g = Dag::with_variables(7);
        let t = DenseNodeId::from_raw(0);
        let y = DenseNodeId::from_raw(1);
        g.insert_directed(t, y).unwrap();
        for i in 2..=5 {
            let c = DenseNodeId::from_raw(i);
            g.insert_directed(c, t).unwrap();
            g.insert_directed(c, y).unwrap();
        }
        g.insert_directed(DenseNodeId::from_raw(6), y).unwrap();
        let mut id = BackdoorIdentifier::new().with_max_candidates(2);
        id.config.forbidden = Arc::from([VariableId::from_raw(6)]);
        let prep = id.prepare(&g).unwrap();
        let q = CausalQuery::average_effect(AverageEffectQuery::binary_ate(
            VariableId::from_raw(0),
            VariableId::from_raw(1),
        ));
        let res = id.identify(&prep, &q, &mut IdentificationWorkspace::default()).unwrap();
        assert_eq!(res.status, IdentificationStatus::NonparametricallyIdentified);
        let expected: Vec<VariableId> = (2..=5).map(VariableId::from_raw).collect();
        assert_eq!(res.estimands[0].adjustment_set.as_ref(), expected.as_slice());
    }

    #[test]
    fn examination_budget_ends_the_search_for_smaller_sets_not_the_decision() {
        // U → T, U → Y, T → Y plus isolates: ∅ is invalid; a tiny budget cannot reach {U}
        // by enumeration, but the ancestral set {U} was already tested valid.
        let mut g = Dag::with_variables(12);
        let t = DenseNodeId::from_raw(0);
        let y = DenseNodeId::from_raw(1);
        let u = DenseNodeId::from_raw(11);
        g.insert_directed(u, t).unwrap();
        g.insert_directed(u, y).unwrap();
        g.insert_directed(t, y).unwrap();
        let mut id = BackdoorIdentifier::new();
        // Request all sets: the mandatory-fork certificate only closes the
        // unique minimal-set query, not enumeration of valid supersets.
        id.config.minimal_only = false;
        id.config.max_examinations = 4;
        let prep = id.prepare(&g).unwrap();
        let q = CausalQuery::average_effect(AverageEffectQuery::binary_ate(
            VariableId::from_raw(0),
            VariableId::from_raw(1),
        ));
        let mut ws = IdentificationWorkspace::default();
        let res = id.identify(&prep, &q, &mut ws).unwrap();
        assert_eq!(res.status, IdentificationStatus::NonparametricallyIdentified);
        assert_eq!(res.estimands[0].adjustment_set.as_ref(), &[VariableId::from_raw(11)]);
        let bounded = res
            .diagnostics
            .iter()
            .find(|d| d.code.as_ref() == BACKDOOR_SEARCH_BOUNDED_DIAGNOSTIC_CODE)
            .expect("the unfinished enumeration is reported");
        assert_eq!(bounded.kind, antecedent_core::DiagnosticKind::Execution);
        assert!(res.performance.candidates_examined <= 4);
    }

    #[test]
    fn superset_skips_stop_at_the_examination_cap() {
        // One backdoor chain T ← N1 ← … ← N30 → Y. Every singleton blocks it, so with
        // `minimal_only` every larger subset is a superset. `max_results` stays above 30, so
        // the search does not stop on the result cap; it used to walk the rest of the power
        // set without charging those skips, and `max_examinations` never fired.
        let n = 30u32;
        let mut g = Dag::with_variables(n + 2);
        let t = DenseNodeId::from_raw(0);
        let y = DenseNodeId::from_raw(1);
        g.insert_directed(t, y).unwrap();
        g.insert_directed(DenseNodeId::from_raw(2), t).unwrap();
        for i in 2..=n {
            g.insert_directed(DenseNodeId::from_raw(i + 1), DenseNodeId::from_raw(i)).unwrap();
        }
        g.insert_directed(DenseNodeId::from_raw(n + 1), y).unwrap();

        let mut id = BackdoorIdentifier::new();
        let cap = 40u64;
        id.config.max_examinations = cap;
        let prep = id.prepare(&g).unwrap();
        let q = CausalQuery::average_effect(AverageEffectQuery::binary_ate(
            VariableId::from_raw(0),
            VariableId::from_raw(1),
        ));
        let res = id.identify(&prep, &q, &mut IdentificationWorkspace::default()).unwrap();
        assert_eq!(res.status, IdentificationStatus::NonparametricallyIdentified);
        assert_eq!(res.performance.candidates_examined, cap);
        let sets: Vec<Vec<VariableId>> =
            res.estimands.iter().map(|e| e.adjustment_set.iter().copied().collect()).collect();
        assert_eq!(sets.len(), 30, "sets={sets:?}");
        assert!(sets.iter().all(|s| s.len() == 1), "sets={sets:?}");
        let bounded = res
            .diagnostics
            .iter()
            .find(|d| d.code.as_ref() == BACKDOOR_SEARCH_BOUNDED_DIAGNOSTIC_CODE)
            .expect("the cap stops the skipped supersets");
        assert_eq!(bounded.kind, antecedent_core::DiagnosticKind::Execution);
    }

    #[test]
    fn distribution_and_path_specific_are_unsupported() {
        use antecedent_core::{
            Intervention, InterventionalDistributionQuery, PathSpecificEffectQuery, Value,
        };

        let mut g = Dag::with_variables(2);
        let t = DenseNodeId::from_raw(0);
        let y = DenseNodeId::from_raw(1);
        g.insert_directed(t, y).unwrap();
        let id = BackdoorIdentifier::new();
        let prep = id.prepare(&g).unwrap();
        let mut ws = IdentificationWorkspace::default();

        let dist = CausalQuery::distribution(InterventionalDistributionQuery::new(
            VariableId::from_raw(1),
            [Intervention::set(VariableId::from_raw(0), Value::f64(1.0))],
        ));
        let err = id.identify(&prep, &dist, &mut ws).unwrap_err();
        assert!(matches!(
            err,
            IdentificationError::UnsupportedQuery { message }
            if message.contains("Distribution")
        ));

        let path = CausalQuery::path_specific(PathSpecificEffectQuery::binary(
            VariableId::from_raw(0),
            VariableId::from_raw(1),
        ));
        let err = id.identify(&prep, &path, &mut ws).unwrap_err();
        assert!(matches!(
            err,
            IdentificationError::UnsupportedQuery { message }
            if message.contains("PathSpecific")
        ));
    }

    fn random_dag(rng: &mut antecedent_core::CausalRng, n: u32) -> Dag {
        let mut g = Dag::with_variables(n);
        let mut order: Vec<u32> = (0..n).collect();
        for i in (1..n as usize).rev() {
            let j = (rng.next_u64() as usize) % (i + 1);
            order.swap(i, j);
        }
        for i in 0..n as usize {
            for j in (i + 1)..n as usize {
                if rng.next_u64() % 3 == 0 {
                    let a = DenseNodeId::from_raw(order[i]);
                    let b = DenseNodeId::from_raw(order[j]);
                    let _ = g.insert_directed(a, b);
                }
            }
        }
        g
    }

    fn descendants_of(g: &Dag, t: DenseNodeId) -> std::collections::HashSet<DenseNodeId> {
        let mut s = std::collections::HashSet::new();
        let mut stack = vec![t];
        while let Some(u) = stack.pop() {
            for &c in g.children(u) {
                if s.insert(c) {
                    stack.push(c);
                }
            }
        }
        s
    }

    fn assert_valid_adjustment_sets(
        g: &Dag,
        t: DenseNodeId,
        y: DenseNodeId,
        estimands: &[antecedent_expr::IdentifiedEstimand],
        dsep: &mut antecedent_graph::DSeparationWorkspace,
    ) {
        let mutilated = remove_outgoing(g, t).unwrap();
        let descendants = descendants_of(g, t);
        for est in estimands {
            let z: Vec<DenseNodeId> =
                est.adjustment_set.iter().map(|v| var_to_dense(*v, g).unwrap()).collect();
            for &zi in &z {
                assert!(
                    !descendants.contains(&zi),
                    "adjustment set contains descendant of T: {zi:?}"
                );
                assert_ne!(zi, t);
                assert_ne!(zi, y);
            }
            assert!(
                is_backdoor_adjustment(&mutilated, t, y, &z, dsep).unwrap(),
                "Z={z:?} does not block backdoors T={} Y={}",
                t.raw(),
                y.raw()
            );
        }
    }

    #[test]
    fn property_backdoor_sets_block_and_exclude_descendants() {
        let mut rng = antecedent_core::CausalRng::from_seed(99);
        let id = BackdoorIdentifier::new();
        let eff = crate::EfficientBackdoorIdentifier::new();
        let mut ws = IdentificationWorkspace::default();
        let mut dsep = antecedent_graph::DSeparationWorkspace::default();
        for _ in 0..40 {
            let n = 4 + (rng.next_u64() % 2) as u32; // 4..=5
            let g = random_dag(&mut rng, n);
            let t = DenseNodeId::from_raw((rng.next_u64() % u64::from(n)) as u32);
            let mut y = DenseNodeId::from_raw((rng.next_u64() % u64::from(n)) as u32);
            if y == t {
                y = DenseNodeId::from_raw((t.raw() + 1) % n);
            }
            let prep = id.prepare(&g).unwrap();
            let q = CausalQuery::average_effect(AverageEffectQuery::with_levels(
                VariableId::from_raw(t.raw()),
                VariableId::from_raw(y.raw()),
                // Non-binary levels: identification is graph-theoretic.
                -1.5,
                2.25,
            ));
            if let Ok(res) = id.identify(&prep, &q, &mut ws) {
                if res.status == IdentificationStatus::NonparametricallyIdentified {
                    assert_valid_adjustment_sets(&g, t, y, &res.estimands, &mut dsep);
                }
            }
            if let Ok(res) = eff.identify(&prep, &q, &mut ws) {
                if res.status == IdentificationStatus::NonparametricallyIdentified {
                    assert_valid_adjustment_sets(&g, t, y, &res.estimands, &mut dsep);
                }
            }
        }
    }

    #[test]
    fn max_history_lag_excludes_deep_covariates() {
        // T <- Z_deep, T <- Z_near -> Y, T -> Y.
        // With history lag metadata, capping max_history_lag drops Z_deep from Z.
        let mut g = Dag::with_variables(4);
        let t = DenseNodeId::from_raw(0);
        let y = DenseNodeId::from_raw(1);
        let z_near = DenseNodeId::from_raw(2);
        let z_deep = DenseNodeId::from_raw(3);
        g.insert_directed(z_near, t).unwrap();
        g.insert_directed(z_near, y).unwrap();
        g.insert_directed(z_deep, t).unwrap();
        g.insert_directed(t, y).unwrap();

        let mut id = BackdoorIdentifier::new();
        id.config.max_history_lag = Some(1);
        id.config.history_lags = Arc::from([
            (VariableId::from_raw(2), 1), // near
            (VariableId::from_raw(3), 3), // deep — excluded
        ]);
        let prep = id.prepare(&g).unwrap();
        let q = CausalQuery::average_effect(AverageEffectQuery::binary_ate(
            VariableId::from_raw(0),
            VariableId::from_raw(1),
        ));
        let mut ws = IdentificationWorkspace::default();
        let res = id.identify(&prep, &q, &mut ws).unwrap();
        assert_eq!(res.status, IdentificationStatus::NonparametricallyIdentified);
        let z = res.estimands[0].adjustment_set.as_ref();
        assert!(z.contains(&VariableId::from_raw(2)));
        assert!(!z.contains(&VariableId::from_raw(3)));
    }

    #[test]
    fn history_lag_and_cost_keep_the_first_listed_value() {
        // A later row for the same variable must not override the first, which is what
        // `exceeds_history_lag` returns from `find`.
        let mut g = Dag::with_variables(3);
        let (t, y, z) =
            (DenseNodeId::from_raw(0), DenseNodeId::from_raw(1), DenseNodeId::from_raw(2));
        g.insert_directed(z, t).unwrap();
        g.insert_directed(z, y).unwrap();
        g.insert_directed(t, y).unwrap();
        let mut id = BackdoorIdentifier::new();
        id.config.max_history_lag = Some(1);
        id.config.history_lags =
            Arc::from([(VariableId::from_raw(2), 0u32), (VariableId::from_raw(2), 9)]);
        id.config.measurement_costs =
            Arc::from([(VariableId::from_raw(2), 0.25), (VariableId::from_raw(2), 4.0)]);
        assert!(!id.config.exceeds_history_lag(VariableId::from_raw(2)));
        let prep = id.prepare(&g).unwrap();
        let q = CausalQuery::average_effect(AverageEffectQuery::binary_ate(
            VariableId::from_raw(0),
            VariableId::from_raw(1),
        ));
        let res = id.identify(&prep, &q, &mut IdentificationWorkspace::default()).unwrap();
        assert_eq!(res.estimands[0].adjustment_set.as_ref(), &[VariableId::from_raw(2)]);
        let problem = AdjustmentProblem::new(
            &g,
            &NodeMap::from_dag(&g),
            VariableId::from_raw(0),
            VariableId::from_raw(1),
            &id.config,
            &mut GraphWorkspace::default(),
        )
        .unwrap();
        assert!((problem.total_cost(&[z]) - 0.25).abs() < 1e-12);
    }

    /// One confounder Z plus `extra` isolated candidates (total candidates = 1+extra).
    /// Minimal valid Z is `{confounder}` at size 1 — safe with large candidate pools.
    fn one_confounder_many_candidates(extra: u32) -> Dag {
        let mut g = Dag::with_variables(3 + extra);
        let t = DenseNodeId::from_raw(0);
        let y = DenseNodeId::from_raw(1);
        let z = DenseNodeId::from_raw(2);
        g.insert_directed(z, t).unwrap();
        g.insert_directed(z, y).unwrap();
        g.insert_directed(t, y).unwrap();
        g
    }

    #[test]
    fn default_declares_no_ceiling_and_allows_twenty_five_candidates() {
        // 1 confounder + 24 isolates → 25 candidates; no ceiling is declared by default.
        let g = one_confounder_many_candidates(24);
        let mut id = BackdoorIdentifier::new();
        assert_eq!(id.config.max_candidates, usize::MAX);
        id.config.max_results = 1;
        let prep = id.prepare(&g).unwrap();
        let q = CausalQuery::average_effect(AverageEffectQuery::binary_ate(
            VariableId::from_raw(0),
            VariableId::from_raw(1),
        ));
        let mut ws = IdentificationWorkspace::default();
        let res = id.identify(&prep, &q, &mut ws).unwrap();
        assert_eq!(res.status, IdentificationStatus::NonparametricallyIdentified);
        assert_eq!(res.estimands.len(), 1);
        assert_eq!(res.estimands[0].adjustment_set.as_ref(), &[VariableId::from_raw(2)]);
    }

    /// `U -> T`, `U -> Y`, `T -> Y` and ten isolates, all sets requested so the mandatory-fork
    /// certificate does not close the query.
    fn confounder_with_isolates() -> (Dag, CausalQuery) {
        let mut g = Dag::with_variables(12);
        let (t, y, u) =
            (DenseNodeId::from_raw(0), DenseNodeId::from_raw(1), DenseNodeId::from_raw(11));
        g.insert_directed(u, t).unwrap();
        g.insert_directed(u, y).unwrap();
        g.insert_directed(t, y).unwrap();
        let q = CausalQuery::average_effect(AverageEffectQuery::binary_ate(
            VariableId::from_raw(0),
            VariableId::from_raw(1),
        ));
        (g, q)
    }

    fn identify_in_pool(
        g: &Dag,
        q: &CausalQuery,
        pool: &antecedent_core::execution::AdmissionPool,
    ) -> Result<IdentificationResult, IdentificationError> {
        let ctx = ExecutionContext::for_tests(1).with_admission(pool.clone());
        let mut id = BackdoorIdentifier::new().with_context(ctx);
        id.config.minimal_only = false;
        let prep = id.prepare(g).unwrap();
        id.identify(&prep, q, &mut IdentificationWorkspace::default())
    }

    #[test]
    fn enumeration_storage_is_leased_and_a_pool_that_cannot_fund_it_falls_back_to_a_certified_set()
    {
        use antecedent_core::execution::AdmissionPool;
        let (g, q) = confounder_with_isolates();
        let funded = AdmissionPool::fixed(1, 1 << 30);
        let full = identify_in_pool(&g, &q, &funded).unwrap();
        assert_eq!(full.estimands[0].adjustment_set.as_ref(), &[VariableId::from_raw(11)]);
        assert!(
            !full.derivation.steps.iter().any(|s| s.detail.contains("exact enumeration skipped"))
        );
        assert!(full.performance.candidates_examined > 2);
        assert_eq!(funded.available().bytes(), 1 << 30);

        // The graph is funded, the ten-candidate enumeration is not, and the ancestral
        // screen (one candidate) is: the search screens, then enumerates the screen.
        let screen = AdmissionPool::fixed(1, 1_000);
        let screened = identify_in_pool(&g, &q, &screen).unwrap();
        assert_eq!(screened.estimands[0].adjustment_set.as_ref(), &[VariableId::from_raw(11)]);
        assert!(screened.derivation.steps.iter().any(|s| s.detail.contains("backdoor.screen")));
        assert_eq!(screen.available().bytes(), 1_000);

        // Only the graph is funded: the search returns the O-set after one more test and
        // examines no subset.
        let tight = AdmissionPool::fixed(1, 800);
        let o_set = identify_in_pool(&g, &q, &tight).unwrap();
        assert_eq!(o_set.estimands[0].adjustment_set.as_ref(), &[VariableId::from_raw(11)]);
        assert!(
            o_set.derivation.steps.iter().any(|s| s.detail.contains("exact enumeration skipped"))
        );
        assert_eq!(o_set.performance.candidates_examined, 2);
        assert_eq!(tight.available().bytes(), 800);

        // A pool that cannot hold the graph refuses before anything is built.
        let none = AdmissionPool::fixed(1, 100);
        let refused = identify_in_pool(&g, &q, &none).unwrap_err();
        assert!(matches!(
            refused,
            IdentificationError::Budget { budget: crate::IdentificationBudget::Memory }
        ));
        assert_eq!(none.available().bytes(), 100);
    }

    #[test]
    fn a_cancelled_context_stops_the_search() {
        let (g, q) = confounder_with_isolates();
        let ctx = ExecutionContext::for_tests(1);
        let id = BackdoorIdentifier::new().with_context(ctx.clone());
        let prep = id.prepare(&g).unwrap();
        ctx.cancellation.cancel();
        let err = id.identify(&prep, &q, &mut IdentificationWorkspace::default()).unwrap_err();
        assert!(matches!(err, IdentificationError::Cancelled));
    }

    #[test]
    fn above_default_cap_screens_isolates_and_identifies() {
        // 1 confounder + 40 isolates → 41 raw candidates; An({T,Y}) screen → {Z}.
        let g = one_confounder_many_candidates(40);
        let id = BackdoorIdentifier::new();
        let prep = id.prepare(&g).unwrap();
        let q = CausalQuery::average_effect(AverageEffectQuery::binary_ate(
            VariableId::from_raw(0),
            VariableId::from_raw(1),
        ));
        let mut ws = IdentificationWorkspace::default();
        let res = id.identify(&prep, &q, &mut ws).unwrap();
        assert_eq!(res.status, IdentificationStatus::NonparametricallyIdentified);
        assert_eq!(res.estimands[0].adjustment_set.as_ref(), &[VariableId::from_raw(2)]);
        assert!(res.derivation.steps.iter().any(|s| s.detail.contains("An({T,Y})")));
    }

    #[test]
    fn lower_custom_cap_screens_then_identifies() {
        let g = one_confounder_many_candidates(24);
        let id = BackdoorIdentifier::new().with_max_candidates(20);
        let prep = id.prepare(&g).unwrap();
        let q = CausalQuery::average_effect(AverageEffectQuery::binary_ate(
            VariableId::from_raw(0),
            VariableId::from_raw(1),
        ));
        let mut ws = IdentificationWorkspace::default();
        let res = id.identify(&prep, &q, &mut ws).unwrap();
        assert_eq!(res.status, IdentificationStatus::NonparametricallyIdentified);
        assert_eq!(res.estimands[0].adjustment_set.as_ref(), &[VariableId::from_raw(2)]);
    }
}

/// Checked sufficient joint-backdoor certificate for an explicitly declared adjustment set.
/// This certificate retains the graph and role ids, independent of numerical row data.
#[derive(Clone, Debug)]
pub struct CheckedJointAdjustment {
    graph: Dag,
    treatments: Arc<[VariableId]>,
    outcome: VariableId,
    adjustment: Arc<[VariableId]>,
}
impl CheckedJointAdjustment {
    /// Graph whose joint backdoor paths were checked.
    #[must_use]
    pub fn graph(&self) -> &Dag {
        &self.graph
    }
    /// Joint intervention roles in declared order.
    #[must_use]
    pub fn treatments(&self) -> &[VariableId] {
        &self.treatments
    }
    /// Outcome role.
    #[must_use]
    pub const fn outcome(&self) -> VariableId {
        self.outcome
    }
    /// Validated common adjustment roles.
    #[must_use]
    pub fn adjustment(&self) -> &[VariableId] {
        &self.adjustment
    }
}
/// Check a declared common adjustment set using the joint sufficient backdoor criterion.
/// Returns `None` when a forbidden descendant or an unblocked backdoor path is found.
/// Unlike subset enumeration, this evaluates the complete supplied set, including valid
/// nonminimal sets, and has no combinatorial search budget.
pub fn check_joint_adjustment(
    graph: &Dag,
    treatments: &[VariableId],
    outcome: VariableId,
    adjustment: &[VariableId],
) -> Result<Option<CheckedJointAdjustment>, IdentificationError> {
    antecedent_core::execution_attempt::run_operation(
        antecedent_core::execution_attempt::Operation::Identification,
        || {
            let valid = |id: VariableId| (id.raw() as usize) < graph.node_count();
            let treatment_set: std::collections::BTreeSet<_> = treatments.iter().copied().collect();
            let adjustment_set: std::collections::BTreeSet<_> =
                adjustment.iter().copied().collect();
            if treatments.is_empty()
                || treatment_set.len() != treatments.len()
                || adjustment_set.len() != adjustment.len()
                || !valid(outcome)
                || treatments.iter().chain(adjustment).any(|id| !valid(*id))
                || treatments.contains(&outcome)
                || adjustment.iter().any(|id| treatments.contains(id) || *id == outcome)
            {
                return Err(IdentificationError::InvalidQuery {
                    message: "invalid joint adjustment roles".into(),
                });
            }
            let nodes: Vec<_> =
                treatments.iter().map(|id| DenseNodeId::from_raw(id.raw())).collect();
            let mut downstream = BitSet::with_len(graph.node_count());
            graph.descendants_of(&nodes, &mut downstream, &mut GraphWorkspace::default());
            // Adjustment roles never include a treatment (checked above), so the
            // reflexive descendant set decides exactly the strict-descendant test.
            if adjustment.iter().any(|id| downstream.contains(DenseNodeId::from_raw(id.raw()))) {
                return Ok(None);
            }
            let mutilated = remove_outgoing_set(graph, &nodes)?;
            let z: Vec<_> = adjustment.iter().map(|id| DenseNodeId::from_raw(id.raw())).collect();
            let mut ws = DSeparationWorkspace::default();
            for t in nodes {
                if !is_backdoor_adjustment(
                    &mutilated,
                    t,
                    DenseNodeId::from_raw(outcome.raw()),
                    &z,
                    &mut ws,
                )? {
                    return Ok(None);
                }
            }
            Ok(Some(CheckedJointAdjustment {
                graph: graph.clone(),
                treatments: Arc::from(treatments),
                outcome,
                adjustment: Arc::from(adjustment),
            }))
        },
    )
}
