//! Temporal identification over finite unfolded graphs.
//!
//! **Pulse / single-step Sustained / single-step Dynamic:** a stationary
//! [`TemporalDag`] is unfolded to a static [`Dag`](antecedent_graph::Dag), then
//! [`BackdoorIdentifier`] runs on the treatment/outcome nodes. Single-step
//! schedules are the same contrast as a pulse at that offset; they must not
//! take the empty-`Z` general-ID path.
//!
//! **Multi-step Sustained / Dynamic:** the same unfolding covers every
//! treatment-time node; identification uses [`IdIdentifier`] (sequential /
//! g-formula). The 0.7 linear estimator still refuses those multi-step
//! schedules at estimate time.
//!
//! Finiteness and stationarity of the template become declared assumptions on the
//! result. History depth grows until ancestral closure of `{treatment, outcome}`
//! no longer touches the truncated boundary (or until `max_history_lag` / a
//! derived cap refuses certification).
//!
//! **Parent adjustment:** when that cap refuses a single-step Pulse and the
//! caller enabled [`TemporalBackdoorIdentifier::parent_adjustment_fallback`],
//! the pulse is identified by adjusting for the treatment's own parents
//! (derivation rule [`PARENT_ADJUSTMENT_RULE`]), which is valid in the unrolled
//! DAG whatever the depth of the treatment's ancestry.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::too_many_lines, clippy::unused_self)]
#![cfg_attr(
    test,
    allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "test fixtures compare exact constants and index with small literals"
    )
)]

use std::sync::Arc;

use antecedent_core::{
    Assumption, AssumptionRecord, AssumptionScope, AssumptionSource, AssumptionStatus,
    AverageEffectQuery, CausalQuery, Intervention, TemporalEffectQuery, TemporalPolicy, VariableId,
};
use antecedent_data::{TemporalIndexer, TemporalNodeKey};
use antecedent_graph::{
    BitSet, DenseNodeId, GraphWorkspace, NodeRef, TemporalDag, UnfoldedTemporalGraph,
};

use crate::backdoor::BackdoorIdentifier;
use crate::error::IdentificationError;
use crate::id::IdIdentifier;
use crate::identifier::IdentificationWorkspace;
use crate::prepared::PreparedAdmg;
use crate::result::IdentificationResult;

/// Identifies [`TemporalEffectQuery`]s via backdoor adjustment over a finite
/// unfolding of a stationary [`TemporalDag`] template.
#[derive(Clone, Debug, Default)]
pub struct TemporalBackdoorIdentifier {
    /// Static backdoor identifier applied to the unfolded graph.
    pub inner: BackdoorIdentifier,
    /// When set, a single-step [`TemporalPolicy::Pulse`] whose unfolding
    /// cannot certify (for example an autoregressive treatment edge) is
    /// identified by adjusting for the treatment's own parents instead
    /// ([`Self::identify_pulse_by_parent_adjustment`], derivation rule
    /// [`PARENT_ADJUSTMENT_RULE`]). Off by default: callers enable it only
    /// where the graph is one fully oriented [`TemporalDag`] the analysis
    /// accepted, never for completions of a class or posterior atoms.
    pub parent_adjustment_fallback: bool,
}

/// Derivation rule id of a single-step pulse identified by adjusting for the
/// treatment's parents `pa(T[t])`.
pub const PARENT_ADJUSTMENT_RULE: &str = "temporal.parent_adjustment";

/// Backdoor identification result paired with the finite-unfolding context
/// needed to reinterpret dense adjustment-set ids as `(variable, offset)`
/// pairs.
#[derive(Clone, Debug)]
pub struct TemporalIdentificationResult {
    /// Backdoor identification result over the unfolded static DAG. Its
    /// `treatment`/`outcome`/adjustment-set ids are dense unfolded node ids,
    /// not the original template [`VariableId`]s.
    pub result: IdentificationResult,
    /// Indexer used for the finite unfolding (dense id <-> temporal key).
    pub indexer: TemporalIndexer,
    /// Temporal key of the treatment node used for identification.
    pub treatment_key: TemporalNodeKey,
    /// Temporal key of the outcome node used for identification.
    pub outcome_key: TemporalNodeKey,
}

impl TemporalBackdoorIdentifier {
    /// Create with default config.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable the parent-adjustment fallback for single-step pulses
    /// ([`Self::parent_adjustment_fallback`]).
    #[must_use]
    pub fn with_parent_adjustment_fallback(mut self) -> Self {
        self.parent_adjustment_fallback = true;
        self
    }

    /// Unfold `template` to a finite static DAG sized for `query`, then run
    /// backdoor identification for the treatment/outcome nodes implied by the
    /// query's temporal policy and horizon.
    ///
    /// The unfolding window starts from the policy offset, `horizon_steps`,
    /// and the template's own maximum lag, then grows one slice at a time
    /// until no ancestor of the treatment or outcome in the unfolded graph
    /// sits at the deepest slice with truncated in-template parents. At that
    /// fixed point, deeper windows cannot add backdoor paths, so the finite
    /// identification is exact. Growth is capped by the query's
    /// `max_history_lag` when set, otherwise by
    /// `variable_count * template_max_lag + |min_offset|` (which bounds
    /// simple confounder chains).
    ///
    /// # Errors
    ///
    /// Invalid query, unfolding failures, backdoor / general-ID identification
    /// errors, or [`IdentificationError::NotCertified`] when the history
    /// cap is reached while confounder ancestry still crosses the truncated
    /// boundary (a clean result cannot be certified).
    pub fn identify_temporal(
        &self,
        template: &TemporalDag,
        query: &TemporalEffectQuery,
    ) -> Result<TemporalIdentificationResult, IdentificationError> {
        self.identify_temporal_with_history(template, query, 0)
    }

    /// Reuse a common certified history when comparing completion functionals.
    pub(crate) fn identify_temporal_with_history(
        &self,
        template: &TemporalDag,
        query: &TemporalEffectQuery,
        minimum_history: u32,
    ) -> Result<TemporalIdentificationResult, IdentificationError> {
        query.validate().map_err(|_| IdentificationError::UnsupportedQuery {
            message: "invalid temporal-effect query",
        })?;
        let treatment_at =
            query.try_treatment_offset().map_err(|_| IdentificationError::UnsupportedQuery {
                message: "TemporalPolicy::Dynamic requires a non-empty active_at schedule",
            })?;
        let outcome_at = query.outcome_offset();
        match &query.policy {
            // Multi-step schedules need sequential / g-formula. A one-step
            // Sustained or Dynamic is the same treatment-time node as a Pulse
            // and must use backdoor `Z`, not the empty-adjustment General ID
            // functional `identify_schedule_contrast` emits.
            TemporalPolicy::Sustained { from, until } if from != until => {
                let schedule: Vec<_> =
                    (*from..=*until).map(|offset| (query.treatment, offset)).collect();
                return self.identify_active_offsets(template, query, &schedule, outcome_at);
            }
            TemporalPolicy::Dynamic { active_at, .. } if active_at.len() != 1 => {
                let schedule: Vec<_> =
                    active_at.iter().map(|&offset| (query.treatment, offset)).collect();
                return self.identify_active_offsets(template, query, &schedule, outcome_at);
            }
            TemporalPolicy::Pulse { .. }
            | TemporalPolicy::Sustained { .. }
            | TemporalPolicy::Dynamic { .. } => {}
            _ => {
                return Err(IdentificationError::UnsupportedQuery {
                    message: "unsupported temporal policy for identification",
                });
            }
        }

        let min_offset = treatment_at.min(outcome_at).min(0);
        let max_offset = treatment_at.max(outcome_at).max(0);
        let horizon = u32::try_from(max_offset)
            .map_err(|_| IdentificationError::msg("negative horizon"))?
            .saturating_add(1);

        let variable_count = required_variable_count(template, [query.treatment, query.outcome]);
        let max_lag = template_max_lag(template);
        let base_history = min_offset.unsigned_abs().max(max_lag).max(minimum_history);
        // The user's max_history_lag, when set, caps window growth; otherwise
        // bound simple confounder chains through every template variable.
        let chain_cap =
            variable_count.saturating_mul(max_lag).saturating_add(min_offset.unsigned_abs());
        let history_cap = query.max_history_lag.unwrap_or(chain_cap).max(base_history);

        let treatment_key = TemporalNodeKey { variable: query.treatment, offset: treatment_at };
        let outcome_key = TemporalNodeKey { variable: query.outcome, offset: outcome_at };
        let truncatable = truncatable_variables(template, variable_count);

        // Grow the history until no ancestor of {treatment, outcome} in the
        // unfolded graph sits at the deepest slice with cut template parents;
        // at that point deeper windows cannot add backdoor paths.
        let mut history = base_history;
        let (history, unfolded, treatment_dense, outcome_dense) = loop {
            let indexer = TemporalIndexer::new(variable_count, history, horizon)
                .map_err(|e| IdentificationError::msg(e.to_string()))?;
            let unfolded =
                template.unfold(indexer).map_err(|e| IdentificationError::msg(e.to_string()))?;

            let treatment_dense = unfolded
                .indexer
                .dense_id(treatment_key)
                .map_err(|_| IdentificationError::UnknownVariable { id: query.treatment })?;
            let outcome_dense = unfolded
                .indexer
                .dense_id(outcome_key)
                .map_err(|_| IdentificationError::UnknownVariable { id: query.outcome })?;

            if !ancestry_touches_boundary(
                &unfolded,
                &[treatment_dense, outcome_dense],
                history,
                &truncatable,
            ) {
                break (history, unfolded, treatment_dense, outcome_dense);
            }
            if history >= history_cap {
                let refusal = history_cap_refusal(history_cap, chain_cap);
                if self.parent_adjustment_fallback
                    && matches!(query.policy, TemporalPolicy::Pulse { .. })
                {
                    let IdentificationError::NotCertified { message } = refusal else {
                        return Err(refusal);
                    };
                    return self.parent_adjustment(template, query, Some(message));
                }
                return Err(refusal);
            }
            history += 1;
        };

        // Unfolded DAGs are built via `Dag::with_variables`, so dense node i is labeled
        // `VariableId::from_raw(i)`. These synthetic ids are only for the unfolded graph;
        // `annotate_temporal` remaps results back to `TemporalNodeKey`s.
        let treatment_var = VariableId::from_raw(treatment_dense);
        let outcome_var = VariableId::from_raw(outcome_dense);

        let ate = AverageEffectQuery::new(
            treatment_var,
            outcome_var,
            Arc::from([]),
            retarget(&query.control, treatment_var)?,
            retarget(&query.active, treatment_var)?,
            query.target_population.clone(),
        );

        let mut identifier = self.inner.clone();
        apply_history_lag_filter(
            &mut identifier.config,
            &unfolded.indexer,
            treatment_key.offset.max(outcome_key.offset),
            query.max_history_lag,
        );
        let prepared = identifier.prepare(&unfolded.dag)?;
        let mut id_ws = IdentificationWorkspace::default();
        let mut result =
            identifier.identify(&prepared, &CausalQuery::average_effect(ate), &mut id_ws)?;
        annotate_temporal(&mut result, query, treatment_key, outcome_key, history, horizon);

        Ok(TemporalIdentificationResult {
            result,
            indexer: unfolded.indexer,
            treatment_key,
            outcome_key,
        })
    }

    /// Identify a single-step pulse `do(T[t] = x)` on `Y[t + h]` by adjusting
    /// for the treatment's own parents `pa(T[t])`, lagged and contemporaneous.
    ///
    /// In a temporal DAG with no latent structure, every back-door path into
    /// `T[t]` enters through a parent, and a parent is a non-collider on that
    /// path, so `pa(T[t])` blocks every back-door path; no parent is a
    /// descendant of `T[t]`. This holds in the infinite unrolled graph, so it
    /// needs no finite-window certificate and stands where unfolding cannot
    /// certify (an autoregressive treatment edge makes the treatment's ancestry
    /// unbounded). The window is sized to contain every parent, and the set is
    /// checked against the back-door criterion on that window.
    ///
    /// The result records derivation rule [`PARENT_ADJUSTMENT_RULE`] and its
    /// premises (causal sufficiency; every parent oriented and observed at its
    /// lag). The estimand method stays `temporal.backdoor.unfolded`, so the
    /// temporal linear adjustment estimator fits the declared set unchanged.
    ///
    /// # Errors
    ///
    /// [`IdentificationError::UnsupportedQuery`] for any policy other than a
    /// [`TemporalPolicy::Pulse`] (time-varying confounding makes parent
    /// adjustment insufficient for multi-step schedules), and
    /// [`IdentificationError::NotCertified`] when a parent lies beyond the
    /// query's `max_history_lag` (not available within the permitted history)
    /// or the outcome node is itself a parent of the treatment.
    pub fn identify_pulse_by_parent_adjustment(
        &self,
        template: &TemporalDag,
        query: &TemporalEffectQuery,
    ) -> Result<TemporalIdentificationResult, IdentificationError> {
        self.parent_adjustment(template, query, None)
    }

    fn parent_adjustment(
        &self,
        template: &TemporalDag,
        query: &TemporalEffectQuery,
        unfolding_refusal: Option<&'static str>,
    ) -> Result<TemporalIdentificationResult, IdentificationError> {
        query.validate().map_err(|_| IdentificationError::UnsupportedQuery {
            message: "invalid temporal-effect query",
        })?;
        if !matches!(query.policy, TemporalPolicy::Pulse { .. }) {
            return Err(IdentificationError::UnsupportedQuery {
                message: "parent adjustment identifies only a single-step Pulse; multi-step \
                          schedules need sequential (g-formula) identification",
            });
        }
        let treatment_at = query.try_treatment_offset().map_err(|_| {
            IdentificationError::UnsupportedQuery { message: "invalid pulse treatment offset" }
        })?;
        let outcome_at = query.outcome_offset();
        let min_offset = treatment_at.min(outcome_at).min(0);
        let max_offset = treatment_at.max(outcome_at).max(0);
        let horizon = u32::try_from(max_offset)
            .map_err(|_| IdentificationError::msg("negative horizon"))?
            .saturating_add(1);
        let variable_count = required_variable_count(template, [query.treatment, query.outcome]);
        // Every template parent of T[treatment_at] sits at most template_max_lag
        // slices earlier, so this window contains all of pa(T).
        let history = min_offset.unsigned_abs().saturating_add(template_max_lag(template));

        let treatment_key = TemporalNodeKey { variable: query.treatment, offset: treatment_at };
        let outcome_key = TemporalNodeKey { variable: query.outcome, offset: outcome_at };
        let indexer = TemporalIndexer::new(variable_count, history, horizon)
            .map_err(|e| IdentificationError::msg(e.to_string()))?;
        let unfolded =
            template.unfold(indexer).map_err(|e| IdentificationError::msg(e.to_string()))?;
        let treatment_dense = unfolded
            .indexer
            .dense_id(treatment_key)
            .map_err(|_| IdentificationError::UnknownVariable { id: query.treatment })?;
        let outcome_dense = unfolded
            .indexer
            .dense_id(outcome_key)
            .map_err(|_| IdentificationError::UnknownVariable { id: query.outcome })?;
        let t = DenseNodeId::from_raw(treatment_dense);
        let y = DenseNodeId::from_raw(outcome_dense);

        let mut parents: Vec<DenseNodeId> = unfolded.dag.parents(t).to_vec();
        parents.sort_unstable();
        if parents.contains(&y) {
            return Err(IdentificationError::NotCertified {
                message: "parent adjustment cannot identify this pulse: the outcome node is a \
                          parent of the treatment",
            });
        }
        let reference = treatment_at.max(outcome_at);
        let mut parent_keys = Vec::with_capacity(parents.len());
        for &parent in &parents {
            let key = unfolded
                .indexer
                .key_of(parent.raw())
                .map_err(|e| IdentificationError::msg(e.to_string()))?;
            let lag = u32::try_from(reference.saturating_sub(key.offset)).unwrap_or(0);
            if query.max_history_lag.is_some_and(|cap| lag > cap) {
                return Err(IdentificationError::NotCertified {
                    message: "temporal unfolding cannot certify backdoor identification, and \
                              parent adjustment cannot stand in: a parent of the treatment lies \
                              beyond max_history_lag, so it is not observed within the permitted \
                              history (raise max_history_lag)",
                });
            }
            parent_keys.push(key);
        }
        let mutilated = crate::backdoor::remove_outgoing(&unfolded.dag, t)?;
        let mut dsep = antecedent_graph::DSeparationWorkspace::default();
        if !crate::backdoor::is_backdoor_adjustment(&mutilated, t, y, &parents, &mut dsep)? {
            return Err(IdentificationError::NotCertified {
                message: "the treatment's parents do not satisfy the back-door criterion on the \
                          unfolded window",
            });
        }

        let treatment_var = VariableId::from_raw(treatment_dense);
        let outcome_var = VariableId::from_raw(outcome_dense);
        let active = retarget(&query.active, treatment_var)?;
        let control = retarget(&query.control, treatment_var)?;
        let ate = AverageEffectQuery::new(
            treatment_var,
            outcome_var,
            Arc::from([]),
            control.clone(),
            active.clone(),
            query.target_population.clone(),
        );
        let active_value = crate::intervention_support::require_set_value(&active, "backdoor")?;
        let control_value = crate::intervention_support::require_set_value(&control, "backdoor")?;
        let adjustment: Vec<VariableId> =
            parents.iter().map(|parent| VariableId::from_raw(parent.raw())).collect();
        let mut arena = antecedent_expr::CausalExprArena::new();
        let functional = arena.backdoor_ate(
            treatment_var,
            outcome_var,
            &adjustment,
            active_value,
            control_value,
        );
        let estimand = crate::result::IdentifiedEstimand::backdoor(
            "backdoor.adjustment",
            Arc::from(adjustment),
            functional,
        );

        let mut assumptions = antecedent_core::AssumptionSet::new();
        assumptions.push(crate::assumptions::causal_markov(PARENT_ADJUSTMENT_RULE));
        assumptions.push(AssumptionRecord {
            assumption: Assumption::CausalSufficiency,
            source: AssumptionSource::AlgorithmDefault {
                algorithm: Arc::from(PARENT_ADJUSTMENT_RULE),
            },
            scope: AssumptionScope::Identification,
            status: AssumptionStatus::Declared,
        });
        assumptions.push(AssumptionRecord {
            assumption: Assumption::Custom {
                id: Arc::from("temporal.parent_adjustment.parents_oriented_observed"),
                description: Arc::from(
                    "every parent of the treatment, lagged and contemporaneous, is a fully \
                     oriented edge of the accepted temporal DAG and is observed at its lag",
                ),
            },
            source: AssumptionSource::AlgorithmDefault {
                algorithm: Arc::from(PARENT_ADJUSTMENT_RULE),
            },
            scope: AssumptionScope::Identification,
            status: AssumptionStatus::Declared,
        });

        let rendered: Vec<String> =
            parent_keys.iter().map(|key| format!("{}@{}", key.variable, key.offset)).collect();
        let mut derivation = crate::result::DerivationTrace::default();
        derivation.push(
            PARENT_ADJUSTMENT_RULE,
            match unfolding_refusal {
                Some(refusal) => format!(
                    "unfolding could not certify ({refusal}); Z = pa(treatment) = \
                     {{{}}} blocks every back-door path in the unrolled temporal DAG",
                    rendered.join(", ")
                ),
                None => format!(
                    "Z = pa(treatment) = {{{}}} blocks every back-door path in the unrolled \
                     temporal DAG",
                    rendered.join(", ")
                ),
            },
        );
        derivation.push("backdoor.adjustment_set", format!("|Z|={}", parents.len()));

        let mut result = IdentificationResult::identified(
            CausalQuery::average_effect(ate),
            vec![estimand],
            arena,
            derivation,
            assumptions,
            crate::result::IdentificationPerformanceRecord {
                candidates_examined: 1,
                sets_returned: 1,
            },
        );
        annotate_temporal(&mut result, query, treatment_key, outcome_key, history, horizon);
        Ok(TemporalIdentificationResult {
            result,
            indexer: unfolded.indexer,
            treatment_key,
            outcome_key,
        })
    }

    /// Identify a joint schedule of `(variable, offset, level)` treatment nodes.
    ///
    /// `level` is the hard `Set` value the caller actually requested at that
    /// step (`None` for a Soft/shift step, which carries no fixed value).
    /// Used by multi-step Sustained / Dynamic and by multi-step / joint Sequence
    /// overlays. The identifier remains `temporal.backdoor.unfolded`.
    ///
    /// The active side of the emitted contrast is the schedule's own requested
    /// level, not a fabricated 0-vs-1 pair: [`resolve_schedule_active_level`]
    /// requires every `Some` level in the schedule to agree, since
    /// [`IdIdentifier::identify_schedule_contrast`] bakes a single literal into
    /// every schedule node.
    ///
    /// # Errors
    ///
    /// In addition to the usual unfolding / identification failures, refuses a
    /// schedule that requests two different `Set` levels at different offsets
    /// (a genuine per-node dose schedule), because the general-ID contrast
    /// cannot express more than one literal active level today.
    pub fn identify_temporal_schedule(
        &self,
        template: &TemporalDag,
        outcome: VariableId,
        outcome_at: i32,
        schedule: &[(VariableId, i32, Option<f64>)],
        max_history_lag: Option<u32>,
        target_population: antecedent_core::TargetPopulation,
    ) -> Result<TemporalIdentificationResult, IdentificationError> {
        if schedule.is_empty() {
            return Err(IdentificationError::msg("empty treatment schedule"));
        }
        let (treatment, first_offset, _) = schedule[0];
        let offsets: Vec<i32> = {
            let mut offsets: Vec<i32> = schedule.iter().map(|&(_, offset, _)| offset).collect();
            offsets.sort_unstable();
            offsets.dedup();
            offsets
        };
        let same_var = schedule.iter().all(|(variable, _, _)| *variable == treatment);
        let contiguous =
            same_var && offsets.len() > 1 && offsets.windows(2).all(|w| w[1] == w[0] + 1);
        let policy = if contiguous {
            TemporalPolicy::sustained(offsets[0], *offsets.last().expect("non-empty"))
        } else if offsets.len() == 1 {
            TemporalPolicy::pulse(offsets[0])
        } else {
            TemporalPolicy::dynamic(antecedent_core::DynamicRuleId::from_raw(0), offsets)
        };
        let active_level = resolve_schedule_active_level(schedule)?;
        let query = TemporalEffectQuery {
            treatment,
            outcome,
            policy,
            control: Intervention::set(treatment, antecedent_core::Value::f64(0.0)),
            active: Intervention::set(treatment, antecedent_core::Value::f64(active_level)),
            horizon_steps: u32::try_from(outcome_at.saturating_add(1))
                .map_err(|_| IdentificationError::msg("outcome offset does not fit horizon"))?,
            max_history_lag,
            target_population,
        };
        let _ = first_offset;
        query.validate().map_err(|_| IdentificationError::UnsupportedQuery {
            message: "invalid temporal schedule query",
        })?;
        let schedule_pairs: Vec<(VariableId, i32)> =
            schedule.iter().map(|&(variable, offset, _)| (variable, offset)).collect();
        self.identify_active_offsets(template, &query, &schedule_pairs, outcome_at)
    }

    /// Multi-time-point interventions (sustained windows or dynamic schedules):
    /// unfold, then identify via general ID (sequential / g-formula).
    fn identify_active_offsets(
        &self,
        template: &TemporalDag,
        query: &TemporalEffectQuery,
        schedule: &[(VariableId, i32)],
        outcome_at: i32,
    ) -> Result<TemporalIdentificationResult, IdentificationError> {
        if schedule.is_empty() {
            return Err(IdentificationError::msg("empty treatment schedule"));
        }
        let from = schedule.iter().map(|&(_, offset)| offset).min().expect("non-empty");
        let until = schedule.iter().map(|&(_, offset)| offset).max().expect("non-empty");
        let min_offset = from.min(outcome_at).min(0);
        let max_offset = until.max(outcome_at).max(0);
        let horizon = u32::try_from(max_offset)
            .map_err(|_| IdentificationError::msg("negative horizon"))?
            .saturating_add(1);

        let mut variables = vec![query.treatment, query.outcome];
        variables.extend(schedule.iter().map(|&(variable, _)| variable));
        let variable_count = required_variable_count(template, variables);
        let max_lag = template_max_lag(template);
        let base_history = min_offset.unsigned_abs().max(max_lag);
        let chain_cap =
            variable_count.saturating_mul(max_lag).saturating_add(min_offset.unsigned_abs());
        let history_cap = query.max_history_lag.unwrap_or(chain_cap).max(base_history);

        let treatment_key = TemporalNodeKey { variable: query.treatment, offset: from };
        let outcome_key = TemporalNodeKey { variable: query.outcome, offset: outcome_at };
        let truncatable = truncatable_variables(template, variable_count);

        let mut history = base_history;
        let (history, unfolded, treatment_nodes, outcome_dense) = loop {
            let indexer = TemporalIndexer::new(variable_count, history, horizon)
                .map_err(|e| IdentificationError::msg(e.to_string()))?;
            let unfolded =
                template.unfold(indexer).map_err(|e| IdentificationError::msg(e.to_string()))?;

            let mut treatment_nodes = Vec::with_capacity(schedule.len());
            for &(variable, offset) in schedule {
                let key = TemporalNodeKey { variable, offset };
                let dense = unfolded
                    .indexer
                    .dense_id(key)
                    .map_err(|_| IdentificationError::UnknownVariable { id: variable })?;
                treatment_nodes.push(dense);
            }
            let outcome_dense = unfolded
                .indexer
                .dense_id(outcome_key)
                .map_err(|_| IdentificationError::UnknownVariable { id: query.outcome })?;

            if treatment_nodes.is_empty() {
                return Err(IdentificationError::msg("empty treatment schedule"));
            }
            let mut boundary_nodes = treatment_nodes.clone();
            boundary_nodes.push(outcome_dense);
            if !ancestry_touches_boundary(&unfolded, &boundary_nodes, history, &truncatable) {
                break (history, unfolded, treatment_nodes, outcome_dense);
            }
            if history >= history_cap {
                return Err(history_cap_refusal(history_cap, chain_cap));
            }
            history += 1;
        };

        // Both sides of the temporal contrast, not just the active level: the
        // historical path applied only `active` to every treatment-time node
        // and relabeled the one-sided interventional distribution as the
        // temporal *effect* — a contrast the emitted functional never encoded.
        let active = match &query.active {
            Intervention::Set { value, .. } => value.clone(),
            _ => {
                return Err(IdentificationError::unsupported(
                    "multi-time temporal ID requires Set interventions",
                ));
            }
        };
        let control = match &query.control {
            Intervention::Set { value, .. } => value.clone(),
            _ => {
                return Err(IdentificationError::unsupported(
                    "multi-time temporal ID requires Set interventions on both contrast sides",
                ));
            }
        };
        let schedule: Vec<VariableId> =
            treatment_nodes.iter().map(|&d| VariableId::from_raw(d)).collect();
        let outcome_var = VariableId::from_raw(outcome_dense);

        let prepared = PreparedAdmg::from_dag(&unfolded.dag)?;
        let id = IdIdentifier::new();
        let mut ws = IdentificationWorkspace::default();
        let mut result = id.identify_schedule_contrast(
            &prepared,
            outcome_var,
            &schedule,
            &active,
            &control,
            CausalQuery::TemporalEffect(query.clone()),
            &mut ws,
        )?;
        result.derivation.push(
            "temporal.schedule",
            format!(
                "sequential / g-formula contrast on unfolded window history={history} \
                 schedule={schedule:?} ({} treatment nodes, both contrast levels)",
                treatment_nodes.len()
            ),
        );
        annotate_temporal(&mut result, query, treatment_key, outcome_key, history, horizon);
        Ok(TemporalIdentificationResult {
            result,
            indexer: unfolded.indexer,
            treatment_key,
            outcome_key,
        })
    }
}

/// The single literal `Set` level to bake into every node of a joint schedule.
///
/// `IdIdentifier::identify_schedule_contrast` assigns one `active_level` to
/// every schedule node (`assign_for` in `id.rs`), so a schedule can only be
/// certified against the *actual* requested regime when every `Some` level it
/// carries agrees. A step with `level = None` (a Soft/shift overlay) supplies
/// no fixed value and is not counted: the estimator downstream reads the
/// overlay's own shift, never this certificate's literal. A schedule with no
/// `Set` step at all keeps the historical `1.0` reference level purely to
/// certify structural identifiability, which is level-free.
///
/// # Errors
///
/// [`IdentificationError::unsupported`] when two schedule steps request
/// different `Set` levels: a genuine per-node dose schedule, which the
/// current single-literal contrast cannot express without silently
/// substituting one side's request for another's.
fn resolve_schedule_active_level(
    schedule: &[(VariableId, i32, Option<f64>)],
) -> Result<f64, IdentificationError> {
    let mut distinct: Vec<f64> = Vec::new();
    for &(_, _, level) in schedule {
        if let Some(value) = level {
            if !distinct.contains(&value) {
                distinct.push(value);
            }
        }
    }
    match distinct.as_slice() {
        [] => Ok(1.0),
        [only] => Ok(*only),
        _ => Err(IdentificationError::unsupported(
            "temporal schedule requests different Set levels at different offsets; \
             a single joint-schedule contrast cannot certify a per-node dose schedule",
        )),
    }
}

/// The refusal when window growth stops at `history_cap` with ancestry still
/// crossing the truncated boundary.
///
/// A confounder chain that repeats no variable reaches at most `chain_cap`
/// slices back (`variable_count * template_max_lag + |min_offset|`). Reaching
/// that depth therefore means an ancestor lies on a lagged cycle, such as an
/// autoregressive edge `x(t-1) -> x(t)`, whose ancestry never ends: no finite
/// window certifies it, and a larger `max_history_lag` cannot help. Only a
/// caller-set `max_history_lag` below `chain_cap` can stop growth early, and
/// then raising it is the remedy.
fn history_cap_refusal(history_cap: u32, chain_cap: u32) -> IdentificationError {
    let message = if history_cap < chain_cap {
        "temporal unfolding reached the query's max_history_lag while confounder ancestry \
         still crossed the truncated boundary; cannot certify backdoor identification over \
         the finite window (raise max_history_lag)"
    } else {
        "temporal unfolding reached its history cap while confounder ancestry still crossed \
         the truncated boundary: an ancestor of the treatment or outcome lies on a lagged \
         cycle (for example an autoregressive edge x(t-1) -> x(t)), so no finite window \
         certifies backdoor identification; review whether that lagged edge belongs in the \
         graph"
    };
    IdentificationError::NotCertified { message }
}

fn template_max_lag(template: &TemporalDag) -> u32 {
    template
        .nodes()
        .iter()
        .filter_map(|n| match n {
            NodeRef::Lagged { lag, .. } => Some(lag.raw()),
            _ => None,
        })
        .max()
        .unwrap_or(0)
}

/// Per-variable flag: `true` when some template edge into that variable spans
/// strictly backwards in time (parent lag greater than child lag), i.e. when
/// an unfolded node of the variable at the deepest window slice would have
/// parents cut off by the truncation.
fn truncatable_variables(template: &TemporalDag, variable_count: u32) -> Vec<bool> {
    let mut truncatable = vec![false; variable_count as usize];
    for edge in template.edges() {
        let Some((from, to)) = edge.parent_child() else {
            continue;
        };
        let (Some(from_key), Some(to_key)) =
            (template.temporal_key(from), template.temporal_key(to))
        else {
            continue;
        };
        if from_key.offset < to_key.offset {
            if let Some(slot) = truncatable.get_mut(to_key.variable.raw() as usize) {
                *slot = true;
            }
        }
    }
    truncatable
}

/// Whether any ancestor of `nodes` in the unfolded graph sits at the deepest
/// slice (`offset == -history`) with in-template parents that the truncation
/// cut off. When this returns `false`, growing the window further cannot add
/// backdoor paths through those nodes.
///
/// Callers must pass **every** treatment-time node in a multi-time schedule
/// plus the outcome. Checking only the first offset can stop growth while a
/// later treatment time still has cut template parents.
fn ancestry_touches_boundary(
    unfolded: &UnfoldedTemporalGraph,
    nodes: &[u32],
    history: u32,
    truncatable: &[bool],
) -> bool {
    let dag = &unfolded.dag;
    let mut ancestors = BitSet::with_len(dag.node_count());
    let mut gws = GraphWorkspace::default();
    let dense: Vec<DenseNodeId> = nodes.iter().copied().map(DenseNodeId::from_raw).collect();
    dag.ancestors_of(&dense, &mut ancestors, &mut gws);
    let boundary = -i64::from(history);
    for i in 0..dag.node_count() {
        let id = DenseNodeId::from_raw(u32::try_from(i).expect("fit"));
        if !ancestors.contains(id) {
            continue;
        }
        let Ok(key) = unfolded.indexer.key_of(id.raw()) else {
            continue;
        };
        if i64::from(key.offset) == boundary
            && truncatable.get(key.variable.raw() as usize).copied().unwrap_or(false)
        {
            return true;
        }
    }
    false
}

/// Populate [`AdjustmentSearchConfig`] history-lag filter from an unfolded indexer.
///
/// Lag for dense node `i` is `max(0, reference_offset - node.offset)`. When
/// `max_history_lag` is set, covariates older than that many steps are excluded
/// from static backdoor enumeration on the unfolded DAG.
///
/// `offset` is the node's signed absolute temporal position (more negative = further past,
/// `reference_offset` = contemporaneous with the query); `lag` is the derived non-negative
/// "steps into the past relative to `reference_offset`". Any node with `key.offset >
/// reference_offset` — i.e. one that sits in the *future* relative to the reference point —
/// produces a negative `reference_offset - key.offset` that `.max(0)` collapses to `lag = 0`.
/// Since `max_history_lag` only excludes lags *greater* than the cap, such a node is
/// indistinguishable from a genuinely-contemporaneous (`lag = 0`) one and can never be
/// excluded by the history cap, no matter how far "in the future" it actually sits. This is
/// not a backdoor-criterion bug — Pearl's criterion is graph-theoretic and does not require Z
/// to (temporally) precede X — but callers relying on `max_history_lag` to bound *all*
/// covariates by recency should be aware that future-offset covariates are never subject to
/// this cap.
fn apply_history_lag_filter(
    config: &mut crate::backdoor::AdjustmentSearchConfig,
    indexer: &TemporalIndexer,
    reference_offset: i32,
    max_history_lag: Option<u32>,
) {
    config.max_history_lag = max_history_lag;
    let mut lags = Vec::with_capacity(indexer.dense_len());
    #[allow(
        clippy::cast_possible_truncation,
        reason = "the indexer addresses its dense nodes with u32 keys, so its length fits u32"
    )]
    let dense_len = indexer.dense_len() as u32;
    for dense in 0..dense_len {
        let Ok(key) = indexer.key_of(dense) else {
            continue;
        };
        let lag = u32::try_from(reference_offset.saturating_sub(key.offset).max(0)).unwrap_or(0);
        lags.push((VariableId::from_raw(dense), lag));
    }
    config.history_lags = Arc::from(lags);
}

fn required_variable_count(
    template: &TemporalDag,
    variables: impl IntoIterator<Item = VariableId>,
) -> u32 {
    let mut max_id = 0;
    for variable in variables {
        max_id = max_id.max(variable.raw());
    }
    for node in template.nodes() {
        if let NodeRef::Lagged { variable, .. } = node {
            max_id = max_id.max(variable.raw());
        }
    }
    max_id.saturating_add(1)
}

pub(crate) fn retarget(
    intervention: &Intervention,
    variable: VariableId,
) -> Result<Intervention, IdentificationError> {
    match intervention {
        Intervention::Set { value, .. } => Ok(Intervention::set(variable, value.clone())),
        _ => Err(IdentificationError::UnsupportedQuery {
            message: "temporal backdoor requires Set interventions",
        }),
    }
}

pub(crate) fn annotate_temporal(
    result: &mut IdentificationResult,
    query: &TemporalEffectQuery,
    treatment_key: TemporalNodeKey,
    outcome_key: TemporalNodeKey,
    history: u32,
    horizon: u32,
) {
    result.required_assumptions.push(AssumptionRecord {
        assumption: Assumption::Stationarity,
        source: AssumptionSource::AlgorithmDefault {
            algorithm: Arc::from("temporal.backdoor.unfolded"),
        },
        scope: AssumptionScope::Identification,
        status: AssumptionStatus::Declared,
    });
    let treatment = query.treatment;
    let outcome = query.outcome;
    let t_offset = treatment_key.offset;
    let o_offset = outcome_key.offset;
    result.derivation.push(
        "temporal.unfold",
        format!(
            "finite window history={history} horizon={horizon}; \
             treatment={treatment}@{t_offset} outcome={outcome}@{o_offset}"
        ),
    );
    for e in &mut result.estimands {
        e.method = Arc::from("temporal.backdoor.unfolded");
    }
}

#[cfg(test)]
mod tests {
    use antecedent_core::Lag;

    use super::*;
    use crate::result::IdentificationStatus;

    #[test]
    fn chain_identifies_with_empty_adjustment() {
        // Template: X_{t-1} -> Y_t (no confounding).
        let mut template = TemporalDag::empty();
        let x = template.add_lagged(VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
        let y = template.add_lagged(VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
        template.insert_directed(x, y).unwrap();

        let query =
            TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0)
                .with_policy(TemporalPolicy::pulse(-1))
                .with_horizon_steps(1);

        let identifier = TemporalBackdoorIdentifier::new();
        let temporal_result = identifier.identify_temporal(&template, &query).unwrap();
        assert_eq!(
            temporal_result.result.status,
            IdentificationStatus::NonparametricallyIdentified
        );
        assert!(temporal_result.result.estimands[0].adjustment_set.is_empty());
        assert!(
            temporal_result
                .result
                .required_assumptions
                .entries
                .iter()
                .any(|a| a.assumption == Assumption::Stationarity)
        );
    }

    #[test]
    fn autoregressive_ancestry_refusal_names_the_lagged_cycle() {
        // Template: X_{t-1} -> X_t, X_{t-1} -> Y_t. The treatment's ancestry is
        // an unbounded autoregressive chain, so no window certifies it and the
        // refusal must not advise a max_history_lag that cannot help.
        let mut template = TemporalDag::empty();
        let x_lag = template.add_lagged(VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
        let x_now = template.add_lagged(VariableId::from_raw(0), Lag::CONTEMPORANEOUS).unwrap();
        let y = template.add_lagged(VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
        template.insert_directed(x_lag, x_now).unwrap();
        template.insert_directed(x_lag, y).unwrap();

        let query =
            TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0)
                .with_policy(TemporalPolicy::pulse(-1))
                .with_horizon_steps(1);
        for query in [query.clone(), query.with_max_history_lag(Some(50))] {
            let Err(IdentificationError::NotCertified { message }) =
                TemporalBackdoorIdentifier::new().identify_temporal(&template, &query)
            else {
                panic!("an autoregressive treatment chain must not certify");
            };
            assert!(message.contains("lagged cycle"), "{message}");
            assert!(message.contains("history cap"), "{message}");
            assert!(!message.contains("max_history_lag"), "{message}");
        }
    }

    /// `T@1 -> T@0` (autoregressive), `Z@1 -> T@0`, `Z@2 -> Y@0`, `T@1 -> Y@0`,
    /// with variables `T = 0`, `Y = 1`, `Z = 2`. For the pulse on `T[-1]` and
    /// `Y[0]`, `Z[-2]` confounds, and the treatment's ancestry is unbounded.
    fn autoregressive_confounded_template() -> TemporalDag {
        let mut template = TemporalDag::empty();
        let t_lag = template.add_lagged(VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
        let t_now = template.add_lagged(VariableId::from_raw(0), Lag::CONTEMPORANEOUS).unwrap();
        let y = template.add_lagged(VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
        let z_lag = template.add_lagged(VariableId::from_raw(2), Lag::from_raw(1)).unwrap();
        let z_lag2 = template.add_lagged(VariableId::from_raw(2), Lag::from_raw(2)).unwrap();
        template.insert_directed(t_lag, t_now).unwrap();
        template.insert_directed(z_lag, t_now).unwrap();
        template.insert_directed(z_lag2, y).unwrap();
        template.insert_directed(t_lag, y).unwrap();
        template
    }

    fn lag_one_pulse() -> TemporalEffectQuery {
        TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0)
            .with_policy(TemporalPolicy::pulse(-1))
            .with_horizon_steps(1)
    }

    fn adjustment_keys(identified: &TemporalIdentificationResult) -> Vec<(u32, i32)> {
        let mut keys: Vec<(u32, i32)> = identified.result.estimands[0]
            .adjustment_set
            .iter()
            .map(|dense| {
                let key = identified.indexer.key_of(dense.raw()).unwrap();
                (key.variable.raw(), key.offset)
            })
            .collect();
        keys.sort_unstable();
        keys
    }

    #[test]
    fn autoregressive_pulse_is_identified_by_parent_adjustment() {
        let template = autoregressive_confounded_template();
        let query = lag_one_pulse();
        assert!(matches!(
            TemporalBackdoorIdentifier::new().identify_temporal(&template, &query),
            Err(IdentificationError::NotCertified { .. })
        ));

        let identified = TemporalBackdoorIdentifier::new()
            .with_parent_adjustment_fallback()
            .identify_temporal(&template, &query)
            .expect("parent adjustment identifies the single-step pulse");
        let result = &identified.result;
        assert_eq!(result.status, IdentificationStatus::NonparametricallyIdentified);
        assert_eq!(result.estimands.len(), 1);
        assert_eq!(result.estimands[0].method.as_ref(), "temporal.backdoor.unfolded");
        // pa(T[-1]) = {T[-2], Z[-2]}.
        assert_eq!(adjustment_keys(&identified), [(0, -2), (2, -2)]);
        let rules: Vec<&str> = result.derivation.steps.iter().map(|s| s.rule.as_ref()).collect();
        assert!(rules.contains(&PARENT_ADJUSTMENT_RULE), "{rules:?}");
        assert!(!rules.contains(&"backdoor.criterion"), "{rules:?}");
        let step =
            result.derivation.steps.iter().find(|s| s.rule.as_ref() == PARENT_ADJUSTMENT_RULE);
        assert!(step.unwrap().detail.contains("lagged cycle"));
        assert!(
            result
                .required_assumptions
                .entries
                .iter()
                .any(|record| matches!(record.assumption, Assumption::CausalSufficiency))
        );
        assert!(result.required_assumptions.entries.iter().any(|record| matches!(
            &record.assumption,
            Assumption::Custom { id, .. }
                if id.as_ref() == "temporal.parent_adjustment.parents_oriented_observed"
        )));
    }

    #[test]
    fn parent_adjustment_refuses_a_parent_beyond_max_history_lag() {
        let template = autoregressive_confounded_template();
        // Both parents sit two slices before the outcome; a one-slice cap does
        // not observe them.
        let query = lag_one_pulse().with_max_history_lag(Some(1));
        let Err(IdentificationError::NotCertified { message }) = TemporalBackdoorIdentifier::new()
            .with_parent_adjustment_fallback()
            .identify_temporal(&template, &query)
        else {
            panic!("a parent outside the permitted history must not be adjusted for");
        };
        assert!(message.contains("beyond max_history_lag"), "{message}");
        // At the parents' own depth the same query identifies.
        TemporalBackdoorIdentifier::new()
            .with_parent_adjustment_fallback()
            .identify_temporal(&template, &lag_one_pulse().with_max_history_lag(Some(2)))
            .expect("parents inside the cap");
    }

    #[test]
    fn parent_adjustment_does_not_reach_multi_step_schedules() {
        let template = autoregressive_confounded_template();
        let sustained = TemporalEffectQuery::sustained(
            VariableId::from_raw(0),
            VariableId::from_raw(1),
            -2,
            1.0,
        )
        .with_policy(TemporalPolicy::sustained(-2, -1))
        .with_horizon_steps(1);
        let identifier = TemporalBackdoorIdentifier::new().with_parent_adjustment_fallback();
        assert!(matches!(
            identifier.identify_temporal(&template, &sustained),
            Err(IdentificationError::NotCertified { .. })
        ));
        assert!(matches!(
            identifier.identify_pulse_by_parent_adjustment(&template, &sustained),
            Err(IdentificationError::UnsupportedQuery { .. })
        ));
        // A single-step Sustained is not a Pulse query and keeps the refusal.
        let single = TemporalEffectQuery::sustained(
            VariableId::from_raw(0),
            VariableId::from_raw(1),
            -1,
            1.0,
        )
        .with_policy(TemporalPolicy::sustained(-1, -1))
        .with_horizon_steps(1);
        assert!(matches!(
            identifier.identify_temporal(&template, &single),
            Err(IdentificationError::NotCertified { .. })
        ));
    }

    #[test]
    fn parent_adjustment_on_a_certifiable_graph_names_the_parents() {
        // Z_{t-1} -> X_t, X_{t-1} -> Y_t, Z_{t-2} -> Y_t: unfolding certifies,
        // and the direct parent-adjustment derivation adjusts pa(X[-1]) = {Z[-2]}.
        let mut template = TemporalDag::empty();
        let z1 = template.add_lagged(VariableId::from_raw(2), Lag::from_raw(1)).unwrap();
        let z2 = template.add_lagged(VariableId::from_raw(2), Lag::from_raw(2)).unwrap();
        let x1 = template.add_lagged(VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
        let x0 = template.add_lagged(VariableId::from_raw(0), Lag::CONTEMPORANEOUS).unwrap();
        let y = template.add_lagged(VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
        template.insert_directed(z1, x0).unwrap();
        template.insert_directed(x1, y).unwrap();
        template.insert_directed(z2, y).unwrap();
        let identifier = TemporalBackdoorIdentifier::new().with_parent_adjustment_fallback();
        let unfolded = identifier.identify_temporal(&template, &lag_one_pulse()).unwrap();
        let rules: Vec<&str> =
            unfolded.result.derivation.steps.iter().map(|s| s.rule.as_ref()).collect();
        assert!(!rules.contains(&PARENT_ADJUSTMENT_RULE), "unfolding certified: {rules:?}");
        let parents =
            identifier.identify_pulse_by_parent_adjustment(&template, &lag_one_pulse()).unwrap();
        assert_eq!(adjustment_keys(&parents), [(2, -2)]);
    }

    #[test]
    fn caller_history_cap_below_the_chain_bound_names_max_history_lag() {
        // Template: W_{t-1} -> Z_t, Z_{t-1} -> X_t, X_{t-1} -> Y_t. The chain is
        // finite (three slices deep) and certifies under the default cap; a
        // caller cap of one slice stops growth early, and only then is raising
        // max_history_lag the remedy.
        let mut template = TemporalDag::empty();
        let w = template.add_lagged(VariableId::from_raw(3), Lag::from_raw(1)).unwrap();
        let z_lag = template.add_lagged(VariableId::from_raw(2), Lag::from_raw(1)).unwrap();
        let z_now = template.add_lagged(VariableId::from_raw(2), Lag::CONTEMPORANEOUS).unwrap();
        let x_lag = template.add_lagged(VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
        let x_now = template.add_lagged(VariableId::from_raw(0), Lag::CONTEMPORANEOUS).unwrap();
        let y = template.add_lagged(VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
        template.insert_directed(w, z_now).unwrap();
        template.insert_directed(z_lag, x_now).unwrap();
        template.insert_directed(x_lag, y).unwrap();

        let query =
            TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0)
                .with_policy(TemporalPolicy::pulse(-1))
                .with_horizon_steps(1);
        let identifier = TemporalBackdoorIdentifier::new();
        identifier.identify_temporal(&template, &query).expect("finite chain certifies");
        let Err(IdentificationError::NotCertified { message }) =
            identifier.identify_temporal(&template, &query.with_max_history_lag(Some(1)))
        else {
            panic!("a one-slice cap must stop growth before the chain ends");
        };
        assert!(message.contains("raise max_history_lag"), "{message}");
        assert!(!message.contains("lagged cycle"), "{message}");
    }

    #[test]
    fn confounded_chain_requires_lagged_confounder() {
        // Template: Z_{t-1} -> X_{t-1}, Z_{t-1} -> Y_t, X_{t-1} -> Y_t.
        let mut template = TemporalDag::empty();
        let z = template.add_lagged(VariableId::from_raw(2), Lag::from_raw(1)).unwrap();
        let x = template.add_lagged(VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
        let y = template.add_lagged(VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
        template.insert_directed(z, x).unwrap();
        template.insert_directed(z, y).unwrap();
        template.insert_directed(x, y).unwrap();

        let query =
            TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0)
                .with_policy(TemporalPolicy::pulse(-1))
                .with_horizon_steps(1);

        let identifier = TemporalBackdoorIdentifier::new();
        let temporal_result = identifier.identify_temporal(&template, &query).unwrap();
        assert_eq!(
            temporal_result.result.status,
            IdentificationStatus::NonparametricallyIdentified
        );
        let z_key = TemporalNodeKey { variable: VariableId::from_raw(2), offset: -1 };
        let z_dense = temporal_result.indexer.dense_id(z_key).unwrap();
        assert_eq!(
            temporal_result.result.estimands[0].adjustment_set.as_ref(),
            &[VariableId::from_raw(z_dense)]
        );
    }

    #[test]
    fn single_step_sustained_and_dynamic_reuse_pulse_backdoor_set() {
        // Same confounded template as above. Licensed single-step Sustained
        // (and a one-offset Dynamic) must not relabel General ID as
        // temporal.backdoor.unfolded with Z = ∅.
        let mut template = TemporalDag::empty();
        let z = template.add_lagged(VariableId::from_raw(2), Lag::from_raw(1)).unwrap();
        let x = template.add_lagged(VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
        let y = template.add_lagged(VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
        template.insert_directed(z, x).unwrap();
        template.insert_directed(z, y).unwrap();
        template.insert_directed(x, y).unwrap();

        let pulse =
            TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0)
                .with_policy(TemporalPolicy::pulse(-1))
                .with_horizon_steps(1);
        let sustained = pulse.clone().with_policy(TemporalPolicy::sustained(-1, -1));
        let dynamic = pulse.clone().with_policy(TemporalPolicy::dynamic(
            antecedent_core::DynamicRuleId::from_raw(0),
            [-1],
        ));

        let identifier = TemporalBackdoorIdentifier::new();
        let pulse_id = identifier.identify_temporal(&template, &pulse).unwrap();
        let z_key = TemporalNodeKey { variable: VariableId::from_raw(2), offset: -1 };
        let z_dense = VariableId::from_raw(pulse_id.indexer.dense_id(z_key).unwrap());
        assert_eq!(pulse_id.result.estimands[0].adjustment_set.as_ref(), &[z_dense]);

        for query in [&sustained, &dynamic] {
            let got = identifier.identify_temporal(&template, query).unwrap();
            assert_eq!(got.result.estimands[0].adjustment_set.as_ref(), &[z_dense]);
            assert_eq!(got.result.estimands[0].method.as_ref(), "temporal.backdoor.unfolded");
        }
    }

    /// Template with all lag-1 edges `B->A`, `A->T`, `B->C`, `C->Y`: the true
    /// backdoor path `T_0 <- A_{-1} <- B_{-2} -> C_{-1} -> Y_0` needs history
    /// 2, one more than the template's single-edge max lag.
    fn deep_confounder_template() -> TemporalDag {
        let mut template = TemporalDag::empty();
        let t_var = VariableId::from_raw(0);
        let y_var = VariableId::from_raw(1);
        let a_var = VariableId::from_raw(2);
        let b_var = VariableId::from_raw(3);
        let c_var = VariableId::from_raw(4);
        let a_lag = template.add_lagged(a_var, Lag::from_raw(1)).unwrap();
        let b_lag = template.add_lagged(b_var, Lag::from_raw(1)).unwrap();
        let c_lag = template.add_lagged(c_var, Lag::from_raw(1)).unwrap();
        let a_now = template.add_lagged(a_var, Lag::CONTEMPORANEOUS).unwrap();
        let c_now = template.add_lagged(c_var, Lag::CONTEMPORANEOUS).unwrap();
        let t_now = template.add_lagged(t_var, Lag::CONTEMPORANEOUS).unwrap();
        let y_now = template.add_lagged(y_var, Lag::CONTEMPORANEOUS).unwrap();
        template.insert_directed(b_lag, a_now).unwrap();
        template.insert_directed(a_lag, t_now).unwrap();
        template.insert_directed(b_lag, c_now).unwrap();
        template.insert_directed(c_lag, y_now).unwrap();
        template
    }

    #[test]
    fn deep_confounder_chain_grows_window_and_adjusts() {
        let template = deep_confounder_template();
        let query =
            TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0);

        let identifier = TemporalBackdoorIdentifier::new();
        let temporal_result = identifier.identify_temporal(&template, &query).unwrap();
        assert_eq!(
            temporal_result.result.status,
            IdentificationStatus::NonparametricallyIdentified
        );
        // The window must have grown to history 2 to expose B_{-2}.
        assert!(
            temporal_result.result.derivation.steps.iter().any(|s| s.detail.contains("history=2"))
        );
        // The confounding must not vanish: no empty adjustment set, and the
        // minimal blockers are exactly A_{-1}, B_{-2}, and C_{-1}.
        assert!(!temporal_result.result.estimands.is_empty());
        let dense = |var: u32, offset: i32| {
            let key = TemporalNodeKey { variable: VariableId::from_raw(var), offset };
            VariableId::from_raw(temporal_result.indexer.dense_id(key).unwrap())
        };
        let expected = [vec![dense(2, -1)], vec![dense(3, -2)], vec![dense(4, -1)]];
        for estimand in &temporal_result.result.estimands {
            assert!(!estimand.adjustment_set.is_empty());
            assert!(expected.iter().any(|e| e.as_slice() == estimand.adjustment_set.as_ref()));
        }
    }

    #[test]
    fn deep_confounder_chain_refuses_at_capped_history() {
        let template = deep_confounder_template();
        let query =
            TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0)
                .with_max_history_lag(Some(1));

        let identifier = TemporalBackdoorIdentifier::new();
        assert!(matches!(
            identifier.identify_temporal(&template, &query),
            Err(IdentificationError::NotCertified { .. })
        ));
    }

    #[test]
    fn sustained_policy_identifies_on_simple_chain() {
        let mut template = TemporalDag::empty();
        let x = template.add_lagged(VariableId::from_raw(0), Lag::CONTEMPORANEOUS).unwrap();
        let y = template.add_lagged(VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
        template.insert_directed(x, y).unwrap();
        // Multi-step window: sequential / g-formula path (`temporal.schedule`).
        // Single-step Sustained is covered by
        // `single_step_sustained_and_dynamic_reuse_pulse_backdoor_set`.
        let query = TemporalEffectQuery::sustained(
            VariableId::from_raw(0),
            VariableId::from_raw(1),
            1,
            1.0,
        );
        let identifier = TemporalBackdoorIdentifier::new();
        let res = identifier.identify_temporal(&template, &query).unwrap();
        assert_eq!(res.result.status, IdentificationStatus::NonparametricallyIdentified);
        assert!(res.result.derivation.steps.iter().any(|s| s.rule.as_ref() == "temporal.schedule"));
    }

    /// P1 regression: the sustained/dynamic path identified only the *active*
    /// level and relabeled the one-sided distribution as a temporal effect.
    /// The functional must be the two-sided contrast: evaluated on a provider
    /// with E[Y|do(T=1)] = 0.9 and E[Y|do(T=0)] = 0.2 it returns 0.7 — the
    /// pre-fix emission had no contrast (and no expectation) to evaluate.
    #[test]
    fn sustained_contrast_encodes_both_levels_numerically() {
        use antecedent_core::Value;
        use antecedent_expr::{
            Assignment, DomainRef, EmpiricalTableProvider, EvalContext, FactorSpec,
            InterventionAssignment,
        };

        let mut template = TemporalDag::empty();
        let x = template.add_lagged(VariableId::from_raw(0), Lag::CONTEMPORANEOUS).unwrap();
        let y = template.add_lagged(VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
        template.insert_directed(x, y).unwrap();
        let query = TemporalEffectQuery::sustained(
            VariableId::from_raw(0),
            VariableId::from_raw(1),
            1,
            1.0,
        );
        let identifier = TemporalBackdoorIdentifier::new();
        let res = identifier.identify_temporal(&template, &query).unwrap();
        assert_eq!(res.result.status, IdentificationStatus::NonparametricallyIdentified);
        assert!(
            res.result
                .derivation
                .steps
                .iter()
                .any(|s| s.detail.as_ref().contains("both contrast levels")),
            "derivation must record the two-sided schedule contrast"
        );

        // Synthetic unfolded ids for T@0 and the outcome node.
        let t0 = VariableId::from_raw(
            res.indexer
                .dense_id(TemporalNodeKey { variable: VariableId::from_raw(0), offset: 0 })
                .unwrap(),
        );
        let y_out = VariableId::from_raw(res.indexer.dense_id(res.outcome_key).unwrap());

        let mut p = EmpiricalTableProvider::new();
        p.set_domain(t0, [Value::f64(0.0), Value::f64(1.0)]);
        p.set_domain(y_out, [Value::f64(0.0), Value::f64(1.0)]);
        for (tlev, p_y1) in [(1.0, 0.9), (0.0, 0.2)] {
            let interv = [InterventionAssignment { variable: t0, value: Value::f64(tlev) }];
            for (yval, prob) in [(1.0, p_y1), (0.0, 1.0 - p_y1)] {
                let spec = FactorSpec {
                    variables: &[y_out],
                    conditioned_on: &[t0],
                    intervention: &interv,
                    domain: DomainRef::Interventional,
                    population: "",
                    regime: None,
                };
                let assign =
                    Assignment::from_pairs([(y_out, Value::f64(yval)), (t0, Value::f64(tlev))]);
                p.insert_probability(&spec, &assign, prob).unwrap();
            }
        }
        let est = res.result.estimands.first().expect("schedule estimand");
        let ate = res
            .result
            .arena
            .compile(est.functional)
            .unwrap()
            .evaluate(&res.result.arena, &p, &EvalContext::default())
            .unwrap();
        assert!(
            (ate - 0.7).abs() < 1e-12,
            "sustained contrast must encode both levels: got {ate}, expected 0.7"
        );
    }

    #[test]
    fn dynamic_policy_identifies_like_schedule() {
        use antecedent_core::{DynamicRuleId, TemporalPolicy};
        let mut template = TemporalDag::empty();
        let x = template.add_lagged(VariableId::from_raw(0), Lag::CONTEMPORANEOUS).unwrap();
        let y = template.add_lagged(VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
        template.insert_directed(x, y).unwrap();
        let query =
            TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0)
                .with_policy(TemporalPolicy::dynamic(DynamicRuleId::from_raw(1), [0, 1]));
        let identifier = TemporalBackdoorIdentifier::new();
        let res = identifier.identify_temporal(&template, &query).unwrap();
        assert_eq!(res.result.status, IdentificationStatus::NonparametricallyIdentified);
        assert!(res.result.derivation.steps.iter().any(|s| s.rule.as_ref() == "temporal.schedule"));
    }

    #[test]
    fn dynamic_schedule_grows_for_every_treatment_offset() {
        use antecedent_core::{DynamicRuleId, TemporalPolicy};
        // offsets[0] is the *later* time; the deep chain is into T_0. Checking only
        // treatment_nodes[0] would under-grow; the full schedule must still hit history=2.
        let template = deep_confounder_template();
        let query =
            TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0)
                .with_policy(TemporalPolicy::dynamic(DynamicRuleId::from_raw(1), [1, 0]));
        let identifier = TemporalBackdoorIdentifier::new();
        let temporal_result = identifier.identify_temporal(&template, &query).unwrap();
        assert_eq!(
            temporal_result.result.status,
            IdentificationStatus::NonparametricallyIdentified
        );
        assert!(
            temporal_result.result.derivation.steps.iter().any(|s| s.detail.contains("history=2")),
            "multi-time ID must grow for every treatment offset, not only offsets[0]"
        );
    }

    #[test]
    fn dynamic_schedule_refuses_at_capped_history() {
        use antecedent_core::{DynamicRuleId, TemporalPolicy};
        let template = deep_confounder_template();
        let query =
            TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0)
                .with_policy(TemporalPolicy::dynamic(DynamicRuleId::from_raw(1), [1, 0]))
                .with_max_history_lag(Some(1));
        let identifier = TemporalBackdoorIdentifier::new();
        assert!(matches!(
            identifier.identify_temporal(&template, &query),
            Err(IdentificationError::NotCertified { .. })
        ));
    }

    /// A joint schedule sustained at a non-0/1 dose must be certified as a
    /// contrast against *that* dose, not against a fabricated 1.0. Before the
    /// fix, `identify_temporal_schedule` hard-coded `active = 1.0` regardless
    /// of the level every schedule step actually requested.
    #[test]
    fn schedule_identifies_the_requested_dose_not_a_fabricated_one() {
        let mut template = TemporalDag::empty();
        let x = template.add_lagged(VariableId::from_raw(0), Lag::CONTEMPORANEOUS).unwrap();
        let y = template.add_lagged(VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
        template.insert_directed(x, y).unwrap();

        let schedule =
            [(VariableId::from_raw(0), -1, Some(2.5)), (VariableId::from_raw(0), 0, Some(2.5))];
        let identifier = TemporalBackdoorIdentifier::new();
        let identified = identifier
            .identify_temporal_schedule(
                &template,
                VariableId::from_raw(1),
                0,
                &schedule,
                None,
                antecedent_core::TargetPopulation::AllObserved,
            )
            .unwrap();
        let CausalQuery::TemporalEffect(recorded) = &identified.result.query else {
            panic!("schedule contrast must record a TemporalEffectQuery");
        };
        assert_eq!(
            recorded.active,
            Intervention::set(VariableId::from_raw(0), antecedent_core::Value::f64(2.5)),
            "the certified active arm must be the dose the schedule actually requested"
        );
        assert_eq!(
            recorded.control,
            Intervention::set(VariableId::from_raw(0), antecedent_core::Value::f64(0.0))
        );
    }

    /// A schedule that requests two different `Set` levels at different
    /// offsets (a genuine per-node dose schedule) cannot be certified by a
    /// single-literal contrast; refusing is honest, silently picking one
    /// side's number is not.
    #[test]
    fn schedule_refuses_mixed_dose_levels_instead_of_picking_one() {
        let mut template = TemporalDag::empty();
        let x = template.add_lagged(VariableId::from_raw(0), Lag::CONTEMPORANEOUS).unwrap();
        let y = template.add_lagged(VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
        template.insert_directed(x, y).unwrap();

        let schedule =
            [(VariableId::from_raw(0), -1, Some(2.0)), (VariableId::from_raw(0), 0, Some(5.0))];
        let identifier = TemporalBackdoorIdentifier::new();
        let err = identifier
            .identify_temporal_schedule(
                &template,
                VariableId::from_raw(1),
                0,
                &schedule,
                None,
                antecedent_core::TargetPopulation::AllObserved,
            )
            .unwrap_err();
        assert!(
            matches!(err, IdentificationError::UnsupportedQuery { .. }),
            "mixed-level schedules must be refused, not silently identified: {err:?}"
        );
    }
}
