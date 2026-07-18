//! Temporal backdoor identification over finite unfolded graphs.
//!
//! A stationary [`TemporalDag`] template is materialised into a static [`Dag`]
//! over a finite window (history/horizon), the treatment and outcome are
//! mapped to their dense unfolded nodes, and the existing [`BackdoorIdentifier`]
//! runs unchanged on that static graph. Finiteness and stationarity of the
//! template become declared assumptions on the result.
//!
//! The history depth is not fixed by the template's single-edge maximum lag:
//! backdoor paths can chain template edges arbitrarily far into the past, so
//! the window is grown iteratively until no ancestor of the treatment or
//! outcome sits in the deepest time slice with in-template parents that were
//! truncated away. Once the ancestral closure of `{treatment, outcome}` no
//! longer touches the truncated boundary, enlarging the window further cannot
//! add backdoor paths, so identification over the finite window is exact. If
//! the growth cap (the query's `max_history_lag` when set, otherwise
//! `variable_count * template_max_lag + |min_offset|`) is reached while the
//! ancestral closure still touches the boundary, identification is refused
//! rather than reported vacuously clean.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use causal_core::{
    Assumption, AssumptionRecord, AssumptionScope, AssumptionSource, AssumptionStatus,
    AverageEffectQuery, CausalQuery, Intervention, TemporalEffectQuery, TemporalPolicy, VariableId,
};
use causal_data::{TemporalIndexer, TemporalNodeKey};
use causal_graph::{
    BitSet, DenseNodeId, GraphWorkspace, NodeRef, TemporalDag, UnfoldedTemporalGraph,
};

use crate::identifier::IdentificationWorkspace;
use crate::backdoor::BackdoorIdentifier;
use crate::error::IdentificationError;
use crate::result::IdentificationResult;

/// Identifies [`TemporalEffectQuery`]s via backdoor adjustment over a finite
/// unfolding of a stationary [`TemporalDag`] template.
#[derive(Clone, Debug, Default)]
pub struct TemporalBackdoorIdentifier {
    /// Static backdoor identifier applied to the unfolded graph.
    pub inner: BackdoorIdentifier,
}

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
    /// Invalid query, unfolding failures, sustained policies (not yet
    /// supported; they require sequential/g-formula identification rather
    /// than a single-node backdoor criterion), backdoor identification
    /// errors, or [`IdentificationError::NotIdentified`] when the history
    /// cap is reached while confounder ancestry still crosses the truncated
    /// boundary (a clean result cannot be certified).
    pub fn identify_temporal(
        &self,
        template: &TemporalDag,
        query: &TemporalEffectQuery,
    ) -> Result<TemporalIdentificationResult, IdentificationError> {
        query.validate().map_err(|_| IdentificationError::UnsupportedQuery {
            message: "invalid temporal-effect query",
        })?;
        let treatment_at = query.treatment_offset();
        let outcome_at = query.outcome_offset();
        // Pulse-only for single-node backdoor.
        if matches!(query.policy, TemporalPolicy::Sustained { .. }) {
            return Err(IdentificationError::SustainedPolicyUnsupported);
        }
        if !matches!(query.policy, TemporalPolicy::Pulse { .. }) {
            return Err(IdentificationError::UnsupportedQuery {
                message: "unsupported temporal policy for backdoor identification",
            });
        }

        let min_offset = treatment_at.min(outcome_at).min(0);
        let max_offset = treatment_at.max(outcome_at).max(0);
        let horizon = u32::try_from(max_offset)
            .map_err(|_| IdentificationError::msg("negative horizon"))?
            .saturating_add(1);

        let variable_count = required_variable_count(template, query.treatment, query.outcome);
        let max_lag = template_max_lag(template);
        let base_history = min_offset.unsigned_abs().max(max_lag);
        // The user's max_history_lag, when set, caps window growth; otherwise
        // bound simple confounder chains through every template variable.
        let history_cap = query
            .max_history_lag
            .unwrap_or_else(|| {
                variable_count.saturating_mul(max_lag).saturating_add(min_offset.unsigned_abs())
            })
            .max(base_history);

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
                treatment_dense,
                outcome_dense,
                history,
                &truncatable,
            ) {
                break (history, unfolded, treatment_dense, outcome_dense);
            }
            if history >= history_cap {
                return Err(IdentificationError::NotIdentified {
                    message: "temporal unfolding reached its history cap while confounder \
                              ancestry still crossed the truncated boundary; cannot certify \
                              backdoor identification over the finite window (raise \
                              max_history_lag or shorten confounder chains)",
                });
            }
            history += 1;
        };

        // Unfolded DAGs are built via `Dag::with_variables`, so dense node i is labeled
        // `VariableId::from_raw(i)`. These synthetic ids are only for the unfolded graph;
        // `annotate_temporal` remaps results back to `TemporalNodeKey`s.
        let treatment_var = VariableId::from_raw(treatment_dense);
        let outcome_var = VariableId::from_raw(outcome_dense);

        let ate = AverageEffectQuery {
            treatment: treatment_var,
            outcome: outcome_var,
            effect_modifiers: Arc::from([]),
            control: retarget(&query.control, treatment_var)?,
            active: retarget(&query.active, treatment_var)?,
            target_population: query.target_population.clone(),
        };

        let prepared = self.inner.prepare(&unfolded.dag)?;
        let mut id_ws = IdentificationWorkspace::default();
        let mut result = self.inner.identify(&prepared, &CausalQuery::average_effect(ate), &mut id_ws)?;
        annotate_temporal(&mut result, query, treatment_key, outcome_key, history, horizon);

        Ok(TemporalIdentificationResult {
            result,
            indexer: unfolded.indexer,
            treatment_key,
            outcome_key,
        })
    }
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

/// Whether any ancestor of `{treatment, outcome}` in the unfolded graph sits
/// at the deepest slice (`offset == -history`) with in-template parents that
/// the truncation cut off. When this returns `false`, growing the window
/// further cannot add backdoor paths between treatment and outcome.
fn ancestry_touches_boundary(
    unfolded: &UnfoldedTemporalGraph,
    treatment_dense: u32,
    outcome_dense: u32,
    history: u32,
    truncatable: &[bool],
) -> bool {
    let dag = &unfolded.dag;
    let mut ancestors = BitSet::with_len(dag.node_count());
    let mut gws = GraphWorkspace::default();
    dag.ancestors_of(
        &[DenseNodeId::from_raw(treatment_dense), DenseNodeId::from_raw(outcome_dense)],
        &mut ancestors,
        &mut gws,
    );
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

fn required_variable_count(
    template: &TemporalDag,
    treatment: VariableId,
    outcome: VariableId,
) -> u32 {
    let mut max_id = treatment.raw().max(outcome.raw());
    for node in template.nodes() {
        if let NodeRef::Lagged { variable, .. } = node {
            max_id = max_id.max(variable.raw());
        }
    }
    max_id.saturating_add(1)
}

fn retarget(
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

fn annotate_temporal(
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
    use causal_core::Lag;

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
            Err(IdentificationError::NotIdentified { .. })
        ));
    }

    #[test]
    fn sustained_policy_is_unsupported() {
        let template = TemporalDag::empty();
        let query = TemporalEffectQuery::sustained(
            VariableId::from_raw(0),
            VariableId::from_raw(1),
            2,
            1.0,
        );
        let identifier = TemporalBackdoorIdentifier::new();
        assert!(matches!(
            identifier.identify_temporal(&template, &query),
            Err(IdentificationError::SustainedPolicyUnsupported)
        ));
    }
}
