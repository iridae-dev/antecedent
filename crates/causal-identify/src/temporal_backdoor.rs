//! Temporal backdoor identification over finite unfolded graphs.
//!
//! A stationary [`TemporalDag`] template is materialised into a static [`Dag`]
//! over a finite window (history/horizon), the treatment and outcome are
//! mapped to their dense unfolded nodes, and the existing [`BackdoorIdentifier`]
//! runs unchanged on that static graph. Finiteness and stationarity of the
//! template become declared assumptions on the result.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use causal_core::{
    Assumption, AssumptionRecord, AssumptionScope, AssumptionSource, AssumptionStatus,
    AverageEffectQuery, CausalQuery, Intervention, TemporalEffectQuery, TemporalPolicy, VariableId,
};
use causal_data::{TemporalIndexer, TemporalNodeKey};
use causal_graph::{NodeRef, TemporalDag};

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
    /// The unfolding window (history/horizon) is derived from the policy
    /// offset, `horizon_steps`, `max_history_lag`, and the template's own
    /// maximum lag so that in-window confounders are not truncated.
    ///
    /// # Errors
    ///
    /// Invalid query, unfolding failures, sustained policies (not yet
    /// supported; they require sequential/g-formula identification rather
    /// than a single-node backdoor criterion), or backdoor identification
    /// errors.
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
        // Pulse-only for Phase 3 single-node backdoor.
        if matches!(query.policy, TemporalPolicy::Sustained { .. }) {
            return Err(IdentificationError::UnsupportedQuery {
                message: "temporal backdoor identification supports Pulse policies only; \
                          sustained interventions require sequential (g-formula) identification",
            });
        }
        if !matches!(query.policy, TemporalPolicy::Pulse { .. }) {
            return Err(IdentificationError::UnsupportedQuery {
                message: "unsupported temporal policy for Phase 3 backdoor identification",
            });
        }

        let min_offset = treatment_at.min(outcome_at).min(0);
        let max_offset = treatment_at.max(outcome_at).max(0);
        let history = min_offset
            .unsigned_abs()
            .max(template_max_lag(template))
            .max(query.max_history_lag.unwrap_or(0));
        let horizon = u32::try_from(max_offset)
            .map_err(|_| IdentificationError::Graph("negative horizon".into()))?
            .saturating_add(1);

        let variable_count = required_variable_count(template, query.treatment, query.outcome);
        let indexer = TemporalIndexer::new(variable_count, history, horizon)
            .map_err(|e| IdentificationError::Graph(e.to_string()))?;

        let unfolded =
            template.unfold(indexer).map_err(|e| IdentificationError::Graph(e.to_string()))?;

        let treatment_key = TemporalNodeKey { variable: query.treatment, offset: treatment_at };
        let outcome_key = TemporalNodeKey { variable: query.outcome, offset: outcome_at };

        let treatment_dense = unfolded
            .indexer
            .dense_id(treatment_key)
            .map_err(|_| IdentificationError::UnknownVariable { id: query.treatment })?;
        let outcome_dense = unfolded
            .indexer
            .dense_id(outcome_key)
            .map_err(|_| IdentificationError::UnknownVariable { id: query.outcome })?;

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
        let mut result = self.inner.identify(&prepared, &CausalQuery::average_effect(ate))?;
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
            Err(IdentificationError::UnsupportedQuery { .. })
        ));
    }
}
