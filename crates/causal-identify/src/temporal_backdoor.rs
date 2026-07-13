//! Temporal backdoor identification over finite unfolded graphs.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]

use std::sync::Arc;

use causal_core::{
    Assumption, AssumptionRecord, AssumptionScope, AssumptionSource, AssumptionStatus,
    AverageEffectQuery, CausalQuery, TemporalEffectQuery, TemporalPolicy, VariableId,
};
use causal_data::{TemporalIndexer, TemporalNodeKey};
use causal_graph::{TemporalDag, UnfoldedTemporalGraph};

use crate::backdoor::BackdoorIdentifier;
use crate::error::IdentificationError;
use crate::result::IdentificationResult;

/// Temporal backdoor identifier: unfold then static backdoor.
#[derive(Clone, Debug, Default)]
pub struct TemporalBackdoorIdentifier {
    /// Underlying static backdoor engine.
    pub inner: BackdoorIdentifier,
}

impl TemporalBackdoorIdentifier {
    /// Default engine.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Identify a temporal effect over a finite unfolding of `graph`.
    ///
    /// Treatment and outcome are mapped to unfolded dense nodes; adjustment sets
    /// are returned as [`VariableId`]s equal to those dense ids (caller maps via
    /// the returned indexer).
    ///
    /// # Errors
    ///
    /// Unfold / mapping / backdoor failures.
    pub fn identify(
        &self,
        graph: &TemporalDag,
        query: &TemporalEffectQuery,
        variable_count: u32,
    ) -> Result<(IdentificationResult, UnfoldedTemporalGraph), IdentificationError> {
        query.validate().map_err(|_| IdentificationError::UnsupportedQuery {
            message: "invalid temporal-effect query",
        })?;
        let history = query.max_history_lag.unwrap_or_else(|| max_template_lag(graph)).max(1);
        let horizon = query.horizon_steps.max(1);
        let indexer = TemporalIndexer::new(variable_count, history, horizon).map_err(|e| {
            IdentificationError::Graph(format!("temporal indexer: {e}"))
        })?;
        let unfolded = graph
            .unfold(indexer)
            .map_err(|e| IdentificationError::Graph(e.to_string()))?;

        let (t_off, y_off) = intervention_outcome_offsets(query);
        let t_key = TemporalNodeKey { variable: query.treatment, offset: t_off };
        let y_key = TemporalNodeKey { variable: query.outcome, offset: y_off };
        let t_dense = unfolded.indexer.dense_id(t_key).map_err(|e| {
            IdentificationError::Graph(format!("treatment node outside window: {e}"))
        })?;
        let y_dense = unfolded.indexer.dense_id(y_key).map_err(|e| {
            IdentificationError::Graph(format!("outcome node outside window: {e}"))
        })?;

        let ate = AverageEffectQuery::with_levels(
            VariableId::from_raw(t_dense),
            VariableId::from_raw(y_dense),
            intervention_f64(&query.control).unwrap_or(0.0),
            intervention_f64(&query.active).unwrap_or(1.0),
        );

        let prepared = self
            .inner
            .prepare(&unfolded.dag)
            .map_err(|e| IdentificationError::Graph(e.to_string()))?;
        let mut result = self
            .inner
            .identify(&prepared, &CausalQuery::AverageEffect(ate))?;

        result.query = CausalQuery::TemporalEffect(query.clone());
        for e in &mut result.estimands {
            e.method = Arc::from("temporal.backdoor.unfolded");
        }
        result.required_assumptions.push(AssumptionRecord {
            assumption: Assumption::Stationarity,
            source: AssumptionSource::AlgorithmDefault {
                algorithm: Arc::from("temporal.backdoor.unfolded"),
            },
            scope: AssumptionScope::Identification,
            status: AssumptionStatus::Declared,
        });
        result.derivation.push(
            "temporal.unfold",
            "finite unfolding of temporal DAG; backdoor on unfolded static graph",
        );

        Ok((result, unfolded))
    }
}

fn intervention_f64(i: &causal_core::Intervention) -> Option<f64> {
    match i {
        causal_core::Intervention::Set { value, .. } => value.as_f64(),
        _ => None,
    }
}

fn intervention_outcome_offsets(query: &TemporalEffectQuery) -> (i32, i32) {
    let t_off = match query.policy {
        TemporalPolicy::Pulse { at } => at,
        TemporalPolicy::Sustained { from, .. } => from,
        _ => 0,
    };
    let y_off = t_off + (query.horizon_steps as i32 - 1).max(0);
    (t_off, y_off)
}

fn max_template_lag(graph: &TemporalDag) -> u32 {
    use causal_graph::NodeRef;
    let mut max_lag = 0u32;
    for n in graph.nodes() {
        if let NodeRef::Lagged { lag, .. } = n {
            max_lag = max_lag.max(lag.raw());
        }
    }
    max_lag.max(1)
}

#[cfg(test)]
mod tests {
    use causal_core::{Lag, TemporalEffectQuery, VariableId};
    use causal_graph::{TemporalDag, ensure_lagged};

    use super::*;
    use crate::result::IdentificationStatus;

    #[test]
    fn lagged_chain_identifies() {
        let mut g = TemporalDag::empty();
        let x1 = ensure_lagged(&mut g, VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
        let y0 = ensure_lagged(&mut g, VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
        g.insert_directed(x1, y0).unwrap();

        // Pulse X at t=-1 (history), outcome Y at t=0 with horizon covering [0].
        let q = TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0)
            .with_policy(causal_core::TemporalPolicy::pulse(-1))
            .with_horizon_steps(2)
            .with_max_history_lag(Some(1));

        let id = TemporalBackdoorIdentifier::new();
        let (res, unfolded) = id.identify(&g, &q, 2).unwrap();
        assert_eq!(res.status, IdentificationStatus::NonparametricallyIdentified);
        assert!(!res.estimands.is_empty());
        assert!(unfolded.dag.node_count() >= 2);
        assert!(res.required_assumptions.entries.iter().any(|a| {
            matches!(a.assumption, Assumption::Stationarity)
        }));
    }
}
