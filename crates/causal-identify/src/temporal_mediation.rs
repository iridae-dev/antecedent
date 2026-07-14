//! Linear temporal mediation identification on stationary templates (Phase 9).
//!
//! For a linear SEM on a [`TemporalDag`], total / direct / mediated effects
//! decompose via path products once a mediator set participates on treatment→
//! outcome pathways in the template.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use causal_core::{
    Assumption, AssumptionRecord, AssumptionScope, AssumptionSet, AssumptionSource,
    AssumptionStatus, CausalQuery, MediationContrast, MediationQuery, TemporalEffectQuery,
    TemporalPolicy, VariableId,
};
use causal_expr::{CausalExprArena, IdentifiedEstimand};
use causal_graph::{NodeRef, TemporalDag};

use crate::error::IdentificationError;
use crate::result::{
    DerivationTrace, IdentificationPerformanceRecord, IdentificationResult, IdentificationStatus,
};
use crate::temporal_backdoor::{TemporalBackdoorIdentifier, TemporalIdentificationResult};

/// Temporal linear mediation identifier.
#[derive(Clone, Debug, Default)]
pub struct TemporalMediationIdentifier {
    /// Reuses temporal unfolding / backdoor machinery for optional horizon checks.
    pub temporal: TemporalBackdoorIdentifier,
}

impl TemporalMediationIdentifier {
    /// Create with defaults.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Identify a mediation query on a stationary temporal template.
    ///
    /// # Errors
    ///
    /// Invalid query, or mediators that do not participate on T→Y pathways.
    pub fn identify(
        &self,
        template: &TemporalDag,
        query: &MediationQuery,
    ) -> Result<IdentificationResult, IdentificationError> {
        query.validate().map_err(|_| IdentificationError::UnsupportedQuery {
            message: "invalid mediation query",
        })?;
        Self::ensure_mediators_intercept(template, query)?;

        let method: Arc<str> = match query.contrast {
            MediationContrast::Total => Arc::from("temporal_mediation.total"),
            MediationContrast::Direct | MediationContrast::NaturalDirect => {
                Arc::from("temporal_mediation.direct")
            }
            MediationContrast::Mediated | MediationContrast::NaturalIndirect => {
                Arc::from("temporal_mediation.mediated")
            }
        };

        let (active, control) = match (&query.active, &query.control) {
            (
                causal_core::Intervention::Set { value: active, .. },
                causal_core::Intervention::Set { value: control, .. },
            ) => (active.clone(), control.clone()),
            _ => {
                return Err(IdentificationError::UnsupportedQuery {
                    message: "mediation requires Set interventions",
                });
            }
        };

        let mut arena = CausalExprArena::new();
        let functional =
            arena.frontdoor_ate(query.treatment, query.outcome, &query.mediators, active, control);
        let estimand = IdentifiedEstimand::frontdoor(
            Arc::clone(&method),
            Arc::clone(&query.mediators),
            functional,
        );

        let mut assumptions = AssumptionSet::new();
        assumptions.push(AssumptionRecord {
            assumption: Assumption::Stationarity,
            source: AssumptionSource::AlgorithmDefault {
                algorithm: Arc::from("temporal_mediation"),
            },
            scope: AssumptionScope::Identification,
            status: AssumptionStatus::Declared,
        });
        assumptions.push(AssumptionRecord {
            assumption: Assumption::ParametricRestriction(causal_core::ParametricAssumption {
                id: Arc::from("linear_sem"),
                description: Arc::from("linear temporal SEM path-product mediation"),
            }),
            source: AssumptionSource::AlgorithmDefault {
                algorithm: Arc::from("temporal_mediation"),
            },
            scope: AssumptionScope::Identification,
            status: AssumptionStatus::Declared,
        });

        let mut derivation = DerivationTrace::default();
        derivation.push(
            method.as_ref(),
            format!(
                "mediators={:?} contrast={:?}",
                query.mediators.iter().map(|v| v.raw()).collect::<Vec<_>>(),
                query.contrast
            ),
        );

        Ok(IdentificationResult {
            status: IdentificationStatus::PartiallyIdentified,
            query: CausalQuery::mediation(query.clone()),
            estimands: vec![estimand],
            arena,
            derivation,
            required_assumptions: assumptions,
            diagnostics: Vec::new(),
            performance: IdentificationPerformanceRecord {
                candidates_examined: 1,
                sets_returned: 1,
            },
        })
    }

    /// Identify using a temporal-effect shell (horizon/policy) plus mediator set.
    ///
    /// # Errors
    ///
    /// Propagates temporal unfolding / mediation failures.
    pub fn identify_with_horizon(
        &self,
        template: &TemporalDag,
        mediation: &MediationQuery,
        horizon_steps: u32,
    ) -> Result<(IdentificationResult, TemporalIdentificationResult), IdentificationError> {
        let te = TemporalEffectQuery {
            treatment: mediation.treatment,
            outcome: mediation.outcome,
            policy: TemporalPolicy::Pulse { at: 0 },
            control: mediation.control.clone(),
            active: mediation.active.clone(),
            horizon_steps,
            max_history_lag: None,
            target_population: mediation.target_population.clone(),
        };
        let temporal = self.temporal.identify_temporal(template, &te)?;
        let id = self.identify(template, mediation)?;
        Ok((id, temporal))
    }

    fn ensure_mediators_intercept(
        template: &TemporalDag,
        query: &MediationQuery,
    ) -> Result<(), IdentificationError> {
        let med: std::collections::HashSet<VariableId> = query.mediators.iter().copied().collect();
        let mut has_path = false;
        let mut has_direct = false;
        for e in template.edges() {
            let Some((from, to)) = e.parent_child() else {
                continue;
            };
            let (
                Some(NodeRef::Lagged { variable: src, .. }),
                Some(NodeRef::Lagged { variable: tgt, .. }),
            ) = (template.nodes().get(from.as_usize()), template.nodes().get(to.as_usize()))
            else {
                continue;
            };
            if *tgt == query.outcome && *src == query.treatment {
                has_direct = true;
                has_path = true;
            }
            if *tgt == query.outcome && med.contains(src) {
                has_path = true;
            }
            if *src == query.treatment && med.contains(tgt) {
                has_path = true;
            }
        }
        if matches!(
            query.contrast,
            MediationContrast::Mediated | MediationContrast::NaturalIndirect
        ) && has_direct
        {
            return Err(IdentificationError::NotIdentified {
                message: "mediated contrast requires no direct treatment→outcome edge",
            });
        }
        if !has_path {
            return Err(IdentificationError::NotIdentified {
                message: "no treatment–mediator–outcome path found in temporal template",
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use causal_core::{Lag, MediationContrast, VariableId};
    use causal_graph::TemporalDag;

    use super::*;

    fn chain_template() -> TemporalDag {
        let mut g = TemporalDag::empty();
        let t1 = g.add_lagged(VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
        let m0 = g.add_lagged(VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
        let y0 = g.add_lagged(VariableId::from_raw(2), Lag::CONTEMPORANEOUS).unwrap();
        g.insert_directed(t1, m0).unwrap();
        g.insert_directed(m0, y0).unwrap();
        g
    }

    #[test]
    fn identifies_mediated_chain() {
        let g = chain_template();
        let q = MediationQuery::binary(
            VariableId::from_raw(0),
            VariableId::from_raw(2),
            [VariableId::from_raw(1)],
            MediationContrast::Mediated,
        );
        let id = TemporalMediationIdentifier::new().identify(&g, &q).unwrap();
        assert!(matches!(id.status, IdentificationStatus::PartiallyIdentified));
        assert_eq!(id.estimands[0].mediators.len(), 1);
    }
}
