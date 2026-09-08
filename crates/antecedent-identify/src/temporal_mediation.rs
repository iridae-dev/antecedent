//! Linear temporal mediation identification on stationary templates.
//!
//! For a linear SEM on a [`TemporalDag`], total / direct / mediated effects
//! decompose via path products once a mediator set participates on treatment→
//! outcome pathways in the template.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::too_many_lines)]

use std::sync::Arc;

use antecedent_core::{
    Assumption, AssumptionRecord, AssumptionScope, AssumptionSet, AssumptionSource,
    AssumptionStatus, CausalQuery, MediationContrast, MediationQuery, TemporalEffectQuery,
    TemporalNodeKey, TemporalPolicy, VariableId,
};
use antecedent_expr::{CausalExprArena, IdentifiedEstimand};
use antecedent_graph::{NodeRef, TemporalDag};

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
    /// When true, [`MediationContrast::NaturalDirect`] / [`MediationContrast::NaturalIndirect`]
    /// are treated as their controlled counterparts (linear alias).
    pub allow_natural_controlled_alias: bool,
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
        if matches!(
            query.contrast,
            MediationContrast::NaturalDirect | MediationContrast::NaturalIndirect
        ) && !self.allow_natural_controlled_alias
        {
            return Err(IdentificationError::unsupported(
                "NaturalDirect/NaturalIndirect require allow_natural_controlled_alias; \
                 natural effects alias controlled effects in linear temporal mediation",
            ));
        }
        Self::ensure_mediators_intercept(template, query)?;
        let adjustment = Self::adjustment_nodes(template, query)?;

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
                antecedent_core::Intervention::Set { value: active, .. },
                antecedent_core::Intervention::Set { value: control, .. },
            ) => (active.clone(), control.clone()),
            _ => {
                return Err(IdentificationError::UnsupportedQuery {
                    message: "mediation requires Set interventions",
                });
            }
        };

        let mut arena = CausalExprArena::new();
        let functional = arena.temporal_mediation_ate(
            query.treatment,
            query.outcome,
            &query.mediators,
            active,
            control,
        );
        let estimand = IdentifiedEstimand::temporal_mediation(
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
            assumption: Assumption::ParametricRestriction(antecedent_core::ParametricAssumption {
                id: Arc::from("linear_sem"),
                description: Arc::from("linear temporal SEM path-product mediation"),
            }),
            source: AssumptionSource::AlgorithmDefault {
                algorithm: Arc::from("temporal_mediation"),
            },
            scope: AssumptionScope::Identification,
            status: AssumptionStatus::Declared,
        });
        if matches!(
            query.contrast,
            MediationContrast::NaturalDirect | MediationContrast::NaturalIndirect
        ) {
            assumptions.push(AssumptionRecord {
                assumption: Assumption::Custom {
                    id: Arc::from("natural_controlled_alias"),
                    description: Arc::from(
                        "natural direct/indirect effects are aliased to controlled \
                         direct/mediated effects under linear temporal mediation",
                    ),
                },
                source: AssumptionSource::AlgorithmDefault {
                    algorithm: Arc::from("temporal_mediation"),
                },
                scope: AssumptionScope::Identification,
                status: AssumptionStatus::Declared,
            });
        }

        let mut derivation = DerivationTrace::default();
        derivation
            .push("temporal_mediation.adjustment", format!("baseline parents={adjustment:?}"));
        derivation.push(
            method.as_ref(),
            format!(
                "mediators={:?} contrast={:?}",
                query.mediators.iter().map(|v| v.raw()).collect::<Vec<_>>(),
                query.contrast
            ),
        );

        Ok(IdentificationResult {
            status: IdentificationStatus::IdentifiedUnderParametricRestrictions,
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
            hedge: None,
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
        let mut has_t_to_m = false;
        let mut has_m_to_y = false;
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
            if *tgt == query.outcome && med.contains(src) {
                has_m_to_y = true;
            }
            if *src == query.treatment && med.contains(tgt) {
                has_t_to_m = true;
            }
        }
        if !(has_t_to_m && has_m_to_y) {
            return Err(IdentificationError::NotCertified {
                message: "no treatment–mediator–outcome path found in temporal template",
            });
        }
        Ok(())
    }

    /// Baseline parents required by both linear mediation regressions.
    ///
    /// Edges are normalized to the mediator/outcome time origin. Conditioning
    /// on the union is valid for the two-equation product only when these
    /// additional parents are not descendants of the treatment or mediator.
    /// Treatment-induced intermediate variables require a larger path model.
    pub fn adjustment_nodes(
        template: &TemporalDag,
        query: &MediationQuery,
    ) -> Result<Vec<TemporalNodeKey>, IdentificationError> {
        let [mediator] = query.mediators.as_ref() else {
            return Err(IdentificationError::unsupported(
                "temporal mediation requires one mediator",
            ));
        };
        let treatment = TemporalNodeKey { variable: query.treatment, offset: -1 };
        let mediator_key = TemporalNodeKey { variable: *mediator, offset: 0 };
        let mut adjustment = Vec::new();
        let mut has_t_m = false;
        let mut has_m_y = false;
        let mut history = 1;
        let mut variable_count =
            query.treatment.raw().max(query.outcome.raw()).max(mediator.raw()) + 1;
        for node in template.nodes() {
            if let NodeRef::Lagged { variable, .. } = node {
                variable_count = variable_count.max(variable.raw() + 1);
            }
        }
        for edge in template.edges() {
            let (from, to) = edge.parent_child().expect("directed temporal edge");
            let (
                NodeRef::Lagged { variable: source, lag: source_lag },
                NodeRef::Lagged { variable: target, lag: target_lag },
            ) = (template.nodes()[from.as_usize()], template.nodes()[to.as_usize()])
            else {
                continue;
            };
            if target != *mediator && target != query.outcome {
                continue;
            }
            let lag = source_lag.raw() - target_lag.raw();
            history = history.max(lag);
            let key = TemporalNodeKey {
                variable: source,
                offset: -i32::try_from(lag).map_err(|e| IdentificationError::msg(e.to_string()))?,
            };
            has_t_m |= target == *mediator && key == treatment;
            has_m_y |= target == query.outcome && key == mediator_key;
            if key != treatment && key != mediator_key && !adjustment.contains(&key) {
                adjustment.push(key);
            }
        }
        if !has_t_m || !has_m_y {
            return Err(IdentificationError::unsupported(
                "temporal mediation requires T at lag one and M/Y contemporaneous",
            ));
        }
        let indexer = antecedent_data::TemporalIndexer::new(variable_count, history, 1)
            .map_err(|e| IdentificationError::msg(e.to_string()))?;
        let unfolded = template.unfold(indexer)?;
        let dense = |key| {
            unfolded
                .indexer
                .dense_id(key)
                .map(antecedent_graph::DenseNodeId::from_raw)
                .map_err(|e| IdentificationError::msg(e.to_string()))
        };
        let t = dense(treatment)?;
        let m = dense(mediator_key)?;
        for &key in &adjustment {
            let z = dense(key)?;
            if unfolded.dag.reaches(t, z) || unfolded.dag.reaches(m, z) {
                return Err(IdentificationError::unsupported(
                    "treatment-induced mediation covariates require a larger path model",
                ));
            }
        }
        adjustment.sort_by_key(|key| (key.variable.raw(), key.offset));
        Ok(adjustment)
    }
}

#[cfg(test)]
mod tests {
    use antecedent_core::{Lag, MediationContrast, VariableId};
    use antecedent_graph::TemporalDag;

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
        assert!(matches!(id.status, IdentificationStatus::IdentifiedUnderParametricRestrictions));
        assert_eq!(id.estimands[0].mediators.len(), 1);
        assert!(id.estimands[0].method.as_ref().starts_with("temporal_mediation."));
        assert_eq!(
            id.arena.derivation(id.estimands[0].functional).map(|d| d.rule.as_ref()),
            Some("temporal_mediation")
        );
    }

    #[test]
    fn identifies_mediated_with_direct_edge() {
        let mut g = TemporalDag::empty();
        let t1 = g.add_lagged(VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
        let m0 = g.add_lagged(VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
        let y0 = g.add_lagged(VariableId::from_raw(2), Lag::CONTEMPORANEOUS).unwrap();
        g.insert_directed(t1, m0).unwrap();
        g.insert_directed(m0, y0).unwrap();
        g.insert_directed(t1, y0).unwrap();
        let q = MediationQuery::binary(
            VariableId::from_raw(0),
            VariableId::from_raw(2),
            [VariableId::from_raw(1)],
            MediationContrast::Mediated,
        );
        let id = TemporalMediationIdentifier::new().identify(&g, &q).unwrap();
        assert!(matches!(id.status, IdentificationStatus::IdentifiedUnderParametricRestrictions));
    }

    #[test]
    fn incomplete_path_not_identified() {
        let mut g = TemporalDag::empty();
        let t1 = g.add_lagged(VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
        let m0 = g.add_lagged(VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
        let _y0 = g.add_lagged(VariableId::from_raw(2), Lag::CONTEMPORANEOUS).unwrap();
        g.insert_directed(t1, m0).unwrap();
        // Missing M→Y.
        let q = MediationQuery::binary(
            VariableId::from_raw(0),
            VariableId::from_raw(2),
            [VariableId::from_raw(1)],
            MediationContrast::Mediated,
        );
        let err = TemporalMediationIdentifier::new().identify(&g, &q).unwrap_err();
        assert!(matches!(err, IdentificationError::NotCertified { .. }));
    }

    #[test]
    fn natural_contrast_without_flag_errors() {
        let g = chain_template();
        let q = MediationQuery::binary(
            VariableId::from_raw(0),
            VariableId::from_raw(2),
            [VariableId::from_raw(1)],
            MediationContrast::NaturalIndirect,
        );
        let err = TemporalMediationIdentifier::new().identify(&g, &q).unwrap_err();
        assert!(matches!(err, IdentificationError::UnsupportedQuery { .. }));
    }

    #[test]
    fn natural_contrast_with_flag_succeeds() {
        let g = chain_template();
        let q = MediationQuery::binary(
            VariableId::from_raw(0),
            VariableId::from_raw(2),
            [VariableId::from_raw(1)],
            MediationContrast::NaturalIndirect,
        );
        let mut ider = TemporalMediationIdentifier::new();
        ider.allow_natural_controlled_alias = true;
        let id = ider.identify(&g, &q).unwrap();
        assert!(id.required_assumptions.entries.iter().any(|a| matches!(
            &a.assumption,
            Assumption::Custom { id, .. } if id.as_ref() == "natural_controlled_alias"
        )));
    }
}
