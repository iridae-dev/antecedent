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
use antecedent_graph::{
    BitSet, DSeparationWorkspace, DenseNodeId, GraphOverlay, GraphWorkspace, NodeRef, TemporalDag,
};

use crate::error::IdentificationError;
use crate::result::{
    DerivationTrace, IdentificationPerformanceRecord, IdentificationResult, IdentificationStatus,
};
use crate::temporal_backdoor::{TemporalBackdoorIdentifier, TemporalIdentificationResult};

/// Certified shared adjustment set for one horizon (see
/// [`TemporalMediationIdentifier::identify_with_horizon`]).
struct MediationAdjustment {
    /// `S(h)` as unfolded dense ids (the estimand's `adjustment_set`).
    dense: Arc<[VariableId]>,
    /// `S(h)` as temporal keys.
    shared: Vec<TemporalNodeKey>,
    /// The `T→Y` back-door set `I(h)`.
    backdoor: Vec<TemporalNodeKey>,
    /// Mediator/outcome parents added to `I(h)`.
    mediator_outcome: Vec<TemporalNodeKey>,
}

/// `{v3[t-1], …}` with lags relative to the outcome time.
fn describe_keys(keys: &[TemporalNodeKey], outcome_offset: i32) -> String {
    let items: Vec<String> = keys
        .iter()
        .map(|key| {
            let lag = outcome_offset - key.offset;
            if lag == 0 {
                format!("v{}[t]", key.variable.raw())
            } else {
                format!("v{}[t-{lag}]", key.variable.raw())
            }
        })
        .collect();
    format!("{{{}}}", items.join(", "))
}

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
        let horizon = query.horizons.first().copied().unwrap_or(1);
        self.identify_horizon(template, query, horizon).map(|(id, _)| id)
    }

    /// Shared body of [`Self::identify`] and [`Self::identify_with_horizon`].
    fn identify_horizon(
        &self,
        template: &TemporalDag,
        query: &MediationQuery,
        horizon: u32,
    ) -> Result<(IdentificationResult, TemporalIdentificationResult), IdentificationError> {
        Self::ensure_mediators_intercept(template, query)?;
        // Template premises (one mediator, T@lag-1 and M/Y contemporaneous,
        // no treatment-induced mediator/outcome parent) stay required for the
        // linear path-product and keep their refusal.
        let _premises = Self::adjustment_nodes(template, query)?;
        let temporal = self.identify_backdoor(template, query, horizon)?;
        let adjustment = Self::mediation_adjustment(template, query, &temporal)?;
        let id = Self::mediation_result(query, horizon, &temporal, &adjustment)?;
        Ok((id, temporal))
    }

    /// Shared adjustment set `S(h)` for the three mechanism regressions at one
    /// horizon, in the coordinates of `temporal.indexer`.
    ///
    /// Under the linear SEM decomposition the mediator model `M ~ T + S` needs
    /// a valid `T→M` adjustment set, and the outcome model `Y ~ T + M + S` needs
    /// a set that blocks `M←…→Y` confounding as well as the `T→Y` backdoor
    /// paths. `I(h)` alone certifies only the last. `S(h)` is `I(h)` plus every
    /// parent of `M` and `Y` other than `T` and `M` that is not a descendant of
    /// the treatment or the mediator, and it is certified on the unfolded DAG:
    ///
    /// * no member descends from `T` or `M`;
    /// * `T ⊥ M | S` with `T`'s outgoing edges removed (mediator model);
    /// * `{T, M} ⊥ Y | S` with the outgoing edges of `T` and `M` removed
    ///   (outcome model: joint back-door for the controlled `(T, M)` contrast);
    /// * `T ⊥ Y | S` with `T`'s outgoing edges removed (reduced-form Total).
    ///
    /// Using one set in all three regressions keeps `Total = Direct + Mediated`
    /// exact in sample.
    fn mediation_adjustment(
        template: &TemporalDag,
        query: &MediationQuery,
        temporal: &TemporalIdentificationResult,
    ) -> Result<MediationAdjustment, IdentificationError> {
        let [mediator] = query.mediators.as_ref() else {
            return Err(IdentificationError::unsupported(
                "temporal mediation requires one mediator",
            ));
        };
        let unfolded = template.unfold(temporal.indexer.clone())?;
        let dag = &unfolded.dag;
        let dense = |key: TemporalNodeKey| {
            temporal
                .indexer
                .dense_id(key)
                .map(DenseNodeId::from_raw)
                .map_err(|e| IdentificationError::msg(e.to_string()))
        };
        let t = dense(temporal.treatment_key)?;
        let y = dense(temporal.outcome_key)?;
        let m =
            dense(TemporalNodeKey { variable: *mediator, offset: temporal.outcome_key.offset })?;

        let mut gws = GraphWorkspace::default();
        let mut downstream = BitSet::with_len(dag.node_count());
        dag.descendants_of(&[t, m], &mut downstream, &mut gws);

        let backdoor: Vec<DenseNodeId> = temporal
            .result
            .estimands
            .first()
            .map(|e| e.adjustment_set.iter().map(|v| DenseNodeId::from_raw(v.raw())).collect())
            .unwrap_or_default();
        let mut set: Vec<DenseNodeId> =
            backdoor.iter().copied().filter(|&z| !downstream.contains(z)).collect();
        let mut mediator_outcome = Vec::new();
        for &p in dag.parents(m).iter().chain(dag.parents(y)) {
            if p == t || p == m || downstream.contains(p) || set.contains(&p) {
                continue;
            }
            set.push(p);
            mediator_outcome.push(p);
        }
        set.sort_unstable();
        mediator_outcome.sort_unstable();

        let n = dag.node_count();
        let mut dws = DSeparationWorkspace::default();
        let treatment_cut = GraphOverlay::remove_outgoing(n, &[t]);
        let joint_cut = GraphOverlay::remove_outgoing(n, &[t, m]);
        let mediator_ok = dag.view(&treatment_cut).is_d_separated(t, m, &set, &mut dws)?;
        let outcome_ok = dag.view(&joint_cut).is_d_separated(t, y, &set, &mut dws)?
            && dag.view(&joint_cut).is_d_separated(m, y, &set, &mut dws)?;
        let total_ok = dag.view(&treatment_cut).is_d_separated(t, y, &set, &mut dws)?;
        if !(mediator_ok && outcome_ok && total_ok) {
            return Err(IdentificationError::NotCertified {
                message: "temporal mediation: I(h) plus the mediator/outcome parents that are \
                          not treatment or mediator descendants does not block \
                          treatment–mediator, mediator–outcome and treatment–outcome confounding \
                          at this horizon; blocking it would condition on a treatment descendant",
            });
        }
        let key = |d: &DenseNodeId| temporal.indexer.key_of(d.raw()).ok();
        Ok(MediationAdjustment {
            dense: set.iter().map(|d| VariableId::from_raw(d.raw())).collect(),
            shared: set.iter().filter_map(key).collect(),
            backdoor: backdoor.iter().filter_map(key).collect(),
            mediator_outcome: mediator_outcome.iter().filter_map(key).collect(),
        })
    }

    fn identify_backdoor(
        &self,
        template: &TemporalDag,
        mediation: &MediationQuery,
        horizon_steps: u32,
    ) -> Result<TemporalIdentificationResult, IdentificationError> {
        // Pulse at -1, outcome at h-1: treatment is h steps before the
        // outcome (estimator lag h). This is the temporal_confounded_pulse
        // convention — I(1)={Z@-1}, I(2) typically empty — not Pulse at 0
        // (T@0, Y@h-1), which would break the lag-1 mediation pins.
        let te = TemporalEffectQuery {
            treatment: mediation.treatment,
            outcome: mediation.outcome,
            policy: TemporalPolicy::Pulse { at: -1 },
            control: mediation.control.clone(),
            active: mediation.active.clone(),
            horizon_steps,
            max_history_lag: None,
            target_population: mediation.target_population.clone(),
        };
        self.temporal.identify_temporal(template, &te)
    }

    fn mediation_result(
        query: &MediationQuery,
        horizon: u32,
        temporal: &TemporalIdentificationResult,
        adjustment: &MediationAdjustment,
    ) -> Result<IdentificationResult, IdentificationError> {
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
        let mut estimand = IdentifiedEstimand::temporal_mediation(
            Arc::clone(&method),
            Arc::clone(&query.mediators),
            functional,
        );
        estimand.adjustment_set = Arc::clone(&adjustment.dense);

        let mut assumptions = AssumptionSet::new();
        let outcome_offset = temporal.outcome_key.offset;
        let describe = |keys: &[TemporalNodeKey]| describe_keys(keys, outcome_offset);
        assumptions.push(AssumptionRecord {
            assumption: Assumption::Custom {
                id: Arc::from("temporal_mediation.adjustment_sets"),
                description: Arc::from(format!(
                    "horizon {horizon}: mediator model M ~ T + S, outcome model Y ~ T + M + S and \
                     reduced form Y ~ T + S share S = {} (lags relative to the outcome time); \
                     S = I({horizon}) {} plus mediator/outcome parents {} that are not \
                     descendants of the treatment or mediator; certified on the unfolded DAG as \
                     a T→M back-door set, a joint (T, M)→Y back-door set and a T→Y back-door set",
                    describe(&adjustment.shared),
                    describe(&adjustment.backdoor),
                    describe(&adjustment.mediator_outcome),
                )),
            },
            source: AssumptionSource::AlgorithmDefault {
                algorithm: Arc::from("temporal_mediation"),
            },
            scope: AssumptionScope::Identification,
            status: AssumptionStatus::Declared,
        });
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
        derivation.push(
            "temporal.backdoor.unfolded",
            format!("I({horizon}) adjustment={:?}", adjustment.backdoor),
        );
        derivation.push(
            "temporal_mediation.adjustment",
            format!(
                "S({horizon}) = I({horizon}) ∪ mediator/outcome parents {:?} = {:?}; shared by \
                 the mediator, outcome and reduced-form regressions",
                adjustment.mediator_outcome, adjustment.shared,
            ),
        );
        derivation.push(
            method.as_ref(),
            format!(
                "mediators={:?} contrast={:?} horizons={:?}",
                query.mediators.iter().map(|v| v.raw()).collect::<Vec<_>>(),
                query.contrast,
                query.horizons.as_ref(),
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
        mediation.validate().map_err(|_| IdentificationError::UnsupportedQuery {
            message: "invalid mediation query",
        })?;
        if matches!(
            mediation.contrast,
            MediationContrast::NaturalDirect | MediationContrast::NaturalIndirect
        ) && !self.allow_natural_controlled_alias
        {
            return Err(IdentificationError::unsupported(
                "NaturalDirect/NaturalIndirect require allow_natural_controlled_alias; \
                 natural effects alias controlled effects in linear temporal mediation",
            ));
        }
        self.identify_horizon(template, mediation, horizon_steps)
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

    fn confounded_mediator_template() -> TemporalDag {
        let mut g = TemporalDag::empty();
        let t0 = g.add_lagged(VariableId::from_raw(0), Lag::CONTEMPORANEOUS).unwrap();
        let t1 = g.add_lagged(VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
        let m0 = g.add_lagged(VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
        let y0 = g.add_lagged(VariableId::from_raw(2), Lag::CONTEMPORANEOUS).unwrap();
        let z0 = g.add_lagged(VariableId::from_raw(3), Lag::CONTEMPORANEOUS).unwrap();
        let z1 = g.add_lagged(VariableId::from_raw(3), Lag::from_raw(1)).unwrap();
        // Same geometry as temporal_confounded_pulse, plus T→M→Y.
        g.insert_directed(z0, t0).unwrap();
        g.insert_directed(z1, y0).unwrap();
        g.insert_directed(t1, y0).unwrap();
        g.insert_directed(t1, m0).unwrap();
        g.insert_directed(m0, y0).unwrap();
        g
    }

    fn named_z(temporal: &TemporalIdentificationResult) -> Vec<(u32, i32)> {
        temporal
            .result
            .estimands
            .first()
            .map(|e| {
                e.adjustment_set
                    .iter()
                    .filter_map(|&dense| {
                        let key = temporal.indexer.key_of(dense.raw()).ok()?;
                        Some((key.variable.raw(), key.offset))
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    #[test]
    fn identifies_i1_not_equal_i2_on_confounded_mediator() {
        let g = confounded_mediator_template();
        let q = MediationQuery::binary(
            VariableId::from_raw(0),
            VariableId::from_raw(2),
            [VariableId::from_raw(1)],
            MediationContrast::Mediated,
        );
        let ider = TemporalMediationIdentifier::new();
        let (id1, t1) = ider.identify_with_horizon(&g, &q, 1).unwrap();
        let (id2, t2) = ider.identify_with_horizon(&g, &q, 2).unwrap();
        assert!(id1.estimands[0].method.as_ref().starts_with("temporal_mediation."));
        assert!(id2.estimands[0].method.as_ref().starts_with("temporal_mediation."));
        let z1 = named_z(&t1);
        let z2 = named_z(&t2);
        assert_eq!(z1, vec![(3, -1)], "I(1) must be Z@-1, got {z1:?}");
        assert!(z2.is_empty(), "I(2) must be empty, got {z2:?}");
        assert_ne!(z1, z2, "I(1)={z1:?} must differ from I(2)={z2:?}");
        // h=1: Z@-1 is both the T→Y back-door set and the only extra parent of
        // M and Y, so S(1) = I(1).
        assert_eq!(
            id1.estimands[0].adjustment_set.as_ref(),
            t1.result.estimands[0].adjustment_set.as_ref()
        );
        // h=2 (outcome at offset 1 in the unfolded window): I(2) is empty, but
        // T@0 and Z@0 are parents of M@1/Y@1 that do not descend from T@-1, so
        // they confound the mediator-outcome relation and join S(2).
        assert_eq!(mediation_keys(&id2, &t2), vec![(0, 0), (3, 0)]);
        assert!(
            id1.derivation.steps.iter().any(|s| s.rule.as_ref() == "temporal.backdoor.unfolded"),
            "certificate must show I(h) via temporal.backdoor.unfolded"
        );
        assert!(
            id1.derivation.steps.iter().any(|s| s.rule.as_ref() == "temporal_mediation.adjustment"),
            "certificate must show the shared mediation adjustment set"
        );
    }

    fn mediation_keys(
        id: &IdentificationResult,
        temporal: &TemporalIdentificationResult,
    ) -> Vec<(u32, i32)> {
        let mut keys: Vec<_> = id.estimands[0]
            .adjustment_set
            .iter()
            .filter_map(|&dense| {
                let key = temporal.indexer.key_of(dense.raw()).ok()?;
                Some((key.variable.raw(), key.offset))
            })
            .collect();
        keys.sort_unstable();
        keys
    }

    /// `t@1 -> m`, `t@1 -> y`, `m -> y`, `z@1 -> m`, `w@1 -> y`, `z@1 -> w@1`:
    /// `m <- z[t-1] -> w[t-1] -> y` confounds the mediator-outcome relation
    /// without confounding `t -> y`.
    fn mediator_outcome_confounded_template() -> TemporalDag {
        let mut g = TemporalDag::empty();
        let t1 = g.add_lagged(VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
        let m0 = g.add_lagged(VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
        let y0 = g.add_lagged(VariableId::from_raw(2), Lag::CONTEMPORANEOUS).unwrap();
        let z1 = g.add_lagged(VariableId::from_raw(3), Lag::from_raw(1)).unwrap();
        let w1 = g.add_lagged(VariableId::from_raw(4), Lag::from_raw(1)).unwrap();
        g.insert_directed(t1, m0).unwrap();
        g.insert_directed(t1, y0).unwrap();
        g.insert_directed(m0, y0).unwrap();
        g.insert_directed(z1, m0).unwrap();
        g.insert_directed(w1, y0).unwrap();
        g.insert_directed(z1, w1).unwrap();
        g
    }

    #[test]
    fn mediator_outcome_confounders_join_the_backdoor_set() {
        let g = mediator_outcome_confounded_template();
        let q = MediationQuery::binary(
            VariableId::from_raw(0),
            VariableId::from_raw(2),
            [VariableId::from_raw(1)],
            MediationContrast::Mediated,
        );
        let (id, temporal) =
            TemporalMediationIdentifier::new().identify_with_horizon(&g, &q, 1).unwrap();
        assert!(named_z(&temporal).is_empty(), "t -> y has no back-door path");
        assert_eq!(mediation_keys(&id, &temporal), vec![(3, -1), (4, -1)]);
        let record = id
            .required_assumptions
            .entries
            .iter()
            .find_map(|a| match &a.assumption {
                Assumption::Custom { id, description }
                    if id.as_ref() == "temporal_mediation.adjustment_sets" =>
                {
                    Some(description.clone())
                }
                _ => None,
            })
            .expect("adjustment sets are recorded as an assumption");
        assert!(record.contains("S = {v3[t-1], v4[t-1]}"), "{record}");
        assert!(record.contains("I(1) {}"), "{record}");
    }

    #[test]
    fn treatment_induced_mediator_outcome_parent_is_refused() {
        // t@1 -> q -> y with q contemporaneous: q is a treatment descendant.
        let mut g = chain_template();
        let t1 = g.add_lagged(VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
        let q0 = g.add_lagged(VariableId::from_raw(3), Lag::CONTEMPORANEOUS).unwrap();
        let y0 = g.add_lagged(VariableId::from_raw(2), Lag::CONTEMPORANEOUS).unwrap();
        g.insert_directed(t1, q0).unwrap();
        g.insert_directed(q0, y0).unwrap();
        let q = MediationQuery::binary(
            VariableId::from_raw(0),
            VariableId::from_raw(2),
            [VariableId::from_raw(1)],
            MediationContrast::Mediated,
        );
        let err = TemporalMediationIdentifier::new().identify_with_horizon(&g, &q, 1).unwrap_err();
        assert!(
            err.to_string()
                .contains("treatment-induced mediation covariates require a larger path model"),
            "{err}"
        );
    }

    #[test]
    fn horizon_two_refuses_treatment_induced_mediator_outcome_confounding() {
        // w[t-1] -> m[t] and w[t-1] -> y[t] confound M-Y; t[t-1] -> w[t].
        // At h=1 w[t-1] is not a descendant of t[t-1] and joins S(1). At h=2
        // it descends from t[t-2], so no admissible set blocks m <- w -> y.
        let mut g = chain_template();
        let t1 = g.add_lagged(VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
        let m0 = g.add_lagged(VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
        let y0 = g.add_lagged(VariableId::from_raw(2), Lag::CONTEMPORANEOUS).unwrap();
        let w0 = g.add_lagged(VariableId::from_raw(3), Lag::CONTEMPORANEOUS).unwrap();
        let w1 = g.add_lagged(VariableId::from_raw(3), Lag::from_raw(1)).unwrap();
        g.insert_directed(t1, w0).unwrap();
        g.insert_directed(w1, m0).unwrap();
        g.insert_directed(w1, y0).unwrap();
        let q = MediationQuery::binary(
            VariableId::from_raw(0),
            VariableId::from_raw(2),
            [VariableId::from_raw(1)],
            MediationContrast::Mediated,
        );
        let ider = TemporalMediationIdentifier::new();
        let (id1, t1) = ider.identify_with_horizon(&g, &q, 1).unwrap();
        assert_eq!(mediation_keys(&id1, &t1), vec![(3, -1)]);
        let err = ider.identify_with_horizon(&g, &q, 2).unwrap_err();
        assert!(matches!(err, IdentificationError::NotCertified { .. }), "{err}");
    }
}
