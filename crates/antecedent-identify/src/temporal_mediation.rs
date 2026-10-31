//! Linear temporal mediation identification on stationary templates.
//!
//! For a linear SEM on a [`TemporalDag`], total / direct / mediated effects
//! decompose via path products once a mediator set participates on treatment→
//! outcome pathways in the template.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

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

// Bound temporary parent/mediator collections and scans before premise
// checks or reachability allocate structures derived from the template.
fn admit_mediation_scratch(
    template: &TemporalDag,
    query: &MediationQuery,
    budget: &mut antecedent_core::TemporalBudget,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<antecedent_core::execution::AdmissionLease, IdentificationError> {
    use crate::error::IdentificationBudget;
    if ctx.cancellation.is_cancelled() {
        return Err(IdentificationError::Cancelled);
    }
    let nodes = template.nodes().len() as u64;
    let mediators = query.mediators.len() as u64;
    budget.charge_work(nodes).map_err(crate::error::temporal_materialization_error)?;
    let mut edges = 0u64;
    for i in 0..template.nodes().len() {
        if ctx.cancellation.is_cancelled() {
            return Err(IdentificationError::Cancelled);
        }
        edges = edges
            .checked_add(
                template.children(antecedent_graph::DenseNodeId::from_index(i)).len() as u64
            )
            .ok_or_else(|| IdentificationError::budget(IdentificationBudget::Steps))?;
    }
    let work = nodes
        .checked_add(mediators)
        .and_then(|v| v.checked_add(edges))
        .and_then(|v| v.checked_add(1))
        .and_then(|v| v.checked_mul(mediators.checked_add(edges)?.checked_add(1)?))
        .and_then(|v| v.checked_mul(8))
        .ok_or_else(|| IdentificationError::budget(IdentificationBudget::Steps))?;
    budget.charge_work(work).map_err(crate::error::temporal_materialization_error)?;
    let bytes = mediators
        .checked_add(edges)
        .and_then(|v| v.checked_mul(256))
        .and_then(|v| v.checked_add(4096))
        .ok_or_else(|| IdentificationError::budget(IdentificationBudget::Memory))?;
    if bytes > budget.limits().max_bytes {
        return Err(IdentificationError::budget(IdentificationBudget::Memory));
    }
    ctx.acquire_admission(0, bytes).map_err(|error| match error {
        antecedent_core::execution::AdmissionError::Cancelled => IdentificationError::Cancelled,
        _ => IdentificationError::budget(IdentificationBudget::Memory),
    })
}

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
            let lag = i64::from(outcome_offset) - i64::from(key.offset);
            if lag == 0 {
                format!("v{}[t]", key.variable.raw())
            } else {
                format!("v{}[t-{lag}]", key.variable.raw())
            }
        })
        .collect();
    format!("{{{}}}", items.join(", "))
}

/// An admitted baseline adjustment set. Clones share both the nodes and their lease.
/// Caller-created copies of the borrowed slice are outside this owner's accounting.
#[derive(Clone, Debug)]
pub struct AdmittedTemporalAdjustmentNodes {
    inner: Arc<AdmittedTemporalAdjustmentNodesInner>,
}

#[derive(Debug)]
struct AdmittedTemporalAdjustmentNodesInner {
    nodes: Vec<TemporalNodeKey>,
    _lease: antecedent_core::execution::AdmissionLease,
}

impl std::ops::Deref for AdmittedTemporalAdjustmentNodes {
    type Target = [TemporalNodeKey];

    fn deref(&self) -> &Self::Target {
        &self.inner.nodes
    }
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
        // One result cannot carry several horizons; answering for the first and dropping the
        // rest would silently ignore part of the declared grid.
        if query.horizons.len() > 1 {
            return Err(IdentificationError::unsupported(
                "identify answers one horizon; call identify_with_horizon once per declared horizon",
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
        let ctx = self
            .temporal
            .inner
            .config
            .context
            .clone()
            .unwrap_or_else(|| antecedent_core::ExecutionContext::production_default(0));
        let mut budget = self.temporal.temporal_budget().unwrap_or_else(|| {
            antecedent_core::TemporalBudget::new(ctx.temporal_materialization_limits())
        });
        let _premise_lease = admit_mediation_scratch(template, query, &mut budget, &ctx)?;
        Self::ensure_mediators_intercept(template, query, &ctx)?;
        // Use the same caller budget for premise traversal and backdoor growth.
        let _premises = Self::adjustment_nodes_admitted(template, query, &mut budget, &ctx)?;
        let temporal_identifier = self.temporal.clone().with_temporal_context(&ctx, &budget);
        let temporal = self.identify_backdoor(&temporal_identifier, template, query, horizon)?;
        let adjustment = Self::mediation_adjustment(query, &temporal)?;
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
    /// the treatment or the mediator, and it is certified on the unfolded DAG. For a
    /// mediator set `M = {M₁…M_q}` every statement holds for each member (the
    /// regressions are `M_j ~ T + S` for each `j` and `Y ~ T + M₁…M_q + S`, whose product
    /// sum `Σ_j a_j b_j` is the block-mediated effect):
    ///
    /// * no member descends from `T` or any `M_j`;
    /// * `T ⊥ M_j | S` with `T`'s outgoing edges removed (mediator models);
    /// * `{T, M₁…M_q} ⊥ Y | S` with the outgoing edges of `T` and every `M_j` removed
    ///   (outcome model: joint back-door for the controlled `(T, M)` contrast);
    /// * `T ⊥ Y | S` with `T`'s outgoing edges removed (reduced-form Total).
    ///
    /// Using one set in all three regressions keeps `Total = Direct + Mediated`
    /// exact in sample.
    fn mediation_adjustment(
        query: &MediationQuery,
        temporal: &TemporalIdentificationResult,
    ) -> Result<MediationAdjustment, IdentificationError> {
        if query.mediators.is_empty() {
            return Err(IdentificationError::unsupported(
                "temporal mediation requires at least one mediator",
            ));
        }
        let dag = &temporal.unfolded.dag;
        let dense = |key: TemporalNodeKey| {
            temporal
                .indexer
                .dense_id(key)
                .map(DenseNodeId::from_raw)
                .map_err(|e| IdentificationError::construction(e))
        };
        let t = dense(temporal.treatment_key)?;
        let y = dense(temporal.outcome_key)?;
        let mediator_nodes = query
            .mediators
            .iter()
            .map(|&mediator| {
                dense(TemporalNodeKey { variable: mediator, offset: temporal.outcome_key.offset })
            })
            .collect::<Result<Vec<_>, _>>()?;

        let mut gws = GraphWorkspace::default();
        // The path-product estimator models mediators at the outcome slice.
        // An earlier mediator is covered when all its outcome paths pass through
        // one of those modeled nodes; only bypassing paths need extra offsets.
        let mut treatment_descendants = BitSet::with_len(dag.node_count());
        let mut outcome_ancestors = BitSet::with_len(dag.node_count());
        dag.descendants_of(&[t], &mut treatment_descendants, &mut gws);
        let mediator_cut = GraphOverlay::remove_outgoing(dag.node_count(), &mediator_nodes);
        dag.view(&mediator_cut).ancestors_of(&[y], &mut outcome_ancestors, &mut gws);
        for index in 0..dag.node_count() {
            let node = DenseNodeId::from_index(index);
            if !treatment_descendants.contains(node) || !outcome_ancestors.contains(node) {
                continue;
            }
            let key =
                temporal.indexer.key_of(node.raw()).map_err(IdentificationError::construction)?;
            if key.offset >= temporal.treatment_key.offset
                && key.offset < temporal.outcome_key.offset
                && query.mediators.contains(&key.variable)
            {
                return Err(IdentificationError::NotCertified {
                    message: "temporal mediation: a treatment-descendant mediator at an earlier \
                              slice reaches the outcome without a modeled outcome-slice mediator; \
                              the current path-product estimand cannot represent that path",
                });
            }
        }
        let mut downstream = BitSet::with_len(dag.node_count());
        let mut sources = Vec::with_capacity(1 + mediator_nodes.len());
        sources.push(t);
        sources.extend_from_slice(&mediator_nodes);
        dag.descendants_of(&sources, &mut downstream, &mut gws);

        let backdoor: Vec<DenseNodeId> = temporal
            .result
            .estimands
            .first()
            .map(|e| e.adjustment_set.iter().map(|v| DenseNodeId::from_raw(v.raw())).collect())
            .unwrap_or_default();
        let mut set: Vec<DenseNodeId> =
            backdoor.iter().copied().filter(|&z| !downstream.contains(z)).collect();
        let mut mediator_outcome = Vec::new();
        let mediator_parents: Vec<DenseNodeId> =
            mediator_nodes.iter().flat_map(|&m| dag.parents(m).iter().copied()).collect();
        for &p in mediator_parents.iter().chain(dag.parents(y)) {
            if p == t || mediator_nodes.contains(&p) || downstream.contains(p) || set.contains(&p) {
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
        let joint_cut = GraphOverlay::remove_outgoing(n, &sources);
        let mut mediator_ok = true;
        for &m in &mediator_nodes {
            mediator_ok &= dag.view(&treatment_cut).is_d_separated(t, m, &set, &mut dws)?;
        }
        let mut outcome_ok = dag.view(&joint_cut).is_d_separated(t, y, &set, &mut dws)?;
        for &m in &mediator_nodes {
            outcome_ok &= dag.view(&joint_cut).is_d_separated(m, y, &set, &mut dws)?;
        }
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
        temporal_identifier: &TemporalBackdoorIdentifier,
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
            effect_modifiers: std::sync::Arc::from([]),
        };
        temporal_identifier.identify_temporal(template, &te)
    }

    #[allow(
        clippy::too_many_lines,
        reason = "one linear identification pipeline whose stages share the prepared graph and workspace"
    )]
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
            estimand_claims: Vec::new(),
            temporal_leases: Vec::new(),
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
        ctx: &antecedent_core::ExecutionContext,
    ) -> Result<(), IdentificationError> {
        let mut med = std::collections::HashSet::new();
        for &variable in query.mediators.iter() {
            if ctx.cancellation.is_cancelled() {
                return Err(IdentificationError::Cancelled);
            }
            med.insert(variable);
        }
        let mut has_t_to_m = false;
        let mut has_m_to_y = false;
        for e in template.edges() {
            if ctx.cancellation.is_cancelled() {
                return Err(IdentificationError::Cancelled);
            }
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
    ) -> Result<AdmittedTemporalAdjustmentNodes, IdentificationError> {
        let ctx = antecedent_core::ExecutionContext::production_default(0);
        let mut budget =
            antecedent_core::TemporalBudget::new(ctx.temporal_materialization_limits());
        Self::adjustment_nodes_within(template, query, &mut budget, &ctx)
    }

    /// Resolve baseline parents under a caller-provided cumulative temporal budget.
    pub fn adjustment_nodes_within(
        template: &TemporalDag,
        query: &MediationQuery,
        budget: &mut antecedent_core::TemporalBudget,
        ctx: &antecedent_core::ExecutionContext,
    ) -> Result<AdmittedTemporalAdjustmentNodes, IdentificationError> {
        let lease = admit_mediation_scratch(template, query, budget, ctx)?;
        let nodes = Self::adjustment_nodes_admitted(template, query, budget, ctx)?;
        Ok(AdmittedTemporalAdjustmentNodes {
            inner: Arc::new(AdmittedTemporalAdjustmentNodesInner { nodes, _lease: lease }),
        })
    }

    // The caller retains the scratch admission across this entire traversal.
    fn adjustment_nodes_admitted(
        template: &TemporalDag,
        query: &MediationQuery,
        budget: &mut antecedent_core::TemporalBudget,
        ctx: &antecedent_core::ExecutionContext,
    ) -> Result<Vec<TemporalNodeKey>, IdentificationError> {
        if query.mediators.is_empty() {
            return Err(IdentificationError::unsupported(
                "temporal mediation requires at least one mediator",
            ));
        }
        let treatment = TemporalNodeKey { variable: query.treatment, offset: -1 };
        let mediator_keys: Vec<TemporalNodeKey> = query
            .mediators
            .iter()
            .map(|&mediator| TemporalNodeKey { variable: mediator, offset: 0 })
            .collect();
        let mut adjustment = Vec::new();
        let mut has_t_m = false;
        let mut has_m_y = false;
        let mut history = 1;
        let mut variable_count = query
            .mediators
            .iter()
            .map(|mediator| mediator.raw())
            .fold(query.treatment.raw().max(query.outcome.raw()), u32::max)
            .checked_add(1)
            .ok_or_else(|| IdentificationError::out_of_range("temporal variable count overflow"))?;
        for node in template.nodes() {
            if ctx.cancellation.is_cancelled() {
                return Err(IdentificationError::Cancelled);
            }
            if let NodeRef::Lagged { variable, .. } = node {
                let count = variable.raw().checked_add(1).ok_or_else(|| {
                    IdentificationError::out_of_range("temporal variable count overflow")
                })?;
                variable_count = variable_count.max(count);
            }
        }
        for edge in template.edges() {
            if ctx.cancellation.is_cancelled() {
                return Err(IdentificationError::Cancelled);
            }
            let (from, to) = edge.parent_child().expect("directed temporal edge");
            let (
                NodeRef::Lagged { variable: source, lag: source_lag },
                NodeRef::Lagged { variable: target, lag: target_lag },
            ) = (template.nodes()[from.as_usize()], template.nodes()[to.as_usize()])
            else {
                continue;
            };
            if !query.mediators.contains(&target) && target != query.outcome {
                continue;
            }
            let lag = source_lag.raw() - target_lag.raw();
            history = history.max(lag);
            let key = TemporalNodeKey {
                variable: source,
                offset: -i32::try_from(lag).map_err(|e| IdentificationError::construction(e))?,
            };
            has_t_m |= query.mediators.contains(&target) && key == treatment;
            has_m_y |= target == query.outcome && mediator_keys.contains(&key);
            if key != treatment && !mediator_keys.contains(&key) && !adjustment.contains(&key) {
                adjustment.push(key);
            }
        }
        if !has_t_m || !has_m_y {
            return Err(IdentificationError::unsupported(
                "temporal mediation requires T at lag one and M/Y contemporaneous",
            ));
        }
        let indexer = antecedent_data::TemporalIndexer::new(variable_count, history, 1)
            .map_err(|e| IdentificationError::construction(e))?;
        let dense = |key| indexer.dense_id(key).map_err(|e| IdentificationError::construction(e));
        let t = dense(treatment)?;
        let mediator_nodes =
            mediator_keys.iter().map(|&key| dense(key)).collect::<Result<Vec<_>, _>>()?;
        for &key in &adjustment {
            if ctx.cancellation.is_cancelled() {
                return Err(IdentificationError::Cancelled);
            }
            let z = dense(key)?;
            let mut induced = t == z || mediator_nodes.contains(&z);
            if !induced {
                induced = crate::temporal_window::reaches_in_window(
                    template,
                    treatment,
                    key,
                    variable_count,
                    history,
                    1,
                    budget,
                    ctx,
                )?;
            }
            if !induced {
                for &mediator_key in &mediator_keys {
                    if crate::temporal_window::reaches_in_window(
                        template,
                        mediator_key,
                        key,
                        variable_count,
                        history,
                        1,
                        budget,
                        ctx,
                    )? {
                        induced = true;
                        break;
                    }
                }
            }
            if induced {
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
    fn sparse_variable_reachability_obeys_caller_limits() {
        let mut graph = chain_template();
        let sparse = graph.add_lagged(VariableId::from_raw(16), Lag::from_raw(1)).unwrap();
        let outcome = antecedent_graph::DenseNodeId::from_index(2);
        graph.insert_directed(sparse, outcome).unwrap();
        let query = MediationQuery::binary(
            VariableId::from_raw(0),
            VariableId::from_raw(2),
            [VariableId::from_raw(1)],
            MediationContrast::Mediated,
        );
        let ctx = antecedent_core::ExecutionContext::for_tests(4);
        let mut limits = ctx.temporal_materialization_limits();
        limits.max_graph_nodes = 10;
        let mut budget = antecedent_core::TemporalBudget::new(limits);
        assert!(matches!(
            TemporalMediationIdentifier::adjustment_nodes_within(&graph, &query, &mut budget, &ctx),
            Err(IdentificationError::Budget { budget: crate::error::IdentificationBudget::Memory })
        ));
        limits.max_graph_nodes = 32;
        let mut budget = antecedent_core::TemporalBudget::new(limits);
        let parents =
            TemporalMediationIdentifier::adjustment_nodes_within(&graph, &query, &mut budget, &ctx)
                .unwrap();
        assert!(
            parents.contains(&TemporalNodeKey { variable: VariableId::from_raw(16), offset: -1 })
        );
        let mut small = ctx.temporal_materialization_limits();
        small.max_work = 1;
        let shared = antecedent_core::TemporalBudget::new(small);
        let identifier = TemporalMediationIdentifier {
            temporal: TemporalBackdoorIdentifier::new().with_temporal_context(&ctx, &shared),
            ..TemporalMediationIdentifier::new()
        };
        assert!(matches!(
            identifier.identify_with_horizon(&graph, &query, 1),
            Err(IdentificationError::Budget { .. })
        ));
        // Clones of the supplied allowance see the work charged by premise
        // checks and backdoor growth, rather than an independent default budget.
        let mut shared =
            antecedent_core::TemporalBudget::new(ctx.temporal_materialization_limits());
        let identifier = TemporalMediationIdentifier {
            temporal: TemporalBackdoorIdentifier::new().with_temporal_context(&ctx, &shared),
            ..TemporalMediationIdentifier::new()
        };
        identifier.identify_with_horizon(&chain_template(), &query, 1).unwrap();
        let entire_allowance = shared.limits().max_work;
        assert!(shared.charge_work(entire_allowance).is_err());
        ctx.cancellation.cancel();
        let mut budget = antecedent_core::TemporalBudget::new(limits);
        assert!(matches!(
            TemporalMediationIdentifier::adjustment_nodes_within(&graph, &query, &mut budget, &ctx),
            Err(IdentificationError::Cancelled)
        ));
    }

    #[test]
    fn adjustment_nodes_clones_retain_the_admission() {
        let query = MediationQuery::binary(
            VariableId::from_raw(0),
            VariableId::from_raw(2),
            [VariableId::from_raw(1)],
            MediationContrast::Mediated,
        );
        let pool = antecedent_core::execution::AdmissionPool::fixed(1, 100_000);
        let ctx = antecedent_core::ExecutionContext::for_tests(5).with_admission(pool.clone());
        let mut budget =
            antecedent_core::TemporalBudget::new(ctx.temporal_materialization_limits());
        let nodes = TemporalMediationIdentifier::adjustment_nodes_within(
            &chain_template(),
            &query,
            &mut budget,
            &ctx,
        )
        .unwrap();
        let cloned = nodes.clone();
        let remaining = pool.available().bytes();
        assert!(remaining < 100_000);
        drop(nodes);
        assert_eq!(pool.available().bytes(), remaining);
        drop(cloned);
        assert_eq!(pool.available().bytes(), 100_000);
    }

    #[test]
    fn extreme_sparse_variable_refuses_without_dense_allocation() {
        let mut graph = chain_template();
        let sparse =
            graph.add_lagged(VariableId::from_raw(1_000_000_000), Lag::from_raw(1)).unwrap();
        graph.insert_directed(sparse, antecedent_graph::DenseNodeId::from_index(2)).unwrap();
        let query = MediationQuery::binary(
            VariableId::from_raw(0),
            VariableId::from_raw(2),
            [VariableId::from_raw(1)],
            MediationContrast::Mediated,
        );
        assert!(matches!(
            TemporalMediationIdentifier::adjustment_nodes(&graph, &query),
            Err(IdentificationError::Budget { budget: crate::error::IdentificationBudget::Memory })
        ));
    }

    #[test]
    fn maximum_variable_id_refuses_before_unfolding() {
        let query = MediationQuery::binary(
            VariableId::from_raw(0),
            VariableId::from_raw(2),
            [VariableId::from_raw(u32::MAX)],
            MediationContrast::Mediated,
        );
        assert!(matches!(
            TemporalMediationIdentifier::adjustment_nodes(&chain_template(), &query),
            Err(IdentificationError::OutOfRange { .. })
        ));
        let mut graph = chain_template();
        graph.add_lagged(VariableId::from_raw(u32::MAX), Lag::CONTEMPORANEOUS).unwrap();
        let query = MediationQuery::binary(
            VariableId::from_raw(0),
            VariableId::from_raw(2),
            [VariableId::from_raw(1)],
            MediationContrast::Mediated,
        );
        assert!(matches!(
            TemporalMediationIdentifier::adjustment_nodes(&graph, &query),
            Err(IdentificationError::OutOfRange { .. })
        ));
    }

    #[test]
    fn earlier_mediator_on_causal_path_refuses_without_blocking_supported_horizons() {
        // T(-1) -> M(0) -> Y(1) via the lag-one M -> Y edge. The
        // estimator's M(1) coefficient alone cannot represent that path.
        let mut graph = TemporalDag::empty();
        let t1 = graph.add_lagged(VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
        let m0 = graph.add_lagged(VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
        let m1 = graph.add_lagged(VariableId::from_raw(1), Lag::from_raw(1)).unwrap();
        let y0 = graph.add_lagged(VariableId::from_raw(2), Lag::CONTEMPORANEOUS).unwrap();
        graph.insert_directed(t1, m0).unwrap();
        graph.insert_directed(m0, y0).unwrap();
        graph.insert_directed(m1, y0).unwrap();
        let q = MediationQuery::binary(
            VariableId::from_raw(0),
            VariableId::from_raw(2),
            [VariableId::from_raw(1)],
            MediationContrast::Mediated,
        );
        let id = TemporalMediationIdentifier::new();
        let error = id.identify_with_horizon(&graph, &q, 2).unwrap_err();
        assert!(
            matches!(error, IdentificationError::NotCertified { message } if message.contains("earlier"))
        );
        // At h=1 the same graph's mediated path goes through M at Y's slice.
        assert!(id.identify_with_horizon(&graph, &q, 1).is_ok());
        // At h=2 no earlier mediator lies on a causal path in the chain-only graph.
        assert!(id.identify_with_horizon(&chain_template(), &q, 2).is_ok());
        // An earlier M(0) reaching Y(1) through the modeled N(1) is covered.
        // Use an acyclic mediator chain: autoregressive templates have a separate
        // existing finite-history refusal and cannot serve as a positive control.
        let mut intercepted = chain_template();
        let prior_m = intercepted.add_lagged(VariableId::from_raw(1), Lag::from_raw(1)).unwrap();
        let current_n =
            intercepted.add_lagged(VariableId::from_raw(3), Lag::CONTEMPORANEOUS).unwrap();
        let current_y =
            intercepted.add_lagged(VariableId::from_raw(2), Lag::CONTEMPORANEOUS).unwrap();
        intercepted.insert_directed(prior_m, current_n).unwrap();
        intercepted.insert_directed(current_n, current_y).unwrap();
        let block_query = MediationQuery::binary(
            VariableId::from_raw(0),
            VariableId::from_raw(2),
            [VariableId::from_raw(1), VariableId::from_raw(3)],
            MediationContrast::Mediated,
        );
        let result = id.identify_with_horizon(&intercepted, &block_query, 2);
        assert!(result.is_ok(), "{result:?}");
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

    /// `T@-1 -> M1 -> M2 -> Y` with `T@-1 -> Y`: M1 and M2 are ordered mediators.
    fn ordered_mediators_template() -> TemporalDag {
        let mut g = TemporalDag::empty();
        let t1 = g.add_lagged(VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
        let m1 = g.add_lagged(VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
        let m2 = g.add_lagged(VariableId::from_raw(2), Lag::CONTEMPORANEOUS).unwrap();
        let y0 = g.add_lagged(VariableId::from_raw(3), Lag::CONTEMPORANEOUS).unwrap();
        g.insert_directed(t1, m1).unwrap();
        g.insert_directed(m1, m2).unwrap();
        g.insert_directed(m2, y0).unwrap();
        g.insert_directed(t1, y0).unwrap();
        g
    }

    #[test]
    fn identifies_an_ordered_mediator_set() {
        let g = ordered_mediators_template();
        let q = MediationQuery::binary(
            VariableId::from_raw(0),
            VariableId::from_raw(3),
            [VariableId::from_raw(1), VariableId::from_raw(2)],
            MediationContrast::Mediated,
        );
        let id = TemporalMediationIdentifier::new().identify(&g, &q).unwrap();
        assert!(matches!(id.status, IdentificationStatus::IdentifiedUnderParametricRestrictions));
        assert_eq!(id.estimands[0].mediators.len(), 2);
        // The ordered mediators are not adjustment covariates of one another.
        assert!(id.estimands[0].adjustment_set.is_empty());
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
