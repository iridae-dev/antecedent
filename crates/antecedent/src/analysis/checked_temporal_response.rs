//! Sealed temporal response execution for a fixed TemporalDag and dose × horizon grid.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{
    AssumptionSet, CausalResponse, ExecutionContext, IdentificationStatus, ResponseFunctional,
    ResponseQuery, TemporalEffectQuery, TemporalNodeKey,
};
use antecedent_data::{TemporalIndexer, TimeSeriesData};
use antecedent_expr::IdentifiedEstimand;
use antecedent_graph::{NodeRef, TemporalDag};
use antecedent_identify::result::IdentificationResult;

use crate::error::CausalError;
use crate::strategy_table::{EstimatorId, IdentifierId};

/// Proof and finite-unfolding context for one requested horizon.
#[derive(Clone, Debug)]
pub(crate) struct TemporalResponseHorizonEvidence {
    horizon: u32,
    effect_query: TemporalEffectQuery,
    identification: IdentificationResult,
    estimand: IdentifiedEstimand,
    indexer: TemporalIndexer,
}

impl TemporalResponseHorizonEvidence {
    /// Bind a temporal backdoor proof to the selected estimand for this horizon.
    pub(crate) fn checked(
        horizon: u32,
        effect_query: TemporalEffectQuery,
        identification: IdentificationResult,
        estimand: IdentifiedEstimand,
        indexer: TemporalIndexer,
    ) -> Result<Self, CausalError> {
        if !identification.estimands.iter().any(|candidate| {
            candidate.functional == estimand.functional
                && candidate.method == estimand.method
                && candidate.adjustment_set == estimand.adjustment_set
        }) {
            return Err(CausalError::Compile {
                message: "temporal response estimand is not supplied by its horizon proof".into(),
            });
        }
        let expected_offset = effect_query.outcome_offset();
        let treatment_key = TemporalNodeKey {
            variable: effect_query.treatment,
            offset: effect_query
                .try_treatment_offset()
                .map_err(|e| CausalError::Compile { message: e.to_string() })?,
        };
        let outcome_key =
            TemporalNodeKey { variable: effect_query.outcome, offset: expected_offset };
        let mapped_treatment = indexer.dense_id(treatment_key).ok();
        let mapped_outcome = indexer.dense_id(outcome_key).ok();
        let proof_query_matches = identification.average_effect().is_some_and(|proof_query| {
            let treatment_id = mapped_treatment.map(antecedent_core::VariableId::from_raw);
            Some(proof_query.treatment.raw()) == mapped_treatment
                && Some(proof_query.outcome.raw()) == mapped_outcome
                && proof_query.target_population == effect_query.target_population
                && treatment_id.is_some_and(|treatment_id| {
                    let set_matches =
                        |proof: &antecedent_core::Intervention,
                         requested: &antecedent_core::Intervention| {
                            matches!(
                                (proof, requested),
                                (
                                    antecedent_core::Intervention::Set {
                                        variable: proof_variable,
                                        value: proof_value
                                    },
                                    antecedent_core::Intervention::Set {
                                        variable: requested_variable,
                                        value: requested_value
                                    }
                                ) if *proof_variable == treatment_id
                                    && *requested_variable == effect_query.treatment
                                    && proof_value == requested_value
                            )
                        };
                    set_matches(&proof_query.control, &effect_query.control)
                        && set_matches(&proof_query.active, &effect_query.active)
                })
        });
        if effect_query.horizon_steps != horizon {
            return Err(CausalError::Compile {
                message: "temporal response proof is bound to a different outcome horizon".into(),
            });
        }
        if !proof_query_matches {
            return Err(CausalError::Compile {
                message: format!(
                    "temporal response proof query does not bind to its retained horizon keys and intervention levels: proof={:?}, retained={effect_query:?}",
                    identification.query
                ),
            });
        }
        if indexer.dense_id(treatment_key).is_err() || indexer.dense_id(outcome_key).is_err() {
            return Err(CausalError::Compile {
                message: format!(
                    "temporal response proof indexer lacks treatment/outcome keys: treatment={treatment_key:?}, outcome={outcome_key:?}"
                ),
            });
        }
        Ok(Self { horizon, effect_query, identification, estimand, indexer })
    }

    /// Requested horizon retained by this proof member.
    #[must_use]
    #[cfg(test)]
    pub(crate) const fn horizon(&self) -> u32 {
        self.horizon
    }

    /// Retained finite-unfolding proof.
    #[must_use]
    pub(crate) const fn identification(&self) -> &IdentificationResult {
        &self.identification
    }

    /// Retained target and dense adjustment IDs.
    #[must_use]
    pub(crate) const fn estimand(&self) -> &IdentifiedEstimand {
        &self.estimand
    }

    /// Retained dense-to-temporal mapping.
    #[must_use]
    pub(crate) const fn indexer(&self) -> &TemporalIndexer {
        &self.indexer
    }
}

/// Checked TemporalDag dose-by-horizon response operation.
///
/// The operation fixes the ordered response grid, every horizon's identification
/// proof and estimand, the temporal indexer used to bind lagged columns, and the
/// frequentist procedure/uncertainty method before execution.
#[derive(Clone, Debug)]
pub(crate) struct CheckedTemporalResponseOperation {
    graph: TemporalDag,
    graph_signature: TemporalGraphSignature,
    query: ResponseQuery,
    evidence: Arc<[TemporalResponseHorizonEvidence]>,
    assumptions: AssumptionSet,
    identifier: IdentifierId,
    estimator: EstimatorId,
    fitter: antecedent_estimate::TemporalResponseEstimator,
    bootstrap_replicates: u32,
    grid: Arc<[f64]>,
}

impl CheckedTemporalResponseOperation {
    /// Seal a point-identified frequentist temporal response route.
    pub(crate) fn checked(
        graph: &TemporalDag,
        query: &ResponseQuery,
        evidence: Vec<TemporalResponseHorizonEvidence>,
        identifier: IdentifierId,
        estimator: EstimatorId,
        bootstrap_replicates: u32,
    ) -> Result<Self, CausalError> {
        query.validate().map_err(|error| CausalError::Compile { message: error.to_string() })?;
        let spec = query.temporal.as_ref().ok_or_else(|| CausalError::Compile {
            message: "checked temporal response requires a temporal response specification".into(),
        })?;
        let (treatment, outcome, grid) = match &query.functional {
            ResponseFunctional::MeanCurve { treatment, outcome } => {
                let values = treatment
                    .grid
                    .values()
                    .map_err(|error| CausalError::Compile { message: error.to_string() })?;
                (treatment.variable, *outcome, values)
            }
            _ => {
                return Err(CausalError::Unsupported {
                    message: "checked temporal operation currently supports MeanCurve only",
                });
            }
        };
        if grid.is_empty()
            || grid.len().saturating_mul(spec.horizons.len())
                > antecedent_core::MAX_TEMPORAL_RESPONSE_CELLS
        {
            return Err(CausalError::Unsupported {
                message: "temporal response dose-by-horizon grid exceeds its materialization limit",
            });
        }
        if identifier != IdentifierId::TemporalBackdoorUnfolded
            || estimator != EstimatorId::TemporalResponseGcomp
        {
            return Err(CausalError::Compile {
                message: "checked temporal MeanCurve requires temporal backdoor and temporal response g-computation".into(),
            });
        }
        if spec.horizons.as_ref()
            != evidence.iter().map(|member| member.horizon).collect::<Vec<_>>()
        {
            return Err(CausalError::Compile {
                message:
                    "temporal response proof members must match the ordered requested horizons"
                        .into(),
            });
        }
        if evidence.is_empty() {
            return Err(CausalError::Compile {
                message: "temporal response requires one identification proof per horizon".into(),
            });
        }
        let mut assumptions = AssumptionSet::new();
        let mut status = None;
        for member in &evidence {
            if member.effect_query.treatment != treatment
                || member.effect_query.outcome != outcome
                || member.effect_query.horizon_steps != member.horizon
                || member.effect_query.policy != spec.policy
            {
                return Err(CausalError::Compile {
                    message: "temporal response query and horizon proof disagree on treatment, outcome, policy, or horizon".into(),
                });
            }
            if !matches!(
                member.identification.status,
                IdentificationStatus::NonparametricallyIdentified
                    | IdentificationStatus::IdentifiedUnderParametricRestrictions
            ) {
                return Err(CausalError::Unsupported {
                    message: "temporal response operation requires point identification at every horizon",
                });
            }
            if status.is_some_and(|known| known != member.identification.status) {
                return Err(CausalError::Compile {
                    message: "temporal response horizon proofs disagree on identification status"
                        .into(),
                });
            }
            status = Some(member.identification.status);
            assumptions.extend_unique(&member.identification.required_assumptions.entries);
        }
        let signature = temporal_graph_signature(graph);
        let fitter = antecedent_estimate::TemporalResponseEstimator::new()
            .with_bootstrap_replicates(bootstrap_replicates);
        Ok(Self {
            graph: graph.clone(),
            graph_signature: signature,
            query: query.clone(),
            evidence: evidence.into(),
            assumptions,
            identifier,
            estimator,
            fitter,
            bootstrap_replicates,
            grid: grid.into(),
        })
    }

    /// Exact retained query and ordered dose grid.
    #[must_use]
    pub(crate) fn query(&self) -> &ResponseQuery {
        &self.query
    }

    /// Primary horizon identification copied into the full result claim.
    #[must_use]
    pub(crate) fn primary_identification(&self) -> &IdentificationResult {
        &self.evidence[0].identification
    }

    /// Primary horizon target copied into the full result claim.
    #[must_use]
    pub(crate) fn primary_estimand(&self) -> &IdentifiedEstimand {
        &self.evidence[0].estimand
    }

    /// Ordered requested doses.
    #[must_use]
    pub(crate) fn grid(&self) -> &[f64] {
        &self.grid
    }

    /// Ordered horizon-proof members.
    #[must_use]
    pub(crate) fn evidence(&self) -> &[TemporalResponseHorizonEvidence] {
        &self.evidence
    }

    /// Fixed method, backend and uncertainty identity.
    #[must_use]
    pub(crate) fn procedure(&self) -> (IdentifierId, EstimatorId, &'static str, u32) {
        let uncertainty = if self.bootstrap_replicates == 0 {
            "frequentist.no_interval"
        } else {
            "frequentist.circular_block_bootstrap"
        };
        (self.identifier, self.estimator, uncertainty, self.bootstrap_replicates)
    }

    /// Verify that refresh keeps the same temporal graph.
    #[must_use]
    pub(crate) fn matches_graph(&self, graph: &TemporalDag) -> bool {
        temporal_graph_signature(graph) == self.graph_signature
            && temporal_graph_signature(&self.graph) == self.graph_signature
    }

    /// Execute from retained proof members and procedure, without re-identification.
    pub(crate) fn execute(
        &self,
        data: &TimeSeriesData,
        ctx: &ExecutionContext,
    ) -> Result<CausalResponse, CausalError> {
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled {
                stage: crate::analysis::stage::STAGE_ESTIMATE_POINT,
            });
        }
        let pairs = self
            .evidence
            .iter()
            .map(|member| (&member.estimand, &member.indexer))
            .collect::<Vec<_>>();
        let status = self.evidence[0].identification.status;
        self.fitter
            .estimate(data, &pairs, &self.query, status, self.assumptions.clone(), ctx)
            .map_err(CausalError::from)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct TemporalGraphSignature {
    nodes: Arc<[(u32, u32)]>,
    edges: Arc<[(u32, u32)]>,
}

fn temporal_graph_signature(graph: &TemporalDag) -> TemporalGraphSignature {
    let nodes = graph
        .nodes()
        .iter()
        .map(|node| match node {
            NodeRef::Lagged { variable, lag } => (variable.raw(), lag.raw()),
            _ => unreachable!("TemporalDag stores lagged nodes only"),
        })
        .collect::<Vec<_>>();
    let mut edges = graph.edges().map(|edge| (edge.a.raw(), edge.b.raw())).collect::<Vec<_>>();
    edges.sort_unstable();
    TemporalGraphSignature { nodes: nodes.into(), edges: edges.into() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use antecedent_core::{
        ContinuousDomain, GridSpec, Lag, ResponseFunctional, TemporalEffectQuery, TemporalPolicy,
        TemporalResponseSpec, VariableId,
    };
    use antecedent_data::TimeSeriesData;
    use antecedent_graph::{TemporalDag, ensure_lagged};
    use antecedent_identify::temporal_backdoor::TemporalBackdoorIdentifier;

    fn operation(bootstrap_replicates: u32) -> (CheckedTemporalResponseOperation, TimeSeriesData) {
        let n = 800;
        let treatment = (0..n).map(|i| ((i * 31 % 997) as f64 - 498.0) / 250.0).collect::<Vec<_>>();
        let outcome = std::iter::once(0.0)
            .chain(treatment.iter().take(n - 1).map(|value| 1.5 + 2.0 * value))
            .collect::<Vec<_>>();
        let data = TimeSeriesData::from_f64_columns(
            [("t", treatment.as_slice()), ("y", outcome.as_slice())],
            1,
        )
        .unwrap();
        let t = VariableId::from_raw(0);
        let y = VariableId::from_raw(1);
        let mut graph = TemporalDag::empty();
        let t_lag = ensure_lagged(&mut graph, t, Lag::from_raw(1)).unwrap();
        let y_now = ensure_lagged(&mut graph, y, Lag::CONTEMPORANEOUS).unwrap();
        graph.insert_directed(t_lag, y_now).unwrap();
        let policy = TemporalPolicy::pulse(-1);
        let grid = GridSpec::Values(Arc::from([-0.5, 0.0, 0.5]));
        let query = ResponseQuery::new(ResponseFunctional::MeanCurve {
            outcome: y,
            treatment: ContinuousDomain::new(t, grid),
        })
        .with_temporal(TemporalResponseSpec::new(vec![1], policy.clone(), None).unwrap());
        let mut effect = TemporalEffectQuery::pulse(t, y, 1.0).with_horizon_steps(1);
        effect.policy = policy;
        let identified =
            TemporalBackdoorIdentifier::new().identify_temporal(&graph, &effect).unwrap();
        let estimand = identified.result.estimands[0].clone();
        let member = TemporalResponseHorizonEvidence::checked(
            1,
            effect,
            identified.result,
            estimand,
            identified.indexer,
        )
        .unwrap();
        let operation = CheckedTemporalResponseOperation::checked(
            &graph,
            &query,
            vec![member],
            IdentifierId::TemporalBackdoorUnfolded,
            EstimatorId::TemporalResponseGcomp,
            bootstrap_replicates,
        )
        .unwrap();
        (operation, data)
    }

    #[test]
    fn temporal_curve_uses_retained_grid_and_horizon_proof_against_linear_truth() {
        let (operation, data) = operation(0);
        let ctx = ExecutionContext::for_tests(41);
        let response = operation.execute(&data, &ctx).unwrap();
        assert_eq!(operation.grid(), [-0.5, 0.0, 0.5]);
        assert_eq!(operation.evidence().iter().map(|m| m.horizon()).collect::<Vec<_>>(), [1]);
        let antecedent_core::ResponseIdentification::PointIdentified(
            antecedent_core::ResponseValue::Surface { mean, .. },
        ) = response.estimate
        else {
            panic!("mean-curve operation returns a surface");
        };
        for (value, level) in mean.iter().zip([-0.5, 0.0, 0.5]) {
            assert!((*value - (1.5 + 2.0 * level)).abs() < 0.08);
        }
        assert_eq!(operation.procedure().3, 0);
    }

    #[test]
    fn operation_rejects_a_different_temporal_graph() {
        let (operation, _) = operation(0);
        let mut changed = TemporalDag::empty();
        let t = ensure_lagged(&mut changed, VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
        let y = ensure_lagged(&mut changed, VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
        let z = ensure_lagged(&mut changed, VariableId::from_raw(1), Lag::from_raw(1)).unwrap();
        changed.insert_directed(t, y).unwrap();
        changed.insert_directed(z, y).unwrap();
        assert!(!operation.matches_graph(&changed));
    }
}
