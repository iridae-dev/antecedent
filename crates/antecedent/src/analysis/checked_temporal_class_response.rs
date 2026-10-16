//! Retained completion proofs and execution settings for an incomplete temporal
//! graph response (mean curve or single-step intervention response).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use crate::analysis::prepared::Inspect;
use crate::analysis::route_guards::{complete_mean_response, inference_label};
use antecedent_core::{CausalQuery, ResponseQuery, TemporalEffectQuery, TemporalNodeKey};
use std::sync::Arc;

use crate::error::CausalError;
use crate::planner::PhysicalExecutionPlan;
use crate::strategy_table::{EstimatorId, IdentifierId};
use crate::{AcceptedGraph, ClassPrior, GraphClass, InferenceMode};

use super::builder::RefuteSuite;
use super::checked_temporal_response::temporal_response_is_direct;
use super::prepared::CachedTemporalClassIdentification;
use super::proof_support::{Procedure, require_primary_pair};
use super::route_guards::same_accepted_graph;

/// A complete, checked completion envelope per requested horizon for a
/// `TemporalCpdag` or `TemporalPag` response. The per-horizon envelopes and their
/// aligned unfolding indexers stay together with the source graph, so no
/// completion can be evaluated against another completion's time window and no
/// execution can re-derive the class from a different graph.
#[derive(Clone)]
pub(crate) struct CheckedTemporalClassResponseOperation {
    source_graph: AcceptedGraph,
    max_completions: Option<usize>,
    query: ResponseQuery,
    bundle: Arc<CachedTemporalClassIdentification>,
    identifier: IdentifierId,
    estimator: EstimatorId,
    inference: InferenceMode,
    bootstrap_replicates: u32,
    class_prior: Option<ClassPrior>,
    validation: RefuteSuite,
    physical: std::sync::Arc<PhysicalExecutionPlan>,
}

/// The frozen premises of [`CheckedTemporalClassResponseOperation::checked`], carried together.
pub(crate) struct CheckedTemporalClassResponseOperationPremises<'a> {
    pub(crate) source_graph: AcceptedGraph,
    pub(crate) query: &'a ResponseQuery,
    pub(crate) bundle: &'a Arc<CachedTemporalClassIdentification>,
    pub(crate) max_completions: Option<usize>,
    pub(crate) inference: &'a InferenceMode,
    pub(crate) bootstrap_replicates: u32,
    pub(crate) class_prior: Option<ClassPrior>,
    pub(crate) validation: RefuteSuite,
    pub(crate) physical: std::sync::Arc<PhysicalExecutionPlan>,
}

impl std::fmt::Debug for CheckedTemporalClassResponseOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CheckedTemporalClassResponseOperation")
            .field("graph_class", &self.source_graph.class())
            .field("structure_version", &self.source_graph.version())
            .field("query", &self.query)
            .field("horizon_count", &self.bundle.by_horizon.len())
            .field("identifier", &self.identifier)
            .field("estimator", &self.estimator)
            .field("bootstrap_replicates", &self.bootstrap_replicates)
            .field("validation", &self.validation)
            .finish_non_exhaustive()
    }
}

/// Checks one horizon's completion proof: it must hold completions with one
/// unfolding indexer each, and every completion's certificate must bind back
/// to the requested treatment, outcome, policy origin and horizon.
fn check_class_response_horizon(
    bundle: &CachedTemporalClassIdentification,
    spec: &antecedent_core::TemporalResponseSpec,
    query: &ResponseQuery,
    (treatment, outcome): (antecedent_core::VariableId, antecedent_core::VariableId),
    horizon: u32,
) -> Result<(), CausalError> {
    let (_, envelope) = bundle
        .by_horizon
        .iter()
        .find(|(cached, _)| *cached == horizon)
        .ok_or_else(|| CausalError::Compile {
            message: format!(
                "temporal class response proof lacks a completion envelope for horizon {horizon}"
            ),
        })?;
    if envelope.envelope.cases.is_empty() {
        return Err(CausalError::Compile {
            message: format!(
                "temporal class response proof has no completion cases at horizon {horizon}"
            ),
        });
    }
    if envelope.indexers.len() != envelope.envelope.cases.len() {
        return Err(CausalError::Compile {
            message:
                "temporal class response proof does not bind one unfolding indexer per completion"
                    .into(),
        });
    }
    // Every completion proof is an unfolded (dense) adjustment certificate;
    // bind its treatment/outcome nodes back through that completion's own
    // indexer to the requested variables, policy origin, and horizon.
    let witness = TemporalEffectQuery::pulse(treatment, outcome, 1.0)
        .with_policy(spec.policy.clone())
        .with_horizon_steps(horizon)
        .with_max_history_lag(spec.max_history_lag)
        .with_target_population(query.target_population.clone());
    let treatment_key =
        TemporalNodeKey { variable: treatment, offset: witness.try_treatment_offset()? };
    let outcome_key = TemporalNodeKey { variable: outcome, offset: witness.outcome_offset() };
    let witness_binds =
        envelope.envelope.cases.iter().zip(&envelope.indexers).all(|(case, indexer)| {
            case.result.average_effect().is_none_or(|proof| {
                Some(proof.treatment.raw()) == indexer.dense_id(treatment_key).ok()
                    && Some(proof.outcome.raw()) == indexer.dense_id(outcome_key).ok()
            })
        });
    if !witness_binds {
        return Err(CausalError::Compile {
                message: "temporal class response proof was certified for a different treatment, outcome, policy, or horizon".into(),
            });
    }
    Ok(())
}

impl CheckedTemporalClassResponseOperation {
    /// Check and freeze the per-horizon completion proofs and selected procedure.
    pub(crate) fn checked(
        premises: CheckedTemporalClassResponseOperationPremises<'_>,
    ) -> Result<Self, CausalError> {
        let CheckedTemporalClassResponseOperationPremises {
            source_graph,
            query,
            bundle,
            max_completions,
            inference,
            bootstrap_replicates,
            class_prior,
            validation,
            physical,
        } = premises;
        query.validate()?;
        if !matches!(source_graph.class(), GraphClass::TemporalCpdag | GraphClass::TemporalPag) {
            return Err(CausalError::Compile {
                message: "temporal class response requires a TemporalCpdag or TemporalPag proof"
                    .into(),
            });
        }
        let spec = query.temporal.as_ref().ok_or_else(|| CausalError::Compile {
            message: "temporal class response requires a temporal response specification".into(),
        })?;
        if !temporal_response_is_direct(query) {
            return Err(CausalError::Unsupported {
                message: "checked temporal class response supports MeanCurve and single-step InterventionResponse only",
            });
        }
        if !complete_mean_response(query) {
            return Err(CausalError::Unsupported {
                message: "checked temporal class response requires complete observation, the all-observed population, and the mean outcome functional",
            });
        }
        if !spec.policy.is_single_step() {
            return Err(CausalError::Unsupported {
                message: "checked temporal class response supports Pulse or single-step Sustained policies",
            });
        }
        if validation != RefuteSuite::None {
            return Err(CausalError::Unsupported {
                message: "checked temporal class response supports validation suite none only",
            });
        }
        let (treatment, outcome) = require_primary_pair(&query.functional)?;
        let identifier = IdentifierId::GeneralizedAdjustment;
        let estimator = EstimatorId::temporal_response_for(inference);
        if physical.logical.query != CausalQuery::Response(query.clone())
            || !physical.binding().names(identifier, estimator)
        {
            return Err(CausalError::Compile {
                message:
                    "temporal class response plan disagrees with its retained query or procedure"
                        .into(),
            });
        }
        for &horizon in spec.horizons.iter() {
            check_class_response_horizon(bundle, spec, query, (treatment, outcome), horizon)?;
        }
        Ok(Self {
            source_graph,
            max_completions,
            query: query.clone(),
            bundle: Arc::clone(bundle),
            identifier,
            estimator,
            inference: inference.clone(),
            bootstrap_replicates,
            class_prior,
            validation,
            physical,
        })
    }

    #[must_use]
    pub(crate) fn source_graph(&self) -> &AcceptedGraph {
        &self.source_graph
    }

    #[must_use]
    pub(crate) fn graph_class(&self) -> GraphClass {
        self.source_graph.class()
    }

    #[must_use]
    pub(crate) fn structure_version(&self) -> u32 {
        self.source_graph.version()
    }

    /// Verify that a click or refresh still carries the source graph the proof
    /// was enumerated from.
    #[must_use]
    pub(crate) fn matches_source_graph(&self, graph: &AcceptedGraph) -> bool {
        same_accepted_graph(&self.source_graph, graph)
    }

    #[must_use]
    pub(crate) fn max_completions(&self) -> Option<usize> {
        self.max_completions
    }

    #[must_use]
    pub(crate) fn query(&self) -> &ResponseQuery {
        &self.query
    }

    #[must_use]
    pub(crate) fn bundle(&self) -> &Arc<CachedTemporalClassIdentification> {
        &self.bundle
    }

    #[must_use]
    pub(crate) const fn inference(&self) -> &InferenceMode {
        &self.inference
    }

    #[must_use]
    pub(crate) fn procedure(&self) -> Procedure {
        Procedure {
            identifier: self.identifier,
            estimator: self.estimator,
            bootstrap_replicates: self.bootstrap_replicates,
            validation: self.validation,
        }
    }

    #[must_use]
    pub(crate) fn class_prior(&self) -> Option<&ClassPrior> {
        self.class_prior.as_ref()
    }

    #[must_use]
    pub(crate) fn physical(&self) -> &PhysicalExecutionPlan {
        &self.physical
    }

    /// Ordered requested horizons.
    #[must_use]
    pub(crate) fn horizons(&self) -> &[u32] {
        self.query.temporal.as_ref().map_or(&[], |spec| spec.horizons.as_ref())
    }

    /// Number of completion cases retained for the first requested horizon.
    #[must_use]
    pub(crate) fn completion_count(&self) -> usize {
        self.horizons()
            .first()
            .and_then(|horizon| self.bundle.by_horizon.iter().find(|(cached, _)| cached == horizon))
            .map_or(0, |(_, envelope)| envelope.envelope.cases.len())
    }

    /// Identified and unidentified enumeration mass retained for the first horizon.
    #[must_use]
    pub(crate) fn enumeration_mass(&self) -> (f64, f64) {
        self.horizons()
            .first()
            .and_then(|horizon| self.bundle.by_horizon.iter().find(|(cached, _)| cached == horizon))
            .map_or((0.0, 0.0), |(_, envelope)| {
                (envelope.envelope.identified_weight.0, envelope.envelope.unidentified_weight.0)
            })
    }
}

/// Read-only source graph, completion proof, and procedure of a retained
/// TemporalCpdag/TemporalPag response.
#[derive(Clone, Debug)]
pub struct CheckedTemporalClassResponseInfo {
    /// Exact response functional, policy, and requested horizons.
    pub query: antecedent_core::ResponseQuery,
    /// Source class retained with the completion proof.
    pub graph_class: crate::accepted::GraphClass,
    /// Identifier fixed during preparation.
    pub identifier: crate::strategy_table::IdentifierId,
    /// Estimator fixed during preparation.
    pub estimator: crate::strategy_table::EstimatorId,
    /// Inference mode label fixed during preparation.
    pub inference: std::sync::Arc<str>,
    /// Validation suite fixed during preparation.
    pub validation: crate::analysis::builder::RefuteSuite,
    /// Circular-block bootstrap budget fixed during preparation.
    pub bootstrap_replicates: u32,
    /// Ordered requested horizons, each with a retained completion envelope.
    pub horizons: std::sync::Arc<[u32]>,
    /// Number of completion atoms retained for the first requested horizon.
    pub completion_count: usize,
    /// Maximum number of class completions searched, when configured.
    pub completion_limit: Option<usize>,
    /// Identified enumeration mass retained for the first requested horizon.
    pub identified_mass: f64,
    /// Unidentified enumeration mass retained for the first requested horizon.
    pub unidentified_mass: f64,
    /// Source graph version sealed with the proof.
    pub graph_version: u64,
}

impl Inspect for CheckedTemporalClassResponseOperation {
    type Info = CheckedTemporalClassResponseInfo;

    fn info(&self) -> CheckedTemporalClassResponseInfo {
        let Procedure { identifier, estimator, bootstrap_replicates, validation } =
            self.procedure();
        let (identified_mass, unidentified_mass) = self.enumeration_mass();
        CheckedTemporalClassResponseInfo {
            query: self.query().clone(),
            graph_class: self.graph_class(),
            identifier,
            estimator,
            inference: inference_label(self.inference()),
            validation,
            bootstrap_replicates,
            horizons: Arc::from(self.horizons()),
            completion_count: self.completion_count(),
            completion_limit: self.max_completions(),
            identified_mass,
            unidentified_mass,
            graph_version: u64::from(self.structure_version()),
        }
    }
}
