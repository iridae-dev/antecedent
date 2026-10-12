//! Shared prepare-time metadata for checked static operations.
//!
//! The block is behind an [`Arc`] so rebind replaces a data binding without
//! cloning the plan, registry, or validator names.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use super::{
    EstimatorId, GraphClass, IdentifierId, InferenceMode, PhysicalExecutionPlan, RefuteSuite, Study,
};
use crate::analysis::latency::LatencyMode;

/// Frozen procedure, graph, and result metadata shared by checked static operations.
///
/// `latency_bootstrap_not_applied` belongs with this block. Copies that omitted
/// it still store the study value here; only routes that already published the
/// diagnostic read it.
#[derive(Clone, Debug)]
pub(crate) struct CheckedStaticCommon {
    pub(crate) identifier: IdentifierId,
    pub(crate) estimator: EstimatorId,
    pub(crate) physical: std::sync::Arc<PhysicalExecutionPlan>,
    pub(crate) inference: InferenceMode,
    pub(crate) refute: RefuteSuite,
    pub(crate) graph_class: GraphClass,
    pub(crate) graph_version: u32,
    pub(crate) support_status: Option<crate::support::CellStatus>,
    pub(crate) structure_source: crate::support::StructureSource,
    pub(crate) population_registry: Option<antecedent_core::PopulationRegistry>,
    pub(crate) latency_mode: Option<LatencyMode>,
    pub(crate) custom_validator_names: Arc<[Arc<str>]>,
    /// `(tier replicates, configured replicates)` when a latency tier's count
    /// was set aside for the configured estimator's own count.
    pub(crate) latency_bootstrap_not_applied: Option<(u32, u32)>,
}

impl CheckedStaticCommon {
    /// Freeze the study fields every checked static operation used to copy by hand.
    pub(crate) fn from_analysis(
        analysis: &Study,
        physical: &std::sync::Arc<PhysicalExecutionPlan>,
        identifier: IdentifierId,
        estimator: EstimatorId,
    ) -> Arc<Self> {
        Arc::new(Self {
            identifier,
            estimator,
            physical: physical.clone(),
            inference: analysis.inference.clone(),
            refute: analysis.refute,
            graph_class: analysis.graph.class(),
            graph_version: analysis.graph.version(),
            support_status: analysis.support_status,
            structure_source: analysis.structure_source,
            population_registry: analysis.population_registry.clone(),
            latency_mode: analysis.latency_mode,
            custom_validator_names: Arc::from(
                analysis
                    .custom_validators
                    .iter()
                    .map(|validator| Arc::from(validator.name()))
                    .collect::<Vec<_>>(),
            ),
            latency_bootstrap_not_applied: analysis.latency_bootstrap_not_applied,
        })
    }
}
