//! Unified `CausalAnalysis` facade (DESIGN.md §21).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

//! Builder types.

#![allow(
    clippy::similar_names,
    clippy::too_many_lines,
    clippy::doc_markdown,
    clippy::too_many_arguments,
    clippy::cast_precision_loss
)]


use causal_core::{
    AverageEffectQuery, CausalQuery,
    TemporalEffectQuery, VariableId,
};
use causal_data::{DiscoveryEstimationSplit, MultiEnvironmentData, TabularData, TimeSeriesData};
use causal_discovery::{MultiDatasetConstraints, RegimeAssignment};
use causal_estimate::OverlapPolicy;
use causal_graph::{Dag, Pag, TemporalDag, TemporalPag};

use crate::error::AnalysisError;
use crate::inference::InferenceMode;
use crate::planner::GraphInput;
use crate::strategy_table::{
    EstimatorId,
    IdentifierId,
};

use super::execute::CausalAnalysis;

/// Which refuters to run (static ATE path).
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum RefuteSuite {
    /// Skip refutation.
    None,
    /// Placebo + random common cause (linear backdoor only).
    PlaceboAndRcc,
    /// Full validation suite (applicable validators only; others NotApplicable).
    Full,
}

#[derive(Clone, Debug)]
pub(crate) enum DataInput {
    Tabular(TabularData),
    Temporal(TimeSeriesData),
    /// Multi-environment series (J-PCMCI+ discover path).
    MultiEnv(MultiEnvironmentData),
}

/// Running-variable configuration for the `rd.sharp` estimator; required when `rd.sharp` is
/// selected as the estimator (see [`CausalAnalysisBuilder::rd_config`]).
#[derive(Clone, Copy, Debug)]
pub struct RdConfig {
    /// Running (assignment) variable.
    pub running_variable: VariableId,
    /// Discontinuity cutoff.
    pub cutoff: f64,
    /// Symmetric bandwidth around the cutoff (`|R − cutoff| ≤ bandwidth` is retained).
    pub bandwidth: f64,
}

/// Builder for static or temporal analysis.
#[derive(Clone, Debug)]
pub struct CausalAnalysisBuilder {
    data: Option<DataInput>,
    graph: Option<GraphInput>,
    query: Option<CausalQuery>,
    refute: RefuteSuite,
    bootstrap_replicates: u32,
    split: Option<DiscoveryEstimationSplit>,
    identifier: Option<IdentifierId>,
    estimator: Option<EstimatorId>,
    rd: Option<RdConfig>,
    inference: InferenceMode,
    /// Optional override for propensity / AIPW overlap (clip/trim). `None` keeps estimator defaults.
    overlap_policy: Option<OverlapPolicy>,
}

impl Default for CausalAnalysisBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl CausalAnalysisBuilder {
    /// Start a builder.
    #[must_use]
    pub fn new() -> Self {
        Self {
            data: None,
            graph: None,
            query: None,
            refute: RefuteSuite::PlaceboAndRcc,
            bootstrap_replicates: 100,
            split: None,
            identifier: None,
            estimator: None,
            rd: None,
            inference: InferenceMode::Frequentist,
            overlap_policy: None,
        }
    }

    /// Supply tabular data.
    #[must_use]
    pub fn data(mut self, data: TabularData) -> Self {
        self.data = Some(DataInput::Tabular(data));
        self
    }

    /// Supply temporal series data.
    #[must_use]
    pub fn series(mut self, data: TimeSeriesData) -> Self {
        self.data = Some(DataInput::Temporal(data));
        self
    }

    /// Supply multi-environment series (required for J-PCMCI+ discovery).
    #[must_use]
    pub fn series_multi(mut self, data: MultiEnvironmentData) -> Self {
        self.data = Some(DataInput::MultiEnv(data));
        self
    }

    /// Supply a validated static DAG.
    #[must_use]
    pub fn graph(mut self, graph: Dag) -> Self {
        self.graph = Some(GraphInput::Static(graph));
        self
    }

    /// Supply a temporal DAG template.
    #[must_use]
    pub fn temporal_graph(mut self, graph: TemporalDag) -> Self {
        self.graph = Some(GraphInput::Temporal(graph));
        self
    }

    /// Discover with PCMCI (typically yields [`CompiledAnalysis::ReviewRequired`]).
    #[must_use]
    pub fn discover_pcmci(
        mut self,
        max_lag: u32,
        alpha: f64,
        fdr: crate::options::FdrControl,
        accept: crate::options::DiscoveryAccept,
    ) -> Self {
        self.graph = Some(GraphInput::DiscoverPcmci {
            max_lag,
            alpha,
            fdr: fdr.adjustment(),
            accept_discovered: accept.auto(),
        });
        self
    }

    /// Discover with PCMCI+ (typically yields [`CompiledAnalysis::ReviewRequiredCpdag`]).
    ///
    /// `accept` only auto-completes when the oriented CPDAG has no undirected marks;
    /// otherwise compile still returns review-required (no silent coercion).
    #[must_use]
    pub fn discover_pcmci_plus(
        mut self,
        max_lag: u32,
        alpha: f64,
        fdr: crate::options::FdrControl,
        accept: crate::options::DiscoveryAccept,
    ) -> Self {
        self.graph = Some(GraphInput::DiscoverPcmciPlus {
            max_lag,
            alpha,
            fdr: fdr.adjustment(),
            accept_discovered: accept.auto(),
        });
        self
    }

    /// Discover with J-PCMCI+ (requires [`Self::series_multi`]; typically review-required).
    #[must_use]
    pub fn discover_jpcmci_plus(
        mut self,
        max_lag: u32,
        alpha: f64,
        fdr: crate::options::FdrControl,
        accept: crate::options::DiscoveryAccept,
        multi_dataset: MultiDatasetConstraints,
    ) -> Self {
        self.graph = Some(GraphInput::DiscoverJpcmciPlus {
            max_lag,
            alpha,
            fdr: fdr.adjustment(),
            accept_discovered: accept.auto(),
            multi_dataset,
        });
        self
    }

    /// Discover with RPCMCI (requires caller-supplied regime assignment).
    #[must_use]
    pub fn discover_rpcmci(
        mut self,
        max_lag: u32,
        alpha: f64,
        fdr: crate::options::FdrControl,
        accept: crate::options::DiscoveryAccept,
        regime_assignment: RegimeAssignment,
    ) -> Self {
        self.graph = Some(GraphInput::DiscoverRpcmci {
            max_lag,
            alpha,
            fdr: fdr.adjustment(),
            accept_discovered: accept.auto(),
            regime_assignment,
        });
        self
    }

    /// Discover with LPCMCI (temporal PAG; typically [`CompiledAnalysis::ReviewRequiredPag`]).
    #[must_use]
    pub fn discover_lpcmci(
        mut self,
        max_lag: u32,
        alpha: f64,
        fdr: crate::options::FdrControl,
        accept: crate::options::DiscoveryAccept,
    ) -> Self {
        self.graph = Some(GraphInput::DiscoverLpcmci {
            max_lag,
            alpha,
            fdr: fdr.adjustment(),
            accept_discovered: accept.auto(),
        });
        self
    }

    /// Discover with static PC (tabular CPDAG; auto-finishes only when fully oriented).
    #[must_use]
    pub fn discover_pc(
        mut self,
        alpha: f64,
        max_cond_size: usize,
        fdr: crate::options::FdrControl,
        accept: crate::options::DiscoveryAccept,
    ) -> Self {
        self.graph = Some(GraphInput::DiscoverPc {
            alpha,
            max_cond_size,
            fdr: fdr.adjustment(),
            accept_discovered: accept.auto(),
        });
        self
    }

    /// Supply a static PAG (class-aware identification required; DAG-only IDs are refused).
    #[must_use]
    pub fn pag(mut self, graph: Pag) -> Self {
        self.graph = Some(GraphInput::Pag(graph));
        self
    }

    /// Supply a temporal PAG (review / class-aware identification required).
    #[must_use]
    pub fn temporal_pag(mut self, graph: TemporalPag) -> Self {
        self.graph = Some(GraphInput::TemporalPag(graph));
        self
    }

    /// Average-effect query (static).
    #[must_use]
    pub fn query(mut self, query: AverageEffectQuery) -> Self {
        self.query = Some(CausalQuery::AverageEffect(query));
        self
    }

    /// Generic causal query (static or temporal).
    #[must_use]
    pub fn causal_query(mut self, query: CausalQuery) -> Self {
        self.query = Some(query);
        self
    }

    /// Temporal effect query.
    #[must_use]
    pub fn temporal_query(mut self, query: TemporalEffectQuery) -> Self {
        self.query = Some(CausalQuery::TemporalEffect(query));
        self
    }

    /// Discovery / estimation temporal-gap split.
    #[must_use]
    pub fn split(mut self, split: DiscoveryEstimationSplit) -> Self {
        self.split = Some(split);
        self
    }

    /// Configure refutation suite (static path).
    #[must_use]
    pub fn refute(mut self, suite: RefuteSuite) -> Self {
        self.refute = suite;
        self
    }

    /// Bootstrap replicates for the primary estimate.
    #[must_use]
    pub fn bootstrap_replicates(mut self, n: u32) -> Self {
        self.bootstrap_replicates = n;
        self
    }

    /// Select the identification strategy for the static ATE path.
    ///
    /// Defaults to [`IdentifierId::BackdoorAdjustment`] when unset. Wire strings such as
    /// `"backdoor.adjustment"` are accepted via [`From<&str>`]. `compile` refuses any
    /// identifier/estimator pair outside the allowlist. Ignored on the temporal path (which
    /// always uses [`IdentifierId::TemporalBackdoorUnfolded`]).
    #[must_use]
    pub fn identifier(mut self, id: impl Into<IdentifierId>) -> Self {
        self.identifier = Some(id.into());
        self
    }

    /// Select the estimator for the static ATE path.
    ///
    /// Defaults to [`EstimatorId::LinearAdjustmentAte`] when unset. Wire strings such as
    /// `"linear.adjustment.ate"` are accepted via [`From<&str>`]. `compile` refuses any
    /// identifier/estimator pair outside the allowlist. Ignored on the temporal path (which
    /// always uses [`EstimatorId::TemporalLinearAdjustment`]).
    #[must_use]
    pub fn estimator(mut self, id: impl Into<EstimatorId>) -> Self {
        self.estimator = Some(id.into());
        self
    }

    /// Configure frequentist vs Bayesian inference (DESIGN.md §34.1).
    ///
    /// [`InferenceMode::Bayesian`] selects estimator [`EstimatorId::BayesianGcomp`].
    #[must_use]
    pub fn inference(mut self, mode: InferenceMode) -> Self {
        if matches!(mode, InferenceMode::Bayesian(_)) {
            self.estimator = Some(EstimatorId::BayesianGcomp);
        }
        self.inference = mode;
        self
    }

    /// Overlap / positivity policy for propensity and AIPW estimators (DESIGN.md §14.3).
    ///
    /// When unset, those estimators keep their built-in defaults (clip = 0.01, no trim).
    /// Ignored by estimators that require [`OverlapPolicy::ExplicitOverride`] (linear, GLM, IV,
    /// front-door, RD).
    #[must_use]
    pub fn overlap_policy(mut self, policy: OverlapPolicy) -> Self {
        self.overlap_policy = Some(policy);
        self
    }

    /// Configure the running variable / cutoff / bandwidth required by the `rd.sharp`
    /// estimator. `compile` refuses `rd.sharp` without this.
    #[must_use]
    pub fn rd_config(mut self, running_variable: VariableId, cutoff: f64, bandwidth: f64) -> Self {
        self.rd = Some(RdConfig { running_variable, cutoff, bandwidth });
        self
    }

    /// Build the analysis object.
    ///
    /// # Errors
    ///
    /// Missing required fields.
    pub fn build(self) -> Result<CausalAnalysis, AnalysisError> {
        Ok(CausalAnalysis {
            data: self.data.ok_or(AnalysisError::Missing { field: "data" })?,
            graph: self.graph.ok_or(AnalysisError::Missing { field: "graph" })?,
            query: self.query.ok_or(AnalysisError::Missing { field: "query" })?,
            refute: self.refute,
            bootstrap_replicates: self.bootstrap_replicates,
            split: self.split,
            identifier: self.identifier,
            estimator: self.estimator,
            rd: self.rd,
            inference: self.inference,
            overlap_policy: self.overlap_policy,
        })
    }
}
