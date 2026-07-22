//! Unified `CausalAnalysis` facade.
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

use std::sync::Arc;

use causal_core::{
    AverageEffectQuery, CausalQuery, PopulationRegistry, TemporalEffectQuery, VariableId,
};
use causal_data::{
    DiscoveryEstimationSplit, EventData, MultiEnvironmentData, PanelData, TabularData,
    TimeSeriesData,
};
use causal_discovery::{MultiDatasetConstraints, RegimeAssignment};
use causal_estimate::OverlapPolicy;
use causal_graph::{Admg, Cpdag, Dag, Pag, TemporalCpdag, TemporalDag, TemporalPag};
use causal_stats::ConditionalIndependence;
use causal_validate::CustomEffectValidator;

use crate::error::AnalysisError;
use crate::inference::InferenceMode;
use crate::planner::GraphInput;
use crate::strategy_table::{EstimatorId, IdentifierId};

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
    /// Event data aligned onto a regular duration grid (stored as series).
    Event(TimeSeriesData),
    /// Multi-environment series (J-PCMCI+ discover path).
    MultiEnv(MultiEnvironmentData),
    /// Multi-unit panel (pooled discover + stacked cluster-HAC estimate).
    Panel(PanelData),
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
#[derive(Clone)]
pub struct CausalAnalysisBuilder {
    data: Option<DataInput>,
    /// Pending event alignment applied in [`Self::build`].
    event_pending: Option<(EventData, u64)>,
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
    /// Optional bindings for named predicates / custom target distributions.
    population_registry: Option<PopulationRegistry>,
    /// Optional CI test for discovery paths (defaults to partial correlation).
    discovery_ci: Option<Arc<dyn ConditionalIndependence + Send + Sync>>,
    /// Custom slow-path validators appended after the built-in refute suite.
    custom_validators: Vec<Arc<dyn CustomEffectValidator>>,
}

impl std::fmt::Debug for CausalAnalysisBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CausalAnalysisBuilder")
            .field("data", &self.data.as_ref().map(|_| "<data>"))
            .field("event_pending", &self.event_pending.as_ref().map(|_| "<event>"))
            .field("graph", &self.graph)
            .field("query", &self.query.as_ref().map(|_| "<query>"))
            .field("refute", &self.refute)
            .field("bootstrap_replicates", &self.bootstrap_replicates)
            .field("split", &self.split)
            .field("identifier", &self.identifier)
            .field("estimator", &self.estimator)
            .field("rd", &self.rd)
            .field("inference", &self.inference)
            .field("overlap_policy", &self.overlap_policy)
            .field("population_registry", &self.population_registry.as_ref().map(|_| "<registry>"))
            .field("discovery_ci", &self.discovery_ci.as_ref().map(|_| "<dyn CI>"))
            .field("custom_validators", &self.custom_validators.len())
            .finish()
    }
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
            event_pending: None,
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
            population_registry: None,
            discovery_ci: None,
            custom_validators: Vec::new(),
        }
    }

    /// Supply tabular data.
    #[must_use]
    pub fn data(mut self, data: TabularData) -> Self {
        self.event_pending = None;
        self.data = Some(DataInput::Tabular(data));
        self
    }

    /// Supply temporal series data.
    #[must_use]
    pub fn series(mut self, data: TimeSeriesData) -> Self {
        self.event_pending = None;
        self.data = Some(DataInput::Temporal(data));
        self
    }

    /// Supply multi-environment series (required for J-PCMCI+ discovery).
    #[must_use]
    pub fn series_multi(mut self, data: MultiEnvironmentData) -> Self {
        self.event_pending = None;
        self.data = Some(DataInput::MultiEnv(data));
        self
    }

    /// Supply irregular event data; aligned onto a regular duration grid at [`Self::build`].
    ///
    /// `align_interval_ns` is the bin width (§5.4). Integer-lag algorithms then run on
    /// the aligned series; raw event indices are never treated as lags.
    #[must_use]
    pub fn events(mut self, data: EventData, align_interval_ns: u64) -> Self {
        self.data = None;
        self.event_pending = Some((data, align_interval_ns));
        self
    }

    /// Supply multi-unit panel data (J-PCMCI+ discover; stacked PanelClusterHac estimate).
    #[must_use]
    pub fn panel(mut self, data: PanelData) -> Self {
        self.event_pending = None;
        self.data = Some(DataInput::Panel(data));
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

    /// Discover with classic static FCI (tabular PAG).
    ///
    /// With [`crate::options::DiscoveryAccept::AutoAccept`], the PAG is accepted
    /// as-is (circle marks go through generalized adjustment). With
    /// [`crate::options::DiscoveryAccept::Review`], compile yields a review-required plan.
    #[must_use]
    pub fn discover_fci(
        mut self,
        alpha: f64,
        max_cond_size: usize,
        fdr: crate::options::FdrControl,
        accept: crate::options::DiscoveryAccept,
    ) -> Self {
        self.graph = Some(GraphInput::DiscoverFci {
            alpha,
            max_cond_size,
            fdr: fdr.adjustment(),
            accept_discovered: accept.auto(),
        });
        self
    }

    /// Discover with classic static RFCI (tabular PAG).
    ///
    /// Same accept/review semantics as [`Self::discover_fci`].
    #[must_use]
    pub fn discover_rfci(
        mut self,
        alpha: f64,
        max_cond_size: usize,
        fdr: crate::options::FdrControl,
        accept: crate::options::DiscoveryAccept,
    ) -> Self {
        self.graph = Some(GraphInput::DiscoverRfci {
            alpha,
            max_cond_size,
            fdr: fdr.adjustment(),
            accept_discovered: accept.auto(),
        });
        self
    }

    /// Discover with GES (tabular CPDAG; auto-finishes only when fully oriented).
    #[must_use]
    pub fn discover_ges(
        mut self,
        alpha: f64,
        max_cond_size: usize,
        fdr: crate::options::FdrControl,
        accept: crate::options::DiscoveryAccept,
    ) -> Self {
        self.graph = Some(GraphInput::DiscoverGes {
            alpha,
            max_cond_size,
            fdr: fdr.adjustment(),
            accept_discovered: accept.auto(),
        });
        self
    }

    /// Discover with DirectLiNGAM (tabular DAG; auto-accept clears pending edges).
    #[must_use]
    pub fn discover_lingam(
        mut self,
        max_cond_size: usize,
        prune_threshold: f64,
        accept: crate::options::DiscoveryAccept,
    ) -> Self {
        self.graph = Some(GraphInput::DiscoverLingam {
            max_cond_size,
            prune_threshold,
            accept_discovered: accept.auto(),
        });
        self
    }

    /// Discover with NOTEARS (tabular continuous SEM → DAG).
    #[must_use]
    pub fn discover_notears(
        mut self,
        max_cond_size: usize,
        lambda: f64,
        threshold: f64,
        standardize: bool,
        accept: crate::options::DiscoveryAccept,
    ) -> Self {
        self.graph = Some(GraphInput::DiscoverNotears {
            max_cond_size,
            lambda,
            threshold,
            standardize,
            accept_discovered: accept.auto(),
        });
        self
    }

    /// Exact DAG posterior → Bayesian effect envelope (requires `inference=Bayesian`).
    #[must_use]
    pub fn discover_exact_dag_posterior(mut self) -> Self {
        self.graph = Some(GraphInput::DiscoverExactDagPosterior);
        self
    }

    /// Order MCMC DAG posterior → Bayesian effect envelope.
    #[must_use]
    pub fn discover_order_mcmc(
        mut self,
        n_chains: u32,
        n_warmup: u32,
        n_draws: u32,
        thin: u32,
        require_diagnostics_gate: bool,
    ) -> Self {
        self.graph = Some(GraphInput::DiscoverOrderMcmc {
            n_chains,
            n_warmup,
            n_draws,
            thin,
            require_diagnostics_gate,
        });
        self
    }

    /// Structure MCMC DAG posterior → Bayesian effect envelope.
    #[must_use]
    pub fn discover_structure_mcmc(
        mut self,
        n_chains: u32,
        n_warmup: u32,
        n_draws: u32,
        thin: u32,
    ) -> Self {
        self.graph = Some(GraphInput::DiscoverStructureMcmc { n_chains, n_warmup, n_draws, thin });
        self
    }

    /// CI-screened structure MCMC posterior → Bayesian effect envelope.
    #[must_use]
    pub fn discover_ci_screened_posterior(
        mut self,
        alpha: f64,
        max_cond_size: usize,
        fdr: crate::options::FdrControl,
        soft_weight: causal_discovery::CiSoftWeight,
        n_chains: u32,
        n_warmup: u32,
        n_draws: u32,
        thin: u32,
    ) -> Self {
        self.graph = Some(GraphInput::DiscoverCiScreenedPosterior {
            alpha,
            fdr: fdr.adjustment(),
            max_cond_size,
            soft_weight,
            n_chains,
            n_warmup,
            n_draws,
            thin,
        });
        self
    }

    /// DBN template posterior → temporal Bayesian effect envelope.
    #[must_use]
    pub fn discover_dbn_posterior(
        mut self,
        max_lag: u32,
        force_mcmc: bool,
        n_chains: u32,
        n_warmup: u32,
        n_draws: u32,
    ) -> Self {
        self.graph = Some(GraphInput::DiscoverDbnPosterior {
            max_lag,
            force_mcmc,
            n_chains,
            n_warmup,
            n_draws,
        });
        self
    }

    /// Override the CI test used by discovery paths (defaults to partial correlation).
    #[must_use]
    pub fn discovery_ci(mut self, ci: Arc<dyn ConditionalIndependence + Send + Sync>) -> Self {
        self.discovery_ci = Some(ci);
        self
    }

    /// Append custom effect validators ( slow path).
    #[must_use]
    pub fn custom_validators(mut self, validators: Vec<Arc<dyn CustomEffectValidator>>) -> Self {
        self.custom_validators = validators;
        self
    }

    /// Supply a static PAG (class-aware identification required; DAG-only IDs are refused).
    #[must_use]
    pub fn pag(mut self, graph: Pag) -> Self {
        self.graph = Some(GraphInput::Pag(graph));
        self
    }

    /// Supply a static CPDAG (auto-completes to a DAG when fully oriented).
    #[must_use]
    pub fn cpdag(mut self, graph: Cpdag) -> Self {
        self.graph = Some(GraphInput::Cpdag(graph));
        self
    }

    /// Supply a static ADMG (general ID when bidirected edges are present).
    #[must_use]
    pub fn admg(mut self, graph: Admg) -> Self {
        self.graph = Some(GraphInput::Admg(graph));
        self
    }

    /// Supply a temporal PAG (review / class-aware identification required).
    #[must_use]
    pub fn temporal_pag(mut self, graph: TemporalPag) -> Self {
        self.graph = Some(GraphInput::TemporalPag(graph));
        self
    }

    /// Supply a temporal CPDAG (auto-completes when fully oriented).
    #[must_use]
    pub fn temporal_cpdag(mut self, graph: TemporalCpdag) -> Self {
        self.graph = Some(GraphInput::TemporalCpdag(graph));
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
        // Bayesian inference must not force the static BayesianGcomp estimator on temporal.
        if matches!(self.estimator, Some(EstimatorId::BayesianGcomp)) {
            self.estimator = Some(EstimatorId::TemporalLinearAdjustment);
        }
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

    /// Configure frequentist vs Bayesian inference.
    ///
    /// For static ATE, [`InferenceMode::Bayesian`] selects estimator [`EstimatorId::BayesianGcomp`].
    /// Temporal queries keep [`EstimatorId::TemporalLinearAdjustment`]; Bayesian mode is applied
    /// at execute time on the lag-aligned design.
    #[must_use]
    pub fn inference(mut self, mode: InferenceMode) -> Self {
        if matches!(mode, InferenceMode::Bayesian(_))
            && !matches!(self.query, Some(CausalQuery::TemporalEffect(_)))
        {
            self.estimator = Some(EstimatorId::BayesianGcomp);
        }
        self.inference = mode;
        self
    }

    /// Overlap / positivity policy for propensity and AIPW estimators.
    ///
    /// When unset, those estimators keep their built-in defaults (clip = 0.01, no trim).
    /// Ignored by estimators that require [`OverlapPolicy::ExplicitOverride`] (linear, GLM, IV,
    /// front-door, RD).
    #[must_use]
    pub fn overlap_policy(mut self, policy: OverlapPolicy) -> Self {
        self.overlap_policy = Some(policy);
        self
    }

    /// Bindings for named predicates and custom target-distribution weights.
    #[must_use]
    pub fn population_registry(mut self, registry: PopulationRegistry) -> Self {
        self.population_registry = Some(registry);
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
    /// Missing required fields, or event alignment failure.
    pub fn build(self) -> Result<CausalAnalysis, AnalysisError> {
        let data = if let Some((event, interval_ns)) = self.event_pending {
            let aligned = event.align_to_grid(interval_ns).map_err(|e| AnalysisError::Compile {
                message: format!("event align_to_grid: {e}"),
            })?;
            DataInput::Event(aligned)
        } else {
            self.data.ok_or(AnalysisError::Missing { field: "data" })?
        };
        Ok(CausalAnalysis {
            data,
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
            population_registry: self.population_registry,
            discovery_ci: self.discovery_ci,
            custom_validators: self.custom_validators,
        })
    }
}
