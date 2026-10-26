//! Compiled structure discovery, attribution and simulation.
//!
//! [`DiscoverySpec`] is the executable configuration of a structure search and
//! [`MechanismProgram`] the mechanism fit a fitted-model route runs. A study
//! carries one of them beside its [`CausalQuery::Discover`],
//! [`CausalQuery::Attribute`] or [`CausalQuery::Simulate`] query: compile lowers
//! the route to a complete program, preparation retains it, and an estimate runs
//! the existing search, attribution or simulation engine under the lowered
//! program's dispatch scope, leases its buffers from the admission pool before
//! allocating them, and stamps the answer with the lowered identity and the
//! executed contract.
//!
//! A discovered structure reaches a downstream study only through its review
//! gate, [`accept_discovered`], which records the search, its bounds, the data
//! snapshot it read, the discovery program and claim, and the review on the
//! [`AcceptedGraph`]. Every study prepared on that graph carries the record in
//! its contract and binds it into its program identity.
//!
//! None of these routes is a geometric support-matrix cell. They execute and
//! stay unlicensed until a coverage record licenses them.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_contract::ClaimEnvelope;
use antecedent_core::{
    AttributeQuery, AttributeTarget, CanonicalWriter, CausalQuery, CausalRng, ChangeEstimator,
    ChangeMeasure, DiscoverQuery, DiscoveryMethod, ExecutionContext, InfluenceAllocation,
    MechanismChangeTest, NoiseDeclaration, ScreenWeight, SearchBounds, SearchSchedule,
    SimulateQuery, SimulationTarget, VariableId,
};
use antecedent_data::{
    MissingPolicy, MultiEnvironmentData, PanelData, PanelUnit, TableView, TabularData,
    TimeSeriesData,
};
use antecedent_graph::{Admg, Dag, DenseNodeId};
use antecedent_model::ValueBatch;

use crate::accepted::{AcceptedGraph, StructureProvenance};
use crate::discovery::{
    BayesianDiscoverParams, CiSoftWeight, CpdagDiscoveryResult, DagDiscoveryResult, DiscoverParams,
    ExactDagMarginals, GraphMcmcSchedule, GraphPosterior, NotearsDiscoveryResult,
    PagDiscoveryResult, RegimeAssignment, RpcmciDiscoveryResult, StaticCpdagDiscoveryResult,
    StaticDagDiscoveryResult, StaticDiscoverParams, StaticPagDiscoveryResult, StructureStability,
};
use crate::error::CausalError;
use crate::gcm::{
    AnomalyScores, ChangeAttributionResult, DifferenceMeasure, DistributionChangeOptions,
    FeatureRelevance, FitResampling, FittedGcm, IntrinsicAllocation, IntrinsicInfluenceQuery,
    IntrinsicInfluenceResult, IteResult, MechanismAssignment, MechanismChangeDetection,
    MechanismChangeMethod, MechanismRegistry, NoiseIndependence, PolicyValue, RobustChangeOptions,
    RootCauseRank, SelectionOptions, UnitChangeResult,
};
use crate::result::StudyResult;

/// The bytes of one `f64` cell, as a quote factor.
const CELL_BYTES: u64 = 8;

// ---------------------------------------------------------------------------
// Discovery specification
// ---------------------------------------------------------------------------

/// The executable parameter set of one search family.
#[derive(Clone, Debug)]
enum SearchParams {
    /// Static constraint- or score-based search.
    Static(StaticDiscoverParams),
    /// PCMCI-family temporal search.
    Temporal(DiscoverParams),
    /// Bayesian structure posterior.
    Bayesian(BayesianDiscoverParams),
    /// PC-skeleton screening followed by a structure posterior.
    Screened {
        /// Prior and score of the posterior.
        bayesian: BayesianDiscoverParams,
        /// Screening search.
        screen: StaticDiscoverParams,
    },
}

/// The executable configuration of a structure search.
///
/// The search method and its options, the conditional-independence test, the
/// multiple-testing rule, background knowledge, multi-dataset constraints,
/// regime assignment, prior, score and MCMC schedule. [`Self::query`] states
/// the search as a [`DiscoverQuery`] whose identity covers all of them: the
/// method and bounds by value, the rest through the configuration digest.
#[derive(Clone, Debug)]
pub struct DiscoverySpec {
    method: DiscoveryMethod,
    search: SearchParams,
    ci_label: Arc<str>,
    max_lag: u32,
    schedule: Option<GraphMcmcSchedule>,
    regimes: Option<RegimeAssignment>,
    min_regime_len: Option<usize>,
    missing: Option<MissingPolicy>,
}

fn saturating_u32(value: usize) -> u32 {
    u32::try_from(value).unwrap_or(u32::MAX)
}

impl DiscoverySpec {
    fn new(method: DiscoveryMethod, search: SearchParams, ci_label: impl Into<Arc<str>>) -> Self {
        let max_lag = match &search {
            SearchParams::Temporal(params) => params.max_lag,
            SearchParams::Static(_) | SearchParams::Bayesian(_) | SearchParams::Screened { .. } => {
                0
            }
        };
        Self {
            method,
            search,
            ci_label: ci_label.into(),
            max_lag,
            schedule: None,
            regimes: None,
            min_regime_len: None,
            missing: None,
        }
    }

    /// Static PC over tabular data; returns a CPDAG.
    #[must_use]
    pub fn pc(params: StaticDiscoverParams, ci_label: impl Into<Arc<str>>) -> Self {
        Self::new(DiscoveryMethod::Pc, SearchParams::Static(params), ci_label)
    }

    /// Static FCI over tabular data; returns a PAG.
    #[must_use]
    pub fn fci(params: StaticDiscoverParams, ci_label: impl Into<Arc<str>>) -> Self {
        Self::new(DiscoveryMethod::Fci, SearchParams::Static(params), ci_label)
    }

    /// Static RFCI over tabular data; returns a PAG.
    #[must_use]
    pub fn rfci(params: StaticDiscoverParams, ci_label: impl Into<Arc<str>>) -> Self {
        Self::new(DiscoveryMethod::Rfci, SearchParams::Static(params), ci_label)
    }

    /// Greedy equivalence search (Gaussian BIC); returns a CPDAG. The screening
    /// and subset options are read from `params`.
    #[must_use]
    pub fn ges(params: StaticDiscoverParams, ci_label: impl Into<Arc<str>>) -> Self {
        let method = DiscoveryMethod::Ges {
            screen_pc: params.screen_pc,
            max_subset: params.max_subset.map(saturating_u32),
        };
        Self::new(method, SearchParams::Static(params), ci_label)
    }

    /// `DirectLiNGAM`; returns a DAG.
    #[must_use]
    pub fn lingam(
        params: StaticDiscoverParams,
        prune_threshold: f64,
        ci_label: impl Into<Arc<str>>,
    ) -> Self {
        Self::new(
            DiscoveryMethod::DirectLingam { prune_threshold },
            SearchParams::Static(params),
            ci_label,
        )
    }

    /// NOTEARS continuous structure search; returns a DAG. `missing` declares
    /// the missing-value policy (`None` refuses a missing value).
    #[must_use]
    pub fn notears(
        params: StaticDiscoverParams,
        lambda: f64,
        threshold: f64,
        standardize: bool,
        missing: Option<MissingPolicy>,
        ci_label: impl Into<Arc<str>>,
    ) -> Self {
        let mut spec = Self::new(
            DiscoveryMethod::Notears { lambda, threshold, standardize },
            SearchParams::Static(params),
            ci_label,
        );
        spec.missing = missing;
        spec
    }

    /// Lagged PCMCI; returns a temporal DAG.
    #[must_use]
    pub fn pcmci(params: DiscoverParams, ci_label: impl Into<Arc<str>>) -> Self {
        Self::new(DiscoveryMethod::Pcmci, SearchParams::Temporal(params), ci_label)
    }

    /// PCMCI+; returns a temporal CPDAG.
    #[must_use]
    pub fn pcmci_plus(params: DiscoverParams, ci_label: impl Into<Arc<str>>) -> Self {
        Self::new(DiscoveryMethod::PcmciPlus, SearchParams::Temporal(params), ci_label)
    }

    /// LPCMCI; returns a temporal PAG.
    #[must_use]
    pub fn lpcmci(params: DiscoverParams, ci_label: impl Into<Arc<str>>) -> Self {
        Self::new(DiscoveryMethod::Lpcmci, SearchParams::Temporal(params), ci_label)
    }

    /// J-PCMCI+ over several environments; returns a temporal CPDAG.
    #[must_use]
    pub fn jpcmci_plus(params: DiscoverParams, ci_label: impl Into<Arc<str>>) -> Self {
        Self::new(DiscoveryMethod::JpcmciPlus, SearchParams::Temporal(params), ci_label)
    }

    /// Regime PCMCI over a supplied regime assignment; returns one temporal
    /// CPDAG per regime.
    #[must_use]
    pub fn rpcmci(
        params: DiscoverParams,
        regimes: RegimeAssignment,
        min_regime_len: Option<usize>,
        ci_label: impl Into<Arc<str>>,
    ) -> Self {
        let mut spec = Self::new(
            DiscoveryMethod::Rpcmci { min_regime_len: min_regime_len.map(saturating_u32) },
            SearchParams::Temporal(params),
            ci_label,
        );
        spec.regimes = Some(regimes);
        spec.min_regime_len = min_regime_len;
        spec
    }

    /// Exact DAG posterior by enumeration.
    #[must_use]
    pub fn exact_dag_posterior(params: BayesianDiscoverParams) -> Self {
        Self::new(DiscoveryMethod::ExactDagPosterior, SearchParams::Bayesian(params), "none")
    }

    /// Exact posterior over PAG classes of the Gaussian maximal ancestral graphs of at most four
    /// variables (latent confounding allowed), the Bayesian counterpart of FCI and RFCI.
    #[must_use]
    pub fn exact_pag_posterior(params: BayesianDiscoverParams) -> Self {
        Self::new(DiscoveryMethod::ExactPagPosterior, SearchParams::Bayesian(params), "none")
    }

    /// Metropolis posterior over PAG classes of the Gaussian maximal ancestral graphs of two to
    /// eight variables, by single-pair link moves.
    #[must_use]
    pub fn pag_mcmc(params: BayesianDiscoverParams, schedule: GraphMcmcSchedule) -> Self {
        let mut spec = Self::new(DiscoveryMethod::PagMcmc, SearchParams::Bayesian(params), "none");
        spec.schedule = Some(schedule);
        spec
    }

    /// Exact DAG posterior edge marginals.
    #[must_use]
    pub fn exact_dag_marginals(params: BayesianDiscoverParams) -> Self {
        Self::new(DiscoveryMethod::ExactDagMarginals, SearchParams::Bayesian(params), "none")
    }

    /// Order MCMC DAG posterior.
    #[must_use]
    pub fn order_mcmc(
        params: BayesianDiscoverParams,
        schedule: GraphMcmcSchedule,
        diagnostics_gate: bool,
    ) -> Self {
        let mut spec = Self::new(
            DiscoveryMethod::OrderMcmc { diagnostics_gate },
            SearchParams::Bayesian(params),
            "none",
        );
        spec.schedule = Some(schedule);
        spec
    }

    /// Structure MCMC DAG posterior.
    #[must_use]
    pub fn structure_mcmc(params: BayesianDiscoverParams, schedule: GraphMcmcSchedule) -> Self {
        let mut spec =
            Self::new(DiscoveryMethod::StructureMcmc, SearchParams::Bayesian(params), "none");
        spec.schedule = Some(schedule);
        spec
    }

    /// PC-skeleton screening followed by structure MCMC.
    #[must_use]
    pub fn ci_screened_posterior(
        params: BayesianDiscoverParams,
        screen: StaticDiscoverParams,
        schedule: GraphMcmcSchedule,
        soft_weight: CiSoftWeight,
        ci_label: impl Into<Arc<str>>,
    ) -> Self {
        let weight = match soft_weight {
            CiSoftWeight::None => ScreenWeight::None,
            CiSoftWeight::BayesFactor => ScreenWeight::BayesFactor,
            CiSoftWeight::PosteriorDependence => ScreenWeight::PosteriorDependence,
        };
        let mut spec = Self::new(
            DiscoveryMethod::CiScreenedPosterior { weight },
            SearchParams::Screened { bayesian: params, screen },
            ci_label,
        );
        spec.schedule = Some(schedule);
        spec
    }

    /// Bounded-lag DBN template posterior.
    #[must_use]
    pub fn dbn_posterior(
        params: BayesianDiscoverParams,
        max_lag: u32,
        force_mcmc: bool,
        schedule: GraphMcmcSchedule,
    ) -> Self {
        let mut spec = Self::new(
            DiscoveryMethod::DbnPosterior { force_mcmc },
            SearchParams::Bayesian(params),
            "none",
        );
        spec.max_lag = max_lag;
        spec.schedule = Some(schedule);
        spec
    }

    /// The posterior over Markov equivalence classes this DAG-posterior search induces.
    ///
    /// The search runs unchanged; every DAG it lists is replaced by its CPDAG and the mass
    /// of the DAGs that share one is summed, so the answer is a posterior whose atoms are
    /// classes. The base must list its graphs: an exact, order-MCMC, structure-MCMC or
    /// CI-screened posterior. Any other base is refused when the study compiles.
    #[must_use]
    pub fn equivalence_classes(self) -> Self {
        let method = DiscoveryMethod::ClassPosterior { base: Box::new(self.method.clone()) };
        Self { method, ..self }
    }

    /// The sampling distribution of this frequentist search over `replicates` resamples.
    ///
    /// The search is re-run on resamples drawn under the study's declared sampling design
    /// (independent subjects: rows; a dependent series: circular blocks; a panel: whole
    /// units), and the answer tabulates how often each node pair was adjacent, carried each
    /// pair of endpoint marks, and how often each whole structure came back. The base is PC,
    /// FCI, RFCI, GES, `DirectLiNGAM`, NOTEARS, PCMCI or PCMCI+; any other base is refused
    /// when the study compiles.
    #[must_use]
    pub fn bootstrap_stability(self, replicates: u32) -> Self {
        let method =
            DiscoveryMethod::BootstrapStability { base: Box::new(self.method.clone()), replicates };
        Self { method, ..self }
    }

    /// The search method.
    #[must_use]
    pub const fn method(&self) -> &DiscoveryMethod {
        &self.method
    }

    /// The same configuration with the wrapper removed: the search a class posterior
    /// collapses or a stability search re-runs.
    pub(crate) fn unwrapped(&self) -> Option<Self> {
        match &self.method {
            DiscoveryMethod::ClassPosterior { base }
            | DiscoveryMethod::BootstrapStability { base, .. } => {
                Some(Self { method: (**base).clone(), ..self.clone() })
            }
            _ => None,
        }
    }

    /// The regime labels of a regime-PCMCI search.
    pub(crate) fn regime_assignment(&self) -> Option<&RegimeAssignment> {
        self.regimes.as_ref()
    }

    /// The same search over other regime labels.
    pub(crate) fn with_regime_assignment(&self, regimes: RegimeAssignment) -> Self {
        Self { regimes: Some(regimes), ..self.clone() }
    }

    /// Resamples a stability search draws, or `None` for any other search.
    pub(crate) const fn stability_replicates(&self) -> Option<u32> {
        match &self.method {
            DiscoveryMethod::BootstrapStability { replicates, .. } => Some(*replicates),
            _ => None,
        }
    }

    /// The marginal-likelihood name a Bayesian search's program declares: the Gaussian BIC
    /// score unless the posterior was built under the Laplace-noise score.
    pub(crate) fn score_label(&self) -> &'static str {
        let family = match &self.search {
            SearchParams::Bayesian(params) | SearchParams::Screened { bayesian: params, .. } => {
                Some(params.score_family)
            }
            _ => None,
        };
        if matches!(self.method, DiscoveryMethod::ExactPagPosterior | DiscoveryMethod::PagMcmc) {
            return "discovery.gaussian_mag_bic_marginal";
        }
        match family {
            Some(antecedent_state::GraphScoreFamily::LaplaceBic) => {
                "discovery.laplace_bic_marginal"
            }
            _ => "discovery.gaussian_bic_marginal",
        }
    }

    /// The label of the conditional-independence test (`none` for a search that tests nothing).
    #[must_use]
    pub fn ci_label(&self) -> &str {
        &self.ci_label
    }

    /// The query this search answers over `variables`.
    #[must_use]
    pub fn query(&self, variables: impl Into<Arc<[VariableId]>>) -> DiscoverQuery {
        DiscoverQuery::new(variables, self.method.clone(), self.bounds())
    }

    fn bounds(&self) -> SearchBounds {
        let (alpha, max_cond_size) = match &self.search {
            SearchParams::Static(params) => (params.alpha, saturating_u32(params.max_cond_size)),
            SearchParams::Temporal(params) => (params.alpha, saturating_u32(params.max_cond_size)),
            SearchParams::Screened { screen, .. } => {
                (screen.alpha, saturating_u32(screen.max_cond_size))
            }
            // A posterior that tests nothing has no significance level; zero states it.
            SearchParams::Bayesian(_) => (0.0, 0),
        };
        SearchBounds {
            alpha,
            max_cond_size,
            max_lag: self.max_lag,
            schedule: self.schedule.map(|schedule| SearchSchedule {
                chains: schedule.n_chains,
                warmup: schedule.n_warmup,
                draws: schedule.n_draws,
                thin: schedule.thin,
            }),
            configuration: self.configuration(),
        }
    }

    /// Digest of everything the search runs that the method and bounds do not
    /// state by value.
    fn configuration(&self) -> [u8; 32] {
        fn knowledge(
            writer: &mut CanonicalWriter,
            knowledge: &crate::discovery::BackgroundKnowledge,
        ) {
            if knowledge.is_empty() {
                writer.word(0);
            } else {
                writer.word(1).bytes(&knowledge.identity());
            }
        }
        fn static_params(writer: &mut CanonicalWriter, params: &StaticDiscoverParams) {
            writer
                .text("static")
                .text(&format!("{:?}", params.fdr))
                .word(u64::from(params.screen_pc))
                .opt_word(params.max_subset.map(|value| value as u64));
            knowledge(writer, &params.knowledge);
        }
        let mut writer = CanonicalWriter::new("antecedent.discovery_configuration");
        writer.text(&self.ci_label);
        match &self.search {
            SearchParams::Static(params) => static_params(&mut writer, params),
            SearchParams::Temporal(params) => {
                writer
                    .text("temporal")
                    .text(&format!("{:?}", params.fdr))
                    .text(&format!("{:?}", params.multi_dataset));
                knowledge(&mut writer, &params.knowledge);
            }
            SearchParams::Bayesian(params) => {
                writer
                    .text("bayesian")
                    .text(&format!("{:?}", params.prior))
                    .text(&format!("{:?}", params.score_family));
            }
            SearchParams::Screened { bayesian, screen } => {
                writer
                    .text("screened")
                    .text(&format!("{:?}", bayesian.prior))
                    .text(&format!("{:?}", bayesian.score_family));
                static_params(&mut writer, screen);
            }
        }
        match &self.regimes {
            Some(assignment) => {
                writer.word(1).word(assignment.regimes.len() as u64);
                for regime in assignment.regimes.iter() {
                    writer.word(u64::from(regime.raw()));
                }
            }
            None => {
                writer.word(0);
            }
        }
        writer.opt_word(self.min_regime_len.map(|value| value as u64));
        writer.opt_text(self.missing.map(MissingPolicy::name));
        writer.finish()
    }

    /// Refuse a spec that does not state `query`.
    pub(crate) fn check_query(&self, query: &DiscoverQuery) -> Result<(), CausalError> {
        if self.query(Arc::clone(&query.variables)) != *query {
            return Err(crate::compile_reason!(
                "invalid_argument",
                "the discovery specification does not state the study's {} query: the method, \
                 bounds or configuration differ",
                query.method.id()
            ));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Mechanism program
// ---------------------------------------------------------------------------

/// How a fitted-model route obtains its mechanisms: the candidate families,
/// the per-node constraints and validation splits of the held-out selection,
/// and optionally a fit already made on the same rows.
///
/// The route fits through the compiler: every node's family is selected by its
/// held-out score on the declared splits, and the selection record of every
/// node is published on the result. A supplied fit is reused only on the rows
/// it was fitted on (by content digest); any other rows are refitted under the
/// same program.
#[derive(Clone, Debug)]
pub struct MechanismProgram {
    registry: MechanismRegistry,
    options: SelectionOptions,
    candidates: Option<Arc<[Arc<str>]>>,
    prefit: Option<(Arc<FittedGcm>, [u8; 32])>,
    /// Variables whose fitted mechanism the caller replaced with its own code.
    wrapped: Option<Arc<[Arc<str>]>>,
}

// A user-supplied mechanism is only ever called through a boundary that contains its panic and
// publishes no partially updated state, so the program may be shared across an unwind boundary.
impl std::panic::RefUnwindSafe for MechanismProgram {}

// The fitted mechanisms a payload may carry include user-supplied ones. They are only called
// through a boundary that contains a panic and publishes no partially updated state, so a
// payload may be shared across an unwind boundary.
impl std::panic::RefUnwindSafe for DiscoveryPayload {}
impl std::panic::RefUnwindSafe for AttributionPayload {}
impl std::panic::RefUnwindSafe for SimulationPayload {}
impl std::panic::UnwindSafe for DiscoveryPayload {}
impl std::panic::UnwindSafe for AttributionPayload {}
impl std::panic::UnwindSafe for SimulationPayload {}

impl Default for MechanismProgram {
    fn default() -> Self {
        Self::standard()
    }
}

impl MechanismProgram {
    /// The standard candidate registry, selected by best held-out score on shuffled folds.
    #[must_use]
    pub fn standard() -> Self {
        Self {
            registry: MechanismRegistry::standard(),
            options: SelectionOptions::default(),
            candidates: None,
            prefit: None,
            wrapped: None,
        }
    }

    /// The families `candidates` name (see [`crate::gcm::candidate_families`]),
    /// narrowed per node and scored on the splits `options` declares.
    ///
    /// # Errors
    ///
    /// An unknown token or an empty list (`mechanism_candidates_invalid`).
    pub fn from_candidates(
        candidates: &[&str],
        options: SelectionOptions,
    ) -> Result<Self, CausalError> {
        Ok(Self {
            registry: crate::gcm::registry_from_candidates(candidates)?,
            options,
            candidates: Some(candidates.iter().map(|token| Arc::from(*token)).collect()),
            prefit: None,
            wrapped: None,
        })
    }

    /// The standard registry under `options`.
    #[must_use]
    pub fn with_options(mut self, options: SelectionOptions) -> Self {
        self.options = options;
        self
    }

    /// Reuse `fitted`, made under this program on `rows`, whenever the route
    /// executes on those same rows.
    #[must_use]
    pub fn with_prefit(mut self, fitted: Arc<FittedGcm>, rows: &TabularData) -> Self {
        self.prefit = Some((fitted, rows.storage().content_digest()));
        self
    }

    /// Reuse `fitted`, made under this program on `rows` and then with the
    /// mechanisms of `wrapped` replaced by caller code. The route never
    /// refits such a model: on any other rows it refuses. The wrapped
    /// variables enter the program's label, so the program identity says that
    /// caller code, not a fitted family, produced those mechanisms.
    #[must_use]
    pub fn with_wrapped_prefit(
        mut self,
        fitted: Arc<FittedGcm>,
        rows: &TabularData,
        wrapped: &[&str],
    ) -> Self {
        let mut names: Vec<Arc<str>> = wrapped.iter().map(|name| Arc::from(*name)).collect();
        names.sort_unstable();
        names.dedup();
        self.prefit = Some((fitted, rows.storage().content_digest()));
        self.wrapped = Some(Arc::from(names));
        self
    }

    /// Whether the program is the standard registry with default selection.
    fn is_standard(&self) -> bool {
        self.candidates.is_none()
            && self.wrapped.is_none()
            && self.options == SelectionOptions::default()
    }

    /// Stable label of the candidate set and selection rule, for the
    /// statistical program's mechanism slots.
    #[must_use]
    pub fn label(&self) -> String {
        if self.is_standard() {
            return "gcm.registry.standard.best_score".to_owned();
        }
        let mut writer = CanonicalWriter::new("antecedent.mechanism_program");
        match &self.candidates {
            Some(tokens) => {
                writer.word(tokens.len() as u64);
                for token in tokens.iter() {
                    writer.text(token);
                }
            }
            None => {
                writer.word(0);
            }
        }
        writer.text(&format!("{:?}", self.options));
        if let Some(wrapped) = &self.wrapped {
            writer.texts(&wrapped[..]);
            return format!("gcm.registry.wrapped.{}", &writer.finish_hex()[..16]);
        }
        format!("gcm.registry.declared.{}", &writer.finish_hex()[..16])
    }

    /// Fit the program's mechanisms to `data` on `graph`, reusing the supplied
    /// fit on the rows it was made on.
    pub(crate) fn fit(
        &self,
        graph: &Dag,
        data: &TabularData,
    ) -> Result<Arc<FittedGcm>, CausalError> {
        if let Some((fitted, rows)) = &self.prefit {
            if *rows == data.storage().content_digest() {
                return Ok(Arc::clone(fitted));
            }
        }
        if self.wrapped.is_some() {
            return Err(crate::compile_reason!(
                "invalid_argument",
                "a model with caller-wrapped mechanisms runs only on the rows it was fitted on; \
                 refit and wrap it on these rows"
            ));
        }
        let fitted = if self.is_standard() {
            crate::gcm::fit_gcm(graph.clone(), data)?
        } else {
            crate::gcm::fit_gcm_with(graph.clone(), data, &self.registry, &self.options)?
        };
        Ok(Arc::new(fitted))
    }
}

// ---------------------------------------------------------------------------
// Typed answers
// ---------------------------------------------------------------------------

/// What a compiled structure search returned: the search engine's own result.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum DiscoveredStructure {
    /// PC or GES: a static CPDAG and its review.
    StaticCpdag(StaticCpdagDiscoveryResult),
    /// FCI or RFCI: a static PAG and its review.
    StaticPag(StaticPagDiscoveryResult),
    /// `DirectLiNGAM`: a static DAG and its review.
    StaticDag(StaticDagDiscoveryResult),
    /// NOTEARS: a static DAG review with its soft weights and row accounting.
    Notears(NotearsDiscoveryResult),
    /// PCMCI: a temporal DAG and its review.
    TemporalDag(DagDiscoveryResult),
    /// PCMCI+ or J-PCMCI+: a temporal CPDAG and its review.
    TemporalCpdag(CpdagDiscoveryResult),
    /// LPCMCI: a temporal PAG and its review.
    TemporalPag(PagDiscoveryResult),
    /// RPCMCI: one temporal CPDAG per regime.
    Regimes(RpcmciDiscoveryResult),
    /// A posterior over structures.
    Posterior(GraphPosterior),
    /// Exact posterior edge marginals.
    EdgeMarginals(ExactDagMarginals),
    /// The sampling distribution of a frequentist search over resamples.
    Stability(StructureStability),
    /// The sampling distribution of regime PCMCI: one table per labelled regime.
    RegimeStability(Vec<(antecedent_core::RegimeId, StructureStability)>),
}

impl DiscoveredStructure {
    /// Algorithm id the engine recorded.
    #[must_use]
    pub fn algorithm_id(&self) -> Option<&str> {
        match self {
            Self::StaticCpdag(result) => Some(&*result.algorithm.id),
            Self::StaticPag(result) => Some(&*result.algorithm.id),
            Self::StaticDag(result) => Some(&*result.algorithm.id),
            Self::Notears(result) => Some(&*result.discovery.algorithm.id),
            Self::TemporalDag(result) => Some(&*result.algorithm.id),
            Self::TemporalCpdag(result) => Some(&*result.algorithm.id),
            Self::TemporalPag(result) => Some(&*result.algorithm.id),
            Self::Regimes(result) => Some(&*result.algorithm.id),
            Self::Posterior(posterior) => posterior.algorithm.as_deref(),
            Self::EdgeMarginals(_) => Some("exact_dag_marginals"),
            Self::Stability(_) | Self::RegimeStability(_) => Some("bootstrap_stability"),
        }
    }

    /// Pending marks the review artifact lists before acceptance.
    #[must_use]
    pub fn pending_marks(&self) -> u64 {
        let count = match self {
            Self::StaticCpdag(result) => result.review.pending_edges.len(),
            Self::StaticPag(result) => result.review.pending_circles.len(),
            Self::StaticDag(result) => result.review.pending_edges.len(),
            Self::Notears(result) => result.discovery.review.pending_edges.len(),
            Self::TemporalDag(result) => result.review.pending_edges.len(),
            Self::TemporalCpdag(result) => result.review.pending_edges.len(),
            Self::TemporalPag(result) => result.review.pending_circles.len(),
            Self::Regimes(_)
            | Self::Posterior(_)
            | Self::EdgeMarginals(_)
            | Self::Stability(_)
            | Self::RegimeStability(_) => 0,
        };
        count as u64
    }

    /// Accept one structure through its review gate.
    ///
    /// `accept_discovered` auto-accepts what the review allows: pending
    /// directed edges are accepted, and the undirected marks of a CPDAG and the
    /// circle marks of a PAG stay class information. `false` refuses with
    /// `ReviewRequired` while anything is pending.
    ///
    /// # Errors
    ///
    /// `ReviewRequired` for pending marks; `route_not_supported` for a regime
    /// collection, a structure posterior or edge marginals, which are not one structure.
    pub fn accept(&self, accept_discovered: bool) -> Result<AcceptedGraph, CausalError> {
        let accepted = match self {
            Self::StaticDag(result) => {
                let review = result.review.clone();
                let review = if accept_discovered { review.accept_all() } else { review };
                AcceptedGraph::accept(review)?
            }
            Self::Notears(result) => {
                let review = result.discovery.review.clone();
                let review = if accept_discovered { review.accept_all() } else { review };
                AcceptedGraph::accept(review)?
            }
            Self::StaticCpdag(result) => {
                let mut review = result.review.clone();
                if accept_discovered {
                    review.pending_edges = Arc::from([]);
                }
                AcceptedGraph::accept(review)?
            }
            Self::StaticPag(result) => {
                if accept_discovered {
                    AcceptedGraph::pag(result.evidence.graph.clone())
                } else {
                    AcceptedGraph::accept(result.review.clone())?
                }
            }
            Self::TemporalDag(result) => {
                let pending = crate::review::PendingGraphReview::new(result.review.clone());
                let pending = if accept_discovered { pending.accept_all() } else { pending };
                pending.finish()?
            }
            Self::TemporalCpdag(result) => {
                let pending = crate::review::PendingCpdagReview::new(result.review.clone());
                let pending =
                    if accept_discovered { pending.accept_all_directed() } else { pending };
                pending.finish()?
            }
            Self::TemporalPag(result) => {
                if accept_discovered {
                    AcceptedGraph::temporal_pag(result.evidence.graph.clone())
                } else {
                    AcceptedGraph::accept(result.review.clone())?
                }
            }
            Self::Regimes(_) | Self::Posterior(_) | Self::EdgeMarginals(_) => {
                return Err(crate::unsupported_reason!(
                    "route_not_supported",
                    "a regime collection, structure posterior or edge marginals is not one \
                     structure; analyse a posterior with StudyBuilder::graph_posterior, or \
                     accept one regime's graph"
                ));
            }
            Self::Stability(_) | Self::RegimeStability(_) => {
                return Err(crate::unsupported_reason!(
                    "route_not_supported",
                    "a stability table is the sampling distribution of a search, not one \
                     structure; accept the structure the search returns on the full sample by \
                     running that search itself, and read this table beside it"
                ));
            }
        };
        Ok(match self.algorithm_id() {
            Some(algorithm) => accepted.with_discovery_algorithm(algorithm),
            None => accepted,
        })
    }
}

/// The answer of a compiled structure search.
#[derive(Clone, Debug)]
pub struct DiscoveryPayload {
    /// The search engine's result.
    pub structure: DiscoveredStructure,
    /// The query the search answered.
    pub query: DiscoverQuery,
}

/// What a compiled attribution returned.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum AttributionOutcome {
    /// Intrinsic causal influence of the upstream noise terms.
    IntrinsicInfluence(IntrinsicInfluenceResult),
    /// Distribution-change attribution with its components ranked as root causes.
    RootCauses {
        /// The change attribution.
        attribution: ChangeAttributionResult,
        /// Components ranked by their contribution.
        ranks: Vec<RootCauseRank>,
    },
    /// Distribution-change attribution or path decomposition.
    Change(ChangeAttributionResult),
    /// Interventional feature relevance.
    FeatureRelevance(Vec<FeatureRelevance>),
    /// Anomaly scores per target.
    Anomaly(Vec<AnomalyScores>),
    /// Per-unit change attribution.
    UnitChange(UnitChangeResult),
    /// Mechanism-change detections.
    MechanismChange(Vec<MechanismChangeDetection>),
}

/// The answer of a compiled attribution and the mechanisms it ran on.
#[derive(Clone, Debug)]
pub struct AttributionPayload {
    /// The attribution.
    pub outcome: AttributionOutcome,
    /// Per-node mechanism selection record of the fit the attribution ran on.
    pub mechanisms: Arc<[MechanismAssignment]>,
}

/// What a compiled simulation returned.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum SimulationOutcome {
    /// Interventional draws.
    Sample(ValueBatch),
    /// Model-based policy value with its Monte Carlo error.
    PolicyValue(PolicyValue),
    /// Unit-level effects after abduction.
    UnitEffects(IteResult),
    /// Interventional draws with the exogenous noise that produced them.
    NoiseSample {
        /// The draws.
        draws: ValueBatch,
        /// The noise, replayable through [`SimulationTarget::NoiseReplay`].
        noise: crate::gcm::NoiseTable,
    },
    /// Counterfactual values of the factual rows (world × node × unit).
    Counterfactual(crate::gcm::CounterfactualResult),
    /// Realised values of every step of a rollout or panel replay.
    Temporal(crate::gcm::TemporalPath),
}

/// The answer of a compiled simulation and the mechanisms it ran on.
#[derive(Clone, Debug)]
pub struct SimulationPayload {
    /// The simulation.
    pub outcome: SimulationOutcome,
    /// Per-node mechanism selection record of the fit the simulation ran on.
    pub mechanisms: Arc<[MechanismAssignment]>,
}

// ---------------------------------------------------------------------------
// Execution
// ---------------------------------------------------------------------------

/// The data a structure search reads.
pub(crate) enum DiscoverData<'a> {
    /// Static rows.
    Tabular(&'a TabularData),
    /// One time series.
    Series(&'a TimeSeriesData),
    /// The units of a panel, each its own series.
    Panel(&'a PanelData),
    /// Several environments' series.
    MultiEnv(&'a MultiEnvironmentData),
}

impl DiscoverData<'_> {
    fn modality(&self) -> &'static str {
        match self {
            Self::Tabular(_) => "tabular",
            Self::Series(_) => "series",
            Self::Panel(_) => "panel",
            Self::MultiEnv(_) => "multi-environment",
        }
    }

    /// Rows the search reads, for its quote.
    pub(crate) fn rows(&self) -> u64 {
        match self {
            Self::Tabular(data) => data.row_count() as u64,
            Self::Series(data) => data.row_count() as u64,
            Self::Panel(panel) => panel.total_rows() as u64,
            Self::MultiEnv(multi) => {
                multi.environments().iter().map(|env| env.row_count() as u64).sum()
            }
        }
    }
}

/// The units of a panel as the per-unit series a panel search reads.
fn panel_units(panel: &PanelData) -> Vec<TimeSeriesData> {
    panel.units().iter().map(|unit| unit.series.clone()).collect()
}

/// A panel of `units`, unit `i` labelled `i`.
///
/// # Errors
///
/// No unit, or units whose schemas differ.
pub fn panel_of_units(units: Vec<TimeSeriesData>) -> Result<PanelData, CausalError> {
    let units: Vec<PanelUnit> = units
        .into_iter()
        .enumerate()
        .map(|(index, series)| PanelUnit { unit_id: saturating_u32(index), series })
        .collect();
    PanelData::try_new(units).map_err(CausalError::rejected)
}

/// Bytes the search leases before it runs: the lagged frame it builds and the
/// link, separating-set and draw tables it keeps.
pub(crate) fn discovery_quote(query: &DiscoverQuery, rows: u64) -> u64 {
    let width = query.variables.len() as u64;
    let lags = u64::from(query.bounds.max_lag).saturating_add(1);
    let frame = rows.saturating_mul(width).saturating_mul(lags).saturating_mul(CELL_BYTES);
    let nodes = width.saturating_mul(lags);
    let links = nodes.saturating_mul(nodes).saturating_mul(64);
    let draws = query.bounds.schedule.map_or(0, |schedule| {
        u64::from(schedule.chains)
            .saturating_mul(u64::from(schedule.draws))
            .saturating_mul(width.saturating_mul(width).div_ceil(8).saturating_add(16))
    });
    frame.saturating_add(links).saturating_add(draws)
}

/// Lease `bytes` from the context's admission pool.
pub(crate) fn admit(
    ctx: &ExecutionContext,
    bytes: u64,
    what: &'static str,
) -> Result<antecedent_core::execution::AdmissionLease, CausalError> {
    ctx.acquire_admission(0, bytes).map_err(|error| match error {
        antecedent_core::execution::AdmissionError::Cancelled => {
            CausalError::Cancelled { stage: what }
        }
        other => CausalError::Resource {
            message: format!(
                "the {what} needs {bytes} bytes and the admission pool refused them: {other}; \
                 next attempt: narrow the request or raise the pool"
            ),
        },
    })
}

fn wrong_data(method: &DiscoveryMethod, data: &DiscoverData<'_>) -> CausalError {
    crate::compile_reason!(
        "data_modality_not_licensed",
        "the {} search does not run on {} data",
        method.id(),
        data.modality()
    )
}

fn schedule_of(spec: &DiscoverySpec) -> Result<GraphMcmcSchedule, CausalError> {
    spec.schedule.ok_or_else(|| {
        crate::compile_reason!(
            "required_option_missing",
            "the {} posterior needs an MCMC schedule",
            spec.method.id()
        )
    })
}

/// Run the search `spec` states over `data`, with the engine every
/// `discover_*` entry runs.
pub(crate) fn run_discovery(
    spec: &DiscoverySpec,
    query: &DiscoverQuery,
    data: &DiscoverData<'_>,
    design: Option<antecedent_core::DeclaredSamplingDesign>,
    ctx: &ExecutionContext,
) -> Result<DiscoveredStructure, CausalError> {
    if let Some(inner) = spec.unwrapped() {
        // A wrapper runs the search it wraps and reads its answer.
        return match spec.stability_replicates() {
            Some(replicates) => super::structure_stability::run_stability(
                &inner, replicates, query, data, design, ctx,
            ),
            None => match run_discovery(&inner, query, data, design, ctx)? {
                DiscoveredStructure::Posterior(posterior) => {
                    Ok(DiscoveredStructure::Posterior(posterior.to_equivalence_classes()?))
                }
                other => Err(crate::compile_reason!(
                    "route_not_supported",
                    "the {} search returned no listed graphs to collapse into equivalence \
                     classes (it returned {})",
                    spec.method().id(),
                    other.algorithm_id().unwrap_or("no structure")
                )),
            },
        };
    }
    let search = Search { spec, vars: &query.variables, data, ctx };
    for algorithm in DISCOVERY_ALGORITHMS {
        if let Some(found) = algorithm(&search)? {
            return Ok(found);
        }
    }
    Err(wrong_data(&spec.method, data))
}

/// The request one discovery algorithm reads.
struct Search<'a> {
    spec: &'a DiscoverySpec,
    vars: &'a [VariableId],
    data: &'a DiscoverData<'a>,
    ctx: &'a ExecutionContext,
}

/// One algorithm's entry: `Ok(None)` when the request's method, parameter family or data
/// modality is not this algorithm's cell, so the next entry is tried.
type DiscoveryAlgorithm = fn(&Search<'_>) -> Result<Option<DiscoveredStructure>, CausalError>;

/// Every algorithm cell, tried in order; adding an algorithm adds one entry here.
const DISCOVERY_ALGORITHMS: &[DiscoveryAlgorithm] = &[
    run_pc,
    run_fci,
    run_rfci,
    run_ges,
    run_lingam,
    run_notears,
    run_pcmci,
    run_pcmci_plus,
    run_lpcmci,
    run_jpcmci_plus,
    run_rpcmci,
    run_exact_dag_posterior,
    run_exact_pag_posterior,
    run_pag_mcmc,
    run_exact_dag_marginals,
    run_order_mcmc,
    run_structure_mcmc,
    run_ci_screened_posterior,
    run_dbn_posterior,
];

type TabularEngine<P, R> =
    fn(&TabularData, &[VariableId], &P, &ExecutionContext) -> Result<R, CausalError>;

/// A static search run over tabular rows.
fn static_tabular<R>(
    search: &Search<'_>,
    method_matches: bool,
    engine: TabularEngine<StaticDiscoverParams, R>,
    wrap: fn(R) -> DiscoveredStructure,
) -> Result<Option<DiscoveredStructure>, CausalError> {
    let (true, SearchParams::Static(p), DiscoverData::Tabular(d)) =
        (method_matches, &search.spec.search, search.data)
    else {
        return Ok(None);
    };
    Ok(Some(wrap(engine(d, search.vars, p, search.ctx)?)))
}

fn run_pc(search: &Search<'_>) -> Result<Option<DiscoveredStructure>, CausalError> {
    static_tabular(
        search,
        matches!(search.spec.method, DiscoveryMethod::Pc),
        crate::discovery::discover_pc,
        DiscoveredStructure::StaticCpdag,
    )
}

fn run_fci(search: &Search<'_>) -> Result<Option<DiscoveredStructure>, CausalError> {
    static_tabular(
        search,
        matches!(search.spec.method, DiscoveryMethod::Fci),
        crate::discovery::discover_fci,
        DiscoveredStructure::StaticPag,
    )
}

fn run_rfci(search: &Search<'_>) -> Result<Option<DiscoveredStructure>, CausalError> {
    static_tabular(
        search,
        matches!(search.spec.method, DiscoveryMethod::Rfci),
        crate::discovery::discover_rfci,
        DiscoveredStructure::StaticPag,
    )
}

fn run_ges(search: &Search<'_>) -> Result<Option<DiscoveredStructure>, CausalError> {
    static_tabular(
        search,
        matches!(search.spec.method, DiscoveryMethod::Ges { .. }),
        crate::discovery::discover_ges,
        DiscoveredStructure::StaticCpdag,
    )
}

fn run_lingam(search: &Search<'_>) -> Result<Option<DiscoveredStructure>, CausalError> {
    let (
        DiscoveryMethod::DirectLingam { prune_threshold },
        SearchParams::Static(p),
        DiscoverData::Tabular(d),
    ) = (&search.spec.method, &search.spec.search, search.data)
    else {
        return Ok(None);
    };
    let found = crate::discovery::discover_lingam(d, search.vars, p, *prune_threshold, search.ctx)?;
    Ok(Some(DiscoveredStructure::StaticDag(found)))
}

fn run_notears(search: &Search<'_>) -> Result<Option<DiscoveredStructure>, CausalError> {
    let (
        DiscoveryMethod::Notears { lambda, threshold, standardize },
        SearchParams::Static(p),
        DiscoverData::Tabular(d),
    ) = (&search.spec.method, &search.spec.search, search.data)
    else {
        return Ok(None);
    };
    let found = crate::discovery::discover_notears_with(
        d,
        search.vars,
        p,
        *lambda,
        *threshold,
        *standardize,
        search.spec.missing,
        search.ctx,
    )?;
    Ok(Some(DiscoveredStructure::Notears(found)))
}

/// A temporal search that runs on one series or on a panel's per-unit series.
fn temporal_series_or_panel<R>(
    search: &Search<'_>,
    method_matches: bool,
    on_series: SeriesEngine<R>,
    on_panel: PanelEngine<R>,
    wrap: fn(R) -> DiscoveredStructure,
) -> Result<Option<DiscoveredStructure>, CausalError> {
    let (true, SearchParams::Temporal(p)) = (method_matches, &search.spec.search) else {
        return Ok(None);
    };
    let found = match search.data {
        DiscoverData::Series(s) => on_series(s, search.vars, p, search.ctx)?,
        DiscoverData::Panel(panel) => on_panel(&panel_units(panel), search.vars, p, search.ctx)?,
        _ => return Ok(None),
    };
    Ok(Some(wrap(found)))
}

type SeriesEngine<R> = fn(
    &TimeSeriesData,
    &[VariableId],
    &DiscoverParams,
    &ExecutionContext,
) -> Result<R, CausalError>;
type PanelEngine<R> = fn(
    &[TimeSeriesData],
    &[VariableId],
    &DiscoverParams,
    &ExecutionContext,
) -> Result<R, CausalError>;

fn run_pcmci(search: &Search<'_>) -> Result<Option<DiscoveredStructure>, CausalError> {
    use crate::discovery as engine;
    temporal_series_or_panel(
        search,
        matches!(search.spec.method, DiscoveryMethod::Pcmci),
        engine::discover_pcmci,
        engine::discover_pcmci_panel,
        DiscoveredStructure::TemporalDag,
    )
}

fn run_pcmci_plus(search: &Search<'_>) -> Result<Option<DiscoveredStructure>, CausalError> {
    use crate::discovery as engine;
    temporal_series_or_panel(
        search,
        matches!(search.spec.method, DiscoveryMethod::PcmciPlus),
        engine::discover_pcmci_plus,
        engine::discover_pcmci_plus_panel,
        DiscoveredStructure::TemporalCpdag,
    )
}

fn run_lpcmci(search: &Search<'_>) -> Result<Option<DiscoveredStructure>, CausalError> {
    use crate::discovery as engine;
    temporal_series_or_panel(
        search,
        matches!(search.spec.method, DiscoveryMethod::Lpcmci),
        engine::discover_lpcmci,
        engine::discover_lpcmci_panel,
        DiscoveredStructure::TemporalPag,
    )
}

fn run_jpcmci_plus(search: &Search<'_>) -> Result<Option<DiscoveredStructure>, CausalError> {
    let (DiscoveryMethod::JpcmciPlus, SearchParams::Temporal(p), DiscoverData::MultiEnv(m)) =
        (&search.spec.method, &search.spec.search, search.data)
    else {
        return Ok(None);
    };
    let found = crate::discovery::discover_jpcmci_plus(m, search.vars, p, search.ctx)?;
    Ok(Some(DiscoveredStructure::TemporalCpdag(found)))
}

fn run_rpcmci(search: &Search<'_>) -> Result<Option<DiscoveredStructure>, CausalError> {
    let (DiscoveryMethod::Rpcmci { .. }, SearchParams::Temporal(p), DiscoverData::Series(s)) =
        (&search.spec.method, &search.spec.search, search.data)
    else {
        return Ok(None);
    };
    let regimes = search.spec.regimes.as_ref().ok_or_else(|| {
        crate::compile_reason!("required_option_missing", "regime PCMCI needs a regime assignment")
    })?;
    let found = crate::discovery::discover_rpcmci(
        s,
        search.vars,
        regimes,
        p,
        search.spec.min_regime_len,
        search.ctx,
    )?;
    Ok(Some(DiscoveredStructure::Regimes(found)))
}

fn run_exact_dag_posterior(
    search: &Search<'_>,
) -> Result<Option<DiscoveredStructure>, CausalError> {
    let (DiscoveryMethod::ExactDagPosterior, SearchParams::Bayesian(b), DiscoverData::Tabular(d)) =
        (&search.spec.method, &search.spec.search, search.data)
    else {
        return Ok(None);
    };
    let found = crate::discovery::discover_exact_dag_posterior(d, search.vars, b, search.ctx)?;
    Ok(Some(DiscoveredStructure::Posterior(found)))
}

fn run_exact_pag_posterior(
    search: &Search<'_>,
) -> Result<Option<DiscoveredStructure>, CausalError> {
    let (DiscoveryMethod::ExactPagPosterior, SearchParams::Bayesian(_), DiscoverData::Tabular(d)) =
        (&search.spec.method, &search.spec.search, search.data)
    else {
        return Ok(None);
    };
    let found = crate::discovery::discover_exact_pag_posterior(d, search.vars, search.ctx)?;
    Ok(Some(DiscoveredStructure::Posterior(found)))
}

fn run_pag_mcmc(search: &Search<'_>) -> Result<Option<DiscoveredStructure>, CausalError> {
    let (DiscoveryMethod::PagMcmc, SearchParams::Bayesian(_), DiscoverData::Tabular(d)) =
        (&search.spec.method, &search.spec.search, search.data)
    else {
        return Ok(None);
    };
    let found = crate::discovery::discover_pag_mcmc(
        d,
        search.vars,
        &schedule_of(search.spec)?,
        search.ctx,
    )?;
    Ok(Some(DiscoveredStructure::Posterior(found)))
}

fn run_exact_dag_marginals(
    search: &Search<'_>,
) -> Result<Option<DiscoveredStructure>, CausalError> {
    let (DiscoveryMethod::ExactDagMarginals, SearchParams::Bayesian(b), DiscoverData::Tabular(d)) =
        (&search.spec.method, &search.spec.search, search.data)
    else {
        return Ok(None);
    };
    let found = crate::discovery::discover_exact_dag_marginals(d, search.vars, b, search.ctx)?;
    Ok(Some(DiscoveredStructure::EdgeMarginals(found)))
}

fn run_order_mcmc(search: &Search<'_>) -> Result<Option<DiscoveredStructure>, CausalError> {
    let (
        DiscoveryMethod::OrderMcmc { diagnostics_gate },
        SearchParams::Bayesian(b),
        DiscoverData::Tabular(d),
    ) = (&search.spec.method, &search.spec.search, search.data)
    else {
        return Ok(None);
    };
    let found = crate::discovery::discover_order_mcmc(
        d,
        search.vars,
        b,
        &schedule_of(search.spec)?,
        *diagnostics_gate,
        search.ctx,
    )?;
    Ok(Some(DiscoveredStructure::Posterior(found)))
}

fn run_structure_mcmc(search: &Search<'_>) -> Result<Option<DiscoveredStructure>, CausalError> {
    let (DiscoveryMethod::StructureMcmc, SearchParams::Bayesian(b), DiscoverData::Tabular(d)) =
        (&search.spec.method, &search.spec.search, search.data)
    else {
        return Ok(None);
    };
    let found = crate::discovery::discover_structure_mcmc(
        d,
        search.vars,
        b,
        &schedule_of(search.spec)?,
        search.ctx,
    )?;
    Ok(Some(DiscoveredStructure::Posterior(found)))
}

fn run_ci_screened_posterior(
    search: &Search<'_>,
) -> Result<Option<DiscoveredStructure>, CausalError> {
    let (
        DiscoveryMethod::CiScreenedPosterior { weight },
        SearchParams::Screened { bayesian, screen },
        DiscoverData::Tabular(d),
    ) = (&search.spec.method, &search.spec.search, search.data)
    else {
        return Ok(None);
    };
    let soft_weight = match weight {
        ScreenWeight::None => CiSoftWeight::None,
        ScreenWeight::BayesFactor => CiSoftWeight::BayesFactor,
        ScreenWeight::PosteriorDependence => CiSoftWeight::PosteriorDependence,
    };
    let found = crate::discovery::discover_ci_screened_posterior(
        d,
        search.vars,
        bayesian,
        screen,
        &schedule_of(search.spec)?,
        soft_weight,
        search.ctx,
    )?;
    Ok(Some(DiscoveredStructure::Posterior(found)))
}

fn run_dbn_posterior(search: &Search<'_>) -> Result<Option<DiscoveredStructure>, CausalError> {
    use crate::discovery as engine;
    let (DiscoveryMethod::DbnPosterior { force_mcmc }, SearchParams::Bayesian(b)) =
        (&search.spec.method, &search.spec.search)
    else {
        return Ok(None);
    };
    let spec = search.spec;
    let found = match search.data {
        DiscoverData::Series(s) => engine::discover_dbn_posterior(
            s,
            search.vars,
            b,
            spec.max_lag,
            *force_mcmc,
            &schedule_of(spec)?,
            search.ctx,
        )?,
        DiscoverData::Panel(panel) => engine::discover_dbn_posterior_panel(
            &panel_units(panel),
            search.vars,
            b,
            spec.max_lag,
            *force_mcmc,
            &schedule_of(spec)?,
            search.ctx,
        )?,
        _ => return Ok(None),
    };
    Ok(Some(DiscoveredStructure::Posterior(found)))
}

/// Bytes a fitted-model route leases before it runs.
pub(crate) fn model_quote(target_rows: u64, nodes: u64, draws: u64) -> u64 {
    target_rows.max(draws).saturating_mul(nodes.max(1)).saturating_mul(CELL_BYTES).saturating_mul(3)
}

pub(crate) fn difference_measure(measure: ChangeMeasure) -> DifferenceMeasure {
    match measure {
        ChangeMeasure::MeanDifference => DifferenceMeasure::MeanDiff,
        ChangeMeasure::VarianceDifference => DifferenceMeasure::VarianceDiff,
        ChangeMeasure::GaussianKl => DifferenceMeasure::GaussianKl,
    }
}

fn change_attribution(
    fitted: &FittedGcm,
    data: &TabularData,
    change: &antecedent_core::ChangeAttributionQuery,
    estimator: &ChangeEstimator,
    ctx: &ExecutionContext,
) -> Result<ChangeAttributionResult, CausalError> {
    match estimator {
        ChangeEstimator::Sampled { measure, n_samples, seed } => {
            crate::gcm::attribute_distribution_change(
                &fitted.model,
                data,
                change,
                &DistributionChangeOptions {
                    measure: difference_measure(*measure),
                    n_samples: *n_samples,
                    seed: *seed,
                },
                ctx,
            )
        }
        ChangeEstimator::Robust { max_rows, n_samples, seed } => {
            crate::gcm::attribute_distribution_change_robust(
                &fitted.model,
                data,
                change,
                &RobustChangeOptions { max_rows: *max_rows, n_samples: *n_samples, seed: *seed },
                ctx,
            )
        }
    }
}

/// The structure an intrinsic-influence query declares the noise on: the
/// model's directed edges and the declared bidirected pairs.
fn noise_structure(
    graph: &Dag,
    bidirected: &[(VariableId, VariableId)],
) -> Result<Admg, CausalError> {
    let nodes = u32::try_from(graph.node_count()).map_err(|_| CausalError::Resource {
        message: "the model has more nodes than a node id holds".into(),
    })?;
    let mut admg = Admg::with_variables(nodes);
    for (from, to) in graph.directed_edge_pairs_sorted() {
        admg.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to))
            .map_err(CausalError::rejected)?;
    }
    for (a, b) in bidirected {
        admg.insert_bidirected(DenseNodeId::from_raw(a.raw()), DenseNodeId::from_raw(b.raw()))
            .map_err(CausalError::rejected)?;
    }
    Ok(admg)
}

/// How a mechanism-change detection draws its null law and reads its answer.
///
/// `block` is the block length of a dependent series (the null law is then a block permutation
/// of the residuals, not an independent-residual one); `bayesian` reads the answer as the
/// posterior probability of a change under the prior the query declares.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct ChangeReading {
    /// Block length of a dependent series, when the populations are segments of one.
    pub(crate) block: Option<usize>,
    /// Read the answer as a posterior probability of a change.
    pub(crate) bayesian: bool,
}

/// Block permutations a dependent-series mechanism-change test runs.
pub(crate) const CHANGE_BLOCK_PERMUTATIONS: usize = 499;

/// The detections of a mechanism-change query under `reading`: the posterior probability of a
/// change under the declared prior, the block-permutation test of a dependent series, or the
/// independent-residual test.
pub(crate) fn run_mechanism_change(
    fitted: &FittedGcm,
    data: &TabularData,
    detection: &antecedent_core::MechanismChangeQuery,
    method: MechanismChangeMethod,
    reading: ChangeReading,
    ctx: &ExecutionContext,
) -> Result<Vec<antecedent_attribution::MechanismChangeDetection>, CausalError> {
    if reading.bayesian {
        let prior = detection
            .change_prior
            .map(antecedent_core::OrderedFloatBits::to_f64)
            .ok_or_else(|| {
                crate::unsupported_reason!(
                    "option_not_applicable",
                    "the posterior probability of a mechanism change needs a prior over the change \
                 indicator, and none is assumed. Next attempt: declare it with \
                 MechanismChangeQuery::with_change_prior"
                )
            })?;
        return Ok(antecedent_attribution::mechanism_change_posterior(
            &fitted.model,
            data,
            detection,
            prior,
            reading.block.unwrap_or(1),
            ctx,
        )?
        .into_iter()
        .map(antecedent_attribution::MechanismChangePosterior::into_detection)
        .collect());
    }
    if let Some(block) = reading.block {
        return Ok(antecedent_attribution::detect_mechanism_changes_block(
            &fitted.model,
            data,
            detection,
            method,
            block,
            CHANGE_BLOCK_PERMUTATIONS,
            ctx,
        )?);
    }
    crate::gcm::mechanism_change_detection(&fitted.model, data, detection, method, ctx)
}

/// The mechanism-fit refits an attribution carries: how many and how they weight the rows.
///
/// The study's inference mode sets it (a posterior over the mechanisms under Bayesian
/// inference), so the query itself, whose identity is canonical, does not change with it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct AttributionFit {
    /// Refits to run.
    pub(crate) replicates: u32,
    /// The row weighting of each refit.
    pub(crate) resampling: FitResampling,
}

/// The named scalar quantities of a change attribution: its total, one entry per component, and
/// one per path when it decomposes along paths.
pub(crate) fn change_quantities(result: &ChangeAttributionResult) -> Vec<(String, f64)> {
    let mut values = vec![("total_change".to_owned(), result.total_change)];
    values.extend(
        result
            .contributions
            .iter()
            .map(|c| (format!("component[{}]", c.component.variable().raw()), c.contribution)),
    );
    values.extend(result.path_breakdown.iter().map(|path| {
        let nodes: Vec<String> = path.path.iter().map(|v| v.raw().to_string()).collect();
        (format!("path[{}]", nodes.join(">")), path.contribution)
    }));
    values
}

/// The named scalar quantities of a simulation, all smooth functionals of the fitted mechanisms:
/// the mean of each variable of a draw batch (interventional draws, noise samples), the policy
/// and baseline means and their paired difference, the mean unit effect, the mean of each
/// outcome in each world of a counterfactual, and, for a rollout, the mean of each of the first
/// `nodes` variables at each step (step-major).
pub(crate) fn simulation_quantities(
    outcome: &SimulationOutcome,
    nodes: usize,
) -> Option<Vec<(String, f64)>> {
    // An empty column has no mean: summarising it is unavailable rather than a zero mean.
    let mean = |column: &[f64]| -> Option<f64> {
        (!column.is_empty()).then(|| antecedent_kernels::mean(column))
    };
    let batch_means = |batch: &ValueBatch| -> Option<Vec<(String, f64)>> {
        (0..batch.n_nodes)
            .map(|node| Some((format!("draw_mean[{node}]"), mean(batch.column(node).ok()?)?)))
            .collect()
    };
    match outcome {
        SimulationOutcome::Sample(batch) => batch_means(batch),
        SimulationOutcome::NoiseSample { draws, .. } => batch_means(draws),
        SimulationOutcome::PolicyValue(value) => Some(vec![
            ("policy_mean".to_owned(), value.policy_mean),
            ("baseline_mean".to_owned(), value.baseline_mean),
            ("mean_difference".to_owned(), value.mean_difference),
        ]),
        SimulationOutcome::UnitEffects(ite) => Some(vec![("mean_ite".to_owned(), ite.mean_ite)]),
        SimulationOutcome::Counterfactual(result) => {
            let mut out = Vec::new();
            for world in 0..result.n_worlds {
                for &outcome in result.outcomes.iter() {
                    let column = result.outcome_column(world, outcome).ok()?;
                    out.push((
                        format!("world[{world}].outcome[{}]", outcome.as_usize()),
                        mean(column)?,
                    ));
                }
            }
            Some(out)
        }
        SimulationOutcome::Temporal(path) => {
            let limit = nodes.min(path.n_nodes);
            let mut out = Vec::with_capacity(path.steps.len() * limit);
            for (step, batch) in path.steps.iter().enumerate() {
                for node in 0..limit {
                    out.push((
                        format!("step[{step}].node[{node}]"),
                        mean(batch.column(node).ok()?)?,
                    ));
                }
            }
            Some(out)
        }
    }
}

/// One simulated path of a rollout: the value of each of the first `nodes` variables at each step
/// (step-major, first row), innovations included. Rolled out under a fresh seed in every refit,
/// these are draws from the predictive law of a path: the refit uncertainty mixed with the
/// innovation noise.
pub(crate) fn simulated_path_quantities(
    outcome: &SimulationOutcome,
    nodes: usize,
) -> Option<Vec<(String, f64)>> {
    let SimulationOutcome::Temporal(path) = outcome else {
        return None;
    };
    let limit = nodes.min(path.n_nodes);
    let mut out = Vec::with_capacity(path.steps.len() * limit);
    for (step, batch) in path.steps.iter().enumerate() {
        for node in 0..limit {
            out.push((format!("step[{step}].path[{node}]"), *batch.column(node).ok()?.first()?));
        }
    }
    Some(out)
}

/// The named scalar quantities of an attribution under the row `weights` of a replicate (the
/// anomaly scores average under them), or `None` for an outcome without scalar quantities
/// (intrinsic influence returns its own refits; a detection is a test).
pub(crate) fn attribution_quantities_weighted(
    outcome: &AttributionOutcome,
    weights: &[f64],
) -> Option<Vec<(String, f64)>> {
    Some(match outcome {
        AttributionOutcome::Change(result) => change_quantities(result),
        AttributionOutcome::RootCauses { attribution, .. } => change_quantities(attribution),
        AttributionOutcome::FeatureRelevance(features) => {
            features.iter().map(|f| (format!("feature[{}]", f.feature.raw()), f.score)).collect()
        }
        AttributionOutcome::Anomaly(scores) => scores
            .iter()
            .map(|s| {
                let (mut numerator, mut denominator) = (0.0_f64, 0.0_f64);
                for (score, &row) in s.scores.iter().zip(s.rows.iter()) {
                    let weight = weights.get(row).copied().unwrap_or(0.0);
                    numerator += weight * score;
                    denominator += weight;
                }
                let mean = if denominator > 0.0 { numerator / denominator } else { f64::NAN };
                (format!("mean_anomaly_score[{}]", s.target.raw()), mean)
            })
            .collect(),
        AttributionOutcome::UnitChange(result) => result
            .components
            .iter()
            .zip(result.mean_contributions.iter())
            .map(|(component, mean)| (format!("unit_mean[{}]", component.raw()), *mean))
            .collect(),
        AttributionOutcome::IntrinsicInfluence(result) => result
            .sources
            .iter()
            .map(|source| (format!("source[{}]", source.source.raw()), source.contribution))
            .collect(),
        _ => return None,
    })
}

/// The named scalar quantities of an attribution at its point answer, in the order its posterior
/// and bootstrap columns use: a change's total, components and paths; feature scores; the mean
/// anomaly score per target; the mean unit contributions; each source's variance share. `None`
/// for a mechanism-change detection, which is a test and has no scalar quantities.
#[must_use]
pub fn attribution_quantities(outcome: &AttributionOutcome) -> Option<Vec<(String, f64)>> {
    let rows = match outcome {
        AttributionOutcome::Anomaly(scores) => {
            scores.iter().flat_map(|s| s.rows.iter().copied()).max().map_or(0, |max| max + 1)
        }
        _ => 0,
    };
    attribution_quantities_weighted(outcome, &vec![1.0; rows])
}

/// The compiled model of the comparison structure a structure-change attribution declares, over
/// the variables of `graph`.
///
/// # Errors
///
/// A model with more nodes than a node id holds, or a comparison edge the graph refuses.
pub(crate) fn structure_comparison_model(
    graph: &Dag,
    comparison: &[(VariableId, VariableId)],
) -> Result<crate::gcm::CompiledCausalModel, CausalError> {
    let nodes = u32::try_from(graph.node_count()).map_err(|_| CausalError::Resource {
        message: "the model has more nodes than a node id holds".into(),
    })?;
    let mut comparison_graph = Dag::with_variables(nodes);
    for (from, to) in comparison {
        comparison_graph
            .insert_directed(DenseNodeId::from_raw(from.raw()), DenseNodeId::from_raw(to.raw()))
            .map_err(CausalError::rejected)?;
    }
    Ok(crate::gcm::CompiledCausalModel::compile(comparison_graph)?)
}

/// Run the attribution `query` states on `fitted`, with the engine every `attribute_*` entry
/// runs. The mechanism-fit refits of an intrinsic-influence target are the query's own unless
/// `fit` sets them.
#[allow(clippy::too_many_lines, reason = "one exhaustive match over every attribution target")]
pub(crate) fn run_attribute_with_fit(
    fitted: &FittedGcm,
    graph: &Dag,
    data: &TabularData,
    query: &AttributeQuery,
    fit: Option<AttributionFit>,
    reading: ChangeReading,
    ctx: &ExecutionContext,
) -> Result<AttributionOutcome, CausalError> {
    Ok(match &query.target {
        AttributeTarget::IntrinsicInfluence(target) => {
            let structure = match &target.noise {
                NoiseDeclaration::Structure { bidirected } => {
                    Some(noise_structure(graph, bidirected)?)
                }
                NoiseDeclaration::Undeclared | NoiseDeclaration::Independent => None,
            };
            let noise = match (&structure, &target.noise) {
                (Some(admg), _) => NoiseIndependence::Structure(admg),
                (None, NoiseDeclaration::Independent) => NoiseIndependence::Declared,
                (None, _) => NoiseIndependence::Undeclared,
            };
            let allocation = match target.allocation {
                InfluenceAllocation::Auto => IntrinsicAllocation::Auto,
                InfluenceAllocation::Exact => IntrinsicAllocation::Exact,
                InfluenceAllocation::Permutation { n_permutations } => {
                    IntrinsicAllocation::Permutation { n_permutations }
                }
            };
            let engine_query = IntrinsicInfluenceQuery::new(target.target)
                .with_allocation(allocation)
                .with_simulation(target.outer_draws, target.inner_draws, target.replicates)
                .with_fit_bootstrap(fit.map_or(target.fit_bootstrap_replicates, |f| f.replicates))
                .with_fit_resampling(fit.map_or(FitResampling::RowBootstrap, |f| f.resampling))
                .with_level(target.level)
                .with_max_sources(target.max_sources)
                .with_seed(target.seed);
            AttributionOutcome::IntrinsicInfluence(crate::gcm::attribute_intrinsic_influence(
                &fitted.model,
                noise,
                Some(data),
                &engine_query,
                ctx,
            )?)
        }
        AttributeTarget::RootCauses { change, estimator } => {
            let attribution = change_attribution(fitted, data, change, estimator, ctx)?;
            let ranks = crate::gcm::rank_root_causes(&attribution, ctx)?;
            AttributionOutcome::RootCauses { attribution, ranks }
        }
        AttributeTarget::DistributionChange { change, estimator } => {
            AttributionOutcome::Change(change_attribution(fitted, data, change, estimator, ctx)?)
        }
        AttributeTarget::Paths { sources, outcome, max_paths, max_len, through } => {
            let mut result = crate::gcm::attribute_paths(
                &fitted.model,
                sources,
                *outcome,
                *max_paths,
                *max_len,
                ctx,
            )?;
            if !through.is_empty() {
                // The same filter and total `attribute_path_specific` applies.
                let filtered: Vec<_> = result
                    .path_breakdown
                    .iter()
                    .filter(|path| through.iter().all(|node| path.path.iter().any(|v| v == node)))
                    .cloned()
                    .collect();
                result.total_change = filtered.iter().map(|path| path.contribution).sum();
                result.path_breakdown = Arc::from(filtered);
            }
            AttributionOutcome::Change(result)
        }
        AttributeTarget::FeatureRelevance { outcome, features, delta, n_samples, max_features } => {
            AttributionOutcome::FeatureRelevance(crate::gcm::attribute_feature_relevance_with(
                &fitted.model,
                data,
                *outcome,
                features,
                &antecedent_attribution::FeatureRelevanceOptions {
                    delta: *delta,
                    n_samples: *n_samples,
                    max_features: *max_features,
                },
                ctx,
            )?)
        }
        AttributeTarget::Anomaly(anomaly) => AttributionOutcome::Anomaly(
            crate::gcm::anomaly_attribution_query_with(&fitted.model, data, anomaly, ctx)?,
        ),
        AttributeTarget::UnitChange(unit) => AttributionOutcome::UnitChange(
            crate::gcm::attribute_unit_change(&fitted.model, data, unit, ctx)?,
        ),
        AttributeTarget::StructureChange { change, comparison, measure, n_samples, seed } => {
            let comparison_model = structure_comparison_model(graph, comparison)?;
            AttributionOutcome::Change(crate::gcm::attribute_structure_change(
                &fitted.model,
                &comparison_model,
                data,
                change,
                &crate::gcm::StructureChangeOptions {
                    measure: difference_measure(*measure),
                    n_samples: *n_samples,
                    seed: *seed,
                },
                ctx,
            )?)
        }
        AttributeTarget::MechanismChange { query: detection, test } => {
            let method = match test {
                MechanismChangeTest::LikelihoodRatio => MechanismChangeMethod::LikelihoodRatio,
                MechanismChangeTest::MeanDifference => MechanismChangeMethod::MeanDiff,
                MechanismChangeTest::ClassifierTwoSample => {
                    MechanismChangeMethod::ClassifierTwoSample
                }
                MechanismChangeTest::KernelTwoSample => MechanismChangeMethod::KernelTwoSample,
                MechanismChangeTest::ChangePoint => MechanismChangeMethod::ChangePoint,
            };
            AttributionOutcome::MechanismChange(run_mechanism_change(
                fitted, data, detection, method, reading, ctx,
            )?)
        }
    })
}

/// Run the simulation `query` states on `fitted`, with the engine every
/// `sample_do`, `policy_value` and `counterfactual_ite` entry runs.
pub(crate) fn run_simulate(
    fitted: &FittedGcm,
    data: &TabularData,
    query: &SimulateQuery,
    ctx: &ExecutionContext,
) -> Result<SimulationOutcome, CausalError> {
    Ok(match &query.target {
        SimulationTarget::InterventionalSample { interventions, draws, seed } => {
            let mut rng = CausalRng::from_seed(*seed);
            SimulationOutcome::Sample(crate::gcm::sample_do(
                &fitted.model,
                interventions,
                *draws,
                &mut rng,
                ctx,
            )?)
        }
        SimulationTarget::PolicyValue { interventions, outcome, draws } => {
            SimulationOutcome::PolicyValue(crate::gcm::policy_value(
                &fitted.model,
                interventions,
                *outcome,
                *draws,
                ctx,
            )?)
        }
        SimulationTarget::UnitEffects { treatment, outcome, active, control } => {
            SimulationOutcome::UnitEffects(crate::gcm::counterfactual_ite(
                fitted.model.clone(),
                data,
                *treatment,
                *outcome,
                *active,
                *control,
                ctx,
            )?)
        }
        SimulationTarget::NoiseSample { interventions, draws } => {
            let (draws, noise) =
                crate::gcm::sample_with_noise(&fitted.model, interventions, *draws, ctx)?;
            SimulationOutcome::NoiseSample { draws, noise }
        }
        SimulationTarget::NoiseReplay { interventions, noise, n_rows, policy_salt } => {
            let table = crate::gcm::noise_table_from_columns(&fitted.model, *n_rows, noise, ctx)?;
            SimulationOutcome::Sample(crate::gcm::replay_noise(
                &fitted.model,
                interventions,
                &table,
                *policy_salt,
                ctx,
            )?)
        }
        SimulationTarget::Counterfactual { interventions, outcomes } => {
            SimulationOutcome::Counterfactual(crate::gcm::counterfactual_policy(
                fitted.model.clone(),
                data,
                interventions,
                outcomes,
                ctx,
            )?)
        }
        SimulationTarget::TemporalRollout { schedule, n_rows, n_steps, window, history, seed } => {
            let initial = history_batches(fitted, history, *n_rows)?;
            let plan = crate::gcm::TemporalPlan {
                sequence: schedule,
                n_steps: *n_steps,
                n_rows: *n_rows,
                window: *window,
                initial_history: &initial,
            };
            let mut rng = CausalRng::from_seed(*seed);
            SimulationOutcome::Temporal(crate::gcm::sample_temporal_policy(
                &fitted.model,
                &plan,
                &mut rng,
                ctx,
            )?)
        }
        SimulationTarget::TemporalCounterfactual {
            schedule,
            n_rows,
            n_steps,
            window,
            panel,
            history,
        } => {
            let initial = history_batches(fitted, history, *n_rows)?;
            let tables = panel_tables(data, panel, *n_rows, *n_steps)?;
            let plan = crate::gcm::TemporalPlan {
                sequence: schedule,
                n_steps: *n_steps,
                n_rows: *n_rows,
                window: *window,
                initial_history: &initial,
            };
            SimulationOutcome::Temporal(crate::gcm::counterfactual_temporal_policy(
                fitted.model.clone(),
                &tables,
                &plan,
                ctx,
            )?)
        }
    })
}

/// Rows a simulation draws or replays, for its admission quote.
pub(crate) fn simulated_rows(target: &SimulationTarget, data_rows: u64) -> u64 {
    match target {
        SimulationTarget::InterventionalSample { draws, .. }
        | SimulationTarget::PolicyValue { draws, .. }
        | SimulationTarget::NoiseSample { draws, .. } => *draws as u64,
        SimulationTarget::NoiseReplay { n_rows, .. } => *n_rows as u64,
        SimulationTarget::UnitEffects { .. } | SimulationTarget::Counterfactual { .. } => data_rows,
        SimulationTarget::TemporalRollout { n_rows, n_steps, .. }
        | SimulationTarget::TemporalCounterfactual { n_rows, n_steps, .. } => {
            (*n_rows as u64).saturating_mul(*n_steps as u64)
        }
    }
}

/// Steps before time zero, most recent first, as one batch per step over every model node.
fn history_batches(
    fitted: &FittedGcm,
    history: &[f64],
    n_rows: usize,
) -> Result<Vec<ValueBatch>, CausalError> {
    let n_nodes = fitted.model.n_nodes();
    let width = n_nodes.saturating_mul(n_rows);
    if width == 0 || history.len() % width != 0 {
        return Err(crate::compile_reason!(
            "invalid_argument",
            "rollout history needs {n_rows} values for each of the model's {n_nodes} variables \
             at every step"
        ));
    }
    Ok(history
        .chunks(width)
        .map(|step| ValueBatch { n_rows, n_nodes, values: Arc::from(step) })
        .collect())
}

/// The factual panel as one table per step, columns named as the study's schema.
fn panel_tables(
    data: &TabularData,
    panel: &[f64],
    n_rows: usize,
    n_steps: usize,
) -> Result<Vec<TabularData>, CausalError> {
    let names: Vec<String> =
        data.schema().variables().iter().map(|variable| variable.name.to_string()).collect();
    let width = names.len().saturating_mul(n_rows);
    if width == 0 || panel.len() != width.saturating_mul(n_steps) {
        return Err(crate::compile_reason!(
            "invalid_argument",
            "the panel needs {n_rows} values for each of the {} schema variables at each of \
             {n_steps} steps",
            names.len()
        ));
    }
    panel
        .chunks(width)
        .map(|step| {
            TabularData::from_f64_columns(
                names.iter().zip(step.chunks(n_rows)).map(|(name, column)| (name.as_str(), column)),
            )
            .map_err(CausalError::from)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Review gate and provenance
// ---------------------------------------------------------------------------

/// The search bounds of `query` as a stable label.
fn bounds_label(query: &DiscoverQuery) -> String {
    let bounds = &query.bounds;
    let schedule = bounds.schedule.map_or_else(
        || "none".to_owned(),
        |s| format!("{}x{}+{}/{}", s.chains, s.draws, s.warmup, s.thin),
    );
    format!(
        "alpha={};max_cond_size={};max_lag={};schedule={schedule}",
        bounds.alpha, bounds.max_cond_size, bounds.max_lag
    )
}

/// Canonical digest of a discovery query.
fn query_digest(query: &DiscoverQuery) -> [u8; 32] {
    use antecedent_core::CanonicalEncode;
    let mut writer = CanonicalWriter::new("antecedent.discover_query");
    query.encode(&mut writer);
    writer.finish()
}

/// Accept the structure an executed discovery returned through its review
/// gate, and record where it came from.
///
/// `result` must be the stamped result of a prepared discovery handle and
/// `claim` the claim sealed over it. The accepted graph carries a
/// [`StructureProvenance`]: the method, its declared inference and returned
/// class, the query digest and search bounds, the data snapshot, the lowered
/// program and the claim, and how the review accepted it. A study prepared on
/// the graph carries that record in its contract.
///
/// # Errors
///
/// A result that is not an executed discovery, that no prepared handle
/// stamped, or whose claim seals another execution; and the review-gate
/// refusals of [`DiscoveredStructure::accept`].
pub fn accept_discovered(
    result: &StudyResult,
    claim: &ClaimEnvelope,
    accept_discovered: bool,
) -> Result<AcceptedGraph, CausalError> {
    let payload = result.discovery().ok_or(CausalError::Conflict {
        what: "result",
        detail: "the result is not an executed structure search",
    })?;
    let executed = result.executed_contract.as_ref().ok_or(CausalError::Conflict {
        what: "result",
        detail: "the discovery result carries no execution stamp; accept the result of a \
                 prepared discovery handle",
    })?;
    if executed.identities != claim.identities {
        return Err(CausalError::Conflict {
            what: "claim",
            detail: "the claim seals another execution than this discovery result",
        });
    }
    let program = result.lowered_identity.ok_or(CausalError::Conflict {
        what: "result",
        detail: "the discovery result carries no lowered program identity",
    })?;
    let pending = payload.structure.pending_marks();
    let accepted = payload.structure.accept(accept_discovered)?;
    let provenance = StructureProvenance {
        method: Arc::from(payload.query.method.id()),
        inference: Arc::from(payload.query.inference().as_str()),
        discovered_class: Arc::from(payload.query.output().as_str()),
        query: query_digest(&payload.query),
        search_bounds: Arc::from(bounds_label(&payload.query)),
        data_snapshot: *executed.identities.data_snapshot.as_bytes(),
        program: *program.as_bytes(),
        claim: *claim.claim_id.as_bytes(),
        review: Arc::from(if pending == 0 { "accepted" } else { "accepted_after_review" }),
        resolved_marks: pending,
    };
    Ok(accepted.with_provenance(Arc::new(provenance)))
}

/// A structure posterior accepted from a compiled discovery, with the search's
/// provenance.
///
/// A study takes it through [`crate::StudyBuilder::accepted_graph_posterior`]:
/// the posterior's atoms and weights become the study's graph posterior and
/// the provenance enters its contract and program identity.
/// [`Self::map_dag`] instead keeps the single most probable DAG as an
/// accepted graph.
#[derive(Clone, Debug)]
pub struct AcceptedPosterior {
    posterior: GraphPosterior,
    provenance: Arc<StructureProvenance>,
    temporal: bool,
}

impl AcceptedPosterior {
    /// The posterior over structures.
    #[must_use]
    pub const fn posterior(&self) -> &GraphPosterior {
        &self.posterior
    }

    /// The search and acceptance this posterior came from.
    #[must_use]
    pub fn provenance(&self) -> &StructureProvenance {
        &self.provenance
    }

    /// The posterior and its provenance.
    pub(crate) fn into_parts(self) -> (GraphPosterior, Arc<StructureProvenance>) {
        (self.posterior, self.provenance)
    }

    /// The most probable DAG atom as an accepted static graph, carrying the
    /// posterior's provenance with the review recorded as `posterior_map`.
    /// Ties go to the first atom in the posterior's order.
    ///
    /// # Errors
    ///
    /// `route_not_supported` for a temporal posterior, for atoms that are not
    /// DAGs, or for a search over a variable list that is not the schema's
    /// columns in order (the atoms index the searched variables by position);
    /// the posterior's own refusals for an atom that is not acyclic.
    pub fn map_dag(&self) -> Result<AcceptedGraph, CausalError> {
        if self.temporal {
            return Err(crate::compile_reason!(
                "route_not_supported",
                "a temporal structure posterior has no single static DAG; supply it as a graph \
                 posterior instead"
            ));
        }
        if self.posterior.atom_kind != antecedent_discovery::GraphPosteriorAtomKind::Dag {
            return Err(crate::compile_reason!(
                "route_not_supported",
                "the posterior's atoms are not DAGs; supply it as a graph posterior instead"
            ));
        }
        let best = self
            .posterior
            .weights
            .iter()
            .enumerate()
            .fold(None::<(usize, f64)>, |best, (index, &weight)| match best {
                Some((_, top)) if top >= weight => best,
                _ => Some((index, weight)),
            })
            .map(|(index, _)| index)
            .ok_or_else(|| {
                crate::compile_reason!("invalid_argument", "the structure posterior has no atoms")
            })?;
        let mask = self.posterior.atoms[best];
        let dag = antecedent_discovery::adjacency_mask::dag_from_adjacency_mask(
            mask,
            self.posterior.n_vars,
        )?;
        let mut provenance = (*self.provenance).clone();
        provenance.review = Arc::from("posterior_map");
        let mut accepted = AcceptedGraph::dag(dag);
        if let Some(algorithm) = self.posterior.algorithm.as_deref() {
            accepted = accepted.with_discovery_algorithm(algorithm);
        }
        Ok(accepted.with_provenance(Arc::new(provenance)))
    }
}

/// Accept the structure posterior an executed discovery returned, recording
/// where it came from.
///
/// `result` must be the stamped result of a prepared discovery handle whose
/// method samples or enumerates a posterior, and `claim` the claim sealed over
/// it. A posterior has no pending marks, so the review is recorded as
/// `posterior`.
///
/// # Errors
///
/// A result that is not an executed structure posterior, that no prepared
/// handle stamped, or whose claim seals another execution; and a searched
/// variable list that is not the schema's columns in order.
pub fn accept_discovered_posterior(
    result: &StudyResult,
    claim: &ClaimEnvelope,
) -> Result<AcceptedPosterior, CausalError> {
    let payload = result.discovery().ok_or(CausalError::Conflict {
        what: "result",
        detail: "the result is not an executed structure search",
    })?;
    let DiscoveredStructure::Posterior(posterior) = &payload.structure else {
        return Err(crate::compile_reason!(
            "route_not_supported",
            "the {} search returned no structure posterior; accept it with accept_discovered",
            payload.query.method.id()
        ));
    };
    let executed = result.executed_contract.as_ref().ok_or(CausalError::Conflict {
        what: "result",
        detail: "the discovery result carries no execution stamp; accept the result of a \
                 prepared discovery handle",
    })?;
    if executed.identities != claim.identities {
        return Err(CausalError::Conflict {
            what: "claim",
            detail: "the claim seals another execution than this discovery result",
        });
    }
    let program = result.lowered_identity.ok_or(CausalError::Conflict {
        what: "result",
        detail: "the discovery result carries no lowered program identity",
    })?;
    let positional = payload
        .query
        .variables
        .iter()
        .enumerate()
        .all(|(index, variable)| variable.raw() as usize == index);
    if !positional {
        return Err(crate::compile_reason!(
            "route_not_supported",
            "the posterior's atoms index the searched variables by position; search the \
             schema's columns in order to accept the posterior as the study's structure"
        ));
    }
    let provenance = StructureProvenance {
        method: Arc::from(payload.query.method.id()),
        inference: Arc::from(payload.query.inference().as_str()),
        discovered_class: Arc::from(payload.query.output().as_str()),
        query: query_digest(&payload.query),
        search_bounds: Arc::from(bounds_label(&payload.query)),
        data_snapshot: *executed.identities.data_snapshot.as_bytes(),
        program: *program.as_bytes(),
        claim: *claim.claim_id.as_bytes(),
        review: Arc::from("posterior"),
        resolved_marks: 0,
    };
    Ok(AcceptedPosterior {
        posterior: posterior.clone(),
        provenance: Arc::new(provenance),
        temporal: payload.query.method.is_temporal(),
    })
}

impl crate::Study {
    /// Provenance of the structure this study reads, when it was accepted
    /// from a compiled discovery: on its accepted graph, or on its accepted
    /// graph posterior.
    pub(crate) fn structure_provenance(&self) -> Option<&StructureProvenance> {
        self.graph.provenance().or(self.config.posterior_provenance.as_deref())
    }

    /// Shared handle of [`Self::structure_provenance`].
    pub(crate) fn structure_provenance_shared(&self) -> Option<Arc<StructureProvenance>> {
        self.graph.provenance_shared().or_else(|| self.config.posterior_provenance.clone())
    }
}

/// Attach the executable specifications a builder carried to the study it built.
///
/// # Errors
///
/// A specification beside a query it does not execute, a discovery query
/// without its specification (unless only inspecting), or a specification
/// that does not state the study's query.
pub(crate) fn attach_specifications(
    study: &mut crate::Study,
    discovery: Option<Arc<DiscoverySpec>>,
    mechanisms: Option<Arc<MechanismProgram>>,
    inspect_only: bool,
) -> Result<(), CausalError> {
    match (&study.query, &discovery) {
        (CausalQuery::Discover(query), Some(spec)) => spec.check_query(query)?,
        (CausalQuery::Discover(_), None) if !inspect_only => {
            return Err(crate::compile_reason!(
                "required_option_missing",
                "a structure-discovery study needs its executable search; build it with \
                 StudyBuilder::discovery"
            ));
        }
        (_, Some(_)) => {
            return Err(crate::compile_reason!(
                "option_not_applicable",
                "a discovery specification applies to a structure-discovery query only"
            ));
        }
        (_, None) => {}
    }
    if mechanisms.is_some()
        && !matches!(study.query, CausalQuery::Attribute(_) | CausalQuery::Simulate(_))
    {
        return Err(crate::compile_reason!(
            "option_not_applicable",
            "a mechanism program applies to an attribution or simulation query only"
        ));
    }
    if discovery.is_some() || mechanisms.is_some() {
        let config = Arc::make_mut(&mut study.config);
        config.discovery = discovery;
        config.mechanisms = mechanisms;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// One-call execution
// ---------------------------------------------------------------------------

/// Stage a cancelled structure search reports, as the search engines report it.
const DISCOVER_CANCELLED_STAGE: &str = "discover";

/// A compiled route prepared and executed once on its retained data: the
/// prepared handle and the result it stamped.
///
/// Language bindings and one-call entry points run a discovery, attribution
/// or simulation through this, so every answer they return keeps the handle
/// that can state its contract, its claim and its inspection record.
#[derive(Clone, Debug)]
pub struct CompiledExecution {
    prepared: Arc<crate::PreparedStudy>,
    result: Arc<StudyResult>,
}

impl CompiledExecution {
    /// Compile and prepare `study`, then estimate it once on its retained data.
    ///
    /// # Errors
    ///
    /// The refusals of [`crate::Study::prepare`] and
    /// [`crate::PreparedStudy::estimate_retained`].
    pub fn run(study: &crate::Study, ctx: &ExecutionContext) -> Result<Self, CausalError> {
        // A structure search is one stage to its caller: cancellation anywhere in its
        // preparation or execution is reported as cancelled during the search, as the
        // search engines report it.
        let searching = matches!(study.query, CausalQuery::Discover(_));
        let stage = |error: CausalError| match error {
            CausalError::Cancelled { .. } if searching => {
                CausalError::Cancelled { stage: DISCOVER_CANCELLED_STAGE }
            }
            other => other,
        };
        if searching && ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled { stage: DISCOVER_CANCELLED_STAGE });
        }
        let prepared = study.prepare(ctx).map_err(stage)?;
        let result = prepared.estimate_retained(ctx).map_err(stage)?;
        Ok(Self { prepared: Arc::new(prepared), result: Arc::new(result) })
    }

    /// The prepared handle.
    #[must_use]
    pub fn prepared(&self) -> &crate::PreparedStudy {
        &self.prepared
    }

    /// The stamped result.
    #[must_use]
    pub fn result(&self) -> &StudyResult {
        &self.result
    }

    /// Identity of the lowered program the result executed.
    #[must_use]
    pub fn lowered_identity(&self) -> Option<antecedent_core::SemanticDigest> {
        self.result.lowered_identity
    }

    /// The contract the result executed under and the claim sealed over it.
    ///
    /// # Errors
    ///
    /// The refusals of [`crate::PreparedStudy::execution_contract`].
    pub fn contract_and_claim(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<(crate::CausalContract, ClaimEnvelope), CausalError> {
        self.prepared.execution_contract(&self.result, ctx)
    }

    /// The cheap inspection record of the study, without identification products.
    ///
    /// # Errors
    ///
    /// Canonical-encoding failures.
    pub fn inspect(&self) -> Result<crate::CausalContract, CausalError> {
        self.prepared.inspect()
    }

    /// Accept the discovered structure through its review gate, recording its provenance.
    ///
    /// # Errors
    ///
    /// The refusals of [`accept_discovered`] and of the contract and claim.
    pub fn accept(
        &self,
        accept_discovered_marks: bool,
        ctx: &ExecutionContext,
    ) -> Result<AcceptedGraph, CausalError> {
        let (_, claim) = self.contract_and_claim(ctx)?;
        accept_discovered(&self.result, &claim, accept_discovered_marks)
    }
}

impl CompiledExecution {
    /// Accept the discovered structure posterior, recording its provenance.
    ///
    /// # Errors
    ///
    /// The refusals of [`accept_discovered_posterior`] and of the contract and claim.
    pub fn accept_posterior(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<AcceptedPosterior, CausalError> {
        let (_, claim) = self.contract_and_claim(ctx)?;
        accept_discovered_posterior(&self.result, &claim)
    }

    /// Accept the discovered structure as a fully oriented static DAG, the
    /// structure a mechanism fit needs, keeping its algorithm id and provenance.
    ///
    /// Pending directed edges are accepted; a CPDAG with no undirected mark is
    /// read as the DAG it states. Undirected or circle marks are never oriented
    /// here.
    ///
    /// # Errors
    ///
    /// The refusals of [`Self::accept`], and `route_not_supported` for a
    /// structure that keeps undirected or circle marks or is temporal.
    pub fn accept_dag(&self, ctx: &ExecutionContext) -> Result<AcceptedGraph, CausalError> {
        let accepted = self.accept(true, ctx)?;
        let dag = match accepted.graph_ref() {
            crate::accepted::GraphRef::Dag(_) => return Ok(accepted),
            crate::accepted::GraphRef::Cpdag(cpdag) => cpdag.try_into_dag().map_err(|_| {
                crate::compile_reason!(
                    "route_not_supported",
                    "cannot coerce the discovered CPDAG into a DAG: undirected marks remain; \
                     orient them (or supply a DAG) before fitting mechanisms"
                )
            })?,
            _ => {
                return Err(crate::compile_reason!(
                    "route_not_supported",
                    "a fitted model needs a fully oriented static DAG; the discovered {} keeps \
                     marks a DAG cannot state",
                    accepted.class().as_str()
                ));
            }
        };
        let mut oriented = AcceptedGraph::dag(dag);
        if let Some(algorithm) = accepted.algorithm_id() {
            oriented = oriented.with_discovery_algorithm(algorithm);
        }
        if let Some(provenance) = accepted.provenance_shared() {
            oriented = oriented.with_provenance(provenance);
        }
        Ok(oriented)
    }
}
