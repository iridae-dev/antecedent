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
//! carry calibration status `not_measured`.
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

    /// The checked query this search answers over `variables`.
    ///
    /// # Errors
    ///
    /// Returns a typed query error when the variables, method, or bounds are invalid.
    /// No configuration identity is produced for an invalid declaration.
    pub fn query(
        &self,
        variables: impl Into<Arc<[VariableId]>>,
    ) -> Result<DiscoverQuery, antecedent_core::QueryError> {
        let mut query = DiscoverQuery::new(variables, self.method.clone(), self.bounds());
        query.validate()?;
        query.bounds.configuration = self.configuration();
        Ok(query)
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
            configuration: [0; 32],
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
        fn uses_ges(method: &DiscoveryMethod) -> bool {
            match method {
                DiscoveryMethod::Ges { .. } => true,
                DiscoveryMethod::BootstrapStability { base, .. } => uses_ges(base),
                _ => false,
            }
        }
        fn uses_dag_bic(method: &DiscoveryMethod) -> bool {
            match method {
                DiscoveryMethod::Ges { .. }
                | DiscoveryMethod::ExactDagPosterior
                | DiscoveryMethod::ExactDagMarginals
                | DiscoveryMethod::OrderMcmc { .. }
                | DiscoveryMethod::StructureMcmc
                | DiscoveryMethod::CiScreenedPosterior { .. }
                | DiscoveryMethod::DbnPosterior { .. } => true,
                DiscoveryMethod::ClassPosterior { base }
                | DiscoveryMethod::BootstrapStability { base, .. } => uses_dag_bic(base),
                _ => false,
            }
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
        // The historical Insert-Turn-Delete search is a different program.
        if uses_ges(&self.method) {
            writer.text("ges.schedule.insert-delete.v2");
        }
        let gaussian_score = match &self.search {
            SearchParams::Static(_) => uses_ges(&self.method),
            SearchParams::Bayesian(params) => {
                params.score_family == antecedent_state::GraphScoreFamily::GaussianBic
            }
            SearchParams::Screened { bayesian, .. } => {
                bayesian.score_family == antecedent_state::GraphScoreFamily::GaussianBic
            }
            SearchParams::Temporal(_) => false,
        };
        if gaussian_score && uses_dag_bic(&self.method) {
            writer.text("antecedent.discovery.gaussian_bic.coefficients_and_variance.v2");
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
        query.validate().map_err(CausalError::from)?;
        if self.query(Arc::clone(&query.variables)).map_err(CausalError::from)? != *query {
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
            Self::Posterior(posterior) => posterior.algorithm().map(|algorithm| &**algorithm),
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

mod discovery_route;

pub use discovery_route::panel_of_units;
pub(crate) use discovery_route::{admit, discovery_quote, run_discovery};

mod model_routes;

pub use model_routes::attribution_quantities;
pub(crate) use model_routes::{
    AttributionFit, ChangeReading, attribution_quantities_weighted, change_quantities,
    difference_measure, model_quote, run_attribute_with_fit, run_mechanism_change, run_simulate,
    simulated_path_quantities, simulated_rows, simulation_quantities, structure_comparison_model,
};

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
mod acceptance;

pub(crate) use acceptance::attach_specifications;
pub use acceptance::{AcceptedPosterior, accept_discovered, accept_discovered_posterior};

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

#[cfg(test)]
mod ges_schedule_identity_tests {
    use super::*;

    fn params() -> StaticDiscoverParams {
        StaticDiscoverParams {
            alpha: 0.05,
            max_cond_size: 2,
            fdr: None,
            ci: crate::discovery_defaults::resolve_ci("parcorr", None).unwrap(),
            screen_pc: false,
            max_subset: None,
            knowledge: antecedent_discovery::BackgroundKnowledge::default(),
        }
    }

    fn historical_bayesian_configuration(params: &BayesianDiscoverParams) -> [u8; 32] {
        let mut writer = CanonicalWriter::new("antecedent.discovery_configuration");
        writer
            .text("none")
            .text("bayesian")
            .text(&format!("{:?}", params.prior))
            .text(&format!("{:?}", params.score_family))
            .word(0)
            .opt_word(None)
            .opt_text(None);
        writer.finish()
    }

    #[test]
    fn gaussian_score_revision_changes_only_affected_configuration_identities() {
        let gaussian = BayesianDiscoverParams::default();
        let old = historical_bayesian_configuration(&gaussian);
        let dag = DiscoverySpec::exact_dag_posterior(gaussian.clone());
        let current = dag.configuration();
        assert_ne!(current, old);
        let variables: Arc<[VariableId]> =
            Arc::from([VariableId::from_raw(0), VariableId::from_raw(1)]);
        let query = dag.query(variables).unwrap();
        assert!(dag.check_query(&query).is_ok());
        let mut historical = query;
        historical.bounds.configuration = old;
        assert!(dag.check_query(&historical).is_err());
        assert_eq!(dag.clone().equivalence_classes().configuration(), current);
        assert_eq!(DiscoverySpec::exact_dag_marginals(gaussian.clone()).configuration(), current);
        let mut laplace = gaussian.clone();
        laplace.score_family = antecedent_state::GraphScoreFamily::LaplaceBic;
        let laplace_old = historical_bayesian_configuration(&laplace);
        assert_eq!(
            DiscoverySpec::order_mcmc(laplace, GraphMcmcSchedule::default(), false).configuration(),
            laplace_old
        );
        // The PAG score uses a distinct multivariate implementation.
        assert_eq!(DiscoverySpec::exact_pag_posterior(gaussian).configuration(), old);
        let ges_params = params();
        let mut old_ges = CanonicalWriter::new("antecedent.discovery_configuration");
        old_ges
            .text("parcorr")
            .text("static")
            .text(&format!("{:?}", ges_params.fdr))
            .word(u64::from(ges_params.screen_pc))
            .opt_word(ges_params.max_subset.map(|value| value as u64))
            .word(0) // empty background knowledge
            .text("ges.schedule.insert-delete.v2")
            .word(0) // no regime assignment
            .opt_word(None)
            .opt_text(None);
        let ges = DiscoverySpec::ges(ges_params, "parcorr");
        assert_ne!(ges.configuration(), old_ges.finish());
        assert_eq!(ges.clone().bootstrap_stability(5).configuration(), ges.configuration());
    }

    #[test]
    fn ges_schedule_changes_identity_and_refuses_a_historical_query() {
        // Captured from the previous compiled implementation with these options.
        let historical_hex = "db511fe8b5b11e0f1cfc943df8ab15ded030b27fdf976f5b00e38492192570fe";
        let mut historical = [0_u8; 32];
        for (index, byte) in historical.iter_mut().enumerate() {
            *byte = u8::from_str_radix(&historical_hex[index * 2..index * 2 + 2], 16).unwrap();
        }
        let variables: Arc<[VariableId]> =
            Arc::from([VariableId::from_raw(0), VariableId::from_raw(1)]);
        let ges = DiscoverySpec::ges(params(), "parcorr");
        let query = ges.query(Arc::clone(&variables)).unwrap();
        assert_ne!(query.bounds.configuration, historical);
        assert!(ges.check_query(&query).is_ok());
        let mut legacy = query.clone();
        legacy.bounds.configuration = historical;
        assert!(ges.check_query(&legacy).is_err());
        let wrapped = ges.clone().bootstrap_stability(5).query(Arc::clone(&variables)).unwrap();
        assert_eq!(wrapped.bounds.configuration, query.bounds.configuration);
        for spec in [
            DiscoverySpec::pc(params(), "parcorr"),
            DiscoverySpec::pc(params(), "parcorr").bootstrap_stability(5),
        ] {
            assert_eq!(
                spec.query(Arc::clone(&variables)).unwrap().bounds.configuration,
                historical
            );
        }
    }
}
