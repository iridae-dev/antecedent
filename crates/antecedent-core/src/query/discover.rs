//! Structure-discovery query: the variables searched, the search method with
//! its declared inference mode, and the bounds the search runs under.
//!
//! A discovery query names a structure search, not a causal contrast. Its
//! answer is a graph or graph class (a DAG, a CPDAG or PAG that stays a class,
//! their temporal counterparts, one temporal class per regime) or a posterior
//! over structures, and it reaches downstream analyses only through the review
//! gate of an accepted graph.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use crate::canonical_encode::{CanonicalEncode, count, encode_variables, flag};
use crate::identity::CanonicalWriter;
use crate::ids::VariableId;

use super::error::QueryError;

/// Inference mode a discovery method declares.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum DiscoveryInference {
    /// Constraint- or score-based search that returns one structure or class.
    Frequentist,
    /// Posterior over structures under a declared prior and score.
    Bayesian,
}

impl DiscoveryInference {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Frequentist => "frequentist",
            Self::Bayesian => "bayesian",
        }
    }
}

/// The structure class a discovery method returns.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum DiscoveredClass {
    /// One static DAG.
    Dag,
    /// A static Markov equivalence class; undirected marks stay part of the class.
    Cpdag,
    /// A static partial ancestral graph; circle marks stay part of the class.
    Pag,
    /// One lagged temporal DAG.
    TemporalDag,
    /// A temporal Markov equivalence class.
    TemporalCpdag,
    /// A temporal partial ancestral graph.
    TemporalPag,
    /// One temporal CPDAG per declared regime.
    RegimeTemporalCpdags,
    /// A posterior over static DAGs.
    DagPosterior,
    /// The exact edge marginals of a static DAG posterior, without graph atoms.
    DagEdgeMarginals,
    /// A posterior over bounded-lag DBN templates.
    TemporalDagPosterior,
}

impl DiscoveredClass {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Dag => "dag",
            Self::Cpdag => "cpdag",
            Self::Pag => "pag",
            Self::TemporalDag => "temporal_dag",
            Self::TemporalCpdag => "temporal_cpdag",
            Self::TemporalPag => "temporal_pag",
            Self::RegimeTemporalCpdags => "regime_temporal_cpdags",
            Self::DagPosterior => "dag_posterior",
            Self::DagEdgeMarginals => "dag_edge_marginals",
            Self::TemporalDagPosterior => "temporal_dag_posterior",
        }
    }

    /// Whether the class is over lagged (temporal) structures.
    #[must_use]
    pub const fn is_temporal(self) -> bool {
        matches!(
            self,
            Self::TemporalDag
                | Self::TemporalCpdag
                | Self::TemporalPag
                | Self::RegimeTemporalCpdags
                | Self::TemporalDagPosterior
        )
    }

    /// Whether the answer is an equivalence class whose unresolved marks are
    /// part of the class (a CPDAG or PAG, static or temporal), not a set of
    /// orientations still to be chosen.
    #[must_use]
    pub const fn is_equivalence_class(self) -> bool {
        matches!(
            self,
            Self::Cpdag
                | Self::Pag
                | Self::TemporalCpdag
                | Self::TemporalPag
                | Self::RegimeTemporalCpdags
        )
    }
}

/// Soft weighting of a CI-screened posterior's candidate edges.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub enum ScreenWeight {
    /// Uniform among the screened candidates.
    #[default]
    None,
    /// Log Bayes factor for dependence.
    BayesFactor,
    /// Posterior probability of dependence.
    PosteriorDependence,
}

impl ScreenWeight {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::BayesFactor => "bayes_factor",
            Self::PosteriorDependence => "posterior_dependence",
        }
    }
}

impl CanonicalEncode for ScreenWeight {
    fn encode(&self, writer: &mut CanonicalWriter) {
        writer.word(match self {
            Self::None => 1,
            Self::BayesFactor => 2,
            Self::PosteriorDependence => 3,
        });
    }
}

/// A structure-search method and the options that change what it returns.
///
/// The conditional-independence test, multiple-testing rule, background
/// knowledge, prior and score enter through [`SearchBounds::configuration`].
#[derive(Clone, Debug, PartialEq)]
pub enum DiscoveryMethod {
    /// Static PC: a CPDAG.
    Pc,
    /// Static FCI: a PAG.
    Fci,
    /// Static RFCI: a PAG.
    Rfci,
    /// Greedy equivalence search (Gaussian BIC): a CPDAG.
    Ges {
        /// Soft PC-skeleton screening of insert candidates.
        screen_pc: bool,
        /// Cap on the T/H subset enumeration; `None` uses the method default.
        max_subset: Option<u32>,
    },
    /// `DirectLiNGAM`: a DAG under a linear non-Gaussian model.
    DirectLingam {
        /// Absolute coefficient below which an edge is pruned.
        prune_threshold: f64,
    },
    /// NOTEARS continuous structure search: a DAG under a linear SEM.
    Notears {
        /// L1 penalty.
        lambda: f64,
        /// Absolute weight below which an edge is dropped.
        threshold: f64,
        /// Standardize columns before the search.
        standardize: bool,
    },
    /// Lagged PCMCI: a temporal DAG of lagged links.
    Pcmci,
    /// PCMCI+: a temporal CPDAG including contemporaneous links.
    PcmciPlus,
    /// LPCMCI: a temporal PAG under latent confounding.
    Lpcmci,
    /// J-PCMCI+ over several environments: a temporal CPDAG.
    JpcmciPlus,
    /// Regime PCMCI over a supplied regime assignment: one temporal CPDAG per regime.
    Rpcmci {
        /// Shortest regime segment admitted; `None` uses the method default.
        min_regime_len: Option<u32>,
    },
    /// Exact DAG posterior by enumeration (small graphs).
    ExactDagPosterior,
    /// Exact DAG posterior edge marginals (medium graphs, no atoms).
    ExactDagMarginals,
    /// Order MCMC DAG posterior.
    OrderMcmc {
        /// Refuse a chain set whose convergence diagnostics fail.
        diagnostics_gate: bool,
    },
    /// Structure MCMC DAG posterior.
    StructureMcmc,
    /// PC-skeleton screening followed by structure MCMC.
    CiScreenedPosterior {
        /// Soft weighting of the screened candidates.
        weight: ScreenWeight,
    },
    /// Bounded-lag DBN template posterior.
    DbnPosterior {
        /// Sample by MCMC even where exact enumeration is feasible.
        force_mcmc: bool,
    },
}

impl DiscoveryMethod {
    /// Stable algorithm id, written to the logical plan's `discovery_algorithm`.
    #[must_use]
    pub const fn id(&self) -> &'static str {
        match self {
            Self::Pc => "pc",
            Self::Fci => "fci",
            Self::Rfci => "rfci",
            Self::Ges { .. } => "ges",
            Self::DirectLingam { .. } => "direct_lingam",
            Self::Notears { .. } => "notears",
            Self::Pcmci => "pcmci",
            Self::PcmciPlus => "pcmci_plus",
            Self::Lpcmci => "lpcmci",
            Self::JpcmciPlus => "jpcmci_plus",
            Self::Rpcmci { .. } => "rpcmci",
            Self::ExactDagPosterior => "exact_dag_posterior",
            Self::ExactDagMarginals => "exact_dag_marginals",
            Self::OrderMcmc { .. } => "order_mcmc",
            Self::StructureMcmc => "structure_mcmc",
            Self::CiScreenedPosterior { .. } => "ci_screened_posterior",
            Self::DbnPosterior { .. } => "dbn_posterior",
        }
    }

    /// The inference mode the method declares.
    #[must_use]
    pub const fn inference(&self) -> DiscoveryInference {
        match self {
            Self::ExactDagPosterior
            | Self::ExactDagMarginals
            | Self::OrderMcmc { .. }
            | Self::StructureMcmc
            | Self::CiScreenedPosterior { .. }
            | Self::DbnPosterior { .. } => DiscoveryInference::Bayesian,
            Self::Pc
            | Self::Fci
            | Self::Rfci
            | Self::Ges { .. }
            | Self::DirectLingam { .. }
            | Self::Notears { .. }
            | Self::Pcmci
            | Self::PcmciPlus
            | Self::Lpcmci
            | Self::JpcmciPlus
            | Self::Rpcmci { .. } => DiscoveryInference::Frequentist,
        }
    }

    /// The structure class the method returns.
    #[must_use]
    pub const fn output(&self) -> DiscoveredClass {
        match self {
            Self::Pc | Self::Ges { .. } => DiscoveredClass::Cpdag,
            Self::Fci | Self::Rfci => DiscoveredClass::Pag,
            Self::DirectLingam { .. } | Self::Notears { .. } => DiscoveredClass::Dag,
            Self::Pcmci => DiscoveredClass::TemporalDag,
            Self::PcmciPlus | Self::JpcmciPlus => DiscoveredClass::TemporalCpdag,
            Self::Lpcmci => DiscoveredClass::TemporalPag,
            Self::Rpcmci { .. } => DiscoveredClass::RegimeTemporalCpdags,
            Self::ExactDagPosterior
            | Self::OrderMcmc { .. }
            | Self::StructureMcmc
            | Self::CiScreenedPosterior { .. } => DiscoveredClass::DagPosterior,
            Self::ExactDagMarginals => DiscoveredClass::DagEdgeMarginals,
            Self::DbnPosterior { .. } => DiscoveredClass::TemporalDagPosterior,
        }
    }

    /// Whether the method searches lagged structure over time series.
    #[must_use]
    pub const fn is_temporal(&self) -> bool {
        self.output().is_temporal()
    }

    /// Whether the method decides adjacencies with conditional-independence tests,
    /// so its significance level is part of what it returns.
    #[must_use]
    pub const fn uses_independence_tests(&self) -> bool {
        matches!(
            self,
            Self::Pc
                | Self::Fci
                | Self::Rfci
                | Self::Ges { .. }
                | Self::Pcmci
                | Self::PcmciPlus
                | Self::Lpcmci
                | Self::JpcmciPlus
                | Self::Rpcmci { .. }
                | Self::CiScreenedPosterior { .. }
        )
    }

    /// Whether the method samples its posterior by MCMC and needs a schedule.
    #[must_use]
    pub const fn samples_posterior(&self) -> bool {
        matches!(
            self,
            Self::OrderMcmc { .. }
                | Self::StructureMcmc
                | Self::CiScreenedPosterior { .. }
                | Self::DbnPosterior { .. }
        )
    }
}

/// The tag, then every option that changes the search.
impl CanonicalEncode for DiscoveryMethod {
    fn encode(&self, writer: &mut CanonicalWriter) {
        match self {
            Self::Pc => {
                writer.word(1);
            }
            Self::Fci => {
                writer.word(2);
            }
            Self::Rfci => {
                writer.word(3);
            }
            Self::Ges { screen_pc, max_subset } => {
                writer.word(4).word(flag(*screen_pc)).opt_word(max_subset.map(u64::from));
            }
            Self::DirectLingam { prune_threshold } => {
                writer.word(5).real(*prune_threshold);
            }
            Self::Notears { lambda, threshold, standardize } => {
                writer.word(6).real(*lambda).real(*threshold).word(flag(*standardize));
            }
            Self::Pcmci => {
                writer.word(7);
            }
            Self::PcmciPlus => {
                writer.word(8);
            }
            Self::Lpcmci => {
                writer.word(9);
            }
            Self::JpcmciPlus => {
                writer.word(10);
            }
            Self::Rpcmci { min_regime_len } => {
                writer.word(11).opt_word(min_regime_len.map(u64::from));
            }
            Self::ExactDagPosterior => {
                writer.word(12);
            }
            Self::ExactDagMarginals => {
                writer.word(13);
            }
            Self::OrderMcmc { diagnostics_gate } => {
                writer.word(14).word(flag(*diagnostics_gate));
            }
            Self::StructureMcmc => {
                writer.word(15);
            }
            Self::CiScreenedPosterior { weight } => {
                writer.word(16);
                weight.encode(writer);
            }
            Self::DbnPosterior { force_mcmc } => {
                writer.word(17).word(flag(*force_mcmc));
            }
        }
    }
}

/// MCMC schedule of a sampled structure posterior.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct SearchSchedule {
    /// Independent chains.
    pub chains: u32,
    /// Warmup draws discarded per chain.
    pub warmup: u32,
    /// Post-warmup draws per chain, before thinning.
    pub draws: u32,
    /// Keep every `thin`-th post-warmup draw.
    pub thin: u32,
}

impl CanonicalEncode for SearchSchedule {
    fn encode(&self, writer: &mut CanonicalWriter) {
        writer
            .word(u64::from(self.chains))
            .word(u64::from(self.warmup))
            .word(u64::from(self.draws))
            .word(u64::from(self.thin));
    }
}

/// The bounds a structure search runs under.
#[derive(Clone, Debug, PartialEq)]
pub struct SearchBounds {
    /// Significance level of the independence tests (constraint methods and
    /// screening); ignored by methods that test nothing.
    pub alpha: f64,
    /// Largest conditioning set a test may use.
    pub max_cond_size: u32,
    /// Largest lag searched; `0` for a static search.
    pub max_lag: u32,
    /// MCMC schedule of a sampled posterior.
    pub schedule: Option<SearchSchedule>,
    /// Digest of the executable configuration the search runs: the
    /// conditional-independence test, the multiple-testing rule, the
    /// background knowledge, the multi-dataset or regime constraints, the
    /// structural prior and the score. Two searches that differ in any of
    /// them are different queries.
    pub configuration: [u8; 32],
}

impl CanonicalEncode for SearchBounds {
    fn encode(&self, writer: &mut CanonicalWriter) {
        writer.real(self.alpha).word(u64::from(self.max_cond_size)).word(u64::from(self.max_lag));
        match &self.schedule {
            Some(schedule) => {
                writer.word(1);
                schedule.encode(writer);
            }
            None => {
                writer.word(0);
            }
        }
        writer.bytes(&self.configuration);
    }
}

/// Structure-discovery query.
#[derive(Clone, Debug, PartialEq)]
pub struct DiscoverQuery {
    /// Variables searched, in column order.
    pub variables: Arc<[VariableId]>,
    /// Search method.
    pub method: DiscoveryMethod,
    /// Search bounds.
    pub bounds: SearchBounds,
}

/// Every field is written; the variables keep their order, which is the
/// column order the search reads.
impl CanonicalEncode for DiscoverQuery {
    fn encode(&self, writer: &mut CanonicalWriter) {
        encode_variables(writer, &self.variables);
        self.method.encode(writer);
        self.bounds.encode(writer);
    }
}

impl DiscoverQuery {
    /// A search over `variables` with `method` under `bounds`.
    #[must_use]
    pub fn new(
        variables: impl Into<Arc<[VariableId]>>,
        method: DiscoveryMethod,
        bounds: SearchBounds,
    ) -> Self {
        Self { variables: variables.into(), method, bounds }
    }

    /// The inference mode the method declares.
    #[must_use]
    pub const fn inference(&self) -> DiscoveryInference {
        self.method.inference()
    }

    /// The structure class the search returns.
    #[must_use]
    pub const fn output(&self) -> DiscoveredClass {
        self.method.output()
    }

    /// Validate the variables, the method options and the bounds.
    ///
    /// # Errors
    ///
    /// [`QueryError::InvalidDiscover`] for no variables or a repeated one, a
    /// significance level outside `(0, 1)` on a testing method, a lag on a
    /// static method, a non-finite or negative method threshold, or a sampled
    /// posterior without a schedule (or with a zero chain, draw or thinning count).
    pub fn validate(&self) -> Result<(), QueryError> {
        let invalid = |message: &str| Err(QueryError::InvalidDiscover(message.to_owned()));
        if self.variables.is_empty() {
            return invalid("a structure search needs at least one variable");
        }
        let mut sorted = self.variables.to_vec();
        sorted.sort_unstable();
        if sorted.windows(2).any(|pair| pair[0] == pair[1]) {
            return invalid("a searched variable is listed twice");
        }
        if self.method.uses_independence_tests()
            && !(self.bounds.alpha.is_finite()
                && self.bounds.alpha > 0.0
                && self.bounds.alpha < 1.0)
        {
            return invalid("the significance level must be in (0, 1)");
        }
        if !self.method.is_temporal() && self.bounds.max_lag != 0 {
            return invalid("a static search takes no lag; set max_lag to 0");
        }
        match &self.method {
            DiscoveryMethod::DirectLingam { prune_threshold }
                if !(prune_threshold.is_finite() && *prune_threshold >= 0.0) =>
            {
                return invalid("the LiNGAM prune threshold must be finite and non-negative");
            }
            DiscoveryMethod::Notears { lambda, threshold, .. }
                if !(lambda.is_finite()
                    && *lambda >= 0.0
                    && threshold.is_finite()
                    && *threshold >= 0.0) =>
            {
                return invalid(
                    "the NOTEARS penalty and threshold must be finite and non-negative",
                );
            }
            DiscoveryMethod::Ges { max_subset: Some(0), .. } => {
                return invalid("the GES subset cap must be at least 1");
            }
            _ => {}
        }
        if self.method.samples_posterior() {
            match self.bounds.schedule {
                Some(schedule)
                    if schedule.chains >= 1 && schedule.draws >= 1 && schedule.thin >= 1 => {}
                Some(_) => {
                    return invalid("an MCMC schedule needs at least one chain, draw and thin");
                }
                None => return invalid("a sampled structure posterior needs an MCMC schedule"),
            }
        }
        Ok(())
    }

    /// Number of searched variables, as a canonical count.
    #[must_use]
    pub fn width(&self) -> u64 {
        count(self.variables.len())
    }
}
