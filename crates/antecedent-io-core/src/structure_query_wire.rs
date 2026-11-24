//! Wire forms of the structure-discovery, attribution and simulation queries.
//!
//! Embedded attribution and change queries reuse their own
//! [`CausalQueryWire`] forms, so one decoder validates each of them.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{
    AttributeQuery, AttributeTarget, CausalQuery, ChangeAttributionQuery, ChangeEstimator,
    ChangeMeasure, DiscoverQuery, DiscoveryMethod, InfluenceAllocation, Intervention,
    InterventionSequence, IntrinsicInfluenceTarget, MechanismChangeTest,
    NestedCounterfactualTarget, NoiseDeclaration, PopulationDoContrastTarget, ScreenWeight,
    SearchBounds, SearchSchedule, SimulateQuery, SimulationTarget, StructurePriorSpec, VariableId,
};
use serde::{Deserialize, Serialize};

use crate::convert::{vars_from_raw, vars_to_raw};
use crate::error::{IoError, fit_u64};
use crate::query_wire::{
    CausalQueryWire, InterventionWire, causal_query_from_wire, causal_query_to_wire,
};

fn fit_usize(value: u64) -> Result<usize, IoError> {
    usize::try_from(value).map_err(|_| IoError::TooLarge)
}

/// Wire form of [`StructurePriorSpec`]. Every field is absent from the document when unset.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct StructurePriorWire {
    /// Independent inclusion probability of every edge the map does not list.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub edge_inclusion: Option<f64>,
    /// `[from, to, probability]` of each listed directed edge (raw variable ids).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub edges: Vec<(u32, u32, f64)>,
    /// Inclusion probability of the edges not listed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_edge: Option<f64>,
    /// Soft tiers of raw variable ids, earliest first.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub soft_tiers: Vec<Vec<u32>>,
    /// Log-penalty of an edge against the soft tier order.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub soft_tier_log_penalty: Option<f64>,
}

impl StructurePriorWire {
    fn from_domain(prior: &StructurePriorSpec) -> Self {
        Self {
            edge_inclusion: prior.edge_inclusion,
            edges: prior.edges.iter().map(|(from, to, p)| (from.raw(), to.raw(), *p)).collect(),
            default_edge: prior.default_edge,
            soft_tiers: prior.soft_tiers.iter().map(|tier| vars_to_raw(tier)).collect(),
            soft_tier_log_penalty: prior.soft_tier_log_penalty,
        }
    }

    fn to_domain(&self) -> StructurePriorSpec {
        StructurePriorSpec {
            edge_inclusion: self.edge_inclusion,
            edges: self
                .edges
                .iter()
                .map(|&(from, to, p)| (VariableId::from_raw(from), VariableId::from_raw(to), p))
                .collect(),
            default_edge: self.default_edge,
            soft_tiers: self.soft_tiers.iter().map(|tier| vars_from_raw(tier)).collect(),
            soft_tier_log_penalty: self.soft_tier_log_penalty,
        }
    }
}

fn prior_to_wire(prior: Option<&StructurePriorSpec>) -> Option<StructurePriorWire> {
    prior.map(StructurePriorWire::from_domain)
}

fn prior_from_wire(prior: Option<&StructurePriorWire>) -> Option<StructurePriorSpec> {
    prior.map(StructurePriorWire::to_domain)
}

/// Wire form of [`DiscoveryMethod`].
///
/// A method that declares no structure prior keeps its original document form; the `*_with_prior`
/// forms and the optional `prior` member exist only when a prior is declared.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
#[serde(rename_all = "snake_case")]
pub enum DiscoveryMethodWire {
    /// PC.
    Pc,
    /// FCI.
    Fci,
    /// RFCI.
    Rfci,
    /// GES.
    Ges {
        /// Soft PC-skeleton screening.
        screen_pc: bool,
        /// Subset cap.
        max_subset: Option<u32>,
    },
    /// `DirectLiNGAM`.
    DirectLingam {
        /// Prune threshold.
        prune_threshold: f64,
    },
    /// NOTEARS.
    Notears {
        /// L1 penalty.
        lambda: f64,
        /// Edge threshold.
        threshold: f64,
        /// Standardize columns.
        standardize: bool,
    },
    /// PCMCI.
    Pcmci,
    /// PCMCI+.
    PcmciPlus,
    /// LPCMCI.
    Lpcmci,
    /// J-PCMCI+.
    JpcmciPlus,
    /// Regime PCMCI.
    Rpcmci {
        /// Shortest regime segment.
        min_regime_len: Option<u32>,
    },
    /// Exact DAG posterior.
    ExactDagPosterior,
    /// Exact DAG posterior under a per-edge prior.
    ExactDagPosteriorWithPrior {
        /// The declared structure prior.
        prior: StructurePriorWire,
    },
    /// Exact DAG posterior edge marginals.
    ExactDagMarginals,
    /// Exact DAG posterior edge marginals under a per-edge prior.
    ExactDagMarginalsWithPrior {
        /// The declared structure prior.
        prior: StructurePriorWire,
    },
    /// Order MCMC.
    OrderMcmc {
        /// Convergence-diagnostics gate.
        diagnostics_gate: bool,
        /// The declared structure prior, when there is one.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        prior: Option<StructurePriorWire>,
    },
    /// Structure MCMC.
    StructureMcmc,
    /// Structure MCMC under a per-edge prior.
    StructureMcmcWithPrior {
        /// The declared structure prior.
        prior: StructurePriorWire,
    },
    /// CI-screened posterior.
    CiScreenedPosterior {
        /// Candidate weighting (`none`, `bayes_factor`, `posterior_dependence`).
        weight: String,
        /// The declared structure prior, when there is one.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        prior: Option<StructurePriorWire>,
    },
    /// CI screen, then exact marginals of the focus neighbourhood.
    ScreenedExactMarginals {
        /// Variables of interest (raw ids).
        focus: Vec<u32>,
        /// Most variables the exact computation runs on.
        max_variables: u32,
        /// Prior inclusion probability of a screened-in pair.
        in_probability: f64,
        /// Prior inclusion probability of any other pair.
        out_probability: f64,
        /// The declared structure prior, when there is one.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        prior: Option<StructurePriorWire>,
    },
    /// DBN template posterior.
    DbnPosterior {
        /// Force MCMC.
        force_mcmc: bool,
        /// The declared structure prior, when there is one.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        prior: Option<StructurePriorWire>,
    },
    /// Exact posterior over PAG classes of Gaussian maximal ancestral graphs.
    ExactPagPosterior,
    /// Metropolis posterior over PAG classes of maximal ancestral graphs.
    PagMcmc,
    /// Posterior over Markov equivalence classes of a DAG posterior.
    ClassPosterior {
        /// The DAG-posterior method whose graphs are collapsed.
        base: Box<DiscoveryMethodWire>,
    },
    /// Sampling distribution of a frequentist search over resamples.
    BootstrapStability {
        /// The frequentist search that is re-run.
        base: Box<DiscoveryMethodWire>,
        /// Resamples drawn.
        replicates: u32,
    },
}

impl DiscoveryMethodWire {
    fn from_domain(method: &DiscoveryMethod) -> Self {
        match method {
            DiscoveryMethod::Pc => Self::Pc,
            DiscoveryMethod::Fci => Self::Fci,
            DiscoveryMethod::Rfci => Self::Rfci,
            DiscoveryMethod::Ges { screen_pc, max_subset } => {
                Self::Ges { screen_pc: *screen_pc, max_subset: *max_subset }
            }
            DiscoveryMethod::DirectLingam { prune_threshold } => {
                Self::DirectLingam { prune_threshold: *prune_threshold }
            }
            DiscoveryMethod::Notears { lambda, threshold, standardize } => {
                Self::Notears { lambda: *lambda, threshold: *threshold, standardize: *standardize }
            }
            DiscoveryMethod::Pcmci => Self::Pcmci,
            DiscoveryMethod::PcmciPlus => Self::PcmciPlus,
            DiscoveryMethod::Lpcmci => Self::Lpcmci,
            DiscoveryMethod::JpcmciPlus => Self::JpcmciPlus,
            DiscoveryMethod::Rpcmci { min_regime_len } => {
                Self::Rpcmci { min_regime_len: *min_regime_len }
            }
            DiscoveryMethod::ExactDagPosterior { prior } => match prior_to_wire(prior.as_ref()) {
                Some(prior) => Self::ExactDagPosteriorWithPrior { prior },
                None => Self::ExactDagPosterior,
            },
            DiscoveryMethod::ExactDagMarginals { prior } => match prior_to_wire(prior.as_ref()) {
                Some(prior) => Self::ExactDagMarginalsWithPrior { prior },
                None => Self::ExactDagMarginals,
            },
            DiscoveryMethod::OrderMcmc { diagnostics_gate, prior } => Self::OrderMcmc {
                diagnostics_gate: *diagnostics_gate,
                prior: prior_to_wire(prior.as_ref()),
            },
            DiscoveryMethod::StructureMcmc { prior } => match prior_to_wire(prior.as_ref()) {
                Some(prior) => Self::StructureMcmcWithPrior { prior },
                None => Self::StructureMcmc,
            },
            DiscoveryMethod::CiScreenedPosterior { weight, prior } => Self::CiScreenedPosterior {
                weight: weight.as_str().into(),
                prior: prior_to_wire(prior.as_ref()),
            },
            DiscoveryMethod::ScreenedExactMarginals {
                focus,
                max_variables,
                in_probability,
                out_probability,
                prior,
            } => Self::ScreenedExactMarginals {
                focus: vars_to_raw(focus),
                max_variables: *max_variables,
                in_probability: *in_probability,
                out_probability: *out_probability,
                prior: prior_to_wire(prior.as_ref()),
            },
            DiscoveryMethod::DbnPosterior { force_mcmc, prior } => {
                Self::DbnPosterior { force_mcmc: *force_mcmc, prior: prior_to_wire(prior.as_ref()) }
            }
            DiscoveryMethod::ExactPagPosterior => Self::ExactPagPosterior,
            DiscoveryMethod::PagMcmc => Self::PagMcmc,
            DiscoveryMethod::ClassPosterior { base } => {
                Self::ClassPosterior { base: Box::new(Self::from_domain(base)) }
            }
            DiscoveryMethod::BootstrapStability { base, replicates } => Self::BootstrapStability {
                base: Box::new(Self::from_domain(base)),
                replicates: *replicates,
            },
        }
    }

    fn to_domain(&self) -> Result<DiscoveryMethod, IoError> {
        Ok(match self {
            Self::Pc => DiscoveryMethod::Pc,
            Self::Fci => DiscoveryMethod::Fci,
            Self::Rfci => DiscoveryMethod::Rfci,
            Self::Ges { screen_pc, max_subset } => {
                DiscoveryMethod::Ges { screen_pc: *screen_pc, max_subset: *max_subset }
            }
            Self::DirectLingam { prune_threshold } => {
                DiscoveryMethod::DirectLingam { prune_threshold: *prune_threshold }
            }
            Self::Notears { lambda, threshold, standardize } => DiscoveryMethod::Notears {
                lambda: *lambda,
                threshold: *threshold,
                standardize: *standardize,
            },
            Self::Pcmci => DiscoveryMethod::Pcmci,
            Self::PcmciPlus => DiscoveryMethod::PcmciPlus,
            Self::Lpcmci => DiscoveryMethod::Lpcmci,
            Self::JpcmciPlus => DiscoveryMethod::JpcmciPlus,
            Self::Rpcmci { min_regime_len } => {
                DiscoveryMethod::Rpcmci { min_regime_len: *min_regime_len }
            }
            Self::ExactDagPosterior => DiscoveryMethod::ExactDagPosterior { prior: None },
            Self::ExactDagPosteriorWithPrior { prior } => {
                DiscoveryMethod::ExactDagPosterior { prior: Some(prior.to_domain()) }
            }
            Self::ExactDagMarginals => DiscoveryMethod::ExactDagMarginals { prior: None },
            Self::ExactDagMarginalsWithPrior { prior } => {
                DiscoveryMethod::ExactDagMarginals { prior: Some(prior.to_domain()) }
            }
            Self::OrderMcmc { diagnostics_gate, prior } => DiscoveryMethod::OrderMcmc {
                diagnostics_gate: *diagnostics_gate,
                prior: prior_from_wire(prior.as_ref()),
            },
            Self::StructureMcmc => DiscoveryMethod::StructureMcmc { prior: None },
            Self::StructureMcmcWithPrior { prior } => {
                DiscoveryMethod::StructureMcmc { prior: Some(prior.to_domain()) }
            }
            Self::CiScreenedPosterior { weight, prior } => DiscoveryMethod::CiScreenedPosterior {
                prior: prior_from_wire(prior.as_ref()),
                weight: match weight.as_str() {
                    "none" => ScreenWeight::None,
                    "bayes_factor" => ScreenWeight::BayesFactor,
                    "posterior_dependence" => ScreenWeight::PosteriorDependence,
                    other => {
                        return Err(IoError::Convert(format!(
                            "unknown CI-screened posterior weight `{other}`"
                        )));
                    }
                },
            },
            Self::ScreenedExactMarginals {
                focus,
                max_variables,
                in_probability,
                out_probability,
                prior,
            } => DiscoveryMethod::ScreenedExactMarginals {
                focus: vars_from_raw(focus),
                max_variables: *max_variables,
                in_probability: *in_probability,
                out_probability: *out_probability,
                prior: prior_from_wire(prior.as_ref()),
            },
            Self::DbnPosterior { force_mcmc, prior } => DiscoveryMethod::DbnPosterior {
                force_mcmc: *force_mcmc,
                prior: prior_from_wire(prior.as_ref()),
            },
            Self::ExactPagPosterior => DiscoveryMethod::ExactPagPosterior,
            Self::PagMcmc => DiscoveryMethod::PagMcmc,
            Self::ClassPosterior { base } => {
                DiscoveryMethod::ClassPosterior { base: Box::new(base.to_domain()?) }
            }
            Self::BootstrapStability { base, replicates } => DiscoveryMethod::BootstrapStability {
                base: Box::new(base.to_domain()?),
                replicates: *replicates,
            },
        })
    }
}

/// Wire form of [`DiscoverQuery`].
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct DiscoverQueryWire {
    /// Searched variables, in column order.
    pub variables: Vec<u32>,
    /// Search method.
    pub method: DiscoveryMethodWire,
    /// Significance level.
    pub alpha: f64,
    /// Largest conditioning set.
    pub max_cond_size: u32,
    /// Largest lag.
    pub max_lag: u32,
    /// MCMC schedule `[chains, warmup, draws, thin]`.
    pub schedule: Option<[u32; 4]>,
    /// Digest of the executable search configuration.
    pub configuration: [u8; 32],
}

impl DiscoverQueryWire {
    /// Encode.
    #[must_use]
    pub fn from_domain(query: &DiscoverQuery) -> Self {
        Self {
            variables: vars_to_raw(&query.variables),
            method: DiscoveryMethodWire::from_domain(&query.method),
            alpha: query.bounds.alpha,
            max_cond_size: query.bounds.max_cond_size,
            max_lag: query.bounds.max_lag,
            schedule: query.bounds.schedule.map(|s| [s.chains, s.warmup, s.draws, s.thin]),
            configuration: query.bounds.configuration,
        }
    }

    /// Decode and validate.
    ///
    /// # Errors
    ///
    /// An unknown method option, or a query that fails validation.
    pub fn to_domain(&self) -> Result<DiscoverQuery, IoError> {
        let query = DiscoverQuery {
            variables: vars_from_raw(&self.variables),
            method: self.method.to_domain()?,
            bounds: SearchBounds {
                alpha: self.alpha,
                max_cond_size: self.max_cond_size,
                max_lag: self.max_lag,
                schedule: self.schedule.map(|[chains, warmup, draws, thin]| SearchSchedule {
                    chains,
                    warmup,
                    draws,
                    thin,
                }),
                configuration: self.configuration,
            },
        };
        query.validate().map_err(|e| IoError::Convert(e.to_string()))?;
        Ok(query)
    }

    /// Variable ids the query names.
    #[must_use]
    pub fn variable_ids(&self) -> Vec<u32> {
        self.variables.clone()
    }
}

/// Wire form of [`ChangeEstimator`].
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
#[serde(rename_all = "snake_case")]
pub enum ChangeEstimatorWire {
    /// Sampled mechanism hybrids.
    Sampled {
        /// Measure (`mean_difference`, `variance_difference`, `gaussian_kl`).
        measure: String,
        /// Samples per coalition.
        n_samples: u64,
        /// Seed.
        seed: u64,
    },
    /// Robust regression hybrids.
    Robust {
        /// Row cap.
        max_rows: u64,
        /// Monte Carlo draws.
        n_samples: u64,
        /// Seed.
        seed: u64,
    },
}

impl ChangeEstimatorWire {
    fn from_domain(estimator: &ChangeEstimator) -> Result<Self, IoError> {
        Ok(match estimator {
            ChangeEstimator::Sampled { measure, n_samples, seed } => Self::Sampled {
                measure: measure_name(*measure).into(),
                n_samples: fit_u64(*n_samples)?,
                seed: *seed,
            },
            ChangeEstimator::Robust { max_rows, n_samples, seed } => Self::Robust {
                max_rows: fit_u64(*max_rows)?,
                n_samples: fit_u64(*n_samples)?,
                seed: *seed,
            },
        })
    }

    fn to_domain(&self) -> Result<ChangeEstimator, IoError> {
        Ok(match self {
            Self::Sampled { measure, n_samples, seed } => ChangeEstimator::Sampled {
                measure: measure_from_name(measure)?,
                n_samples: fit_usize(*n_samples)?,
                seed: *seed,
            },
            Self::Robust { max_rows, n_samples, seed } => ChangeEstimator::Robust {
                max_rows: fit_usize(*max_rows)?,
                n_samples: fit_usize(*n_samples)?,
                seed: *seed,
            },
        })
    }
}

/// Wire form of [`AttributeTarget`].
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
#[serde(rename_all = "snake_case")]
pub enum AttributeTargetWire {
    /// Intrinsic influence.
    IntrinsicInfluence {
        /// Attributed variable.
        target: u32,
        /// `auto`, `exact`, or `permutation`.
        allocation: String,
        /// Permutations of a permutation allocation.
        n_permutations: Option<u64>,
        /// Outer draws.
        outer_draws: u64,
        /// Inner draws.
        inner_draws: u64,
        /// Replicate batches.
        replicates: u64,
        /// Row-bootstrap refits.
        fit_bootstrap_replicates: u32,
        /// Interval level.
        level: f64,
        /// Source limit.
        max_sources: u64,
        /// Seed.
        seed: u64,
        /// `undeclared`, `independent`, or `structure`.
        noise: String,
        /// Bidirected pairs of a `structure` declaration.
        bidirected: Vec<(u32, u32)>,
    },
    /// Ranked root causes of a distribution change.
    RootCauses {
        /// The change, as its own query wire.
        change: Box<CausalQueryWire>,
        /// Coalition evaluation.
        estimator: ChangeEstimatorWire,
    },
    /// Distribution change.
    DistributionChange {
        /// The change, as its own query wire.
        change: Box<CausalQueryWire>,
        /// Coalition evaluation.
        estimator: ChangeEstimatorWire,
    },
    /// Path decomposition.
    Paths {
        /// Sources.
        sources: Vec<u32>,
        /// Outcome.
        outcome: u32,
        /// Path limit.
        max_paths: u64,
        /// Length limit.
        max_len: u64,
        /// Required intermediate nodes.
        through: Vec<u32>,
    },
    /// Feature relevance.
    FeatureRelevance {
        /// Outcome.
        outcome: u32,
        /// Features.
        features: Vec<u32>,
        /// Shift.
        delta: f64,
        /// Samples per coalition.
        n_samples: u64,
        /// Feature limit.
        max_features: u64,
    },
    /// Anomaly scores, as the anomaly query wire.
    Anomaly {
        /// The anomaly query.
        query: Box<CausalQueryWire>,
    },
    /// Unit change, as the unit-change query wire.
    UnitChange {
        /// The unit-change query.
        query: Box<CausalQueryWire>,
    },
    /// Mechanism-change detection, as the detection query wire and its test.
    MechanismChange {
        /// The detection query.
        query: Box<CausalQueryWire>,
        /// `likelihood_ratio`, `mean_difference`, `classifier_two_sample`,
        /// `kernel_two_sample`, or `change_point`.
        test: String,
    },
    /// Structure-change attribution.
    StructureChange {
        /// The change, as its own query wire.
        change: Box<CausalQueryWire>,
        /// Comparison structure's directed edges.
        comparison: Vec<(u32, u32)>,
        /// Measure (`mean_difference`, `variance_difference`, `gaussian_kl`).
        measure: String,
        /// Samples per coalition.
        n_samples: u64,
        /// Seed.
        seed: u64,
    },
    /// Population do-contrast of one parent on one child.
    PopulationDoContrast {
        /// Intervened variable.
        parent: u32,
        /// Variable read.
        child: u32,
        /// Width of the contrast.
        delta: f64,
        /// Most observed rows accepted; absent accepts every row.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        max_units: Option<u64>,
        /// Seed of the generator both arms draw from.
        seed: u64,
    },
}

/// Wire form of [`AttributeQuery`].
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct AttributeQueryWire {
    /// What is attributed.
    pub target: AttributeTargetWire,
}

fn change_wire(change: &ChangeAttributionQuery) -> Result<Box<CausalQueryWire>, IoError> {
    Ok(Box::new(causal_query_to_wire(&CausalQuery::ChangeAttribution(change.clone()))?))
}

fn change_from_wire(wire: &CausalQueryWire) -> Result<ChangeAttributionQuery, IoError> {
    match causal_query_from_wire(wire)? {
        CausalQuery::ChangeAttribution(change) => Ok(change),
        _ => {
            Err(IoError::Convert("an attribution change must be a change-attribution query".into()))
        }
    }
}

fn measure_name(measure: ChangeMeasure) -> &'static str {
    match measure {
        ChangeMeasure::MeanDifference => "mean_difference",
        ChangeMeasure::VarianceDifference => "variance_difference",
        ChangeMeasure::GaussianKl => "gaussian_kl",
    }
}

fn measure_from_name(name: &str) -> Result<ChangeMeasure, IoError> {
    match name {
        "mean_difference" => Ok(ChangeMeasure::MeanDifference),
        "variance_difference" => Ok(ChangeMeasure::VarianceDifference),
        "gaussian_kl" => Ok(ChangeMeasure::GaussianKl),
        other => Err(IoError::Convert(format!("unknown change measure `{other}`"))),
    }
}

fn mechanism_test_name(test: MechanismChangeTest) -> &'static str {
    match test {
        MechanismChangeTest::LikelihoodRatio => "likelihood_ratio",
        MechanismChangeTest::MeanDifference => "mean_difference",
        MechanismChangeTest::ClassifierTwoSample => "classifier_two_sample",
        MechanismChangeTest::KernelTwoSample => "kernel_two_sample",
        MechanismChangeTest::ChangePoint => "change_point",
    }
}

impl AttributeQueryWire {
    /// Encode.
    ///
    /// # Errors
    ///
    /// An embedded query that has no wire form, or a count that does not fit.
    pub fn from_domain(query: &AttributeQuery) -> Result<Self, IoError> {
        let target = match &query.target {
            AttributeTarget::IntrinsicInfluence(target) => {
                let (allocation, n_permutations) = match target.allocation {
                    InfluenceAllocation::Auto => ("auto", None),
                    InfluenceAllocation::Exact => ("exact", None),
                    InfluenceAllocation::Permutation { n_permutations } => {
                        ("permutation", Some(fit_u64(n_permutations)?))
                    }
                };
                let (noise, bidirected) = match &target.noise {
                    NoiseDeclaration::Undeclared => ("undeclared", Vec::new()),
                    NoiseDeclaration::Independent => ("independent", Vec::new()),
                    NoiseDeclaration::Structure { bidirected } => {
                        ("structure", bidirected.iter().map(|(a, b)| (a.raw(), b.raw())).collect())
                    }
                };
                AttributeTargetWire::IntrinsicInfluence {
                    target: target.target.raw(),
                    allocation: allocation.into(),
                    n_permutations,
                    outer_draws: fit_u64(target.outer_draws)?,
                    inner_draws: fit_u64(target.inner_draws)?,
                    replicates: fit_u64(target.replicates)?,
                    fit_bootstrap_replicates: target.fit_bootstrap_replicates,
                    level: target.level,
                    max_sources: fit_u64(target.max_sources)?,
                    seed: target.seed,
                    noise: noise.into(),
                    bidirected,
                }
            }
            AttributeTarget::RootCauses { change, estimator } => AttributeTargetWire::RootCauses {
                change: change_wire(change)?,
                estimator: ChangeEstimatorWire::from_domain(estimator)?,
            },
            AttributeTarget::DistributionChange { change, estimator } => {
                AttributeTargetWire::DistributionChange {
                    change: change_wire(change)?,
                    estimator: ChangeEstimatorWire::from_domain(estimator)?,
                }
            }
            AttributeTarget::Paths { sources, outcome, max_paths, max_len, through } => {
                AttributeTargetWire::Paths {
                    sources: vars_to_raw(sources),
                    outcome: outcome.raw(),
                    max_paths: fit_u64(*max_paths)?,
                    max_len: fit_u64(*max_len)?,
                    through: vars_to_raw(through),
                }
            }
            AttributeTarget::FeatureRelevance {
                outcome,
                features,
                delta,
                n_samples,
                max_features,
            } => AttributeTargetWire::FeatureRelevance {
                outcome: outcome.raw(),
                features: vars_to_raw(features),
                delta: *delta,
                n_samples: fit_u64(*n_samples)?,
                max_features: fit_u64(*max_features)?,
            },
            AttributeTarget::Anomaly(query) => AttributeTargetWire::Anomaly {
                query: Box::new(causal_query_to_wire(&CausalQuery::AnomalyAttribution(
                    query.clone(),
                ))?),
            },
            AttributeTarget::UnitChange(query) => AttributeTargetWire::UnitChange {
                query: Box::new(causal_query_to_wire(&CausalQuery::UnitChange(query.clone()))?),
            },
            AttributeTarget::MechanismChange { query, test } => {
                AttributeTargetWire::MechanismChange {
                    query: Box::new(causal_query_to_wire(&CausalQuery::MechanismChange(
                        query.clone(),
                    ))?),
                    test: mechanism_test_name(*test).into(),
                }
            }
            AttributeTarget::StructureChange { change, comparison, measure, n_samples, seed } => {
                AttributeTargetWire::StructureChange {
                    change: change_wire(change)?,
                    comparison: comparison.iter().map(|(a, b)| (a.raw(), b.raw())).collect(),
                    measure: measure_name(*measure).into(),
                    n_samples: fit_u64(*n_samples)?,
                    seed: *seed,
                }
            }
            AttributeTarget::PopulationDoContrast(target) => {
                AttributeTargetWire::PopulationDoContrast {
                    parent: target.parent.raw(),
                    child: target.child.raw(),
                    delta: target.delta,
                    max_units: target.max_units.map(fit_u64).transpose()?,
                    seed: target.seed,
                }
            }
        };
        Ok(Self { target })
    }

    /// Decode and validate.
    ///
    /// # Errors
    ///
    /// An unknown option name, an embedded query of the wrong kind, a count
    /// that does not fit, or a query that fails validation.
    #[allow(
        clippy::too_many_lines,
        reason = "one arm per attribution target, each decoding that target's fields"
    )]
    pub fn to_domain(&self) -> Result<AttributeQuery, IoError> {
        let target = match &self.target {
            AttributeTargetWire::IntrinsicInfluence {
                target,
                allocation,
                n_permutations,
                outer_draws,
                inner_draws,
                replicates,
                fit_bootstrap_replicates,
                level,
                max_sources,
                seed,
                noise,
                bidirected,
            } => AttributeTarget::IntrinsicInfluence(IntrinsicInfluenceTarget {
                target: VariableId::from_raw(*target),
                allocation: match (allocation.as_str(), n_permutations) {
                    ("auto", None) => InfluenceAllocation::Auto,
                    ("exact", None) => InfluenceAllocation::Exact,
                    ("permutation", Some(n)) => {
                        InfluenceAllocation::Permutation { n_permutations: fit_usize(*n)? }
                    }
                    _ => {
                        return Err(IoError::Convert(format!(
                            "unknown intrinsic-influence allocation `{allocation}`"
                        )));
                    }
                },
                outer_draws: fit_usize(*outer_draws)?,
                inner_draws: fit_usize(*inner_draws)?,
                replicates: fit_usize(*replicates)?,
                fit_bootstrap_replicates: *fit_bootstrap_replicates,
                level: *level,
                max_sources: fit_usize(*max_sources)?,
                seed: *seed,
                noise: match noise.as_str() {
                    "undeclared" => NoiseDeclaration::Undeclared,
                    "independent" => NoiseDeclaration::Independent,
                    "structure" => NoiseDeclaration::Structure {
                        bidirected: bidirected
                            .iter()
                            .map(|(a, b)| (VariableId::from_raw(*a), VariableId::from_raw(*b)))
                            .collect::<Vec<_>>()
                            .into(),
                    },
                    other => {
                        return Err(IoError::Convert(format!(
                            "unknown noise declaration `{other}`"
                        )));
                    }
                },
            }),
            AttributeTargetWire::RootCauses { change, estimator } => AttributeTarget::RootCauses {
                change: change_from_wire(change)?,
                estimator: estimator.to_domain()?,
            },
            AttributeTargetWire::DistributionChange { change, estimator } => {
                AttributeTarget::DistributionChange {
                    change: change_from_wire(change)?,
                    estimator: estimator.to_domain()?,
                }
            }
            AttributeTargetWire::Paths { sources, outcome, max_paths, max_len, through } => {
                AttributeTarget::Paths {
                    sources: vars_from_raw(sources),
                    outcome: VariableId::from_raw(*outcome),
                    max_paths: fit_usize(*max_paths)?,
                    max_len: fit_usize(*max_len)?,
                    through: vars_from_raw(through),
                }
            }
            AttributeTargetWire::FeatureRelevance {
                outcome,
                features,
                delta,
                n_samples,
                max_features,
            } => AttributeTarget::FeatureRelevance {
                outcome: VariableId::from_raw(*outcome),
                features: vars_from_raw(features),
                delta: *delta,
                n_samples: fit_usize(*n_samples)?,
                max_features: fit_usize(*max_features)?,
            },
            AttributeTargetWire::Anomaly { query } => match causal_query_from_wire(query)? {
                CausalQuery::AnomalyAttribution(query) => AttributeTarget::Anomaly(query),
                _ => {
                    return Err(IoError::Convert(
                        "an anomaly target must be an anomaly-attribution query".into(),
                    ));
                }
            },
            AttributeTargetWire::UnitChange { query } => match causal_query_from_wire(query)? {
                CausalQuery::UnitChange(query) => AttributeTarget::UnitChange(query),
                _ => {
                    return Err(IoError::Convert(
                        "a unit-change target must be a unit-change query".into(),
                    ));
                }
            },
            AttributeTargetWire::MechanismChange { query, test } => {
                let CausalQuery::MechanismChange(query) = causal_query_from_wire(query)? else {
                    return Err(IoError::Convert(
                        "a mechanism-change target must be a mechanism-change query".into(),
                    ));
                };
                let test = match test.as_str() {
                    "likelihood_ratio" => MechanismChangeTest::LikelihoodRatio,
                    "mean_difference" => MechanismChangeTest::MeanDifference,
                    "classifier_two_sample" => MechanismChangeTest::ClassifierTwoSample,
                    "kernel_two_sample" => MechanismChangeTest::KernelTwoSample,
                    "change_point" => MechanismChangeTest::ChangePoint,
                    other => {
                        return Err(IoError::Convert(format!(
                            "unknown mechanism-change test `{other}`"
                        )));
                    }
                };
                AttributeTarget::MechanismChange { query, test }
            }
            AttributeTargetWire::StructureChange {
                change,
                comparison,
                measure,
                n_samples,
                seed,
            } => AttributeTarget::StructureChange {
                change: change_from_wire(change)?,
                comparison: comparison
                    .iter()
                    .map(|(a, b)| (VariableId::from_raw(*a), VariableId::from_raw(*b)))
                    .collect::<Vec<_>>()
                    .into(),
                measure: measure_from_name(measure)?,
                n_samples: fit_usize(*n_samples)?,
                seed: *seed,
            },
            AttributeTargetWire::PopulationDoContrast { parent, child, delta, max_units, seed } => {
                AttributeTarget::PopulationDoContrast(PopulationDoContrastTarget {
                    parent: VariableId::from_raw(*parent),
                    child: VariableId::from_raw(*child),
                    delta: *delta,
                    max_units: max_units.map(fit_usize).transpose()?,
                    seed: *seed,
                })
            }
        };
        let query = AttributeQuery { target };
        query.validate().map_err(|e| IoError::Convert(e.to_string()))?;
        Ok(query)
    }

    /// Variable ids the query names, embedded queries included.
    #[must_use]
    pub fn variable_ids(&self) -> Vec<u32> {
        match &self.target {
            AttributeTargetWire::IntrinsicInfluence { target, bidirected, .. } => {
                std::iter::once(*target)
                    .chain(bidirected.iter().flat_map(|(a, b)| [*a, *b]))
                    .collect()
            }
            AttributeTargetWire::Paths { sources, outcome, through, .. } => {
                sources.iter().chain(through).copied().chain(std::iter::once(*outcome)).collect()
            }
            AttributeTargetWire::FeatureRelevance { outcome, features, .. } => {
                features.iter().copied().chain(std::iter::once(*outcome)).collect()
            }
            AttributeTargetWire::StructureChange { comparison, .. } => {
                comparison.iter().flat_map(|(a, b)| [*a, *b]).collect()
            }
            AttributeTargetWire::PopulationDoContrast { parent, child, .. } => {
                vec![*parent, *child]
            }
            AttributeTargetWire::RootCauses { .. }
            | AttributeTargetWire::DistributionChange { .. }
            | AttributeTargetWire::Anomaly { .. }
            | AttributeTargetWire::UnitChange { .. }
            | AttributeTargetWire::MechanismChange { .. } => Vec::new(),
        }
    }

    /// Embedded query wires, for the validators that walk them.
    #[must_use]
    pub fn embedded(&self) -> Option<&CausalQueryWire> {
        match &self.target {
            AttributeTargetWire::RootCauses { change, .. }
            | AttributeTargetWire::DistributionChange { change, .. }
            | AttributeTargetWire::StructureChange { change, .. } => Some(&**change),
            AttributeTargetWire::Anomaly { query }
            | AttributeTargetWire::UnitChange { query }
            | AttributeTargetWire::MechanismChange { query, .. } => Some(&**query),
            AttributeTargetWire::IntrinsicInfluence { .. }
            | AttributeTargetWire::Paths { .. }
            | AttributeTargetWire::FeatureRelevance { .. }
            | AttributeTargetWire::PopulationDoContrast { .. } => None,
        }
    }
}

/// Wire form of [`SimulationTarget`].
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
#[serde(rename_all = "snake_case")]
pub enum SimulationTargetWire {
    /// Interventional sample.
    InterventionalSample {
        /// Interventions.
        interventions: Vec<InterventionWire>,
        /// Rows drawn.
        draws: u64,
        /// Seed.
        seed: u64,
    },
    /// Policy value.
    PolicyValue {
        /// Interventions.
        interventions: Vec<InterventionWire>,
        /// Outcome.
        outcome: u32,
        /// Rows drawn.
        draws: u64,
    },
    /// Unit effects.
    UnitEffects {
        /// Treatment.
        treatment: u32,
        /// Outcome.
        outcome: u32,
        /// Active level.
        active: f64,
        /// Control level.
        control: f64,
    },
    /// Draws with their exogenous noise.
    NoiseSample {
        /// Interventions.
        interventions: Vec<InterventionWire>,
        /// Rows drawn.
        draws: u64,
    },
    /// Replay of retained noise.
    NoiseReplay {
        /// Interventions.
        interventions: Vec<InterventionWire>,
        /// Noise, `n_rows` values per variable.
        noise: Vec<f64>,
        /// Rows replayed.
        n_rows: u64,
        /// Stochastic-policy stream key.
        policy_salt: u64,
    },
    /// Counterfactual values of the factual rows.
    Counterfactual {
        /// Interventions.
        interventions: Vec<InterventionWire>,
        /// Outcomes.
        outcomes: Vec<u32>,
    },
    /// Temporal rollout.
    TemporalRollout {
        /// The schedule, as a sequence intervention.
        schedule: InterventionWire,
        /// Rows per step.
        n_rows: u64,
        /// Steps.
        n_steps: u64,
        /// Lag window.
        window: u32,
        /// History before time zero.
        history: Vec<f64>,
        /// Seed.
        seed: u64,
    },
    /// Temporal counterfactual over a factual panel.
    TemporalCounterfactual {
        /// The schedule, as a sequence intervention.
        schedule: InterventionWire,
        /// Units per step.
        n_rows: u64,
        /// Steps.
        n_steps: u64,
        /// Lag window.
        window: u32,
        /// Factual panel.
        panel: Vec<f64>,
        /// History before the panel.
        history: Vec<f64>,
    },
    /// A simulation whose program holds an in-process callback. It runs in the process
    /// that holds the callback; the wire records the target and the canonical digest of
    /// the whole query, which commits to every callback's declaration (name, inputs,
    /// output, pinned dependencies, implementation kind), and refuses to decode: no
    /// durable encoding of the callback exists.
    InProcess {
        /// Target kind (`SimulationTarget::as_str`).
        target: String,
        /// Lower-case hex canonical digest of the domain query.
        digest: String,
        /// Names of the in-process callbacks.
        callbacks: Vec<String>,
    },
    /// Nested counterfactual mean over the study's rows.
    NestedCounterfactual {
        /// Outer-world interventions.
        outer: Vec<InterventionWire>,
        /// Inner-world interventions.
        inner: Vec<InterventionWire>,
        /// Variables held at their outer-world values.
        frozen: Vec<u32>,
        /// Outcome.
        outcome: u32,
    },
}

/// Portability of every intervention a simulation target declares.
fn target_portability(target: &SimulationTarget) -> antecedent_core::Portability {
    let fold = |interventions: &[Intervention]| {
        interventions
            .iter()
            .fold(antecedent_core::Portability::Portable, |acc, iv| acc.and(iv.portability()))
    };
    match target {
        SimulationTarget::InterventionalSample { interventions, .. }
        | SimulationTarget::PolicyValue { interventions, .. }
        | SimulationTarget::NoiseSample { interventions, .. }
        | SimulationTarget::NoiseReplay { interventions, .. }
        | SimulationTarget::Counterfactual { interventions, .. } => fold(interventions),
        SimulationTarget::NestedCounterfactual(target) => {
            fold(&target.outer).and(fold(&target.inner))
        }
        SimulationTarget::TemporalRollout { schedule, .. }
        | SimulationTarget::TemporalCounterfactual { schedule, .. } => {
            Intervention::sequence(schedule.clone()).portability()
        }
        SimulationTarget::UnitEffects { .. } => antecedent_core::Portability::Portable,
    }
}

fn schedule_to_wire(schedule: &InterventionSequence) -> Result<InterventionWire, IoError> {
    InterventionWire::from_domain_recoverable(&Intervention::sequence(schedule.clone()))
}

fn schedule_from_wire(schedule: &InterventionWire) -> Result<InterventionSequence, IoError> {
    match schedule.to_domain() {
        Intervention::Sequence(sequence) => Ok(sequence),
        other => Err(IoError::Convert(format!("a schedule must be a sequence, got {other:?}"))),
    }
}

/// Wire form of [`SimulateQuery`].
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct SimulateQueryWire {
    /// What is simulated.
    pub target: SimulationTargetWire,
}

/// A recoverable callback is written as its verified recovery record; a closed-form
/// intervention as itself.
fn interventions_to_wire(interventions: &[Intervention]) -> Result<Vec<InterventionWire>, IoError> {
    interventions.iter().map(InterventionWire::from_domain_recoverable).collect()
}

fn interventions_from_wire(interventions: &[InterventionWire]) -> Arc<[Intervention]> {
    interventions.iter().map(InterventionWire::to_domain).collect::<Vec<_>>().into()
}

/// The refusal to decode a simulation that ran on in-process callbacks: only the callbacks'
/// declaration digest was recorded, so the query cannot be rebuilt.
fn in_process_refusal(target: &str, callbacks: &[String]) -> IoError {
    let refusal = antecedent_core::Portability::RuntimeBound { callbacks: callbacks.to_vec() }
        .export_refusal()
        .map_or_else(
            || ("policy_callback_not_portable", String::new()),
            |refusal| (refusal.code, refusal.message),
        );
    IoError::Refused {
        code: refusal.0,
        message: format!(
            "{}; the {target} simulation ran in the process that held the callback and its \
             record keeps only the callbacks' declaration digest",
            refusal.1
        ),
    }
}

impl SimulateQueryWire {
    /// Encode.
    ///
    /// # Errors
    ///
    /// An intervention holding a runtime-bound callback, or a count that does not fit.
    pub fn from_domain(query: &SimulateQuery) -> Result<Self, IoError> {
        if let antecedent_core::Portability::RuntimeBound { callbacks } =
            target_portability(&query.target)
        {
            use antecedent_core::CanonicalEncode;
            let mut writer =
                antecedent_core::CanonicalWriter::new("antecedent.simulate_query.in_process");
            query.encode(&mut writer);
            return Ok(Self {
                target: SimulationTargetWire::InProcess {
                    target: query.target.as_str().to_owned(),
                    digest: antecedent_core::to_hex(&writer.finish()),
                    callbacks,
                },
            });
        }
        let target = match &query.target {
            SimulationTarget::InterventionalSample { interventions, draws, seed } => {
                SimulationTargetWire::InterventionalSample {
                    interventions: interventions_to_wire(interventions)?,
                    draws: fit_u64(*draws)?,
                    seed: *seed,
                }
            }
            SimulationTarget::PolicyValue { interventions, outcome, draws } => {
                SimulationTargetWire::PolicyValue {
                    interventions: interventions_to_wire(interventions)?,
                    outcome: outcome.raw(),
                    draws: fit_u64(*draws)?,
                }
            }
            SimulationTarget::UnitEffects { treatment, outcome, active, control } => {
                SimulationTargetWire::UnitEffects {
                    treatment: treatment.raw(),
                    outcome: outcome.raw(),
                    active: *active,
                    control: *control,
                }
            }
            SimulationTarget::NoiseSample { interventions, draws } => {
                SimulationTargetWire::NoiseSample {
                    interventions: interventions_to_wire(interventions)?,
                    draws: fit_u64(*draws)?,
                }
            }
            SimulationTarget::NoiseReplay { interventions, noise, n_rows, policy_salt } => {
                SimulationTargetWire::NoiseReplay {
                    interventions: interventions_to_wire(interventions)?,
                    noise: noise.to_vec(),
                    n_rows: fit_u64(*n_rows)?,
                    policy_salt: *policy_salt,
                }
            }
            SimulationTarget::Counterfactual { interventions, outcomes } => {
                SimulationTargetWire::Counterfactual {
                    interventions: interventions_to_wire(interventions)?,
                    outcomes: vars_to_raw(outcomes),
                }
            }
            SimulationTarget::TemporalRollout {
                schedule,
                n_rows,
                n_steps,
                window,
                history,
                seed,
            } => SimulationTargetWire::TemporalRollout {
                schedule: schedule_to_wire(schedule)?,
                n_rows: fit_u64(*n_rows)?,
                n_steps: fit_u64(*n_steps)?,
                window: *window,
                history: history.to_vec(),
                seed: *seed,
            },
            SimulationTarget::TemporalCounterfactual {
                schedule,
                n_rows,
                n_steps,
                window,
                panel,
                history,
            } => SimulationTargetWire::TemporalCounterfactual {
                schedule: schedule_to_wire(schedule)?,
                n_rows: fit_u64(*n_rows)?,
                n_steps: fit_u64(*n_steps)?,
                window: *window,
                panel: panel.to_vec(),
                history: history.to_vec(),
            },
            SimulationTarget::NestedCounterfactual(target) => {
                SimulationTargetWire::NestedCounterfactual {
                    outer: interventions_to_wire(&target.outer)?,
                    inner: interventions_to_wire(&target.inner)?,
                    frozen: vars_to_raw(&target.frozen),
                    outcome: target.outcome.raw(),
                }
            }
        };
        Ok(Self { target })
    }

    /// Decode and validate.
    ///
    /// # Errors
    ///
    /// A count that does not fit, or a query that fails validation.
    pub fn to_domain(&self) -> Result<SimulateQuery, IoError> {
        let target = match &self.target {
            SimulationTargetWire::InterventionalSample { interventions, draws, seed } => {
                SimulationTarget::InterventionalSample {
                    interventions: interventions_from_wire(interventions),
                    draws: fit_usize(*draws)?,
                    seed: *seed,
                }
            }
            SimulationTargetWire::PolicyValue { interventions, outcome, draws } => {
                SimulationTarget::PolicyValue {
                    interventions: interventions_from_wire(interventions),
                    outcome: VariableId::from_raw(*outcome),
                    draws: fit_usize(*draws)?,
                }
            }
            SimulationTargetWire::UnitEffects { treatment, outcome, active, control } => {
                SimulationTarget::UnitEffects {
                    treatment: VariableId::from_raw(*treatment),
                    outcome: VariableId::from_raw(*outcome),
                    active: *active,
                    control: *control,
                }
            }
            SimulationTargetWire::NoiseSample { interventions, draws } => {
                SimulationTarget::NoiseSample {
                    interventions: interventions_from_wire(interventions),
                    draws: fit_usize(*draws)?,
                }
            }
            SimulationTargetWire::NoiseReplay { interventions, noise, n_rows, policy_salt } => {
                SimulationTarget::NoiseReplay {
                    interventions: interventions_from_wire(interventions),
                    noise: Arc::from(noise.as_slice()),
                    n_rows: fit_usize(*n_rows)?,
                    policy_salt: *policy_salt,
                }
            }
            SimulationTargetWire::Counterfactual { interventions, outcomes } => {
                SimulationTarget::Counterfactual {
                    interventions: interventions_from_wire(interventions),
                    outcomes: vars_from_raw(outcomes),
                }
            }
            SimulationTargetWire::TemporalRollout {
                schedule,
                n_rows,
                n_steps,
                window,
                history,
                seed,
            } => SimulationTarget::TemporalRollout {
                schedule: schedule_from_wire(schedule)?,
                n_rows: fit_usize(*n_rows)?,
                n_steps: fit_usize(*n_steps)?,
                window: *window,
                history: Arc::from(history.as_slice()),
                seed: *seed,
            },
            SimulationTargetWire::TemporalCounterfactual {
                schedule,
                n_rows,
                n_steps,
                window,
                panel,
                history,
            } => SimulationTarget::TemporalCounterfactual {
                schedule: schedule_from_wire(schedule)?,
                n_rows: fit_usize(*n_rows)?,
                n_steps: fit_usize(*n_steps)?,
                window: *window,
                panel: Arc::from(panel.as_slice()),
                history: Arc::from(history.as_slice()),
            },
            SimulationTargetWire::NestedCounterfactual { outer, inner, frozen, outcome } => {
                SimulationTarget::NestedCounterfactual(NestedCounterfactualTarget {
                    outer: interventions_from_wire(outer),
                    inner: interventions_from_wire(inner),
                    frozen: vars_from_raw(frozen),
                    outcome: VariableId::from_raw(*outcome),
                })
            }
            SimulationTargetWire::InProcess { target, callbacks, .. } => {
                return Err(in_process_refusal(target, callbacks));
            }
        };
        let query = SimulateQuery { target };
        query.validate().map_err(|e| IoError::Convert(e.to_string()))?;
        Ok(query)
    }

    /// Plain variable ids the query names (intervention reads are walked by the caller).
    #[must_use]
    pub fn variable_ids(&self) -> Vec<u32> {
        match &self.target {
            SimulationTargetWire::InterventionalSample { .. }
            | SimulationTargetWire::NoiseSample { .. }
            | SimulationTargetWire::NoiseReplay { .. }
            | SimulationTargetWire::TemporalRollout { .. }
            | SimulationTargetWire::TemporalCounterfactual { .. }
            | SimulationTargetWire::InProcess { .. } => Vec::new(),
            SimulationTargetWire::Counterfactual { outcomes, .. } => outcomes.clone(),
            SimulationTargetWire::PolicyValue { outcome, .. } => vec![*outcome],
            SimulationTargetWire::UnitEffects { treatment, outcome, .. } => {
                vec![*treatment, *outcome]
            }
            SimulationTargetWire::NestedCounterfactual { frozen, outcome, .. } => {
                frozen.iter().copied().chain(std::iter::once(*outcome)).collect()
            }
        }
    }

    /// Every intervention the query declares, in both worlds of a nested counterfactual.
    #[must_use]
    pub fn all_interventions(&self) -> Vec<&InterventionWire> {
        match &self.target {
            SimulationTargetWire::NestedCounterfactual { outer, inner, .. } => {
                outer.iter().chain(inner).collect()
            }
            _ => self.interventions().iter().collect(),
        }
    }

    /// Interventions the query declares: of a nested counterfactual, the outer world's
    /// ([`Self::all_interventions`] also walks the inner world's).
    #[must_use]
    pub fn interventions(&self) -> &[InterventionWire] {
        match &self.target {
            SimulationTargetWire::InterventionalSample { interventions, .. }
            | SimulationTargetWire::PolicyValue { interventions, .. }
            | SimulationTargetWire::NoiseSample { interventions, .. }
            | SimulationTargetWire::NoiseReplay { interventions, .. }
            | SimulationTargetWire::Counterfactual { interventions, .. } => {
                interventions.as_slice()
            }
            SimulationTargetWire::NestedCounterfactual { outer, .. } => outer.as_slice(),
            SimulationTargetWire::TemporalRollout { schedule, .. }
            | SimulationTargetWire::TemporalCounterfactual { schedule, .. } => {
                std::slice::from_ref(schedule)
            }
            SimulationTargetWire::UnitEffects { .. } | SimulationTargetWire::InProcess { .. } => {
                &[]
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use antecedent_core::{AnomalyAttributionQuery, PopulationSelector, UnitChangeQuery, Value};

    fn v(raw: u32) -> VariableId {
        VariableId::from_raw(raw)
    }

    fn round_trip(query: &CausalQuery) {
        let wire = causal_query_to_wire(query).expect("encode");
        let json = serde_json::to_string(&wire).expect("serialize");
        let back: CausalQueryWire = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(&causal_query_from_wire(&back).expect("decode"), query);
    }

    #[derive(Debug)]
    struct Identity(antecedent_core::CallbackSpec);

    impl antecedent_core::BatchCallback for Identity {
        fn spec(&self) -> &antecedent_core::CallbackSpec {
            &self.0
        }

        fn call(
            &self,
            inputs: &[&[f64]],
            out: &mut [f64],
        ) -> Result<(), antecedent_core::CallbackError> {
            out.copy_from_slice(inputs[0]);
            Ok(())
        }
    }

    fn in_process_sample(name: &str) -> CausalQuery {
        let spec = antecedent_core::CallbackSpec {
            name: name.into(),
            inputs: Arc::from(["x".to_string()]),
            output: antecedent_core::CallbackOutput::Real,
            dependencies: Arc::from([]),
            implementation: antecedent_core::CallbackImplementation::Runtime,
        };
        let callback: Arc<dyn antecedent_core::BatchCallback> = Arc::new(Identity(spec));
        let expr =
            antecedent_core::ValueExpr::callback(callback, [antecedent_core::ValueExpr::var(v(0))]);
        CausalQuery::Simulate(SimulateQuery::new(SimulationTarget::InterventionalSample {
            interventions: Arc::from([Intervention::assign(v(1), expr)]),
            draws: 4,
            seed: 1,
        }))
    }

    #[test]
    fn an_in_process_callback_is_recorded_by_digest_and_refuses_to_decode() {
        let wire = causal_query_to_wire(&in_process_sample("f")).expect("local execution encodes");
        let CausalQueryWire::Simulate(simulate) = &wire else {
            panic!("a simulate query encodes as a simulate wire");
        };
        let SimulationTargetWire::InProcess { target, digest, callbacks } = &simulate.target else {
            panic!("an in-process callback is recorded by digest");
        };
        assert_eq!(target, "interventional_sample");
        assert_eq!(digest.len(), 64);
        assert_eq!(callbacks, &vec!["f".to_string()]);
        let other = causal_query_to_wire(&in_process_sample("g")).unwrap();
        assert_ne!(wire, other, "the digest commits to the callback's declaration");
        let Err(IoError::Refused { code, .. }) = causal_query_from_wire(&wire) else {
            panic!("export or replay of an in-process program is refused");
        };
        assert_eq!(code, "policy_callback_not_portable");
    }

    #[test]
    #[allow(clippy::too_many_lines, reason = "one decoder arm per discovery method")]
    fn the_structure_and_model_queries_round_trip() {
        round_trip(&CausalQuery::Discover(DiscoverQuery::new(
            vec![v(0), v(1), v(2)],
            DiscoveryMethod::CiScreenedPosterior { weight: ScreenWeight::BayesFactor, prior: None },
            SearchBounds {
                alpha: 0.05,
                max_cond_size: 2,
                max_lag: 0,
                schedule: Some(SearchSchedule { chains: 2, warmup: 5, draws: 10, thin: 1 }),
                configuration: [3; 32],
            },
        )));
        round_trip(&CausalQuery::Discover(DiscoverQuery::new(
            vec![v(0), v(1), v(2)],
            DiscoveryMethod::BootstrapStability {
                base: Box::new(DiscoveryMethod::Ges { screen_pc: false, max_subset: None }),
                replicates: 40,
            },
            SearchBounds {
                alpha: 0.05,
                max_cond_size: 2,
                max_lag: 0,
                schedule: None,
                configuration: [4; 32],
            },
        )));
        round_trip(&CausalQuery::Discover(DiscoverQuery::new(
            vec![v(0), v(1), v(2)],
            DiscoveryMethod::ClassPosterior {
                base: Box::new(DiscoveryMethod::StructureMcmc { prior: None }),
            },
            SearchBounds {
                alpha: 0.0,
                max_cond_size: 0,
                max_lag: 0,
                schedule: Some(SearchSchedule { chains: 2, warmup: 5, draws: 10, thin: 1 }),
                configuration: [5; 32],
            },
        )));
        let prior = StructurePriorSpec {
            edge_inclusion: Some(0.25),
            edges: Arc::from([(v(0), v(1), 0.1), (v(1), v(2), 0.7)]),
            default_edge: Some(0.3),
            soft_tiers: Arc::from([Arc::from([v(0)]), Arc::from([v(1), v(2)])]),
            soft_tier_log_penalty: Some(2.0),
        };
        for method in [
            DiscoveryMethod::ExactDagPosterior { prior: Some(prior.clone()) },
            DiscoveryMethod::ExactDagMarginals { prior: Some(prior.clone()) },
            DiscoveryMethod::OrderMcmc { diagnostics_gate: true, prior: Some(prior.clone()) },
            DiscoveryMethod::StructureMcmc { prior: Some(prior.clone()) },
            DiscoveryMethod::CiScreenedPosterior {
                weight: ScreenWeight::None,
                prior: Some(prior.clone()),
            },
            DiscoveryMethod::DbnPosterior { force_mcmc: false, prior: Some(prior.clone()) },
            DiscoveryMethod::ClassPosterior {
                base: Box::new(DiscoveryMethod::StructureMcmc { prior: Some(prior.clone()) }),
            },
        ] {
            // A screen runs independence tests, so it needs a significance level in (0, 1).
            let alpha = if matches!(method, DiscoveryMethod::CiScreenedPosterior { .. }) {
                0.05
            } else {
                0.0
            };
            round_trip(&CausalQuery::Discover(DiscoverQuery::new(
                vec![v(0), v(1), v(2)],
                method,
                SearchBounds {
                    alpha,
                    max_cond_size: 0,
                    max_lag: 0,
                    schedule: Some(SearchSchedule { chains: 2, warmup: 5, draws: 10, thin: 1 }),
                    configuration: [6; 32],
                },
            )));
        }
        // A method without a prior keeps its original document form.
        let plain = DiscoveryMethodWire::from_domain(&DiscoveryMethod::OrderMcmc {
            diagnostics_gate: true,
            prior: None,
        });
        assert_eq!(
            serde_json::to_string(&plain).expect("serialize"),
            r#"{"order_mcmc":{"diagnostics_gate":true}}"#
        );
        assert_eq!(
            serde_json::to_string(&DiscoveryMethodWire::from_domain(
                &DiscoveryMethod::ExactDagPosterior { prior: None }
            ))
            .expect("serialize"),
            r#""exact_dag_posterior""#
        );
        round_trip(&CausalQuery::Attribute(AttributeQuery::new(AttributeTarget::Anomaly(
            AnomalyAttributionQuery::new(vec![v(1)], 10),
        ))));
        round_trip(&CausalQuery::Attribute(AttributeQuery::new(AttributeTarget::UnitChange(
            UnitChangeQuery::new(v(1), 10),
        ))));
        round_trip(&CausalQuery::Attribute(AttributeQuery::new(AttributeTarget::RootCauses {
            change: ChangeAttributionQuery::new(
                v(1),
                PopulationSelector::TimeRange { start: 0, end: 10 },
                PopulationSelector::TimeRange { start: 10, end: 20 },
            ),
            estimator: ChangeEstimator::Robust { max_rows: 100, n_samples: 50, seed: 1 },
        })));
        round_trip(&CausalQuery::Simulate(SimulateQuery::new(
            SimulationTarget::InterventionalSample {
                interventions: Arc::from([Intervention::set(v(0), Value::f64(1.0))]),
                draws: 10,
                seed: 2,
            },
        )));
        round_trip(&CausalQuery::Simulate(SimulateQuery::new(SimulationTarget::NoiseReplay {
            interventions: Arc::from([Intervention::set(v(0), Value::f64(1.0))]),
            noise: Arc::from([0.1, -0.2, 0.3, 0.4]),
            n_rows: 2,
            policy_salt: 5,
        })));
        round_trip(&CausalQuery::Simulate(SimulateQuery::new(SimulationTarget::TemporalRollout {
            schedule: InterventionSequence::new(vec![antecedent_core::SequencedIntervention::new(
                Intervention::set(v(0), Value::f64(1.0)),
                antecedent_core::TemporalPolicy::sustained(0, 3),
            )]),
            n_rows: 2,
            n_steps: 3,
            window: 1,
            history: Arc::from([]),
            seed: 9,
        })));
        round_trip(&CausalQuery::Simulate(SimulateQuery::new(SimulationTarget::UnitEffects {
            treatment: v(0),
            outcome: v(1),
            active: 1.0,
            control: 0.0,
        })));
    }
}

#[cfg(test)]
#[path = "structure_query_wire_fitted_model_tests.rs"]
mod fitted_model_tests;
