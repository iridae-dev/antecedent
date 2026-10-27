//! Attribution query on a fitted structural causal model: intrinsic influence,
//! root causes, distribution change, path decomposition, feature relevance,
//! anomaly scores, unit change and mechanism-change detection.
//!
//! Each target is computed under the mechanisms fitted on the supplied
//! structure. None is a geometric support-matrix cell; the route executes and
//! stays unlicensed until a coverage record licenses it.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::fault::AttributeFault;
use std::sync::Arc;

use crate::canonical_encode::{CanonicalEncode, count, encode_variables};
use crate::identity::CanonicalWriter;
use crate::ids::VariableId;

use super::attribution::{
    AnomalyAttributionQuery, ChangeAttributionQuery, MechanismChangeQuery, UnitChangeQuery,
};
use super::error::QueryError;

/// Allocation of the intrinsic-influence variance game among its sources.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum InfluenceAllocation {
    /// Exact up to the exact-player limit, permutation sampling above it.
    Auto,
    /// Exact Shapley over every coalition.
    Exact,
    /// Shapley by permutation sampling.
    Permutation {
        /// Permutations drawn (at least 2).
        n_permutations: usize,
    },
}

impl CanonicalEncode for InfluenceAllocation {
    fn encode(&self, writer: &mut CanonicalWriter) {
        match self {
            Self::Auto => {
                writer.word(1);
            }
            Self::Exact => {
                writer.word(2);
            }
            Self::Permutation { n_permutations } => {
                writer.word(3).word(count(*n_permutations));
            }
        }
    }
}

/// How an intrinsic-influence query declares the exogenous noise.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub enum NoiseDeclaration {
    /// No declaration; the attribution refuses, as it requires one.
    Undeclared,
    /// Every mechanism's noise is independent of the others' (no latent common cause).
    Independent,
    /// The structure the model was built from, with its bidirected pairs; any
    /// bidirected pair refuses the attribution.
    Structure {
        /// Latent common-cause pairs of the structure.
        bidirected: Arc<[(VariableId, VariableId)]>,
    },
}

impl CanonicalEncode for NoiseDeclaration {
    fn encode(&self, writer: &mut CanonicalWriter) {
        match self {
            Self::Undeclared => {
                writer.word(1);
            }
            Self::Independent => {
                writer.word(2);
            }
            Self::Structure { bidirected } => {
                writer.word(3).word(count(bidirected.len()));
                for (a, b) in bidirected.iter() {
                    writer.word(u64::from(a.raw())).word(u64::from(b.raw()));
                }
            }
        }
    }
}

/// Intrinsic causal influence of the upstream noise terms on a target's variance.
#[derive(Clone, Debug, PartialEq)]
pub struct IntrinsicInfluenceTarget {
    /// Variable whose variance is attributed.
    pub target: VariableId,
    /// Allocation rule.
    pub allocation: InfluenceAllocation,
    /// Outer noise draws per replicate batch (at least 2).
    pub outer_draws: usize,
    /// Inner draws per outer draw (at least 2).
    pub inner_draws: usize,
    /// Independent replicate batches (at least 2).
    pub replicates: usize,
    /// Row-bootstrap refits for mechanism-fitting uncertainty; `0` for none.
    pub fit_bootstrap_replicates: u32,
    /// Two-sided level of every reported interval, in `(0, 1)`.
    pub level: f64,
    /// Most sources accepted.
    pub max_sources: usize,
    /// Seed of the noise draws and the permutation stream.
    pub seed: u64,
    /// Noise declaration.
    pub noise: NoiseDeclaration,
}

impl CanonicalEncode for IntrinsicInfluenceTarget {
    fn encode(&self, writer: &mut CanonicalWriter) {
        self.target.encode(writer);
        self.allocation.encode(writer);
        writer
            .word(count(self.outer_draws))
            .word(count(self.inner_draws))
            .word(count(self.replicates))
            .word(u64::from(self.fit_bootstrap_replicates))
            .real(self.level)
            .word(count(self.max_sources))
            .word(self.seed);
        self.noise.encode(writer);
    }
}

/// Difference measure of a sampled distribution-change attribution.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum ChangeMeasure {
    /// Difference of means.
    MeanDifference,
    /// Difference of variances.
    VarianceDifference,
    /// Gaussian Kullback-Leibler divergence from the all-baseline coalition.
    GaussianKl,
}

impl CanonicalEncode for ChangeMeasure {
    fn encode(&self, writer: &mut CanonicalWriter) {
        writer.word(match self {
            Self::MeanDifference => 1,
            Self::VarianceDifference => 2,
            Self::GaussianKl => 3,
        });
    }
}

/// How a distribution-change attribution evaluates its coalitions.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum ChangeEstimator {
    /// Mechanism hybrids sampled under the declared measure.
    Sampled {
        /// Difference measure.
        measure: ChangeMeasure,
        /// Samples per coalition evaluation.
        n_samples: usize,
        /// Sampling seed.
        seed: u64,
    },
    /// Robust regression hybrids.
    Robust {
        /// Cap on the units used for regression and evaluation.
        max_rows: usize,
        /// Monte Carlo draws of the nonlinear path.
        n_samples: usize,
        /// Seed of the nonlinear path.
        seed: u64,
    },
}

impl CanonicalEncode for ChangeEstimator {
    fn encode(&self, writer: &mut CanonicalWriter) {
        match self {
            Self::Sampled { measure, n_samples, seed } => {
                writer.word(1);
                measure.encode(writer);
                writer.word(count(*n_samples)).word(*seed);
            }
            Self::Robust { max_rows, n_samples, seed } => {
                writer.word(2).word(count(*max_rows)).word(count(*n_samples)).word(*seed);
            }
        }
    }
}

/// Two-sample test of a mechanism-change detection.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum MechanismChangeTest {
    /// Residual distributions compared by a Gaussian likelihood ratio.
    LikelihoodRatio,
    /// Mean difference of structural residuals.
    MeanDifference,
    /// Classifier two-sample proxy on residuals.
    ClassifierTwoSample,
    /// Kernel two-sample test on residuals.
    KernelTwoSample,
    /// Known-split change-point test.
    ChangePoint,
}

impl CanonicalEncode for MechanismChangeTest {
    fn encode(&self, writer: &mut CanonicalWriter) {
        writer.word(match self {
            Self::LikelihoodRatio => 1,
            Self::MeanDifference => 2,
            Self::ClassifierTwoSample => 3,
            Self::KernelTwoSample => 4,
            Self::ChangePoint => 5,
        });
    }
}

/// What an attribution query attributes.
#[derive(Clone, Debug, PartialEq)]
pub enum AttributeTarget {
    /// Shapley split of a target's variance among upstream noise terms.
    IntrinsicInfluence(IntrinsicInfluenceTarget),
    /// Distribution-change attribution with its components ranked as root causes.
    RootCauses {
        /// The change being attributed.
        change: ChangeAttributionQuery,
        /// How the coalitions are evaluated.
        estimator: ChangeEstimator,
    },
    /// Distribution-change attribution between two populations.
    DistributionChange {
        /// The change being attributed.
        change: ChangeAttributionQuery,
        /// How the coalitions are evaluated.
        estimator: ChangeEstimator,
    },
    /// Decomposition of the effect of `sources` on `outcome` along directed paths.
    Paths {
        /// Source variables.
        sources: Arc<[VariableId]>,
        /// Outcome variable.
        outcome: VariableId,
        /// Most paths enumerated.
        max_paths: usize,
        /// Longest path enumerated.
        max_len: usize,
        /// Keep only paths through every listed node; empty keeps every path.
        through: Arc<[VariableId]>,
    },
    /// Interventional relevance of features for an outcome, by Shapley values.
    FeatureRelevance {
        /// Outcome variable.
        outcome: VariableId,
        /// Features scored.
        features: Arc<[VariableId]>,
        /// Shift applied to each intervened feature from its empirical mean.
        delta: f64,
        /// Monte Carlo samples per coalition.
        n_samples: usize,
        /// Most features accepted.
        max_features: usize,
    },
    /// Anomaly scores of observed units.
    Anomaly(AnomalyAttributionQuery),
    /// Per-unit change attribution.
    UnitChange(UnitChangeQuery),
    /// Detection of changed mechanisms (not attribution).
    MechanismChange {
        /// Populations, targets and level.
        query: MechanismChangeQuery,
        /// Two-sample test.
        test: MechanismChangeTest,
    },
    /// Attribution of a distribution change to the parent sets that differ
    /// between the supplied structure and a comparison structure.
    StructureChange {
        /// The change being attributed.
        change: ChangeAttributionQuery,
        /// Directed edges of the comparison structure.
        comparison: Arc<[(VariableId, VariableId)]>,
        /// Difference measure.
        measure: ChangeMeasure,
        /// Samples per coalition evaluation.
        n_samples: usize,
        /// Sampling seed.
        seed: u64,
    },
}

impl AttributeTarget {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::IntrinsicInfluence(_) => "intrinsic_influence",
            Self::RootCauses { .. } => "root_causes",
            Self::DistributionChange { .. } => "distribution_change",
            Self::Paths { .. } => "paths",
            Self::FeatureRelevance { .. } => "feature_relevance",
            Self::Anomaly(_) => "anomaly",
            Self::UnitChange(_) => "unit_change",
            Self::MechanismChange { .. } => "mechanism_change",
            Self::StructureChange { .. } => "structure_change",
        }
    }
}

impl CanonicalEncode for AttributeTarget {
    fn encode(&self, writer: &mut CanonicalWriter) {
        match self {
            Self::IntrinsicInfluence(target) => {
                writer.word(1);
                target.encode(writer);
            }
            Self::RootCauses { change, estimator } => {
                writer.word(2);
                change.encode(writer);
                estimator.encode(writer);
            }
            Self::DistributionChange { change, estimator } => {
                writer.word(3);
                change.encode(writer);
                estimator.encode(writer);
            }
            Self::Paths { sources, outcome, max_paths, max_len, through } => {
                writer.word(4);
                encode_variables(writer, sources);
                outcome.encode(writer);
                writer.word(count(*max_paths)).word(count(*max_len));
                encode_variables(writer, through);
            }
            Self::FeatureRelevance { outcome, features, delta, n_samples, max_features } => {
                writer.word(5);
                outcome.encode(writer);
                encode_variables(writer, features);
                writer.real(*delta).word(count(*n_samples)).word(count(*max_features));
            }
            Self::Anomaly(query) => {
                writer.word(6);
                query.encode(writer);
            }
            Self::UnitChange(query) => {
                writer.word(7);
                query.encode(writer);
            }
            Self::MechanismChange { query, test } => {
                writer.word(8);
                query.encode(writer);
                test.encode(writer);
            }
            Self::StructureChange { change, comparison, measure, n_samples, seed } => {
                writer.word(9);
                change.encode(writer);
                writer.word(count(comparison.len()));
                for (from, to) in comparison.iter() {
                    writer.word(u64::from(from.raw())).word(u64::from(to.raw()));
                }
                measure.encode(writer);
                writer.word(count(*n_samples)).word(*seed);
            }
        }
    }
}

/// Attribution query on a fitted structural causal model.
#[derive(Clone, Debug, PartialEq)]
pub struct AttributeQuery {
    /// What is attributed.
    pub target: AttributeTarget,
}

impl CanonicalEncode for AttributeQuery {
    fn encode(&self, writer: &mut CanonicalWriter) {
        self.target.encode(writer);
    }
}

impl AttributeQuery {
    /// Attribute `target`.
    #[must_use]
    pub const fn new(target: AttributeTarget) -> Self {
        Self { target }
    }

    /// The variable the attribution is about: the attributed target, the
    /// change outcome, the path outcome, the first anomaly or detection target.
    ///
    /// # Errors
    ///
    /// [`QueryError::InvalidAttribute`] for an anomaly or detection query with no target.
    pub fn outcome(&self) -> Result<VariableId, QueryError> {
        match &self.target {
            AttributeTarget::IntrinsicInfluence(target) => Ok(target.target),
            AttributeTarget::RootCauses { change, .. }
            | AttributeTarget::DistributionChange { change, .. }
            | AttributeTarget::StructureChange { change, .. } => Ok(change.outcome),
            AttributeTarget::Paths { outcome, .. }
            | AttributeTarget::FeatureRelevance { outcome, .. } => Ok(*outcome),
            AttributeTarget::UnitChange(query) => Ok(query.outcome),
            AttributeTarget::Anomaly(query) => {
                query.targets.first().copied().ok_or_else(|| {
                    QueryError::InvalidAttribute(AttributeFault::AnomalyWithoutTarget)
                })
            }
            AttributeTarget::MechanismChange { query, .. } => {
                query.targets.first().copied().ok_or_else(|| {
                    QueryError::InvalidAttribute(AttributeFault::MechanismChangeWithoutTarget)
                })
            }
        }
    }

    /// Validate the target.
    ///
    /// # Errors
    ///
    /// The embedded query's own validation errors, and
    /// [`QueryError::InvalidAttribute`] for too few draws or replicates, a
    /// level outside `(0, 1)`, no sources or features, a zero path or feature
    /// limit, a non-finite shift, or no Monte Carlo samples.
    pub fn validate(&self) -> Result<(), QueryError> {
        let invalid = |fault: AttributeFault| Err(QueryError::InvalidAttribute(fault));
        match &self.target {
            AttributeTarget::IntrinsicInfluence(target) => {
                if target.outer_draws < 2 || target.inner_draws < 2 || target.replicates < 2 {
                    return invalid(AttributeFault::IntrinsicDraws);
                }
                if !(target.level.is_finite() && target.level > 0.0 && target.level < 1.0) {
                    return invalid(AttributeFault::IntervalLevelRange);
                }
                if target.max_sources == 0 {
                    return invalid(AttributeFault::SourceLimit);
                }
                if let InfluenceAllocation::Permutation { n_permutations } = target.allocation {
                    if n_permutations < 2 {
                        return invalid(AttributeFault::PermutationCount);
                    }
                }
                Ok(())
            }
            AttributeTarget::RootCauses { change, estimator }
            | AttributeTarget::DistributionChange { change, estimator } => {
                change.validate()?;
                let samples = match estimator {
                    ChangeEstimator::Sampled { n_samples, .. }
                    | ChangeEstimator::Robust { n_samples, .. } => *n_samples,
                };
                if samples == 0 {
                    return invalid(AttributeFault::ChangeSampleCount);
                }
                if let ChangeEstimator::Robust { max_rows: 0, .. } = estimator {
                    return invalid(AttributeFault::RobustRowCap);
                }
                Ok(())
            }
            AttributeTarget::Paths { sources, max_paths, max_len, .. } => {
                if sources.is_empty() {
                    return invalid(AttributeFault::PathSourcesEmpty);
                }
                if *max_paths == 0 || *max_len == 0 {
                    return Err(QueryError::NonPositivePathLimit);
                }
                Ok(())
            }
            AttributeTarget::FeatureRelevance {
                outcome,
                features,
                delta,
                n_samples,
                max_features,
            } => {
                if features.is_empty() || features.contains(outcome) {
                    return invalid(AttributeFault::FeatureRelevanceFeatures);
                }
                if !delta.is_finite() || *n_samples == 0 || *max_features == 0 {
                    return invalid(AttributeFault::FeatureRelevanceSampling);
                }
                Ok(())
            }
            AttributeTarget::Anomaly(query) => query.validate(),
            AttributeTarget::UnitChange(query) => query.validate(),
            AttributeTarget::MechanismChange { query, .. } => query.validate(),
            AttributeTarget::StructureChange { change, comparison, n_samples, .. } => {
                change.validate()?;
                if comparison.iter().any(|(from, to)| from == to) {
                    return invalid(AttributeFault::ComparisonSelfLoop);
                }
                if *n_samples == 0 {
                    return invalid(AttributeFault::StructureChangeSampleCount);
                }
                Ok(())
            }
        }
    }
}
