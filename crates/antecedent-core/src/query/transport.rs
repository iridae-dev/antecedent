//! Structural transportability queries.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::fault::TransportFault;
use std::sync::Arc;

use crate::VariableId;
use crate::canonical_encode::{CanonicalEncode, encode_option, encode_variable_set};
use crate::identity::CanonicalWriter;

use super::transport_catalog::{EvidenceCatalog, FactorNeed, UnmetDependency, same_variable_set};
use super::{QueryError, ResponseQuery};

/// Transport a response from one explicitly named population to another.
///
/// `source_experiments` is the compatibility view. When [`Self::catalog`] is
/// absent, those variables become source experimental regimes (historical
/// behaviour). When a catalog is present, `source_experiments` is derived from
/// available experimental regimes and must not disagree with an explicit list.
#[derive(Clone, Debug, PartialEq)]
pub struct TransportQuery {
    /// Response functional requested in the target population.
    pub response: ResponseQuery,
    /// Source population key.
    pub source_population: Arc<str>,
    /// Target population key.
    pub target_population: Arc<str>,
    /// Compatibility view of source experimental variables.
    pub source_experiments: Arc<[VariableId]>,
    /// Optional supplied evidence catalog. Absent means the compatibility view
    /// is the entire experimental setting.
    pub catalog: Option<EvidenceCatalog>,
    /// A weighted target population over the target rows (a kernel over covariate distance, or
    /// declared row weights). Absent keeps the labelled target population and every existing
    /// identity: nothing is encoded for it.
    pub weighted_target: Option<TransportWeightedTarget>,
}

/// The kernel of a weighted transport target, as declared.
///
/// Reals are carried as their IEEE bits so the declaration is hashed and compared exactly.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TransportTargetKernel {
    /// Kernel family id (`gaussian`, `epanechnikov` or `tricube`).
    pub family: Arc<str>,
    /// Bandwidth bits.
    pub bandwidth_bits: u64,
    /// Center coordinate bits, one per covariate.
    pub center_bits: Arc<[u64]>,
    /// Scale bits, one per covariate.
    pub scale_bits: Arc<[u64]>,
}

/// A weighted target population of a trial-to-target transport.
///
/// The target is the non-participant rows reweighted by a function of `covariates`. The
/// `identity` is the digest of the declaration and the exact weight bits, so two targets that
/// weight the rows differently never share a query identity. A kernel target carries its
/// declaration so an artifact replays the same target against its data; explicit row weights
/// are caller data like the trial columns, verified against `identity`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TransportWeightedTarget {
    /// Digest of the declaration and the exact weight bits.
    pub identity: [u8; 32],
    /// Covariates the weights are a function of.
    pub covariates: Arc<[VariableId]>,
    /// Bits of the ESS-fraction floor of the overlap gate.
    pub min_ess_fraction_bits: u64,
    /// The declared kernel; `None` for explicit row weights.
    pub kernel: Option<TransportTargetKernel>,
}

impl CanonicalEncode for TransportWeightedTarget {
    fn encode(&self, writer: &mut CanonicalWriter) {
        writer.bytes(&self.identity);
        encode_variable_set(writer, &self.covariates);
        writer.word(self.min_ess_fraction_bits);
        match &self.kernel {
            Some(kernel) => {
                writer.word(1).text(&kernel.family).word(kernel.bandwidth_bits);
                writer.words(&kernel.center_bits).words(&kernel.scale_bits);
            }
            None => {
                writer.word(0);
            }
        }
    }
}

/// Domain tag of the weighted-target section of a transport query's encoding.
const WEIGHTED_TARGET_TAG: u64 = 0x7767_7474_6172_6774;

/// Every field is written. `source_experiments` is a variable set (it is
/// compared to the catalog as a set) and is encoded in ascending id order.
impl CanonicalEncode for TransportQuery {
    fn encode(&self, writer: &mut CanonicalWriter) {
        self.response.encode(writer);
        writer.text(&self.source_population).text(&self.target_population);
        encode_variable_set(writer, &self.source_experiments);
        encode_option(writer, self.catalog.as_ref());
        // Written only when declared, so a query without a weighted target keeps its identity.
        if let Some(target) = &self.weighted_target {
            writer.word(WEIGHTED_TARGET_TAG);
            target.encode(writer);
        }
    }
}

impl TransportQuery {
    /// Construct a single-source transport query without a catalog.
    #[must_use]
    pub fn new(
        response: ResponseQuery,
        source_population: impl Into<Arc<str>>,
        target_population: impl Into<Arc<str>>,
        source_experiments: impl Into<Arc<[VariableId]>>,
    ) -> Self {
        Self {
            response,
            source_population: source_population.into(),
            target_population: target_population.into(),
            source_experiments: source_experiments.into(),
            catalog: None,
            weighted_target: None,
        }
    }

    /// Attach a validated catalog. Derived source experiments must not disagree.
    ///
    /// # Errors
    ///
    /// [`QueryError::InvalidTransport`] when the catalog is invalid or the
    /// explicit `source_experiments` list disagrees with available source
    /// experimental regimes.
    pub fn with_catalog(mut self, catalog: EvidenceCatalog) -> Result<Self, QueryError> {
        catalog.validate()?;
        let derived = catalog.source_experiment_variables(&self.source_population);
        if !self.source_experiments.is_empty()
            && !same_variable_set(&self.source_experiments, &derived)
        {
            return Err(QueryError::InvalidTransport(TransportFault::SourceExperimentsDisagree));
        }
        if self.source_experiments.is_empty() {
            self.source_experiments = derived;
        }
        self.catalog = Some(catalog);
        Ok(self)
    }

    /// Declare a weighted target population for the target rows.
    ///
    /// # Errors
    ///
    /// [`QueryError::InvalidTransport`] when the target names no covariate or its floor is
    /// outside `[0, 1]`.
    pub fn with_weighted_target(
        mut self,
        target: TransportWeightedTarget,
    ) -> Result<Self, QueryError> {
        target.validate()?;
        self.weighted_target = Some(target);
        Ok(self)
    }

    /// Compatibility catalog: supplied catalog, or one synthesized regime per
    /// listed source experiment.
    #[must_use]
    fn compatibility_catalog(&self) -> EvidenceCatalog {
        self.catalog.clone().unwrap_or_else(|| {
            EvidenceCatalog::from_source_experiments(
                Arc::clone(&self.source_population),
                &self.source_experiments,
            )
        })
    }

    /// Unmet executable-factor dependencies in stable order.
    #[must_use]
    pub fn unmet_factor_dependencies(
        &self,
        needed: &[(Arc<str>, FactorNeed<'_>)],
    ) -> Arc<[UnmetDependency]> {
        self.compatibility_catalog().unmet_factor_dependencies(needed)
    }

    /// Validate population keys, response semantics, experiment uniqueness, and catalog.
    ///
    /// # Errors
    ///
    /// [`QueryError::InvalidTransport`] or the nested response validation error.
    pub fn validate(&self) -> Result<(), QueryError> {
        self.response.validate()?;
        if self.source_population.trim().is_empty()
            || self.target_population.trim().is_empty()
            || self.source_population == self.target_population
        {
            return Err(QueryError::InvalidTransport(TransportFault::PopulationKeys));
        }
        let mut experiments = self.source_experiments.to_vec();
        experiments.sort_unstable_by_key(|id| id.raw());
        if experiments.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(QueryError::InvalidTransport(TransportFault::SourceExperimentsNotUnique));
        }
        if let Some(target) = &self.weighted_target {
            target.validate()?;
        }
        if let Some(catalog) = &self.catalog {
            catalog.validate()?;
            let derived = catalog.source_experiment_variables(&self.source_population);
            if !same_variable_set(&self.source_experiments, &derived) {
                return Err(QueryError::InvalidTransport(
                    TransportFault::SourceExperimentsDisagree,
                ));
            }
        }
        Ok(())
    }
}

impl TransportWeightedTarget {
    /// The ESS-fraction floor.
    #[must_use]
    pub fn min_ess_fraction(&self) -> f64 {
        f64::from_bits(self.min_ess_fraction_bits)
    }

    /// Validate the covariates and the floor.
    ///
    /// # Errors
    ///
    /// [`QueryError::InvalidTransport`] for no covariates, or a floor outside `[0, 1]`.
    pub fn validate(&self) -> Result<(), QueryError> {
        let floor = self.min_ess_fraction();
        if self.covariates.is_empty() || !(0.0..=1.0).contains(&floor) {
            return Err(QueryError::InvalidTransport(TransportFault::WeightedTarget));
        }
        Ok(())
    }
}
