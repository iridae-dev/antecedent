//! Structural transportability queries.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

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
}

/// Every field is written. `source_experiments` is a variable set (it is
/// compared to the catalog as a set) and is encoded in ascending id order.
impl CanonicalEncode for TransportQuery {
    fn encode(&self, writer: &mut CanonicalWriter) {
        self.response.encode(writer);
        writer.text(&self.source_population).text(&self.target_population);
        encode_variable_set(writer, &self.source_experiments);
        encode_option(writer, self.catalog.as_ref());
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
