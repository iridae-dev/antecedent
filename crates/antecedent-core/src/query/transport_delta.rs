//! Hypothetical evidence changes for transport planning.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::{collections::BTreeSet, sync::Arc};

use crate::{EvidenceCatalog, EvidenceKind, EvidenceRegime, QueryError, RegimeId};

/// A proposed set of new study laws. It is never itself available evidence.
#[derive(Clone, Debug, PartialEq)]
pub struct EvidenceCatalogDelta {
    /// Proposed regimes, each with an identity absent from the base catalog.
    pub proposed_regimes: Arc<[EvidenceRegime]>,
}

impl EvidenceCatalogDelta {
    /// Validate a hypothetical addition without changing the source catalog.
    ///
    /// # Errors
    /// A proposal is already available, has a duplicate id, or violates the
    /// catalog's population and variable contract.
    pub fn try_new(
        base: &EvidenceCatalog,
        proposed_regimes: impl Into<Arc<[EvidenceRegime]>>,
    ) -> Result<Self, QueryError> {
        let proposed_regimes = proposed_regimes.into();
        let mut ids: BTreeSet<RegimeId> = base.regimes.iter().map(|regime| regime.id).collect();
        for regime in proposed_regimes.iter() {
            if regime.evidence_kind != EvidenceKind::Proposed || !ids.insert(regime.id) {
                return Err(QueryError::InvalidTransport(
                    "catalog delta requires distinct proposed regimes".into(),
                ));
            }
        }
        let mut combined = base.regimes.to_vec();
        combined.extend(proposed_regimes.iter().cloned());
        EvidenceCatalog::try_new(
            Arc::clone(&base.environments),
            combined,
            Arc::clone(&base.bindings),
            base.target_sampling,
        )?;
        Ok(Self { proposed_regimes })
    }

    /// Build a temporary catalog for structural identification and factor
    /// binding. No data or dataset binding is created by this preview.
    ///
    /// # Errors
    /// Invalid base catalog or changed proposal identity.
    pub fn preview_catalog(&self, base: &EvidenceCatalog) -> Result<EvidenceCatalog, QueryError> {
        Self::try_new(base, Arc::clone(&self.proposed_regimes))?;
        let mut regimes = base.regimes.to_vec();
        regimes.extend(self.proposed_regimes.iter().cloned().map(|mut regime| {
            regime.evidence_kind = EvidenceKind::Available;
            regime
        }));
        EvidenceCatalog::try_new(
            Arc::clone(&base.environments),
            regimes,
            Arc::clone(&base.bindings),
            base.target_sampling,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DistributionAvailability, RegimeKind, VariableId};

    #[test]
    fn preview_is_available_only_in_the_clone() {
        let base = EvidenceCatalog::empty();
        let proposed = EvidenceRegime::try_new(
            RegimeId::from_raw(7),
            RegimeKind::Experimental,
            EvidenceKind::Proposed,
            [VariableId::from_raw(0)],
            [],
            [VariableId::from_raw(1)],
            "source",
            DistributionAvailability::Joint,
        )
        .unwrap();
        let delta = EvidenceCatalogDelta::try_new(&base, [proposed]).unwrap();
        let preview = delta.preview_catalog(&base).unwrap();
        assert!(!base.has_available_experiment("source", &[VariableId::from_raw(0)]));
        assert!(preview.has_available_experiment("source", &[VariableId::from_raw(0)]));
        assert_eq!(delta.proposed_regimes[0].evidence_kind, EvidenceKind::Proposed);
    }

    #[test]
    fn delta_rejects_available_and_duplicate_proposals() {
        let base = EvidenceCatalog::empty();
        let mut proposed = EvidenceRegime::try_new(
            RegimeId::from_raw(7),
            RegimeKind::Experimental,
            EvidenceKind::Proposed,
            [VariableId::from_raw(0)],
            [],
            [VariableId::from_raw(1)],
            "source",
            DistributionAvailability::Joint,
        )
        .unwrap();
        proposed.evidence_kind = EvidenceKind::Available;
        assert!(EvidenceCatalogDelta::try_new(&base, [proposed.clone()]).is_err());
        proposed.evidence_kind = EvidenceKind::Proposed;
        assert!(EvidenceCatalogDelta::try_new(&base, [proposed.clone(), proposed]).is_err());
    }
}
