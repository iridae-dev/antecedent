//! The portable form of a discovery run: configuration identities, statistic receipts and
//! graph identities. It carries no raw data.
//!
//! # What a consumer may do with it
//!
//! * Inspect it without data: the route, the declared inputs as digests, the structure, the
//!   unresolved edges, the bounded-search receipts, the producer's work counters.
//! * Resume from it with the raw data and the same contract: the algorithm is replayed over
//!   the receipts only (a statistic it does not hold is a refusal) and must reproduce the
//!   recorded result identity.
//!
//! It is not a transcript the consumer trusts on its own say-so: statistics are bound to the
//! content they were computed from, so missing raw data, different data, a different
//! contract, a receipt that is not a usable statistic or a replay that reaches another result
//! are each refused.
//!
//! # Limit
//!
//! A receipt the replay never consumes is retained but is not re-derived: replay validates
//! the receipts the algorithm reads, through the result it must reproduce. A producer that
//! wants every receipt checked recomputes from the raw data with an empty store.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeSet;

use super::identity::{DataIdentity, Digest};
use super::outcome::DiscoveryOutcome;
use super::refusal::SelectiveDiscoveryError;
use super::stages::{DiscoveryStage, StageInputs};
use super::store::{MAX_PORTABLE_RECEIPTS, StatisticReceipt, StatisticStore};
use super::structure::DiscoveredStructure;
use super::work::WorkUnits;

/// The retained state of one discovery run in portable form.
#[derive(Clone, Debug)]
pub struct PortableDiscoveryState {
    /// Route label (`static_pc`, `static_fci`, `temporal_pcmci`, `temporal_pcmci_plus`).
    pub route: String,
    /// Every declared stage input as `(name, digest)` in canonical name order.
    pub components: Vec<(String, Digest)>,
    /// Identity of the contract (everything but data and kernel policy).
    pub contract_identity: Digest,
    /// Identity of the data the statistics were computed from.
    pub data_identity: Digest,
    /// Rows (or time points) of that data.
    pub rows: u64,
    /// Searched variables in search order.
    pub variables: Vec<u32>,
    /// Retained statistics, strictly ascending by coordinate.
    pub receipts: Vec<StatisticReceipt>,
    /// Identity of the receipt list.
    pub receipts_identity: Digest,
    /// The discovered structure.
    pub structure: DiscoveredStructure,
    /// Identity of everything the algorithm returned except execution detail.
    pub result_identity: Digest,
    /// Whether a search bound was exhausted.
    pub search_incomplete: bool,
    /// Whether orientation evidence conflicted.
    pub orientation_conflict: bool,
    /// `(code, message)` of the bounded-search diagnostics.
    pub bounded_search_receipts: Vec<(String, String)>,
    /// The producer's work counters for the run that produced the state.
    pub work: WorkUnits,
}

fn refuse(detail: &'static str, remedy: impl Into<String>) -> SelectiveDiscoveryError {
    SelectiveDiscoveryError::refused(
        antecedent_core::reason_code!("discovery_statistics_invalid"),
        detail,
        remedy,
    )
}

impl PortableDiscoveryState {
    pub(crate) fn from_live(
        inputs: &StageInputs,
        data: &DataIdentity,
        contract_identity: Digest,
        store: &StatisticStore,
        outcome: &DiscoveryOutcome,
        work: WorkUnits,
    ) -> Result<Self, SelectiveDiscoveryError> {
        if store.len() > MAX_PORTABLE_RECEIPTS {
            return Err(refuse(
                "selective_discovery.receipt_limit",
                format!(
                    "the run retains {} statistics; a portable state carries at most \
                     {MAX_PORTABLE_RECEIPTS}. Export a smaller search (fewer variables, a lower \
                     conditioning-set limit) or recompute from the raw data",
                    store.len()
                ),
            ));
        }
        let receipts = store.receipts();
        let receipts_identity = StatisticStore::receipts_identity(&receipts);
        Ok(Self {
            route: inputs.route().to_owned(),
            components: inputs.components().map(|(n, d)| (n.to_owned(), *d)).collect(),
            contract_identity,
            data_identity: data.digest,
            rows: data.rows,
            variables: data.columns.iter().map(|(v, _)| v.raw()).collect(),
            receipts,
            receipts_identity,
            structure: outcome.structure.clone(),
            result_identity: outcome.result_identity,
            search_incomplete: outcome.search_incomplete,
            orientation_conflict: outcome.orientation_conflict,
            bounded_search_receipts: outcome.bounded_search_receipts.clone(),
            work,
        })
    }

    /// Recompute the receipt identity and the contract identity from the fields.
    ///
    /// For a producer that assembles a state by hand and for tests that edit a field and
    /// reseal it. A consumer never calls this: it recomputes and compares.
    pub fn reseal(&mut self) {
        self.receipts_identity = StatisticStore::receipts_identity(&self.receipts);
        let mut ordered: Vec<(&str, &Digest)> =
            self.components.iter().map(|(n, d)| (n.as_str(), d)).collect();
        ordered.sort_unstable_by_key(|(name, _)| *name);
        self.contract_identity = super::contract::contract_identity_of(&self.route, ordered);
    }

    /// Identity of the stored structure.
    #[must_use]
    pub fn graph_identity(&self) -> Digest {
        self.structure.identity()
    }

    /// Check the state is internally consistent: usable receipts in strictly ascending
    /// coordinate order, a matching receipt identity, and every declared input present once.
    ///
    /// # Errors
    ///
    /// The first inconsistency found.
    pub fn validate(&self) -> Result<(), SelectiveDiscoveryError> {
        if self.receipts.len() > MAX_PORTABLE_RECEIPTS {
            return Err(refuse(
                "selective_discovery.receipt_limit",
                format!("a portable state carries at most {MAX_PORTABLE_RECEIPTS} receipts"),
            ));
        }
        if self.receipts.iter().any(|r| !r.is_valid()) {
            return Err(refuse(
                "selective_discovery.receipt_invalid",
                "a retained statistic is not finite or its p-value is outside [0, 1]; \
                 recompute it from the raw data",
            ));
        }
        if self.receipts.windows(2).any(|w| w[0].coordinate >= w[1].coordinate) {
            return Err(refuse(
                "selective_discovery.receipt_repeated",
                "receipts are not strictly ascending by coordinate: the state is not the \
                 record of one run",
            ));
        }
        if StatisticStore::receipts_identity(&self.receipts) != self.receipts_identity {
            return Err(refuse(
                "selective_discovery.receipts_identity",
                "the receipt identity does not match the receipts",
            ));
        }
        let mut ordered: Vec<(&str, &Digest)> =
            self.components.iter().map(|(n, d)| (n.as_str(), d)).collect();
        ordered.sort_unstable_by_key(|(name, _)| *name);
        if super::contract::contract_identity_of(&self.route, ordered) != self.contract_identity {
            return Err(refuse(
                "selective_discovery.contract_identity",
                "the contract identity does not match the declared inputs",
            ));
        }
        let expected: BTreeSet<&str> = DiscoveryStage::ALL
            .into_iter()
            .flat_map(|stage| stage.own_components().iter().copied())
            .collect();
        let named: BTreeSet<&str> = self.components.iter().map(|(n, _)| n.as_str()).collect();
        if named != expected || self.components.len() != expected.len() {
            return Err(refuse(
                "selective_discovery.components_incomplete",
                "the state does not declare every stage input exactly once",
            ));
        }
        Ok(())
    }

    /// Check the state belongs to a request: same route, same data content, same contract.
    pub(crate) fn verify_binding(
        &self,
        inputs: &StageInputs,
        data: &DataIdentity,
    ) -> Result<(), SelectiveDiscoveryError> {
        self.validate()?;
        if self.route != inputs.route() {
            return Err(refuse(
                "selective_discovery.contract_mismatch",
                format!(
                    "the state was produced by route `{}`, this request is `{}`",
                    self.route,
                    inputs.route()
                ),
            ));
        }
        if self.data_identity != data.digest {
            return Err(refuse(
                "selective_discovery.data_mismatch",
                "the supplied data differ from the data the statistics were computed from; \
                 new data invalidates the statistics and discovery is rerun, not resumed",
            ));
        }
        let differing: Vec<&str> = self
            .components
            .iter()
            .filter(|(name, digest)| inputs.component(name) != Some(digest))
            .map(|(name, _)| name.as_str())
            .collect();
        if !differing.is_empty() {
            return Err(refuse(
                "selective_discovery.contract_mismatch",
                format!(
                    "the request differs from the state in: {}. Resume with the contract and \
                     kernel policy the state was produced under, or rerun discovery",
                    differing.join(", ")
                ),
            ));
        }
        Ok(())
    }

    /// Check a replayed outcome is the recorded one.
    pub(crate) fn verify_replayed(
        &self,
        outcome: &DiscoveryOutcome,
    ) -> Result<(), SelectiveDiscoveryError> {
        if outcome.result_identity != self.result_identity
            || outcome.structure_identity != self.structure.identity()
        {
            return Err(refuse(
                "selective_discovery.replay_diverged",
                "replaying the algorithm over the retained statistics did not reproduce the \
                 recorded result: a receipt or the recorded result was changed. Recompute from \
                 the raw data",
            ));
        }
        Ok(())
    }
}
