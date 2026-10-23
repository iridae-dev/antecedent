//! Layer four of a planning program: the declared planning request and the
//! Monte Carlo store its simulation is admitted for.
//!
//! The request's identity covers the design, every planning assumption
//! (effect, outcome law, intracluster correlation, serial dependence,
//! attrition, allocation ratio, level and tails, priors), the criterion, the
//! goal and the simulation size and seed, so two plans that differ in any of
//! them have different lowered identities.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{CanonicalWriter, SemanticDigest};
use antecedent_design::planning::PlanningRequest;

/// Bytes of one replicate's retained outcome: decision flags, the estimate, its
/// variance, the stopping look and the enrolled units.
pub const SIMULATION_RECORD_BYTES: u64 = 64;

/// The Monte Carlo store a planning simulation holds: one record per replicate
/// and one data scratch per worker, quoted with checked arithmetic.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct SimulationQuote {
    pub(crate) replicates: u64,
    pub(crate) record_bytes: u64,
    pub(crate) scratch_bytes: u64,
    pub(crate) workers: u64,
    pub(crate) total_bytes: u64,
}

impl SimulationQuote {
    /// Quote `replicates` records and `workers` scratches of `scratch_bytes`,
    /// or `None` when a product or the sum overflows `u64`.
    #[must_use]
    pub const fn quote(replicates: u64, scratch_bytes: u64, workers: u64) -> Option<Self> {
        let Some(records) = replicates.checked_mul(SIMULATION_RECORD_BYTES) else {
            return None;
        };
        let Some(scratch) = scratch_bytes.checked_mul(workers) else {
            return None;
        };
        let Some(total_bytes) = records.checked_add(scratch) else {
            return None;
        };
        Some(Self {
            replicates,
            record_bytes: SIMULATION_RECORD_BYTES,
            scratch_bytes,
            workers,
            total_bytes,
        })
    }

    /// Replicates the simulation runs.
    #[must_use]
    pub const fn replicates(&self) -> u64 {
        self.replicates
    }

    /// Bytes of one replicate record.
    #[must_use]
    pub const fn record_bytes(&self) -> u64 {
        self.record_bytes
    }

    /// Bytes of one worker's data scratch at the largest planned size.
    #[must_use]
    pub const fn scratch_bytes(&self) -> u64 {
        self.scratch_bytes
    }

    /// Workers the quote provisions scratch for.
    #[must_use]
    pub const fn workers(&self) -> u64 {
        self.workers
    }

    /// Bytes the simulation leases before it allocates.
    #[must_use]
    pub const fn total_bytes(&self) -> u64 {
        self.total_bytes
    }
}

/// A planning request carried by a lowered program.
#[derive(Clone, Debug, PartialEq)]
pub struct PlanningProgram {
    pub(crate) request: Arc<PlanningRequest>,
    pub(crate) request_identity: SemanticDigest,
    pub(crate) simulation: Option<SimulationQuote>,
}

impl PlanningProgram {
    /// The declared request.
    #[must_use]
    pub fn request(&self) -> &PlanningRequest {
        &self.request
    }

    /// Identity of the declared request.
    #[must_use]
    pub const fn request_identity(&self) -> SemanticDigest {
        self.request_identity
    }

    /// Quoted simulation store, when the request simulates.
    #[must_use]
    pub const fn simulation(&self) -> Option<SimulationQuote> {
        self.simulation
    }

    /// Write the program to a lowered identity. The quote is physical
    /// accounting derived from the request, which the identity already covers.
    pub(crate) fn write(&self, writer: &mut CanonicalWriter) {
        writer.text("planning").bytes(self.request_identity.as_bytes());
    }
}
