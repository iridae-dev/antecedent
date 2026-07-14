//! Identifier contract (DESIGN.md §10.3).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use causal_core::{AssumptionSet, CausalQuery};
use causal_graph::Dag;

use crate::backdoor::{BackdoorIdentifier, PreparedIdentificationGraph};
use crate::efficient::EfficientBackdoorIdentifier;
use crate::error::IdentificationError;
use crate::frontdoor::FrontDoorIdentifier;
use crate::iv::InstrumentalVariableIdentifier;
use crate::result::IdentificationResult;

/// Scratch buffers for identification algorithms (DESIGN §10.3).
///
/// Phase-1 identifiers do not yet reuse this; it exists so the trait matches
/// DESIGN and callers can thread a workspace through polymorphic dispatch.
#[derive(Clone, Debug, Default)]
pub struct IdentificationWorkspace {
    _private: (),
}

/// Identification algorithm over graph type `G` (DESIGN §10.3).
///
/// Concrete identifiers keep inherent `prepare` / `identify` methods for
/// ergonomic use; this trait is the extension / dispatch surface. `assumptions`
/// and `workspace` are accepted for DESIGN parity even when a given algorithm
/// does not yet consume them.
pub trait Identifier<G> {
    /// Compile `graph` (+ declared assumptions) into a reusable prepared form.
    ///
    /// # Errors
    ///
    /// Graph validation failures or unsupported graph shape.
    fn prepare(
        &self,
        graph: &G,
        assumptions: &AssumptionSet,
    ) -> Result<PreparedIdentificationGraph, IdentificationError>;

    /// Identify `query` against a prepared graph.
    ///
    /// # Errors
    ///
    /// Unsupported query, unknown variables, or algorithm limits.
    fn identify(
        &self,
        prepared: &PreparedIdentificationGraph,
        query: &CausalQuery,
        workspace: &mut IdentificationWorkspace,
    ) -> Result<IdentificationResult, IdentificationError>;
}

impl Identifier<Dag> for BackdoorIdentifier {
    fn prepare(
        &self,
        graph: &Dag,
        _assumptions: &AssumptionSet,
    ) -> Result<PreparedIdentificationGraph, IdentificationError> {
        Self::prepare(self, graph)
    }

    fn identify(
        &self,
        prepared: &PreparedIdentificationGraph,
        query: &CausalQuery,
        _workspace: &mut IdentificationWorkspace,
    ) -> Result<IdentificationResult, IdentificationError> {
        Self::identify(self, prepared, query)
    }
}

impl Identifier<Dag> for EfficientBackdoorIdentifier {
    fn prepare(
        &self,
        graph: &Dag,
        _assumptions: &AssumptionSet,
    ) -> Result<PreparedIdentificationGraph, IdentificationError> {
        Self::prepare(self, graph)
    }

    fn identify(
        &self,
        prepared: &PreparedIdentificationGraph,
        query: &CausalQuery,
        _workspace: &mut IdentificationWorkspace,
    ) -> Result<IdentificationResult, IdentificationError> {
        Self::identify(self, prepared, query)
    }
}

impl Identifier<Dag> for FrontDoorIdentifier {
    fn prepare(
        &self,
        graph: &Dag,
        _assumptions: &AssumptionSet,
    ) -> Result<PreparedIdentificationGraph, IdentificationError> {
        Self::prepare(self, graph)
    }

    fn identify(
        &self,
        prepared: &PreparedIdentificationGraph,
        query: &CausalQuery,
        _workspace: &mut IdentificationWorkspace,
    ) -> Result<IdentificationResult, IdentificationError> {
        Self::identify(self, prepared, query)
    }
}

impl Identifier<Dag> for InstrumentalVariableIdentifier {
    fn prepare(
        &self,
        graph: &Dag,
        _assumptions: &AssumptionSet,
    ) -> Result<PreparedIdentificationGraph, IdentificationError> {
        Self::prepare(self, graph)
    }

    fn identify(
        &self,
        prepared: &PreparedIdentificationGraph,
        query: &CausalQuery,
        _workspace: &mut IdentificationWorkspace,
    ) -> Result<IdentificationResult, IdentificationError> {
        Self::identify(self, prepared, query)
    }
}
