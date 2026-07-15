//! Identifier contract (DESIGN.md §10.3).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use causal_core::{AssumptionSet, CausalQuery};
use causal_graph::{DSeparationWorkspace, Dag, GraphWorkspace};

use crate::backdoor::{BackdoorIdentifier, PreparedIdentificationGraph};
use crate::efficient::EfficientBackdoorIdentifier;
use crate::error::IdentificationError;
use crate::frontdoor::FrontDoorIdentifier;
use crate::iv::InstrumentalVariableIdentifier;
use crate::result::IdentificationResult;

/// Scratch buffers for identification algorithms (DESIGN §10.3).
///
/// Shared across `identify` calls so ancestry / d-separation workspaces are not
/// reallocated per query.
#[derive(Clone, Debug, Default)]
pub struct IdentificationWorkspace {
    /// Ancestry / descendant traversal scratch.
    pub graph: GraphWorkspace,
    /// d-separation / m-separation scratch.
    pub dsep: DSeparationWorkspace,
}

/// Identification algorithm over graph type `G` (DESIGN §10.3).
///
/// Concrete identifiers keep inherent `prepare` / `identify` methods as the
/// primary API. This trait is the extension / dispatch surface. Declared
/// `assumptions` are stored on the prepared graph and merged into the result;
/// `workspace` carries reusable graph-search scratch.
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
        assumptions: &AssumptionSet,
    ) -> Result<PreparedIdentificationGraph, IdentificationError> {
        Self::prepare_with_assumptions(self, graph, assumptions.clone())
    }

    fn identify(
        &self,
        prepared: &PreparedIdentificationGraph,
        query: &CausalQuery,
        workspace: &mut IdentificationWorkspace,
    ) -> Result<IdentificationResult, IdentificationError> {
        Self::identify(self, prepared, query, workspace)
    }
}

impl Identifier<Dag> for EfficientBackdoorIdentifier {
    fn prepare(
        &self,
        graph: &Dag,
        assumptions: &AssumptionSet,
    ) -> Result<PreparedIdentificationGraph, IdentificationError> {
        Self::prepare_with_assumptions(self, graph, assumptions.clone())
    }

    fn identify(
        &self,
        prepared: &PreparedIdentificationGraph,
        query: &CausalQuery,
        workspace: &mut IdentificationWorkspace,
    ) -> Result<IdentificationResult, IdentificationError> {
        Self::identify(self, prepared, query, workspace)
    }
}

impl Identifier<Dag> for FrontDoorIdentifier {
    fn prepare(
        &self,
        graph: &Dag,
        assumptions: &AssumptionSet,
    ) -> Result<PreparedIdentificationGraph, IdentificationError> {
        Self::prepare_with_assumptions(self, graph, assumptions.clone())
    }

    fn identify(
        &self,
        prepared: &PreparedIdentificationGraph,
        query: &CausalQuery,
        workspace: &mut IdentificationWorkspace,
    ) -> Result<IdentificationResult, IdentificationError> {
        Self::identify(self, prepared, query, workspace)
    }
}

impl Identifier<Dag> for InstrumentalVariableIdentifier {
    fn prepare(
        &self,
        graph: &Dag,
        assumptions: &AssumptionSet,
    ) -> Result<PreparedIdentificationGraph, IdentificationError> {
        Self::prepare_with_assumptions(self, graph, assumptions.clone())
    }

    fn identify(
        &self,
        prepared: &PreparedIdentificationGraph,
        query: &CausalQuery,
        workspace: &mut IdentificationWorkspace,
    ) -> Result<IdentificationResult, IdentificationError> {
        Self::identify(self, prepared, query, workspace)
    }
}
