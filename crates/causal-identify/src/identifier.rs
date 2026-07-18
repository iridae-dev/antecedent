//! Identifier contract (DESIGN.md §10.3).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use causal_core::{AssumptionSet, CausalQuery};
use causal_graph::{Admg, DSeparationWorkspace, Dag, GraphWorkspace};

use crate::backdoor::{BackdoorIdentifier, PreparedIdentificationGraph};
use crate::efficient::EfficientBackdoorIdentifier;
use crate::error::IdentificationError;
use crate::frontdoor::FrontDoorIdentifier;
use crate::id::IdIdentifier;
use crate::iv::InstrumentalVariableIdentifier;
use crate::prepared::PreparedAdmg;
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
    /// Prepared graph type for this identifier.
    type Prepared;

    /// Compile `graph` (+ declared assumptions) into a reusable prepared form.
    ///
    /// # Errors
    ///
    /// Graph validation failures or unsupported graph shape.
    fn prepare(
        &self,
        graph: &G,
        assumptions: &AssumptionSet,
    ) -> Result<Self::Prepared, IdentificationError>;

    /// Identify `query` against a prepared graph.
    ///
    /// # Errors
    ///
    /// Unsupported query, unknown variables, or algorithm limits.
    fn identify(
        &self,
        prepared: &Self::Prepared,
        query: &CausalQuery,
        workspace: &mut IdentificationWorkspace,
    ) -> Result<IdentificationResult, IdentificationError>;
}

impl Identifier<Dag> for BackdoorIdentifier {
    type Prepared = PreparedIdentificationGraph;

    fn prepare(
        &self,
        graph: &Dag,
        assumptions: &AssumptionSet,
    ) -> Result<Self::Prepared, IdentificationError> {
        Self::prepare_with_assumptions(self, graph, assumptions.clone())
    }

    fn identify(
        &self,
        prepared: &Self::Prepared,
        query: &CausalQuery,
        workspace: &mut IdentificationWorkspace,
    ) -> Result<IdentificationResult, IdentificationError> {
        Self::identify(self, prepared, query, workspace)
    }
}

impl Identifier<Dag> for EfficientBackdoorIdentifier {
    type Prepared = PreparedIdentificationGraph;

    fn prepare(
        &self,
        graph: &Dag,
        assumptions: &AssumptionSet,
    ) -> Result<Self::Prepared, IdentificationError> {
        Self::prepare_with_assumptions(self, graph, assumptions.clone())
    }

    fn identify(
        &self,
        prepared: &Self::Prepared,
        query: &CausalQuery,
        workspace: &mut IdentificationWorkspace,
    ) -> Result<IdentificationResult, IdentificationError> {
        Self::identify(self, prepared, query, workspace)
    }
}

impl Identifier<Dag> for FrontDoorIdentifier {
    type Prepared = PreparedIdentificationGraph;

    fn prepare(
        &self,
        graph: &Dag,
        assumptions: &AssumptionSet,
    ) -> Result<Self::Prepared, IdentificationError> {
        Self::prepare_with_assumptions(self, graph, assumptions.clone())
    }

    fn identify(
        &self,
        prepared: &Self::Prepared,
        query: &CausalQuery,
        workspace: &mut IdentificationWorkspace,
    ) -> Result<IdentificationResult, IdentificationError> {
        Self::identify(self, prepared, query, workspace)
    }
}

impl Identifier<Dag> for InstrumentalVariableIdentifier {
    type Prepared = PreparedIdentificationGraph;

    fn prepare(
        &self,
        graph: &Dag,
        assumptions: &AssumptionSet,
    ) -> Result<Self::Prepared, IdentificationError> {
        Self::prepare_with_assumptions(self, graph, assumptions.clone())
    }

    fn identify(
        &self,
        prepared: &Self::Prepared,
        query: &CausalQuery,
        workspace: &mut IdentificationWorkspace,
    ) -> Result<IdentificationResult, IdentificationError> {
        Self::identify(self, prepared, query, workspace)
    }
}

impl Identifier<Admg> for IdIdentifier {
    type Prepared = PreparedAdmg;

    fn prepare(
        &self,
        graph: &Admg,
        assumptions: &AssumptionSet,
    ) -> Result<Self::Prepared, IdentificationError> {
        Self::prepare_with_assumptions(self, graph, assumptions.clone())
    }

    fn identify(
        &self,
        prepared: &Self::Prepared,
        query: &CausalQuery,
        workspace: &mut IdentificationWorkspace,
    ) -> Result<IdentificationResult, IdentificationError> {
        Self::identify(self, prepared, query, workspace)
    }
}

impl Identifier<Dag> for IdIdentifier {
    type Prepared = PreparedAdmg;

    fn prepare(
        &self,
        graph: &Dag,
        assumptions: &AssumptionSet,
    ) -> Result<Self::Prepared, IdentificationError> {
        PreparedAdmg::from_dag_with_assumptions(graph, assumptions.clone())
    }

    fn identify(
        &self,
        prepared: &Self::Prepared,
        query: &CausalQuery,
        workspace: &mut IdentificationWorkspace,
    ) -> Result<IdentificationResult, IdentificationError> {
        Self::identify(self, prepared, query, workspace)
    }
}
