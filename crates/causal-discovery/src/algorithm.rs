//! Discovery algorithm trait (DESIGN.md §5.1).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use causal_core::{ExecutionContext, VariableId};
use causal_data::{MultiEnvironmentData, TabularData, TimeSeriesData};

use crate::engine::DiscoveryWorkspace;
use crate::error::DiscoveryError;
use crate::jpcmci_plus::JpcmciPlus;
use crate::lpcmci::Lpcmci;
use crate::pc::{Pc, StaticCpdagDiscoveryResult};
use crate::pcmci::Pcmci;
use crate::pcmci_plus::PcmciPlus;
use crate::result::{CpdagDiscoveryResult, DagDiscoveryResult, PagDiscoveryResult};
use crate::rpcmci::{RegimeAssignment, Rpcmci, RpcmciDiscoveryResult};

/// Algorithm that accepts a concrete dataset type `D` (DESIGN.md §5.1).
///
/// `variables` and `workspace` are part of the discovery contract in this crate:
/// PCMCI-family engines need an explicit variable subset and reusable buffers.
/// Callers that only have `data` + `ctx` should store those on the algorithm and
/// forward through [`Self::discover`].
pub trait DiscoveryAlgorithm<D> {
    /// Typed discovery output (DAG / CPDAG / PAG result, …).
    type Output;

    /// Run discovery on `data`.
    ///
    /// # Errors
    ///
    /// Engine, orientation, or configuration failures.
    fn discover(
        &mut self,
        data: &D,
        variables: &[VariableId],
        workspace: &mut DiscoveryWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<Self::Output, DiscoveryError>;
}

impl DiscoveryAlgorithm<TimeSeriesData> for Pcmci {
    type Output = DagDiscoveryResult;

    fn discover(
        &mut self,
        data: &TimeSeriesData,
        variables: &[VariableId],
        workspace: &mut DiscoveryWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<Self::Output, DiscoveryError> {
        self.run(data, variables, workspace, ctx)
    }
}

impl DiscoveryAlgorithm<TimeSeriesData> for PcmciPlus {
    type Output = CpdagDiscoveryResult;

    fn discover(
        &mut self,
        data: &TimeSeriesData,
        variables: &[VariableId],
        workspace: &mut DiscoveryWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<Self::Output, DiscoveryError> {
        self.run(data, variables, workspace, ctx)
    }
}

impl DiscoveryAlgorithm<TimeSeriesData> for Lpcmci {
    type Output = PagDiscoveryResult;

    fn discover(
        &mut self,
        data: &TimeSeriesData,
        variables: &[VariableId],
        workspace: &mut DiscoveryWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<Self::Output, DiscoveryError> {
        self.run(data, variables, workspace, ctx)
    }
}

impl DiscoveryAlgorithm<MultiEnvironmentData> for JpcmciPlus {
    type Output = CpdagDiscoveryResult;

    fn discover(
        &mut self,
        data: &MultiEnvironmentData,
        variables: &[VariableId],
        workspace: &mut DiscoveryWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<Self::Output, DiscoveryError> {
        self.run(data, variables, workspace, ctx)
    }
}

impl DiscoveryAlgorithm<TabularData> for Pc {
    type Output = StaticCpdagDiscoveryResult;

    fn discover(
        &mut self,
        data: &TabularData,
        variables: &[VariableId],
        workspace: &mut DiscoveryWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<Self::Output, DiscoveryError> {
        self.run(data, variables, workspace, ctx)
    }
}

impl DiscoveryAlgorithm<TimeSeriesData> for Rpcmci {
    type Output = RpcmciDiscoveryResult;

    fn discover(
        &mut self,
        data: &TimeSeriesData,
        variables: &[VariableId],
        workspace: &mut DiscoveryWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<Self::Output, DiscoveryError> {
        let assignment = self
            .assignment
            .as_ref()
            .ok_or(DiscoveryError::unsupported(
                "Rpcmci::discover requires with_assignment(...) before discover",
            ))?;
        // Need clone of assignment for run if run takes &
        let assignment: RegimeAssignment = assignment.clone();
        self.run(data, variables, &assignment, workspace, ctx)
    }
}
