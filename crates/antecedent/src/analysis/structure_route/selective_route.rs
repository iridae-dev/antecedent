//! Selective discovery through the compiled structure-discovery route.
//!
//! A selective search is a structure search under a frozen contract of the selective adapter
//! (`antecedent_discovery::selective`): PC or FCI over a table, PCMCI or PCMCI+ over one series.
//! The adapter answers a test from a retained statistic only at an identical input coordinate,
//! so a fresh selective run and a resumed one return the same structure; resuming from a
//! [`PortableDiscoveryState`] reuses the statistics of the run that produced it.
//!
//! The search compiles as the `DiscoverStatic` or `DiscoverTemporal` route of its algorithm.
//! Its configuration digest binds the contract identity and, when it resumes, the identities of
//! the retained state (route, contract, data, receipts and result), so a resumed search is a
//! different program from a fresh one. A specification without a selective contract writes
//! nothing here and keeps the digest it always had.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{CanonicalWriter, DiscoverQuery, DiscoveryMethod, ExecutionContext};
use antecedent_discovery::DiscoveryConstraints;
use antecedent_discovery::selective::{
    DiscoveryOutcome, PortableDiscoveryState, SelectiveDiscoveryError, SelectiveRunReport,
    StaticAlgorithm, StaticDiscoveryContract, StaticSelectiveDiscovery, TemporalAlgorithm,
    TemporalDiscoveryContract, TemporalSelectiveDiscovery,
};

use super::{DiscoverData, DiscoveredStructure, DiscoverySpec, SearchParams};
use crate::error::CausalError;

/// The frozen contract of one selective search.
#[derive(Clone, Debug)]
pub(crate) enum SelectiveContract {
    /// PC or FCI over a table.
    Static(StaticDiscoveryContract),
    /// PCMCI or PCMCI+ over one series.
    Temporal(TemporalDiscoveryContract),
}

/// A selective search: its contract, the contract's identity and the state it resumes from.
#[derive(Clone, Debug)]
pub(crate) struct SelectiveSearch {
    contract: SelectiveContract,
    identity: [u8; 32],
    resume: Option<PortableDiscoveryState>,
}

impl SelectiveSearch {
    pub(super) fn constraints(&self) -> &DiscoveryConstraints {
        match &self.contract {
            SelectiveContract::Static(contract) => &contract.constraints,
            SelectiveContract::Temporal(contract) => &contract.constraints,
        }
    }

    pub(super) fn max_lag(&self) -> u32 {
        match &self.contract {
            SelectiveContract::Static(_) => 0,
            SelectiveContract::Temporal(contract) => contract.constraints.temporal.max_lag.raw(),
        }
    }

    fn variables(&self) -> &[antecedent_core::VariableId] {
        match &self.contract {
            SelectiveContract::Static(contract) => &contract.variables,
            SelectiveContract::Temporal(contract) => &contract.variables,
        }
    }

    /// The configuration this search adds to the specification's digest.
    pub(super) fn write(&self, writer: &mut CanonicalWriter) {
        writer.text("selective").bytes(&self.identity);
        match &self.resume {
            None => {
                writer.word(0);
            }
            Some(state) => {
                writer
                    .word(1)
                    .text(&state.route)
                    .bytes(&state.contract_identity)
                    .bytes(&state.data_identity)
                    .bytes(&state.receipts_identity)
                    .bytes(&state.result_identity);
            }
        }
    }
}

/// What a compiled selective search returned.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct SelectiveSearchResult {
    /// The outcome the adapter retained.
    pub outcome: DiscoveryOutcome,
    /// What the request reused and computed.
    pub report: SelectiveRunReport,
    /// The portable state a later search resumes from; `None` when the retained statistics
    /// exceed the adapter's portable limit, which is the adapter's export refusal.
    pub state: Option<PortableDiscoveryState>,
}

/// A selective refusal or failure as a compile error: an algorithm failure keeps its own
/// error, a refusal keeps its registered code and namespaced detail.
fn selective_error(error: SelectiveDiscoveryError) -> CausalError {
    match error {
        SelectiveDiscoveryError::Discovery(error) => CausalError::from(error),
        other => {
            let code = other.code().unwrap_or("invalid_argument");
            CausalError::Compile {
                message: format!("{}{code}: {other}", crate::error::REASON_PREFIX),
            }
        }
    }
}

impl DiscoverySpec {
    fn selective(
        method: DiscoveryMethod,
        contract: SelectiveContract,
        identity: [u8; 32],
        ci_identity: String,
    ) -> Self {
        Self::new(
            method,
            SearchParams::Selective(Arc::new(SelectiveSearch { contract, identity, resume: None })),
            ci_identity,
        )
    }

    /// A static selective search (PC or FCI over a table) under `contract`.
    ///
    /// The study's search variables must be the contract's, in its order, and the execution
    /// context's master seed must be the contract's seed.
    ///
    /// # Errors
    ///
    /// The contract's own refusals (an invalid contract or an unknown test).
    pub fn selective_static(contract: StaticDiscoveryContract) -> Result<Self, CausalError> {
        let identity = contract.identity().map_err(selective_error)?;
        let (_, ci_identity) = contract.ci_test.resolve().map_err(selective_error)?;
        let method = match contract.algorithm {
            StaticAlgorithm::Pc => DiscoveryMethod::Pc,
            StaticAlgorithm::Fci => DiscoveryMethod::Fci,
        };
        Ok(Self::selective(method, SelectiveContract::Static(contract), identity, ci_identity))
    }

    /// A temporal selective search (PCMCI or PCMCI+ over one series) under `contract`.
    ///
    /// The study's search variables must be the contract's, in its order, and the execution
    /// context's master seed must be the contract's seed.
    ///
    /// # Errors
    ///
    /// The contract's own refusals (an invalid contract or an unknown test).
    pub fn selective_temporal(contract: TemporalDiscoveryContract) -> Result<Self, CausalError> {
        let identity = contract.identity().map_err(selective_error)?;
        let (_, ci_identity) = contract.ci_test.resolve().map_err(selective_error)?;
        let method = match contract.algorithm {
            TemporalAlgorithm::Pcmci => DiscoveryMethod::Pcmci,
            TemporalAlgorithm::PcmciPlus => DiscoveryMethod::PcmciPlus,
        };
        Ok(Self::selective(method, SelectiveContract::Temporal(contract), identity, ci_identity))
    }

    /// The same selective search resumed from `state`: the adapter verifies the state against
    /// the contract and the study's data and reuses its statistics.
    ///
    /// # Errors
    ///
    /// A specification that is not a selective search.
    pub fn resuming(mut self, state: PortableDiscoveryState) -> Result<Self, CausalError> {
        let SearchParams::Selective(selective) = &self.search else {
            return Err(crate::compile_reason!(
                "invalid_argument",
                "only a selective search resumes from a portable state; build it with \
                 DiscoverySpec::selective_static or DiscoverySpec::selective_temporal"
            ));
        };
        let mut resumed = SelectiveSearch::clone(selective);
        resumed.resume = Some(state);
        self.search = SearchParams::Selective(Arc::new(resumed));
        Ok(self)
    }
}

fn finish(
    outcome: Option<&DiscoveryOutcome>,
    state: Result<PortableDiscoveryState, SelectiveDiscoveryError>,
    report: SelectiveRunReport,
) -> Result<DiscoveredStructure, CausalError> {
    let outcome = outcome.cloned().ok_or_else(|| {
        crate::compile_reason!("invalid_argument", "the selective search retained no outcome")
    })?;
    Ok(DiscoveredStructure::Selective(Box::new(SelectiveSearchResult {
        outcome,
        report,
        state: state.ok(),
    })))
}

/// Run `selective` on the study's data through its adapter: a fresh run, or a resume from the
/// retained state.
pub(super) fn run_selective(
    selective: &SelectiveSearch,
    query: &DiscoverQuery,
    data: &DiscoverData<'_>,
    ctx: &ExecutionContext,
) -> Result<DiscoveredStructure, CausalError> {
    if *query.variables != *selective.variables() {
        return Err(crate::compile_reason!(
            "invalid_argument",
            "the study searches other variables than the selective contract declares; build the \
             study over the contract's variables in its order"
        ));
    }
    match (&selective.contract, data) {
        (SelectiveContract::Static(contract), DiscoverData::Tabular(table)) => {
            let (adapter, report) = match &selective.resume {
                Some(state) => StaticSelectiveDiscovery::resume(contract, Some(*table), state, ctx)
                    .map_err(selective_error)?,
                None => {
                    let mut adapter = StaticSelectiveDiscovery::new();
                    let report = adapter.run(contract, table, ctx).map_err(selective_error)?;
                    (adapter, report)
                }
            };
            finish(adapter.outcome(), adapter.export_state(), report)
        }
        (SelectiveContract::Temporal(contract), DiscoverData::Series(series)) => {
            let (adapter, report) = match &selective.resume {
                Some(state) => {
                    TemporalSelectiveDiscovery::resume(contract, Some(*series), state, ctx)
                        .map_err(selective_error)?
                }
                None => {
                    let mut adapter = TemporalSelectiveDiscovery::new();
                    let report = adapter.run(contract, series, ctx).map_err(selective_error)?;
                    (adapter, report)
                }
            };
            finish(adapter.outcome(), adapter.export_state(), report)
        }
        _ => Err(crate::compile_reason!(
            "data_modality_not_supported",
            "a static selective search reads a table and a temporal selective search reads one \
             series"
        )),
    }
}
