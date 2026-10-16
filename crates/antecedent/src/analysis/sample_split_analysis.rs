//! Discovery followed by analysis on a row split, in the analysis facade.
//!
//! [`discover_on_split`] partitions the rows of a table by a declared [`SplitRule`], runs
//! static discovery on the discovery part only, and keeps the [`SampleSplitReceipt`] that binds
//! the analysis part to that discovery. [`SplitDiscovery::check_analysis_table`] is the gate a
//! downstream analysis passes through: it refuses a table that is not the receipt's analysis
//! rows, and returns an [`AnalysisBinding`] that a claim carries so the discovery and the
//! split it rests on are part of the claim's identity.
//!
//! # What the binding licenses
//!
//! Because discovery saw none of the analysis rows, an analysis of those rows that treats the
//! discovered (or declared) graph as given is calibrated for the analysis procedure's own
//! sampling error *given that graph*. The binding says exactly that and no more
//! ([`AnalysisBinding::conditional_on_selected_graph`]). It does not by itself license "the
//! interval covers the true effect": that is a property of the whole method (split,
//! discovery, graph acceptance, estimation), and it is licensed only for the one construction
//! that was measured, through [`super::sample_split_effect`], whose interval is attested from
//! a coverage record keyed `discovered_sample_split`. For any other analysis of the analysis
//! rows the whole-method calibration stays `unmeasured`
//! ([`AnalysisBinding::whole_method_calibration`]).
//!
//! No per-dataset statement is made either way: whether a discovered graph recovers the true
//! one rests on faithfulness and causal sufficiency, which cannot be tested from the table
//! that is being discovered on.
//!
//! The same rows used to select and to analyse are the reuse case:
//! [`SplitDiscovery::request_unsplit_interval`] returns the refusal, and the evidence ledger
//! sees the shared observations ([`split_evidence_sources`]).
//!
//! # Refresh with outcome-only data
//!
//! A binding belongs to its receipt. New outcome-only rows are a new split: call
//! [`discover_on_split`] again with the declared rule, and the discovery, receipt, binding and
//! every downstream claim are recomputed against it. Nothing is patched in place, and a
//! refresh never converts reuse into a split.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_contract::evidence_ledger::SourceDeclaration;
use antecedent_core::{CanonicalWriter, ExecutionContext, VariableId};
use antecedent_data::TabularData;
use antecedent_discovery::selective::{
    DiscoveryOutcome, SelectiveDiscoveryError, SelectiveRunReport, StaticDiscoveryContract,
    StaticSelectiveDiscovery, request_downstream_interval,
};

pub use antecedent_discovery::selective::reuse::{
    SampleSplitReceipt, SplitRows, SplitRule, split_rows,
};

use super::selective_discovery::discovery_rows_source;

const BINDING_TAG: &str = "antecedent.sample_split_analysis.binding.v1";

/// The standing of whole-method calibration for a split analysis: recorded, not measured.
pub const WHOLE_METHOD_CALIBRATION: &str = "unmeasured";

/// A split, the discovery run on its discovery part, and the receipt tying them together.
#[derive(Debug)]
pub struct SplitDiscovery {
    split: SplitRows,
    outcome: DiscoveryOutcome,
    report: SelectiveRunReport,
    variables: Vec<VariableId>,
}

/// What an analysis table passed [`SplitDiscovery::check_analysis_table`] under.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AnalysisBinding {
    /// Identity of the split receipt (rule, row sets and both parts' content).
    pub receipt_identity: [u8; 32],
    /// Identity of the structure discovery returned on the discovery part.
    pub structure_identity: [u8; 32],
    /// Identity of the discovery result.
    pub result_identity: [u8; 32],
    /// Content identity of the analysis part that was checked.
    pub analysis_data_identity: [u8; 32],
    /// Identity of this binding: all of the above.
    pub identity: [u8; 32],
}

impl AnalysisBinding {
    /// Whether the licensed statement is conditional on the selected graph. Always true: the
    /// route removes the dependence between selection and analysis, it does not state that the
    /// selected graph is right.
    #[must_use]
    pub const fn conditional_on_selected_graph(&self) -> bool {
        true
    }

    /// Standing of the whole-method calibration: always [`WHOLE_METHOD_CALIBRATION`].
    #[must_use]
    pub const fn whole_method_calibration(&self) -> &'static str {
        WHOLE_METHOD_CALIBRATION
    }
}

/// Split `data` by `rule` and run `contract` on the discovery part.
///
/// The receipt records the content of both parts over `contract.variables`. The analysis part
/// carries every column of `data`, so an analysis may use covariates discovery did not search.
///
/// # Errors
///
/// Weighted or masked rows, a non-numeric column, a rule that leaves a part empty or names a
/// row twice or out of range (all refused before discovery runs), or a refused contract or an
/// algorithm failure or cancellation in discovery.
pub fn discover_on_split(
    contract: &StaticDiscoveryContract,
    data: &TabularData,
    rule: SplitRule,
    ctx: &ExecutionContext,
) -> Result<SplitDiscovery, SelectiveDiscoveryError> {
    let split = split_rows(data, &contract.variables, rule)?;
    let mut adapter = StaticSelectiveDiscovery::new();
    let report = adapter.run(contract, &split.discovery, ctx)?;
    let outcome = adapter.outcome().cloned().ok_or_else(|| {
        SelectiveDiscoveryError::refused(
            antecedent_core::reason_code!("invalid_argument"),
            "selective_discovery.nothing_retained",
            "the discovery run retained no outcome",
        )
    })?;
    Ok(SplitDiscovery { split, outcome, report, variables: contract.variables.clone() })
}

impl SplitDiscovery {
    /// The rows discovery used.
    #[must_use]
    pub const fn discovery_rows(&self) -> &TabularData {
        &self.split.discovery
    }

    /// The only rows a downstream analysis may use.
    #[must_use]
    pub const fn analysis_rows(&self) -> &TabularData {
        &self.split.analysis
    }

    /// The record of the split.
    #[must_use]
    pub const fn receipt(&self) -> &SampleSplitReceipt {
        &self.split.receipt
    }

    /// The structure discovery returned on the discovery part.
    #[must_use]
    pub const fn outcome(&self) -> &DiscoveryOutcome {
        &self.outcome
    }

    /// What the discovery request did.
    #[must_use]
    pub const fn report(&self) -> &SelectiveRunReport {
        &self.report
    }

    /// Re-derive the partition from the rule and check the stored identities.
    ///
    /// # Errors
    ///
    /// As [`SampleSplitReceipt::verify`].
    pub fn verify(&self) -> Result<(), SelectiveDiscoveryError> {
        self.split.receipt.verify()
    }

    /// Check that `table` is the receipt's analysis part and bind it to this discovery.
    ///
    /// # Errors
    ///
    /// A receipt that does not re-derive, or a table whose searched columns differ in content
    /// or row count from the recorded analysis part: the discovery rows, the unsplit table or
    /// any other rows (`selective_discovery.analysis_rows_not_the_split_part`).
    pub fn check_analysis_table(
        &self,
        table: &TabularData,
    ) -> Result<AnalysisBinding, SelectiveDiscoveryError> {
        let receipt = &self.split.receipt;
        receipt.check_analysis_table(table, &self.variables)?;
        let mut writer = CanonicalWriter::new(BINDING_TAG);
        writer
            .bytes(&receipt.identity)
            .bytes(&self.outcome.structure_identity)
            .bytes(&self.outcome.result_identity)
            .bytes(&receipt.analysis_data.digest);
        Ok(AnalysisBinding {
            receipt_identity: receipt.identity,
            structure_identity: self.outcome.structure_identity,
            result_identity: self.outcome.result_identity,
            analysis_data_identity: receipt.analysis_data.digest,
            identity: writer.finish(),
        })
    }

    /// Ask for an interval on account of discovery for an analysis that reused the rows
    /// discovery saw (the unsplit table, or the discovery part).
    ///
    /// # Errors
    ///
    /// Always `cell_not_licensed`
    /// (`selective_discovery.downstream_interval_not_licensed`): split the rows.
    pub fn request_unsplit_interval(
        &self,
    ) -> Result<std::convert::Infallible, SelectiveDiscoveryError> {
        request_downstream_interval(&self.outcome)
    }
}

/// The evidence-ledger declarations of the split's two parts under one snapshot id.
///
/// Returns `(discovery, analysis)`. Declare the unsplit table with the same `snapshot_id` (for
/// example [`super::selective_discovery::discovery_data_source`] over `0..n_rows`): either
/// part then overlaps it on exactly the rows it holds, and the two parts are independent of
/// each other.
///
/// # Errors
///
/// A receipt whose rule no longer assigns rows (as [`SampleSplitReceipt::verify`]).
pub fn split_evidence_sources(
    receipt: &SampleSplitReceipt,
    snapshot_id: &[u8],
) -> Result<(SourceDeclaration, SourceDeclaration), SelectiveDiscoveryError> {
    receipt.verify()?;
    let (discovery, analysis) = receipt.rule.assign(receipt.n_rows)?;
    Ok((
        discovery_rows_source("discovery rows of the split", snapshot_id, &discovery),
        discovery_rows_source("analysis rows of the split", snapshot_id, &analysis),
    ))
}
