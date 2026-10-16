//! Discovery followed by analysis on a row split, in the analysis facade.
//!
//! [`discover_on_split`] partitions the rows of a table by a declared [`SplitRule`], runs
//! static discovery on the discovery part only, and keeps the [`SampleSplitReceipt`] that binds
//! the analysis part to that discovery. [`SplitDiscovery::check_analysis_table`] is the gate a
//! downstream analysis passes through: it refuses a table that is not the receipt's analysis
//! rows, and returns an [`AnalysisBinding`] that a claim carries so the discovery and the
//! split it rests on are part of the claim's identity.
//!
//! # Order of operations, masks and weights
//!
//! Rows are filtered before they are split: first the analysis mask, then complete cases over
//! the searched variables; the excluded rows are recorded in the receipt and belong to neither
//! part, and the rule assigns only the rows that remain. Weights are gathered with their rows,
//! unnormalized; weighted discovery reads the discovery part's weights through
//! `weighted_parcorr`, and the analysis part's weights, original row ids and Kish effective
//! sample size are exposed for a downstream weighted estimate
//! ([`SplitDiscovery::analysis_weights`], [`SplitDiscovery::analysis_row_ids`],
//! [`SplitDiscovery::analysis_effective_sample_size`]). The derivation, the estimands and the
//! normalization contract are in `antecedent_discovery::selective::reuse`.
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

use std::sync::Arc;

use antecedent_contract::evidence_ledger::SourceDeclaration;
use antecedent_core::{CanonicalWriter, ExecutionContext, VariableId};
use antecedent_data::TabularData;
use antecedent_discovery::selective::{
    CiTestSpec, DiscoveryOutcome, SelectiveDiscoveryError, SelectiveRunReport,
    StaticDiscoveryContract, StaticSelectiveDiscovery, column_digest, digest_hex,
    request_downstream_interval,
};
use antecedent_stats::WeightedPartialCorrelation;

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

/// Test names that read the table's observation weights.
const WEIGHTED_TESTS: [&str; 2] = ["weighted_parcorr", "weighted_partial_corr"];

/// The contract discovery runs under on `part`.
///
/// Weighted discovery uses the weighted partial-correlation test on the discovery part's own
/// weights (its identity names the weights, so a retained statistic is served only to the same
/// weights). A weight that no declared test reads, or a weighted test without weights, is
/// refused: a split must not drop weights silently or invent them.
fn contract_on_part(
    contract: &StaticDiscoveryContract,
    part: &TabularData,
) -> Result<StaticDiscoveryContract, SelectiveDiscoveryError> {
    let weighted_test = matches!(
        &contract.ci_test,
        CiTestSpec::Named(name)
            if WEIGHTED_TESTS.contains(&name.trim().to_ascii_lowercase().as_str())
    );
    let refuse = |detail, remedy: &str| {
        SelectiveDiscoveryError::refused(
            antecedent_core::reason_code!("option_not_applicable"),
            detail,
            remedy,
        )
    };
    match (part.storage().weights(), weighted_test) {
        (None, false) => Ok(contract.clone()),
        (Some(weights), true) => {
            let mut weighted = contract.clone();
            weighted.ci_test = CiTestSpec::declared(
                format!("weighted_parcorr:weights={}", digest_hex(&column_digest(weights))),
                Arc::new(WeightedPartialCorrelation::new(weights.to_vec())),
            );
            Ok(weighted)
        }
        (Some(_), false) => Err(refuse(
            "selective_discovery.weights_not_read_by_test",
            "the table carries observation weights but the declared test ignores them; declare \
             `weighted_parcorr`, or split an unweighted table",
        )),
        (None, true) => Err(refuse(
            "selective_discovery.weighted_test_without_weights",
            "`weighted_parcorr` needs observation weights and the table carries none",
        )),
    }
}

/// Split `data` by `rule` and run `contract` on the discovery part.
///
/// Rows are filtered (analysis mask, then complete cases over `contract.variables`) before they
/// are split; see [`split_rows`]. The receipt records the exclusions and the content of the
/// source and of both parts over `contract.variables`. The analysis part carries every column
/// of `data`, so an analysis may use covariates discovery did not search, and the weights of
/// its rows, unnormalized ([`SplitDiscovery::analysis_weights`]).
///
/// # Errors
///
/// A non-numeric column, a rule that leaves a part empty or names a row twice, out of range or
/// excluded, a part without positive weight, weights that the declared test does not read (or a
/// weighted test without weights; all refused before discovery runs), or a refused contract or
/// an algorithm failure or cancellation in discovery.
pub fn discover_on_split(
    contract: &StaticDiscoveryContract,
    data: &TabularData,
    rule: SplitRule,
    ctx: &ExecutionContext,
) -> Result<SplitDiscovery, SelectiveDiscoveryError> {
    let split = split_rows(data, &contract.variables, rule)?;
    let contract = contract_on_part(contract, &split.discovery)?;
    let mut adapter = StaticSelectiveDiscovery::new();
    let report = adapter.run(&contract, &split.discovery, ctx)?;
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

    /// The original row index of each analysis row, in the part's row order.
    #[must_use]
    pub fn analysis_row_ids(&self) -> &[usize] {
        &self.split.analysis_row_ids
    }

    /// The original row index of each discovery row, in the part's row order.
    #[must_use]
    pub fn discovery_row_ids(&self) -> &[usize] {
        &self.split.discovery_row_ids
    }

    /// The weights of the analysis rows, unnormalized, or `None` for an unweighted source.
    #[must_use]
    pub fn analysis_weights(&self) -> Option<&[f64]> {
        self.split.analysis_weights()
    }

    /// Kish effective sample size of the analysis weights (`None` when unweighted).
    #[must_use]
    pub fn analysis_effective_sample_size(&self) -> Option<f64> {
        self.split.analysis_effective_sample_size()
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

    /// Re-derive the partition from the rule and the recorded exclusions, and check the stored
    /// identities, the parts' row ids, masks, weights and content against the receipt.
    ///
    /// # Errors
    ///
    /// As [`SplitRows::verify`].
    pub fn verify(&self) -> Result<(), SelectiveDiscoveryError> {
        self.split.verify(&self.variables)
    }

    /// Replay the split on `source` and require it to be the table the receipt describes.
    ///
    /// # Errors
    ///
    /// As [`SampleSplitReceipt::verify_against_source`].
    pub fn verify_against_source(
        &self,
        source: &TabularData,
    ) -> Result<(), SelectiveDiscoveryError> {
        self.split.receipt.verify_against_source(source, &self.variables)
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
/// The rows are the original row indices of the split table, so an excluded row belongs to
/// neither part.
///
/// # Errors
///
/// A receipt whose rule no longer assigns rows (as [`SampleSplitReceipt::verify`]).
pub fn split_evidence_sources(
    receipt: &SampleSplitReceipt,
    snapshot_id: &[u8],
) -> Result<(SourceDeclaration, SourceDeclaration), SelectiveDiscoveryError> {
    let (discovery, analysis) = receipt.assigned_rows()?;
    Ok((
        discovery_rows_source("discovery rows of the split", snapshot_id, &discovery),
        discovery_rows_source("analysis rows of the split", snapshot_id, &analysis),
    ))
}
