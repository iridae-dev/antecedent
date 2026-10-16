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
//! # Weighted estimation on the analysis part
//!
//! [`SplitDiscovery::estimate_analysis_effect`] estimates a linear treatment effect on the
//! analysis part with the part's own weights declared as sampling weights, adjusting for a
//! declared set (typically the treatment's parents in the discovered graph). Its estimand,
//! normalization and uncertainty are stated on [`SplitAnalysisEffect`]: the effect in the
//! weighted population the analysis part's weights represent, conditional on the selected
//! graph; weights are normalized to mean one over the rows used (the estimate is the same
//! under any rescaling); the standard error is the sampling-weighted sandwich (HC1, or the
//! cluster sandwich over declared analysis-part clusters); rows of weight zero are excluded and
//! counted; and the Kish effective sample size is reported beside the row count. The interval
//! it supports is a separate, unmeasured claim: no interval is published.
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
use antecedent_estimate::{
    AnalyticSeKind, EstimationError, EstimationWorkspace, LinearAdjustmentAte,
    SamplingWeightRecord, SamplingWeights,
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

/// What to estimate on the analysis part of a split.
#[derive(Clone, Debug, PartialEq)]
pub struct AnalysisEffectRequest {
    /// Treatment column.
    pub treatment: VariableId,
    /// Outcome column.
    pub outcome: VariableId,
    /// Adjustment columns; the discovered treatment parents are the declared choice.
    pub adjustment: Vec<VariableId>,
    /// Control and active treatment levels of the contrast.
    pub levels: (f64, f64),
    /// Cluster label of every analysis row, in the part's row order, for the cluster
    /// sandwich; `None` treats rows as the independent units.
    pub clusters: Option<Vec<u32>>,
}

/// A linear effect estimated on the analysis part of a split, weighted by that part's weights.
///
/// **Estimand.** The coefficient of the treatment contrast in the weighted least-squares
/// regression of the outcome on the treatment and the adjustment columns: the effect in the
/// weighted population the analysis part's weights represent, conditional on the graph that
/// chose the adjustment set (discovery saw none of these rows). Without weights it is the
/// ordinary least-squares coefficient.
///
/// **Normalization.** Estimates and standard errors depend on the weights only up to scale;
/// they are normalized to mean one over the rows used. The weight sum is reported unnormalized
/// and is never read as a population size.
///
/// **Uncertainty.** The standard error is the sampling-weighted HC1 sandwich, or the cluster
/// sandwich when clusters are declared; no interval is published (its coverage is unmeasured).
///
/// **Effective sample size.** The Kish size of the weights of the rows used is in
/// [`Self::sampling_weights`]; `None` for an unweighted source.
#[derive(Clone, Debug, PartialEq)]
pub struct SplitAnalysisEffect {
    /// The estimated effect of the declared contrast.
    pub effect: f64,
    /// Sandwich standard error of [`Self::effect`].
    pub standard_error: f64,
    /// The standard-error construction that ran.
    pub se_kind: AnalyticSeKind,
    /// What the part's weights did: rows used, zero-weight rows excluded, effective size.
    pub sampling_weights: Option<SamplingWeightRecord>,
    /// Rows the effect was estimated on.
    pub rows: u64,
    /// Original row index of each analysis row, in the part's row order (including rows that
    /// had weight zero).
    pub analysis_row_ids: Vec<usize>,
    /// The binding of this analysis to the split receipt and the discovery it followed.
    pub binding: AnalysisBinding,
}

impl SplitAnalysisEffect {
    /// The population the effect speaks for.
    pub const TARGET_POPULATION: &'static str =
        "the weighted population the analysis part's weights represent, given the selected graph";
}

impl SplitDiscovery {
    /// Estimate a linear effect on the analysis part, weighted by the part's own weights.
    ///
    /// The estimate rests on the analysis rows alone, so it is conditional on the selected
    /// graph and never reuses the rows discovery saw; the returned binding says so.
    ///
    /// # Errors
    ///
    /// A receipt that does not re-derive, weights that are invalid on the analysis part, a
    /// singular design, or fewer than two clusters when clusters are declared.
    pub fn estimate_analysis_effect(
        &self,
        request: &AnalysisEffectRequest,
        ctx: &ExecutionContext,
    ) -> Result<SplitAnalysisEffect, EstimationError> {
        let binding =
            self.check_analysis_table(self.analysis_rows()).map_err(|error| match error {
                SelectiveDiscoveryError::Refused { code, detail, remedy } => {
                    EstimationError::refused(code, format!("{detail}: {remedy}"))
                }
                other => EstimationError::data_msg(other.to_string()),
            })?;
        let (control, active) = request.levels;
        let query = antecedent_core::AverageEffectQuery::with_levels(
            request.treatment,
            request.outcome,
            control,
            active,
        );
        let mut arena = antecedent_expr::CausalExprArena::new();
        let functional = arena.backdoor_ate(
            request.treatment,
            request.outcome,
            &request.adjustment,
            antecedent_core::Value::f64(active),
            antecedent_core::Value::f64(control),
        );
        let estimand = antecedent_expr::IdentifiedEstimand::backdoor(
            "backdoor.adjustment",
            Arc::from(request.adjustment.clone()),
            functional,
        );
        let mut estimator = LinearAdjustmentAte::new()
            .with_bootstrap_replicates(0)
            .with_se_kind(AnalyticSeKind::Hc1);
        if let Some(weights) = self.analysis_weights() {
            estimator = estimator.with_sampling_weights(SamplingWeights::new(weights.to_vec())?);
        }
        if let Some(clusters) = &request.clusters {
            estimator = estimator
                .with_se_kind(AnalyticSeKind::Cluster)
                .with_source_cluster_ids(clusters.clone());
        }
        let prepared = estimator.prepare(self.analysis_rows(), &estimand, &query)?;
        let fit = estimator.fit(
            &prepared,
            &mut EstimationWorkspace::default(),
            ctx,
            antecedent_core::AssumptionSet::new(),
        )?;
        Ok(SplitAnalysisEffect {
            effect: fit.ate,
            standard_error: fit.se_analytic(),
            se_kind: fit.uncertainty.analytic.kind.unwrap_or(AnalyticSeKind::Hc1),
            sampling_weights: fit.diagnostics.sampling_weights,
            rows: fit.provenance.n_obs.unwrap_or(0),
            analysis_row_ids: self.analysis_row_ids().to_vec(),
            binding,
        })
    }
}
