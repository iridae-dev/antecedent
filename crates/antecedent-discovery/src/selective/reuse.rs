//! Discovery followed by analysis: how the downstream error depends on how the discovery rows
//! are used, and the one bounded route that keeps the dependence simple, a row split.
//!
//! # Derivation
//!
//! Let `D` be the rows, `Ĝ = S(D_disc)` the graph that discovery on rows `D_disc` and the
//! declared acceptance rule select, `A(D_ana; g)` the analysis (an adjustment set, an
//! estimator and an interval procedure) that treats `g` as the causal graph, and `θ` the
//! estimand of the analysis question under the true graph `G`. Let `𝒢*(θ)` be the set of
//! graphs under which `A(·; g)` targets `θ` (for example, graphs that give the same valid
//! adjustment set as `G` for this treatment and outcome). The reported interval is
//! `[L, U] = A(D_ana; Ĝ)` at nominal level `1 - α`.
//!
//! **Split (disjoint rows).** Rows are exchangeable and `D_ana` is independent of `D_disc`,
//! so conditional on `Ĝ = g` the analysis rows still follow the sampling law. Then
//!
//! ```text
//! P(θ ∉ [L, U] | Ĝ = g) = α_n(g)                    if g ∈ 𝒢*(θ)
//!                        = m(g)                      otherwise (bias of an unlicensed g)
//! P(θ ∉ [L, U])        ≤ α_n + P(Ĝ ∉ 𝒢*(θ))         (union bound over the selection event)
//! ```
//!
//! where `α_n` is the analysis procedure's own finite-sample miscoverage at the analysis row
//! count `n_ana`. The first term is what an interval states. The second is the probability that
//! discovery did not return a graph that licenses the question: it is not controlled by `α`,
//! it depends on the sample size, the faithfulness of the data-generating process and the CI
//! test, and no interval procedure reports it. A split therefore licenses exactly one
//! statement: an interval for the effect *given the selected graph*. It does not license "the
//! interval covers the true effect with probability `1 - α`" without a measured bound on
//! `P(Ĝ ∉ 𝒢*(θ))` for this method at this coordinate.
//!
//! **Reuse (the same rows select and analyse).** `Ĝ` is a function of `D`, so conditional
//! on `Ĝ = g` the rows follow the sampling law *restricted to the selection event
//! `{S(D) = g}`*. The analysis' own interval is calibrated for the unrestricted law and is
//! generally anti-conservative after selection: the data that made an edge significant also
//! make the downstream coefficient look stable. A correction exists only where the selection
//! event has a tractable truncated law (a closed polyhedral selection, for example); the
//! event "PC with these conditional-independence tests returned `g`" is not of that kind and
//! no such correction is implemented. An interval under reuse is therefore refused (see
//! [`super::claims::request_downstream_interval`]), and the selective-inference-free path is
//! the split below.
//!
//! **Cost of the split.** With discovery fraction `f`, `n_disc = f n` and `n_ana = (1 - f) n`.
//! Discovery error `P(Ĝ ∉ 𝒢*)` rises as `f` falls, and the analysis interval width scales as
//! `1 / sqrt(1 - f)`. The split trades a measurable discovery term for an interval that needs
//! no selection adjustment. There is no free fraction; `f` is declared and enters the
//! identity.
//!
//! # What the route is
//!
//! [`split_rows`] partitions the rows of a static table into a discovery table and an
//! analysis table by a declared rule (an explicit row assignment, or a seeded permutation
//! with a declared fraction). The discovery table is the only input discovery sees; the
//! analysis table is the only input the downstream analysis may use. The
//! [`SampleSplitReceipt`] records the rule, the rows excluded before the split, the row-set
//! identities and the content identities of the source and of both tables so a consumer can
//! re-derive the partition from the rule and the exclusions instead of trusting a flag
//! ([`SampleSplitReceipt::verify`], [`SampleSplitReceipt::verify_against_source`],
//! [`SplitRows::verify`]).
//!
//! # Order of operations
//!
//! 1. **Filter by the analysis mask.** A row outside the table's analysis mask is excluded.
//! 2. **Filter to complete cases** over the declared variables (the searched variables):
//!    a row of the mask with a missing cell in any of them is excluded. Covariates the
//!    declared variables do not include keep their missing cells in the analysis part, where
//!    the analysis' own complete-case rule applies on top.
//! 3. **Record the exclusions.** The excluded rows, by original index and by reason, are in
//!    the receipt and in neither part.
//! 4. **Split the eligible rows.** A seeded permutation permutes only the eligible rows and
//!    takes `round(fraction * n_eligible)` of them for discovery; an explicit rule names
//!    original row indices and refuses one that was excluded. The rule never sees an excluded
//!    row, so which rows are excluded cannot move the assignment of the others.
//!
//! The order matters for the claim: filtering after the split would let a row's completeness
//! decide its part, and a mask applied inside each part would make the part sizes depend on
//! the data the rule is meant to be independent of. Original row identities are kept: each
//! part's `*_row_ids` and [`SampleSplitReceipt::assigned_rows`] give the source row of every
//! part row.
//!
//! # Weights
//!
//! Observation weights are carried by gather: each part holds, bit for bit, the weights of its
//! rows in the source. The split never normalizes them, so a part's weights do not sum to one
//! and do not sum to the source total. Zero weights are legitimate and stay in their part (they
//! contribute nothing to a weighted statistic and nothing to the effective sample size); a part
//! whose weights have no positive mass is refused.
//!
//! * **Estimand.** The weights define the population the rows are meant to represent. A
//!   weighted statistic on a part estimates the weighted-population quantity, for example the
//!   weighted partial correlation that [`antecedent_stats::WeightedPartialCorrelation`] uses in
//!   discovery, or a weighted mean in the analysis. Unit weights give the unweighted target.
//!   The split is exchangeable in the weighted sense only when the weights are fixed design
//!   weights (functions of the sampling design, not of the outcomes), which the caller asserts.
//! * **Normalization is the consumer's.** What the weights mean (inverse selection
//!   probability, survey expansion factors, precision weights) decides whether to divide by
//!   `Σw` over the analysis part (a weighted mean) or to keep the totals (a population total).
//!   The receipt supplies the pieces for either: [`SplitRows::analysis_weights`] unscaled and
//!   the source digest to show they were not altered.
//! * **Effective sample size.** An interval for a weighted analysis must use the Kish effective
//!   sample size `(Σw)² / Σw²` of the analysis weights, not the row count, because unequal
//!   weights lower the information per row. It is recorded per part as
//!   [`SplitWeights::analysis_effective_sample_size`] (computed by
//!   `antecedent_kernels::kish_ess`, which yields `0` for no mass and `NaN` for a non-finite
//!   weight). Downstream weighted estimation on the analysis part is the sampling-weights
//!   work's to wire; this module exposes the weights, the row ids and the effective size for it.
//!
//! # What the route is not
//!
//! * **Not a coverage claim.** The route removes a dependence; it does not measure error.
//!   No interval produced downstream of it is labelled calibrated for the whole method. The
//!   calibration design is recorded below, to be measured, and the calibration lookup for any
//!   such interval stays unmeasured until a record exists for this coordinate.
//! * **Not for dependent rows.** A random row partition of temporal, panel, clustered or
//!   otherwise dependent rows leaves dependence between the parts, so the independence the
//!   derivation uses does not hold. The route is for rows declared exchangeable; a
//!   contiguous temporal window with a gap is a different design that this route does not
//!   cover.
//!
//! # Whole-method calibration design (recorded, not measured)
//!
//! * **Claim.** For the declared discovery contract, acceptance rule, estimator and interval
//!   procedure at the split fraction `f`: the probability that the interval for the effect
//!   covers the true effect, jointly over the random split, discovery and analysis.
//! * **Truth.** Linear-Gaussian SCMs with known graphs and known total effects, spanning
//!   variable counts, edge density and effect sizes, with weakly faithful (near-cancelling)
//!   members and a misspecified-test member; a nonlinear-additive member for the
//!   conditional-independence test the contract declares.
//! * **Nominal level.** The analysis interval level, `0.95` and `0.90`.
//! * **Sampling model.** Independent rows; `n` on a grid that brackets the supported range;
//!   `f` in `{0.3, 0.5, 0.7}`; at least 2000 replicates per cell with Monte Carlo error bounds
//!   reported.
//! * **Measured quantities.** `P(Ĝ ∈ 𝒢*)` (adequate-graph probability), conditional
//!   coverage given `Ĝ ∈ 𝒢*`, conditional coverage given `Ĝ ∉ 𝒢*`, and unconditional
//!   coverage, each with bounds, so the decomposition above is checked rather than assumed.
//! * **Negative control.** The same procedure with reuse (no split) at the same `n`, to
//!   confirm the anti-conservative direction the derivation predicts; a negative control that
//!   does not undercover somewhere in the grid is a defect to investigate, not a result.
//! * **Boundary probes.** `n_ana` at the smallest size the estimator supports, a graph whose
//!   true parents are weakly detectable at `n_disc`, and a class whose completion is not
//!   unique.
//! * **Support range.** Coordinates outside the measured grid are outside the claim.
//!
//! # Refresh with outcome-only data
//!
//! A downstream claim is bound to the split receipt it was made under, and
//! [`SampleSplitReceipt::check_analysis_table`] refuses an analysis table that is not the
//! receipt's analysis rows. New outcome-only rows are therefore a new split: the receipt is
//! recomputed with the declared fraction and seed, and the discovered graph and the
//! downstream estimate are recomputed against it. Nothing is patched in place. Rows that
//! both select and analyse are the reuse case and refuse an interval; a refresh never
//! converts reuse into a split.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{CanonicalWriter, VariableId};
use antecedent_data::{
    DataError, Float64Column, OwnedColumn, OwnedColumnarStorage, TableView, TabularData,
    ValidityBitmap,
};

use super::identity::{
    DataIdentity, Digest, SELECTIVE_IDENTITY_VERSION, column_digest, static_data_identity,
};
use super::refusal::SelectiveDiscoveryError;
use crate::error::DiscoveryError;

const SPLIT_TAG: &str = "antecedent.selective_discovery.sample_split.v1";
const ROWS_TAG: &str = "antecedent.selective_discovery.sample_split.rows.v1";
const MASK_TAG: &str = "antecedent.selective_discovery.sample_split.mask.v1";

fn not_supported(detail: &'static str, remedy: impl Into<String>) -> SelectiveDiscoveryError {
    SelectiveDiscoveryError::refused(
        antecedent_core::reason_code!("route_not_supported"),
        detail,
        remedy,
    )
}

fn invalid(detail: &'static str, remedy: impl Into<String>) -> SelectiveDiscoveryError {
    SelectiveDiscoveryError::refused(
        antecedent_core::reason_code!("invalid_argument"),
        detail,
        remedy,
    )
}

/// How rows are assigned to the discovery and analysis parts.
#[derive(Clone, Debug, PartialEq)]
pub enum SplitRule {
    /// The caller names the discovery rows by their index in the table that is split; every
    /// other eligible row is an analysis row.
    Explicit {
        /// Rows used for discovery, any order; repeats, out-of-range rows and rows excluded
        /// before the split are refused.
        discovery_rows: Vec<usize>,
    },
    /// A seeded permutation of the eligible rows; the first `round(fraction * n_eligible)` of
    /// the permutation are discovery rows. Deterministic and independent of platform.
    SeededPermutation {
        /// Permutation seed.
        seed: u64,
        /// Share of the eligible rows used for discovery, in `(0, 1)`.
        discovery_fraction: f64,
    },
}

impl SplitRule {
    /// The discovery rows (ascending) and analysis rows (ascending) for `n_rows` rows that are
    /// all eligible.
    ///
    /// # Errors
    ///
    /// As [`Self::assign_eligible`].
    pub fn assign(
        &self,
        n_rows: usize,
    ) -> Result<(Vec<usize>, Vec<usize>), SelectiveDiscoveryError> {
        let all: Vec<usize> = (0..n_rows).collect();
        self.assign_eligible(n_rows, &all)
    }

    /// The discovery rows (ascending) and analysis rows (ascending) over the `eligible` rows of
    /// a table with `n_rows` rows. Both are original row indices of that table.
    ///
    /// `eligible` is strictly ascending and below `n_rows`: the rows left once the analysis
    /// mask and incomplete cases were filtered out. A seeded permutation permutes only those
    /// rows, so excluded rows never consume a draw.
    ///
    /// # Errors
    ///
    /// A row out of range, repeated or not eligible, a fraction outside `(0, 1)`, or a part
    /// with no rows.
    pub fn assign_eligible(
        &self,
        n_rows: usize,
        eligible: &[usize],
    ) -> Result<(Vec<usize>, Vec<usize>), SelectiveDiscoveryError> {
        let mut discovery: Vec<usize> = match self {
            Self::Explicit { discovery_rows } => {
                let mut rows = discovery_rows.clone();
                rows.sort_unstable();
                if rows.windows(2).any(|w| w[0] == w[1]) {
                    return Err(invalid(
                        "selective_discovery.split_rows_repeated",
                        "a row is named twice in the discovery rows",
                    ));
                }
                if rows.last().is_some_and(|&r| r >= n_rows) {
                    return Err(invalid(
                        "selective_discovery.split_row_out_of_range",
                        format!("a discovery row is not below the row count {n_rows}"),
                    ));
                }
                if let Some(row) = rows.iter().find(|r| eligible.binary_search(r).is_err()) {
                    return Err(invalid(
                        "selective_discovery.split_row_excluded",
                        format!(
                            "discovery row {row} was excluded before the split (outside the \
                             analysis mask, or incomplete in a searched variable); name only \
                             eligible rows"
                        ),
                    ));
                }
                rows
            }
            Self::SeededPermutation { seed, discovery_fraction } => {
                if !(*discovery_fraction > 0.0 && *discovery_fraction < 1.0) {
                    return Err(invalid(
                        "selective_discovery.split_fraction_invalid",
                        "the discovery fraction must lie strictly between 0 and 1",
                    ));
                }
                let m = eligible.len();
                let mut order: Vec<usize> = (0..m).collect();
                let mut state = *seed;
                for i in (1..m).rev() {
                    let j = usize::try_from(splitmix64(&mut state) % (i as u64 + 1)).unwrap_or(i);
                    order.swap(i, j);
                }
                // The rounded product is a whole number in [0, m]; count the
                // integers up to it instead of casting the float.
                let scaled = ((m as f64) * discovery_fraction).round();
                let take = (0..=m).take_while(|&t| (t as f64) <= scaled).count().saturating_sub(1);
                let take = take.clamp(1, m.saturating_sub(1).max(1));
                let mut rows: Vec<usize> =
                    order[..take.min(m)].iter().map(|&position| eligible[position]).collect();
                rows.sort_unstable();
                rows
            }
        };
        discovery.dedup();
        let in_discovery: std::collections::BTreeSet<usize> = discovery.iter().copied().collect();
        let analysis: Vec<usize> =
            eligible.iter().copied().filter(|r| !in_discovery.contains(r)).collect();
        if discovery.is_empty() || analysis.is_empty() {
            return Err(invalid(
                "selective_discovery.split_part_empty",
                "both the discovery rows and the analysis rows must be non-empty",
            ));
        }
        Ok((discovery, analysis))
    }

    fn write(&self, writer: &mut CanonicalWriter) {
        match self {
            Self::Explicit { discovery_rows } => {
                let mut rows = discovery_rows.clone();
                rows.sort_unstable();
                writer.word(0).word(rows.len() as u64);
                for row in rows {
                    writer.word(row as u64);
                }
            }
            Self::SeededPermutation { seed, discovery_fraction } => {
                writer.word(1).word(*seed).real(*discovery_fraction);
            }
        }
    }
}

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn rows_identity(tag: &str, rows: &[usize]) -> Digest {
    let mut writer = CanonicalWriter::new(tag);
    writer.word(SELECTIVE_IDENTITY_VERSION).word(rows.len() as u64);
    for &row in rows {
        writer.word(row as u64);
    }
    writer.finish()
}

/// Identity of an analysis mask: the inclusion bit of every row.
fn mask_identity(mask: &ValidityBitmap) -> Digest {
    let bits: Vec<u8> = (0..mask.len()).map(|i| u8::from(mask.is_valid(i))).collect();
    CanonicalWriter::digest_bytes(MASK_TAG, &bits)
}

/// The observation weights of a split table, as the receipt records them.
///
/// Weights are carried by gather and never rescaled, so a part's weights sum to whatever its
/// rows held in the source. See the module documentation for normalization.
#[derive(Clone, Debug, PartialEq)]
pub struct SplitWeights {
    /// Identity of the source table's weights over all of its rows.
    pub source: Digest,
    /// Identity of the weights gathered into the discovery part.
    pub discovery: Digest,
    /// Identity of the weights gathered into the analysis part.
    pub analysis: Digest,
    /// Kish effective sample size `(Σw)² / Σw²` of the discovery part's weights.
    pub discovery_effective_sample_size: f64,
    /// Kish effective sample size of the analysis part's weights.
    pub analysis_effective_sample_size: f64,
}

/// Record of one row split: enough to re-derive the partition and to check a table against it.
#[derive(Clone, Debug, PartialEq)]
pub struct SampleSplitReceipt {
    /// The declared rule.
    pub rule: SplitRule,
    /// Row count of the table that was split.
    pub n_rows: usize,
    /// Rows the analysis mask excluded before the split (original indices, ascending).
    pub masked_out_rows: Vec<usize>,
    /// Rows the mask kept but that were incomplete in a declared variable, excluded before the
    /// split (original indices, ascending).
    pub incomplete_rows: Vec<usize>,
    /// Rows in the discovery part.
    pub n_discovery: usize,
    /// Rows in the analysis part.
    pub n_analysis: usize,
    /// Identity of the discovery row set (original indices).
    pub discovery_rows_identity: Digest,
    /// Identity of the analysis row set (original indices).
    pub analysis_rows_identity: Digest,
    /// Identity of the source table's analysis mask, when it had one.
    pub mask_identity: Option<Digest>,
    /// The source table's weights and the parts', when it had weights.
    pub weights: Option<SplitWeights>,
    /// Content identity of the source table over the declared variables (mask and weights
    /// included).
    pub source_data: DataIdentity,
    /// Content identity of the discovery part over the declared variables.
    pub discovery_data: DataIdentity,
    /// Content identity of the analysis part over the declared variables.
    pub analysis_data: DataIdentity,
    /// Identity of the whole receipt.
    pub identity: Digest,
}

/// Whether `rows` is strictly ascending and every entry is below `n_rows`.
fn ascending_below(rows: &[usize], n_rows: usize) -> bool {
    rows.windows(2).all(|w| w[0] < w[1]) && rows.last().is_none_or(|&r| r < n_rows)
}

impl SampleSplitReceipt {
    /// The rows that survived filtering, ascending: every row but the recorded exclusions.
    fn eligible_rows(&self) -> Result<Vec<usize>, SelectiveDiscoveryError> {
        let consistent = ascending_below(&self.masked_out_rows, self.n_rows)
            && ascending_below(&self.incomplete_rows, self.n_rows)
            && (self.mask_identity.is_some() || self.masked_out_rows.is_empty())
            && self.masked_out_rows.iter().all(|r| self.incomplete_rows.binary_search(r).is_err());
        if !consistent {
            return Err(invalid(
                "selective_discovery.split_receipt_mismatch",
                "the receipt's excluded rows are not valid for its row count and mask",
            ));
        }
        Ok((0..self.n_rows)
            .filter(|r| {
                self.masked_out_rows.binary_search(r).is_err()
                    && self.incomplete_rows.binary_search(r).is_err()
            })
            .collect())
    }

    /// The rows each part holds, derived from the rule and the recorded exclusions and checked
    /// against the receipt: `(discovery, analysis)` as ascending original row indices. This is
    /// the row identity a downstream analysis keys per-row quantities on (weights, outcomes).
    ///
    /// # Errors
    ///
    /// As [`Self::verify`].
    pub fn assigned_rows(&self) -> Result<(Vec<usize>, Vec<usize>), SelectiveDiscoveryError> {
        self.verify_rows()
    }

    fn verify_rows(&self) -> Result<(Vec<usize>, Vec<usize>), SelectiveDiscoveryError> {
        let eligible = self.eligible_rows()?;
        let (discovery, analysis) = self.rule.assign_eligible(self.n_rows, &eligible)?;
        let overlap = discovery.iter().any(|row| analysis.binary_search(row).is_ok());
        if overlap || discovery.len() + analysis.len() != eligible.len() {
            return Err(invalid(
                "selective_discovery.split_not_disjoint",
                "the discovery and analysis rows overlap or do not cover the eligible rows",
            ));
        }
        if discovery.len() != self.n_discovery
            || analysis.len() != self.n_analysis
            || rows_identity(ROWS_TAG, &discovery) != self.discovery_rows_identity
            || rows_identity(ROWS_TAG, &analysis) != self.analysis_rows_identity
        {
            return Err(invalid(
                "selective_discovery.split_receipt_mismatch",
                "the receipt's row sets are not the ones its rule assigns over its recorded \
                 exclusions; re-split from the declared rule",
            ));
        }
        if self.identity != receipt_identity(self) {
            return Err(invalid(
                "selective_discovery.split_receipt_mismatch",
                "the receipt's identity does not cover its recorded contents",
            ));
        }
        Ok((discovery, analysis))
    }

    /// Re-derive the partition from the rule, the row count and the recorded exclusions and
    /// check it against the receipt.
    ///
    /// Does not trust the stored sizes or row-set identities: it recomputes the assignment over
    /// the rows the exclusions leave, requires the two parts to be disjoint and to cover them,
    /// and requires the stored identities (rows, exclusions, mask, weights, source and part
    /// contents) to be the ones the receipt identity covers. It cannot see the tables; see
    /// [`Self::verify_against_source`] and [`SplitRows::verify`] for those.
    ///
    /// # Errors
    ///
    /// A rule that no longer assigns rows, parts that overlap or do not cover the eligible
    /// rows, exclusions that are not valid for the row count, or stored identities that differ
    /// from the re-derived ones.
    pub fn verify(&self) -> Result<(), SelectiveDiscoveryError> {
        self.verify_rows().map(|_| ())
    }

    /// Replay the split on `source` and require it to be the table this receipt describes.
    ///
    /// A source whose analysis mask or weights differ from the recorded ones is refused by
    /// name before the split is replayed; any other difference (a changed value, an added
    /// missing cell, a different row count) shows as a replayed receipt that differs from this
    /// one.
    ///
    /// # Errors
    ///
    /// As [`Self::verify`], plus `selective_discovery.split_mask_mismatch`,
    /// `selective_discovery.split_weights_mismatch` and
    /// `selective_discovery.split_source_mismatch`, or the refusal the replayed split raises.
    pub fn verify_against_source(
        &self,
        source: &TabularData,
        variables: &[VariableId],
    ) -> Result<(), SelectiveDiscoveryError> {
        self.verify()?;
        let storage = source.storage();
        if storage.analysis_mask().map(mask_identity) != self.mask_identity {
            return Err(invalid(
                "selective_discovery.split_mask_mismatch",
                "the source table's analysis mask is not the mask this split filtered by",
            ));
        }
        if storage.weights().map(column_digest) != self.weights.as_ref().map(|w| w.source) {
            return Err(invalid(
                "selective_discovery.split_weights_mismatch",
                "the source table's weights are not the weights this split carried",
            ));
        }
        let replay = split_rows(source, variables, self.rule.clone())?;
        if replay.receipt == *self {
            Ok(())
        } else {
            Err(invalid(
                "selective_discovery.split_source_mismatch",
                "replaying the rule on the source table does not reproduce the receipt; the \
                 source is not the table that was split",
            ))
        }
    }

    /// Check that `table` is exactly the analysis part this receipt describes, over
    /// `variables`.
    ///
    /// The comparison covers the searched columns, the analysis mask and the weights: an
    /// analysis table with re-weighted rows is another table.
    ///
    /// # Errors
    ///
    /// A table whose searched columns, mask or weights differ in content or row count from the
    /// recorded analysis part, which is the signature of an analysis on other rows (for example
    /// the discovery rows or the unsplit table).
    pub fn check_analysis_table(
        &self,
        table: &TabularData,
        variables: &[VariableId],
    ) -> Result<(), SelectiveDiscoveryError> {
        self.verify()?;
        let identity =
            static_data_identity(table, variables).map_err(SelectiveDiscoveryError::from)?;
        if identity.digest == self.analysis_data.digest {
            Ok(())
        } else {
            Err(invalid(
                "selective_discovery.analysis_rows_not_the_split_part",
                "the analysis table is not the analysis part of this split; an analysis on the \
                 discovery rows or on the unsplit table reuses the rows discovery saw and has \
                 no interval on account of discovery",
            ))
        }
    }
}

fn receipt_identity(receipt: &SampleSplitReceipt) -> Digest {
    let mut writer = CanonicalWriter::new(SPLIT_TAG);
    writer.word(SELECTIVE_IDENTITY_VERSION);
    receipt.rule.write(&mut writer);
    writer
        .word(receipt.n_rows as u64)
        .word(receipt.n_discovery as u64)
        .word(receipt.n_analysis as u64)
        .bytes(&receipt.discovery_rows_identity)
        .bytes(&receipt.analysis_rows_identity)
        .bytes(&receipt.discovery_data.digest)
        .bytes(&receipt.analysis_data.digest);
    // Exclusions, mask and weights enter only when present, so a split of a complete,
    // unmasked, unweighted table keeps the identity it always had.
    let filtered = !receipt.masked_out_rows.is_empty()
        || !receipt.incomplete_rows.is_empty()
        || receipt.mask_identity.is_some()
        || receipt.weights.is_some();
    if filtered {
        writer.word(1).bytes(&receipt.source_data.digest);
        for excluded in [&receipt.masked_out_rows, &receipt.incomplete_rows] {
            writer.word(excluded.len() as u64);
            for &row in excluded {
                writer.word(row as u64);
            }
        }
        match &receipt.mask_identity {
            Some(mask) => {
                writer.word(1).bytes(mask);
            }
            None => {
                writer.word(0);
            }
        }
        match &receipt.weights {
            Some(weights) => {
                writer
                    .word(1)
                    .bytes(&weights.source)
                    .bytes(&weights.discovery)
                    .bytes(&weights.analysis)
                    .real(weights.discovery_effective_sample_size)
                    .real(weights.analysis_effective_sample_size);
            }
            None => {
                writer.word(0);
            }
        }
    }
    writer.finish()
}

/// The two parts of a split table.
#[derive(Clone, Debug)]
pub struct SplitRows {
    /// The only rows discovery may use.
    pub discovery: TabularData,
    /// The only rows the downstream analysis may use.
    pub analysis: TabularData,
    /// The original row index of each discovery row, in the part's row order (ascending).
    pub discovery_row_ids: Vec<usize>,
    /// The original row index of each analysis row, in the part's row order (ascending).
    pub analysis_row_ids: Vec<usize>,
    /// The record of the split.
    pub receipt: SampleSplitReceipt,
}

impl SplitRows {
    /// The weights of the discovery rows (`None` for an unweighted source), unnormalized.
    #[must_use]
    pub fn discovery_weights(&self) -> Option<&[f64]> {
        self.discovery.storage().weights()
    }

    /// The weights of the analysis rows (`None` for an unweighted source), as the source held
    /// them for those rows: unnormalized. Normalization belongs to the consumer.
    #[must_use]
    pub fn analysis_weights(&self) -> Option<&[f64]> {
        self.analysis.storage().weights()
    }

    /// Kish effective sample size of the analysis weights, `(Σw)² / Σw²`
    /// (`antecedent_kernels::kish_ess`), or `None` for an unweighted source. A weighted
    /// analysis has this many equally weighted observations' worth of information when the
    /// weights are unrelated to the outcome; the row count is `receipt.n_analysis`.
    #[must_use]
    pub fn analysis_effective_sample_size(&self) -> Option<f64> {
        self.analysis_weights().map(antecedent_kernels::kish_ess)
    }

    /// Check the parts against the receipt: the receipt itself, the row ids of both parts, the
    /// analysis mask and the weights each part holds, and the content of both parts over
    /// `variables`.
    ///
    /// # Errors
    ///
    /// As [`SampleSplitReceipt::verify`], plus `selective_discovery.split_rows_mismatch` for row
    /// ids that are not the ones the receipt assigns, `selective_discovery.split_mask_mismatch`
    /// and `selective_discovery.split_weights_mismatch` for a part whose mask or weights are not
    /// the recorded ones, and `selective_discovery.split_receipt_mismatch` for part content
    /// that differs from the recorded identity.
    pub fn verify(&self, variables: &[VariableId]) -> Result<(), SelectiveDiscoveryError> {
        let (discovery, analysis) = self.receipt.verify_rows()?;
        if discovery != self.discovery_row_ids || analysis != self.analysis_row_ids {
            return Err(invalid(
                "selective_discovery.split_rows_mismatch",
                "the parts' original row ids are not the rows the receipt assigns",
            ));
        }
        let recorded = self.receipt.weights.as_ref();
        for (part, rows, digest) in [
            (&self.discovery, discovery.len(), recorded.map(|w| w.discovery)),
            (&self.analysis, analysis.len(), recorded.map(|w| w.analysis)),
        ] {
            let storage = part.storage();
            if part.row_count() != rows {
                return Err(invalid(
                    "selective_discovery.split_rows_mismatch",
                    "a part does not hold as many rows as the receipt assigns it",
                ));
            }
            if storage.analysis_mask().is_some() != self.receipt.mask_identity.is_some()
                || storage.analysis_mask().is_some_and(|m| (0..m.len()).any(|i| !m.is_valid(i)))
            {
                return Err(invalid(
                    "selective_discovery.split_mask_mismatch",
                    "a part's analysis mask is not the all-included mask of rows that passed \
                     the recorded mask",
                ));
            }
            if storage.weights().map(column_digest) != digest {
                return Err(invalid(
                    "selective_discovery.split_weights_mismatch",
                    "a part's weights are not the weights the receipt recorded for its rows",
                ));
            }
        }
        for (part, recorded) in [
            (&self.discovery, &self.receipt.discovery_data),
            (&self.analysis, &self.receipt.analysis_data),
        ] {
            let identity =
                static_data_identity(part, variables).map_err(SelectiveDiscoveryError::from)?;
            if identity.digest != recorded.digest {
                return Err(invalid(
                    "selective_discovery.split_receipt_mismatch",
                    "a part's content is not the content the receipt recorded",
                ));
            }
        }
        Ok(())
    }
}

/// Which rows survive filtering, and why the others do not.
struct Eligibility {
    eligible: Vec<usize>,
    masked_out: Vec<usize>,
    incomplete: Vec<usize>,
}

/// Filter first by the analysis mask, then to complete cases over `variables`.
fn eligibility(
    data: &TabularData,
    variables: &[VariableId],
) -> Result<Eligibility, SelectiveDiscoveryError> {
    let validities = variables
        .iter()
        .map(|&v| data.column(v).map(antecedent_data::ColumnView::validity))
        .collect::<Result<Vec<_>, DataError>>()
        .map_err(|error| SelectiveDiscoveryError::from(DiscoveryError::from(error)))?;
    let mask = data.storage().analysis_mask();
    let mut out =
        Eligibility { eligible: Vec::new(), masked_out: Vec::new(), incomplete: Vec::new() };
    for row in 0..data.row_count() {
        if mask.is_some_and(|m| !m.is_valid(row)) {
            out.masked_out.push(row);
        } else if validities.iter().any(|v| !v.is_valid(row)) {
            out.incomplete.push(row);
        } else {
            out.eligible.push(row);
        }
    }
    Ok(out)
}

/// Partition the rows of `data` into discovery and analysis parts.
///
/// Rows are filtered before they are split: first the analysis mask, then complete cases over
/// `variables`; the excluded rows are recorded in the receipt and belong to neither part. The
/// rule then assigns the eligible rows. Every schema variable must be a float64-valued column;
/// the parts carry all of them, so the analysis may use covariates discovery did not search,
/// with their missing cells intact. Weights are gathered with their rows, unnormalized, and
/// each part carries the (all-included) analysis mask when the source had one. `variables`
/// names the columns whose completeness and content identity the receipt records (normally the
/// searched variables).
///
/// # Errors
///
/// A rule that does not assign both parts non-empty rows, names a row twice, out of range or
/// excluded (`selective_discovery.split_row_excluded`), a part without positive weight
/// (`selective_discovery.split_part_weight_invalid`), a variable that is not numeric
/// (`selective_discovery.split_unsupported_rows`), or a variable that is not in the schema.
pub fn split_rows(
    data: &TabularData,
    variables: &[VariableId],
    rule: SplitRule,
) -> Result<SplitRows, SelectiveDiscoveryError> {
    let storage = data.storage();
    let n_rows = data.row_count();
    let Eligibility { eligible, masked_out, incomplete } = eligibility(data, variables)?;
    let (discovery_rows, analysis_rows) = rule.assign_eligible(n_rows, &eligible)?;

    let schema = storage.schema().clone();
    let mut columns: Vec<(VariableId, Vec<f64>)> = Vec::with_capacity(schema.len());
    for spec in schema.variables() {
        let values = data.float64_values(spec.id).map_err(|_| {
            not_supported(
                "selective_discovery.split_unsupported_rows",
                format!("variable `{}` is not a float64 column; split a numeric table", spec.name),
            )
        })?;
        columns.push((spec.id, values));
    }
    let part = |rows: &[usize]| -> Result<TabularData, SelectiveDiscoveryError> {
        let gathered = columns
            .iter()
            .map(|(id, values)| {
                let values: Arc<[f64]> = rows.iter().map(|&r| values[r]).collect();
                Float64Column::new(*id, values, ValidityBitmap::all_valid(rows.len()))
                    .map(OwnedColumn::Float64)
            })
            .collect::<Result<Vec<_>, DataError>>()
            .map_err(|error| {
                not_supported("selective_discovery.split_unsupported_rows", error.to_string())
            })?;
        let mask = storage.analysis_mask().map(|_| ValidityBitmap::all_valid(rows.len()));
        let weights: Option<Arc<[f64]>> =
            storage.weights().map(|w| rows.iter().map(|&r| w[r]).collect());
        OwnedColumnarStorage::try_new(schema.clone(), gathered, mask, weights)
            .map(TabularData::new)
            .map_err(|error| match error {
                DataError::InvalidWeights { .. } => invalid(
                    "selective_discovery.split_part_weight_invalid",
                    format!(
                        "a part has no positive weight ({error}); give the rule rows that carry \
                         weight, or drop the zero-weight rows before splitting"
                    ),
                ),
                other => {
                    not_supported("selective_discovery.split_unsupported_rows", other.to_string())
                }
            })
    };
    let discovery = part(&discovery_rows)?;
    let analysis = part(&analysis_rows)?;

    let identity_of = |table: &TabularData| {
        static_data_identity(table, variables).map_err(SelectiveDiscoveryError::from)
    };
    let weights =
        match (storage.weights(), discovery.storage().weights(), analysis.storage().weights()) {
            (Some(source), Some(d), Some(a)) => Some(SplitWeights {
                source: column_digest(source),
                discovery: column_digest(d),
                analysis: column_digest(a),
                discovery_effective_sample_size: antecedent_kernels::kish_ess(d),
                analysis_effective_sample_size: antecedent_kernels::kish_ess(a),
            }),
            _ => None,
        };
    let mut receipt = SampleSplitReceipt {
        rule,
        n_rows,
        masked_out_rows: masked_out,
        incomplete_rows: incomplete,
        n_discovery: discovery_rows.len(),
        n_analysis: analysis_rows.len(),
        discovery_rows_identity: rows_identity(ROWS_TAG, &discovery_rows),
        analysis_rows_identity: rows_identity(ROWS_TAG, &analysis_rows),
        mask_identity: storage.analysis_mask().map(mask_identity),
        weights,
        source_data: identity_of(data)?,
        discovery_data: identity_of(&discovery)?,
        analysis_data: identity_of(&analysis)?,
        identity: [0; 32],
    };
    receipt.identity = receipt_identity(&receipt);
    Ok(SplitRows {
        discovery,
        analysis,
        discovery_row_ids: discovery_rows,
        analysis_row_ids: analysis_rows,
        receipt,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_rule_partitions_the_rows_exactly() {
        let rule = SplitRule::Explicit { discovery_rows: vec![4, 0, 2] };
        let (discovery, analysis) = rule.assign(6).unwrap();
        assert_eq!(discovery, vec![0, 2, 4]);
        assert_eq!(analysis, vec![1, 3, 5]);
    }

    #[test]
    fn seeded_permutation_is_deterministic_disjoint_and_sized() {
        let rule = SplitRule::SeededPermutation { seed: 11, discovery_fraction: 0.4 };
        let (d1, a1) = rule.assign(50).unwrap();
        let (d2, a2) = rule.assign(50).unwrap();
        assert_eq!((&d1, &a1), (&d2, &a2));
        assert_eq!(d1.len(), 20);
        assert_eq!(a1.len(), 30);
        assert!(d1.iter().all(|r| a1.binary_search(r).is_err()));
        let other = SplitRule::SeededPermutation { seed: 12, discovery_fraction: 0.4 };
        assert_ne!(other.assign(50).unwrap().0, d1);
    }

    #[test]
    fn malformed_rules_are_refused_with_registered_details() {
        let repeated = SplitRule::Explicit { discovery_rows: vec![1, 1] };
        assert_eq!(
            repeated.assign(4).unwrap_err().detail(),
            Some("selective_discovery.split_rows_repeated")
        );
        let out = SplitRule::Explicit { discovery_rows: vec![9] };
        assert_eq!(
            out.assign(4).unwrap_err().detail(),
            Some("selective_discovery.split_row_out_of_range")
        );
        let all = SplitRule::Explicit { discovery_rows: vec![0, 1, 2, 3] };
        assert_eq!(
            all.assign(4).unwrap_err().detail(),
            Some("selective_discovery.split_part_empty")
        );
        let fraction = SplitRule::SeededPermutation { seed: 1, discovery_fraction: 1.0 };
        assert_eq!(
            fraction.assign(4).unwrap_err().detail(),
            Some("selective_discovery.split_fraction_invalid")
        );
    }
}
