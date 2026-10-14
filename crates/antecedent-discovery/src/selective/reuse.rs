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
//! [`SampleSplitReceipt`] records the rule, the row-set identities and the content identities
//! of both tables so a consumer can re-derive the partition from the rule and the row count
//! instead of trusting a flag ([`SampleSplitReceipt::verify`]).
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
//!   cover. Weighted rows and rows carrying an analysis mask are refused: how a weight or a
//!   mask should be split is not determined by the rows.
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

use antecedent_core::{CanonicalWriter, VariableId};
use antecedent_data::{TableView, TabularData};

use super::identity::{DataIdentity, Digest, SELECTIVE_IDENTITY_VERSION, static_data_identity};
use super::refusal::SelectiveDiscoveryError;

const SPLIT_TAG: &str = "antecedent.selective_discovery.sample_split.v1";
const ROWS_TAG: &str = "antecedent.selective_discovery.sample_split.rows.v1";

fn invalid(detail: &'static str, remedy: impl Into<String>) -> SelectiveDiscoveryError {
    SelectiveDiscoveryError::refused(
        antecedent_core::reason_code!("invalid_argument"),
        detail,
        remedy,
    )
}

fn not_supported(detail: &'static str, remedy: impl Into<String>) -> SelectiveDiscoveryError {
    SelectiveDiscoveryError::refused(
        antecedent_core::reason_code!("route_not_supported"),
        detail,
        remedy,
    )
}

/// How rows are assigned to the discovery and analysis parts.
#[derive(Clone, Debug, PartialEq)]
pub enum SplitRule {
    /// The caller names the discovery rows; every other row is an analysis row.
    Explicit {
        /// Rows used for discovery, any order; repeats and out-of-range rows are refused.
        discovery_rows: Vec<usize>,
    },
    /// A seeded permutation of the rows; the first `round(fraction * n)` of the permutation
    /// are discovery rows. Deterministic and independent of platform.
    SeededPermutation {
        /// Permutation seed.
        seed: u64,
        /// Share of rows used for discovery, in `(0, 1)`.
        discovery_fraction: f64,
    },
}

impl SplitRule {
    /// The discovery rows (ascending) and analysis rows (ascending) for `n_rows` rows.
    ///
    /// # Errors
    ///
    /// A row out of range or repeated, a fraction outside `(0, 1)`, or a part with no rows.
    pub fn assign(&self, n_rows: usize) -> Result<(Vec<usize>, Vec<usize>), SelectiveDiscoveryError> {
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
                rows
            }
            Self::SeededPermutation { seed, discovery_fraction } => {
                if !(*discovery_fraction > 0.0 && *discovery_fraction < 1.0) {
                    return Err(invalid(
                        "selective_discovery.split_fraction_invalid",
                        "the discovery fraction must lie strictly between 0 and 1",
                    ));
                }
                let mut order: Vec<usize> = (0..n_rows).collect();
                let mut state = *seed;
                for i in (1..n_rows).rev() {
                    let j = (splitmix64(&mut state) % (i as u64 + 1)) as usize;
                    order.swap(i, j);
                }
                let take = ((n_rows as f64) * discovery_fraction).round() as usize;
                let take = take.clamp(1, n_rows.saturating_sub(1).max(1));
                let mut rows: Vec<usize> = order[..take.min(n_rows)].to_vec();
                rows.sort_unstable();
                rows
            }
        };
        discovery.dedup();
        let in_discovery: std::collections::BTreeSet<usize> = discovery.iter().copied().collect();
        let analysis: Vec<usize> = (0..n_rows).filter(|r| !in_discovery.contains(r)).collect();
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

/// Record of one row split: enough to re-derive the partition and to check a table against it.
#[derive(Clone, Debug, PartialEq)]
pub struct SampleSplitReceipt {
    /// The declared rule.
    pub rule: SplitRule,
    /// Row count of the table that was split.
    pub n_rows: usize,
    /// Rows in the discovery part.
    pub n_discovery: usize,
    /// Rows in the analysis part.
    pub n_analysis: usize,
    /// Identity of the discovery row set.
    pub discovery_rows_identity: Digest,
    /// Identity of the analysis row set.
    pub analysis_rows_identity: Digest,
    /// Content identity of the discovery part over the declared variables.
    pub discovery_data: DataIdentity,
    /// Content identity of the analysis part over the declared variables.
    pub analysis_data: DataIdentity,
    /// Identity of the whole receipt.
    pub identity: Digest,
}

impl SampleSplitReceipt {
    /// Re-derive the partition from the rule and row count and check it against the receipt.
    ///
    /// Does not trust the stored sizes or row-set identities: it recomputes the assignment,
    /// requires the two parts to be disjoint and to cover `n_rows`, and requires the stored
    /// identities to match.
    ///
    /// # Errors
    ///
    /// A rule that no longer assigns rows, parts that overlap or do not cover the rows, or
    /// stored identities that differ from the re-derived ones.
    pub fn verify(&self) -> Result<(), SelectiveDiscoveryError> {
        let (discovery, analysis) = self.rule.assign(self.n_rows)?;
        let overlap = discovery.iter().any(|row| analysis.binary_search(row).is_ok());
        if overlap || discovery.len() + analysis.len() != self.n_rows {
            return Err(invalid(
                "selective_discovery.split_not_disjoint",
                "the discovery and analysis rows overlap or do not cover the rows",
            ));
        }
        if discovery.len() != self.n_discovery
            || analysis.len() != self.n_analysis
            || rows_identity(ROWS_TAG, &discovery) != self.discovery_rows_identity
            || rows_identity(ROWS_TAG, &analysis) != self.analysis_rows_identity
        {
            return Err(invalid(
                "selective_discovery.split_receipt_mismatch",
                "the receipt's row sets are not the ones its rule assigns; re-split from the \
                 declared rule",
            ));
        }
        if self.identity != receipt_identity(self) {
            return Err(invalid(
                "selective_discovery.split_receipt_mismatch",
                "the receipt's identity does not cover its recorded contents",
            ));
        }
        Ok(())
    }

    /// Check that `table` is exactly the analysis part this receipt describes, over
    /// `variables`.
    ///
    /// # Errors
    ///
    /// A table whose searched columns differ in content or row count from the recorded
    /// analysis part, which is the signature of an analysis on other rows (for example the
    /// discovery rows or the unsplit table).
    pub fn check_analysis_table(
        &self,
        table: &TabularData,
        variables: &[VariableId],
    ) -> Result<(), SelectiveDiscoveryError> {
        self.verify()?;
        let identity = static_data_identity(table, variables).map_err(SelectiveDiscoveryError::from)?;
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
    writer.finish()
}

/// The two parts of a split table.
#[derive(Clone, Debug)]
pub struct SplitRows {
    /// The only rows discovery may use.
    pub discovery: TabularData,
    /// The only rows the downstream analysis may use.
    pub analysis: TabularData,
    /// The record of the split.
    pub receipt: SampleSplitReceipt,
}

/// Partition the rows of `data` into discovery and analysis parts.
///
/// Every schema variable must be a float64-valued column; the parts carry all of them, so the
/// analysis may use covariates discovery did not search. `variables` names the columns whose
/// content identity the receipt records (normally the searched variables).
///
/// # Errors
///
/// Weighted rows or rows with an analysis mask (`selective_discovery.split_unsupported_rows`),
/// a rule that does not assign both parts non-empty rows, a variable that is not numeric, or
/// a variable that is not in the schema.
pub fn split_rows(
    data: &TabularData,
    variables: &[VariableId],
    rule: SplitRule,
) -> Result<SplitRows, SelectiveDiscoveryError> {
    let storage = data.storage();
    if storage.weights().is_some() || storage.analysis_mask().is_some() {
        return Err(not_supported(
            "selective_discovery.split_unsupported_rows",
            "rows carrying weights or an analysis mask are not split: how a weight or a mask \
             belongs to a part is not determined by the rows. Materialize the weighted or \
             masked rows into a plain table, then split",
        ));
    }
    let n_rows = data.row_count();
    let (discovery_rows, analysis_rows) = rule.assign(n_rows)?;

    let schema = storage.schema().clone();
    let mut columns: Vec<(String, Vec<f64>)> = Vec::with_capacity(schema.len());
    for spec in schema.variables() {
        let values = data.float64_values(spec.id).map_err(|_| {
            invalid(
                "selective_discovery.split_unsupported_rows",
                format!("variable `{}` is not a float64 column; split a numeric table", spec.name),
            )
        })?;
        columns.push((spec.name.to_string(), values));
    }
    let part = |rows: &[usize]| -> Result<TabularData, SelectiveDiscoveryError> {
        let gathered: Vec<(String, Vec<f64>)> = columns
            .iter()
            .map(|(name, values)| (name.clone(), rows.iter().map(|&r| values[r]).collect()))
            .collect();
        TabularData::try_from_schema_f64(
            schema.clone(),
            gathered.iter().map(|(name, values)| (name.as_str(), values.as_slice())),
        )
        .map_err(|error| {
            invalid("selective_discovery.split_unsupported_rows", error.to_string())
        })
    };
    let discovery = part(&discovery_rows)?;
    let analysis = part(&analysis_rows)?;

    let discovery_data =
        static_data_identity(&discovery, variables).map_err(SelectiveDiscoveryError::from)?;
    let analysis_data =
        static_data_identity(&analysis, variables).map_err(SelectiveDiscoveryError::from)?;
    let mut receipt = SampleSplitReceipt {
        rule,
        n_rows,
        n_discovery: discovery_rows.len(),
        n_analysis: analysis_rows.len(),
        discovery_rows_identity: rows_identity(ROWS_TAG, &discovery_rows),
        analysis_rows_identity: rows_identity(ROWS_TAG, &analysis_rows),
        discovery_data,
        analysis_data,
        identity: [0; 32],
    };
    receipt.identity = receipt_identity(&receipt);
    Ok(SplitRows { discovery, analysis, receipt })
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
