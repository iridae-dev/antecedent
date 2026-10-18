//! Exact graphless design-family license predicate.
//!
//! The license table is generated from `parity/support_graphless.toml` by
//! `scripts/generate_graphless_support.py`. This module is the single owner of
//! the predicate: the facade's checked execution and artifact verification
//! both classify through [`classify_graphless`] / [`graphless_admits`], so the
//! two cannot drift apart.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#[path = "support_graphless_data.rs"]
mod data;

use data::GraphlessLicenseRow;
use data::LICENSES;

/// Exact graphless design, method, and inference claim to classify.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GraphlessSupportKey<'a> {
    /// Query family, such as `randomized_effect`.
    pub family: &'a str,
    /// Assignment design, such as `complete` or `cluster`.
    pub design: &'a str,
    /// Concrete estimator method; similar methods cannot inherit a license.
    pub method: &'a str,
    /// Exact inference claim, including confidence level and pointwise scope.
    pub inference_claim: &'a str,
}

/// Observed design and inference support for an exact graphless license.
#[allow(
    clippy::struct_excessive_bools,
    reason = "each flag records a distinct observed support condition; they are not a state enum"
)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GraphlessAssignmentSupport {
    /// `unit`, `cluster`, or independent `sequence`, matched to the licensed design.
    pub assignment_unit: &'static str,
    /// Number assigned to the treatment arm.
    pub treated: usize,
    /// Number assigned to the reference arm.
    pub control: usize,
    /// True only after the estimator actually published its 95% interval.
    pub interval_95_published: bool,
    /// Number of analyzed outcome rows.
    pub rows: usize,
    /// Number of separate randomization blocks or independent sequences.
    pub blocks: usize,
    /// Smallest realized assignment count in any block arm.
    pub min_block_arm: usize,
    /// Smallest realized count in any factorial cell.
    pub min_factorial_cell: usize,
    /// Smallest realized count across every named action.
    pub min_action_rows: usize,
    /// Smallest declared probability across every action and row.
    pub min_probability: f64,
    /// Smaller of the expected treated and expected control rows: the arm's summed known
    /// assignment probabilities. Zero when the design declares no per-row probabilities.
    pub min_expected_arm_units: f64,
    /// Number of distinct published pointwise intervals.
    pub reported_intervals: usize,
    /// Number of fitted pre-assignment ANCOVA covariates.
    pub covariates: usize,
    /// Whether every contrast named by the query has an interval.
    pub all_reported_intervals: bool,
    /// Realized matches for the evaluated policy and reference.
    pub policy_matches: usize,
    /// Realized reference matches.
    pub reference_matches: usize,
    /// Smallest held-out rank bin and observed arm count within a bin.
    pub min_bin_rows: usize,
    /// Smallest observed arm count within a held-out rank bin.
    pub min_bin_arm_rows: usize,
    /// Smallest fixed baseline group and observed compared-arm count.
    pub min_group_rows: usize,
    /// Smallest compared-arm count within a fixed baseline group.
    pub min_group_arm_rows: usize,
    /// Smallest local kernel window across fixed dose-policy targets.
    pub min_local_rows: usize,
    /// Smallest effective sample size across fixed dose-policy windows.
    pub min_effective_sample_size: f64,
    /// Largest normalized kernel weight across fixed dose-policy windows.
    pub max_normalized_weight: f64,
    /// Smallest observed-dose density across fixed dose-policy windows.
    pub min_dose_density: f64,
    /// Caller declares the supplied observed-dose density known, not fitted here.
    pub known_density: bool,
    /// Global action constraints do not couple subject recommendations.
    pub uncoupled_constraints: bool,
    /// Outcome nuisance predictions use declared disjoint training subjects.
    pub disjoint_nuisance_training: bool,
    /// Ranking training IDs are declared and disjoint from evaluation IDs.
    pub rank_ownership: bool,
    /// Every independent sequence has the same observed number of periods.
    pub balanced_sequences: bool,
}

/// Runtime status for one exact graphless support key.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GraphlessSupportStatus {
    /// Exact license and assignment support matched.
    Licensed {
        /// Scope restrictions published with the license.
        limitations: &'static str,
        /// Known-truth calibration test used for the row.
        known_truth_test: &'static str,
        /// Retained result/artifact route test used for the row.
        retained_route_test: &'static str,
    },
    /// No exact row or too little independent assignment support.
    Refused,
}

/// The licensed row that admits `key` under `observed`, if any.
fn find_license(
    key: GraphlessSupportKey<'_>,
    observed: GraphlessAssignmentSupport,
) -> Option<&'static GraphlessLicenseRow> {
    LICENSES.iter().find(|row| row.admits(key, observed))
}

/// Whether some licensed row matches `key` and is satisfied by `observed`.
#[must_use]
pub fn graphless_admits(
    key: GraphlessSupportKey<'_>,
    observed: GraphlessAssignmentSupport,
) -> bool {
    find_license(key, observed).is_some()
}

/// Classify a graphless design against the separate machine-readable table.
/// This lookup cannot grant a geometric support-matrix cell by analogy.
#[must_use]
pub fn classify_graphless(
    key: GraphlessSupportKey<'_>,
    observed: GraphlessAssignmentSupport,
) -> GraphlessSupportStatus {
    find_license(key, observed).map_or(GraphlessSupportStatus::Refused, |row| {
        GraphlessSupportStatus::Licensed {
            limitations: row.limitations,
            known_truth_test: row.known_truth_test,
            retained_route_test: row.retained_route_test,
        }
    })
}

impl GraphlessLicenseRow {
    /// Whether this row matches `key` and every threshold is met by `observed`.
    fn admits(&self, key: GraphlessSupportKey<'_>, observed: GraphlessAssignmentSupport) -> bool {
        let row = self;
        row.family == key.family
            && row.design == key.design
            && row.method == key.method
            && row.inference_claim == key.inference_claim
            && row.assignment_unit == observed.assignment_unit
            && observed.treated >= row.min_assignment_units_per_arm
            && observed.control >= row.min_assignment_units_per_arm
            && observed.interval_95_published
            && observed.rows >= row.min_rows
            && observed.blocks >= row.min_blocks
            && observed.min_block_arm >= row.min_block_arm
            && observed.min_factorial_cell >= row.min_factorial_cell
            && observed.min_action_rows >= row.min_action_rows
            && observed.min_probability + 1e-12 >= row.min_probability
            && observed.min_expected_arm_units + 1e-9 >= row.min_expected_arm_units
            && observed.reported_intervals >= row.min_reported_intervals
            && (row.max_covariates == 0 || observed.covariates <= row.max_covariates)
            && (!row.all_reported_intervals || observed.all_reported_intervals)
            && observed.policy_matches >= row.min_policy_matches
            && observed.reference_matches >= row.min_reference_matches
            && observed.min_bin_rows >= row.min_bin_rows
            && observed.min_bin_arm_rows >= row.min_bin_arm_rows
            && observed.min_group_rows >= row.min_group_rows
            && observed.min_group_arm_rows >= row.min_group_arm_rows
            && observed.min_local_rows >= row.min_local_rows
            && observed.min_effective_sample_size + 1e-12 >= row.min_effective_sample_size
            && (row.max_normalized_weight == 0.0
                || observed.max_normalized_weight <= row.max_normalized_weight + 1e-12)
            && observed.min_dose_density + 1e-12 >= row.min_dose_density
            && (!row.requires_uncoupled_constraints || observed.uncoupled_constraints)
            && (!row.requires_disjoint_nuisance_training || observed.disjoint_nuisance_training)
            && (!row.requires_rank_ownership || observed.rank_ownership)
            && (!row.requires_balanced_sequences || observed.balanced_sequences)
            && (!row.requires_known_density || observed.known_density)
    }
}
