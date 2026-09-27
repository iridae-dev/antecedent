//! Generated from parity/support_graphless.toml; do not edit by hand.
pub(super) struct GraphlessLicenseRow {
    pub(super) family: &'static str,
    pub(super) design: &'static str,
    pub(super) method: &'static str,
    pub(super) inference_claim: &'static str,
    pub(super) assignment_unit: &'static str,
    pub(super) known_truth_test: &'static str,
    pub(super) retained_route_test: &'static str,
    pub(super) limitations: &'static str,
    pub(super) min_assignment_units_per_arm: usize,
}
pub(super) const LICENSES: &[GraphlessLicenseRow] = &[
    GraphlessLicenseRow {
        family: "randomized_effect",
        design: "cluster",
        method: "neyman_unit_weighted_cluster_totals",
        inference_claim: "pointwise_95_normal_interval",
        assignment_unit: "cluster",
        known_truth_test: "crates/antecedent-estimate/src/randomized_neyman.rs::cluster_interval_support_boundary_covers_known_itt",
        retained_route_test: "crates/antecedent/tests/randomized_trial_route.rs::calibrated_randomized_intervals_round_trip_and_reject_tampering",
        limitations: "Known complete cluster assignment, independent clusters, binary unit-weighted ITT, at least 30 clusters per arm, and an actually published interval (positive realized variance). Conservative cluster-total variance and a pointwise 95% normal interval; no observational network exposure or other assignment design is licensed by this row.",
        min_assignment_units_per_arm: 30,
    },
    GraphlessLicenseRow {
        family: "randomized_effect",
        design: "complete",
        method: "neyman_difference_in_means",
        inference_claim: "pointwise_95_normal_interval",
        assignment_unit: "unit",
        known_truth_test: "crates/antecedent-estimate/src/randomized_neyman.rs::complete_interval_support_boundary_covers_known_itt",
        retained_route_test: "crates/antecedent/tests/randomized_trial_route.rs::calibrated_randomized_intervals_round_trip_and_reject_tampering",
        limitations: "Known complete assignment, independent units, binary ITT, at least 30 units per arm, and an actually published interval (positive realized variance). Conservative Neyman variance and a pointwise 95% normal interval; no simultaneous interval, CACE, adjustment, or other assignment design is licensed by this row.",
        min_assignment_units_per_arm: 30,
    },
];
