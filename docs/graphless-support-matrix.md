# Graphless design-family support matrix

This table is separate from the graph/structure support axes in
[the geometric matrix](support-matrix.md). A query is licensed only when
its exact family, design, method, inference claim, and observed
assignment-unit counts match a row. All other combinations are refused.
That refusal means no matrix license; off-axis point results may still
execute. The route must validate its stated design and assumptions.

| Family | Design | Method | Inference claim | Assignment support | Evidence |
| --- | --- | --- | --- | --- | --- |
| `randomized_effect` | `cluster` | `neyman_unit_weighted_cluster_totals` | `pointwise_95_normal_interval` | >= 30 clusters per arm; interval published | [`cluster_interval_support_boundary_covers_known_itt`](../crates/antecedent-estimate/src/randomized_neyman.rs), [`calibrated_randomized_intervals_round_trip_and_reject_tampering`](../crates/antecedent/tests/randomized_trial_route.rs) |
| `randomized_effect` | `complete` | `neyman_difference_in_means` | `pointwise_95_normal_interval` | >= 30 units per arm; interval published | [`complete_interval_support_boundary_covers_known_itt`](../crates/antecedent-estimate/src/randomized_neyman.rs), [`calibrated_randomized_intervals_round_trip_and_reject_tampering`](../crates/antecedent/tests/randomized_trial_route.rs) |

**cluster limits:** Known complete cluster assignment, independent clusters, binary unit-weighted ITT, at least 30 clusters per arm, and an actually published interval (positive realized variance). Conservative cluster-total variance and a pointwise 95% normal interval; no observational network exposure or other assignment design is licensed by this row.

**complete limits:** Known complete assignment, independent units, binary ITT, at least 30 units per arm, and an actually published interval (positive realized variance). Conservative Neyman variance and a pointwise 95% normal interval; no simultaneous interval, CACE, adjustment, or other assignment design is licensed by this row.
