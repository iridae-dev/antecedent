# Graphless design-family support matrix

This table is separate from the graph/structure support axes in
[the geometric matrix](support-matrix.md). A query is licensed only when
its exact family, design, method, inference claim, and observed
assignment-unit counts match a row. All other combinations are refused.
That refusal means no matrix license; off-axis point results may still
execute. The route must validate its stated design and assumptions.

| Family | Design | Method | Inference claim | Assignment support | Evidence |
| --- | --- | --- | --- | --- | --- |
| `randomized_effect` | `bernoulli` | `independent_action_ht_score` | `pointwise_95_normal_interval` | >= 400 rows, >= 30 assignment units per arm, >= 1 reported intervals, probability >= 0.2, interval published | [`independent_bernoulli_and_multi_arm_coverage`](../crates/antecedent-estimate/src/randomized_scores.rs), [`calibrated_randomized_intervals_round_trip_and_reject_tampering`](../crates/antecedent/tests/randomized_trial_route.rs) |
| `randomized_effect` | `cluster` | `neyman_unit_weighted_cluster_totals` | `pointwise_95_normal_interval` | >= 30 assignment units per arm, >= 1 reported intervals, interval published | [`cluster_interval_support_boundary_covers_known_itt`](../crates/antecedent-estimate/src/randomized_neyman.rs), [`calibrated_randomized_intervals_round_trip_and_reject_tampering`](../crates/antecedent/tests/randomized_trial_route.rs) |
| `randomized_effect` | `complete` | `neyman_difference_in_means` | `pointwise_95_normal_interval` | >= 30 assignment units per arm, >= 1 reported intervals, interval published | [`complete_interval_support_boundary_covers_known_itt`](../crates/antecedent-estimate/src/randomized_neyman.rs), [`calibrated_randomized_intervals_round_trip_and_reject_tampering`](../crates/antecedent/tests/randomized_trial_route.rs) |
| `randomized_effect` | `factorial_2x2` | `fixed_cell_neyman_contrasts` | `three_pointwise_95_normal_intervals` | >= 30 factorial cell, >= 3 reported intervals, all intervals | [`factorial_cell_randomization_covers_three_known_pointwise_effects`](../crates/antecedent-estimate/src/randomized_neyman.rs), [`calibrated_randomized_intervals_round_trip_and_reject_tampering`](../crates/antecedent/tests/randomized_trial_route.rs) |
| `randomized_effect` | `multi_arm` | `independent_action_ht_scores` | `all_action_pointwise_95_normal_intervals` | >= 400 rows, >= 30 action rows, >= 2 reported intervals, probability >= 0.2, all intervals | [`independent_bernoulli_and_multi_arm_coverage`](../crates/antecedent-estimate/src/randomized_scores.rs), [`calibrated_randomized_intervals_round_trip_and_reject_tampering`](../crates/antecedent/tests/randomized_trial_route.rs) |
| `randomized_effect` | `stratified` | `blocked_neyman_difference_in_means` | `pointwise_95_normal_interval` | >= 60 assignment units per arm, >= 4 blocks, >= 15 block arm, >= 1 reported intervals, interval published | [`blocked_randomization_covers_known_itt_and_withholds_sparse_blocks`](../crates/antecedent-estimate/src/randomized_neyman.rs), [`calibrated_randomized_intervals_round_trip_and_reject_tampering`](../crates/antecedent/tests/randomized_trial_route.rs) |

**bernoulli limits:** Independent Bernoulli binary ITT with known probabilities at least 0.2 in both arms on every row, 400 rows, 30 realized assignments per arm, and a published positive-variance pointwise 95% score interval. Fixed CUPED, ANCOVA, CACE/LATE, and lower-probability designs are outside this license.

**cluster limits:** Known complete cluster assignment, independent clusters, binary unit-weighted ITT, at least 30 clusters per arm, and an actually published interval (positive realized variance). Conservative cluster-total variance and a pointwise 95% normal interval; no observational network exposure or other assignment design is licensed by this row.

**complete limits:** Known complete assignment, independent units, binary ITT, at least 30 units per arm, and an actually published interval (positive realized variance). Conservative Neyman variance and a pointwise 95% normal interval; no simultaneous interval, CACE, adjustment, or other assignment design is licensed by this row.

**factorial_2x2 limits:** Known fixed-cell 2×2 randomization with at least 30 units in every cell and published positive-variance intervals for both main effects and the interaction. Each 95% interval is pointwise; joint or simultaneous coverage is not licensed.

**multi_arm limits:** Independent known-probability assignment to at least three actions, 400 rows, at least 30 realized assignments to every action, probabilities at least 0.2 for every action on every row, and a published interval for every action-versus-reference contrast. The intervals are separate pointwise 95% claims; simultaneous coverage is not licensed.

**stratified limits:** Known independent complete assignment within at least four blocks, at least 15 units in every block arm and 60 per overall arm, and a published positive-variance pointwise 95% interval. This row licenses the block-weighted ITT only.
