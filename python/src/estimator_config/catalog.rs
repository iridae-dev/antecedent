//! Supported estimator configuration keys.
// SPDX-License-Identifier: MIT OR Apache-2.0

/// One `(estimator_id, valid_keys)` row. Table-driven: adding a configurable estimator or a
/// key to an existing one is a data change here, not a new branch scattered through the
/// parser. The unknown-key / wrong-estimator error messages are both derived from this table.
const ESTIMATOR_KEYS: &[(&str, &[&str])] = &[
    (
        "linear.adjustment.ate",
        &[
            "bootstrap_replicates",
            "bootstrap_interval",
            "se_kind",
            "se_lag",
            "cluster_ids",
            "source_cluster_ids",
            "source_sampling_weights",
            "multiway_ids",
            "panel_times",
            "fit_kind",
            "fit_lambda",
            "fit_c",
            "other_levels",
        ],
    ),
    (
        "propensity.weighting",
        &[
            "bootstrap_replicates",
            "bootstrap_interval",
            "glm_options",
            "overlap",
            "other_levels",
            "cluster_ids",
            "source_cluster_ids",
            "source_sampling_weights",
        ],
    ),
    (
        "propensity.matching",
        &[
            "bootstrap_replicates",
            "se_kind",
            "se_lag",
            "cluster_ids",
            "multiway_ids",
            "panel_times",
            "glm_options",
            "caliper",
            "caliper_scale",
            "overlap",
            "other_levels",
            "source_sampling_weights",
        ],
    ),
    (
        "propensity.stratification",
        &[
            "bootstrap_replicates",
            "bootstrap_interval",
            "glm_options",
            "n_strata",
            "overlap",
            "other_levels",
            "cluster_ids",
            "source_cluster_ids",
            "source_sampling_weights",
        ],
    ),
    (
        "distance.matching",
        &[
            "bootstrap_replicates",
            "se_kind",
            "se_lag",
            "cluster_ids",
            "multiway_ids",
            "panel_times",
            "glm_options",
            "caliper",
            "overlap",
            "source_sampling_weights",
        ],
    ),
    (
        "aipw",
        &[
            "bootstrap_replicates",
            "bootstrap_interval",
            "se_kind",
            "se_lag",
            "cluster_ids",
            "source_cluster_ids",
            "multiway_ids",
            "panel_times",
            "glm_options",
            "overlap",
            "propensity_penalty",
            "nuisance_fallback",
            "cluster_dml",
            "other_levels",
            "source_sampling_weights",
        ],
    ),
    (
        "glm.adjustment",
        &[
            "bootstrap_replicates",
            "bootstrap_interval",
            "se_kind",
            "se_lag",
            "cluster_ids",
            "source_cluster_ids",
            "multiway_ids",
            "panel_times",
            "glm_options",
            "family",
            "other_levels",
            "source_sampling_weights",
        ],
    ),
    (
        "frontdoor.linear_two_stage",
        &["bootstrap_replicates", "se_kind", "se_lag", "cluster_ids", "source_cluster_ids"],
    ),
    (
        "iv.wald",
        &[
            "bootstrap_replicates",
            "se_kind",
            "se_lag",
            "cluster_ids",
            "multiway_ids",
            "panel_times",
        ],
    ),
    (
        "iv.2sls",
        &[
            "bootstrap_replicates",
            "se_kind",
            "se_lag",
            "cluster_ids",
            "multiway_ids",
            "panel_times",
        ],
    ),
    // `rd.sharp` has no `EstimatorSpec` variant (see estimator_spec.rs doc comment) — its
    // triple is applied via the pre-existing `StudyBuilder::rd_config` path instead, merged
    // with the loose `running_variable`/`cutoff`/`bandwidth` kwargs by `merge_rd_triple`.
    // `se_kind` selects the jump coefficient's residual-based SE (`RdConfig::with_se_kind`).
    ("rd.sharp", &["running_variable", "cutoff", "bandwidth", "se_kind"]),
    (
        "dml",
        &[
            "learner",
            "outcome",
            "treatment",
            "score",
            "folds",
            "overlap",
            "other_levels",
            "cluster_dml",
            "source_sampling_weights",
        ],
    ),
    (
        "dr.learner",
        &[
            "learner",
            "outcome",
            "treatment",
            "final_learner",
            "folds",
            "overlap",
            "cluster_dml",
            "source_sampling_weights",
        ],
    ),
    (
        "causal.forest",
        &["n_trees", "min_leaf", "max_depth", "honesty", "source_cluster_ids", "min_clusters"],
    ),
];

pub(super) fn valid_keys_for(estimator_id: &str) -> Option<&'static [&'static str]> {
    ESTIMATOR_KEYS.iter().find(|(id, _)| *id == estimator_id).map(|(_, keys)| *keys)
}

pub(super) fn estimators_accepting(key: &str) -> Vec<&'static str> {
    ESTIMATOR_KEYS.iter().filter(|(_, keys)| keys.contains(&key)).map(|(id, _)| *id).collect()
}

pub(super) fn supported_estimator_ids() -> Vec<&'static str> {
    ESTIMATOR_KEYS.iter().map(|(id, _)| *id).collect()
}
