//! The one definition of the inference-declaration builders that estimators share.
//!
//! Every estimator that publishes a standard error or a resampling interval carries the same
//! declarations as public fields (bootstrap size, analytic SE kind, clustering, panel times,
//! sampling design). The `with_*` builders over those fields are generated here, once, and
//! expanded inside each estimator's `impl` block, so a change to one builder's contract is a
//! change to one definition.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

/// `with_bootstrap_replicates`: the bootstrap replicate count (`0` skips the bootstrap).
macro_rules! with_bootstrap_replicates {
    () => {
        /// Set the number of bootstrap replicates; `0` skips the bootstrap and reports only the
        /// analytic standard error.
        #[must_use]
        pub const fn with_bootstrap_replicates(mut self, replicates: u32) -> Self {
            self.bootstrap_replicates = replicates;
            self
        }
    };
}

/// `with_se_kind`: the analytic standard-error kind.
macro_rules! with_se_kind {
    () => {
        /// Set the analytic standard-error kind.
        #[must_use]
        pub const fn with_se_kind(mut self, se_kind: $crate::se::AnalyticSeKind) -> Self {
            self.se_kind = se_kind;
            self
        }
    };
}

/// `with_cluster_ids`: cluster labels aligned to the prepared rows.
macro_rules! with_cluster_ids {
    () => {
        /// Set cluster ids (aligned to the prepared rows) for the cluster standard error.
        #[must_use]
        pub fn with_cluster_ids(mut self, cluster_ids: Vec<u32>) -> Self {
            self.cluster_ids = Some(cluster_ids);
            self
        }
    };
}

/// `with_source_cluster_ids`: cluster labels of every row of the source table.
macro_rules! with_source_cluster_ids {
    () => {
        /// Set cluster labels for every row of the source table (length must equal its row
        /// count); each complete-case row keeps the label of its original row.
        #[must_use]
        pub fn with_source_cluster_ids(mut self, cluster_ids: Vec<u32>) -> Self {
            self.source_cluster_ids = Some(cluster_ids);
            self
        }
    };
}

/// `with_multiway_ids`: one label vector per clustering dimension.
macro_rules! with_multiway_ids {
    () => {
        /// Set multiway cluster ids (one `Vec<u32>` per clustering dimension).
        #[must_use]
        pub fn with_multiway_ids(mut self, multiway_ids: Vec<Vec<u32>>) -> Self {
            self.multiway_ids = Some(multiway_ids);
            self
        }
    };
}

/// `with_panel_times`: panel time labels for the panel HAC standard error.
macro_rules! with_panel_times {
    () => {
        /// Set panel time labels for the panel cluster-HAC standard error.
        #[must_use]
        pub fn with_panel_times(mut self, panel_times: Vec<i64>) -> Self {
            self.panel_times = Some(panel_times);
            self
        }
    };
}

/// `with_sampling_design`: declare the sampling design the bootstrap must respect.
macro_rules! with_sampling_design {
    () => {
        /// Declare the sampling design of the rows. A bootstrap resample the design does not
        /// allow (a row-level draw under independent subject trajectories, a non-block draw of
        /// a dependent series, or a draw that ignores a panel's clusters) is refused.
        #[must_use]
        pub fn with_sampling_design(mut self, design: antecedent_core::DeclaredSamplingDesign) -> Self {
            self.sampling_design = $crate::util::SamplingDeclaration::Declared(design);
            self
        }
    };
}

/// `with_bootstrap_interval`: request a bootstrap interval from the wave procedure.
macro_rules! with_bootstrap_interval {
    () => {
        /// Request a bootstrap interval produced by the complete fit/resample/stop/interval
        /// procedure (see the `bootstrap_interval` field).
        #[must_use]
        pub const fn with_bootstrap_interval(mut self, request: $crate::util::IntervalRequest) -> Self {
            self.bootstrap_interval = Some(request);
            self
        }
    };
}

pub(crate) use {
    with_bootstrap_interval, with_bootstrap_replicates, with_cluster_ids, with_multiway_ids,
    with_panel_times, with_sampling_design, with_se_kind, with_source_cluster_ids,
};
