//! The one definition of the inference-declaration builders that estimators share.
//!
//! Every estimator that publishes a standard error or a resampling interval carries the same
//! declarations as public fields (bootstrap size, analytic SE kind, clustering, panel times,
//! sampling design). The `with_*` builders over those fields are generated here, once, and
//! expanded inside each estimator's `impl` block, so a change to one builder's contract is a
//! change to one definition.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use crate::se::AnalyticSeKind;
use crate::util::{IntervalRequest, SamplingDeclaration};

/// The inference declarations a resampling or standard-error estimator carries: how the
/// uncertainty is produced (bootstrap size and interval request, analytic standard-error kind),
/// which dependence structure the rows have (cluster labels, multiway labels, panel times) and
/// the sampling design the bootstrap must respect.
///
/// The estimators that publish a standard error or a resampling interval embed one value of
/// this type as their public `inference` field. Its members are public so the declarations can
/// be read and assigned directly; the `with_*` builders on each estimator forward to them.
#[derive(Clone, Debug)]
pub struct InferenceSpec {
    /// Bootstrap replicates (`0` skips the bootstrap and reports only the analytic standard
    /// error).
    pub bootstrap_replicates: u32,
    /// A requested bootstrap interval. When set, the non-penalized bootstrap runs as the wave
    /// procedure that stops on the request's Monte Carlo tolerances (capped by
    /// [`Self::bootstrap_replicates`]) and reports its interval, waves, Monte Carlo error and
    /// stop reason in [`crate::EstimateDiagnostics::resampling`]. `None` runs the fixed-count
    /// bootstrap.
    pub bootstrap_interval: Option<IntervalRequest>,
    /// Analytic standard-error kind.
    pub se_kind: AnalyticSeKind,
    /// Cluster labels aligned to the prepared rows, for [`AnalyticSeKind::Cluster`].
    pub cluster_ids: Option<Vec<u32>>,
    /// Cluster labels of every row of the source table, subset to the complete-case rows by
    /// their original index. Exclusive with [`Self::cluster_ids`].
    pub source_cluster_ids: Option<Vec<u32>>,
    /// Multiway cluster labels (one `Vec<u32>` per clustering dimension).
    pub multiway_ids: Option<Vec<Vec<u32>>>,
    /// Panel time labels for [`AnalyticSeKind::PanelClusterHac`].
    pub panel_times: Option<Vec<i64>>,
    /// Sampling design of the rows, for the bootstrap. Undeclared, the design is the one the
    /// cluster labels or panel times imply; declared, a resample the design does not allow
    /// (a row-level draw under a trajectory design, say) is refused.
    pub sampling_design: SamplingDeclaration,
}

impl InferenceSpec {
    /// Declarations with `bootstrap_replicates` replicates, the homoskedastic analytic standard
    /// error, no clustering and an undeclared sampling design.
    #[must_use]
    pub const fn new(bootstrap_replicates: u32) -> Self {
        Self {
            bootstrap_replicates,
            bootstrap_interval: None,
            se_kind: AnalyticSeKind::Homoskedastic,
            cluster_ids: None,
            source_cluster_ids: None,
            multiway_ids: None,
            panel_times: None,
            sampling_design: SamplingDeclaration::Undeclared,
        }
    }
}

impl Default for InferenceSpec {
    /// Two hundred replicates, matching the estimators' default bootstrap size.
    fn default() -> Self {
        Self::new(200)
    }
}

/// The declarations of an [`InferenceSpec`] one estimator reads. The bootstrap size and the
/// analytic standard-error kind are read by every estimator that embeds the spec; the rest are
/// listed here, and a declaration an estimator does not read is refused, never ignored.
#[derive(Clone, Copy, Debug, Default)]
#[allow(
    clippy::struct_excessive_bools,
    reason = "one independent flag per declaration of the shared specification"
)]
pub(crate) struct ReadDeclarations {
    /// A requested bootstrap interval is read.
    pub(crate) bootstrap_interval: bool,
    /// Cluster labels aligned to the prepared rows are read.
    pub(crate) cluster_ids: bool,
    /// Source-table cluster labels are read.
    pub(crate) source_cluster_ids: bool,
    /// Multiway cluster labels are read.
    pub(crate) multiway_ids: bool,
    /// Panel time labels are read.
    pub(crate) panel_times: bool,
    /// A declared sampling design is read.
    pub(crate) sampling_design: bool,
}

impl InferenceSpec {
    /// Refuse a declaration `estimator` does not read, instead of ignoring it: every field of
    /// the spec that `read` leaves out must stay at its undeclared default.
    ///
    /// # Errors
    ///
    /// `estimator_inference_mismatch` naming the declarations that were set but are not read.
    pub(crate) fn require_read_declarations(
        &self,
        estimator: &str,
        read: ReadDeclarations,
    ) -> Result<(), crate::error::EstimationError> {
        let unread = [
            ("bootstrap_interval", !read.bootstrap_interval && self.bootstrap_interval.is_some()),
            ("cluster_ids", !read.cluster_ids && self.cluster_ids.is_some()),
            ("source_cluster_ids", !read.source_cluster_ids && self.source_cluster_ids.is_some()),
            ("multiway_ids", !read.multiway_ids && self.multiway_ids.is_some()),
            ("panel_times", !read.panel_times && self.panel_times.is_some()),
            (
                "sampling_design",
                !read.sampling_design
                    && !matches!(self.sampling_design, SamplingDeclaration::Undeclared),
            ),
        ]
        .into_iter()
        .filter_map(|(name, declared)| declared.then_some(name))
        .collect::<Vec<_>>();
        if unread.is_empty() {
            return Ok(());
        }
        Err(crate::error::EstimationError::refused(
            antecedent_core::reason_code!("estimator_inference_mismatch"),
            format!(
                "{estimator} does not read the inference declaration(s) {}; they would be \
                 silently ignored, so they must stay undeclared",
                unread.join(", ")
            ),
        ))
    }
}

/// `with_bootstrap_replicates`: the bootstrap replicate count (`0` skips the bootstrap).
macro_rules! with_bootstrap_replicates {
    ($($via:ident)?) => {
        /// Set the number of bootstrap replicates; `0` skips the bootstrap and reports only the
        /// analytic standard error.
        #[must_use]
        pub const fn with_bootstrap_replicates(mut self, replicates: u32) -> Self {
            self$(.$via)?.bootstrap_replicates = replicates;
            self
        }
    };
}

/// `with_se_kind`: the analytic standard-error kind.
macro_rules! with_se_kind {
    ($($via:ident)?) => {
        /// Set the analytic standard-error kind.
        #[must_use]
        pub const fn with_se_kind(mut self, se_kind: $crate::se::AnalyticSeKind) -> Self {
            self$(.$via)?.se_kind = se_kind;
            self
        }
    };
}

/// `with_cluster_ids`: cluster labels aligned to the prepared rows.
macro_rules! with_cluster_ids {
    ($($via:ident)?) => {
        /// Set cluster ids (aligned to the prepared rows) for the cluster standard error.
        #[must_use]
        pub fn with_cluster_ids(mut self, cluster_ids: Vec<u32>) -> Self {
            self$(.$via)?.cluster_ids = Some(cluster_ids);
            self
        }
    };
}

/// `with_source_cluster_ids`: cluster labels of every row of the source table.
macro_rules! with_source_cluster_ids {
    ($($via:ident)?) => {
        /// Set cluster labels for every row of the source table (length must equal its row
        /// count); each complete-case row keeps the label of its original row.
        #[must_use]
        pub fn with_source_cluster_ids(mut self, cluster_ids: Vec<u32>) -> Self {
            self$(.$via)?.source_cluster_ids = Some(cluster_ids);
            self
        }
    };
}

/// `with_multiway_ids`: one label vector per clustering dimension.
macro_rules! with_multiway_ids {
    ($($via:ident)?) => {
        /// Set multiway cluster ids (one `Vec<u32>` per clustering dimension).
        #[must_use]
        pub fn with_multiway_ids(mut self, multiway_ids: Vec<Vec<u32>>) -> Self {
            self$(.$via)?.multiway_ids = Some(multiway_ids);
            self
        }
    };
}

/// `with_panel_times`: panel time labels for the panel HAC standard error.
macro_rules! with_panel_times {
    ($($via:ident)?) => {
        /// Set panel time labels for the panel cluster-HAC standard error.
        #[must_use]
        pub fn with_panel_times(mut self, panel_times: Vec<i64>) -> Self {
            self$(.$via)?.panel_times = Some(panel_times);
            self
        }
    };
}

/// `with_sampling_design`: declare the sampling design the bootstrap must respect.
macro_rules! with_sampling_design {
    ($($via:ident)?) => {
        /// Declare the sampling design of the rows. A bootstrap resample the design does not
        /// allow (a row-level draw under independent subject trajectories, a non-block draw of
        /// a dependent series, or a draw that ignores a panel's clusters) is refused.
        #[must_use]
        pub fn with_sampling_design(mut self, design: antecedent_core::DeclaredSamplingDesign) -> Self {
            self$(.$via)?.sampling_design = $crate::util::SamplingDeclaration::Declared(design);
            self
        }
    };
}

/// `with_bootstrap_interval`: request a bootstrap interval from the wave procedure.
macro_rules! with_bootstrap_interval {
    ($($via:ident)?) => {
        /// Request a bootstrap interval produced by the complete fit/resample/stop/interval
        /// procedure (see the `bootstrap_interval` field).
        #[must_use]
        pub const fn with_bootstrap_interval(mut self, request: $crate::util::IntervalRequest) -> Self {
            self$(.$via)?.bootstrap_interval = Some(request);
            self
        }
    };
}

#[allow(unused_imports)]
pub(crate) use {
    with_bootstrap_interval, with_bootstrap_replicates, with_cluster_ids, with_multiway_ids,
    with_panel_times, with_sampling_design, with_se_kind, with_source_cluster_ids,
};
