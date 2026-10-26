//! Temporal bootstrap / resampling index plans.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![cfg_attr(
    test,
    allow(
        clippy::cast_possible_truncation,
        reason = "test fixtures compare exact constants and index with small literals"
    )
)]

use std::sync::Arc;

use antecedent_core::{
    CausalRng, DeclaredSamplingDesign, ExecutionContext, ResamplingRule, SamplingDesignDeclaration,
    SamplingDesignError, ScratchLease, ScratchRefusal, SplittingRule, StreamDomain, VariableId,
    derive_stream,
};
use antecedent_kernels::{fill_bayesian_bootstrap_weights, unbiased_index};

use crate::dataset::TimeSeriesData;
use crate::error::DataError;
use crate::global_index::GlobalId;
use crate::split::TemporalRandomPolicy;
use crate::storage::OwnedColumnarStorage;
use crate::table::TableView;

mod indexes;
mod replicates;
mod series;
#[cfg(test)]
mod tests;

pub use self::indexes::circular_block_length;
pub use self::indexes::fill_circular_block_indexes;
pub use self::replicates::fill_resample_index_batch;
pub use self::replicates::fill_resample_index_replicate;
#[cfg(test)]
pub(crate) use self::replicates::fill_resample_weight_batch;
pub use self::replicates::fill_resample_weight_replicate;
pub(crate) use self::replicates::fill_resample_weights;
pub use self::series::resample_timeseries;
pub(crate) use self::series::resample_timeseries_grouped;
pub(crate) use self::series::resample_timeseries_with_policy;

use self::indexes::{
    declared_design_error, design_independent_of_time_column, draw_contiguous_block,
    draw_grouped_units, fill_indexes_prepared, prepare_clusters, refuse_row_iid,
};

fn row_id(index: usize) -> Result<GlobalId, DataError> {
    u64::try_from(index)
        .map(GlobalId::new)
        .map_err(|_| DataError::overflow("row index as a GlobalId"))
}

fn row_pos(id: GlobalId) -> Result<usize, DataError> {
    usize::try_from(id.get()).map_err(|_| {
        DataError::overflow("GlobalId as a row position (keep ChunkIndex inside one bounded chunk)")
    })
}

/// Null / permutation scheme for [`ResamplingPlan::Permutation`].
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum PermutationScheme {
    /// Full Fisher–Yates (Fisher & Yates 1938; Durstenfeld 1964) shuffle of row indexes `0..n`.
    Full,
    /// Shuffle within each cluster (ids supplied to the fill helper).
    WithinCluster,
}

/// Resampling plan producing row-index or weight replicates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ResamplingPlan {
    /// IID with-replacement row bootstrap.
    IidBootstrap,
    /// Bayesian bootstrap: Dirichlet(1,…,1) observation weights (Rubin).
    BayesianBootstrap,
    /// Moving-block bootstrap (contiguous blocks).
    MovingBlock {
        /// Block length in rows.
        length: usize,
    },
    /// Circular block bootstrap (wraps the series).
    CircularBlock {
        /// Block length in rows.
        length: usize,
    },
    /// Cluster bootstrap: draw G whole clusters with replacement from G observed clusters.
    /// Replicate row counts vary when cluster sizes differ (ids via grouped fill).
    ///
    /// A cluster drawn `k` times contributes its rows `k` times under one original
    /// label: the replicate is a row-index list, so a cluster-aware estimator (cluster-robust
    /// variance, cluster fixed effects) must treat the repeats as distinct clusters
    /// rather than merge them by label.
    ///
    /// `cluster` names the variable whose labels are supplied as `cluster_ids`
    /// to the fill helpers (labels remain caller-resolved).
    ClusterBootstrap {
        /// Variable that holds cluster membership labels.
        cluster: VariableId,
    },
    /// Stationary block bootstrap (Politis & Romano 1994 geometric lengths).
    StationaryBlock {
        /// Expected block length (mean of geometric distribution).
        expected_length: f64,
    },
    /// Permutation (shuffle) index plan.
    Permutation(PermutationScheme),
}

impl ResamplingPlan {
    /// Whether this plan yields observation weights rather than row indexes.
    #[must_use]
    pub const fn is_weight_plan(self) -> bool {
        matches!(self, Self::BayesianBootstrap)
    }

    /// Resampling rule this plan applies, for a declared design to accept or refuse.
    ///
    /// An iid bootstrap, a Bayesian bootstrap, and a full-row permutation draw rows.
    /// A trajectory or clustered panel refuses that draw. A within-cluster permutation
    /// keeps each unit's rows together and is the subject rule. Whole-cluster and
    /// contiguous-block plans keep their own rules.
    #[must_use]
    pub const fn resampling_rule(self) -> ResamplingRule {
        match self {
            Self::IidBootstrap
            | Self::BayesianBootstrap
            | Self::Permutation(PermutationScheme::Full) => ResamplingRule::RowIid,
            Self::Permutation(PermutationScheme::WithinCluster) => ResamplingRule::RowSubjectIid,
            Self::MovingBlock { .. }
            | Self::CircularBlock { .. }
            | Self::StationaryBlock { .. } => ResamplingRule::ContiguousBlock,
            Self::ClusterBootstrap { .. } => ResamplingRule::WholeCluster,
        }
    }

    /// Whether this plan requires per-row cluster labels.
    #[must_use]
    pub const fn needs_clusters(self) -> bool {
        matches!(
            self,
            Self::ClusterBootstrap { .. } | Self::Permutation(PermutationScheme::WithinCluster)
        )
    }
}

/// Fill `out` with a length-`n` row-index plan under `plan`.
///
/// Plans that need cluster labels ([`ResamplingPlan::ClusterBootstrap`],
/// [`PermutationScheme::WithinCluster`]) must use
/// [`fill_resample_indexes_grouped`] instead.
///
/// # Errors
///
/// Zero series length, zero block length, weight-only plan, or a clustered plan.
pub fn fill_resample_indexes(
    plan: ResamplingPlan,
    n: usize,
    rng: &mut CausalRng,
    out: &mut Vec<GlobalId>,
) -> Result<(), DataError> {
    if plan.needs_clusters() {
        return Err(DataError::precondition(
            "a clustered resampling plan",
            "fill_resample_indexes_grouped",
        ));
    }
    fill_resample_indexes_grouped(plan, n, None, rng, out)
}

/// Fill `out` with a row-index plan, optionally using `cluster_ids`.
///
/// Cluster bootstrap draws as many clusters as were observed, so unequal cluster
/// sizes produce variable-length output. All other index plans return `n` rows.
///
/// # Errors
///
/// Shape / plan mismatches as in [`fill_resample_indexes`], plus missing or
/// wrong-length cluster ids when required.
///
/// # Panics
///
/// Propagates panic from [`unbiased_index`] if a zero-length draw is requested
/// after the `n > 0` guard (should be unreachable).
pub(crate) fn fill_resample_indexes_grouped(
    plan: ResamplingPlan,
    n: usize,
    cluster_ids: Option<&[u32]>,
    rng: &mut CausalRng,
    out: &mut Vec<GlobalId>,
) -> Result<(), DataError> {
    let clusters = prepare_clusters(plan, n, cluster_ids)?;
    fill_indexes_prepared(plan, n, clusters.as_ref(), rng, out)
}

/// Fill `out` with a row-index plan under `plan`, after the declared design accepts the
/// plan's resampling rule.
///
/// A dependent series refuses a plan that drops its dependence. That refusal is
/// [`DataError::SamplingDesign`], whose [`DataError::route_diagnostic`] is the failed
/// dependent-resampling check.
///
/// # Errors
///
/// A plan whose resampling rule the declared design does not allow, then the errors of
/// [`fill_resample_indexes_grouped`]'s plan and shape checks.
pub fn fill_declared_plan(
    declaration: SamplingDesignDeclaration,
    plan: ResamplingPlan,
    n: usize,
    cluster_ids: Option<&[u32]>,
    rng: &mut CausalRng,
    out: &mut Vec<GlobalId>,
) -> Result<(), DataError> {
    let design = design_independent_of_time_column(declaration);
    let rule = plan.resampling_rule();
    if matches!(rule, ResamplingRule::RowIid) {
        return Err(refuse_row_iid(design));
    }
    design.apply_resampling(rule).map_err(declared_design_error)?;
    fill_resample_indexes_grouped(plan, n, cluster_ids, rng, out)
}

/// Draw one replicate for a declared sampling design.
///
/// The design selects the draw. [`SamplingDesignDeclaration::time_column_present`]
/// is recorded and is not that selection. A trajectory without subject labels is a
/// row bootstrap and is refused; with labels, the whole trajectory is the draw.
/// A panel draws whole clusters. A series draws a contiguous block on the
/// interior left after the embargo on each side.
///
/// # Errors
///
/// A row-IID attempt on a trajectory or panel, an embargo on a design with no block
/// boundary, a panel without cluster labels, a block that does not fit after
/// the embargo, or a row id that does not fit in [`GlobalId`].
pub fn fill_declared_resample(
    declaration: SamplingDesignDeclaration,
    n: usize,
    cluster_ids: Option<&[u32]>,
    block_length: usize,
    embargo: u32,
    rng: &mut CausalRng,
    out: &mut Vec<GlobalId>,
) -> Result<(), DataError> {
    let design = design_independent_of_time_column(declaration);
    match design.rules().resampling() {
        ResamplingRule::RowIid => Err(refuse_row_iid(design)),
        ResamplingRule::RowSubjectIid => draw_grouped_units(
            design,
            ResamplingRule::RowSubjectIid,
            SplittingRule::RowSubjectIid,
            n,
            cluster_ids,
            embargo,
            true,
            rng,
            out,
        ),
        ResamplingRule::WholeCluster => draw_grouped_units(
            design,
            ResamplingRule::WholeCluster,
            SplittingRule::WholeCluster,
            n,
            cluster_ids,
            embargo,
            false,
            rng,
            out,
        ),
        ResamplingRule::ContiguousBlock => {
            draw_contiguous_block(design, n, block_length, embargo, rng, out)
        }
    }
}
