//! Cluster-aware cross-fitted AIPW (2.2 E4): the independence unit is a declared cluster.
//!
//! Rows of one cluster are never split across cross-fit folds, so the nuisances scoring a
//! cluster are fit on other clusters only, and the dependence enters the *score* level: the
//! scores of a cluster are summed before they are squared. This is not an IID cross-fit
//! followed by a cluster standard error (`AnalyticSeKind::Cluster` on `AipwAte`): there, a
//! cluster's rows train the nuisances that score its own rows, which the cluster sandwich
//! cannot undo.
//!
//! The route publishes the cross-fitted point estimate and the score table and **no
//! interval**. The cluster-sandwich standard error of the scores is returned only as a named
//! [`ClusterDmlReceipt`] value (the derivation and its conditions are in
//! `docs/guides/clustered-dml.md`); no coverage record exists for it, so it is never turned
//! into an interval here. A cluster count below the declared minimum is a typed refusal, and
//! a dyadic independence unit (rows sharing endpoints) is a declared but closed option.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use antecedent_core::TargetPopulation;

use crate::error::EstimationError;
use crate::overlap::OverlapPolicy;
use crate::propensity::{refuse, trim_of};
use crate::scores::ScoreTable;
use crate::se::AnalyticSeKind;

/// Smallest cluster count a declaration may require. Below it the sandwich over cluster sums
/// has too few terms to mean anything, and a fold's training set would hold under eight
/// clusters at the default five folds.
pub const MIN_CLUSTERS_FLOOR: usize = 10;

/// Declared minimum cluster count when a caller does not choose one.
pub const DEFAULT_MIN_CLUSTERS: usize = 20;

/// Provenance tag a cluster-DML score table carries; a table bearing it publishes no interval.
pub(crate) const CLUSTER_DML_PROVENANCE_TAG: &str = ";cluster_dml=";

/// What is independent across the rows of the data.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum IndependenceUnit {
    /// Disjoint clusters (households, sites, patients): whole clusters are independent.
    Cluster,
    /// Rows are dyads whose endpoints repeat across rows. Declarable but closed: the fold
    /// ownership and covariance of dyadic data are a separate cell.
    Dyad,
}

impl IndependenceUnit {
    /// Stable wire name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Cluster => "cluster",
            Self::Dyad => "dyad",
        }
    }
}

/// Declared cluster-DML configuration of an `AipwAte`: the independence unit and the smallest
/// cluster count the route accepts. The cluster labels are the estimator's `cluster_ids`
/// (aligned to complete-case rows).
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ClusterDml {
    unit: IndependenceUnit,
    min_clusters: usize,
}

impl ClusterDml {
    /// Cluster independence with a declared minimum cluster count.
    ///
    /// # Errors
    ///
    /// `invalid_argument` when `min_clusters` is below [`MIN_CLUSTERS_FLOOR`].
    pub fn new(min_clusters: usize) -> Result<Self, EstimationError> {
        if min_clusters < MIN_CLUSTERS_FLOOR {
            return Err(refuse(
                antecedent_core::reason_code!("invalid_argument"),
                "cluster_dml.invalid_min_clusters",
                &format!(
                    "the declared minimum cluster count {min_clusters} is below the floor \
                     {MIN_CLUSTERS_FLOOR}"
                ),
            ));
        }
        Ok(Self { unit: IndependenceUnit::Cluster, min_clusters })
    }

    /// A dyadic independence declaration: accepted as a declaration, refused at execution.
    #[must_use]
    pub const fn dyadic(min_clusters: usize) -> Self {
        Self { unit: IndependenceUnit::Dyad, min_clusters }
    }

    /// The declared independence unit.
    #[must_use]
    pub const fn independence_unit(&self) -> IndependenceUnit {
        self.unit
    }

    /// The declared minimum cluster count.
    #[must_use]
    pub const fn min_clusters(&self) -> usize {
        self.min_clusters
    }

    /// Stable key of the declaration, part of the estimator-spec identity.
    #[must_use]
    pub fn canonical_key(&self) -> String {
        format!(
            "cluster_dml.aipw.v1;unit={};min_clusters={}",
            self.unit.as_str(),
            self.min_clusters
        )
    }

    /// Refuse the closed independence units before any work is done.
    ///
    /// # Errors
    ///
    /// `dyadic_dependence_not_licensed` for a dyadic unit.
    pub fn validate_for_execution(&self) -> Result<(), EstimationError> {
        match self.unit {
            IndependenceUnit::Cluster => Ok(()),
            IndependenceUnit::Dyad => Err(refuse(
                antecedent_core::reason_code!("dyadic_dependence_not_licensed"),
                "cluster_dml.dyadic_closed",
                "rows that share endpoints are dependent through both endpoints, so disjoint \
                 clusters do not describe them: dyadic fold ownership and two-way covariance \
                 are not licensed, use the screen/estimate split of connected endpoint \
                 components for selection and a cluster declaration only when rows are \
                 disjoint clusters",
            )),
        }
    }

    /// Check the declared labels against the prepared rows and the declared minimum, and
    /// return them as the fold units.
    ///
    /// # Errors
    ///
    /// `required_option_missing` without labels, a length mismatch, or `too_few_clusters`
    /// below the declared minimum.
    pub fn declare_units(
        &self,
        cluster_ids: Option<&[u32]>,
        nrows: usize,
    ) -> Result<Arc<[u32]>, EstimationError> {
        self.validate_for_execution()?;
        let Some(ids) = cluster_ids else {
            return Err(refuse(
                antecedent_core::reason_code!("required_option_missing"),
                "cluster_dml.cluster_ids_missing",
                "cluster-DML AIPW needs the cluster label of every complete-case row \
                 (estimator cluster_ids)",
            ));
        };
        if ids.len() != nrows {
            return Err(EstimationError::data_msg(format!(
                "cluster_ids length {} != nrows {nrows}",
                ids.len()
            )));
        }
        let found = distinct_count(ids);
        if found < self.min_clusters {
            return Err(refuse(
                antecedent_core::reason_code!("too_few_clusters"),
                "cluster_dml.too_few_clusters",
                &format!(
                    "{found} clusters are below the declared minimum {}; a sandwich over \
                     cluster sums needs many independent clusters and no interval or \
                     standard error is formed",
                    self.min_clusters
                ),
            ));
        }
        Ok(Arc::from(ids))
    }

    /// The cluster-sandwich standard error of a cluster-DML score table with its receipt.
    ///
    /// `clusters` are the labels of the table's rows. The table must hold the mean
    /// functional's two arm columns and every cluster's rows must lie in one fold; a table
    /// whose clusters were split across folds is not cluster-DML evidence and is refused.
    ///
    /// # Errors
    ///
    /// A closed independence unit, a label/row mismatch, too few clusters, a cluster split
    /// across folds, or a table that is not a single mean contrast.
    pub fn receipt(
        &self,
        table: &ScoreTable,
        clusters: &[u32],
    ) -> Result<ClusterDmlReceipt, EstimationError> {
        self.validate_for_execution()?;
        if clusters.len() != table.n_rows || table.fold_ids.len() != table.n_rows {
            return Err(EstimationError::data_msg(
                "cluster labels and fold ids must align with the score table rows",
            ));
        }
        if table.columns.len() != 2 {
            return Err(EstimationError::unsupported(
                "the cluster-DML receipt covers the mean functional's two arm columns",
            ));
        }
        let found = distinct_count(clusters);
        if found < self.min_clusters {
            return Err(refuse(
                antecedent_core::reason_code!("too_few_clusters"),
                "cluster_dml.too_few_clusters",
                &format!("{found} clusters are below the declared minimum {}", self.min_clusters),
            ));
        }
        require_whole_clusters_per_fold(clusters, &table.fold_ids)?;
        let psi: Vec<f64> =
            table.column(0)?.iter().zip(table.column(1)?).map(|(&a, &b)| b - a).collect();
        Ok(ClusterDmlReceipt {
            independence_unit: self.unit,
            n_clusters: found,
            min_clusters: self.min_clusters,
            n_rows: table.n_rows,
            folds: table.n_folds as usize,
            unit_digest: unit_digest(clusters),
            cluster_sandwich_se: crate::se::cluster_influence_se(&psi, clusters)?,
        })
    }
}

/// Declared unit, counts and the cluster-sandwich standard error of the cross-fitted scores.
///
/// `cluster_sandwich_se` is `sqrt(G/(G-1) * sum_g S_g^2) / n` with `S_g` the cluster sum of
/// the centered contrast scores. It is a variance receipt of the stored scores under the
/// declared independence unit, not an interval and not a calibrated claim.
#[derive(Clone, Debug, PartialEq)]
pub struct ClusterDmlReceipt {
    /// The declared independence unit.
    pub independence_unit: IndependenceUnit,
    /// Distinct clusters.
    pub n_clusters: usize,
    /// The declared minimum the count satisfied.
    pub min_clusters: usize,
    /// Complete-case rows.
    pub n_rows: usize,
    /// Cross-fit folds.
    pub folds: usize,
    /// Fingerprint of the sorted distinct cluster labels (row-order invariant).
    pub unit_digest: String,
    /// Cluster-sandwich standard error of the mean contrast.
    pub cluster_sandwich_se: f64,
}

/// Seeded whole-cluster fold plan: every distinct cluster is dealt to one fold, so a fold
/// depends on the seed and the cluster label, never on row order or on how many rows a
/// cluster has. Clusters are not arm-stratified (arms vary inside a cluster).
///
/// # Errors
///
/// Fewer than two folds, or fewer distinct clusters than folds.
pub fn cluster_fold_plan(
    clusters: &[u32],
    folds: usize,
    seed: u64,
) -> Result<Vec<u32>, EstimationError> {
    crate::learn_nuisance::crossfit_fold_plan(&vec![0; clusters.len()], clusters, folds, seed)
}

/// The route is licensed for the untrimmed `AllObserved` mean ATE only.
pub(crate) fn require_scope(
    mean_functional: bool,
    population: &TargetPopulation,
    overlap: OverlapPolicy,
) -> Result<(), EstimationError> {
    if !mean_functional
        || !matches!(population, TargetPopulation::AllObserved)
        || trim_of(overlap).is_some()
    {
        return Err(refuse(
            antecedent_core::reason_code!("route_not_supported"),
            "cluster_dml.scope",
            "cluster-DML AIPW is licensed only for the untrimmed AllObserved mean ATE: other \
             functionals and populations, and trimmed fits, refit full-sample nuisances that \
             have no whole-cluster cross-fit",
        ));
    }
    Ok(())
}

/// A penalized propensity keeps points only and its remainder is not shown negligible, so it
/// is not combined with the cluster claim.
pub(crate) fn require_unpenalized(penalized: bool) -> Result<(), EstimationError> {
    if penalized {
        return Err(refuse(
            antecedent_core::reason_code!("route_not_supported"),
            "cluster_dml.penalized_propensity",
            "a penalized propensity publishes a point only and is not combined with the \
             cluster-DML declaration",
        ));
    }
    Ok(())
}

/// The route publishes no interval: a bootstrap (row resampling would break clusters) or an
/// analytic SE kind is refused instead of ignored.
pub(crate) fn refuse_interval_request(
    bootstrap_replicates: u32,
    se_kind: AnalyticSeKind,
) -> Result<(), EstimationError> {
    if bootstrap_replicates != 0 || !matches!(se_kind, AnalyticSeKind::Homoskedastic) {
        return Err(refuse(
            antecedent_core::reason_code!("cluster_interval_not_licensed"),
            "cluster_dml.interval_withheld",
            "no interval is licensed for cluster-DML AIPW (no coverage record exists, and a \
             row bootstrap would split clusters); set bootstrap_replicates to 0 and keep the \
             default se_kind to receive the point estimate, the score table and the \
             cluster-sandwich receipt",
        ));
    }
    Ok(())
}

/// Provenance suffix a cluster-DML score table carries.
pub(crate) fn provenance_suffix(units: &[u32]) -> String {
    format!(
        "{CLUSTER_DML_PROVENANCE_TAG}unit=cluster;clusters={};digest={}",
        distinct_count(units),
        unit_digest(units)
    )
}

/// Whether a score table's provenance marks whole-cluster cross-fitting.
pub(crate) fn provenance_marks_cluster_units(provenance: &str) -> bool {
    provenance.contains(CLUSTER_DML_PROVENANCE_TAG)
}

fn distinct_count(ids: &[u32]) -> usize {
    ids.iter().collect::<BTreeSet<_>>().len()
}

/// FNV-1a over the sorted distinct labels: a fingerprint, not a cryptographic digest (the
/// estimator identity digests the full label vector).
fn unit_digest(ids: &[u32]) -> String {
    let distinct: BTreeSet<u32> = ids.iter().copied().collect();
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for id in distinct {
        for byte in id.to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    format!("{hash:016x}")
}

fn require_whole_clusters_per_fold(
    clusters: &[u32],
    fold_ids: &[u32],
) -> Result<(), EstimationError> {
    let mut fold_of: BTreeMap<u32, u32> = BTreeMap::new();
    for (&cluster, &fold) in clusters.iter().zip(fold_ids) {
        let first = *fold_of.entry(cluster).or_insert(fold);
        if first != fold {
            return Err(EstimationError::data_msg(format!(
                "cluster {cluster} is split across cross-fit folds {first} and {fold}: its rows \
                 are not independent of the nuisances that score them"
            )));
        }
    }
    Ok(())
}
